"""GUI 后台控制器：线程 + queue，Tk 主线程零阻塞。

铁律：
- pipeline/Provider/FFmpeg 只在工作线程调用；控件永远不被工作线程触碰。
- 工作线程只产出事件 tuple 放入 queue.Queue；Tk 用 after() 轮询排空。
- 暂停/取消无法撤回已发出的云端请求（RunControl 只门控新提交）。

生成 → 拼接（本控制器职责，CLI 另有 main._auto_merge）：
- 文本/章节/失败重试/局部重生成，片段全部成功后自动 run_merge；
- 只有 ffprobe 能识别最终输出文件才报"已完成"；
- 拼接失败保留全部片段，发 merge_failed 事件，用户可只重试拼接（不调 API）。
"""
from __future__ import annotations

import queue
import threading
import traceback
from pathlib import Path

from podcast_tts.audio import MergeError, run_merge
from podcast_tts.config import ConfigError
from podcast_tts.gui.settings import GuiSettings
from podcast_tts.parser import ParseError, match_speaker_header
from podcast_tts.pipeline import (
    RunControl,
    all_speech_clean,
    chapter_freshness,
    inspect_segments,
    plan_segment,
    run_project,
    run_regenerate,
    run_retry,
    source_snapshot_for_edit,
    validate_emotags,
)
from podcast_tts.project import (
    WorkingScript,
    parse_working,
)
from podcast_tts.models import Segment
from podcast_tts.providers import get_provider

# 事件类型：status/progress/log/failed/finished/fatal/audition_done/merge_failed


def build_script_text(settings: GuiSettings, text: str) -> str:
    """把文本框内容落成标准脚本文本。

    - 首行已是"已注册角色"的角色头（支持中文显示名/[旁白] 形式）时原样保留；
    - 否则必须显式选择「默认角色」，补一个不含导演指令的纯角色头——
      项目级导演指令走配置层级（cfg.default_instruction），不塞进角色头，
      否则它会被当成行内指令而错误地覆盖角色默认/章节指令。
    """
    text = text.strip("\n")
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    name_to_id = settings.speaker_name_to_id()
    if match_speaker_header(first_line.strip(), name_to_id) is not None:
        return text if text.endswith("\n") else text + "\n"
    role = (settings.default_role or "").strip()
    if not role:
        raise ConfigError(
            "文本没有角色标记（如 \"角色名：台词\"），且未选择「默认角色」。"
            "请在角色区选择一个默认角色，或在每段台词前加上已注册的角色名；"
            "不会静默使用任何角色。"
        )
    return f"{role}：\n{text.strip()}\n"


def preview_working(
    settings: GuiSettings, ws: WorkingScript
) -> tuple[list[dict], list[str]]:
    """工作脚本生成预览（粘贴/文件统一入口）：返回 (片段行列表, 错误列表)。

    纯本地，不触网、不落盘；行内带"状态/本次动作"，汇总由 preview_summary() 取。
    """
    cfg = settings.to_app_config()
    try:
        rows, _manifest = inspect_segments(
            ws.episode, cfg, ws,
            default_speaker_id=settings.default_role_id() or None,
            persist=False,
        )
        # 标签校验与正式生成同一套（拼写/模型支持/SSML 组合）
        segs = parse_working(
            ws, cfg, default_speaker_id=settings.default_role_id() or None)
        validate_emotags(segs, cfg)
    except (ParseError, ConfigError) as exc:
        return [], [str(exc)]
    except Exception as exc:  # noqa: BLE001 —— 预览中把解析错误直接展示
        return [], [f"{type(exc).__name__}: {exc}"]
    out_rows: list[dict] = []
    for r in rows:
        ext = r.get("audio_format", "wav")
        if r["kind"] == "pause":
            output = (r.get("file")
                      or f"segments/{r['index']:04d}_PAUSE_{r['pause_seconds']:g}s.wav")
            original = f"[PAUSE:{r['pause_seconds']:g}]"
            text = f"停顿 {r['pause_seconds']:g} 秒"
        else:
            output = r.get("file") or f"segments/{r['index']:04d}_{r['speaker']}.{ext}"
            original = r.get("raw_text") or r["text"]
            text = r["text"]
        row = {
            "index": r["index"], "kind": r["kind"],
            "speaker_id": r["speaker"], "speaker_name": r["speaker_name"],
            "original": original, "text": text,
            "injected": r.get("injected"), "origin": r.get("origin"),
            "tags": r.get("tags", ""),
            "instruction": r.get("instruction"),
            "rate": r.get("rate"), "pitch": r.get("pitch"),
            "volume": r.get("volume"), "output": output,
            "error": r.get("error"),
            "state": r["state"], "state_zh": r["state_zh"],
            "action_zh": r["action_zh"],
        }
        if r["kind"] == "pause":
            row["pause_seconds"] = r["pause_seconds"]
        out_rows.append(row)
    return out_rows, []


def preview_summary(rows: list[dict]) -> dict:
    """预览顶部汇总：将生成 N / 将重试 N / 跳过 N / 停顿 N / 待更新 N。"""
    summary = {"generate": 0, "retry": 0, "skip": 0, "pause": 0,
               "dirty": 0, "errors": 0}
    for r in rows:
        state = r.get("state")
        if r.get("error"):
            summary["errors"] += 1
        if state == "pause":
            summary["pause"] += 1
        elif state == "missing":
            summary["generate"] += 1
        elif state == "failed":
            summary["retry"] += 1
        elif state == "dirty":
            summary["dirty"] += 1
            summary["generate"] += 1
        elif state == "clean":
            summary["skip"] += 1
    return summary


def chapter_summary(settings: GuiSettings, ws: WorkingScript) -> dict:
    """主界面状态摘要（纯本地不触网）；配置不完整时返回 {"error": ...}。"""
    try:
        cfg = settings.to_app_config()
        return chapter_freshness(
            ws.episode, cfg, ws,
            default_speaker_id=settings.default_role_id() or None)
    except (ConfigError, ParseError) as exc:
        return {"error": str(exc)}


def preview_text(settings: GuiSettings, text: str) -> tuple[list[dict], list[str]]:
    """文本模式预览：粘贴内容即工作脚本（不落盘）。纯本地。"""
    ws = WorkingScript.from_paste(settings.episode_name, text)
    return preview_working(settings, ws)


def preview_book(settings: GuiSettings, book_path: str) -> tuple[list[dict], list[str]]:
    """章节文件模式预览：文件内容复制为工作脚本（不触碰源文件）。纯本地。"""
    path = Path(book_path)
    if not path.is_file():
        return [], [f"章节文件不存在：{book_path}"]
    try:
        ws = WorkingScript.import_file(path.stem, path, mode="book")
    except Exception as exc:  # noqa: BLE001 —— 读取/编码错误直接展示
        return [], [f"{type(exc).__name__}: {exc}"]
    return preview_working(settings, ws)


class GuiController:
    def __init__(self, event_queue: "queue.Queue | None" = None,
                 provider_factory=None, merge_func=None):
        self.events: queue.Queue = event_queue or queue.Queue()
        self._provider_factory = provider_factory or get_provider
        # 可注入拼接函数（测试替身）；默认真实 run_merge（本地 FFmpeg，不触网）
        self._merge_func = merge_func or run_merge
        self._thread: threading.Thread | None = None
        self.control: RunControl | None = None
        self.state = "idle"      # idle|running|paused|cancelling|merging|done|
                                 # merge_failed|cancelled|error
        self.episode: str | None = None
        # 当前工作脚本（粘贴/导入两个入口的唯一汇合形态）
        self.ws: WorkingScript | None = None

    # —— 事件（工作线程 → queue → Tk 主线程） ————————————————————————

    def _emit(self, kind: str, *payload) -> None:
        self.events.put((kind,) + payload)

    def _set_state(self, state: str) -> None:
        self.state = state
        self._emit("status", state)

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # —— 启动（粘贴/文件统一走 WorkingScript → run_project） ———————————————

    def build_script(self, settings: GuiSettings, text: str) -> str:
        """兼容保留：角色头拼装（预览/旧测试用）；正式生成已直接走工作脚本。"""
        return build_script_text(settings, text)

    def start_text(self, settings: GuiSettings, text: str) -> None:
        """粘贴文本直接成为工作脚本（source_type=paste）。"""
        self.set_settings(settings)
        episode = settings.episode_name
        ws = WorkingScript.from_paste(episode, text)
        self.ws = ws
        self.episode = episode
        self._start(
            lambda cfg, provider, control: run_project(
                episode, cfg, ws,
                default_speaker_id=settings.default_role_id() or None,
                provider=provider, control=control, persist=True),
            mode="generate")

    def start_book(self, settings: GuiSettings, book_path: str) -> None:
        """章节文件内容复制为工作脚本（只读源文件，绝不回写）。"""
        path = Path(book_path)
        if not path.is_file():
            raise FileNotFoundError(f"章节文件不存在：{book_path}")
        self.set_settings(settings)
        episode = path.stem
        ws = WorkingScript.import_file(episode, path, mode="book")
        self.ws = ws
        self.episode = episode
        self._start(
            lambda cfg, provider, control: run_project(
                episode, cfg, ws,
                default_speaker_id=settings.default_role_id() or None,
                provider=provider, control=control, persist=True),
            mode="generate")

    def start_working(self, settings: GuiSettings, ws: WorkingScript) -> None:
        """通用入口：用界面当前工作脚本生成（片段管理器修改后继续生成等）。"""
        self.set_settings(settings)
        self.episode = ws.episode
        self.ws = ws
        self._start(
            lambda cfg, provider, control: run_project(
                ws.episode, cfg, ws,
                default_speaker_id=settings.default_role_id() or None,
                provider=provider, control=control, persist=True),
            mode="generate")

    def retry_failed(self) -> None:
        if not self.episode:
            raise ConfigError("尚无 episode，无法重试。")
        self._start(lambda cfg, provider, control: run_retry(
            self.episode, cfg, provider=provider, control=control), mode="generate")

    def regenerate(self, indices: list[int]) -> None:
        """旧契约：按序号重生成（CLI/兼容测试用），文本以 manifest 为准。"""
        if not self.episode:
            raise ConfigError("尚无 episode，无法局部重生成。")
        self._start(lambda cfg, provider, control: run_regenerate(
            self.episode, indices, cfg, provider=provider, control=control),
            mode="generate")

    def regenerate_segments(
        self, settings: GuiSettings, seg_ids: list[str], ws: WorkingScript | None = None
    ) -> None:
        """单句重生成：只对指定 seg_id 发 TTS 请求，其他片段音频不动。"""
        ws = ws or self.ws
        if ws is None:
            raise ConfigError("当前没有工作脚本，无法局部重生成。")
        episode = ws.episode
        self.set_settings(settings)
        self.episode = episode
        self.ws = ws
        self._start(
            lambda cfg, provider, control: run_regenerate(
                episode, [], cfg, seg_ids=seg_ids, ws=ws,
                provider=provider, control=control),
            mode="generate")

    # —— 片段管理器（纯本地，不触网） ————————————————————————————————————

    def segment_rows(
        self, settings: GuiSettings, ws: WorkingScript | None = None
    ) -> list[dict]:
        """返回当前工作脚本每个片段的状态/请求规划行（管理器表格用）。

        先落盘规划（新片段由此获得持久化 seg_id），再统一走 inspect_segments：
        状态判定全应用唯一来源（content_hash + 文件存在性）。
        """
        ws = ws or self.ws
        if ws is None:
            raise ConfigError("当前没有工作脚本。")
        cfg = settings.to_app_config()
        default_id = settings.default_role_id() or None
        rows, _manifest = inspect_segments(
            ws.episode, cfg, ws, default_speaker_id=default_id, persist=True)
        return rows

    def edit_segment(
        self, settings: GuiSettings, seg_id: str, patch: dict,
        ws: WorkingScript | None = None,
    ) -> list[dict]:
        """保存单句编辑到工作脚本覆盖层并落盘；返回刷新后的片段行。

        保存前记录每个被改字段的"源脚本当前值"快照（base）：之后若主文本框
        修改了同一字段（源值偏离 base），该字段覆盖在规划时自动失效，
        实现两个编辑入口的 latest-edit-wins。覆盖只影响最终 TTS 请求字段
        → 仅该片段 content_hash 变化（dirty）。
        """
        ws = ws or self.ws
        if ws is None:
            raise ConfigError("当前没有工作脚本。")
        cfg = settings.to_app_config()
        default_id = settings.default_role_id() or None
        base = source_snapshot_for_edit(
            ws.episode, cfg, ws, seg_id, set(patch.keys()),
            default_speaker_id=default_id)
        ws.set_override(seg_id, patch, base=base)
        ws.save(cfg.projects_dir)
        return self.segment_rows(settings, ws)

    def merge_only(self, *, force: bool = True) -> None:
        """只重新拼接（不调用任何云端 API）。片段生成后拼接失败时的恢复入口。"""
        if not self.episode:
            raise ConfigError("尚无 episode，无法合并。")
        if not getattr(self, "_settings", None):
            raise ConfigError("当前没有可用的设置，请先在界面确认后再重试拼接。")
        self._start(lambda cfg, provider, control: self._do_merge(cfg, force=force),
                    mode="merge", needs_provider=False)

    def audition(self, settings: GuiSettings, sample_text: str) -> None:
        """试听：用当前设置合成一小段（走真实 pipeline 参数构造）。"""
        self.set_settings(settings)
        self.episode = settings.episode_name
        self._start(lambda cfg, provider, control: self._run_audition(
            settings, sample_text, cfg, provider), mode="audition")

    def _run_audition(self, settings, sample_text, cfg, provider):
        from podcast_tts.models import Segment, TTSRequest

        role = (settings.default_role or "").strip()
        if not role:
            raise ConfigError(
                "试听需要明确的角色：请在「默认角色（可选）」中选择一个角色，"
                "不会静默使用角色列表中的第一个。"
            )
        rv = settings.role_by_name(role)
        sid = settings.default_role_id()
        if not rv or sid not in cfg.voices:
            raise ConfigError("请先选择一个有效角色再试听。")
        voice_setting = cfg.voices[sid]
        # 试听与正式生成走同一规划：指令按 项目级 < 角色默认级 取最终一条，
        # 参数按 项目默认 < 角色默认 合并，绝不另搞一套
        seg = Segment(
            index=1, kind="speech", speaker=sid, speaker_name=rv.role,
            text=sample_text.strip() or "试听样例。",
        )
        # 试听与正式生成走同一套标签校验（拼写/模型支持/SSML 组合），
        # 避免“试听可用、正式生成失败”
        validate_emotags([seg], cfg)
        _, direction, extras = plan_segment(cfg, seg)
        out_dir = cfg.projects_dir / "_audition"
        out_dir.mkdir(parents=True, exist_ok=True)
        ext = str(extras.get("format", "wav"))
        out = out_dir / f"audition_{sid}.{ext}"
        # 音色试听目录只保留"本次角色 + 本次格式"这一份最新文件，
        # 删除该角色历史格式残留（不碰其他角色试听文件与正式项目）
        for old in out_dir.glob(f"audition_{sid}.*"):
            try:
                if old.resolve() != out.resolve():
                    old.unlink()
            except OSError:
                pass
        req = TTSRequest(
            text=seg.text,
            voice=voice_setting.voice,
            language_type=voice_setting.language_type,
            direction=direction,
            output_path=str(out),
            audio_params=extras,
        )
        result = provider.generate(req)
        self._emit("audition_done", result.output_path)
        return None

    # —— 拼接（工作线程） ——————————————————————————————————————————————

    def _expected_output(self, cfg) -> str:
        return str(cfg.projects_dir / self.episode /
                   f"output/{self.episode}.{cfg.merge_format}")

    def build_srt(self) -> tuple[str, int]:
        """生成与最终音频同源的 SRT 文本（纯本地，主线程调用即可）。

        守卫与普通自动合并一致：有未生成/待更新/失败的台词片段时拒绝，避免字幕
        与用户实际听到的成品不一致。返回 (srt 文本, 字幕条数)；纯停顿项目条数为 0。
        """
        from podcast_tts.subtitles import SrtError, render_for_episode

        cfg = self._settings.to_app_config()
        if self.ws is not None:
            default_id = (
                self._settings.default_role_id() if self._settings else None
            ) or None
            rows, _m = inspect_segments(
                self.episode, cfg, self.ws,
                default_speaker_id=default_id, persist=False)
            pending = [
                f"{r['index']:04d}（{r['state_zh']}）"
                for r in rows
                if r["kind"] == "speech" and r["state"] != "clean"
            ]
            if pending:
                raise ConfigError(
                    "还有台词片段不是最新状态，字幕必须与最终音频一致，暂不导出：\n"
                    + "、".join(pending)
                    + "\n请先点「生成 / 更新音频」完成这些片段并成功拼接整章。"
                )
        try:
            return render_for_episode(self.episode, cfg)
        except SrtError:
            raise
        except MergeError as exc:
            raise ConfigError(str(exc)) from exc

    def _do_merge(self, cfg, *, force: bool, guarded: bool = True):
        """执行拼接并回报。

        返回 MergeOutcome（成功）/ False（严格守卫跳过，未产出最新成品）/
        None（拼接失败，merge_failed 事件已发）。

        guarded=True（普通"生成 / 更新音频"流程）：合并前统一检查全部台词片段
        必须已生成，并把当前请求指纹 expected_hashes 传给 run_merge——
        绝不让 dirty/missing/failed 片段的旧音频静默拼进"最新整章"。
        force=True（高级菜单明确要求强制重拼）：不守卫，使用现有片段文件。
        """
        self._set_state("merging")
        self._emit("log", "正在拼接最终音频（按脚本顺序，不使用请求完成顺序）…")

        expected_hashes = None
        if guarded and not force and self.ws is not None:
            default_id = (
                self._settings.default_role_id() if self._settings else None
            ) or None
            rows, _m = inspect_segments(
                self.episode, cfg, self.ws, default_speaker_id=default_id,
                persist=False)
            if not all_speech_clean(rows):
                pending = [
                    f"{r['index']:04d}（{r['state_zh']}）"
                    for r in rows
                    if r["kind"] == "speech" and r["state"] != "clean"
                ]
                self._emit(
                    "log",
                    f"还有 {len(pending)} 个片段待更新（{'、'.join(pending)}），"
                    "已保留本次生成结果，但不产出最新整章；待全部片段更新后会自动拼接。",
                )
                self._emit("merge_skipped", {
                    "episode": self.episode, "pending": pending,
                    "output": self._expected_output(cfg),
                })
                self._set_state("done")
                return False
            expected_hashes = {
                r["seg_id"]: r["expected_hash"]
                for r in rows
                if r["kind"] == "speech" and r.get("expected_hash")
            }

        try:
            kwargs = {"force": force}
            if expected_hashes is not None:
                kwargs["expected_hashes"] = expected_hashes
            outcome = self._merge_func(self.episode, cfg, **kwargs)
        except MergeError as exc:
            self._emit("log", f"[拼接失败] {exc}")
            self._emit("merge_failed", {
                "episode": self.episode,
                "error": str(exc),
                "output": self._expected_output(cfg),
            })
            self._set_state("merge_failed")
            return None
        except FileNotFoundError as exc:
            self._emit("log", f"[拼接失败] {exc}")
            self._emit("merge_failed", {
                "episode": self.episode,
                "error": str(exc),
                "output": self._expected_output(cfg),
            })
            self._set_state("merge_failed")
            return None

        out_abs = cfg.projects_dir / self.episode / outcome.output
        # run_merge 内部已用 ffprobe 识别过最终文件；这里再确认文件存在
        if not Path(out_abs).is_file():
            msg = f"拼接流程未报错但找不到最终输出：{out_abs}"
            self._emit("log", f"[拼接失败] {msg}")
            self._emit("merge_failed", {
                "episode": self.episode, "error": msg,
                "output": str(out_abs)})
            self._set_state("merge_failed")
            return None
        dur = f"，时长 {outcome.duration:.1f}s" if outcome.duration else ""
        tag = "（片段未变化，沿用已验证成品）" if outcome.skipped else ""
        self._emit("log", f"拼接成功{tag}：成品输出 {out_abs}{dur}")
        self._emit("merged", {"episode": self.episode, "output": str(out_abs),
                              "duration": outcome.duration, "skipped": outcome.skipped,
                              "srt": outcome.srt})
        return outcome

    def _start(self, target, *, needs_provider: bool = True,
               mode: str = "generate") -> None:
        if self.busy:
            raise ConfigError("已有任务在运行，请先暂停/取消或等待完成。")
        control = RunControl()
        control.on_event = self._on_record
        self.control = control
        self._set_state("running")

        def worker():
            try:
                cfg = self._settings.to_app_config()
                provider = self._provider_factory(cfg) if needs_provider else None
                outcome = target(cfg, provider, control)

                if mode == "audition":
                    self._set_state("done")
                    self._emit("finished", {"succeeded": 0, "failed": 0,
                                            "skipped": 0, "cancelled": False,
                                            "merged": None})
                    return

                if mode == "merge":
                    # _do_merge 已发 merged/merge_failed 并设置状态
                    if self.state != "merge_failed":
                        self._set_state("done")
                        self._emit("finished", {
                            "succeeded": 0, "failed": 0, "skipped": 0,
                            "cancelled": False, "merged": True,
                            "episode": self.episode})
                    else:
                        self._emit("finished", {
                            "succeeded": 0, "failed": 0, "skipped": 0,
                            "cancelled": False, "merged": False,
                            "episode": self.episode})
                    return

                # mode == "generate"
                failed = [
                    {"index": r.index,
                     "speaker": r.speaker_name or r.speaker,
                     "error": r.error}
                    for r in outcome.manifest.segments
                    if r.status == "failed"
                ]
                if failed:
                    self._emit("failed", failed)

                # 取消：不拼接；有失败片段：不拼接（run_merge 也会拒绝）
                if outcome.cancelled:
                    self._set_state("cancelled")
                    self._emit("finished", {
                        "succeeded": outcome.succeeded, "failed": outcome.failed,
                        "skipped": outcome.skipped, "total": outcome.total,
                        "cancelled": True, "merged": False, "episode": self.episode})
                    return
                if failed:
                    self._emit("log",
                               "片段未全部成功，保留已生成片段，未进行拼接；"
                               "可在修复后点「重试失败片段」，成功后会自动拼接。")
                    self._set_state("done")
                    self._emit("finished", {
                        "succeeded": outcome.succeeded, "failed": outcome.failed,
                        "skipped": outcome.skipped, "total": outcome.total,
                        "cancelled": False, "merged": False, "episode": self.episode})
                    return

                # 全部片段成功 → 自动拼接；严格守卫确认没有遗留待更新片段
                merge_outcome = self._do_merge(cfg, force=False)
                if merge_outcome is None:
                    # 拼接真正失败（ffmpeg 等），merge_failed 已发
                    self._emit("finished", {
                        "succeeded": outcome.succeeded, "failed": outcome.failed,
                        "skipped": outcome.skipped, "total": outcome.total,
                        "cancelled": False, "merged": False, "episode": self.episode})
                    return
                if merge_outcome is False:
                    # 仍有其他片段未更新（典型：只重生成了一个片段）：
                    # 保留片段、不产出最新整章，不算失败
                    self._emit("finished", {
                        "succeeded": outcome.succeeded, "failed": outcome.failed,
                        "skipped": outcome.skipped, "total": outcome.total,
                        "cancelled": False, "merged": False,
                        "merge_skipped": True, "episode": self.episode})
                    return
                self._set_state("done")
                self._emit("finished", {
                    "succeeded": outcome.succeeded, "failed": outcome.failed,
                    "skipped": outcome.skipped, "total": outcome.total,
                    "cancelled": False,
                    "merged": True,
                    "merge_skipped": bool(merge_outcome and merge_outcome.skipped),
                    "output": str(cfg.projects_dir / self.episode /
                                  merge_outcome.output),
                    "episode": self.episode,
                })
            except Exception as exc:  # noqa: BLE001 —— 工作线程异常转事件
                self._set_state("error")
                self._emit("fatal", f"{type(exc).__name__}: {exc}",
                           traceback.format_exc())

        self._thread = threading.Thread(target=worker, name="podvia-worker",
                                        daemon=True)
        self._thread.start()

    def set_settings(self, settings: GuiSettings) -> None:
        self._settings = settings

    # —— 暂停/恢复/取消 ————————————————————————————————————————————————

    def pause(self) -> None:
        if self.control and self.busy and self.state == "running":
            self.control.pause()
            self._set_state("paused")
            self._emit("log", "已请求暂停：正在合成的请求会继续完成，之后停止提交新片段。")

    def resume(self) -> None:
        if self.control and self.busy and self.state == "paused":
            self.control.resume()
            self._set_state("running")
            self._emit("log", "已恢复。")

    def cancel(self) -> None:
        if self.control and self.busy and self.state in ("running", "paused"):
            self._set_state("cancelling")
            self.control.cancel()
            self._emit("log", "已请求取消：已发出的云端请求无法撤回，将在其完成后停止。")

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    # —— 进度回调（在工作线程触发，只入队） ——————————————————————————————

    def _on_record(self, record) -> None:
        self._emit("progress", record.index, record.status,
                   record.speaker_name or record.speaker,
                   record.error)
