"""批量生成 Pipeline（文档第 11 / 12 / 13 / 14 节）。

片段状态机：pending → generating → success / failed（failed 可被 generate/retry 再次拾起）。

- 线程池按脚本顺序提交，max_workers=cfg.max_concurrency；manifest 更新在主线程
  as_completed 中串行执行 → 无并发写问题
- 断点续生成 = manifest + content_hash：status==success 且 hash 一致且文件存在 → 跳过
- generate_one 永不抛出（内部捕获一切并记入 manifest）
- PAUSE 段不进入 TTS 队列（manifest 初始化时已记 success，合并阶段处理）
- run_retry：只依赖 manifest 重新生成 failed/pending 台词（文档 17 retry 命令）
- RunControl（阶段 6 GUI）：暂停时不再提交新片段、取消时停止排队；
  已发出的云端 HTTP 请求无法撤回（官方无取消接口），会自然完成并如实记账。
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

from podcast_tts import log
from podcast_tts import capabilities as caps
from podcast_tts import emotags
from podcast_tts.capabilities import resolve_input_extras
from podcast_tts.config import AppConfig, ConfigError
from podcast_tts.project import (
    WorkingScript,
    apply_overrides,
    parse_working,
    sync_records_from_segments,
)
from podcast_tts.audio import MergeError, run_merge, segments_signature
from podcast_tts.ssml import SSMLError, validate_ssml
from podcast_tts.manifest import (
    MAX_TEXT_LEN,
    Manifest,
    cleanup_orphan_audio,
    content_hash,
    fresh_episode_state,
    load as load_manifest,
    manifest_path,
    reconcile_segments,
    save,
    sync_manifest,
)
from podcast_tts.models import Segment, SegmentRecord, TTSRequest, TTSResult
from podcast_tts.providers import get_provider
from podcast_tts.providers.base import TTSProvider
from podcast_tts.retry import retry_with_backoff


@dataclass
class GenerateOutcome:
    """一轮生成结果摘要（main.py 据此决定退出码）。"""

    total: int
    speech_total: int
    skipped: int
    succeeded: int
    failed: int
    manifest: Manifest
    cancelled: bool = False  # 用户取消（已在途请求仍会跑完，未提交片段保持原状）

    @property
    def exit_code(self) -> int:
        return 4 if self.failed else 0


@dataclass
class RunControl:
    """GUI 协作控制（阶段 6）。

    pause_event：置位后暂停提交新片段（已在途请求继续跑完，无法撤回）。
    cancel_event：置位后取消运行：在途请求跑完即停止，未提交片段不改状态。
    on_event：每完成一个片段回调一次（在工作线程被调用，GUI 必须经队列转发）。
    """

    pause_event: threading.Event = field(default_factory=threading.Event)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    on_event: object = None  # Callable[[SegmentRecord], None] | None
    cancelled: bool = False

    def pause(self) -> None:
        self.pause_event.set()

    def resume(self) -> None:
        self.pause_event.clear()

    def cancel(self) -> None:
        self.cancel_event.set()
        self.pause_event.clear()  # 取消不能被暂停挡住


def validate_emotags(segments: list[Segment], cfg: AppConfig) -> None:
    """生成前整任务标签校验：任何错误一次性抛出，绝不先生成部分片段再报错。

    校验项（规则见 podcast_tts.emotags）：
    - 标签拼写（疑似写错的方括号标签）；
    - 当前 provider/模型是否官方确认支持标签（未确认不静默透传）；
    - 情绪/拟声标签与 SSML 的同片段组合（官方未说明，不支持）。
    """
    caps_obj = caps.get_capabilities(cfg.model) if cfg.provider == "qwen_audio" else None
    supports = None if caps_obj is None else caps_obj.supports_emotags
    errors: list[str] = []
    for seg in segments:
        if seg.kind != "speech" or not seg.text:
            continue
        where = f"片段 {seg.index:04d}（{speaker_label(seg)}）"
        errors.extend(emotags.check_text_tags(seg.text, where=where))
        errors.extend(emotags.check_ssml_combo(seg.text, where=where))
        errors.extend(
            emotags.check_model_support(
                seg.text, cfg.provider, cfg.model, where=where, supports=supports
            )
        )
    if errors:
        raise ConfigError("情绪/拟声标签校验未通过：\n" + "\n".join(errors))


def _speaker_map(cfg: AppConfig) -> dict[str, str]:
    """{角色显示名: 稳定ID}；未设 display_name（CLI 配置）时显示名即 ID。"""
    return {(s.display_name or sid): sid for sid, s in cfg.voices.items()}


def resolve_speaker_id(
    cfg: AppConfig, speaker: str | None, speaker_name: str | None = None
) -> str | None:
    """把记录/片段中的说话人标识解析为当前配置里的稳定 ID。

    兼容旧 manifest：旧记录按显示名保存 speaker 时，用 display_name 反查 ID。
    """
    if speaker in cfg.voices:
        return speaker
    for sid, setting in cfg.voices.items():
        if setting.display_name and (
            setting.display_name == speaker
            or (speaker_name and setting.display_name == speaker_name)
        ):
            return sid
    return None


def speaker_label(seg: Segment | SegmentRecord) -> str:
    """日志/GUI 显示用的角色名（优先显示名，可中文）。"""
    return seg.speaker_name or seg.speaker or ""


def run_generate(
    script_path: Path,
    cfg: AppConfig,
    *,
    force: bool = False,
    no_resume: bool = False,
    provider: TTSProvider | None = None,
    control: RunControl | None = None,
) -> GenerateOutcome:
    """CLI 脚本入口：文件内容包装为内存 WorkingScript（不落工作脚本文件），
    走与 GUI 完全相同的 run_project 核心。"""
    provider = provider or get_provider(cfg)
    log.info(f"解析脚本：{script_path}")
    ws = WorkingScript(
        episode=script_path.stem, mode="script",
        text=script_path.read_text(encoding="utf-8-sig"),
        source_type="file", source_name=script_path.name,
    )
    return run_project(
        script_path.stem, cfg, ws,
        force=force, no_resume=no_resume, provider=provider, control=control,
        persist=False, script_label=_rel_label(script_path),
        kind_label="脚本",
    )


def run_book(
    book_path: Path,
    cfg: AppConfig,
    *,
    force: bool = False,
    no_resume: bool = False,
    default_speaker: str | None = None,
    provider: TTSProvider | None = None,
    control: RunControl | None = None,
) -> GenerateOutcome:
    """CLI 章节入口：同样汇入 run_project（章节模式解析）。

    CLI 同文件二次运行走 reconcile 增量缓存（fresh=False），
    不等于 GUI 的"重新导入新文件"。
    """
    provider = provider or get_provider(cfg)
    log.info(f"导入章节文件：{book_path}")
    ws = WorkingScript.import_file(book_path.stem, book_path, mode="book", fresh=False)
    return run_project(
        book_path.stem, cfg, ws,
        force=force, no_resume=no_resume,
        default_speaker_id=default_speaker,
        provider=provider, control=control,
        persist=False, script_label=_rel_label(book_path),
        kind_label="章节文件",
    )


def _rel_label(path: Path) -> str:
    from podcast_tts.config import ROOT_DIR
    import os
    try:
        return Path(os.path.relpath(path, ROOT_DIR)).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _reconcile_current(
    episode: str,
    cfg: AppConfig,
    ws: WorkingScript,
    *,
    default_speaker_id: str | None = None,
    script_label: str | None = None,
    use_fresh: bool = False,
) -> tuple[Path, list[Segment], Manifest]:
    """当前工作状态的唯一规划内核：parse → reconcile → 覆盖层 → 记录同步。

    不落盘、不触网。所有生成/预览/状态检查入口都经过这里，保证数据源唯一。
    副作用：给新片段分配 seg_id；latest-edit-wins 冲突时就地清理 ws 失效覆盖。
    """
    ws.episode = episode
    episode_dir = cfg.projects_dir / episode
    if use_fresh:
        # 重新导入：丢弃上一份工作状态（旧 manifest + 旧片段音频），output/ 保留
        fresh_episode_state(episode_dir)
    segments = parse_working(ws, cfg, default_speaker_id=default_speaker_id)
    manifest = sync_manifest(
        episode_dir,
        script_label=script_label or f"projects/{episode}/working_script.json",
        script_text=ws.text, cfg=cfg, segments=segments, fresh=use_fresh,
    )
    apply_overrides(segments, ws, cfg)
    sync_records_from_segments(manifest.segments, segments)
    return episode_dir, segments, manifest


def plan_and_persist(
    episode: str,
    cfg: AppConfig,
    ws: WorkingScript,
    *,
    default_speaker_id: str | None = None,
    script_label: str | None = None,
    persist: bool = True,
    fresh: bool = False,
) -> tuple[list[Segment], Manifest]:
    """把当前工作脚本规划为 Segment[] + manifest 并落盘（不触网、不生成）。

    fresh=True（或 ws.fresh=True）：重新导入，忽略旧 manifest，所有片段全新。
    fresh=False（默认）：编辑当前脚本，reconcile 沿用旧 seg_id 与音频。
    规划后 ws.fresh 复位为 False：同一个工作脚本后续编辑走 reconcile。

    persist=True 时另做 orphan 清理（只删 segments/ 中无 manifest 引用的受管音频，
    output/ 成品绝不触碰）；写入只新增/更新记录，不改 status、不清音频归属。
    """
    use_fresh = fresh or ws.fresh
    episode_dir, segments, manifest = _reconcile_current(
        episode, cfg, ws, default_speaker_id=default_speaker_id,
        script_label=script_label, use_fresh=use_fresh,
    )
    ws.prune_overrides({s.seg_id for s in segments})
    ws.fresh = False  # 首次规划后不再是"全新导入"，后续编辑应 reconcile
    if persist:
        ws.save(cfg.projects_dir)
        referenced = {r.file for r in manifest.segments if r.file}
        removed = cleanup_orphan_audio(episode_dir, referenced)
        if removed:
            log.info(f"已清理 {len(removed)} 个无引用的旧片段音频：{removed}")
    save(episode_dir, manifest)
    return segments, manifest


# —— 统一用户可见状态（全 GUI 唯一判定入口） ————————————————————————————————

SEG_STATE_PAUSE = "pause"      # 停顿（不调 TTS）
SEG_STATE_MISSING = "missing"  # 未生成
SEG_STATE_FAILED = "failed"    # 生成失败
SEG_STATE_DIRTY = "dirty"      # 待更新（有旧音频，但当前请求已变化）
SEG_STATE_CLEAN = "clean"      # 已生成（请求与音频一致）
SEG_STATE_GENERATING = "generating"  # 生成中（运行态叠加，不持久化）

SEG_STATE_ZH = {
    SEG_STATE_PAUSE: "停顿",
    SEG_STATE_MISSING: "未生成",
    SEG_STATE_FAILED: "生成失败",
    SEG_STATE_DIRTY: "待更新",
    SEG_STATE_CLEAN: "已生成",
    SEG_STATE_GENERATING: "生成中",
}

# 状态 → 本次生成动作（预览/确认用）
SEG_ACTION_ZH = {
    SEG_STATE_PAUSE: "不调用 TTS",
    SEG_STATE_MISSING: "将生成",
    SEG_STATE_FAILED: "将重试",
    SEG_STATE_DIRTY: "将重新生成",
    SEG_STATE_CLEAN: "跳过（已是最新）",
    SEG_STATE_GENERATING: "生成中…",
}


def classify_segment_state(
    record: SegmentRecord | None, seg: Segment, cfg: AppConfig, episode_dir: Path
) -> str:
    """统一片段状态判定：所有 GUI 状态/按钮/合并决策都使用本函数。

    依据 content_hash（请求指纹）+ 文件存在性，绝不只看 manifest.status：
    - 无记录/pending        → 未生成
    - failed                → 生成失败
    - success 且 hash 一致且文件存在 → 已生成
    - success 但 hash 不一致/文件缺失 → 待更新
    - pause                 → 停顿
    """
    if seg.kind == "pause":
        return SEG_STATE_PAUSE
    if record is None or record.status == "pending":
        return SEG_STATE_MISSING
    if record.status == "failed":
        return SEG_STATE_FAILED
    if _is_cached(record, seg, cfg, episode_dir):
        return SEG_STATE_CLEAN
    return SEG_STATE_DIRTY


def inspect_segments(
    episode: str,
    cfg: AppConfig,
    ws: WorkingScript,
    *,
    default_speaker_id: str | None = None,
    persist: bool = False,
    fresh: bool = False,
) -> tuple[list[dict], Manifest]:
    """规划并返回当前每个片段的"状态 + 最终 TTS 请求规划"行（不触网）。

    persist=True：先落盘（新片段获得稳定 seg_id、orphan 清理）；
    persist=False：纯只读检查，不改变任何生成状态（生成预览/状态概览用）。

    每行含：seg_id/index/kind/角色/text/direction/rate/pitch/volume/
    state（内部态）/state_zh/action_zh/file/error/instruction/voice/
    expected_hash（当前请求指纹）/origin/tags/injected/raw_text/pause_seconds。
    """
    if persist or fresh or ws.fresh:
        segments, manifest = plan_and_persist(
            episode, cfg, ws, default_speaker_id=default_speaker_id,
            persist=persist, fresh=fresh,
        )
        episode_dir = cfg.projects_dir / episode
    else:
        episode_dir, segments, manifest = _reconcile_current(
            episode, cfg, ws, default_speaker_id=default_speaker_id,
        )
    by_id = {r.seg_id: r for r in manifest.segments if r.seg_id}
    rows: list[dict] = []
    for seg in segments:
        rec = by_id.get(seg.seg_id)
        state = classify_segment_state(rec, seg, cfg, episode_dir)
        row: dict = {
            "seg_id": seg.seg_id, "index": seg.index, "kind": seg.kind,
            "speaker": seg.speaker, "speaker_name": seg.speaker_name or seg.speaker,
            "text": seg.text, "raw_text": seg.raw_text,
            "injected": seg.injected_prefix, "origin": seg.origin,
            "pause_seconds": seg.pause_seconds,
            "direction": seg.direction,
            "chapter_direction": seg.chapter_direction,
            "rate": seg.rate, "pitch": seg.pitch, "volume": seg.volume,
            "state": state, "state_zh": SEG_STATE_ZH[state],
            "action_zh": SEG_ACTION_ZH[state],
            "file": (rec.file if rec else None),
            "error": (rec.error if rec else None),
            "instruction": None, "voice": None, "language_type": None,
            "expected_hash": None,
        }
        if seg.kind == "speech":
            tags = []
            try:
                from podcast_tts import emotags
                tags = [t[2] for t in emotags.iter_tags(seg.text or "")]
            except Exception:  # noqa: BLE001 — 标签仅展示，不影响状态
                pass
            row["tags"] = " ".join(dict.fromkeys(tags))
            try:
                setting, instruction, extras = plan_segment(cfg, seg)
                row["instruction"] = instruction
                row["voice"] = setting.voice
                row["language_type"] = setting.language_type
                row["rate"] = extras.get("rate", seg.rate)
                row["pitch"] = extras.get("pitch", seg.pitch)
                row["volume"] = extras.get("volume", seg.volume)
                row["expected_hash"] = content_hash(
                    cfg, seg.text, setting.voice, setting.language_type,
                    instruction, extras,
                )
                row["audio_format"] = str(extras.get("format", "wav")).lower()
            except ConfigError as exc:
                row["error"] = str(exc)
        else:
            row["tags"] = ""
            row["audio_format"] = "wav"
        rows.append(row)
    return rows, manifest


def segment_states(
    episode: str,
    cfg: AppConfig,
    ws: WorkingScript,
    *,
    default_speaker_id: str | None = None,
) -> list[dict]:
    """兼容包装：纯本地（不落盘）返回片段状态行。新代码请用 inspect_segments。"""
    rows, _manifest = inspect_segments(
        episode, cfg, ws, default_speaker_id=default_speaker_id, persist=False)
    return rows


# —— 整章成品 freshness —————————————————————————————————————————————————————

def chapter_freshness(
    episode: str,
    cfg: AppConfig,
    ws: WorkingScript,
    *,
    default_speaker_id: str | None = None,
) -> dict:
    """整章成品状态：missing（未生成）/ fresh（最新）/ stale（待更新）。

    - 存在 missing/failed/dirty 台词片段 → stale；
    - PAUSE 时长/片段顺序/文件变化 → segments_signature 变化 → stale（不需 TTS）；
    - 所有台词 clean 且签名与上次合并一致且成品文件存在 → fresh。
    """
    rows, manifest = inspect_segments(
        episode, cfg, ws, default_speaker_id=default_speaker_id, persist=False)
    counts = {"clean": 0, "dirty": 0, "missing": 0, "failed": 0, "pause": 0}
    for r in rows:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    episode_dir = cfg.projects_dir / episode
    out_rel = f"output/{episode}.{cfg.merge_format}"
    out_path = episode_dir / out_rel
    blockers = counts["dirty"] + counts["missing"] + counts["failed"]
    if not out_path.is_file():
        state = "missing"
    elif blockers > 0:
        state = "stale"
    else:
        sig = segments_signature(episode_dir, manifest)
        state = "fresh" if (
            manifest.merge.get("status") == "success"
            and manifest.merge.get("segments_signature") == sig
        ) else "stale"
    return {
        "state": state,
        "state_zh": {"missing": "未生成", "fresh": "最新", "stale": "待更新"}[state],
        "counts": counts, "total": len(rows),
        "speech_total": counts["clean"] + counts["dirty"]
        + counts["missing"] + counts["failed"],
        "output": out_rel, "output_exists": out_path.is_file(),
    }


def all_speech_clean(rows: list[dict]) -> bool:
    """所有台词片段都已生成（clean）才允许产出"最新整章"。"""
    return all(
        r["kind"] == "pause" or r["state"] == SEG_STATE_CLEAN for r in rows
    )


def guarded_run_merge(
    episode: str,
    cfg: AppConfig,
    ws: WorkingScript,
    *,
    default_speaker_id: str | None = None,
    force: bool = False,
):
    """普通用户流程的唯一合并入口。

    force=False：任何台词片段不是"已生成"都拒绝合并——绝不用旧音频冒充最新成品
    （PAUSE 时长变化不阻止：停顿无 TTS，签名变化会触发重新合并）。
    force=True：高级"强制合并"，允许使用现有片段文件，但调用方必须明确告知用户
    结果可能不是最新。
    """
    rows, _manifest = inspect_segments(
        episode, cfg, ws, default_speaker_id=default_speaker_id, persist=False)
    expected_hashes: dict[str, str] = {}
    if not force:
        blockers = [
            f"{r['index']:04d} {r['speaker_name']}（{SEG_STATE_ZH[r['state']]}）"
            for r in rows
            if r["kind"] == "speech" and r["state"] != SEG_STATE_CLEAN
        ]
        if blockers:
            raise MergeError(
                f"存在 {len(blockers)} 个未更新片段，暂不能生成最新整章："
                + "、".join(blockers)
                + "。请先点「生成 / 更新音频」或在片段管理中生成这些片段。"
            )
        for r in rows:
            if r["kind"] == "speech" and r.get("expected_hash"):
                expected_hashes[r["seg_id"]] = r["expected_hash"]
    return run_merge(
        episode, cfg, force=force,
        expected_hashes=expected_hashes or None,
    )


def source_snapshot_for_edit(
    episode: str,
    cfg: AppConfig,
    ws: WorkingScript,
    seg_id: str,
    fields: set[str],
    *,
    default_speaker_id: str | None = None,
) -> dict:
    """片段编辑保存前，取这些字段"当前源脚本（未叠加覆盖）"的值作为 base 快照。

    用于 latest-edit-wins：之后主文本框改了同字段，本次覆盖自动失效。
    """
    segments = parse_working(ws, cfg, default_speaker_id=default_speaker_id)
    episode_dir = cfg.projects_dir / episode
    existing = load_manifest(episode_dir)
    if existing is not None:
        reconcile_segments(existing.segments, segments)
    else:
        reconcile_segments(None, segments)
    target = next((s for s in segments if s.seg_id == seg_id), None)
    if target is None:
        return {}
    attr_map = {
        "text": "text", "speaker": "speaker", "direction": "direction",
        "rate": "rate", "pitch": "pitch", "volume": "volume",
        "pause_seconds": "pause_seconds",
    }
    return {f: getattr(target, attr_map[f], None) for f in fields if f in attr_map}


def run_project(
    episode: str,
    cfg: AppConfig,
    ws: WorkingScript,
    *,
    default_speaker_id: str | None = None,
    only_seg_ids: set[str] | list[str] | None = None,
    force: bool = False,
    no_resume: bool = False,
    provider: TTSProvider | None = None,
    control: RunControl | None = None,
    persist: bool = True,
    script_label: str | None = None,
    kind_label: str = "工作脚本",
    fresh: bool = False,
) -> GenerateOutcome:
    """统一生成核心（GUI/CLI、粘贴/导入共用）。

    fresh=True 仅用于"重新导入新文件"场景：丢弃旧工作状态，全部未生成。
    """
    provider = provider or get_provider(cfg)
    episode_dir = cfg.projects_dir / episode

    segments, manifest = plan_and_persist(
        episode, cfg, ws, default_speaker_id=default_speaker_id,
        script_label=script_label, persist=persist, fresh=fresh)
    speech_total = sum(1 for s in segments if s.kind == "speech")
    log.info(
        f"{kind_label}共 {len(segments)} 段"
        f"（{speech_total} 台词 / {len(segments) - speech_total} 停顿）"
    )

    # 选择本轮要生成的片段
    todo: list[Segment] = []
    skipped = 0
    if only_seg_ids is not None:
        wanted = set(only_seg_ids)
        by_id = {s.seg_id: s for s in segments}
        missing = sorted(i for i in wanted if i not in by_id)
        if missing:
            raise ConfigError(f"片段 ID 不存在：{missing}")
        for sid in wanted:
            seg = by_id[sid]
            if seg.kind != "speech":
                raise ConfigError(f"片段 {seg.index:04d} 是停顿，不能调用 TTS。")
            todo.append(seg)
        # 单段重生成只校验该段，避免被其他无关坏行拦截
        validate_emotags(todo, cfg)
        log.info(f"指定重生成 {len(todo)} 段（忽略缓存，按当前工作脚本与配置）")
    else:
        validate_emotags(segments, cfg)  # 整任务先校验标签，通过后才进入生成
        for seg in segments:
            if seg.kind != "speech":
                continue
            if not (force or no_resume) and _is_cached(
                manifest.record(seg.index), seg, cfg, episode_dir
            ):
                skipped += 1
                continue
            todo.append(seg)
        log.info(f"待生成 {len(todo)}，跳过（缓存命中）{skipped}")

    succeeded, failed, cancelled = _generate_batch(
        todo, manifest, episode_dir, cfg, provider, control=control
    )

    log.info(
        f"本轮完成：成功 {succeeded}，失败 {failed}，缓存跳过 {skipped}"
        + ("（已取消）" if cancelled else "")
    )
    if failed:
        failed_list = "、".join(
            f"{r.index:04d} {speaker_label(r)}"
            for r in manifest.segments if r.status == "failed"
        )
        log.warning(f"存在失败片段：{failed_list}。可运行 retry/regen 命令重试。")
    return GenerateOutcome(
        total=len(segments), speech_total=speech_total,
        skipped=skipped, succeeded=succeeded, failed=failed, manifest=manifest,
        cancelled=cancelled,
    )


def run_regenerate(
    episode: str,
    indices: list[int] | set[int] | tuple[int, ...],
    cfg: AppConfig,
    *,
    seg_ids: set[str] | list[str] | None = None,
    ws: WorkingScript | None = None,
    provider: TTSProvider | None = None,
    control: RunControl | None = None,
) -> GenerateOutcome:
    """局部重生成。

    - seg_ids + ws：按工作脚本最新内容重生成指定片段（GUI 单句编辑后的路径）；
    - indices：兼容 CLI/旧契约，文本与参数以 manifest 记录为准，不重新解析。
    """
    if seg_ids is not None or ws is not None:
        if ws is None:
            raise ConfigError("按 seg_id 重生成必须提供 WorkingScript。")
        provider = provider or get_provider(cfg)
        return run_project(
            episode, cfg, ws, only_seg_ids=seg_ids,
            provider=provider, control=control, persist=True,
        )

    episode_dir = cfg.projects_dir / episode
    manifest = load_manifest(episode_dir)
    if manifest is None:
        raise FileNotFoundError(
            f"未找到 {manifest_path(episode_dir)}（episode 可能尚未生成，"
            "先运行 generate/book 命令）。"
        )
    provider = provider or get_provider(cfg)
    wanted = set(indices)

    todo: list[Segment] = []
    missing: list[int] = []
    for idx in sorted(wanted):
        rec = manifest.record(idx)
        if rec is None or rec.kind != "speech":
            missing.append(idx)
            continue
        sid = resolve_speaker_id(cfg, rec.speaker, rec.speaker_name)
        if sid is None:
            raise ConfigError(
                f"config.yaml 中没有角色 {rec.speaker!r} 的音色配置"
                f"（manifest 片段 {rec.index:04d}）。请补全 voices 后再局部重生成。"
            )
        todo.append(Segment(
            index=rec.index, kind="speech", seg_id=rec.seg_id, speaker=sid,
            speaker_name=rec.speaker_name,
            direction=rec.direction, chapter_direction=rec.chapter_direction,
            rate=rec.rate, pitch=rec.pitch, volume=rec.volume,
            text=rec.text,
            chapter=rec.chapter, source_file=rec.source_file,
            char_start=rec.char_start, char_end=rec.char_end,
        ))
    if missing:
        raise ConfigError(
            f"以下片段编号不存在或不是台词片段：{missing}（可用 status 命令查看编号）。"
        )
    if not todo:
        raise ConfigError("没有指定任何可重生成的台词片段。")

    validate_emotags(todo, cfg)  # 模型切换后旧文本可能含不支持标签，重新门控
    log.info(f"局部重生成 {len(todo)} 段：{sorted(wanted)}（忽略缓存，按当前配置）")
    succeeded, failed, cancelled = _generate_batch(
        todo, manifest, episode_dir, cfg, provider, control=control
    )
    log.info(f"局部重生成完成：成功 {succeeded}，失败 {failed}")
    return GenerateOutcome(
        total=len(manifest.segments),
        speech_total=sum(1 for r in manifest.segments if r.kind == "speech"),
        skipped=0, succeeded=succeeded, failed=failed, manifest=manifest,
        cancelled=cancelled,
    )


def run_retry(
    episode: str,
    cfg: AppConfig,
    *,
    provider: TTSProvider | None = None,
    control: RunControl | None = None,
) -> GenerateOutcome:
    """重新生成 manifest 中 failed/pending 的台词片段（文档 17 retry 命令）。

    只依赖 manifest，不重新解析脚本：文本/导演指令以 manifest 记录为准。
    manifest 不存在抛 FileNotFoundError；角色不在当前 config 抛 ConfigError。
    """
    episode_dir = cfg.projects_dir / episode
    manifest = load_manifest(episode_dir)
    if manifest is None:
        raise FileNotFoundError(
            f"未找到 {manifest_path(episode_dir)}（episode 可能尚未生成，"
            "先运行 generate 命令）。"
        )
    provider = provider or get_provider(cfg)

    todo: list[Segment] = []
    failed_n = pending_n = 0
    for rec in manifest.segments:
        if rec.kind != "speech" or rec.status not in ("failed", "pending"):
            continue
        if rec.status == "failed":
            failed_n += 1
        else:
            pending_n += 1
        sid = resolve_speaker_id(cfg, rec.speaker, rec.speaker_name)
        if sid is None:
            raise ConfigError(
                f"config.yaml 中没有角色 {rec.speaker!r} 的音色配置"
                f"（manifest 片段 {rec.index:04d}）。请补全 voices 后重试。"
            )
        todo.append(
            Segment(
                index=rec.index, kind="speech", seg_id=rec.seg_id, speaker=sid,
                speaker_name=rec.speaker_name,
                direction=rec.direction, chapter_direction=rec.chapter_direction,
                rate=rec.rate, pitch=rec.pitch, volume=rec.volume,
                text=rec.text,
                chapter=rec.chapter, source_file=rec.source_file,
                char_start=rec.char_start, char_end=rec.char_end,
            )
        )

    if not todo:
        log.info("没有需要重试的片段（无 failed/pending 台词）。")
        return GenerateOutcome(
            total=len(manifest.segments),
            speech_total=sum(1 for r in manifest.segments if r.kind == "speech"),
            skipped=0, succeeded=0, failed=0, manifest=manifest,
        )

    log.info(f"待重试 {len(todo)}（失败 {failed_n} / 待处理 {pending_n}）")
    validate_emotags(todo, cfg)  # 模型切换后旧文本可能含不支持标签，重新门控
    succeeded, failed, cancelled = _generate_batch(
        todo, manifest, episode_dir, cfg, provider, control=control
    )
    log.info(f"重试完成：成功 {succeeded}，失败 {failed}")
    if failed:
        failed_list = "、".join(
            f"{r.index:04d} {speaker_label(r)}"
            for r in manifest.segments if r.status == "failed"
        )
        log.warning(f"仍有失败片段：{failed_list}。可再次运行 retry 命令。")
    return GenerateOutcome(
        total=len(manifest.segments),
        speech_total=sum(1 for r in manifest.segments if r.kind == "speech"),
        skipped=0, succeeded=succeeded, failed=failed, manifest=manifest,
        cancelled=cancelled,
    )


def _generate_batch(
    todo: list[Segment],
    manifest: Manifest,
    episode_dir: Path,
    cfg: AppConfig,
    provider: TTSProvider,
    *,
    control: RunControl | None = None,
) -> tuple[int, int, bool]:
    """并发生成 todo 中的片段，逐段更新并落盘 manifest。

    返回（成功数, 失败数, 是否被取消）。提交是门控的：
    - 暂停（control.pause_event 置位）→ 不再提交新片段，在途请求自然跑完；
    - 取消（cancel_event 置位）→ 停止提交，在途跑完后退出，未提交片段保持原状。
    manifest 写入始终在本调用线程串行执行，无并发写。
    """
    succeeded = failed = 0
    if not todo:
        return succeeded, failed, False

    pending = list(todo)
    in_flight = {}
    cancelled = False
    with ThreadPoolExecutor(max_workers=cfg.max_concurrency) as pool:
        while pending or in_flight:
            # 按并发上限补位；暂停时等待；取消时停止补位
            while pending and len(in_flight) < cfg.max_concurrency:
                if control is not None and control.cancel_event.is_set():
                    cancelled = True
                    control.cancelled = True
                    pending.clear()
                    break
                if control is not None and control.pause_event.is_set():
                    time.sleep(0.1)
                    continue
                seg = pending.pop(0)
                fut = pool.submit(
                    _generate_one, seg, manifest.record(seg.index), cfg,
                    episode_dir, provider, manifest
                )
                in_flight[fut] = seg

            if not in_flight:
                break  # 取消且无在途请求

            done, _ = wait(
                in_flight, timeout=0.1, return_when="FIRST_COMPLETED"
            )
            for fut in done:
                seg = in_flight.pop(fut)
                record = fut.result()  # 内部已含重试，永不抛出
                manifest.replace(record)
                save(episode_dir, manifest)  # 本线程串行原子写
                if record.status == "success":
                    succeeded += 1
                else:
                    failed += 1
                if control is not None and control.on_event is not None:
                    try:
                        control.on_event(record)
                    except Exception as exc:  # noqa: BLE001 —— 回调故障不影响生成
                        log.debug(f"GUI 进度回调异常（已忽略）：{exc}")
    return succeeded, failed, cancelled


def _resolve_extras(cfg: AppConfig, setting) -> dict:
    """解析"项目默认级"音频附加参数（不含角色/行内覆盖）。

    audio_params 是 qwen_audio provider 专属；provider=qwen（Qwen-TTS 端点）
    不支持这些 input 字段，返回空 dict（其适配器也不读取 audio_params）。
    SSML 开关：角色级 setting.ssml 优先，否则跟随全局 enable_ssml。
    """
    if cfg.provider != "qwen_audio":
        return {}
    ssml_on = setting.ssml if setting.ssml is not None else cfg.audio_params.enable_ssml
    extras = resolve_input_extras(
        cfg.audio_params, cfg.model,
        language_type=setting.language_type, language_hint=setting.language_hint,
    )
    if ssml_on:
        extras["enable_ssml"] = True
    return extras


def final_instruction(cfg: AppConfig, setting, seg: Segment) -> str | None:
    """片段最终生效的导演指令——四级层级，后者覆盖前者，只取一条，绝不拼接：

        项目级 cfg.default_instruction
        < 角色默认级 setting.instruction
        < 章节级 seg.chapter_direction
        < 行内片段级 seg.direction
    """
    return (
        seg.direction
        or seg.chapter_direction
        or setting.instruction
        or cfg.default_instruction
    )


_PLANNED_NUMERIC = ("rate", "pitch", "volume")


def plan_segment(
    cfg: AppConfig, seg: Segment, *, snapshot: dict | None = None
) -> tuple[object, str | None, dict]:
    """计算一个片段最终生效的 (角色配置, instruction, 音频参数)。

    - instruction：四级层级取唯一一条（见 final_instruction）；
    - 音频参数：项目默认（_resolve_extras）→ 角色默认 → 行内片段，后者非空即覆盖；
    - snapshot（retry 失败段）存在时音频参数完全沿用失败时快照，指令仍按层级
      重算（确定性结果，与失败时一致）；
    - 角色/行内显式指定了 rate/pitch/volume，但当前 provider/模型不支持时
      抛 ConfigError——不静默忽略（项目级参数的模型不支持由能力矩阵 + WARNING 处理）。

    生成链路与 GUI 生成预览共用本函数，保证"所见即所发"。
    """
    sid = resolve_speaker_id(cfg, seg.speaker, seg.speaker_name)
    if sid is None:
        raise ConfigError(
            f"片段 {seg.index:04d} 的角色 {speaker_label(seg)!r} 不在当前角色配置中；"
            "请在角色表补全该角色后再生成。"
        )
    setting = cfg.voices[sid]
    instruction = final_instruction(cfg, setting, seg)

    if snapshot is not None:
        return setting, instruction, dict(snapshot)

    extras = _resolve_extras(cfg, setting)
    display = seg.speaker_name or setting.display_name or sid
    for key in _PLANNED_NUMERIC:
        inline_val = getattr(seg, key)
        val = inline_val if inline_val is not None else getattr(setting, key)
        if val is None:
            continue  # 未覆盖：沿用项目默认（已在 extras 中）或不发送
        if cfg.provider != "qwen_audio":
            raise ConfigError(
                f"角色 {display!r} 设置了 {key}={val}，但当前接口 {cfg.provider} "
                "不支持语速/音调/音量参数；请去掉该设置或改用 qwen-audio-tts 模型。"
            )
        mc = caps.get_capabilities(cfg.model)
        if mc is None:
            raise ConfigError(
                f"角色 {display!r} 设置了 {key}={val}，但模型 {cfg.model!r} 不在已核实"
                "能力矩阵中，无法确认其支持该参数；请改用已核实模型，或去掉该设置"
                "（不会静默忽略后继续生成）。"
            )
        if key not in mc.supports_optional:
            raise ConfigError(
                f"模型 {cfg.model} 不支持参数 {key}（角色 {display!r} 设置了 "
                f"{key}={val}）；请去掉该设置或更换模型。"
            )
        extras[key] = val
    return setting, instruction, extras


def _record_base(seg: Segment) -> dict:
    """Segment → SegmentRecord 共通字段（保证层级字段随记录持久化）。"""
    return dict(
        seg_id=seg.seg_id, index=seg.index, kind="speech", speaker=seg.speaker,
        speaker_name=seg.speaker_name,
        direction=seg.direction, chapter_direction=seg.chapter_direction,
        rate=seg.rate, pitch=seg.pitch, volume=seg.volume, text=seg.text,
        chapter=seg.chapter, source_file=seg.source_file,
        char_start=seg.char_start, char_end=seg.char_end,
    )


def _target_file(
    manifest: Manifest, record: SegmentRecord | None, seg: Segment, ext: str
) -> str:
    """决定片段音频落盘相对路径。

    - 已有同扩展名文件（含编辑后重生成、序号因插入而重排）：沿用，保证该
      seg_id 的音频归属稳定，不与其他片段撞文件；
    - 新片段：segments/序号_角色.ext；若该名已被其他记录占用，加 seg_id 短后缀。
    """
    if record is not None and record.file and record.file.endswith(f".{ext}"):
        return record.file
    candidate = f"segments/{seg.index:04d}_{seg.speaker}.{ext}"
    owned = {r.file for r in manifest.segments
             if r is not record and r.file and r.seg_id != seg.seg_id}
    if candidate in owned:
        suffix = (seg.seg_id or "")[:6]
        candidate = f"segments/{seg.index:04d}_{seg.speaker}_{suffix}.{ext}"
    return candidate


def _is_cached(
    record: SegmentRecord | None, seg: Segment, cfg: AppConfig, episode_dir: Path
) -> bool:
    """缓存命中判定（文档 14.1）：success + hash 一致 + 文件存在且非空。"""
    if record is None or record.status != "success":
        return False
    try:
        setting, instruction, extras = plan_segment(cfg, seg)
    except ConfigError:
        return False  # 规划失败的片段交给生成阶段如实记账
    expected = content_hash(
        cfg, seg.text, setting.voice, setting.language_type, instruction, extras
    )
    if record.content_hash != expected:
        return False
    if not record.file:
        return False
    audio_file = episode_dir / record.file
    try:
        return audio_file.is_file() and audio_file.stat().st_size > 0
    except OSError:
        return False


def _generate_one(
    seg: Segment,
    record: SegmentRecord | None,
    cfg: AppConfig,
    episode_dir: Path,
    provider: TTSProvider,
    manifest: Manifest | None = None,
) -> SegmentRecord:
    """生成单个片段：规划最终指令/参数 → hash → 重试调用 → 返回记录。永不抛出。"""
    label = speaker_label(seg)

    # 音频参数来源：
    # - failed 记录（retry/续跑在途片段）→ 以 manifest 快照为准，配置随后再改不影响它；
    # - 其他（新片段、或因配置变化 hash 不一致而需重新生成的 success 记录）
    #   → 一律按当前配置规划，否则改参数不会真正生效。
    snapshot = None
    if record is not None and record.status == "failed" and record.audio_params:
        snapshot = record.audio_params
    try:
        setting, instruction, extras = plan_segment(cfg, seg, snapshot=snapshot)
    except ConfigError as exc:
        log.error(f"{seg.index:04d} {label} 配置错误，未调用 API：{exc}")
        return SegmentRecord(
            **_record_base(seg),
            status="failed",
            file=record.file if record is not None else None,
            attempts=(record.attempts if record is not None else 0) + 1,
            error=str(exc),
            audio_params={},
        )

    # 片段扩展名必须与实际请求的 format 一致；PAUSE 始终是 WAV（见 manifest 初始化）
    ext = str(extras.get("format", "wav")).lower()
    if manifest is not None:
        file_rel = _target_file(manifest, record, seg, ext)
    elif record is not None and record.status == "failed" and record.file \
            and record.file.endswith(f".{ext}"):
        file_rel = record.file  # 兜底：retry 沿用同格式的失败记录路径
    else:
        file_rel = f"segments/{seg.index:04d}_{seg.speaker}.{ext}"

    # SSML：显式开启时逐片段校验（原文不改写、不自动包装）。校验失败不调用 API，
    # 直接记为 failed；错误信息定位到片段。普通文本模式永远不做 SSML 校验/包装。
    if extras.get("enable_ssml"):
        try:
            validated = validate_ssml(seg.text)
        except SSMLError as exc:
            log.error(f"{seg.index:04d} {label} SSML 校验失败：{exc}")
            return SegmentRecord(
                **_record_base(seg),
                status="failed", file=file_rel,
                attempts=(record.attempts if record is not None else 0) + 1,
                error=f"SSML 校验失败：{exc}",
                audio_params=extras,
            )
        # 冲突规则（官方英文标签页）：<speak rate/pitch/volume> 会覆盖同名 HTTP
        # 请求参数。为避免双重指定引起歧义，SSML 中出现的属性，HTTP 侧不再发送。
        # instruction（方言/情感/角色）与 SSML（韵律/停顿）正交，继续原文发送。
        for attr, api_key in (
            (validated.has_rate, "rate"),
            (validated.has_pitch, "pitch"),
            (validated.has_volume, "volume"),
        ):
            if attr and api_key in extras:
                log.info(
                    f"{seg.index:04d} {label} SSML <speak> 已指定 {api_key}，"
                    f"不再发送 HTTP {api_key} 参数（官方规定标签属性优先）。"
                )
                extras.pop(api_key, None)

    request = TTSRequest(
        text=seg.text,
        voice=setting.voice,
        language_type=setting.language_type,
        direction=instruction,  # 每请求只发最终生效的一条指令
        output_path=str(episode_dir / file_rel),
        audio_params=extras,
    )
    if len(seg.text) > MAX_TEXT_LEN:
        log.warning(
            f"{seg.index:04d} {label} 文本长度 {len(seg.text)} 超过百炼实测上限 "
            f"{MAX_TEXT_LEN}，可能被 API 拒绝（错误将如实记录）。"
        )

    old_attempts = record.attempts if record is not None else 0
    retry_used = 0
    started = time.perf_counter()

    def _on_retry(attempt: int, total_attempts: int, delay: float, exc: Exception) -> None:
        nonlocal retry_used
        retry_used = attempt
        log.retry(
            f"{seg.index:04d} {label} 第 {attempt}/{total_attempts} 次尝试失败，"
            f"等待 {delay:.1f}s 后重试：{exc}"
        )

    param_desc = " ".join(
        f"{k}={extras[k]}" for k in _PLANNED_NUMERIC if k in extras
    )
    log.info(
        f"生成 {seg.index:04d} {label}（音色 {setting.voice}）"
        f" instruction={instruction!r}"
        + (f" {param_desc}" if param_desc else "")
    )
    try:
        result: TTSResult = retry_with_backoff(
            lambda: provider.generate(request),
            max_retries=cfg.max_retries,
            base_delay=cfg.retry_base_delay,
            on_retry=_on_retry,
        )
    except Exception as exc:  # noqa: BLE001 —— 记入 manifest，继续其他片段
        total_attempts = old_attempts + max(retry_used, 0) + 1
        log.error(
            f"{seg.index:04d} {label} 生成失败（共 {total_attempts} 次尝试）：{exc}"
        )
        return SegmentRecord(
            **_record_base(seg),
            status="failed", file=file_rel, attempts=total_attempts, error=str(exc),
            audio_params=extras,
        )

    elapsed = time.perf_counter() - started
    total_attempts = old_attempts + retry_used + 1
    log.ok(f"{seg.index:04d} {label} 完成（{elapsed:.1f}s）")
    return SegmentRecord(
        **_record_base(seg),
        status="success", file=file_rel,
        content_hash=content_hash(
            cfg, seg.text, setting.voice, setting.language_type, instruction, extras
        ),
        audio_params=extras,
        attempts=total_attempts, request_id=result.request_id,
    )
