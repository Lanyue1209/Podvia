"""FFmpeg 合并（文档第 16 节）：ffprobe 探测、静音生成、无损拼接 + 一次性编码。

- 决策树（16.1）：全部片段为一致 PCM WAV → 生成同参静音 → concat demuxer 无损拼接
  → 最终一次性编码为 MP3（全片唯一一次有损编码）；格式不一致（含 MP3/混合）→
  在【独立临时目录】把每个片段转码为 24kHz/mono/s16 WAV 后再拼接（绝不覆盖原始
  片段，哪怕它的扩展名是 .mp3），临时目录在成功/失败后都清理。
- 静音（16.2）：anullsrc 按探测到的采样率/声道生成（不硬编码 24000）；
  PAUSE 文件已存在且非空 → 视为缓存跳过
- 拼接（16.3）：concat_list.txt（UTF-8、绝对路径、-safe 0）；segments_signature
  （文件名+大小+hash 清单摘要）一致且输出存在且可被 ffprobe 识别 → 跳过，--force
  强制重并
- 合并前置校验（16.3）：manifest 存在未完成片段 → 拒绝并提示先 retry
- FFmpeg 缺失（16.4）：-version 探测失败 → MergeError + 安装指引（main.py 映射退出码 5）

本模块不感知脚本/Provider 概念（不 import parser/providers，架构约束见文档 0-5）。
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from podcast_tts import log
from podcast_tts.config import AppConfig
from podcast_tts.manifest import (
    Manifest,
    content_hash,
    load as load_manifest,
    manifest_path,
    save,
    utc_now_iso,
)
from podcast_tts.subtitles import (
    SrtError,
    build_timeline,
    cues_from_timeline,
    render_srt,
    write_srt,
)


class MergeError(Exception):
    """合并失败（FFmpeg 缺失/片段未完成/文件缺失/命令失败）。main.py 映射为退出码 5。"""


@dataclass
class AudioFormat:
    """一个音频文件的流参数（ffprobe 探测结果）。"""

    codec: str
    sample_rate: int
    channels: int

    @property
    def is_pcm_wav(self) -> bool:
        return self.codec.startswith("pcm_")

    @property
    def channel_layout(self) -> str:
        if self.channels == 1:
            return "mono"
        if self.channels == 2:
            return "stereo"
        raise MergeError(f"不支持的声道数 {self.channels}（V1 仅支持 mono/stereo）。")


@dataclass
class MergeOutcome:
    """一次合并的结果摘要（main.py 据此决定提示与退出码）。"""

    episode: str
    output: str            # 相对 episode 目录，如 "output/episode001.mp3"
    duration: float | None
    skipped: bool          # True = 签名一致跳过合并
    srt: str | None = None  # 相对 episode 目录的字幕路径，如 "output/episode001.srt"


# —— 工具定位与执行 ————————————————————————————————————————————————


def resolve_tools(cfg: AppConfig) -> tuple[str, str]:
    """定位 ffmpeg/ffprobe：cfg.ffmpeg_dir 内有 exe 则用之，否则回退 PATH（裸名）。"""
    return _resolve_tool(cfg.ffmpeg_dir, "ffmpeg"), _resolve_tool(cfg.ffmpeg_dir, "ffprobe")


def _resolve_tool(directory: Path, name: str) -> str:
    exe = directory / f"{name}.exe"
    if exe.is_file():
        return str(exe)
    return name  # 依赖系统 PATH


def verify_tools(ffmpeg: str, ffprobe: str) -> None:
    """探测两个可执行文件可用（16.4）。失败给出 Windows 安装指引。"""
    for tool in (ffprobe, ffmpeg):
        try:
            _run_command([tool, "-version"])
        except MergeError as exc:
            raise MergeError(
                f"FFmpeg/ffprobe 不可用（{tool}）：{exc}\n"
                "       Windows 安装指引：从 https://www.gyan.dev/ffmpeg/builds/ 下载"
                " release-full 解压，把 bin 目录加入 PATH，"
                "或在 config.yaml 的 paths.ffmpeg_dir 指定目录。"
            ) from exc


def _run_command(cmd: list[str]) -> None:
    """执行外部命令；非 0 退出或启动失败抛 MergeError（stderr 末尾几行入错误信息）。"""
    log.debug("执行：" + " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
    except OSError as exc:
        raise MergeError(f"命令执行失败：{exc}") from exc
    if proc.returncode != 0:
        tail = [line for line in (proc.stderr or "").strip().splitlines() if line][-3:]
        raise MergeError(
            f"FFmpeg 命令失败（exit {proc.returncode}）：{' | '.join(tail) or '无 stderr 输出'}"
        )


# —— 探测 ————————————————————————————————————————————————————————


def probe(ffprobe: str, file: Path) -> AudioFormat:
    """探测音频流参数（取第一个音频流）。"""
    cmd = [
        ffprobe, "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,sample_rate,channels",
        "-of", "json", str(file),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
    except OSError as exc:
        raise MergeError(f"ffprobe 执行失败：{exc}") from exc
    if proc.returncode != 0:
        raise MergeError(f"ffprobe 探测失败：{file}（{(proc.stderr or '').strip()[:200]}）")
    try:
        streams = json.loads(proc.stdout).get("streams") or []
        stream = streams[0]
        return AudioFormat(
            codec=str(stream["codec_name"]),
            sample_rate=int(stream["sample_rate"]),
            channels=int(stream["channels"]),
        )
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise MergeError(f"ffprobe 输出无法解析：{file}（{exc}）") from exc


def probe_duration(ffprobe: str, file: Path) -> float | None:
    """探测时长（秒）；失败仅 DEBUG 记录，不影响合并结果。"""
    cmd = [ffprobe, "-v", "error", "-show_entries", "format=duration",
           "-of", "json", str(file)]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        return float(json.loads(proc.stdout)["format"]["duration"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        log.debug(f"时长探测失败（不影响合并）：{file}（{exc}）")
        return None


# —— 静音 / 标准化 ——————————————————————————————————————————————————


def make_silence(ffmpeg: str, seconds: float, fmt: AudioFormat, out_path: Path) -> None:
    """生成 PCM 静音（16.2）：anullsrc 参数取自探测结果，不硬编码。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run_command([
        ffmpeg, "-y", "-f", "lavfi",
        "-i", f"anullsrc=r={fmt.sample_rate}:cl={fmt.channel_layout}",
        "-t", f"{max(seconds, 0.0):.3f}", "-c:a", "pcm_s16le", str(out_path),
    ])


def _normalize_to(ffmpeg: str, src: Path, dst: Path) -> None:
    """把任意片段转码为 24kHz/mono/s16 WAV，写入独立临时目录（不动原片段）。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run_command([
        ffmpeg, "-y", "-i", str(src),
        "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(dst),
    ])


# 规范化后的统一参数（混合格式分支使用）
_NORMALIZED_REF = AudioFormat(codec="pcm_s16le", sample_rate=24000, channels=1)


# —— 签名与合并 ————————————————————————————————————————————————————


def segments_signature(episode_dir: Path, manifest: Manifest) -> str:
    """16.3：对"片段文件名+大小+hash 清单"取 SHA256；PAUSE 以时长参与。

    台词片段文件内容变化（重新生成）→ 大小变化 → 签名变化 → 触发重新合并。
    """
    lines = []
    for rec in manifest.segments:
        if rec.kind == "pause":
            lines.append(f"{rec.index:04d}|PAUSE|{rec.pause_seconds}")
            continue
        size = -1
        if rec.file:
            try:
                size = (episode_dir / rec.file).stat().st_size
            except OSError:
                pass
        lines.append(f"{rec.index:04d}|{rec.file}|{size}|{rec.content_hash}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def run_merge(
    episode: str,
    cfg: AppConfig,
    *,
    force: bool = False,
    expected_hashes: dict[str, str] | None = None,
) -> MergeOutcome:
    """合并一个 episode：校验 → 探测 → 静音 → 拼接编码 → 更新 manifest.merge。

    expected_hashes：{seg_id: 当前规划出的 content_hash}。传入时，每个台词片段
    manifest 中记录的 content_hash 必须与其一致，否则拒绝合并——防止把"待更新"
    片段的旧音频静默拼进成品（需求十五）。force=True 且 expected_hashes=None
    表示高级用户明确要求用现有片段文件强制重拼（不标记为最新由调用方负责提示）。

    抛 FileNotFoundError（manifest 缺失）或 MergeError（一切合并失败）。
    """
    episode_dir = cfg.projects_dir / episode
    manifest = load_manifest(episode_dir)
    if manifest is None:
        raise FileNotFoundError(
            f"未找到 {manifest_path(episode_dir)}（episode 可能尚未生成，"
            "先运行 generate 命令）。"
        )

    undone = [r for r in manifest.segments if r.kind == "speech" and r.status != "success"]
    if undone:
        raise MergeError(
            f"存在 {len(undone)} 个未完成片段"
            f"（{'、'.join(f'{r.index:04d} {r.speaker}' for r in undone)}），"
            "请先运行 retry 命令再合并。"
        )

    if expected_hashes is not None:
        speech_recs = [r for r in manifest.segments if r.kind == "speech"]
        stale = [
            r for r in speech_recs
            if r.seg_id in expected_hashes and r.content_hash != expected_hashes[r.seg_id]
        ]
        if stale:
            raise MergeError(
                f"存在 {len(stale)} 个待更新片段，不能用旧音频合并最新整章"
                f"（{'、'.join(f'{r.index:04d} {r.speaker}' for r in stale)}），"
                "请先生成这些片段。"
            )
        missing_plan = [r for r in speech_recs if r.seg_id not in expected_hashes]
        if missing_plan:
            raise MergeError(
                "部分片段缺少当前生成规划，无法确认音频是否最新："
                + "、".join(f"{r.index:04d}" for r in missing_plan)
                + "。请重新点「生成 / 更新音频」后再合并。"
            )
    elif not force:
        # 调用方未提供当前规划（CLI/恢复场景）：用 manifest 记录的当前规划字段
        # 自行重算请求指纹。status==success 但指纹不匹配即为"待更新"，
        # 绝不把旧音频静默合并为最新成品（需求十五）。
        from podcast_tts.pipeline import plan_segment  # 延迟导入避免循环
        from podcast_tts.models import Segment as _Segment
        from podcast_tts.config import ConfigError as _ConfigError

        stale = []
        for r in manifest.segments:
            if r.kind != "speech":
                continue
            seg = _Segment(
                index=r.index, kind="speech", seg_id=r.seg_id,
                speaker=r.speaker, speaker_name=r.speaker_name,
                text=r.text or "", direction=r.direction,
                chapter_direction=r.chapter_direction,
                rate=r.rate, pitch=r.pitch, volume=r.volume,
            )
            try:
                setting, instruction, extras = plan_segment(cfg, seg)
            except _ConfigError:
                continue  # 规划无效的片段交由生成阶段处理，此处不误放
            want = content_hash(
                cfg, seg.text, setting.voice, setting.language_type,
                instruction, extras,
            )
            if r.content_hash != want:
                stale.append(r)
        if stale:
            raise MergeError(
                f"存在 {len(stale)} 个待更新片段，不能用旧音频合并最新整章"
                f"（{'、'.join(f'{r.index:04d} {r.speaker}' for r in stale)}），"
                "请先重新生成（generate）这些片段。"
            )

    out_rel = f"output/{episode}.{cfg.merge_format}"
    out_path = episode_dir / out_rel

    sig = segments_signature(episode_dir, manifest)
    if (
        not force
        and manifest.merge.get("status") == "success"
        and manifest.merge.get("segments_signature") == sig
        and out_path.is_file()
    ):
        # 已声明最新还不够：必须能被 ffprobe 真实识别才算数（防止空文件/伪装扩展名）
        _ffmpeg, _ffprobe = resolve_tools(cfg)
        try:
            probe(_ffprobe, out_path)
        except MergeError as exc:
            log.warning(f"既有合并产物无法被 ffprobe 识别，重新合并：{out_path}（{exc}）")
        else:
            duration = probe_duration(_ffprobe, out_path)
            log.info(f"合并产物已是最新：{out_rel}（--force 可强制重新合并）。")
            existing_srt = out_path.with_suffix(".srt")
            srt_rel = (f"output/{existing_srt.name}"
                       if existing_srt.is_file() else None)
            return MergeOutcome(episode=episode, output=out_rel, duration=duration,
                                skipped=True, srt=srt_rel)

    ffmpeg, ffprobe = resolve_tools(cfg)
    verify_tools(ffmpeg, ffprobe)

    # 16.1 决策树第 1 步：逐个探测台词片段
    speech_recs = [r for r in manifest.segments if r.kind == "speech"]
    formats: dict[int, AudioFormat] = {}
    for rec in speech_recs:
        seg_file = episode_dir / (rec.file or "")
        if not seg_file.is_file():
            raise MergeError(
                f"片段文件缺失：{rec.file}（manifest 记录 success 但文件不存在），"
                "请重新生成该片段。"
            )
        formats[rec.index] = probe(ffprobe, seg_file)

    # 规范化工作目录：独立临时目录，绝不写回 segments/（成功或失败都清理）
    tmp_dir = episode_dir / "metadata" / "merge_tmp"
    distinct = {(f.codec, f.sample_rate, f.channels) for f in formats.values()}
    need_normalize = len(distinct) > 1 or not all(f.is_pcm_wav for f in formats.values())
    normalized = False
    try:
        if need_normalize:
            # 16.1 分支 3：存在 MP3 或参数不一致 → 全部转码进临时目录再拼接。
            # 原始片段（含 .mp3）的文件名/扩展名/大小/内容一律不变。
            log.warning(
                "片段格式不一致或包含非 WAV，在独立临时目录标准化为 "
                "24kHz/mono/s16 WAV 后再拼接（不修改任何原始片段）。"
            )
            shutil.rmtree(tmp_dir, ignore_errors=True)
            tmp_dir.mkdir(parents=True, exist_ok=True)
            ordered_sources: list[Path] = []
            for rec in manifest.segments:
                src = episode_dir / (rec.file or "")
                if rec.kind == "speech":
                    dst = tmp_dir / f"{rec.index:04d}.wav"
                    _normalize_to(ffmpeg, src, dst)
                    ordered_sources.append(dst)
                else:
                    # PAUSE 恒为 WAV 并缓存在 segments/：缺失则按统一参数生成到
                    # segments（可复用）；已存在且参数一致则直接引用；历史遗留
                    # 参数不同才在临时目录转码（同样不改原文件）。
                    if not (src.is_file() and src.stat().st_size > 0):
                        log.info(f"生成静音 {rec.index:04d}（{rec.pause_seconds:.1f}s）...")
                        make_silence(
                            ffmpeg, float(rec.pause_seconds or 0.0), _NORMALIZED_REF, src
                        )
                    pf = probe(ffprobe, src)
                    if (pf.codec, pf.sample_rate, pf.channels) == (
                        "pcm_s16le", 24000, 1
                    ):
                        ordered_sources.append(src)
                    else:
                        dst = tmp_dir / f"{rec.index:04d}.wav"
                        _normalize_to(ffmpeg, src, dst)
                        ordered_sources.append(dst)
            ref = _NORMALIZED_REF
            normalized = True
        else:
            ref = next(iter(formats.values()), _NORMALIZED_REF)
            # 16.2：生成静音到 segments 目录（PAUSE 恒为 WAV，已有非空文件视为缓存）
            pause_recs = [r for r in manifest.segments if r.kind == "pause"]
            for rec in pause_recs:
                out = episode_dir / (rec.file or "")
                if force or not (out.is_file() and out.stat().st_size > 0):
                    log.info(f"生成静音 {rec.index:04d}（{rec.pause_seconds:.1f}s）...")
                    make_silence(ffmpeg, float(rec.pause_seconds or 0.0), ref, out)
            ordered_sources = [
                episode_dir / (rec.file or "") for rec in manifest.segments
            ]

        # 16.3：concat 无损拼接 + 最终一次编码。
        # 顺序严格按 manifest.segments（脚本顺序），与请求完成先后无关。
        concat_list = episode_dir / "metadata" / "concat_list.txt"
        lines = []
        for src_path in ordered_sources:
            absolute = src_path.resolve()
            lines.append("file '" + str(absolute).replace("'", "'\\''") + "'")
        concat_list.write_text("\n".join(lines) + "\n", encoding="utf-8")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list)]
        if cfg.merge_format == "mp3":
            cmd += ["-c:a", "libmp3lame", "-b:a", cfg.bitrate]
        else:
            cmd += ["-c:a", "copy"]
        cmd += [str(out_path)]

        log.info(
            f"合并 {len(manifest.segments)} 段"
            f"（{len(speech_recs)} 台词 + {len(manifest.segments) - len(speech_recs)} 静音）"
            f"→ {out_rel} ..."
        )
        _run_command(cmd)

        if not out_path.is_file() or out_path.stat().st_size == 0:
            raise MergeError(f"合并输出异常（文件缺失或为空）：{out_path}")
        # 只有 ffprobe 能真实识别最终文件，才算合并成功（扩展名不能说明内容）
        try:
            final_fmt = probe(ffprobe, out_path)
        except MergeError as exc:
            raise MergeError(
                f"合并产物无法被 ffprobe 识别：{out_path}（{exc}）"
            ) from exc

        # 句级 SRT：直接遍历本次写进 concat_list 的同一份 ordered_sources
        # （规范化临时文件此刻仍存在）。失败不阻断音频合并，只放弃本次字幕。
        srt_rel = None
        srt_cues = None
        try:
            timeline = build_timeline(
                episode_dir, manifest, ffprobe, probe_duration,
                sources=ordered_sources)
            srt_cues = cues_from_timeline(timeline)
        except SrtError as exc:
            log.warning(f"字幕未随本次合并生成：{exc}")

        if srt_cues is not None and srt_cues:
            srt_path = out_path.with_suffix(".srt")
            write_srt(srt_path, render_srt(srt_cues))
            srt_rel = f"output/{srt_path.name}"
            log.ok(f"字幕已生成：{srt_rel}（{len(srt_cues)} 条，时间轴与本次合并同源）")
        elif srt_cues is not None:
            log.info("本项目没有台词片段（纯停顿），不生成 SRT 字幕。")
    finally:
        # 无论成功失败都清理临时规范化目录（PAUSE/台词原片段不受影响）
        shutil.rmtree(tmp_dir, ignore_errors=True)

    duration = probe_duration(ffprobe, out_path)
    manifest.merge = {
        "status": "success",
        "output": out_rel,
        "merged_at": utc_now_iso(),
        "segments_signature": sig,
        "codec": final_fmt.codec,
        "sample_rate": final_fmt.sample_rate,
        "channels": final_fmt.channels,
    }
    if normalized:
        manifest.merge["normalized"] = True
    if srt_rel:
        # 字幕条目快照（与本次 concat 同源计算），便于诊断与未来其它字幕格式复用
        manifest.merge["subtitles"] = {
            "file": srt_rel,
            "cues": [
                {"n": c.number, "start_ms": c.start_ms,
                 "end_ms": c.end_ms, "text": c.text}
                for c in (srt_cues or [])
            ],
        }
    save(episode_dir, manifest)

    dur_text = f"，时长 {duration:.1f}s" if duration else ""
    srt_text = "，含 SRT 字幕" if srt_rel else ""
    log.ok(f"合并完成：{out_rel}（{len(manifest.segments)} 段{dur_text}{srt_text}）")
    return MergeOutcome(episode=episode, output=out_rel, duration=duration,
                        skipped=False, srt=srt_rel)
