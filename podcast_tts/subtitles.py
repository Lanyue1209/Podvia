"""句级 SRT 字幕：时间轴直接来自最终音频合并所用的同一份播放序列。

唯一事实来源（与 audio.run_merge 完全一致）：
- 顺序 = manifest.segments 的顺序（merge 写 concat_list.txt 的顺序）；
- 每条台词的时长 = 实际被拼接进去的那个音频文件经 ffprobe 读出的时长
  （merge 路径传入 ordered_sources，规范化临时文件也与之一一对应）；
- 每条停顿 = 实际静音文件的时长；独立导出而静音尚未生成时，使用
  pause_seconds（anullsrc -t 的输入值，精确到毫秒）。

全程使用整数毫秒累加，避免长音频浮点漂移。PAUSE 不产生字幕，只推进时间轴。
本模块不 import audio（audio 反向 import 本模块），ffprobe 路径与探测函数由
调用方注入；独立导出时在函数内延迟 import audio.resolve_tools/probe_duration。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from podcast_tts import log
from podcast_tts.manifest import Manifest, load as load_manifest

# ffprobe 可执行路径, 文件 -> 时长（秒）；读不出返回 None
DurationProber = Callable[[str, Path], "float | None"]


class SrtError(Exception):
    """字幕无法可靠生成（缺音频 / 时长不可读 / 无项目 / 无台词）。"""


@dataclass
class TimelineEntry:
    """最终播放序列中的一个条目（一条台词音频或一段停顿）。"""

    index: int
    seg_id: str | None
    kind: str            # "speech" | "pause"
    duration_ms: int
    text: str | None = None  # 仅 speech：清洗后的可见正文


@dataclass
class SubtitleCue:
    """一条句级字幕（闭区间 [start_ms, end_ms)，单位毫秒）。"""

    number: int
    start_ms: int
    end_ms: int
    text: str


# SSML / XML 控制标签（<speak>、<break …/> 等）；官方情绪标签走 emotags 精确表
_SSML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\n\u3000]+")


def format_srt_time(ms: int) -> str:
    """毫秒 → HH:MM:SS,mmm（小时不封顶，>1 小时同样正确；毫秒恒为 3 位）。"""
    ms = max(int(ms), 0)
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def clean_spoken_text(text: str | None) -> str:
    """提取观众应看到的正文：去官方 TTS 标签与 SSML，不动文字/翻译/标点。

    - 只删除 emotags 官方表内的标签（如 [laughing]/[excited]），普通正文里的
      方括号（如 [注]）原样保留；
    - 删除 <speak> 等 SSML/XML 标签；
    - 换行/制表/连续空白折成单个空格并去首尾空白。
    """
    if not text:
        return ""
    from podcast_tts import emotags

    parts: list[str] = []
    last = 0
    for start, end, _tag, _kind in emotags.iter_tags(text):
        parts.append(text[last:start])
        last = end
    parts.append(text[last:])
    out = _SSML_TAG_RE.sub("", "".join(parts))
    return _WS_RE.sub(" ", out).strip()


def build_timeline(
    episode_dir: Path,
    manifest: Manifest,
    ffprobe: str,
    prober: DurationProber,
    *,
    sources: "list[Path | str] | None" = None,
) -> list[TimelineEntry]:
    """按最终拼接顺序构造播放时间线（整数毫秒）。

    sources 给定时（merge 路径）必须与 manifest.segments 一一对应，且条目就是
    实际写进 concat_list 的文件（含规范化临时文件）——字幕与成品音频因此严格同源。
    sources 为 None 时（独立导出）：台词读 manifest 记录的原始片段文件，停顿优先
    读已生成的静音文件，缺失时用 pause_seconds 精确值。

    台词缺音频或时长不可读 → SrtError（绝不以 0 秒静默产出错误字幕）。
    """
    records = manifest.segments
    if sources is not None and len(sources) != len(records):
        raise SrtError(
            f"内部错误：合并序列（{len(sources)} 条）与 manifest 片段"
            f"（{len(records)} 条）数量不一致，拒绝生成字幕。"
        )

    entries: list[TimelineEntry] = []
    missing: list[int] = []
    unreadable: list[int] = []

    for i, rec in enumerate(records):
        if sources is not None:
            src = Path(sources[i])
        elif rec.kind == "pause":
            src = episode_dir / rec.file if rec.file else None
        else:
            src = episode_dir / rec.file if rec.file else None

        if rec.kind == "pause":
            seconds: "float | None" = None
            if src is not None and src.is_file() and src.stat().st_size > 0:
                seconds = prober(ffprobe, src)
            if seconds is None:
                # 静音尚未生成（用户未合并就导出）：anullsrc -t 的目标值，精确到毫秒
                seconds = float(rec.pause_seconds or 0.0)
            entries.append(TimelineEntry(
                index=rec.index, seg_id=rec.seg_id, kind="pause",
                duration_ms=max(round(seconds * 1000), 0)))
            continue

        # speech
        if rec.status != "success" or src is None or not src.is_file() \
                or src.stat().st_size == 0:
            missing.append(rec.index)
            continue
        seconds = prober(ffprobe, src)
        if seconds is None or seconds <= 0:
            unreadable.append(rec.index)
            continue
        entries.append(TimelineEntry(
            index=rec.index, seg_id=rec.seg_id, kind="speech",
            duration_ms=max(round(seconds * 1000), 1),
            text=clean_spoken_text(rec.text)))

    if missing:
        raise SrtError(
            "以下台词片段还没有可用音频，无法生成与成品一致的字幕："
            f"{_fmt_indices(missing)}。请先点「生成 / 更新音频」完成这些片段，"
            "再导出 SRT（停顿不会列入）。"
        )
    if unreadable:
        raise SrtError(
            "以下片段的音频文件存在，但 ffmpeg/ffprobe 无法读取其时长，已拒绝"
            f"生成错误字幕：{_fmt_indices(unreadable)}。请检查 ffmpeg 是否可用、"
            "片段文件是否损坏，必要时重新生成该片段。"
        )
    return entries


def cues_from_timeline(entries: list[TimelineEntry]) -> list[SubtitleCue]:
    """把播放时间线转成句级字幕：整数毫秒累加，停顿只推进游标、不出字幕。

    异常的「有音频但正文为空」台词：同样占用时间轴（保证后续不错位），但不产生
    字幕条目，编号从 1 连续编号。
    """
    cues: list[SubtitleCue] = []
    cursor_ms = 0
    number = 0
    for entry in entries:
        start_ms = cursor_ms
        cursor_ms += entry.duration_ms
        if entry.kind != "speech":
            continue
        if not entry.text:
            log.warning(
                f"片段 {entry.index:04d} 有音频但正文为空，已跳过其字幕条目"
                "（时间轴仍计入该片段时长）。"
            )
            continue
        number += 1
        cues.append(SubtitleCue(number, start_ms, cursor_ms, entry.text))
    return cues


def render_srt(cues: list[SubtitleCue]) -> str:
    """序列化为标准 SRT（UTF-8 文本，LF，条目间空一行）。"""
    blocks = [
        f"{c.number}\n"
        f"{format_srt_time(c.start_ms)} --> {format_srt_time(c.end_ms)}\n"
        f"{c.text}"
        for c in cues
    ]
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def write_srt(path: Path, content: str) -> Path:
    """以 UTF-8（无 BOM）写出 .srt；返回路径。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    return path


def render_for_episode(
    episode: str,
    cfg,
    *,
    sources: "list[Path | str] | None" = None,
) -> tuple[str, int]:
    """独立导出入口：加载项目 → 同源时间线 → SRT 文本。

    返回 (srt 文本, 字幕条数)。0 条（纯停顿项目）时文本为空串，由调用方决定
    提示方式。merge 流程不经过这里（它直接用本次 ordered_sources）。
    """
    # 延迟导入：audio 顶层 import 本模块，避免循环
    from podcast_tts.audio import probe_duration, resolve_tools

    episode_dir = cfg.projects_dir / episode
    manifest = load_manifest(episode_dir)
    if manifest is None:
        raise SrtError(
            f"项目 {episode} 还没有 manifest.json（尚未生成过任何片段），"
            "无法导出字幕。"
        )
    _ffmpeg, ffprobe = resolve_tools(cfg)
    entries = build_timeline(
        episode_dir, manifest, ffprobe, probe_duration, sources=sources)
    cues = cues_from_timeline(entries)
    return render_srt(cues), len(cues)


def default_srt_path(episode: str, cfg) -> Path:
    """与最终音频同目录、同名：output/<episode>.srt。"""
    return cfg.projects_dir / episode / "output" / f"{episode}.srt"


def export_srt(episode: str, cfg, out_path: "Path | str | None" = None) -> tuple[Path, int]:
    """生成并写出 SRT（CLI/测试便捷入口），返回 (路径, 条数)。"""
    content, count = render_for_episode(episode, cfg)
    path = Path(out_path) if out_path else default_srt_path(episode, cfg)
    write_srt(path, content)
    return path, count


def _fmt_indices(indices: list[int]) -> str:
    return "、".join(f"{i:04d}" for i in indices)
