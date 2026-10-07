"""manifest.json 读写：断点续生成/缓存/状态查询的唯一依据（文档 5.3 / 14 / 15）。

规则：
- 读写集中于本模块，其他模块不得直接操作 manifest 文件
- 原子写：写 *.tmp 后 os.replace；ensure_ascii=False，UTF-8
- manifest 是唯一断点依据；不以"文件是否存在"作为成功依据
- PAUSE 段在 manifest 中直接记 success（合并阶段生成静音，不调 API）
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field, fields
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from podcast_tts import log
from podcast_tts.capabilities import HASH_VERSION
from podcast_tts.config import AppConfig, ROOT_DIR
from podcast_tts.models import Segment, SegmentRecord
from podcast_tts.project import new_seg_id

MANIFEST_NAME = "manifest.json"
# 百炼单条文本实测上限（文档 8.5：API 报 "Range of input length should be [0, 600]"）
MAX_TEXT_LEN = 600


def utc_now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_hash(
    cfg: AppConfig,
    text: str,
    voice: str,
    language_type: str | None,
    direction: str | None,
    audio_params: dict | None = None,
) -> str:
    """缓存键（版本见 capabilities.HASH_VERSION，当前 v3）：

    SHA256(HASH_VERSION + "|" + provider|model|voice|language_type|text|direction
              + "|" + JSON(audio_params, sort_keys))

    audio_params 是该片段实际发送的全部音频附加参数（format/sample_rate/rate/
    pitch/volume/seed/hot_fix/language_hints，已合并项目/角色/行内三层覆盖）。
    direction 是该片段最终生效的唯一一条导演指令（四级层级解析结果）。
    任何音频参数或最终指令变化 → hash 变化 → 自动重新生成。v3 前缀使旧版
    （v1 无参数 / v2 仅项目级参数与原始 direction）缓存一次性失效。
    """
    params_key = json.dumps(
        audio_params or {}, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    payload = "|".join([
        HASH_VERSION,
        cfg.provider, cfg.model, voice, language_type or "",
        text, direction or "",
        params_key,
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class Manifest:
    """projects/<episode>/metadata/manifest.json 的内存表示。"""

    episode: str
    created_at: str
    updated_at: str
    script_file: str                    # 相对项目根（跨盘符等异常时为绝对路径）
    script_hash: str                    # 脚本内容 SHA256，用于检测脚本变更
    provider: str
    model: str
    voices: dict[str, str]              # {角色: Voice ID}
    segments: list[SegmentRecord]
    merge: dict = field(default_factory=dict)  # Phase 7 填充
    hash_version: str = HASH_VERSION    # content_hash 公式版本（旧文件缺失视为 v1）

    def record(self, index: int) -> SegmentRecord | None:
        for rec in self.segments:
            if rec.index == index:
                return rec
        return None

    def replace(self, record: SegmentRecord) -> None:
        """按 index 替换片段记录（主线程串行调用，无并发问题）。"""
        for i, rec in enumerate(self.segments):
            if rec.index == record.index:
                self.segments[i] = record
                return
        raise KeyError(f"manifest 中不存在 index={record.index} 的记录")


def manifest_path(episode_dir: Path) -> Path:
    return episode_dir / "metadata" / MANIFEST_NAME


def fresh_episode_state(episode_dir: Path) -> None:
    """重新导入章节文件时调用：丢弃上一份工作状态，开始全新 Segment 集合。

    删除（仅 Podvia 管理的中间产物）：
    - metadata/manifest.json（旧 Segment 的 seg_id/音频/hash/状态全部废弃）
    - segments/ 目录下所有片段音频（已失去引用的生成缓存）

    保留：
    - output/ 下的最终合并成品（用户可能已导出/需参考；下次合并会覆盖同名文件）
    - working_script.json（由调用方用新内容覆盖，不在这里删）

    不删除任何用户目录或源码。
    """
    manifest_file = manifest_path(episode_dir)
    if manifest_file.exists():
        try:
            manifest_file.unlink()
        except OSError:
            pass
    seg_dir = episode_dir / "segments"
    if seg_dir.is_dir():
        shutil.rmtree(seg_dir, ignore_errors=True)


# Podvia 受管片段音频扩展名（orphan 清理白名单，绝不碰其他文件）
_SEG_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}


def cleanup_orphan_audio(episode_dir: Path, referenced_files: set[str]) -> list[str]:
    """删除 segments/ 中已无任何 manifest 记录引用的受管片段音频。

    安全边界：
    - 只扫描 <episode>/segments/ 目录；
    - 只删白名单扩展名的音频文件（.part 等临时文件不在此列）；
    - output/、原始输入、其他项目文件一律不碰。
    返回被删除文件的相对路径（POSIX 风格）列表。
    """
    seg_dir = episode_dir / "segments"
    if not seg_dir.is_dir():
        return []
    referenced = {p.replace("\\", "/") for p in referenced_files if p}
    removed: list[str] = []
    for f in seg_dir.iterdir():
        if not f.is_file() or f.suffix.lower() not in _SEG_AUDIO_EXTS:
            continue
        rel = f.relative_to(episode_dir).as_posix()
        if rel in referenced:
            continue
        try:
            f.unlink()
            removed.append(rel)
        except OSError:
            pass
    return removed


def load(episode_dir: Path) -> Manifest | None:
    """加载已有 manifest；不存在返回 None。"""
    path = manifest_path(episode_dir)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    record_fields = {f.name for f in fields(SegmentRecord)}
    return Manifest(
        episode=data["episode"],
        created_at=data["created_at"],
        updated_at=data["updated_at"],
        script_file=data.get("script_file", ""),
        script_hash=data.get("script_hash", ""),
        provider=data.get("provider", ""),
        model=data.get("model", ""),
        voices=data.get("voices", {}),
        segments=[
            SegmentRecord(**{k: v for k, v in rec.items() if k in record_fields})
            for rec in data.get("segments", [])
        ],
        merge=data.get("merge", {}),
        hash_version=data.get("hash_version", "v1"),
    )


def _identity(kind: str, speaker: str | None, text: str | None,
              pause_seconds: float | None) -> tuple:
    """重解析时用于对齐旧记录的片段身份键。

    speech = (类型, 角色稳定ID, 发送文本)；pause 统一为 ("pause",)
    —— 改停顿秒数不改变身份（只影响合并，不调 TTS）。
    """
    if kind == "pause":
        return ("pause",)
    return ("speech", speaker or "", text or "")


def _fresh_record(seg: Segment, *, used_files: set[str]) -> SegmentRecord:
    """全新片段的初始记录：speech=pending（未生成）；pause=success（不调 API）。"""
    seg_id = new_seg_id()
    seg.seg_id = seg_id
    if seg.kind == "pause":
        # 静音文件名包含时长：时长变化 → 文件名变化 → 合并时重新生成对应长度的静音，
        # 旧时长文件成为 orphan 由 plan_and_persist 安全清理（绝不复用错误时长）
        file_rel = f"segments/{seg.index:04d}_PAUSE_{seg.pause_seconds:g}s.wav"
        if file_rel in used_files:  # 序号重排导致与历史文件重名：加短 id 消歧
            file_rel = f"segments/{seg.index:04d}_PAUSE_{seg.pause_seconds:g}s_{seg_id[:6]}.wav"
        used_files.add(file_rel)
        return SegmentRecord(
            seg_id=seg_id, index=seg.index, kind="pause",
            pause_seconds=seg.pause_seconds, status="success", file=file_rel,
            chapter=seg.chapter, source_file=seg.source_file,
            char_start=seg.char_start, char_end=seg.char_end,
        )
    return SegmentRecord(
        seg_id=seg_id, index=seg.index, kind="speech",
        speaker=seg.speaker, speaker_name=seg.speaker_name,
        direction=seg.direction, chapter_direction=seg.chapter_direction,
        rate=seg.rate, pitch=seg.pitch, volume=seg.volume, text=seg.text,
        status="pending",
        chapter=seg.chapter, source_file=seg.source_file,
        char_start=seg.char_start, char_end=seg.char_end,
    )


def _inherit_record(old: SegmentRecord, seg: Segment,
                    used_files: set[str]) -> SegmentRecord:
    """旧记录沿用到重解析后的同身份/同位置片段：保留生成态（音频/hash/次数），
    更新请求相关字段。是否需要重新生成交给 content_hash 指纹比对。"""
    seg_id = old.seg_id or new_seg_id()
    seg.seg_id = seg_id
    if seg.kind == "pause":
        # 时长未变：沿用旧文件（含历史无时长的 0001_PAUSE.wav 命名，向后兼容）；
        # 时长已变：改用带时长的新文件名，迫使合并阶段重新生成对应长度的静音
        if old.file and old.pause_seconds == seg.pause_seconds:
            file_rel = old.file
        else:
            file_rel = f"segments/{seg.index:04d}_PAUSE_{seg.pause_seconds:g}s.wav"
            if file_rel in used_files:
                file_rel = (f"segments/{seg.index:04d}_PAUSE_"
                            f"{seg.pause_seconds:g}s_{seg_id[:6]}.wav")
        used_files.add(file_rel)
        return SegmentRecord(
            seg_id=seg_id, index=seg.index, kind="pause",
            pause_seconds=seg.pause_seconds, status="success", file=file_rel,
            attempts=old.attempts,
            chapter=seg.chapter, source_file=seg.source_file,
            char_start=seg.char_start, char_end=seg.char_end,
        )
    return SegmentRecord(
        seg_id=seg_id, index=seg.index, kind="speech",
        speaker=seg.speaker, speaker_name=seg.speaker_name,
        direction=seg.direction, chapter_direction=seg.chapter_direction,
        rate=seg.rate, pitch=seg.pitch, volume=seg.volume, text=seg.text,
        # —— 生成态全部沿用：是否 dirty 由 content_hash/文件存在性判定 ——
        status=old.status, file=old.file, content_hash=old.content_hash,
        audio_params=dict(old.audio_params), attempts=old.attempts,
        request_id=old.request_id, error=old.error,
        chapter=seg.chapter, source_file=seg.source_file,
        char_start=seg.char_start, char_end=seg.char_end,
    )


def reconcile_segments(
    old_records: list[SegmentRecord] | None,
    new_segments: list[Segment],
) -> list[SegmentRecord]:
    """把重解析结果与历史 manifest 记录对齐（稳定 seg_id 的核心）。

    用 difflib 对"身份键序列"做 LCS 对齐：
    - equal：同身份片段，沿用记录（文本/参数 hash 一致即 clean）；
    - replace 1:1 且同种类：视为"编辑了同一句"——沿用 seg_id 与音频关联，
      请求字段已更新，hash 不一致即 dirty，只重生成这一句；
    - insert：新片段（pending）；delete：旧记录移除（音频文件保留在磁盘不删除）；
    - replace 中数量不等或种类互变：逐个按上述规则处理，多余新片段 pending，
      多余旧记录移除。

    副作用：给每个 new_segment 分配/沿用 seg_id。返回与 new_segments 同序的记录。
    """
    if not old_records:
        used: set[str] = set()
        return [_fresh_record(seg, used_files=used) for seg in new_segments]

    old_keys = [
        _identity(r.kind, r.speaker, r.text, r.pause_seconds) for r in old_records
    ]
    new_keys = [
        _identity(s.kind, s.speaker, s.text, s.pause_seconds) for s in new_segments
    ]
    used_files = {r.file for r in old_records if r.file}
    out: list[SegmentRecord] = []
    matcher = difflib.SequenceMatcher(a=old_keys, b=new_keys, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                out.append(_inherit_record(
                    old_records[i1 + k], new_segments[j1 + k], used_files))
        elif tag == "insert":
            for j in range(j1, j2):
                out.append(_fresh_record(new_segments[j], used_files=used_files))
        elif tag == "delete":
            continue  # 旧片段删除：记录移除，磁盘音频保留（不做破坏性删除）
        else:  # replace：优先 1:1 沿用（编辑），处理不了的按新增/删除
            olds = list(range(i1, i2))
            news = list(range(j1, j2))
            for oi, nj in zip(olds, news):
                old, seg = old_records[oi], new_segments[nj]
                if old.kind == seg.kind:
                    out.append(_inherit_record(old, seg, used_files))
                else:
                    out.append(_fresh_record(seg, used_files=used_files))
            for nj in news[len(olds):]:
                out.append(_fresh_record(new_segments[nj], used_files=used_files))
    return out


def load_or_init(
    episode_dir: Path,
    script_path: Path,
    cfg: AppConfig,
    segments: list[Segment],
) -> Manifest:
    """兼容入口（CLI/旧测试）：从脚本文件读取文本并同步 manifest。"""
    return sync_manifest(
        episode_dir,
        script_label=_rel_script_label(script_path),
        script_text=script_path.read_text(encoding="utf-8"),
        cfg=cfg, segments=segments,
    )


def _rel_script_label(script_path: Path) -> str:
    try:
        return Path(os.path.relpath(script_path, ROOT_DIR)).as_posix()
    except ValueError:  # Windows 跨盘符
        return script_path.resolve().as_posix()


def sync_manifest(
    episode_dir: Path,
    *,
    script_label: str,
    script_text: str,
    cfg: AppConfig,
    segments: list[Segment],
    fresh: bool = False,
) -> Manifest:
    """按当前解析结果同步 manifest（加载或新建 → reconcile 稳定对齐）。

    fresh=True 表示"重新导入=全新工作状态"：忽略磁盘上的旧 manifest，
    为所有片段创建全新记录（speech=pending/未生成，pause=success），
    不与旧 Segment 做 diff。
    fresh=False（默认）表示"编辑当前工作脚本"：reconcile 沿用旧 seg_id 与音频。

    纯函数，不落盘，由调用方 save。
    """
    existing = None if fresh else load(episode_dir)
    script_hash = sha256_text(script_text)

    if existing is not None:
        if existing.hash_version != HASH_VERSION:
            log.warning(
                f"manifest 为旧版缓存公式（{existing.hash_version}），当前版本为 {HASH_VERSION}"
                "（content_hash 新纳入 format/sample_rate/rate/pitch/volume/seed/hot_fix/"
                "language_hints 等音频参数）。所有已成功片段的 content_hash 将不一致，"
                "本次运行会全部重新生成。"
            )
            existing.hash_version = HASH_VERSION
        if existing.script_hash and existing.script_hash != script_hash:
            log.warning(
                "工作脚本与上次生成时相比已变化，正在按片段内容对齐：未变化的片段"
                "继续沿用已有音频，变化/新增的片段才会重新生成。"
            )
        existing.script_hash = script_hash
        existing.script_file = script_label
        if (existing.provider, existing.model) != (cfg.provider, cfg.model):
            log.warning(
                f"manifest 记录的 provider/model（{existing.provider}/{existing.model}）与"
                f"当前配置（{cfg.provider}/{cfg.model}）不一致，content_hash 将不同，"
                "全部片段会重新生成。"
            )
        existing.provider = cfg.provider
        existing.model = cfg.model
        existing.voices = {name: setting.voice for name, setting in cfg.voices.items()}
        # 稳定对齐：seg_id 沿用/新增，记录字段更新，生成态保留
        existing.segments = reconcile_segments(existing.segments, segments)
        return existing

    now = utc_now_iso()
    used: set[str] = set()
    records = [_fresh_record(seg, used_files=used) for seg in segments]
    return Manifest(
        episode=episode_dir.name,
        created_at=now,
        updated_at=now,
        script_file=script_label,
        script_hash=script_hash,
        provider=cfg.provider,
        model=cfg.model,
        voices={name: setting.voice for name, setting in cfg.voices.items()},
        segments=records,
    )


def save(episode_dir: Path, manifest: Manifest) -> None:
    """原子写 manifest（tmp + os.replace）。"""
    path = manifest_path(episode_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.updated_at = utc_now_iso()
    data = asdict(manifest)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)
    log.debug(f"manifest 已保存：{path}")
