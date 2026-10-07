"""Working Script（工作脚本）：所有输入入口统一后的项目内部形态。

数据流（本模块是"来源 → 工作脚本"的唯一汇合点）：

    粘贴文本 ─┐
             ├─→ WorkingScript（projects/<episode>/working_script.json）
    导入文件 ─┘        │ 只在 Podvia 项目内保存/编辑，原始来源文件只读、永不回写
                      ↓ parse（script 模式 / book 模式共用同一条解析链）
                   Segment[] → manifest → 逐段生成 → 合并

设计纪律：
- 原始 .md/.txt 只作为 Source（source_type/source_name/imported_at 元信息），
  导入时把内容复制进工作脚本；之后一切编辑、解析、生成只使用工作脚本。
- 工作脚本文本是用户编辑层的 source of truth；片段级单句编辑以"按 seg_id 的
  覆盖层（overrides）"保存在同一文件中（章节正文存在句切分，无法稳定回写为
  源文行；覆盖层在每次解析后叠加，不另起平行系统）。
- 不引入数据库：沿用项目目录 JSON 持久化（与 manifest.json 同目录）。
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from podcast_tts.book import parse_book
from podcast_tts import log
from podcast_tts.config import ConfigError
from podcast_tts.models import Segment, SegmentRecord
from podcast_tts.parser import PAUSE_LINE, ScriptParser, match_speaker_header

WORKING_NAME = "working_script.json"
MODE_SCRIPT = "script"   # 文本框脚本（角色头语法）
MODE_BOOK = "book"       # 章节文件（# 章节 / - 角色: / 正文行级角色）


def working_path(projects_dir: Path, episode: str) -> Path:
    return projects_dir / episode / WORKING_NAME


def utc_now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class WorkingScript:
    """一份可编辑的项目工作脚本。"""

    episode: str
    mode: str = MODE_SCRIPT          # "script" | "book"
    text: str = ""                   # 当前工作脚本文本
    # 来源（只读，仅展示）：type = "paste" | "file" | "legacy"
    source_type: str = "paste"
    source_name: str | None = None   # 原始文件名（file 模式）
    imported_at: str = ""
    modified: bool = False           # 导入后用户是否改过（重新导入保护用）
    updated_at: str = ""
    # True = 刚从文件重新导入，应丢弃旧 manifest，所有片段全新（未生成）；
    # 首次规划/生成后会被置为 False，后续编辑走 reconcile 沿用 seg_id。
    fresh: bool = False
    # 片段级覆盖层：{seg_id: {"text"?/"speaker"?/"direction"?/"rate"?/...}}
    # key 存在即覆盖；number/direction 为 None 表示"清除行内指定，回退到角色/项目级"
    overrides: dict[str, dict] = field(default_factory=dict)
    # 每个覆盖字段创建时所基于的"源脚本值"快照：{seg_id: {"text"?/"rate"?/...}}
    # 用于 latest-edit-wins：主文本框随后修改了同一字段（当前源值 != base）时，
    # 该字段覆盖自动失效，用户在文本框的最新修改优先。旧项目无此字段 → 不做冲突检测。
    override_base: dict[str, dict] = field(default_factory=dict)

    # —— 构造 ————————————————————————————————————————————————————————

    @classmethod
    def from_paste(cls, episode: str, text: str) -> "WorkingScript":
        now = utc_now_iso()
        return cls(episode=episode, mode=MODE_SCRIPT, text=text,
                   source_type="paste", imported_at=now, updated_at=now)

    @classmethod
    def import_file(cls, episode: str, path: str | Path,
                    *, mode: str | None = None, fresh: bool = True) -> "WorkingScript":
        """读入原始文件内容并复制为工作脚本。

        绝不写回源文件。.md/.txt 按章节模式解析；mode 可显式指定。
        fresh=True（默认）：这是一次"重新导入"，应丢弃旧工作状态、全部未生成。
        fresh=False：仅读取内容生成工作脚本（如 CLI run_book 同文件二次运行，
        应走 reconcile 增量缓存，不视为全新导入）。
        """
        p = Path(path)
        content = p.read_text(encoding="utf-8-sig")
        now = utc_now_iso()
        guessed = MODE_BOOK if p.suffix.lower() in (".md", ".markdown", ".txt") else MODE_SCRIPT
        return cls(
            episode=episode, mode=mode or guessed, text=content,
            source_type="file", source_name=p.name,
            imported_at=now, updated_at=now, fresh=fresh,
        )

    @classmethod
    def from_manifest(cls, episode: str, manifest) -> "WorkingScript":
        """旧项目兼容：没有 working_script.json 时，从 manifest 记录重建工作脚本。

        重建为脚本形态（"角色: 台词" + [PAUSE:x]），标注 legacy 来源。
        """
        lines: list[str] = []
        for rec in manifest.segments:
            if rec.kind == "pause":
                lines.append(f"[PAUSE:{rec.pause_seconds:g}]")
            else:
                name = rec.speaker_name or rec.speaker or ""
                lines.append(f"{name}: {rec.text or ''}")
        now = utc_now_iso()
        return cls(
            episode=episode, mode=MODE_SCRIPT, text="\n\n".join(lines) + ("\n" if lines else ""),
            source_type="legacy", source_name=manifest.script_file or None,
            imported_at=manifest.created_at or now, updated_at=now,
        )

    # —— 编辑 ————————————————————————————————————————————————————————

    def update_text(self, text: str) -> bool:
        """更新工作脚本文本；与当前内容不同则标记 modified。返回是否发生变化。"""
        if text == self.text:
            return False
        self.text = text
        self.modified = True
        self.updated_at = utc_now_iso()
        return True

    def set_override(
        self, seg_id: str, patch: dict, base: dict | None = None
    ) -> None:
        """写入/合并一个片段的覆盖（单句编辑）；只含显式给出的键。

        speech 可覆盖 text/speaker/direction/rate/pitch/volume；
        pause 只可覆盖 pause_seconds（仅影响合并静音，不调用 TTS）。

        base：本次各覆盖字段创建时对应的"源脚本当前值"快照，用于
        latest-edit-wins 冲突消解（主文本框后续改了同字段则覆盖失效）。
        同一字段再次编辑时以最新源值刷新 base。
        """
        allowed = ("text", "speaker", "direction", "rate", "pitch", "volume",
                   "pause_seconds")
        clean = {k: v for k, v in patch.items() if k in allowed}
        existing = self.overrides.get(seg_id, {})
        existing.update(clean)
        # 全为"清除/回退"且没有正文等实质内容时，仍保留记录（空串/None 是显式语义）
        self.overrides[seg_id] = existing
        if base:
            base_clean = {k: v for k, v in base.items() if k in allowed}
            base_existing = self.override_base.get(seg_id, {})
            base_existing.update(base_clean)
            self.override_base[seg_id] = base_existing
        self.modified = True
        self.updated_at = utc_now_iso()

    def drop_override_fields(self, seg_id: str, fields: set[str]) -> None:
        """删除指定片段上已失效的覆盖字段（latest-edit-wins 冲突消解用）。"""
        patch = self.overrides.get(seg_id)
        if not patch:
            return
        for f in fields:
            patch.pop(f, None)
            b = self.override_base.get(seg_id)
            if b:
                b.pop(f, None)
        if not patch:
            self.overrides.pop(seg_id, None)
            self.override_base.pop(seg_id, None)

    def prune_overrides(self, alive_seg_ids: set[str]) -> None:
        """删除已不存在片段的残留覆盖（重解析后调用）。"""
        dead = {k for k in self.overrides if k not in alive_seg_ids}
        for k in dead:
            self.overrides.pop(k, None)
            self.override_base.pop(k, None)

    # —— 持久化 ——————————————————————————————————————————————————————

    def save(self, projects_dir: Path | str) -> Path:
        base = Path(projects_dir) / self.episode
        base.mkdir(parents=True, exist_ok=True)
        target = base / WORKING_NAME
        data = asdict(self)
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, target)
        log.debug(f"工作脚本已保存：{target}")
        return target

    @classmethod
    def load(cls, projects_dir: Path | str, episode: str) -> "WorkingScript | None":
        target = Path(projects_dir) / episode / WORKING_NAME
        if not target.exists():
            return None
        data = json.loads(target.read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


# —— 解析（两条入口共用） ——————————————————————————————————————————————

def speaker_name_to_id(cfg) -> dict[str, str]:
    """{角色显示名: 稳定ID}；未设 display_name（CLI）时显示名即 ID。"""
    return {(s.display_name or sid): sid for sid, s in cfg.voices.items()}


def effective_script_text(
    ws: WorkingScript, cfg, default_speaker_id: str | None = None
) -> str:
    """解析前实际使用的脚本文本（粘贴/文件统一入口）。

    - book 模式：原样返回（章首默认角色由 parse_book 内部处理）；
    - script 模式：首行已是注册角色头则原样；否则用默认角色补一个纯角色头
      （项目级导演指令走配置层级，不塞进角色头）；
      既没有角色头也没有默认角色时抛 ConfigError——绝不静默使用角色。
    注意：ws.text（用户编辑原文）不变，manifest 的 script_hash 仍基于原文。
    """
    if ws.mode == MODE_BOOK:
        return ws.text
    text = ws.text.strip("\n")
    stripped = text.strip()
    name_map = speaker_name_to_id(cfg)
    lines = stripped.splitlines()
    # 开头的空行与合法 PAUSE 行是独立停顿段，与角色无关（可出现在脚本最开头）；
    # "是否需要包默认角色头"取决于第一个真正需要朗读的内容行
    first_idx = 0
    while first_idx < len(lines):
        s = lines[first_idx].strip()
        if not s or PAUSE_LINE.match(s):
            first_idx += 1
            continue
        break
    first_content = lines[first_idx].strip() if first_idx < len(lines) else ""
    if first_content and match_speaker_header(first_content, name_map) is not None:
        return text if text.endswith("\n") else text + "\n"
    if default_speaker_id:
        if not first_content:
            # 纯停顿/空脚本：没有任何台词需要角色，原样解析（只产生 pause 段）
            return text if text.endswith("\n") else text + "\n"
        setting = cfg.voices.get(default_speaker_id)
        role = (setting.display_name if setting else None) or default_speaker_id
        # 角色头插在开头 PAUSE 之后，避免"空台词块 + PAUSE"解析冲突
        head = "\n".join(lines[:first_idx])
        rest = "\n".join(lines[first_idx:])
        prefix = f"{head}\n" if head.strip() else ""
        return f"{prefix}{role}：\n{rest}\n"
    raise ConfigError(
        "文本没有角色标记（如 \"角色名：台词\"），且未选择「默认角色」。"
        "请在角色区选择一个默认角色，或在每段台词前加上已注册的角色名；"
        "不会静默使用任何角色。"
    )


def parse_working(
    ws: WorkingScript,
    cfg,
    *,
    default_speaker_id: str | None = None,
) -> list[Segment]:
    """把工作脚本解析为 Segment[]（不触网）。

    script 模式：角色头脚本语法（ScriptParser）；无角色头时统一按
    effective_script_text 用默认角色包装（粘贴/文件两条入口在此汇合）。
    book 模式：章节语法（parse_book，正文行级角色 + 章首默认角色 fallback）。
    两种模式共用 scriptlines 行级语法，解析结果结构完全一致。
    """
    name_map = speaker_name_to_id(cfg)
    if ws.mode == MODE_BOOK:
        parsed = parse_book(
            ws.text, name_map,
            source_file=ws.source_name,
            default_speaker=default_speaker_id,
            default_title=ws.episode,
        )
        return parsed.segments
    return ScriptParser(name_map).parse(
        effective_script_text(ws, cfg, default_speaker_id)
    )


# —— 片段级覆盖层叠加（reconcile 分配 seg_id 之后） ————————————————————————

_OVERRIDE_FIELDS = ("speaker", "direction", "rate", "pitch", "volume")
# 覆盖字段 → Segment 属性
_FIELD_ATTR = {
    "text": "text", "speaker": "speaker", "direction": "direction",
    "rate": "rate", "pitch": "pitch", "volume": "volume",
    "pause_seconds": "pause_seconds",
}


def _norm_field(field: str, value):
    """归一化字段值用于 base/源值比较（'' 与 None 等价，数字按数值比较）。"""
    if field in ("rate", "pitch"):
        return float(value) if value not in (None, "") else None
    if field == "volume":
        return int(value) if value not in (None, "") else None
    if field == "pause_seconds":
        return float(value) if value not in (None, "") else None
    return str(value).strip() if value not in (None, "") else None


def apply_overrides(segments: list[Segment], ws: WorkingScript, cfg) -> list[str]:
    """把单句编辑覆盖叠加到解析结果上（就地修改 Segment）。

    返回实际生效的 seg_id 列表。

    latest-edit-wins（字段级冲突消解）：
    每个覆盖字段保存时记录其基于的源值（ws.override_base）。叠加时若该字段的
    当前源值已与 base 不同，说明用户之后又在主文本框改了同一字段——
    删除该字段覆盖（源值胜出），其余字段不受影响。旧项目无 base 快照时
    不做冲突检测（向后兼容，覆盖直接生效）。

    覆盖的字段即最终 TTS 请求字段，后续 content_hash（请求指纹）自然变化 →
    仅这些片段 dirty；pause 只有时长覆盖：不调 TTS，只在下次合并时改变静音长度。
    """
    if not ws.overrides:
        return []
    applied: list[str] = []
    id_to_name = {sid: (s.display_name or sid) for sid, s in cfg.voices.items()}
    for seg in segments:
        sid = seg.seg_id or ""
        patch = ws.overrides.get(sid)
        if not patch:
            continue
        base = ws.override_base.get(sid, {})

        if seg.kind == "pause":
            # 主文本框后来改了停顿时长 → 覆盖失效，源值优先
            if "pause_seconds" in base and _norm_field(
                    "pause_seconds", seg.pause_seconds) != _norm_field(
                    "pause_seconds", base.get("pause_seconds")):
                ws.drop_override_fields(sid, {"pause_seconds"})
                continue
            if patch.get("pause_seconds") not in (None, ""):
                try:
                    seg.pause_seconds = float(patch["pause_seconds"])
                except (TypeError, ValueError):
                    pass
            applied.append(seg.seg_id)
            continue

        # 冲突消解：逐字段比较"当前源值"与"编辑时基值"
        dropped: set[str] = set()
        live: dict[str, object] = {}
        for field, value in patch.items():
            if field == "pause_seconds":
                continue
            attr = _FIELD_ATTR.get(field)
            if attr is None:
                continue
            if field in base:
                current_source = _norm_field(field, getattr(seg, attr, None))
                if current_source != _norm_field(field, base[field]):
                    # 主文本框后来修改了同一字段 → 覆盖失效，源值优先
                    dropped.add(field)
                    continue
            live[field] = value
        if dropped:
            ws.drop_override_fields(sid, dropped)

        if not live:
            continue
        if live.get("text"):
            seg.text = str(live["text"])
        spk = live.get("speaker")
        if spk:
            seg.speaker = spk
            seg.speaker_name = id_to_name.get(spk, seg.speaker_name)
        if "direction" in live:
            val = live["direction"]
            seg.direction = str(val).strip() if val else None
        for key in ("rate", "pitch"):
            if key in live:
                val = live[key]
                setattr(seg, key, float(val) if val not in (None, "") else None)
        if "volume" in live:
            val = live["volume"]
            seg.volume = int(val) if val not in (None, "") else None
        applied.append(seg.seg_id)
    return applied


def sync_records_from_segments(
    records: list[SegmentRecord], segments: list[Segment]
) -> None:
    """把 Segment 的请求相关字段写回同序 manifest 记录（覆盖层/重解析后保持同步）。

    status/file/content_hash 等生成态字段不动——是否需要重新生成由
    content_hash（请求指纹）比对决定。
    """
    by_id = {r.seg_id: r for r in records if r.seg_id}
    for seg in segments:
        rec = by_id.get(seg.seg_id)
        if rec is None:
            continue
        rec.index = seg.index
        rec.kind = seg.kind
        rec.speaker = seg.speaker
        rec.speaker_name = seg.speaker_name
        rec.direction = seg.direction
        rec.chapter_direction = seg.chapter_direction
        rec.rate = seg.rate
        rec.pitch = seg.pitch
        rec.volume = seg.volume
        rec.text = seg.text
        if seg.kind == "pause":
            rec.pause_seconds = seg.pause_seconds
        rec.chapter = seg.chapter
        rec.source_file = seg.source_file
        rec.char_start = seg.char_start
        rec.char_end = seg.char_end


def new_seg_id() -> str:
    return uuid.uuid4().hex
