"""章节导入工作流：章节文件解析 + 规则化安全分段（通用语音合成的一种输入方式）。

与脚本解析器（parser.py）并列，**不改动现有 pipeline**：
本模块只负责把整本/单章文本切成 Segment 列表，生成/缓存/重试/合并全部复用
pipeline 既有能力。

硬约束：
- 纯本地规则，无网络、无 LLM、绝不改写原文；发送文本只做空白折叠
  （连续空白 → 单个空格），不增删/替换任何字词。
- 每个片段带原文追溯坐标：(source_file, chapter, char_start, char_end)，
  source[char_start:char_end] 去空白后与 raw_text 完全一致。
- 分段只在句读/停顿处断开，绝不切进：数字（3.14 / 1,000）、缩写
  （Mr. / U.S.A. / e.g.）、专有名词与人名（词内不含句读，天然不拆）、
  SSML <speak>...</speak> 结构（整块原子化）。

章节文件格式（.md/.txt，UTF-8）：
    # 第一章 标题
    - 角色: Lily            # 可选；朗读者/说话人/speaker 同义；整章默认角色（fallback）
    - 导演: 平静、温柔地讲述    # 可选；导演指令/指令/direction 同义（章节级 instruction）
    正文段落……               # 普通正文 → 章节默认角色
    Lily(轻快地): [excited] 台词   # 显式角色行 → 该角色对白（写法与脚本模式完全一致）
    Leo: 台词
    旁白又说话了。            # 下一段普通正文回到章节默认角色
    # 第二章 标题
角色优先级：行内显式角色 > 章首 "- 角色:"（默认角色）> 全局默认角色；
显式角色行只作用于自己的台词块（到空行为止），不污染后续普通正文。
章首无 "- 角色:" 且无全局默认时，仅当正文出现普通正文才报错
（纯显式角色对话的章节可以没有默认角色）。
.txt 无 Markdown 标题时，可用换页符 \\f 分章；两者都没有则视为单章。

正文行级语法（角色头/行内参数/PAUSE/情绪标签/Markdown 清洗）与脚本模式共享
同一套实现（scriptlines 模块）：同一种写法在文本框与章节文件中解析结果一致。
差异约定：章节正文中疑似角色声明但未注册/格式错误的行，按普通正文处理并记录
WARNING（章节有"默认角色兜底"契约；脚本模式则直接报错，避免读出角色名）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from podcast_tts import emotags
from podcast_tts import log
from podcast_tts.models import Segment
from podcast_tts.scriptlines import (
    PAUSE_LINE,
    _join_lines,
    looks_like_unregistered_speaker,
    match_speaker_header,
    strip_inline_markdown,
)

# 单段最大字符数：留余量低于百炼实测单条 600 字上限（manifest.MAX_TEXT_LEN）
BOOK_MAX_SEGMENT_CHARS = 480
# 章节之间插入的确定性停顿（仅章节之间；章内停顿仍由作者用 [PAUSE:x] 控制）
CHAPTER_PAUSE_SECONDS = 1.0

_HEADING = re.compile(r"^#\s+(.+?)\s*$")
# PAUSE 行正则与脚本模式共享（scriptlines.PAUSE_LINE），旧名保留为别名
_PAUSE_LINE = PAUSE_LINE
_META_SPEAKER = re.compile(
    r"^[-*]\s*(?:角色|朗读者|说话人|speaker)\s*[:：]\s*(.+)$", re.IGNORECASE
)
_META_DIRECTION = re.compile(
    r"^[-*]\s*(?:导演指令|导演|指令|direction)\s*[:：]\s*(.+)$", re.IGNORECASE
)
_SSML_BLOCK = re.compile(r"<speak\b[^>]*>.*?</speak>", re.IGNORECASE | re.DOTALL)
_WS_RUN = re.compile(r"\s+")

# 句点不视为句子边界的常见缩写（小写匹配）；不含句点的写法也在此（如 etc）
_ABBREV_WORDS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "ave", "blvd",
    "etc", "fig", "no", "vol", "pp", "inc", "ltd", "co", "corp", "approx",
    "cf", "al", "ed", "eds", "trans", "dept", "est",
}
# e.g. / i.e. / a.m. / p.m. 这类带点缩写：句点前的 token 本身含点
_DOTTED_TOKEN = re.compile(r"^(?:[A-Za-z]\.)+[A-Za-z]?$")
# 次级断句位置（长句超限时使用）：中文逗号/顿号/分号/冒号、英文 ; : 与空格
_SECONDARY = "，、；：;:"

# 闭合引号/括号：句末停连带吞下，避免下一段以引号开头
_CLOSING = "\"”’）)】」』"


class BookError(Exception):
    """章节文件结构/分段错误（消息可定位到章节）。"""


@dataclass
class Chapter:
    """一个章节的结构信息（追溯用）。"""

    title: str | None
    speaker: str | None
    direction: str | None
    body_start: int           # 正文在源文件中的字符偏移（含）
    body_end: int             # 正文结束偏移（不含）


@dataclass
class BookParse:
    """章节导入结果。"""

    segments: list[Segment]
    chapters: list[Chapter] = field(default_factory=list)


def normalize_ws(text: str) -> str:
    """空白折叠：唯一允许的"格式性"处理，绝不改动字词。"""
    return _WS_RUN.sub(" ", text).strip()


# —— 章节切分 ————————————————————————————————————————————————————————————


def _split_chapter_blocks(source: str, default_title: str) -> list[tuple[str | None, int, int]]:
    """返回 [(标题, 块起始, 块结束), ...]，块含标题行（分段时跳过）。

    优先按换页符 \\f 分章（.txt），其次按 Markdown 一级标题；
    首个标题之前若有正文，作为无标题前言章。
    """
    if "\f" in source:
        blocks: list[tuple[str | None, int, int]] = []
        pos = 0
        idx = 0
        for match in re.finditer("\f", source):
            part = source[pos:match.start()]
            blocks.append(_titled_block(part, pos, default_title, idx, allow_heading_title=True))
            pos = match.end()
            idx += 1
        part = source[pos:]
        if part.strip():
            blocks.append(_titled_block(part, pos, default_title, idx, allow_heading_title=True))
        return [b for b in blocks if source[b[1]:b[2]].strip()]

    blocks = []
    heading_spans: list[tuple[int, int, str]] = []
    for lineno_match in re.finditer(r"(?m)^(.+?)$", source):
        line = lineno_match.group(1)
        m = _HEADING.match(line.strip())
        if m:
            heading_spans.append((lineno_match.start(1), lineno_match.end(1), m.group(1).strip()))

    if not heading_spans:
        return [(default_title, 0, len(source))] if source.strip() else []

    # 前言（第一个标题之前的非空内容）
    first_start = heading_spans[0][0]
    if source[:first_start].strip():
        blocks.append((None, 0, first_start))

    for i, (hstart, hend, title) in enumerate(heading_spans):
        block_end = heading_spans[i + 1][0] if i + 1 < len(heading_spans) else len(source)
        # 块结束回退尾部空白
        blocks.append((title, hstart, block_end))
    return blocks


def _titled_block(part: str, base: int, default_title: str,
                  idx: int, allow_heading_title: bool) -> tuple[str | None, int, int]:
    title: str | None = default_title
    lines = list(re.finditer(r"(?m)^(.+?)$", part))
    for lm in lines:
        m = _HEADING.match(lm.group(1).strip())
        if m and allow_heading_title:
            title = m.group(1).strip()
            break
        if lm.group(1).strip():
            break
    return (title, base, base + len(part))


def _parse_chapter_meta(
    source: str, block_start: int, block_end: int
) -> tuple[str | None, str | None, int]:
    """解析章首元数据行，返回 (speaker, direction, body_start)。

    只吃掉章首连续的空行与两条已识别元数据；遇到第一条普通正文即停止，
    无法识别的 "- xxx: yyy" 行视为正文，绝不静默吞掉用户内容。
    """
    speaker: str | None = None
    direction: str | None = None
    body_start = block_end

    pos = block_start
    line_re = re.compile(r"(?m)^[^\r\n]*$")
    for lm in line_re.finditer(source, block_start, block_end):
        line = lm.group(0).strip()
        if not line:
            continue
        if _HEADING.match(line):
            # 块起始可能包含章节标题行（# 分章模式）；标题不是正文也不是元数据
            pos = lm.end()
            continue
        m_sp = _META_SPEAKER.match(line)
        m_dir = _META_DIRECTION.match(line)
        if m_sp:
            speaker = m_sp.group(1).strip() or speaker
            pos = lm.end()
            continue
        if m_dir:
            direction = m_dir.group(1).strip() or direction
            pos = lm.end()
            continue
        body_start = lm.start()
        break
    else:
        body_start = pos
    return speaker, direction, body_start


# —— 句子边界识别 ————————————————————————————————————————————————————————


def _period_is_boundary(text: str, i: int) -> bool:
    """text[i] == '.' 时，判断是否为句间边界（保护数字/缩写/缩略词）。"""
    n = len(text)
    # 小数 3.14 / 版本号 v3.1
    if i > 0 and i + 1 < n and text[i - 1].isdigit() and text[i + 1].isdigit():
        return False
    # 千分位 1,000. 后接数字不影响（逗号不在边界符号内）；序号 1. 后空白仍算边界
    # 找紧邻句点前的单词 token
    j = i - 1
    while j >= 0 and (text[j].isalnum() or text[j] == "."):
        j -= 1
    token = text[j + 1:i]
    if not token:
        return True
    # U.S. / e.g. / a.m. 这类"字母.字母."序列
    if _DOTTED_TOKEN.match(token + ".") or _DOTTED_TOKEN.match(token):
        return False
    last_word = token.rsplit(".", 1)[-1]
    if last_word.lower() in _ABBREV_WORDS:
        return False
    # 句点后必须是空白（可隔着闭合引号）才是句界；file.name 这种不断
    k = i + 1
    while k < n and text[k] in _CLOSING:
        k += 1
    if k < n and not text[k].isspace():
        return False
    return True


def _find_sentence_end(text: str, start: int) -> int:
    """从 start 起找下一个句子边界，返回边界之后的位置；找不到返回 len(text)。"""
    i = start
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "。！？；":
            i += 1
            while i < n and text[i] in _CLOSING:
                i += 1
            return i
        if ch in ".!?;":
            if ch != "." or _period_is_boundary(text, i):
                i += 1
                while i < n and text[i] in _CLOSING:
                    i += 1
                return i
        i += 1
    return n


def _split_long_sentence(text: str, base: int, max_len: int) -> list[tuple[int, int]]:
    """单个句子超长时按次级停点（逗号/冒号/分号/空格）贪心切；无停点才硬切。

    情绪/拟声标签（含 [clears throat] 等带空格标签）是原子单元：切点绝不落在
    标签内部；窗口内唯一停点在标签内部时，向标签后延伸找合法停点（仍不超
    max_len + 标签全长，远低于 API 600 字上限）。
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    n = len(text)
    protected = emotags.tag_spans(text)

    def _inside_tag(p: int) -> bool:
        """切点 p（前段在 p 前结束）落在某标签内部时返回 True。"""
        return any(s < p < e for s, e in protected)

    while n - pos > max_len:
        window = text[pos:pos + max_len]
        cut = -1
        for i, ch in enumerate(window):
            if ch in _SECONDARY or ch.isspace():
                if not _inside_tag(pos + i + 1):
                    cut = i  # 取窗口内最后一个可行停点
        if cut <= 0:
            # 窗口内没有标签外的停点：向标签末尾之后延伸寻找（保证不拆标签）
            ext_end = min(n, pos + max_len + 64)
            for j in range(max_len, ext_end - pos):
                ch = text[pos + j]
                if (ch in _SECONDARY or ch.isspace()) and not _inside_tag(pos + j + 1):
                    cut = j
                    break
        if cut <= 0:
            cut = max_len  # 兜底硬切（无任何停点）
            log.warning(
                f"章节分段：原文在偏移 {base + pos} 附近无句读停点，按 {max_len} 字"
                "硬切（内容未改写，仅为满足单条长度上限）。"
            )
        # 停点字符（逗号/空格）归入前一段，保证字符不丢失、可完整回溯
        spans.append((base + pos, base + pos + cut + 1))
        pos += cut + 1
        # 跳过前导空白，避免段首空格
        while pos < n and text[pos].isspace():
            pos += 1
    if pos < n and text[pos:].strip():
        spans.append((base + pos, base + n))
    return spans


# —— 正文 → 原子单元 —————————————————————————————————————————————————————


def _prose_spans(raw: str, base: int, max_len: int) -> list[tuple[int, int]]:
    """把普通正文（非 SSML）切成 (start,end) 单元，坐标基于源文件。"""
    units: list[tuple[int, int]] = []
    pos = 0
    n = len(raw)
    while pos < n:
        end = _find_sentence_end(raw, pos)
        sent_start, sent_end = pos, end
        # 跳过纯空白句
        if not raw[sent_start:sent_end].strip():
            pos = sent_end
            continue
        if len(normalize_ws(raw[sent_start:sent_end])) > max_len:
            units.extend(_split_long_sentence(
                raw[sent_start:sent_end], base + sent_start, max_len
            ))
        else:
            units.append((base + sent_start, base + sent_end))
        pos = sent_end

    # 相邻短句贪心合并到 max_len（直接合并，天然不会切开任何词）
    merged: list[tuple[int, int]] = []
    for s, e in units:
        if merged and len(normalize_ws(raw[merged[-1][0] - base:e - base])) <= max_len:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def _segment_body(
    source: str, body_start: int, body_end: int, max_len: int
) -> list[tuple[int, int, str]]:
    """正文区域 → [(start,end,kind)]，kind: 'text' | 'ssml' | 'pause'。

    SSML <speak> 块整块原子化；作者显式写的 [PAUSE:x] 行也保留为停顿。
    """
    raw = source[body_start:body_end]
    pieces: list[tuple[int, int, str]] = []
    cursor = 0
    for m in _SSML_BLOCK.finditer(raw):
        if m.start() > cursor:
            _collect_with_pauses(raw, cursor, m.start(), body_start, max_len, pieces)
        pieces.append((body_start + m.start(), body_start + m.end(), "ssml"))
        cursor = m.end()
    if cursor < len(raw):
        _collect_with_pauses(raw, cursor, len(raw), body_start, max_len, pieces)
    return pieces


def _collect_with_pauses(raw, start, end, base, max_len, pieces):
    """在一段普通正文中识别作者显式 [PAUSE:x] 行，其余按句切分（保持顺序）。"""
    region = raw[start:end]
    line_cursor = 0

    def flush_prose(upto: int) -> None:
        nonlocal line_cursor
        if upto > line_cursor and region[line_cursor:upto].strip():
            for s, e in _prose_spans(
                raw[start + line_cursor:start + upto], base + start + line_cursor, max_len
            ):
                pieces.append((s, e, "text"))

    for lm in re.finditer(r"(?m)^[^\r\n]*$", region):
        line = lm.group(0).strip()
        pm = _PAUSE_LINE.match(line)
        if pm:
            flush_prose(lm.start())  # 先冲刷停顿之前的散文，保证顺序
            pieces.append((
                base + start + lm.start(), base + start + lm.end(),
                "pause:" + pm.group(1),
            ))
            line_cursor = lm.end()
    flush_prose(len(region))


# —— 对外入口 ————————————————————————————————————————————————————————————


def normalize_speakers(speakers) -> dict[str, str]:
    """把角色入参归一化为 {显示名: 稳定ID}。

    - dict：原样（GUI：显示名→稳定 id）
    - set/list/tuple：{名字: 名字}（CLI：id 即名字）
    """
    if isinstance(speakers, dict):
        return {str(k).strip(): v for k, v in speakers.items()}
    return {str(n).strip(): str(n).strip() for n in speakers}


def split_speech_text_detailed(
    text: str, max_len: int = BOOK_MAX_SEGMENT_CHARS
) -> list[tuple[str, str]]:
    """把一段台词安全切成若干 ≤max_len 字的片段，返回 [(发送文本, 补入前缀)]。

    - SSML <speak> 块原子化，绝不拆坏标签；
    - 只在句子边界断开，保护数字/缩写/专有名词；超长句才按次级停点切，
      且切点绝不落在情绪/拟声标签内部；
    - 情绪连续性【官方语义：控制类标签作用于其后文本，直到下一个控制类标签或
      被切分为止】：同一台词块被拆成多个 API 请求时，后续片段开头补入仍生效的
      控制类标签（计入发送文本，即返回值第二元素）；拟声类一次性动作绝不复制；
    - 作者显式 [PAUSE:x] 行保留为独立片段时由调用方区分，本函数把它当普通文本
      （脚本路径中 PAUSE 行在进入本函数前已被解析为 pause 段）。
    """
    pieces = _segment_body(text, 0, len(text), max_len)
    out: list[tuple[str, str]] = []
    for s, e, kind in pieces:
        if kind.startswith("pause:"):
            continue  # 脚本路径中 PAUSE 已独立成段；公共切分函数只返回朗读文本
        piece = normalize_ws(text[s:e])
        if not piece:
            continue
        # SSML 片段不与情绪标签组合（官方未说明，pipeline 会报错拦截），不补前缀
        prefix = "" if kind == "ssml" else emotags.continuation_prefix(text[:s], piece)
        out.append((prefix + piece if prefix else piece, prefix))
    return out


def split_speech_text(text: str, max_len: int = BOOK_MAX_SEGMENT_CHARS) -> list[str]:
    """split_speech_text_detailed 的简版：只返回发送文本列表。"""
    return [p for p, _ in split_speech_text_detailed(text, max_len)]


def parse_book(
    source: str,
    valid_speakers: set[str] | list[str] | tuple[str, ...] | dict[str, str],
    *,
    source_file: str | None = None,
    default_speaker: str | None = None,
    default_title: str = "未命名章节",
    max_len: int = BOOK_MAX_SEGMENT_CHARS,
    insert_chapter_pause: bool = True,
) -> BookParse:
    """把章节源文解析为 Segment 列表（顺序编号，含章节间 PAUSE）。

    - valid_speakers：角色集合或 {显示名: 稳定ID} 映射；章首 "- 角色:" 与正文
      行内角色行均用显示名精确匹配（与脚本模式共享 match_speaker_header）。
    - default_speaker：章节未声明角色时，普通正文使用的【稳定ID】fallback；
      既无声明又无默认、且正文出现普通正文时 → BookError（纯显式对话章节可用）。
    - 正文角色优先级：行内显式角色 > 章首 "- 角色:" > default_speaker。
    """
    name_to_id = normalize_speakers(valid_speakers)
    if default_speaker is not None and default_speaker not in name_to_id.values():
        raise BookError(f"默认朗读者 {default_speaker!r} 不在角色列表中。")

    blocks = _split_chapter_blocks(source, default_title)
    chapters: list[Chapter] = []
    segments: list[Segment] = []

    for block_title, bs, be in blocks:
        meta_speaker, chapter_dir, body_start = _parse_chapter_meta(source, bs, be)
        chapter_speaker_id: str | None = None
        chapter_speaker_name: str | None = None
        if meta_speaker is not None:
            chapter_speaker_id = name_to_id.get(meta_speaker)
            if chapter_speaker_id is None:
                raise BookError(
                    f"章节 {block_title!r} 的朗读者 {meta_speaker!r} 不在角色列表中"
                    f"（合法角色：{', '.join(sorted(name_to_id))}）。"
                )
            chapter_speaker_name = meta_speaker
        elif default_speaker is not None:
            chapter_speaker_id = default_speaker
            chapter_speaker_name = next(
                (n for n, sid in name_to_id.items() if sid == default_speaker), None
            )
        # 无声明且无默认：延迟到正文真的出现普通正文时再报错（支持纯显式对话章节）
        if not source[body_start:be].strip():
            raise BookError(f"章节 {block_title!r} 正文为空。")

        chapters.append(
            Chapter(block_title, chapter_speaker_name, chapter_dir, body_start, be)
        )

        # —— 正文逐行状态机（与 ScriptParser 共享行级解析） ————————————
        # 对话块：显式注册角色行开启，延续到空行/下一角色行/PAUSE/章末；
        # 普通正文 run：无角色行的连续正文，按句切分并贪心合并，归章节默认角色。
        dlg: dict | None = None
        prose_start: int | None = None
        emo_ctx = ""   # 本章普通正文情绪上下文；对话块是屏障（清空，不跨角色延续）

        def close_prose(upto: int) -> None:
            nonlocal prose_start, emo_ctx
            if prose_start is None:
                return
            start = prose_start
            prose_start = None
            if upto <= start or not source[start:upto].strip():
                return
            if chapter_speaker_id is None:
                lineno = source.count("\n", 0, start) + 1
                raise BookError(
                    f"章节 {block_title!r} 第 {lineno} 行附近有普通正文，"
                    "但本章未声明朗读者（章首写 “- 角色: 角色名”），且未提供默认朗读者。"
                )
            region_prefix = emo_ctx
            for s, e, kind in _segment_body(source, start, upto, max_len):
                raw_piece = source[s:e]
                if kind == "ssml":
                    segments.append(Segment(
                        index=0, kind="speech", speaker=chapter_speaker_id,
                        speaker_name=chapter_speaker_name,
                        chapter_direction=chapter_dir, origin="prose",
                        text=normalize_ws(raw_piece), raw_text=raw_piece.strip(),
                        chapter=block_title, source_file=source_file,
                        char_start=s, char_end=e,
                    ))
                else:
                    text = normalize_ws(strip_inline_markdown(raw_piece))
                    if not text:
                        continue
                    # 情绪连续：同一章节同一角色正文被拆段时，后续片段补入
                    # 仍生效的控制类标签（对话块出现即重置，绝不跨角色延续）
                    prefix = emotags.continuation_prefix(region_prefix, text)
                    segments.append(Segment(
                        index=0, kind="speech", speaker=chapter_speaker_id,
                        speaker_name=chapter_speaker_name,
                        chapter_direction=chapter_dir, origin="prose",
                        text=prefix + text if prefix else text,
                        injected_prefix=prefix or None,
                        raw_text=raw_piece.strip(),
                        chapter=block_title, source_file=source_file,
                        char_start=s, char_end=e,
                    ))
                region_prefix += raw_piece
            emo_ctx = region_prefix

        def close_dialogue() -> None:
            nonlocal dlg
            if dlg is None:
                return
            block = dlg
            dlg = None
            raw = _join_lines([ln for _, _, ln in block["lines"]]).strip()
            text = _join_lines(
                [strip_inline_markdown(ln) for _, _, ln in block["lines"]]
            ).strip()
            if not text:
                raise BookError(
                    f"章节 {block_title!r} 第 {block['lineno']} 行："
                    f"{block['name']} 的台词为空。"
                )
            # 追溯坐标只覆盖台词内容（不含 "角色名:" 头部），切片与 raw_text 对应
            span_start = block["spans"][0][0]
            span_end = block["spans"][-1][1]
            detailed = (
                split_speech_text_detailed(text, max_len)
                if len(text) > max_len else [(text, "")]
            )
            for piece, injected in detailed:
                segments.append(Segment(
                    index=0, kind="speech", speaker=block["id"],
                    speaker_name=block["name"], direction=block["direction"],
                    rate=block["overrides"].get("rate"),
                    pitch=block["overrides"].get("pitch"),
                    volume=block["overrides"].get("volume"),
                    chapter_direction=chapter_dir, origin="dialogue",
                    text=piece, raw_text=raw,
                    injected_prefix=injected or None,
                    chapter=block_title, source_file=source_file,
                    char_start=span_start, char_end=span_end,
                ))

        line_re = re.compile(r"(?m)^[^\r\n]*$")
        base_lineno = source.count("\n", 0, body_start) + 1
        for i, lm in enumerate(line_re.finditer(source, body_start, be)):
            lineno = base_lineno + i
            # finditer(source, pos, endpos) 的匹配坐标是全文绝对偏移，直接使用
            ls, le = lm.start(), lm.end()
            line = lm.group(0).strip()

            if not line:
                close_dialogue()   # 空行结束对话块；不结束正文 run（保持跨段合并）
                continue

            mp = PAUSE_LINE.match(line)
            if mp:
                close_dialogue()
                close_prose(ls)
                seconds = float(mp.group(1))
                if not 0 < seconds <= 10:
                    raise BookError(
                        f"章节 {block_title!r} 中 [PAUSE:{seconds:g}] 超出允许范围 (0, 10]。"
                    )
                segments.append(Segment(
                    index=0, kind="pause", pause_seconds=seconds,
                    chapter=block_title, source_file=source_file,
                    char_start=ls, char_end=le,
                ))
                emo_ctx += source[ls:le] + "\n"   # 旧版语义：上下文包含 PAUSE 行原文
                prose_start = le
                continue

            header = match_speaker_header(line, name_to_id)
            if header is not None:
                close_dialogue()
                close_prose(ls)
                emo_ctx = ""   # 对话块 = 情绪屏障：不继承前文，也不外泄给后文
                dlg = {
                    "id": header["speaker"], "name": header["speaker_name"],
                    "direction": header["direction"],
                    "overrides": header["overrides"],
                    "lines": [], "lineno": lineno, "spans": [],
                }
                if header["inline_text"]:
                    dlg["lines"].append((ls, le, header["inline_text"]))
                    # 台词内容 span 从"行尾内容"倒推：不含 "角色名(...):" 头部，
                    # 使切片与 raw_text（纯台词）对应
                    trailing = len(lm.group(0)) - len(lm.group(0).rstrip())
                    content_end = le - trailing
                    dlg["spans"].append(
                        (content_end - len(header["inline_text"]), content_end)
                    )
                continue

            if dlg is not None:
                # 对话块延续行（与脚本模式一致：空行前同属一个台词块）；
                # 疑似未注册角色行不吞内容、告警后并入台词
                susp = looks_like_unregistered_speaker(line)
                if susp:
                    log.warning(
                        f"章节 {block_title!r} 第 {lineno} 行：疑似角色声明 {susp!r}"
                        f"（未注册或格式错误），已并入 {dlg['name']} 的台词。"
                    )
                dlg["lines"].append((ls, le, line))
                dlg["spans"].append((ls, le))
                continue

            # 普通正文行 → 章节默认角色；疑似角色声明给 WARNING 后仍按正文处理
            susp = looks_like_unregistered_speaker(line)
            if susp:
                log.warning(
                    f"章节 {block_title!r} 第 {lineno} 行：疑似角色声明 {susp!r}"
                    "（未注册或格式错误），已按普通正文处理"
                    "（如需多角色请先在角色表注册，或修正书写格式）。"
                )
            if prose_start is None:
                prose_start = ls

        close_dialogue()
        close_prose(be)

        # 章节之间插入确定性停顿（末章之后不加）
        if insert_chapter_pause and (bs, be) != blocks[-1][1:]:
            segments.append(Segment(index=0, kind="pause",
                                    pause_seconds=CHAPTER_PAUSE_SECONDS,
                                    chapter=block_title, source_file=source_file))

    for i, seg in enumerate(segments, start=1):
        seg.index = i

    log.info(
        f"章节文件解析完成：{len(chapters)} 章 / {len(segments)} 段"
        f"（含 {sum(1 for s in segments if s.kind == 'pause')} 停顿）。"
    )
    return BookParse(segments=segments, chapters=chapters)
