"""行级共享解析：脚本输入（文本框/CLI）与章节正文共用的同一套行语法。

这里只放"一行文本"级别的解析能力，供 parser.ScriptParser 与 book.parse_book
复用，保证同一种写法（角色头、行内参数、PAUSE、Markdown 清洗、疑似角色识别）
在两种入口下行为一致：

- match_speaker_header：已注册角色的说话人行（Name: / Name(指令; rate=1.2): /
  [名字] 台词，中英文冒号与括号均可）
- extract_inline_params：导演指令括号内的 rate/pitch/volume 行内覆盖
- PAUSE_LINE：整行 [PAUSE:x] 停顿指令
- strip_inline_markdown：行内 Markdown 格式符清洗（只删格式符，不改文字）
- _join_lines：多行台词合并（中文直接连接、西文加空格）
- looks_like_unregistered_speaker：疑似角色声明但未注册/格式错误的保守识别

纯本地规则，无网络、无 LLM。
"""
from __future__ import annotations

import re

from podcast_tts import capabilities as caps
from podcast_tts import emotags

# PAUSE 行：整行仅为 [PAUSE:x]，单位秒，允许范围 (0, 10]
PAUSE_LINE = re.compile(r"^\[PAUSE:(\d+(?:\.\d+)?)\]$", re.IGNORECASE)

# 方括号说话人：[名字] 文本（名字不含方括号/换行）
BRACKET_SPEAKER = re.compile(r"^\[(?P<name>[^\[\]\r\n]{1,64})\]\s*(?P<inline>.*)$")
# 角色名尾部的导演指令括号：Name(指令) / Name（指令）
TRAIL_PAREN = re.compile(r"^(?P<name>.*?)[\(（](?P<direction>.*)[\)）]\s*$")
_FIRST_COLON = re.compile(r"[:：]")

# 行内参数标记：rate=1.2 / pitch=0.9 / volume=80（边界：前面不能是字母数字）
_INLINE_PARAM = re.compile(
    r"(?<![A-Za-z0-9_])(rate|pitch|volume)\s*=\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
# 抽取参数后用于清理残余分隔符
_SEP_RUN = re.compile(r"[\s;；,，、]{2,}")

# 疑似角色名中不允许出现的字符（句读/引号/书名号/括号/URL 痕迹）
_NAME_FORBIDDEN = set("，。！？；、,;!?\"'“”‘’《》<>()（）[]【】/\\@#\t")


# —— 行内 Markdown 清洗 ——————————————————————————————————————
_CODE_SPAN = re.compile(r"`([^`]*)`")
_BOLD_ITALIC = re.compile(r"(\*\*\*|___)(?=\S)(.+?)(?<=\S)\1")
_BOLD = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")
_BOLD_US = re.compile(r"(?<!_)__(?=\S)(.+?)(?<=\S)__(?!_)")
_ITALIC = re.compile(r'(?<!\*)\*(?=\S)([^"]+?)(?<=\S)\*(?!\*)')
_ITALIC_US = re.compile(r'(?<![\w_])_(?=\S)([^"]+?)(?<=\S)_(?![\w_])')


def strip_inline_markdown(text: str) -> str:
    """剥离行内 Markdown 格式控制符（粗斜体/粗体/斜体/行内代码装饰符）。

    只删格式符本身，文字、标点、空白一律保留；无配对符号时保持原样。
    """
    placeholders: dict[str, str] = {}

    def _keep_code(match: re.Match) -> str:
        key = f"\x00{len(placeholders)}\x00"
        placeholders[key] = match.group(1)
        return key

    text = _CODE_SPAN.sub(_keep_code, text)  # 行内代码：装饰符去掉，内容原样保留
    text = _BOLD_ITALIC.sub(r"\2", text)
    text = _BOLD.sub(r"\1", text)
    text = _BOLD_US.sub(r"\1", text)
    text = _ITALIC.sub(r"\1", text)
    text = _ITALIC_US.sub(r"\1", text)
    for key, content in placeholders.items():
        text = text.replace(key, content)
    return text


# —— 导演指令括号内的行内参数 ————————————————————————————————————————


def extract_inline_params(raw_direction: str):
    """从导演指令文本中抽取 rate/pitch/volume 行内覆盖。

    返回 (清洗后的指令或 None, {"rate": float, ...})。数值超范围抛
    caps.validate_* 的 ConfigError（调用方负责补充位置信息）。
    不认识的 "xxx=yyy" 原样保留在指令中。
    """
    overrides: dict[str, float | int] = {}

    def _sub(m: re.Match) -> str:
        key = m.group(1).lower()
        raw = m.group(2)
        if key == "volume":
            overrides[key] = caps.validate_volume(int(float(raw)))
        elif key == "rate":
            overrides[key] = caps.validate_rate(float(raw))
        else:
            overrides[key] = caps.validate_pitch(float(raw))
        return " "

    cleaned = _INLINE_PARAM.sub(_sub, raw_direction)
    cleaned = _SEP_RUN.sub(" ", cleaned).strip(" \t;；,，、")
    return (cleaned or None), overrides


# —— 说话人行识别 ——————————————————————————————————————————————


def match_speaker_header(line: str, name_to_id: dict[str, str]) -> dict | None:
    """若该行是"已注册角色"的说话人行，返回解析结果；否则返回 None。

    返回字段：speaker(id)/speaker_name/direction/overrides/inline_text。
    本函数只认已注册名字（精确匹配），不做猜测。
    """
    # 方括号形式：[名字] 文本
    mb = BRACKET_SPEAKER.match(line)
    if mb:
        name = mb.group("name").strip()
        if name in name_to_id:
            return {
                "speaker": name_to_id[name], "speaker_name": name,
                "direction": None, "overrides": {},
                "inline_text": mb.group("inline").strip(),
            }
        return None

    mc = _FIRST_COLON.search(line)
    if not mc:
        return None
    prefix = line[:mc.start()]
    inline_text = line[mc.end():].strip()

    direction = None
    name_part = prefix.strip()
    mp = TRAIL_PAREN.match(name_part)
    if mp:
        name_part = mp.group("name").strip()
        direction = mp.group("direction").strip() or None
    # 无括号但整段就是注册名（无指令）
    if name_part not in name_to_id and prefix.strip() in name_to_id:
        name_part = prefix.strip()
    if name_part not in name_to_id:
        return None
    overrides: dict[str, float | int] = {}
    if direction:
        direction, overrides = extract_inline_params(direction)
    return {
        "speaker": name_to_id[name_part], "speaker_name": name_part,
        "direction": direction, "overrides": overrides,
        "inline_text": inline_text,
    }


def looks_like_unregistered_speaker(line: str) -> str | None:
    """行首疑似角色标记但未注册时，返回候选名字；否则 None。

    保守策略：只有"短、无空白、无句读标点、含字母/汉字"的冒号前缀，或
    方括号短标记才算疑似，避免误伤普通叙述。

    特殊放行：行首是官方情绪/拟声标签（如 [laughing]）不算角色名；
    方括号内内容含空格且不是官方标签的仍按普通文本处理（不做未注册角色报错）。
    """
    mb = BRACKET_SPEAKER.match(line)
    if mb and not PAUSE_LINE.match(line):
        name = mb.group("name").strip()
        # 先放行官方标签（如 [laughing] 独占一行）
        if f"[{name}]" in emotags.ALL_TAGS:
            return None
        if 0 < len(name) <= 32 and " " not in name and name != "全局配置":
            return name
        # 方括号内含空格但不是官方标签：可能是普通英文短语，不做未注册报错
        if " " in name and f"[{name}]" not in emotags.ALL_TAGS:
            return None
        return None
    mc = _FIRST_COLON.search(line)
    if not mc:
        return None
    candidate = mc.string[:mc.start()].strip()
    mp = TRAIL_PAREN.match(candidate)
    if mp:
        candidate = mp.group("name").strip()
    if not candidate or len(candidate) > 32:
        return None
    if any(ch.isspace() for ch in candidate):
        return None
    # 含括号（尤其是未闭合括号）→ 形似写坏的角色头，按疑似处理
    if any(ch in "()（）" for ch in candidate):
        return candidate
    if any(ch in _NAME_FORBIDDEN for ch in candidate):
        return None
    if not any(ch.isalnum() or "一" <= ch <= "鿿" for ch in candidate):
        return None
    return candidate


# 旧名兼容（parser 原内部函数名）
_looks_like_unregistered_speaker = looks_like_unregistered_speaker


# —— 多行台词合并 ————————————————————————————————————————————
_CJK_END = re.compile(r"(?:[一-鿿㐀-䶿]|[，。！？；：、）】》”’…—])$")
_CJK_START = re.compile(r"^(?:[一-鿿㐀-䶿]|[（【《“‘])")


def _join_lines(lines: list[str]) -> str:
    """合并多行台词：前一行以中文（汉字/中文标点）结尾且当前行以中文开头时直接连接，
    否则用单个空格连接。这是格式性处理，不改动任何文字与标点。"""
    joined = ""
    for line in lines:
        if not joined:
            joined = line
        elif _CJK_END.search(joined) and _CJK_START.match(line):
            joined += line
        else:
            joined += " " + line
    return joined
