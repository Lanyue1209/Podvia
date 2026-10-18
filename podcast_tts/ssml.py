"""SSML 安全校验模块（阶段 3）。

只做【校验】，不做任何文本改写：合法 SSML 原文透传给百炼；非法输入直接报错，
绝不把普通文本自动包装/改写成 SSML。

官方依据（2026-10-01 核实）：
- 《SSML 与 LaTeX》https://help.aliyun.com/zh/model-studio/ssml-latex-user-guide
  （更新 2026-09-18）：HTTP API 需 input.enable_ssml=true；支持模型含
  qwen-audio-3.1-tts-flash / qwen-audio-3.0-tts-flash / qwen-audio-3.0-tts-plus
  及 CosyVoice 系列；音色限复刻音色或标注支持 SSML 的系统音色。
- 《SSML标记语言介绍》
  中文 https://help.aliyun.com/zh/model-studio/introduction-to-cosyvoice-ssml-markup-language
  英文 https://help.aliyun.com/en/isi/developer-reference/introduction-to-cosyvoice-ssml-markup-language
  （更新 2026-08-26）：标签与属性规格。

【重要结论】阿里云（CosyVoice/Qwen-Audio）SSML 是 W3C SSML 1.0 的子集，
**不支持 <prosody> 与 <emphasis> 标签**。语速/音调/音量通过 <speak> 的
rate/pitch/volume 属性控制（百炼 Model Studio 下取值 0.5~2 / 0.5~2 / 0~100，
官方英文页明确说明会覆盖同名 HTTP 请求参数）。

本阶段最小白名单（均官方明确列出）：
    <speak>   根节点（唯一），属性白名单 rate/pitch/volume
    <break>   空标签，time="Ns"(1-10) 或 "Nms"(50-10000)；连续停顿累计 ≤10s
    <phoneme> alphabet="py|cmu" ph="..."，仅含文本
    <sub>     alias="..."，仅含文本
    <say-as>  interpret-as=12 个官方枚举，仅含文本
未纳入：<soundEvent>(外链音频)、<s>/<w>、effect/bgm 等（最小白名单，后续再扩）。
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

# —— 安全：XML 解析前的硬性禁区 ————————————————————————————————————————
# ElementTree 不加载外部实体，但内部 DTD 实体仍可能被 expat 展开（如
# billion laughs）。在解析前直接拒绝一切 DOCTYPE/ENTITY 声明，杜绝 XXE
# 与实体膨胀攻击。
_FORBIDDEN_DECL = re.compile(r"<!\s*(doctype|entity)\b", re.IGNORECASE)

# 标签白名单
SPEAK = "speak"
BREAK = "break"
PHONEME = "phoneme"
SUB = "sub"
SAY_AS = "say-as"
_ALLOWED_TAGS = frozenset({SPEAK, BREAK, PHONEME, SUB, SAY_AS})

# <speak> 属性白名单（voice/effect/bgm 等本阶段不开放）
_SPEAK_ATTRS = frozenset({"rate", "pitch", "volume"})
# say-as 官方枚举（2026-08-26 英文标签页）
_SAY_AS_TYPES = frozenset({
    "cardinal", "digits", "telephone", "name", "address", "id",
    "characters", "punctuation", "date", "time", "currency", "measure",
})
_PHONEME_ALPHABETS = frozenset({"py", "cmu"})
_TEXT_ONLY_TAGS = frozenset({PHONEME, SUB, SAY_AS})

_BREAK_TIME = re.compile(r"^(\d+)(ms|s)$")
# 连续多个 break 累计上限 10s（官方）
_BREAK_TOTAL_CAP_MS = 10_000

# 片段级文本模式（Segment.text_mode）
TEXT_MODE_AUTO = "auto"    # 按内容自动判断（默认）
TEXT_MODE_PLAIN = "plain"  # 强制普通文本（不做 SSML 校验，原文发送）
TEXT_MODE_SSML = "ssml"    # 强制 SSML（必须通过校验，失败则该片段报错）
TEXT_MODES = (TEXT_MODE_AUTO, TEXT_MODE_PLAIN, TEXT_MODE_SSML)

# auto 判定：整段正文就是一份 <speak>…</speak> 文档（形状匹配；XML 合法性由
# validate_ssml 把关——形状像但格式错误 → 明确报错，绝不静默退回普通文本）。
# 文本里偶尔出现 <xxx> 不会命中（必须以 <speak 根开头、以 </speak> 收尾）。
_SPEAK_DOC_RE = re.compile(
    r"^\s*<speak\b[^>]*>.*</speak>\s*$", re.IGNORECASE | re.DOTALL
)
# 形似 SSML：出现 <speak 开标签但不满足完整文档形状（如未闭合）。
# 同样视为 SSML 意图，交由 validate_ssml 报"SSML 格式错误"，
# 绝不静默退回普通文本发送（否则用户以为在发 SSML 实际发的是原文）。
_SPEAK_OPEN_ANY_RE = re.compile(r"<speak\b", re.IGNORECASE)


class SSMLError(ValueError):
    """SSML 校验错误；消息必须能定位到具体标签/属性/原因。"""


def looks_like_ssml(text: str | None) -> bool:
    """文本是否应按 SSML 处理（形状判断，不做 XML 校验）。

    - 完整 <speak>…</speak> 文档 → True；
    - 含 <speak 开标签但文档不完整（如未闭合）→ True（SSML 意图，
      XML 错误由 validate_ssml 明确报"SSML 格式错误"，绝不静默按 plain 发送）；
    - 偶尔出现 <xxx> 等普通尖括号内容（无 <speak）→ False，按普通文本。
    """
    if not text:
        return False
    if _SPEAK_DOC_RE.match(text):
        return True
    return _SPEAK_OPEN_ANY_RE.search(text) is not None


def resolve_text_mode(
    text: str | None,
    text_mode: str | None = None,
    *,
    role_ssml: "bool | None" = None,
    global_ssml: bool = False,
) -> str:
    """解析片段文本模式 → "plain" | "ssml"（逐片段独立，绝不整章统一）。

    优先级（高 → 低）：
    1. text_mode 显式 "ssml"/"plain"（片段管理中的强制选择，最高）；
    2. auto（含 None/空/未知值）：
       a. 角色级显式 ssml 开关（旧语义：True=该角色全部按 SSML 校验，
          False=该角色永不使用 SSML）→ 直接生效，与历史行为一致；
       b. 整段正文是 <speak>…</speak> 文档（或含 <speak 开标签）→ ssml；
       c. 全局 enable_ssml（旧项目兼容）→ ssml；
       d. 否则 plain。
    强制/auto 判定为 ssml 后是否合法交给 validate_ssml——形状像但格式错误
    会明确报错，绝不静默退回普通文本。
    """
    mode = (text_mode or "").strip().lower()
    if mode == TEXT_MODE_SSML:
        return TEXT_MODE_SSML
    if mode == TEXT_MODE_PLAIN:
        return TEXT_MODE_PLAIN
    if role_ssml is not None:
        return TEXT_MODE_SSML if role_ssml else TEXT_MODE_PLAIN
    if looks_like_ssml(text):
        return TEXT_MODE_SSML
    return TEXT_MODE_SSML if global_ssml else TEXT_MODE_PLAIN


@dataclass(frozen=True)
class ValidatedSSML:
    """校验结果（原文不做任何改写）。"""

    text: str
    has_rate: bool
    has_pitch: bool
    has_volume: bool


def _fail(tag: str, msg: str) -> None:
    raise SSMLError(f"SSML 标签 <{tag}> {msg}")


def _speak_rate(tag: str, raw: str) -> str:
    # 百炼 Model Studio：0.5~2 倍速（官方英文页 Important 说明）
    try:
        v = float(raw)
    except ValueError:
        _fail(tag, f"rate 必须是 0.5~2 的数字（当前 {raw!r}）。")
        raise  # pragma: no cover
    if not (0.5 <= v <= 2.0):
        _fail(tag, f"rate 取值范围为 0.5~2（当前 {raw!r}）。")
    return raw


def _speak_pitch(tag: str, raw: str) -> str:
    try:
        v = float(raw)
    except ValueError:
        _fail(tag, f"pitch 必须是 0.5~2 的数字（当前 {raw!r}）。")
        raise  # pragma: no cover
    if not (0.5 <= v <= 2.0):
        _fail(tag, f"pitch 取值范围为 0.5~2（当前 {raw!r}）。")
    return raw


def _speak_volume(tag: str, raw: str) -> str:
    try:
        v = int(raw)
    except ValueError:
        _fail(tag, f"volume 必须是 0~100 的整数（当前 {raw!r}）。")
        raise  # pragma: no cover
    if not (0 <= v <= 100):
        _fail(tag, f"volume 取值范围为 0~100（当前 {raw!r}）。")
    return raw


def _break_ms(tag: str, raw: str | None) -> int:
    """返回毫秒数；无属性时官方默认 1000ms。"""
    if raw is None:
        return 1000
    m = _BREAK_TIME.match(raw.strip())
    if not m:
        _fail(tag, f'time 格式必须是 "Ns"(1~10) 或 "Nms"(50~10000)（当前 {raw!r}）。')
    n, unit = int(m.group(1)), m.group(2)
    if unit == "s":
        if not (1 <= n <= 10):
            _fail(tag, f"time 秒数必须为 1~10 的整数（当前 {n}）。")
        return n * 1000
    if not (50 <= n <= 10_000):
        _fail(tag, f"time 毫秒数必须为 50~10000 的整数（当前 {n}）。")
    return n


def validate_ssml(text: str) -> ValidatedSSML:
    """校验 SSML 文本，返回属性标记；任何不合法情形抛 SSMLError（不改写原文）。

    安全保证：拒绝 DOCTYPE/ENTITY；仅白名单标签与属性；结构（空标签/
    纯文本子节点）强校验。
    """
    if not isinstance(text, str) or not text.strip():
        raise SSMLError("SSML 文本为空。")
    if _FORBIDDEN_DECL.search(text):
        raise SSMLError(
            "SSML 不允许 DOCTYPE/ENTITY 声明（防 XXE/实体注入），请移除 <!DOCTYPE> / <!ENTITY>。"
        )
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise SSMLError(f"SSML 不是合法 XML：{exc}") from None

    if root.tag != SPEAK:
        raise SSMLError(
            f"SSML 根节点必须是 <speak>，当前为 <{root.tag}>；普通文本请勿开启 SSML 模式。"
        )

    # speak 属性白名单
    unknown = set(root.attrib) - _SPEAK_ATTRS
    if unknown:
        _fail(SPEAK, f"存在不支持的属性 {sorted(unknown)}；本阶段仅开放 rate/pitch/volume"
                     "（voice/effect/bgm 等暂不支持；另注意阿里云不支持 prosody/emphasis）。")
    has_rate = "rate" in root.attrib
    has_pitch = "pitch" in root.attrib
    has_volume = "volume" in root.attrib
    if has_rate:
        _speak_rate(SPEAK, root.attrib["rate"])
    if has_pitch:
        _speak_pitch(SPEAK, root.attrib["pitch"])
    if has_volume:
        _speak_volume(SPEAK, root.attrib["volume"])

    # 必须有可朗读内容（非空白文本或 break 等标签）
    if not (root.text and root.text.strip()) and len(root) == 0:
        _fail(SPEAK, "内容为空，至少需要文本或 <break/>。")

    # 白名单下结构必为平铺：speak 下直接跟 break/phoneme/sub/say-as 与文本；
    # 后三者不允许嵌套，break 为空标签。因此只需顺序遍历直接子节点。
    run_ms = 0
    if root.text and root.text.strip():
        run_ms = 0
    for child in root:
        if child.tag not in _ALLOWED_TAGS:
            raise SSMLError(
                f"不支持的 SSML 标签 <{child.tag}>（白名单：speak/break/phoneme/sub/say-as）。"
                "注意：阿里云 Qwen-Audio/CosyVoice 不支持 <prosody>/<emphasis>，"
                "请改用 <speak rate/pitch/volume> 或 HTTP 的 rate/pitch/volume 参数。"
            )
        if child.tag == BREAK:
            _check_attrs(child, {"time"}, BREAK)
            if list(child) or (child.text and child.text.strip()):
                _fail(BREAK, "是空标签，不能包含文本或子标签。")
            run_ms += _break_ms(BREAK, child.attrib.get("time"))
            if run_ms > _BREAK_TOTAL_CAP_MS:
                _fail(BREAK, f"连续停顿累计不得超过 10 秒（当前累计 {run_ms}ms）。")
        else:
            _validate_text_tag(child)
            run_ms = 0  # 含文本标签打断连续停顿
        if child.tail and child.tail.strip():
            run_ms = 0

    return ValidatedSSML(text=text, has_rate=has_rate, has_pitch=has_pitch,
                         has_volume=has_volume)


def _validate_text_tag(node: ET.Element) -> None:
    if node.tag == PHONEME:
        _check_attrs(node, {"alphabet", "ph"}, PHONEME, required={"alphabet", "ph"})
        if node.attrib["alphabet"] not in _PHONEME_ALPHABETS:
            _fail(PHONEME, f'alphabet 仅支持 "py"/"cmu"（当前 {node.attrib["alphabet"]!r}）。')
        if not node.attrib["ph"].strip():
            _fail(PHONEME, "ph 不能为空。")
    elif node.tag == SUB:
        _check_attrs(node, {"alias"}, SUB, required={"alias"})
        if not node.attrib["alias"].strip():
            _fail(SUB, "alias 不能为空。")
    elif node.tag == SAY_AS:
        _check_attrs(node, {"interpret-as"}, SAY_AS, required={"interpret-as"})
        value = node.attrib["interpret-as"]
        if value not in _SAY_AS_TYPES:
            _fail(SAY_AS, f"interpret-as={value!r} 不在官方支持集合内："
                          f"{sorted(_SAY_AS_TYPES)}。")
    if list(node):
        _fail(node.tag, f"只能包含文本，不能嵌套标签（发现 {[c.tag for c in node]}）。")
    if not (node.text or "").strip():
        _fail(node.tag, "必须包含非空文本。")


def _check_attrs(node: ET.Element, allowed: set[str], tag: str,
                 required: set[str] | None = None) -> None:
    required = required or set()
    keys = set(node.attrib)
    unknown = keys - allowed
    if unknown:
        _fail(tag, f"存在不支持的属性 {sorted(unknown)}（允许：{sorted(allowed)}）。")
    missing = required - keys
    if missing:
        _fail(tag, f"缺少必填属性 {sorted(missing)}。")


def _require_text_only(node: ET.Element) -> None:
    """phoneme/sub/say-as 只能包含文本，不能嵌套标签。"""
    if list(node):
        nested = [c.tag for c in node]
        _fail(node.tag, f"只能包含文本，不能嵌套标签（发现 {nested}）。")
    if not (node.text or "").strip():
        _fail(node.tag, "必须包含非空文本。")
