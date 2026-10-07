"""情绪与人声拟声标签（Qwen-Audio-TTS 文本内嵌标签）的唯一定义与校验。

来源【官方】：阿里云百炼《非实时语音合成》用户指南「情感与富语言标签」一节
https://help.aliyun.com/zh/model-studio/non-realtime-tts-user-guide
（官方更新时间 2026-09-28；本仓库核对日期 2026-10-03，逐字录入下方两张表）

官方明确的事实：
- 支持模型：仅 qwen-audio-3.1-tts-flash、qwen-audio-3.0-tts-plus、
  qwen-audio-3.0-tts-flash。其它模型（Qwen-TTS / CosyVoice / MiniMax / 未知模型）
  一律按不支持处理，不静默透传。
- 控制类标签（23 个）：设定情感/风格，作用于其后所有文本，直到遇到下一个
  控制类标签，"或因句子较长被自动切分为止"。因此 Podvia 手动把长台词拆成多个
  API 请求时，会在后续请求开头补入仍生效的控制类标签以保持情绪连续（只补入
  已确认的控制类标签，并计入实际发送文本）。
- 富语言类标签（7 个）：在当前位置插入一次性拟声效果，不影响前后文本。
  自动分段绝不把拟声标签复制到其它片段（避免重复生成笑声/咳嗽）。
- 官方未说明标签与 SSML（enable_ssml）能否组合 → Podvia 不支持同一片段混用，
  生成前明确报错。
- 官方未规定标签与 instruction/rate/pitch/volume 的优先级 → Podvia 原样同时
  发送，不宣称保证某种叠加效果。
"""
from __future__ import annotations

import re

# —— 官方标签表（2026-10-03 核对，逐字录入；不为凑数增删） ————————————————

# 控制类标签：作用于其后所有文本，直到下一个控制类标签（情绪/风格状态）
CONTROL_TAGS: dict[str, str] = {
    "[sad]": "悲伤",
    "[amazed]": "惊叹",
    "[deep and loud shouting]": "深沉大声呐喊",
    "[trembling]": "颤抖",
    "[angry]": "愤怒",
    "[excited]": "兴奋",
    "[sarcastic]": "讽刺",
    "[curious]": "好奇",
    "[like dracula]": "德古拉风格（低沉、阴森）",
    "[bored]": "无聊",
    "[tired]": "疲惫",
    "[scornful]": "轻蔑",
    "[shouting]": "大喊",
    "[asmr]": "ASMR 轻柔耳语",
    "[panicked]": "恐慌",
    "[mischievously]": "调皮",
    "[empathetic]": "共情",
    "[whispers]": "耳语",
    "[reluctantly]": "不情愿",
    "[crying]": "哭泣",
    "[serious]": "严肃",
    "[very slowly]": "非常缓慢地说话",
    "[very fast]": "非常快速地说话",
}

# 富语言类标签：在当前位置插入一次性拟声效果，不影响前后文本（人声动作）
PARALINGUISTIC_TAGS: dict[str, str] = {
    "[gasp]": "倒吸一口气",
    "[sighing]": "叹息",
    "[clears throat]": "清嗓",
    "[giggles]": "咯咯笑",
    "[laughing]": "大笑",
    "[cough]": "咳嗽",
    "[snorts]": "哼声、嗤笑",
}

ALL_TAGS: dict[str, str] = {**CONTROL_TAGS, **PARALINGUISTIC_TAGS}

# 官方标签匹配（长的在前，保证 [deep and loud shouting] 优先；标签拼写区分大小写，
# 官方全部为小写，拼写错误按"未识别标签"报错，不静默放行）
_TAG_RE = re.compile(
    "|".join(re.escape(t) for t in sorted(ALL_TAGS, key=len, reverse=True))
)

# 疑似想写标签但不在官方表的方括号内容：ASCII 字母开头、仅含字母/数字/空格、
# 长度有限（[PAUSE:x]、中文方括号、纯数字编号等不在此列，避免误伤普通文本）
_TAG_CANDIDATE_RE = re.compile(r"\[([A-Za-z][A-Za-z0-9 ]{0,30})\]")
_PAUSE_INNER_RE = re.compile(r"^PAUSE:\d+(?:\.\d+)?$", re.IGNORECASE)

_SSML_SPEAK_RE = re.compile(r"<\s*speak\b", re.IGNORECASE)


def iter_tags(text: str):
    """产出文本中所有官方标签：(start, end, tag, kind)，kind ∈ control | vocal。"""
    for m in _TAG_RE.finditer(text):
        tag = m.group(0)
        kind = "control" if tag in CONTROL_TAGS else "vocal"
        yield m.start(), m.end(), tag, kind


def has_tags(text: str) -> bool:
    return _TAG_RE.search(text) is not None


def tag_spans(text: str) -> list[tuple[int, int]]:
    """官方标签的 (start, end) 区间，供分段器保护（绝不从标签内部切开）。"""
    return [(m.start(), m.end()) for m in _TAG_RE.finditer(text)]


def leading_tag(line: str) -> str | None:
    """行首是官方标签时返回该标签（含 [laughing] 独占一行的情况），否则 None。"""
    m = _TAG_RE.match(line)
    return m.group(0) if m else None


def unknown_tag_candidates(text: str) -> list[tuple[int, str]]:
    """疑似标签拼写错误的方括号内容：[(位置, 内容)]。官方标签与 [PAUSE:x] 除外。"""
    out: list[tuple[int, str]] = []
    for m in _TAG_CANDIDATE_RE.finditer(text):
        inner = m.group(1)
        if f"[{inner}]" in ALL_TAGS or _PAUSE_INNER_RE.match(inner):
            continue
        out.append((m.start(), inner))
    return out


def active_emotion_before(text: str, pos: int) -> str | None:
    """text[:pos] 中最后一个控制类标签（该位置仍然生效的情绪状态）；无则 None。"""
    active: str | None = None
    for start, _end, tag, kind in iter_tags(text):
        if start >= pos:
            break
        if kind == "control":
            active = tag
    return active


def continuation_prefix(prior_text: str, piece: str) -> str:
    """长台词被拆分后，后续片段需补入的情绪前缀（不含时返回 ""）。

    - prior_text：同一角色同一台词块/章节正文中、本片段之前的全部原文；
    - 只补入控制类标签（情绪状态），拟声类一次性动作绝不复制；
    - 片段本身已以控制类标签开头时不再补入（新状态覆盖旧状态）。
    """
    lead = leading_tag(piece)
    if lead is not None and lead in CONTROL_TAGS:
        return ""  # 片段自带控制类标签开头：新状态覆盖旧状态，不再补入
    active = active_emotion_before(prior_text, len(prior_text))
    return f"{active} " if active else ""


# —— 校验（生成前整任务调用；返回错误描述列表，空列表 = 通过） ————————————

_SOURCE = "（官方《非实时语音合成》页，2026-10-03 核对）"


def check_text_tags(text: str, *, where: str) -> list[str]:
    """标签拼写校验：疑似写错的标签直接报错，给出最接近的官方写法提示。"""
    import difflib

    errors: list[str] = []
    officials = list(ALL_TAGS)
    for pos, inner in unknown_tag_candidates(text):
        suggestion = ""
        lowered = f"[{inner.lower()}]"
        if lowered in ALL_TAGS:
            suggestion = f"。官方标签为小写，是否想写 {lowered}（{ALL_TAGS[lowered]}）"
        else:
            close = difflib.get_close_matches(
                lowered, officials, n=1, cutoff=0.55)
            if close:
                suggestion = f"。是否想写 {close[0]}（{ALL_TAGS[close[0]]}）"
        errors.append(
            f"{where}：第 {pos + 1} 个字符附近的方括号内容 [{inner}] 不是官方情绪/"
            f"拟声标签{suggestion}。若是标签请改正拼写{_SOURCE}；"
            "若是普通文本，请改用【】或（）以免被误当作控制标签。"
        )
    return errors


def check_ssml_combo(text: str, *, where: str) -> list[str]:
    """SSML 与情绪/拟声标签的组合：官方未说明支持，同一片段混用直接报错。"""
    if has_tags(text) and _SSML_SPEAK_RE.search(text):
        return [
            f"{where}：同一片段同时包含 SSML（<speak>…）与情绪/拟声标签；"
            "官方文档未说明二者可组合使用，Podvia 不支持该组合。"
            "请将该片段改为纯 SSML 或纯标签文本后分片段书写。"
        ]
    return []


def check_model_support(
    text: str, provider: str, model: str, *, where: str,
    supports: bool | None,
) -> list[str]:
    """模型支持门控：含官方标签但当前 provider/模型未确认支持时阻止发送。

    supports=None 表示模型不在已核实矩阵（未确认，不静默透传）。
    """
    first = next(iter_tags(text), None)
    if first is None:
        return []
    tag = first[2]
    if provider != "qwen_audio":
        return [
            f"{where}：文本含情绪/拟声标签 {tag}，但当前接口 {provider!r} 不是 "
            f"Qwen-Audio-TTS；官方仅确认 qwen-audio-3.1-tts-flash / "
            f"qwen-audio-3.0-tts-plus / qwen-audio-3.0-tts-flash 支持标签{_SOURCE}。"
        ]
    if supports is None:
        return [
            f"{where}：文本含情绪/拟声标签 {tag}，但模型 {model!r} 不在已核实能力"
            "矩阵中，是否支持标签未确认；Podvia 不静默透传，请改用上述已确认模型"
            "或删除标签。"
        ]
    if not supports:
        return [
            f"{where}：模型 {model!r} 不在官方标签支持列表中"
            f"（仅 qwen-audio-3.1-tts-flash / qwen-audio-3.0-tts-plus / "
            f"qwen-audio-3.0-tts-flash{_SOURCE}），请更换模型或删除标签 {tag}。"
        ]
    return []
