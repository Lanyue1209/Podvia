"""Script Parser：将带「角色: 台词」标记的 Markdown 脚本解析为 Segment 列表。

设计要点：
- 提取三类音频内容：角色台词（可含导演指令与行内参数）与 [PAUSE:x]
- 角色由调用方提供的"显示名 → 稳定ID"映射决定（支持中文角色名）；
  只有已注册的显示名才能作为说话人标记，杜绝把普通句首词误判成角色
- 支持的说话人行形式（中英文冒号、中英文括号均可）：
    Lily: 台词            Lily：台词           Lily(小声点): 台词
    莉莉: 台词            莉莉：台词           莉莉(温柔地; rate=1.2): 台词
    [旁白] 台词           （仅当"旁白"已在角色表注册）
- 疑似角色标记但未注册（如 "张三：你好" / "[旁白] …"）：直接 ParseError，
  绝不静默整行送合成（避免把角色名读出来）
- 行内参数写在导演指令括号内：rate=0.5~2 / pitch=0.5~2 / volume=0~100，
  如 Lily(轻快地; rate=1.2, volume=80): …；属片段级覆盖
- 情绪与人声拟声标签（官方 Qwen-Audio-TTS text 内嵌标签，见 emotags 模块）：
  可出现在台词开头/中间/独占一行（归属当前角色）；台词块之外独立成行报错；
  疑似写错的标签（如 [laugh]）报拼写错误；标签绝不当作角色名
- 忽略 Markdown 结构、[全局配置] 块、元数据行（Visible:/自检/状态等）、代码块
- 音效标记（【音效:…】/[BGM:…]/[音效：…]）：当前版本明确不支持，直接 ParseError
  拒绝（不忽略、不合成），提示用户删除该行
- 绝不修改台词内容：仅剥离 Markdown 格式控制符，标点/破折号等原样保留
- 单个台词块超过安全长度（480 字）时按句子边界安全切分（复用 book 分段器，
  保护数字/缩写/SSML 块），不等 API 报错
- 纯本地解析：无网络、无 LLM
"""
from __future__ import annotations

import re

from podcast_tts import emotags
from podcast_tts import log
from podcast_tts.book import normalize_speakers, split_speech_text_detailed
from podcast_tts.models import Segment
# 行级语法（角色头/行内参数/PAUSE/Markdown 清洗/疑似角色识别）与章节正文共享，
# 见 scriptlines 模块——两种入口的同一写法必须得到同一解析结果
from podcast_tts.scriptlines import (  # noqa: F401  （部分名称仅供外部兼容导入）
    BRACKET_SPEAKER,
    PAUSE_LINE,
    TRAIL_PAREN,
    _FIRST_COLON,
    _join_lines,
    extract_inline_params,
    looks_like_unregistered_speaker as _looks_like_unregistered_speaker,
    match_speaker_header,
    strip_inline_markdown,
)

# 脚本台词块安全分段长度（低于百炼实测 600 字上限，留余量）
SAFE_SEGMENT_CHARS = 480


class ParseError(Exception):
    """脚本解析错误。消息包含行号，便于定位修改。"""


# —— 行级模式（匹配前各行已 strip）—————————————————————————————
CONFIG_HEADER = re.compile(r"^\[全局配置\]$")
SFX_LINE = re.compile(r"^【音效[:：][^】]*】.*$")
# 显式音效/音乐标记的方括号写法（当前版本不支持，给明确错误而不是读出）
SFX_BRACKET_LINE = re.compile(r"^\[\s*(?:音效|背景音乐|BGM|环境音|SFX)\b[:：]?", re.IGNORECASE)
FENCE_LINE = re.compile(r"^```")
HEADING_LINE = re.compile(r"^#{1,6}\s")
QUOTE_LINE = re.compile(r"^>")
TABLE_LINE = re.compile(r"^\|.*\|")
HR_LINE = re.compile(r"^[-*=_]{3,}$")
BULLET_LINE = re.compile(r"^[-*+]\s")  # 含任务列表在内的各类列表
HTML_COMMENT_LINE = re.compile(r"^<!--.*-->$")

# 元数据行前缀（明确要求：不读取 Visible: 等非音频内容）
META_LINE = re.compile(
    r"^(?:Visible\s*[:：]|自检|状态|策划|目标时长|时长|版本|日期|备注|说明"
    r"|Episode\s*[:：]|EP\s*[:：])",
    re.IGNORECASE,
)

# SSML <speak> 块的行级开/闭标记（用于把跨行 SSML 收进同一台词块）
_SSML_OPEN_RE = re.compile(r"<speak\b", re.IGNORECASE)
_SSML_CLOSE_RE = re.compile(r"</speak\s*>", re.IGNORECASE)

# [全局配置] 块小节名；VOICE / LANGUAGE 下的 "Name = 值" 会被收集用于交叉校验
CONFIG_SECTION_LINE = re.compile(
    r"^(VOICE|LANGUAGE|CHARACTER|GLOBAL\s*STYLE)\s*[:：]$", re.IGNORECASE
)
CONFIG_KV_LINE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.+)$")


class ScriptParser:
    """逐行状态机解析器。

    用法：
        parser = ScriptParser({"Lily": "id-lily", "莉莉": "id-lily"})
        segments = parser.parse(source_text)
        # 也兼容旧调用：ScriptParser({"Lily", "Leo"})（id 即名字）
    """

    def __init__(self, valid_speakers: set[str] | list[str] | tuple[str, ...] | dict):
        self.name_to_id: dict[str, str] = normalize_speakers(valid_speakers)
        self.valid_speakers = set(self.name_to_id)
        self.voice_map: dict[str, str] = {}
        self.language_map: dict[str, str] = {}

    def parse(self, source: str) -> list[Segment]:
        segments: list[Segment] = []
        self.voice_map = {}
        self.language_map = {}

        # 当前收集中的台词块状态
        speaker: str | None = None
        speaker_name: str | None = None
        direction: str | None = None
        overrides: dict = {}
        dialogue_lines: list[str] = []
        speaker_lineno = 0
        in_config = False
        in_fence = False
        config_section = ""
        # 跨行 SSML 收集：进入 <speak> 块后，直到 </speak> 的所有行
        #（含空行/标签/属性）都属于当前台词块，绝不拆段、绝不丢弃标签
        ssml_open = False

        def _check_ssml_boundary(piece: str) -> None:
            """按本行/本段内容维护 ssml_open（含开含闭则视为单行完整块）。"""
            nonlocal ssml_open
            if _SSML_CLOSE_RE.search(piece):
                ssml_open = False
            elif _SSML_OPEN_RE.search(piece):
                ssml_open = True

        def flush() -> None:
            """结束当前台词块并产出 Segment（超长块安全切多段；无台词块为空操作）。"""
            nonlocal speaker, speaker_name, direction, overrides, dialogue_lines
            nonlocal ssml_open
            if speaker is None:
                return
            raw = _join_lines(dialogue_lines).strip()
            text = _join_lines(
                [strip_inline_markdown(line) for line in dialogue_lines]
            ).strip()
            if not text:
                raise ParseError(f"第 {speaker_lineno} 行：{speaker_name} 的台词为空。")
            if len(text) > SAFE_SEGMENT_CHARS:
                detailed = split_speech_text_detailed(text)
            else:
                detailed = [(text, "")]
            for piece, injected in detailed:
                segments.append(
                    Segment(
                        index=len(segments) + 1,
                        kind="speech",
                        speaker=speaker,
                        speaker_name=speaker_name,
                        direction=direction,
                        rate=overrides.get("rate"),
                        pitch=overrides.get("pitch"),
                        volume=overrides.get("volume"),
                        text=piece,
                        raw_text=raw,
                        injected_prefix=injected or None,
                    )
                )
            speaker, speaker_name, direction, overrides, dialogue_lines = (
                None, None, None, {}, []
            )
            ssml_open = False

        for lineno, raw_line in enumerate(source.splitlines(), start=1):
            line = raw_line.strip()

            # 代码围栏：内部内容全部忽略（围栏也会结束配置块与台词收集）
            if FENCE_LINE.match(line):
                flush()
                in_fence = not in_fence
                in_config = False
                continue
            if in_fence:
                continue

            # 跨行 SSML：块内所有行（含空行/标签/属性/看似结构的行）都收进
            # 当前台词块，直到 </speak> 为止——不拆段、不当作正文结构处理
            if ssml_open:
                if line:
                    dialogue_lines.append(line)
                    if _SSML_CLOSE_RE.search(line):
                        ssml_open = False
                continue

            # 空行：结束台词收集；配置块内部的空行不结束配置块
            if not line:
                if not in_config:
                    flush()
                continue

            # [全局配置] 块入口
            if CONFIG_HEADER.match(line):
                flush()
                in_config = True
                config_section = ""
                continue

            # 配置块内部：遇到 标题/PAUSE/音效/已知角色台词行 时退出配置块，
            # 该行继续按下方正常逻辑处理；其余行作为配置内容收集或忽略
            if in_config:
                speaker_match = match_speaker_header(line, self.name_to_id)
                if (
                    HEADING_LINE.match(line)
                    or PAUSE_LINE.match(line)
                    or SFX_LINE.match(line)
                    or speaker_match is not None
                ):
                    in_config = False
                else:
                    section_match = CONFIG_SECTION_LINE.match(line)
                    if section_match:
                        config_section = section_match.group(1).upper().replace(" ", "")
                    else:
                        kv = CONFIG_KV_LINE.match(line)
                        if kv and config_section == "VOICE":
                            self.voice_map[kv.group(1)] = kv.group(2).strip()
                        elif kv and config_section == "LANGUAGE":
                            self.language_map[kv.group(1)] = kv.group(2).strip()
                        # CHARACTER / GLOBAL STYLE 及其他内容：人读的导演资料，忽略
                    continue

            # PAUSE：机器执行指令（与导演指令中的"停顿"字样是两回事，不混淆）
            m = PAUSE_LINE.match(line)
            if m:
                flush()
                seconds = float(m.group(1))
                if not 0 < seconds <= 10:
                    raise ParseError(
                        f"第 {lineno} 行：PAUSE 时长 {seconds:g}s 超出允许范围 (0, 10]。"
                    )
                segments.append(
                    Segment(index=len(segments) + 1, kind="pause",
                            pause_seconds=seconds)
                )
                continue

            # 音效标记：明确不支持
            if SFX_LINE.match(line) or SFX_BRACKET_LINE.match(line):
                raise ParseError(
                    f"第 {lineno} 行：检测到音效/背景音乐标记（{line}）。"
                    "当前版本不支持音效（导演指令、停顿 [PAUSE:x] 与音量都不是音效）；"
                    "请删除该行后再生成。"
                )

            # 元数据行：忽略（Visible:/自检/状态/策划/目标时长 等）
            if META_LINE.match(line):
                flush()
                log.debug(f"第 {lineno} 行：忽略元数据行：{line}")
                continue

            # 说话人行：先精确匹配已注册角色（支持中文显示名/方括号形式）
            header = match_speaker_header(line, self.name_to_id)
            if header is not None:
                flush()
                speaker = header["speaker"]
                speaker_name = header["speaker_name"]
                direction = header["direction"]
                overrides = header["overrides"]
                dialogue_lines = []
                speaker_lineno = lineno
                inline_text = header["inline_text"]
                if inline_text:
                    dialogue_lines.append(inline_text)
                _check_ssml_boundary(inline_text or "")
                continue

            # 行首方括号但不是已注册角色：区分官方标签 / 疑似写错的标签 / 未注册角色
            bracket = BRACKET_SPEAKER.match(line)
            if bracket and not PAUSE_LINE.match(line):
                bname = bracket.group("name").strip()
                if f"[{bname}]" in emotags.ALL_TAGS:
                    # 官方情绪/拟声标签（含 [laughing] 独占一行）：归属当前台词块
                    if speaker is not None:
                        if not bracket.group("inline").strip():
                            log.warning(
                                f"第 {lineno} 行：拟声/情绪标签 [{bname}] 独立成行。"
                                "官方示例均为“标签+台词”组合，独立成行的效果未经官方确认；"
                                f"已按角色 {speaker_name} 的台词片段处理。"
                            )
                        dialogue_lines.append(line)
                        continue
                    raise ParseError(
                        f"第 {lineno} 行：情绪/拟声标签 [{bname}] 之前没有任何角色台词，"
                        "无法确定由谁发出。请把它写入某个角色的台词"
                        f"（如 “角色名: [{bname}] 台词”），不要独立成行。"
                    )
                if _FIRST_COLON.search(line) is None and emotags.unknown_tag_candidates(
                    f"[{bname}]"
                ):
                    # 形如 [laugh] / [veryfast]：疑似官方标签拼写错误
                    raise ParseError(
                        emotags.check_text_tags(f"[{bname}]", where=f"第 {lineno} 行")[0]
                    )

            # 未注册但疑似角色标记：硬报错，绝不静默整行合成
            suspicious = _looks_like_unregistered_speaker(line)
            if suspicious is not None:
                raise ParseError(
                    f"第 {lineno} 行：疑似角色标记 {suspicious!r}，但该名称不在角色表中"
                    f"（已注册角色：{', '.join(sorted(self.valid_speakers)) or '（空）'}）。\n"
                    "处理办法：① 在角色表添加该角色；② 改用已注册的角色名；"
                    "③ 若这是普通台词，请删去行首的冒号标记或改写该句。"
                )

            # Markdown 结构行：忽略
            if (
                HEADING_LINE.match(line)
                or QUOTE_LINE.match(line)
                or TABLE_LINE.match(line)
                or HR_LINE.match(line)
                or BULLET_LINE.match(line)
                or HTML_COMMENT_LINE.match(line)
            ):
                flush()
                log.debug(f"第 {lineno} 行：忽略 Markdown 结构行：{line}")
                continue

            # 台词行：并入当前台词块
            if speaker is not None:
                dialogue_lines.append(line)
                _check_ssml_boundary(line)
                continue

            # 未知结构：默认忽略；疑似格式错误时警告
            if line.startswith(tuple(self.valid_speakers)):
                log.warning(f"第 {lineno} 行：疑似格式错误，已忽略：{line}")
            else:
                log.debug(f"第 {lineno} 行：忽略无法识别的行：{line}")

        flush()
        return segments
