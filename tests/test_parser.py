"""Phase 2 单元测试：ScriptParser。

覆盖（脚本解析器全功能）：
- 样例全断言（说话人行、导演指令、PAUSE、行内 Markdown 清洗、标点保留）
- 全特性 fixture：同行写法、多行台词合并（中文直连/英文空格）、忽略清单、
  【音效】警告、代码块、[全局配置] 收集、带空格说话人行
- PAUSE 正常/大小写/小数/越界报错
- 未知说话人报错、空台词报错
- 行内 Markdown 清洗（保留 — ? 等标点）
- 疑似格式错误警告（括号未闭合、缺少冒号）
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from podcast_tts.parser import (
    ParseError,
    ScriptParser,
    _join_lines,
    strip_inline_markdown,
)

FIXTURES = Path(__file__).parent / "fixtures"


def make_parser() -> ScriptParser:
    return ScriptParser({"Lily", "Leo"})


# —— 6.6 验收样例（文档原文） ———————————————————————————————————————

DOC_66_SAMPLE = """# 英语播客 · 绿野仙踪 第01期

状态：已完成导演 | 目标时长：12 分钟
策划文件：/docs/plan001.md

[全局配置]

VOICE:
Lily = lily_voice_id
Leo = leo_voice_id

Lily(声音亮起来,像欢迎老朋友):
Hello everyone, and welcome back to English Story Radio!

[PAUSE:0.8]

Leo(带着一点笑意):
哈喽大家好，我是Leo。今天我们要开始一个全新的故事 —— *The Wonderful Wizard of Oz*!

Lily(压低一点声音,像说秘密):
Wait — seriously?
"""


def test_66_reference_sample():
    segments = make_parser().parse(DOC_66_SAMPLE)
    assert len(segments) == 4

    s1, s2, s3, s4 = segments
    assert (s1.index, s1.kind, s1.speaker) == (1, "speech", "Lily")
    assert s1.direction == "声音亮起来,像欢迎老朋友"
    assert s1.text == "Hello everyone, and welcome back to English Story Radio!"

    assert (s2.index, s2.kind, s2.pause_seconds) == (2, "pause", 0.8)

    assert (s3.index, s3.kind, s3.speaker) == (3, "speech", "Leo")
    assert s3.direction == "带着一点笑意"
    assert s3.text == (
        "哈喽大家好，我是Leo。今天我们要开始一个全新的故事 —— The Wonderful Wizard of Oz!"
    )
    assert "*" not in s3.text
    assert s3.raw_text is not None and "*" in s3.raw_text

    assert (s4.index, s4.kind, s4.speaker) == (4, "speech", "Lily")
    assert s4.direction == "压低一点声音,像说秘密"
    assert s4.text == "Wait — seriously?"  # 长破折号与问号原样保留


# —— 全特性 fixture ————————————————————————————————————————————————


def test_fixture_full_features():
    parser = make_parser()
    source = (FIXTURES / "sample_script.md").read_text(encoding="utf-8")
    segments = parser.parse(source)

    assert [s.kind for s in segments] == [
        "speech", "pause", "speech", "speech", "pause",
        "speech", "speech", "speech", "speech",
    ]
    assert [s.index for s in segments] == list(range(1, 10))

    s1 = segments[0]
    assert s1.speaker == "Lily" and s1.direction == "声音亮起来,像欢迎老朋友"
    assert s1.text == (
        "Hello everyone, and welcome back to English Story Radio! "
        "I'm Lily, and this time we're diving into a brand new story."
    )

    assert segments[1].pause_seconds == 0.8

    s3 = segments[2]
    assert s3.speaker == "Leo" and s3.direction == "带着一点笑意"
    # 前一行以汉字结尾、本行以英文开头 → 用单个空格连接
    assert s3.text == (
        "哈喽大家好，我是 Leo。今天我们要开始一个全新的故事 —— The Wonderful Wizard of Oz!"
    )

    s4 = segments[3]
    assert s4.speaker == "Lily" and s4.text == "Wait — seriously?"

    assert segments[4].pause_seconds == 1.5

    s6 = segments[5]
    # 同行写法 "Leo(指令): 台词"
    assert s6.speaker == "Leo" and s6.direction == "激动地，语速稍快"
    assert s6.text == "Nope! But my grandma used to talk about it all the time."

    s7 = segments[6]
    assert s7.speaker == "Lily" and s7.direction is None
    assert s7.text == "Here is a line with inline code and bold italic text."

    s8 = segments[7]
    assert s8.speaker == "Leo" and s8.direction == "停顿，难以置信"
    # 两个中文行直接连接（前一行以汉字结尾 + 本行以汉字开头）
    assert s8.text == "Wow, really? 你就这么走了？"

    s9 = segments[8]
    # 说话人名与括号之间允许空格
    assert s9.speaker == "Lily" and s9.direction == "收尾，语速放慢，温暖一点"
    assert s9.text == "That's all for today — see you next week?"

    # [全局配置] 收集结果（供 main.py 与 config.yaml 交叉校验）
    assert parser.voice_map == {"Lily": "lily_voice_id", "Leo": "leo_voice_id"}
    assert parser.language_map == {"Lily": "English", "Leo": "Chinese"}


# —— 行内 Markdown 清洗 ———————————————————————————————————————————


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("**bold**", "bold"),
        ("__bold__", "bold"),
        ("*italic*", "italic"),
        ("_italic_", "italic"),
        ("***both***", "both"),
        ("`code`", "code"),
        ("I'm **Lily**, a *new* `test` ___x___", "I'm Lily, a new test x"),
        ("Wait — seriously?", "Wait — seriously?"),  # 标点原样保留
        ("2 * 3 = 6", "2 * 3 = 6"),  # 无强调语义的星号不剥离
        ("plain text", "plain text"),
    ],
)
def test_inline_markdown_cleaning(raw: str, expected: str):
    assert strip_inline_markdown(raw) == expected


# —— 多行台词合并规则 ——————————————————————————————————————————————


def test_multiline_join_rules():
    # 中文行直接连接
    assert _join_lines(["你就这么走", "了？"]) == "你就这么走了？"
    assert _join_lines(["你好，", "世界"]) == "你好，世界"
    # 英文行用单个空格连接
    assert _join_lines(["Hello", "world"]) == "Hello world"
    assert _join_lines(["你好，", "Leo"]) == "你好， Leo"
    # 边界
    assert _join_lines(["only"]) == "only"
    assert _join_lines([]) == ""


# —— PAUSE ——————————————————————————————————————————————————————————


def test_pause_variants():
    source = "Lily:\nhi\n\n[PAUSE:2]\n\n[pause:2.5]\n\nLeo:\nyo\n"
    segments = make_parser().parse(source)
    assert [s.kind for s in segments] == ["speech", "pause", "pause", "speech"]
    assert segments[1].pause_seconds == 2.0
    assert segments[2].pause_seconds == 2.5


@pytest.mark.parametrize("bad", ["[PAUSE:0]", "[PAUSE:11]", "[PAUSE:10.1]"])
def test_pause_out_of_range(bad: str):
    with pytest.raises(ParseError, match="超出允许范围"):
        make_parser().parse(f"Lily:\nhi\n\n{bad}\n")


# —— 报错场景 ——————————————————————————————————————————————————————


def test_unknown_speaker_error():
    with pytest.raises(ParseError, match="不在角色表") as exc_info:
        make_parser().parse("Max:\nHello there.\n")
    assert "Max" in str(exc_info.value)


def test_empty_speech_error():
    with pytest.raises(ParseError, match="台词为空"):
        make_parser().parse("Lily:\n\nLeo:\nyo\n")


# —— 忽略清单 ——————————————————————————————————————————————————————


def test_ignore_non_audio_content():
    source = "\n".join(
        [
            "# 标题",
            "> 引用",
            "| 表格 | 行 |",
            "- 列表项",
            "* 星号列表",
            "1. 有序列表（未知结构默认忽略）",
            "<!-- 注释 -->",
            "---",
            "***",
            "```",
            "Lily: 代码块内伪台词",
            "[PAUSE:99]",
            "```",
            "Visible: false",
            "自检：第 3 段重录",
            "状态：已完成",
            "策划文件：/docs/plan001.md",
            "目标时长：12 分钟",
            "版本：V1.0",
        ]
    )
    assert make_parser().parse(source) == []


def test_sfx_explicitly_rejected():
    """当前版本明确不做音效：音效标记必须报错，绝不静默合成或读出。"""
    with pytest.raises(ParseError, match="不支持音效"):
        make_parser().parse("Lily:\nhi\n\n【音效:风声，持续2秒】\n")
    with pytest.raises(ParseError, match="不支持音效"):
        make_parser().parse("[BGM: 背景音乐]\nLily:\nhi\n")
    with pytest.raises(ParseError, match="不支持音效"):
        make_parser().parse("[音效：脚步声]\n")


def test_unclosed_speaker_paren_is_suspicious_error():
    """未闭合的角色括号（疑似角色标记）：硬报错，不静默丢弃也不整行合成。"""
    with pytest.raises(ParseError, match="疑似角色标记"):
        make_parser().parse("Lily(指令:\n未完的台词\n")


def test_plain_line_without_colon_ignored_outside_dialogue():
    """没有冒号、没有方括号标记的普通行，在无角色上下文时忽略，不误判为角色。"""
    assert make_parser().parse("Leo 没有冒号\n") == []


# —— [全局配置] 块 ——————————————————————————————————————————————————


def test_config_block_collection_and_exit():
    source = "\n".join(
        [
            "[全局配置]",
            "",
            "VOICE:",
            "Lily = lily_voice_id",
            "Leo = leo_voice_id",
            "",
            "LANGUAGE:",
            "Lily = English",
            "Leo = Chinese",
            "",
            "CHARACTER:",
            "Lily = 青年女性",
            "",
            "GLOBAL STYLE:",
            "节奏明快。",
            "",
            "Leo(带笑):",
            "配置块之后的第一段台词。",
        ]
    )
    parser = make_parser()
    segments = parser.parse(source)

    # CHARACTER / GLOBAL STYLE 不进入映射；VOICE / LANGUAGE 正确收集
    assert parser.voice_map == {"Lily": "lily_voice_id", "Leo": "leo_voice_id"}
    assert parser.language_map == {"Lily": "English", "Leo": "Chinese"}

    # 配置块被合法角色台词行正常退出
    assert len(segments) == 1
    assert segments[0].speaker == "Leo"
    assert segments[0].direction == "带笑"
    assert segments[0].text == "配置块之后的第一段台词。"


def test_parse_is_stateless_between_calls():
    parser = make_parser()
    parser.parse("Lily:\nfirst\n")
    assert len(parser.parse("Leo:\nsecond\n")) == 1
    # 重复解析不累积 segments；配置映射被重置
    assert parser.voice_map == {}
