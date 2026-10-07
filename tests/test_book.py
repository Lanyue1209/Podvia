"""阶段 5 单元测试：读书音频（章节导入、安全分段、原文追溯、局部重生成）。

纯本地 + FakeProvider，不触网、无 LLM。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from podcast_tts.book import BOOK_MAX_SEGMENT_CHARS, BookError, normalize_ws, parse_book
from podcast_tts.config import AppConfig, ConfigError, QwenAudioParams, VoiceSetting
from podcast_tts.models import TTSResult
from podcast_tts.pipeline import run_book, run_regenerate
from podcast_tts.providers.base import TTSProvider

BOOK = """# 第一章 出发

- 角色: Lily
- 导演: 平静地讲述

清晨，Mr. Smith 带着 3.14 美元出门了。他住在 Washington D.C. 的郊区。

[PAUSE:2]

# 第二章 到达

- 角色: Leo
- 导演: 略带兴奋

终于，他们抵达了 U.S.A. 东海岸！天气真好。
"""


def _parse(source: str, **kw):
    return parse_book(source, {"Lily", "Leo"}, source_file="novel.md",
                      default_speaker="Lily", **kw)


# —— 章节结构 ————————————————————————————————————————————————————————


def test_chapters_meta_and_pause_between():
    result = _parse(BOOK)
    assert [c.title for c in result.chapters] == ["第一章 出发", "第二章 到达"]
    assert result.chapters[0].speaker == "Lily"
    assert result.chapters[0].direction == "平静地讲述"
    assert result.chapters[1].speaker == "Leo"

    kinds = [(s.kind, s.speaker, s.direction) for s in result.segments]
    # 章间确定性停顿恰好一个，且位于两章之间；末章之后无停顿
    pauses = [s for s in result.segments if s.kind == "pause"]
    assert len(pauses) == 2  # 一个作者 [PAUSE:2]，一个章间停顿
    explicit = next(s for s in pauses if s.pause_seconds == 2.0)
    assert explicit.chapter == "第一章 出发"
    # 顺序：第一章台词 → PAUSE → 第二章台词
    speakers = [s.speaker for s in result.segments if s.kind == "speech"]
    assert speakers[0] == "Lily" and speakers[-1] == "Leo"


def test_indices_are_sequential():
    result = _parse(BOOK)
    assert [s.index for s in result.segments] == list(range(1, len(result.segments) + 1))


def test_unknown_speaker_rejected():
    with pytest.raises(BookError, match="不在角色列表"):
        parse_book("# 一\n\n- 角色: Nobody\n\n正文。", {"Lily"})


def test_missing_speaker_without_default_rejected():
    with pytest.raises(BookError, match="未声明朗读者"):
        parse_book("# 一\n\n正文。", {"Lily"})


def test_empty_chapter_rejected():
    with pytest.raises(BookError, match="正文为空"):
        _parse("# 第一章 出发\n\n- 角色: Lily\n\n# 第二章\n\n- 角色: Leo\n\n有正文。")


def test_form_feed_split_txt():
    txt = "第一章内容。这是第一句。\f第二章内容。第二句。"
    result = parse_book(txt, {"Lily"}, default_speaker="Lily",
                        default_title="小说")
    assert len(result.chapters) == 2
    assert [c.title for c in result.chapters] == ["小说", "小说"]


# —— 原文追溯 ————————————————————————————————————————————————————————


def test_every_segment_traces_to_source():
    result = _parse(BOOK)
    for seg in result.segments:
        assert seg.source_file == "novel.md"
        if seg.kind != "speech":
            continue  # 章间结构性停顿无原文坐标；作者 [PAUSE] 行有坐标
        assert seg.char_start is not None and seg.char_end is not None
        raw_slice = BOOK[seg.char_start:seg.char_end]
        # 发送文本 = 原文切片的空白折叠版（无任何字词改写）
        assert seg.text == normalize_ws(raw_slice)
        assert seg.raw_text == raw_slice.strip()
        assert seg.text  # 非空
    # 作者显式 PAUSE 行也必须带坐标
    explicit = next(s for s in result.segments if s.kind == "pause" and s.pause_seconds == 2.0)
    assert explicit.char_start is not None
    assert BOOK[explicit.char_start:explicit.char_end].strip().startswith("[PAUSE:2]")


def test_no_implicit_rewrite_keeps_punctuation():
    result = _parse("你好，世界——这是一个测试！第二句呢？当然继续。")
    text = "".join(s.text for s in result.segments if s.kind == "speech")
    assert "——" in text and "！" in text and "？" in text


# —— 安全分段：数字/缩写/专名/SSML ————————————————————————————————————


def test_abbreviations_and_numbers_not_split():
    src = ("Mr. Smith paid 3.14 dollars in Washington D.C. yesterday; "
           "the price was 1,000 yen in the U.S.A. and e.g. taxes applied.")
    result = _parse(src)
    text = result.segments[0].text
    assert "Mr. Smith" in text and "3.14" in text
    assert "D.C." in text and "U.S.A." in text and "e.g." in text
    # 全部在同一段（缩写句点没有被当句界）
    assert len([s for s in result.segments if s.kind == "speech"]) == 1


def test_sentence_boundary_still_works():
    result = _parse("第一句。第二句！第三句？")
    # 三个句子；是否合并取决于总长，但每个句子必须完整出现且顺序不乱
    joined = "".join(s.text for s in result.segments if s.kind == "speech")
    assert joined.replace(" ", "") == "第一句。第二句！第三句？"


def test_ssml_block_is_atomic():
    src = ('<speak rate="1.2">你好。世界。</speak>这是标签外的一句。')
    result = _parse(src)
    ssml_segs = [s for s in result.segments if "<speak" in (s.text or "")]
    assert len(ssml_segs) == 1
    assert ssml_segs[0].text == '<speak rate="1.2">你好。世界。</speak>'
    # 标签外正文独立成段，未被吞进 SSML
    others = [s.text for s in result.segments if s.kind == "speech" and "<speak" not in s.text]
    assert any("标签外" in t for t in others)


def test_long_paragraph_split_under_limit():
    # 无句末标点、只有逗号的长句（500 字以上）
    body = "数字与名词不应被拆坏，" * 40  # ~ 400 字？实际约 440
    body += "张三带着12345.67元走了" * 20
    result = _parse(body)
    texts = [s.text for s in result.segments if s.kind == "speech"]
    assert len(texts) > 1
    assert all(len(t) <= BOOK_MAX_SEGMENT_CHARS for t in texts)
    # 顺序拼接后只发生空白折叠，不丢字、不改字（原文无空白，应逐字相等）
    joined = "".join(texts)
    assert joined == normalize_ws(body)
    assert "张三带着12345.67元走了" in joined


# —— 章节内多角色（正文行级角色，与脚本模式共享解析） ————————————————————

_MULTI = {"Narrator": "Narrator", "Lily": "Lily", "Leo": "Leo"}


def _parse_multi(source: str, **kw):
    return parse_book(source, _MULTI, source_file="novel.md",
                      default_speaker="Narrator", **kw)


def test_single_role_chapter_stays_all_default():
    """旧单角色格式完全兼容：无行内角色时全部由章节默认角色朗读。"""
    src = "# 第一章\n\n- 角色: Lily\n\n你好。\n今天天气不错。\n"
    result = _parse(src)
    speech = [s for s in result.segments if s.kind == "speech"]
    # 相邻短句被既有贪心合并规则并为一段（旧版行为，保持不变）
    assert len(speech) == 1
    assert speech[0].speaker == "Lily" and speech[0].origin == "prose"
    assert "你好" in speech[0].text and "今天天气不错" in speech[0].text


def test_multi_role_chapter_inline_speakers_and_prose_fallback():
    src = (
        "# 第一章\n\n- 角色: Narrator\n\n"
        "旁白第一句。\n\n"
        "Lily: 你好。\n\n"
        "Leo: 你好。\n\n"
        "旁白第二句。\n"
    )
    result = _parse_multi(src)
    speech = [s for s in result.segments if s.kind == "speech"]
    assert [(s.speaker, s.text) for s in speech] == [
        ("Narrator", "旁白第一句。"),
        ("Lily", "你好。"),
        ("Leo", "你好。"),
        ("Narrator", "旁白第二句。"),
    ]
    # 类型标记：显式角色行=对白，默认角色正文=正文
    assert [s.origin for s in speech] == [
        "prose", "dialogue", "dialogue", "prose",
    ]


def test_inline_role_does_not_pollute_following_prose():
    src = "# 第一章\n\n- 角色: Narrator\n\nLily: 你好。\n\n夜色越来越深。\n"
    result = _parse_multi(src)
    speech = [s for s in result.segments if s.kind == "speech"]
    assert [s.speaker for s in speech] == ["Lily", "Narrator"]


def test_dialogue_with_direction_and_params_and_tags():
    src = (
        "# 第一章\n\n- 角色: Lily\n- 导演: 悬疑风格\n\n"
        "Leo(轻快地): [excited] 你终于来了！\n\n"
        "Leo(神秘地; rate=0.9): [whispers] 小声点。\n"
    )
    result = _parse(src)
    speech = [s for s in result.segments if s.kind == "speech"]
    assert len(speech) == 2
    first, second = speech
    assert first.speaker == "Leo" and first.direction == "轻快地"
    assert first.rate is None and first.text.startswith("[excited] ")
    assert second.direction == "神秘地" and second.rate == 0.9
    assert second.text.startswith("[whispers] ")
    # 章节级导演指令保留在 chapter_direction（与行内指令分层合并，互不破坏）
    assert all(s.chapter_direction == "悬疑风格" for s in speech)


def test_dialogue_multiple_vocal_tags_line_not_swallowed():
    src = "# 第一章\n\n- 角色: Lily\n\nLeo(坚定地): [serious] 我知道了。[clears throat] 我们走。\n"
    result = _parse(src)
    speech = [s for s in result.segments if s.kind == "speech"]
    assert len(speech) == 1
    assert "[serious]" in speech[0].text and "[clears throat]" in speech[0].text
    assert "我知道了。" in speech[0].text and "我们走。" in speech[0].text


def test_pause_line_in_chapter_body_is_independent():
    src = "# 第一章\n\n- 角色: Lily\n\n第一句。\n\n[PAUSE:1.5]\n\nLeo: 第二句。\n"
    result = _parse(src)
    pauses = [s for s in result.segments if s.kind == "pause"]
    explicit = next(s for s in pauses if s.pause_seconds == 1.5)
    assert explicit.speaker is None and explicit.pause_seconds == 1.5
    # PAUSE 不绑定角色、不影响下一角色
    speech = [s for s in result.segments if s.kind == "speech"]
    assert [s.speaker for s in speech] == ["Lily", "Leo"]
    assert src[explicit.char_start:explicit.char_end].strip() == "[PAUSE:1.5]"


def test_dialogue_emotion_does_not_leak_into_prose():
    """对话块是情绪屏障：[sad] 不得延续到其后的普通正文。"""
    src = "# 第一章\n\n- 角色: Lily\n\nLeo: [sad] 我有点害怕。\n\n门外没有回应。\n"
    result = _parse(src)
    prose = next(s for s in result.segments if s.kind == "speech" and s.origin == "prose")
    assert prose.speaker == "Lily"
    assert not prose.text.startswith("[sad]")


def test_chapter_without_default_but_all_explicit_roles():
    result = parse_book(
        "# 第一章\n\nLily: 你好。\nLeo: 你好。\n",
        {"Lily", "Leo"},
        insert_chapter_pause=False,
    )
    speech = [s for s in result.segments if s.kind == "speech"]
    assert [s.speaker for s in speech] == ["Lily", "Leo"]


def test_chapter_without_default_and_prose_still_rejected():
    with pytest.raises(BookError, match="未声明朗读者"):
        parse_book("# 第一章\n\nLily: 你好。\n\n旁白正文。\n", {"Lily"})


def test_malformed_speaker_line_warns_and_falls_back_to_prose(caplog):
    """Leo(低沉: 你好 → 不静默吞掉：WARNING（含行号/原文/原因）+ 按普通正文处理。"""
    import logging as _logging

    src = "# 第一章\n\n- 角色: Lily\n\nLeo(低沉: 你好\n"
    with caplog.at_level(_logging.WARNING, logger="podcast_tts"):
        result = _parse(src)
    speech = [s for s in result.segments if s.kind == "speech"]
    assert len(speech) == 1
    assert speech[0].speaker == "Lily"           # 按普通正文 → 章节默认角色
    assert "Leo(低沉" in speech[0].text          # 内容不丢
    assert any("疑似角色声明" in r.message for r in caplog.records)
    assert any("第 5 行" in r.message for r in caplog.records)  # 含行号


def test_unregistered_speaker_line_warns_and_falls_back_to_prose(caplog):
    import logging as _logging

    src = "# 第一章\n\n- 角色: Lily\n\n张三: 你好\n"
    with caplog.at_level(_logging.WARNING, logger="podcast_tts"):
        result = _parse(src)
    speech = [s for s in result.segments if s.kind == "speech"]
    assert speech[0].speaker == "Lily"
    assert "张三" in speech[0].text
    assert any("疑似角色声明" in r.message for r in caplog.records)


def test_dialogue_multi_line_block_keeps_script_semantics():
    """与脚本模式一致：角色头开启的台词块延续到空行为止。"""
    src = "# 第一章\n\n- 角色: Lily\n\nLeo: 你好\n今天天气不错。\n\n旁白继续。\n"
    result = _parse(src)
    speech = [s for s in result.segments if s.kind == "speech"]
    assert [s.speaker for s in speech] == ["Leo", "Lily"]
    assert "你好" in speech[0].text and "今天天气不错" in speech[0].text


def test_book_line_syntax_matches_script_parser():
    """同一行写法在章节正文与脚本解析器中得到相同 Segment 结果（共享 scriptlines）。"""
    line_block = "Lily(轻快地; rate=1.1): [excited] 太好了！\n继续。\n"
    book = parse_book("# 一\n\n" + line_block, {"Lily"}, insert_chapter_pause=False)
    from podcast_tts.parser import ScriptParser

    script = ScriptParser({"Lily": "Lily"}).parse(line_block)
    bs = [s for s in book.segments if s.kind == "speech"]
    assert len(bs) == 1 and len(script) == 1
    b, sc = bs[0], script[0]
    assert b.speaker == sc.speaker == "Lily"
    assert b.direction == sc.direction == "轻快地"
    assert b.rate == sc.rate == 1.1
    assert b.text == sc.text


def test_dialogue_segments_trace_to_source():
    """对白段坐标可回溯：切片去空白后与 raw_text 一致。"""
    src = "# 第一章\n\n- 角色: Lily\n\n旁白。\n\nLeo(低沉地): 你好。\n"
    result = _parse(src)
    dlg = next(s for s in result.segments if s.kind == "speech" and s.origin == "dialogue")
    raw_slice = src[dlg.char_start:dlg.char_end]
    assert "".join(raw_slice.split()) == "".join((dlg.raw_text or "").split())


# —— pipeline 集成（FakeProvider） ————————————————————————————————————


class FakeProvider(TTSProvider):
    name = "fake"

    def __init__(self):
        self.calls: list = []

    def generate(self, request):
        self.calls.append(request)
        out = Path(request.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"FAKEWAVDATA")
        return TTSResult(output_path=request.output_path, request_id=f"req-{len(self.calls):03d}")


def make_cfg(tmp_path: Path) -> AppConfig:
    return AppConfig(
        provider="qwen_audio",
        model="qwen-audio-3.0-tts-plus",
        voices={
            "Lily": VoiceSetting("longanlingxin", "English"),
            "Leo": VoiceSetting("longanlufeng", "Chinese"),
        },
        max_concurrency=2, request_timeout=5, max_retries=1, retry_base_delay=0.01,
        optimize_instructions=True,
        qwen_base_url="https://fake.example/api",
        qwen_audio_base_url="https://fake.example/audio-tts",
        audio_params=QwenAudioParams(),
        merge_format="mp3", bitrate="192k",
        scripts_dir=tmp_path / "scripts", projects_dir=tmp_path / "projects",
        logs_dir=tmp_path / "logs", ffmpeg_dir=tmp_path / "tools",
        api_key="sk-test",
    )


def _write_book(tmp_path: Path) -> Path:
    scripts = tmp_path / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    path = scripts / "novel.md"
    path.write_text(BOOK, encoding="utf-8")
    return path


def test_run_book_generates_and_records_traceability(tmp_path):
    cfg = make_cfg(tmp_path)
    path = _write_book(tmp_path)
    provider = FakeProvider()

    outcome = run_book(path, cfg, provider=provider, default_speaker="Lily")
    speech_n = len([s for s in outcome.manifest.segments if s.kind == "speech"])
    assert outcome.succeeded == speech_n and outcome.failed == 0
    assert len(provider.calls) == speech_n
    rec = outcome.manifest.record(1)
    assert rec.chapter == "第一章 出发"
    assert rec.source_file == "novel.md"
    assert rec.char_start is not None and rec.char_end is not None
    assert rec.text == normalize_ws(BOOK[rec.char_start:rec.char_end])


def test_run_book_cache_skip_on_second_run(tmp_path):
    cfg = make_cfg(tmp_path)
    path = _write_book(tmp_path)

    first = run_book(path, cfg, provider=FakeProvider(), default_speaker="Lily")
    provider2 = FakeProvider()
    second = run_book(path, cfg, provider=provider2, default_speaker="Lily")
    assert second.succeeded == 0 and second.skipped == first.speech_total
    assert provider2.calls == []


def test_run_regenerate_only_selected_indices(tmp_path):
    cfg = make_cfg(tmp_path)
    path = _write_book(tmp_path)
    run_book(path, cfg, provider=FakeProvider(), default_speaker="Lily")

    provider = FakeProvider()
    outcome = run_regenerate("novel", [1], cfg, provider=provider)
    assert outcome.succeeded == 1 and outcome.failed == 0
    assert len(provider.calls) == 1
    assert provider.calls[0].text  # 文本来自 manifest 快照

    # 未选择的片段再次局部重生成时不受影响
    provider2 = FakeProvider()
    run_regenerate("novel", [1], cfg, provider=provider2)
    assert len(provider2.calls) == 1


def test_run_regenerate_bad_index_raises(tmp_path):
    cfg = make_cfg(tmp_path)
    path = _write_book(tmp_path)
    run_book(path, cfg, provider=FakeProvider(), default_speaker="Lily")
    with pytest.raises(ConfigError, match="不存在"):
        run_regenerate("novel", [999], cfg, provider=FakeProvider())


def test_run_regenerate_missing_episode_raises(tmp_path):
    cfg = make_cfg(tmp_path)
    with pytest.raises(FileNotFoundError):
        run_regenerate("ghost", [1], cfg, provider=FakeProvider())
