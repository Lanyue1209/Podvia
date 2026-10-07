"""Tests for podcast_tts.emotags — 情绪与人声拟声标签的校验、分段与缓存。"""
from pathlib import Path

import pytest

from podcast_tts import emotags
from podcast_tts.config import (
    AppConfig,
    ConfigError,
    QwenAudioParams,
    VoiceSetting,
)
from podcast_tts.models import Segment


def make_cfg(tmp_path: Path, **overrides) -> AppConfig:
    """隔离目录 + 虚假 Key 的 qwen_audio 配置（封网，不调真实 API）。"""
    defaults = dict(
        provider="qwen_audio",
        model="qwen-audio-3.0-tts-plus",
        voices={"Lily": VoiceSetting("longanlingxin", "Chinese"),
                "Leo": VoiceSetting("longanlufeng", "Chinese")},
        max_concurrency=1, request_timeout=5, max_retries=1, retry_base_delay=0.01,
        optimize_instructions=False,
        qwen_base_url="https://fake.example/api",
        qwen_audio_base_url="https://fake.example/audio-tts",
        audio_params=QwenAudioParams(),
        merge_format="mp3", bitrate="192k",
        scripts_dir=tmp_path / "scripts", projects_dir=tmp_path / "projects",
        logs_dir=tmp_path / "logs", ffmpeg_dir=tmp_path / "tools",
        api_key="sk-test",
    )
    defaults.update(overrides)
    return AppConfig(**defaults)



# —— 标签识别 —————————————————————————————————————————————————————————

class TestTagRecognition:
    def test_all_tags_are_lowercase(self):
        for t in emotags.ALL_TAGS:
            assert t == t.lower(), f"标签必须全小写：{t}"

    def test_control_tag_count(self):
        assert len(emotags.CONTROL_TAGS) == 23

    def test_vocal_tag_count(self):
        assert len(emotags.PARALINGUISTIC_TAGS) == 7

    def test_iter_tags_order(self):
        text = "[excited]你好[laughing]再见"
        tags = list(emotags.iter_tags(text))
        assert tags == [
            (0, 9, "[excited]", "control"),
            (11, 21, "[laughing]", "vocal"),
        ]

    def test_no_tags(self):
        assert emotags.has_tags("今天天气不错。") is False

    def test_has_tags(self):
        assert emotags.has_tags("[excited]天气不错。") is True

    def test_active_emotion_before(self):
        assert emotags.active_emotion_before("[excited]你好[sad]再见", 8) == "[excited]"
        assert emotags.active_emotion_before("[excited]你好[sad]再见", 15) == "[sad]"
        assert emotags.active_emotion_before("你好", 5) is None

    def test_unknown_candidates(self):
        assert emotags.unknown_tag_candidates("[laugh] [unknown]") == [
            (0, "laugh"), (8, "unknown")
        ]

    def test_unknown_candidates_ignore_official(self):
        assert emotags.unknown_tag_candidates("[laughing] [cough]") == []

    def test_unknown_candidates_ignore_pause(self):
        assert emotags.unknown_tag_candidates("[PAUSE:1.5]") == []

    def test_unknown_candidates_ignore_chinese_brackets(self):
        # 中文方括号或纯数字不视为标签候选
        assert emotags.unknown_tag_candidates("【音效：脚步】[123]") == []

    def test_leading_tag_match(self):
        assert emotags.leading_tag("[excited] 你好") == "[excited]"

    def test_leading_tag_no_match(self):
        assert emotags.leading_tag("你好 [excited]") is None

    def test_continuation_prefix(self):
        prior = "[excited]你好"
        assert emotags.continuation_prefix(prior, "再见") == "[excited] "

    def test_continuation_prefix_new_control_overrides(self):
        prior = "[excited]你好"
        assert emotags.continuation_prefix(prior, "[sad]再见") == ""

    def test_continuation_prefix_no_active(self):
        assert emotags.continuation_prefix("你好", "再见") == ""

    def test_continuation_prefix_vocal_not_inherited(self):
        prior = "[laughing]你好"
        assert emotags.continuation_prefix(prior, "再见") == ""

    def test_tag_spans(self):
        assert emotags.tag_spans("[very fast]你好") == [(0, 11)]


# —— 校验函数 —————————————————————————————————————————————————————————

class TestCheckTextTags:
    def test_ok_no_tags(self):
        assert emotags.check_text_tags("你好", where="第 1 段") == []

    def test_ok_official_tags(self):
        assert emotags.check_text_tags("[excited]你好", where="第 1 段") == []

    def test_unknown_tag_lowercase_hint(self):
        errors = emotags.check_text_tags("[laugh]", where="第 1 段")
        assert len(errors) == 1
        assert "是否想写 [laughing]" in errors[0]  # 给出最接近的官方写法

    def test_unknown_tag_exact_lowercase_hint(self):
        errors = emotags.check_text_tags("[Laughing]", where="第 1 段")
        assert len(errors) == 1
        assert "官方标签为小写" in errors[0]

    def test_unknown_tag_generic(self):
        errors = emotags.check_text_tags("[happy]", where="第 1 段")
        assert len(errors) == 1
        assert "不是官方情绪/拟声标签" in errors[0]


class TestCheckSsmlCombo:
    def test_ok_separate(self):
        assert emotags.check_ssml_combo("<speak>你好</speak>", where="第 1 段") == []
        assert emotags.check_ssml_combo("[excited]你好", where="第 1 段") == []

    def test_error_when_both_present(self):
        errors = emotags.check_ssml_combo(
            "<speak>[excited]你好</speak>", where="第 1 段")
        assert len(errors) == 1
        assert "不支持该组合" in errors[0]


class TestCheckModelSupport:
    def test_ok_no_tags(self):
        assert emotags.check_model_support(
            "你好", "qwen_audio", "qwen-audio-3.1-tts-flash", where="第 1 段",
            supports=True,
        ) == []

    def test_error_unsupported_provider(self):
        errors = emotags.check_model_support(
            "[excited]你好", "qwen", "qwen3-tts-flash", where="第 1 段",
            supports=False,
        )
        assert len(errors) == 1
        assert "当前接口" in errors[0]

    def test_error_unknown_model(self):
        errors = emotags.check_model_support(
            "[excited]你好", "qwen_audio", "some-new-model", where="第 1 段",
            supports=None,
        )
        assert len(errors) == 1
        assert "未确认" in errors[0]


# —— 整任务 validate_emotags ——————————————————————————————————————————

class TestValidateEmotags:
    def test_pass(self, tmp_path):
        cfg = make_cfg(tmp_path, model="qwen-audio-3.1-tts-flash")
        segs = [Segment(index=1, kind="speech", text="你好")]
        # 不抛即为通过
        from podcast_tts.pipeline import validate_emotags
        validate_emotags(segs, cfg)

    def test_fail_unknown_tag(self, tmp_path):
        from podcast_tts.pipeline import validate_emotags
        cfg = make_cfg(tmp_path, model="qwen-audio-3.1-tts-flash")
        segs = [Segment(index=1, kind="speech", text="[happy]你好")]
        with pytest.raises(ConfigError) as exc_info:
            validate_emotags(segs, cfg)
        assert "情绪/拟声标签校验未通过" in str(exc_info.value)

    def test_fail_model_unsupported(self, tmp_path):
        from podcast_tts.pipeline import validate_emotags
        cfg = make_cfg(tmp_path, provider="qwen", model="qwen3-tts-flash")
        segs = [Segment(index=1, kind="speech", text="[excited]你好")]
        with pytest.raises(ConfigError) as exc_info:
            validate_emotags(segs, cfg)
        assert "当前接口" in str(exc_info.value)

    def test_fail_ssml_combo(self, tmp_path):
        from podcast_tts.pipeline import validate_emotags
        cfg = make_cfg(tmp_path, model="qwen-audio-3.1-tts-flash")
        segs = [Segment(index=1, kind="speech",
                        text="<speak>[excited]你好</speak>")]
        with pytest.raises(ConfigError) as exc_info:
            validate_emotags(segs, cfg)
        assert "不支持该组合" in str(exc_info.value)


# —— 解析器集成（标签归属 / 拼写错误 / 分段连续） ————————————————————————

from podcast_tts.parser import ParseError, ScriptParser

SPEAKERS = {"Lily": "Lily", "Leo": "Leo"}


def parse(text: str):
    return ScriptParser(SPEAKERS).parse(text)


class TestParserTags:
    def test_tag_at_dialogue_start(self):
        segs = parse("Lily: [excited] 太好了！")
        assert len(segs) == 1
        assert segs[0].text == "[excited] 太好了！"
        assert segs[0].speaker == "Lily"

    def test_tag_mid_dialogue(self):
        segs = parse("Lily: 天气真好。[laughing] 我们出去玩吧！")
        assert segs[0].text == "天气真好。[laughing] 我们出去玩吧！"

    def test_tag_with_spaces_intact(self):
        segs = parse("Lily: [clears throat] 各位注意，[very fast] 现在开始！")
        assert "[clears throat]" in segs[0].text
        assert "[very fast]" in segs[0].text

    def test_standalone_tag_line_inside_block(self):
        segs = parse("Lily: 你听我说。\n[laughing]\n真好笑。")
        assert len(segs) == 1
        assert "[laughing]" in segs[0].text
        assert segs[0].speaker == "Lily"

    def test_standalone_tag_line_without_speaker_errors(self):
        with pytest.raises(ParseError) as exc_info:
            parse("[laughing]")
        assert "无法确定" in str(exc_info.value)
        assert "角色" in str(exc_info.value)

    def test_tag_typo_at_line_start_errors(self):
        with pytest.raises(ParseError) as exc_info:
            parse("Lily: 你好\n[laugh] 哈哈")
        assert "不是官方情绪/拟声标签" in str(exc_info.value)
        assert "[laughing]" in str(exc_info.value)  # 给出小写官方写法建议

    def test_unregistered_speaker_still_errors(self):
        with pytest.raises(ParseError):
            parse("[旁白] 大家好")
        with pytest.raises(ParseError):
            parse("张三: 你好")

    def test_pause_still_works(self):
        segs = parse("Lily: 你好\n\n[PAUSE:1.5]\n\nLeo: 在的")
        assert [s.kind for s in segs] == ["speech", "pause", "speech"]

    def test_plain_bracket_number_not_tag(self):
        # [3] 纯数字编号不是标签候选，作为普通台词内容放行
        segs = parse("Lily: 参考文献 [3] 提到了这一点。")
        assert "[3]" in segs[0].text

    def test_sfx_still_rejected(self):
        with pytest.raises(ParseError):
            parse("Lily: 你好\n[BGM: 轻快]")


def _long_emotion_text() -> str:
    # [excited] 开头 + 超过 480 字的台词，触发自动分段
    body = "今天发生了一件特别有意思的事情，" * 40  # 15 字 × 40 = 600 字
    return f"Lily: [excited] {body}。"


class TestEmotionContinuation:
    def test_long_dialogue_injects_emotion_prefix(self):
        segs = parse(_long_emotion_text())
        assert len(segs) >= 2
        assert segs[0].text.startswith("[excited]")
        assert segs[0].injected_prefix is None  # 第一段用原文标签，不算补入
        for seg in segs[1:]:
            assert seg.injected_prefix == "[excited] "
            assert seg.text.startswith("[excited] ")
            assert len(seg.text) <= 600  # 补入后仍满足模型长度限制

    def test_vocal_tag_not_duplicated(self):
        body = "他说了一句很长的话，" * 50  # 900 字
        segs = parse(f"Lily: [laughing] 然后{body}。")
        joined = "".join(s.text for s in segs)
        assert joined.count("[laughing]") == 1  # 一次性拟声绝不因分段重复

    def test_emotion_switch_mid_text(self):
        # 情绪在同一段台词内从 [sad] 切到 [excited]，且文本超长被拆分
        part = "后面的内容越来越激动人心，" * 50  # ~650 字
        segs = parse(f"Lily: [sad] 开头很低落。[excited] {part}。")
        assert len(segs) >= 2
        # 最后一个控制类标签是 [excited]，其后的拆分片段必须继承 [excited]
        injected = [s.injected_prefix for s in segs if s.injected_prefix]
        assert injected, "应当存在补入片段"
        assert set(injected) == {"[excited] "}
        # [sad] 绝不能被补入到任何片段
        assert not any(s.text.startswith("[sad]") for s in segs[1:])

    def test_emotion_does_not_leak_across_roles(self):
        body = "这件事说来话长，" * 40
        segs = parse(f"Lily: [excited] {body}。\n\nLeo: 我听着呢。")
        leo = [s for s in segs if s.speaker == "Leo"]
        assert leo
        for seg in leo:
            assert "[excited]" not in seg.text
            assert seg.injected_prefix is None

    def test_multi_role_each_own_emotion(self):
        segs = parse("Lily: [excited] 太棒了！\n\nLeo: [sad] 我却高兴不起来。")
        assert segs[0].text.startswith("[excited]")
        assert segs[1].text.startswith("[sad]")


class TestBookPath:
    """章节导入路径：分段同样保护标签并补入情绪状态。"""

    def test_book_long_chapter_injects_prefix(self):
        from podcast_tts.book import parse_book
        body = "这一章的内容非常丰富，" * 50  # ~550 字
        source = f"# 第一章\n- 角色: Lily\n[excited] {body}。"
        parsed = parse_book(source, {"Lily": "Lily"}, default_speaker="Lily")
        speech = [s for s in parsed.segments if s.kind == "speech"]
        assert len(speech) >= 2
        assert speech[0].text.startswith("[excited]")
        for seg in speech[1:]:
            assert seg.injected_prefix == "[excited] "
            assert seg.text.startswith("[excited] ")

    def test_book_tag_not_split(self):
        from podcast_tts.book import parse_book
        # 把带空格标签顶到 480 边界附近，验证不被切开
        filler = "他慢慢地讲述着往事，" * 46  # 460 字
        source = f"# 第一章\n- 角色: Lily\n{filler}[clears throat] 然后继续讲，" + "继续讲，" * 20
        parsed = parse_book(source, {"Lily": "Lily"}, default_speaker="Lily")
        for seg in parsed.segments:
            if seg.kind != "speech":
                continue
            assert "[clears" not in seg.text or "[clears throat]" in seg.text
            assert "throat]" not in seg.text or "[clears throat]" in seg.text


# —— Pipeline 门控（mock provider，封网） ————————————————————————————————

from podcast_tts.models import TTSResult
from podcast_tts.pipeline import run_generate
from podcast_tts.providers.base import TTSProvider


class FakeProvider(TTSProvider):
    name = "fake"

    def __init__(self):
        self.calls: list = []

    def generate(self, request):
        self.calls.append(request)
        out = Path(request.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"FAKEWAVDATA")
        return TTSResult(output_path=request.output_path, request_id="req-emu")


def _write_script(tmp_path: Path, text: str) -> Path:
    d = tmp_path / "scripts"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "episode001.md"
    p.write_text(text, encoding="utf-8")
    return p


class TestPipelineGate:
    def test_tags_pass_and_reach_provider_verbatim(self, tmp_path):
        cfg = make_cfg(tmp_path)
        script = _write_script(tmp_path, "Lily: [excited] 太好了！[laughing] 走吧！\n")
        provider = FakeProvider()
        outcome = run_generate(script, cfg, provider=provider)
        assert outcome.failed == 0
        assert provider.calls[0].text == "[excited] 太好了！[laughing] 走吧！"

    def test_unknown_tag_blocks_before_any_call(self, tmp_path):
        cfg = make_cfg(tmp_path)
        script = _write_script(tmp_path, "Lily: [happy] 你好\n\nLily: 第二段\n")
        provider = FakeProvider()
        with pytest.raises(ConfigError):
            run_generate(script, cfg, provider=provider)
        assert provider.calls == []  # 整任务先校验，不生成任何片段

    def test_unsupported_model_blocks(self, tmp_path):
        cfg = make_cfg(tmp_path, provider="qwen", model="qwen3-tts-flash")
        # qwen 端点音色规则不同，此处只验证标签门控先于音色问题触发
        script = _write_script(tmp_path, "Lily: [excited] 你好\n")
        provider = FakeProvider()
        with pytest.raises(ConfigError) as exc_info:
            run_generate(script, cfg, provider=provider)
        assert "当前接口" in str(exc_info.value)
        assert provider.calls == []

    def test_tag_change_invalidates_cache(self, tmp_path):
        from podcast_tts.manifest import content_hash
        cfg = make_cfg(tmp_path)
        h1 = content_hash(cfg, "[excited] 你好", "longanlingxin", "Chinese", None, {})
        h2 = content_hash(cfg, "你好", "longanlingxin", "Chinese", None, {})
        h3 = content_hash(cfg, "[excited] 你好", "longanlingxin", "Chinese", None, {})
        assert h1 != h2  # 标签新增 → hash 变化
        assert h1 == h3  # 同样文本 → 可复用

    def test_injected_prefix_participates_in_cache(self, tmp_path):
        from podcast_tts.manifest import content_hash
        cfg = make_cfg(tmp_path)
        h_plain = content_hash(cfg, "第二段内容", "longanlingxin", "Chinese", None, {})
        h_injected = content_hash(
            cfg, "[excited] 第二段内容", "longanlingxin", "Chinese", None, {})
        assert h_plain != h_injected

    def test_retry_keeps_manifest_text(self, tmp_path):
        """重试使用 manifest 保存的实际发送文本（含补入前缀），不随界面变化。"""
        from podcast_tts.manifest import load_or_init, save
        cfg = make_cfg(tmp_path)
        script = _write_script(tmp_path, _long_emotion_text() + "\n")
        segments = ScriptParser({"Lily": "Lily"}).parse(
            script.read_text(encoding="utf-8"))
        assert any(s.injected_prefix for s in segments)
        episode_dir = cfg.projects_dir / script.stem
        manifest = load_or_init(episode_dir, script, cfg, segments)
        save(episode_dir, manifest)
        # manifest 记录的最终文本必须含补入前缀（重试同源）
        rec_texts = [r.text for r in manifest.segments if r.kind == "speech"]
        assert any(t.startswith("[excited] ") for t in rec_texts[1:])
