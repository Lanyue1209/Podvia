"""审查修复验收测试：导演指令四级层级 / 语音参数三层 / 中文角色名 / 多角色规则。

对应本轮验收清单（全程 mock，不触网、不读取真实 API Key）：
 1- 5 说话人标记：英文半角/全角冒号、中文显示名、未注册报错、[旁白]、《》不误判
 6- 7 默认角色：留空报错不静默回退；设置后正常回退
 8-11 多角色：分段、不同音色、复用音色、失效音色（失效音色另见 test_gui）
12-16 导演指令：项目/角色/章节/行内四级 + 覆盖优先级 + 每请求仅一条 instruction
17-18 rate/pitch/volume：项目<角色<行内（含 SSML 最高优先）；每段实际请求参数
19-20 长文本/数字缩写保护（另见 test_pipeline/test_book）
21    API 完成顺序不影响脚本顺序
22-24 拼接失败/仅重拼/音效拒绝（另见 test_gui、test_parser）
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from podcast_tts import config as config_mod
from podcast_tts.book import parse_book
from podcast_tts.config import (
    AppConfig,
    ConfigError,
    QwenAudioParams,
    VoiceSetting,
)
from podcast_tts.gui.controller import preview_text
from podcast_tts.gui.settings import GuiSettings, RoleVoice
from podcast_tts.models import Segment, TTSResult
from podcast_tts.parser import ParseError, ScriptParser
from podcast_tts.pipeline import final_instruction, plan_segment, run_generate
from podcast_tts.providers.base import TTSProvider

PROJECT_INSTRUCTION = "项目指令：使用活泼、轻快的语气"
ROLE_LILY_INSTRUCTION = "角色默认：Lily 声音甜美"
CHAPTER_INSTRUCTION = "章节级：本章娓娓道来"

# 用户给定的三句验收样例
THREE_LINES = "Lily：第一句\n\nLeo：第二句\n\nLily(轻快地)：第三句\n"


# —— 公共夹具 ——————————————————————————————————————————————————————————


def make_cfg(tmp_path: Path, **overrides) -> AppConfig:
    voices = {
        "id-lily": VoiceSetting(
            voice="longanlingxin", language_type="Chinese",
            display_name="Lily", instruction=ROLE_LILY_INSTRUCTION,
        ),
        "id-leo": VoiceSetting(
            voice="longanlufeng", language_type="Chinese",
            display_name="Leo",
        ),
        "id-narr": VoiceSetting(
            voice="longanlingxin", language_type="Chinese",
            display_name="旁白", custom=True,
        ),
    }
    defaults = dict(
        provider="qwen_audio",
        model="qwen-audio-3.0-tts-plus",
        voices=voices,
        max_concurrency=3,
        request_timeout=5,
        max_retries=1,
        retry_base_delay=0.01,
        optimize_instructions=True,
        qwen_base_url="https://fake.example/api",
        qwen_audio_base_url="https://fake.example/audio-tts",
        audio_params=QwenAudioParams(),
        merge_format="mp3",
        bitrate="192k",
        scripts_dir=tmp_path / "scripts",
        projects_dir=tmp_path / "projects",
        logs_dir=tmp_path / "logs",
        ffmpeg_dir=tmp_path / "tools",
        api_key="sk-test",
        default_instruction=PROJECT_INSTRUCTION,
    )
    defaults.update(overrides)
    return AppConfig(**defaults)


def write_script(tmp_path: Path, text: str, name: str = "ep.md") -> Path:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    path = scripts_dir / name
    path.write_text(text, encoding="utf-8")
    return path


class RecordingProvider(TTSProvider):
    """记录每个 TTSRequest；可按文本/音色注入延迟模拟乱序完成。"""

    name = "recording"

    def __init__(self, delays: dict[str, float] | None = None):
        self.calls: list = []
        self._delays = delays or {}
        self._lock = threading.Lock()
        self.completed: list[str] = []

    def generate(self, request):
        delay = 0.0
        for key, val in self._delays.items():
            if key in request.text:
                delay = val
        if delay:
            time.sleep(delay)
        out = Path(request.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"FAKE")
        with self._lock:
            self.calls.append(request)
            self.completed.append(request.text)
        return TTSResult(output_path=request.output_path, request_id="r1")


def generate(tmp_path, source, cfg=None, **kw):
    provider = kw.pop("provider", None) or RecordingProvider()
    cfg = cfg or make_cfg(tmp_path, **kw)
    path = write_script(tmp_path, source)
    outcome = run_generate(path, cfg, provider=provider)
    return cfg, provider, outcome


# —— 1-5：说话人标记格式 ————————————————————————————————————————————————


@pytest.mark.parametrize(
    ("line", "expected_id", "expected_name", "expected_text"),
    [
        ("Lily: 今天星期几", "id-lily", "Lily", "今天星期几"),       # 英文半角冒号
        ("Lily：今天星期几", "id-lily", "Lily", "今天星期几"),       # 中文全角冒号
        ("莉莉：今天星期几", "id-lily", "莉莉", "今天星期几"),       # 中文显示名
        ("旁白：这是旁白内容", "id-narr", "旁白", "这是旁白内容"),    # 中文角色名
        ("[旁白] 这是旁白内容", "id-narr", "旁白", "这是旁白内容"),   # 方括号形式
    ],
)
def test_speaker_header_formats(line, expected_id, expected_name, expected_text):
    parser = ScriptParser({"Lily": "id-lily", "莉莉": "id-lily",
                           "Leo": "id-leo", "旁白": "id-narr"})
    segs = parser.parse(line)
    assert len(segs) == 1
    seg = segs[0]
    assert (seg.speaker, seg.speaker_name, seg.text) == (
        expected_id, expected_name, expected_text
    )


def test_unregistered_chinese_speaker_is_hard_error():
    """未注册中文名：硬报错，绝不把 '张三：' 当文本整行送合成。"""
    parser = ScriptParser({"Lily": "id-lily"})
    with pytest.raises(ParseError, match="不在角色表"):
        parser.parse("张三：你好\n")


def test_unregistered_lily_line_never_sent_to_tts():
    """角色表只有 Leo 时，'Lily：今天星期几' 必须报错而不是读出 'Lily'。"""
    parser = ScriptParser({"Leo": "id-leo"})
    with pytest.raises(ParseError, match="不在角色表") as exc:
        parser.parse("Lily：今天星期几\n")
    assert "Lily" in str(exc.value)


def test_bracket_narrator_unregistered_is_hard_error():
    """[旁白] 只有在'旁白'已注册时才支持；未注册 → 疑似标记硬报错。"""
    parser = ScriptParser({"Lily": "id-lily"})
    with pytest.raises(ParseError, match="不在角色表"):
        parser.parse("[旁白] 今天的新闻开始了\n")


def test_book_title_marks_are_not_speakers():
    """《》 是书名号，绝不能被当作角色名。"""
    parser = ScriptParser({"Lily": "id-lily", "旁白": "id-narr"})
    segs = parser.parse("《旁白》：今天的新闻开始了\n")
    assert segs == []


# —— 6-7：默认角色回退 ————————————————————————————————————————————————


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "ROOT_DIR", tmp_path)
    (tmp_path / "config.yaml").write_text(
        """
provider: qwen_audio
model: qwen-audio-3.0-tts-plus
voices:
  lilyseed: { voice: "longanlingxin" }
  leoseed:  { voice: "longanlufeng" }
generate:
  max_concurrency: 1
""",
        encoding="utf-8",
    )
    return tmp_path


def _gui_settings(**kw) -> GuiSettings:
    s = GuiSettings(
        api_key="sk-test",
        provider="qwen_audio",
        model="qwen-audio-3.0-tts-plus",
        projects_dir="",
        roles=[
            RoleVoice(role="莉莉", voice="longanlingxin", id="id-lily"),
            RoleVoice(role="Leo", voice="longanlufeng", id="id-leo"),
        ],
        direction=PROJECT_INSTRUCTION,
    )
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_preview_plain_text_without_default_role_reports_error(isolated_config):
    """无角色头且无默认角色：预览报错，绝不静默挑选任何角色。"""
    s = _gui_settings(default_role="")
    rows, errors = preview_text(s, "没有角色头的普通文本。")
    assert rows == [] and errors
    assert "默认角色" in errors[0]


def test_preview_plain_text_with_default_role_falls_back(isolated_config):
    s = _gui_settings(default_role="莉莉")
    rows, errors = preview_text(s, "普通文本内容。")
    assert errors == [] and len(rows) == 1
    row = rows[0]
    assert row["speaker_name"] == "莉莉"
    assert row["text"] == "普通文本内容。"
    assert row["instruction"] == PROJECT_INSTRUCTION  # 项目级指令经层级生效
    assert row["output"].endswith(".wav")


def test_preview_flags_unregistered_speaker(isolated_config):
    s = _gui_settings(default_role="莉莉")
    rows, errors = preview_text(s, "张三：你好\n")
    assert rows == [] and any("不在角色表" in e for e in errors)


# —— 8-11：多角色与音色 ————————————————————————————————————————————————


def test_multi_role_segments_independent_requests(tmp_path):
    """一个模型多次请求服务多个角色；角色数不被限制为 2。"""
    cfg, provider, outcome = generate(tmp_path, THREE_LINES)
    assert outcome.failed == 0 and len(provider.calls) == 3
    by_index = sorted(provider.calls, key=lambda r: r.output_path)
    assert [Path(r.output_path).stem.split("_", 1)[1] for r in by_index] == [
        "id-lily", "id-leo", "id-lily"
    ]


def test_multi_role_distinct_voices_per_segment(tmp_path):
    cfg, provider, _ = generate(tmp_path, THREE_LINES)
    voices = [r.voice for r in sorted(provider.calls, key=lambda r: r.output_path)]
    assert voices == ["longanlingxin", "longanlufeng", "longanlingxin"]
    # 不同音色数量与角色数量是两个概念（Lily 复用同一音色）
    assert len({r.voice for r in provider.calls}) == 2


def test_reused_voice_allowed_and_same_voice_sent(tmp_path):
    """允许复用：Leo 也绑定 longanlingxin，两个角色各发各的请求、同一音色。"""
    cfg = make_cfg(tmp_path)
    cfg.voices["id-leo"] = VoiceSetting(
        voice="longanlingxin", language_type="Chinese", display_name="Leo"
    )
    _, provider, outcome = generate(tmp_path, THREE_LINES, cfg=cfg)
    assert outcome.failed == 0
    assert {r.voice for r in provider.calls} == {"longanlingxin"}
    assert len(provider.calls) == 3


def test_reused_voice_passes_gui_validation_when_toggle_off(isolated_config):
    s = _gui_settings(distinct_voices=False)
    s.roles[1].voice = "longanlingxin"  # 与莉莉同一系统音色
    assert s.validate() == []
    # 开启独立音色时必须报错
    s.distinct_voices = True
    assert any("不同音色" in e or "同时分配" in e for e in s.validate())


def test_counts_separate_roles_voices_system_and_concurrency(isolated_config):
    s = _gui_settings()
    counts = s.counts()
    assert counts["roles"] == 2 and counts["distinct_voices"] == 2
    assert counts["system_voices"] == 2  # plus 仅 2 个系统音色
    # 并发数不在 counts 内（由 generate.max_concurrency 单独控制）
    assert "concurrency" not in counts


# —— 12-16：导演指令四级层级 ————————————————————————————————————————————


def test_four_level_instruction_priority(tmp_path):
    cfg = make_cfg(tmp_path)
    lily = cfg.voices["id-lily"]
    leo = cfg.voices["id-leo"]

    seg_project = Segment(index=1, kind="speech", speaker="id-leo",
                          speaker_name="Leo", text="x")
    seg_role = Segment(index=2, kind="speech", speaker="id-lily",
                       speaker_name="Lily", text="x")
    seg_chapter = Segment(index=3, kind="speech", speaker="id-lily",
                          speaker_name="Lily", text="x",
                          chapter_direction=CHAPTER_INSTRUCTION)
    seg_inline = Segment(index=4, kind="speech", speaker="id-lily",
                         speaker_name="Lily", text="x",
                         chapter_direction=CHAPTER_INSTRUCTION,
                         direction="行内：小声说")

    # 项目级 < 角色默认级 < 章节级 < 行内片段级
    assert final_instruction(cfg, leo, seg_project) == PROJECT_INSTRUCTION
    assert final_instruction(cfg, lily, seg_role) == ROLE_LILY_INSTRUCTION
    assert final_instruction(cfg, lily, seg_chapter) == CHAPTER_INSTRUCTION
    assert final_instruction(cfg, lily, seg_inline) == "行内：小声说"
    # 指令是"取一条"，绝不拼接：行内结果不含任何上级指令文本
    assert CHAPTER_INSTRUCTION not in final_instruction(cfg, lily, seg_inline)
    assert PROJECT_INSTRUCTION not in final_instruction(cfg, lily, seg_inline)


def test_user_sample_each_request_carries_only_final_instruction(tmp_path):
    """用户验收样例：三段各自的最终 instruction 与实际发送值。"""
    cfg, provider, outcome = generate(tmp_path, THREE_LINES)
    assert outcome.failed == 0
    reqs = sorted(provider.calls, key=lambda r: r.output_path)

    # 第一段：无行内/章节/角色以外更细指令？Lily 有角色默认指令 → 角色默认生效
    assert reqs[0].direction == ROLE_LILY_INSTRUCTION
    # 第二段：Leo 无角色默认 → 继承项目级
    assert reqs[1].direction == PROJECT_INSTRUCTION
    # 第三段：行内"轻快地"覆盖一切
    assert reqs[2].direction == "轻快地"

    # 每请求只有唯一一条 instruction（TTSRequest.direction 单字段，且不拼接）
    for req, expected_text in zip(reqs, ["第一句", "第二句", "第三句"]):
        assert isinstance(req.direction, str) and req.direction
        assert req.text == expected_text
        assert "instruction" not in req.audio_params  # 指令不混入参数桶


def test_no_instruction_sends_none(tmp_path):
    cfg = make_cfg(tmp_path, default_instruction=None)
    cfg.voices["id-leo"] = VoiceSetting(
        voice="longanlufeng", language_type="Chinese", display_name="Leo"
    )
    _, provider, _ = generate(
        tmp_path, "Leo：只有一句，没有任何层级指令。\n", cfg=cfg
    )
    assert provider.calls[0].direction is None


def test_chapter_level_instruction_from_book(tmp_path):
    """章节级指令：parse_book 产出 chapter_direction 并在请求中生效。"""
    cfg = make_cfg(tmp_path, default_instruction=None)
    cfg.voices["id-lily"] = VoiceSetting(
        voice="longanlingxin", language_type="Chinese", display_name="Lily"
    )
    source = (
        "# 一\n\n"
        "- 角色: Lily\n"
        "- 导演: 本章平静讲述\n\n"
        "正文内容。\n"
    )
    parsed = parse_book(
        source, {"Lily": "id-lily"}, source_file="ch.md"
    )
    seg = parsed.segments[0]
    assert seg.chapter_direction == "本章平静讲述"
    setting, instruction, _ = plan_segment(cfg, seg)
    assert instruction == "本章平静讲述"


# —— 17-18：rate/pitch/volume 三层 ————————————————————————————————————


PARAM_SCRIPT = (
    "Lily(轻快地; rate=1.5, volume=70)：第一句\n\n"
    "Leo：第二句\n\n"
    "Lily：第三句\n"
)


def test_param_three_levels_and_no_leak_between_segments(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.audio_params = QwenAudioParams(rate=1.1, volume=50)  # 项目默认
    cfg.voices["id-lily"] = VoiceSetting(
        voice="longanlingxin", language_type="Chinese",
        display_name="Lily", rate=1.3,
    )
    path = write_script(tmp_path, PARAM_SCRIPT)
    provider = RecordingProvider()
    outcome = run_generate(path, cfg, provider=provider)
    assert outcome.failed == 0
    reqs = sorted(provider.calls, key=lambda r: r.output_path)

    # 段1：行内覆盖（rate 1.5 / volume 70）
    assert reqs[0].audio_params["rate"] == 1.5
    assert reqs[0].audio_params["volume"] == 70
    # 段2：Leo 无角色默认 → 项目默认；上一段的行内值不得泄露
    assert reqs[1].audio_params["rate"] == 1.1
    assert reqs[1].audio_params["volume"] == 50
    # 段3：Lily 角色默认（无行内）
    assert reqs[2].audio_params["rate"] == 1.3
    assert reqs[2].audio_params["volume"] == 50
    # 未设置 pitch：任何一段都不发送
    assert all("pitch" not in r.audio_params for r in reqs)


def test_role_param_unsupported_provider_is_config_error(tmp_path):
    cfg = make_cfg(
        tmp_path,
        provider="qwen",
        model="qwen3-tts-instruct-flash",
        voices={"id-lily": VoiceSetting(
            voice="Cherry", language_type="English",
            display_name="Lily", rate=1.2)},
    )
    seg = Segment(index=1, kind="speech", speaker="id-lily",
                  speaker_name="Lily", text="x")
    with pytest.raises(ConfigError, match="不支持语速"):
        plan_segment(cfg, seg)


def test_role_param_unknown_model_is_config_error(tmp_path):
    cfg = make_cfg(
        tmp_path,
        model="qwen-audio-9.9-tts",
        voices={"id-lily": VoiceSetting(
            voice="clone-001", language_type=None, custom=True,
            display_name="Lily", volume=80)},
    )
    seg = Segment(index=1, kind="speech", speaker="id-lily",
                  speaker_name="Lily", text="x")
    with pytest.raises(ConfigError, match="能力矩阵"):
        plan_segment(cfg, seg)


def test_param_change_enters_cache_hash_and_regenerates(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.audio_params = QwenAudioParams(rate=1.1, volume=50)
    cfg.voices["id-lily"] = VoiceSetting(
        voice="longanlingxin", language_type="Chinese",
        display_name="Lily", rate=1.3,
    )
    path = write_script(tmp_path, PARAM_SCRIPT)

    p1 = RecordingProvider()
    run_generate(path, cfg, provider=p1)
    assert len(p1.calls) == 3

    # 同配置二次运行：全部缓存命中，零请求
    p2 = RecordingProvider()
    run_generate(path, cfg, provider=p2)
    assert p2.calls == []

    # 仅修改 Lily 的角色默认语速：
    # 段1 行内 rate=1.5 已覆盖角色默认 → 不重生成；段3 无行内 → 必须重生成；
    # 段2 是 Leo → 仍命中缓存。
    cfg.voices["id-lily"] = VoiceSetting(
        voice="longanlingxin", language_type="Chinese",
        display_name="Lily", rate=1.6,
    )
    p3 = RecordingProvider()
    run_generate(path, cfg, provider=p3)
    regenerated = sorted(
        Path(r.output_path).stem.split("_", 1)[1] for r in p3.calls
    )
    assert regenerated == ["id-lily"]
    # 段3（第三句）的请求确实带着新角色默认值
    req3 = next(r for r in p3.calls if r.text == "第三句")
    assert req3.audio_params["rate"] == 1.6


def test_ssml_attribute_takes_precedence_over_http_param(tmp_path):
    """SSML <speak rate=...> 优先级最高：HTTP 侧不再发送同名 rate。"""
    cfg = make_cfg(tmp_path)
    cfg.audio_params = QwenAudioParams(rate=1.2, enable_ssml=True)
    cfg.voices["id-lily"] = VoiceSetting(
        voice="clone-001", language_type="Chinese", custom=True,
        display_name="Lily", ssml=True,
    )
    _, provider, outcome = generate(
        tmp_path, "Lily：<speak rate=\"1.5\">你好</speak>\n", cfg=cfg
    )
    assert outcome.failed == 0
    params = provider.calls[0].audio_params
    assert params.get("enable_ssml") is True
    assert "rate" not in params  # SSML 标签属性优先，避免双重指定


def test_utf8_bom_in_script_file_is_stripped(tmp_path):
    """Windows 记事本/PowerShell 常写出 UTF-8 BOM：首行角色不得变成 '\\ufeffLily'。"""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True)
    path = scripts_dir / "bom.md"
    path.write_bytes(("\ufeff" + THREE_LINES).encode("utf-8"))
    cfg = make_cfg(tmp_path)
    provider = RecordingProvider()
    outcome = run_generate(path, cfg, provider=provider)
    assert outcome.failed == 0
    reqs = sorted(provider.calls, key=lambda r: r.output_path)
    assert [r.text for r in reqs] == ["第一句", "第二句", "第三句"]


# —— 21：完成顺序与脚本顺序无关 ————————————————————————————————————————


def test_completion_order_does_not_change_segment_order(tmp_path):
    cfg = make_cfg(tmp_path, max_concurrency=3)
    path = write_script(tmp_path, THREE_LINES)
    provider = RecordingProvider(delays={"第二句": 0.4, "第一句": 0.1,
                                          "第三句": 0.1})
    run_generate(path, cfg, provider=provider)
    # 实际完成顺序：段1/段3 先回，段2 最后
    assert provider.completed[0] != "第二句"
    assert provider.completed[-1] == "第二句"
    # 但 manifest（拼接依据）严格按脚本顺序
    from podcast_tts.manifest import load as load_manifest
    manifest = load_manifest(cfg.projects_dir / "ep")
    speech = [r for r in manifest.segments if r.kind == "speech"]
    assert ([r.speaker_name for r in speech]
            == ["Lily", "Leo", "Lily"])
    assert [r.text for r in speech] == ["第一句", "第二句", "第三句"]
