"""阶段 2 单元测试：config.yaml 启动期校验（qwen_audio 段/音色/能力矩阵）。

全程使用临时目录隔离的 ROOT_DIR（含临时 .env），不读取仓库真实 .env，
不触网；校验失败必须抛 ConfigError 且信息可指导修复。
"""
from __future__ import annotations

import pytest

from podcast_tts import config as config_mod
from podcast_tts.config import ConfigError, QwenAudioParams, load_config

_BASE = """
provider: qwen_audio
model: qwen-audio-3.0-tts-plus
voices:
  Lily: { voice: "longanlingxin", language_type: "English" }
  Leo:  { voice: "longanlufeng", language_type: "Chinese" }
"""


@pytest.fixture
def isolated_root(tmp_path, monkeypatch):
    """把 config.ROOT_DIR 指向临时目录并准备 .env；返回 (root, write_cfg)。"""
    monkeypatch.setattr(config_mod, "ROOT_DIR", tmp_path)
    (tmp_path / ".env").write_text("DASHSCOPE_API_KEY=sk-test\n", encoding="utf-8")

    def write_cfg(body: str = _BASE):
        (tmp_path / "config.yaml").write_text(body, encoding="utf-8")

    return tmp_path, write_cfg


def test_default_config_loads_with_baseline_params(isolated_root):
    _, write_cfg = isolated_root
    write_cfg()
    cfg = load_config()
    assert cfg.provider == "qwen_audio"
    assert cfg.audio_params == QwenAudioParams()  # 全缺省
    assert cfg.audio_params.format == "wav" and cfg.audio_params.sample_rate == 24000


# —— 数值/枚举范围 ————————————————————————————————————————————————————


@pytest.mark.parametrize("key,bad", [("rate", 0.4), ("rate", 2.1), ("rate", "x")])
def test_rate_out_of_range(isolated_root, key, bad):
    _, write_cfg = isolated_root
    write_cfg(_BASE + f"qwen_audio:\n  {key}: {bad}\n")
    with pytest.raises(ConfigError, match="rate"):
        load_config()


def test_pitch_out_of_range(isolated_root):
    _, write_cfg = isolated_root
    write_cfg(_BASE + "qwen_audio:\n  pitch: 2.5\n")
    with pytest.raises(ConfigError, match="pitch"):
        load_config()


@pytest.mark.parametrize("bad", [-1, 101, 50.5, "loud"])
def test_volume_invalid(isolated_root, bad):
    _, write_cfg = isolated_root
    write_cfg(_BASE + f"qwen_audio:\n  volume: {bad!r}\n".replace("'loud'", "loud"))
    with pytest.raises(ConfigError, match="volume"):
        load_config()


@pytest.mark.parametrize("bad", [-1, 65536, 1.5])
def test_seed_invalid(isolated_root, bad):
    _, write_cfg = isolated_root
    write_cfg(_BASE + f"qwen_audio:\n  seed: {bad}\n")
    with pytest.raises(ConfigError, match="seed"):
        load_config()


def test_sample_rate_invalid(isolated_root):
    _, write_cfg = isolated_root
    write_cfg(_BASE + "qwen_audio:\n  sample_rate: 11025\n")
    with pytest.raises(ConfigError, match="sample_rate"):
        load_config()


@pytest.mark.parametrize("bad", ["pcm", "opus", "flac"])
def test_format_rejected(isolated_root, bad):
    _, write_cfg = isolated_root
    write_cfg(_BASE + f"qwen_audio:\n  format: {bad}\n")
    with pytest.raises(ConfigError):
        load_config()


def test_format_mp3_allowed(isolated_root):
    _, write_cfg = isolated_root
    write_cfg(_BASE + "qwen_audio:\n  format: mp3\n  sample_rate: 48000\n")
    cfg = load_config()
    assert cfg.audio_params.format == "mp3"
    assert cfg.audio_params.sample_rate == 48000


# —— language_hint —————————————————————————————————————————————————————


def test_language_hint_invalid_code(isolated_root):
    _, write_cfg = isolated_root
    write_cfg(_BASE + "qwen_audio:\n  language_hint: xx\n")
    with pytest.raises(ConfigError, match="language_hint"):
        load_config()


def test_voice_language_hint_and_type_conflict(isolated_root):
    _, write_cfg = isolated_root
    body = _BASE.replace(
        'Lily: { voice: "longanlingxin", language_type: "English" }',
        'Lily: { voice: "longanlingxin", language_type: "English", language_hint: "en" }',
    )
    write_cfg(body)
    with pytest.raises(ConfigError, match="不能同时配置"):
        load_config()


# —— hot_fix 结构（严格同构官方示例） ———————————————————————————————————


def test_hot_fix_valid(isolated_root):
    _, write_cfg = isolated_root
    write_cfg(
        _BASE
        + "qwen_audio:\n"
        + "  hot_fix:\n"
        + "    pronunciation:\n"
        + "      - {天气: 'tian1 qi4'}\n"
        + "    replace:\n"
        + "      - {今天: 金天}\n"
    )
    cfg = load_config()
    assert cfg.audio_params.hot_fix == {
        "pronunciation": [{"天气": "tian1 qi4"}],
        "replace": [{"今天": "金天"}],
    }


@pytest.mark.parametrize(
    "snippet",
    [
        "  hot_fix: []\n",                                  # 类型错（非映射）
        "  hot_fix: {spell: [{a: b}]}\n",                  # 未知键
        "  hot_fix: {pronunciation: []}\n",                # 空数组
        "  hot_fix: {pronunciation: [{a: b, c: d}]}\n",    # 多键对象
        "  hot_fix: {pronunciation: [{a: 5}]}\n",          # 值非字符串
        "  hot_fix: {pronunciation: [{'': b}]}\n",         # 空词
        "  hot_fix: {}\n",                                  # 空映射
    ],
)
def test_hot_fix_bad_structures(isolated_root, snippet):
    _, write_cfg = isolated_root
    write_cfg(_BASE + "qwen_audio:\n" + snippet)
    with pytest.raises(ConfigError):
        load_config()


# —— 边界键与跨 provider ————————————————————————————————————————————————


def test_enable_ssml_accepted_for_verified_model(isolated_root):
    _, write_cfg = isolated_root
    write_cfg(_BASE + "qwen_audio:\n  enable_ssml: true\n")
    cfg = load_config()
    assert cfg.audio_params.enable_ssml is True


def test_enable_ssml_non_bool_rejected(isolated_root):
    _, write_cfg = isolated_root
    write_cfg(_BASE + "qwen_audio:\n  enable_ssml: perhaps\n")  # 非布尔（注：YAML 中 yes/no 会被解析为布尔）
    with pytest.raises(ConfigError, match="true/false"):
        load_config()


def test_enable_ssml_rejected_for_unknown_model(isolated_root):
    _, write_cfg = isolated_root
    body = _voices_body("some-brand-new-model", "any-voice")
    write_cfg(body + "qwen_audio:\n  enable_ssml: true\n")
    with pytest.raises(ConfigError, match="SSML"):
        load_config()


def test_unknown_qwen_audio_key_rejected(isolated_root):
    _, write_cfg = isolated_root
    write_cfg(_BASE + "qwen_audio:\n  volumn: 80\n")  # 故意拼错
    with pytest.raises(ConfigError, match="未知配置项"):
        load_config()


def test_unknown_voice_key_rejected(isolated_root):
    _, write_cfg = isolated_root
    body = _BASE.replace(
        'Leo:  { voice: "longanlufeng", language_type: "Chinese" }',
        'Leo:  { voice: "longanlufeng", language_type: "Chinese", spd: 1.0 }',
    )
    write_cfg(body)
    with pytest.raises(ConfigError, match="未知配置项"):
        load_config()


def test_qwen_audio_section_rejected_for_qwen_provider(isolated_root):
    _, write_cfg = isolated_root
    body = _BASE.replace("provider: qwen_audio", "provider: qwen").replace(
        "model: qwen-audio-3.0-tts-plus", "model: qwen3-tts-instruct-flash"
    )
    body = body.replace(
        'voice: "longanlingxin"', 'voice: "Cherry"'
    ).replace('voice: "longanlufeng"', 'voice: "Ethan"')
    write_cfg(body + "qwen_audio:\n  rate: 1.2\n")
    with pytest.raises(ConfigError, match="专属参数"):
        load_config()


# —— 音色跨模型校验 ————————————————————————————————————————————————————


def _voices_body(model: str, lily_voice: str) -> str:
    return (
        f"provider: qwen_audio\nmodel: {model}\n"
        "voices:\n"
        f"  Lily: {{ voice: \"{lily_voice}\", language_type: \"English\" }}\n"
    )


def test_plus_rejects_flash_voice(isolated_root):
    _, write_cfg = isolated_root
    write_cfg(_voices_body("qwen-audio-3.0-tts-plus", "longanfengyue"))
    with pytest.raises(ConfigError, match="跨模型混用"):
        load_config()


def test_flash_rejects_plus_voice(isolated_root):
    _, write_cfg = isolated_root
    write_cfg(_voices_body("qwen-audio-3.0-tts-flash", "longanlingxin"))
    with pytest.raises(ConfigError):
        load_config()


def test_31_rejects_v36_voice(isolated_root):
    _, write_cfg = isolated_root
    write_cfg(_voices_body("qwen-audio-3.1-tts-flash", "longanhuan_v3.6"))
    with pytest.raises(ConfigError):
        load_config()


def test_model_specific_voices_accepted(isolated_root):
    _, write_cfg = isolated_root
    write_cfg(_voices_body("qwen-audio-3.0-tts-flash", "longanfengyue"))
    assert load_config().voices["Lily"].voice == "longanfengyue"
    write_cfg(_voices_body("qwen-audio-3.1-tts-flash", "longanhuan_v3.1"))
    assert load_config().voices["Lily"].voice == "longanhuan_v3.1"


def test_custom_voice_skips_whitelist(isolated_root):
    _, write_cfg = isolated_root
    body = (
        "provider: qwen_audio\nmodel: qwen-audio-3.0-tts-plus\n"
        "voices:\n"
        "  Narrator: { voice: \"qwen-audio-3.0-tts-plus-myclone\", "
        "language_type: \"Chinese\", custom: true }\n"
    )
    write_cfg(body)
    cfg = load_config()
    assert cfg.voices["Narrator"].custom is True


def test_custom_does_not_bypass_empty_voice(isolated_root):
    _, write_cfg = isolated_root
    body = (
        "provider: qwen_audio\nmodel: qwen-audio-3.0-tts-plus\n"
        'voices:\n  Narrator: { voice: "  ", custom: true }\n'
    )
    write_cfg(body)
    with pytest.raises(ConfigError, match="voice"):
        load_config()


def test_custom_must_be_bool(isolated_root):
    _, write_cfg = isolated_root
    body = (
        "provider: qwen_audio\nmodel: qwen-audio-3.0-tts-plus\n"
        'voices:\n  Narrator: { voice: "clone-x", custom: "yes" }\n'
    )
    write_cfg(body)
    with pytest.raises(ConfigError, match="custom"):
        load_config()


def test_unknown_model_skips_voice_whitelist(isolated_root):
    """未知模型无法建白名单：不阻断启动（运行期按基线降级）。"""
    _, write_cfg = isolated_root
    write_cfg(_voices_body("some-brand-new-model", "any-voice"))
    cfg = load_config()
    assert cfg.model == "some-brand-new-model"
