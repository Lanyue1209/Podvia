"""Phase 9 单元测试：Qwen-Audio-TTS 适配器（qwen_audio provider）。

覆盖与 Qwen-TTS 的差异点：
- 端点使用 qwen_audio_base_url（非 multimodal-generation）
- 请求体：input 内含 text/voice/format=wav/sample_rate=24000
- 缺省请求体快照（旧配置逐键不变）；rate/pitch/volume/seed/hot_fix 透传
- language_hint（16 码）→ language_hints 单元素数组；language_type 旧映射
- 导演指令原文传入 input.instruction（该系列原生支持，非数值参数）
- 未知模型仅基线参数 + WARNING；TTSRequest.audio_params（pipeline 路径）
- 成功链路复用基类（响应结构一致）
- 工厂按 provider 分发
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from podcast_tts.config import QwenAudioParams
from podcast_tts.providers import get_provider
from podcast_tts.providers.base import FatalProviderError
from podcast_tts.providers.qwen import QwenProvider
from podcast_tts.providers.qwen_audio import QwenAudioProvider
from test_providers import SUCCESS_BODY, FakeResponse, make_cfg, make_request, patch_success

import podcast_tts.providers.qwen_audio as qwen_audio_mod


def make_audio_cfg(**overrides):
    """qwen_audio 专用配置：模型与端点均为 Qwen-Audio-TTS 系列。"""
    overrides.setdefault("model", "qwen-audio-3.0-tts-plus")
    overrides.setdefault("audio_params", QwenAudioParams())
    return make_cfg(**overrides)


# —— 请求构造 ————————————————————————————————————————————————————————


def test_payload_basic_shape():
    provider = QwenAudioProvider(make_audio_cfg())
    payload = provider._build_payload(make_request(Path(".")))

    assert payload["model"] == "qwen-audio-3.0-tts-plus"
    inp = payload["input"]
    assert inp["text"] == "Hello everyone!"
    assert inp["voice"] == "Cherry"
    assert inp["format"] == "wav"
    assert inp["sample_rate"] == 24000
    assert inp["language_hints"] == ["en"]  # English → en


def test_payload_language_chinese_and_none():
    provider = QwenAudioProvider(make_audio_cfg())

    payload = provider._build_payload(make_request(Path("."), language_type="Chinese"))
    assert payload["input"]["language_hints"] == ["zh"]

    payload = provider._build_payload(make_request(Path("."), language_type=None))
    assert "language_hints" not in payload["input"]


def test_payload_unknown_language_omits_and_warns(caplog):
    provider = QwenAudioProvider(make_audio_cfg())
    with caplog.at_level(logging.WARNING, logger="podcast_tts"):
        payload = provider._build_payload(
            make_request(Path("."), language_type="Japanese")
        )
    assert "language_hints" not in payload["input"]
    assert any("Japanese" in rec.message for rec in caplog.records)


def test_payload_instruction_passthrough_verbatim():
    provider = QwenAudioProvider(make_cfg())
    payload = provider._build_payload(
        make_request(Path("."), direction="压低一点声音,像说秘密")
    )
    # 原文传入，绝不改写
    assert payload["input"]["instruction"] == "压低一点声音,像说秘密"


def test_payload_no_direction_no_instruction_key():
    provider = QwenAudioProvider(make_cfg())
    payload = provider._build_payload(make_request(Path("."), direction=None))
    assert "instruction" not in payload["input"]


def test_endpoint_uses_audio_base_url(monkeypatch, tmp_path):
    captured = patch_success(monkeypatch)
    provider = QwenAudioProvider(make_cfg())
    provider.generate(make_request(tmp_path))
    assert captured["url"] == "https://fake.example/audio-tts"


def test_empty_text_fatal():
    provider = QwenAudioProvider(make_cfg())
    with pytest.raises(FatalProviderError):
        provider.generate(make_request(Path("."), text="  "))


# —— 继承与工厂 ————————————————————————————————————————————————————————


def test_inherits_qwen_provider():
    assert issubclass(QwenAudioProvider, QwenProvider)
    assert QwenAudioProvider.name == "qwen_audio"


def test_factory_dispatch():
    provider = get_provider(make_cfg(provider="qwen_audio"))
    assert isinstance(provider, QwenAudioProvider)


def test_generate_success_uses_base_flow(monkeypatch, tmp_path):
    """响应结构与 Qwen-TTS 一致 → 成功链路直接复用基类。"""
    out = tmp_path / "out.wav"
    captured = patch_success(monkeypatch)
    provider = QwenAudioProvider(make_audio_cfg())
    result = provider.generate(make_request(tmp_path, output_path=str(out)))

    assert captured["url"] == "https://fake.example/audio-tts"
    assert out.read_bytes() == b"RIFFwave"
    assert result.request_id == SUCCESS_BODY["request_id"]


# —— 阶段 2：音频参数增强 ——————————————————————————————————————————————


def test_default_payload_exact_snapshot():
    """旧配置请求体逐键快照：只有 text/voice/format/sample_rate/language_hints。"""
    provider = QwenAudioProvider(make_audio_cfg())
    payload = provider._build_payload(make_request(Path(".")))
    assert payload == {
        "model": "qwen-audio-3.0-tts-plus",
        "input": {
            "text": "Hello everyone!",
            "voice": "Cherry",
            "format": "wav",
            "sample_rate": 24000,
            "language_hints": ["en"],
        },
    }
    # 缺省可选键绝不发送
    for key in ("rate", "pitch", "volume", "seed", "hot_fix", "instruction"):
        assert key not in payload["input"]


@pytest.mark.parametrize(
    "field,value,api_key,api_value",
    [
        ("rate", 1.5, "rate", 1.5),
        ("pitch", 0.8, "pitch", 0.8),
        ("volume", 80, "volume", 80),
        ("seed", 42, "seed", 42),
    ],
)
def test_optional_numeric_params_passthrough(field, value, api_key, api_value):
    cfg = make_audio_cfg(audio_params=QwenAudioParams(**{field: value}))
    payload = QwenAudioProvider(cfg)._build_payload(make_request(Path(".")))
    assert payload["input"][api_key] == api_value


def test_optional_params_absent_when_none():
    cfg = make_audio_cfg(audio_params=QwenAudioParams(rate=None, seed=None))
    payload = QwenAudioProvider(cfg)._build_payload(make_request(Path(".")))
    assert "rate" not in payload["input"] and "seed" not in payload["input"]


def test_hot_fix_structure_matches_official_example():
    hot_fix = {
        "pronunciation": [{"天气": "tian1 qi4"}],
        "replace": [{"今天": "金天"}],
    }
    cfg = make_audio_cfg(audio_params=QwenAudioParams(hot_fix=hot_fix))
    payload = QwenAudioProvider(cfg)._build_payload(make_request(Path(".")))
    assert payload["input"]["hot_fix"] == hot_fix


def test_language_hint_short_code_maps_to_array():
    # language_hint 是 Podvia 配置名，API 字段必须是 language_hints 数组
    cfg = make_audio_cfg(audio_params=QwenAudioParams(language_hint="fr"))
    payload = QwenAudioProvider(cfg)._build_payload(
        make_request(Path("."), language_type=None)
    )
    assert payload["input"]["language_hints"] == ["fr"]


def test_format_mp3_in_payload():
    cfg = make_audio_cfg(audio_params=QwenAudioParams(format="mp3"))
    payload = QwenAudioProvider(cfg)._build_payload(make_request(Path(".")))
    assert payload["input"]["format"] == "mp3"


def test_unknown_model_baseline_only_and_warns(caplog):
    """未知模型：只有 format/sample_rate 基线字段，可选参数（含 hints）一律不发。"""
    cfg = make_audio_cfg(
        model="some-future-tts-model",
        audio_params=QwenAudioParams(rate=1.5, seed=7, language_hint="fr"),
    )
    with caplog.at_level(logging.WARNING, logger="podcast_tts"):
        payload = QwenAudioProvider(cfg)._build_payload(
            make_request(Path("."), language_type="English")
        )
    inp = payload["input"]
    assert set(inp) == {"text", "voice", "format", "sample_rate"}
    assert "rate" not in inp and "seed" not in inp and "language_hints" not in inp
    assert any("能力矩阵" in r.message for r in caplog.records)


def test_unknown_model_keeps_instruction():
    """基线保留导演指令（instruction 属基线字段）。"""
    cfg = make_audio_cfg(model="some-future-tts-model")
    payload = QwenAudioProvider(cfg)._build_payload(
        make_request(Path("."), direction="小声一点")
    )
    assert payload["input"]["instruction"] == "小声一点"


def test_request_audio_params_pipeline_path():
    """pipeline 路径：TTSRequest.audio_params 原样成为 input 附加字段。"""
    provider = QwenAudioProvider(make_audio_cfg())
    extras = {
        "format": "mp3", "sample_rate": 48000, "rate": 1.2,
        "hot_fix": {"replace": [{"AI": "A I"}]}, "language_hints": ["ja"],
    }
    payload = provider._build_payload(make_request(Path("."), audio_params=extras))
    assert payload["input"]["format"] == "mp3"
    assert payload["input"]["sample_rate"] == 48000
    assert payload["input"]["rate"] == 1.2
    assert payload["input"]["language_hints"] == ["ja"]
