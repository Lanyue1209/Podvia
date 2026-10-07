"""阶段 4 单元测试：模型切换（provider/模型一致性、注册表、hash 变化）。

不触网；配置加载使用临时目录隔离 ROOT_DIR，不读取真实 .env。
"""
from __future__ import annotations

import pytest

from podcast_tts import capabilities as caps
from podcast_tts.config import ConfigError, QwenAudioParams, load_config
from podcast_tts.manifest import content_hash


def _cfg(provider: str, model: str, voice: str = "longanlingxin") -> str:
    return (
        f"provider: {provider}\nmodel: {model}\n"
        "voices:\n"
        f'  Lily: {{ voice: "{voice}" }}\n'
    )


@pytest.fixture
def isolated_root(tmp_path, monkeypatch):
    from podcast_tts import config as config_mod
    monkeypatch.setattr(config_mod, "ROOT_DIR", tmp_path)
    (tmp_path / ".env").write_text("DASHSCOPE_API_KEY=sk-test\n", encoding="utf-8")

    def write_cfg(body: str):
        (tmp_path / "config.yaml").write_text(body, encoding="utf-8")
    return write_cfg


# —— provider/模型一致性 ——————————————————————————————————————————————


def test_qwen_audio_model_with_qwen_provider_rejected(isolated_root):
    isolated_root(_cfg("qwen", "qwen-audio-3.0-tts-plus", voice="Cherry"))
    with pytest.raises(ConfigError, match="provider: qwen_audio"):
        load_config()


def test_qwen_tts_model_with_audio_provider_rejected(isolated_root):
    isolated_root(_cfg("qwen_audio", "qwen3-tts-flash"))
    with pytest.raises(ConfigError, match="provider: qwen"):
        load_config()


@pytest.mark.parametrize("model", [
    "cosyvoice-v3.5-flash", "cosyvoice-v2", "MiniMax/speech-2.8-hd", "speech-2.8-hd",
])
def test_unimplemented_models_refused(isolated_root, model):
    isolated_root(_cfg("qwen_audio", model))
    with pytest.raises(ConfigError, match="尚未实现|待实现"):
        load_config()


def test_qwen3_tts_models_load(isolated_root):
    isolated_root(_cfg("qwen", "qwen3-tts-instruct-flash", voice="Cherry"))
    cfg = load_config()
    assert cfg.provider == "qwen" and cfg.model == "qwen3-tts-instruct-flash"


def test_cross_family_voice_rejected_on_qwen_provider(isolated_root):
    # 从 qwen_audio 切到 qwen 后沿用 longan 音色 → 启动失败
    isolated_root(_cfg("qwen", "qwen3-tts-flash", voice="longanlingxin"))
    with pytest.raises(ConfigError, match="系统音色"):
        load_config()


def test_unknown_audio_model_still_baseline(isolated_root):
    isolated_root(_cfg("qwen_audio", "qwen-audio-9.9-tts-plus", voice="custom-voice"))
    cfg = load_config()  # 不阻断（custom 跳过白名单）；运行期只发基线字段
    assert cfg.model == "qwen-audio-9.9-tts-plus"


# —— 注册表（GUI 唯一事实来源） ————————————————————————————————————————


def test_registry_lists_all_implemented_and_planned():
    profiles = caps.list_model_profiles()
    by_name = {p["model"]: p for p in profiles}

    for m in ("qwen-audio-3.0-tts-plus", "qwen-audio-3.0-tts-flash",
              "qwen-audio-3.1-tts-flash", "qwen3-tts-flash",
              "qwen3-tts-instruct-flash"):
        assert by_name[m]["status"] == "implemented"

    planned = [p for p in profiles if p["status"] == "planned"]
    labels = " ".join(p["model"] for p in planned)
    assert "cosyvoice" in labels.lower() and "minimax" in labels.lower()
    # 关键约束：没有任何 CosyVoice/MiniMax 条目伪装成 implemented
    assert all(
        not (p["status"] == "implemented" and ("cosyvoice" in p["model"].lower()
                                               or "minimax" in p["model"].lower()))
        for p in profiles
    )


def test_registry_voice_sets_do_not_overlap():
    profiles = {p["model"]: p for p in caps.list_model_profiles()
                if p.get("provider") == "qwen_audio"}
    plus = set(profiles["qwen-audio-3.0-tts-plus"]["voices"])
    flash30 = set(profiles["qwen-audio-3.0-tts-flash"]["voices"])
    flash31 = set(profiles["qwen-audio-3.1-tts-flash"]["voices"])
    # 3.0 plus 与 flash 的系统音色集合不相同（官方禁止跨模型混用）
    assert plus != flash31 and flash30 != flash31
    # 每个配置音色都能在注册表内找到所属模型
    for m, voices in (("qwen-audio-3.0-tts-plus", plus),
                      ("qwen-audio-3.0-tts-flash", flash30),
                      ("qwen-audio-3.1-tts-flash", flash31)):
        for v in voices:
            caps.validate_voice(m, v, custom=False)


def test_describe_model_status():
    assert caps.describe_model("qwen-audio-3.1-tts-flash") == "已实现"
    assert caps.describe_model("qwen3-tts-flash") == "已实现"
    assert caps.describe_model("cosyvoice-v3-flash") == "待实现"
    assert caps.describe_model("qwen-audio-9.9-tts-plus") == "待确认"


# —— 切换模型必须改变缓存 hash ————————————————————————————————————————


def test_hash_changes_when_model_or_provider_changes():
    from types import SimpleNamespace
    from dataclasses import asdict

    text = "同一句话"
    base = QwenAudioParams()
    cfg1 = SimpleNamespace(provider="qwen_audio", model="qwen-audio-3.0-tts-plus")
    cfg2 = SimpleNamespace(provider="qwen_audio", model="qwen-audio-3.1-tts-flash")
    cfg3 = SimpleNamespace(provider="qwen_audio", model="qwen-audio-3.0-tts-plus")
    h1 = content_hash(cfg1, text, "longanlingxin", None, None, asdict(base))
    h2 = content_hash(cfg2, text, "longwanxiaozhu_v3.1", None, None, asdict(base))
    h3 = content_hash(cfg3, text, "longanlingxin", None, None,
                      asdict(QwenAudioParams(enable_ssml=True)))
    assert h1 != h2 and h1 != h3
    assert len(h1) == 64 and h1 != h2  # sha256 hex；模型/参数不同 hash 必不同
