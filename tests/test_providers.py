"""Phase 3 单元测试：TTS Provider 抽象层与 Qwen 适配器。

全部 mock HTTP（monkeypatch requests.post / requests.get），不真实调用网络。
覆盖（Provider 抽象层与 Qwen 适配器）：
- 成功链路：payload 构造（text/voice/language_type/instructions 原文/optimize_instructions）
  、下载落盘、.part 原子改名、TTSResult 字段
- instruct / 非 instruct 模型的导演指令行为
- 错误映射表（业务码优先于 HTTP 状态）：
  401/403 鉴权与权限、400 参数错误、Arrearage 欠费、AllocationQuota.FreeTierOnly
  免费额度耗尽、AccessDenied 系列权限/未开通 → Fatal；
  400+Throttling/RateLimit、429+Retry-After、5xx、网络异常 → Retryable；
  body 层 code 分类；缺 audio.url / 非 JSON body → Retryable
- 下载失败、空文件 → Retryable；空文本 / 缺 API Key → Fatal
- 工厂 get_provider 与基类 warmup 默认实现
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
import requests

from podcast_tts.config import AppConfig, ConfigError, QwenAudioParams, VoiceSetting
from podcast_tts.models import TTSRequest
from podcast_tts.providers import get_provider
from podcast_tts.providers.base import (
    FatalProviderError,
    ProviderError,
    RetryableError,
    TTSProvider,
)
from podcast_tts.providers import qwen as qwen_mod
from podcast_tts.providers.qwen import QwenProvider

SUCCESS_BODY = {
    "status_code": 200,
    "request_id": "req-123",
    "output": {
        "finish_reason": "stop",
        "audio": {"url": "https://cdn.example/audio.wav", "expires_at": 1750000000},
    },
    "usage": {"characters": 123},
}


# —— 测试脚手架 ———————————————————————————————————————————————————


def make_cfg(**overrides) -> AppConfig:
    defaults = dict(
        provider="qwen",
        model="qwen3-tts-instruct-flash",
        voices={
            "Lily": VoiceSetting("Cherry", "English"),
            "Leo": VoiceSetting("Ethan", "Chinese"),
        },
        max_concurrency=3,
        request_timeout=5,
        max_retries=3,
        retry_base_delay=0.1,
        optimize_instructions=True,
        qwen_base_url="https://fake.example/api",
        qwen_audio_base_url="https://fake.example/audio-tts",
        audio_params=QwenAudioParams(),
        merge_format="mp3",
        bitrate="192k",
        scripts_dir=Path("scripts"),
        projects_dir=Path("projects"),
        logs_dir=Path("logs"),
        ffmpeg_dir=Path("tools/bin"),
        api_key="sk-test",
    )
    defaults.update(overrides)
    return AppConfig(**defaults)


def make_request(tmp_path: Path, **overrides) -> TTSRequest:
    defaults = dict(
        text="Hello everyone!",
        voice="Cherry",
        language_type="English",
        direction=None,
        output_path=str(tmp_path / "out.wav"),
    )
    defaults.update(overrides)
    return TTSRequest(**defaults)


class FakeResponse:
    """requests.Response 的最小替身（非流式）。"""

    def __init__(self, status_code=200, json_data=None, headers=None, text=""):
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else {}
        self.headers = headers or {}
        self.text = text

    def json(self):
        return self._json_data


class FakeStreamResponse:
    """requests.Response 的最小替身（流式下载）。"""

    def __init__(self, status_code=200, data: bytes = b""):
        self.status_code = status_code
        self.headers = {}
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_content(self, chunk_size: int):
        for i in range(0, len(self._data), chunk_size):
            yield self._data[i:i + chunk_size]


def patch_success(monkeypatch, *, data: bytes = b"RIFFwave") -> dict:
    """mock 完整成功链路，返回 captured 字典供断言 payload。"""
    captured: dict = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(url=url, json=json, headers=headers, timeout=timeout)
        return FakeResponse(200, SUCCESS_BODY)

    def fake_get(url, stream=False, timeout=None):
        captured["download_url"] = url
        return FakeStreamResponse(200, data)

    monkeypatch.setattr(qwen_mod.requests, "post", fake_post)
    monkeypatch.setattr(qwen_mod.requests, "get", fake_get)
    return captured


# —— 成功链路 ——————————————————————————————————————————————————————


def test_generate_success(tmp_path, monkeypatch):
    captured = patch_success(monkeypatch)
    provider = QwenProvider(make_cfg())
    out = tmp_path / "0001_Lily.wav"

    result = provider.generate(TTSRequest(
        text="Hello everyone!",
        voice="Cherry",
        language_type="English",
        direction="声音亮起来,像欢迎老朋友",
        output_path=str(out),
    ))

    assert result.output_path == str(out)
    assert result.request_id == "req-123"
    assert result.provider_data["usage"] == {"characters": 123}
    assert result.provider_data["audio_url"] == "https://cdn.example/audio.wav"

    assert out.read_bytes() == b"RIFFwave"          # 内容完整落盘
    assert not Path(str(out) + ".part").exists()    # 无 .part 残留

    # 请求体逐字段校验
    assert captured["url"] == "https://fake.example/api"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["timeout"] == 5
    payload = captured["json"]
    assert payload["model"] == "qwen3-tts-instruct-flash"
    assert payload["input"]["text"] == "Hello everyone!"
    assert payload["input"]["voice"] == "Cherry"
    assert payload["input"]["language_type"] == "English"
    # 导演指令原文传入，绝不改写
    assert payload["input"]["instructions"] == "声音亮起来,像欢迎老朋友"
    assert payload["input"]["optimize_instructions"] is True


def test_optional_fields_omitted(tmp_path, monkeypatch):
    """direction=None、language_type=None、optimize 关闭 → 相应字段不出现在请求中。"""
    captured = patch_success(monkeypatch)
    provider = QwenProvider(make_cfg(optimize_instructions=False))
    provider.generate(make_request(tmp_path, language_type=None, direction=None))

    assert set(captured["json"]["input"]) == {"text", "voice"}


def test_non_instruct_model_ignores_direction(tmp_path, monkeypatch, caplog):
    """非 instruct 模型：不发送 instructions，记录 WARNING，不报错不中断。"""
    captured = patch_success(monkeypatch)
    provider = QwenProvider(make_cfg(model="qwen3-tts-flash"))
    with caplog.at_level(logging.WARNING, logger="podcast_tts"):
        provider.generate(make_request(tmp_path, direction="开心一点"))

    assert "instructions" not in captured["json"]["input"]
    assert "optimize_instructions" not in captured["json"]["input"]
    assert any("不支持" in r.message for r in caplog.records)


# —— 错误映射表（HTTP 层） ——————————————————————————————————————————


@pytest.mark.parametrize("status", [401, 403])
def test_auth_error_fatal(tmp_path, monkeypatch, status):
    monkeypatch.setattr(
        qwen_mod.requests, "post",
        lambda *a, **k: FakeResponse(status, {"code": "InvalidApiKey", "message": "bad key"}),
    )
    provider = QwenProvider(make_cfg())
    with pytest.raises(FatalProviderError) as exc_info:
        provider.generate(make_request(tmp_path))
    assert exc_info.value.http_status == status
    assert exc_info.value.retryable is False


def test_400_invalid_parameter_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(
        qwen_mod.requests, "post",
        lambda *a, **k: FakeResponse(400, {"code": "InvalidParameter.voice", "message": "no such voice"}),
    )
    provider = QwenProvider(make_cfg())
    with pytest.raises(FatalProviderError, match="参数错误"):
        provider.generate(make_request(tmp_path))


def test_429_retryable_with_retry_after(tmp_path, monkeypatch):
    monkeypatch.setattr(
        qwen_mod.requests, "post",
        lambda *a, **k: FakeResponse(
            429, {"code": "Throttling", "message": "limit"}, headers={"Retry-After": "7"}
        ),
    )
    provider = QwenProvider(make_cfg())
    with pytest.raises(RetryableError, match="Retry-After: 7") as exc_info:
        provider.generate(make_request(tmp_path))
    assert exc_info.value.http_status == 429
    assert exc_info.value.retryable is True


@pytest.mark.parametrize("status", [500, 502, 503])
def test_5xx_retryable(tmp_path, monkeypatch, status):
    monkeypatch.setattr(
        qwen_mod.requests, "post",
        lambda *a, **k: FakeResponse(status, {"code": "InternalError", "message": "oops"}),
    )
    provider = QwenProvider(make_cfg())
    with pytest.raises(RetryableError) as exc_info:
        provider.generate(make_request(tmp_path))
    assert exc_info.value.http_status == status
    assert exc_info.value.retryable is True


@pytest.mark.parametrize(
    "exc",
    [
        requests.exceptions.ConnectionError("connection refused"),
        requests.exceptions.ReadTimeout("timed out"),
    ],
)
def test_network_exception_retryable(tmp_path, monkeypatch, exc):
    def fake_post(*a, **k):
        raise exc

    monkeypatch.setattr(qwen_mod.requests, "post", fake_post)
    provider = QwenProvider(make_cfg())
    with pytest.raises(RetryableError, match="网络请求失败"):
        provider.generate(make_request(tmp_path))


# —— 错误映射表（body 层：HTTP 200 但含 code） ——————————————————————


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("InvalidApiKey", FatalProviderError),
        ("InvalidParameter", FatalProviderError),
        ("ThrottlingException", RetryableError),
        ("RateLimitExceeded", RetryableError),
        ("InternalError", RetryableError),  # 未知 code 保守按可重试
    ],
)
def test_body_error_codes(tmp_path, monkeypatch, code, expected):
    monkeypatch.setattr(
        qwen_mod.requests, "post",
        lambda *a, **k: FakeResponse(
            200, {"code": code, "message": "boom", "request_id": "req-x"}
        ),
    )
    provider = QwenProvider(make_cfg())
    with pytest.raises(expected, match=code) as exc_info:
        provider.generate(make_request(tmp_path))
    assert exc_info.value.request_id == "req-x"


# —— 欠费 / 额度 / 权限细分（官方错误码，2026-10 核实） ——————————————————


def _post_returning(monkeypatch, status, body):
    monkeypatch.setattr(
        qwen_mod.requests, "post", lambda *a, **k: FakeResponse(status, body)
    )


_ARREARAGE_BODY = {
    "code": "Arrearage",
    "message": "Access denied, please make sure your account is in good standing.",
}


def test_400_arrearage_fatal(tmp_path, monkeypatch):
    """Arrearage 欠费在官方错误表中以 HTTP 400 返回 → 不可重试的 Fatal。"""
    _post_returning(monkeypatch, 400, _ARREARAGE_BODY)
    provider = QwenProvider(make_cfg())
    with pytest.raises(FatalProviderError, match="欠费|余额") as exc_info:
        provider.generate(make_request(tmp_path))
    assert exc_info.value.http_status == 400
    assert exc_info.value.retryable is False


def test_body_arrearage_fatal(tmp_path, monkeypatch):
    """HTTP 200 但 body 携带 Arrearage 同样致命（body 层错误码优先）。"""
    _post_returning(monkeypatch, 200, {"code": "Arrearage", "message": "overdue"})
    provider = QwenProvider(make_cfg())
    with pytest.raises(FatalProviderError, match="欠费|余额") as exc_info:
        provider.generate(make_request(tmp_path))
    assert exc_info.value.retryable is False


def test_403_free_tier_quota_fatal(tmp_path, monkeypatch):
    """免费额度耗尽（用完即停）：403 AllocationQuota.FreeTierOnly → Fatal 额度类。"""
    _post_returning(
        monkeypatch, 403,
        {"code": "AllocationQuota.FreeTierOnly", "message": "free tier quota used up"},
    )
    provider = QwenProvider(make_cfg())
    with pytest.raises(FatalProviderError, match="免费额度已耗尽") as exc_info:
        provider.generate(make_request(tmp_path))
    assert exc_info.value.http_status == 403
    # 额度耗尽不是欠费，错误文案必须与欠费区分
    assert "欠费" not in str(exc_info.value)


@pytest.mark.parametrize(
    "code",
    ["AccessDenied", "Workspace.AccessDenied", "Model.AccessDenied"],
)
def test_403_access_denied_permission_fatal(tmp_path, monkeypatch, code):
    """403 AccessDenied 系列是权限问题 → Fatal，且不归类为欠费/额度。"""
    _post_returning(monkeypatch, 403, {"code": code, "message": "Access denied."})
    provider = QwenProvider(make_cfg())
    with pytest.raises(FatalProviderError, match="无访问权限") as exc_info:
        provider.generate(make_request(tmp_path))
    text = str(exc_info.value)
    assert "欠费" not in text
    assert "免费额度" not in text


def test_403_unpurchased_fatal_not_arrears(tmp_path, monkeypatch):
    """AccessDenied.Unpurchased 表示模型服务未开通，不是欠费。"""
    _post_returning(
        monkeypatch, 403,
        {"code": "AccessDenied.Unpurchased", "message": "Access to model denied."},
    )
    provider = QwenProvider(make_cfg())
    with pytest.raises(FatalProviderError, match="未开通") as exc_info:
        provider.generate(make_request(tmp_path))
    assert "欠费" not in str(exc_info.value)


def test_403_without_code_fatal_not_arrears(tmp_path, monkeypatch):
    """403 无业务码：Fatal（不可重试），但不臆断为欠费。"""
    _post_returning(monkeypatch, 403, {"message": "Access denied."})
    provider = QwenProvider(make_cfg())
    with pytest.raises(FatalProviderError, match="鉴权失败或访问被拒绝") as exc_info:
        provider.generate(make_request(tmp_path))
    assert exc_info.value.http_status == 403
    assert "欠费" not in str(exc_info.value)


@pytest.mark.parametrize("code", ["Throttling", "RateLimitExceeded"])
def test_400_throttling_code_retryable(tmp_path, monkeypatch, code):
    """核心修正：HTTP 400 + 限流业务码时，业务码优先 → Retryable。"""
    _post_returning(
        monkeypatch, 400, {"code": code, "message": "rate increased too quickly"}
    )
    provider = QwenProvider(make_cfg())
    with pytest.raises(RetryableError) as exc_info:
        provider.generate(make_request(tmp_path))
    assert exc_info.value.http_status == 400
    assert exc_info.value.retryable is True


def test_429_without_code_still_retryable(tmp_path, monkeypatch):
    """429 无 body 业务码：按 HTTP 状态兜底为 Retryable。"""
    _post_returning(monkeypatch, 429, {"message": "Requests rate limit exceeded"})
    provider = QwenProvider(make_cfg())
    with pytest.raises(RetryableError) as exc_info:
        provider.generate(make_request(tmp_path))
    assert exc_info.value.http_status == 429
    assert exc_info.value.retryable is True


@pytest.mark.parametrize("code", ["ModelNotFound", "model_not_found", "WorkSpaceNotFound"])
def test_404_model_not_found_fatal(tmp_path, monkeypatch, code):
    """官方 404 模型/业务空间不存在：配置错误 → Fatal，重试无意义。"""
    _post_returning(monkeypatch, 404, {"code": code, "message": "Model can not be found."})
    provider = QwenProvider(make_cfg())
    with pytest.raises(FatalProviderError, match="模型或业务空间不存在") as exc_info:
        provider.generate(make_request(tmp_path))
    assert exc_info.value.http_status == 404
    assert exc_info.value.retryable is False


@pytest.mark.parametrize("body", [{"message": "not found"}, {"code": "SomeUnknown404", "message": "x"}])
def test_404_unknown_keeps_retryable(tmp_path, monkeypatch, body):
    """未知 404（无官方码或未知码）不扩大匹配，保守保持 Retryable 兜底。"""
    _post_returning(monkeypatch, 404, body)
    provider = QwenProvider(make_cfg())
    with pytest.raises(RetryableError) as exc_info:
        provider.generate(make_request(tmp_path))
    assert exc_info.value.http_status == 404
    assert exc_info.value.retryable is True


# —— 响应异常格式 / 下载 ————————————————————————————————————————————


def test_missing_audio_url_retryable(tmp_path, monkeypatch):
    body = {"request_id": "req-y", "output": {"finish_reason": "stop"}}
    monkeypatch.setattr(
        qwen_mod.requests, "post", lambda *a, **k: FakeResponse(200, body)
    )
    provider = QwenProvider(make_cfg())
    with pytest.raises(RetryableError, match="缺少 output.audio.url"):
        provider.generate(make_request(tmp_path))


def test_malformed_json_body_retryable(tmp_path, monkeypatch):
    class BadResponse(FakeResponse):
        def json(self):
            raise ValueError("not json")

    monkeypatch.setattr(
        qwen_mod.requests, "post", lambda *a, **k: BadResponse(200, text="<html>bad</html>")
    )
    provider = QwenProvider(make_cfg())
    with pytest.raises(RetryableError, match="不是合法 JSON"):
        provider.generate(make_request(tmp_path))


def test_download_http_error_retryable(tmp_path, monkeypatch):
    monkeypatch.setattr(
        qwen_mod.requests, "post", lambda *a, **k: FakeResponse(200, SUCCESS_BODY)
    )
    monkeypatch.setattr(
        qwen_mod.requests, "get", lambda *a, **k: FakeStreamResponse(404)
    )
    provider = QwenProvider(make_cfg())
    with pytest.raises(RetryableError, match="下载音频失败"):
        provider.generate(make_request(tmp_path))


def test_download_empty_file_retryable(tmp_path, monkeypatch):
    patch_success(monkeypatch, data=b"")
    monkeypatch.setattr(
        qwen_mod.requests, "get", lambda *a, **k: FakeStreamResponse(200, b"")
    )
    provider = QwenProvider(make_cfg())
    out = tmp_path / "out.wav"
    with pytest.raises(RetryableError, match="音频文件为空"):
        provider.generate(make_request(tmp_path))
    assert not out.exists()
    assert not Path(str(out) + ".part").exists()  # 空 .part 已清理


def test_download_network_error_retryable(tmp_path, monkeypatch):
    patch_success(monkeypatch)

    def fake_get(*a, **k):
        raise requests.exceptions.ConnectionError("cdn down")

    monkeypatch.setattr(qwen_mod.requests, "get", fake_get)
    provider = QwenProvider(make_cfg())
    with pytest.raises(RetryableError, match="下载音频失败"):
        provider.generate(make_request(tmp_path))


def test_download_midstream_error_cleans_part(tmp_path, monkeypatch):
    """下载到一半断连：已落盘的 .part 半截文件必须被清掉，不残留 segments/。"""
    patch_success(monkeypatch)

    class BrokenStream(FakeStreamResponse):
        def iter_content(self, chunk_size: int):
            yield b"RIFFpartial-data"
            raise requests.exceptions.ConnectionError("cdn cut mid-stream")

    monkeypatch.setattr(
        qwen_mod.requests, "get", lambda *a, **k: BrokenStream(200))
    provider = QwenProvider(make_cfg())
    out = tmp_path / "out.wav"
    with pytest.raises(RetryableError, match="下载音频失败"):
        provider.generate(make_request(tmp_path))
    assert not out.exists()
    assert not Path(str(out) + ".part").exists()


# —— Fatal 前置校验 ————————————————————————————————————————————————


def test_empty_text_fatal(tmp_path):
    provider = QwenProvider(make_cfg())
    with pytest.raises(FatalProviderError, match="文本为空"):
        provider.generate(make_request(tmp_path, text="   "))


def test_missing_api_key_fatal():
    with pytest.raises(FatalProviderError, match="DASHSCOPE_API_KEY"):
        QwenProvider(make_cfg(api_key=None))


# —— 异常属性约定 ——————————————————————————————————————————————————


def test_provider_error_attributes():
    err = ProviderError("base", request_id="r1", http_status=500, retryable=True)
    assert (err.request_id, err.http_status, err.retryable) == ("r1", 500, True)
    assert RetryableError("x").retryable is True
    assert FatalProviderError("x").retryable is False


# —— 工厂与基类 ————————————————————————————————————————————————————


def test_get_provider_qwen():
    provider = get_provider(make_cfg())
    assert isinstance(provider, QwenProvider)
    assert provider.name == "qwen"
    assert provider.model == "qwen3-tts-instruct-flash"


def test_get_provider_unknown():
    with pytest.raises(ConfigError, match="未知 provider"):
        get_provider(make_cfg(provider="cosyvoice"))


def test_warmup_default_noop():
    class DummyProvider(TTSProvider):
        def generate(self, request):
            raise NotImplementedError

    assert DummyProvider().warmup() is None
