"""API Key 本机安全存储单元测试（不触网、不使用真实 Key）。

Windows 上验证真实 DPAPI roundtrip；非 Windows 降级路径通过 monkeypatch 覆盖。
"""
from __future__ import annotations

import base64

import pytest

from podcast_tts.gui import secret_store

FAKE_KEY = "sk-test-aaaaaaaaaaaaaaaaaaaaaaaa"


def test_protect_unprotect_roundtrip_windows():
    token = secret_store.protect(FAKE_KEY)
    # Windows 走 DPAPI；其他环境退化为显式 plain: 标记
    assert token.startswith(("dpapi:", "plain:"))
    assert secret_store.unprotect(token) == FAKE_KEY


def test_token_never_contains_plaintext_bytes():
    token = secret_store.protect(FAKE_KEY)
    if token.startswith("dpapi:"):
        # DPAPI 密文是加密二进制的 base64，不含明文片段
        assert FAKE_KEY.encode() not in base64.b64decode(token[len("dpapi:"):])
    assert FAKE_KEY not in token


def test_empty_string_roundtrip():
    assert secret_store.unprotect(secret_store.protect("")) == ""
    assert secret_store.unprotect("") == ""


def test_legacy_plaintext_token_passthrough():
    # 旧版直接落盘的明文 Key：unprotect 原样返回，由 settings 层迁移
    assert secret_store.unprotect(FAKE_KEY) == FAKE_KEY


def test_mask_key_never_reveals_full_value():
    masked = secret_store.mask_key(FAKE_KEY)
    assert masked and FAKE_KEY not in masked
    assert masked.startswith(FAKE_KEY[:4]) and masked.endswith(FAKE_KEY[-3:])
    assert secret_store.mask_key("short") == "*****"
    assert secret_store.mask_key("") == ""


def test_non_windows_fallback_is_explicitly_marked(monkeypatch):
    monkeypatch.setattr(secret_store, "_dpapi_available", lambda: False)
    token = secret_store.protect(FAKE_KEY)
    assert token.startswith("plain:")
    assert secret_store.unprotect(token) == FAKE_KEY


def test_dpapi_token_cannot_unprotect_without_dpapi(monkeypatch):
    token = secret_store.protect(FAKE_KEY)
    if not token.startswith("dpapi:"):
        pytest.skip("当前环境未产生 DPAPI 令牌")
    monkeypatch.setattr(secret_store, "_dpapi_available", lambda: False)
    with pytest.raises(ValueError):
        secret_store.unprotect(token)


def test_restrict_file_permissions_non_windows_returns_false(monkeypatch, tmp_path):
    monkeypatch.setattr(secret_store, "_dpapi_available", lambda: False)
    p = tmp_path / "x.json"
    p.write_text("{}")
    assert secret_store.restrict_file_permissions(p) is False
