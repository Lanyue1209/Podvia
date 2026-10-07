"""阶段 7 单元测试：冻结运行时引导（不触网、不依赖 PyInstaller）。"""
from __future__ import annotations

import sys

import pytest

from podcast_tts import runtime


def test_ensure_user_root_copies_template_and_creates_dirs(tmp_path):
    template = tmp_path / "tpl"
    template.mkdir()
    cfg = template / "config.yaml"
    cfg.write_text("provider: qwen_audio\n", encoding="utf-8")

    root = tmp_path / "appdata" / "Podvia"
    result = runtime.ensure_user_root(root, cfg)

    assert result == root
    assert (root / "config.yaml").read_text(encoding="utf-8").startswith("provider:")
    assert (root / "scripts").is_dir()
    assert (root / "projects").is_dir()
    assert (root / "logs").is_dir()
    assert (root / "tools" / "bin").is_dir()
    env_text = (root / ".env").read_text(encoding="utf-8")
    assert "DASHSCOPE_API_KEY=" in env_text
    # 模板式首启：.env 里绝不允许出现任何密钥
    assert "DASHSCOPE_API_KEY=\n" in env_text.replace("\r\n", "\n")


def test_ensure_user_root_never_overwrites_user_config(tmp_path):
    template = tmp_path / "tpl"
    template.mkdir()
    (template / "config.yaml").write_text("provider: qwen_audio\n", encoding="utf-8")
    root = tmp_path / "root"
    runtime.ensure_user_root(root, template / "config.yaml")

    (root / "config.yaml").write_text("provider: qwen  # 用户自定义", encoding="utf-8")
    (root / ".env").write_text("DASHSCOPE_API_KEY=sk-user-own\n", encoding="utf-8")
    runtime.ensure_user_root(root, template / "config.yaml")  # 再次启动

    assert "用户自定义" in (root / "config.yaml").read_text(encoding="utf-8")
    assert "sk-user-own" in (root / ".env").read_text(encoding="utf-8")


def test_ensure_user_root_missing_template_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        runtime.ensure_user_root(tmp_path / "root", tmp_path / "no-config.yaml")


def test_bundled_ffmpeg_dir_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.sys, "frozen", True, raising=False)
    monkeypatch.setattr(runtime, "bundle_dir", lambda: tmp_path)
    assert runtime.bundled_ffmpeg_dir() is None


def test_bundled_ffmpeg_dir_detected(tmp_path, monkeypatch):
    tb = tmp_path / "tools" / "bin"
    tb.mkdir(parents=True)
    (tb / "ffmpeg.exe").write_bytes(b"x")
    monkeypatch.setattr(runtime, "bundle_dir", lambda: tmp_path)
    assert runtime.bundled_ffmpeg_dir() == tb


def test_user_data_dir_uses_appdata(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    assert runtime.user_data_dir() == tmp_path / "Podvia"


def test_frozen_flag_reflects_sys(monkeypatch):
    assert runtime.is_frozen() is bool(getattr(sys, "frozen", False))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert runtime.is_frozen() is True
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert runtime.is_frozen() is False
