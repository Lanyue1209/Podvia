"""TTS Provider 抽象层：统一接口、异常体系与工厂（文档第 7 节）。

新增 Provider 的步骤（核心生产模块零改动）：
  1. 新增 providers/xxx.py 实现 TTSProvider 接口
  2. 在 get_provider 增加一个分支
  3. config.py / config.yaml 增加对应配置支持
"""
from __future__ import annotations

from podcast_tts.config import AppConfig, ConfigError
from podcast_tts.providers.base import (
    FatalProviderError,
    ProviderError,
    RetryableError,
    TTSProvider,
)

__all__ = [
    "TTSProvider",
    "ProviderError",
    "RetryableError",
    "FatalProviderError",
    "get_provider",
]


def get_provider(cfg: AppConfig) -> TTSProvider:
    """按配置构造 Provider。延迟导入：核心模块不加载厂商依赖。"""
    if cfg.provider == "qwen":
        from podcast_tts.providers.qwen import QwenProvider

        return QwenProvider(cfg)
    if cfg.provider == "qwen_audio":
        from podcast_tts.providers.qwen_audio import QwenAudioProvider

        return QwenAudioProvider(cfg)
    raise ConfigError(f"未知 provider: {cfg.provider}")
