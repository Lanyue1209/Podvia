"""TTS Provider 抽象基类与异常体系（文档 7.1 / 7.2）。

约定：
- Provider 只负责"调 API → 下载音频 → 落盘"，不做重试（pipeline 统一调度，便于统计 attempts）
- 失败一律抛 ProviderError 子类；retryable 属性是重试模块判断的唯一依据
- 核心生产模块只 import 本模块与 models，不直接依赖任何厂商 SDK
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from podcast_tts.models import TTSRequest, TTSResult


class ProviderError(Exception):
    """Provider 异常基类，携带排查所需的上下文属性。"""

    def __init__(
        self,
        message: str,
        *,
        request_id: str | None = None,
        http_status: int | None = None,
        retryable: bool = False,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.request_id = request_id      # 厂商 request_id，便于工单排查
        self.http_status = http_status    # HTTP 状态码（网络异常时为 None）
        self.retryable = retryable        # 重试模块判断的唯一依据
        self.retry_after = retry_after    # 429 时服务端建议的等待秒数（None=未提供）


class RetryableError(ProviderError):
    """可重试：网络超时、5xx、429 限流、下载失败、响应异常格式等瞬时故障。"""

    def __init__(self, message: str, **kwargs) -> None:
        kwargs.pop("retryable", None)
        super().__init__(message, retryable=True, **kwargs)


class FatalProviderError(ProviderError):
    """不可重试：Key 无效、参数/voice 错误、文本为空等——重试无意义。"""

    def __init__(self, message: str, **kwargs) -> None:
        kwargs.pop("retryable", None)
        super().__init__(message, retryable=False, **kwargs)


class TTSProvider(ABC):
    """所有 TTS 厂商适配器的统一接口。"""

    name: str = ""    # 注册名，如 "qwen"（与 config.yaml 的 provider 字段对应）
    model: str = ""

    @abstractmethod
    def generate(self, request: TTSRequest) -> TTSResult:
        """同步生成：调 API → 下载音频 → 写入 request.output_path。

        失败时抛 ProviderError 子类。重试逻辑不在此层。
        """

    def warmup(self) -> None:
        """可选：启动时校验 API Key / 连通性。默认空实现。"""
        return None
