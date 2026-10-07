"""重试调度：仅对 RetryableError 指数退避重试（文档第 13 节）。

- FatalProviderError 立即向上抛（重试无意义）
- delay = base_delay * factor**(attempt-1)；429 带 Retry-After 时直接使用该值
- 重试次数耗尽后抛出最后一次的 RetryableError，由 pipeline 记入 manifest
- Provider 层不做重试；本模块也只做调度，不碰 manifest / 文件
"""
from __future__ import annotations

import time
from typing import Callable, TypeVar

from podcast_tts.providers.base import ProviderError, RetryableError

T = TypeVar("T")


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    max_retries: int,
    base_delay: float,
    factor: float = 2.0,
    on_retry: Callable[[int, int, float, ProviderError], None] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> T:
    """调用 fn()，失败时按策略重试。

    on_retry(attempt_total, total_attempts, delay, exc)：
      attempt_total 从 1 计（1=首次尝试），total_attempts = max_retries + 1。
    """
    total_attempts = max_retries + 1
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except RetryableError as exc:
            if attempt >= total_attempts:
                raise
            if exc.retry_after is not None:
                delay = exc.retry_after
            else:
                delay = base_delay * factor ** (attempt - 1)
            if on_retry is not None:
                on_retry(attempt, total_attempts, delay, exc)
            sleep_fn(delay)
        except ProviderError:
            # FatalProviderError 等不可重试错误立即上抛
            raise
