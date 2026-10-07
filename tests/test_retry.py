"""Phase 5 单元测试：retry_with_backoff（文档第 13 节）。"""
from __future__ import annotations

import pytest

from podcast_tts.retry import retry_with_backoff
from podcast_tts.providers.base import FatalProviderError, RetryableError


def make_sleep_log():
    return lambda: []


def test_first_attempt_success_no_retry():
    calls = []
    sleeps = []

    def fn():
        calls.append(1)
        return "ok"

    result = retry_with_backoff(
        fn, max_retries=3, base_delay=2.0, sleep_fn=sleeps.append
    )
    assert result == "ok"
    assert len(calls) == 1
    assert sleeps == []


def test_retryable_then_success():
    calls = []
    sleeps = []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise RetryableError("transient")
        return "ok"

    result = retry_with_backoff(
        fn, max_retries=3, base_delay=2.0, sleep_fn=sleeps.append
    )
    assert result == "ok"
    assert len(calls) == 3
    # 指数退避：2 * 2^0=2, 2 * 2^1=4
    assert sleeps == [2.0, 4.0]


def test_exhausted_retries_raises_last_error():
    calls = []
    sleeps = []

    def fn():
        calls.append(1)
        raise RetryableError(f"still down {len(calls)}")

    with pytest.raises(RetryableError, match="still down 4"):
        retry_with_backoff(fn, max_retries=3, base_delay=2.0, sleep_fn=sleeps.append)
    # 首次 + 3 次重试 = 4 次尝试；3 次退避：2, 4, 8
    assert len(calls) == 4
    assert sleeps == [2.0, 4.0, 8.0]


def test_fatal_error_raises_immediately():
    calls = []
    sleeps = []

    def fn():
        calls.append(1)
        raise FatalProviderError("bad api key")

    with pytest.raises(FatalProviderError, match="bad api key"):
        retry_with_backoff(fn, max_retries=3, base_delay=2.0, sleep_fn=sleeps.append)
    assert len(calls) == 1
    assert sleeps == []


def test_retry_after_header_takes_precedence():
    calls = []
    sleeps = []

    def fn():
        calls.append(1)
        if len(calls) == 1:
            raise RetryableError("throttled", retry_after=7.0)
        return "ok"

    result = retry_with_backoff(
        fn, max_retries=3, base_delay=2.0, sleep_fn=sleeps.append
    )
    assert result == "ok"
    assert sleeps == [7.0]  # 服务端建议值优先于指数退避


def test_on_retry_callback_args():
    events = []
    sleeps = []
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise RetryableError("x", request_id="req-9")
        return "ok"

    def on_retry(attempt, total, delay, exc):
        events.append((attempt, total, delay, exc.request_id))

    retry_with_backoff(
        fn, max_retries=3, base_delay=1.0,
        on_retry=on_retry, sleep_fn=sleeps.append,
    )
    # attempt 从 1 计（1=首次尝试）；total = max_retries + 1 = 4
    assert events == [(1, 4, 1.0, "req-9"), (2, 4, 2.0, "req-9")]


def test_unexpected_exception_propagates():
    """非 ProviderError 的意外异常直接上抛（pipeline 会兜底捕获）。"""
    calls = []

    def fn():
        calls.append(1)
        raise ValueError("bug")

    with pytest.raises(ValueError, match="bug"):
        retry_with_backoff(fn, max_retries=3, base_delay=1.0, sleep_fn=lambda s: None)
    assert len(calls) == 1
