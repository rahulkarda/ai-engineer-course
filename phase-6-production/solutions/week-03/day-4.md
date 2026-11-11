# Solutions — Week 03 · Day 4: Rate Limiting & Backpressure

---

## Exercise 1 — `try_acquire`

```python
async def try_acquire(self, tokens: float = 1.0) -> bool:
    """Non-blocking acquire. Returns True if tokens acquired, False if insufficient."""
    async with self._lock:
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False


# Tests
@pytest.mark.asyncio
async def test_try_acquire_success() -> None:
    bucket = TokenBucket(rate=10.0, capacity=5.0)
    assert await bucket.try_acquire(3.0) is True


@pytest.mark.asyncio
async def test_try_acquire_insufficient() -> None:
    bucket = TokenBucket(rate=1.0, capacity=2.0)
    await bucket.try_acquire(2.0)   # drain
    assert await bucket.try_acquire(1.0) is False
```

---

## Exercise 2 — Sliding window rate limiter

```python
import time
from collections import deque
from unittest.mock import patch


class SlidingWindowRateLimiter:
    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._timestamps: deque[float] = deque()

    def allow(self) -> bool:
        now = time.monotonic()
        cutoff = now - self._window
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
        if len(self._timestamps) >= self._max:
            return False
        self._timestamps.append(now)
        return True


def test_sliding_window_allows_within_limit() -> None:
    limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=1.0)
    t0 = 1000.0
    with patch("time.monotonic", return_value=t0):
        assert limiter.allow() is True
        assert limiter.allow() is True
        assert limiter.allow() is True
        assert limiter.allow() is False  # 4th within same second


def test_sliding_window_resets_after_window() -> None:
    limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=1.0)
    with patch("time.monotonic", return_value=1000.0):
        limiter.allow()
        limiter.allow()
    with patch("time.monotonic", return_value=1001.1):  # 1.1s later, outside window
        assert limiter.allow() is True
```

---

## Exercise 3 — Token-aware batch caller

```python
from dataclasses import dataclass


def estimate_tokens(prompt: str) -> int:
    return max(1, len(prompt) // 4)


async def rate_limited_llm_call(
    prompt: str,
    limiter: TokenBucket,
    model_tpm_limit: int = 100_000,
) -> str:
    estimated_tokens = estimate_tokens(prompt)
    await limiter.acquire(float(estimated_tokens))
    logger.info("LLM called: tokens=%d", estimated_tokens)
    return f"Response to: {prompt}"


class BatchLLMCaller:
    def __init__(self, tpm_limit: int = 60_000) -> None:
        # Convert TPM → tokens per second
        self._limiter = TokenBucket(rate=tpm_limit / 60.0, capacity=tpm_limit / 60.0 * 10)
        self._tpm_limit = tpm_limit

    async def call_all(self, prompts: list[str]) -> list[str]:
        tasks = [
            rate_limited_llm_call(p, self._limiter, self._tpm_limit)
            for p in prompts
        ]
        return await asyncio.gather(*tasks)
```

---

## Exercise 4 — Cancelled while blocked in `queue.put`

**Explanation:**

`asyncio.Queue.put()` is a coroutine that suspends when the queue is full. If the task is cancelled while suspended in `put()`, Python raises `asyncio.CancelledError` inside the coroutine at the `await` point. A `try/except asyncio.CancelledError` block will catch it. If you swallow `CancelledError` (i.e., don't re-raise), the task continues running — this is almost always a bug because the cancellation request is ignored and the task cannot be shut down cleanly.

**Rule**: Always re-raise `CancelledError` after any cleanup. Never swallow it.

```python
@pytest.mark.asyncio
async def test_cancelled_in_put_propagates() -> None:
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
    await queue.put("fill")  # queue full

    cancelled_correctly = False

    async def slow_producer() -> None:
        nonlocal cancelled_correctly
        try:
            await queue.put("blocked item")
        except asyncio.CancelledError:
            cancelled_correctly = True
            raise  # must re-raise

    task = asyncio.create_task(slow_producer())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled_correctly
```
