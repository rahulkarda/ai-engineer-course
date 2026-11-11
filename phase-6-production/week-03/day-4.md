# Phase 6 · Week 03 · Day 4 — Rate Limiting & Backpressure

> Objective: Implement token-bucket and leaky-bucket rate limiters and apply backpressure to agent pipelines under load.  ·  Time: ~1 hr  ·  Prereqs: [[day-3]] (circuit breakers, asyncio)

---

## Concept

### Rate limiting

Rate limiting caps how many requests a system sends or accepts over time. Two classic algorithms:

**Token bucket** — a bucket holds up to `capacity` tokens; tokens refill at `rate` per second. Each request consumes one token. If the bucket is empty, the request is rejected or delayed.

- Allows **bursting** up to `capacity` requests at once
- Long-term average capped at `rate` req/s
- Used by LLM APIs to enforce TPM/RPM limits

**Leaky bucket** — requests enter a fixed-size queue and drain at a constant rate. No bursting. Smooths traffic into steady output.

### Backpressure

Backpressure is the mechanism by which a slow consumer signals a fast producer to slow down. Without it, a fast producer fills queues until memory exhausts or latency spikes.

In agent systems, backpressure manifests as:
- `asyncio.Queue(maxsize=N)` blocking a producer when the queue is full
- An HTTP client receiving 429 and slowing its request rate
- A semaphore with `max_wait_seconds=0` rejecting overflow immediately

The goal is to **fail fast at the edge** rather than let pressure propagate through the system.

---

## How it works internally

**Token bucket refill formula:**

```
now = monotonic()
tokens += (now - last_refill) * rate
tokens = min(tokens, capacity)
last_refill = now
```

Each `acquire(n)` call: if `tokens >= n`, subtract and allow; else reject or sleep until enough tokens accumulate.

**Async-safe token bucket**: use `asyncio.Lock` to make `acquire` atomic; use `asyncio.sleep` to wait without blocking the event loop.

**`asyncio.Queue` backpressure**: producer calls `await queue.put(item)` — it blocks (suspends the coroutine) when `maxsize` is reached. Consumer drains at its own pace. This naturally throttles the producer without polling.

---

## Worked example

```python
"""
day4_rate_limiting.py — Token bucket, leaky bucket, and backpressure queue.

Install: pip install pydantic
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Token bucket ───────────────────────────────────────────────────────────────

class RateLimitExceeded(Exception):
    """Raised when no tokens are available and blocking is not allowed."""


class TokenBucket:
    """
    Thread/async-safe token bucket rate limiter.

    Args:
        rate: tokens added per second
        capacity: maximum burst size
    """

    def __init__(self, rate: float, capacity: float) -> None:
        self._rate = rate
        self._capacity = capacity
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now

    async def acquire(self, tokens: float = 1.0, block: bool = True) -> None:
        """
        Consume `tokens` from the bucket.

        Args:
            tokens: number of tokens to consume (e.g. token count for LLM)
            block: if True, sleep until tokens are available; else raise RateLimitExceeded
        """
        async with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return
            if not block:
                raise RateLimitExceeded(
                    f"Rate limit: need {tokens} tokens, have {self._tokens:.2f}"
                )
            # Calculate wait and sleep outside the lock to avoid blocking others
            deficit = tokens - self._tokens
            wait_seconds = deficit / self._rate

        logger.debug("Rate limiter: waiting %.3fs for %s tokens", wait_seconds, tokens)
        await asyncio.sleep(wait_seconds)

        async with self._lock:
            self._refill()
            self._tokens -= tokens

    @property
    def available_tokens(self) -> float:
        self._refill()
        return self._tokens


# ── Leaky bucket ───────────────────────────────────────────────────────────────

class LeakyBucket:
    """
    Leaky bucket: enforces constant output rate with bounded queue.
    Producer puts items; drain loop processes at `drain_rate` items/sec.
    """

    def __init__(self, capacity: int, drain_rate: float) -> None:
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=capacity)
        self._drain_rate = drain_rate

    async def put(self, item: Any) -> None:
        """Add item — raises asyncio.QueueFull if bucket is full (fast-fail)."""
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull as exc:
            raise RateLimitExceeded("Leaky bucket full — request dropped") from exc

    async def drain(self, handler: Any) -> None:
        """Drain loop: call `handler(item)` at drain_rate items/second."""
        interval = 1.0 / self._drain_rate
        while True:
            item = await self._queue.get()
            await handler(item)
            self._queue.task_done()
            await asyncio.sleep(interval)


# ── Backpressure demo with asyncio.Queue ──────────────────────────────────────

async def producer(queue: asyncio.Queue[str], n: int) -> None:
    """Fast producer — will block when queue is full (backpressure applied)."""
    for i in range(n):
        item = f"prompt-{i}"
        logger.info("Producer: queuing %s", item)
        await queue.put(item)  # blocks here when queue full
        logger.debug("Producer: enqueued %s", item)
    await queue.put(None)  # sentinel


async def consumer(queue: asyncio.Queue[str], limiter: TokenBucket) -> None:
    """Slow consumer with rate limiter."""
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        await limiter.acquire(1.0)
        logger.info("Consumer: processed %s", item)
        queue.task_done()


# ── LLM token-aware rate limiter ──────────────────────────────────────────────

async def rate_limited_llm_call(
    prompt: str,
    limiter: TokenBucket,
    estimated_tokens: int = 100,
) -> str:
    """
    Acquire `estimated_tokens` from the rate limiter before calling the LLM.
    Models like Claude have tokens-per-minute (TPM) limits — this simulates that.
    """
    await limiter.acquire(float(estimated_tokens))
    # Mock LLM — replace with real client in production
    logger.info("LLM called: '%s' (tokens=%d)", prompt[:40], estimated_tokens)
    return f"Response to: {prompt}"


# ── Demo ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    # Token bucket: 5 req/s, burst of 3
    limiter = TokenBucket(rate=5.0, capacity=3.0)

    # Burst 3 requests immediately (tokens available)
    for i in range(3):
        await limiter.acquire()
        logger.info("Burst request %d processed", i + 1)

    # 4th request: bucket empty, raises immediately in non-blocking mode
    try:
        await limiter.acquire(block=False)
    except RateLimitExceeded as e:
        logger.warning("Expected rejection: %s", e)

    # Backpressure: bounded queue of 3
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=3)
    slow_limiter = TokenBucket(rate=10.0, capacity=10.0)

    await asyncio.gather(
        producer(queue, n=6),
        consumer(queue, slow_limiter),
    )
    logger.info("Backpressure demo complete")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Exercises

1. Add a `TokenBucket.try_acquire(tokens) -> bool` method that returns `True` if tokens were acquired and `False` otherwise (never blocks). Write tests for both outcomes.

2. Implement a `SlidingWindowRateLimiter` that uses a `collections.deque` to track request timestamps and rejects requests if more than `max_requests` occurred in the last `window_seconds`. Write tests with simulated time using `unittest.mock.patch`.

3. Modify `rate_limited_llm_call` to accept a `model_tpm_limit: int` parameter and calculate `estimated_tokens` from the prompt length using a simple heuristic (`len(prompt) // 4`). Then build a `BatchLLMCaller` that takes a list of prompts and calls them all, respecting the TPM limit.

4. In the backpressure demo, the producer blocks on `await queue.put(item)`. What happens if the producer is inside a `try/except asyncio.CancelledError` block and the task is cancelled while blocked in `put`? Write a brief explanation and a test that demonstrates it.

(Solutions: ../solutions/week-03/day-4.md)

---

## Self-check

- [ ] I can implement a token bucket from scratch with async-safe locking
- [ ] I can explain the difference between token bucket (bursting) and leaky bucket (smoothing)
- [ ] I can use `asyncio.Queue(maxsize=N)` for producer-consumer backpressure
- [ ] I can apply token-aware rate limiting for LLM TPM/RPM quotas
- [ ] I can describe what happens to a bounded queue when both producer and consumer fail

---

## Supplementary links

- [Token bucket algorithm — Wikipedia](https://en.wikipedia.org/wiki/Token_bucket)
- [asyncio.Queue docs](https://docs.python.org/3/library/asyncio-queue.html)
- [Anthropic rate limits reference](https://docs.anthropic.com/en/api/rate-limits)
