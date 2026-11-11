# Phase 6 · Week 03 · Day 3 — Circuit Breakers, Timeouts & Bulkheads

> Objective: Implement circuit breakers to stop hammering failing services, set per-call timeouts, and isolate components with bulkheads.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]] (retries), [[day-2]] (idempotency)

---

## Concept

### Circuit breaker

When a downstream service is down, retries pile up and slow everything. A **circuit breaker** tracks failure rate and "opens" (blocks calls entirely) when failures exceed a threshold — giving the downstream time to recover.

States:

```
CLOSED ──(failures > threshold)──► OPEN ──(timeout elapsed)──► HALF-OPEN
   ▲                                                                │
   └────────────────(probe succeeds)──────────────────────────────┘
         (probe fails → OPEN again)
```

| State | Behavior |
|---|---|
| CLOSED | All calls pass through; failures tracked |
| OPEN | All calls fail immediately (fast-fail) |
| HALF-OPEN | Single probe call allowed; success → CLOSED, failure → OPEN |

### Timeout

Every network call needs a **deadline**. Without one, a hung LLM request blocks a thread/coroutine forever. Set timeouts at three layers:

1. **Connect timeout** — max time to establish a TCP connection
2. **Read timeout** — max time waiting for the first byte
3. **Total timeout** — end-to-end wall clock budget for the full call

### Bulkhead

From shipbuilding: compartmentalize so one flooding compartment doesn't sink the ship. In software: **isolate** resource pools (thread pools, semaphores, connection pools) so one slow dependency can't exhaust all shared resources.

For agents: give each LLM call type (fast/structured vs slow/generative) its own semaphore so a long generation doesn't block health checks.

---

## How it works internally

**Circuit breaker failure count** uses a sliding window — typically last N calls or last T seconds. Simple version: count failures in a fixed window.

**Half-open probe**: after `reset_timeout` seconds in OPEN, one request is let through. If it succeeds, the breaker closes; if it fails, it reopens for another `reset_timeout`.

**Bulkhead with `asyncio.Semaphore`**: a semaphore with capacity N allows at most N concurrent acquirers. Tasks beyond N wait (or fail fast if a timeout is set on the acquire).

---

## Worked example

```python
"""
day3_circuit_breaker.py — Circuit breaker, timeout, and bulkhead for LLM calls.

Install: pip install pydantic
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Callable, Coroutine, TypeVar

from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

T = TypeVar("T")


# ── Circuit breaker state ──────────────────────────────────────────────────────

class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerConfig(BaseModel):
    failure_threshold: int = 5       # failures before opening
    success_threshold: int = 2       # consecutive successes to close from half-open
    reset_timeout: float = 30.0      # seconds to stay OPEN before probe
    window_seconds: float = 60.0     # sliding window for failure counting


class CircuitOpenError(Exception):
    """Raised when a call is rejected because the circuit is OPEN."""


class CircuitBreaker:
    def __init__(self, name: str, config: CircuitBreakerConfig | None = None) -> None:
        self.name = name
        self.cfg = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0.0
        self._opened_at: float = 0.0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._opened_at >= self.cfg.reset_timeout:
                logger.info("CircuitBreaker[%s]: OPEN → HALF_OPEN (probe)", self.name)
                self._state = CircuitState.HALF_OPEN
        return self._state

    def _on_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.cfg.success_threshold:
                logger.info("CircuitBreaker[%s]: HALF_OPEN → CLOSED", self.name)
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
        else:
            self._failure_count = 0

    def _on_failure(self) -> None:
        self._last_failure_time = time.monotonic()
        if self._state == CircuitState.HALF_OPEN:
            logger.warning("CircuitBreaker[%s]: HALF_OPEN probe failed → OPEN", self.name)
            self._open()
            return
        self._failure_count += 1
        if self._failure_count >= self.cfg.failure_threshold:
            logger.warning(
                "CircuitBreaker[%s]: threshold reached (%d) → OPEN",
                self.name,
                self._failure_count,
            )
            self._open()

    def _open(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = time.monotonic()

    async def call(self, fn: Callable[[], Coroutine[Any, Any, T]]) -> T:
        """Execute `fn` through the circuit breaker."""
        if self.state == CircuitState.OPEN:
            raise CircuitOpenError(f"CircuitBreaker[{self.name}] is OPEN")
        try:
            result = await fn()
            self._on_success()
            return result
        except CircuitOpenError:
            raise
        except Exception:
            self._on_failure()
            raise


# ── Timeout wrapper ────────────────────────────────────────────────────────────

async def with_timeout(coro: Coroutine[Any, Any, T], seconds: float) -> T:
    """Wrap a coroutine with a total timeout. Raises asyncio.TimeoutError."""
    return await asyncio.wait_for(coro, timeout=seconds)


# ── Bulkhead ───────────────────────────────────────────────────────────────────

class Bulkhead:
    """
    Limits concurrent calls to `max_concurrent`.
    Calls beyond that fail fast (or wait up to `max_wait_seconds`).
    """

    def __init__(self, name: str, max_concurrent: int, max_wait_seconds: float = 0.0) -> None:
        self.name = name
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_wait = max_wait_seconds

    async def call(self, fn: Callable[[], Coroutine[Any, Any, T]]) -> T:
        try:
            acquired = await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self._max_wait if self._max_wait > 0 else None,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"Bulkhead[{self.name}] full — rejected call"
            ) from exc
        try:
            return await fn()
        finally:
            self._semaphore.release()


# ── Mock LLM call ─────────────────────────────────────────────────────────────

_call_count = 0


async def flaky_llm_call(prompt: str, should_fail: bool = False) -> str:
    global _call_count
    _call_count += 1
    await asyncio.sleep(0.01)  # simulate latency
    if should_fail:
        raise ConnectionError("LLM API unavailable")
    return f"Response #{_call_count}: {prompt}"


# ── Demo ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    breaker = CircuitBreaker("llm-api", CircuitBreakerConfig(failure_threshold=3))
    bulkhead = Bulkhead("llm-generative", max_concurrent=2, max_wait_seconds=1.0)

    # Normal calls
    for i in range(2):
        result = await breaker.call(lambda: flaky_llm_call(f"Q{i}"))
        logger.info("OK: %s", result)

    # Trigger failures to open the circuit
    for i in range(3):
        try:
            await breaker.call(lambda: flaky_llm_call("fail", should_fail=True))
        except ConnectionError:
            logger.warning("Call failed (expected)")

    # Circuit should now be open — calls rejected immediately
    try:
        await breaker.call(lambda: flaky_llm_call("after open"))
    except CircuitOpenError as e:
        logger.error("Rejected: %s", e)

    # Bulkhead: only 2 concurrent calls allowed
    async def slow_llm() -> str:
        await asyncio.sleep(0.5)
        return "done"

    tasks = [bulkhead.call(slow_llm) for _ in range(4)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, Exception):
            logger.warning("Bulkhead rejected: %s", r)
        else:
            logger.info("Bulkhead OK: %s", r)


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Exercises

1. Add a `half_open_max_calls: int = 1` config field to `CircuitBreakerConfig` and enforce it: if the circuit is HALF_OPEN, at most `half_open_max_calls` probes are allowed concurrently. All others should raise `CircuitOpenError`. Write a test.

2. Implement a `CircuitBreaker.metrics()` method returning a `dict` with keys `state`, `failure_count`, `success_count`, `opened_at`. Write a test that verifies `opened_at` is `None` when CLOSED and a float when OPEN.

3. Modify `Bulkhead` to track `rejected_count` and `total_calls`. Add a `stats()` method. Write a test that runs 10 concurrent calls with `max_concurrent=3` (fast fail mode) and verifies 7 calls were rejected.

4. Combine circuit breaker + bulkhead + timeout into a single `resilient_call(fn, timeout_s, breaker, bulkhead)` function. The order should be: bulkhead acquire → circuit breaker check → timeout-wrapped call. Explain in a comment why that order matters.

(Solutions: ../solutions/week-03/day-3.md)

---

## Self-check

- [ ] I can explain the three circuit breaker states and transitions
- [ ] I can implement a circuit breaker with sliding failure window
- [ ] I can apply `asyncio.wait_for` for per-call timeouts
- [ ] I can use a semaphore-based bulkhead to cap concurrency
- [ ] I understand why bulkhead isolation prevents cascading failures

---

## Supplementary links

- [Martin Fowler — Circuit Breaker pattern](https://martinfowler.com/bliki/CircuitBreaker.html)
- [Python `asyncio.Semaphore` docs](https://docs.python.org/3/library/asyncio-sync.html#asyncio.Semaphore)
- [Netflix Hystrix design doc](https://github.com/Netflix/Hystrix/wiki/How-it-Works)
