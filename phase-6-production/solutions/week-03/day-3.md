# Solutions — Week 03 · Day 3: Circuit Breakers, Timeouts & Bulkheads

---

## Exercise 1 — HALF_OPEN concurrent probe limit

```python
from day3_circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitOpenError
import asyncio
import pytest


class CircuitBreakerV2(CircuitBreaker):
    """Extended circuit breaker with half-open concurrency limit."""

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None) -> None:
        super().__init__(name, config)
        self._half_open_active = 0

    async def call(self, fn):
        current_state = self.state
        if current_state == CircuitState.OPEN:
            raise CircuitOpenError(f"CircuitBreaker[{self.name}] is OPEN")
        if current_state == CircuitState.HALF_OPEN:
            max_probes = getattr(self.cfg, "half_open_max_calls", 1)
            if self._half_open_active >= max_probes:
                raise CircuitOpenError(
                    f"CircuitBreaker[{self.name}] HALF_OPEN probe limit reached"
                )
            self._half_open_active += 1
        try:
            result = await fn()
            self._on_success()
            return result
        except CircuitOpenError:
            raise
        except Exception:
            self._on_failure()
            raise
        finally:
            if current_state == CircuitState.HALF_OPEN:
                self._half_open_active -= 1


@pytest.mark.asyncio
async def test_half_open_max_calls() -> None:
    from pydantic import BaseModel

    class ExtConfig(CircuitBreakerConfig):
        half_open_max_calls: int = 1

    breaker = CircuitBreakerV2("test", ExtConfig(failure_threshold=1, reset_timeout=0.01))

    # Trip the breaker
    try:
        await breaker.call(lambda: (_ for _ in ()).throw(ConnectionError("fail")))
    except ConnectionError:
        pass

    await asyncio.sleep(0.02)  # let reset timeout pass → HALF_OPEN

    # Two concurrent probes — second should be rejected
    results = await asyncio.gather(
        breaker.call(lambda: asyncio.sleep(0.05)),
        breaker.call(lambda: asyncio.sleep(0.05)),
        return_exceptions=True,
    )
    assert any(isinstance(r, CircuitOpenError) for r in results)
```

---

## Exercise 2 — `CircuitBreaker.metrics()`

```python
def metrics(self) -> dict[str, Any]:
    return {
        "state": self._state.value,
        "failure_count": self._failure_count,
        "success_count": self._success_count,
        "opened_at": self._opened_at if self._state != CircuitState.CLOSED else None,
    }


def test_metrics_opened_at() -> None:
    from day3_circuit_breaker import CircuitBreaker, CircuitBreakerConfig
    breaker = CircuitBreaker("m", CircuitBreakerConfig(failure_threshold=1))

    # CLOSED — opened_at is None
    assert breaker.metrics()["opened_at"] is None

    # Force open
    breaker._failure_count = 1
    breaker._on_failure()

    m = breaker.metrics()
    assert m["state"] == "open"
    assert isinstance(m["opened_at"], float)
```

---

## Exercise 3 — Bulkhead with stats

```python
import asyncio
import pytest
from dataclasses import dataclass


@dataclass
class BulkheadStats:
    total_calls: int = 0
    rejected_count: int = 0


class TrackedBulkhead(Bulkhead):
    def __init__(self, name: str, max_concurrent: int, max_wait_seconds: float = 0.0) -> None:
        super().__init__(name, max_concurrent, max_wait_seconds)
        self._stats = BulkheadStats()

    def stats(self) -> BulkheadStats:
        return BulkheadStats(
            total_calls=self._stats.total_calls,
            rejected_count=self._stats.rejected_count,
        )

    async def call(self, fn):
        self._stats.total_calls += 1
        try:
            return await super().call(fn)
        except RuntimeError:
            self._stats.rejected_count += 1
            raise


@pytest.mark.asyncio
async def test_bulkhead_rejected_count() -> None:
    bulkhead = TrackedBulkhead("test", max_concurrent=3, max_wait_seconds=0.001)

    async def slow():
        await asyncio.sleep(0.5)
        return "ok"

    results = await asyncio.gather(*[bulkhead.call(slow) for _ in range(10)], return_exceptions=True)
    s = bulkhead.stats()
    assert s.total_calls == 10
    assert s.rejected_count == 7  # 3 concurrent + 7 rejected
```

---

## Exercise 4 — `resilient_call` composition

```python
async def resilient_call(
    fn: Callable[[], Coroutine[Any, Any, T]],
    timeout_s: float,
    breaker: CircuitBreaker,
    bulkhead: Bulkhead,
) -> T:
    """
    Order: bulkhead → circuit breaker → timeout-wrapped call

    Rationale:
    1. Bulkhead first: resource gating happens at the outermost layer.
       If we're over capacity, reject before consuming any other resources
       (no token acquisition, no circuit breaker state change).

    2. Circuit breaker second: if the downstream is known-broken, fast-fail
       immediately after we've confirmed we have a slot. This avoids burning
       a bulkhead slot on a call we know will fail.

    3. Timeout innermost: wraps only the actual I/O, not the queueing time.
       Measuring timeout from inside the bulkhead gives accurate per-call
       latency data; measuring from outside would include queue wait time.
    """
    async def cb_and_timeout() -> T:
        return await breaker.call(lambda: with_timeout(fn(), timeout_s))

    return await bulkhead.call(cb_and_timeout)
```
