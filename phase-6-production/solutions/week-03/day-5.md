# Solutions — Week 03 · Day 5: Graceful Degradation, Fallbacks, Sagas & Chaos

---

## Exercise 1 — `fallback_chain` with deadline

```python
import time


async def fallback_chain_with_deadline(
    fns: list[AsyncFn[T]],
    label: str = "fallback_chain",
    max_fallback_latency_ms: float | None = None,
) -> T:
    start = time.monotonic()
    last_exc: Exception | None = None

    for i, fn in enumerate(fns):
        if max_fallback_latency_ms is not None:
            elapsed_ms = (time.monotonic() - start) * 1000
            if elapsed_ms >= max_fallback_latency_ms:
                raise TimeoutError(
                    f"{label}: fallback latency budget exhausted after {elapsed_ms:.0f}ms"
                )
        try:
            result = await fn()
            if i > 0:
                logger.warning("%s: using fallback #%d", label, i)
            return result
        except Exception as exc:
            logger.warning("%s: option %d failed — %s", label, i, exc)
            last_exc = exc

    raise RuntimeError(f"{label}: all options exhausted") from last_exc


@pytest.mark.asyncio
async def test_fallback_deadline_exceeded() -> None:
    async def slow_fail() -> str:
        await asyncio.sleep(0.2)
        raise RuntimeError("slow failure")

    with pytest.raises(TimeoutError):
        await fallback_chain_with_deadline(
            [slow_fail, slow_fail, slow_fail],
            max_fallback_latency_ms=150,
        )
```

---

## Exercise 2 — Choreography saga

```python
async def choreography_saga() -> None:
    events: asyncio.Queue[str] = asyncio.Queue()
    results: list[str] = []
    compensation_events: list[str] = []

    async def step1() -> None:
        results.append("slot_reserved")
        await events.put("step1.done")

    async def step2_listener() -> None:
        event = await events.get()
        if event == "step1.done":
            try:
                results.append("credits_deducted")
                await events.put("step2.done")
            except Exception:
                await events.put("step2.failed")

    async def compensation_listener() -> None:
        event = await events.get()
        if event == "step2.failed":
            compensation_events.append("slot_released")

    await asyncio.gather(step1(), step2_listener(), compensation_listener())
    return results, compensation_events


"""
Why choreography is harder to debug:
When you have 5+ steps spread across services, the causal chain of "which event
triggered which action" is implicit in the event names. There is no single place
that shows the saga's current state or history. A distributed trace must be
reconstructed from logs across multiple services. With orchestration, a single
coordinator object has the full execution history in memory.
"""
```

---

## Exercise 3 — `ChaosConfig.jitter_ms` with mock

```python
import asyncio
from unittest.mock import AsyncMock, patch

from day5_degradation_saga_chaos import ChaosConfig, chaos_wrap


@pytest.mark.asyncio
async def test_chaos_jitter_sleep_range() -> None:
    async def noop() -> str:
        return "ok"

    config = ChaosConfig(jitter_ms=200.0, error_rate=0.0, latency_p=0.0, seed=99)

    sleep_calls: list[float] = []

    async def fake_sleep(duration: float) -> None:
        sleep_calls.append(duration)

    with patch("asyncio.sleep", side_effect=fake_sleep):
        wrapped = chaos_wrap_with_jitter(noop, config)
        for _ in range(10):
            await wrapped()

    for duration in sleep_calls:
        assert 0 <= duration <= 0.200 + 1e-9, f"Jitter out of range: {duration}"
```

Modified `chaos_wrap` to add jitter:

```python
def chaos_wrap_with_jitter(
    fn: AsyncFn[T],
    config: ChaosConfig,
    label: str = "chaos",
) -> AsyncFn[T]:
    rng = random.Random(config.seed)

    async def wrapped() -> T:
        # Always apply jitter if configured
        if hasattr(config, "jitter_ms") and config.jitter_ms > 0:
            jitter = rng.uniform(0, config.jitter_ms / 1000.0)
            await asyncio.sleep(jitter)
        if rng.random() < config.error_rate:
            raise ConnectionError(f"CHAOS: injected failure in {label}")
        if rng.random() < config.latency_p:
            await asyncio.sleep(config.latency_ms / 1000.0)
        return await fn()

    return wrapped
```

---

## Exercise 4 — Saga compensation with retry

```python
import asyncio
import logging

logger = logging.getLogger(__name__)


class RetryingSagaOrchestrator(SagaOrchestrator):
    """Saga orchestrator that retries compensations up to 3 times."""

    async def _compensate(self, completed: list[int]) -> None:
        for i in reversed(completed):
            step = self._steps[i]
            if step.compensate is None:
                continue
            for attempt in range(1, 4):
                try:
                    logger.info(
                        "Saga: compensating step [%d] %s (attempt %d)", i, step.name, attempt
                    )
                    await step.compensate()
                    break
                except Exception:
                    if attempt == 3:
                        logger.exception(
                            "Saga: compensation failed 3 times for step [%d] %s — manual intervention needed",
                            i,
                            step.name,
                        )
                    else:
                        await asyncio.sleep(1.0)


@pytest.mark.asyncio
async def test_compensation_retry_succeeds_on_second_attempt() -> None:
    call_count = 0

    async def flaky_compensate() -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("compensation network error")

    async def failing_step() -> None:
        raise RuntimeError("step failed")

    saga = RetryingSagaOrchestrator([
        SagaStep("step1", lambda: asyncio.sleep(0), flaky_compensate),
        SagaStep("step2", failing_step),
    ])

    with pytest.raises(RuntimeError, match="step2"):
        await saga.run()

    assert call_count == 2  # failed once, succeeded on second attempt
```
