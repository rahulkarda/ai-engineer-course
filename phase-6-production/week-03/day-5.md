# Phase 6 · Week 03 · Day 5 — Graceful Degradation, Fallbacks, Sagas & Chaos Intro

> Objective: Implement fallback chains for LLM failures, model the saga pattern for distributed agent transactions, and run a first chaos experiment.  ·  Time: ~1 hr  ·  Prereqs: [[day-3]] (circuit breakers), [[day-4]] (rate limiting)

---

## Concept

### Graceful degradation & fallbacks

A system degrades gracefully when it provides reduced-but-useful service instead of failing completely. For LLM agents:

| Failure | Graceful degradation |
|---|---|
| Primary model 503 | Retry with smaller/cheaper backup model |
| All models rate-limited | Return cached/stale result with staleness note |
| Tool call fails | Return partial result + "tool unavailable" annotation |
| Complete outage | Return static safe response |

A **fallback chain** tries each option in sequence, returning the first success.

### Saga pattern

A saga is a sequence of local transactions, each with a **compensating transaction** that undoes its effect if a later step fails. It replaces distributed 2PC for long-running workflows.

```
Step 1: book_flight()    → compensate: cancel_flight()
Step 2: charge_card()    → compensate: refund_card()
Step 3: send_receipt()   → compensate: (none — idempotent)
```

If step 3 fails: no compensation needed.  
If step 2 fails: compensate step 1 (cancel flight).

Two styles:
- **Choreography** — each step publishes an event; the next step listens. Decoupled but hard to trace.
- **Orchestration** — a central saga orchestrator drives each step and compensation. Easier to reason about.

### Chaos engineering intro

Chaos engineering deliberately injects failures to verify that reliability mechanisms work before a real outage does. Key principle: test in a controlled environment with a clear hypothesis.

**Chaos loop:**
1. Define steady-state behavior
2. Hypothesize: "adding X% failures to service Y won't degrade p99 latency"
3. Inject failures (latency, errors, resource exhaustion)
4. Measure: did steady state hold?
5. Fix if not; document if yes

---

## How it works internally

**Fallback chain**: a list of callables tried in order; each is wrapped in a try/except so failure advances to the next. Optionally decorated with circuit breakers so a broken primary doesn't add latency on every call.

**Orchestrated saga**: the orchestrator maintains a `steps_completed: list[int]` log. On failure, it iterates backward over completed steps and calls each compensating function. Compensation itself is retried with backoff since it must succeed.

**Chaos injection**: a `chaos_middleware` wraps any async callable and randomly injects errors, delays, or both. Seed with `random.seed(N)` for deterministic test replay.

---

## Worked example

```python
"""
day5_degradation_saga_chaos.py — Fallback chain, orchestrated saga, chaos middleware.

Install: pip install pydantic
"""

import asyncio
import logging
import random
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

T = TypeVar("T")


# ── Fallback chain ─────────────────────────────────────────────────────────────

type AsyncFn[T] = Callable[[], Coroutine[Any, Any, T]]


async def fallback_chain(
    fns: list[AsyncFn[T]],
    label: str = "fallback_chain",
) -> T:
    """
    Try each function in order. Return first success.
    Raises the last exception if all fail.
    """
    last_exc: Exception | None = None
    for i, fn in enumerate(fns):
        try:
            result = await fn()
            if i > 0:
                logger.warning("%s: primary failed, using fallback #%d", label, i)
            return result
        except Exception as exc:
            logger.warning("%s: option %d failed — %s", label, i, exc)
            last_exc = exc
    raise RuntimeError(f"{label}: all options exhausted") from last_exc


# ── Mock LLM providers ─────────────────────────────────────────────────────────

async def primary_llm(prompt: str) -> str:
    raise ConnectionError("Primary model overloaded")


async def backup_llm(prompt: str) -> str:
    await asyncio.sleep(0.05)
    return f"[backup] {prompt}"


async def static_fallback(prompt: str) -> str:
    return "[degraded] Service temporarily unavailable. Please try again shortly."


# ── Orchestrated saga ──────────────────────────────────────────────────────────

@dataclass
class SagaStep:
    name: str
    execute: Callable[[], Coroutine[Any, Any, Any]]
    compensate: Callable[[], Coroutine[Any, Any, None]] | None = None


class SagaOrchestrator:
    """
    Runs saga steps in order. On failure, compensates completed steps in reverse.
    """

    def __init__(self, steps: list[SagaStep]) -> None:
        self._steps = steps

    async def run(self) -> list[Any]:
        results: list[Any] = []
        completed: list[int] = []

        for i, step in enumerate(self._steps):
            try:
                logger.info("Saga: executing step [%d] %s", i, step.name)
                result = await step.execute()
                results.append(result)
                completed.append(i)
            except Exception as exc:
                logger.error("Saga: step [%d] %s failed — %s", i, step.name, exc)
                await self._compensate(completed)
                raise RuntimeError(f"Saga failed at step '{step.name}'") from exc

        logger.info("Saga: all %d steps completed", len(self._steps))
        return results

    async def _compensate(self, completed: list[int]) -> None:
        for i in reversed(completed):
            step = self._steps[i]
            if step.compensate is None:
                continue
            try:
                logger.info("Saga: compensating step [%d] %s", i, step.name)
                await step.compensate()
            except Exception:
                logger.exception(
                    "Saga: compensation failed for step [%d] %s — manual intervention needed",
                    i,
                    step.name,
                )


# ── Chaos middleware ───────────────────────────────────────────────────────────

@dataclass
class ChaosConfig:
    error_rate: float = 0.0       # 0.0–1.0 probability of injecting an error
    latency_p: float = 0.0        # probability of injecting extra latency
    latency_ms: float = 500.0     # extra latency in milliseconds
    seed: int | None = None       # for reproducible tests


def chaos_wrap(
    fn: AsyncFn[T],
    config: ChaosConfig,
    label: str = "chaos",
) -> AsyncFn[T]:
    """Wrap `fn` with probabilistic error and latency injection."""
    rng = random.Random(config.seed)

    async def wrapped() -> T:
        if rng.random() < config.error_rate:
            logger.warning("CHAOS[%s]: injecting error", label)
            raise ConnectionError(f"CHAOS: injected failure in {label}")
        if rng.random() < config.latency_p:
            delay = config.latency_ms / 1000.0
            logger.warning("CHAOS[%s]: injecting %.0fms latency", label, config.latency_ms)
            await asyncio.sleep(delay)
        return await fn()

    return wrapped


# ── Demo ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    prompt = "Summarize the risk report"

    # Fallback chain
    result = await fallback_chain(
        [
            lambda: primary_llm(prompt),
            lambda: backup_llm(prompt),
            lambda: static_fallback(prompt),
        ],
        label="llm",
    )
    logger.info("Final response: %s", result)

    # Orchestrated saga
    booked = []

    async def book_flight() -> str:
        booked.append("flight-XY123")
        return "flight-XY123"

    async def cancel_flight() -> None:
        booked.remove("flight-XY123")
        logger.info("Compensation: flight cancelled")

    async def charge_card() -> str:
        raise RuntimeError("Card declined")  # Simulate step 2 failure

    saga = SagaOrchestrator([
        SagaStep("book_flight", book_flight, cancel_flight),
        SagaStep("charge_card", charge_card),
    ])
    try:
        await saga.run()
    except RuntimeError as e:
        logger.error("Saga rolled back: %s", e)
    logger.info("Booked items after rollback: %s", booked)

    # Chaos: inject 50% errors on backup LLM for chaos testing
    chaotic_backup = chaos_wrap(
        lambda: backup_llm("chaos test"),
        ChaosConfig(error_rate=0.5, seed=42),
        label="backup_llm",
    )
    for i in range(5):
        try:
            r = await chaotic_backup()
            logger.info("Chaos test %d: %s", i, r)
        except ConnectionError as e:
            logger.warning("Chaos test %d: injected failure: %s", i, e)


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Exercises

1. Add a `max_fallback_latency_ms` option to `fallback_chain` that raises `TimeoutError` if the total time across all attempts exceeds the limit. Use `time.monotonic()`.

2. Implement a `choreography_saga` using `asyncio.Queue` events: `step1` publishes `"step1.done"`, a listener triggers `step2`, and so on. Compare to the orchestration model: in what scenario is choreography harder to debug?

3. Add a `ChaosConfig.jitter_ms` option that adds `random(0, jitter_ms)` latency on every call (not just probabilistic). Write a test using `unittest.mock.patch('asyncio.sleep')` to verify the sleep duration is within the expected range.

4. Extend the `SagaOrchestrator` to support async compensation with retry: each compensation step should retry up to 3 times with 1-second delays before logging a "manual intervention needed" warning. Write a test where the first compensation attempt fails but the second succeeds.

(Solutions: ../solutions/week-03/day-5.md)

---

## Self-check

- [ ] I can build a fallback chain that tries alternatives on failure
- [ ] I can implement an orchestrated saga with compensation on failure
- [ ] I can distinguish choreography sagas from orchestration sagas
- [ ] I can write a chaos middleware that injects errors and latency probabilistically
- [ ] I can run a chaos experiment with a hypothesis and verify the result with tests

---

## Supplementary links

- [Saga Pattern — Microservices.io](https://microservices.io/patterns/data/saga.html)
- [Principles of Chaos Engineering](https://principlesofchaos.org/)
- [Netflix Chaos Monkey](https://netflix.github.io/chaosmonkey/)
