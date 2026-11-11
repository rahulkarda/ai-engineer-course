# Solutions — Week 03 Project: Reliability Toolkit

> Complete reference implementation. Read only after building your own version.

---

## Project structure

```
reliability_toolkit/
├── pyproject.toml
├── .env.example
└── src/
    └── reliability_toolkit/
        ├── __init__.py
        ├── config.py
        ├── retry.py
        ├── idempotency.py
        ├── circuit_breaker.py
        ├── timeout.py
        ├── bulkhead.py
        ├── rate_limiter.py
        ├── fallback.py
        ├── client.py
        └── chaos.py
```

---

## `.env.example`

```bash
# Reliability Toolkit — environment config
# Copy to .env and adjust values. Never commit .env.
RELIABILITY_RETRY_MAX_ATTEMPTS=5
RELIABILITY_RETRY_MAX_DELAY_S=30.0
RELIABILITY_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
RELIABILITY_CIRCUIT_BREAKER_RESET_TIMEOUT_S=30.0
RELIABILITY_RATE_LIMIT_RPM=60.0
RELIABILITY_RATE_LIMIT_BURST=10
RELIABILITY_BULKHEAD_MAX_CONCURRENT=20
RELIABILITY_CALL_TIMEOUT_S=30.0
RELIABILITY_IDEMPOTENCY_TTL_S=86400.0
```

---

## `src/reliability_toolkit/__init__.py`

```python
"""Reliability toolkit for LLM/tool clients."""

from reliability_toolkit.client import LLMRequest, LLMResponse, ResilientLLMClient
from reliability_toolkit.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitOpenError
from reliability_toolkit.chaos import ChaosConfig, ChaosMiddleware
from reliability_toolkit.config import ReliabilityConfig
from reliability_toolkit.idempotency import IdempotencyStore
from reliability_toolkit.rate_limiter import RateLimitExceeded, TokenBucket

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitOpenError",
    "ChaosConfig",
    "ChaosMiddleware",
    "IdempotencyStore",
    "LLMRequest",
    "LLMResponse",
    "RateLimitExceeded",
    "ReliabilityConfig",
    "ResilientLLMClient",
    "TokenBucket",
]
```

---

## `src/reliability_toolkit/retry.py`

```python
"""Retry policy with exponential backoff and full jitter."""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TransientError(Exception):
    """Retryable transient failure."""


class PermanentError(Exception):
    """Non-retryable failure."""


@dataclass
class RetryMetrics:
    total_attempts: int = 0
    total_successes: int = 0
    total_failures: int = 0


class RetryPolicy:
    def __init__(
        self,
        max_attempts: int = 5,
        base_delay_s: float = 0.5,
        max_delay_s: float = 30.0,
        retryable: type[Exception] | tuple[type[Exception], ...] = TransientError,
    ) -> None:
        self._max = max_attempts
        self._base = base_delay_s
        self._cap = max_delay_s
        self._retryable = retryable
        self._metrics = RetryMetrics()

    def _jitter_wait(self, attempt: int) -> float:
        ceiling = min(self._cap, self._base * (2 ** attempt))
        return random.uniform(0, ceiling)

    async def execute(
        self,
        fn: Callable[[], Coroutine[Any, Any, T]],
    ) -> tuple[int, T]:
        """
        Execute fn with retry. Returns (attempt_count, result).
        Raises the last exception after max_attempts.
        """
        last_exc: Exception | None = None
        for attempt in range(1, self._max + 1):
            self._metrics.total_attempts += 1
            try:
                result = await fn()
                self._metrics.total_successes += 1
                return attempt, result
            except PermanentError:
                self._metrics.total_failures += 1
                raise
            except Exception as exc:
                if not isinstance(exc, self._retryable):
                    self._metrics.total_failures += 1
                    raise
                last_exc = exc
                wait = self._jitter_wait(attempt - 1)
                logger.warning(
                    "RetryPolicy: attempt %d/%d failed (%s) — retrying in %.2fs",
                    attempt, self._max, exc, wait,
                )
                if attempt < self._max:
                    await asyncio.sleep(wait)
        self._metrics.total_failures += 1
        raise RuntimeError(f"Exhausted {self._max} attempts") from last_exc

    def metrics(self) -> dict[str, int]:
        return {
            "total_attempts": self._metrics.total_attempts,
            "total_successes": self._metrics.total_successes,
            "total_failures": self._metrics.total_failures,
        }
```

---

## `src/reliability_toolkit/chaos.py`

```python
"""Chaos middleware for test injection."""

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ChaosConfig:
    error_rate: float = 0.0
    latency_p: float = 0.0
    latency_ms: float = 500.0
    jitter_ms: float = 0.0
    seed: int | None = None


class ChaosMiddleware:
    def __init__(self, config: ChaosConfig, label: str = "chaos") -> None:
        self._cfg = config
        self._label = label
        self._rng = random.Random(config.seed)

    def wrap(
        self,
        fn: Callable[[], Coroutine[Any, Any, T]],
    ) -> Callable[[], Coroutine[Any, Any, T]]:
        rng = self._rng
        cfg = self._cfg
        label = self._label

        async def wrapped() -> T:
            if cfg.jitter_ms > 0:
                await asyncio.sleep(rng.uniform(0, cfg.jitter_ms / 1000.0))
            if rng.random() < cfg.error_rate:
                logger.warning("CHAOS[%s]: injecting error", label)
                raise ConnectionError(f"CHAOS injected failure in {label}")
            if rng.random() < cfg.latency_p:
                logger.warning(
                    "CHAOS[%s]: injecting %.0fms latency", label, cfg.latency_ms
                )
                await asyncio.sleep(cfg.latency_ms / 1000.0)
            return await fn()

        return wrapped
```

---

## `scenarios/chaos_scenario.py`

```python
"""
Chaos scenario: 70% primary failure rate, verify 80%+ requests served.

Run: python scenarios/chaos_scenario.py
"""

import asyncio
import logging
import time
from unittest.mock import AsyncMock

from reliability_toolkit import ChaosConfig, ChaosMiddleware, LLMRequest, ResilientLLMClient

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def make_real_transport() -> AsyncMock:
    transport = AsyncMock()
    transport.complete.return_value = "primary response"
    return transport


async def make_backup_transport() -> AsyncMock:
    transport = AsyncMock()
    transport.complete.return_value = "[backup] response"
    return transport


async def main() -> None:
    primary_raw = await make_real_transport()
    backup = await make_backup_transport()

    chaos = ChaosMiddleware(ChaosConfig(error_rate=0.7, latency_ms=200, seed=123), label="primary")

    # Wrap primary transport with chaos
    original_complete = primary_raw.complete

    async def chaotic_complete(prompt: str, model: str) -> str:
        return await chaos.wrap(lambda: original_complete(prompt, model))()

    primary_raw.complete = chaotic_complete

    client = ResilientLLMClient(primary=primary_raw, fallbacks=[backup])

    results = {"success": 0, "degraded": 0, "failed": 0}
    latencies: list[float] = []

    async def one_call(i: int) -> None:
        start = time.monotonic()
        try:
            resp = await client.complete(LLMRequest(prompt=f"prompt-{i}"))
            latencies.append((time.monotonic() - start) * 1000)
            if resp.degraded:
                results["degraded"] += 1
            else:
                results["success"] += 1
        except Exception:
            results["failed"] += 1

    await asyncio.gather(*[one_call(i) for i in range(20)])

    served = results["success"] + results["degraded"]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    print(
        f"total=20 success={results['success']} degraded={results['degraded']} "
        f"failed={results['failed']} avg_latency_ms={avg_latency:.0f}"
    )
    assert served >= 16, f"Expected ≥16 served, got {served}"
    print("Chaos hypothesis PASSED: 80%+ requests served despite 70% failure rate")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## `scenarios/saga_scenario.py`

```python
"""
Saga scenario: 3-step booking saga with rollback on step 2 failure.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class BookingState:
    def __init__(self) -> None:
        self.slot_reserved: bool = False
        self.credits_deducted: bool = False


state = BookingState()


async def reserve_slot() -> str:
    state.slot_reserved = True
    logger.info("Slot reserved")
    return "SLOT-42"


async def release_slot() -> None:
    state.slot_reserved = False
    logger.info("Compensation: slot released")


async def deduct_credits() -> str:
    raise RuntimeError("Insufficient credits")


async def refund_credits() -> None:
    state.credits_deducted = False
    logger.info("Compensation: credits refunded")


async def send_confirmation() -> str:
    return "CONF-001"


async def main() -> None:
    from reliability_toolkit.fallback import SagaOrchestrator, SagaStep

    saga = SagaOrchestrator([
        SagaStep("reserve_slot", reserve_slot, release_slot),
        SagaStep("deduct_credits", deduct_credits, refund_credits),
        SagaStep("send_confirmation", send_confirmation),
    ])

    try:
        await saga.run()
    except RuntimeError as exc:
        logger.error("Saga failed: %s", exc)

    logger.info("Final state: slot_reserved=%s credits_deducted=%s",
                state.slot_reserved, state.credits_deducted)
    assert not state.slot_reserved, "Slot should have been released by compensation"


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Grading guide

| Dimension | Full marks indicators |
|---|---|
| Correctness (40%) | All F1–F10 pass; chaos scenario assertion passes; saga compensation verified |
| Architecture (25%) | Call-stack order comment in client.py matches reference; idempotency before bulkhead |
| Test quality (20%) | Deterministic chaos tests with seed; concurrent idempotency tested; CB state transitions tested |
| Code quality (15%) | Pydantic v2 everywhere; `logging` not `print`; no bare `except`; `model_config = {"frozen": True}` on config |
