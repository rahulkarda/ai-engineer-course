# Solutions — Day 3: Cross-Provider Fallback

---

## Base types (shared reference)

```python
# gateway/fallback/base.py
from dataclasses import dataclass, field
from typing import Any


class ProviderError(Exception):
    """Base class for provider errors."""


class RateLimitError(ProviderError):
    """Provider is rate-limited."""


class ServiceUnavailableError(ProviderError):
    """Provider returned 5xx / is down."""


@dataclass
class AttemptResult:
    provider: str
    success: bool
    error: str | None = None
```

---

## Exercise 1: Async retry with exponential backoff in `FallbackChain`

```python
# gateway/fallback/chain.py
import asyncio
import logging
from collections.abc import Callable, Awaitable

from gateway.fallback.base import AttemptResult, ProviderError
from gateway.providers.base import BaseProvider, CompletionRequest, CompletionResponse

logger = logging.getLogger(__name__)

# Errors that are worth retrying
_RETRYABLE = (ProviderError,)


class FallbackChain:
    """
    Try providers in order. If a provider raises a retryable error, wait with
    exponential backoff then move on to the next provider.
    """

    def __init__(
        self,
        providers: list[BaseProvider],
        max_retries_per_provider: int = 2,
        base_backoff_s: float = 0.5,
    ) -> None:
        if not providers:
            raise ValueError("FallbackChain requires at least one provider")
        self._providers = providers
        self._max_retries = max_retries_per_provider
        self._base_backoff_s = base_backoff_s
        self.attempts: list[AttemptResult] = []

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.attempts.clear()
        last_exc: Exception | None = None

        for provider in self._providers:
            for attempt in range(self._max_retries + 1):
                if attempt > 0:
                    wait = self._base_backoff_s * (2 ** (attempt - 1))
                    logger.info(
                        "fallback backoff provider=%s attempt=%d wait=%.2fs",
                        provider.provider_name(),
                        attempt,
                        wait,
                    )
                    await asyncio.sleep(wait)

                try:
                    response = await provider.complete(request)
                    self.attempts.append(
                        AttemptResult(provider=provider.provider_name(), success=True)
                    )
                    return response
                except _RETRYABLE as exc:
                    last_exc = exc
                    logger.warning(
                        "fallback provider=%s attempt=%d error=%s",
                        provider.provider_name(),
                        attempt,
                        exc,
                    )
                    self.attempts.append(
                        AttemptResult(
                            provider=provider.provider_name(),
                            success=False,
                            error=str(exc),
                        )
                    )
                    # If more retries remain for this provider, loop; otherwise break
                    # to the next provider after the last attempt
                    continue
                break  # non-retryable exception falls through — break inner loop

        raise RuntimeError(
            f"All providers exhausted after {len(self.attempts)} attempts"
        ) from last_exc
```

---

## Exercise 2: Context manager that logs total fallback attempts on exit

```python
# gateway/fallback/chain.py  (extend FallbackChain)
from types import TracebackType


class FallbackChain:
    """
    Usage as context manager:

        async with FallbackChain(providers) as chain:
            response = await chain.complete(request)
        # → logs total attempts on __aexit__
    """

    def __init__(
        self,
        providers: list[BaseProvider],
        max_retries_per_provider: int = 2,
        base_backoff_s: float = 0.5,
    ) -> None:
        if not providers:
            raise ValueError("FallbackChain requires at least one provider")
        self._providers = providers
        self._max_retries = max_retries_per_provider
        self._base_backoff_s = base_backoff_s
        self.attempts: list[AttemptResult] = []

    async def __aenter__(self) -> "FallbackChain":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        total = len(self.attempts)
        failures = sum(1 for a in self.attempts if not a.success)
        logger.info(
            "FallbackChain exiting: total_attempts=%d failures=%d",
            total,
            failures,
        )
        return False  # do not suppress exceptions

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.attempts.clear()
        last_exc: Exception | None = None

        for provider in self._providers:
            for attempt in range(self._max_retries + 1):
                if attempt > 0:
                    wait = self._base_backoff_s * (2 ** (attempt - 1))
                    logger.info(
                        "fallback backoff provider=%s attempt=%d wait=%.2fs",
                        provider.provider_name(),
                        attempt,
                        wait,
                    )
                    await asyncio.sleep(wait)

                try:
                    response = await provider.complete(request)
                    self.attempts.append(
                        AttemptResult(provider=provider.provider_name(), success=True)
                    )
                    return response
                except _RETRYABLE as exc:
                    last_exc = exc
                    logger.warning(
                        "fallback provider=%s attempt=%d error=%s",
                        provider.provider_name(),
                        attempt,
                        exc,
                    )
                    self.attempts.append(
                        AttemptResult(
                            provider=provider.provider_name(),
                            success=False,
                            error=str(exc),
                        )
                    )

        raise RuntimeError(
            f"All providers exhausted after {len(self.attempts)} attempts"
        ) from last_exc
```

---

## Exercise 3: pytest — first provider raises `RateLimitError`, second succeeds

```python
# tests/unit/test_fallback_chain.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.fallback.base import AttemptResult, RateLimitError
from gateway.fallback.chain import FallbackChain
from gateway.providers.base import CompletionRequest, CompletionResponse


def _make_request() -> CompletionRequest:
    return CompletionRequest(
        model="claude-haiku-5-5",
        messages=[{"role": "user", "content": "hello"}],
    )


def _mock_provider(name: str, side_effect: Exception | None = None) -> MagicMock:
    provider = MagicMock()
    provider.provider_name.return_value = name
    if side_effect is not None:
        provider.complete = AsyncMock(side_effect=side_effect)
    else:
        provider.complete = AsyncMock(
            return_value=CompletionResponse(
                content="ok",
                model="claude-haiku-5-5",
                provider=name,
                input_tokens=5,
                output_tokens=5,
                latency_ms=10.0,
            )
        )
    return provider


@pytest.mark.asyncio
async def test_fallback_first_rate_limited_second_succeeds() -> None:
    """First provider raises RateLimitError; second provider succeeds."""
    provider_a = _mock_provider("provider_a", side_effect=RateLimitError("rate limited"))
    provider_b = _mock_provider("provider_b")

    # max_retries=0 so we fail fast and fall through to provider_b immediately
    chain = FallbackChain(
        providers=[provider_a, provider_b],
        max_retries_per_provider=0,
        base_backoff_s=0.0,
    )
    request = _make_request()
    response = await chain.complete(request)

    assert response.content == "ok"
    assert response.provider == "provider_b"

    # attempts list: 1 failure from provider_a, 1 success from provider_b
    assert len(chain.attempts) == 2
    failure = chain.attempts[0]
    success = chain.attempts[1]
    assert failure.provider == "provider_a"
    assert failure.success is False
    assert success.provider == "provider_b"
    assert success.success is True


@pytest.mark.asyncio
async def test_fallback_both_fail_raises() -> None:
    provider_a = _mock_provider("provider_a", side_effect=RateLimitError("429"))
    provider_b = _mock_provider("provider_b", side_effect=RateLimitError("429"))

    chain = FallbackChain(
        providers=[provider_a, provider_b],
        max_retries_per_provider=0,
        base_backoff_s=0.0,
    )
    with pytest.raises(RuntimeError, match="All providers exhausted"):
        await chain.complete(_make_request())


@pytest.mark.asyncio
async def test_context_manager_logs_on_exit(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    provider_a = _mock_provider("provider_a", side_effect=RateLimitError("429"))
    provider_b = _mock_provider("provider_b")

    with caplog.at_level(logging.INFO, logger="gateway.fallback.chain"):
        async with FallbackChain(
            providers=[provider_a, provider_b],
            max_retries_per_provider=0,
            base_backoff_s=0.0,
        ) as chain:
            await chain.complete(_make_request())

    assert any("total_attempts=2" in record.message for record in caplog.records)
```

---

## Exercise 4: `CircuitBreaker` per provider — opens after 3 failures, stays open 60 s

```python
# gateway/fallback/circuit_breaker.py
import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum

from gateway.fallback.base import ProviderError
from gateway.providers.base import BaseProvider, CompletionRequest, CompletionResponse

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"      # normal — requests pass through
    OPEN = "open"          # tripped — requests fail fast
    HALF_OPEN = "half_open"  # probe — one request allowed to test recovery


@dataclass
class _ProviderCircuit:
    state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    opened_at: float | None = None


class CircuitBreaker:
    """
    Wraps a BaseProvider. Opens after `failure_threshold` consecutive failures.
    Stays open for `recovery_timeout_s` seconds, then moves to HALF_OPEN to
    probe whether the provider has recovered.
    """

    def __init__(
        self,
        provider: BaseProvider,
        failure_threshold: int = 3,
        recovery_timeout_s: float = 60.0,
    ) -> None:
        self._provider = provider
        self._threshold = failure_threshold
        self._recovery_timeout = recovery_timeout_s
        self._circuit = _ProviderCircuit()

    @property
    def state(self) -> CircuitState:
        return self._circuit.state

    def _try_transition_to_half_open(self) -> None:
        """Move from OPEN → HALF_OPEN if recovery timeout has elapsed."""
        if (
            self._circuit.state == CircuitState.OPEN
            and self._circuit.opened_at is not None
            and (time.monotonic() - self._circuit.opened_at) >= self._recovery_timeout
        ):
            logger.info(
                "circuit_breaker provider=%s OPEN→HALF_OPEN",
                self._provider.provider_name(),
            )
            self._circuit.state = CircuitState.HALF_OPEN

    def _record_success(self) -> None:
        self._circuit.consecutive_failures = 0
        if self._circuit.state != CircuitState.CLOSED:
            logger.info(
                "circuit_breaker provider=%s %s→CLOSED",
                self._provider.provider_name(),
                self._circuit.state,
            )
        self._circuit.state = CircuitState.CLOSED
        self._circuit.opened_at = None

    def _record_failure(self) -> None:
        self._circuit.consecutive_failures += 1
        if (
            self._circuit.consecutive_failures >= self._threshold
            and self._circuit.state != CircuitState.OPEN
        ):
            logger.warning(
                "circuit_breaker provider=%s CLOSED→OPEN after %d failures",
                self._provider.provider_name(),
                self._circuit.consecutive_failures,
            )
            self._circuit.state = CircuitState.OPEN
            self._circuit.opened_at = time.monotonic()

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self._try_transition_to_half_open()

        if self._circuit.state == CircuitState.OPEN:
            raise ProviderError(
                f"Circuit OPEN for provider '{self._provider.provider_name()}'"
            )

        try:
            response = await self._provider.complete(request)
            self._record_success()
            return response
        except ProviderError as exc:
            self._record_failure()
            raise
```

Tests:

```python
# tests/unit/test_circuit_breaker.py
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from gateway.fallback.base import ProviderError, RateLimitError
from gateway.fallback.circuit_breaker import CircuitBreaker, CircuitState
from gateway.providers.base import CompletionRequest, CompletionResponse


def _make_request() -> CompletionRequest:
    return CompletionRequest(
        model="claude-haiku-5-5",
        messages=[{"role": "user", "content": "hi"}],
    )


def _mock_provider(name: str = "p1") -> MagicMock:
    p = MagicMock()
    p.provider_name.return_value = name
    return p


@pytest.mark.asyncio
async def test_circuit_opens_after_threshold_failures() -> None:
    provider = _mock_provider()
    provider.complete = AsyncMock(side_effect=RateLimitError("429"))

    cb = CircuitBreaker(provider, failure_threshold=3, recovery_timeout_s=60.0)
    request = _make_request()

    for _ in range(3):
        with pytest.raises(RateLimitError):
            await cb.complete(request)

    assert cb.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_open_circuit_fails_fast_without_calling_provider() -> None:
    provider = _mock_provider()
    provider.complete = AsyncMock(side_effect=RateLimitError("429"))

    cb = CircuitBreaker(provider, failure_threshold=3, recovery_timeout_s=60.0)
    request = _make_request()

    # Trip the circuit
    for _ in range(3):
        with pytest.raises(RateLimitError):
            await cb.complete(request)

    provider.complete.reset_mock()

    # Next call should fail fast (ProviderError) WITHOUT calling the provider
    with pytest.raises(ProviderError):
        await cb.complete(request)

    provider.complete.assert_not_called()


@pytest.mark.asyncio
async def test_circuit_recovers_after_timeout() -> None:
    provider = _mock_provider()
    good_response = CompletionResponse(
        content="ok",
        model="claude-haiku-5-5",
        provider="p1",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1.0,
    )
    provider.complete = AsyncMock(side_effect=RateLimitError("429"))

    cb = CircuitBreaker(provider, failure_threshold=3, recovery_timeout_s=0.01)
    request = _make_request()

    # Trip the circuit
    for _ in range(3):
        with pytest.raises(RateLimitError):
            await cb.complete(request)

    assert cb.state == CircuitState.OPEN

    # Wait for recovery timeout to elapse
    await asyncio.sleep(0.05)

    # Now allow success
    provider.complete = AsyncMock(return_value=good_response)
    response = await cb.complete(request)

    assert response.content == "ok"
    assert cb.state == CircuitState.CLOSED
```
