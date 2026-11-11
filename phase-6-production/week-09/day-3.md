# Phase 6 · Week 09 · Day 3 — Cross-Provider Fallback

> Objective: Build a fallback chain that retries across providers/models on failure, distinguishing retryable from non-retryable errors.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

## Concept

Any provider can fail: rate limits hit, a model goes offline, a transient 5xx arrives. A **fallback chain** encodes an ordered preference list of `(provider, model)` pairs. When the first option fails with a retryable error the chain tries the next; if the error is non-retryable (authentication failure, invalid API key) the chain stops immediately rather than wasting further attempts.

Design decisions:

- **Error hierarchy** — a typed exception tree (not bare strings) lets `except` clauses be precise. Each error carries a `retryable` flag.
- **Attempt audit trail** — every attempt (success or failure) is recorded as a `FallbackAttempt`. Callers can log exactly which providers were tried and why.
- **No magic retry loops** — the chain iterates the list exactly once. Back-off / sleep belongs in a separate `RetryPolicy` layer (see Exercise 3), keeping `FallbackChain` readable.
- **Separation of concerns** — `FallbackChain` is not a router; routing (which models to try) is still handled by `ModelRouter`. The chain receives a pre-built `chain` list.

## How it works internally

```
FallbackChain.complete(request)
  │
  ├─ Attempt 1: provider_a / model_a
  │     └─ RateLimitError (retryable=True)  → log, continue
  │
  ├─ Attempt 2: provider_b / model_b
  │     └─ AuthError (retryable=False)      → re-raise immediately
  │
  └─ Attempt 3: provider_c / model_c
        └─ Success                          → return (response, attempts)
```

The `retryable` flag on each `ProviderError` drives the branch:

```python
if not e.retryable:
    raise   # bubble up immediately — no point trying next provider
logger.warning(...)
# fall through to next iteration
```

If the loop exhausts all options without a success, it raises `ProviderError("all", "all providers exhausted", retryable=False)`.

`FallbackAttempt` is a `dataclass` (not a Pydantic model) because it is an internal audit record, not a data-boundary type. Pydantic models are reserved for data crossing system boundaries (API requests/responses).

The chain calls `request.model_copy(update={"model": model})` to get a fresh request with the correct model id for each attempt. `model_copy` is the Pydantic v2 method for immutable updates.

## Worked example

```python
# gateway/fallback.py
import logging
from dataclasses import dataclass, field

from gateway.providers import BaseProvider, CompletionRequest, CompletionResponse, ProviderRegistry

logger = logging.getLogger(__name__)


# ── Error hierarchy ───────────────────────────────────────────────────────────

class ProviderError(Exception):
    """Base class for all provider-level errors.

    Args:
        provider: Name of the provider that raised this error.
        message:  Human-readable description.
        retryable: If True the fallback chain will try the next option.
                   If False the chain stops and re-raises immediately.
    """
    def __init__(self, provider: str, message: str, retryable: bool = True) -> None:
        self.provider = provider
        self.message = message
        self.retryable = retryable
        super().__init__(f"{provider}: {message}")


class RateLimitError(ProviderError):
    """Provider returned HTTP 429 or equivalent."""
    def __init__(self, provider: str) -> None:
        super().__init__(provider, "rate limit exceeded", retryable=True)


class AuthError(ProviderError):
    """Provider rejected the API key (HTTP 401/403)."""
    def __init__(self, provider: str) -> None:
        super().__init__(provider, "authentication failed", retryable=False)


class ModelNotFoundError(ProviderError):
    """Requested model id does not exist on this provider."""
    def __init__(self, provider: str, model: str) -> None:
        super().__init__(provider, f"model '{model}' not found", retryable=False)


class ServiceUnavailableError(ProviderError):
    """Provider is temporarily unavailable (HTTP 503)."""
    def __init__(self, provider: str) -> None:
        super().__init__(provider, "service unavailable", retryable=True)


# ── Attempt record ────────────────────────────────────────────────────────────

@dataclass
class FallbackAttempt:
    """Audit record for one attempt in a fallback chain."""
    provider: str
    model: str
    error: str | None = None
    success: bool = False


# ── Fallback chain ────────────────────────────────────────────────────────────

class FallbackChain:
    """Tries each (provider, model) pair in order until one succeeds.

    Args:
        chain:    Ordered list of (provider_name, model_id) tuples.
        registry: ProviderRegistry to resolve provider names to instances.
    """

    def __init__(
        self,
        chain: list[tuple[str, str]],
        registry: ProviderRegistry,
    ) -> None:
        if not chain:
            raise ValueError("FallbackChain requires at least one entry.")
        self.chain = chain
        self.registry = registry

    async def complete(
        self,
        request: CompletionRequest,
    ) -> tuple[CompletionResponse, list[FallbackAttempt]]:
        """Attempt each chain entry in order.

        Returns:
            (CompletionResponse, list of all FallbackAttempt records)

        Raises:
            ProviderError: if a non-retryable error occurs, or all options exhausted.
        """
        attempts: list[FallbackAttempt] = []

        for provider_name, model in self.chain:
            request_copy = request.model_copy(update={"model": model})
            attempt = FallbackAttempt(provider=provider_name, model=model)

            try:
                provider = self.registry.get(provider_name)
                logger.info("fallback.attempt provider=%s model=%s", provider_name, model)
                response = await provider.complete(request_copy)
                attempt.success = True
                attempts.append(attempt)
                return response, attempts

            except ProviderError as exc:
                attempt.error = str(exc)
                attempts.append(attempt)
                if not exc.retryable:
                    logger.error(
                        "fallback.non_retryable provider=%s error=%s",
                        provider_name, exc.message,
                    )
                    raise
                logger.warning(
                    "fallback.retry provider=%s error=%s trying_next=True",
                    provider_name, exc.message,
                )

        raise ProviderError("all", "all providers exhausted", retryable=False)
```

```python
# tests/unit/test_fallback.py
import pytest
from unittest.mock import AsyncMock, MagicMock

from gateway.fallback import (
    AuthError,
    FallbackAttempt,
    FallbackChain,
    ProviderError,
    RateLimitError,
)
from gateway.providers import CompletionRequest, CompletionResponse, ProviderRegistry


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_request() -> CompletionRequest:
    return CompletionRequest(
        model="claude-haiku-5-5",
        messages=[{"role": "user", "content": "Ping"}],
    )


def make_mock_response(content: str = "ok") -> CompletionResponse:
    return CompletionResponse(
        content=content,
        model="claude-haiku-5-5",
        provider="mock",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1.0,
    )


def make_registry(*providers) -> ProviderRegistry:
    registry = ProviderRegistry()
    for p in providers:
        registry.register(p)
    return registry


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_first_provider_succeeds_no_fallback():
    """If the primary provider succeeds, no secondary is called."""
    primary = MagicMock()
    primary.provider_name.return_value = "primary"
    primary.complete = AsyncMock(return_value=make_mock_response("primary ok"))

    registry = make_registry(primary)
    chain = FallbackChain([("primary", "claude-haiku-5-5")], registry)

    response, attempts = await chain.complete(make_request())

    assert response.content == "primary ok"
    assert len(attempts) == 1
    assert attempts[0].success is True
    primary.complete.assert_called_once()


@pytest.mark.asyncio
async def test_falls_back_after_rate_limit():
    """RateLimitError (retryable) causes chain to try next provider."""
    primary = MagicMock()
    primary.provider_name.return_value = "primary"
    primary.complete = AsyncMock(side_effect=RateLimitError("primary"))

    secondary = MagicMock()
    secondary.provider_name.return_value = "secondary"
    secondary.complete = AsyncMock(return_value=make_mock_response("secondary ok"))

    registry = make_registry(primary, secondary)
    chain = FallbackChain(
        [("primary", "claude-haiku-5-5"), ("secondary", "claude-haiku-5-5")],
        registry,
    )

    response, attempts = await chain.complete(make_request())

    assert response.content == "secondary ok"
    assert len(attempts) == 2
    assert attempts[0].success is False
    assert attempts[1].success is True


@pytest.mark.asyncio
async def test_auth_error_stops_chain_immediately():
    """AuthError (retryable=False) must not try subsequent providers."""
    primary = MagicMock()
    primary.provider_name.return_value = "primary"
    primary.complete = AsyncMock(side_effect=AuthError("primary"))

    secondary = MagicMock()
    secondary.provider_name.return_value = "secondary"
    secondary.complete = AsyncMock(return_value=make_mock_response("should not reach"))

    registry = make_registry(primary, secondary)
    chain = FallbackChain(
        [("primary", "claude-haiku-5-5"), ("secondary", "claude-haiku-5-5")],
        registry,
    )

    with pytest.raises(AuthError):
        await chain.complete(make_request())

    secondary.complete.assert_not_called()


@pytest.mark.asyncio
async def test_all_providers_exhausted_raises():
    """ProviderError('all', ...) raised when every option fails."""
    p = MagicMock()
    p.provider_name.return_value = "flaky"
    p.complete = AsyncMock(side_effect=RateLimitError("flaky"))

    registry = make_registry(p)
    chain = FallbackChain([("flaky", "claude-haiku-5-5")], registry)

    with pytest.raises(ProviderError, match="all providers exhausted"):
        await chain.complete(make_request())


def test_empty_chain_raises_on_construction():
    registry = ProviderRegistry()
    with pytest.raises(ValueError, match="at least one entry"):
        FallbackChain([], registry)


@pytest.mark.asyncio
async def test_attempt_records_error_messages():
    """Error message stored in FallbackAttempt for observability."""
    p = MagicMock()
    p.provider_name.return_value = "p1"
    p.complete = AsyncMock(side_effect=RateLimitError("p1"))

    registry = make_registry(p)
    chain = FallbackChain([("p1", "claude-haiku-5-5")], registry)

    with pytest.raises(ProviderError):
        await chain.complete(make_request())
```

Usage (wiring the chain at startup):

```python
from gateway.fallback import FallbackChain
from gateway.providers import AnthropicProvider, ProviderRegistry, CompletionRequest

registry = ProviderRegistry()
registry.register(AnthropicProvider())
# Register a secondary provider here when available.

# Prefer opus; fall back to sonnet, then haiku.
chain = FallbackChain(
    chain=[
        ("anthropic", "claude-opus-5-5"),
        ("anthropic", "claude-sonnet-5-5"),
        ("anthropic", "claude-haiku-5-5"),
    ],
    registry=registry,
)

request = CompletionRequest(
    model="",   # FallbackChain sets model per attempt
    messages=[{"role": "user", "content": "Explain entropy."}],
)

response, attempts = await chain.complete(request)
```

## Exercises

1. Add a `RetryPolicy` dataclass with fields `max_attempts: int`, `backoff_factor: float`, and `retryable_types: list[type[ProviderError]]`. Write a `with_retry()` async helper that wraps a single provider call, sleeping `backoff_factor * attempt_number` seconds between retries, up to `max_attempts`. Write a test that mocks `asyncio.sleep` and asserts it is called the correct number of times.

2. Add a `ServiceUnavailableError` (already defined in the module) to your fallback tests. Write a test that chains three providers: the first raises `ServiceUnavailableError`, the second raises `RateLimitError`, and the third succeeds. Assert all three attempts are recorded and only the third has `success=True`.

3. Implement a `FallbackChainBuilder` that provides a fluent API:
   ```python
   chain = (
       FallbackChainBuilder(registry)
       .add("anthropic", "claude-opus-5-5")
       .add("anthropic", "claude-sonnet-5-5")
       .build()
   )
   ```
   Write a test verifying that `build()` produces a `FallbackChain` with the entries in the correct order.

4. Add an `on_fallback` callback parameter to `FallbackChain.__init__` with type `Callable[[FallbackAttempt], None] | None`. When a retryable error triggers a fallback, call the hook (if set) with the failed attempt. Write a test asserting the hook is called exactly once when one provider fails and a second succeeds.

(Solutions: ../solutions/week-09/day-3.md)

## Self-check

- [ ] I can explain why `AuthError` should stop the chain while `RateLimitError` should not
- [ ] I can implement `FallbackChain.complete()` that records every attempt and returns the audit trail
- [ ] I can write tests that inject specific error types without any real provider calls
