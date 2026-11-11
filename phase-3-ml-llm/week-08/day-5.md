# Phase 3 · Week 08 · Day 5 — Provider-Agnostic LLM Client Wrapper

> Objective: Design a mockable LLM client abstraction that decouples application logic from the Anthropic SDK.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

---

## Concept

Direct dependency on `anthropic.Anthropic` in application code creates two problems:

1. **Testability**: every test either hits the real API (slow, costs money, flaky) or requires patching internal SDK internals with `unittest.mock.patch`.
2. **Portability**: switching providers (Anthropic → OpenAI → local model) requires changes throughout the codebase, not just at the boundary.

The fix is a **thin abstraction layer**: define the interface you need, implement it for each provider, inject it into components that need it. Tests use a `MockLLMClient` that returns pre-configured responses — no patching, no network.

This is the Dependency Inversion Principle applied to LLM calls. High-level modules (your extraction logic, your agents) depend on the abstraction. Low-level modules (Anthropic SDK, OpenAI SDK) implement it.

---

## How it works internally

### Protocol vs ABC

Python offers two ways to define interfaces:

**`typing.Protocol`** — structural subtyping. A class satisfies the protocol if it has the right methods, regardless of inheritance. No need to explicitly inherit. Checked by type checkers like mypy/pyright, not at runtime by default.

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class LLMClient(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse: ...
```

**`abc.ABC`** — nominal subtyping. Subclasses must inherit and implement all `@abstractmethod` methods. Raises `TypeError` at instantiation if abstract methods are missing. Enforced at runtime.

Use `Protocol` for external code you cannot control (third-party adapters). Use `ABC` when you own all implementations and want runtime enforcement. For our injectable LLM client, `Protocol` is cleaner: mock and production implementations don't need to share a base class.

### Data models at the boundary

Define explicit Pydantic models for request and response instead of passing raw dicts. Benefits:

- Validation happens at the boundary, not deep inside your pipeline
- Type information is preserved through the call chain
- Mocking is trivial: just construct a `LLMResponse` directly
- If the provider changes response shape, the adapter handles it — consumers see the same model

### Dependency injection

Pass the client as a constructor argument:

```python
class ExtractionService:
    def __init__(self, client: LLMClient) -> None:
        self._client = client
```

In production: `ExtractionService(AnthropicClient(api_key=...))`
In tests: `ExtractionService(MockLLMClient([mock_response_1, mock_response_2]))`

No patching. No `unittest.mock.patch`. The test controls exactly what the "model" returns.

### The MockLLMClient pattern

The mock takes a list of responses and returns them in order. If exhausted, it raises `StopIteration` (useful for catching tests that make unexpected extra calls). Optionally configurable to raise an exception to simulate API errors.

```python
mock = MockLLMClient(responses=[
    LLMResponse(content='{"name": "Alice"}', ...),
    LLMResponse(content='{"name": "Bob"}', ...),
])
```

Calls `complete()` return responses in sequence. This is simpler and more readable than `MagicMock(side_effect=[...])`.

### Middleware pattern

Wrap a client with a middleware client that adds cross-cutting behavior:

```python
class LoggingClient:
    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner

    def complete(self, request: LLMRequest) -> LLMResponse:
        logger.info("LLM request: model=%s tokens=%d", request.model, request.max_tokens)
        response = self._inner.complete(request)
        logger.info("LLM response: tokens=%d", response.output_tokens)
        return response
```

Chain middlewares: `LoggingClient(CostTrackingClient(AnthropicClient(...)))`. Each wraps the next. Open-closed: add behavior without modifying existing clients.

---

## Worked example

```python
"""
week-08/day-5: Provider-agnostic LLM client with Protocol interface and MockLLMClient.

Requirements:
    pip install anthropic pydantic
"""

from __future__ import annotations

import logging
import os
from typing import Iterator, Literal, Protocol, runtime_checkable

import anthropic
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")


# ---------------------------------------------------------------------------
# 1. Data models
# ---------------------------------------------------------------------------


class LLMMessage(BaseModel):
    """A single message in a conversation."""

    role: Literal["user", "assistant"]
    content: str


class LLMRequest(BaseModel):
    """
    Provider-agnostic request model.

    All provider-specific parameters (temperature, top_p, etc.) are omitted
    intentionally. Add them as Optional fields if your abstraction needs them.
    """

    messages: list[LLMMessage]
    system: str | None = None
    max_tokens: int = 1024
    model: str = "claude-opus-5-5"
    stop_sequences: list[str] = Field(default_factory=list)


class LLMResponse(BaseModel):
    """Provider-agnostic response model."""

    content: str
    input_tokens: int
    output_tokens: int
    model: str
    stop_reason: str = "end_turn"


# ---------------------------------------------------------------------------
# 2. Client Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMClient(Protocol):
    """
    Structural interface for all LLM client implementations.

    Any class with a `complete` method matching this signature satisfies
    the protocol — no inheritance required.
    """

    def complete(self, request: LLMRequest) -> LLMResponse:
        """
        Send a completion request and return the response.

        Implementations must:
        - Raise anthropic.APIError (or equivalent) on transient failures
        - Always return a fully-populated LLMResponse on success
        - Never mutate the request object
        """
        ...


# ---------------------------------------------------------------------------
# 3. Anthropic implementation
# ---------------------------------------------------------------------------


class AnthropicClient:
    """
    LLMClient backed by the Anthropic SDK.

    Reads ANTHROPIC_API_KEY from the environment. Does not accept the key
    as a constructor argument to prevent accidental logging.
    """

    def __init__(self, model_override: str | None = None) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model_override = model_override
        logger.debug("AnthropicClient initialized")

    def complete(self, request: LLMRequest) -> LLMResponse:
        model = self._model_override or request.model

        create_kwargs: dict = {
            "model": model,
            "max_tokens": request.max_tokens,
            "messages": [m.model_dump() for m in request.messages],
        }
        if request.system:
            create_kwargs["system"] = request.system
        if request.stop_sequences:
            create_kwargs["stop_sequences"] = request.stop_sequences

        logger.debug("AnthropicClient.complete: model=%s max_tokens=%d", model, request.max_tokens)

        response = self._client.messages.create(**create_kwargs)

        content_blocks = [b.text for b in response.content if hasattr(b, "text")]
        content = "\n".join(content_blocks)

        return LLMResponse(
            content=content,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
            stop_reason=response.stop_reason or "end_turn",
        )


# ---------------------------------------------------------------------------
# 4. Mock implementation for testing
# ---------------------------------------------------------------------------


class MockLLMClientError(Exception):
    """Raised when the MockLLMClient is configured to simulate an error."""


class MockLLMClient:
    """
    In-memory LLMClient for testing. Returns responses from a pre-configured list.

    Usage:
        responses = [
            LLMResponse(content='{"ok": true}', input_tokens=10, output_tokens=5,
                        model="mock"),
        ]
        client = MockLLMClient(responses=responses)
        svc = MyService(llm_client=client)
        result = svc.run(...)

    If error_on_call is set, that call index will raise MockLLMClientError instead
    of returning a response. Useful for testing retry and repair logic.
    """

    def __init__(
        self,
        responses: list[LLMResponse],
        error_on_calls: list[int] | None = None,
    ) -> None:
        self._responses = list(responses)
        self._error_on_calls: set[int] = set(error_on_calls or [])
        self._call_count = 0
        self._recorded_requests: list[LLMRequest] = []

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def recorded_requests(self) -> list[LLMRequest]:
        """All requests made to this client, in order. Useful for assertions."""
        return list(self._recorded_requests)

    def complete(self, request: LLMRequest) -> LLMResponse:
        idx = self._call_count
        self._call_count += 1
        self._recorded_requests.append(request)

        if idx in self._error_on_calls:
            raise MockLLMClientError(f"Simulated error on call {idx}")

        if idx >= len(self._responses):
            raise StopIteration(
                f"MockLLMClient exhausted: {idx} calls made but only "
                f"{len(self._responses)} responses configured"
            )

        response = self._responses[idx]
        logger.debug(
            "MockLLMClient returning response %d: content=%s...",
            idx,
            response.content[:50],
        )
        return response


# ---------------------------------------------------------------------------
# 5. Middleware examples
# ---------------------------------------------------------------------------


class LoggingLLMClient:
    """
    Middleware that logs every request/response pair.
    Wraps any LLMClient implementation.
    """

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner

    def complete(self, request: LLMRequest) -> LLMResponse:
        logger.info(
            "LLM request | model=%s max_tokens=%d messages=%d",
            request.model,
            request.max_tokens,
            len(request.messages),
        )
        response = self._inner.complete(request)
        logger.info(
            "LLM response | input_tokens=%d output_tokens=%d stop=%s",
            response.input_tokens,
            response.output_tokens,
            response.stop_reason,
        )
        return response


class CostTrackingLLMClient:
    """
    Middleware that accumulates token usage across calls.
    Can be combined with LoggingLLMClient via composition.
    """

    INPUT_PRICE_PER_M: float = 15.00   # claude-opus-5-5 input
    OUTPUT_PRICE_PER_M: float = 75.00  # claude-opus-5-5 output

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self._total_input: int = 0
        self._total_output: int = 0

    @property
    def total_cost_usd(self) -> float:
        return (self._total_input / 1_000_000) * self.INPUT_PRICE_PER_M + (
            self._total_output / 1_000_000
        ) * self.OUTPUT_PRICE_PER_M

    def complete(self, request: LLMRequest) -> LLMResponse:
        response = self._inner.complete(request)
        self._total_input += response.input_tokens
        self._total_output += response.output_tokens
        return response

    def usage_summary(self) -> dict:
        return {
            "total_input_tokens": self._total_input,
            "total_output_tokens": self._total_output,
            "estimated_cost_usd": round(self.total_cost_usd, 6),
        }


# ---------------------------------------------------------------------------
# 6. Example service that depends on LLMClient (injectable)
# ---------------------------------------------------------------------------


class SummarizationService:
    """
    Summarizes text using an injected LLM client.

    The client can be any object satisfying LLMClient — production or mock.
    """

    SYSTEM_PROMPT = "You are a concise summarization assistant."

    def __init__(self, client: LLMClient, model: str = "claude-opus-5-5") -> None:
        self._client = client
        self._model = model

    def summarize(self, text: str, max_words: int = 50) -> str:
        request = LLMRequest(
            messages=[
                LLMMessage(
                    role="user",
                    content=f"Summarize the following in at most {max_words} words:\n\n{text}",
                )
            ],
            system=self.SYSTEM_PROMPT,
            max_tokens=256,
            model=self._model,
        )
        response = self._client.complete(request)
        return response.content.strip()
```

Tests using `MockLLMClient` — no patching needed:

```python
"""
tests/test_day5_client.py
All tests use MockLLMClient. No real API calls, no mock.patch.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from week_08.day_5 import (
    AnthropicClient,
    CostTrackingLLMClient,
    LLMClient,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LoggingLLMClient,
    MockLLMClient,
    MockLLMClientError,
    SummarizationService,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_response(content: str = "hello", tokens: int = 10) -> LLMResponse:
    return LLMResponse(
        content=content,
        input_tokens=tokens,
        output_tokens=tokens // 2,
        model="mock",
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_mock_client_satisfies_protocol() -> None:
    """MockLLMClient satisfies the LLMClient Protocol at runtime."""
    mock = MockLLMClient(responses=[make_response()])
    assert isinstance(mock, LLMClient)


# ---------------------------------------------------------------------------
# MockLLMClient behaviour
# ---------------------------------------------------------------------------


def test_mock_returns_responses_in_order() -> None:
    responses = [make_response(f"response_{i}") for i in range(3)]
    mock = MockLLMClient(responses=responses)
    req = LLMRequest(messages=[LLMMessage(role="user", content="hi")])

    for i in range(3):
        resp = mock.complete(req)
        assert resp.content == f"response_{i}"

    assert mock.call_count == 3


def test_mock_raises_when_exhausted() -> None:
    mock = MockLLMClient(responses=[make_response()])
    req = LLMRequest(messages=[LLMMessage(role="user", content="hi")])
    mock.complete(req)  # first call succeeds

    with pytest.raises(StopIteration):
        mock.complete(req)  # no more responses


def test_mock_raises_on_configured_error_call() -> None:
    mock = MockLLMClient(responses=[make_response(), make_response()], error_on_calls=[1])
    req = LLMRequest(messages=[LLMMessage(role="user", content="hi")])

    mock.complete(req)  # call 0: success
    with pytest.raises(MockLLMClientError):
        mock.complete(req)  # call 1: configured error


def test_mock_records_requests() -> None:
    mock = MockLLMClient(responses=[make_response(), make_response()])
    req1 = LLMRequest(messages=[LLMMessage(role="user", content="first")])
    req2 = LLMRequest(messages=[LLMMessage(role="user", content="second")])

    mock.complete(req1)
    mock.complete(req2)

    assert mock.recorded_requests[0].messages[0].content == "first"
    assert mock.recorded_requests[1].messages[0].content == "second"


# ---------------------------------------------------------------------------
# SummarizationService with injected mock
# ---------------------------------------------------------------------------


def test_summarization_service_uses_mock_response() -> None:
    expected_summary = "This is a short summary."
    mock = MockLLMClient(responses=[make_response(content=expected_summary)])
    service = SummarizationService(client=mock)

    result = service.summarize("A long piece of text that needs summarizing...")
    assert result == expected_summary
    assert mock.call_count == 1


def test_summarization_service_includes_max_words_in_prompt() -> None:
    mock = MockLLMClient(responses=[make_response()])
    service = SummarizationService(client=mock)

    service.summarize("some text", max_words=25)

    prompt = mock.recorded_requests[0].messages[0].content
    assert "25" in prompt


def test_summarization_service_sets_system_prompt() -> None:
    mock = MockLLMClient(responses=[make_response()])
    service = SummarizationService(client=mock)

    service.summarize("some text")

    req = mock.recorded_requests[0]
    assert req.system is not None
    assert len(req.system) > 0


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


def test_cost_tracking_accumulates_over_calls() -> None:
    responses = [make_response("r1", tokens=1000), make_response("r2", tokens=2000)]
    mock = MockLLMClient(responses=responses)
    tracker = CostTrackingLLMClient(inner=mock)

    req = LLMRequest(messages=[LLMMessage(role="user", content="hi")])
    tracker.complete(req)
    tracker.complete(req)

    summary = tracker.usage_summary()
    assert summary["total_input_tokens"] == 3000
    assert summary["estimated_cost_usd"] > 0


def test_logging_client_forwards_response() -> None:
    expected = make_response("forwarded content")
    mock = MockLLMClient(responses=[expected])
    logging_client = LoggingLLMClient(inner=mock)

    req = LLMRequest(messages=[LLMMessage(role="user", content="hi")])
    resp = logging_client.complete(req)

    assert resp.content == "forwarded content"


def test_middleware_chain() -> None:
    """LoggingClient wrapping CostTrackingClient wrapping MockClient."""
    mock = MockLLMClient(responses=[make_response("chained", tokens=500)])
    tracker = CostTrackingLLMClient(inner=mock)
    logged = LoggingLLMClient(inner=tracker)

    req = LLMRequest(messages=[LLMMessage(role="user", content="hi")])
    resp = logged.complete(req)

    assert resp.content == "chained"
    assert tracker.usage_summary()["total_input_tokens"] == 500


# ---------------------------------------------------------------------------
# Data model validation
# ---------------------------------------------------------------------------


def test_llm_message_rejects_invalid_role() -> None:
    with pytest.raises(ValidationError):
        LLMMessage(role="system", content="oops")  # type: ignore[arg-type]


def test_llm_request_defaults() -> None:
    req = LLMRequest(messages=[LLMMessage(role="user", content="hi")])
    assert req.max_tokens == 1024
    assert req.model == "claude-opus-5-5"
    assert req.stop_sequences == []
```

---

## Exercises

(Solutions: ../solutions/week-08/day-5.md)

1. **Streaming support**: Add a `stream` method to the `LLMClient` protocol with signature `def stream(self, request: LLMRequest) -> Iterator[str]`. Implement it in `AnthropicClient` using `client.messages.stream(...)`. Implement it in `MockLLMClient` so it yields each character of the configured response one at a time (simulating token-by-token streaming).

2. **OpenAI-compatible client**: Implement `OpenAIClient` satisfying `LLMClient` using the `openai` SDK. Map `LLMRequest` fields to `openai.chat.completions.create` parameters. Map the response back to `LLMResponse`. The only code that should differ from `AnthropicClient` is inside `OpenAIClient.complete`.

3. **Rate-limiting middleware**: Implement `RateLimitingLLMClient` that wraps any `LLMClient` and enforces a maximum of N calls per minute using a token bucket algorithm. Block (sleep) rather than raise when the limit is reached. Use `time.monotonic()` for the bucket refill calculation. Write tests that verify N+1 calls take measurably longer than N calls.

4. **Integration-style pipeline test**: Using `MockLLMClient` only (no patches), write a test for a two-step pipeline:
   - Step 1: classify a document (returns `{"category": "invoice"}`)
   - Step 2: extract structured data based on category (returns `{"vendor": "Acme", "amount": 42.0}`)
   Verify the second request's prompt references the category from step 1. Configure `MockLLMClient` with exactly two responses and assert `call_count == 2` at the end.

---

## Self-check

- [ ] I can explain the difference between `Protocol` (structural) and `ABC` (nominal) subtyping, and when to choose each.
- [ ] I can describe why injecting a mock client is preferable to patching the SDK with `unittest.mock.patch`.
- [ ] I can implement a middleware client that wraps another `LLMClient` and adds cross-cutting behavior.
- [ ] I can design `MockLLMClient` to support error injection and request recording for test assertions.
- [ ] I can draw the dependency graph: application code → `LLMClient` protocol ← `AnthropicClient` / `MockLLMClient`.
- [ ] I can explain why `LLMRequest`/`LLMResponse` as Pydantic models at the boundary is better than raw dicts.

---

## Supplementary reading

- [PEP 544 — Protocols: Structural subtyping](https://peps.python.org/pep-0544/)
- [Python typing.Protocol documentation](https://docs.python.org/3/library/typing.html#typing.Protocol)
- [Dependency Injection in Python — Real Python](https://realpython.com/dependency-injection-python/)
- [Anthropic SDK streaming guide](https://docs.anthropic.com/en/api/messages-streaming)
