# Week 8 Day 5 — Solutions: Provider-Agnostic LLM Client

## Exercise 1: Add Streaming Support to LLMClient Protocol

The key insight: streaming is an alternate execution path, not a separate client. Add
`stream_complete` to the Protocol returning an `Iterator[str]`, so callers can consume
token-by-token without buffering the full response.

```python
from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import anthropic

logger = logging.getLogger(__name__)


class LLMResponse:
    """Structured response from any LLM provider."""

    def __init__(
        self,
        content: str,
        input_tokens: int,
        output_tokens: int,
        model: str,
        stop_reason: str,
    ) -> None:
        self.content = content
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.model = model
        self.stop_reason = stop_reason

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __repr__(self) -> str:
        return (
            f"LLMResponse(model={self.model!r}, "
            f"tokens={self.input_tokens}+{self.output_tokens})"
        )


@runtime_checkable
class LLMClient(Protocol):
    """Provider-agnostic interface for LLM completions."""

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Single-shot completion. Returns full response."""
        ...

    def stream_complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Iterator[str]:
        """Streaming completion. Yields text chunks as they arrive."""
        ...


class AnthropicClient:
    """Production Anthropic implementation of LLMClient."""

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self._client = anthropic.Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )
        logger.info("AnthropicClient initialised with model=%s", model)

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        logger.debug("complete() called, system=%d chars", len(system))
        message = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return LLMResponse(
            content=message.content[0].text,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            model=message.model,
            stop_reason=message.stop_reason or "end_turn",
        )

    def stream_complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Iterator[str]:
        """Yields text deltas. Caller is responsible for joining if needed."""
        logger.debug("stream_complete() called, system=%d chars", len(system))
        with self._client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            for text in stream.text_stream:
                yield text
        logger.debug("stream_complete() finished")
```

**Why `Iterator[str]` not `AsyncIterator`?**
The Anthropic SDK's synchronous streaming context manager returns a regular iterator.
Async streaming requires `anthropic.AsyncAnthropic` — covered in Day 6. Keeping the
Protocol synchronous first avoids mixing async concerns into the base abstraction.

---

## Exercise 2: OpenAI-Compatible Client

The OpenAI SDK uses `client.chat.completions.create()` with a `messages` list. Streaming
uses `stream=True` and iterates `chunk.choices[0].delta.content`.

```python
from __future__ import annotations

import logging
import os
from collections.abc import Iterator

from openai import OpenAI

from .protocol import LLMClient, LLMResponse  # same module as above

logger = logging.getLogger(__name__)


class OpenAIClient:
    """OpenAI implementation of LLMClient."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        # base_url allows pointing at any OpenAI-compatible endpoint (Ollama, vLLM, etc.)
        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
        )
        logger.info("OpenAIClient initialised with model=%s", model)

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            content=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=response.model,
            stop_reason=choice.finish_reason or "stop",
        )

    def stream_complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta is not None:
                yield delta
```

**Runtime check — both satisfy the Protocol:**
```python
assert isinstance(AnthropicClient(), LLMClient)
assert isinstance(OpenAIClient(), LLMClient)
```

---

## Exercise 3: Middleware Pattern

Each middleware wraps an `LLMClient` and delegates to it, adding cross-cutting behaviour.
This is the Decorator pattern — no subclassing, fully composable.

```python
from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from threading import Lock

from .protocol import LLMClient, LLMResponse

logger = logging.getLogger(__name__)


class LoggingMiddleware:
    """Logs every call and its latency."""

    def __init__(self, inner: LLMClient, name: str = "llm") -> None:
        self._inner = inner
        self._name = name

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        start = time.perf_counter()
        logger.info("[%s] complete() start", self._name)
        try:
            response = self._inner.complete(
                system, user, max_tokens=max_tokens, temperature=temperature
            )
            elapsed = time.perf_counter() - start
            logger.info(
                "[%s] complete() ok  tokens=%d  latency=%.3fs",
                self._name,
                response.total_tokens,
                elapsed,
            )
            return response
        except Exception:
            elapsed = time.perf_counter() - start
            logger.exception(
                "[%s] complete() error  latency=%.3fs", self._name, elapsed
            )
            raise

    def stream_complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Iterator[str]:
        logger.info("[%s] stream_complete() start", self._name)
        chunk_count = 0
        for chunk in self._inner.stream_complete(
            system, user, max_tokens=max_tokens, temperature=temperature
        ):
            chunk_count += 1
            yield chunk
        logger.info("[%s] stream_complete() done  chunks=%d", self._name, chunk_count)


# ---------------------------------------------------------------------------

# Approximate Anthropic pricing (USD per million tokens) as of mid-2025.
# Replace with your own pricing table as needed.
_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5-5":    (15.00, 75.00),
    "claude-sonnet-5-5":  (3.00,  15.00),
    "claude-haiku-3-5":   (0.80,   4.00),
    "gpt-4o":             (5.00,  15.00),
    "gpt-4o-mini":        (0.15,   0.60),
}


class CostTrackingMiddleware:
    """Accumulates USD cost across all calls."""

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self._lock = Lock()
        self.total_cost_usd: float = 0.0
        self.total_calls: int = 0

    def _add_cost(self, response: LLMResponse) -> None:
        input_price, output_price = _PRICING.get(response.model, (0.0, 0.0))
        cost = (
            response.input_tokens * input_price / 1_000_000
            + response.output_tokens * output_price / 1_000_000
        )
        with self._lock:
            self.total_cost_usd += cost
            self.total_calls += 1
        logger.debug(
            "cost +$%.6f  (total=$%.4f  calls=%d)",
            cost,
            self.total_cost_usd,
            self.total_calls,
        )

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        response = self._inner.complete(
            system, user, max_tokens=max_tokens, temperature=temperature
        )
        self._add_cost(response)
        return response

    def stream_complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Iterator[str]:
        # Streaming does not give token counts until the stream ends,
        # so we cannot track cost for streaming calls here without
        # collecting the final usage event. This is a known limitation.
        logger.warning("CostTrackingMiddleware: streaming cost not tracked")
        yield from self._inner.stream_complete(
            system, user, max_tokens=max_tokens, temperature=temperature
        )


# ---------------------------------------------------------------------------

class RateLimitError(RuntimeError):
    """Raised when the rate limit would be exceeded."""


class RateLimitMiddleware:
    """Simple token-bucket rate limiter (calls per minute)."""

    def __init__(self, inner: LLMClient, calls_per_minute: int = 60) -> None:
        self._inner = inner
        self._interval = 60.0 / calls_per_minute  # seconds between calls
        self._last_call: float = 0.0
        self._lock = Lock()

    def _wait_for_slot(self) -> None:
        with self._lock:
            now = time.time()
            wait = self._interval - (now - self._last_call)
            if wait > 0:
                logger.debug("rate limit: sleeping %.3fs", wait)
                time.sleep(wait)
            self._last_call = time.time()

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self._wait_for_slot()
        return self._inner.complete(
            system, user, max_tokens=max_tokens, temperature=temperature
        )

    def stream_complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Iterator[str]:
        self._wait_for_slot()
        yield from self._inner.stream_complete(
            system, user, max_tokens=max_tokens, temperature=temperature
        )
```

**Composing middleware (order matters — outermost runs first):**
```python
base_client: LLMClient = AnthropicClient()
cost_tracker = CostTrackingMiddleware(base_client)
rate_limited = RateLimitMiddleware(cost_tracker, calls_per_minute=30)
logged = LoggingMiddleware(rate_limited, name="prod")

# logged -> rate_limited -> cost_tracker -> base_client
result = logged.complete(system="You are helpful.", user="Hello")
print(f"Total cost so far: ${cost_tracker.total_cost_usd:.4f}")
```

---

## Exercise 4: Integration Tests Using MockLLMClient

No `mock.patch` needed — we inject `MockLLMClient` at construction time.

```python
# tests/test_pipeline_integration.py
from __future__ import annotations

import json
import logging
from collections.abc import Iterator

import pytest

from week08.middleware import CostTrackingMiddleware, LoggingMiddleware
from week08.protocol import LLMClient, LLMResponse


# ---------------------------------------------------------------------------
# MockLLMClient — injectable test double, no network required
# ---------------------------------------------------------------------------

class MockLLMClient:
    """Configurable fake LLM for unit and integration tests."""

    def __init__(
        self,
        responses: list[str] | None = None,
        model: str = "claude-opus-5-5",
        input_tokens: int = 100,
        output_tokens: int = 50,
    ) -> None:
        self._responses = list(responses or ['{"result": "ok"}'])
        self._call_index = 0
        self._model = model
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self.calls: list[dict] = []  # introspect what was sent

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self.calls.append({"system": system, "user": user})
        idx = min(self._call_index, len(self._responses) - 1)
        content = self._responses[idx]
        self._call_index += 1
        return LLMResponse(
            content=content,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            model=self._model,
            stop_reason="end_turn",
        )

    def stream_complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Iterator[str]:
        response = self.complete(system, user, max_tokens=max_tokens, temperature=temperature)
        # Simulate streaming by yielding word-by-word
        for word in response.content.split():
            yield word + " "


# ---------------------------------------------------------------------------
# A trivial "pipeline" that uses LLMClient — exercises the whole stack
# ---------------------------------------------------------------------------

class SummarisationPipeline:
    def __init__(self, client: LLMClient) -> None:
        self._client = client
        self._system = "Summarise the following text in one sentence."

    def summarise(self, text: str) -> str:
        response = self._client.complete(system=self._system, user=text)
        return response.content.strip()

    def summarise_streaming(self, text: str) -> str:
        chunks = list(
            self._client.stream_complete(system=self._system, user=text)
        )
        return "".join(chunks).strip()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipelineWithMock:
    def test_summarise_returns_llm_content(self) -> None:
        mock = MockLLMClient(responses=["Short summary."])
        pipeline = SummarisationPipeline(mock)
        result = pipeline.summarise("Long article about AI...")
        assert result == "Short summary."
        assert len(mock.calls) == 1

    def test_system_prompt_passed_correctly(self) -> None:
        mock = MockLLMClient(responses=["ok"])
        pipeline = SummarisationPipeline(mock)
        pipeline.summarise("anything")
        assert "Summarise" in mock.calls[0]["system"]

    def test_streaming_joins_chunks(self) -> None:
        mock = MockLLMClient(responses=["chunk one two"])
        pipeline = SummarisationPipeline(mock)
        result = pipeline.summarise_streaming("input text")
        assert "chunk" in result
        assert "one" in result
        assert "two" in result

    def test_multiple_calls_cycle_responses(self) -> None:
        mock = MockLLMClient(responses=["first", "second", "third"])
        pipeline = SummarisationPipeline(mock)
        assert pipeline.summarise("a") == "first"
        assert pipeline.summarise("b") == "second"
        assert pipeline.summarise("c") == "third"
        # Beyond the list: stays at last response
        assert pipeline.summarise("d") == "third"


class TestMiddlewareStack:
    def test_cost_tracking_accumulates(self) -> None:
        mock = MockLLMClient(
            responses=["response"],
            model="claude-opus-5-5",
            input_tokens=1000,
            output_tokens=500,
        )
        tracker = CostTrackingMiddleware(mock)
        pipeline = SummarisationPipeline(tracker)

        pipeline.summarise("doc 1")
        pipeline.summarise("doc 2")

        assert tracker.total_calls == 2
        # 2 * (1000 * 15/1M + 500 * 75/1M) = 2 * (0.015 + 0.0375) = 0.105
        assert abs(tracker.total_cost_usd - 0.105) < 1e-6

    def test_logging_middleware_does_not_swallow_errors(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        class FailingClient:
            def complete(self, *args, **kwargs) -> LLMResponse:  # type: ignore[override]
                raise ValueError("upstream error")

            def stream_complete(self, *args, **kwargs) -> Iterator[str]:  # type: ignore[override]
                raise ValueError("upstream error")
                yield  # make it a generator

        logged = LoggingMiddleware(FailingClient(), name="test")
        pipeline = SummarisationPipeline(logged)

        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError, match="upstream error"):
                pipeline.summarise("anything")

    def test_full_stack_integration(self) -> None:
        """LoggingMiddleware -> CostTrackingMiddleware -> MockLLMClient."""
        mock = MockLLMClient(
            responses=["result"],
            model="claude-opus-5-5",
            input_tokens=200,
            output_tokens=100,
        )
        tracker = CostTrackingMiddleware(mock)
        logged = LoggingMiddleware(tracker)
        pipeline = SummarisationPipeline(logged)

        result = pipeline.summarise("test input")

        assert result == "result"
        assert tracker.total_calls == 1
        assert tracker.total_cost_usd > 0
```

**Key points:**
- `MockLLMClient.calls` lets tests assert on what was *sent* to the LLM, not just what came back.
- Middleware tests use the same injection pattern — stack them in the test, not in production code.
- `FailingClient` inline class tests error propagation without any `mock.patch`.
