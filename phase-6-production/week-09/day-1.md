# Phase 6 · Week 09 · Day 1 — Gateway Pattern + Provider Abstraction

> Objective: Understand the LLM gateway pattern and implement a provider-agnostic abstraction layer.  ·  Time: ~1 hr  ·  Prereqs: [[phase-5-week-08-day-5]]

## Concept

An **LLM gateway** is a single entry-point that routes requests to one or more model providers (Anthropic, OpenAI, Cohere, …) behind a unified interface. Every caller in your system talks to the gateway; the gateway translates to the provider-specific SDK.

Why this matters:

- **Vendor lock-in avoidance** — swap Anthropic for OpenAI by changing one config line, not 40 call-sites.
- **Centralized control plane** — logging, rate limiting, caching, cost tracking all live in one place.
- **Uniform API surface** — downstream code works with `CompletionRequest` / `CompletionResponse`; it never sees provider SDKs.
- **Testability** — a `MockProvider` lets every test run without real API calls.

The pattern is structurally identical to the **Adapter** / **Strategy** design patterns: define an abstract interface, implement one concrete class per provider, inject the concrete instance at startup.

## How it works internally

```
Caller
  │
  ▼
GatewayRouter  ←── ProviderRegistry (name → BaseProvider)
  │
  ├─ AnthropicProvider  →  anthropic.AsyncAnthropic
  ├─ OpenAIProvider     →  openai.AsyncOpenAI
  └─ MockProvider       →  in-process stub
```

1. **Request enters** as a `CompletionRequest` (Pydantic model — validated, typed).
2. **Registry lookup** resolves the provider name to a concrete `BaseProvider` instance.
3. **Provider translates** the generic request to its SDK's format, calls the network, wraps the response as `CompletionResponse`.
4. **Latency / token metadata** attach here — callers never measure raw SDK calls.

The abstract base class (`BaseProvider`) uses Python's `abc.ABC` + `@abstractmethod` to enforce the interface at class-definition time, not at call time. This means a missing `complete()` raises `TypeError` on instantiation, before any request arrives.

Keys are loaded via `os.getenv()` — never hardcoded. The provider reads the key in `__init__`; if the env var is absent the client initialises with `None` and will fail on the first real call (appropriate: fail fast, not silently).

## Worked example

```python
# gateway/providers.py
import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field
import anthropic

logger = logging.getLogger(__name__)


# ── Data models ──────────────────────────────────────────────────────────────

class CompletionRequest(BaseModel):
    """Provider-agnostic completion request."""
    model: str
    messages: list[dict[str, str]]
    max_tokens: int = 1024
    temperature: float = 0.7
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompletionResponse(BaseModel):
    """Normalised response returned by every provider."""
    content: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


# ── Abstract interface ────────────────────────────────────────────────────────

class BaseProvider(ABC):
    """Provider-agnostic LLM interface.

    Every concrete provider must implement `complete` and `provider_name`.
    The gateway calls only these two methods, so providers are interchangeable.
    """

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Send a completion request and return a normalised response."""
        ...

    @abstractmethod
    def provider_name(self) -> str:
        """Return a stable lowercase identifier, e.g. 'anthropic'."""
        ...


# ── Anthropic concrete provider ───────────────────────────────────────────────

class AnthropicProvider(BaseProvider):
    """Wraps the Anthropic Messages API."""

    def __init__(self) -> None:
        # Key read from env — never hardcoded.
        self._client = anthropic.AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

    def provider_name(self) -> str:
        return "anthropic"

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        start = time.monotonic()
        logger.info("anthropic.complete model=%s tokens_max=%d", request.model, request.max_tokens)

        response = await self._client.messages.create(
            model=request.model,
            max_tokens=request.max_tokens,
            messages=request.messages,
        )

        latency_ms = (time.monotonic() - start) * 1000

        return CompletionResponse(
            content=response.content[0].text,
            model=request.model,
            provider=self.provider_name(),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
        )


# ── Mock provider (tests / local dev) ────────────────────────────────────────

class MockProvider(BaseProvider):
    """Returns deterministic responses without any network calls.

    Pass `responses` to control what `complete()` returns.
    """

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = responses or ["mock response"]
        self._call_count = 0

    def provider_name(self) -> str:
        return "mock"

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        idx = self._call_count % len(self._responses)
        text = self._responses[idx]
        self._call_count += 1
        logger.debug("mock.complete call=%d model=%s", self._call_count, request.model)
        return CompletionResponse(
            content=text,
            model=request.model,
            provider=self.provider_name(),
            input_tokens=10,
            output_tokens=len(text.split()),
            latency_ms=1.0,
        )


# ── Provider registry ─────────────────────────────────────────────────────────

class ProviderRegistry:
    """Maps provider name → BaseProvider instance."""

    def __init__(self) -> None:
        self._providers: dict[str, BaseProvider] = {}

    def register(self, provider: BaseProvider) -> None:
        name = provider.provider_name()
        logger.info("registry.register provider=%s", name)
        self._providers[name] = provider

    def get(self, name: str) -> BaseProvider:
        if name not in self._providers:
            raise KeyError(f"Provider '{name}' not registered. Available: {list(self._providers)}")
        return self._providers[name]

    def list_providers(self) -> list[str]:
        return list(self._providers.keys())
```

```python
# tests/unit/test_providers.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.providers import (
    AnthropicProvider,
    CompletionRequest,
    CompletionResponse,
    MockProvider,
    ProviderRegistry,
)


# ── Helper fixtures ───────────────────────────────────────────────────────────

def make_request(model: str = "claude-haiku-5-5") -> CompletionRequest:
    return CompletionRequest(
        model=model,
        messages=[{"role": "user", "content": "Hello"}],
    )


def make_sdk_response(text: str = "Hi") -> MagicMock:
    """Build a MagicMock that looks like an anthropic.types.Message."""
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    msg.usage.input_tokens = 5
    msg.usage.output_tokens = 2
    return msg


# ── AnthropicProvider ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_anthropic_complete_returns_response():
    """complete() wraps SDK output into CompletionResponse."""
    sdk_msg = make_sdk_response("Hello from Claude")

    with patch("gateway.providers.anthropic.AsyncAnthropic") as mock_cls:
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=sdk_msg)
        mock_cls.return_value = mock_client

        provider = AnthropicProvider()
        result = await provider.complete(make_request())

    assert isinstance(result, CompletionResponse)
    assert result.content == "Hello from Claude"
    assert result.provider == "anthropic"
    assert result.input_tokens == 5
    assert result.output_tokens == 2
    assert result.latency_ms > 0  # must be positive


@pytest.mark.asyncio
async def test_anthropic_complete_passes_model_id():
    """The model id in the request is forwarded to the SDK and reflected in the response."""
    sdk_msg = make_sdk_response()

    with patch("gateway.providers.anthropic.AsyncAnthropic") as mock_cls:
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=sdk_msg)
        mock_cls.return_value = mock_client

        provider = AnthropicProvider()
        result = await provider.complete(make_request("claude-sonnet-5-5"))

    assert result.model == "claude-sonnet-5-5"


# ── MockProvider ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_mock_provider_cycles_responses():
    provider = MockProvider(responses=["first", "second"])
    r1 = await provider.complete(make_request())
    r2 = await provider.complete(make_request())
    r3 = await provider.complete(make_request())  # wraps back to "first"

    assert r1.content == "first"
    assert r2.content == "second"
    assert r3.content == "first"


# ── ProviderRegistry ──────────────────────────────────────────────────────────

def test_registry_register_and_get():
    registry = ProviderRegistry()
    mock = MockProvider()
    registry.register(mock)

    assert registry.get("mock") is mock


def test_registry_get_unknown_raises():
    registry = ProviderRegistry()
    with pytest.raises(KeyError, match="not registered"):
        registry.get("unknown")
```

Usage:

```python
import asyncio
from gateway.providers import AnthropicProvider, ProviderRegistry, CompletionRequest

async def main() -> None:
    registry = ProviderRegistry()
    registry.register(AnthropicProvider())

    provider = registry.get("anthropic")
    response = await provider.complete(
        CompletionRequest(
            model="claude-haiku-5-5",
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )
    )
    print(response.content)  # replace with logger in production

asyncio.run(main())
```

## Exercises

1. Add an `OpenAIProvider` stub class that inherits `BaseProvider`, returns `provider_name() = "openai"`, and raises `NotImplementedError` from `complete()`. Register it alongside `AnthropicProvider` in a `ProviderRegistry` and verify `list_providers()` returns both names.

2. Add a `list_models()` abstract method to `BaseProvider` with return type `list[str]`. Implement it on `AnthropicProvider` returning `["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]`, and on `MockProvider` returning `["mock-model"]`. Add a test that calls `list_models()` on both concrete types.

3. Write a `GatewayRouter` class that accepts a `ProviderRegistry` and exposes `async def route(self, provider_name: str, request: CompletionRequest) -> CompletionResponse`. It should look up the provider, delegate to `complete()`, and re-raise any exception with the provider name included in the message.

4. Write a pytest test for `AnthropicProvider.complete()` that:
   - patches `anthropic.AsyncAnthropic` so no real network call is made
   - asserts `result.latency_ms > 0`
   - asserts `result.input_tokens == <value you set in the mock>`

(Solutions: ../solutions/week-09/day-1.md)

## Self-check

- [ ] I can explain why a gateway needs a provider abstraction (one concrete reason per stakeholder: ops, dev, test)
- [ ] I can implement `BaseProvider` with full type hints and `@abstractmethod` decorators from memory
- [ ] I can write a mock test that patches the Anthropic client and never calls the real API
