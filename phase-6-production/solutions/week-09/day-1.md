# Solutions — Day 1: Gateway Pattern + Provider Abstraction

---

## Exercise 1: OpenAI Provider Stub

Full implementation of `OpenAIProvider` conforming to `BaseProvider`. No real API calls — returns a deterministic stub response. In production, swap the stub body for an `openai.AsyncOpenAI` client call.

```python
# gateway/providers/openai_provider.py
import logging
import time

from gateway.providers.base import BaseProvider, CompletionRequest, CompletionResponse

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """Stub OpenAI provider — no real API calls."""

    def provider_name(self) -> str:
        return "openai"

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        # In production: call openai.AsyncOpenAI client
        logger.info("openai.complete stub model=%s", request.model)
        return CompletionResponse(
            content="[openai stub response]",
            model=request.model,
            provider=self.provider_name(),
            input_tokens=50,
            output_tokens=20,
            latency_ms=100.0,
        )

    def list_models(self) -> list[str]:
        return ["gpt-4o", "gpt-4o-mini"]
```

Key points:
- `provider_name()` returns a lowercase string used as the registry key.
- `complete()` is `async` — always await it.
- `list_models()` is synchronous; no I/O needed.

---

## Exercise 2: `list_models` on `BaseProvider` and `AnthropicProvider`

### `BaseProvider` — add abstract method

```python
# gateway/providers/base.py
from abc import ABC, abstractmethod
from pydantic import BaseModel


class CompletionRequest(BaseModel):
    model: str
    messages: list[dict[str, str]]
    max_tokens: int = 1024
    temperature: float = 0.7


class CompletionResponse(BaseModel):
    content: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class BaseProvider(ABC):
    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse: ...

    @abstractmethod
    def provider_name(self) -> str: ...

    @abstractmethod
    def list_models(self) -> list[str]: ...
```

### `AnthropicProvider` — implement `list_models`

```python
# gateway/providers/anthropic_provider.py
import logging
import os
import time

import anthropic

from gateway.providers.base import BaseProvider, CompletionRequest, CompletionResponse

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseProvider):
    def __init__(self) -> None:
        self._client = anthropic.AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

    def provider_name(self) -> str:
        return "anthropic"

    def list_models(self) -> list[str]:
        return ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        start = time.perf_counter()
        message = await self._client.messages.create(
            model=request.model,
            max_tokens=request.max_tokens,
            messages=request.messages,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        content = message.content[0].text if message.content else ""
        logger.info(
            "anthropic.complete model=%s latency_ms=%.1f",
            request.model,
            latency_ms,
        )
        return CompletionResponse(
            content=content,
            model=request.model,
            provider=self.provider_name(),
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            latency_ms=latency_ms,
        )
```

---

## Exercise 3: `ProviderRegistry`

```python
# gateway/registry.py
import logging

from gateway.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, BaseProvider] = {}

    def register(self, provider: BaseProvider) -> None:
        self._providers[provider.provider_name()] = provider
        logger.info("registered provider=%s", provider.provider_name())

    def get(self, name: str) -> BaseProvider:
        if name not in self._providers:
            raise KeyError(f"Provider '{name}' not registered")
        return self._providers[name]

    def list_providers(self) -> list[str]:
        return list(self._providers.keys())
```

Usage:

```python
registry = ProviderRegistry()
registry.register(AnthropicProvider())
registry.register(OpenAIProvider())

print(registry.list_providers())  # ["anthropic", "openai"]
provider = registry.get("anthropic")
```

`get()` raises `KeyError` (not `ValueError`) so callers can distinguish "unknown provider" from other errors.

---

## Exercise 4: pytest test for `AnthropicProvider` with mock

```python
# tests/unit/test_anthropic_provider.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.providers.anthropic_provider import AnthropicProvider
from gateway.providers.base import CompletionRequest


@pytest.mark.asyncio
async def test_anthropic_complete_mocked() -> None:
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="Hello from mock")]
    mock_message.usage.input_tokens = 10
    mock_message.usage.output_tokens = 5

    with patch("anthropic.AsyncAnthropic") as mock_cls:
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_message)
        mock_cls.return_value = mock_client

        provider = AnthropicProvider()
        request = CompletionRequest(
            model="claude-haiku-5-5",
            messages=[{"role": "user", "content": "hi"}],
        )
        response = await provider.complete(request)

    assert response.content == "Hello from mock"
    assert response.provider == "anthropic"
    assert response.latency_ms >= 0.0
    assert response.input_tokens == 10
    assert response.output_tokens == 5


@pytest.mark.asyncio
async def test_anthropic_list_models() -> None:
    with patch("anthropic.AsyncAnthropic"):
        provider = AnthropicProvider()

    models = provider.list_models()
    assert "claude-haiku-5-5" in models
    assert "claude-sonnet-5-5" in models
    assert "claude-opus-5-5" in models


@pytest.mark.asyncio
async def test_provider_registry_get_raises_on_missing() -> None:
    from gateway.registry import ProviderRegistry

    registry = ProviderRegistry()
    with pytest.raises(KeyError, match="unknown"):
        registry.get("unknown")
```

### Running the tests

```bash
pytest tests/unit/test_anthropic_provider.py -v
```

No real credentials needed — all Anthropic SDK calls are patched at `anthropic.AsyncAnthropic`.
