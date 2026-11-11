# Solutions — Week 9 Project: Production Model Gateway Service

## Package Structure

```
gateway/
├── __init__.py
├── providers/
│   ├── __init__.py
│   ├── base.py              # BaseProvider, CompletionRequest, CompletionResponse, ProviderRegistry
│   ├── anthropic_provider.py  # AnthropicProvider (real Anthropic SDK calls)
│   └── mock_provider.py     # MockProvider (for tests — no API calls)
├── routing/
│   ├── __init__.py
│   ├── router.py            # ModelRouter, RoutingPolicy, COST_TABLE
│   └── fallback.py          # FallbackChain, ProviderError hierarchy
├── caching/
│   ├── __init__.py
│   └── exact_cache.py       # ExactCache with TTL
├── quota/
│   ├── __init__.py
│   ├── rate_limiter.py      # RateLimiter — token bucket per tenant
│   └── quota_tracker.py     # QuotaTracker — per-tenant daily budget
├── logging_/
│   ├── __init__.py
│   └── cost_attributor.py   # CostAttributor, CostRecord
└── api/
    ├── __init__.py
    └── main.py              # FastAPI app — fully wired

tests/
├── conftest.py              # MockProvider fixture, shared registry
├── test_providers.py
├── test_routing.py
├── test_fallback.py
├── test_caching.py
├── test_quota.py
└── test_api.py

pyproject.toml
.env.example
```

---

## `pyproject.toml`

```toml
[project]
name = "model-gateway"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "anthropic>=0.40",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "httpx>=0.27",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.24",
    "pytest-cov>=5",
    "ruff>=0.6",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py312"
```

---

## `gateway/providers/base.py`

```python
import time
import logging
import os
from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CompletionRequest(BaseModel):
    model: str
    messages: list[dict[str, str]]
    max_tokens: int = 1024
    temperature: float = 0.7
    metadata: dict[str, Any] = Field(default_factory=dict)


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

---

## `gateway/providers/anthropic_provider.py`

```python
import os
import time
import logging
import anthropic

from gateway.providers.base import BaseProvider, CompletionRequest, CompletionResponse
from gateway.routing.fallback import RateLimitError, ProviderError

logger = logging.getLogger(__name__)

ANTHROPIC_MODELS = [
    "claude-haiku-5-5",
    "claude-sonnet-5-5",
    "claude-opus-5-5",
]


class AnthropicProvider(BaseProvider):
    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.AsyncAnthropic(api_key=key)

    def provider_name(self) -> str:
        return "anthropic"

    def list_models(self) -> list[str]:
        return ANTHROPIC_MODELS

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        start = time.monotonic()
        try:
            response = await self._client.messages.create(
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                messages=request.messages,  # type: ignore[arg-type]
            )
        except anthropic.RateLimitError as exc:
            raise RateLimitError("anthropic") from exc
        except anthropic.APIError as exc:
            raise ProviderError("anthropic", str(exc)) from exc

        latency_ms = (time.monotonic() - start) * 1000
        content = response.content[0].text if response.content else ""

        return CompletionResponse(
            content=content,
            model=request.model,
            provider="anthropic",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=round(latency_ms, 1),
        )
```

---

## `gateway/providers/mock_provider.py`

```python
import time
from gateway.providers.base import BaseProvider, CompletionRequest, CompletionResponse


class MockProvider(BaseProvider):
    """
    In-memory provider for unit tests.
    Does not call any external API.
    """

    def __init__(
        self,
        should_fail: bool = False,
        fail_count: int = 0,
        latency_ms: float = 5.0,
    ) -> None:
        self._should_fail = should_fail
        self._fail_count = fail_count
        self._latency_ms = latency_ms
        self._call_count = 0

    def provider_name(self) -> str:
        return "mock"

    def list_models(self) -> list[str]:
        return ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self._call_count += 1
        if self._should_fail or self._call_count <= self._fail_count:
            from gateway.routing.fallback import RateLimitError
            raise RateLimitError("mock")

        return CompletionResponse(
            content=f"mock: {request.messages[-1]['content']}",
            model=request.model,
            provider="mock",
            input_tokens=10,
            output_tokens=20,
            latency_ms=self._latency_ms,
        )
```

---

## `gateway/routing/fallback.py`

```python
import logging
from gateway.providers.base import BaseProvider, CompletionRequest, CompletionResponse, ProviderRegistry

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    def __init__(self, provider: str, message: str) -> None:
        self.provider = provider
        super().__init__(f"[{provider}] {message}")


class RateLimitError(ProviderError):
    def __init__(self, provider: str) -> None:
        super().__init__(provider, "rate limit exceeded")


class ContextLengthError(ProviderError):
    def __init__(self, provider: str) -> None:
        super().__init__(provider, "context length exceeded")


class FallbackChain:
    """
    Try each (provider_name, model) pair in order.
    Return the first successful response plus attempt count.
    """

    def __init__(
        self,
        chain: list[tuple[str, str]],
        registry: ProviderRegistry,
    ) -> None:
        self.chain = chain  # [(provider_name, model), ...]
        self.registry = registry

    async def complete(
        self, request: CompletionRequest
    ) -> tuple[CompletionResponse, int]:
        last_exc: Exception | None = None

        for attempt, (provider_name, model) in enumerate(self.chain, start=1):
            try:
                provider = self.registry.get(provider_name)
                req = request.model_copy(update={"model": model})
                response = await provider.complete(req)
                if attempt > 1:
                    logger.info(
                        "fallback succeeded attempt=%d provider=%s model=%s",
                        attempt,
                        provider_name,
                        model,
                    )
                return response, attempt
            except ProviderError as exc:
                logger.warning(
                    "attempt %d failed provider=%s model=%s err=%s",
                    attempt,
                    provider_name,
                    model,
                    exc,
                )
                last_exc = exc
            except KeyError as exc:
                logger.error("provider not registered: %s", exc)
                last_exc = exc

        raise ProviderError("fallback_chain", f"all {len(self.chain)} attempts failed") from last_exc
```

---

## `gateway/routing/router.py`

```python
import logging
from enum import Enum
from gateway.providers.base import ProviderRegistry

logger = logging.getLogger(__name__)


class RoutingPolicy(str, Enum):
    COST = "cost"
    LATENCY = "latency"
    CAPABILITY = "capability"


# Cost per 1K output tokens (USD)
COST_TABLE: dict[str, float] = {
    "claude-haiku-5-5": 0.00125,
    "claude-sonnet-5-5": 0.015,
    "claude-opus-5-5": 0.075,
}

# Ordered by capability (ascending)
CAPABILITY_ORDER = ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]

# Ordered by latency (fastest first)
LATENCY_ORDER = ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]


class ModelRouter:
    def __init__(self, policy: RoutingPolicy = RoutingPolicy.COST) -> None:
        self.policy = policy

    def select_model(
        self,
        available_models: list[str],
        policy: RoutingPolicy | None = None,
    ) -> str:
        """Select the best model from available_models given the routing policy."""
        effective_policy = policy or self.policy

        if not available_models:
            raise ValueError("No models available for routing")

        if effective_policy == RoutingPolicy.COST:
            return min(available_models, key=lambda m: COST_TABLE.get(m, float("inf")))

        if effective_policy == RoutingPolicy.LATENCY:
            for model in LATENCY_ORDER:
                if model in available_models:
                    return model
            return available_models[0]

        if effective_policy == RoutingPolicy.CAPABILITY:
            for model in reversed(CAPABILITY_ORDER):
                if model in available_models:
                    return model
            return available_models[-1]

        return available_models[0]
```

---

## `gateway/caching/exact_cache.py`

```python
import hashlib
import json
import time
import logging
from gateway.providers.base import CompletionRequest, CompletionResponse

logger = logging.getLogger(__name__)


class ExactCache:
    """
    In-memory exact-match cache keyed on SHA-256(canonical JSON of request).
    Entries expire after ttl_seconds.
    """

    def __init__(self, ttl_seconds: int = 300) -> None:
        self.ttl_seconds = ttl_seconds
        self._store: dict[str, tuple[CompletionResponse, float]] = {}

    def make_key(self, request: CompletionRequest) -> str:
        payload = json.dumps(
            {"messages": request.messages, "model": request.model},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> CompletionResponse | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        response, expires_at = entry
        if time.monotonic() > expires_at:
            del self._store[key]
            logger.debug("cache expired key=%s", key[:8])
            return None
        logger.debug("cache hit key=%s", key[:8])
        return response

    def set(self, key: str, response: CompletionResponse) -> None:
        expires_at = time.monotonic() + self.ttl_seconds
        self._store[key] = (response, expires_at)
        logger.debug("cache set key=%s ttl=%ds", key[:8], self.ttl_seconds)

    def clear(self) -> None:
        self._store.clear()

    def size(self) -> int:
        return len(self._store)
```

---

## `gateway/quota/rate_limiter.py`

```python
import time
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Sliding-window rate limiter per tenant.
    Tracks request timestamps in a deque per tenant_id.
    """

    def __init__(self, rpm: int = 60) -> None:
        self.rpm = rpm
        self._window_seconds = 60.0
        self._timestamps: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, tenant_id: str) -> bool:
        now = time.monotonic()
        window_start = now - self._window_seconds

        # evict expired timestamps
        self._timestamps[tenant_id] = [
            t for t in self._timestamps[tenant_id] if t > window_start
        ]

        if len(self._timestamps[tenant_id]) >= self.rpm:
            logger.warning(
                "rate limit hit tenant=%s count=%d rpm=%d",
                tenant_id,
                len(self._timestamps[tenant_id]),
                self.rpm,
            )
            return False

        self._timestamps[tenant_id].append(now)
        return True

    def reset(self, tenant_id: str) -> None:
        """Clear rate limit state for a tenant (testing / admin use)."""
        self._timestamps.pop(tenant_id, None)

    def reset_all(self) -> None:
        self._timestamps.clear()
```

---

## `gateway/quota/quota_tracker.py`

```python
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class QuotaTracker:
    """
    Per-tenant daily budget tracker (in-memory).
    In production, persist to Redis or a database.
    """

    def __init__(self, daily_budget_usd: float = 10.0) -> None:
        self.daily_budget_usd = daily_budget_usd
        self._costs: dict[str, float] = defaultdict(float)
        self._tokens: dict[str, int] = defaultdict(int)

    def check_budget(self, tenant_id: str) -> bool:
        """Returns True if tenant is within budget."""
        return self._costs[tenant_id] < self.daily_budget_usd

    def deduct(self, tenant_id: str, cost_usd: float, tokens: int = 0) -> None:
        self._costs[tenant_id] += cost_usd
        self._tokens[tenant_id] += tokens
        logger.info(
            "cost deducted tenant=%s cost=%.6f total=%.6f remaining=%.6f",
            tenant_id,
            cost_usd,
            self._costs[tenant_id],
            max(0.0, self.daily_budget_usd - self._costs[tenant_id]),
        )

    def get_cost(self, tenant_id: str) -> float:
        return self._costs[tenant_id]

    def get_remaining(self, tenant_id: str) -> float:
        return max(0.0, self.daily_budget_usd - self._costs[tenant_id])

    def reset(self, tenant_id: str) -> None:
        self._costs.pop(tenant_id, None)
        self._tokens.pop(tenant_id, None)
```

---

## `gateway/logging_/cost_attributor.py`

```python
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CostRecord:
    tenant_id: str
    model: str
    cost_usd: float
    input_tokens: int
    output_tokens: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


class CostAttributor:
    def __init__(self) -> None:
        self._records: list[CostRecord] = []
        self._tenant_totals: dict[str, float] = defaultdict(float)

    def record(self, cost: CostRecord) -> None:
        self._records.append(cost)
        self._tenant_totals[cost.tenant_id] += cost.cost_usd
        logger.debug(
            "cost recorded tenant=%s model=%s cost=%.6f",
            cost.tenant_id,
            cost.model,
            cost.cost_usd,
        )

    def get_tenant_cost(self, tenant_id: str) -> float:
        return self._tenant_totals.get(tenant_id, 0.0)

    def top_spenders(self, n: int) -> list[tuple[str, float]]:
        sorted_tenants = sorted(
            self._tenant_totals.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_tenants[:n]

    def all_records(self) -> list[CostRecord]:
        return list(self._records)
```

---

## `gateway/api/main.py`

```python
import os
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from gateway.providers.base import CompletionRequest, CompletionResponse, ProviderRegistry
from gateway.providers.mock_provider import MockProvider
from gateway.routing.router import ModelRouter, RoutingPolicy
from gateway.routing.fallback import FallbackChain, ProviderError
from gateway.caching.exact_cache import ExactCache
from gateway.quota.rate_limiter import RateLimiter
from gateway.quota.quota_tracker import QuotaTracker
from gateway.logging_.cost_attributor import CostAttributor, CostRecord

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# ---------------------------------------------------------------------------
# Singletons (replace MockProvider with AnthropicProvider in production)
# ---------------------------------------------------------------------------

registry = ProviderRegistry()
router = ModelRouter(policy=RoutingPolicy.COST)
cache = ExactCache(ttl_seconds=300)
rate_limiter = RateLimiter(rpm=int(os.getenv("RATE_LIMIT_RPM", "60")))
quota_tracker = QuotaTracker(daily_budget_usd=float(os.getenv("DAILY_BUDGET_USD", "10.0")))
cost_attributor = CostAttributor()

TIER_TO_MODEL: dict[str, str] = {
    "fast": "claude-haiku-5-5",
    "smart": "claude-sonnet-5-5",
    "best": "claude-opus-5-5",
}

COST_OUT_PER_1K: dict[str, float] = {
    "claude-haiku-5-5": 0.00125,
    "claude-sonnet-5-5": 0.015,
    "claude-opus-5-5": 0.075,
}

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

class TenantMiddleware(BaseHTTPMiddleware):
    EXEMPT_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        tenant_id = request.headers.get("X-Tenant-ID")
        if not tenant_id:
            return JSONResponse(
                status_code=400,
                content={"error": "X-Tenant-ID header required"},
            )

        request.state.tenant_id = tenant_id
        return await call_next(request)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    registry.register(MockProvider())  # swap for AnthropicProvider() in prod
    logger.info("gateway started providers=%s", registry.list_providers())
    yield
    logger.info("gateway shutting down")


app = FastAPI(title="Model Gateway", version="0.1.0", lifespan=lifespan)
app.add_middleware(TenantMiddleware)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class GatewayRequest(BaseModel):
    model_tier: str = "smart"  # fast | smart | best
    messages: list[dict[str, str]]


class GatewayResponse(BaseModel):
    content: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cache_hit: bool = False
    cost_usd: float


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "providers": registry.list_providers()}


@app.get("/v1/tenants/{tenant_id}/usage")
async def tenant_usage(tenant_id: str):
    return {
        "tenant_id": tenant_id,
        "daily_cost_usd": round(quota_tracker.get_cost(tenant_id), 6),
        "budget_remaining_usd": round(quota_tracker.get_remaining(tenant_id), 6),
        "top_spenders": cost_attributor.top_spenders(5),
    }


@app.post("/v1/complete", response_model=GatewayResponse)
async def complete(
    req: GatewayRequest,
    x_tenant_id: str = Header(...),
    x_user_id: str = Header(default="anonymous"),
):
    if req.model_tier not in TIER_TO_MODEL:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown tier '{req.model_tier}'. Valid: {list(TIER_TO_MODEL)}",
        )

    if not rate_limiter.is_allowed(x_tenant_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    if not quota_tracker.check_budget(x_tenant_id):
        raise HTTPException(status_code=402, detail="Daily budget exceeded")

    model = TIER_TO_MODEL[req.model_tier]
    completion_req = CompletionRequest(model=model, messages=req.messages)

    # --- Cache check ---
    cache_key = cache.make_key(completion_req)
    cached = cache.get(cache_key)
    if cached:
        logger.info("cache hit tenant=%s model=%s", x_tenant_id, model)
        return GatewayResponse(**cached.model_dump(), cache_hit=True, cost_usd=0.0)

    # --- Fallback chain: preferred model → haiku fallback ---
    fallback_model = "claude-haiku-5-5" if model != "claude-haiku-5-5" else model
    chain = FallbackChain(
        chain=[("mock", model), ("mock", fallback_model)],
        registry=registry,
    )

    try:
        response, attempts = await chain.complete(completion_req)
    except ProviderError as exc:
        logger.error("all fallbacks failed tenant=%s err=%s", x_tenant_id, exc)
        raise HTTPException(status_code=502, detail="All providers failed") from exc

    cache.set(cache_key, response)

    cost = (response.output_tokens / 1000) * COST_OUT_PER_1K.get(response.model, 0.015)
    quota_tracker.deduct(x_tenant_id, cost, response.output_tokens)

    from datetime import datetime
    cost_attributor.record(
        CostRecord(
            tenant_id=x_tenant_id,
            model=response.model,
            cost_usd=cost,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
    )

    logger.info(
        "complete tenant=%s model=%s attempts=%d tokens_out=%d latency_ms=%.1f cost=%.6f",
        x_tenant_id,
        response.model,
        attempts,
        response.output_tokens,
        response.latency_ms,
        cost,
    )

    return GatewayResponse(**response.model_dump(), cost_usd=cost)
```

---

## `tests/conftest.py`

```python
import pytest
from gateway.providers.base import BaseProvider, CompletionRequest, CompletionResponse, ProviderRegistry
from gateway.routing.fallback import RateLimitError


class MockProvider(BaseProvider):
    """
    Configurable mock provider for tests.
    - should_fail=True → always raises RateLimitError
    - fail_count=N → raises on first N calls, succeeds after
    """

    def __init__(
        self,
        should_fail: bool = False,
        fail_count: int = 0,
        latency_ms: float = 5.0,
    ) -> None:
        self._should_fail = should_fail
        self._fail_count = fail_count
        self._latency_ms = latency_ms
        self._call_count = 0

    def provider_name(self) -> str:
        return "mock"

    def list_models(self) -> list[str]:
        return ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self._call_count += 1
        if self._should_fail or self._call_count <= self._fail_count:
            raise RateLimitError("mock")

        return CompletionResponse(
            content=f"mock: {request.messages[-1]['content']}",
            model=request.model,
            provider="mock",
            input_tokens=10,
            output_tokens=20,
            latency_ms=self._latency_ms,
        )


@pytest.fixture
def mock_provider() -> MockProvider:
    return MockProvider()


@pytest.fixture
def failing_provider() -> MockProvider:
    return MockProvider(should_fail=True)


@pytest.fixture
def flaky_provider() -> MockProvider:
    """Fails first call, succeeds after."""
    return MockProvider(fail_count=1)


@pytest.fixture
def registry(mock_provider: MockProvider) -> ProviderRegistry:
    reg = ProviderRegistry()
    reg.register(mock_provider)
    return reg
```

---

## `tests/test_providers.py`

```python
import pytest
from gateway.providers.base import CompletionRequest, ProviderRegistry
from tests.conftest import MockProvider


@pytest.mark.asyncio
async def test_mock_provider_returns_completion(mock_provider):
    req = CompletionRequest(
        model="claude-sonnet-5-5",
        messages=[{"role": "user", "content": "hello"}],
    )
    response = await mock_provider.complete(req)
    assert response.content == "mock: hello"
    assert response.model == "claude-sonnet-5-5"
    assert response.provider == "mock"
    assert response.input_tokens > 0
    assert response.output_tokens > 0
    assert response.latency_ms >= 0


@pytest.mark.asyncio
async def test_failing_provider_raises(failing_provider):
    from gateway.routing.fallback import RateLimitError
    req = CompletionRequest(
        model="claude-haiku-5-5",
        messages=[{"role": "user", "content": "hi"}],
    )
    with pytest.raises(RateLimitError):
        await failing_provider.complete(req)


def test_provider_registry_register_and_get():
    reg = ProviderRegistry()
    provider = MockProvider()
    reg.register(provider)
    assert reg.get("mock") is provider
    assert "mock" in reg.list_providers()


def test_provider_registry_raises_for_unknown():
    reg = ProviderRegistry()
    with pytest.raises(KeyError):
        reg.get("does-not-exist")
```

---

## `tests/test_routing.py`

```python
import pytest
from gateway.routing.router import ModelRouter, RoutingPolicy


def test_cost_routing_selects_cheapest():
    router = ModelRouter(policy=RoutingPolicy.COST)
    models = ["claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5-5"]
    assert router.select_model(models) == "claude-haiku-5-5"


def test_capability_routing_selects_most_powerful():
    router = ModelRouter(policy=RoutingPolicy.CAPABILITY)
    models = ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]
    assert router.select_model(models) == "claude-opus-5-5"


def test_latency_routing_selects_fastest():
    router = ModelRouter(policy=RoutingPolicy.LATENCY)
    models = ["claude-sonnet-5-5", "claude-opus-5-5"]
    assert router.select_model(models) == "claude-sonnet-5-5"


def test_router_raises_on_empty_model_list():
    router = ModelRouter()
    with pytest.raises(ValueError):
        router.select_model([])


def test_per_request_policy_override():
    router = ModelRouter(policy=RoutingPolicy.COST)
    models = ["claude-haiku-5-5", "claude-opus-5-5"]
    # override to capability for this call
    assert router.select_model(models, policy=RoutingPolicy.CAPABILITY) == "claude-opus-5-5"
```

---

## `tests/test_fallback.py`

```python
import pytest
from gateway.routing.fallback import FallbackChain, RateLimitError, ProviderError
from gateway.providers.base import CompletionRequest, ProviderRegistry
from tests.conftest import MockProvider


@pytest.mark.asyncio
async def test_fallback_chain_succeeds_on_first():
    provider = MockProvider()
    reg = ProviderRegistry()
    reg.register(provider)

    chain = FallbackChain(chain=[("mock", "claude-sonnet-5-5")], registry=reg)
    req = CompletionRequest(model="claude-sonnet-5-5", messages=[{"role": "user", "content": "hi"}])
    response, attempts = await chain.complete(req)

    assert attempts == 1
    assert response.content == "mock: hi"


@pytest.mark.asyncio
async def test_fallback_chain_falls_back_on_error():
    flaky = MockProvider(fail_count=1)
    reg = ProviderRegistry()
    reg.register(flaky)

    chain = FallbackChain(
        chain=[("mock", "claude-opus-5-5"), ("mock", "claude-haiku-5-5")],
        registry=reg,
    )
    req = CompletionRequest(model="claude-opus-5-5", messages=[{"role": "user", "content": "hi"}])
    response, attempts = await chain.complete(req)

    assert attempts == 2
    assert response.model == "claude-haiku-5-5"


@pytest.mark.asyncio
async def test_fallback_chain_raises_when_all_fail():
    failing = MockProvider(should_fail=True)
    reg = ProviderRegistry()
    reg.register(failing)

    chain = FallbackChain(
        chain=[("mock", "claude-opus-5-5"), ("mock", "claude-haiku-5-5")],
        registry=reg,
    )
    req = CompletionRequest(model="claude-opus-5-5", messages=[{"role": "user", "content": "hi"}])
    with pytest.raises(ProviderError):
        await chain.complete(req)
```

---

## `tests/test_caching.py`

```python
import pytest
import time
from gateway.caching.exact_cache import ExactCache
from gateway.providers.base import CompletionRequest, CompletionResponse


def _make_response(model: str = "claude-sonnet-5-5") -> CompletionResponse:
    return CompletionResponse(
        content="cached answer",
        model=model,
        provider="mock",
        input_tokens=10,
        output_tokens=20,
        latency_ms=5.0,
    )


def _make_request(content: str = "hello") -> CompletionRequest:
    return CompletionRequest(
        model="claude-sonnet-5-5",
        messages=[{"role": "user", "content": content}],
    )


def test_cache_miss_returns_none():
    cache = ExactCache(ttl_seconds=60)
    key = cache.make_key(_make_request("miss"))
    assert cache.get(key) is None


def test_cache_set_and_get():
    cache = ExactCache(ttl_seconds=60)
    req = _make_request("hello")
    key = cache.make_key(req)
    resp = _make_response()
    cache.set(key, resp)
    cached = cache.get(key)
    assert cached is not None
    assert cached.content == "cached answer"


def test_cache_key_is_deterministic():
    cache = ExactCache()
    req = _make_request("same")
    assert cache.make_key(req) == cache.make_key(req)


def test_different_messages_produce_different_keys():
    cache = ExactCache()
    assert cache.make_key(_make_request("a")) != cache.make_key(_make_request("b"))


def test_cache_expiry():
    cache = ExactCache(ttl_seconds=1)
    req = _make_request("expiry test")
    key = cache.make_key(req)
    cache.set(key, _make_response())
    assert cache.get(key) is not None
    time.sleep(1.1)
    assert cache.get(key) is None


def test_cache_size():
    cache = ExactCache()
    for i in range(5):
        req = _make_request(f"msg-{i}")
        cache.set(cache.make_key(req), _make_response())
    assert cache.size() == 5
    cache.clear()
    assert cache.size() == 0
```

---

## `tests/test_quota.py`

```python
import pytest
from gateway.quota.rate_limiter import RateLimiter
from gateway.quota.quota_tracker import QuotaTracker


def test_rate_limiter_allows_within_limit():
    limiter = RateLimiter(rpm=5)
    for _ in range(5):
        assert limiter.is_allowed("tenant-a") is True


def test_rate_limiter_blocks_over_limit():
    limiter = RateLimiter(rpm=3)
    for _ in range(3):
        limiter.is_allowed("tenant-b")
    assert limiter.is_allowed("tenant-b") is False


def test_rate_limiter_is_per_tenant():
    limiter = RateLimiter(rpm=2)
    limiter.is_allowed("a")
    limiter.is_allowed("a")
    # a is now blocked
    assert limiter.is_allowed("a") is False
    # b is unaffected
    assert limiter.is_allowed("b") is True


def test_rate_limiter_reset():
    limiter = RateLimiter(rpm=1)
    limiter.is_allowed("tenant-c")
    assert limiter.is_allowed("tenant-c") is False
    limiter.reset("tenant-c")
    assert limiter.is_allowed("tenant-c") is True


def test_quota_tracker_allows_within_budget():
    tracker = QuotaTracker(daily_budget_usd=1.0)
    assert tracker.check_budget("t1") is True


def test_quota_tracker_blocks_over_budget():
    tracker = QuotaTracker(daily_budget_usd=0.01)
    tracker.deduct("t1", 0.01)
    assert tracker.check_budget("t1") is False


def test_quota_tracker_get_remaining():
    tracker = QuotaTracker(daily_budget_usd=10.0)
    tracker.deduct("t2", 3.0)
    assert abs(tracker.get_remaining("t2") - 7.0) < 1e-9


def test_quota_tracker_remaining_never_negative():
    tracker = QuotaTracker(daily_budget_usd=1.0)
    tracker.deduct("t3", 5.0)  # overspend
    assert tracker.get_remaining("t3") == 0.0
```

---

## `tests/test_api.py`

```python
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, AsyncMock

from gateway.api.main import app, rate_limiter, quota_tracker, cache


@pytest.fixture(autouse=True)
def reset_state():
    """Reset all in-memory state between tests."""
    rate_limiter.reset_all()
    quota_tracker.reset("tenant-test")
    quota_tracker.reset("tenant-rl")
    cache.clear()
    yield
    rate_limiter.reset_all()
    cache.clear()


@pytest.mark.asyncio
async def test_health_returns_ok():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_complete_fast_tier():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/complete",
            json={"model_tier": "fast", "messages": [{"role": "user", "content": "hello"}]},
            headers={"x-tenant-id": "tenant-test"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "claude-haiku-5-5"
    assert data["cost_usd"] > 0
    assert data["cache_hit"] is False


@pytest.mark.asyncio
async def test_complete_cache_hit_on_second_request():
    payload = {"model_tier": "smart", "messages": [{"role": "user", "content": "cached?"}]}
    headers = {"x-tenant-id": "tenant-test"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        first = await client.post("/v1/complete", json=payload, headers=headers)
        second = await client.post("/v1/complete", json=payload, headers=headers)

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["cache_hit"] is True
    assert second.json()["cost_usd"] == 0.0


@pytest.mark.asyncio
async def test_rate_limit_returns_429():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # patch rpm to 1 for this test
        rate_limiter.rpm = 1
        await client.post(
            "/v1/complete",
            json={"model_tier": "fast", "messages": [{"role": "user", "content": "first"}]},
            headers={"x-tenant-id": "tenant-rl"},
        )
        resp = await client.post(
            "/v1/complete",
            json={"model_tier": "fast", "messages": [{"role": "user", "content": "second"}]},
            headers={"x-tenant-id": "tenant-rl"},
        )
        rate_limiter.rpm = 60  # restore default
    assert resp.status_code == 429


@pytest.mark.asyncio
async def test_budget_exceeded_returns_402():
    quota_tracker.daily_budget_usd = 0.0
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/complete",
            json={"model_tier": "fast", "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-tenant-id": "tenant-test"},
        )
    quota_tracker.daily_budget_usd = 10.0  # restore
    assert resp.status_code == 402


@pytest.mark.asyncio
async def test_missing_tenant_header_returns_400():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/complete",
            json={"model_tier": "fast", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 400
    assert "X-Tenant-ID" in resp.json()["error"]


@pytest.mark.asyncio
async def test_unknown_tier_returns_400():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/complete",
            json={"model_tier": "turbo", "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-tenant-id": "tenant-test"},
        )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_tenant_usage_endpoint():
    quota_tracker.deduct("tenant-test", 1.50, tokens=1000)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            "/v1/tenants/tenant-test/usage",
            headers={"x-tenant-id": "tenant-test"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["daily_cost_usd"] == pytest.approx(1.50, abs=1e-6)
    assert data["budget_remaining_usd"] == pytest.approx(8.50, abs=1e-6)
```

---

## `.env.example`

```
ANTHROPIC_API_KEY=your-key-here
LOG_LEVEL=INFO
RATE_LIMIT_RPM=60
DAILY_BUDGET_USD=10.0
```

---

## Running the gateway locally

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v --cov=gateway

# Start the server (uses MockProvider by default)
LOG_LEVEL=DEBUG uvicorn gateway.api.main:app --reload

# Example request
curl -X POST http://localhost:8000/v1/complete \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: acme-corp" \
  -d '{"model_tier": "smart", "messages": [{"role": "user", "content": "What is 2+2?"}]}'
```

---

## Key design decisions

| Decision | Rationale |
|---|---|
| `BaseProvider` ABC | Enforces interface contract; enables MockProvider in tests without patching SDK |
| `FallbackChain` takes `(provider_name, model)` pairs | Provider and model are decoupled — can fall back to a different model on the same provider, or to a different provider |
| `ExactCache` uses SHA-256 of canonical JSON | Deterministic, collision-resistant, model is included in key so sonnet and haiku requests for same prompt are cached separately |
| `RateLimiter` uses sliding window | Sliding window avoids the "reset cliff" problem of fixed-window limiters (burst at window boundary) |
| `QuotaTracker` is in-memory | Correct for a single instance; in multi-instance deployments, replace with Redis INCRBYFLOAT atomic operations |
| `TenantMiddleware` exempts `/health` | Health checks (load balancer probes) must not require tenant headers |
| All costs via `COST_OUT_PER_1K` dict | Single source of truth; update prices here when Anthropic changes pricing |
