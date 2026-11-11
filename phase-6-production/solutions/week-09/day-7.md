# Solutions — Day 7: Integration Patterns

## Exercise 1: FastAPI Tenant Middleware

Middleware that extracts `X-Tenant-ID` from request headers and rejects requests missing it (except health/docs routes).

```python
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)


class TenantMiddleware(BaseHTTPMiddleware):
    """
    Require X-Tenant-ID header on all API routes.
    Sets request.state.tenant_id for downstream handlers.
    """

    EXEMPT_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        tenant_id = request.headers.get("X-Tenant-ID")
        if not tenant_id:
            logger.warning("missing X-Tenant-ID path=%s", request.url.path)
            return JSONResponse(
                status_code=400,
                content={"error": "X-Tenant-ID header required"},
            )

        request.state.tenant_id = tenant_id
        logger.info(
            "request path=%s tenant=%s method=%s",
            request.url.path,
            tenant_id,
            request.method,
        )
        return await call_next(request)
```

**Registration:**
```python
from fastapi import FastAPI

app = FastAPI()
app.add_middleware(TenantMiddleware)
```

**Accessing tenant in a route handler:**
```python
from fastapi import Request

@app.post("/v1/complete")
async def complete(request: Request, body: CompletionBody):
    tenant_id = request.state.tenant_id  # set by middleware
    ...
```

**Test:**
```python
import pytest
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

app = FastAPI()
app.add_middleware(TenantMiddleware)

@app.get("/v1/ping")
async def ping(request: Request):
    return PlainTextResponse(f"tenant={request.state.tenant_id}")

@pytest.mark.asyncio
async def test_missing_tenant_id_rejected():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/v1/ping")
    assert resp.status_code == 400
    assert "X-Tenant-ID" in resp.json()["error"]

@pytest.mark.asyncio
async def test_tenant_id_set_on_state():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/v1/ping", headers={"X-Tenant-ID": "acme"})
    assert resp.status_code == 200
    assert "tenant=acme" in resp.text

@pytest.mark.asyncio
async def test_health_exempt():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    # 404 (route not defined) — not 400. Middleware didn't block it.
    assert resp.status_code != 400
```

---

## Exercise 2: Health Endpoint with Provider Availability Check

```python
from fastapi import APIRouter, Depends
from pydantic import BaseModel
import asyncio
import time
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


class ProviderStatus(BaseModel):
    name: str
    status: str  # "ok" | "degraded" | "down"
    latency_ms: float | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
    providers: list[ProviderStatus]


async def _ping_provider(name: str, provider: "BaseProvider") -> ProviderStatus:
    """Ping a provider with a minimal 1-token request and measure latency."""
    try:
        start = time.monotonic()
        await provider.complete(
            CompletionRequest(
                model=provider.list_models()[0],
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0.0,
            )
        )
        latency_ms = (time.monotonic() - start) * 1000
        return ProviderStatus(name=name, status="ok", latency_ms=round(latency_ms, 1))
    except Exception as exc:
        logger.warning("provider %s health check failed: %s", name, exc)
        return ProviderStatus(name=name, status="down", error=str(exc))


@router.get("/health", response_model=HealthResponse)
async def health(registry: "ProviderRegistry" = Depends(get_registry)):
    """
    Check all registered providers in parallel.
    Returns 200 with status="degraded" if any provider is down —
    the gateway itself is up even when a provider is degraded.
    """
    tasks = [
        _ping_provider(name, registry.get(name))
        for name in registry.list_providers()
    ]
    statuses: list[ProviderStatus] = await asyncio.gather(*tasks)

    overall = "ok" if all(s.status == "ok" for s in statuses) else "degraded"
    return HealthResponse(status=overall, providers=statuses)
```

**Design notes:**
- `asyncio.gather` pings all providers in parallel — no serial blocking.
- Returns HTTP 200 even when `status="degraded"` so load balancer health checks don't pull the gateway pod. Add `status_code=503` if you prefer the LB to stop routing to degraded instances.
- `max_tokens=1` minimises cost of health-check calls. Some providers charge for health checks; consider a mock health endpoint on the provider adapter instead for high-frequency checks.

**Provider-adapter pattern (skip real API calls in health):**
```python
class AnthropicProvider(BaseProvider):
    async def ping(self) -> bool:
        """Lightweight check: verify API key is valid without a completion call."""
        # anthropic.models.list() is cheaper than a completion
        try:
            self._client.models.list()
            return True
        except Exception:
            return False
```

---

## Exercise 3: `GatewayConfig` with pydantic-settings

```python
from pydantic import Field
from pydantic_settings import BaseSettings


class GatewayConfig(BaseSettings):
    """
    Gateway configuration loaded from environment variables and .env file.
    All secrets come from env vars — never hardcode.
    """

    # Provider credentials
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    # Routing
    default_model: str = "claude-sonnet-5-5"
    fallback_model: str = "claude-haiku-5-5"

    # Caching
    cache_ttl_seconds: int = 300

    # Rate limiting
    rate_limit_rpm: int = 60          # requests per minute per tenant
    rate_limit_burst: int = 10        # token bucket burst capacity

    # Budget
    daily_budget_usd: float = 10.0   # per tenant

    # Observability
    log_level: str = "INFO"
    enable_request_logging: bool = True

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "populate_by_name": True,   # accept both alias and field name
    }


def get_config() -> GatewayConfig:
    """Factory — creates once, reuse the instance in production."""
    return GatewayConfig()
```

**Usage:**
```python
config = get_config()
print(config.default_model)       # "claude-sonnet-5-5"
print(config.anthropic_api_key)   # loaded from ANTHROPIC_API_KEY env var
```

**FastAPI dependency injection:**
```python
from functools import lru_cache
from fastapi import Depends

@lru_cache(maxsize=1)
def get_config() -> GatewayConfig:
    return GatewayConfig()

@app.get("/debug/config")
async def debug_config(config: GatewayConfig = Depends(get_config)):
    # NEVER expose secrets — return only non-sensitive fields
    return {
        "default_model": config.default_model,
        "cache_ttl_seconds": config.cache_ttl_seconds,
        "rate_limit_rpm": config.rate_limit_rpm,
        "log_level": config.log_level,
    }
```

**`.env.example`** (always commit this, never `.env`):
```
ANTHROPIC_API_KEY=your-key-here
LOG_LEVEL=INFO
DAILY_BUDGET_USD=10.0
RATE_LIMIT_RPM=60
CACHE_TTL_SECONDS=300
```

**Test (no real env required):**
```python
import pytest
from gateway.config import GatewayConfig

def test_config_defaults():
    config = GatewayConfig(anthropic_api_key="test-key")
    assert config.default_model == "claude-sonnet-5-5"
    assert config.cache_ttl_seconds == 300
    assert config.daily_budget_usd == 10.0

def test_config_override_via_env(monkeypatch):
    monkeypatch.setenv("DAILY_BUDGET_USD", "99.0")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    config = GatewayConfig()
    assert config.daily_budget_usd == 99.0

def test_config_missing_api_key_defaults_empty():
    config = GatewayConfig()  # no env var set in test isolation
    assert config.anthropic_api_key == ""
```
