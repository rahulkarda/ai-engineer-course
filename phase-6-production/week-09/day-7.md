# Phase 6 · Week 09 · Day 7 — Week Review + Integration Patterns

> Objective: Wire all gateway components into a cohesive FastAPI service and trace the full request lifecycle end-to-end.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]], [[day-2]], [[day-3]], [[day-4]], [[day-5]], [[day-6]]

---

## Concept

Individual components — rate limiter, cache, router, cost attributor, logger — are useless in isolation. Production value comes from composition: a single request flowing through every layer in the correct order, with clean handoffs and consistent error semantics.

Today maps the request lifecycle, then shows how FastAPI middleware and dependency injection connect the components you built.

---

## How it works internally

### Full gateway request lifecycle

```
Incoming HTTP request
        │
        ▼
 [1] Middleware: extract X-Tenant-ID, X-User-ID headers
        │
        ▼
 [2] RateLimiter.check(tenant_id)
     → 429 Too Many Requests if exceeded
        │
        ▼
 [3] QuotaTracker.check(tenant_id)
     → 402 Payment Required if budget exhausted
        │
        ▼
 [4] ExactCache.lookup(prompt_hash)
     → 200 (cache hit) or continue
        │
        ▼
 [5] ModelRouter.select(policy) → (provider, model)
        │
        ▼
 [6] ProviderBase.complete(model, messages)
     → if error: FallbackChain.next() → retry step 6
        │
        ▼
 [7] ExactCache.store(prompt_hash, response)
        │
        ▼
 [8] CostAttributor.record(CostRecord)
        │
        ▼
 [9] Logger.info(LogEntry)
        │
        ▼
 [10] Return HTTP 200 with CompletionResponse
```

Order matters. The cache lookup (step 4) must come **after** auth/rate checks (steps 2–3) to prevent unauthenticated cache reads. Cost recording (step 8) must come **after** the provider call so token counts are known.

### FastAPI middleware for tenant extraction

FastAPI middleware runs before routing — ideal for cross-cutting concerns like header extraction and request ID generation:

```python
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class TenantMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        tenant_id = request.headers.get("X-Tenant-ID", "anonymous")
        user_id   = request.headers.get("X-User-ID", "anonymous")
        request_id = str(uuid.uuid4())

        # Attach to request state — accessible in route handlers
        request.state.tenant_id  = tenant_id
        request.state.user_id    = user_id
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
```

Route handlers access these via `request.state.tenant_id` or via `Header(...)` parameter declarations.

### GatewayConfig — config-driven initialization

All runtime parameters come from environment variables, not hardcoded values:

```python
from pydantic_settings import BaseSettings
from pydantic import Field

class GatewayConfig(BaseSettings):
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    rate_limit_rpm: int = Field(default=60, alias="RATE_LIMIT_RPM")
    quota_daily_usd: float = Field(default=10.0, alias="QUOTA_DAILY_USD")
    cache_ttl_seconds: int = Field(default=300, alias="CACHE_TTL_SECONDS")
    default_model: str = Field(default="claude-haiku-5-5", alias="DEFAULT_MODEL")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    model_config = {"populate_by_name": True}
```

`pydantic-settings` reads from environment automatically — `os.getenv()` is not needed when using this pattern.

### Deployment topology

```
┌─────────────────────────────────────────────┐
│  Client (any HTTP client)                   │
└────────────────┬────────────────────────────┘
                 │ HTTP
                 ▼
┌─────────────────────────────────────────────┐
│  Gateway Pod (stateless FastAPI)            │
│  - TenantMiddleware                         │
│  - Rate limiter (reads/writes Redis)        │
│  - Quota tracker (reads/writes Redis)       │
│  - Exact cache (reads/writes Redis)         │
│  - ModelRouter + FallbackChain              │
│  - CostAttributor (writes to Redis stream)  │
│  - Structured logger (stdout → log sink)    │
└────────────┬──────────────┬─────────────────┘
             │              │
             ▼              ▼
     ┌──────────────┐  ┌──────────────┐
     │  Redis       │  │  Anthropic   │
     │  (cache +    │  │  API         │
     │   quota +    │  └──────────────┘
     │   rate limit)│
     └──────────────┘
```

Stateless gateway pods scale horizontally. Redis is the shared state layer — all pods see the same rate limit counters, cache entries, and quota balances.

---

## Worked example

Minimal FastAPI app wiring all components together:

```python
# gateway/api/app.py
from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("gateway")


# ── Config ────────────────────────────────────────────────────────

class GatewayConfig(BaseSettings):
    anthropic_api_key: str = ""
    rate_limit_rpm: int = 60
    quota_daily_usd: float = 10.0
    cache_ttl_seconds: int = 300
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", populate_by_name=True)


# ── Request / Response schemas ────────────────────────────────────

class CompletionRequest(BaseModel):
    messages: list[dict[str, str]]
    model_tier: str = "fast"   # "fast" | "smart" | "best"
    feature_tag: str = "default"


class CompletionResponse(BaseModel):
    request_id: str
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cached: bool = False


# ── Simplified in-process stubs (replace with real impls from days 1-5) ──

class _RateLimiter:
    """Token-bucket per tenant (in-memory stub)."""
    def __init__(self, rpm: int) -> None:
        self._rpm = rpm
        self._counts: dict[str, list[float]] = {}

    def check(self, tenant_id: str) -> bool:
        now = time.time()
        window = self._counts.setdefault(tenant_id, [])
        self._counts[tenant_id] = [t for t in window if now - t < 60]
        if len(self._counts[tenant_id]) >= self._rpm:
            return False
        self._counts[tenant_id].append(now)
        return True


_TIER_MODEL = {
    "fast": "claude-haiku-5-5",
    "smart": "claude-sonnet-5-5",
    "best": "claude-opus-5-5",
}

_COST_PER_TOKEN = 0.000_000_25  # flat stub — use real pricing in production


# ── Lifespan: initialise shared resources ─────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    config = GatewayConfig()
    logging.basicConfig(level=config.log_level)
    app.state.config = config
    app.state.rate_limiter = _RateLimiter(config.rate_limit_rpm)
    logger.info("Gateway started", extra={"rate_limit_rpm": config.rate_limit_rpm})
    yield
    logger.info("Gateway shutting down")


# ── Middleware ────────────────────────────────────────────────────

class TenantMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request.state.tenant_id  = request.headers.get("X-Tenant-ID", "anonymous")
        request.state.user_id    = request.headers.get("X-User-ID", "anonymous")
        request.state.request_id = str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response


# ── App ───────────────────────────────────────────────────────────

app = FastAPI(title="Model Gateway", version="0.1.0", lifespan=lifespan)
app.add_middleware(TenantMiddleware)


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """Liveness + provider connectivity status."""
    providers: dict[str, str] = {"anthropic": "ok", "mock": "ok"}
    api_key = request.app.state.config.anthropic_api_key
    if not api_key:
        providers["anthropic"] = "no_key_configured"
    return {"status": "ok", "providers": providers}


@app.post("/v1/complete", response_model=CompletionResponse)
async def complete(
    body: CompletionRequest,
    request: Request,
    x_tenant_id: str = Header(default="anonymous"),
    x_user_id: str = Header(default="anonymous"),
) -> CompletionResponse:
    tenant_id  = request.state.tenant_id
    request_id = request.state.request_id
    t_start    = time.perf_counter()

    # [1] Rate limit
    if not request.app.state.rate_limiter.check(tenant_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded for tenant {tenant_id}",
        )

    # [2] Route
    model = _TIER_MODEL.get(body.model_tier, "claude-haiku-5-5")

    # [3] Provider call (real implementation uses ProviderBase + FallbackChain)
    # Stub response for illustration:
    content       = f"[stub response from {model}]"
    input_tokens  = sum(len(m["content"].split()) for m in body.messages)
    output_tokens = 42
    cost_usd      = (input_tokens + output_tokens) * _COST_PER_TOKEN

    latency_ms = (time.perf_counter() - t_start) * 1000

    # [4] Structured log
    logger.info(
        "request_complete",
        extra={
            "request_id":   request_id,
            "tenant_id":    tenant_id,
            "model":        model,
            "latency_ms":   round(latency_ms, 1),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd":     cost_usd,
        },
    )

    return CompletionResponse(
        request_id=request_id,
        content=content,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
    )
```

Run locally:
```bash
uvicorn gateway.api.app:app --reload --port 8080
```

Test with curl:
```bash
curl -s -X POST http://localhost:8080/v1/complete \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: acme" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"model_tier":"fast"}' | python -m json.tool
```

---

## Exercises

1. **Health endpoint hardening**: extend the `/health` endpoint to actually check provider connectivity by attempting a minimal API call (e.g., list models or a 1-token completion). Return `{"status": "degraded", "providers": {"anthropic": "error: timeout"}}` if the check fails. Write a pytest test that mocks the provider check and verifies `degraded` status is returned correctly.

2. **Tenant usage endpoint**: implement `GET /v1/tenants/{tenant_id}/usage` that returns `{tenant_id, total_cost_usd, request_count, breakdown_by_feature: {...}}`. Wire it to `CostAttributor` from day 5. Write a test that pre-populates `CostAttributor` with three records and asserts correct aggregation.

3. **Middleware test**: write a pytest test using FastAPI's `TestClient` that sends a request without `X-Tenant-ID` header and asserts: (a) the request succeeds (defaults to `"anonymous"`), and (b) the response contains an `X-Request-ID` header with a valid UUID.

(Solutions: ../solutions/week-09/day-7.md)

---

## Self-check

1. The lifecycle places the cache lookup (step 4) **after** rate limiting (step 2). What security problem would occur if you swapped the order?

2. `GatewayConfig` uses `pydantic-settings` with `env_file=".env"`. In a Kubernetes deployment there is no `.env` file — values come from environment variables injected by the pod spec. Does the app still work? Why?

3. Name two things that must happen to make the in-memory `_RateLimiter` work correctly when the gateway scales to three pods behind a load balancer — without any code changes to the rate limiter class itself.
