# Week 09 Project — Production Model Gateway Service

> Time: ~4 hr (Sat) + 3 hr (Sun)  ·  Builds on: days 1–7

---

## Overview

Build a complete, production-quality model gateway service exposing a FastAPI REST API. The gateway abstracts LLM providers behind a unified interface, enforces per-tenant rate limits and quotas, caches responses, routes by policy, records costs, and logs structured audit events — all wired together as a deployable Python package.

By the end of the weekend you will have a service you could realistically deploy as a sidecar or internal platform API.

---

## Deliverables

```
gateway/
├── __init__.py
├── providers/
│   ├── __init__.py
│   ├── base.py           # ProviderBase ABC
│   ├── anthropic.py      # AnthropicProvider (calls Anthropic API)
│   └── mock.py           # MockProvider (for tests; no HTTP calls)
├── routing/
│   ├── __init__.py
│   ├── router.py         # ModelRouter (CHEAPEST/FASTEST/BEST/AUTO policies)
│   ├── fallback.py       # FallbackChain
│   └── ab_router.py      # ModelABRouter (deterministic hash split)
├── caching/
│   ├── __init__.py
│   └── exact_cache.py    # ExactCache (in-memory, TTL=300s)
├── quota/
│   ├── __init__.py
│   ├── rate_limiter.py   # RateLimiter (token bucket, 60 rpm per tenant)
│   └── quota_tracker.py  # QuotaTracker (per-tenant daily USD budget)
├── logging/
│   ├── __init__.py
│   ├── models.py         # LogEntry, CostRecord Pydantic models
│   └── attributor.py     # CostAttributor
└── api/
    ├── __init__.py
    ├── app.py            # FastAPI app, lifespan, middleware
    ├── routes.py         # /v1/complete, /health, /v1/tenants/{id}/usage
    └── schemas.py        # CompletionRequest, CompletionResponse, UsageResponse

tests/
├── unit/
│   ├── test_providers.py
│   ├── test_router.py
│   ├── test_fallback.py
│   ├── test_ab_router.py
│   ├── test_cache.py
│   ├── test_rate_limiter.py
│   ├── test_quota_tracker.py
│   └── test_attributor.py
└── integration/
    └── test_api.py       # FastAPI TestClient tests, all providers mocked

pyproject.toml
.env.example
.gitignore
```

---

## Architecture diagram (ASCII)

```
┌──────────────────────────────────────────────────────────────────────┐
│  Client (curl / SDK / another service)                               │
└───────────────────────────┬──────────────────────────────────────────┘
                            │  POST /v1/complete
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  FastAPI App                                                         │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  TenantMiddleware                                           │    │
│  │  Extracts X-Tenant-ID, X-User-ID → request.state           │    │
│  │  Stamps X-Request-ID on response                           │    │
│  └─────────────────────────┬───────────────────────────────────┘    │
│                            │                                         │
│  ┌─────────────────────────▼───────────────────────────────────┐    │
│  │  RateLimiter.check(tenant_id)                               │    │
│  │  Token bucket · 60 rpm per tenant                           │    │
│  │  → HTTP 429 if exceeded                                     │    │
│  └─────────────────────────┬───────────────────────────────────┘    │
│                            │                                         │
│  ┌─────────────────────────▼───────────────────────────────────┐    │
│  │  QuotaTracker.check(tenant_id)                              │    │
│  │  Per-tenant daily USD budget                                │    │
│  │  → HTTP 402 if budget exhausted                             │    │
│  └─────────────────────────┬───────────────────────────────────┘    │
│                            │                                         │
│  ┌─────────────────────────▼───────────────────────────────────┐    │
│  │  ExactCache.lookup(prompt_hash)                             │    │
│  │  SHA-256 of (model + messages) · TTL=300s                   │    │
│  │  → return cached response (HTTP 200) or continue            │    │
│  └─────────────────────────┬───────────────────────────────────┘    │
│                            │  cache miss                             │
│  ┌─────────────────────────▼───────────────────────────────────┐    │
│  │  ModelRouter.select(policy)          ABRouter.assign(user)  │    │
│  │  CHEAPEST / FASTEST / BEST / AUTO    deterministic hash      │    │
│  │  → (provider, model)                                        │    │
│  └─────────────────────────┬───────────────────────────────────┘    │
│                            │                                         │
│  ┌─────────────────────────▼───────────────────────────────────┐    │
│  │  FallbackChain                                              │    │
│  │  opus-5-5 → sonnet-5-5 → haiku-5-5                         │    │
│  │  Retries next on ProviderError                              │    │
│  └──────────┬─────────────────────────────┬────────────────────┘    │
│             │                             │                          │
│  ┌──────────▼──────────┐     ┌────────────▼────────────┐            │
│  │  AnthropicProvider  │     │  MockProvider           │            │
│  │  os.getenv(         │     │  (tests only)           │            │
│  │  ANTHROPIC_API_KEY) │     └─────────────────────────┘            │
│  └──────────┬──────────┘                                            │
│             │  response + token counts                               │
│  ┌──────────▼──────────────────────────────────────────────────┐    │
│  │  ExactCache.store(prompt_hash, response)                    │    │
│  └──────────┬──────────────────────────────────────────────────┘    │
│             │                                                        │
│  ┌──────────▼──────────────────────────────────────────────────┐    │
│  │  CostAttributor.record(CostRecord)                          │    │
│  │  Accumulates cost per (tenant_id, feature_tag)              │    │
│  └──────────┬──────────────────────────────────────────────────┘    │
│             │                                                        │
│  ┌──────────▼──────────────────────────────────────────────────┐    │
│  │  Logger.info(LogEntry)                                      │    │
│  │  Structured JSON · no message content · no PII              │    │
│  └──────────┬──────────────────────────────────────────────────┘    │
│             │                                                        │
└─────────────┼────────────────────────────────────────────────────────┘
              │
              ▼
    HTTP 200 CompletionResponse
```

---

## Step-by-step build guide

### Step 1 — Scaffold the package (30 min)

Create the directory structure above. Start with `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "model-gateway"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.111",
    "uvicorn[standard]>=0.29",
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "anthropic>=0.25",
    "httpx>=0.27",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-asyncio>=0.23", "httpx>=0.27"]
```

Create `.env.example`:
```
ANTHROPIC_API_KEY=sk-ant-...
RATE_LIMIT_RPM=60
QUOTA_DAILY_USD=10.0
CACHE_TTL_SECONDS=300
LOG_LEVEL=INFO
```

Ensure `.env` is in `.gitignore`.

### Step 2 — Providers (45 min)

Implement `ProviderBase` as an abstract class in `providers/base.py`. Define at minimum:

```python
from abc import ABC, abstractmethod
from pydantic import BaseModel

class ProviderResponse(BaseModel):
    content: str
    model: str
    input_tokens: int
    output_tokens: int

class ProviderBase(ABC):
    @abstractmethod
    async def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
    ) -> ProviderResponse: ...

    @property
    @abstractmethod
    def name(self) -> str: ...
```

Implement `AnthropicProvider` using `os.getenv("ANTHROPIC_API_KEY")`. Implement `MockProvider` that returns a configurable stub response with no HTTP calls.

Write `tests/unit/test_providers.py` — test `MockProvider` directly; test `AnthropicProvider` with `unittest.mock.AsyncMock` patching the `anthropic.AsyncAnthropic` client.

### Step 3 — Routing + Fallback (45 min)

`ModelRouter` maps a routing policy enum to a `(provider_name, model)` tuple. Policies:

- `CHEAPEST` → `claude-haiku-5-5`
- `FASTEST` → `claude-haiku-5-5`
- `BEST` → `claude-opus-5-5`
- `AUTO` → `claude-sonnet-5-5`

`FallbackChain` accepts an ordered list of `(provider, model)` pairs. On `ProviderError`, it advances to the next pair. Raises `AllProvidersFailedError` when the chain is exhausted.

Default chain: `anthropic/claude-opus-5-5` → `anthropic/claude-sonnet-5-5` → `anthropic/claude-haiku-5-5`.

Write unit tests: verify fallback advances on error, exhausted chain raises, successful first call does not advance.

### Step 4 — Cache + Rate Limit + Quota (45 min)

**ExactCache**: SHA-256 hash of `json.dumps({"model": model, "messages": messages}, sort_keys=True)` as key. Store response with TTL. Use `time.monotonic()` for TTL tracking.

**RateLimiter**: sliding-window token bucket. Store timestamps per `tenant_id`. On each `check()`, evict timestamps older than 60 s, count remaining, reject if ≥ `rpm_limit`.

**QuotaTracker**: per-tenant daily spend. `check(tenant_id, estimated_cost)` returns `True` if remaining budget allows. `record(tenant_id, actual_cost)` deducts from balance. Reset at midnight UTC.

Write unit tests for all three — use `unittest.mock.patch` on `time.monotonic` and `datetime.now` where needed.

### Step 5 — Cost Attribution + A/B Router (30 min)

Port `CostRecord`, `CostAttributor`, `ABVariant`, `ABExperiment`, and `ModelABRouter` from day-5 worked example into the appropriate modules. No changes needed — just placement and imports.

Write or move the day-5 tests into `tests/unit/test_attributor.py` and `tests/unit/test_ab_router.py`.

### Step 6 — FastAPI wiring (60 min)

In `api/app.py`:

- Create `FastAPI` app with `lifespan` context manager that initialises all components and attaches them to `app.state`
- Register `TenantMiddleware`
- Import routes from `api/routes.py`

Implement three routes in `api/routes.py`:

```
POST /v1/complete        — full lifecycle (steps 1–10 from day-7)
GET  /health             — provider ping + config summary
GET  /v1/tenants/{tenant_id}/usage  — CostAttributor.aggregate(tenant_id)
```

Use `CompletionRequest` / `CompletionResponse` / `UsageResponse` Pydantic models from `api/schemas.py`.

### Step 7 — Integration tests + final check (45 min)

Write `tests/integration/test_api.py` using FastAPI `TestClient`. Override `app.state` providers with `MockProvider` instances using pytest fixtures. Test:

- Normal request returns 200 with correct model
- Rate limit: 11th request returns 429
- Fallback: first provider raises, second succeeds, response is 200
- `/health` returns 200
- `/v1/tenants/{id}/usage` returns correct aggregated cost

Run full test suite:
```bash
pytest tests/ -v --tb=short
```

Target: ≥80% coverage (`pytest-cov`). Check with:
```bash
pytest tests/ --cov=gateway --cov-report=term-missing
```

---

## Acceptance criteria

- [ ] All pytest tests pass; zero real HTTP calls to any LLM provider in the test suite
- [ ] `GET /health` returns HTTP 200 with `{"status": "ok", ...}`
- [ ] Sending 11 identical requests with the same `X-Tenant-ID` in under 60 s: first 10 return 200, 11th returns 429
- [ ] Per-tenant cost is tracked and returned by `GET /v1/tenants/{tenant_id}/usage`
- [ ] FallbackChain advances to next model when `AnthropicProvider` raises `ProviderError`; response still returns 200 with fallback model name

---

## Stretch goals

- **Semantic cache**: add `SemanticCache` using sentence-transformers embeddings; return cached response when cosine similarity > 0.95; wire as optional replacement for `ExactCache` via config flag
- **Redis backend**: replace in-memory dicts in `RateLimiter`, `ExactCache`, and `QuotaTracker` with Redis via `redis.asyncio`; add `REDIS_URL` to `GatewayConfig`
- **Prometheus metrics**: add `GET /metrics` endpoint (Prometheus exposition format) exposing: `gateway_requests_total{tenant, model, status}`, `gateway_latency_seconds{model}`, `gateway_cost_usd_total{tenant}`
- **Streaming responses**: add `POST /v1/complete/stream` that yields `text/event-stream` using FastAPI `StreamingResponse` and `anthropic.AsyncAnthropic.messages.stream()`
