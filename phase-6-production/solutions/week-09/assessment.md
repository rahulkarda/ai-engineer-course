# Solutions — Week 9 Assessment

## Part 1: Conceptual Answers

### Q1: Why does provider abstraction matter? What interface methods are required?

Provider abstraction solves **vendor lock-in** and enables **multi-provider routing** without the caller knowing which LLM backend is in use. By hiding the Anthropic SDK (or any SDK) behind an interface, you can:

- Swap providers without changing any business logic.
- Route to different providers based on cost, latency, or capability.
- Inject mock providers in tests without patching SDK internals.
- Add observability (latency tracking, cost recording) at a single chokepoint.

**Required interface methods:**

```python
from abc import ABC, abstractmethod
from gateway.providers.base import CompletionRequest, CompletionResponse

class BaseProvider(ABC):
    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Execute a chat completion. Raises ProviderError on failure."""
        ...

    @abstractmethod
    def provider_name(self) -> str:
        """Unique identifier, e.g. 'anthropic', 'openai'. Used by registry."""
        ...

    @abstractmethod
    def list_models(self) -> list[str]:
        """Models this provider can serve, e.g. ['claude-haiku-5-5', ...]."""
        ...
```

---

### Q2: Three routing dimensions — when to use each

| Dimension | Primary signal | When to choose |
|---|---|---|
| **Cost** | Output token price per model | Bulk tasks, low-stakes generation, high volume (FAQ bots, batch summarization). Use cheapest model that meets quality bar. |
| **Latency** | Median/p99 time-to-first-token | Real-time UX, streaming chat, interactive features. Haiku-class models (claude-haiku-5-5) are fastest. |
| **Capability** | Model benchmark scores, context window | Complex reasoning, multi-step code generation, tasks that require deep understanding. Use most powerful model (claude-opus-5-5) only when cheaper models provably fail. |

**Decision heuristic:**
- Default to **cost** routing — start cheap, upgrade only when quality drops.
- Switch to **latency** routing when user-facing p99 > 3s.
- Switch to **capability** routing for structured extraction, long-doc reasoning, or agentic tasks where errors are expensive.

---

### Q3: Token bucket algorithm — parameters and mechanics

**How it works:**

A bucket holds tokens up to `capacity`. Every second, `refill_rate` tokens are added (up to `capacity`). Each request consumes 1 token. If the bucket is empty, the request is rejected.

```
capacity     → max burst: how many requests can arrive in a spike
refill_rate  → sustained rate: tokens/second = rpm / 60
```

**Example:** 60 rpm = `refill_rate = 1.0 token/second`, `capacity = 10` (allow a burst of 10 before throttling).

```python
import time

class TokenBucket:
    def __init__(self, capacity: float, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens per second
        self._tokens = capacity
        self._last_refill = time.monotonic()

    def is_allowed(self) -> bool:
        self._refill()
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)
        self._last_refill = now
```

**Two parameters:**
- `capacity` — burst limit (short spike tolerance)
- `refill_rate` — sustained rate limit (`rpm / 60`)

---

### Q4: Exact vs. semantic caching — keys, applicability, trade-offs

**Exact cache:**

- **Key:** `SHA-256(canonical_json(sorted(messages)) + model)`
- **When it applies:** Identical requests — same messages in same order, same model. Common in FAQ bots, fixed-prompt pipelines, retry patterns.
- **Complexity:** O(1) hash lookup.

```python
import hashlib, json

def make_exact_key(messages: list[dict], model: str) -> str:
    payload = json.dumps({"messages": messages, "model": model}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()
```

**Semantic cache:**

- **Key:** Embedding vector of the user query, stored in a vector DB (e.g., Redis with `redis-semantic` backend).
- **Lookup:** Cosine similarity search — if nearest cached query exceeds threshold (e.g., 0.95), return cached response.
- **When it applies:** Paraphrased questions with the same answer — "What's the capital of France?" ≈ "Tell me France's capital city."
- **Complexity:** O(embedding time) + O(ANN search) — typically 10–50 ms overhead.

**Trade-offs:**

| | Exact | Semantic |
|---|---|---|
| Hit rate | Low (identical only) | Higher (near-duplicates) |
| Lookup cost | O(1), microseconds | Embedding + vector search, ~10-50 ms |
| False positives | None — mathematically exact | Possible if threshold is too low |
| Infrastructure | In-memory dict or Redis string | Vector DB required |
| Correctness risk | Zero | Low — tune threshold carefully for factual queries |

---

### Q5: Why must A/B variant assignment be deterministic?

**User experience reason:** Users expect consistent behavior. If a user assigned to "sonnet" gets "haiku" on their next request, they experience different quality/speed and cannot reason about the product.

**Statistical validity reason:** A/B testing measures the effect of a treatment on the same user over time. If a user switches variants between sessions, their responses are attributed to both variants — contaminating both groups and invalidating the experiment's statistical assumptions.

**How to achieve determinism:**

```python
import hashlib

def assign_variant(user_id: str, experiment_id: str) -> int:
    """Returns a stable bucket 0–999 for this user/experiment pair."""
    key = f"{experiment_id}:{user_id}".encode()
    digest = int(hashlib.sha256(key).hexdigest(), 16)
    return digest % 1000
```

- SHA-256 is a pure function: same input → same output, always.
- No database lookup, no clock dependency, no mutable state.
- The `experiment_id` salt ensures the same user gets different assignments in different experiments (prevents correlation between concurrent experiments).

---

## Part 2: Coding Challenge Solution

### Full wired FastAPI gateway with tier routing, rate limiting, fallback, and cost tracking

```python
# gateway/api/chat.py
import os
import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


# --- Data models ---

class ChatRequest(BaseModel):
    model_tier: str  # "fast" | "smart" | "best"
    messages: list[dict[str, str]]


class ChatResponse(BaseModel):
    content: str
    model: str
    cost_usd: float
    latency_ms: float


# --- Configuration ---

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

FALLBACK: dict[str, str] = {
    "best": "smart",
    "smart": "fast",
}

RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "10"))


# --- In-memory rate limiter (token bucket, per tenant) ---

_request_timestamps: dict[str, list[float]] = defaultdict(list)


def is_rate_limited(tenant_id: str, rpm: int = RATE_LIMIT_RPM) -> bool:
    """Sliding window rate limiter — returns True if tenant is over limit."""
    now = time.monotonic()
    window = 60.0
    # evict timestamps outside the window
    _request_timestamps[tenant_id] = [
        t for t in _request_timestamps[tenant_id] if now - t < window
    ]
    if len(_request_timestamps[tenant_id]) >= rpm:
        return True
    _request_timestamps[tenant_id].append(now)
    return False


# --- Mock provider (replace with AnthropicProvider in production) ---

async def _mock_complete(model: str, messages: list[dict]) -> dict:
    """Simulates an LLM completion. Returns content + token counts."""
    return {
        "content": f"[mock] response from {model} to: {messages[-1]['content']}",
        "output_tokens": 50,
    }


# --- App ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("gateway started tier_map=%s", list(TIER_TO_MODEL.keys()))
    yield


app = FastAPI(title="Model Gateway", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    x_tenant_id: str = Header(...),
):
    if req.model_tier not in TIER_TO_MODEL:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown tier '{req.model_tier}'. Valid: {list(TIER_TO_MODEL)}",
        )

    if is_rate_limited(x_tenant_id):
        logger.warning("rate limit exceeded tenant=%s", x_tenant_id)
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    model = TIER_TO_MODEL[req.model_tier]
    start = time.monotonic()

    try:
        result = await _mock_complete(model, req.messages)
    except Exception as primary_err:
        fallback_tier = FALLBACK.get(req.model_tier)
        if not fallback_tier:
            raise HTTPException(status_code=502, detail=str(primary_err))
        logger.warning(
            "primary failed, falling back: %s -> %s err=%s",
            req.model_tier,
            fallback_tier,
            primary_err,
        )
        model = TIER_TO_MODEL[fallback_tier]
        try:
            result = await _mock_complete(model, req.messages)
        except Exception as fallback_err:
            raise HTTPException(status_code=502, detail=str(fallback_err))

    latency_ms = (time.monotonic() - start) * 1000
    cost = (result["output_tokens"] / 1000) * COST_OUT_PER_1K[model]

    logger.info(
        "complete tenant=%s model=%s tokens=%d latency_ms=%.1f cost=%.6f",
        x_tenant_id,
        model,
        result["output_tokens"],
        latency_ms,
        cost,
    )

    return ChatResponse(
        content=result["content"],
        model=model,
        cost_usd=cost,
        latency_ms=round(latency_ms, 1),
    )
```

### Test suite

```python
# tests/test_chat_api.py
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, AsyncMock
from gateway.api.chat import app, _request_timestamps


@pytest.fixture(autouse=True)
def clear_rate_limits():
    _request_timestamps.clear()
    yield
    _request_timestamps.clear()


@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_chat_fast_tier_returns_haiku():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat",
            json={"model_tier": "fast", "messages": [{"role": "user", "content": "hello"}]},
            headers={"x-tenant-id": "tenant-1"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "claude-haiku-5-5"
    assert data["cost_usd"] > 0
    assert data["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_chat_smart_tier_returns_sonnet():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat",
            json={"model_tier": "smart", "messages": [{"role": "user", "content": "hello"}]},
            headers={"x-tenant-id": "tenant-1"},
        )
    assert resp.status_code == 200
    assert resp.json()["model"] == "claude-sonnet-5-5"


@pytest.mark.asyncio
async def test_chat_best_tier_returns_opus():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat",
            json={"model_tier": "best", "messages": [{"role": "user", "content": "hello"}]},
            headers={"x-tenant-id": "tenant-1"},
        )
    assert resp.status_code == 200
    assert resp.json()["model"] == "claude-opus-5-5"


@pytest.mark.asyncio
async def test_unknown_tier_returns_400():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat",
            json={"model_tier": "turbo", "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-tenant-id": "tenant-1"},
        )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_missing_tenant_header_returns_422():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat",
            json={"model_tier": "fast", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 422  # FastAPI validation: required header missing


@pytest.mark.asyncio
async def test_rate_limit_exceeded_returns_429():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # exhaust the 10-rpm limit
        for _ in range(10):
            await client.post(
                "/v1/chat",
                json={"model_tier": "fast", "messages": [{"role": "user", "content": "hi"}]},
                headers={"x-tenant-id": "tenant-rl"},
            )
        resp = await client.post(
            "/v1/chat",
            json={"model_tier": "fast", "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-tenant-id": "tenant-rl"},
        )
    assert resp.status_code == 429


@pytest.mark.asyncio
async def test_rate_limit_is_per_tenant():
    """Exhausting one tenant's limit should not affect another tenant."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for _ in range(10):
            await client.post(
                "/v1/chat",
                json={"model_tier": "fast", "messages": [{"role": "user", "content": "hi"}]},
                headers={"x-tenant-id": "tenant-a"},
            )
        # tenant-b should still be allowed
        resp = await client.post(
            "/v1/chat",
            json={"model_tier": "fast", "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-tenant-id": "tenant-b"},
        )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_fallback_triggered_on_provider_error():
    """When primary model raises, the fallback model should be used."""
    call_count = 0

    async def failing_then_succeeding(model: str, messages: list) -> dict:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("provider timeout")
        return {"content": f"fallback from {model}", "output_tokens": 10}

    with patch("gateway.api.chat._mock_complete", side_effect=failing_then_succeeding):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat",
                json={"model_tier": "smart", "messages": [{"role": "user", "content": "hi"}]},
                headers={"x-tenant-id": "tenant-1"},
            )
    assert resp.status_code == 200
    # smart falls back to fast (haiku)
    assert resp.json()["model"] == "claude-haiku-5-5"
```
