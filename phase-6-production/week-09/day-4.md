# Phase 6 · Week 09 · Day 4 — Caching (Exact + Semantic) + Rate Limiting / Quota

> Objective: Add exact-match and semantic caching, per-tenant rate limiting, and daily budget enforcement to the gateway.  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

## Concept

Two orthogonal cost-reduction techniques:

**Caching** avoids calling the LLM at all when an equivalent response already exists:

- **Exact cache** — hash the full request (model + sorted messages). Identical requests return the cached response instantly. Cache entries expire after a TTL to avoid stale answers.
- **Semantic cache** — embed the user query, store the embedding alongside the response. On a new request, compute the query embedding and find the nearest neighbour. If cosine similarity exceeds a threshold (e.g. 0.95), the stored response is "close enough" and is returned without an LLM call. Useful when users paraphrase the same question repeatedly.

**Rate limiting and quotas** protect your cost budget and ensure fair multi-tenant use:

- **Rate limiting** — token bucket per `tenant_id`: each tenant has a `rpm_limit` (requests per minute). The bucket refills every minute. If empty, the request is rejected with a `429`-equivalent error.
- **Quota tracking** — each tenant has a `daily_budget_usd`. After every successful completion, deduct the estimated cost. If the budget is exhausted, further requests are blocked until the next day.

All state is in-memory in this lesson. In production, use Redis so multiple gateway replicas share the same counters.

## How it works internally

**Exact cache key:**

```
key = SHA-256(json.dumps({"model": ..., "messages": [...]}, sort_keys=True))
```

Sorting keys ensures `{"role":"user","content":"hi"}` and `{"content":"hi","role":"user"}` produce the same hash.

**Semantic cache flow:**

```
query_embedding = embed(last_user_message)
for (stored_embedding, cached_response) in semantic_store:
    sim = cosine_similarity(query_embedding, stored_embedding)
    if sim >= threshold:
        return cached_response  # cache hit
# cache miss → call LLM → store (embedding, response)
```

Cosine similarity with numpy:

```python
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
```

**Token bucket** (rate limiter):

```
bucket[tenant_id] = {"tokens": rpm_limit, "last_refill": now}
on request:
    elapsed = now - last_refill
    tokens += elapsed_minutes * rpm_limit  (cap at rpm_limit)
    if tokens < 1: reject
    tokens -= 1
```

**Quota deduction** happens after a successful (non-cached) completion. If the deduction would push `spent_usd > daily_budget_usd`, the call is blocked before it reaches the provider.

## Worked example

```python
# gateway/cache.py
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gateway.providers import CompletionRequest, CompletionResponse

logger = logging.getLogger(__name__)


# ── Exact cache ───────────────────────────────────────────────────────────────

@dataclass
class _CacheEntry:
    response: CompletionResponse
    expires_at: float  # monotonic timestamp


class ExactCache:
    """SHA-256-keyed in-memory cache with per-entry TTL.

    Production note: replace `_store` with a Redis client.
    """

    def __init__(self, default_ttl_seconds: float = 300.0) -> None:
        self.default_ttl = default_ttl_seconds
        self._store: dict[str, _CacheEntry] = {}

    @staticmethod
    def make_key(request: CompletionRequest) -> str:
        """Stable SHA-256 hash of model + messages (order-normalised)."""
        payload = json.dumps(
            {"model": request.model, "messages": request.messages},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> CompletionResponse | None:
        """Return cached response if present and not expired."""
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.monotonic() > entry.expires_at:
            del self._store[key]
            logger.debug("cache.exact.expired key=%s", key[:8])
            return None
        logger.debug("cache.exact.hit key=%s", key[:8])
        return entry.response

    def set(
        self,
        key: str,
        response: CompletionResponse,
        ttl: float | None = None,
    ) -> None:
        """Store a response under key with a TTL."""
        expires_at = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
        self._store[key] = _CacheEntry(response=response, expires_at=expires_at)
        logger.debug("cache.exact.set key=%s ttl=%.0fs", key[:8], ttl or self.default_ttl)

    def invalidate(self, key: str) -> None:
        self._store.pop(key, None)

    @property
    def size(self) -> int:
        return len(self._store)
```

```python
# gateway/semantic_cache.py
import logging
import time

import numpy as np

from gateway.providers import CompletionResponse

logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [0, 1]. Returns 0 if either vector is zero."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


class SimpleSemanticCache:
    """Nearest-neighbour cache backed by an in-memory list of embeddings.

    Args:
        threshold:         Minimum cosine similarity to treat as a hit (0–1).
        default_ttl:       Seconds before an entry expires.
        embed_fn:          Callable[str] → np.ndarray.  In production, use a
                           real embedding model.  Pass a stub for testing.

    Production note: replace the list scan with a vector DB (Pinecone, pgvector).
    """

    def __init__(
        self,
        embed_fn,   # Callable[[str], np.ndarray]
        threshold: float = 0.95,
        default_ttl: float = 600.0,
    ) -> None:
        self._embed_fn = embed_fn
        self.threshold = threshold
        self.default_ttl = default_ttl
        # Each entry: (embedding, response, expires_at)
        self._store: list[tuple[np.ndarray, CompletionResponse, float]] = []

    def _purge_expired(self) -> None:
        now = time.monotonic()
        self._store = [(e, r, exp) for e, r, exp in self._store if exp > now]

    def get(self, query: str) -> CompletionResponse | None:
        """Return cached response if a similar query exists."""
        self._purge_expired()
        query_emb = self._embed_fn(query)

        best_sim, best_resp = 0.0, None
        for stored_emb, resp, _ in self._store:
            sim = cosine_similarity(query_emb, stored_emb)
            if sim > best_sim:
                best_sim, best_resp = sim, resp

        if best_sim >= self.threshold and best_resp is not None:
            logger.debug("cache.semantic.hit similarity=%.4f", best_sim)
            return best_resp

        logger.debug("cache.semantic.miss best_similarity=%.4f", best_sim)
        return None

    def set(self, query: str, response: CompletionResponse, ttl: float | None = None) -> None:
        """Store a response alongside the query embedding."""
        emb = self._embed_fn(query)
        expires_at = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
        self._store.append((emb, response, expires_at))
        logger.debug("cache.semantic.set store_size=%d", len(self._store))

    @property
    def size(self) -> int:
        return len(self._store)
```

```python
# gateway/rate_limit.py
import logging
import time
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Tenant configuration ──────────────────────────────────────────────────────

class TenantQuota(BaseModel):
    """Per-tenant limits enforced by RateLimiter and QuotaTracker."""
    tenant_id: str
    rpm_limit: int = 60               # requests per minute
    daily_token_limit: int = 1_000_000
    daily_budget_usd: float = 10.0


# ── Token-bucket rate limiter ─────────────────────────────────────────────────

class _BucketState:
    __slots__ = ("tokens", "last_refill")

    def __init__(self, capacity: float) -> None:
        self.tokens = capacity
        self.last_refill = time.monotonic()


class RateLimiter:
    """Per-tenant token-bucket rate limiter.

    Production note: replace `_buckets` with Redis + Lua script for atomicity.
    """

    def __init__(self, quotas: list[TenantQuota]) -> None:
        self._quotas: dict[str, TenantQuota] = {q.tenant_id: q for q in quotas}
        self._buckets: dict[str, _BucketState] = {}

    def _bucket(self, tenant_id: str) -> _BucketState:
        if tenant_id not in self._buckets:
            capacity = self._quotas[tenant_id].rpm_limit if tenant_id in self._quotas else 60
            self._buckets[tenant_id] = _BucketState(capacity)
        return self._buckets[tenant_id]

    def is_allowed(self, tenant_id: str) -> bool:
        """Return True if the tenant has quota for one more request this minute."""
        quota = self._quotas.get(tenant_id)
        rpm = quota.rpm_limit if quota else 60

        bucket = self._bucket(tenant_id)
        now = time.monotonic()
        elapsed_minutes = (now - bucket.last_refill) / 60.0

        # Refill tokens proportionally (capped at rpm).
        bucket.tokens = min(rpm, bucket.tokens + elapsed_minutes * rpm)
        bucket.last_refill = now

        if bucket.tokens >= 1:
            bucket.tokens -= 1
            logger.debug("rate_limit.allowed tenant=%s tokens_remaining=%.1f", tenant_id, bucket.tokens)
            return True

        logger.warning("rate_limit.rejected tenant=%s", tenant_id)
        return False


# ── Quota tracker (daily budget) ──────────────────────────────────────────────

class _TenantUsage:
    __slots__ = ("spent_usd", "tokens_used", "reset_at")

    def __init__(self, reset_at: float) -> None:
        self.spent_usd: float = 0.0
        self.tokens_used: int = 0
        self.reset_at = reset_at


class QuotaTracker:
    """Tracks per-tenant daily spend and token usage.

    Production note: persist to Redis or a DB; reset daily via cron.
    """

    _DAY_SECONDS = 86_400

    def __init__(self, quotas: list[TenantQuota]) -> None:
        self._quotas: dict[str, TenantQuota] = {q.tenant_id: q for q in quotas}
        self._usage: dict[str, _TenantUsage] = {}

    def _usage_for(self, tenant_id: str) -> _TenantUsage:
        usage = self._usage.get(tenant_id)
        now = time.monotonic()
        if usage is None or now >= usage.reset_at:
            self._usage[tenant_id] = _TenantUsage(reset_at=now + self._DAY_SECONDS)
        return self._usage[tenant_id]

    def check_budget(self, tenant_id: str, estimated_cost_usd: float) -> bool:
        """Return True if the tenant can afford estimated_cost_usd."""
        quota = self._quotas.get(tenant_id)
        if quota is None:
            return True  # unknown tenant: allow (configure a default quota in prod)
        usage = self._usage_for(tenant_id)
        allowed = (usage.spent_usd + estimated_cost_usd) <= quota.daily_budget_usd
        if not allowed:
            logger.warning(
                "quota.budget_exceeded tenant=%s spent=%.4f budget=%.4f",
                tenant_id, usage.spent_usd, quota.daily_budget_usd,
            )
        return allowed

    def deduct(self, tenant_id: str, cost_usd: float, tokens: int) -> None:
        """Record actual spend after a successful completion."""
        usage = self._usage_for(tenant_id)
        usage.spent_usd += cost_usd
        usage.tokens_used += tokens
        logger.debug(
            "quota.deduct tenant=%s cost=%.6f tokens=%d total_spent=%.4f",
            tenant_id, cost_usd, tokens, usage.spent_usd,
        )

    def get_remaining_budget(self, tenant_id: str) -> float:
        quota = self._quotas.get(tenant_id)
        if quota is None:
            return float("inf")
        usage = self._usage_for(tenant_id)
        return max(0.0, quota.daily_budget_usd - usage.spent_usd)
```

```python
# tests/unit/test_cache.py
import time
import pytest
import numpy as np

from gateway.cache import ExactCache
from gateway.semantic_cache import SimpleSemanticCache, cosine_similarity
from gateway.providers import CompletionRequest, CompletionResponse


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_response(content: str = "ok") -> CompletionResponse:
    return CompletionResponse(
        content=content,
        model="claude-haiku-5-5",
        provider="mock",
        input_tokens=5,
        output_tokens=2,
        latency_ms=1.0,
    )


def make_request(content: str = "hello") -> CompletionRequest:
    return CompletionRequest(
        model="claude-haiku-5-5",
        messages=[{"role": "user", "content": content}],
    )


# ── ExactCache ────────────────────────────────────────────────────────────────

def test_exact_cache_hit():
    cache = ExactCache()
    req = make_request()
    key = cache.make_key(req)
    resp = make_response("cached answer")
    cache.set(key, resp)
    assert cache.get(key) is not None
    assert cache.get(key).content == "cached answer"


def test_exact_cache_miss_returns_none():
    cache = ExactCache()
    assert cache.get("nonexistent") is None


def test_exact_cache_expiry():
    cache = ExactCache(default_ttl_seconds=0.05)  # 50 ms TTL
    key = cache.make_key(make_request())
    cache.set(key, make_response())
    time.sleep(0.1)
    assert cache.get(key) is None  # expired


def test_exact_cache_key_deterministic():
    r1 = make_request("deterministic")
    r2 = make_request("deterministic")
    cache = ExactCache()
    assert cache.make_key(r1) == cache.make_key(r2)


def test_exact_cache_different_requests_differ():
    cache = ExactCache()
    k1 = cache.make_key(make_request("hello"))
    k2 = cache.make_key(make_request("goodbye"))
    assert k1 != k2


# ── cosine_similarity ─────────────────────────────────────────────────────────

def test_cosine_identical_vectors():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)


def test_cosine_orthogonal_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)


def test_cosine_zero_vector_returns_zero():
    a = np.zeros(3)
    b = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(a, b) == 0.0


# ── SimpleSemanticCache ───────────────────────────────────────────────────────

def fixed_embed(text: str) -> np.ndarray:
    """Stub embedder: returns a constant vector (simulates identical queries)."""
    return np.array([1.0, 0.0, 0.0])


def test_semantic_cache_hit_above_threshold():
    sc = SimpleSemanticCache(embed_fn=fixed_embed, threshold=0.95)
    sc.set("what is AI?", make_response("AI is ..."))
    result = sc.get("what is artificial intelligence?")
    assert result is not None
    assert result.content == "AI is ..."


def test_semantic_cache_miss_below_threshold():
    # Two different vectors → low similarity
    call_count = {"n": 0}

    def varying_embed(text: str) -> np.ndarray:
        call_count["n"] += 1
        return np.array([1.0, 0.0, 0.0]) if call_count["n"] == 1 else np.array([0.0, 1.0, 0.0])

    sc = SimpleSemanticCache(embed_fn=varying_embed, threshold=0.95)
    sc.set("question A", make_response("answer A"))
    result = sc.get("completely different question")
    assert result is None
```

```python
# tests/unit/test_rate_limit.py
import pytest

from gateway.rate_limit import QuotaTracker, RateLimiter, TenantQuota


def make_quota(tenant_id: str = "t1", rpm: int = 5, budget: float = 1.0) -> TenantQuota:
    return TenantQuota(tenant_id=tenant_id, rpm_limit=rpm, daily_budget_usd=budget)


# ── RateLimiter ───────────────────────────────────────────────────────────────

def test_rate_limiter_allows_up_to_rpm():
    quota = make_quota(rpm=3)
    limiter = RateLimiter([quota])
    results = [limiter.is_allowed("t1") for _ in range(4)]
    # First 3 should be allowed, 4th rejected.
    assert results[:3] == [True, True, True]
    assert results[3] is False


def test_rate_limiter_unknown_tenant_uses_default():
    limiter = RateLimiter([])
    # Should not raise; uses default 60 rpm.
    assert limiter.is_allowed("unknown") is True


# ── QuotaTracker ──────────────────────────────────────────────────────────────

def test_quota_tracker_allows_within_budget():
    tracker = QuotaTracker([make_quota(budget=1.0)])
    assert tracker.check_budget("t1", 0.50) is True


def test_quota_tracker_blocks_over_budget():
    tracker = QuotaTracker([make_quota(budget=0.01)])
    tracker.deduct("t1", 0.01, 100)
    assert tracker.check_budget("t1", 0.001) is False


def test_quota_tracker_remaining_budget_decreases():
    tracker = QuotaTracker([make_quota(budget=5.0)])
    tracker.deduct("t1", 2.0, 500)
    assert tracker.get_remaining_budget("t1") == pytest.approx(3.0)


def test_quota_tracker_unknown_tenant_always_allowed():
    tracker = QuotaTracker([])
    assert tracker.check_budget("nobody", 999.0) is True
```

Full integration sketch (wiring cache + rate limiter + quota into the gateway):

```python
# gateway/gateway.py (sketch — not complete)
import logging
from gateway.cache import ExactCache
from gateway.rate_limit import RateLimiter, QuotaTracker, TenantQuota
from gateway.routing import estimate_cost, ModelRouter, RoutingPolicy
from gateway.providers import CompletionRequest, CompletionResponse, ProviderRegistry

logger = logging.getLogger(__name__)


class GatewayError(Exception):
    pass


class Gateway:
    def __init__(
        self,
        registry: ProviderRegistry,
        router: ModelRouter,
        exact_cache: ExactCache,
        rate_limiter: RateLimiter,
        quota_tracker: QuotaTracker,
    ) -> None:
        self._registry = registry
        self._router = router
        self._cache = exact_cache
        self._limiter = rate_limiter
        self._quota = quota_tracker

    async def complete(
        self,
        request: CompletionRequest,
        tenant_id: str,
        policy: RoutingPolicy = RoutingPolicy.AUTO,
    ) -> CompletionResponse:
        # 1. Rate limit check.
        if not self._limiter.is_allowed(tenant_id):
            raise GatewayError(f"Rate limit exceeded for tenant {tenant_id}")

        # 2. Exact cache lookup.
        key = self._cache.make_key(request)
        cached = self._cache.get(key)
        if cached is not None:
            logger.info("gateway.cache_hit tenant=%s", tenant_id)
            return cached

        # 3. Route to provider + model.
        provider_name, model_id = self._router.route(request, policy)
        request = request.model_copy(update={"model": model_id})

        # 4. Budget check (estimated).
        estimated_cost = estimate_cost(model_id, request.max_tokens, request.max_tokens // 4)
        if not self._quota.check_budget(tenant_id, estimated_cost):
            raise GatewayError(f"Daily budget exceeded for tenant {tenant_id}")

        # 5. Provider call.
        provider = self._registry.get(provider_name)
        response = await provider.complete(request)

        # 6. Record actual spend.
        actual_cost = estimate_cost(model_id, response.input_tokens, response.output_tokens)
        self._quota.deduct(tenant_id, actual_cost, response.input_tokens + response.output_tokens)

        # 7. Cache the response.
        self._cache.set(key, response)

        return response
```

## Exercises

1. Add a `clear()` method to `ExactCache` that removes all entries, and a `size` property that returns the count of non-expired entries only (i.e., it purges expired entries before counting). Write tests for both.

2. Extend `SimpleSemanticCache` with a `max_size: int` parameter. When `set()` would exceed `max_size`, evict the oldest entry (FIFO). Write a test that adds `max_size + 1` entries and verifies `size == max_size`.

3. Add a `TokenRateLimiter` variant that limits by **tokens per minute** instead of requests per minute. Its `is_allowed(tenant_id, token_count)` returns `True` if the tenant has enough token budget remaining in the current window. Write at least two tests.

4. Write an integration test for `Gateway.complete()` that:
   - Uses a `MockProvider`
   - Confirms the second call with identical messages returns the cached response (assert `MockProvider.complete` is called exactly once across two gateway calls)
   - Uses `pytest.mark.asyncio`

(Solutions: ../solutions/week-09/day-4.md)

## Self-check

- [ ] I can explain when exact caching is insufficient and semantic caching adds value
- [ ] I can implement a SHA-256 cache key from a `CompletionRequest` without using mutable state
- [ ] I can implement a token-bucket rate limiter and a daily budget tracker with in-memory state
