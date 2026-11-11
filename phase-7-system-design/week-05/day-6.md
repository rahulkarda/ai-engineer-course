# Phase 7 · Week 05 · Day 6 (Saturday) — Deep Practice: Two Full Designs + Build

> Objective: Write two complete system design documents + implement a distributed rate limiter and a typeahead service with tests.  ·  Time: ~4 hr  ·  Prereqs: [[week-05/day-1..5]]

---

## Overview

Today is structured practice time. Four hours, four deliverables:

| Block | Time | Deliverable |
|---|---|---|
| A | 60 min | Write full design: Chat + Messaging System |
| B | 60 min | Write full design: Distributed Rate Limiter |
| C | 60 min | Implement rate limiter (token bucket) with pytest suite |
| D | 60 min | Implement typeahead service (trie + Redis) with pytest suite |

The designs follow the 8-step framework from Week 1. The implementations build directly on the reference code from Days 4 and 5.

---

## Block A — Written Design: Chat System (60 min)

Write a full design document covering all 8 steps. Target: 600–900 words + diagrams.

### Constraints for this session
- 50 M DAU, 200 M messages/day.
- 1:1 chat only (no group chat to bound scope).
- Must handle offline delivery.
- No video/voice calls.

### Checklist for your design

- [ ] Functional and non-functional requirements clearly separated.
- [ ] Back-of-envelope estimate (messages/day → storage/day → cache size).
- [ ] REST + WebSocket API sketched with payload examples.
- [ ] Data model: conversations, messages, delivery status tables.
- [ ] Architecture diagram: WebSocket gateway, message service, Cassandra, Redis, push notification service.
- [ ] Deep dive 1: message delivery guarantee (at-least-once with dedup).
- [ ] Deep dive 2: offline message sync (reconnection flow).
- [ ] Bottleneck table: ≥ 3 rows.
- [ ] Trade-off table: ≥ 3 rows.

### Scoring guide (self-assessment)

| Criterion | Points |
|---|---|
| All 8 steps present | 20 |
| Estimates grounded (not made up) | 15 |
| Deep dives show genuine insight (not surface-level) | 25 |
| Trade-offs are real (not generic) | 20 |
| Diagrams are clear and consistent with text | 20 |
| **Total** | **100** |

Pass: ≥ 75.

---

## Block B — Written Design: Distributed Rate Limiter (60 min)

Write a full design document. Target: 500–700 words + diagrams.

### Constraints for this session
- Must work across 30 API pods.
- Per-user limit: 100 req/min.
- Per-IP limit: 1 000 req/min.
- Decision < 2 ms.

### Checklist

- [ ] Requirements: functional + non-functional.
- [ ] Estimate: QPS per pod, Redis ops/s needed.
- [ ] API: how does the limiter expose its decision (middleware vs. sidecar)?
- [ ] Algorithm choice with justification (sliding window vs. token bucket).
- [ ] Redis data model (keys, types, TTL).
- [ ] Architecture diagram: pods → Redis Cluster.
- [ ] Deep dive: Lua script atomicity — why needed?
- [ ] Deep dive: fail-open vs. fail-closed — which and why?
- [ ] Bottleneck + trade-off tables.

---

## Block C — Implementation: Distributed Rate Limiter (60 min)

Build a production-quality token bucket rate limiter backed by Redis, with a pytest suite.

### Deliverables

```
src/rate_limiter/
    __init__.py
    token_bucket.py        # TokenBucketLimiter class from Day 4
    config.py              # RateLimiterConfig (pydantic-settings)
    middleware.py          # FastAPI/Starlette middleware wrapper
tests/unit/
    test_token_bucket.py   # ≥ 8 test cases
tests/integration/
    test_rate_limiter_redis.py  # requires running Redis
```

### `config.py` scaffold

```python
# src/rate_limiter/config.py
from pydantic_settings import BaseSettings


class RateLimiterSettings(BaseSettings):
    redis_url: str = "redis://localhost:6379/0"
    default_capacity: float = 100.0   # tokens
    default_rate: float = 1.67         # tokens/s (100/min)

    model_config = {"env_prefix": "RL_"}
```

### `middleware.py` scaffold

```python
# src/rate_limiter/middleware.py
import logging
from collections.abc import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .token_bucket import TokenBucketLimiter

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limiter: TokenBucketLimiter) -> None:
        super().__init__(app)
        self._limiter = limiter

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Key: prefer authenticated user_id, fall back to client IP
        key = request.headers.get("X-User-Id") or request.client.host
        if not self._limiter.allow(key):
            return Response(
                content='{"detail": "rate limit exceeded"}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": "60"},
            )
        return await call_next(request)
```

### Required test cases (`test_token_bucket.py`)

Write tests for:
1. First request always allowed (bucket starts full).
2. Exactly `capacity` consecutive requests allowed.
3. Request `capacity + 1` is denied.
4. After waiting 1 s, one token refills → one more request allowed.
5. `allow()` with `cost=0.5` allows 2× as many requests.
6. Different keys are isolated (key A exhausted does not affect key B).
7. Concurrent calls do not cause double-spend (use threading).
8. Redis unavailability → fails open (allow request, log warning).

---

## Block D — Implementation: Typeahead Service (60 min)

Build the in-memory trie typeahead service with a simple HTTP API.

### Deliverables

```
src/typeahead/
    __init__.py
    trie.py            # Trie class from Day 5
    service.py         # TypeaheadService: insert, search, bulk_load
    api.py             # FastAPI app with GET /suggest?q=<prefix>
tests/unit/
    test_trie.py       # ≥ 10 test cases
```

### `service.py` scaffold

```python
# src/typeahead/service.py
import logging
from pathlib import Path

from .trie import Trie

logger = logging.getLogger(__name__)


class TypeaheadService:
    def __init__(self, top_k: int = 5) -> None:
        self._trie = Trie()
        self._top_k = top_k

    def suggest(self, prefix: str) -> list[str]:
        """Return top suggestions for prefix."""
        if not prefix:
            return []
        return self._trie.search(prefix.lower())[: self._top_k]

    def record_query(self, query: str, frequency: int = 1) -> None:
        """Record a user query to update frequencies."""
        self._trie.insert(query.lower(), frequency)

    def bulk_load(self, queries: list[tuple[str, int]]) -> None:
        """Load (query, frequency) pairs from offline data."""
        for query, freq in queries:
            self._trie.insert(query.lower(), freq)
        logger.info("Bulk loaded %d queries", len(queries))
```

### `api.py` scaffold

```python
# src/typeahead/api.py
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query
from pydantic import BaseModel

from .service import TypeaheadService

logger = logging.getLogger(__name__)
_service: TypeaheadService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service
    _service = TypeaheadService(top_k=5)
    # In production: load from snapshot file here
    logger.info("Typeahead service ready")
    yield
    logger.info("Typeahead service shutting down")


app = FastAPI(lifespan=lifespan)


class SuggestResponse(BaseModel):
    prefix: str
    suggestions: list[str]


@app.get("/suggest", response_model=SuggestResponse)
def suggest(q: str = Query(min_length=1, max_length=100)) -> SuggestResponse:
    assert _service is not None
    return SuggestResponse(prefix=q, suggestions=_service.suggest(q))
```

### Required test cases (`test_trie.py`)

Write tests for:
1. Empty trie returns empty list for any prefix.
2. Insert one word; prefix matches exact word.
3. Insert "apple" and "apply"; prefix "appl" returns both.
4. Top-K limit is respected (insert 10 words, search returns ≤ K).
5. Higher-frequency word appears first.
6. Insert same word twice; frequency accumulates.
7. Prefix longer than any word returns empty list.
8. Case: "Apple" and "apple" should be normalized to same entry (if service lowercases).
9. Delete a word; it no longer appears in results.
10. Unicode prefix ("café") works correctly.

---

## Exercises

1. In Block C test 7 (concurrency): what is the exact race condition the Lua script prevents? Explain at the Redis command level.
2. In Block D: how would you persist the trie state to disk so the service survives a restart without re-loading all history? What serialization format would you use?
3. Your typeahead service is deployed in 3 regions. How do you keep them in sync? What consistency model is acceptable?
4. Rate limiter Block C: change the implementation to support per-endpoint limits (different capacity/rate for `/api/v1/search` vs. `/api/v1/upload`). Sketch the config model and key change.

(Solutions: ../solutions/week-05/day-6.md)

---

## Self-check
- [ ] I can write a full system design in 60 min with all 8 steps.
- [ ] My rate limiter implementation handles concurrency correctly via Lua.
- [ ] My trie implementation correctly maintains top-K across insertions.
- [ ] I have ≥ 18 passing tests across both implementations.
