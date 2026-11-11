# Solutions — Day 1: Horizontal vs Vertical Scaling & Stateless Services

## Exercise 1 — Stateful vs stateless audit

**Example audit of a typical FastAPI app:**

Suppose the app has:
```python
_rate_limit_counts: dict[str, int] = {}        # (a) must be externalised
_config = load_config()                         # (a) safe — read-only after startup
_ml_model = torch.load("model.pt")             # (a) safe — read-only weights
_active_jobs: dict[str, Job] = {}              # (b) must be externalised
```

Classifications:
- `_rate_limit_counts`: must be externalised → Redis INCR with TTL.
- `_config`: safe — read-only, same value on all replicas.
- `_ml_model`: safe — model weights never change at runtime.
- `_active_jobs`: must be externalised → Redis hash or task queue (Celery + Redis).

**Refactor plan**: move `_rate_limit_counts` to Redis with key `ratelimit:{user_id}:{window}` and `_active_jobs` to a Celery task queue. Each replica gets a Redis client from the connection pool at startup. No other changes needed; the code is otherwise stateless.

---

## Exercise 2 — Cost model

Two t3.larges at $0.166/hr handle ~400 req/s.  
One t3.xlarge at $0.166/hr handles ~420 req/s.

Same cost, but two t3.larges give:
- **Fault tolerance**: losing one leaves ~200 req/s still serving; the xlarge goes to 0.
- **Headroom**: two larges can scale to three (600 req/s) by adding $0.083/hr; the xlarge must jump to a 2xlarge (8x cost nonlinearity).
- **Rolling deploys**: update one replica at a time; the xlarge requires a maintenance window.

Horizontal wins at scale because each doubling of vertical capacity costs >2× beyond a threshold, while horizontal scales linearly and adds fault isolation.

---

## Exercise 3 — Externalised counter

```python
"""
test_external_counter.py
"""

import pytest
from unittest.mock import AsyncMock, patch

# Minimal FastAPI app with externalised counter
from fastapi import FastAPI
from fastapi.testclient import TestClient
import asyncio

# The implementation under test:
"""
# counter_app.py
import os
import redis.asyncio as aioredis
from fastapi import FastAPI

app = FastAPI()
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

@app.post("/hit")
async def hit(page: str) -> dict:
    r = aioredis.from_url(REDIS_URL, decode_responses=True)
    count = await r.incr(f"hits:{page}")
    await r.aclose()
    return {"page": page, "count": count}
"""

import fakeredis.aioredis as fakeredis


@pytest.mark.asyncio
async def test_counter_is_shared_across_instances():
    """Two 'instances' sharing the same Redis see a consistent count."""
    fake_redis = fakeredis.FakeRedis(decode_responses=True)

    async def incr_counter(key: str) -> int:
        return await fake_redis.incr(f"hits:{key}")

    # Simulate two instances both hitting the same key
    r1 = await incr_counter("homepage")
    r2 = await incr_counter("homepage")
    r3 = await incr_counter("homepage")

    assert r1 == 1
    assert r2 == 2
    assert r3 == 3
    await fake_redis.aclose()
```

---

## Exercise 4 — Spot the flaw

**Bug**: `_sessions` is a module-level `dict`. With 3 replicas, each replica has its own copy. A login on replica-1 stores the token in replica-1's memory. If a subsequent `/me` request routes to replica-2, its `_sessions` dict is empty → 401.

**Fix**: move sessions to Redis:
```python
import redis.asyncio as aioredis
import os, secrets

r = aioredis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=True)
SESSION_TTL = 3600

@app.post("/login")
async def login(user_id: str) -> str:
    token = secrets.token_hex(16)
    await r.setex(f"session:{token}", SESSION_TTL, user_id)
    return token

@app.get("/me")
async def me(token: str) -> str:
    uid = await r.get(f"session:{token}")
    if not uid:
        raise HTTPException(401)
    return uid
```

The O(N) scan on the old dict is also eliminated — Redis `GET session:{token}` is O(1).
