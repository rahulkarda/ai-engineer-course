# Phase 7 · Week 02 · Day 1 — Horizontal vs Vertical Scaling & Stateless Services

> Objective: Understand when and how to scale out vs up, and why statelessness is the prerequisite for horizontal scale.  ·  Time: ~1 hr  ·  Prereqs: [[week-01/day-7]]

## Concept

**Scaling** is the act of increasing a system's capacity to handle more load.

### Vertical scaling (scale-up)
Add more resources to an existing machine: bigger CPU, more RAM, faster disk.

- **Ceiling**: hardware limits exist; the largest available instance is finite.
- **Simplicity**: no code changes — the OS and runtime just see bigger hardware.
- **Single point of failure**: one machine means one outage.
- **Cost**: each doubling of a cloud instance roughly doubles cost, then jumps nonlinearly.

### Horizontal scaling (scale-out)
Add more machines and distribute load across them.

- **Near-unlimited ceiling**: add nodes as demand grows.
- **Fault tolerance**: losing one node does not take the service down.
- **Complexity**: requires a load balancer, coordination, and — critically — **stateless application code**.
- **Cost**: commodity instances are cheaper per unit of capacity beyond a threshold.

### When each applies

| Signal | Prefer vertical | Prefer horizontal |
|--------|----------------|-------------------|
| Database with complex transactions | Yes | Risky without sharding |
| Legacy single-threaded app | Yes (stop-gap) | Requires refactor |
| Stateless API / microservice | Either at small scale | Yes at medium-large |
| ML inference serving | Yes (GPU memory bound) | Yes (replicate stateless workers) |
| Event-driven workers | Either | Yes (per-topic consumer groups) |

## How it works internally

### Why stateless is mandatory for horizontal scale

A **stateful** server stores session data (login token, shopping cart, in-progress workflow) in its own process memory or local disk.

```
Client ─→ Server-A  (user session stored here)
         ↑
         ✗ Cannot route to Server-B — session is missing
```

A **stateless** server holds NO per-user state in memory. Every request carries all context needed, or context is stored in a shared backing store (Redis, DB, JWT).

```
Client ─→ Load Balancer ─→ Server-A  (reads session from Redis)
                        ─→ Server-B  (same Redis — works fine)
                        ─→ Server-C  (same Redis — works fine)
```

### The Twelve-Factor App principle III — Config / XII — Processes

Twelve-factor (Heroku, now widely adopted) states processes must be **stateless and share-nothing**. Any data that must persist goes into a backing service (DB, cache, object store).

### Stateless patterns for common "state" needs

| Need | Stateless approach |
|------|--------------------|
| Auth session | JWT (client carries token) or central session store (Redis) |
| File uploads | Stream directly to object store (S3/GCS); server is pass-through |
| In-progress job state | Persist to task queue (Celery + Redis, SQS) |
| Sticky computation (ML model) | Model is read-only; load it once at startup — fine |

## Worked example

```python
"""
day1_stateless_demo.py
Demonstrates a stateless FastAPI service that externalises session state to Redis.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import redis.asyncio as aioredis
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

redis_client: aioredis.Redis | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client
    redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
    log.info("Redis connected: %s", REDIS_URL)
    yield
    await redis_client.aclose()


app = FastAPI(lifespan=lifespan)


class CartItem(BaseModel):
    product_id: str
    quantity: int


@app.post("/cart/{user_id}/add")
async def add_to_cart(
    user_id: str,
    item: CartItem,
    x_instance_id: str = Header(default="unknown"),
) -> dict[str, Any]:
    """
    Stateless handler — no instance-local state.
    The cart lives in Redis; any replica handles any request.
    """
    assert redis_client is not None
    key = f"cart:{user_id}"
    await redis_client.hset(key, item.product_id, item.quantity)
    await redis_client.expire(key, 3600)  # TTL: 1 hr
    log.info("instance=%s user=%s added product=%s", x_instance_id, user_id, item.product_id)
    return {"instance": x_instance_id, "user": user_id, "cart_key": key}


@app.get("/cart/{user_id}")
async def get_cart(user_id: str) -> dict[str, Any]:
    assert redis_client is not None
    key = f"cart:{user_id}"
    cart = await redis_client.hgetall(key)
    if not cart:
        raise HTTPException(status_code=404, detail="Cart not found or expired")
    return {"user": user_id, "items": cart}
```

Key observations:
- `redis_client` is the only shared resource — it is **connection pool only**, not per-user data.
- Two instances of this service share the same Redis; either can serve any user.
- `X-Instance-Id` header in the response lets you verify different servers served consecutive requests.

### Simulating scale-out without a load balancer

```python
"""
day1_scale_simulation.py
Simulate two 'instances' processing requests round-robin.
Run: python -I day1_scale_simulation.py
"""

import asyncio
import logging
from collections import defaultdict

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


async def fake_instance(instance_id: str, request: str) -> str:
    """Stateless handler — result depends only on input, not instance memory."""
    await asyncio.sleep(0.01)  # simulate I/O
    return f"[instance={instance_id}] processed: {request!r}"


async def main() -> None:
    requests = [f"req-{i}" for i in range(8)]
    instances = ["A", "B"]
    # Round-robin dispatch — works because instances are stateless
    tasks = [
        fake_instance(instances[i % len(instances)], req)
        for i, req in enumerate(requests)
    ]
    results = await asyncio.gather(*tasks)
    for r in results:
        log.info(r)

    # Verify: requests routed to both instances
    counts: dict[str, int] = defaultdict(int)
    for r in results:
        inst = r.split("instance=")[1].split("]")[0]
        counts[inst] += 1
    log.info("Distribution: %s", dict(counts))


if __name__ == "__main__":
    asyncio.run(main())
```

## Exercises

1. **Stateful vs stateless audit**: take a FastAPI app you've written (or sketch one). List every place in-process state is stored. Classify each as: (a) safe because read-only, (b) must be externalised, (c) can be eliminated. Write a one-paragraph refactor plan.

2. **Cost model**: a t3.large (2 vCPU, 8 GB, $0.083/hr) handles 200 req/s. A t3.xlarge (4 vCPU, 16 GB) handles 420 req/s at $0.166/hr. Two t3.larges handle ~400 req/s at $0.166/hr. Which option gives more headroom for the same cost, and why does horizontal win at higher scale?

3. **Implement an externalised counter**: rewrite a simple hit-counter FastAPI endpoint that currently uses a module-level `count = 0` to use Redis INCR instead. Write a pytest that starts two instances (via two `TestClient` objects sharing a mocked Redis) and verifies counts are consistent.

4. **Spot the flaw**: the following code is deployed on 3 replicas. Explain why it breaks:
   ```python
   _sessions: dict[str, str] = {}  # user_id -> token

   @app.post("/login")
   def login(user_id: str) -> str:
       token = secrets.token_hex(16)
       _sessions[user_id] = token
       return token

   @app.get("/me")
   def me(token: str) -> str:
       for uid, t in _sessions.items():
           if t == token:
               return uid
       raise HTTPException(401)
   ```

(Solutions: ../solutions/week-02/day-1.md)

## Self-check
- [ ] I can explain the ceiling and SPOF trade-offs of vertical scaling
- [ ] I can identify stateful code and describe how to externalise the state
- [ ] I can implement a stateless FastAPI service backed by Redis
- [ ] I can calculate cost/capacity trade-offs between scale-up and scale-out

## Supplementary links
- [The Twelve-Factor App — Processes](https://12factor.net/processes)
- [AWS: Vertical vs Horizontal Scaling](https://aws.amazon.com/blogs/database/scaling-your-amazon-rds-instance-vertically-and-horizontally/)
- [Redis: Data types quick reference](https://redis.io/docs/data-types/)
