# Phase 7 · Week 02 · Day 4 — Cache Invalidation, Stampede & Eviction

> Objective: Master the hard problems of caching — keeping data fresh, preventing stampede under high load, and choosing eviction policies.  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

## Concept

Phil Karlton's quote: *"There are only two hard things in computer science: cache invalidation and naming things."* This lesson covers the first.

### Cache invalidation

Invalidation = ensuring the cache does not serve stale data after the underlying data changes.

#### Strategies

**Time-based (TTL)**  
Every entry expires after a fixed duration. Simple, but stale data is served until expiry.  
Use when: eventual consistency is acceptable.

**Event-driven invalidation**  
The service that writes to the DB also deletes/updates the cache key immediately.

```python
async def update_product(product_id: int, new_price: float) -> None:
    await db.execute("UPDATE products SET price=? WHERE id=?", new_price, product_id)
    await cache.delete(f"product:{product_id}")   # hard invalidation
```

Use when: data is critical and stale reads have business cost.

**Versioned keys**  
Instead of invalidating, bump a version number. Old readers still see old data until TTL, but new readers get the new version key immediately.

```python
# Write
version = await cache.incr("product:version")
await cache.set(f"product:{product_id}:v{version}", data, ttl=300)

# Read
version = await cache.get("product:version")
data = await cache.get(f"product:{product_id}:v{version}")
```

**Tag-based invalidation**  
Associate cache entries with tags. Invalidate a tag → all associated entries expire.  
Supported by: Symfony Cache, some Redis-based libraries.

#### Race condition: read-modify-write

```
T0: Thread-A reads: cache miss
T0: Thread-B reads: cache miss (same key, same instant)
T1: Thread-A queries DB, gets val=10
T1: Thread-B queries DB, gets val=10
T2: Thread-A writes cache: key=10, TTL=30s
T2: Thread-B writes cache: key=10 (duplicate work, but consistent here)
```

This is the **cache stampede** — next section.

---

### Cache stampede (thundering herd)

When a popular cache entry expires, many concurrent requests all see a miss simultaneously. Each fires a DB query. The DB receives N×normal load — potentially causing a cascade failure.

```
       Cache expires
            │
   ┌────────┼────────┐
   req-1    req-2    req-3    (all miss simultaneously)
     │        │        │
     ▼        ▼        ▼
  DB query  DB query  DB query  ← thundering herd
```

#### Mitigation 1: Mutex / cache lock

Only one request populates the cache; others wait for it.

```python
async def get_with_lock(key: str) -> Any:
    val = await cache.get(key)
    if val is not None:
        return val
    lock_key = f"lock:{key}"
    acquired = await cache.set(lock_key, "1", nx=True, ex=5)  # atomic
    if acquired:
        try:
            val = await db.get(key)
            await cache.set(key, val, ex=TTL)
            return val
        finally:
            await cache.delete(lock_key)
    else:
        # Another thread is fetching — wait briefly and retry
        await asyncio.sleep(0.05)
        return await get_with_lock(key)  # retry
```

#### Mitigation 2: Probabilistic early recompute (XFetch)

Before the TTL hits zero, some requests probabilistically recompute early, so the entry is refreshed before expiry. The probability increases as expiry approaches.

```python
import math, random

def should_recompute(ttl_remaining: float, beta: float = 1.0, delta: float = 0.1) -> bool:
    """
    XFetch: recompute early with probability that grows as expiry nears.
    ttl_remaining: seconds until expiry
    delta: last recompute cost in seconds
    beta: tuning factor (higher = more eager recomputation)
    """
    return random.random() < beta * delta * math.log(max(ttl_remaining, 1e-9))
```

#### Mitigation 3: Stale-while-revalidate

Serve the stale entry to the current request, trigger a background refresh. Client sees fast (slightly stale) response; next request sees fresh data.

This is a standard HTTP cache directive: `Cache-Control: max-age=60, stale-while-revalidate=30`.

---

### Eviction policies

When cache memory is full, the eviction policy decides what to discard.

| Policy | Description | When to use |
|--------|-------------|-------------|
| **LRU** (least recently used) | Evict the entry not accessed for the longest time | General purpose; good for temporal locality |
| **LFU** (least frequently used) | Evict the entry accessed fewest times | Skewed access patterns (Zipf) |
| **FIFO** | Evict oldest inserted entry | Simple; rarely optimal |
| **TTL-aware LRU** | Expire TTL entries first, then LRU | Redis default (`allkeys-lru`) |
| **Random** | Evict a random entry | Fast; ok for uniform access |
| **No-eviction** | Return error when full | Databases-as-cache where loss is unacceptable |

Redis exposes this via `maxmemory-policy`:
- `allkeys-lru` — recommended for pure caches
- `volatile-lru` — only evict keys with TTL set
- `allkeys-lfu` — best for hot-spot workloads (Zipf)
- `noeviction` — for durability (Redis as primary store)

## How it works internally

### LRU implementation

Classic O(1) LRU uses a **hashmap + doubly-linked list**:
- Hashmap: O(1) lookup by key.
- Doubly-linked list: MRU at head, LRU at tail. On access, move node to head. On eviction, remove tail.

```
HEAD ←→ [most-recent] ←→ ... ←→ [least-recent] ←→ TAIL
```

### Redis LRUCLOCK approximation

Redis does not use a full LRU linked list. Instead, each key stores a 24-bit `lru_clock` (seconds, wraps every ~194 days). On eviction, Redis samples N random keys (default 5), picks the one with the oldest `lru_clock`. This is O(1) per eviction with good practical behaviour.

## Worked example

```python
"""
day4_stampede.py
Demonstrates cache stampede and two mitigations:
1. Redis SET NX lock
2. Stale-while-revalidate (background refresh)
"""

import asyncio
import json
import logging
import os
import time
from typing import Any

import redis.asyncio as aioredis

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
TTL = 5  # Short TTL to make stampede easy to trigger

_db_hits = 0


async def slow_db_query(key: str) -> dict[str, Any]:
    global _db_hits
    _db_hits += 1
    await asyncio.sleep(0.1)  # simulate 100 ms DB query
    log.info("DB HIT #%d for key=%s", _db_hits, key)
    return {"key": key, "value": f"data-for-{key}", "ts": time.time()}


# ---------------------------------------------------------------------------
# Naive cache-aside (stampede-prone)
# ---------------------------------------------------------------------------

async def naive_get(redis: aioredis.Redis, key: str) -> Any:
    raw = await redis.get(key)
    if raw:
        return json.loads(raw)
    val = await slow_db_query(key)
    await redis.setex(key, TTL, json.dumps(val))
    return val


# ---------------------------------------------------------------------------
# Mutex-protected get
# ---------------------------------------------------------------------------

async def locked_get(redis: aioredis.Redis, key: str, retries: int = 10) -> Any:
    raw = await redis.get(key)
    if raw:
        return json.loads(raw)

    lock_key = f"lock:{key}"
    acquired = await redis.set(lock_key, "1", nx=True, ex=5)
    if acquired:
        try:
            val = await slow_db_query(key)
            await redis.setex(key, TTL, json.dumps(val))
            return val
        finally:
            await redis.delete(lock_key)
    else:
        for _ in range(retries):
            await asyncio.sleep(0.02)
            raw = await redis.get(key)
            if raw:
                return json.loads(raw)
        raise TimeoutError(f"Lock wait timeout for key={key}")


# ---------------------------------------------------------------------------
# Stampede simulation
# ---------------------------------------------------------------------------

async def simulate(get_fn, redis: aioredis.Redis, key: str, concurrency: int = 20) -> None:
    global _db_hits
    _db_hits = 0
    await redis.delete(key)  # ensure cold cache

    async def worker(i: int) -> None:
        result = await get_fn(redis, key)
        log.debug("worker-%02d got %s", i, result.get("key"))

    tasks = [worker(i) for i in range(concurrency)]
    await asyncio.gather(*tasks)
    log.info("Strategy=%s concurrency=%d DB calls=%d", get_fn.__name__, concurrency, _db_hits)


async def main() -> None:
    redis = aioredis.from_url(REDIS_URL, decode_responses=True)

    log.info("=== Naive (stampede-prone) ===")
    await simulate(naive_get, redis, "hot-key", concurrency=20)

    log.info("=== Mutex-locked (stampede-safe) ===")
    await simulate(locked_get, redis, "hot-key", concurrency=20)

    await redis.aclose()


if __name__ == "__main__":
    asyncio.run(main())
```

Expected output:
- Naive: DB is hit ~20 times (all concurrent requests miss simultaneously).
- Locked: DB is hit exactly once; other requests wait for the lock.


## Exercises

1. **LRU cache**: implement an `LRUCache[K, V]` class (no external libraries) using `collections.OrderedDict`. Support `get(key)`, `set(key, val)`, and a `capacity` limit. Write pytest tests covering: capacity enforcement, LRU eviction order, and a hit promoting an entry to MRU.

2. **Probabilistic early recompute**: implement `xfetch_get` that stores `(value, expiry_ts, last_delta)` in Redis. Before returning a value, check `should_recompute` and if true, refresh in the background. Write a test showing that with 1000 concurrent readers and beta=2, fewer than 5 thundering-herd DB calls occur on a 5-second TTL.

3. **Tag-based invalidation**: design (in code) a `TaggedCache` wrapper around Redis. `set(key, val, tags=["product", "category:electronics"])`. `invalidate_tag("product")` deletes all keys tagged "product". Use Redis sets to track tag→key membership. Write tests.

4. **Eviction policy selection**: a social media platform stores user feed items (high churn, Zipf-distributed popularity) in Redis. Should it use `allkeys-lru` or `allkeys-lfu`? Justify with a concrete access pattern argument.

(Solutions: ../solutions/week-02/day-4.md)

## Self-check
- [ ] I can describe three cache invalidation strategies and their trade-offs
- [ ] I can implement a mutex-based stampede mitigation using Redis SET NX
- [ ] I can explain O(1) LRU using hashmap + doubly-linked list
- [ ] I can select an appropriate Redis eviction policy for a given workload

## Supplementary links
- [Redis: Key expiration](https://redis.io/docs/manual/keyspace-notifications/)
- [XFetch paper: Optimal Probabilistic Cache Stampede Prevention](https://cseweb.ucsd.edu/~avattani/papers/cache_stampede.pdf)
- [Redis: maxmemory-policy](https://redis.io/docs/manual/eviction/)
- [Cloudflare: stale-while-revalidate explained](https://web.dev/articles/stale-while-revalidate)
