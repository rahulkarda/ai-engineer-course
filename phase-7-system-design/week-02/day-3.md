# Phase 7 · Week 02 · Day 3 — Caching Layers & Strategies

> Objective: Map the four caching layers (client, CDN, app, DB), understand cache-aside / write-through / write-back strategies, and implement cache-aside with TTL.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

## Concept

A **cache** is a faster, smaller store that sits in front of a slower, larger store. A hit returns data cheaply; a miss falls through to the origin.

```
                 ┌──────────┐
Browser / Client │  L1 cache│  (memory, ~nanoseconds)
                 └────┬─────┘
                      │ miss
                 ┌────▼─────┐
CDN edge node    │  L2 cache│  (geographic PoP, ~milliseconds)
                 └────┬─────┘
                      │ miss
                 ┌────▼─────┐
App server       │  L3 cache│  (Redis / Memcached, ~sub-millisecond)
                 └────┬─────┘
                      │ miss
                 ┌────▼─────┐
Database         │  origin  │  (disk, ~10s of ms)
                 └──────────┘
```

Each layer trades cost for speed. Hits that stop at L1 never reach the network; L2 hits stop at the CDN edge.

### The four layers in detail

**Client / browser cache**
- HTTP `Cache-Control`, `ETag`, `Last-Modified` headers drive it.
- Zero network cost on a hit.
- Controlled by the server (response headers), not the client.

**CDN cache**
- Geographically distributed edge nodes store copies of static and sometimes dynamic content.
- Hit → response travels a short distance.
- Miss → CDN fetches from origin, stores the response, serves future hits.
- Key = URL + Vary headers (e.g., `Accept-Encoding`, `Accept-Language`).

**Application cache (Redis, Memcached)**
- In-process or sidecar cache backing a service.
- The service code decides what to cache, for how long, and with what key.
- Supports rich data structures (Redis hashes, sorted sets).
- Can be shared across replicas (unlike in-process `dict`).

**Database / query cache**
- Built into the DB engine (e.g., MySQL query cache — deprecated; PostgreSQL plan cache).
- Index buffer pools (InnoDB buffer pool, PostgreSQL shared_buffers) are the dominant form.
- Operator-level: you don't cache here explicitly; you tune buffer sizes and query plans.

---

### Caching strategies

#### Cache-aside (lazy loading)

Application code manages the cache manually.

```
read(key):
  val = cache.get(key)
  if val is None:        # cache miss
      val = db.get(key)
      cache.set(key, val, ttl=T)
  return val

write(key, val):
  db.set(key, val)
  cache.delete(key)      # invalidate, not update
```

- **Pros**: only caches what's actually read; resilient — cache failure just means slower reads.
- **Cons**: first read after expiry is slow (miss penalty); risk of stale data between `db.set` and `cache.delete` under concurrent writes.

#### Write-through

Every write goes to cache AND database synchronously.

```
write(key, val):
  db.set(key, val)
  cache.set(key, val, ttl=T)   # keep cache warm
```

- **Pros**: cache is always fresh; reads are always fast.
- **Cons**: every write is slower (two stores); rarely-read data pollutes cache.

#### Write-back (write-behind)

Writes land in cache only; a background process flushes to DB asynchronously.

```
write(key, val):
  cache.set(key, val)   # immediate, fast
  enqueue(flush_job, key)   # async flush

flush_job(key):
  val = cache.get(key)
  db.set(key, val)
```

- **Pros**: extremely low write latency; DB write batching possible.
- **Cons**: crash between write and flush = data loss; complex recovery.

### Choosing a strategy

| Scenario | Recommended |
|----------|-------------|
| Read-heavy, infrequent writes (product catalogue) | Cache-aside or write-through |
| Write-heavy, reads also hot (leaderboard, counters) | Write-through or write-back |
| Write latency is the bottleneck | Write-back (with durability trade-off acknowledged) |
| Cache is optional / you cannot afford stale reads | Cache-aside with short TTL |

## How it works internally

### Cache key design

Cache key = the minimal set of inputs that uniquely determines the output.

Bad key: `"user_profile"` — collides across users.  
Good key: `f"user:{user_id}:profile"` — unique per user.

Namespace prefixes prevent key collisions across services when sharing one Redis instance: `payments:invoice:42` vs `orders:invoice:42`.

### TTL choice

- Too short → high miss rate → DB load not reduced.
- Too long → stale data shown to users.
- Rule of thumb: TTL = acceptable staleness. For a product price that updates once an hour, TTL=300s is fine. For a real-time leaderboard, TTL=5s (or invalidate on write).

### Memory capacity and eviction preview

When cache memory is full, entries must be evicted. Common policies (covered in depth Day 4):
- **LRU**: evict least-recently used.
- **LFU**: evict least-frequently used.
- **TTL-based**: expired entries go first.

## Worked example

```python
"""
day3_cache_aside.py
Cache-aside pattern with TTL, using Redis as the backing cache.
Run: python -I day3_cache_aside.py
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
DEFAULT_TTL = int(os.getenv("CACHE_TTL_SECONDS", "30"))


# ---------------------------------------------------------------------------
# Simulated slow database
# ---------------------------------------------------------------------------

_DB: dict[str, dict[str, Any]] = {
    "product:1": {"id": 1, "name": "Widget", "price": 9.99},
    "product:2": {"id": 2, "name": "Gadget", "price": 24.99},
    "product:3": {"id": 3, "name": "Doohickey", "price": 4.99},
}
_db_call_count = 0


async def db_get(key: str) -> dict[str, Any] | None:
    global _db_call_count
    _db_call_count += 1
    await asyncio.sleep(0.05)  # simulate 50 ms DB query
    result = _DB.get(key)
    log.info("DB read key=%s hit=%s (total_db_calls=%d)", key, result is not None, _db_call_count)
    return result


async def db_set(key: str, value: dict[str, Any]) -> None:
    _DB[key] = value
    log.info("DB write key=%s", key)


# ---------------------------------------------------------------------------
# Cache-aside implementation
# ---------------------------------------------------------------------------

class CacheAsideStore:
    def __init__(self, redis: aioredis.Redis, ttl: int = DEFAULT_TTL) -> None:
        self._redis = redis
        self._ttl = ttl
        self.hits = 0
        self.misses = 0

    def _cache_key(self, key: str) -> str:
        return f"cache:{key}"

    async def get(self, key: str) -> dict[str, Any] | None:
        ckey = self._cache_key(key)
        raw = await self._redis.get(ckey)
        if raw is not None:
            self.hits += 1
            log.info("CACHE HIT  key=%s", key)
            return json.loads(raw)

        # Cache miss — fetch from DB and populate cache
        self.misses += 1
        log.info("CACHE MISS key=%s", key)
        value = await db_get(key)
        if value is not None:
            await self._redis.setex(ckey, self._ttl, json.dumps(value))
        return value

    async def set(self, key: str, value: dict[str, Any]) -> None:
        """Write-through companion: update DB then invalidate cache."""
        await db_set(key, value)
        ckey = self._cache_key(key)
        await self._redis.delete(ckey)  # invalidate; next read repopulates
        log.info("INVALIDATED cache key=%s", key)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


async def main() -> None:
    redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    store = CacheAsideStore(redis, ttl=10)

    # Simulate repeated reads — second read should hit cache
    for round_num in range(3):
        log.info("--- Round %d ---", round_num + 1)
        for key in ["product:1", "product:2", "product:1"]:
            result = await store.get(key)
            assert result is not None, f"Expected {key} to exist"

    log.info(
        "Hit rate: %.0f%% | DB calls: %d | Cache hits: %d | Cache misses: %d",
        store.hit_rate * 100,
        _db_call_count,
        store.hits,
        store.misses,
    )

    # Simulate a write → cache invalidation
    await store.set("product:1", {"id": 1, "name": "Widget v2", "price": 10.99})
    result = await store.get("product:1")
    log.info("After update: %s", result)

    await redis.aclose()


if __name__ == "__main__":
    asyncio.run(main())
```

Expected output shows:
- Round 1: all misses → DB calls.
- Rounds 2–3: hits for `product:1` and `product:2`.
- After `store.set(...)`, the next `get` is a miss (invalidated) → DB read returns new value.

## Exercises

1. **Write-through**: implement a `WriteThroughStore` class that always writes to both DB and cache synchronously. Write a test that verifies after `set(key, val)`, `get(key)` is always a cache hit.

2. **Hit-rate measurement**: wrap `CacheAsideStore.get` in a Prometheus-style counter (or a simple dict of counters). Simulate 1000 reads for 10 keys with Zipf-distributed frequency (key `i` is accessed proportionally to `1/i`). Plot or log the hit rate vs TTL for TTLs of [5s, 30s, 120s, ∞]. What TTL gives 90%+ hit rate?

3. **Key namespace design**: you have two services, `catalog` and `inventory`, sharing one Redis. Devise a key naming scheme that prevents collisions, supports bulk-delete by service, and allows easy debugging. Document with examples.

4. **Write-back risks**: describe two concrete failure scenarios where write-back loses data. For each, propose a mitigation without switching to write-through.

(Solutions: ../solutions/week-02/day-3.md)

## Self-check
- [ ] I can draw the four caching layers and describe the trade-offs at each
- [ ] I can implement cache-aside with TTL and invalidation on write
- [ ] I can choose between cache-aside, write-through, and write-back for a given scenario
- [ ] I understand TTL sizing and its impact on hit rate vs staleness

## Supplementary links
- [Redis: caching patterns](https://redis.io/docs/manual/patterns/)
- [AWS: Caching best practices](https://aws.amazon.com/caching/best-practices/)
- [HTTP caching — MDN](https://developer.mozilla.org/en-US/docs/Web/HTTP/Caching)
