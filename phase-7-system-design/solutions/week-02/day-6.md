# Solutions — Day 6: Build Session (Caching Layer + Consistent Hash LB)

## Exercise 1 — test_caching_layer.py

```python
"""
test_caching_layer.py
Full test suite for CachingLayer.
Run: pytest test_caching_layer.py -v
"""

import asyncio
import json
import pytest
from caching_layer import (
    CachingLayer,
    CacheBackend,
    DataBackend,
    InMemoryCacheBackend,
    InMemoryDataBackend,
    CacheMetrics,
)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

def make_layer(data: dict | None = None, ttl: int = 60):
    db = InMemoryDataBackend(data or {"key:1": "value-one", "key:2": "value-two"})
    cache = InMemoryCacheBackend()
    return CachingLayer(cache, db, default_ttl=ttl), db, cache


# -----------------------------------------------------------------------
# Cold miss → DB called once, result cached
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cold_miss_calls_db_once():
    layer, db, _ = make_layer()
    result = await layer.get("key:1")
    assert result == "value-one"
    assert db.call_count == 1


@pytest.mark.asyncio
async def test_warm_hit_does_not_call_db():
    layer, db, _ = make_layer()
    await layer.get("key:1")         # miss → populates cache
    db.call_count = 0
    result = await layer.get("key:1")  # hit → no DB
    assert result == "value-one"
    assert db.call_count == 0


# -----------------------------------------------------------------------
# Stampede: 50 concurrent gets → DB called exactly once
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stampede_db_called_once():
    layer, db, _ = make_layer()
    results = await asyncio.gather(*[layer.get("key:1") for _ in range(50)])
    assert all(r == "value-one" for r in results)
    assert db.call_count == 1, f"Expected 1 DB call, got {db.call_count}"
    assert layer.metrics.stampedes_averted == 49


# -----------------------------------------------------------------------
# TTL expiry → next get is a miss
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ttl_expiry_causes_miss():
    layer, db, _ = make_layer(ttl=1)
    await layer.get("key:1")
    assert db.call_count == 1

    await asyncio.sleep(1.1)  # let TTL expire

    db.call_count = 0
    await layer.get("key:1")
    assert db.call_count == 1, "Should have called DB again after TTL expiry"


# -----------------------------------------------------------------------
# Invalidation: after invalidate, next get is a miss
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_invalidation_causes_miss():
    layer, db, _ = make_layer()
    await layer.get("key:1")   # populate cache
    db.call_count = 0

    await layer.invalidate("key:1")
    result = await layer.get("key:1")
    assert result == "value-one"
    assert db.call_count == 1, "Should have called DB after invalidation"


@pytest.mark.asyncio
async def test_invalidation_with_new_value():
    layer, db, _ = make_layer()
    await layer.invalidate("key:1", new_value="updated-value")
    result = await layer.get("key:1")
    assert result == "updated-value"


# -----------------------------------------------------------------------
# Metrics: hit_rate correct
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hit_rate_after_warmup():
    layer, _, _ = make_layer()
    await layer.get("key:1")   # miss
    for _ in range(9):
        await layer.get("key:1")  # 9 hits
    assert layer.metrics.hit_rate == pytest.approx(0.9, abs=0.01)


# -----------------------------------------------------------------------
# StampedeGuard waiters receive correct value
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_waiters_get_correct_value():
    layer, _, _ = make_layer(data={"slow": {"complex": "object", "n": 42}})
    results = await asyncio.gather(*[layer.get("slow") for _ in range(10)])
    assert all(r == {"complex": "object", "n": 42} for r in results)


# -----------------------------------------------------------------------
# Protocol conformance
# -----------------------------------------------------------------------

def test_in_memory_cache_satisfies_protocol():
    cache = InMemoryCacheBackend()
    assert isinstance(cache, CacheBackend)


def test_in_memory_db_satisfies_protocol():
    db = InMemoryDataBackend()
    assert isinstance(db, DataBackend)
```

---

## Exercise 2 — test_consistent_hash_lb.py

```python
"""
test_consistent_hash_lb.py
"""

import pytest
from consistent_hash_lb import Backend, ConsistentHashLB


def make_ring(*addresses: tuple[str, int]) -> ConsistentHashLB:
    lb = ConsistentHashLB(vnodes_per_backend=150)
    for host, port in addresses:
        lb.add_backend(Backend(host, port))
    return lb


# -----------------------------------------------------------------------
# Determinism
# -----------------------------------------------------------------------

def test_determinism():
    lb = make_ring(("10.0.0.1", 8001), ("10.0.0.2", 8002), ("10.0.0.3", 8003))
    key = "user:99999"
    first = lb.get_backend(key)
    for _ in range(100):
        assert lb.get_backend(key) == first, "Routing must be deterministic"


# -----------------------------------------------------------------------
# Distribution: 3 backends, 10K keys, each in [20%, 47%]
# -----------------------------------------------------------------------

def test_distribution_3_backends():
    lb = make_ring(("10.0.0.1", 8001), ("10.0.0.2", 8002), ("10.0.0.3", 8003))
    keys = [f"key:{i}" for i in range(10_000)]
    counts: dict[str, int] = {}
    for k in keys:
        b = lb.get_backend(k)
        assert b is not None
        counts[b.address] = counts.get(b.address, 0) + 1
    for addr, count in counts.items():
        pct = count / 10_000
        assert 0.20 <= pct <= 0.47, f"{addr}: {pct:.2%} outside [20%, 47%]"


# -----------------------------------------------------------------------
# Scale-out: adding 4th backend remaps ≤ 30%
# -----------------------------------------------------------------------

def test_scale_out_remapping():
    lb = make_ring(("10.0.0.1", 8001), ("10.0.0.2", 8002), ("10.0.0.3", 8003))
    keys = [f"key:{i}" for i in range(10_000)]
    before = {k: lb.get_backend(k).address for k in keys}

    lb.add_backend(Backend("10.0.0.4", 8004))
    after = {k: lb.get_backend(k).address for k in keys}

    remapped = sum(1 for k in keys if before[k] != after[k])
    assert remapped / 10_000 <= 0.30, f"Too many remapped: {remapped}"


# -----------------------------------------------------------------------
# Scale-in: removing a backend, all keys served by remaining
# -----------------------------------------------------------------------

def test_scale_in_all_keys_served():
    lb = make_ring(("10.0.0.1", 8001), ("10.0.0.2", 8002), ("10.0.0.3", 8003))
    victim = Backend("10.0.0.2", 8002)
    lb.remove_backend(victim)
    keys = [f"key:{i}" for i in range(1_000)]
    for k in keys:
        b = lb.get_backend(k)
        assert b is not None
        assert b.address != victim.address, f"Key {k} still routed to removed backend"


# -----------------------------------------------------------------------
# Weight: weight=2 handles ~2× keys of weight=1
# -----------------------------------------------------------------------

def test_weighted_distribution():
    lb = ConsistentHashLB(vnodes_per_backend=150)
    lb.add_backend(Backend("10.0.0.1", 8001, weight=1))
    lb.add_backend(Backend("10.0.0.2", 8002, weight=2))
    keys = [f"key:{i}" for i in range(10_000)]
    counts: dict[str, int] = {}
    for k in keys:
        b = lb.get_backend(k)
        assert b is not None
        counts[b.address] = counts.get(b.address, 0) + 1
    ratio = counts.get("10.0.0.2:8002", 0) / counts.get("10.0.0.1:8001", 1)
    assert 1.5 <= ratio <= 2.5, f"Expected ~2x, got {ratio:.2f}"


# -----------------------------------------------------------------------
# Empty ring returns None
# -----------------------------------------------------------------------

def test_empty_ring_returns_none():
    lb = ConsistentHashLB()
    assert lb.get_backend("any-key") is None
```

---

## Exercise 3 — Weighted consistent hashing

The solution is already embedded in the `ConsistentHashLB` implementation from Day 6:
```python
vnode_count = self._vnodes_per_backend * backend.weight
```
A backend with `weight=2` receives `150 * 2 = 300` virtual nodes vs `150` for `weight=1`. Because virtual nodes are uniformly distributed on the ring, the weight=2 backend captures approximately twice the key space. The `test_weighted_distribution` test above verifies 1.5×–2.5× ratio.

---

## Exercise 4 — RedisCacheBackend

```python
"""
redis_cache_backend.py
Production Redis implementation of CacheBackend Protocol.
"""

import redis.asyncio as aioredis
from typing import Optional


class RedisCacheBackend:
    def __init__(self, redis: aioredis.Redis) -> None:
        self._redis = redis

    async def get(self, key: str) -> Optional[str]:
        return await self._redis.get(key)

    async def set(self, key: str, value: str, ttl: int) -> None:
        await self._redis.setex(key, ttl, value)

    async def delete(self, key: str) -> None:
        await self._redis.delete(key)

    async def set_nx(self, key: str, value: str, ttl: int) -> bool:
        result = await self._redis.set(key, value, nx=True, ex=ttl)
        return result is not None  # True if set, False if key existed


# Usage: drop-in replacement
# redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
# cache = RedisCacheBackend(redis_client)
# layer = CachingLayer(cache, db, default_ttl=60)
```
