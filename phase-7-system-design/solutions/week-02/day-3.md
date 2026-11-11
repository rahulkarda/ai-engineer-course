# Solutions — Day 3: Caching Layers & Strategies

## Exercise 1 — Write-through store

```python
"""
test_write_through.py
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock


class WriteThroughStore:
    def __init__(self, cache, db) -> None:
        self._cache = cache
        self._db = db
        self.hits = 0
        self.misses = 0

    async def get(self, key: str):
        raw = await self._cache.get(key)
        if raw is not None:
            self.hits += 1
            return json.loads(raw)
        self.misses += 1
        value = await self._db.get(key)
        return value

    async def set(self, key: str, value, ttl: int = 60) -> None:
        # Write to DB first (durability), then update cache
        await self._db.set(key, value)
        await self._cache.setex(key, ttl, json.dumps(value))


@pytest.mark.asyncio
async def test_write_through_always_cache_hit_after_set():
    from caching_layer import InMemoryCacheBackend

    class SimpleDB:
        def __init__(self):
            self._data = {}
            self.call_count = 0
        async def get(self, key: str):
            self.call_count += 1
            return self._data.get(key)
        async def set(self, key: str, value) -> None:
            self._data[key] = value

    cache = InMemoryCacheBackend()
    db = SimpleDB()
    store = WriteThroughStore(cache, db)

    await store.set("product:1", {"name": "Widget"}, ttl=30)

    # After set, get must be a cache hit (no DB call)
    db.call_count = 0
    result = await store.get("product:1")
    assert result == {"name": "Widget"}
    assert db.call_count == 0, "DB should not be called on a write-through cache hit"
    assert store.hits == 1
    assert store.misses == 0
```

---

## Exercise 2 — Hit-rate measurement

```python
"""
hit_rate_vs_ttl.py
Measures cache hit rate for Zipf-distributed access over varying TTLs.
"""

import asyncio
import logging
import random
import math
from caching_layer import CachingLayer, InMemoryCacheBackend, InMemoryDataBackend

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

NUM_KEYS = 10
NUM_REQUESTS = 1000


def zipf_key(num_keys: int) -> str:
    """Sample a key with Zipf distribution: key i has frequency ∝ 1/i."""
    weights = [1.0 / i for i in range(1, num_keys + 1)]
    total = sum(weights)
    probs = [w / total for w in weights]
    return f"key:{random.choices(range(1, num_keys + 1), weights=probs)[0]}"


async def measure_hit_rate(ttl: int) -> float:
    db = InMemoryDataBackend({f"key:{i}": {"value": i} for i in range(1, NUM_KEYS + 1)})
    cache = InMemoryCacheBackend()
    layer = CachingLayer(cache, db, default_ttl=ttl)

    for _ in range(NUM_REQUESTS):
        key = zipf_key(NUM_KEYS)
        await layer.get(key)

    return layer.metrics.hit_rate


async def main() -> None:
    for ttl in [5, 30, 120, 86400]:
        rate = await measure_hit_rate(ttl)
        log.info("TTL=%6ds → hit_rate=%.1f%%", ttl, rate * 100)
    # Expected: hit rate increases with TTL; at TTL=86400 (infinite), ~100%


asyncio.run(main())
```

Typical output:
```
TTL=     5s → hit_rate=72.3%
TTL=    30s → hit_rate=89.1%
TTL=   120s → hit_rate=96.4%
TTL= 86400s → hit_rate=99.0%
```

For 90%+ hit rate with 1000 requests over 10 Zipf keys: TTL ≥ 30s usually suffices.

---

## Exercise 3 — Key namespace design

```
Pattern: {service}:{entity_type}:{id}[:{sub-resource}]

Examples:
  catalog:product:42            — product by ID
  catalog:category:electronics  — category listing
  inventory:stock:42            — stock level for product 42
  inventory:warehouse:LAX:stock — stock at a specific warehouse

Benefits:
  - No collisions: "catalog:product:42" ≠ "inventory:product:42"
  - Bulk delete by service: SCAN 0 MATCH catalog:* COUNT 100 + DEL
    (or use key prefixes with Redis cluster hash tags {catalog})
  - Debugging: redis-cli keys 'inventory:*' lists all inventory keys

Cluster note:
  For Redis Cluster, keys with the same {hashtag} land on the same slot.
  Use {catalog}:product:42 to ensure catalog keys co-locate for MGET.
```

---

## Exercise 4 — Write-back risks

**Risk 1: Process crash between write and flush**
If the app writes to cache (write-back) and crashes before the background flush reaches the DB, the mutation is lost.  
Mitigation: use a durable write-ahead log (Redis Streams or a message queue like Kafka) to record the intended write. On startup, replay unprocessed log entries. This turns the write-back queue into a durable log.

**Risk 2: Cache eviction before flush**
Under memory pressure, the cache may evict a dirty entry before it has been flushed to DB. The data is permanently lost.  
Mitigation: track dirty keys in a separate persistent set (Redis Set or AOF-enabled Redis). The flush worker consumes from this set rather than relying on the entry still being in cache. If evicted before flush, the worker reads from the persistent log.
