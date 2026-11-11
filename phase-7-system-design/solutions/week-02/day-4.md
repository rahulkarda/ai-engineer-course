# Solutions — Day 4: Cache Invalidation, Stampede & Eviction

## Exercise 1 — LRU cache

```python
"""
lru_cache.py
O(1) LRU cache using OrderedDict.
"""

from collections import OrderedDict
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._capacity = capacity
        self._store: OrderedDict[K, V] = OrderedDict()

    def get(self, key: K) -> V | None:
        if key not in self._store:
            return None
        self._store.move_to_end(key)   # mark as most recently used
        return self._store[key]

    def set(self, key: K, val: V) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = val
        if len(self._store) > self._capacity:
            self._store.popitem(last=False)  # evict LRU (first item)


# Tests
import pytest

def test_lru_capacity_enforced():
    c: LRUCache[str, int] = LRUCache(3)
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)
    c.set("d", 4)            # evicts "a"
    assert c.get("a") is None
    assert c.get("b") == 2

def test_lru_eviction_order():
    c: LRUCache[str, int] = LRUCache(2)
    c.set("x", 10)
    c.set("y", 20)
    c.get("x")               # access "x" → "y" is now LRU
    c.set("z", 30)           # evicts "y"
    assert c.get("y") is None
    assert c.get("x") == 10
    assert c.get("z") == 30

def test_lru_hit_promotes_to_mru():
    c: LRUCache[str, int] = LRUCache(3)
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)
    c.get("a")               # "a" is now MRU; "b" is LRU
    c.set("d", 4)            # evicts "b"
    assert c.get("b") is None
    assert c.get("a") == 1
```

---

## Exercise 2 — Probabilistic early recompute (XFetch)

```python
"""
xfetch_cache.py
Demonstrates XFetch probabilistic early recompute.
"""

import asyncio
import json
import logging
import math
import random
import time
from typing import Any

from caching_layer import InMemoryCacheBackend

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

_db_hits = 0


async def slow_db_fetch(key: str) -> dict[str, Any]:
    global _db_hits
    _db_hits += 1
    await asyncio.sleep(0.01)
    return {"key": key, "value": "data", "ts": time.time()}


def should_recompute(ttl_remaining: float, beta: float, delta: float) -> bool:
    if ttl_remaining <= 0:
        return True
    return random.random() < beta * delta * math.log(max(ttl_remaining, 1e-9))


async def xfetch_get(cache: InMemoryCacheBackend, key: str, ttl: int, beta: float = 2.0) -> Any:
    meta_key = f"__xfetch_meta__:{key}"
    raw = await cache.get(key)
    meta_raw = await cache.get(meta_key)

    if raw is not None and meta_raw is not None:
        meta = json.loads(meta_raw)
        expires_at = meta["expires_at"]
        delta = meta["delta"]
        ttl_remaining = expires_at - time.monotonic()

        if not should_recompute(ttl_remaining, beta, delta):
            return json.loads(raw)

    # Cache miss or probabilistic early recompute
    t_start = time.monotonic()
    value = await slow_db_fetch(key)
    delta = time.monotonic() - t_start

    await cache.set(key, json.dumps(value), ttl)
    expires_at = time.monotonic() + ttl
    await cache.set(meta_key, json.dumps({"expires_at": expires_at, "delta": delta}), ttl + 5)
    return value


# Test: with 1000 concurrent readers, DB calls should be << 1000
@pytest.mark.asyncio
async def test_xfetch_limits_db_calls():
    global _db_hits
    _db_hits = 0
    cache = InMemoryCacheBackend()
    # Warm up
    await xfetch_get(cache, "hot-key", ttl=5, beta=2.0)
    _db_hits = 0

    tasks = [xfetch_get(cache, "hot-key", ttl=5, beta=2.0) for _ in range(1000)]
    await asyncio.gather(*tasks)
    assert _db_hits < 5, f"Too many DB calls: {_db_hits}"

import pytest
```

---

## Exercise 3 — Tag-based invalidation

```python
"""
tagged_cache.py
"""

import asyncio
import json
from typing import Any
from caching_layer import InMemoryCacheBackend


class TaggedCache:
    def __init__(self, backend: InMemoryCacheBackend) -> None:
        self._cache = backend

    def _tag_key(self, tag: str) -> str:
        return f"__tag__:{tag}"

    async def set(self, key: str, value: Any, tags: list[str], ttl: int = 60) -> None:
        await self._cache.set(key, json.dumps(value), ttl)
        for tag in tags:
            tag_key = self._tag_key(tag)
            raw = await self._cache.get(tag_key)
            members: set[str] = set(json.loads(raw)) if raw else set()
            members.add(key)
            await self._cache.set(tag_key, json.dumps(list(members)), ttl + 300)

    async def get(self, key: str) -> Any | None:
        raw = await self._cache.get(key)
        return json.loads(raw) if raw else None

    async def invalidate_tag(self, tag: str) -> int:
        """Delete all keys tagged with `tag`. Returns count deleted."""
        tag_key = self._tag_key(tag)
        raw = await self._cache.get(tag_key)
        if not raw:
            return 0
        members: list[str] = json.loads(raw)
        for key in members:
            await self._cache.delete(key)
        await self._cache.delete(tag_key)
        return len(members)


# Tests
import pytest

@pytest.mark.asyncio
async def test_tag_invalidation():
    cache = TaggedCache(InMemoryCacheBackend())
    await cache.set("product:1", {"name": "Widget"}, tags=["product", "category:tools"])
    await cache.set("product:2", {"name": "Gadget"}, tags=["product", "category:tech"])
    await cache.set("category:tools:list", ["Widget"], tags=["category:tools"])

    deleted = await cache.invalidate_tag("product")
    assert deleted == 2
    assert await cache.get("product:1") is None
    assert await cache.get("product:2") is None
    # category:tools:list is still there (tagged differently)
    assert await cache.get("category:tools:list") is not None
```

---

## Exercise 4 — Eviction policy selection

**Recommendation: `allkeys-lfu`**

Social media feed items follow a Zipf distribution: a small fraction of posts (trending, viral) account for most reads. LRU evicts based on recency — a viral post that was last accessed 10 minutes ago might be evicted for a one-off request for a month-old post accessed 5 minutes ago.

LFU tracks access frequency. Viral items accumulate high frequency counts and are retained even during memory pressure. Cold items (posted once, never viewed again) are evicted quickly.

For a cold-start scenario (after a Redis restart): LFU starts with frequency=0 for all items and converges to the true distribution after ~1 cache fill cycle. The `lfu-decay-time` setting prevents stale frequency counts from blocking new trending content.

Concretely: if post A has been viewed 10,000 times in the past hour and post B was viewed 5 minutes ago once, LRU keeps B and evicts A; LFU keeps A and evicts B. LFU is correct here.
