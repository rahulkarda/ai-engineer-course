# Phase 7 · Week 02 · Day 6 — Build: Caching Layer + Consistent-Hash LB (4-hr session)

> Objective: Build a production-quality cache-aside layer with TTL and stampede protection, and a consistent-hashing load balancer, both with full test suites.  ·  Time: ~4 hr  ·  Prereqs: [[day-1]]–[[day-5]]

## Overview

Today you build the two core reference implementations for this week's project:

1. **`CachingLayer`** — cache-aside with TTL, mutex-based stampede protection, and metric tracking.
2. **`ConsistentHashLB`** — consistent hashing ring with configurable virtual nodes, supporting add/remove node with minimal key remapping.

Both become the foundation of the week project (`project.md`).

---

## Part 1: Caching Layer (2 hr)

### Design goals
- Drop-in async cache-aside for any key-value lookup.
- TTL-per-key support.
- Stampede protection via Redis SET NX lock.
- Metrics: hit_rate, miss_rate, stampede_averted_count.
- Clean interfaces for testability (injectable DB and cache backends).

### Architecture

```
CachingLayer
├── CacheBackend (Protocol)         ← Redis implementation + in-memory mock
├── DataBackend (Protocol)          ← DB simulation / real DB calls
├── StampedeGuard                   ← SET NX lock + retry
└── CacheMetrics                    ← counters, exported as dict
```

### Implementation

```python
"""
caching_layer.py
Production-style cache-aside with stampede protection.
Usage: python -I caching_layer.py
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_TTL = int(os.getenv("CACHE_TTL_SECONDS", "60"))
LOCK_TTL = int(os.getenv("CACHE_LOCK_TTL_SECONDS", "10"))


# ---------------------------------------------------------------------------
# Protocols — define the interface, not the implementation
# ---------------------------------------------------------------------------

@runtime_checkable
class CacheBackend(Protocol):
    async def get(self, key: str) -> str | None: ...
    async def set(self, key: str, value: str, ttl: int) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def set_nx(self, key: str, value: str, ttl: int) -> bool: ...


@runtime_checkable
class DataBackend(Protocol):
    async def fetch(self, key: str) -> Any | None: ...
    async def store(self, key: str, value: Any) -> None: ...


# ---------------------------------------------------------------------------
# In-memory implementations for testing / standalone demo
# ---------------------------------------------------------------------------

@dataclass
class _Entry:
    value: str
    expires_at: float


class InMemoryCacheBackend:
    """Thread-safe (asyncio) in-memory cache backend."""

    def __init__(self) -> None:
        self._store: dict[str, _Entry] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> str | None:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None or time.monotonic() > entry.expires_at:
                if entry:
                    del self._store[key]
                return None
            return entry.value

    async def set(self, key: str, value: str, ttl: int) -> None:
        async with self._lock:
            self._store[key] = _Entry(value=value, expires_at=time.monotonic() + ttl)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def set_nx(self, key: str, value: str, ttl: int) -> bool:
        async with self._lock:
            entry = self._store.get(key)
            if entry and time.monotonic() <= entry.expires_at:
                return False  # key exists
            self._store[key] = _Entry(value=value, expires_at=time.monotonic() + ttl)
            return True


class InMemoryDataBackend:
    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = data or {}
        self.call_count = 0
        self._delay = 0.02  # simulate 20 ms DB latency

    async def fetch(self, key: str) -> Any | None:
        self.call_count += 1
        await asyncio.sleep(self._delay)
        return self._data.get(key)

    async def store(self, key: str, value: Any) -> None:
        self._data[key] = value


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class CacheMetrics:
    hits: int = 0
    misses: int = 0
    stampedes_averted: int = 0
    invalidations: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "stampedes_averted": self.stampedes_averted,
            "invalidations": self.invalidations,
        }


# ---------------------------------------------------------------------------
# StampedeGuard
# ---------------------------------------------------------------------------

class StampedeGuard:
    def __init__(
        self,
        cache: CacheBackend,
        lock_ttl: int = LOCK_TTL,
        retry_interval: float = 0.05,
        max_retries: int = 20,
    ) -> None:
        self._cache = cache
        self._lock_ttl = lock_ttl
        self._retry_interval = retry_interval
        self._max_retries = max_retries

    def _lock_key(self, key: str) -> str:
        return f"__lock__:{key}"

    async def acquire(self, key: str) -> bool:
        lkey = self._lock_key(key)
        return await self._cache.set_nx(lkey, "1", self._lock_ttl)

    async def release(self, key: str) -> None:
        await self._cache.delete(self._lock_key(key))

    async def wait_for_value(self, key: str) -> str | None:
        for _ in range(self._max_retries):
            await asyncio.sleep(self._retry_interval)
            val = await self._cache.get(key)
            if val is not None:
                return val
        return None  # timed out waiting


# ---------------------------------------------------------------------------
# CachingLayer — the main API
# ---------------------------------------------------------------------------

class CachingLayer:
    def __init__(
        self,
        cache: CacheBackend,
        db: DataBackend,
        default_ttl: int = DEFAULT_TTL,
    ) -> None:
        self._cache = cache
        self._db = db
        self._ttl = default_ttl
        self.metrics = CacheMetrics()
        self._guard = StampedeGuard(cache)

    async def get(self, key: str, ttl: int | None = None) -> Any | None:
        effective_ttl = ttl if ttl is not None else self._ttl

        # 1. Cache lookup
        raw = await self._cache.get(key)
        if raw is not None:
            self.metrics.hits += 1
            return json.loads(raw)

        self.metrics.misses += 1

        # 2. Try to acquire the population lock
        lock_acquired = await self._guard.acquire(key)

        if not lock_acquired:
            # Another coroutine is populating — wait for it
            self.metrics.stampedes_averted += 1
            log.debug("Stampede guard: waiting for key=%s to be populated", key)
            raw = await self._guard.wait_for_value(key)
            if raw is not None:
                return json.loads(raw)
            # Fallback: try DB directly if wait timed out
            log.warning("Lock wait timed out for key=%s, falling back to DB", key)

        # 3. Fetch from DB and populate cache
        try:
            value = await self._db.fetch(key)
            if value is not None:
                serialised = json.dumps(value)
                await self._cache.set(key, serialised, effective_ttl)
                log.debug("Populated cache key=%s ttl=%d", key, effective_ttl)
            return value
        finally:
            if lock_acquired:
                await self._guard.release(key)

    async def invalidate(self, key: str, new_value: Any | None = None) -> None:
        """Invalidate cache on write. Optionally update DB with new_value."""
        if new_value is not None:
            await self._db.store(key, new_value)
        await self._cache.delete(key)
        self.metrics.invalidations += 1
        log.debug("Invalidated key=%s", key)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

async def main() -> None:
    db = InMemoryDataBackend({
        "product:1": {"id": 1, "name": "Widget", "price": 9.99},
        "product:2": {"id": 2, "name": "Gadget", "price": 24.99},
    })
    cache = InMemoryCacheBackend()
    layer = CachingLayer(cache, db, default_ttl=10)

    log.info("=== Sequential reads (warm-up) ===")
    for key in ["product:1", "product:2", "product:1", "product:2"]:
        result = await layer.get(key)
        log.info("key=%s result=%s", key, result)

    log.info("=== Concurrent reads (stampede test) ===")
    tasks = [layer.get("product:1") for _ in range(20)]
    results = await asyncio.gather(*tasks)
    assert all(r == {"id": 1, "name": "Widget", "price": 9.99} for r in results)

    log.info("=== Metrics: %s ===", layer.metrics.to_dict())
    log.info("DB call count: %d (should be 2 for 2 unique keys)", db.call_count)

    log.info("=== Invalidation ===")
    await layer.invalidate("product:1", new_value={"id": 1, "name": "Widget v2", "price": 10.99})
    result = await layer.get("product:1")
    log.info("After invalidation: %s", result)

    log.info("Final metrics: %s", layer.metrics.to_dict())


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Part 2: Consistent-Hash Load Balancer (2 hr)

### Design goals
- Ring-based consistent hashing with virtual nodes.
- Add/remove backend with minimal key remapping (≤ 2/N fraction).
- Thread-safe (asyncio).
- Observable: dump ring state, show which keys map to which backend.

### Implementation

```python
"""
consistent_hash_lb.py
Consistent-hashing load balancer with virtual nodes.
"""

import bisect
import hashlib
import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass(frozen=True)
class Backend:
    host: str
    port: int
    weight: int = 1  # multiplier for virtual node count

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


class ConsistentHashLB:
    """
    Ring-based consistent hash load balancer.

    Virtual nodes: each backend gets `vnodes_per_backend * weight` positions
    on the hash ring. More virtual nodes = more uniform distribution.
    """

    def __init__(self, vnodes_per_backend: int = 150) -> None:
        self._vnodes_per_backend = vnodes_per_backend
        self._ring: dict[int, Backend] = {}      # hash_point → backend
        self._sorted_keys: list[int] = []         # sorted hash points
        self._backends: set[Backend] = set()

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(key: str) -> int:
        """SHA-256 → 32-bit unsigned int."""
        digest = hashlib.sha256(key.encode()).digest()
        return int.from_bytes(digest[:4], "big")

    # ------------------------------------------------------------------
    # Ring management
    # ------------------------------------------------------------------

    def add_backend(self, backend: Backend) -> None:
        if backend in self._backends:
            log.warning("Backend %s already on ring", backend.address)
            return
        vnode_count = self._vnodes_per_backend * backend.weight
        for i in range(vnode_count):
            vnode_key = f"{backend.address}#vnode{i}"
            point = self._hash(vnode_key)
            self._ring[point] = backend
        self._sorted_keys = sorted(self._ring.keys())
        self._backends.add(backend)
        log.info("Added %s with %d virtual nodes", backend.address, vnode_count)

    def remove_backend(self, backend: Backend) -> None:
        if backend not in self._backends:
            log.warning("Backend %s not found on ring", backend.address)
            return
        vnode_count = self._vnodes_per_backend * backend.weight
        for i in range(vnode_count):
            vnode_key = f"{backend.address}#vnode{i}"
            point = self._hash(vnode_key)
            self._ring.pop(point, None)
        self._sorted_keys = sorted(self._ring.keys())
        self._backends.discard(backend)
        log.info("Removed %s", backend.address)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def get_backend(self, request_key: str) -> Backend | None:
        if not self._ring:
            return None
        point = self._hash(request_key)
        # Find first ring position >= point (clockwise)
        idx = bisect.bisect_left(self._sorted_keys, point)
        if idx == len(self._sorted_keys):
            idx = 0  # wrap around
        return self._ring[self._sorted_keys[idx]]

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def distribution(self) -> dict[str, int]:
        """Count virtual nodes per backend address."""
        counts: dict[str, int] = {}
        for backend in self._ring.values():
            counts[backend.address] = counts.get(backend.address, 0) + 1
        return counts

    def remapping_fraction(
        self,
        keys: list[str],
        new_backend: Backend,
    ) -> float:
        """
        Estimate what fraction of `keys` would remap if `new_backend` is added.
        Returns the fraction that changes assignment.
        """
        before = {k: b.address for k in keys if (b := self.get_backend(k))}
        self.add_backend(new_backend)
        after = {k: b.address for k in keys if (b := self.get_backend(k))}
        self.remove_backend(new_backend)
        changed = sum(1 for k in keys if before.get(k) != after.get(k))
        return changed / len(keys) if keys else 0.0


def demo() -> None:
    lb = ConsistentHashLB(vnodes_per_backend=150)
    backends = [
        Backend("10.0.0.1", 8001),
        Backend("10.0.0.2", 8002),
        Backend("10.0.0.3", 8003),
    ]
    for b in backends:
        lb.add_backend(b)

    log.info("Ring distribution: %s", lb.distribution())

    # Route 1000 sample keys
    test_keys = [f"user:{i}" for i in range(1000)]
    assignments: dict[str, int] = {}
    for key in test_keys:
        backend = lb.get_backend(key)
        if backend:
            assignments[backend.address] = assignments.get(backend.address, 0) + 1
    log.info("Key distribution across backends: %s", assignments)

    # Measure remapping on scale event
    new_b = Backend("10.0.0.4", 8004)
    fraction = lb.remapping_fraction(test_keys, new_b)
    log.info(
        "Adding backend: %.1f%% of keys remap (expected ~%.1f%%)",
        fraction * 100,
        100 / (len(backends) + 1),
    )

    # Remove a backend — surviving backends absorb its keys
    lb.remove_backend(backends[1])
    log.info("After removing %s:", backends[1].address)
    for key in ["user:42", "user:99", "user:500"]:
        log.info("  %s → %s", key, lb.get_backend(key))


if __name__ == "__main__":
    demo()
```

---

## Exercises

1. **Test suite**: write a `pytest` test file (`test_caching_layer.py`) covering at minimum:
   - Cache miss → DB is called once, result is cached.
   - Cache hit → DB is NOT called.
   - Stampede: 50 concurrent `get` calls for the same key → DB is called exactly once.
   - Invalidation: after `invalidate(key)`, next `get` is a miss.
   - TTL: after TTL expires, next `get` is a miss.

2. **Consistent hash tests**: write `test_consistent_hash_lb.py` covering:
   - With N=3 backends and 10,000 keys, each backend handles 25–42% of keys (uniform enough).
   - Adding a 4th backend remaps ≤ 30% of keys.
   - Removing a backend: all keys previously served by it are now served by surviving backends.
   - Same key always maps to the same backend (determinism).

3. **Weighted consistent hashing**: extend `ConsistentHashLB` so that a backend with `weight=2` handles approximately twice the keys of a `weight=1` backend. Verify with a test using 10,000 keys.

4. **Redis CacheBackend**: implement a `RedisCacheBackend` that satisfies `CacheBackend` protocol using `redis.asyncio`. The `set_nx` method must use `redis.set(key, value, nx=True, ex=ttl)`. Make `CachingLayer` work with both `InMemoryCacheBackend` and `RedisCacheBackend`.

(Solutions: ../solutions/week-02/day-6.md)

## Self-check
- [ ] My `CachingLayer` implementation passes all five stampede test cases
- [ ] My `ConsistentHashLB` remaps ≤ 30% on a 3→4 backend scale event
- [ ] Both implementations use `Protocol`-based dependency injection (no hard Redis dependency in logic)
- [ ] Test coverage ≥ 80% on both modules
