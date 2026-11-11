# Solutions — Week 02 Assessment

## Conceptual answers

**1. Stateful replicas**

The `_sessions` dict lives in each process's memory. Login on replica-A stores the token there. A subsequent request routed to replica-B or -C has no knowledge of that token → 401. The minimal fix: externalise sessions to Redis. Use `redis.setex(f"session:{token}", TTL, user_id)` on login; `redis.get(f"session:{token}")` on every authenticated request. Any replica can now verify any token.

**2. L4 vs L7**

L4 routes based on IP/TCP metadata only — faster (no HTTP parsing), but content-blind. L7 parses the full HTTP request, enabling path/host routing, header inspection, and SSL termination. Use L4 for non-HTTP protocols (database proxies, raw TCP services, game servers) where the overhead of HTTP parsing adds nothing. Use L7 whenever you need path-based routing, A/B testing, or SSL offload.

**3. Cache stampede + SET NX fix**

Without protection: all 1000 requests see `cache.get()` return None. Each fires a DB query. The DB receives 1000× its normal load — it may slow down, queue up, or fail. With SET NX: the first request atomically sets a lock key (`SET lock:key 1 NX EX 5`). It receives `True` → proceeds to DB. The other 999 receive `False` → sleep briefly and retry `cache.get()`. By the time they retry, the first request has populated the cache → 999 cache hits, 1 DB call.

**4. LRU vs LFU for Zipf workloads**

After warm-up, **LFU** outperforms LRU. With Zipf distribution, the top 20% of keys account for 80% of reads. LRU can evict a hot key if it wasn't accessed in the last window — a brief lull in one key's access can displace it. LFU retains keys proportional to their historical access frequency. The hot 20% accumulate high frequency counts and are effectively pinned in the cache. LFU converges to a near-optimal cache once frequency counts have stabilised.

**5. Consistent hashing: 3 → 4 backends**

Modulo hashing (`hash(key) % 3` → `hash(key) % 4`): almost every key remaps — a key that resolved to bucket 2 under %3 may resolve to any bucket under %4. In practice ~75% of keys remap.

Consistent hashing: place 3 backends on a ring covering integers [0, 2^32). Adding a 4th backend inserts its virtual nodes at positions distributed around the ring. Only the keys that fall between each new node and its clockwise predecessor remap. In expectation, `10,000 / (3+1) = 2,500 keys` remap — 25%. Virtual nodes tighten this toward the ideal.

**6. Write-back for order management**

Risk: a crash between writing to cache and flushing to DB loses order data — financial and legal liability. Mitigation required: the write-back queue must be a durable log (Redis Streams with AOF enabled, or a proper message queue like Kafka). The flush consumer must write to DB with idempotency (use an order UUID as the idempotency key). Only acknowledge an order as "created" to the client after the durable log write, not the cache write. Consider whether write-back is appropriate at all for financial data — write-through or synchronous DB write may be safer.

**7. CDN stale after bug fix**

Option A — **Purge the CDN cache**: use the CDN provider's purge API to evict `app.js` immediately. Trade-off: fast (< 1 min), but requires API access in the deploy pipeline. All edge nodes discard the cache → a burst of origin fetches (thundering herd on the origin). For a single file, manageable.

Option B — **Cache-busting filename**: redeploy with `app.a4g9c1.js` (new content hash). The HTML references the new filename. Old `app.a3f8b2.js` entries expire naturally. Trade-off: no immediate fix for users who already have the old HTML cached — they reference the old URL. But if HTML has a short TTL (or `no-cache`), this self-heals quickly. Best long-term practice.

---

## Coding challenge solutions

### Challenge A — SafeCache

```python
"""
safe_cache.py
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger(__name__)


@runtime_checkable
class DataBackend(Protocol):
    async def fetch(self, key: str) -> Any | None: ...


@dataclass
class _CacheEntry:
    value: str
    expires_at: float


class SafeCache:
    def __init__(self, db: DataBackend, default_ttl: int = 60) -> None:
        self._db = db
        self._ttl = default_ttl
        self._store: dict[str, _CacheEntry] = {}
        self._locks: dict[str, asyncio.Event] = {}
        self._hits = 0
        self._misses = 0

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    def _is_valid(self, entry: _CacheEntry) -> bool:
        return time.monotonic() < entry.expires_at

    async def get(self, key: str) -> Any | None:
        # Fast path: cache hit
        entry = self._store.get(key)
        if entry and self._is_valid(entry):
            self._hits += 1
            return json.loads(entry.value)

        self._misses += 1

        # Stampede guard: if another coroutine is already populating, wait
        if key in self._locks:
            event = self._locks[key]
            await event.wait()
            entry = self._store.get(key)
            if entry and self._is_valid(entry):
                self._hits += 1
                self._misses -= 1  # correct the double-count
                return json.loads(entry.value)

        # Acquire population lock
        event = asyncio.Event()
        self._locks[key] = event
        try:
            value = await self._db.fetch(key)
            if value is not None:
                serialised = json.dumps(value)
                self._store[key] = _CacheEntry(
                    value=serialised,
                    expires_at=time.monotonic() + self._ttl,
                )
            return value
        finally:
            event.set()
            self._locks.pop(key, None)

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        serialised = json.dumps(value)
        self._store[key] = _CacheEntry(
            value=serialised,
            expires_at=time.monotonic() + (ttl or self._ttl),
        )

    async def invalidate(self, key: str, new_value: Any | None = None) -> None:
        self._store.pop(key, None)
        if new_value is not None:
            await self._db.store(key, new_value)  # type: ignore[attr-defined]
```

### Challenge B — HashRing

```python
"""
hash_ring.py
"""

import bisect
import hashlib


class HashRing:
    def __init__(self, vnodes_per_node: int = 150) -> None:
        self._vnodes = vnodes_per_node
        self._ring: dict[int, str] = {}
        self._keys: list[int] = []
        self._nodes: dict[str, int] = {}  # node_id → weight

    @staticmethod
    def _hash(key: str) -> int:
        digest = hashlib.sha256(key.encode()).digest()
        return int.from_bytes(digest[:4], "big")

    def add_node(self, node_id: str, weight: int = 1) -> None:
        if node_id in self._nodes:
            return
        self._nodes[node_id] = weight
        vnode_count = self._vnodes * weight
        for i in range(vnode_count):
            point = self._hash(f"{node_id}#vnode{i}")
            self._ring[point] = node_id
        self._keys = sorted(self._ring.keys())

    def remove_node(self, node_id: str) -> None:
        if node_id not in self._nodes:
            return
        weight = self._nodes.pop(node_id)
        for i in range(self._vnodes * weight):
            point = self._hash(f"{node_id}#vnode{i}")
            self._ring.pop(point, None)
        self._keys = sorted(self._ring.keys())

    def get_node(self, key: str) -> str | None:
        if not self._ring:
            return None
        point = self._hash(key)
        idx = bisect.bisect_left(self._keys, point)
        if idx == len(self._keys):
            idx = 0
        return self._ring[self._keys[idx]]
```

---

## Debugging challenge solutions

**Bug 1: `_sessions = {}` — in-process state**

Fails under horizontal scaling because each replica maintains its own `_sessions` dict. A login on replica-1 is invisible to replica-2.

Fix: replace with Redis:
```python
r = aioredis.from_url(os.getenv("REDIS_URL"), decode_responses=True)

@app.post("/login")
async def login(user_id: str):
    token = secrets.token_hex(16)
    await r.setex(f"session:{token}", 3600, user_id)
    return {"token": token}

@app.get("/me")
async def me(x_token: str = Header()):
    uid = await r.get(f"session:{x_token}")
    if not uid:
        raise HTTPException(401)
    return {"user_id": uid}
```

**Bug 2: `_sessions.clear()` — clears ALL sessions**

`logout` wipes every user's session, not just the caller's. Even within a single replica this is wrong; under multi-replica deployment it only affects the replica that receives the logout request anyway.

Fix:
```python
@app.delete("/logout")
async def logout(x_token: str = Header()):
    await r.delete(f"session:{x_token}")
    return {"ok": True}
```

**Bug 3: `CACHE = {}` — per-replica in-memory cache**

Each replica has its own `CACHE` dict. A cache miss on one replica populates that replica's local cache; other replicas still miss and hit the DB. Under load balancing, the same product may be fetched from DB once per replica — no cross-replica sharing, and the cache is lost on process restart.

Fix: move to a shared Redis cache (cache-aside pattern):
```python
@app.get("/products/{pid}")
async def get_product(pid: str):
    cached = await r.get(f"product:{pid}")
    if cached:
        return json.loads(cached)
    result = fetch_from_db(pid)
    await r.setex(f"product:{pid}", 300, json.dumps(result))
    return result
```
