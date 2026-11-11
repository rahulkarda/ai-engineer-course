# Solutions — Week 02 Project

## Reference implementation overview

The following files constitute the complete reference solution. They are functionally complete and test-passing. Learner implementations that satisfy the acceptance criteria are equally valid — the goal is the behaviour, not the exact structure.

---

## caching_layer.py (reference)

See `solutions/week-02/day-6.md` for the full `test_caching_layer.py` suite. The `caching_layer.py` module from `week-02/day-6.md` (the 4-hr build session) is the reference implementation. Key design decisions:

- **Protocol-based injection**: `CacheBackend` and `DataBackend` are `typing.Protocol` types. This allows swapping `InMemoryCacheBackend` for `RedisCacheBackend` without changing `CachingLayer`.
- **StampedeGuard uses asyncio.Event-style locking** in the `SafeCache` variant (assessment solution) or Redis SET NX in the `CachingLayer` variant. Both are correct — choose based on whether Redis is available.
- **CacheMetrics** is a simple dataclass, not Prometheus; for production, swap counters for `prometheus_client.Counter`.

---

## consistent_hash_lb.py (reference)

The `ConsistentHashLB` from `week-02/day-6.md` is the reference. Key design decisions:

- **SHA-256 truncated to 32 bits**: gives a well-distributed ring; collision probability is negligible with ≤ 150 × N virtual nodes.
- **Virtual node count = 150 × weight**: empirically 150 vnodes gives ≤ 5% standard deviation in key distribution for 3–10 backends. 50 vnodes is adequate if memory is a constraint.
- **`remapping_fraction` does not permanently mutate the ring**: it adds the new backend, measures, then removes it. Useful for pre-flight capacity planning.

---

## design.md (reference)

```markdown
# Week 02 Design Document: Caching Layer + Consistent-Hash Load Balancer

## Architecture

```
[Client requests]
      │
      ▼
[ConsistentHashLB]  ──hash(request_key)──►  [Backend A]
                                         ►  [Backend B]
                                         ►  [Backend C]
                    (each backend has its own CachingLayer)
                              │
                    ┌─────────▼──────────┐
                    │   CachingLayer     │
                    │  ┌──────────────┐  │
                    │  │ CacheBackend │  │ ◄── InMemory or Redis
                    │  └──────┬───────┘  │
                    │         │ miss     │
                    │  ┌──────▼───────┐  │
                    │  │ DataBackend  │  │ ◄── DB / service
                    │  └──────────────┘  │
                    └────────────────────┘
```

## Data flow

**Read path**:
1. Client sends `GET /product/42`.
2. `ConsistentHashLB.get_backend("product:42")` deterministically selects Backend-B.
3. Backend-B's `CachingLayer.get("product:42")` checks its `CacheBackend`.
4. **Cache hit**: return serialised value directly.
5. **Cache miss**: `StampedeGuard.acquire("product:42")` → only one coroutine queries `DataBackend`; concurrent requests wait for the Event/lock to release.
6. Value stored in cache with TTL, returned to all waiters.

**Write path**:
1. Client sends `PUT /product/42 {new_data}`.
2. Any backend may handle the write.
3. `CachingLayer.invalidate("product:42", new_value=new_data)` → writes to `DataBackend`, deletes cache key.
4. Next read is a miss → re-populates from `DataBackend`.

**Scale event (add Backend-D)**:
1. `ConsistentHashLB.add_backend(Backend-D)` inserts virtual nodes.
2. ~25% of keys now route to Backend-D (new backend gets ideal share).
3. Those keys experience one cache miss on first request → DB fetch → populate Backend-D's cache.
4. Other 75% of keys are unaffected — no cache disruption.

## Trade-off analysis

- **Cache-aside over write-through**: this workload is read-heavy. Write-through keeps cache warm but adds latency to every write and stores data that may never be read again. Cache-aside only caches what is actually read.

- **Consistent hashing over round-robin for caching proxy**: with round-robin, the same product key routes to all three backends. Each backend independently misses on first access → 3 DB calls. With consistent hashing, product:42 always routes to Backend-B. After the first miss, Backend-B's cache handles all subsequent requests for that key.

- **Stampede guard cost vs benefit**: waiters add ~50ms latency per request during a cold start for a given key. Without protection, 50 concurrent misses could each hit the DB (10–100× cost). The trade-off is correct for all but the most latency-sensitive read paths.

- **TTL selection**: product prices change at most once per hour → TTL=300s is safe. For user-specific data (recommended products), TTL=30s balances freshness and hit rate. For static reference data (country codes), TTL=86400s.

- **Virtual node count**: 150 vnodes/backend is a reasonable default. Increase to 300+ for > 10 backends or when you need very tight distribution bounds. Lower to 50 for memory-constrained environments; acceptable with ≥ 5 backends.

## Production gaps

1. **Redis-backed CacheBackend**: `RedisCacheBackend` (see day-6 solutions) replaces `InMemoryCacheBackend` for cross-replica cache sharing.
2. **Circuit breaker on DataBackend**: if DB returns 5xx consecutively, stop sending requests for 30s and return a cached stale value or a graceful error.
3. **Prometheus metrics**: `CacheMetrics` counters → `prometheus_client.Counter`; expose `/metrics` endpoint.
4. **Distributed tracing**: inject `trace_id` at entry, propagate through `CachingLayer.get()` and log at each step.
5. **Graceful backend removal**: before `ConsistentHashLB.remove_backend()`, drain in-flight requests with a timeout.
6. **Ring state synchronisation**: for multi-process deployments, persist ring snapshot to Redis/etcd so all instances share the same view.
```

---

## Acceptance criteria checklist

| Criterion | Notes |
|-----------|-------|
| All tests pass | Run `pytest test_caching_layer.py test_consistent_hash_lb.py -v` |
| Stampede: DB called ≤ 2 times for 50 concurrent gets | asyncio.Event or Redis SET NX |
| Scale remapping ≤ 30% for 3→4 backend | Verified by `test_scale_out_remapping` |
| design.md covers all five sections | Architecture, flows, trade-offs, TTL, production gaps |
| No hardcoded secrets | All config via `os.getenv` |
| Type hints on all public APIs | Enforced by `ruff check --select ANN` |

---

## Common mistakes and corrections

**Mistake**: using `asyncio.sleep(0)` as stampede retry instead of waiting for a real signal.  
**Correction**: use `asyncio.Event.wait()` or poll with exponential backoff up to a timeout.

**Mistake**: consistent hash ring key derived from `str(backend)` (default dataclass repr). If `__repr__` changes, all ring positions change.  
**Correction**: use the explicit `f"{host}:{port}#vnode{i}"` string, not `repr`.

**Mistake**: `set_nx` returning `True` when the key already exists (logic inversion).  
**Correction**: Redis `SET NX` returns `OK` (truthy) on success, `None` on failure. Map `None → False`.

**Mistake**: TTL of 0 in tests causing immediate expiry on every access.  
**Correction**: for stampede tests, use TTL ≥ 10s; test TTL expiry separately with TTL=1 and `asyncio.sleep(1.1)`.
