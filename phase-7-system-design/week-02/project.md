# Week 02 Project — Scalability Building Blocks

> Time: ~4 hr (Saturday build) + review during Sunday  ·  Prereqs: Days 1–7

## Overview

Build a reference implementation of two scalability building blocks:

1. A **caching layer** — cache-aside with TTL, stampede protection, and metrics.
2. A **consistent-hashing load balancer** — ring-based, virtual nodes, add/remove node.

Wrap both in a short written **design document** explaining the architecture, trade-offs, and how they would fit into a production system.

---

## Deliverables

### 1. `caching_layer.py`

A self-contained module with:

- `CacheBackend` — `Protocol` with `get`, `set`, `delete`, `set_nx`.
- `DataBackend` — `Protocol` with `fetch`, `store`.
- `InMemoryCacheBackend` — in-memory implementation for tests.
- `InMemoryDataBackend` — in-memory DB for tests.
- `CacheMetrics` — dataclass tracking `hits`, `misses`, `stampedes_averted`, `invalidations`, `hit_rate`.
- `StampedeGuard` — mutex using `set_nx`; waits with configurable retry.
- `CachingLayer` — main class wiring the above; `get(key, ttl?)` and `invalidate(key, new_value?)`.

Required behaviour:
- Concurrent calls for the same key on a cold cache → exactly one DB call.
- `hit_rate` reflects actual cache behaviour over the session.
- `invalidate(key, new_val)` updates DB and deletes cache key atomically from the caller's perspective.

### 2. `consistent_hash_lb.py`

A self-contained module with:

- `Backend` — frozen dataclass `(host, port, weight)`.
- `ConsistentHashLB` — ring with configurable `vnodes_per_backend`.
  - `add_backend(backend)` — places `vnodes * weight` virtual nodes.
  - `remove_backend(backend)` — removes all virtual nodes.
  - `get_backend(request_key)` — clockwise lookup on sorted ring.
  - `distribution()` — dict of `address → vnode_count`.
  - `remapping_fraction(keys, new_backend)` — estimates remapping % without permanently modifying the ring.

Required behaviour:
- With 3 backends and 150 vnodes each, distribution over 10,000 keys should be within 20–47% per backend.
- Adding a 4th backend remaps ≤ 30% of keys.
- Deterministic: same key, same ring state → always same backend.

### 3. Test suites

`test_caching_layer.py` (at minimum 8 tests):
- [ ] Cold miss → DB called once, value cached.
- [ ] Warm hit → DB not called.
- [ ] Stampede: 50 concurrent gets → DB called exactly once.
- [ ] TTL expiry: after TTL, next get is a miss.
- [ ] Invalidation: after invalidate, next get is a miss, new value returned.
- [ ] Metrics: hit_rate correct after N reads.
- [ ] StampedeGuard: waiters receive correct value.
- [ ] Protocol conformance: `InMemoryCacheBackend` satisfies `CacheBackend`.

`test_consistent_hash_lb.py` (at minimum 6 tests):
- [ ] Determinism: same key always maps to same backend.
- [ ] Distribution: 3 backends, 10K keys, each in [20%, 47%].
- [ ] Scale-out remapping: adding 4th backend remaps ≤ 30%.
- [ ] Scale-in: after removing a backend, all keys served by remaining backends.
- [ ] Weight: weight=2 backend handles ~2× the keys of weight=1.
- [ ] Empty ring: `get_backend` returns `None`.

### 4. `design.md`

A written design document (1–2 pages) covering:

**Architecture diagram** (Mermaid or ASCII):
```
[Client] → [LB: ConsistentHashLB] → [Backend A: CachingLayer + InMemoryDataBackend]
                                   → [Backend B: CachingLayer + InMemoryDataBackend]
                                   → [Backend C: CachingLayer + InMemoryDataBackend]
                                              ↓ miss
                                       [Shared Data Store]
```

**Data flow**:
- Read path: client key → hash ring → backend → cache lookup → (miss) DB fetch → cache set → return.
- Write path: client update → DB write → cache invalidation.
- Scale event: adding a backend → minimal key remapping → cache misses only for remapped keys.

**Trade-off analysis** (bullet points):
- Why cache-aside over write-through for this workload.
- Why consistent hashing over round-robin for a caching proxy.
- Stampede guard cost (latency for waiters) vs benefit (DB protection).
- TTL choice: how staleness tolerance affects TTL selection.
- Virtual node count: more vnodes = more uniform distribution but more memory.

**Production gaps** (what you would add for real deployment):
- Redis-backed `CacheBackend`.
- Circuit breaker on the DB.
- Prometheus metrics export.
- Distributed tracing (trace ID propagation through cache layer).
- Graceful backend removal (drain in-flight requests before remove).

### 5. `demo.py`

A runnable script demonstrating all components together. It should:
1. Build a `ConsistentHashLB` with 3 backends.
2. Give each backend its own `CachingLayer` (simulating per-node cache).
3. Send 100 requests for 10 product keys.
4. Print per-backend metrics.
5. Add a 4th backend, print remapping %.
6. Send 20 more requests and show cache miss spike for remapped keys.

---

## Acceptance criteria

| Criterion | Pass |
|-----------|------|
| All test files pass `pytest` with 0 failures | Required |
| `test_caching_layer.py` stampede test: DB called ≤ 2 times for 50 concurrent gets | Required |
| `test_consistent_hash_lb.py` remapping test: ≤ 30% keys remap on 3→4 backend scale | Required |
| `design.md` covers all five sections | Required |
| No hardcoded secrets or credentials | Required |
| All public functions/classes have type hints and docstrings | Recommended |
| `ruff check` passes with 0 errors | Recommended |

---

## Grading breakdown

| Component | Weight |
|-----------|--------|
| `caching_layer.py` correctness (stampede, TTL, metrics) | 30% |
| `consistent_hash_lb.py` correctness (distribution, remapping) | 30% |
| Test suite coverage and quality | 20% |
| `design.md` completeness and accuracy | 20% |

Pass gate: 75% overall. Both required acceptance criteria must pass.

Solution reference: ../solutions/week-02/project.md
