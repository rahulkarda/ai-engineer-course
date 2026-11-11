# Phase 7 · Week 02 · Day 7 — Project Review & Scaling Reads vs Writes

> Objective: Complete the week project, understand read/write asymmetry at scale, and review all week concepts in preparation for the assessment.  ·  Time: ~3 hr project + 1 hr review  ·  Prereqs: [[day-6]]

## Part 1: Scaling Reads vs Writes (1 hr)

Most real systems are **read-heavy**: a social platform might see 100 reads for every 1 write. This asymmetry drives architectural decisions.

### Read scaling patterns

**Read replicas**  
Writes go to a primary; reads are distributed across N read replicas. Replication lag introduces eventual consistency.

```
Application ──write──► Primary DB
Application ──read───► Replica-1
                    ──► Replica-2
                    ──► Replica-3
```

**Caching layer**  
The caching strategies from Days 3–4. Reads hit cache; writes invalidate/update cache. Can offload 80–95% of DB read load.

**CQRS (Command Query Responsibility Segregation)**  
Separate the write model (normalised, ACID) from the read model (denormalised, optimised for query). A write triggers an async event that updates the read model.

```
Write path:  API ──► Command Handler ──► Write DB ──► Event Bus
Read path:   API ──► Query Handler   ──► Read DB (pre-materialised views)
```

When useful: when read and write shapes differ dramatically (e.g., write = row mutations; read = pre-aggregated dashboard data).

### Write scaling patterns

**Write-ahead log (WAL) fanout**  
The DB's WAL is consumed by a CDC (Change Data Capture) tool (Debezium, AWS DMS) and fanned out to downstream systems.

**Sharding**  
Split data by a shard key across multiple DB instances.
- **Range sharding**: `user_id 1–1M → shard-1`, `1M–2M → shard-2`. Hot-key risk (new users → shard-N).
- **Hash sharding**: `shard = hash(user_id) % N`. Uniform distribution; range queries span shards.
- **Directory sharding**: a routing table maps keys to shards. Flexible; adds indirection.

**Write batching**  
Buffer writes in memory (or queue) and flush in bulk. Dramatically reduces DB round trips. Risk: data loss on crash.

### The read/write trade-off matrix

| Pattern | Improves | Costs |
|---------|---------|-------|
| Read replicas | Read throughput | Replication lag, operational complexity |
| Application cache | Read latency + throughput | Stale data, invalidation complexity |
| CQRS | Query flexibility | Two data models, eventual consistency |
| Sharding | Write throughput + storage | Cross-shard queries, re-sharding |
| Write batching | Write throughput | Durability risk |

---

## Part 2: Week Project Completion (2 hr)

Complete the `project.md` deliverables:

1. **Written design document** (`design.md`): architecture diagram (Mermaid or ASCII), data flow for reads and writes, trade-off analysis.

2. **`caching_layer.py`** (from Day 6): ensure stampede protection, TTL, and metrics are complete.

3. **`consistent_hash_lb.py`** (from Day 6): ensure add/remove/routing are complete.

4. **Test suites**: `test_caching_layer.py` and `test_consistent_hash_lb.py` (from Day 6 exercises).

5. **Integration demo**: a script that wires both components together — a stateless API that routes requests through the consistent-hash LB and serves responses from the caching layer.

### Integration demo scaffold

```python
"""
day7_integration.py
Wires ConsistentHashLB + CachingLayer into a simulated request pipeline.
"""

import asyncio
import logging

from consistent_hash_lb import Backend, ConsistentHashLB
from caching_layer import CachingLayer, InMemoryCacheBackend, InMemoryDataBackend

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


async def handle_request(
    request_key: str,
    lb: ConsistentHashLB,
    layers: dict[str, CachingLayer],
) -> dict | None:
    """Route request_key to a backend via LB, then serve from that backend's cache."""
    backend = lb.get_backend(request_key)
    if backend is None:
        log.error("No backend available")
        return None

    layer = layers[backend.address]
    result = await layer.get(request_key)
    log.info("key=%-20s backend=%-15s result=%s", request_key, backend.address, result)
    return result


async def main() -> None:
    # Shared data store
    shared_db = InMemoryDataBackend({
        f"product:{i}": {"id": i, "name": f"Item-{i}", "price": i * 1.5}
        for i in range(1, 20)
    })

    # Three backends, each with its own cache (simulates per-node cache)
    backends = [
        Backend("10.0.0.1", 8001),
        Backend("10.0.0.2", 8002),
        Backend("10.0.0.3", 8003),
    ]
    lb = ConsistentHashLB(vnodes_per_backend=150)
    layers: dict[str, CachingLayer] = {}
    for b in backends:
        lb.add_backend(b)
        layers[b.address] = CachingLayer(InMemoryCacheBackend(), shared_db, default_ttl=30)

    # Simulate 30 requests
    keys = [f"product:{i % 10 + 1}" for i in range(30)]
    tasks = [handle_request(k, lb, layers) for k in keys]
    await asyncio.gather(*tasks)

    # Aggregate metrics
    log.info("\n=== Backend metrics ===")
    for b in backends:
        m = layers[b.address].metrics
        log.info("%s: %s", b.address, m.to_dict())

    # Scale event: add a 4th backend
    log.info("\n=== Scale event: adding backend 10.0.0.4:8004 ===")
    new_b = Backend("10.0.0.4", 8004)
    remapped = lb.remapping_fraction(keys, new_b)
    log.info("Keys remapped by scale event: %.1f%%", remapped * 100)
    lb.add_backend(new_b)
    layers[new_b.address] = CachingLayer(InMemoryCacheBackend(), shared_db, default_ttl=30)


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Part 3: Week Review (30 min)

### Concept map

```
Scalability
├── Vertical (scale up)
│   └── ceiling → switch to horizontal
└── Horizontal (scale out)
    └── requires stateless services
        └── externalise state to Redis / DB

Traffic distribution
└── Load Balancer
    ├── L4 (TCP) — fast, content-blind
    └── L7 (HTTP) — path/header routing, SSL termination
        ├── Round-robin, Weighted, Least-conn
        └── Consistent hash → stable routing for caches

Caching
├── Layers: client → CDN → app → DB
├── Strategies: cache-aside, write-through, write-back
├── Problems: invalidation, stampede, eviction
│   ├── Invalidation: TTL, event-driven, versioned keys
│   ├── Stampede: mutex lock, XFetch, stale-while-revalidate
│   └── Eviction: LRU, LFU, random (Redis maxmemory-policy)
└── Edge: CDN + Cache-Control headers

Edge layer
├── Reverse proxy: TLS termination, compression, routing
├── Rate limiting: token bucket, sliding window
└── CDN: anycast PoPs, cache-busting, purge API

Read vs Write scaling
├── Reads: replicas, caching, CQRS read models
└── Writes: sharding, WAL/CDC, batching
```

### Common interview questions for this week

1. "How would you scale a read-heavy API from 1 to 100 servers?"
2. "What is a cache stampede and how do you prevent it?"
3. "Explain consistent hashing and why it's better than modulo for cache sharding."
4. "When would you use write-back caching, and what are the risks?"
5. "How does a CDN work, and how do you bust the cache after a deploy?"
6. "What is L4 vs L7 load balancing? When do you choose each?"

---

## Exercises

1. **CQRS sketch**: design a write and read model for a "trending products" feature. Write path: product view events. Read path: pre-aggregated top-10 list per category, updated every 60 seconds. Draw the data flow.

2. **Shard key selection**: for a messaging app with 100M users and 10B messages, propose a shard key for the `messages` table. Justify based on: query patterns (fetch messages by user, fetch thread), hot-key risk, and re-sharding complexity.

3. **Full-stack design**: design a URL shortener that can handle 10K writes/sec and 1M reads/sec. Specify: data model, caching strategy, LB algorithm, CDN usage, write path, read path. One page of prose + one architecture diagram.

4. **Week reflection**: review your `caching_layer.py` and `consistent_hash_lb.py`. For each: identify one thing you would change for production (error handling, observability, configuration) and write the improvement.

(Solutions: ../solutions/week-02/day-7.md)

## Self-check
- [ ] I can explain three patterns for scaling reads and two for scaling writes
- [ ] I can describe CQRS and when it is appropriate
- [ ] My integration demo routes requests consistently via consistent hashing
- [ ] I can answer all six interview questions above from memory

## Supplementary links
- [Martin Fowler: CQRS](https://martinfowler.com/bliki/CQRS.html)
- [Stripe: Scaling — how we do it](https://stripe.com/blog/operational-excellence-at-stripe)
- [Figma: How we scaled Figma's multiplayer infrastructure](https://www.figma.com/blog/how-figmas-multiplayer-technology-works/)
- [High Scalability: Real-world architectures](http://highscalability.com/)
