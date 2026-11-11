# Week 03 Project — Data-Layer Design for a Read-Heavy Product

> Estimated time: 4–6 hr total (across the week)  ·  Checkpoint: Day 7

---

## Overview

You will produce a written design document and two reference implementation modules for the data layer of **FeedApp** — a social media feed service with user posts, likes, follows, and timeline queries. This mirrors the kind of design work expected of a senior or staff engineer before a significant infrastructure project.

The project has two deliverables:

1. **Design document** (`docs/design-doc.md`) — a written, structured explanation of your datastore choices, partitioning strategy, replication approach, and CQRS read model design.
2. **Reference implementations** — two Python modules (`sharding_router.py`, `replica_router.py`) with full pytest suites, demonstrating that your design choices are implementable.

---

## Product Context

**FeedApp** core features:

- Users can post text content (up to 500 chars)
- Users can like posts
- Users can follow other users (up to 10 000 followers per user for this exercise — no celebrity fan-out)
- The timeline shows the N most recent posts from users you follow, sorted by recency
- Profile pages show a user's own posts, sorted by recency

**Load profile:**

| Metric | Value |
|---|---|
| Daily active users | 10 million |
| Writes (posts + likes + follows) | 5 000 / s peak |
| Timeline reads | 200 000 / s peak |
| Read:write ratio | ~40:1 |
| Post retention | 24 months |
| Latency target (reads) | p99 < 150 ms |
| Latency target (writes) | p99 < 300 ms |
| Durability | No data loss on single node failure |
| Consistency | Timeline reads: eventual (≤ 2 s lag acceptable) |

---

## Requirements

### Design document

Your `docs/design-doc.md` must cover all six sections below. Each section has a minimum depth requirement.

**Section 1 — Requirements summary** (5–10 lines)

Restate the functional and non-functional requirements in your own words, as you would at the start of a system-design interview. Include:
- Write and read throughput
- Latency targets
- Consistency expectations per operation type
- Data retention

**Section 2 — Datastore selection** (table + rationale paragraph)

For each of the following use cases, name a specific datastore and give a 1–3 sentence rationale:

| Use case | Store | Rationale |
|---|---|---|
| User profiles + follow graph | ? | |
| Post content + metadata | ? | |
| Timeline cache (hot reads) | ? | |
| Like counters | ? | |
| Analytics / reporting | ? | |

Your choices must be consistent with the load profile. You must mention at least one trade-off for your primary transactional store choice.

**Section 3 — Partitioning strategy** (diagram or table + explanation)

Specify:
- Partition key(s) and why
- Sharding method (range / hash / consistent hash) and why
- Number of shards and growth headroom reasoning
- How you handle uneven distribution (if applicable)

Include a short example: "Key `user:123456` hashes to shard N because..."

**Section 4 — Replication approach** (1–2 paragraphs)

Specify:
- Number of replicas per shard
- Synchronous / asynchronous / semi-synchronous and why
- Failover mechanism and target RTO
- How replication lag interacts with your consistency requirements

**Section 5 — CQRS read model design** (diagram + schema)

Specify:
- The event(s) that trigger projection updates (e.g., `PostCreated`, `PostLiked`)
- The read model schema (table/document structure) optimised for the timeline query
- The index design for that schema
- How staleness is bounded

Include a Mermaid sequence diagram or ASCII diagram of the write path → projection → read path.

**Section 6 — Open questions and risks** (bullet list)

List at least three open questions or risks you would escalate before implementation. For each, state what decision is needed and who should make it.

---

### Reference implementations

**Module 1: `src/data_layer/sharding_router.py`**

Complete implementation of `ShardingRouter` from Day 6 with:
- Consistent hashing with configurable virtual nodes
- `add_node`, `remove_node`, `get_shard`, `get_distribution`
- All TODOs resolved — no stub code remaining

**Module 2: `src/data_layer/replica_router.py`**

Complete implementation of `ReplicaRouter` from Day 6 with:
- Lag-aware read routing with round-robin
- `mark_replica_lagging`, `mark_replica_healthy`, `route_read`, `route_write`, `status`
- All TODOs resolved — no stub code remaining

**Tests:**

- `tests/unit/test_sharding_router.py` — minimum 6 passing tests
- `tests/unit/test_replica_router.py` — minimum 6 passing tests
- `tests/integration/test_data_layer.py` — minimum 1 passing integration test covering both modules together

---

## Deliverables

```
project root/
├── docs/
│   └── design-doc.md           ← written design document (all 6 sections)
├── src/
│   └── data_layer/
│       ├── __init__.py
│       ├── sharding_router.py  ← complete ShardingRouter
│       └── replica_router.py   ← complete ReplicaRouter
├── tests/
│   ├── unit/
│   │   ├── test_sharding_router.py
│   │   └── test_replica_router.py
│   └── integration/
│       └── test_data_layer.py
├── pyproject.toml              ← project config, pytest config, ruff config
└── .env.example                ← any required env vars documented (no real values)
```

Run: `pytest tests/ -v` — all tests must pass with zero errors before submission.

---

## Evaluation criteria

| Criterion | Weight | What full marks looks like |
|---|---|---|
| Design document completeness | 30% | All 6 sections present, meets minimum depth, no hand-wavy answers |
| Design document correctness | 25% | Choices are defensible given the load profile; trade-offs acknowledged; no factual errors about the technologies chosen |
| Code correctness | 25% | Both modules fully implemented, all tests pass, consistent hash ring distributes evenly, replica router correctly filters lagging replicas |
| Test quality | 10% | Tests cover edge cases (empty ring, all replicas lagging, single-node ring, large key set); assertions are specific |
| Code quality | 10% | Type hints throughout, `logging` not `print`, no hardcoded secrets, Pydantic models where appropriate, functions < 30 lines |

**Pass threshold: 75% overall.**

---

## Hints and guidance

**On choosing a partition key**: pick one that spreads load evenly AND is always present in your most frequent query. `user_id` is a natural choice for user-centric data; consider whether your timeline query can always supply it.

**On sharding vs not sharding**: at 10 M DAU and 200 k reads/s, a single Postgres primary will struggle. Sharding is warranted. A useful heuristic: consider sharding when a single node cannot absorb write throughput even with all reasonable vertical scaling options exhausted.

**On CQRS read model schema**: the timeline query is `WHERE follower_id = ? ORDER BY created_at DESC LIMIT 20`. Design the read model so this query is a single-table B-tree scan — no joins, no subqueries. Fan-out on write (precompute timelines per follower at write time) is one approach.

**On replication lag and consistency**: the product says ≤ 2 s lag is acceptable on timelines. That means asynchronous replication is fine for the read path, but your write path (post durability) still needs at least one synchronous acknowledgement.

**On tests**: the most common gap is testing only the happy path. Explicitly test: empty ring → error, all replicas lagging → leader fallback, key determinism across router instances, and removal of a non-existent node → error.

**On the design doc tone**: write as if handing this to a team of three engineers who will implement it next week. Be specific enough that they don't need to make major architectural decisions themselves.

---

## Example design doc structure (abbreviated)

```markdown
# Data-Layer Design — FeedApp

## 1. Requirements summary
Peak write load: 5 000/s, peak read load: 200 000/s (40:1 ratio).
Timeline reads tolerate ≤ 2 s staleness; write durability is non-negotiable.
Data retained for 24 months; p99 read latency target is 150 ms.

## 2. Datastore selection
| Use case        | Store      | Rationale                                   |
|-----------------|------------|---------------------------------------------|
| User profiles   | PostgreSQL | ACID writes; follow graph fits in relational |
| Post content    | PostgreSQL | Structured, needs full-text search option   |
| Timeline cache  | Redis      | Sorted Set gives O(log N) range by score    |
| Like counters   | Redis      | INCR is atomic and fast; eventual write-back|
| Analytics       | ClickHouse | Column store for aggregate queries          |

Trade-off: PostgreSQL requires careful index tuning at this scale.
We accept that complexity in exchange for strong consistency on writes.

## 3. Partitioning strategy
Partition key: user_id (present in all core queries, high cardinality).
Method: consistent hashing, 150 virtual nodes, 16 shards.
16 shards supports ~8× current write throughput before reshard.

Key user:123456 → MD5 → 0x7A3F... → ring position → shard-11

...
```

Use this as a structural reference only — your document must reflect your own reasoning.
