# Week 01 Project — Distributed Key-Value Store

> Phase 6 · Week 01 · Build project  ·  Due: end of Day 7  ·  Time: ~4 hr build + 1 hr tests

## Goal

Build and test a complete in-process simulation of a 3-node distributed key-value store that demonstrates the core trade-offs covered this week: replication lag, eventual consistency, partition handling, last-write-wins conflict resolution, and idempotent writes.

This is the foundation for the Phase 6 portfolio piece: *Production AI Platform*. You will extend this store in later weeks with observability, circuit breakers, and eventually a real network transport.

## Deliverables

```
week-01-kvstore/
├── src/
│   ├── __init__.py
│   ├── models.py          # Entry, WriteRequest, ReadResult, ReplicationMessage
│   ├── network_sim.py     # NetworkSim (drop_rate, partitions, stats)
│   ├── node.py            # KVNode (idempotent writes, LWW replication)
│   └── cluster.py         # Cluster (write, read, quorum_read, sync_replicas)
├── tests/
│   ├── __init__.py
│   ├── test_replication.py    # 4 tests
│   ├── test_consistency.py    # 3 tests
│   └── test_partition.py      # 3 tests
├── scripts/
│   └── smoke.py
└── pyproject.toml
```

Total: 10 tests minimum (more is better), all green.

## Requirements

### Functional requirements

1. **Write** — `cluster.write(key, value, idempotency_key=None)` writes to the primary and attempts async replication to both replicas. Returns the committed `Entry`.

2. **Idempotent write** — calling `write` with the same `idempotency_key` a second time returns the original entry without modifying the store.

3. **Stale read** — `cluster.read(key, node="replica-1")` may return `None` or an older value if replication was dropped. `is_stale=True` on the result.

4. **Quorum read** — `cluster.quorum_read(key)` reads from all three nodes and returns the entry with the highest version. Always returns the latest committed write (since primary always has it).

5. **Partition** — `network_sim.partition(a, b)` prevents all replication messages between node a and node b. Writes to primary still succeed (AP under partition).

6. **Heal + sync** — `cluster.sync_replicas()` force-applies all primary entries to replicas using LWW. After sync, all nodes agree.

7. **LWW conflict resolution** — if a replica receives a replication message with a version older than its current entry for that key, it discards the message.

8. **Stats** — `network_sim.stats` returns `{"delivered": N, "dropped": M}` accurately counting replication messages.

### Non-functional requirements

- Python 3.12+, type hints everywhere
- Pydantic v2 models for all inter-module data
- `logging` only — no `print()`
- `ruff check` passes clean
- No real network I/O, no threading — pure in-process simulation

## Grading criteria

| Criterion | Points |
|---|---|
| All 10 tests pass | 30 |
| LWW correctly prevents old entries overwriting new entries (test) | 15 |
| Partition test: primary available, replicas stale | 15 |
| Quorum read returns latest despite partial replication failure | 15 |
| Idempotency test: function called once for duplicate key | 10 |
| Code quality (types, logging, Pydantic, ruff-clean) | 10 |
| Smoke script runs without assertion errors | 5 |
| **Total** | **100** |

Pass threshold: 75 points.

## Hints

- `time.monotonic()` gives a monotonically increasing float — suitable as a LWW version. Don't use `time.time()` (can go backwards on clock adjustment).
- Pydantic's `model_copy(update={...})` is useful when you need a modified copy of an immutable model (e.g., setting `is_stale=True` on a `ReadResult`).
- Use `frozenset({a, b})` as a partition pair key — sets are unordered, so `{a, b} == {b, a}`.
- For `quorum_read`, handle the case where all nodes return `None` (key doesn't exist).

## Stretch goals

Complete the core first. If time allows:

1. **Read-your-own-writes**: after a successful write, any subsequent read (on any node) within 2 s must return that version or newer. Implement and test.
2. **Leader election stub**: when the primary is partitioned from both replicas, promote `replica-1` to acting primary. Writes route to it. When the partition heals, reconcile using LWW.
3. **Metrics endpoint**: add a `cluster.metrics() -> dict` method that returns total write count, average replication success rate, and current partition pairs.

## Reference

Full implementation guide with step-by-step code: [[day-6]]
Solution: ../solutions/week-01/project.md
