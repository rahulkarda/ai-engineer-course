# Solutions — Day 7: Project Completion + Assessment Prep (Pointer)

Day 7 is a project finalization and review day. There is no new implementation to reveal.

## Project completion self-check

All 10 tests from Day 6 should pass. Verify:

```bash
python -m pytest tests/ -v
# Expected: 10 passed in < 2s
```

Coverage:
```bash
python -m pytest tests/ --cov=src --cov-report=term-missing
# Expected: ≥85% coverage
```

Ruff:
```bash
ruff check src/ tests/
# Expected: All checks passed.
```

## Concept review answers (brief)

**Why distribute?** Capacity (single machine ceiling), availability (SPOF elimination), latency (geo-distribution). AI examples: inference cluster (capacity), replicated vector DB (availability), regional embedding cache (latency).

**CAP placement**:
- Zookeeper: CP (refuses operations during partition to maintain consistency)
- Cassandra: AP (always accepts writes, eventually consistent)
- Redis single-node: not distributed — no P to choose
- Your KV store: AP (primary accepts writes during partition; replicas serve stale reads)

**Consistency spectrum (strongest → weakest)**: Linearisability → Sequential → Causal → Eventual
- Linearisability: global real-time ordering; cost = highest latency, synchronous
- Sequential: total order preserved per-process; cost = coordination overhead
- Causal: only causally-related events ordered; cost = vector clock overhead
- Eventual: any order until convergence; cost = conflict resolution complexity

**Read-your-own-writes mitigation**: (1) sticky session — route user's reads to primary for T seconds after write; (2) version piggybacking — attach write version to response, reader waits until replica reaches that version.

**Raft majority in 5-node cluster**: 3 of 5. Vote restriction: candidate must have `last_log_term >= voter's last_log_term`, or same term and `last_log_index >= voter's last_log_index`.

**Lamport clock on receive**: `L = max(L_local, L_received) + 1`

**Idempotent operations in the KV store**: `apply_replication` (LWW makes repeat application safe), `sync_replicas` (idempotent: applying the same entry twice is a no-op due to version check). Non-idempotent by default: `write` (without the idempotency key, calling twice with different values would write both).
