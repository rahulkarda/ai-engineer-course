# Solutions — Week 01 Project

The full reference implementation is provided in [[day-6]]. This file provides:
1. The complete reference code for all four source files (for comparison after your own attempt)
2. Explanation of design decisions
3. Expected test output

## Reference implementation summary

### Design decisions

**Last-write-wins over version vectors**

LWW was chosen for simplicity. `time.monotonic()` provides a strictly increasing float on a single machine, which is sufficient for an in-process simulation where there is no clock skew. In a real distributed system, you would use hybrid logical clocks (HLC) or version vectors to handle cross-node concurrent writes correctly.

What LWW sacrifices: if two nodes independently write to the same key during a partition (impossible in this single-leader design, but relevant in multi-leader), the write with the higher monotonic timestamp wins regardless of semantic correctness. A user's "undo" operation (lower timestamp) could be silently discarded if a concurrent write has a higher timestamp.

**Single-leader replication**

Eliminates write conflicts: only the primary accepts writes. Trade-off: the primary is a bottleneck and single point of failure for writes. The cluster becomes CP under a primary failure (cannot write until a new leader is elected).

**`frozenset` for partition pairs**

`frozenset({a, b})` is order-independent, so `partition("primary", "replica-1")` and `partition("replica-1", "primary")` refer to the same pair. Without this, you'd need to check both orderings.

**`model_copy(update={"is_stale": True})` on `ReadResult`**

Pydantic v2 models are immutable by default. `model_copy` creates a shallow copy with the specified fields overridden — no need to reconstruct the entire model. This is the idiomatic Pydantic v2 pattern.

## Expected test output

```
tests/test_replication.py::test_write_reaches_primary PASSED
tests/test_replication.py::test_successful_replication_to_both_replicas PASSED
tests/test_replication.py::test_idempotent_write_does_not_duplicate PASSED
tests/test_replication.py::test_replica_may_miss_write_when_dropped PASSED
tests/test_consistency.py::test_stale_read_eventually_consistent PASSED
tests/test_consistency.py::test_quorum_read_returns_latest PASSED
tests/test_consistency.py::test_lww_conflict_resolution PASSED
tests/test_partition.py::test_primary_still_accepts_writes_during_partition PASSED
tests/test_partition.py::test_replica_convergence_after_partition_heal PASSED
tests/test_partition.py::test_network_stats_track_drops PASSED

10 passed in 0.15s
```

## What a production system would additionally need

1. **Durable storage**: entries are lost on process restart. A real system would write to an append-only log on disk (WAL) before acknowledging the write.

2. **True async replication**: replication happens synchronously in the same call stack. A real system replicates via background threads or async I/O, with backpressure and retry queues.

3. **Leader election**: this simulation has a fixed primary. A real system needs Raft or Paxos to elect a new leader when the primary fails.

4. **Network transport**: all "nodes" are in-process. A real system uses TCP/gRPC with connection management, TLS, authentication.

5. **Observability**: no metrics, no tracing, no health checks. A production store would expose Prometheus metrics for replication lag, write latency, quorum hit rate, and partition event counts.

6. **Compaction / TTL**: the in-memory store grows forever. A real system needs compaction, TTL eviction, and bloom filters.
