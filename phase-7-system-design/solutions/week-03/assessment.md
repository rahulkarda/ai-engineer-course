# Solutions — Phase 7 · Week 3 · Assessment — Data Layer at Scale

---

## Conceptual Questions

### Q1. A candidate says: "I'll just use MongoDB because it scales horizontally." What follow-up questions would you ask?

**Model answer:**

Before accepting or rejecting the choice, probe the access patterns and consistency requirements. Ask: What are the primary query patterns — do you need multi-document transactions, or is each document self-contained? What is the read:write ratio and what QPS do you need at steady state? What consistency level do you need — can reads tolerate stale data, or do you need read-your-writes? What is the data model — is it truly document-shaped, or are there many-to-many relationships that will require either denormalisation (with write amplification) or application-level joins?

The phrase "scales horizontally" is not a reason to choose MongoDB over PostgreSQL — modern Postgres with Citus or even careful vertical scaling handles significant QPS. The right follow-up is always workload-first: understand the access patterns, then select the store.

---

### Q2. Index on `(last_name, country)` vs `(country, last_name)` — who is right?

**Model answer:**

It depends on the query's selectivity distribution. If `country = 'DE'` is highly selective (few rows per country), putting `country` first means the index can skip 95%+ of rows on the first column alone — this is almost certainly faster. If `last_name` is more selective (rare surnames like 'Smith' are not, but unusual names are), leading with `last_name` works better.

For `last_name = 'Smith'`: 'Smith' is a common surname; there may be millions of rows. `country = 'DE'` may be more selective. In this specific case `(country, last_name)` is likely faster. The leftmost prefix rule means the database uses the index only if the query includes a leading-column condition; both orderings work for this exact query. The colleague's `(country, last_name)` is more likely correct when `country` has higher selectivity than `last_name`.

The correct answer in an interview is: "measure both with `EXPLAIN ANALYZE` on production data volumes — the right answer depends on actual cardinality."

---

### Q3. Synchronous vs asynchronous replication — difference and when to choose each.

**Model answer:**

In **synchronous replication**, the leader waits for at least one replica to acknowledge the write before confirming success to the client. This guarantees zero data loss on leader failure (RPO = 0) but adds replica network round-trip latency to every write. Choose synchronous when data loss is unacceptable: financial transactions, inventory decrements, user authentication records.

In **asynchronous replication**, the leader confirms to the client immediately and propagates the write to replicas in the background. This gives lowest write latency but risks data loss — if the leader crashes before a replica catches up, those writes are gone. Choose async when writes are high-volume and some lag is acceptable: analytics event ingestion, social media timelines, session activity logs. Semi-synchronous (one replica must ack, rest are async) is a common middle ground: one confirmation guarantees durability without paying full fan-out latency.

---

### Q4. Consistent hash ring: 4 nodes → 5 nodes, what fraction of keys move?

**Model answer:**

With a perfectly balanced consistent hash ring, adding a 5th node to a 4-node ring means the new node takes over approximately 1/5 of the keyspace. Keys that move are only those whose clockwise successor on the ring changes — i.e., keys that now land on the new node's arc instead of the arc they previously belonged to. All other keys stay on their existing nodes.

Expected fraction moved ≈ **1/N_new = 1/5 = 20%**. In practice, with 100 virtual nodes per physical node, the distribution is close to uniform and the actual movement is within a few percentage points of 20%. This is the core advantage over modular hashing, which would move ~80% of keys (all keys are rehashed when N changes from 4 to 5 under `key mod N`).

---

### Q5. Why can't we "just use the primary for reads" at 50 000 reads/s?

**Model answer:**

Two concrete failure modes: (1) **Connection exhaustion** — PostgreSQL uses a process-per-connection model. At 50 000 reads/s even with 1 ms average query time you need thousands of concurrent connections; Postgres struggles above ~500 active connections without a pooler, and even with pgBouncer, the primary must execute all those queries, consuming CPU and I/O that compete directly with writes. (2) **Write contention** — every read on the primary acquires shared locks and competes for buffer cache with in-flight writes. Under peak write load (5 000/s), read latency spikes as writes thrash the shared buffer pool, and you get write amplification on the WAL as MVCC cleanup (autovacuum) runs.

The correct approach is to route reads to read replicas, which serve reads independently without competing with the leader's write path. Read replicas also allow read workloads to scale horizontally: adding a replica adds read capacity without touching the primary.

---

### Q6. CQRS: what it is, what problem it solves, what new problem it creates.

**Model answer:**

**Command Query Responsibility Segregation (CQRS)** separates the write model (commands that mutate state) from the read model (queries that project state). The write model is normalised for consistency; the read model is denormalised for query performance. They are separate data structures, often separate datastores, kept in sync via an event or projection pipeline.

**Problem it solves:** a single model can't be simultaneously optimal for writes (3NF schema, row locks, foreign key constraints) and for reads (denormalised, wide rows, pre-joined). A social feed read that joins users, posts, likes, and follows across 4 tables is 10–100× slower than reading from a pre-computed `timelines` table. CQRS allows each side to evolve independently.

**New problem it introduces:** eventual consistency between write and read models. The projection pipeline has lag; a user who posts may not immediately see their post in their timeline. Application code must handle this window gracefully (e.g., optimistic local update in the UI). Example read model that doesn't exist in the write model: `user_timelines(follower_id, post_id, author_name, body_preview, created_at)` — a denormalised cache of recent posts from followed users, impossible to maintain as a normalised write-model table.

---

### Q7. PostgreSQL replica lag of 8 seconds — three causes and mitigations.

**Model answer:**

1. **High write throughput overwhelming the replica's apply rate.** Cause: replica WAL apply is single-threaded by default in older Postgres versions; if the primary is writing 500 MB/s and the replica can only apply 100 MB/s, lag accumulates. Mitigation: enable parallel replication (`max_worker_processes`, `max_parallel_workers`); upgrade to Postgres 14+ with parallel apply.

2. **Long-running queries on the replica blocking WAL apply.** Cause: `hot_standby_feedback = on` combined with a slow analytics query on the replica can block vacuum and WAL application (conflict resolution). Mitigation: cancel conflicting queries (`max_standby_streaming_delay`), or route analytics to a dedicated replica that doesn't serve latency-sensitive reads.

3. **Network bandwidth saturation between primary and replica.** Cause: a large bulk import or DDL operation generates a burst of WAL that exceeds the cross-AZ or cross-datacenter link capacity. Mitigation: throttle bulk imports during peak hours; use `pg_basebackup` with `--max-rate`; consider WAL compression (`wal_compression = on`) to reduce bytes transferred.

---

### Q8. PostgreSQL for OLTP, ClickHouse for analytics — why not consolidate?

**Model answer:**

OLTP and OLAP workloads have fundamentally opposite I/O access patterns. PostgreSQL uses row-oriented storage: each row is a contiguous unit on disk, ideal for point reads (`SELECT * FROM orders WHERE id = 42`) and row-level mutations. ClickHouse uses columnar storage: all values of a given column are stored contiguously, enabling vectorised scans over millions of rows with hardware SIMD instructions — ideal for `SELECT sum(revenue) FROM orders WHERE date > '2024-01-01'`.

If you run analytics aggregations on PostgreSQL, each `SUM(revenue)` query must read and discard every column in every row (high I/O, high CPU). Conversely, ClickHouse is optimised for batch appends and range scans but has poor support for row-level updates, transactions, and complex joins — running an e-commerce transactional workload on ClickHouse would be an architectural mismatch. The two systems are complements: write to Postgres, stream changes via CDC to ClickHouse, query each for what it does best. Consolidating onto one system means either terrible analytics performance or terrible transactional reliability.

---

## Coding Challenge — Complete Reference Implementation

### `src/feedapp/data_layer.py`

```python
"""FeedApp data layer: shard routing, replica routing, and CQRS read model.

src/feedapp/data_layer.py
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from data_layer.sharding_router import ShardingRouter
from data_layer.replica_router import ReplicaRouter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TimelineEntry(BaseModel):
    """A single entry in a user's timeline read model."""

    post_id: str
    author_id: str
    author_name: str
    body_preview: str        # first 280 chars of post body
    like_count: int
    created_at: datetime


# ---------------------------------------------------------------------------
# Shard router — wraps ShardingRouter with FeedApp-specific API
# ---------------------------------------------------------------------------

class FeedShardRouter:
    """Routes feed operations to the correct shard based on user_id.

    Uses consistent hashing with 150 virtual nodes across 8 shards.
    Partition key is user_id — present in every core FeedApp query.
    """

    SHARD_COUNT: int = 8
    VIRTUAL_NODES: int = 150

    def __init__(self) -> None:
        self._router = ShardingRouter(virtual_nodes=self.VIRTUAL_NODES)
        for i in range(self.SHARD_COUNT):
            self._router.add_node(f"feed-shard-{i}")
        logger.info("FeedShardRouter initialised with %d shards", self.SHARD_COUNT)

    def get_shard_for_user(self, user_id: str) -> str:
        """Return the shard name responsible for `user_id`.

        Deterministic: same user_id always maps to the same shard.
        """
        return self._router.get_shard(user_id)

    def get_distribution(self, user_ids: list[str]) -> dict[str, int]:
        """Return per-shard key counts for load-balance verification."""
        return self._router.get_distribution(user_ids)


# ---------------------------------------------------------------------------
# Replica router — wraps ReplicaRouter with FeedApp-specific API
# ---------------------------------------------------------------------------

class FeedReplicaRouter:
    """Manages replica routing for all shards in the FeedApp cluster.

    3 replicas per shard, max acceptable lag 500 ms.
    Writes always go to the shard leader; reads round-robin over healthy replicas.
    """

    REPLICAS_PER_SHARD: int = 3
    MAX_LAG_MS: int = 500

    def __init__(self, shard_names: list[str]) -> None:
        self._routers: dict[str, ReplicaRouter] = {}
        for shard in shard_names:
            leader = f"{shard}-leader"
            replicas = [f"{shard}-replica-{i}" for i in range(self.REPLICAS_PER_SHARD)]
            self._routers[shard] = ReplicaRouter(
                leader=leader,
                replicas=replicas,
                max_lag_ms=self.MAX_LAG_MS,
            )
        logger.info(
            "FeedReplicaRouter initialised: %d shards × %d replicas",
            len(shard_names),
            self.REPLICAS_PER_SHARD,
        )

    def get_read_node_for_shard(self, shard: str) -> str:
        """Return the best read node for `shard` (replica or leader fallback)."""
        return self._routers[shard].route_read()

    def get_write_node_for_shard(self, shard: str) -> str:
        """Return the leader node for `shard`."""
        return self._routers[shard].route_write()

    def mark_replica_lagging(self, shard: str, replica: str, lag_ms: int) -> None:
        """Record a lag observation; replica is excluded from reads if above threshold."""
        self._routers[shard].mark_replica_lagging(replica, lag_ms)

    def mark_replica_healthy(self, shard: str, replica: str) -> None:
        """Clear lag record — replica re-enters the healthy pool."""
        self._routers[shard].mark_replica_healthy(replica)


# ---------------------------------------------------------------------------
# CQRS read model
# ---------------------------------------------------------------------------

class TimelineReadModel:
    """In-memory CQRS read model for development and testing.

    Production equivalent: Redis Sorted Set (score = created_at timestamp)
    or a denormalised `timelines` Postgres table populated via Kafka projection.

    The model is fan-out-on-write: when a post is created, it is immediately
    projected into each follower's timeline so reads are O(1) list lookups
    rather than scatter-gather across the follow graph.
    """

    def __init__(self) -> None:
        # follower_id → list of TimelineEntry, maintained sorted descending by created_at
        self._timelines: dict[str, list[TimelineEntry]] = defaultdict(list)

    def apply_post_created(
        self, follower_ids: list[str], entry: TimelineEntry
    ) -> None:
        """Project a PostCreated event into each follower's timeline.

        Inserts the entry in sorted order (newest first).
        Called by the event projection worker, not by the request path.
        """
        for follower_id in follower_ids:
            timeline = self._timelines[follower_id]
            # Insert maintaining descending created_at order
            # (linear insert acceptable for in-memory model; production uses Redis ZADD)
            inserted = False
            for i, existing in enumerate(timeline):
                if entry.created_at >= existing.created_at:
                    timeline.insert(i, entry)
                    inserted = True
                    break
            if not inserted:
                timeline.append(entry)
        logger.debug(
            "Projected post_id=%s into %d follower timelines",
            entry.post_id,
            len(follower_ids),
        )

    def get_timeline(
        self,
        user_id: str,
        limit: int = 20,
        before: Optional[datetime] = None,
    ) -> list[TimelineEntry]:
        """Return up to `limit` timeline entries for `user_id`, newest first.

        Args:
            user_id: The viewer whose timeline is requested.
            limit: Maximum number of entries to return.
            before: If supplied, return only entries with created_at < before
                    (keyset pagination cursor).

        Returns:
            List of TimelineEntry sorted descending by created_at.
        """
        entries = self._timelines.get(user_id, [])
        if before is not None:
            entries = [e for e in entries if e.created_at < before]
        return entries[:limit]

    def entry_count(self, user_id: str) -> int:
        """Return the total number of timeline entries stored for `user_id`."""
        return len(self._timelines.get(user_id, []))
```

### `tests/unit/test_feedapp_data_layer.py`

```python
"""Unit tests for FeedApp data layer.

tests/unit/test_feedapp_data_layer.py
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from feedapp.data_layer import (
    FeedShardRouter,
    FeedReplicaRouter,
    TimelineEntry,
    TimelineReadModel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def shard_router() -> FeedShardRouter:
    return FeedShardRouter()


@pytest.fixture()
def shard_names() -> list[str]:
    return [f"feed-shard-{i}" for i in range(8)]


@pytest.fixture()
def replica_router(shard_names: list[str]) -> FeedReplicaRouter:
    return FeedReplicaRouter(shard_names=shard_names)


@pytest.fixture()
def read_model() -> TimelineReadModel:
    return TimelineReadModel()


def _make_entry(post_id: str, created_at: datetime) -> TimelineEntry:
    return TimelineEntry(
        post_id=post_id,
        author_id="author-1",
        author_name="Alice",
        body_preview="Hello world",
        like_count=0,
        created_at=created_at,
    )


# ---------------------------------------------------------------------------
# FeedShardRouter tests
# ---------------------------------------------------------------------------

class TestFeedShardRouter:
    def test_shard_assignment_is_deterministic(self, shard_router: FeedShardRouter) -> None:
        """Same user_id always maps to the same shard."""
        shard_a = shard_router.get_shard_for_user("user-42")
        shard_b = shard_router.get_shard_for_user("user-42")
        assert shard_a == shard_b

    def test_all_shards_receive_traffic(self, shard_router: FeedShardRouter) -> None:
        """All 8 shards receive roughly equal load over 800 distinct user IDs (±10%)."""
        user_ids = [f"user-{i}" for i in range(800)]
        distribution = shard_router.get_distribution(user_ids)
        for shard, count in distribution.items():
            assert 72 <= count <= 128, (
                f"Shard {shard} has {count} keys — expected 72–128 (±10% of 100)"
            )
        assert sum(distribution.values()) == 800

    def test_different_users_can_map_to_different_shards(
        self, shard_router: FeedShardRouter
    ) -> None:
        """Not all user IDs land on the same shard."""
        shards = {shard_router.get_shard_for_user(f"user-{i}") for i in range(50)}
        assert len(shards) > 1, "All 50 users mapped to same shard — hash function broken"

    def test_shard_name_format(self, shard_router: FeedShardRouter) -> None:
        """Returned shard names follow expected naming convention."""
        shard = shard_router.get_shard_for_user("user-999")
        assert shard.startswith("feed-shard-")


# ---------------------------------------------------------------------------
# FeedReplicaRouter tests
# ---------------------------------------------------------------------------

class TestFeedReplicaRouter:
    def test_write_node_is_always_leader(
        self, replica_router: FeedReplicaRouter, shard_names: list[str]
    ) -> None:
        """route_write always returns a node ending in -leader."""
        for shard in shard_names:
            node = replica_router.get_write_node_for_shard(shard)
            assert node.endswith("-leader"), f"Expected leader, got {node}"

    def test_read_node_is_replica_when_healthy(
        self, replica_router: FeedReplicaRouter, shard_names: list[str]
    ) -> None:
        """Under healthy conditions, reads go to replicas, not leaders."""
        for shard in shard_names:
            node = replica_router.get_read_node_for_shard(shard)
            assert "-replica-" in node, f"Expected replica, got {node}"

    def test_all_lagging_replicas_fallback_to_leader(
        self, replica_router: FeedReplicaRouter
    ) -> None:
        """When all replicas exceed lag threshold, reads fall back to the leader."""
        shard = "feed-shard-0"
        for i in range(FeedReplicaRouter.REPLICAS_PER_SHARD):
            replica_router.mark_replica_lagging(shard, f"{shard}-replica-{i}", lag_ms=9999)
        node = replica_router.get_read_node_for_shard(shard)
        assert node == f"{shard}-leader"

    def test_recovering_replica_re_enters_pool(
        self, replica_router: FeedReplicaRouter
    ) -> None:
        """After mark_replica_healthy, the replica is eligible for reads again."""
        shard = "feed-shard-1"
        for i in range(FeedReplicaRouter.REPLICAS_PER_SHARD):
            replica_router.mark_replica_lagging(shard, f"{shard}-replica-{i}", lag_ms=9999)
        replica_router.mark_replica_healthy(shard, f"{shard}-replica-0")
        node = replica_router.get_read_node_for_shard(shard)
        assert node == f"{shard}-replica-0"


# ---------------------------------------------------------------------------
# TimelineReadModel tests
# ---------------------------------------------------------------------------

class TestTimelineReadModel:
    def test_apply_post_created_adds_to_all_followers(
        self, read_model: TimelineReadModel
    ) -> None:
        """Post is added to every follower's timeline."""
        entry = _make_entry("post-1", datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc))
        read_model.apply_post_created(["user-A", "user-B", "user-C"], entry)
        assert read_model.entry_count("user-A") == 1
        assert read_model.entry_count("user-B") == 1
        assert read_model.entry_count("user-C") == 1

    def test_get_timeline_returns_newest_first(self, read_model: TimelineReadModel) -> None:
        """Entries are returned in descending created_at order."""
        t0 = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        t1 = datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc)
        t2 = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        for post_id, ts in [("post-old", t0), ("post-mid", t1), ("post-new", t2)]:
            read_model.apply_post_created(["viewer-1"], _make_entry(post_id, ts))
        timeline = read_model.get_timeline("viewer-1")
        post_ids = [e.post_id for e in timeline]
        assert post_ids == ["post-new", "post-mid", "post-old"]

    def test_get_timeline_respects_limit(self, read_model: TimelineReadModel) -> None:
        """get_timeline(limit=N) returns at most N entries."""
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for i in range(30):
            ts = base + timedelta(minutes=i)
            read_model.apply_post_created(["viewer-2"], _make_entry(f"post-{i}", ts))
        result = read_model.get_timeline("viewer-2", limit=10)
        assert len(result) == 10

    def test_get_timeline_before_paginates(self, read_model: TimelineReadModel) -> None:
        """get_timeline with `before` returns only entries older than cursor."""
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        entries = [_make_entry(f"post-{i}", base + timedelta(hours=i)) for i in range(5)]
        for entry in entries:
            read_model.apply_post_created(["viewer-3"], entry)
        cursor = base + timedelta(hours=3)  # exclude hours 3 and 4
        result = read_model.get_timeline("viewer-3", before=cursor)
        assert all(e.created_at < cursor for e in result)
        assert len(result) == 3  # hours 0, 1, 2

    def test_empty_timeline_returns_empty_list(self, read_model: TimelineReadModel) -> None:
        """User with no timeline entries returns an empty list, not an error."""
        result = read_model.get_timeline("user-nobody")
        assert result == []

    def test_apply_post_to_no_followers_does_not_error(
        self, read_model: TimelineReadModel
    ) -> None:
        """Projecting a post with an empty follower list is a no-op."""
        entry = _make_entry("post-orphan", datetime(2024, 1, 1, tzinfo=timezone.utc))
        read_model.apply_post_created([], entry)  # should not raise

    def test_entry_count_reflects_projections(self, read_model: TimelineReadModel) -> None:
        """entry_count returns the correct total after multiple projections."""
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for i in range(7):
            ts = base + timedelta(minutes=i)
            read_model.apply_post_created(["user-Z"], _make_entry(f"p-{i}", ts))
        assert read_model.entry_count("user-Z") == 7
```

---

## Scoring Guide

### 90–100% — Distinction

All 8 conceptual answers include: precise technical terminology, explicit trade-offs (not just benefits), at least one concrete example or numbers. Answers to Q3, Q4, Q7 reference specific Postgres or Cassandra behaviour, not generic descriptions.

Coding challenge: all 5+ tests pass; `FeedShardRouter` uses `ShardingRouter` composition (not inheritance), not a direct reimplementation; `TimelineReadModel.get_timeline` handles `before=None` and `before=<datetime>` distinctly; entries are truly sorted (not insertion-order by accident); `TimelineEntry` uses Pydantic `BaseModel`; no `print` statements.

Stretch indicators: pagination test uses a non-trivial cursor (not just the oldest entry); test for concurrent follower projection; type annotations use `datetime | None` not `Optional[datetime]`.

### 75–89% — Pass

Conceptual answers are mostly correct but one or two lack trade-offs or are too generic (e.g., Q8 says "ClickHouse is faster for analytics" without explaining columnar storage). Q4 may have the right answer (20%) without correct reasoning.

Coding: tests pass, but edge cases missing (empty timeline, all replicas lagging, zero followers). `TimelineReadModel` works but insertion order may not be guaranteed correct — passes happy-path tests but would fail a thorough edge-case review.

### 60–74% — Near-pass

Two or more conceptual questions have factual errors (e.g., claiming consistent hashing moves 0 keys when adding a node, or misidentifying sync vs async trade-offs). Q6 (CQRS) answer describes event sourcing instead of CQRS.

Coding: `FeedShardRouter` or `FeedReplicaRouter` not implemented (only stubs); fewer than 5 tests; `get_timeline` with `before` not implemented or incorrect; missing type hints on public methods.

### Below 60% — Retry

Recommend re-reading Days 3 (replication), 4 (sharding + consistent hashing), and 5 (CQRS). Focus areas: CAP theorem, why consistent hashing moves 1/N keys (draw the ring), what "eventual consistency" means in a CQRS system. Reattempt the coding challenge after completing the Day 6 implementation exercises.
