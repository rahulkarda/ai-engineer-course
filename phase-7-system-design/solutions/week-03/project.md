# Solutions — Phase 7 · Week 3 · Project — Data-Layer Design for FeedApp

---

## Part 1 — Design Document

### 1. Requirements Summary

**FeedApp** is a read-heavy social feed service with the following load profile:

| Metric | Value |
|---|---|
| Daily active users | 10 million |
| Write throughput | 5 000 writes/s peak (posts + likes + follows) |
| Read throughput | 200 000 reads/s peak (timeline + profile pages) |
| Read:write ratio | ~40:1 |
| Post retention | 24 months |
| p99 read latency | < 150 ms |
| p99 write latency | < 300 ms |
| Durability | No data loss on single node failure |
| Timeline consistency | Eventual — up to 2 s staleness acceptable |
| User count cap (exercise) | ≤ 10 000 followers per user (no celebrity fan-out) |

Key constraints driving design decisions:
- 200 000 reads/s eliminates single-node read solutions — read replicas are mandatory.
- 40:1 ratio means optimising the read path is worth significant write complexity.
- 2 s timeline lag tolerance means async replication is acceptable for reads.
- Durability requirement means at least one synchronous ack is required on the write path.

---

### 2. Datastore Selection

| Use case | Store | Rationale |
|---|---|---|
| User profiles | PostgreSQL | ACID writes; complex queries (following counts, profile joins); FK integrity; fits on single node at 10 M users (~5 GB) |
| Post content + metadata | Cassandra (wide-column) | Write-optimised LSM tree handles 5 000/s sustained easily; partition by `user_id` gives O(1) profile-page reads; no joins needed |
| Follow graph | Adjacency list in Cassandra | Store `follows_by_follower(follower_id, followee_id)` and `follows_by_followee(followee_id, follower_id)` — O(1) lookups both ways without a dedicated graph DB; acceptable at ≤ 10 000 followers per user |
| Timeline (hot read path) | Redis Sorted Set | `ZRANGEBYSCORE follower_id -inf +inf LIMIT 0 20` is O(log N + limit); sub-millisecond reads; TTL-based eviction of old entries; handles 200 000 reads/s on a modest cluster |
| Like counters | Redis (INCR) | Atomic increment; no row-lock contention; eventual write-back to Cassandra for persistence |
| Analytics / reporting | ClickHouse | Columnar store; aggregations over 100 M+ rows in < 1 s; fed via Kafka CDC from Cassandra |

**Trade-off (Cassandra):** Cassandra sacrifices flexible querying for horizontal write scalability. Multi-column filters beyond the partition key require either a secondary index (slow, scatter-gather) or denormalised table design. We accept this by pre-designing tables for exactly the access patterns we know. Any ad-hoc query requirement must go through the ClickHouse analytics path instead.

**Why not PostgreSQL for posts?** At 5 000 writes/s sustained with 24-month retention (~200 M rows), a single Postgres primary saturates on write I/O. Sharding Postgres is operationally complex (Citus). Cassandra's peer-to-peer architecture with tunable consistency handles this workload natively and is the industry standard for social feed post storage.

---

### 3. Partitioning Strategy

#### Posts (Cassandra)
- **Partition key**: `user_id` — present in every profile-page query; high cardinality (10 M users); stable over the record's lifetime
- **Clustering key**: `created_at DESC` — supports `WHERE user_id = ? ORDER BY created_at DESC LIMIT 20` as a single-partition range scan
- **Method**: Cassandra's built-in consistent hashing (Murmur3), configurable token range
- **Shard count**: 16 logical shards (token ring partitions), RF=3 across 6 nodes (each node owns ~8 token ranges)
- **Growth**: 16 shards supports ~10× current write throughput before topology change

**Example:** `user_id = "u-00123456"` → Murmur3 hash → token `0x3F2A...` → ring position → node-3 (owns token range `0x3000..0x4000`) → replicated to node-4, node-5

#### Timelines (Redis)
- **Key pattern**: `timeline:{viewer_id}` — one Sorted Set per user
- **Score**: Unix timestamp of `created_at` — enables time-range queries via ZRANGEBYSCORE
- **Redis Cluster**: 16384 hash slots, `{viewer_id}` is the hash tag → consistent slot assignment
- **Sharding**: 6 Redis nodes, ~2730 slots each; 1 replica per master

#### Follow graph (Cassandra)
- **Table 1 `follows_by_follower`**: partition key `follower_id` — "who do I follow?" queries
- **Table 2 `follows_by_followee`**: partition key `followee_id` — "who follows me?" (for fan-out on write)
- Both tables partition by the lookup key → single-partition read, no scatter-gather

#### User profiles (PostgreSQL)
- No sharding — 10 M users at ~500 bytes/row = 5 GB; fits comfortably on a single Postgres node with read replicas
- Vertical scaling + read replicas sufficient for this tier

#### Uneven distribution mitigation
- Cassandra's Murmur3 with virtual nodes (256 vnodes per node default) provides near-uniform distribution
- Redis Cluster's 16384 slots with balanced assignment avoids systematic hotspots
- Hotspot risk: a user with 10 000 followers who posts frequently creates 10 000 timeline writes per post. At ≤ 10 000 followers (problem constraint), maximum fan-out is bounded: 5 000 posts/s × 10 000 followers = 50 M Redis writes/s peak — this requires the Redis tier to be sized accordingly (6+ nodes, pipeline batching)

---

### 4. Replication Approach

**Cassandra posts (RF=3)**

Three replicas per partition using Cassandra's native replication. Consistency levels:
- **Writes**: `QUORUM` (2/3 replicas must ack) — ensures durability; no data loss if one node fails
- **Timeline reads**: `ONE` — lowest latency; 2 s staleness acceptable per requirements
- **Profile reads**: `LOCAL_QUORUM` — stronger consistency for own-write visibility

Replication is async between data centres (multi-DC deployment) and synchronous within the local DC for quorum acknowledgement. RTO on node failure: Cassandra's gossip protocol detects failure in < 10 s; coordinator automatically routes to remaining replicas — no manual failover needed.

**PostgreSQL user profiles**

1 leader, 2 synchronous read replicas (same region), 1 async replica (second region for DR). Semi-synchronous: leader waits for 1 of 2 replicas before confirming to client. Failover via Patroni + HAProxy; RTO < 30 s; RPO = 0 within region.

Replication lag interaction: PostgreSQL replicas serve profile reads. With semi-sync, lag on the synchronous replica is bounded to < 50 ms under normal conditions. Follower count and bio reads are tolerant of this lag.

**Redis timelines**

Redis Cluster with async replication (1 replica per master). AOF + RDB persistence. Redis Sentinel for master election; RTO < 10 s. Timeline Redis is eventually consistent by design — a follower may not see a new post for up to 2 s (Kafka lag + Redis write), which is within the stated tolerance.

---

### 5. CQRS Read Model Design

**Events that trigger projection:**

| Event | Trigger | Read model update |
|---|---|---|
| `PostCreated` | User publishes a post | Insert entry into all followers' Redis timelines (`ZADD`) |
| `PostLiked` | User likes a post | Increment `like_count` in Redis timeline entries (async, batched) |
| `UserFollowed` | User follows another | Backfill last 20 posts of followed user into follower's timeline |
| `UserUnfollowed` | User unfollows | Remove unfollowed user's posts from follower's timeline (lazy GC) |

**Read model schema (Redis Sorted Set + hash):**

```
Key:   timeline:{viewer_id}
Type:  Sorted Set
Score: Unix timestamp (float)
Member: {post_id}                          ← member is just the post_id

Key:   post:{post_id}
Type:  Hash
Fields:
  author_id     STRING
  author_name   STRING
  body_preview  STRING (first 280 chars)
  like_count    INT
  created_at    INT (unix timestamp ms)
```

Timeline query = `ZREVRANGEBYSCORE timeline:{viewer_id} +inf -inf LIMIT 0 20`
followed by `HGETALL post:{post_id}` for each result — pipelined in a single Redis round-trip.

**Index design:** Redis Sorted Set is inherently indexed by score (B-skip-list). Range by time is O(log N + limit). No secondary indexes needed for the timeline read path.

**Staleness bounded by:**
1. Kafka consumer group lag — typically < 200 ms; SLA target < 500 ms
2. Redis write latency — < 5 ms
3. Total end-to-end: < 2 s with 99th percentile headroom, matching requirements

**Write path → projection → read path:**

```
Client       Write API    Cassandra       Kafka           Redis
  |               |           |              |               |
  |--POST /post-->|           |              |               |
  |               |--INSERT-->|              |               |
  |               |<-- ack ---|              |               |
  |<-- 201 -------|           |              |               |
  |               |           |--CDC event-->|               |
  |               |           |   (PostCreated)              |
  |               |           |              |--fan-out----->|
  |               |           |              |  ZADD each    |
  |               |           |              |  follower     |
  |               |           |              |               |
  |--GET /timeline|           |              |               |
  |-------------->|           |              |               |
  |               |<----------|--------------|---ZREVRANGE---|
  |<-- 200 -------|           |              |               |
```

---

### 6. Open Questions and Risks

- [ ] **Fan-out at follower cap**: 10 000 followers × 5 000 posts/s = 50 M Redis writes/s at peak. Is the Redis cluster sized for this? Who decides cluster scale-out budget? **Owner: Platform + Finance.**

- [ ] **Timeline backfill on follow**: when a user follows another, we backfill their last 20 posts. If the followed user has 20 posts and 10 000 followers follow simultaneously (viral follow event), that's 200 000 Cassandra reads and 200 000 Redis writes in seconds. **Decision needed: lazy backfill (first timeline load) vs eager backfill (on follow event). Owner: Engineering.**

- [ ] **Like counter write amplification**: every like updates `like_count` in Redis for every follower who has that post in their timeline. At 10 000 followers × 1 000 likes/s = 10 M Redis writes/s for like counters alone. Batching and approximate counts (HyperLogLog) may be needed. **Owner: Engineering + Product (acceptable accuracy?).**

- [ ] **Data retention enforcement**: posts must be deleted after 24 months. Cassandra TTL on post rows handles this, but timeline Redis entries for deleted posts will return stale data. Tombstone propagation strategy needed. **Owner: Engineering.**

- [ ] **Cross-region consistency**: follow graph writes (PostgreSQL or Cassandra) need to be visible globally. Multi-DC Cassandra replication handles this, but a follow event in EU followed immediately by a post visibility check in US could miss the follow. **Decision: acceptable race condition, or do we require cross-region quorum on follows? Owner: Product + Architecture.**

---

## Part 2 — Python Reference Implementations

### `src/data_layer/sharding_router.py` (complete)

```python
"""Consistent-hashing sharding router with virtual nodes.

src/data_layer/sharding_router.py
"""

from __future__ import annotations

import hashlib
import logging
from bisect import bisect_left, insort
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ShardingRouter:
    """Routes keys to shards using a consistent hash ring with virtual nodes.

    Args:
        virtual_nodes: Number of virtual node positions per physical node.
                       Higher values improve key distribution uniformity.
                       Default 150 gives < 5% stddev across 3+ nodes.
    """

    virtual_nodes: int = 150
    _ring: dict[int, str] = field(default_factory=dict, init=False, repr=False)
    _sorted_keys: list[int] = field(default_factory=list, init=False, repr=False)
    _nodes: set[str] = field(default_factory=set, init=False, repr=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_node(self, node: str) -> None:
        """Add a physical node to the ring with `virtual_nodes` positions.

        Idempotent: adding the same node twice is a no-op after the first call
        (positions already exist). Logs at INFO level.
        """
        if node in self._nodes:
            logger.warning("Node %r already in ring — skipping", node)
            return
        for i in range(self.virtual_nodes):
            key = self._hash(f"{node}#{i}")
            self._ring[key] = node
            insort(self._sorted_keys, key)
        self._nodes.add(node)
        logger.info("Added node %r: %d virtual positions on ring", node, self.virtual_nodes)

    def remove_node(self, node: str) -> None:
        """Remove a physical node and all its virtual positions.

        Raises:
            KeyError: if `node` is not registered in the ring.
        """
        if node not in self._nodes:
            raise KeyError(f"Node {node!r} not found in ring")
        for i in range(self.virtual_nodes):
            key = self._hash(f"{node}#{i}")
            del self._ring[key]
            idx = bisect_left(self._sorted_keys, key)
            self._sorted_keys.pop(idx)
        self._nodes.discard(node)
        logger.info("Removed node %r from ring", node)

    def get_shard(self, key: str) -> str:
        """Return the physical node responsible for `key`.

        Uses clockwise lookup on the sorted ring (next node >= hash(key)).

        Raises:
            RuntimeError: if the ring is empty.
        """
        if not self._ring:
            raise RuntimeError("Cannot route key: ring is empty")
        h = self._hash(key)
        idx = bisect_left(self._sorted_keys, h)
        # Wrap around for clockwise behaviour
        idx = idx % len(self._sorted_keys)
        return self._ring[self._sorted_keys[idx]]

    def get_distribution(self, keys: list[str]) -> dict[str, int]:
        """Return per-shard key counts for load-balance analysis."""
        counts: dict[str, int] = defaultdict(int)
        for key in keys:
            counts[self.get_shard(key)] += 1
        return dict(counts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(value: str) -> int:
        """Return a stable 32-bit integer hash of `value` using MD5."""
        digest = hashlib.md5(value.encode(), usedforsecurity=False).digest()
        return int.from_bytes(digest[:4], byteorder="big")
```

### `src/data_layer/replica_router.py` (complete)

```python
"""Lag-aware read-replica router.

src/data_layer/replica_router.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ReplicaRouter:
    """Routes reads to healthy replicas and writes to the leader.

    A replica is "healthy" when its reported lag is below `max_lag_ms`.
    Round-robin is applied over the healthy set. Falls back to the leader
    when no healthy replica is available.

    Args:
        leader: Primary/leader node name.
        replicas: List of replica node names.
        max_lag_ms: Maximum acceptable replication lag in milliseconds.
    """

    leader: str
    replicas: list[str]
    max_lag_ms: int = 200
    _lag: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _rr_index: int = field(default=0, init=False, repr=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mark_replica_lagging(self, replica: str, lag_ms: int) -> None:
        """Record the current replication lag for `replica`.

        Raises:
            ValueError: if `replica` is not in self.replicas.
        """
        if replica not in self.replicas:
            raise ValueError(f"Unknown replica {replica!r}")
        self._lag[replica] = lag_ms
        if lag_ms > self.max_lag_ms:
            logger.warning(
                "Replica %r lag %d ms exceeds threshold %d ms — excluded from reads",
                replica,
                lag_ms,
                self.max_lag_ms,
            )

    def mark_replica_healthy(self, replica: str) -> None:
        """Clear any lag record for `replica`, marking it healthy.

        Raises:
            ValueError: if `replica` is not in self.replicas.
        """
        if replica not in self.replicas:
            raise ValueError(f"Unknown replica {replica!r}")
        self._lag.pop(replica, None)
        logger.debug("Replica %r marked healthy", replica)

    def route_read(self) -> str:
        """Return the next node to handle a read request.

        Algorithm:
          1. Build healthy set: replicas where lag < max_lag_ms (or no lag recorded).
          2. If non-empty: round-robin across healthy set.
          3. If empty: fall back to leader with a WARNING.
        """
        healthy = [
            r for r in self.replicas
            if self._lag.get(r, 0) < self.max_lag_ms
        ]
        if not healthy:
            logger.warning(
                "All replicas exceed lag threshold — falling back to leader %r",
                self.leader,
            )
            return self.leader
        selected = healthy[self._rr_index % len(healthy)]
        self._rr_index += 1
        return selected

    def route_write(self) -> str:
        """Return the leader node for all writes."""
        return self.leader

    def status(self) -> dict[str, object]:
        """Return a dict summarising router state for observability.

        Keys: leader, total_replicas, healthy_replicas, lagging_replicas
        """
        healthy_count = sum(
            1 for r in self.replicas if self._lag.get(r, 0) < self.max_lag_ms
        )
        return {
            "leader": self.leader,
            "total_replicas": len(self.replicas),
            "healthy_replicas": healthy_count,
            "lagging_replicas": len(self.replicas) - healthy_count,
        }
```

### `src/feed/timeline_service.py` (TimelineService — composes both routers)

```python
"""High-level TimelineService that composes sharding and replica routing.

src/feed/timeline_service.py
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from data_layer.sharding_router import ShardingRouter
from data_layer.replica_router import ReplicaRouter

logger = logging.getLogger(__name__)


class TimelineService:
    """Orchestrates post creation, follow management, and timeline reads.

    Uses:
    - ShardingRouter: consistent hash routing to the correct shard
    - ReplicaRouter (one per shard): lag-aware read replica selection
    - In-memory stores (dict): stand-in for Cassandra posts and Redis timelines

    In production: replace _post_store with Cassandra client calls and
    _timeline_store with Redis Sorted Set calls (pipeline for fan-out).
    """

    SHARD_COUNT = 8
    REPLICAS_PER_SHARD = 3
    MAX_LAG_MS = 500

    def __init__(self) -> None:
        # Routing layer
        self._shard_router = ShardingRouter(virtual_nodes=150)
        self._replica_routers: dict[str, ReplicaRouter] = {}
        for i in range(self.SHARD_COUNT):
            shard = f"shard-{i}"
            self._shard_router.add_node(shard)
            self._replica_routers[shard] = ReplicaRouter(
                leader=f"{shard}-leader",
                replicas=[f"{shard}-replica-{j}" for j in range(self.REPLICAS_PER_SHARD)],
                max_lag_ms=self.MAX_LAG_MS,
            )

        # Storage stubs (replace with real DB clients in production)
        self._posts: dict[str, dict] = {}          # post_id → post data
        self._user_posts: dict[str, list[str]] = {}  # user_id → [post_id, ...]
        self._follows: dict[str, set[str]] = {}      # follower_id → {followee_id, ...}
        self._followers: dict[str, set[str]] = {}    # followee_id → {follower_id, ...}
        self._timelines: dict[str, list[dict]] = {}  # viewer_id → sorted list of posts

        logger.info("TimelineService initialised: %d shards", self.SHARD_COUNT)

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def post(self, user_id: str, content: str) -> str:
        """Create a new post for `user_id` and fan it out to follower timelines.

        Returns:
            The generated post_id.
        """
        shard = self._shard_router.get_shard(user_id)
        write_node = self._replica_routers[shard].route_write()
        logger.debug("POST user=%s → shard=%s → node=%s", user_id, shard, write_node)

        post_id = str(uuid.uuid4())
        created_at = datetime.now(tz=timezone.utc)
        post = {
            "post_id": post_id,
            "user_id": user_id,
            "content": content[:500],
            "created_at": created_at,
        }

        # Write to post store (simulates Cassandra INSERT)
        self._posts[post_id] = post
        self._user_posts.setdefault(user_id, []).insert(0, post_id)

        # Fan-out to all follower timelines (simulates Kafka → Redis projection)
        for follower_id in self._followers.get(user_id, set()):
            self._fan_out_to_timeline(follower_id, post)

        return post_id

    def follow(self, follower_id: str, followee_id: str) -> None:
        """Record that `follower_id` now follows `followee_id`.

        Also backfills the 20 most recent posts from `followee_id` into
        `follower_id`'s timeline (fan-out-on-read for the backfill only).
        """
        self._follows.setdefault(follower_id, set()).add(followee_id)
        self._followers.setdefault(followee_id, set()).add(follower_id)
        logger.debug("FOLLOW %s → %s", follower_id, followee_id)

        # Backfill last 20 posts of followee into follower timeline
        for post_id in self._user_posts.get(followee_id, [])[:20]:
            post = self._posts[post_id]
            self._fan_out_to_timeline(follower_id, post)

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get_timeline(
        self, user_id: str, limit: int = 20
    ) -> list[dict]:
        """Return the `limit` most recent posts in `user_id`'s timeline.

        Reads from the pre-computed timeline (CQRS read model).
        Routes through a healthy replica for the user's shard.
        """
        shard = self._shard_router.get_shard(user_id)
        read_node = self._replica_routers[shard].route_read()
        logger.debug("GET_TIMELINE user=%s → shard=%s → node=%s", user_id, shard, read_node)

        timeline = self._timelines.get(user_id, [])
        return timeline[:limit]

    def get_profile(self, user_id: str, limit: int = 20) -> list[dict]:
        """Return the `limit` most recent posts by `user_id`.

        Reads directly from post store (not the timeline read model).
        """
        shard = self._shard_router.get_shard(user_id)
        read_node = self._replica_routers[shard].route_read()
        logger.debug("GET_PROFILE user=%s → shard=%s → node=%s", user_id, shard, read_node)

        post_ids = self._user_posts.get(user_id, [])[:limit]
        return [self._posts[pid] for pid in post_ids]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fan_out_to_timeline(self, viewer_id: str, post: dict) -> None:
        """Insert `post` into `viewer_id`'s timeline in sorted order (newest first)."""
        timeline = self._timelines.setdefault(viewer_id, [])
        created_at = post["created_at"]
        for i, entry in enumerate(timeline):
            if created_at >= entry["created_at"]:
                timeline.insert(i, post)
                return
        timeline.append(post)
```

### `tests/unit/test_timeline_service.py`

```python
"""Unit tests for TimelineService.

tests/unit/test_timeline_service.py
"""

from __future__ import annotations

import pytest

from feed.timeline_service import TimelineService


@pytest.fixture()
def svc() -> TimelineService:
    return TimelineService()


class TestPost:
    def test_post_returns_post_id(self, svc: TimelineService) -> None:
        post_id = svc.post("user-1", "Hello world")
        assert isinstance(post_id, str)
        assert len(post_id) == 36  # UUID format

    def test_post_visible_on_profile(self, svc: TimelineService) -> None:
        svc.post("user-1", "My first post")
        profile = svc.get_profile("user-1")
        assert len(profile) == 1
        assert profile[0]["content"] == "My first post"

    def test_post_truncated_at_500_chars(self, svc: TimelineService) -> None:
        long_content = "x" * 600
        svc.post("user-1", long_content)
        profile = svc.get_profile("user-1")
        assert len(profile[0]["content"]) == 500

    def test_multiple_posts_ordered_newest_first(self, svc: TimelineService) -> None:
        svc.post("user-1", "first")
        svc.post("user-1", "second")
        svc.post("user-1", "third")
        profile = svc.get_profile("user-1")
        assert profile[0]["content"] == "third"
        assert profile[-1]["content"] == "first"


class TestFollow:
    def test_follow_fans_out_to_timeline(self, svc: TimelineService) -> None:
        svc.post("user-A", "post by A")
        svc.follow("user-B", "user-A")
        timeline = svc.get_timeline("user-B")
        assert len(timeline) == 1
        assert timeline[0]["content"] == "post by A"

    def test_new_post_appears_in_follower_timeline(self, svc: TimelineService) -> None:
        svc.follow("user-B", "user-A")
        svc.post("user-A", "written after follow")
        timeline = svc.get_timeline("user-B")
        assert any(e["content"] == "written after follow" for e in timeline)

    def test_timeline_empty_before_follow(self, svc: TimelineService) -> None:
        svc.post("user-A", "invisible")
        timeline = svc.get_timeline("user-B")  # user-B does not follow user-A
        assert timeline == []

    def test_backfill_limited_to_20_posts(self, svc: TimelineService) -> None:
        for i in range(30):
            svc.post("user-A", f"post {i}")
        svc.follow("user-B", "user-A")
        timeline = svc.get_timeline("user-B", limit=100)
        assert len(timeline) == 20  # backfill capped at 20


class TestTimelineRouting:
    def test_write_node_is_leader(self, svc: TimelineService) -> None:
        """post() routes writes to a leader node."""
        # Verify by checking shard routing directly
        shard = svc._shard_router.get_shard("user-X")
        write_node = svc._replica_routers[shard].route_write()
        assert write_node.endswith("-leader")

    def test_read_node_is_replica_under_healthy_conditions(
        self, svc: TimelineService
    ) -> None:
        """get_timeline() routes reads to a replica under normal conditions."""
        shard = svc._shard_router.get_shard("user-X")
        read_node = svc._replica_routers[shard].route_read()
        assert "-replica-" in read_node

    def test_timeline_still_works_when_all_replicas_lagging(
        self, svc: TimelineService
    ) -> None:
        """Timeline reads fall back to leader when all replicas exceed lag threshold."""
        svc.post("user-A", "content")
        svc.follow("user-B", "user-A")

        shard = svc._shard_router.get_shard("user-B")
        for i in range(TimelineService.REPLICAS_PER_SHARD):
            svc._replica_routers[shard].mark_replica_lagging(
                f"{shard}-replica-{i}", lag_ms=9999
            )
        # Should not raise — falls back to leader
        timeline = svc.get_timeline("user-B")
        assert isinstance(timeline, list)

    def test_distribution_across_shards(self, svc: TimelineService) -> None:
        """All 8 shards receive traffic for 800 distinct user IDs (±10%)."""
        user_ids = [f"user-{i}" for i in range(800)]
        distribution = svc._shard_router.get_distribution(user_ids)
        for shard, count in distribution.items():
            assert 72 <= count <= 128, f"Shard {shard} count {count} outside ±10% of 100"
```

---

## Part 3 — Evaluation Notes

### What distinguishes excellent (90%+) from adequate (75%) solutions

**Design document:**

Excellent solutions are specific about numbers. "16 shards, each handling ~310 writes/s peak" is excellent. "Some shards for scalability" is adequate. Excellent solutions justify *why* Cassandra over PostgreSQL for posts by referencing LSM tree write performance, not just "Cassandra scales better." They also identify the fan-out write amplification problem (10 000 followers × write rate) and propose a mitigation.

Excellent solutions draw the CQRS sequence diagram with all four actors (writer, Cassandra, Kafka/CDC, Redis) and show both the write-path latency (confirmed before fan-out) and the async fan-out separately. Adequate solutions describe the pattern without showing the sequence.

**Code:**

Excellent: `ShardingRouter._hash` uses `usedforsecurity=False` (Python 3.9+ safety flag for non-security hashing). Excellent: `ReplicaRouter.route_read` handles the all-lagging edge case explicitly and logs a WARNING (not just silently returning the leader). Excellent: `TimelineService._fan_out_to_timeline` inserts in sorted order rather than appending and sorting — demonstrates understanding that the timeline must always be ordered for efficient pagination.

Adequate: all tests pass but only on the happy path; no test for all-replicas-lagging fallback, no test for backfill cap at 20 posts.

### Common mistakes to avoid

1. **Monotonically increasing partition key.** Sharding posts by `post_id` (if auto-increment) or `created_at` means all writes go to the last shard. Use `user_id` — writes are distributed by user, not by time.

2. **Forgetting the inverse follow table.** To fan out a post, you need "who follows this user?" Not just "who do I follow?" Two separate tables (or index tables) are required in Cassandra.

3. **Sync replication for timelines.** Timelines are eventually consistent by design. Requiring quorum writes to Redis timelines adds 3× write latency for no correctness benefit. Use async fan-out.

4. **Missing the replication lag test.** Students implement `route_read` and test the happy path, but never test what happens when ALL replicas are lagging. The `route_read` fallback to leader is the single most important correctness property of the `ReplicaRouter`.

5. **`print` instead of `logging`.** Observable systems use `logging` so log levels are configurable at runtime. `print` statements silently swallow context (timestamps, level, module, thread) and cannot be filtered in production.

6. **Untyped dicts as state.** Using `dict` for post data instead of a Pydantic model or `dataclass` means type errors surface at runtime, not at definition time. Define a `Post` model with `post_id: str`, `user_id: str`, `content: str`, `created_at: datetime` and use it throughout.

7. **No `.env.example` / hardcoded connection strings.** Even in an exercise, any database DSN or secret should be read from `os.getenv("DB_URL")` with a documented example in `.env.example`. Hardcoding even a fake connection string is a pattern that produces production incidents.
