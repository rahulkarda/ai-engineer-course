# Solutions — Phase 7 · Week 3 · Day 4 — Partitioning, Sharding & Rebalancing

---

## Exercise 1 — Range vs Hash for a Time-Series Sensor DB

**Recommended strategy: range partitioning on a compound key.**

Pure timestamp as partition key with range partitioning creates a classic **monotonic hotspot**: every new write hits the newest partition exclusively. As long as sensors emit data in real time, the current-time partition absorbs 100 % of writes while all older partitions sit idle. This destroys write throughput and makes the hot partition a single point of contention.

**Why range over hash for this workload:**

The typical query pattern for time-series data is "give me sensor X readings from T1 to T2". Range partitioning preserves sort order, so such a query touches one or a small set of contiguous partitions — no scatter-gather. Hash partitioning would scatter a time-range scan across every shard.

**Mitigation — random prefix trick (salted range key):**

Prepend a short random bucket prefix to the timestamp key:

```
key = f"{random.randint(0, NUM_BUCKETS - 1):02d}:{sensor_id}:{timestamp_iso}"
```

Writes spread evenly across `NUM_BUCKETS` partitions (e.g., 16 or 32). Time-range reads fan out to all buckets in parallel and merge results at query time. The fan-out is bounded (fan = NUM_BUCKETS, not all shards) and is predictable.

**Alternative: compound key `(sensor_id, timestamp)`:**

If queries are almost always scoped to a single sensor, `(sensor_id, timestamp)` is better — writes spread across as many range partitions as there are active sensors, and a per-sensor range scan touches one partition. Hotspot risk only reappears if one sensor dominates write volume.

**Summary:**

| Scenario | Key choice | Hotspot risk | Mitigation |
|---|---|---|---|
| Pure timestamp | `timestamp` | Severe (all writes on newest partition) | Random prefix |
| Sensor + time | `(sensor_id, timestamp)` | Moderate (high-frequency sensors) | Random prefix on sensor_id |
| Random access only | `hash(sensor_id:timestamp)` | None | Range scans become scatter-gather |

---

## Exercise 2 — Virtual Nodes and Load Imbalance

**Setup:** 3 physical nodes, 150 virtual nodes per node = 450 total vnodes on a 2^32 ring.

**Expected imbalance (mathematical reasoning):**

Each vnode occupies a random position on the ring. The arc length between consecutive vnodes determines how many keys each vnode "owns". For uniformly distributed hash positions, arc lengths follow an exponential distribution. With V vnodes total, the expected fraction of the ring per vnode is 1/V = 1/450 ≈ 0.22 %.

The load on a single physical node is the sum of its V/3 = 150 arc lengths. By the central limit theorem, as V grows large, the distribution of a node's total load approaches a normal distribution centred at 1/3 with standard deviation proportional to 1/sqrt(V).

**Rough calculation:**

Standard deviation of a single arc ≈ 1/V (for exponential arcs). Node load = sum of 150 arcs, so:

```
σ_node ≈ sqrt(150) × (1/450) ≈ 12.25 / 450 ≈ 2.7 %
```

At 3 standard deviations (99.7 % confidence), max deviation from 33.3 % ≈ **±8 %** in the worst case, typically closer to ±5 % in practice.

**With 500 vnodes per node (1 500 total):**

```
σ_node ≈ sqrt(500) / 1500 ≈ 22.4 / 1500 ≈ 1.5 %
```

Max imbalance at 3σ drops to roughly **±4.5 %** — substantially better.

**Memory trade-off:**

Each vnode entry = one `(int, str)` tuple in the sorted ring. At 150 vnodes × 3 nodes = 450 entries, memory is negligible (< 50 KB). At 500 × 100 nodes = 50 000 entries, still only a few MB. The practical limit is routing lookup time: binary search is O(log V_total), so doubling vnodes adds one comparison per lookup — irrelevant in practice. Memory cost is real but small; prefer more vnodes for better balance unless you have thousands of nodes.

---

## Exercise 3 — `add_node`, `remove_node`, and `get_migration_plan`

Full implementation extending the lesson's `ConsistentHashRing` with a `get_migration_plan` method, plus pytest tests.

```python
# consistent_hashing_solution.py
"""
Extended ConsistentHashRing with get_migration_plan.

Usage:
    python consistent_hashing_solution.py
    pytest test_consistent_hashing_solution.py
"""
from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

RING_MAX = 2**32 - 1


def _hash(value: str) -> int:
    """SHA-256 truncated to 32 bits."""
    digest = hashlib.sha256(value.encode()).digest()
    return int.from_bytes(digest[:4], "big")


class ConsistentHashRing:
    """
    Consistent hash ring with configurable virtual-node count.

    Supports add_node, remove_node, get_node, and get_migration_plan.
    """

    def __init__(self, vnodes_per_node: int = 150) -> None:
        self.vnodes_per_node = vnodes_per_node
        # Sorted list of (position, physical_node_name)
        self._ring: list[tuple[int, str]] = []
        self._nodes: set[str] = set()

    # ------------------------------------------------------------------
    # Topology mutations
    # ------------------------------------------------------------------

    def add_node(self, node: str) -> None:
        """Add a physical node with vnodes_per_node virtual positions."""
        if node in self._nodes:
            log.warning("Node %s already in ring — skipping", node)
            return
        self._nodes.add(node)
        for i in range(self.vnodes_per_node):
            position = _hash(f"{node}#vnode{i}")
            self._insert_sorted(position, node)
        log.info("Added node %s (%d vnodes)", node, self.vnodes_per_node)

    def remove_node(self, node: str) -> None:
        """Remove a node and all its virtual positions.

        Keys previously routed to this node automatically re-route to the
        next clockwise vnode owner.

        Raises:
            ValueError: if node is not in the ring.
        """
        if node not in self._nodes:
            raise ValueError(f"Node {node!r} not in ring")
        self._nodes.discard(node)
        self._ring = [(pos, n) for pos, n in self._ring if n != node]
        log.info("Removed node %s. Remaining nodes: %s", node, sorted(self._nodes))

    # ------------------------------------------------------------------
    # Migration planning
    # ------------------------------------------------------------------

    def get_migration_plan(
        self, new_node: str
    ) -> list[tuple[str, str, str]]:
        """
        Compute the migration plan for adding `new_node` to the ring.

        For each virtual position that `new_node` would own, determine
        which existing node currently owns that arc and record the
        (vnode_position_label, old_owner, new_node) tuple.

        The ring is NOT modified. Call add_node separately if you want
        to commit the change.

        Returns:
            List of (position_label, from_node, to_node) describing
            which key ranges move and from where.

        Raises:
            ValueError: if new_node is already in the ring.
        """
        if new_node in self._nodes:
            raise ValueError(f"Node {new_node!r} already in ring")
        if not self._ring:
            # Empty ring — every vnode is a new assignment, no migration
            return []

        plan: list[tuple[str, str, str]] = []

        for i in range(self.vnodes_per_node):
            position = _hash(f"{new_node}#vnode{i}")
            current_owner = self._route_position(position)
            if current_owner and current_owner != new_node:
                label = f"{new_node}#vnode{i}@{position}"
                plan.append((label, current_owner, new_node))

        log.info(
            "Migration plan for %s: %d arcs to move", new_node, len(plan)
        )
        return plan

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def get_node(self, key: str) -> Optional[str]:
        """Return the physical node responsible for `key`."""
        if not self._ring:
            return None
        return self._route_position(_hash(key))

    def _route_position(self, position: int) -> Optional[str]:
        if not self._ring:
            return None
        lo, hi = 0, len(self._ring)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._ring[mid][0] < position:
                lo = mid + 1
            else:
                hi = mid
        idx = lo % len(self._ring)
        return self._ring[idx][1]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def distribution(self, sample_keys: list[str]) -> dict[str, int]:
        """Count sample keys per node."""
        counts: dict[str, int] = defaultdict(int)
        for k in sample_keys:
            node = self.get_node(k)
            if node:
                counts[node] += 1
        return dict(counts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _insert_sorted(self, position: int, node: str) -> None:
        lo, hi = 0, len(self._ring)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._ring[mid][0] < position:
                lo = mid + 1
            else:
                hi = mid
        self._ring.insert(lo, (position, node))
```

**Test file: `test_consistent_hashing_solution.py`**

```python
"""
pytest tests for ConsistentHashRing.add_node, remove_node, get_migration_plan.
"""
import pytest
from consistent_hashing_solution import ConsistentHashRing


# ---------------------------------------------------------------------------
# add_node
# ---------------------------------------------------------------------------

def test_add_node_populates_ring():
    ring = ConsistentHashRing(vnodes_per_node=10)
    ring.add_node("A")
    assert len(ring._ring) == 10
    assert all(node == "A" for _, node in ring._ring)


def test_add_node_idempotent():
    ring = ConsistentHashRing(vnodes_per_node=10)
    ring.add_node("A")
    ring.add_node("A")  # duplicate — should be silently ignored
    assert len(ring._ring) == 10


def test_add_multiple_nodes_scales_ring():
    ring = ConsistentHashRing(vnodes_per_node=50)
    ring.add_node("A")
    ring.add_node("B")
    ring.add_node("C")
    assert len(ring._ring) == 150  # 3 × 50


# ---------------------------------------------------------------------------
# remove_node
# ---------------------------------------------------------------------------

def test_remove_node_shrinks_ring():
    ring = ConsistentHashRing(vnodes_per_node=50)
    ring.add_node("A")
    ring.add_node("B")
    ring.remove_node("A")
    assert len(ring._ring) == 50
    assert all(node == "B" for _, node in ring._ring)


def test_remove_node_raises_for_unknown():
    ring = ConsistentHashRing(vnodes_per_node=10)
    ring.add_node("A")
    with pytest.raises(ValueError, match="not in ring"):
        ring.remove_node("nonexistent")


def test_remove_node_reroutes_keys():
    ring = ConsistentHashRing(vnodes_per_node=100)
    ring.add_node("A")
    ring.add_node("B")

    # Record routing for all sample keys with 2 nodes
    keys = [f"k:{i}" for i in range(1000)]
    before = {k: ring.get_node(k) for k in keys}

    ring.remove_node("A")

    # After removal, every key must route to B (the only survivor)
    for k in keys:
        assert ring.get_node(k) == "B"

    # Keys that were on B before removal must still be on B
    b_keys_before = [k for k, n in before.items() if n == "B"]
    for k in b_keys_before:
        assert ring.get_node(k) == "B"


# ---------------------------------------------------------------------------
# get_migration_plan
# ---------------------------------------------------------------------------

def test_migration_plan_empty_ring_returns_empty():
    ring = ConsistentHashRing(vnodes_per_node=50)
    plan = ring.get_migration_plan("A")
    assert plan == []


def test_migration_plan_does_not_modify_ring():
    ring = ConsistentHashRing(vnodes_per_node=50)
    ring.add_node("A")
    ring.add_node("B")
    before_len = len(ring._ring)
    ring.get_migration_plan("C")
    assert len(ring._ring) == before_len


def test_migration_plan_source_nodes_are_existing():
    ring = ConsistentHashRing(vnodes_per_node=50)
    ring.add_node("A")
    ring.add_node("B")
    plan = ring.get_migration_plan("C")
    for _, from_node, to_node in plan:
        assert from_node in {"A", "B"}
        assert to_node == "C"


def test_migration_plan_length_approximately_one_third():
    """With 3 equal existing nodes + 1 new node, ~1/4 of vnodes should migrate."""
    ring = ConsistentHashRing(vnodes_per_node=200)
    for name in ("A", "B", "C"):
        ring.add_node(name)
    plan = ring.get_migration_plan("D")
    # Each new vnode steals from one existing owner.
    # With 200 vnodes for D and 600 existing, ~200 arcs move (one per vnode of D).
    assert 100 <= len(plan) <= 200  # each D vnode maps to exactly one existing owner


def test_migration_plan_raises_if_node_already_in_ring():
    ring = ConsistentHashRing(vnodes_per_node=10)
    ring.add_node("A")
    with pytest.raises(ValueError, match="already in ring"):
        ring.get_migration_plan("A")


def test_key_stability_after_add_node():
    """Keys NOT in a migrated arc must resolve to the same node before and after."""
    ring = ConsistentHashRing(vnodes_per_node=150)
    ring.add_node("A")
    ring.add_node("B")

    keys = [f"user:{i}" for i in range(2000)]
    before = {k: ring.get_node(k) for k in keys}

    # Get migration plan tells us which arcs move to C
    plan = ring.get_migration_plan("C")
    ring.add_node("C")  # commit the change
    after = {k: ring.get_node(k) for k in keys}

    moved = [k for k in keys if before[k] != after[k]]
    stayed = [k for k in keys if before[k] == after[k]]

    # Keys that moved should go to C
    for k in moved:
        assert after[k] == "C"

    # Keys that stayed must be unchanged
    for k in stayed:
        assert after[k] == before[k]

    # Roughly 1/3 of keys should have moved (±10 %)
    fraction_moved = len(moved) / len(keys)
    assert 0.23 <= fraction_moved <= 0.43, f"Unexpected fraction moved: {fraction_moved:.2%}"
```

---

## Exercise 4 — Global Secondary Index on Email for a 10-Shard User Table

**Schema:** `users` table partitioned by `user_id` across 10 shards. Frequent query: `SELECT * FROM users WHERE email = ?`.

### Design

Create a separate **global secondary index (GSI)** table, itself partitioned by `hash(email)` across (say) 10 GSI shards:

```
GSI shard assignment:
  gsi_shard = hash(email) % 10

GSI entry:
  { email: "alice@example.com", user_id: "u-42" }
```

### Write Path

1. Application writes new user → routed to `user_shard = hash(user_id) % 10`.
2. After the primary write succeeds, write `(email, user_id)` to `gsi_shard = hash(email) % 10`.
3. Write amplification factor = **2** (one primary + one index write per user mutation).

If email changes, the old GSI entry must be deleted and a new one inserted — 3 operations total (delete old, insert new primary, insert new index).

### Read Path

1. Query `email = "alice@example.com"` → route to `gsi_shard = hash("alice@example.com") % 10`.
2. Retrieve `user_id` from GSI — single-shard read.
3. Use `user_id` to fetch full user record from `user_shard = hash(user_id) % 10` — second single-shard read.
4. Total: **2 single-shard reads**, no scatter-gather.

**Contrast with local (scatter-gather) index:** each of the 10 primary shards maintains its own local index on email. A query must fan out to all 10 shards and merge results. Read = 10 parallel shard reads vs. 2 for the GSI.

### Consistency Concern

The GSI write is **not atomic** with the primary write. If the application crashes between the primary write and the GSI write:

- The user record exists in the primary store.
- The GSI is missing the email → user_id mapping.
- Queries by email return no result even though the user exists.

**Mitigation options:**
1. **Two-phase commit** across primary and GSI shards (expensive, reduces availability).
2. **Async GSI update with reconciliation**: primary write succeeds; GSI update is queued in a durable outbox. A background process applies it with at-least-once semantics and an idempotency check.
3. **DynamoDB-style asynchronous GSI**: the database engine manages eventual consistency of the GSI, trading brief inconsistency windows for simplicity. Reads that require strong consistency must go to the primary.

---

## Exercise 5 — Hotspot Mitigation for a Celebrity Post

**Scenario:** Post `post_id=celebrity-123` gets 100 000 writes/sec (likes, comments) all routing to shard 4 (determined by `hash("celebrity-123") % N`).

### Strategy 1 — Write Sharding with Random Suffix

Append a random suffix to the write key:

```python
suffix = random.randint(0, 99)   # 100 buckets
write_key = f"celebrity-123:{suffix}"
```

Writes now distribute across 100 logical keys, spreading load across multiple shards (or at least across multiple storage keys on the same shard, reducing lock contention). Reads aggregate across all 100 buckets:

```python
total = sum(read(f"celebrity-123:{i}") for i in range(100))
```

**Trade-off:** Read amplification (100 reads instead of 1). Works best when writes vastly outnumber reads or when reads can be served from a cache.

### Strategy 2 — Caching Layer (Read Hot Data)

Place a distributed in-memory cache (Redis, Memcached) in front of the shard. The celebrity post's read model (e.g., like count) is cached with a short TTL (1–5 seconds). The vast majority of reads hit the cache, not the shard.

**Trade-off:** Consistency lag (readers see slightly stale counts). Acceptable for social metrics. Cache invalidation complexity increases. Does not help if the workload is write-dominant rather than read-dominant.

### Strategy 3 — Dedicated "Hot Key" Partition (Manual Split)

Detect the hot key (via monitoring or threshold-based alerting) and migrate it to a dedicated shard or a set of replicated shards using directory-based partitioning (a routing table override). The celebrity post gets its own isolated partition with dedicated resources.

**Trade-off:** Requires operational intervention or an automated hot-key detection system. The dedicated shard may be idle when the celebrity stops being hot. Most useful for predictable celebrity events (pre-announced content drops).

### Combined recommendation

For unpredictable spikes: **Strategy 1 (write sharding) + Strategy 2 (caching)** together are the most effective. The write sharding reduces write hotspot; the cache absorbs the read storm. Add Strategy 3 as a longer-term operational response if a specific key sustains high load indefinitely.
