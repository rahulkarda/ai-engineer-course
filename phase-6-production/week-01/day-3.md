# Phase 6 · Week 01 · Day 3 — Replication & Partitioning / Sharding

> Objective: Understand how data is replicated across nodes and partitioned across shards, and reason about the trade-offs for AI workloads.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

## Concept

### Replication

Replication means **keeping a copy of the same data on multiple nodes**. Motivations:

1. **High availability** — if one node fails, others serve traffic
2. **Read scalability** — route reads to any replica, spreading load
3. **Geo-distribution** — put a replica close to each user region

Three replication topologies:

**Single-leader (primary/replica)**
All writes go to one leader. The leader propagates changes to followers. Reads can go to any node (with the consistency caveats from Day 2). Most common. Examples: PostgreSQL streaming replication, Kafka topic partitions.

```
Client ──write──▶ Leader ──replicate──▶ Follower-1
                         └──replicate──▶ Follower-2
Client ──read───▶ Follower-1  (may be stale)
```

**Multi-leader**
Multiple nodes accept writes. Changes are propagated to all others. Used for multi-datacenter active-active setups. Introduces write conflicts that must be resolved.

**Leaderless (Dynamo-style)**
Any node can accept writes. Quorum reads/writes (W+R>N) ensure consistency. Amazon DynamoDB, Cassandra, Riak use variants of this.

### Replication lag

In async replication, followers are always slightly behind the leader. This lag can range from milliseconds to seconds under load. Consequences for AI:

- A vector DB follower may not yet have the embedding you just indexed
- An agent reading its own previous write from a replica may see stale state
- Two agents reading from different replicas may see different versions of a shared document

Mitigation patterns: **read-your-own-writes** (route a user's reads to the leader for 1 second after their write), **monotonic reads** (a single user always reads from the same replica).

### Partitioning (sharding)

Partitioning means **splitting data across nodes so each node holds a subset**. Goal: scale *beyond* what one node can store or serve.

Two fundamental strategies:

**Range partitioning**
Keys are sorted; contiguous ranges go to the same shard.

```
Shard 0: keys  A–F
Shard 1: keys  G–M
Shard 2: keys  N–Z
```

Advantage: range scans are efficient (all keys in "G–J" are on Shard 1).
Disadvantage: **hot spots** — if all writes have keys starting with "A", Shard 0 is overloaded.

**Hash partitioning**
Apply a hash function to the key; the hash determines the shard.

```
shard = hash(key) % num_shards
```

Advantage: even distribution; no hot spots for random keys.
Disadvantage: range scans are expensive (must query all shards).

### Consistent hashing

Consistent hashing places both keys and nodes on a circular ring. A key belongs to the first node clockwise from its hash position. Adding/removing a node rebalances only `1/N` of keys instead of all keys.

Used by: Cassandra, DynamoDB, Redis Cluster, many CDNs.

```
        0°
       node-A
  270° ──●────── 90°
     node-C  node-B
       180°
```

### Sharding for AI workloads

| Data type | Common partition key | Strategy |
|---|---|---|
| User embeddings | user_id | Hash (even distribution) |
| Document chunks | document_id | Hash |
| Time-series metrics | (metric_name, time_bucket) | Range on time (hot recent shard!) |
| Conversation history | session_id | Hash |
| Geo-aware vector search | geo_cell | Range |

**Hot-shard problem in AI**: if all queries target the "latest" time bucket, one shard gets all load. Mitigation: add a random suffix to the partition key (scatter), then read from all suffixed shards and merge (gather).

## How it works internally

### Replication log / WAL

Leaders propagate changes via a replication log (sometimes identical to the write-ahead log used for crash recovery). Each entry is a logical operation ("set key=K value=V") or a physical block change. Followers apply the log in order.

Two formats:
- **Statement-based**: log the SQL/command. Risk: non-deterministic functions (`NOW()`, `RANDOM()`) diverge.
- **Row-based** (logical): log the before/after values of each row. Safe but verbose.

### Resharding

When you add shards, consistent hashing minimises movement. With naive modulo hashing (`hash(key) % N`), going from N=3 to N=4 moves ~75% of keys. Consistent hashing moves ~25% (1/N fraction).

In practice, resharding is still disruptive: it requires a migration window, a dual-read period where both old and new shards are consulted, and a cutover. Plan for it upfront.

## Worked example

```python
"""
day3_sharding.py — Demonstrate hash partitioning and consistent hashing.

Simulates routing keys to shards and measuring distribution evenness.

Run: python day3_sharding.py
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
)
log = logging.getLogger(__name__)

NUM_VNODES = int(os.getenv("NUM_VNODES", "150"))  # virtual nodes per physical node


def _hash_key(key: str) -> int:
    """Stable 32-bit hash of a string key."""
    return int(hashlib.md5(key.encode(), usedforsecurity=False).hexdigest(), 16)


# ── Naive modulo sharding ────────────────────────────────────────────────────

class NaiveShardRouter:
    def __init__(self, num_shards: int) -> None:
        self.num_shards = num_shards

    def shard_for(self, key: str) -> int:
        return _hash_key(key) % self.num_shards


# ── Consistent hashing ───────────────────────────────────────────────────────

@dataclass
class ConsistentHashRing:
    """
    Consistent hash ring with virtual nodes.
    Each physical node occupies NUM_VNODES positions on the ring.
    """

    _ring: dict[int, str] = field(default_factory=dict)
    _sorted_positions: list[int] = field(default_factory=list)
    _nodes: set[str] = field(default_factory=set)

    def add_node(self, node_id: str) -> None:
        self._nodes.add(node_id)
        for i in range(NUM_VNODES):
            vnode_key = f"{node_id}#vnode{i}"
            pos = _hash_key(vnode_key) % (2**32)
            self._ring[pos] = node_id
        self._sorted_positions = sorted(self._ring.keys())
        log.debug("Added node %s (%d vnodes)", node_id, NUM_VNODES)

    def remove_node(self, node_id: str) -> None:
        self._nodes.discard(node_id)
        for i in range(NUM_VNODES):
            vnode_key = f"{node_id}#vnode{i}"
            pos = _hash_key(vnode_key) % (2**32)
            self._ring.pop(pos, None)
        self._sorted_positions = sorted(self._ring.keys())
        log.debug("Removed node %s", node_id)

    def node_for(self, key: str) -> str:
        if not self._ring:
            raise RuntimeError("Ring is empty")
        pos = _hash_key(key) % (2**32)
        # Find first position >= pos (clockwise)
        for ring_pos in self._sorted_positions:
            if ring_pos >= pos:
                return self._ring[ring_pos]
        # Wrap around to the first node
        return self._ring[self._sorted_positions[0]]


def measure_distribution(router: NaiveShardRouter | ConsistentHashRing, keys: list[str], label: str) -> None:
    counts: dict[str, int] = defaultdict(int)
    for key in keys:
        if isinstance(router, NaiveShardRouter):
            shard = str(router.shard_for(key))
        else:
            shard = router.node_for(key)
        counts[shard] += 1

    total = len(keys)
    log.info("── %s ── %d keys across %d buckets", label, total, len(counts))
    for bucket in sorted(counts):
        pct = counts[bucket] / total * 100
        bar = "█" * int(pct / 2)
        log.info("  %-12s %5d  %5.1f%%  %s", bucket, counts[bucket], pct, bar)


def simulate_node_add(ring: ConsistentHashRing, keys: list[str]) -> None:
    """Show how many keys move when a new node is added."""
    before = {k: ring.node_for(k) for k in keys}
    ring.add_node("node-D")
    after = {k: ring.node_for(k) for k in keys}
    moved = sum(1 for k in keys if before[k] != after[k])
    log.info(
        "After adding node-D: %d / %d keys moved (%.1f%%)  expected ≈25%%",
        moved,
        len(keys),
        moved / len(keys) * 100,
    )


if __name__ == "__main__":
    # Generate synthetic document IDs (like vector DB chunk IDs)
    keys = [f"doc_{i:05d}_chunk_{j}" for i in range(200) for j in range(3)]
    log.info("Total keys: %d", len(keys))

    # Naive modulo sharding — 3 shards
    naive = NaiveShardRouter(num_shards=3)
    measure_distribution(naive, keys, "Naive modulo (3 shards)")

    # Consistent hashing — 3 nodes
    ring = ConsistentHashRing()
    for n in ("node-A", "node-B", "node-C"):
        ring.add_node(n)
    measure_distribution(ring, keys, "Consistent hashing (3 nodes)")

    # Add a 4th node and measure key movement
    simulate_node_add(ring, keys)
```

## Exercises

1. Run the script and compare the distribution evenness between naive modulo and consistent hashing. Increase `NUM_VNODES` to 300 and observe whether distribution improves.

2. Implement a `range_shard_for(key: str, boundaries: list[str]) -> int` function that assigns a key to a range shard based on lexicographic ordering. Test it with embedding IDs of the form `user_{uuid}`.

3. Simulate a **replication lag** scenario: create a `ReplicatedStore` with a leader and two followers. Writes go to the leader and are replicated with a random delay of 0–500 ms. Implement `read_your_own_writes(key, write_ts)` that waits (up to 1 s) until the replica has caught up to the given timestamp before returning the value.

4. An AI system stores 50 M user-session embeddings and must add a 4th shard to an existing 3-shard cluster. Estimate what fraction of embeddings must be moved under (a) naive modulo resharding, (b) consistent hashing. Explain in comments.

(Solutions: ../solutions/week-01/day-3.md)

## Self-check

- [ ] I can describe single-leader, multi-leader, and leaderless replication and their trade-offs
- [ ] I can explain replication lag and name two read-consistency mitigation patterns
- [ ] I can implement hash partitioning and explain its distribution properties
- [ ] I can explain why consistent hashing minimises key movement when adding/removing nodes
- [ ] I can identify hot-shard problems in AI workloads and propose a scatter-gather mitigation

## Supplementary reading

- Designing Data-Intensive Applications (Kleppmann) — Chapter 5: Replication, Chapter 6: Partitioning
- [Consistent Hashing and Random Trees](https://www.cs.princeton.edu/courses/archive/fall09/cos518/papers/chash.pdf) — Karger et al., 1997
- [Amazon DynamoDB: A Scalable, Predictably Performant, and Fully Managed NoSQL Database Service](https://www.usenix.org/conference/atc22/presentation/elhemali) — USENIX ATC 2022
