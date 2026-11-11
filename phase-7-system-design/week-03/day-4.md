# Phase 7 · Week 03 · Day 4 — Partitioning, Sharding & Rebalancing

> Objective: Understand why and how to split data across nodes, pick the right strategy, and move data safely when topology changes.  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

---

## Concept

A single database node has hard limits: disk capacity, memory for hot working set, and throughput (IOPS, CPU for query execution). **Partitioning** (also called *sharding* when the partitions live on separate hosts) splits a dataset into disjoint subsets so each node owns a manageable slice.

Key goals:

- **Scale storage and throughput** linearly with node count.
- **Isolate failure domains** — one node's crash affects only its partitions.
- **Enable parallel query** execution across partitions.

The core trade-off: distributing data introduces **coordination overhead** (cross-partition queries, distributed transactions) and **operational complexity** (rebalancing, routing).

---

## How It Works Internally

### Range Partitioning

Rows are assigned to partitions by key ranges: `[A–F]`, `[G–M]`, `[N–Z]`.

```
Partition 0: keys 0000 – 3333
Partition 1: keys 3334 – 6666
Partition 2: keys 6667 – 9999
```

**Pros:** Range scans are efficient (one partition or a small contiguous set).  
**Cons:** Sequential writes (e.g., time-series with timestamp key) pile onto one partition — a **hotspot**.  
Mitigation: prefix keys with a hash bucket, or use a compound key where the first component scatters writes.

### Hash Partitioning

Apply a deterministic hash to the key, then `partition = hash(key) % N`.

**Pros:** Even distribution for random-access workloads.  
**Cons:** Destroys sort order — range queries become scatter-gather over all `N` nodes.

#### Consistent Hashing & Virtual Nodes

Naive modular hashing (`% N`) remaps *all* keys when `N` changes. **Consistent hashing** fixes this:

1. Map the hash space to a conceptual ring `[0, 2^32)`.
2. Each physical node owns one or more **virtual nodes** (vnodes) — positions on the ring.
3. A key belongs to the first vnode clockwise from `hash(key)`.
4. Adding/removing a node only migrates the keys in the affected arcs.

```
        0
       /|\
      / | \
   300  |  60
    |   |   |
   240  |  120
      \ | /
       \|/
       180

Each mark = a vnode position.
Keys route clockwise to the nearest vnode.
```

Virtual nodes serve two purposes:  
- **Even load distribution** — a single physical node gets many scattered vnodes rather than one large arc.  
- **Heterogeneous capacity** — a bigger machine gets proportionally more vnodes.

### Directory-Based Partitioning

A **lookup service** (the "directory") maps each key (or key range) to a specific node. No hash, no range arithmetic — pure indirection.

**Pros:** Maximum flexibility; easy to move individual keys or hot ranges.  
**Cons:** Single point of failure and a write bottleneck if not replicated; every request hits the directory.

Used by: DynamoDB's partition map, MongoDB's config server, Vitess's VTGate routing table.

### Secondary Indexes on Partitioned Data

Two approaches:

| Approach | How | Read | Write |
|---|---|---|---|
| **Local (scatter-gather)** | Each partition indexes its own rows | Read from all partitions, merge | Single-partition write |
| **Global index** | One logical index partitioned separately by index term | Single-partition read | Write fan-out to index partitions |

Elasticsearch uses local indexes (scatter-gather). DynamoDB Global Secondary Indexes use asynchronously replicated global indexes.

---

## Rebalancing Strategies

When nodes join or leave, keys must migrate. Goal: **minimal data movement**, **no downtime**, **balanced load**.

### Fixed Number of Partitions

Pre-create more partitions than nodes (e.g., 1 000 partitions across 10 nodes = 100 each). When a node is added, steal some partitions from existing nodes. No repartitioning — just reassign ownership.

**Risk:** Choosing the partition count upfront; too few → large partitions hard to move; too many → metadata overhead.

Used by: Elasticsearch, Riak, Couchbase.

### Dynamic Partitioning

Split a partition when it exceeds a size threshold; merge when it falls below. The number of partitions grows with data volume.

Used by: HBase, RethinkDB, MongoDB.

### Proportional to Nodes (Cassandra / consistent hashing)

Each new node steals a fixed fraction of keys from existing nodes by inserting vnodes on the ring. Partition count scales with node count.

### Hotspot Mitigation

| Cause | Mitigation |
|---|---|
| Sequential key (timestamp) | Prefix key with `random_prefix(0..N)` |
| Celebrity/viral key | Append random suffix to write key; read from all suffixes and aggregate |
| Skewed range | Manual split; move hot range to dedicated partition |

---

## Worked Example — Consistent Hashing in Python

```python
"""
consistent_hashing.py

A minimal consistent hashing ring with virtual nodes.
Demonstrates key routing, node addition, and node removal
with migration tracking.
"""
from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _hash(value: str) -> int:
    """SHA-256 truncated to 32 bits for a compact ring."""
    digest = hashlib.sha256(value.encode()).digest()
    return int.from_bytes(digest[:4], "big")


RING_MAX = 2**32 - 1


class ConsistentHashRing:
    """
    Hash ring with configurable virtual-node count per physical node.

    Attributes:
        vnodes_per_node: Number of ring positions each physical node occupies.
    """

    def __init__(self, vnodes_per_node: int = 150) -> None:
        self.vnodes_per_node = vnodes_per_node
        # ring: sorted list of (position, physical_node_name)
        self._ring: list[tuple[int, str]] = []
        # physical nodes currently in the ring
        self._nodes: set[str] = set()

    # ------------------------------------------------------------------
    # Topology mutations
    # ------------------------------------------------------------------

    def add_node(self, node: str) -> list[tuple[str, str]]:
        """
        Add a physical node, inserting `vnodes_per_node` positions.

        Returns a list of (key, new_owner) migration events for keys
        that were previously owned by a different node.
        """
        if node in self._nodes:
            log.warning("Node %s already in ring — skipping", node)
            return []

        self._nodes.add(node)
        migrations: list[tuple[str, str]] = []

        for i in range(self.vnodes_per_node):
            vnode_key = f"{node}#vnode{i}"
            position = _hash(vnode_key)
            # Find what node *previously* owned this position
            prev_owner = self._route_position(position)
            if prev_owner and prev_owner != node:
                migrations.append((vnode_key, node))
                log.debug(
                    "Position %d: %s → %s (vnode %s)",
                    position,
                    prev_owner,
                    node,
                    i,
                )
            self._insert_sorted(position, node)

        log.info("Added node %s (%d vnodes, %d migrations)", node, self.vnodes_per_node, len(migrations))
        return migrations

    def remove_node(self, node: str) -> None:
        """Remove a node and its vnodes. Keys re-route to the next vnode owner."""
        if node not in self._nodes:
            raise ValueError(f"Node {node!r} not in ring")
        self._nodes.discard(node)
        self._ring = [(pos, n) for pos, n in self._ring if n != node]
        log.info("Removed node %s. Remaining nodes: %s", node, sorted(self._nodes))

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def get_node(self, key: str) -> Optional[str]:
        """Return the physical node responsible for `key`."""
        if not self._ring:
            return None
        position = _hash(key)
        return self._route_position(position)

    def _route_position(self, position: int) -> Optional[str]:
        if not self._ring:
            return None
        # Binary search for the first vnode position >= `position`
        lo, hi = 0, len(self._ring)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._ring[mid][0] < position:
                lo = mid + 1
            else:
                hi = mid
        # Wrap around to ring start if past the last vnode
        idx = lo % len(self._ring)
        return self._ring[idx][1]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def distribution(self, sample_keys: list[str]) -> dict[str, int]:
        """Count how many sample keys map to each node."""
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
        """Insert (position, node) maintaining sort order."""
        lo, hi = 0, len(self._ring)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._ring[mid][0] < position:
                lo = mid + 1
            else:
                hi = mid
        self._ring.insert(lo, (position, node))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _generate_keys(n: int) -> list[str]:
    return [f"user:{i:06d}" for i in range(n)]


if __name__ == "__main__":
    ring = ConsistentHashRing(vnodes_per_node=100)

    # Start with 3 nodes
    for name in ("node-A", "node-B", "node-C"):
        ring.add_node(name)

    keys = _generate_keys(10_000)
    dist = ring.distribution(keys)
    log.info("Distribution (3 nodes): %s", {k: f"{v/100:.1f}%" for k, v in sorted(dist.items())})

    # Add a 4th node — observe migrations
    migrations = ring.add_node("node-D")
    log.info("Keys migrated to node-D: %d (of %d ring positions)", len(migrations), 100)

    dist2 = ring.distribution(keys)
    log.info("Distribution (4 nodes): %s", {k: f"{v/100:.1f}%" for k, v in sorted(dist2.items())})

    # Remove node-A — keys re-route to surviving nodes
    ring.remove_node("node-A")
    dist3 = ring.distribution(keys)
    log.info("Distribution after removing node-A: %s", {k: f"{v/100:.1f}%" for k, v in sorted(dist3.items())})
```

**Expected output (approximate):**

```
INFO Added node node-A (100 vnodes, 0 migrations)
INFO Added node node-B (100 vnodes, ~100 migrations)
INFO Added node node-C (100 vnodes, ~100 migrations)
INFO Distribution (3 nodes): {'node-A': '33.x%', 'node-B': '33.x%', 'node-C': '33.x%'}
INFO Added node node-D (100 vnodes, ~100 migrations)
INFO Distribution (4 nodes): {'node-A': '25.x%', 'node-B': '25.x%', 'node-C': '25.x%', 'node-D': '25.x%'}
INFO Removed node node-A. Remaining nodes: ['node-B', 'node-C', 'node-D']
INFO Distribution after removing node-A: {...three nodes, ~33% each...}
```

Key observation: adding `node-D` migrates roughly `1/4` of keys (only those in arcs now owned by `node-D`), not all keys.

---

## ASCII Ring Diagram

```
Consistent Hash Ring  (positions 0 → 2^32, wrapping)

              0 / 2^32
                 |
       A-vn5 ---+--- B-vn1
      /                   \
  C-vn3                  A-vn2
    |       Physical       |
  B-vn4      nodes:        C-vn1
    |      A, B, C, D      |
  D-vn2                  D-vn1
      \                   /
       A-vn1 ---+--- C-vn2
                 |
              B-vn3

Key "user:001234" hashes to position X.
Routing: walk clockwise from X to the nearest vnode.
That vnode's physical node owns the key.

Adding node D inserts D-vn* positions.
Keys in arcs between the predecessor and D-vn* migrate to D.
All other keys stay put.
```

---

## Exercises

1. **Range vs hash:** Given a time-series sensor dataset (key = `sensor_id:timestamp`), explain which partition strategy you'd choose and why. What specific hotspot risk arises with a pure timestamp key?

2. **Virtual nodes:** In the `ConsistentHashRing` code above, increase `vnodes_per_node` from 100 to 500 and re-run the distribution test with 10 000 keys and 3 nodes. How does load balance change? What is the memory trade-off?

3. **Rebalancing:** Modify `ConsistentHashRing.remove_node` to return a list of `(key_pattern, old_owner, new_owner)` tuples (hint: iterate over removed vnode positions and find the new owner for each position). Write a pytest test asserting that after removing a node, all previously-routed keys resolve to a different node only if they fall in a migrated arc.

4. **Secondary index design:** You have a partitioned `orders` table (partitioned by `order_id` hash). Users frequently query `orders WHERE customer_id = ?`. Design both a local (scatter-gather) and a global index approach. For each, state the write amplification factor and read latency characteristic.

5. **Hotspot mitigation:** A social media platform stores posts by `user_id`. A celebrity with 50M followers creates a post, generating 2M reads/sec to a single partition. Propose two complementary mitigation strategies without repartitioning the entire dataset.

(Solutions: ../solutions/week-03/day-4.md)

---

## Self-check

- [ ] I can explain why modular hashing (`% N`) causes expensive rebalancing and how consistent hashing avoids it.
- [ ] I can describe the role of virtual nodes and their effect on load distribution and migration cost.
- [ ] I can distinguish range, hash, and directory partitioning by their scan, write, and rebalancing trade-offs.
- [ ] I can implement a consistent hash ring with add/remove operations in Python without reference.
- [ ] I can design a secondary index for a partitioned table and state the read/write trade-offs of local vs. global indexes.
- [ ] I can identify and mitigate hotspots for both sequential-key and high-cardinality-value scenarios.

---

## Supplementary Reading

- *Designing Data-Intensive Applications*, Martin Kleppmann — Chapter 6 (Partitioning)
- [Amazon DynamoDB: How it works — Partitions and data distribution](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/HowItWorks.Partitions.html)
- [Cassandra: Consistent Hashing and Virtual Nodes](https://cassandra.apache.org/doc/latest/cassandra/architecture/dynamo.html)
- [Google Spanner: Partitioning and Rebalancing](https://research.google/pubs/pub39966/)
- [Vitess: VTGate routing and sharding](https://vitess.io/docs/concepts/sharding/)
