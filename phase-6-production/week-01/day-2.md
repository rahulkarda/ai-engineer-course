# Phase 6 · Week 01 · Day 2 — CAP Theorem & Consistency Models

> Objective: Understand CAP's trade-offs and choose the right consistency model for a given AI workload.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

## Concept

### The CAP theorem

In 2000, Eric Brewer conjectured — and in 2002, Gilbert & Lynch proved — that a distributed system can guarantee at most **two** of the following three properties simultaneously:

| Property | Meaning |
|---|---|
| **C**onsistency | Every read receives the most recent write (or an error) |
| **A**vailability | Every request receives a non-error response (may not be latest) |
| **P**artition tolerance | The system continues operating despite network partitions |

**Partition tolerance is non-negotiable** in any real network. Packets get dropped; switches fail; network segments become isolated. You will always experience partitions. The real choice is: **when a partition occurs, do you sacrifice C or A?**

- **CP systems** (prefer consistency): return an error or stale-data refusal rather than serve possibly-outdated data. Example: a distributed lock, a financial ledger.
- **AP systems** (prefer availability): return the best available answer, possibly stale. Example: a shopping cart, a social-media feed, a semantic search cache.

### CAP in the context of AI systems

| Component | Typical choice | Rationale |
|---|---|---|
| Vector database (ANN search) | AP | Stale embeddings → slightly worse recall, not a disaster |
| Feature store (online) | CP | Stale features → wrong model prediction → real cost |
| Agent session state | CP | Two agents seeing different states → conflicting actions |
| LLM response cache | AP | Serving a cached answer is fine even if cache is slightly stale |
| Distributed rate limiter | CP | Under-counting requests → billing/quota violation |

### Consistency models (spectrum)

CAP's "C" is actually the strongest possible consistency. Real systems offer a spectrum:

```
Strong ←————————————————————————————→ Weak
  |          |          |          |
Linearisability  Sequential  Causal  Eventual
```

**Linearisability (strong consistency)**
Every operation appears to take effect instantaneously at some point between its start and end. Reads always see the latest committed write. Cost: high latency, low availability under partitions.

**Sequential consistency**
Operations appear in some global order that respects each process's local order. Reads may not see the absolute latest write but will see writes in order. Weaker than linearisability.

**Causal consistency**
If write A causally precedes write B (B was done *because of* A), all nodes see A before B. Concurrent writes may appear in different orders on different nodes.

**Eventual consistency**
Given no new writes, all replicas will *eventually* converge to the same value. No guarantee on when. Reads may return any version.

### Eventual consistency is not chaos

Eventual consistency is well-defined and widely used successfully. It requires:
1. A conflict-resolution strategy (last-write-wins, merge functions, CRDTs)
2. Idempotent writes (covered Day 5)
3. Application tolerance for temporary divergence

## How it works internally

### What a partition actually looks like

```
Node A ──────────── Node B
   |     ✗ packets    |
   |    dropped here  |
   └── Node C ────────┘
```

Nodes A and B can still reach C but cannot reach each other. Any write to A is not visible to B. A CP system would refuse writes on the minority partition. An AP system would accept writes on both sides, creating **divergent state** that must be reconciled later.

### Conflict resolution strategies

When two nodes both accepted writes to the same key during a partition:

| Strategy | Mechanism | Risk |
|---|---|---|
| Last-write-wins (LWW) | Keep the write with the highest timestamp | Clock skew can cause data loss |
| Version vectors | Track per-node version; detect concurrent writes | More complex; requires merge logic |
| CRDTs | Data types designed for safe merge (counters, sets) | Expressiveness limited |
| Application-level merge | Custom logic knows domain semantics | Most powerful but must be written |

### Read/write quorums

A quorum system with N replicas, write quorum W, read quorum R satisfies strong consistency when **W + R > N**. For N=3, W=2, R=2: any read overlaps with any write by at least 1 replica, guaranteeing the latest write is seen.

- W=3, R=1 → fast reads, slow writes, strong consistency
- W=1, R=3 → fast writes, slow reads, strong consistency
- W=1, R=1 → fastest, but no consistency guarantee

## Worked example

```python
"""
day2_consistency.py — Simulate eventual consistency in a replicated key-value store.

Three "nodes" hold their own in-memory state. Writes go to the primary;
propagation to replicas is asynchronous and occasionally dropped (partition).
Read from any replica may return stale data.

Run: python day2_consistency.py
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)-10s %(message)s",
)

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "7"))
random.seed(RANDOM_SEED)

# Probability that a replication message is dropped (simulates a partition)
REPLICATION_DROP_RATE = float(os.getenv("REPLICATION_DROP_RATE", "0.4"))


@dataclass
class Node:
    node_id: str
    _store: dict[str, tuple[Any, float]] = field(default_factory=dict)  # key → (value, timestamp)
    log: logging.Logger = field(init=False)

    def __post_init__(self) -> None:
        self.log = logging.getLogger(self.node_id)

    def write(self, key: str, value: Any, timestamp: float) -> None:
        existing = self._store.get(key)
        if existing is None or timestamp > existing[1]:
            self._store[key] = (value, timestamp)
            self.log.debug("write  key=%s value=%s ts=%.4f", key, value, timestamp)

    def read(self, key: str) -> tuple[Any | None, float | None]:
        entry = self._store.get(key)
        if entry is None:
            return None, None
        return entry


class EventuallyConsistentCluster:
    """
    Three-node cluster with async replication.
    Primary accepts all writes; replicas receive async propagation.
    Reads can target any node.
    """

    def __init__(self, drop_rate: float = REPLICATION_DROP_RATE) -> None:
        self.primary = Node("primary")
        self.replica_a = Node("replica_a")
        self.replica_b = Node("replica_b")
        self.replicas = [self.replica_a, self.replica_b]
        self.drop_rate = drop_rate
        self.log = logging.getLogger("cluster")

    def write(self, key: str, value: Any) -> float:
        """Write to primary. Propagate asynchronously (with possible drops)."""
        ts = time.monotonic()
        self.primary.write(key, value, ts)

        for replica in self.replicas:
            if random.random() < self.drop_rate:
                self.log.warning(
                    "replication DROP  key=%s → %s", key, replica.node_id
                )
            else:
                replica.write(key, value, ts)

        return ts

    def read(self, key: str, node: str = "primary") -> tuple[Any | None, float | None]:
        """Read from a named node — may return stale data from replicas."""
        target: Node
        match node:
            case "primary":
                target = self.primary
            case "replica_a":
                target = self.replica_a
            case "replica_b":
                target = self.replica_b
            case _:
                raise ValueError(f"Unknown node: {node}")
        return target.read(key)


def demonstrate_stale_reads() -> None:
    cluster = EventuallyConsistentCluster()

    log = logging.getLogger("demo")
    log.info("Writing embedding_version=v2 to cluster")
    cluster.write("embedding_version", "v2")

    log.info("Immediately reading from all nodes:")
    for node_name in ("primary", "replica_a", "replica_b"):
        value, ts = cluster.read("embedding_version", node=node_name)
        log.info("  %-12s → %s  (ts=%s)", node_name, value, f"{ts:.4f}" if ts else "None")

    log.info(
        "\nObservation: replicas that dropped replication still serve 'None' (stale)."
        " Under eventual consistency this is expected — they will converge"
        " once replication succeeds."
    )


if __name__ == "__main__":
    demonstrate_stale_reads()
```

Run several times (or set `REPLICATION_DROP_RATE=0.9`) to observe replica divergence.

## Exercises

1. Modify the cluster to implement **last-write-wins** reconciliation: after a "network heal" (call a `heal()` method), all replicas sync from the primary. Verify with assertions that all three nodes agree on the value after healing.

2. Implement a **quorum read**: read from all three nodes, collect the values, and return the one with the highest timestamp. What is the minimum `drop_rate` at which a quorum read of 2/3 nodes still returns the correct value?

3. Design a `VersionVector` dataclass that tracks per-node write counts. Show how it detects that two concurrent writes (one to replica_a, one to replica_b during a partition) are in conflict rather than one superseding the other.

4. An AI system stores "user preference embeddings" in an eventually-consistent store. A user updates their preferences. List two concrete failure modes caused by stale reads, and propose a mitigation for each.

(Solutions: ../solutions/week-01/day-2.md)

## Self-check

- [ ] I can state the CAP theorem and explain why P is non-negotiable
- [ ] I can classify a system as CP or AP given its failure behaviour
- [ ] I can order the four consistency levels (linearisability → eventual) by strength and cost
- [ ] I can compute whether a quorum configuration W+R>N guarantees strong consistency
- [ ] I can describe last-write-wins and version vectors as conflict-resolution strategies

## Supplementary reading

- Designing Data-Intensive Applications (Kleppmann) — Chapter 5: Replication, Chapter 9: Consistency and Consensus
- [CAP Twelve Years Later: How the "Rules" Have Changed](https://www.infoq.com/articles/cap-twelve-years-later-how-the-rules-have-changed/) — Eric Brewer, 2012
- [Please stop calling databases CP or AP](https://martin.kleppmann.com/2015/05/11/please-stop-calling-databases-cp-or-ap.html) — Martin Kleppmann
