# Solutions — Day 2: CAP Theorem & Consistency Models

## Exercise 1 — Heal and sync with assertions

```python
def heal(self) -> None:
    """Force-sync all replicas from primary (simulates partition heal + catch-up)."""
    import time
    for key, (value, ts) in self.primary._store.items():
        for replica in self.replicas:
            existing = replica._store.get(key)
            if existing is None or ts > existing[1]:
                replica._store[key] = (value, ts)
    self.log.info("All replicas synced from primary")


# Test
cluster = EventuallyConsistentCluster(drop_rate=1.0)
cluster.write("embedding_version", "v2")

# Before heal: replicas may differ
for node_name in ("primary", "replica_a", "replica_b"):
    v, _ = cluster.read("embedding_version", node=node_name)
    print(f"{node_name}: {v}")

cluster.heal()

# After heal: all nodes agree
primary_val, _ = cluster.read("embedding_version", node="primary")
assert cluster.read("embedding_version", node="replica_a")[0] == primary_val
assert cluster.read("embedding_version", node="replica_b")[0] == primary_val
print("All nodes converged:", primary_val)
```

## Exercise 2 — Quorum read analysis

```python
def quorum_read(self, key: str, quorum_size: int = 2) -> tuple[Any | None, float | None]:
    """
    Read from quorum_size nodes and return the value with the highest timestamp.
    For N=3, quorum_size=2: W+R=2+2=4>3 → strong consistency guaranteed.
    """
    nodes = [self.primary, self.replica_a, self.replica_b]
    results = []
    for node in nodes[:quorum_size]:  # simplified: read first quorum_size nodes
        val, ts = node.read(key)
        if val is not None:
            results.append((val, ts))

    if not results:
        return None, None
    # Return the value with the highest timestamp (LWW winner)
    return max(results, key=lambda x: x[1])
```

Minimum `drop_rate` for 2-of-3 quorum to still work: any `drop_rate < 1.0`. As long as at least 2 of 3 replications succeed, a quorum read of 2 is guaranteed to include the primary (which always has the latest write), returning the correct value. The quorum read of 2/3 fails only when both non-primary reads fail, which requires 2 specific drops — independent probability `drop_rate²`.

## Exercise 3 — Version vector for conflict detection

```python
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class VersionVector:
    """
    Per-node write counters. Tracks causal relationships.
    key: identifying which "object" this vector belongs to.
    """
    counters: dict[str, int] = field(default_factory=dict)

    def increment(self, node_id: str) -> None:
        self.counters[node_id] = self.counters.get(node_id, 0) + 1

    def dominates(self, other: "VersionVector") -> bool:
        """Return True if self is strictly causally after other."""
        all_keys = set(self.counters) | set(other.counters)
        return all(
            self.counters.get(k, 0) >= other.counters.get(k, 0) for k in all_keys
        ) and self.counters != other.counters

    def concurrent_with(self, other: "VersionVector") -> bool:
        """Neither dominates the other — writes happened concurrently."""
        return not self.dominates(other) and not other.dominates(self)


# Scenario: replica_a and replica_b each write during a partition
vv_a = VersionVector()
vv_b = VersionVector()

vv_a.increment("replica_a")   # replica_a writes once
vv_b.increment("replica_b")   # replica_b writes once independently

assert vv_a.concurrent_with(vv_b), "Should be concurrent — neither caused the other"
print("Conflict detected: writes are concurrent, application must resolve")
```

## Exercise 4 — Stale preference embeddings: failure modes and mitigations

**Failure mode 1 — Wrong recommendations**
An agent reads a user's stale preference vector (missing a recent "dislike" signal) and recommends items the user just indicated they don't want. The agent sees `embedding_version=v1` while the primary has `v2` with the new signal.
Mitigation: implement read-your-own-writes — route reads for the next 5 seconds after a preference update to the primary, not replicas.

**Failure mode 2 — Model/embedding version mismatch**
The primary was updated to use a new embedding model (`text-embed-3-large`). A replica serving queries still has the old model version in its config. Queries route to both: similarity scores are computed in incompatible embedding spaces, silently producing wrong results.
Mitigation: version the embedding model alongside the vectors. Reject queries where the requested model version doesn't match the stored version — fail loudly rather than silently.
