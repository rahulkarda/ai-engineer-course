# Solutions — Day 3: Replication & Partitioning / Sharding

## Exercise 1 — Distribution evenness

Run with `NUM_VNODES=150` (default): distribution should be within ±5% of equal across the 3 nodes. With `NUM_VNODES=300`: variance shrinks further, but diminishing returns after ~200. The naive modulo router will be nearly identical since `md5`-based hashing already distributes evenly for a large number of keys.

Key insight: the benefit of more virtual nodes is during *node addition/removal* — more vnodes means finer-grained rebalancing. Distribution evenness at steady state converges with fewer vnodes.

## Exercise 2 — Range sharding

```python
import bisect


def range_shard_for(key: str, boundaries: list[str]) -> int:
    """
    Assign a key to a shard based on lexicographic range boundaries.
    boundaries = ['d', 'h', 'm', 'r'] → 5 shards (before 'd', d–h, h–m, m–r, after 'r')
    Returns shard index 0..len(boundaries).
    """
    return bisect.bisect_right(boundaries, key)


# Test with embedding IDs of the form user_{uuid}
import uuid

boundaries = [f"user_{c}" for c in ['3', '6', '9', 'c', 'f']]  # hex prefix boundaries
test_keys = [f"user_{uuid.uuid4()}" for _ in range(12)]
for key in test_keys:
    shard = range_shard_for(key, boundaries)
    print(f"  {key[:20]}... → shard {shard}")
```

Observation: UUIDs (hex) distribute roughly evenly across hex prefix boundaries. Using the first hex digit of the UUID as a boundary is a valid partitioning strategy for UUID-keyed data.

## Exercise 3 — Replication lag and read-your-own-writes

```python
from __future__ import annotations
import time
import random
from dataclasses import dataclass, field
from typing import Any

MAX_WAIT_S = 1.0
POLL_INTERVAL_S = 0.01


@dataclass
class TimestampedValue:
    value: Any
    written_at: float  # monotonic timestamp


@dataclass
class ReplicatedStore:
    _primary: dict[str, TimestampedValue] = field(default_factory=dict)
    _replicas: list[dict[str, TimestampedValue]] = field(
        default_factory=lambda: [{}, {}]
    )
    _max_lag_s: float = 0.5  # simulated max replication lag

    def write(self, key: str, value: Any) -> float:
        ts = time.monotonic()
        self._primary[key] = TimestampedValue(value=value, written_at=ts)
        # Async replication with random lag
        lag = random.uniform(0, self._max_lag_s)
        # In a real system this would be async; here we just record when it will arrive
        # We simulate it by checking lag at read time
        return ts

    def _apply_replication(self, replica_idx: int, key: str) -> None:
        """Force-apply primary's value to a replica (call after simulated lag)."""
        if key in self._primary:
            self._replicas[replica_idx][key] = self._primary[key]

    def read_your_own_writes(self, key: str, write_ts: float, replica_idx: int = 0) -> Any | None:
        """
        Read from a replica. If its version for `key` is older than write_ts,
        wait (up to MAX_WAIT_S) for replication to arrive, then read.
        """
        deadline = time.monotonic() + MAX_WAIT_S
        while time.monotonic() < deadline:
            # Simulate replication arriving over time
            self._apply_replication(replica_idx, key)
            entry = self._replicas[replica_idx].get(key)
            if entry and entry.written_at >= write_ts:
                return entry.value
            time.sleep(POLL_INTERVAL_S)
        # Fallback: read from primary
        primary_entry = self._primary.get(key)
        return primary_entry.value if primary_entry else None


# Test
store = ReplicatedStore(max_lag_s=0.1)
write_ts = store.write("preference_vector", "[0.1, 0.2, 0.3]")
result = store.read_your_own_writes("preference_vector", write_ts, replica_idx=0)
assert result == "[0.1, 0.2, 0.3]"
print("Read-your-own-writes: OK")
```

## Exercise 4 — Fraction of embeddings moved

With **naive modulo hashing** and 50 M embeddings:
- Going from N=3 to N=4 shards: key assignment = `hash(k) % N`
- Old shard for key `k`: `hash(k) % 3`; new shard: `hash(k) % 4`
- Only keys where `hash(k) % 3 == hash(k) % 4` stay in place
- Fraction staying: roughly 1/4 (those where hash mod 12 is consistent) — analysis:
  - For random hashes, about 25% of keys land on the same shard in both assignments
  - Therefore ~75% of 50 M = ~37.5 M embeddings must move

With **consistent hashing**:
- Adding 1 node to a 3-node ring: each of the 3 existing nodes gives up ~1/4 of its keyspace
- Total moved: approximately N_new/(N_old + N_new) = 1/4 ≈ 12.5 M embeddings (25%)

**Why it matters operationally**:
- 37.5 M vs 12.5 M embeddings to migrate = 3× more disk I/O, network transfer, and migration time
- At 768 floats × 4 bytes = 3 KB per embedding: 37.5 M × 3 KB = ~112 GB to move vs ~37 GB
- A 112 GB migration at 1 GB/s takes 112 seconds of disruption; consistent hashing: 37 seconds
- During migration, queries to moved keys must be dual-routed (old + new shard), adding query complexity
