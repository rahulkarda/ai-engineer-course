# Solutions — Week 01 Assessment

## Conceptual answers

### 1. Fallacies violated

Violates Fallacy 1 (network is reliable) and Fallacy 2 (latency is zero). Even on a LAN, NICs fail, switches have buffers that overflow under load, and GC pauses on one node create omission failures from another node's perspective. LAN latency is non-zero: a p99 network call in a busy data center can be 5–50 ms vs. p50 of 0.5 ms.

Concrete failure to show: during load testing, induce 30% CPU load on one tool server. Observe that agent tool calls start timing out despite "same LAN" — the kernel's network receive buffer fills up (omission failure), and the agent either hangs or retries, causing cascading load.

### 2. CP vs AP for agent session store

**CP** is the right choice. Rationale: if two agents read stale state and both believe they should perform action X, the action executes twice. This causes duplicate API calls, conflicting state mutations, or billing errors. The cost of unavailability (one agent waits for a partition to resolve) is lower than the cost of two agents acting on stale, divergent state.

AP sacrifices: during a partition, an agent might not be able to read session state at all (CP refuses stale reads), causing a temporary hang. With AP, the agent reads stale state and proceeds — which may be worse.

### 3. Eventual vs causal consistency

**Eventual**: Two agents A and B both read a shared document. A rewrites section 2 based on section 1. B rewrites section 3 independently. Under eventual consistency, the node serving A and the node serving B may disagree on which sections are "current" until the replicas converge. There is no ordering guarantee: B might see A's new section 2 before or after A sees B's new section 3.

**Causal**: If A's read of section 1 *caused* its write of section 2, causal consistency guarantees that any node serving B will see A's new section 2 *before* B's write to section 3 is applied. The causal chain is preserved.

Why it matters: an agent summarising a document should never see section 3 without also seeing the section 2 it references. Causal consistency prevents this "time travel" read. Eventual consistency does not.

### 4. Replication lag on preference vector

The agent sees the version that was on the replica at read time — possibly the version *before* the write to the primary. If the write happened 200 ms ago and the replica is 200 ms behind, the agent reads the old preference vector.

Two mitigation patterns:

**Read-your-own-writes**: After a preference update, tag the response with the write version. For the next 5 seconds (or until the replica catches up), route that user's reads to the primary. Implementation: store `(user_id, write_version, expiry_ts)` in a fast local cache; if a read for that user arrives before expiry, forward to primary.

**Monotonic reads with version piggybacking**: Include the write version in the API response. The client passes it on subsequent reads as `min_version`. The replica checks its local version; if behind, it either waits for replication or redirects to primary.

### 5. Consistent hashing key movement

Approximately **1/4 (25%)** of embeddings move with consistent hashing, vs. ~75% with naive modulo hashing.

Under consistent hashing, the new node takes over a contiguous arc of the ring from the three existing nodes. Each existing node gives up ~1/(N+1) of its keyspace. For N=3 → N=4: each existing node gives up 1/4 of its keys; total moved = 3 × (1/4) × (1/3 of total) = 1/4 of all keys ≈ 2.5 M.

Under naive modulo (hash % N): almost all keys change shards because their `hash % 4` differs from `hash % 3`. In practice ~75% must be migrated.

Operational significance: at 3 KB/embedding, 2.5 M × 3 KB = 7.5 GB to migrate vs. 22.5 GB. With a 500 MB/s migration bandwidth, 15 s vs. 45 s of migration window. During migration, dual-read latency increases for moved keys.

### 6. Raft with node failures

**Losing 2 of 5 nodes**: 3 nodes remain → majority = 3. The 3 remaining nodes can elect a leader and accept writes. System remains operational.

**Losing 3 of 5 nodes**: 2 nodes remain → majority requires 3. No leader can be elected. The cluster becomes **unavailable** — it refuses writes (CP choice). This prevents split-brain: the remaining 2 nodes cannot safely accept writes without risking divergence from a partition where the "dead" nodes are actually alive.

### 7. Lamport timestamps vs vector clocks

Lamport timestamps give only a **partial order**: `L(A) < L(B)` means either A caused B, or A and B are concurrent and it's a coincidence. It does NOT prove causality.

If A's LT=15 and B's LT=12, and they wrote concurrently during a partition, LWW picks A's write — but this is **arbitrary**. B's write (LT=12) may have happened at the same wall-clock moment or even later, but happened to have a lower Lamport counter at that node.

A vector clock reveals that: if A's VC = (15,0) and B's VC = (0,12) (two-node example), neither dominates — they are provably concurrent. The system must surface this as a **conflict** and apply domain-specific resolution, rather than silently dropping B's write.

### 8. Idempotency for charge_payment

Without idempotency keys:
1. Agent calls `charge_payment(100, "acc-1")`
2. Server processes charge successfully
3. HTTP response is lost in transit (network timeout)
4. Agent times out and retries
5. Server receives second request — no way to tell it's a retry
6. Card is charged again: $200 total, one payment, customer angry

To make it safe:
1. Agent generates `idempotency_key = str(uuid.uuid4())` **before** the first call
2. Passes it in every call: `charge_payment(100, "acc-1", idempotency_key="ikey-abc")`
3. Server checks a durable store (Redis or DB) for the key before processing
4. If found: return the cached result (charge already processed)
5. If not found: process charge, store `(idempotency_key, result)` atomically, return result
6. Key must be stored **before or atomically with** the charge to survive server crash

Result: any number of retries produces exactly one charge.

---

## Coding challenge solution — DistributedRateLimiter

```python
# rate_limiter.py
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class RateLimiterNode:
    node_id: str
    window_seconds: float
    limit: int
    # per-node request timestamps from this node's perspective
    _buckets: dict[str, list[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._buckets[self.node_id] = []

    def record_request(self) -> None:
        self._prune_window()
        self._buckets.setdefault(self.node_id, []).append(time.monotonic())
        log.debug("%s recorded request  total_seen=%d", self.node_id, self._total_count())

    def gossip_state(self) -> dict[str, list[float]]:
        self._prune_window()
        return {k: list(v) for k, v in self._buckets.items()}

    def receive_gossip(self, payload: dict[str, list[float]]) -> None:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        for node_id, timestamps in payload.items():
            existing = set(self._buckets.get(node_id, []))
            merged = sorted(existing | set(ts for ts in timestamps if ts >= cutoff))
            self._buckets[node_id] = merged
        self._prune_window()
        log.debug("%s after gossip: total=%d", self.node_id, self._total_count())

    def is_allowed(self) -> bool:
        self._prune_window()
        return self._total_count() < self.limit

    def _total_count(self) -> int:
        return sum(len(v) for v in self._buckets.values())

    def _prune_window(self) -> None:
        cutoff = time.monotonic() - self.window_seconds
        for node_id in list(self._buckets.keys()):
            self._buckets[node_id] = [ts for ts in self._buckets[node_id] if ts >= cutoff]


@dataclass
class DistributedRateLimiter:
    window_seconds: float = 1.0
    limit: int = 10
    gossip_drop_rate: float = 0.0

    _nodes: list[RateLimiterNode] = field(init=False)

    def __post_init__(self) -> None:
        self._nodes = [
            RateLimiterNode(f"node-{i}", self.window_seconds, self.limit)
            for i in range(3)
        ]

    def request(self, node_idx: int) -> bool:
        node = self._nodes[node_idx]
        if not node.is_allowed():
            log.info("%s: rate limited", node.node_id)
            return False
        node.record_request()
        return True

    def gossip_round(self) -> None:
        payloads = [node.gossip_state() for node in self._nodes]
        for i, receiver in enumerate(self._nodes):
            for j, payload in enumerate(payloads):
                if i == j:
                    continue
                if random.random() < self.gossip_drop_rate:
                    log.debug("GOSSIP DROP node-%d → node-%d", j, i)
                    continue
                receiver.receive_gossip(payload)
```

```python
# test_rate_limiter.py
from __future__ import annotations

import time
import pytest
from rate_limiter import DistributedRateLimiter


def test_after_gossip_all_nodes_see_limit_reached() -> None:
    """Criterion 1: 10 requests on node-0 → all nodes deny after gossip."""
    rl = DistributedRateLimiter(window_seconds=5.0, limit=10, gossip_drop_rate=0.0)
    for _ in range(10):
        assert rl.request(0)
    rl.gossip_round()
    for i in range(3):
        assert not rl._nodes[i].is_allowed(), f"node-{i} should be rate-limited"


def test_isolated_nodes_allow_over_limit() -> None:
    """Criterion 2: 100% drops → node-1 and node-2 unaware, allow over limit."""
    rl = DistributedRateLimiter(window_seconds=5.0, limit=10, gossip_drop_rate=1.0)
    for _ in range(10):
        rl.request(0)
    rl.gossip_round()  # all dropped
    # node-1 still thinks count=0, should allow
    assert rl._nodes[1].is_allowed(), "node-1 should incorrectly allow (demonstrates AP divergence)"


def test_convergence_after_two_gossip_rounds() -> None:
    """Criterion 3: after isolation, 2 gossip rounds with 0% drops → convergence."""
    rl = DistributedRateLimiter(window_seconds=5.0, limit=10, gossip_drop_rate=1.0)
    for _ in range(5):
        rl.request(0)
    rl.gossip_round()  # all dropped — isolation period

    rl.network_drop_rate_override = 0.0
    rl.gossip_drop_rate = 0.0
    rl.gossip_round()
    rl.gossip_round()

    counts = [rl._nodes[i]._total_count() for i in range(3)]
    assert counts[0] == counts[1] == counts[2], f"Should converge: {counts}"


def test_window_expiry_allows_new_requests() -> None:
    """Criterion 4: after window elapses, old requests pruned, new requests allowed."""
    rl = DistributedRateLimiter(window_seconds=0.1, limit=5, gossip_drop_rate=0.0)
    for _ in range(5):
        rl.request(0)
    rl.gossip_round()
    assert not rl._nodes[0].is_allowed()

    time.sleep(0.15)  # wait for window to expire
    rl._nodes[0]._prune_window()
    assert rl._nodes[0].is_allowed(), "Window expired — should allow new requests"
```

### AP trade-off analysis (required for Debugging dimension)

Criterion 2 test demonstrates that the `DistributedRateLimiter` is an **AP system**:
- During the gossip drop period, each node is *available* (accepts requests)
- But they are *inconsistent* — node-0 knows about 10 requests, nodes 1 and 2 know about 0
- This allows total request count to exceed the intended limit (e.g., 30 instead of 10)

This is the fundamental AP trade-off: to remain available (no blocking gossip sync), the limiter accepts temporary over-limit requests. A CP limiter would block requests until a quorum of nodes confirms the count — adding latency and risking unavailability during partitions.

In practice: rate limiters typically accept this trade-off because the over-limit window is short (milliseconds to seconds), and the cost of incorrectly rejecting a valid request (customer impact) exceeds the cost of briefly allowing a few extra requests.
