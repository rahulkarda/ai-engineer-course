# Solutions — Phase 7 · Week 3 · Day 6 — Build Day: ShardingRouter + ReplicaRouter

---

## Part 1 — `ShardingRouter` Reference Implementation

Save as `src/data_layer/sharding_router.py`.

```python
"""
sharding_router.py

Consistent-hashing sharding router with virtual nodes.

Key design decisions:
- MD5 for speed; truncated to 32-bit for a compact ring.
- bisect module for O(log V) routing lookups.
- Pure dataclass to keep internal state explicit.
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
    """
    Routes keys to shards using a consistent hash ring with virtual nodes.

    The ring maps 32-bit integer positions to physical node names. Each
    physical node occupies `virtual_nodes` positions, evenly distributing
    load and minimising hot arcs.

    Args:
        virtual_nodes: Number of ring positions per physical node.
                       Higher values improve uniformity at the cost of
                       slightly more memory and lookup overhead.
                       Recommended: 100–300 for most deployments.

    Raises:
        RuntimeError: get_shard called on an empty ring.
        KeyError: remove_node called for an unknown node.
    """

    virtual_nodes: int = 150

    # Internal ring state — not exposed as constructor arguments.
    # _ring maps ring position → physical node name.
    # _sorted_keys is the sorted list of all ring positions for binary search.
    _ring: dict[int, str] = field(default_factory=dict, init=False, repr=False)
    _sorted_keys: list[int] = field(default_factory=list, init=False, repr=False)
    _nodes: set[str] = field(default_factory=set, init=False, repr=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_node(self, node: str) -> None:
        """
        Add a physical node to the ring.

        Places `virtual_nodes` positions on the ring for `node`.
        Duplicate adds are silently ignored.

        Args:
            node: Unique name for the physical node (e.g. "shard-0").
        """
        if node in self._nodes:
            logger.warning("add_node: %s already in ring — skipping", node)
            return

        self._nodes.add(node)
        for i in range(self.virtual_nodes):
            position = self._hash(f"{node}#{i}")
            self._ring[position] = node
            insort(self._sorted_keys, position)

        logger.info(
            "add_node: added %s (%d vnodes, ring size=%d)",
            node,
            self.virtual_nodes,
            len(self._sorted_keys),
        )

    def remove_node(self, node: str) -> None:
        """
        Remove a physical node and all its virtual positions.

        Keys previously routed to this node automatically re-route to the
        next clockwise owner after removal.

        Args:
            node: Name of the physical node to remove.

        Raises:
            KeyError: if `node` is not currently in the ring.
        """
        if node not in self._nodes:
            raise KeyError(f"remove_node: {node!r} not in ring")

        self._nodes.discard(node)
        for i in range(self.virtual_nodes):
            position = self._hash(f"{node}#{i}")
            if position in self._ring:
                del self._ring[position]
                # Remove from sorted list — O(V) linear scan, acceptable for
                # node-removal frequency (rare operational event).
                self._sorted_keys.remove(position)

        logger.info(
            "remove_node: removed %s, ring size=%d, remaining nodes=%s",
            node,
            len(self._sorted_keys),
            sorted(self._nodes),
        )

    def get_shard(self, key: str) -> str:
        """
        Return the physical node responsible for `key`.

        Performs a clockwise lookup on the sorted ring: find the first
        ring position >= hash(key); if none exists, wrap to the first
        position (ring semantics).

        Args:
            key: The key to route (any string).

        Returns:
            Physical node name.

        Raises:
            RuntimeError: if the ring has no nodes.
        """
        if not self._sorted_keys:
            raise RuntimeError("get_shard: ring is empty — add nodes first")

        position = self._hash(key)
        idx = bisect_left(self._sorted_keys, position)
        # Wrap around to ring start if past the last vnode
        idx = idx % len(self._sorted_keys)
        ring_position = self._sorted_keys[idx]
        return self._ring[ring_position]

    def get_distribution(self, keys: list[str]) -> dict[str, int]:
        """
        Return a count of how many keys from `keys` map to each shard.

        Args:
            keys: List of keys to route.

        Returns:
            Dict mapping node name → count of keys assigned to that node.
        """
        counts: dict[str, int] = defaultdict(int)
        for key in keys:
            counts[self.get_shard(key)] += 1
        return dict(counts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(value: str) -> int:
        """
        Return a stable 32-bit integer hash of `value` using MD5.

        MD5 is chosen for speed, not security. Truncating to 4 bytes gives
        a ring of 2^32 positions, large enough for practical deployments.
        """
        digest = hashlib.md5(value.encode(), usedforsecurity=False).digest()
        return int.from_bytes(digest[:4], "big")
```

---

## Part 2 — `ReplicaRouter` Reference Implementation

Save as `src/data_layer/replica_router.py`.

```python
"""
replica_router.py

Lag-aware read-replica router with round-robin over healthy replicas.

Design decisions:
- Pure dataclass with explicit internal state fields.
- Healthy = lag not recorded OR lag < max_lag_ms (strict less-than).
- Round-robin index is global across calls; filtering means the actual
  replica sequence may not be perfectly uniform when lagging replicas
  change, but it converges to even distribution quickly.
- Leader fallback emits WARNING — callers should alert on this.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ReplicaRouter:
    """
    Routes reads to healthy replicas (round-robin) and writes to the leader.

    A replica is "healthy" if its recorded lag is strictly below
    `max_lag_ms`. Replicas with no lag record are considered healthy.
    When no healthy replica exists, reads fall back to the leader.

    Args:
        leader: Node name for the primary/write leader.
        replicas: List of replica node names (read targets).
        max_lag_ms: Maximum acceptable replication lag in milliseconds.
                    Replicas at or above this threshold are skipped.

    Raises:
        ValueError: mark_replica_lagging / mark_replica_healthy called
                    with a replica not in self.replicas.
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
        """
        Record the current replication lag for `replica`.

        Logs a WARNING if the lag exceeds max_lag_ms (replica becomes
        ineligible for reads until mark_replica_healthy is called or
        lag is updated below threshold).

        Args:
            replica: Replica node name (must be in self.replicas).
            lag_ms: Current replication lag in milliseconds.

        Raises:
            ValueError: if `replica` is not in self.replicas.
        """
        if replica not in self.replicas:
            raise ValueError(
                f"mark_replica_lagging: {replica!r} not in replicas {self.replicas}"
            )
        self._lag[replica] = lag_ms
        if lag_ms >= self.max_lag_ms:
            logger.warning(
                "mark_replica_lagging: %s lag=%dms exceeds threshold=%dms — "
                "excluded from reads",
                replica,
                lag_ms,
                self.max_lag_ms,
            )
        else:
            logger.debug(
                "mark_replica_lagging: %s lag=%dms (below threshold)",
                replica,
                lag_ms,
            )

    def mark_replica_healthy(self, replica: str) -> None:
        """
        Clear any lag record for `replica`, marking it healthy again.

        Args:
            replica: Replica node name (must be in self.replicas).

        Raises:
            ValueError: if `replica` is not in self.replicas.
        """
        if replica not in self.replicas:
            raise ValueError(
                f"mark_replica_healthy: {replica!r} not in replicas {self.replicas}"
            )
        removed = self._lag.pop(replica, None)
        if removed is not None:
            logger.info(
                "mark_replica_healthy: %s cleared (was lag=%dms)", replica, removed
            )

    def route_read(self) -> str:
        """
        Return the next node to handle a read request.

        Algorithm:
          1. Build healthy replica list: replicas where lag is not recorded
             OR recorded lag < max_lag_ms.
          2. If the healthy list is non-empty: round-robin across it.
          3. If empty: fall back to leader and emit a WARNING.

        Returns:
            Node name as a string.
        """
        healthy = [
            r for r in self.replicas
            if self._lag.get(r, 0) < self.max_lag_ms
        ]

        if not healthy:
            logger.warning(
                "route_read: no healthy replicas available — falling back to leader %s",
                self.leader,
            )
            return self.leader

        idx = self._rr_index % len(healthy)
        self._rr_index += 1
        selected = healthy[idx]
        logger.debug("route_read: → %s (rr_index=%d)", selected, self._rr_index)
        return selected

    def route_write(self) -> str:
        """
        Return the leader node for all writes.

        Returns:
            Leader node name.
        """
        return self.leader

    def status(self) -> dict[str, object]:
        """
        Return a dict summarising router state for observability.

        Returns:
            {
                "leader": str,
                "total_replicas": int,
                "healthy_replicas": int,
                "lagging_replicas": int,
            }
        """
        lagging = sum(
            1 for r in self.replicas
            if self._lag.get(r, 0) >= self.max_lag_ms
        )
        return {
            "leader": self.leader,
            "total_replicas": len(self.replicas),
            "healthy_replicas": len(self.replicas) - lagging,
            "lagging_replicas": lagging,
        }
```

---

## Part 3 — Complete pytest Test Suite

### `tests/unit/test_sharding_router.py`

```python
"""
Unit tests for ShardingRouter.

Run:  pytest tests/unit/test_sharding_router.py -v
"""
import pytest
from src.data_layer.sharding_router import ShardingRouter


# ---------------------------------------------------------------------------
# Empty ring / error handling
# ---------------------------------------------------------------------------

def test_get_shard_empty_ring_raises():
    router = ShardingRouter(virtual_nodes=10)
    with pytest.raises(RuntimeError, match="empty"):
        router.get_shard("any-key")


def test_remove_node_not_in_ring_raises():
    router = ShardingRouter(virtual_nodes=10)
    with pytest.raises(KeyError):
        router.remove_node("phantom")


# ---------------------------------------------------------------------------
# Single node
# ---------------------------------------------------------------------------

def test_single_node_all_keys_route_to_it():
    router = ShardingRouter(virtual_nodes=50)
    router.add_node("only-node")
    keys = [f"k:{i}" for i in range(500)]
    for key in keys:
        assert router.get_shard(key) == "only-node"


# ---------------------------------------------------------------------------
# Key determinism
# ---------------------------------------------------------------------------

def test_same_key_always_same_shard():
    router = ShardingRouter(virtual_nodes=100)
    for name in ("A", "B", "C"):
        router.add_node(name)
    key = "user:stable-key"
    first = router.get_shard(key)
    for _ in range(100):
        assert router.get_shard(key) == first


def test_hash_is_deterministic():
    assert ShardingRouter._hash("hello") == ShardingRouter._hash("hello")
    assert ShardingRouter._hash("foo") != ShardingRouter._hash("bar")


# ---------------------------------------------------------------------------
# Ring size invariants
# ---------------------------------------------------------------------------

def test_add_node_ring_size():
    router = ShardingRouter(virtual_nodes=50)
    router.add_node("A")
    assert len(router._sorted_keys) == 50
    router.add_node("B")
    assert len(router._sorted_keys) == 100


def test_remove_node_ring_size():
    router = ShardingRouter(virtual_nodes=50)
    router.add_node("A")
    router.add_node("B")
    router.remove_node("A")
    assert len(router._sorted_keys) == 50
    assert all(router._ring[p] == "B" for p in router._sorted_keys)


def test_add_node_idempotent():
    router = ShardingRouter(virtual_nodes=50)
    router.add_node("A")
    router.add_node("A")  # duplicate — should be ignored
    assert len(router._sorted_keys) == 50


# ---------------------------------------------------------------------------
# Distribution uniformity
# ---------------------------------------------------------------------------

def test_distribution_roughly_uniform_three_nodes():
    router = ShardingRouter(virtual_nodes=150)
    for name in ("A", "B", "C"):
        router.add_node(name)
    keys = [f"user:{i}" for i in range(3000)]
    dist = router.get_distribution(keys)
    assert set(dist.keys()) == {"A", "B", "C"}
    for count in dist.values():
        # Each node should get 1000 ± 10%
        assert 900 <= count <= 1100, f"Uneven distribution: {dist}"


def test_distribution_with_100_vnodes():
    """100 vnodes per node should still give ±15% uniformity."""
    router = ShardingRouter(virtual_nodes=100)
    for name in ("X", "Y", "Z"):
        router.add_node(name)
    keys = [f"item:{i}" for i in range(3000)]
    dist = router.get_distribution(keys)
    for count in dist.values():
        assert 750 <= count <= 1250, f"Distribution too skewed: {dist}"


# ---------------------------------------------------------------------------
# Rebalancing — minimal key movement
# ---------------------------------------------------------------------------

def test_add_node_moves_roughly_one_third_of_keys():
    router = ShardingRouter(virtual_nodes=150)
    router.add_node("A")
    router.add_node("B")

    keys = [f"key:{i}" for i in range(2000)]
    before = {k: router.get_shard(k) for k in keys}

    router.add_node("C")
    after = {k: router.get_shard(k) for k in keys}

    moved = sum(1 for k in keys if before[k] != after[k])
    fraction = moved / len(keys)
    # Expect ~1/3 to move, allow ±12%
    assert 0.21 <= fraction <= 0.45, f"Unexpected fraction moved: {fraction:.2%}"


def test_removed_node_keys_reroute_to_survivors():
    router = ShardingRouter(virtual_nodes=100)
    router.add_node("A")
    router.add_node("B")
    router.add_node("C")

    keys = [f"data:{i}" for i in range(1500)]
    before = {k: router.get_shard(k) for k in keys}

    router.remove_node("C")
    after = {k: router.get_shard(k) for k in keys}

    for k in keys:
        if before[k] == "C":
            # Keys that were on C must now be on A or B
            assert after[k] in {"A", "B"}
        else:
            # Keys on other nodes must NOT have moved
            assert after[k] == before[k]
```

---

### `tests/unit/test_replica_router.py`

```python
"""
Unit tests for ReplicaRouter.

Run:  pytest tests/unit/test_replica_router.py -v
"""
import logging

import pytest
from src.data_layer.replica_router import ReplicaRouter


def _make_router(**kwargs) -> ReplicaRouter:
    """Convenience factory with sensible defaults."""
    return ReplicaRouter(
        leader="leader",
        replicas=["replica-a", "replica-b"],
        max_lag_ms=kwargs.get("max_lag_ms", 200),
    )


# ---------------------------------------------------------------------------
# Writes always go to leader
# ---------------------------------------------------------------------------

def test_write_always_goes_to_leader():
    rr = _make_router()
    for _ in range(10):
        assert rr.route_write() == "leader"


def test_write_unaffected_by_lag():
    rr = _make_router()
    rr.mark_replica_lagging("replica-a", lag_ms=9999)
    rr.mark_replica_lagging("replica-b", lag_ms=9999)
    assert rr.route_write() == "leader"


# ---------------------------------------------------------------------------
# Read round-robin over healthy replicas
# ---------------------------------------------------------------------------

def test_reads_go_to_replicas_not_leader():
    rr = _make_router()
    targets = {rr.route_read() for _ in range(20)}
    assert "leader" not in targets
    assert targets == {"replica-a", "replica-b"}


def test_round_robin_alternates():
    rr = _make_router()
    results = [rr.route_read() for _ in range(4)]
    # With 2 healthy replicas: a, b, a, b (order depends on round-robin start)
    assert results.count("replica-a") == 2
    assert results.count("replica-b") == 2


# ---------------------------------------------------------------------------
# Lag filtering
# ---------------------------------------------------------------------------

def test_lagging_replica_is_skipped():
    rr = _make_router()
    rr.mark_replica_lagging("replica-a", lag_ms=350)
    for _ in range(10):
        assert rr.route_read() == "replica-b"


def test_replica_at_threshold_is_excluded():
    """Lag == max_lag_ms (not strictly less-than) should be excluded."""
    rr = _make_router(max_lag_ms=200)
    rr.mark_replica_lagging("replica-a", lag_ms=200)  # exactly at threshold
    for _ in range(10):
        assert rr.route_read() == "replica-b"


def test_replica_below_threshold_is_included():
    rr = _make_router(max_lag_ms=200)
    rr.mark_replica_lagging("replica-a", lag_ms=199)  # below threshold
    targets = {rr.route_read() for _ in range(20)}
    assert "replica-a" in targets


# ---------------------------------------------------------------------------
# All-lagging fallback to leader
# ---------------------------------------------------------------------------

def test_all_lagging_falls_back_to_leader(caplog):
    rr = _make_router()
    rr.mark_replica_lagging("replica-a", lag_ms=9999)
    rr.mark_replica_lagging("replica-b", lag_ms=9999)
    with caplog.at_level(logging.WARNING, logger="src.data_layer.replica_router"):
        result = rr.route_read()
    assert result == "leader"
    assert any("falling back to leader" in msg for msg in caplog.messages)


def test_all_lagging_every_call_returns_leader():
    rr = _make_router()
    rr.mark_replica_lagging("replica-a", lag_ms=500)
    rr.mark_replica_lagging("replica-b", lag_ms=500)
    for _ in range(5):
        assert rr.route_read() == "leader"


# ---------------------------------------------------------------------------
# mark_replica_healthy clears lag
# ---------------------------------------------------------------------------

def test_mark_healthy_restores_replica_to_pool():
    rr = _make_router()
    rr.mark_replica_lagging("replica-a", lag_ms=9999)
    # Only replica-b available
    for _ in range(6):
        assert rr.route_read() == "replica-b"

    # Restore replica-a
    rr.mark_replica_healthy("replica-a")
    targets = {rr.route_read() for _ in range(20)}
    assert "replica-a" in targets  # back in the pool


def test_mark_healthy_unknown_replica_raises():
    rr = _make_router()
    with pytest.raises(ValueError, match="not in replicas"):
        rr.mark_replica_healthy("phantom-replica")


# ---------------------------------------------------------------------------
# Invalid replica raises ValueError
# ---------------------------------------------------------------------------

def test_mark_lagging_unknown_replica_raises():
    rr = _make_router()
    with pytest.raises(ValueError, match="not in replicas"):
        rr.mark_replica_lagging("unknown", lag_ms=100)


# ---------------------------------------------------------------------------
# Status dict
# ---------------------------------------------------------------------------

def test_status_shape():
    rr = _make_router()
    s = rr.status()
    assert set(s.keys()) == {
        "leader",
        "total_replicas",
        "healthy_replicas",
        "lagging_replicas",
    }


def test_status_counts_all_healthy():
    rr = _make_router()
    s = rr.status()
    assert s["total_replicas"] == 2
    assert s["healthy_replicas"] == 2
    assert s["lagging_replicas"] == 0
    assert s["leader"] == "leader"


def test_status_counts_with_lagging():
    rr = _make_router()
    rr.mark_replica_lagging("replica-a", lag_ms=500)
    s = rr.status()
    assert s["healthy_replicas"] == 1
    assert s["lagging_replicas"] == 1


def test_status_all_lagging():
    rr = _make_router()
    rr.mark_replica_lagging("replica-a", lag_ms=500)
    rr.mark_replica_lagging("replica-b", lag_ms=500)
    s = rr.status()
    assert s["healthy_replicas"] == 0
    assert s["lagging_replicas"] == 2
```

---

### `tests/integration/test_data_layer.py`

```python
"""
Integration test: ShardingRouter + ReplicaRouter working together.

Run:  pytest tests/integration/test_data_layer.py -v
"""
import pytest
from src.data_layer.sharding_router import ShardingRouter
from src.data_layer.replica_router import ReplicaRouter


def _build_system(num_shards: int = 3, replicas_per_shard: int = 2):
    """
    Build a full ShardingRouter + per-shard ReplicaRouter topology.

    Returns:
        (shard_router, replica_routers) where replica_routers is a
        dict mapping shard_name → ReplicaRouter.
    """
    shard_router = ShardingRouter(virtual_nodes=150)
    shard_names = [f"shard-{i}" for i in range(num_shards)]
    for name in shard_names:
        shard_router.add_node(name)

    replica_routers: dict[str, ReplicaRouter] = {}
    for shard_name in shard_names:
        replicas = [
            f"{shard_name}-replica-{chr(ord('a') + j)}"
            for j in range(replicas_per_shard)
        ]
        replica_routers[shard_name] = ReplicaRouter(
            leader=f"{shard_name}-leader",
            replicas=replicas,
            max_lag_ms=200,
        )
    return shard_router, replica_routers


def test_all_writes_go_to_shard_leader():
    shard_router, replica_routers = _build_system()
    write_keys = [f"write:{i}" for i in range(100)]
    for key in write_keys:
        shard = shard_router.get_shard(key)
        target = replica_routers[shard].route_write()
        assert target == f"{shard}-leader"


def test_reads_distributed_across_replicas():
    shard_router, replica_routers = _build_system()
    read_keys = [f"read:{i}" for i in range(1000)]
    read_targets: set[str] = set()
    for key in read_keys:
        shard = shard_router.get_shard(key)
        target = replica_routers[shard].route_read()
        read_targets.add(target)
        # Must never be a leader for normal reads
        assert "leader" not in target or "replica" in target  # safety net

    # With 3 shards × 2 replicas = 6 replica targets — should see most of them
    assert len(read_targets) >= 4  # at minimum most replicas are exercised


def test_lagging_replica_excluded_for_its_shard():
    shard_router, replica_routers = _build_system(num_shards=3)

    # Determine which shard a well-known key maps to
    test_key = "lagging-test:42"
    affected_shard = shard_router.get_shard(test_key)
    rr = replica_routers[affected_shard]
    lagging_replica = rr.replicas[0]

    # Mark the first replica of the affected shard as lagging
    rr.mark_replica_lagging(lagging_replica, lag_ms=500)

    # All reads to the affected shard must skip the lagging replica
    for _ in range(50):
        shard = shard_router.get_shard(test_key)
        assert shard == affected_shard  # same shard (deterministic)
        target = rr.route_read()
        assert target != lagging_replica, (
            f"Expected lagging replica {lagging_replica!r} to be skipped, "
            f"but got {target!r}"
        )

    # Other shards are unaffected
    for shard_name, other_rr in replica_routers.items():
        if shard_name == affected_shard:
            continue
        targets = {other_rr.route_read() for _ in range(10)}
        assert "leader" not in targets  # other shards still use replicas


def test_writes_still_reach_leader_when_replicas_lagging():
    shard_router, replica_routers = _build_system(num_shards=3)
    target_shard = "shard-0"
    rr = replica_routers[target_shard]
    for replica in rr.replicas:
        rr.mark_replica_lagging(replica, lag_ms=9999)

    # Writes must always go to the leader regardless of replica health
    for _ in range(20):
        assert rr.route_write() == f"{target_shard}-leader"

    # Reads fall back to leader when all replicas are lagging
    for _ in range(20):
        assert rr.route_read() == f"{target_shard}-leader"


def test_distribution_across_three_shards():
    shard_router, _ = _build_system(num_shards=3)
    keys = [f"dist:{i}" for i in range(3000)]
    dist = shard_router.get_distribution(keys)
    assert set(dist.keys()) == {"shard-0", "shard-1", "shard-2"}
    for count in dist.values():
        assert 900 <= count <= 1100, f"Uneven shard distribution: {dist}"
```

---

## Key Design Notes

**`ShardingRouter._hash` uses MD5** — faster than SHA-256 for non-security hashing. The `usedforsecurity=False` flag suppresses Python 3.9+ deprecation warnings on FIPS systems. Both SHA-256 (as in the Day 4 lesson) and MD5 are valid; what matters is determinism across runs and processes.

**`ReplicaRouter._rr_index`** is a simple integer monotonically incremented. It is not thread-safe. For concurrent use, replace with an `itertools.cycle` protected by a `threading.Lock`, or use an atomic counter.

**Lag threshold is strict less-than** (`< max_lag_ms`). A replica reporting lag exactly equal to the threshold is excluded. This is a deliberate conservative choice — at the boundary, behaviour is ambiguous and the safe default is to exclude. Document this in your API if you change it.

**Empty ring raises `RuntimeError`**, not returns `None`. Callers should not need to handle `None` from a routing function; it would force every call site to check. A raised exception surfaces the misconfiguration immediately.
