# Phase 6 · Week 01 · Day 6 — Build: Distributed KV Store Simulation (4 hr)

> Objective: Implement a complete in-process simulation of a distributed key-value store with replication, eventual consistency, and partition handling — and write tests that demonstrate the tradeoffs.  ·  Time: ~4 hr  ·  Prereqs: [[day-1]] through [[day-5]]

## What you're building

A Python simulation of a 3-node distributed key-value store. It runs entirely in memory (no sockets), using direct method calls to simulate network communication. A `NetworkSim` layer intercepts calls and can inject:

- Message delays
- Packet drops (simulating unreliable network)
- Full partitions between node pairs

The store implements:
- Single-leader replication with async propagation
- Eventual consistency with last-write-wins conflict resolution
- Read-your-own-writes on the primary
- Idempotent writes via version timestamps
- Basic partition detection and leader fencing

Tests verify each trade-off explicitly.

## Project structure

```
week-01-kvstore/
├── src/
│   ├── __init__.py
│   ├── models.py          # Pydantic types: Entry, WriteRequest, ReadResult
│   ├── network_sim.py     # NetworkSim: delay/drop/partition injection
│   ├── node.py            # KVNode: in-memory store + replication logic
│   └── cluster.py         # Cluster: 3-node coordinator
├── tests/
│   ├── __init__.py
│   ├── test_replication.py
│   ├── test_consistency.py
│   └── test_partition.py
└── pyproject.toml
```

## Guided implementation

### Step 1 — Models (30 min)

Define the data types. Every value crossing a module boundary is a Pydantic model.

```python
# src/models.py
from __future__ import annotations

from pydantic import BaseModel, Field
import time


class Entry(BaseModel):
    """A single key-value entry with its replication metadata."""
    key: str
    value: str
    version: float = Field(default_factory=time.monotonic)  # LWW timestamp
    written_by: str  # node_id of the writer


class WriteRequest(BaseModel):
    key: str
    value: str
    idempotency_key: str  # client-generated UUID per logical write


class ReadResult(BaseModel):
    key: str
    value: str | None
    version: float | None
    from_node: str
    is_stale: bool  # True if this replica may not have the latest write


class ReplicationMessage(BaseModel):
    entry: Entry
    source_node: str
```

### Step 2 — Network simulator (45 min)

```python
# src/network_sim.py
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class NetworkSim:
    """
    In-process network simulator.
    Intercepts replication calls and injects delay, drops, and partitions.
    """

    drop_rate: float = 0.0          # probability a message is dropped
    delay_ms: float = 0.0           # added latency (simulated via counter, not real sleep)
    partitioned_pairs: set[frozenset[str]] = field(default_factory=set)
    _drop_count: int = 0
    _deliver_count: int = 0

    def partition(self, node_a: str, node_b: str) -> None:
        self.partitioned_pairs.add(frozenset({node_a, node_b}))
        log.warning("PARTITION: %s ↔ %s  SEVERED", node_a, node_b)

    def heal(self, node_a: str, node_b: str) -> None:
        self.partitioned_pairs.discard(frozenset({node_a, node_b}))
        log.info("HEAL: %s ↔ %s  RESTORED", node_a, node_b)

    def heal_all(self) -> None:
        self.partitioned_pairs.clear()
        log.info("HEAL ALL: all partitions resolved")

    def should_deliver(self, source: str, destination: str) -> bool:
        if frozenset({source, destination}) in self.partitioned_pairs:
            log.debug("DROP (partition): %s → %s", source, destination)
            self._drop_count += 1
            return False
        if random.random() < self.drop_rate:
            log.debug("DROP (random): %s → %s", source, destination)
            self._drop_count += 1
            return False
        self._deliver_count += 1
        return True

    @property
    def stats(self) -> dict[str, int]:
        return {"delivered": self._deliver_count, "dropped": self._drop_count}
```

### Step 3 — KV Node (60 min)

```python
# src/node.py
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from .models import Entry, ReadResult, ReplicationMessage, WriteRequest


@dataclass
class KVNode:
    """Single node in the distributed KV store."""

    node_id: str
    _store: dict[str, Entry] = field(default_factory=dict)
    _processed_idempotency_keys: set[str] = field(default_factory=set)
    _log: logging.Logger = field(init=False)

    def __post_init__(self) -> None:
        self._log = logging.getLogger(f"node.{self.node_id}")

    # ── Writes ──────────────────────────────────────────────────────────────

    def write(self, request: WriteRequest) -> Entry:
        """
        Accept a write. Idempotent: duplicate idempotency_key returns the
        previously stored entry without overwriting.
        """
        if request.idempotency_key in self._processed_idempotency_keys:
            existing = self._store.get(request.key)
            self._log.debug(
                "IDEMPOTENT skip  key=%s ikey=%s", request.key, request.idempotency_key[:8]
            )
            if existing:
                return existing
            # Key was deleted between writes — treat as new
        self._processed_idempotency_keys.add(request.idempotency_key)
        entry = Entry(key=request.key, value=request.value, written_by=self.node_id)
        self._store[request.key] = entry
        self._log.info("WRITE  key=%s value=%s version=%.6f", entry.key, entry.value, entry.version)
        return entry

    def apply_replication(self, msg: ReplicationMessage) -> None:
        """
        Accept a replicated entry. Last-write-wins: only apply if the
        incoming version is newer than what we have.
        """
        existing = self._store.get(msg.entry.key)
        if existing and existing.version >= msg.entry.version:
            self._log.debug(
                "LWW skip  key=%s  local_v=%.6f  remote_v=%.6f",
                msg.entry.key, existing.version, msg.entry.version,
            )
            return
        self._store[msg.entry.key] = msg.entry
        self._log.debug(
            "REPLICATED  key=%s  from=%s  version=%.6f",
            msg.entry.key, msg.source_node, msg.entry.version,
        )

    # ── Reads ───────────────────────────────────────────────────────────────

    def read(self, key: str) -> ReadResult:
        entry = self._store.get(key)
        return ReadResult(
            key=key,
            value=entry.value if entry else None,
            version=entry.version if entry else None,
            from_node=self.node_id,
            is_stale=False,  # caller marks staleness based on context
        )

    # ── Inspection ──────────────────────────────────────────────────────────

    def all_keys(self) -> dict[str, Entry]:
        return dict(self._store)
```

### Step 4 — Cluster (45 min)

```python
# src/cluster.py
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

from .models import Entry, ReadResult, ReplicationMessage, WriteRequest
from .network_sim import NetworkSim
from .node import KVNode

log = logging.getLogger(__name__)


@dataclass
class Cluster:
    """
    Three-node cluster with single-leader async replication.
    network: NetworkSim controls what gets delivered to replicas.
    """

    network: NetworkSim = field(default_factory=NetworkSim)
    _primary: KVNode = field(init=False)
    _replicas: list[KVNode] = field(init=False)

    def __post_init__(self) -> None:
        self._primary = KVNode("primary")
        self._replicas = [KVNode("replica-1"), KVNode("replica-2")]

    # ── Public API ───────────────────────────────────────────────────────────

    def write(self, key: str, value: str, idempotency_key: str | None = None) -> Entry:
        """Write to primary, then attempt async replication to each replica."""
        ikey = idempotency_key or str(uuid.uuid4())
        req = WriteRequest(key=key, value=value, idempotency_key=ikey)
        entry = self._primary.write(req)
        self._replicate(entry)
        return entry

    def read(self, key: str, node: str = "primary") -> ReadResult:
        """Read from a named node. Non-primary reads may be stale."""
        target = self._get_node(node)
        result = target.read(key)
        result = result.model_copy(update={"is_stale": node != "primary"})
        return result

    def quorum_read(self, key: str) -> ReadResult:
        """
        Read from all three nodes and return the entry with the highest version.
        Guarantees reading the latest committed write.
        """
        all_nodes = [self._primary] + self._replicas
        results = [n.read(key) for n in all_nodes]
        best = max(
            (r for r in results if r.value is not None),
            key=lambda r: r.version or 0.0,
            default=results[0],
        )
        return best.model_copy(update={"is_stale": False})

    def sync_replicas(self) -> None:
        """
        Force-sync all primary entries to replicas (simulates partition heal + catch-up).
        Uses LWW — replicas accept any entry newer than what they have.
        """
        for key, entry in self._primary.all_keys().items():
            msg = ReplicationMessage(entry=entry, source_node="primary")
            for replica in self._replicas:
                replica.apply_replication(msg)
        log.info("Replicas synced from primary")

    # ── Internal ─────────────────────────────────────────────────────────────

    def _replicate(self, entry: Entry) -> None:
        msg = ReplicationMessage(entry=entry, source_node="primary")
        for replica in self._replicas:
            if self.network.should_deliver("primary", replica.node_id):
                replica.apply_replication(msg)

    def _get_node(self, name: str) -> KVNode:
        if name == "primary":
            return self._primary
        for r in self._replicas:
            if r.node_id == name:
                return r
        raise ValueError(f"Unknown node: {name}")

    def node_ids(self) -> list[str]:
        return ["primary"] + [r.node_id for r in self._replicas]
```

### Step 5 — Tests (60 min)

Write tests that demonstrate the trade-offs — not just "it works", but "here is what breaks under these conditions."

```python
# tests/test_replication.py
from __future__ import annotations

import uuid
import pytest
from src.cluster import Cluster
from src.network_sim import NetworkSim


def test_write_reaches_primary() -> None:
    cluster = Cluster()
    entry = cluster.write("model_version", "v3")
    result = cluster.read("model_version", node="primary")
    assert result.value == "v3"
    assert result.version == entry.version


def test_successful_replication_to_both_replicas() -> None:
    cluster = Cluster(network=NetworkSim(drop_rate=0.0))
    cluster.write("feature_store_version", "2024-11")
    for node in ("replica-1", "replica-2"):
        r = cluster.read("feature_store_version", node=node)
        assert r.value == "2024-11", f"Replica {node} should have the value"


def test_idempotent_write_does_not_duplicate() -> None:
    cluster = Cluster()
    ikey = str(uuid.uuid4())
    entry1 = cluster.write("session_state", "step-1", idempotency_key=ikey)
    entry2 = cluster.write("session_state", "step-2", idempotency_key=ikey)
    # Same idempotency key → second write is a no-op
    assert entry1.version == entry2.version
    assert cluster.read("session_state").value == "step-1"


def test_replica_may_miss_write_when_dropped() -> None:
    """Demonstrates that high drop rates cause replicas to miss writes."""
    cluster = Cluster(network=NetworkSim(drop_rate=1.0))  # all replication dropped
    cluster.write("index_version", "v10")
    # Primary has it
    assert cluster.read("index_version", node="primary").value == "v10"
    # Replicas do not
    for node in ("replica-1", "replica-2"):
        r = cluster.read("index_version", node=node)
        assert r.value is None, f"Replica {node} should be stale under 100% drop rate"
```

```python
# tests/test_consistency.py
from __future__ import annotations

from src.cluster import Cluster
from src.network_sim import NetworkSim


def test_stale_read_eventually_consistent() -> None:
    """After healing, replicas converge to primary's state."""
    cluster = Cluster(network=NetworkSim(drop_rate=1.0))
    cluster.write("embedding_model", "text-embed-3-large")

    # Stale
    r = cluster.read("embedding_model", node="replica-1")
    assert r.value is None

    # Heal + sync
    cluster.sync_replicas()
    r = cluster.read("embedding_model", node="replica-1")
    assert r.value == "text-embed-3-large"


def test_quorum_read_returns_latest() -> None:
    """Quorum read returns latest value even if some replicas are stale."""
    cluster = Cluster(network=NetworkSim(drop_rate=0.5))
    # Write multiple times so at least one replica likely has the value
    for _ in range(10):
        cluster.write("rag_index_version", "2024-Q4")

    result = cluster.quorum_read("rag_index_version")
    # At least one node (primary) always has it, so quorum read should find it
    assert result.value == "2024-Q4"


def test_lww_conflict_resolution() -> None:
    """Last-write-wins: higher-version entry wins on replica."""
    from src.models import Entry, ReplicationMessage
    from src.node import KVNode
    import time

    node = KVNode("replica-test")
    old_entry = Entry(key="k", value="old", version=1.0, written_by="primary")
    new_entry = Entry(key="k", value="new", version=2.0, written_by="primary")

    node.apply_replication(ReplicationMessage(entry=new_entry, source_node="primary"))
    node.apply_replication(ReplicationMessage(entry=old_entry, source_node="primary"))

    result = node.read("k")
    assert result.value == "new"  # old entry must not overwrite new entry
```

```python
# tests/test_partition.py
from __future__ import annotations

from src.cluster import Cluster
from src.network_sim import NetworkSim


def test_primary_still_accepts_writes_during_partition() -> None:
    """Under partition, primary remains available for writes (AP choice)."""
    net = NetworkSim()
    cluster = Cluster(network=net)
    net.partition("primary", "replica-1")
    net.partition("primary", "replica-2")

    # Write succeeds — primary is available
    entry = cluster.write("config_flag", "enabled")
    assert entry.value == "enabled"

    # Replicas are stale
    for node in ("replica-1", "replica-2"):
        r = cluster.read("config_flag", node=node)
        assert r.value is None


def test_replica_convergence_after_partition_heal() -> None:
    net = NetworkSim()
    cluster = Cluster(network=net)

    # Write during partition
    net.partition("primary", "replica-1")
    net.partition("primary", "replica-2")
    cluster.write("model_checkpoint", "checkpoint-42")

    # Heal and sync
    net.heal_all()
    cluster.sync_replicas()

    for node in cluster.node_ids():
        r = cluster.read("model_checkpoint", node=node)
        assert r.value == "checkpoint-42", f"Node {node} did not converge"


def test_network_stats_track_drops() -> None:
    net = NetworkSim(drop_rate=1.0)
    cluster = Cluster(network=net)
    cluster.write("key-a", "val-a")
    cluster.write("key-b", "val-b")
    stats = net.stats
    # 2 writes × 2 replicas = 4 replication messages, all dropped
    assert stats["dropped"] == 4
    assert stats["delivered"] == 0
```

### Step 6 — pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "week01-kvstore"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["pydantic>=2.0"]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-cov"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
target-version = "py312"
```

## Build checklist

- [ ] `python -m pytest tests/ -v` — all tests pass
- [ ] All four consistency trade-offs demonstrated by a failing test that is then fixed by a design choice (partition, eventually consistent, idempotent, quorum)
- [ ] `NetworkSim.stats` shows correct drop/deliver counts
- [ ] No `print()` statements — only `logging`
- [ ] All module boundaries use Pydantic models

## Extension challenges (if time allows)

1. Add a **Lamport-timestamp** version to `Entry` alongside `time.monotonic()` and compare their ordering under simulated clock skew (set `version` to a random float occasionally).
2. Implement **read-your-own-writes**: after a write returns, subsequent reads from any node must return that version or newer (within 1 s). Hint: pass the write's version to the read and have non-primary nodes re-request from primary if their local version is older.
3. Add a `leader_election()` method to `Cluster` that marks a replica as the new primary when the current primary is partitioned from the other two nodes. Demonstrate with a test.
