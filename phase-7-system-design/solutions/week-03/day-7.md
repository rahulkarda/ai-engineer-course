# Solutions — Phase 7 · Week 3 · Day 7 — Week Review + Integration

---

## Exercise 1 — Data-Layer Design Document: Read-Heavy E-Commerce Catalog Service

```markdown
# Data-Layer Design — E-Commerce Catalog Service

## 1. Requirements Summary

- Write throughput: ~500 writes/s (product updates, inventory adjustments, price changes)
- Read throughput: ~50 000 reads/s peak (catalog browsing, search, cart operations)
- Read:write ratio: ~100:1 — strongly read-heavy
- Latency targets: p99 read < 80 ms, p99 write < 200 ms
- Consistency: inventory writes strong (overselling is a business error); catalog reads
  eventually consistent (500 ms acceptable); cart/session reads must be own-write consistent
- Data retention: product records indefinite; session data 30-day TTL; analytics 24 months

## 2. Datastore Selection

| Use case              | Store           | Rationale                                                         |
|-----------------------|-----------------|-------------------------------------------------------------------|
| Product catalog       | PostgreSQL      | ACID writes for price/inventory; rich query support; FK integrity |
| Inventory             | PostgreSQL      | Row-level locking for decrement-on-purchase; transactional        |
| Session / cart        | Redis           | Sub-millisecond reads; natural TTL eviction; Hash data structure  |
| Full-text search      | Elasticsearch   | Inverted index for keyword/facet search; relevance scoring        |
| Analytics / reporting | ClickHouse      | Column store; aggregations over 100M+ rows in seconds             |

Trade-off (PostgreSQL): requires careful connection pooling (pgBouncer) and read replica
offloading at this QPS. JSONB columns for flexible product attributes add index complexity.
We accept this over a document store because inventory updates require serialisable transactions.

## 3. Partitioning Strategy

### Product catalog (PostgreSQL)
- **Partition key**: `product_id` (UUID) — high cardinality, always present in reads
- **Method**: hash partitioning (PostgreSQL declarative partitioning, 16 partitions)
- **Rationale**: avoids range hotspots; catalog reads always supply `product_id`
- **Reporting partition**: range by `category_id` on a separate read-replica schema
  (category-level aggregations need range scans — kept off the OLTP path)

### Elasticsearch index
- **Shard key**: `_routing=product_id` default
- **Shards**: 5 primary (supports ~50 GB index with room for 3× growth)
- **Replicas**: 1 per primary per region (2 AZ deployment)

### Redis
- **Partitioning**: Redis Cluster, 16384 hash slots, consistent hash by `session_id` / `cart_id`
- No explicit user sharding needed — Redis Cluster handles it transparently

Key routing example:
`product_id = "prod-00123456"` → MD5 → `0x8FA2...` → partition slot 9 of 16

## 4. Replication Approach

**PostgreSQL (primary datastore)**
- 1 leader per region, 2 synchronous read replicas in the same region
- 1 asynchronous replica in a secondary region (disaster recovery)
- Mode: semi-synchronous — leader waits for 1 replica before confirming write
- Failover: Patroni + HAProxy; RTO < 30 s; RPO = 0 within region
- Read replicas serve all catalog reads; replication lag < 200 ms acceptable for catalog,
  NOT acceptable for inventory (inventory reads go to leader via read-your-writes routing)

**Elasticsearch**
- RF = 2 (1 primary + 1 replica per shard)
- Async replication; near-realtime (1 s refresh interval)
- Indexed asynchronously from PostgreSQL via Debezium CDC pipeline
- Staleness: up to ~2 s acceptable for search results

**Redis**
- Redis Cluster with 3 masters, 1 replica each
- Async replication (Redis default); AOF + RDB persistence for durability
- Sentinel for automatic failover; RTO < 10 s

## 5. CQRS Read Model for Product Listings

The write model is the normalised PostgreSQL schema:
`products(product_id, name, price, category_id, ...)` + `inventory(product_id, qty, ...)`

The CQRS read model is a denormalised `product_listings` view maintained as a separate table
(or materialised in Elasticsearch) populated via event projection:

```sql
-- Read model schema (PostgreSQL materialised or Elasticsearch document)
CREATE TABLE product_listings (
    product_id      UUID PRIMARY KEY,
    name            TEXT NOT NULL,
    price           NUMERIC(10,2) NOT NULL,
    category_path   TEXT[],           -- denormalised hierarchy
    image_url       TEXT,
    in_stock        BOOLEAN NOT NULL,
    stock_qty       INT NOT NULL,
    avg_rating      NUMERIC(3,2),
    review_count    INT DEFAULT 0,
    updated_at      TIMESTAMPTZ NOT NULL
);

CREATE INDEX idx_listings_category ON product_listings USING BTREE (category_path, price);
CREATE INDEX idx_listings_instock  ON product_listings (in_stock, updated_at DESC)
    WHERE in_stock = TRUE;
```

Events that trigger projection updates:
- `ProductUpdated` → update name, price, image_url
- `InventoryChanged` → update in_stock, stock_qty
- `ReviewPosted` → recompute avg_rating, review_count (batched, not per-event)

Sequence diagram:

```
Client          Write API       PostgreSQL      CDC (Debezium)    product_listings
  |                |                |                 |                  |
  |-- POST /update->|               |                 |                  |
  |                |-- UPDATE ------>|                 |                  |
  |                |<-- ack ---------|                 |                  |
  |<-- 200 --------|                |                 |                  |
  |                |                |-- WAL event ---->|                  |
  |                |                |                 |-- upsert -------->|
  |                |                |                 |                  |
  |-- GET /catalog->|               |                 |                  |
  |                |                |                 |  <-- SELECT ------|
  |<-- 200 product--|               |                 |                  |
```

## 6. Open Questions / Risks

- [ ] Inventory consistency during flash sales: do we accept optimistic locking
  (check-and-set in application) or pessimistic row locks? Decision owner: Engineering + Product.
- [ ] Elasticsearch vs PostgreSQL full-text: ES adds operational complexity — is FTS
  via `pg_trgm` + `tsvector` sufficient at this scale? Decision owner: Platform team.
- [ ] Cross-region writes for inventory: two-phase commit across regions is expensive —
  do we restrict inventory writes to primary region only? Decision owner: Architecture board.
```

---

## Exercise 2 — DataLayerClient Facade (Complete Implementation)

```python
"""Facade that routes reads and writes across a sharded, replicated cluster.

src/data_layer/data_layer_client.py
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from data_layer.sharding_router import ShardingRouter
from data_layer.replica_router import ReplicaRouter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base — lets tests swap in a fake backend
# ---------------------------------------------------------------------------

class BaseDataLayerClient(ABC):
    """Abstract interface for a sharded, replicated data-layer client."""

    @abstractmethod
    def get_write_node(self, key: str) -> str:
        """Return the leader node responsible for writing `key`."""

    @abstractmethod
    def get_read_node(self, key: str) -> str:
        """Return the best read node (replica or leader fallback) for `key`."""

    @abstractmethod
    def mark_replica_lagging(self, shard: str, replica: str, lag_ms: int) -> None:
        """Record a lag observation for a replica in a given shard."""

    @abstractmethod
    def cluster_status(self) -> dict[str, object]:
        """Return per-shard health summary."""


# ---------------------------------------------------------------------------
# Production implementation
# ---------------------------------------------------------------------------

@dataclass
class DataLayerClient(BaseDataLayerClient):
    """Unified client for routing reads and writes in a sharded cluster.

    Composes a ShardingRouter (consistent hashing) with one ReplicaRouter per
    shard (lag-aware round-robin). All routing decisions are in-process — no
    network calls are made here.

    Args:
        shard_names: Ordered list of shard identifiers.
        replicas_per_shard: Number of read replicas per shard.
        virtual_nodes: Virtual nodes for the consistent hash ring.
        max_lag_ms: Max acceptable replica lag for reads (ms).
    """

    shard_names: list[str]
    replicas_per_shard: int = 2
    virtual_nodes: int = 150
    max_lag_ms: int = 200

    _shard_router: ShardingRouter = field(init=False, repr=False)
    _replica_routers: dict[str, ReplicaRouter] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not self.shard_names:
            raise ValueError("shard_names must contain at least one shard")
        self._shard_router = ShardingRouter(virtual_nodes=self.virtual_nodes)
        for shard in self.shard_names:
            self._shard_router.add_node(shard)
            leader = f"{shard}-leader"
            replicas = [
                f"{shard}-replica-{i}" for i in range(self.replicas_per_shard)
            ]
            self._replica_routers[shard] = ReplicaRouter(
                leader=leader,
                replicas=replicas,
                max_lag_ms=self.max_lag_ms,
            )
        logger.info(
            "DataLayerClient initialised: %d shards, %d replicas/shard",
            len(self.shard_names),
            self.replicas_per_shard,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_write_node(self, key: str) -> str:
        """Return the leader node responsible for writing `key`.

        Resolves: key → shard (consistent hash) → leader node name.
        """
        shard = self._shard_router.get_shard(key)
        node = self._replica_routers[shard].route_write()
        logger.debug("WRITE key=%s → shard=%s → node=%s", key, shard, node)
        return node

    def get_read_node(self, key: str) -> str:
        """Return the best replica (or leader fallback) for reading `key`.

        Resolves: key → shard (consistent hash) → healthiest replica.
        """
        shard = self._shard_router.get_shard(key)
        node = self._replica_routers[shard].route_read()
        logger.debug("READ  key=%s → shard=%s → node=%s", key, shard, node)
        return node

    def mark_replica_lagging(self, shard: str, replica: str, lag_ms: int) -> None:
        """Propagate a lag observation to the appropriate shard's ReplicaRouter.

        Raises KeyError if `shard` is not in the cluster.
        """
        if shard not in self._replica_routers:
            raise KeyError(f"Unknown shard: {shard!r}")
        self._replica_routers[shard].mark_replica_lagging(replica, lag_ms)

    def mark_replica_healthy(self, shard: str, replica: str) -> None:
        """Clear the lag record for `replica` in `shard`."""
        if shard not in self._replica_routers:
            raise KeyError(f"Unknown shard: {shard!r}")
        self._replica_routers[shard].mark_replica_healthy(replica)

    def cluster_status(self) -> dict[str, object]:
        """Return a per-shard health snapshot.

        Returns:
            dict mapping shard name → ReplicaRouter.status() dict.
        """
        return {
            shard: rr.status()
            for shard, rr in self._replica_routers.items()
        }

    def get_distribution(self, keys: list[str]) -> dict[str, int]:
        """Return shard key-count distribution for observability / load testing."""
        return self._shard_router.get_distribution(keys)
```

**Pytest tests for DataLayerClient:**

```python
"""tests/integration/test_data_layer_client.py"""

from __future__ import annotations

import pytest
from data_layer.data_layer_client import DataLayerClient


@pytest.fixture()
def client() -> DataLayerClient:
    return DataLayerClient(
        shard_names=["shard-0", "shard-1", "shard-2"],
        replicas_per_shard=2,
        max_lag_ms=200,
    )


def test_write_node_is_always_leader(client: DataLayerClient) -> None:
    for i in range(50):
        node = client.get_write_node(f"key:{i}")
        assert node.endswith("-leader"), f"Expected leader, got {node}"


def test_read_node_is_replica_when_healthy(client: DataLayerClient) -> None:
    for i in range(50):
        node = client.get_read_node(f"key:{i}")
        assert "-replica-" in node, f"Expected replica under healthy conditions, got {node}"


def test_read_falls_back_to_leader_when_all_replicas_lagging(
    client: DataLayerClient,
) -> None:
    # Mark all replicas in shard-0 as lagging above threshold
    client.mark_replica_lagging("shard-0", "shard-0-replica-0", lag_ms=500)
    client.mark_replica_lagging("shard-0", "shard-0-replica-1", lag_ms=500)

    # Route many keys — some will land on shard-0
    fallback_to_leader = False
    for i in range(200):
        key = f"user:{i}"
        shard = client._shard_router.get_shard(key)
        if shard == "shard-0":
            node = client.get_read_node(key)
            assert node == "shard-0-leader", f"Expected leader fallback, got {node}"
            fallback_to_leader = True
    assert fallback_to_leader, "No keys routed to shard-0 — increase key range"


def test_cluster_status_keys(client: DataLayerClient) -> None:
    status = client.cluster_status()
    assert set(status.keys()) == {"shard-0", "shard-1", "shard-2"}
    for shard_status in status.values():
        assert "leader" in shard_status
        assert "healthy_replicas" in shard_status


def test_key_routing_is_deterministic(client: DataLayerClient) -> None:
    key = "user:42"
    nodes = {client.get_write_node(key) for _ in range(10)}
    assert len(nodes) == 1, "Write node should be deterministic for a given key"


def test_mark_replica_healthy_restores_routing(client: DataLayerClient) -> None:
    client.mark_replica_lagging("shard-0", "shard-0-replica-0", lag_ms=999)
    client.mark_replica_lagging("shard-0", "shard-0-replica-1", lag_ms=999)
    client.mark_replica_healthy("shard-0", "shard-0-replica-0")

    # After restoring one replica, reads on shard-0 should NOT fall back to leader
    for i in range(200):
        key = f"user:{i}"
        shard = client._shard_router.get_shard(key)
        if shard == "shard-0":
            node = client.get_read_node(key)
            assert node != "shard-0-leader", "Should route to restored replica, not leader"
            break
```

---

## Exercise 3 — How Consistent Hashing + Replica Routing Compose

In a real production system these two mechanisms operate at different layers but are deeply coupled:

**Consistent hashing** answers: *which physical shard owns this data?*

- The hash ring determines data placement at write time and lookup time.
- Adding a node moves only `~1/N` of keys, not all keys — this is the critical property that makes live resharding feasible without a full cluster reindex.
- Virtual nodes smooth load distribution so no single physical node owns a disproportionate arc.

**Replica routing** answers: *which copy of that data should answer this read?*

- Each shard independently maintains a leader and N replicas.
- The `ReplicaRouter` encapsulates the health model: replicas are healthy until proven lagging.
- Round-robin distributes read QPS across replicas; the lag filter prevents stale reads beyond a business-defined threshold.

**Composition at system level:**

```
Read request for key K
  ↓
ShardingRouter.get_shard(K)          # Layer 1: Where is the data?
  → shard-2
  ↓
ReplicaRouter("shard-2").route_read() # Layer 2: Which copy?
  → shard-2-replica-1                 # (healthy, low lag)
  ↓
Network call to shard-2-replica-1
```

The layers are independent — you can swap the sharding algorithm (e.g., range instead of hash) without touching replica routing, and you can change the replica health model without touching shard assignment. This clean separation of concerns is what makes each piece testable in isolation.

The critical coupling point: when a shard is removed (node failure), the `ShardingRouter` reassigns its keys to the next clockwise shard. If the receiving shard's replicas are already lagging, the combined system degrades gracefully — reads fall back to that shard's leader rather than returning stale data or erroring. No cross-layer coordination is needed because the fallback is local to `ReplicaRouter`.

---

## Exercise 4 — Week Synthesis: Highest-Blast-Radius Decision

**Sharding strategy (hash vs range, and partition key choice) has the highest blast radius.**

Reasoning: every other decision is partially reversible at the cost of operational pain:

| Decision       | Can you change it live? | Migration cost |
|----------------|-------------------------|----------------|
| SQL vs NoSQL   | Hard, but bounded — you can migrate one table at a time with dual-write | Weeks to months |
| Indexing       | Yes — `CREATE INDEX CONCURRENTLY` in Postgres | Hours to days |
| Replication topology | Yes — add/remove replicas online | Hours |
| Sharding strategy | **No** — resharding requires full data migration | Months + downtime risk |

**Why sharding is the worst:**

1. **Partition key is permanent in practice.** If you shard posts by `created_at` (range) and later discover you have timestamp hotspots, re-partitioning 100 TB across 16 shards requires coordinated dual-writes, a multi-week data migration, and a cutover with rollback risk. You cannot do it incrementally in most distributed stores.

2. **Wrong key causes cascading failure.** A monotonically increasing key (e.g., auto-increment ID, timestamp) concentrates all writes on the last shard. As that shard fills, you can add more shards — but the hot shard problem is structural, not fixable by tuning.

3. **The error is invisible until scale.** A bad partition key works fine at low load; the hotspot only emerges at 10× scale, at which point you have production data in the wrong layout.

4. **Cross-shard transactions proliferate.** If the partition key doesn't match your most common query pattern, you end up scatter-gathering across all shards for simple queries — killing p99 latency and adding complexity that cannot be easily removed.

**Mitigation:** spend disproportionate time on partition key selection. The rule: partition key must be (a) present in every high-frequency query, (b) high cardinality, (c) not monotonically increasing. `user_id` for user-centric data is almost always correct; `created_at` alone is almost always wrong.
