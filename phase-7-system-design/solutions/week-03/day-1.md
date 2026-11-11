# Solutions — Phase 7 · Week 3 · Day 1 — SQL vs NoSQL Selection

## Exercise 1 — Classify 5 Workloads to DB Type

| Workload | DB Type | Justification |
|---|---|---|
| E-commerce cart | Key-Value (Redis) | Cart is a single-user blob keyed by session ID. O(1) get/set. No joins needed. Sub-ms latency is critical for checkout conversion. Expiry TTL maps naturally to session lifetime. |
| Social graph (friend-of-friend) | Graph DB (Neo4j) | Primary query is multi-hop traversal: "friends of friends who like X." Each hop follows a pointer — index-free adjacency. A relational JOIN chain explodes exponentially at 3+ hops. |
| IoT / clickstream analytics | Wide-Column (Cassandra) | Append-only writes at high volume. Partition key = device_id, clustering key = timestamp gives fast range scans per device. LSM tree handles write-heavy load without B-tree page splits. |
| CMS with variable content fields | Document Store (MongoDB) | Each content type (blog post, product page, FAQ) has a different field set. Document model stores each as a self-contained JSON blob. Schema evolution (adding fields) is cheap — no ALTER TABLE. |
| Banking / financial ledger | Relational SQL (PostgreSQL) | Transfers require multi-table ACID transactions: debit account A, credit account B atomically. Partial failure must roll back entirely. Regulatory audits require complex join-based queries. |

---

## Exercise 2 — Extend WorkloadProfile with `has_graph_queries` + Update `select_database()`

```python
"""
db_selector_v2.py — Extended DB selector with graph query support and
                    consistency-aware wide-column recommendations.

Usage:
    python db_selector_v2.py

Requires: Python 3.12+
"""

import logging
from dataclasses import dataclass
from enum import StrEnum

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


class DBType(StrEnum):
    RELATIONAL = "Relational SQL (PostgreSQL / MySQL)"
    DOCUMENT = "Document Store (MongoDB / Firestore)"
    KEY_VALUE = "Key-Value Store (Redis / DynamoDB)"
    WIDE_COLUMN = "Wide-Column Store (Cassandra / ScyllaDB)"
    GRAPH = "Graph DB (Neo4j / Amazon Neptune)"
    DISTRIBUTED_GRAPH = "Distributed Graph DB (TigerGraph / Amazon Neptune Analytics)"


@dataclass
class WorkloadProfile:
    """Describes the shape of a workload for DB selection."""

    # Core access pattern
    needs_joins: bool
    needs_acid_transactions: bool
    schema_is_fixed: bool

    # Scale and latency
    write_heavy: bool
    read_latency_sub_ms: bool

    # Data shape
    data_is_graph: bool
    wide_time_series_or_append: bool

    # --- New fields ---
    has_graph_queries: bool = False        # Primary queries involve graph traversal
    max_nodes_millions: int = 0            # Estimated graph scale in millions of nodes
    consistency_required: bool = False     # Strict consistency even on wide-column workload


def select_database(profile: WorkloadProfile) -> tuple[DBType, str]:
    """
    Return (recommended DB type, reason) for a given workload profile.

    Decision priority (top wins):
      1. Sub-ms latency  -> KV
      2. Graph queries   -> Graph or Distributed Graph (scale-aware)
      3. Append + write  -> Wide-Column (with consistency warning)
      4. Joins / ACID    -> Relational
      5. Flexible schema -> Document
      6. Default         -> Relational
    """
    logger.debug("Evaluating workload profile: %s", profile)

    # --- Priority 1: cache tier ---
    if profile.read_latency_sub_ms:
        reason = "Sub-ms latency requirement demands an in-memory KV store."
        return DBType.KEY_VALUE, reason

    # --- Priority 2: graph queries (new field takes precedence over data_is_graph flag) ---
    if profile.has_graph_queries or profile.data_is_graph:
        nodes = profile.max_nodes_millions
        if nodes > 10_000:  # > 10 billion nodes
            logger.warning(
                "Graph workload at %.0fM nodes exceeds typical Neo4j limits (~10B nodes). "
                "Consider a distributed graph DB (TigerGraph, Neptune Analytics).",
                nodes,
            )
            reason = (
                f"Graph traversal at {nodes}M nodes exceeds single-instance limits. "
                "Distributed graph engine required for this scale."
            )
            return DBType.DISTRIBUTED_GRAPH, reason

        reason = "Multi-hop graph traversals are index-free in a native graph DB."
        return DBType.GRAPH, reason

    # --- Priority 3: append-heavy wide-column ---
    if profile.wide_time_series_or_append and profile.write_heavy:
        if profile.consistency_required:
            logger.warning(
                "Wide-column workload requested WITH strict consistency. "
                "Cassandra is AP — consider Spanner or CockroachDB for CP + wide-column scale."
            )
            reason = (
                "Append-heavy writes suit Cassandra, but strict consistency conflicts with "
                "its AP design. Consider distributed SQL (Spanner / CockroachDB) instead."
            )
            return DBType.RELATIONAL, reason  # Recommend distributed SQL as safest fallback

        reason = (
            "Append-heavy writes + wide rows suit Cassandra's LSM-tree "
            "and partition-key-based distribution."
        )
        return DBType.WIDE_COLUMN, reason

    # --- Priority 4: relational needs ---
    if profile.needs_joins or profile.needs_acid_transactions:
        reason = (
            "Cross-entity queries or ACID transactions require "
            "a relational engine with full SQL planner support."
        )
        return DBType.RELATIONAL, reason

    # --- Priority 5: flexible schema ---
    if not profile.schema_is_fixed:
        reason = (
            "Flexible / evolving schema with document-level atomicity "
            "fits a document store."
        )
        return DBType.DOCUMENT, reason

    # --- Default ---
    reason = (
        "Fixed schema, moderate scale, no strong relational needs — "
        "relational SQL remains the safest default for operability."
    )
    return DBType.RELATIONAL, reason


def main() -> None:
    test_cases: list[tuple[str, WorkloadProfile]] = [
        (
            "Social graph (small — <1B nodes)",
            WorkloadProfile(
                needs_joins=False,
                needs_acid_transactions=False,
                schema_is_fixed=True,
                write_heavy=False,
                read_latency_sub_ms=False,
                data_is_graph=False,
                wide_time_series_or_append=False,
                has_graph_queries=True,
                max_nodes_millions=500,
            ),
        ),
        (
            "Hyperscale graph (>10B nodes)",
            WorkloadProfile(
                needs_joins=False,
                needs_acid_transactions=False,
                schema_is_fixed=True,
                write_heavy=False,
                read_latency_sub_ms=False,
                data_is_graph=False,
                wide_time_series_or_append=False,
                has_graph_queries=True,
                max_nodes_millions=50_000,  # 50 billion nodes
            ),
        ),
        (
            "IoT time-series (consistent)",
            WorkloadProfile(
                needs_joins=False,
                needs_acid_transactions=False,
                schema_is_fixed=True,
                write_heavy=True,
                read_latency_sub_ms=False,
                data_is_graph=False,
                wide_time_series_or_append=True,
                consistency_required=True,
            ),
        ),
        (
            "IoT time-series (eventual OK)",
            WorkloadProfile(
                needs_joins=False,
                needs_acid_transactions=False,
                schema_is_fixed=True,
                write_heavy=True,
                read_latency_sub_ms=False,
                data_is_graph=False,
                wide_time_series_or_append=True,
                consistency_required=False,
            ),
        ),
    ]

    for name, profile in test_cases:
        db_type, reason = select_database(profile)
        logger.info("[%-40s] -> %-50s | %s", name, db_type, reason)


if __name__ == "__main__":
    main()
```

**Key design decisions:**
- `has_graph_queries` takes precedence over the older `data_is_graph` flag — both trigger the graph branch so the dataclass is backward-compatible.
- `max_nodes_millions` drives the scale warning at 10,000M (10B) threshold — Neo4j clusters start struggling past that point.
- `consistency_required=True` on a wide-column workload produces a redirect to distributed SQL rather than silently returning Cassandra. This is an architectural red flag that must surface loudly, not silently.

---

## Exercise 3 — CAP Theorem: Network Partition in a Distributed KV Store

**Which two properties can you preserve?**

During a network partition you can preserve **Availability (A) + Partition Tolerance (P)** — or **Consistency (C) + Partition Tolerance (P)**. You cannot have all three simultaneously.

**Partition Tolerance is mandatory in any real distributed system.** Networks drop packets. You cannot design a distributed system that simply never partitions — hardware fails, switches drop, data centers lose connectivity. If you sacrifice P, you are building a single-node system.

**Concrete scenario:**

Cluster: nodes A (DC-West) and B (DC-East), both accept writes. Network link between them fails.

- **If you choose CP**: Node B detects it cannot reach Node A to confirm writes. Node B refuses new writes (returns errors) until the partition heals. Users in DC-East see downtime. Consistency preserved — every read still returns the latest confirmed write.

- **If you choose AP**: Node B continues accepting writes. Users in DC-East can read and write freely. When the partition heals, nodes A and B must reconcile diverged state. Some writes may conflict — you need a merge/LWW strategy. Consistency is temporarily violated.

**The tradeoff in plain terms:**
- CP = stop accepting writes when unsure = safe data, unhappy users during partitions.
- AP = keep accepting writes when unsure = happy users during partitions, reconciliation debt afterward.

**For a KV store used as a session cache** (data loss tolerable): choose AP — downtime is worse than a stale session.
**For a KV store used as a financial ledger**: choose CP — data loss is worse than brief unavailability.

---

## Exercise 4 — Document DB over Relational for Multi-Tenant SaaS

**Scenario:** Multi-tenant SaaS where each tenant configures custom fields (tenant A has `industry`, tenant B has `compliance_tier`, tenant C has `custom_field_99`).

**Three arguments for Document DB:**

1. **Schema-per-tenant without DDL hell**: In a relational DB, custom fields require either an EAV (Entity-Attribute-Value) table — which destroys query performance and type safety — or per-tenant schema migrations. A document store lets each tenant's record be a self-contained JSON document with its own shape. Adding a field for one tenant is an application-level change, not a database migration.

2. **Atomic document updates without cross-table transactions**: A tenant's record — including all nested custom fields — lives in one document. An update to a tenant's profile is a single atomic write. In a relational model, custom field values split across a `tenants` table and a `tenant_custom_fields` table require a multi-row transaction just to update one logical entity.

3. **Horizontal scale aligned with tenant isolation**: Documents can be sharded by `tenant_id` — all of tenant A's data lands on the same shard. In a relational multi-tenant model, a row in `custom_fields` for tenant A might not land on the same node as tenant A's rows in `tenants`. Document stores like MongoDB make shard key = tenant ID natural, enabling per-tenant data locality and easier compliance (GDPR data residency).

---

## Exercise 5 — Hybrid SQL + NoSQL for a Ride-Sharing App

**Workloads to split:**

| Data | Store | Reason |
|---|---|---|
| **Driver/rider profiles, payment methods** | PostgreSQL | Fixed schema, ACID for payment data, complex join queries (rider + payment + trip history) |
| **Active trip state** (current location, status, ETA) | Redis | Sub-ms latency. Ephemeral data — trip ends, state evicts. KV with TTL perfectly matches "active trip" lifecycle |
| **Real-time driver location** (1s updates, all active drivers) | Redis Geo / Cassandra | Geospatial proximity queries (GEORADIUS). Write-heavy (every driver pings every second). Cassandra partition key = geohash cell, clustering key = driver_id for region-level scatter-gather |
| **Trip history** (completed rides) | Cassandra | Append-only (trips never mutate after completion). Query patterns: by rider_id+date range, by driver_id+date range — both map to partition-key + clustering-key design |
| **Driver-rider graph** (recurring pairs, referral network) | Neo4j or PostgreSQL | If referral depth > 2 hops is a query, graph DB wins. If analytics only (flat reports), PostgreSQL + materialized view is simpler |
| **Surge pricing events** (stream) | Kafka + Cassandra | Event log is append-only and time-ordered. Cassandra's wide-column model with time clustering is ideal for high-throughput event storage |

**Key principle:** Derive the data store from the *access pattern*, not the data model. Active trip = cache. Historical trip = append-only column store. Profile + payment = ACID relational.

---

## Notes

- **CAP is a choice, not a bug**: Every distributed system designer must consciously pick CP or AP per component. A single app can use both — payments are CP (PostgreSQL), sessions are AP (Redis).
- **Document vs Relational is not "new vs old"**: Choose document when schema diversity and denormalization align with your query patterns. Choose relational when joins, constraints, and multi-entity transactions dominate.
- **The decision tree is a starting point**: Real systems almost always end up polyglot. The skill is knowing which workload goes where, and drawing a clear boundary between stores.
- **Interview tip**: Always justify DB choice with access patterns + consistency + scale constraints. "We used MongoDB because it's flexible" is a red flag. "We used MongoDB because schema varies per tenant and we never need cross-document joins" is an answer.
