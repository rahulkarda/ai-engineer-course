# Phase 7 · Week 03 · Day 1 — SQL vs NoSQL Selection

> Objective: Build a principled framework for choosing a database engine based on access patterns, consistency needs, and scale constraints.  ·  Time: ~1 hr  ·  Prereqs: [[day-0-week-02]]

---

## Concept

Every database design decision is an access-pattern decision. The question is never "which DB is better?" — it is "which DB matches the shape of my reads and writes?"

### Relational Model Recap

Relational databases organize data into **tables** (relations) with a strict schema. Rows share a type; columns are typed. Foreign keys enforce referential integrity. The query engine uses **joins** to combine relations at read time, keeping data normalized at write time.

Key guarantees: **ACID**

| Property | Meaning |
|---|---|
| Atomicity | Transaction completes fully or not at all |
| Consistency | Every transaction moves DB from valid state to valid state |
| Isolation | Concurrent transactions do not see each other's partial state |
| Durability | Committed data survives crashes |

### CAP Theorem (brief)

In a distributed system, you can guarantee at most **two** of:

- **C**onsistency — every read sees the latest write
- **A**vailability — every request gets a (non-error) response
- **P**artition tolerance — the system continues under network splits

Network partitions are unavoidable in practice, so real systems choose between **CP** (prioritize consistency, risk unavailability) and **AP** (prioritize availability, risk stale reads). SQL databases on a single node sidestep CAP — they give full ACID. Distributed SQL (CockroachDB, Spanner) pays latency to keep C. NoSQL systems often relax C for A+P.

---

## How It Works Internally

### Why SQL Wins on Joins

A relational engine maintains **statistics** (row counts, cardinality, histograms) and chooses a **query plan** — hash join, nested loop, merge join — at parse time. The planner can exploit foreign-key indexes to traverse relations in O(log n) per lookup. Normalization means writes update one place; the planner assembles the full picture at query time.

### NoSQL Internal Models

**Document (e.g., MongoDB)** — stores JSON/BSON documents. Each document is self-contained (denormalized). Reads are fast because related data co-locates. Updates to embedded arrays are atomic at the document level. Expensive: multi-document transactions (added later, costly).

**Key-Value (e.g., Redis)** — hash table in memory, optionally persisted. O(1) get/set. No query language. Zero schema. Ideal for caches, sessions, counters, pub-sub. Loses all relational query ability.

**Wide-Column (e.g., Cassandra)** — rows keyed by a **partition key**; within a partition, data sorted by a **clustering key**. Data physically sorted on disk for fast range scans on clustering columns. Writes are cheap (append-only LSM tree). Reads across partition keys require scatter-gather — design schema around query, not entity.

**Graph (e.g., Neo4j)** — nodes and edges stored as pointer-linked records. Traversals follow pointers directly (index-free adjacency) — no join overhead per hop. Ideal when relationship depth (2-4+ hops) is the primary query dimension.

---

## Worked Example

```python
"""
db_selector.py — Decision helper for database type selection.

Usage:
    python db_selector.py

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


@dataclass
class WorkloadProfile:
    """Describes the shape of a workload for DB selection."""

    # Core access pattern
    needs_joins: bool               # Multi-entity relational queries
    needs_acid_transactions: bool   # Strong transactional guarantees
    schema_is_fixed: bool           # Schema known at design time

    # Scale and latency
    write_heavy: bool               # Write throughput > read throughput
    read_latency_sub_ms: bool       # Sub-millisecond read SLA (cache-tier)

    # Data shape
    data_is_graph: bool             # Primary queries are multi-hop traversals
    wide_time_series_or_append: bool  # Append-only, time-series, or event log


def select_database(profile: WorkloadProfile) -> tuple[DBType, str]:
    """
    Return (recommended DB type, reason) for a given workload profile.

    Decision logic follows the ASCII tree below — top conditions win.
    """
    logger.debug("Evaluating workload profile: %s", profile)

    if profile.read_latency_sub_ms:
        reason = "Sub-ms latency requirement demands in-memory KV store."
        return DBType.KEY_VALUE, reason

    if profile.data_is_graph:
        reason = "Multi-hop graph traversals are index-free in a native graph DB."
        return DBType.GRAPH, reason

    if profile.wide_time_series_or_append and profile.write_heavy:
        reason = (
            "Append-heavy writes + wide rows suit Cassandra's LSM-tree "
            "and partition-key-based distribution."
        )
        return DBType.WIDE_COLUMN, reason

    if profile.needs_joins or profile.needs_acid_transactions:
        reason = (
            "Cross-entity queries or ACID transactions require "
            "a relational engine with full SQL planner support."
        )
        return DBType.RELATIONAL, reason

    if not profile.schema_is_fixed:
        reason = (
            "Flexible / evolving schema with document-level atomicity "
            "fits a document store."
        )
        return DBType.DOCUMENT, reason

    # Default: schema fixed, no joins, no ACID, no extreme latency
    reason = (
        "Fixed schema, moderate scale, no strong relational needs — "
        "relational SQL remains the safest default for operability."
    )
    return DBType.RELATIONAL, reason


def main() -> None:
    test_cases: list[tuple[str, WorkloadProfile]] = [
        (
            "Session cache",
            WorkloadProfile(
                needs_joins=False,
                needs_acid_transactions=False,
                schema_is_fixed=False,
                write_heavy=True,
                read_latency_sub_ms=True,
                data_is_graph=False,
                wide_time_series_or_append=False,
            ),
        ),
        (
            "E-commerce orders",
            WorkloadProfile(
                needs_joins=True,
                needs_acid_transactions=True,
                schema_is_fixed=True,
                write_heavy=False,
                read_latency_sub_ms=False,
                data_is_graph=False,
                wide_time_series_or_append=False,
            ),
        ),
        (
            "IoT sensor time-series",
            WorkloadProfile(
                needs_joins=False,
                needs_acid_transactions=False,
                schema_is_fixed=True,
                write_heavy=True,
                read_latency_sub_ms=False,
                data_is_graph=False,
                wide_time_series_or_append=True,
            ),
        ),
        (
            "Social network friend-of-friend",
            WorkloadProfile(
                needs_joins=False,
                needs_acid_transactions=False,
                schema_is_fixed=True,
                write_heavy=False,
                read_latency_sub_ms=False,
                data_is_graph=True,
                wide_time_series_or_append=False,
            ),
        ),
        (
            "CMS with variable fields",
            WorkloadProfile(
                needs_joins=False,
                needs_acid_transactions=False,
                schema_is_fixed=False,
                write_heavy=False,
                read_latency_sub_ms=False,
                data_is_graph=False,
                wide_time_series_or_append=False,
            ),
        ),
    ]

    for name, profile in test_cases:
        db_type, reason = select_database(profile)
        logger.info("[%-35s] -> %-45s | %s", name, db_type, reason)


if __name__ == "__main__":
    main()
```

### ASCII Decision Tree

```
START
  |
  |-- read_latency_sub_ms? --(yes)--> KEY-VALUE (Redis)
  |
  |-- data_is_graph? --------(yes)--> GRAPH (Neo4j)
  |
  |-- write_heavy + append? --(yes)--> WIDE-COLUMN (Cassandra)
  |
  |-- needs_joins OR ACID? --(yes)--> RELATIONAL SQL (PostgreSQL)
  |
  |-- schema flexible? ------(yes)--> DOCUMENT (MongoDB)
  |
  `--(default)----------------(no)--> RELATIONAL SQL (safest default)
```

---

## Exercises

1. Extend `WorkloadProfile` with a `max_nodes_millions: int` field. Update `select_database` so that graph workloads above 10 billion nodes suggest a distributed graph (TigerGraph) with a log message warning about scale limits.

2. Add a `consistency_required: bool` field. If the workload is wide-column but also requires strict consistency, return a warning alongside the recommendation. Write a pytest unit test covering at least four workload profiles.

3. Sketch the schema (tables + foreign keys) for an e-commerce system: `users`, `orders`, `order_items`, `products`. Then redesign it as a MongoDB document model. List the trade-offs of each approach in comments.

4. Given a workload: 500k writes/sec, keys are UUIDs, reads are single-key lookups, no relationships. Which DB family wins? Justify using CAP and the decision tree.

5. Research question: Spanner and CockroachDB claim to be "globally consistent SQL." What mechanism do they use to achieve this at the cost of latency? (Answer in comments, no code required.)

(Solutions: ../solutions/week-03/day-1.md)

---

## Self-check

- [ ] I can explain the ACID properties from first principles without notes.
- [ ] I can describe how a wide-column store's partition key + clustering key differs from a relational primary key.
- [ ] I can walk through the decision tree for a new system design question in an interview setting.
- [ ] I can implement `select_database` from scratch with type hints and logging.
- [ ] I can state the CAP theorem and explain why partition tolerance is not optional in distributed systems.

---

## Supplementary Links

- [PostgreSQL Docs — Query Planning](https://www.postgresql.org/docs/current/planner-optimizer.html)
- [MongoDB Data Modeling](https://www.mongodb.com/docs/manual/core/data-modeling-introduction/)
- [Cassandra Data Modeling — choosing partition keys](https://cassandra.apache.org/doc/latest/cassandra/data_modeling/intro.html)
- [Neo4j Index-Free Adjacency Explained](https://neo4j.com/blog/index-free-adjacency/)
- [Brewer's CAP Theorem (original paper)](https://www.cs.berkeley.edu/~brewer/cs262b-2004/PODC-keynote.pdf)
- [Designing Data-Intensive Applications — Kleppmann (Ch 2)](https://dataintensive.net/)
