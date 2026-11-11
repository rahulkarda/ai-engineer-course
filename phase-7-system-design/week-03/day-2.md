# Phase 7 · Week 03 · Day 2 — Indexing at Scale

> Objective: Understand how indexes are built and maintained internally, when they help vs. hurt, and how to design composite and distributed indexes for high-throughput systems.  ·  Time: ~1 hr  ·  Prereqs: [[day-1-week-03]]

---

## Concept

An index is a separate data structure that maps a subset of column values to row locations. Reading an index is faster than a full table scan because it is smaller and sorted. The cost is **write amplification**: every insert or update must also update every covering index.

### B-Tree Index Internals

The dominant index structure in relational databases (PostgreSQL, MySQL InnoDB) is the **B+ tree** — a balanced tree where:

- **Internal nodes** hold separator keys and child pointers.
- **Leaf nodes** hold the actual indexed values and row pointers (heap pointers or primary key refs).
- All leaf nodes are linked in sorted order, enabling efficient range scans.

**Page splits** are the key overhead. Each node is a fixed-size page (typically 8 KB in PostgreSQL). When a page is full and a new value must be inserted:

1. The page is split into two half-full pages.
2. The median key is promoted to the parent.
3. If the parent is also full, it splits recursively — up to the root.
4. A root split creates a new root, increasing tree height by 1.

Random inserts (e.g., UUIDs as primary keys) cause frequent page splits and leave pages ~50% full. Sequential inserts (e.g., auto-increment integers, timestamps) fill pages ~100% — much more efficient.

### Composite Indexes and Column Order

A composite index on `(a, b, c)` is sorted first by `a`, then by `b` within each `a` group, then by `c`. This means:

- Queries filtering on `a` alone: index used.
- Queries filtering on `a, b`: index used.
- Queries filtering on `b` alone: index **not** used (leftmost prefix rule).
- Queries filtering on `a` and `c` (skipping `b`): index used for `a` only; `c` requires row fetch.

**Rule**: Put the most selective column (highest cardinality, used in equality filters) first; range columns last.

### Covering Indexes

A **covering index** includes all columns needed to satisfy a query — the engine never touches the heap (table). Postgres calls this an "index-only scan."

```sql
-- Without covering index: index scan + heap fetch per row
SELECT email FROM users WHERE tenant_id = 42;

-- With covering index on (tenant_id, email): index-only scan
CREATE INDEX idx_users_tenant_email ON users(tenant_id, email);
```

### Partial Indexes

A **partial index** is built only over rows matching a WHERE predicate. Smaller, faster to build, faster to scan:

```sql
-- Only index active users — ignores the 90% who are inactive
CREATE INDEX idx_active_users ON users(email) WHERE active = true;
```

### Index Selectivity

**Selectivity** = distinct values / total rows. High selectivity (close to 1.0) means the index eliminates most rows — worth using. Low selectivity (boolean column, e.g., `active`) means the index barely filters — a full scan is often cheaper because random I/O for each matching row costs more than a sequential full scan.

### Write Amplification Tradeoff

Each index on a table multiplies write cost. A table with 5 indexes requires 6 writes per insert (1 heap + 5 index pages). For write-heavy workloads (Cassandra, Kafka consumers materializing to DB), minimize indexes to only those required for production queries.

### Secondary Indexes in Distributed Databases

Distributed databases (Cassandra, DynamoDB) distribute data across nodes by **partition key**. Secondary indexes have two strategies:

| Strategy | Where index lives | Read cost | Write cost |
|---|---|---|---|
| **Local** | Same node as data | Must query all nodes (scatter-gather) | Low — co-located write |
| **Global** | Separate index partition | Single node lookup | High — cross-node write, eventual consistency |

DynamoDB's Global Secondary Index (GSI) is eventually consistent: a write to the main table propagates to the GSI asynchronously.

### Bloom Filters as Pre-filters

A **Bloom filter** is a probabilistic bit-array structure: it answers "definitely not in set" or "possibly in set." False positives exist; false negatives do not.

Cassandra uses Bloom filters per SSTable: before doing a disk read, it checks the filter. If the filter says "absent," the SSTable is skipped entirely — reducing I/O dramatically for point lookups of non-existent keys.

---

## How It Works Internally

### B+ Tree Node Layout

```
Internal node (8 KB page):
  [ ptr_0 | key_1 | ptr_1 | key_2 | ptr_2 | ... | key_n | ptr_n ]

Leaf node (8 KB page):
  [ (key, heap_ptr), (key, heap_ptr), ... ] --> [next_leaf_ptr]
```

A lookup for key `k`:
1. Read root page — binary search for `k`, follow pointer.
2. Repeat down each level (typically 3–4 levels for millions of rows).
3. At leaf, return heap pointer → fetch actual row from heap.

**Height of tree**: For a page size of 8 KB and 8-byte keys + 8-byte pointers, each internal node holds ~500 entries. 3-level tree: 500^3 = 125 million rows. 4 I/Os for any lookup.

### Write Path (LSM vs B-Tree)

B-tree writes are **in-place**: find the leaf page, modify it, mark dirty, flush. This causes random I/O — expensive on spinning disks, reasonable on SSDs.

LSM (Log-Structured Merge) trees (used by Cassandra, RocksDB) write to an in-memory buffer first, then flush sorted runs to disk. Compaction merges runs periodically. Writes are **sequential** — very fast. Reads may touch multiple levels — mitigated by Bloom filters and block caches.

---

## Worked Example

```python
"""
index_simulation.py — Compare B-tree lookup cost vs full scan,
and demonstrate composite index column ordering effect.

Requires: Python 3.12+
"""

import logging
import math
import random
import time
from dataclasses import dataclass, field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# B-tree cost model
# ---------------------------------------------------------------------------

PAGE_SIZE_BYTES = 8_192
KEY_SIZE_BYTES = 8
PTR_SIZE_BYTES = 8
ENTRIES_PER_INTERNAL_NODE = PAGE_SIZE_BYTES // (KEY_SIZE_BYTES + PTR_SIZE_BYTES)


def btree_height(num_rows: int) -> int:
    """
    Estimate B+ tree height for a given row count.
    Height = ceil(log_{fan_out}(num_rows)).
    """
    if num_rows <= 1:
        return 1
    height = math.ceil(math.log(num_rows) / math.log(ENTRIES_PER_INTERNAL_NODE))
    return max(1, height)


def btree_lookup_io(num_rows: int) -> int:
    """
    Number of page I/Os for a single point lookup via B-tree index.
    = tree height (traverse) + 1 (heap fetch).
    """
    return btree_height(num_rows) + 1  # +1 for heap page fetch


def full_scan_io(num_rows: int, row_size_bytes: int = 100) -> int:
    """
    Number of page I/Os for a full sequential scan.
    = ceil(num_rows * row_size / page_size)
    """
    total_bytes = num_rows * row_size_bytes
    return math.ceil(total_bytes / PAGE_SIZE_BYTES)


def print_cost_comparison(num_rows: int) -> None:
    index_ios = btree_lookup_io(num_rows)
    scan_ios = full_scan_io(num_rows)
    height = btree_height(num_rows)
    logger.info(
        "rows=%-12d | btree_height=%d | index_lookup_ios=%-5d | full_scan_ios=%-8d | speedup=%.0fx",
        num_rows,
        height,
        index_ios,
        scan_ios,
        scan_ios / index_ios,
    )


# ---------------------------------------------------------------------------
# Composite index ordering simulation
# ---------------------------------------------------------------------------

@dataclass
class Row:
    tenant_id: int   # low cardinality: 100 distinct values
    status: str      # very low cardinality: 3 values
    user_id: int     # high cardinality: unique per row


def generate_dataset(n: int) -> list[Row]:
    statuses = ["active", "inactive", "pending"]
    return [
        Row(
            tenant_id=random.randint(1, 100),
            status=random.choice(statuses),
            user_id=i,
        )
        for i in range(n)
    ]


@dataclass
class SimulatedIndex:
    """
    Simulates a composite index by pre-sorting and binary-searching.
    columns: ordered list of attribute names to index.
    """
    columns: list[str]
    _data: list[Row] = field(default_factory=list, repr=False)

    def build(self, rows: list[Row]) -> None:
        import functools

        def compare(r: Row) -> tuple:
            return tuple(getattr(r, col) for col in self.columns)

        self._data = sorted(rows, key=compare)
        logger.info(
            "Built index on (%s) with %d rows",
            ", ".join(self.columns),
            len(self._data),
        )

    def query(self, **filters: object) -> tuple[list[Row], int]:
        """
        Scan the index. Return (matching rows, pages_read).
        pages_read approximates I/O cost:
          - leading column equality: binary search to start, scan matches
          - non-leading columns: must scan entire index (no prefix pruning)
        """
        # Determine if we can use the leading prefix
        leading_covered = all(
            col in filters for col in self._data and False  # placeholder
        )

        # Simplified: check if first column is in filters
        first_col = self.columns[0]
        if first_col in filters:
            # Binary search simulation: O(log n) pages to find range
            start_io = btree_height(len(self._data))
            matches = [r for r in self._data if self._matches(r, filters)]
            # Pages to scan the matching range
            range_io = max(1, math.ceil(len(matches) / 100))  # 100 rows/page approx
            total_io = start_io + range_io
            method = "index range scan (leading prefix)"
        else:
            # Full index scan required
            matches = [r for r in self._data if self._matches(r, filters)]
            total_io = full_scan_io(len(self._data), row_size_bytes=24)
            method = "full index scan (no leading prefix)"

        logger.info(
            "Index (%s) | query=%s | method=%s | results=%d | est_io=%d",
            ", ".join(self.columns),
            filters,
            method,
            len(matches),
            total_io,
        )
        return matches, total_io

    def _matches(self, row: Row, filters: dict) -> bool:
        return all(getattr(row, col) == val for col, val in filters.items())


def demonstrate_column_order(rows: list[Row]) -> None:
    """
    Same query, two different index column orderings.
    Shows that leading-column prefix determines whether index range scan is used.
    """
    query_filter = {"tenant_id": 5, "status": "active"}

    # Good order: high-selectivity equality first, then low-selectivity
    idx_good = SimulatedIndex(columns=["tenant_id", "status", "user_id"])
    idx_good.build(rows)
    _, io_good = idx_good.query(**query_filter)

    # Bad order: low-selectivity first
    idx_bad = SimulatedIndex(columns=["status", "tenant_id", "user_id"])
    idx_bad.build(rows)
    _, io_bad = idx_bad.query(**query_filter)

    logger.info(
        "Column order impact: good_order_io=%d vs bad_order_io=%d (%.1fx worse)",
        io_good,
        io_bad,
        io_bad / io_good,
    )


def main() -> None:
    logger.info("=== B-Tree Lookup Cost vs Full Scan ===")
    for n in [1_000, 10_000, 100_000, 1_000_000, 10_000_000]:
        print_cost_comparison(n)

    logger.info("")
    logger.info("=== Composite Index Column Order Effect ===")
    rows = generate_dataset(50_000)
    demonstrate_column_order(rows)


if __name__ == "__main__":
    main()
```

### ASCII B-Tree Diagram

```
B+ Tree (fan-out = 4, simplified)

Level 0 (root):
  [ 25 | 50 | 75 ]
   /     |     |    \

Level 1 (internal):
[10|20] [30|40] [55|65] [80|90]
  |  |     |  |    |  |    |  |

Level 2 (leaves, linked):
[5,8,9]->[10,15,19]->[20,22,24]->[25,27,30]-> ... (sorted linked list)
             |
             heap pointer to actual row data
```

Each lookup: root read → internal read → leaf read → heap read = **4 I/Os** for millions of rows.

---

## Exercises

1. Modify `print_cost_comparison` to also compute cost when **selectivity is 10%** (index is used but 10% of rows match). At what table size does a full scan become cheaper than index + heap fetches for 10% selectivity? Hint: index saves work only when matching rows are sparse.

2. Add a `BloomFilter` class with `add(key)` and `might_contain(key) -> bool` methods using Python's `bitarray` (or simulate with a set of hashes). Demonstrate false positive rate at different filter sizes for 10,000 inserted keys.

3. Write a pytest test for `btree_height` asserting:
   - 1 row → height 1
   - 512 rows → height 1
   - 513 rows → height 2
   - 1,000,000 rows → height ≤ 4

4. Design a composite index for this query pattern on a `logs` table: `WHERE service_id = ? AND level = 'ERROR' AND created_at > ?`. Justify the column order. What would a covering index look like?

5. Explain in comments: why does using a UUID (random) as a primary key in PostgreSQL cause more page splits than a BIGSERIAL (sequential)? What is the practical impact on insert throughput?

(Solutions: ../solutions/week-03/day-2.md)

---

## Self-check

- [ ] I can draw a B+ tree and explain what happens during a page split.
- [ ] I can apply the leftmost prefix rule to determine if a composite index is usable for a given query.
- [ ] I can explain write amplification and know when to avoid over-indexing.
- [ ] I can describe the difference between local and global secondary indexes in a distributed DB.
- [ ] I can implement `btree_height` and `btree_lookup_io` from scratch without reference.

---

## Supplementary Links

- [PostgreSQL — Index Internals](https://www.postgresql.org/docs/current/btree-implementation.html)
- [Use the Index, Luke — Visual B-Tree explanation](https://use-the-index-luke.com/sql/anatomy/the-tree)
- [Cassandra — Storage Engine and Bloom Filters](https://cassandra.apache.org/doc/latest/cassandra/architecture/storage-engine.html)
- [DynamoDB — Global vs Local Secondary Indexes](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/SecondaryIndexes.html)
- [RocksDB — LSM Tree Architecture](https://github.com/facebook/rocksdb/wiki/RocksDB-Overview)
- [Designing Data-Intensive Applications — Kleppmann (Ch 3, Storage and Retrieval)](https://dataintensive.net/)
