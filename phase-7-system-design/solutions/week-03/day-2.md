# Solutions — Phase 7 · Week 3 · Day 2 — Indexing at Scale

## Exercise 1 — Optimal Composite Index Column Order

**Query:**
```sql
SELECT * FROM events
WHERE status = 'active'
  AND created_at > '2024-01-01'
ORDER BY user_id;
```

**Optimal column order: `(status, created_at, user_id)`**

**Why — applying the leftmost prefix rule:**

1. **`status` first** — equality filter (`=`). Put equality columns before range columns. The index can binary-search directly to the `status = 'active'` subtree, eliminating all non-active rows before scanning anything.

2. **`created_at` second** — range filter (`>`). Within the `status = 'active'` partition of the index, rows are sorted by `created_at`. The engine performs a range scan starting at `2024-01-01` — no heap fetch needed to evaluate this predicate.

3. **`user_id` third** — ORDER BY column. If `user_id` is in the index, PostgreSQL can return results in sorted order *from the index itself* without a separate sort step (no `Sort` node in EXPLAIN). This is the covering/sorting benefit.

**Why NOT `(created_at, status, user_id)`:**
- Leading range column means the index is sorted by `created_at` first. The engine must scan all rows where `created_at > '2024-01-01'` and then filter by `status` within that range — cannot prune to `status = 'active'` first.

**Why NOT `(user_id, status, created_at)`:**
- `user_id` is not in the WHERE clause at all. The index cannot be used for the filter predicates — full index scan or full table scan.

**Covering index (optional)** — if the query returns only a few columns, add them to avoid heap fetches:
```sql
CREATE INDEX idx_events_status_created_user
  ON events(status, created_at, user_id)
  INCLUDE (event_type, payload_size);
-- INCLUDE columns are stored in leaf nodes only, not used for ordering/filtering
```

---

## Exercise 2 — B-Tree Height Calculation for 1M Records

**Given:**
- Page size: 4 KB = 4,096 bytes
- Key size: 8 bytes
- Pointer size: 8 bytes
- Row count: 1,000,000

**Step 1 — Fan-out (entries per internal node):**

Each internal node entry consists of one key + one pointer = 8 + 8 = 16 bytes.

```
fan_out = floor(page_size / (key_size + pointer_size))
        = floor(4096 / 16)
        = 256
```

**Step 2 — Tree height:**

Height = ceil(log_fan_out(num_rows))

```
height = ceil(log_256(1_000_000))
       = ceil(ln(1_000_000) / ln(256))
       = ceil(13.8155 / 5.5452)
       = ceil(2.492)
       = 3
```

**Verification — how many rows does a 3-level tree cover?**
```
Level 0 (root):       256 pointers
Level 1 (internal):   256 × 256 = 65,536 pointers
Level 2 (leaves):     65,536 × 256 = 16,777,216 rows
```

A 3-level tree with 4KB pages and 8-byte keys covers up to **16.7 million rows** — 1M rows fits comfortably.

**Total I/Os for a point lookup:**
```
I/Os = tree height + 1 (heap fetch)
     = 3 + 1
     = 4 I/Os
```

**Note on PostgreSQL's 8KB page size (for comparison):**
```
fan_out_8kb = floor(8192 / 16) = 512
height      = ceil(log_512(1_000_000)) = ceil(13.8155 / 6.238) = ceil(2.215) = 3
```
Same height because 1M rows is small. At 100M rows with 8KB pages: height = ceil(log_512(100_000_000)) = ceil(18.42 / 6.238) = 3 still. At 1B rows: ceil(20.7 / 6.238) = 4 I/Os.

---

## Exercise 3 — BloomFilter Implementation

```python
"""
bloom_filter.py — A simple Bloom filter using hashlib.

A Bloom filter is a probabilistic data structure that:
  - Guarantees: "definitely not in set" (no false negatives)
  - Accepts:    "probably in set" (false positives possible)

False positive rate: (1 - e^(-k*n/m))^k
  where k = number of hash functions, n = items inserted, m = bit array size.

Usage:
    python bloom_filter.py

Requires: Python 3.12+
"""

import hashlib
import logging
import math
from typing import Union

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


class BloomFilter:
    """
    Probabilistic membership test using a bit array and k hash functions.

    Args:
        size:        Number of bits in the filter. Larger = lower false positive rate.
        hash_count:  Number of independent hash functions to apply per item.

    Notes:
        - Uses SHA-256 seeded with an index to derive k independent hash functions.
        - No external dependencies — pure stdlib.
        - Thread-unsafe (fine for single-threaded usage; add a lock for concurrent access).
    """

    def __init__(self, size: int = 1_000_000, hash_count: int = 7) -> None:
        if size < 1:
            raise ValueError(f"size must be >= 1, got {size}")
        if hash_count < 1:
            raise ValueError(f"hash_count must be >= 1, got {hash_count}")

        self._size = size
        self._hash_count = hash_count
        self._bit_array: list[bool] = [False] * size
        self._items_added: int = 0

        logger.info(
            "BloomFilter initialized: size=%d bits (%.2f KB), hash_count=%d",
            size,
            size / 8 / 1024,
            hash_count,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, item: Union[str, bytes]) -> None:
        """Insert an item into the filter."""
        for bit_index in self._hash_positions(item):
            self._bit_array[bit_index] = True
        self._items_added += 1

    def might_contain(self, item: Union[str, bytes]) -> bool:
        """
        Return True if item *might* be in the set (possible false positive).
        Return False if item is *definitely not* in the set.
        """
        return all(
            self._bit_array[bit_index]
            for bit_index in self._hash_positions(item)
        )

    @property
    def false_positive_rate(self) -> float:
        """
        Theoretical false positive probability given current item count.
        Formula: (1 - e^(-k*n/m))^k
        """
        k = self._hash_count
        n = self._items_added
        m = self._size
        if n == 0:
            return 0.0
        return (1.0 - math.exp(-k * n / m)) ** k

    def __len__(self) -> int:
        return self._items_added

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hash_positions(self, item: Union[str, bytes]) -> list[int]:
        """
        Generate hash_count bit positions for the given item.

        Strategy: for each seed i in range(hash_count), compute
          SHA-256(seed_bytes + item_bytes) and take modulo size.
        This gives k statistically independent hash functions from one primitive.
        """
        if isinstance(item, str):
            item_bytes = item.encode("utf-8")
        else:
            item_bytes = item

        positions: list[int] = []
        for i in range(self._hash_count):
            seed = i.to_bytes(4, byteorder="big")
            digest = hashlib.sha256(seed + item_bytes).digest()
            # Use first 8 bytes of digest as a 64-bit integer
            int_val = int.from_bytes(digest[:8], byteorder="big")
            positions.append(int_val % self._size)
        return positions

    @classmethod
    def optimal_params(cls, expected_items: int, target_fpr: float) -> tuple[int, int]:
        """
        Calculate optimal (size, hash_count) for a given item count and
        target false positive rate.

        m = -n * ln(p) / (ln(2))^2
        k = (m/n) * ln(2)
        """
        if not 0 < target_fpr < 1:
            raise ValueError("target_fpr must be between 0 and 1 exclusive")
        m = math.ceil(-expected_items * math.log(target_fpr) / (math.log(2) ** 2))
        k = max(1, round((m / expected_items) * math.log(2)))
        return m, k


# ------------------------------------------------------------------
# Demo / false positive rate verification
# ------------------------------------------------------------------

def demonstrate_false_positive_rates() -> None:
    """
    Insert 10,000 items into filters of various sizes and measure
    actual false positive rate against theoretical prediction.
    """
    n_items = 10_000
    test_keys = [f"key:{i}" for i in range(n_items)]
    absent_keys = [f"absent:{i}" for i in range(n_items)]  # never inserted

    for size_multiplier, label in [(5, "tight (5x)"), (10, "moderate (10x)"), (20, "loose (20x)")]:
        size = n_items * size_multiplier
        bf = BloomFilter(size=size, hash_count=7)

        for key in test_keys:
            bf.add(key)

        # Measure false positives on keys that were never inserted
        false_positives = sum(1 for k in absent_keys if bf.might_contain(k))
        actual_fpr = false_positives / len(absent_keys)

        # Verify no false negatives (all inserted items must be found)
        false_negatives = sum(1 for k in test_keys if not bf.might_contain(k))

        logger.info(
            "size=%7d | theoretical_fpr=%.4f | actual_fpr=%.4f | false_negatives=%d | label=%s",
            size,
            bf.false_positive_rate,
            actual_fpr,
            false_negatives,
            label,
        )


def demonstrate_optimal_params() -> None:
    """Show optimal filter sizing for given constraints."""
    for n, fpr in [(10_000, 0.01), (100_000, 0.001), (1_000_000, 0.0001)]:
        m, k = BloomFilter.optimal_params(n, fpr)
        logger.info(
            "n=%8d | target_fpr=%.4f | optimal_size=%9d bits (%.1f KB) | optimal_k=%d",
            n, fpr, m, m / 8 / 1024, k,
        )


def main() -> None:
    logger.info("=== False Positive Rate at Different Filter Sizes ===")
    demonstrate_false_positive_rates()

    logger.info("")
    logger.info("=== Optimal Parameter Sizing ===")
    demonstrate_optimal_params()

    logger.info("")
    logger.info("=== Basic Usage Demo ===")
    m, k = BloomFilter.optimal_params(expected_items=1000, target_fpr=0.01)
    bf = BloomFilter(size=m, hash_count=k)
    for i in range(1000):
        bf.add(f"user:{i}")

    logger.info("'user:500' might_contain=%s (should be True)", bf.might_contain("user:500"))
    logger.info("'user:9999' might_contain=%s (likely False)", bf.might_contain("user:9999"))
    logger.info("No false negatives: %s", all(bf.might_contain(f"user:{i}") for i in range(1000)))


if __name__ == "__main__":
    main()
```

**Key implementation notes:**
- No `bitarray` library needed — `list[bool]` works for a demonstration; production code should use `bytearray` or `bitarray` for memory efficiency (8x compression).
- k independent hash functions are derived by seeding SHA-256 with an index — avoids needing k different hash primitives.
- False negatives are **impossible** by construction — if all k bits were set on insertion, `might_contain` will always find them.

---

## Exercise 4 — Local vs Global Secondary Index on `email` in a Distributed Table

**Context:** Table partitioned by `user_id` (hash-based), secondary index on `email` for login lookups.

### Local Secondary Index (LSI)

Index lives on the **same node as the data it indexes**. Each node maintains a local index over only its partition's rows.

| Dimension | Detail |
|---|---|
| **Write cost** | Low — index write is co-located, single-node, synchronous |
| **Read — lookup by email** | Must query **all nodes** (scatter-gather). No node knows which partition holds a given email. Coordinator fans out to all N nodes, waits for all responses, merges results. Latency = slowest node (tail latency problem). |
| **Consistency** | Strong — index update happens in the same transaction as the data write |
| **Storage** | Distributed proportionally — each node stores its own slice of the index |

**When to use:** Queries that always include the partition key (`user_id`) plus a secondary filter. Example: `WHERE user_id = 123 AND email = 'x'` — single-node lookup.

### Global Secondary Index (GSI)

Index lives on **separate partitions**, keyed by the index column (`email`). All nodes write to a shared index partition set.

| Dimension | Detail |
|---|---|
| **Write cost** | Higher — every write to the main table triggers an asynchronous write to the GSI partition. Two-phase: write to main table, then (async) write to index. |
| **Read — lookup by email** | Single-node — email is the partition key of the GSI. One hop. Low, predictable latency. |
| **Consistency** | Eventual — GSI is updated asynchronously. A login immediately after account creation may fail (GSI not yet updated). |
| **Storage** | Duplicates all indexed data on separate nodes. 2x storage overhead for the indexed columns. |

**When to use:** Queries by non-partition-key columns where scatter-gather is unacceptable. DynamoDB's GSI is the canonical example.

### Tradeoff Summary

```
LSI:  write fast, read slow (scatter-gather) + strongly consistent
GSI:  write slow + eventual, read fast (single node)
```

**For email-based login on a high-traffic system:** GSI wins — login must be fast, and eventual consistency is acceptable (a few-millisecond window between account creation and email index propagation is tolerable with a retry on the client).

**For login on a financial system where consistency is critical:** Use a dedicated user-lookup service backed by a separate relational DB keyed on email, rather than relying on GSI eventual consistency.

---

## Exercise 5 — When Does a Partial Index Outperform a Full Index?

A partial index wins when **a small, well-defined subset of rows is queried almost exclusively**, and the rest of the table is effectively "cold" for that query pattern.

### Concrete Example: Active Users

```sql
-- Table: users
-- 10,000,000 total rows
-- 200,000 active (status = 'active') -- 2% of table
-- 9,800,000 inactive

-- Full index on email:
CREATE INDEX idx_users_email ON users(email);
-- Index size: ~10M entries × ~30 bytes = ~300 MB
-- Scan for active user login: index lookup + heap fetch + filter status = 'active'

-- Partial index on email WHERE active:
CREATE INDEX idx_active_users_email ON users(email) WHERE status = 'active';
-- Index size: ~200K entries × ~30 bytes = ~6 MB
-- Scan for active user login: 50x smaller index, fits in buffer cache
```

**Approximate numbers:**

| Metric | Full Index | Partial Index |
|---|---|---|
| Index size | ~300 MB | ~6 MB |
| Fits in shared_buffers (8GB RAM) | Partially | Entirely |
| Login query I/Os | 3-4 (B-tree + heap) | 1-2 (entire index in memory) |
| Index build time | ~60s | ~1.2s |
| Insert overhead | Always update index | Only update for active rows |

**When partial index beats full:**
1. A boolean/enum column with very low cardinality (active=true is 2% of rows).
2. A "hot path" query that filters on a rare status (e.g., `WHERE status = 'pending'` for a job queue).
3. Soft-delete patterns: `WHERE deleted_at IS NULL` — active rows are 5%, deleted rows 95%.

**When partial index does NOT help:**
- The query does not match the partial index predicate (queries for inactive users never use the partial index — need a separate full index or accept a scan).
- The subset is large (>20% of table) — the savings in size are insufficient to justify the operational complexity of maintaining two indexes.

---

## Notes

- **Composite index column order is the single most impactful indexing decision**: equality first, range last, ORDER BY last — always.
- **Write amplification compounds**: 5 indexes on a table = 6x write cost. Profile before adding. Each index must justify its read benefit against the write tax.
- **Bloom filters are not a substitute for indexes**: they eliminate I/Os for "definitely not present" cases but cannot answer "give me the row." They are a pre-filter to avoid disk reads, not a query mechanism.
- **B-tree height grows logarithmically**: a 4-level tree handles billions of rows. Height is almost never the bottleneck — I/O for the heap fetch after the index lookup is.
- **LSM vs B-tree**: if your workload is write-heavy with infrequent point reads, consider Cassandra/RocksDB (LSM). If reads dominate and random access is the pattern, B-tree (PostgreSQL) wins. The index structure should match the access pattern at the storage layer too, not just the query layer.
- **Interview tip**: When asked "how would you optimize this query?", always ask for the access pattern first (frequency, selectivity, cardinality). Index design is access-pattern design.
