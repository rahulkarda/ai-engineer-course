# Phase 2 · Week 08 · Day 5 — Database Internals

> Objective: Understand how databases store, index, plan, and replicate data — the "why" behind slow queries, schema decisions, and distributed consistency.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

---

## Concept

Most application engineers treat the database as a black box. That works until it doesn't: a query inexplicably slows down at 10M rows, a transaction behaves unexpectedly under concurrency, a replica serves stale data during a network partition.

Knowing three layers fixes 80% of surprises:
1. **Storage structure** — how data is laid out on disk (B-tree vs LSM-tree).
2. **Query execution** — how the planner turns SQL into physical operations.
3. **Concurrency and distribution** — isolation levels, replication, sharding, and the constraints they impose.

---

## How it works internally

### B-tree vs LSM-tree

**B-tree (PostgreSQL, MySQL InnoDB, SQLite):**

- Balanced tree of pages (typically 8–16 KB each). Leaf pages hold actual row data or index entries.
- **Reads**: O(log N) page reads from root to leaf.
- **Writes**: update in place. Random I/O — the page containing the row must be read, modified, written back.
- **Read amplification**: low (log N pages).
- **Write amplification**: moderate — writes go to the WAL (Write-Ahead Log) first (sequential), then to the B-tree page (random). A single logical write touches 2–3 physical pages.
- **Space amplification**: low — deletions reclaim space.
- Optimal for: read-heavy workloads, range scans, mixed OLTP.

**LSM-tree (RocksDB, Cassandra, ClickHouse MergeTree):**

- Writes go first to an in-memory buffer (MemTable), then flushed to immutable sorted files (SSTables) on disk.
- Background **compaction** merges SSTables, removes deleted entries, sorts by key.
- **Reads**: may scan multiple SSTables across levels until the key is found. Bloom filters prune most level checks.
- **Write amplification**: high — data written once per level during compaction. 10–30x is common.
- **Read amplification**: higher than B-tree for random point reads (multiple SSTable checks). Range reads can be fast if data is hot in cache.
- **Space amplification**: moderate — deleted data lingers until compaction.
- Optimal for: write-heavy workloads (logs, time-series, event streams), where throughput > latency.

**Compaction strategies (LSM):**

| Strategy       | Write amp | Space amp | Read amp | Use case              |
|----------------|-----------|-----------|----------|-----------------------|
| Size-tiered    | low       | high      | high     | write-heavy ingest    |
| Leveled (LCS)  | high      | low       | low      | read/write balance    |
| TWCS           | low       | low*      | medium   | time-series (TTL data)|

### Query planning and EXPLAIN

The query planner converts SQL into a tree of physical operators. Key operators:

- **Seq Scan** — full table scan. O(N). Chosen when selectivity is low (> ~10% of rows) or no index exists.
- **Index Scan** — traverse B-tree, fetch heap rows. Random I/O per row. Good for high-selectivity queries.
- **Index Only Scan** — all needed columns in the index (covering index). No heap fetch.
- **Bitmap Index Scan** — for moderate selectivity: collect all matching TIDs into a bitmap, then sort and do sequential heap reads.

**Join strategies:**

| Strategy     | Complexity       | When chosen                                        |
|--------------|------------------|---------------------------------------------------|
| Nested Loop  | O(N × M)         | small inner table or index on join key            |
| Hash Join    | O(N + M)         | large tables, equality join, no useful index      |
| Merge Join   | O(N log N + M log M) | both sides pre-sorted (index or ORDER BY)    |

**Cost model**: the planner estimates cost using `pg_statistic` (column histograms, distinct counts, null fraction). Stale statistics → bad plan. Run `ANALYZE` after bulk loads.

**Reading EXPLAIN ANALYZE output** (PostgreSQL):
```
-> Hash Join  (cost=184.00..512.00 rows=1500 width=64)
              (actual time=2.1..8.3 rows=1423 loops=1)
     Hash Cond: (orders.customer_id = customers.id)
     -> Seq Scan on orders  (cost=0..300 rows=20000 ...)
     -> Hash  (cost=100.00..100.00 rows=6720 ...)
          -> Index Scan on customers  ...
```

`cost=estimated_startup..estimated_total`, then `actual time=actual_startup..actual_total`. When `rows` estimate diverges greatly from `actual rows`, the planner is flying blind — add index, run ANALYZE, or add statistics target.

### Isolation levels and MVCC

SQL standard defines four isolation levels (weakest to strongest):

| Level            | Dirty Read | Non-repeatable Read | Phantom Read | Serialization Anomaly |
|------------------|-----------|--------------------|--------------|-----------------------|
| Read Uncommitted | possible  | possible           | possible     | possible              |
| Read Committed   | prevented | possible           | possible     | possible              |
| Repeatable Read  | prevented | prevented          | possible*    | possible              |
| Serializable     | prevented | prevented          | prevented    | prevented             |

*PostgreSQL's Repeatable Read also prevents phantoms due to MVCC.

**MVCC (Multi-Version Concurrency Control)** — PostgreSQL's implementation:

Every row has `xmin` (transaction that created it) and `xmax` (transaction that deleted it, or 0). A reader sees a row only if `xmin` is committed and `xmax` is 0 or not yet committed at snapshot time.

This means:
- **Readers never block writers, writers never block readers.**
- Snapshot isolation is "free" — each transaction gets a consistent snapshot at start.
- Downside: old row versions accumulate until `VACUUM` reclaims them (`dead tuples`).
- `SELECT FOR UPDATE` acquires a row-level lock and prevents concurrent updates.

**Serializable Snapshot Isolation (SSI)** — PostgreSQL's Serializable level. Tracks read/write dependencies between transactions and aborts the cycle-forming transaction. No reader/writer blocking but higher abort rate under contention.

### Sharding

**Hash sharding:**
- `shard = hash(key) % N`
- Uniform distribution; no hot spots.
- Resharding when N changes: most keys move (consistent hashing reduces this — only K/N keys move when adding one node).

**Range sharding:**
- `shard = lookup(key_range_table)`
- Supports efficient range scans that stay within one shard.
- Risk: monotonically increasing keys (timestamps, auto-increment IDs) create write hot spots on the last shard.

**Directory sharding:**
- Explicit map: `key → shard_id` in a routing table.
- Most flexible; can migrate individual keys.
- Routing table becomes a bottleneck; often cached aggressively.

**Cross-shard queries**: require scatter-gather (fan-out to all shards, merge results). Avoid or route through a query coordinator.

### Replication

**Leader/follower (async):**
- Leader accepts all writes. Followers replicate the WAL/binlog.
- Asynchronous: leader commits before followers confirm. **Lag** is the gap between leader and follower state.
- Risk: failover can lose committed transactions (uncommitted in follower).
- Good for: read scaling (route reads to followers), geographic distribution.

**Synchronous replication:**
- Leader waits for at least one follower to confirm before acknowledging commit.
- No data loss on failover (single follower can fail; leader waits for at least one sync follower).
- Write latency += network RTT to nearest sync follower.

**Read replicas**: serve read traffic. Stale by replication lag (milliseconds to seconds). Applications that need read-your-writes must route to leader or wait for replica to catch up.

**Replication failure modes:**
- Lagging replica serves stale data (invisible unless lag monitored).
- Split-brain: two nodes both think they are leader → diverged state. Prevented by requiring a quorum (majority) for leadership.

### CAP theorem + PACELC

**CAP**: a distributed system can guarantee at most two of:
- **Consistency** — every read sees the most recent write.
- **Availability** — every request gets a (non-error) response.
- **Partition tolerance** — system continues operating when network partitions occur.

Since partitions happen in real networks, the practical choice is **CP** (sacrifice availability during partition) or **AP** (sacrifice consistency, serve possibly-stale data).

Examples:
- PostgreSQL with synchronous replication → CP (blocks writes if sync follower unreachable).
- Cassandra (quorum reads/writes tunable) → AP by default.
- ZooKeeper, etcd → CP (leader election; refuses reads from followers during partition).

**PACELC** extends CAP to the non-partition case (the common case):

> If **P**artition: **A**vailability vs **C**onsistency.  
> **E**lse (no partition): **L**atency vs **C**onsistency.

Most systems trade latency for consistency even without partitions (e.g., async replication is faster but can serve stale reads). Cassandra is PA/EL: available during partition, lower latency by default. PostgreSQL synchronous replication is PC/EC: consistent during partition, higher latency.

---

## Worked example

```python
"""
week-08/day-5 examples
- B-tree vs LSM-tree write path simulation
- EXPLAIN output parser (extracts key metrics for monitoring)
- MVCC snapshot isolation simulation
- Shard router with hash and range strategies
"""

from __future__ import annotations

import hashlib
import logging
import time
from bisect import bisect_right
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Storage engine write path — B-tree vs LSM-tree (conceptual simulation)
# ---------------------------------------------------------------------------


class StorageEngine(str, Enum):
    BTREE = "btree"
    LSM = "lsm"


@dataclass
class WriteResult:
    engine: StorageEngine
    key: str
    logical_writes: int  # application-level
    physical_writes: int  # actual I/O amplification
    write_amp_ratio: float


def simulate_write(engine: StorageEngine, key: str, value: str) -> WriteResult:
    """
    Simulates write amplification for each storage engine.
    B-tree: WAL write + page update = 2-3x amplification.
    LSM: MemTable flush + N compaction levels = 10-30x amplification.
    These are approximations based on well-known benchmarks (RocksDB docs).
    """
    if engine == StorageEngine.BTREE:
        # WAL append (sequential) + B-tree page update (random)
        physical = 2 + 1  # WAL + page + possible page split
        return WriteResult(
            engine=engine,
            key=key,
            logical_writes=1,
            physical_writes=physical,
            write_amp_ratio=physical,
        )
    else:  # LSM
        # MemTable (1) + L0 flush (1) + L1 compaction (10x) + L2 (10x) ...
        # Simplified: 10-30x depending on level depth
        levels = 3
        physical = 1 + sum(10**i for i in range(levels))
        write_amp = physical
        return WriteResult(
            engine=engine,
            key=key,
            logical_writes=1,
            physical_writes=physical,
            write_amp_ratio=write_amp,
        )


# ---------------------------------------------------------------------------
# 2. EXPLAIN ANALYZE output parser
# ---------------------------------------------------------------------------


@dataclass
class PlanNode:
    node_type: str
    estimated_rows: int
    actual_rows: int
    actual_time_ms: float
    row_estimation_error: float  # > 10x = stale stats warning


def parse_explain_line(line: str) -> PlanNode | None:
    """
    Parses a single EXPLAIN ANALYZE output line.
    Real implementation would use EXPLAIN (FORMAT JSON) for structured output.
    This demonstrates the metrics to watch.
    """
    import re

    # Match pattern: -> NodeType  (cost=..rows=N..) (actual time=X..Y rows=A loops=L)
    pattern = r"->\s+([\w\s]+?)\s+\(cost=[\d.]+\.\.[\d.]+\s+rows=(\d+)\s+.*?\)\s+\(actual time=([\d.]+)\.\.([\d.]+)\s+rows=(\d+)"
    match = re.search(pattern, line)
    if not match:
        return None

    node_type = match.group(1).strip()
    estimated_rows = int(match.group(2))
    actual_rows = int(match.group(5))
    actual_time = float(match.group(4))

    # Row estimation error: bad estimates mislead the planner
    if estimated_rows > 0:
        error = max(estimated_rows, actual_rows) / min(estimated_rows, actual_rows)
    else:
        error = float("inf")

    return PlanNode(
        node_type=node_type,
        estimated_rows=estimated_rows,
        actual_rows=actual_rows,
        actual_time_ms=actual_time,
        row_estimation_error=error,
    )


def analyze_query_plan(explain_output: str) -> list[str]:
    """Returns human-readable warnings for common query plan problems."""
    warnings: list[str] = []
    for line in explain_output.splitlines():
        node = parse_explain_line(line)
        if node is None:
            continue
        if node.row_estimation_error > 10:
            warnings.append(
                f"Poor row estimate on '{node.node_type}': "
                f"estimated {node.estimated_rows}, got {node.actual_rows} "
                f"({node.row_estimation_error:.0f}x off) — run ANALYZE"
            )
        if "Seq Scan" in node.node_type and node.actual_rows > 100_000:
            warnings.append(
                f"Seq Scan on large table ({node.actual_rows:,} rows) — "
                "consider adding an index on the filter column"
            )
    return warnings


# ---------------------------------------------------------------------------
# 3. MVCC snapshot isolation — minimal simulation
# ---------------------------------------------------------------------------


@dataclass
class RowVersion:
    xmin: int  # transaction that created this version
    xmax: int  # transaction that deleted this version (0 = still live)
    data: dict[str, Any]


@dataclass
class MVCCTable:
    """
    Each row can have multiple versions. A transaction sees the version
    where xmin <= snapshot_txid and (xmax == 0 or xmax > snapshot_txid).
    """

    _versions: dict[str, list[RowVersion]] = field(default_factory=dict)
    _current_txid: int = 0

    def begin(self) -> int:
        """Returns a new transaction ID (snapshot point)."""
        self._current_txid += 1
        return self._current_txid

    def insert(self, txid: int, key: str, data: dict[str, Any]) -> None:
        version = RowVersion(xmin=txid, xmax=0, data=data)
        self._versions.setdefault(key, []).append(version)

    def update(self, txid: int, key: str, new_data: dict[str, Any]) -> None:
        # Mark current visible version as deleted, insert new version
        for v in self._versions.get(key, []):
            if v.xmax == 0 and v.xmin <= txid:
                v.xmax = txid  # logically deleted by this transaction
        self.insert(txid, key, new_data)

    def read(self, snapshot_txid: int, key: str) -> dict[str, Any] | None:
        """
        Returns the row version visible at snapshot_txid.
        MVCC invariant: xmin committed and <= snapshot_txid,
                         xmax == 0 or xmax > snapshot_txid.
        Simplified: treats any xmin <= snapshot as committed.
        """
        for v in reversed(self._versions.get(key, [])):
            if v.xmin <= snapshot_txid and (v.xmax == 0 or v.xmax > snapshot_txid):
                return v.data
        return None


def demo_mvcc() -> None:
    table = MVCCTable()

    tx1 = table.begin()
    table.insert(tx1, "user:1", {"name": "Alice", "balance": 100})

    # tx2 starts after tx1 inserts — sees the row
    tx2 = table.begin()
    val = table.read(tx2, "user:1")
    logger.info("tx2 reads: %s", val)  # Alice, balance 100

    # tx3 updates — tx2's snapshot still sees old version
    tx3 = table.begin()
    table.update(tx3, "user:1", {"name": "Alice", "balance": 150})

    val_before = table.read(tx2, "user:1")
    val_after = table.read(tx3, "user:1")
    logger.info("tx2 (snapshot) sees: %s", val_before)  # balance 100
    logger.info("tx3 (own write) sees: %s", val_after)   # balance 150
    assert val_before is not None and val_before["balance"] == 100
    assert val_after is not None and val_after["balance"] == 150


# ---------------------------------------------------------------------------
# 4. Shard router — hash and range strategies
# ---------------------------------------------------------------------------


class ShardRouter:
    """Routes a key to a shard_id using hash or range strategy."""

    def __init__(self, shard_count: int) -> None:
        self._shard_count = shard_count

    def hash_shard(self, key: str) -> int:
        """
        Consistent hash using SHA-256 mod N.
        Note: NOT consistent hashing (ring); just simple modulo.
        Consistent hashing would reduce key movement on scale-out.
        """
        digest = int(hashlib.sha256(key.encode()).hexdigest(), 16)
        return digest % self._shard_count

    def range_shard(self, key: int, boundaries: list[int]) -> int:
        """
        Range-based sharding.
        boundaries = [1000, 2000, 3000] → shards [0-999, 1000-1999, 2000-2999, 3000+]
        bisect_right returns the index to insert key to keep boundaries sorted,
        which equals the shard number.
        """
        return bisect_right(boundaries, key)


def demo_shard_distribution() -> None:
    router = ShardRouter(shard_count=4)

    # Simulate 10_000 keys — expect roughly uniform distribution with hash
    from collections import Counter

    counts: Counter[int] = Counter()
    for i in range(10_000):
        shard = router.hash_shard(f"user:{i}")
        counts[shard] += 1

    for shard_id, count in sorted(counts.items()):
        logger.info("Hash shard %d: %d keys (%.1f%%)", shard_id, count, count / 100)

    # Range sharding example — time-series by year
    boundaries = [2020, 2022, 2024]  # splits: ..2020, 2021-2022, 2023-2024, 2025+
    for year in [2019, 2021, 2023, 2025]:
        shard = router.range_shard(year, boundaries)
        logger.info("Year %d → shard %d", year, shard)


if __name__ == "__main__":
    # Write amplification comparison
    btree_result = simulate_write(StorageEngine.BTREE, "user:1", "Alice")
    lsm_result = simulate_write(StorageEngine.LSM, "user:1", "Alice")
    logger.info("B-tree write amp: %.1fx", btree_result.write_amp_ratio)
    logger.info("LSM write amp: %.1fx", lsm_result.write_amp_ratio)

    # MVCC demo
    demo_mvcc()

    # Shard distribution
    demo_shard_distribution()
```

---

## Exercises

1. **Index selection**: Given a table `events(id, user_id, event_type, created_at)` and these three queries, decide which index (or composite index) to add for each, and explain why:
   - `SELECT * FROM events WHERE user_id = $1 ORDER BY created_at DESC LIMIT 20`
   - `SELECT COUNT(*) FROM events WHERE event_type = $1 AND created_at > now() - interval '7 days'`
   - `SELECT * FROM events WHERE id = $1`

2. **EXPLAIN analysis**: Run `EXPLAIN (ANALYZE, BUFFERS) <query>` in PostgreSQL for a query on a table with > 100K rows. Identify: which scan type was chosen, whether it matches your expectation, and the row estimation error. If the estimate is off by > 5x, run `ANALYZE events` and re-check.

3. **Isolation levels**: Write two concurrent Python threads (or asyncio tasks) that demonstrate a non-repeatable read: thread A reads a value, thread B updates and commits it, thread A reads again — values differ. Then switch the connection to `REPEATABLE READ` and observe that the second read returns the original value.

4. **Sharding math**: Implement consistent hashing (a hash ring with virtual nodes) for a `ShardRouter`. Add a method `add_node(node_id)` that shows how many keys need to move vs modulo hashing when going from 4 to 5 shards.

(Solutions: ../solutions/week-08/day-5.md)

---

## Self-check

- [ ] I can explain why LSM-tree has higher write amplification than B-tree, and when to prefer each.
- [ ] I can read EXPLAIN ANALYZE output and identify a bad row estimate and its fix.
- [ ] I can describe how MVCC allows readers and writers to not block each other.
- [ ] I can explain the difference between async and sync replication and the failure modes of each.
- [ ] I can place PostgreSQL, Cassandra, and etcd on the PACELC grid and justify the placement.

---

## Supplementary reading

- [The Design and Implementation of Modern Column-Oriented Database Systems (paper summary)](https://www.cs.umd.edu/~abadi/papers/abadi-column-stores.pdf)
- [PostgreSQL MVCC documentation](https://www.postgresql.org/docs/current/mvcc-intro.html)
- [PACELC theorem (Daniel Abadi)](https://dbmsmusings.blogspot.com/2010/04/problems-with-cap-and-yahoos-little.html)
