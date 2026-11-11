# Solutions — Week 08 · Day 5

## Exercise 1

**For each workload, choose B-tree or LSM and justify.**

### (a) Point-of-sale reads (frequent random reads)
**Choice: B-tree**

Justification: B-trees store data in sorted page order and support O(log n) point lookups with predictable I/O — typically 3-4 page reads for a billion-row table. POS systems need low-latency random reads (customer lookup, inventory check). B-trees have no read amplification from compaction. PostgreSQL, MySQL InnoDB, and SQLite all use B-tree primary indexes by default.

### (b) IoT sensor write-heavy time-series
**Choice: LSM**

Justification: LSM trees batch writes in memory (memtable) and flush sequentially to disk — sequential I/O is 10-100x faster than random I/O on spinning disk and still faster on SSDs. Frequent small random writes on a B-tree cause page splits and fragmentation. LSM write throughput scales linearly with hardware. Used by InfluxDB, Cassandra, RocksDB, ScyllaDB for exactly this pattern.

### (c) Banking ledger (read-heavy with occasional writes)
**Choice: B-tree**

Justification: Ledger reads (balance queries, transaction history, audits) dominate. B-tree read performance is optimal. The occasional writes (transactions) are infrequent enough that write amplification is not a bottleneck. ACID guarantees and point-in-time consistency are critical for banking — B-tree databases (PostgreSQL) have mature transactional semantics. LSM read amplification (scanning multiple SSTables) is undesirable here.

### (d) Analytics event store (write-once, bulk reads)
**Choice: LSM (column-store variant) or specialized OLAP**

Justification: Events are append-only (no updates) — exactly the LSM sweet spot. Bulk reads for analytics are typically full-range scans or aggregations, not point lookups, so B-tree's random-access advantage is irrelevant. Column-oriented LSM stores (Parquet-backed, ClickHouse, Apache Cassandra wide rows) allow compression and vectorized scans. Pure LSM (RocksDB) is also fine for the write-once constraint.

---

## Exercise 2

**EXPLAIN analysis and index strategy for a 10M-row orders table.**

### Query
```sql
SELECT * FROM orders
WHERE user_id = 42
  AND status = 'PENDING'
ORDER BY created_at DESC
LIMIT 20;
```

### Simulated EXPLAIN output (PostgreSQL)

```
QUERY PLAN
---------------------------------------------------------------------------
 Limit  (cost=0.00..45210.00 rows=20 width=256)
         (actual time=12450.3..12451.1 rows=20 loops=1)
   ->  Sort  (cost=0.00..89200.00 rows=35000 width=256)
             (actual time=12450.2..12450.3 rows=20 loops=1)
         Sort Key: created_at DESC
         Sort Method: external merge  Disk: 2800kB
         ->  Seq Scan on orders  (cost=0.00..220000.00 rows=35000 width=256)
                                  (actual time=0.05..12430.0 rows=35000 loops=1)
               Filter: ((user_id = 42) AND (status = 'PENDING'::text))
               Rows Removed by Filter: 9965000
 Planning Time: 0.8 ms
 Execution Time: 12451.5 ms
```

### Analysis of the bad plan

| Symptom | Cause |
|---|---|
| `Seq Scan` on 10M rows | No usable index on `user_id` or `status` |
| `external merge Disk` | Sort spill to disk — can't fit 35k rows in `work_mem` |
| 12 seconds execution | Full table scan + sort on unindexed columns |

### Recommended index strategy

**Option A: Composite index (best for this query)**

```sql
-- Covers the filter + sort in one index scan
CREATE INDEX CONCURRENTLY idx_orders_user_status_created
    ON orders (user_id, status, created_at DESC);
```

Resulting plan after index creation:

```
QUERY PLAN
---------------------------------------------------------------------------
 Limit  (cost=0.00..4.20 rows=20 width=256)
         (actual time=0.18..0.22 rows=20 loops=1)
   ->  Index Scan using idx_orders_user_status_created on orders
                         (cost=0.56..420.00 rows=2000 width=256)
                         (actual time=0.17..0.21 rows=20 loops=1)
         Index Cond: ((user_id = 42) AND (status = 'PENDING'))
         Scan direction: Forward
 Planning Time: 0.4 ms
 Execution Time: 0.3 ms
```

Down from 12 451 ms to 0.3 ms. The planner satisfies `ORDER BY created_at DESC LIMIT 20` by reading only 20 rows from the index in the correct order — no sort step.

**Why this column order:**
- `user_id` first — highest selectivity equality predicate.
- `status` second — further filters down to pending rows.
- `created_at DESC` last — allows the planner to traverse the index in sorted order and short-circuit at LIMIT 20.

**Option B: Partial index (smaller, if status set is stable)**

```sql
CREATE INDEX CONCURRENTLY idx_orders_pending_user_created
    ON orders (user_id, created_at DESC)
    WHERE status = 'PENDING';
```

Smaller index footprint (only indexes pending rows), same query speed. Trade-off: the partial index only helps queries that hard-code `status = 'PENDING'`; the composite index (Option A) handles any status value.

**General rules applied:**
1. Equality predicates before range/sort columns in composite index.
2. `CONCURRENTLY` avoids table lock on production.
3. Avoid `SELECT *` in production — project only needed columns to reduce index page reads if a covering index is used.

---

## Exercise 3

**Hot-user shard problem and two solutions.**

### Setup

```
Sharding key: hash(user_id) % N_SHARDS

Shard 0: users 0, N, 2N, ...
Shard 1: users 1, N+1, 2N+1, ...
...
```

### What happens during a hot-user shard problem

A celebrity or viral account (say `user_id=1001`) generates 10 million post reads/writes per hour. `hash(1001) % 8 = 5` — all traffic lands on Shard 5.

```
Shard 0: 1,200 req/s  [OK]
Shard 1: 1,100 req/s  [OK]
Shard 2: 1,050 req/s  [OK]
Shard 3:   980 req/s  [OK]
Shard 4: 1,150 req/s  [OK]
Shard 5: 87,000 req/s [CPU 100%, latency 8s, OOM, cascading failures]
Shard 6: 1,000 req/s  [OK]
Shard 7: 1,080 req/s  [OK]
```

Shard 5 becomes a bottleneck: high CPU, lock contention, write queue backlog, queries time out. Other users whose posts happen to hash to Shard 5 also experience degraded performance. This is the "hot shard" or "hot partition" problem.

### Solution 1: Read replicas + read-your-writes routing

Add N read replicas per shard. Route read traffic for hot users to replicas; route writes to the primary.

```
write(user_id=1001)  -->  Shard 5 PRIMARY
read(user_id=1001)   -->  Shard 5 REPLICA-1
                     -->  Shard 5 REPLICA-2
                     -->  Shard 5 REPLICA-3  (round-robin)
```

**Trade-off:** Replication lag means reads may return slightly stale data. Implement read-your-writes consistency by routing a user's own reads to the primary (or a replica that has caught up past the write LSN) for a short window after each write.

This scales read capacity linearly with replica count at the cost of replication lag and extra hardware.

### Solution 2: Shard splitting (logical re-shard) + consistent hashing

Split Shard 5 into two sub-shards (5a, 5b) using range or secondary hash:

```python
def route_shard(user_id: int, post_id: int) -> str:
    base_shard = hash(user_id) % NUM_SHARDS          # -> 5
    if is_hot_user(user_id):
        # Further split by post_id to distribute within the hot user
        sub = hash(post_id) % HOT_USER_SPLIT_FACTOR  # -> 0 or 1
        return f"shard-5-{sub}"
    return f"shard-{base_shard}"
```

Or use **consistent hashing with virtual nodes**: assign hot users more virtual-node slots spread across physical shards, so their data is distributed across multiple shards.

**Trade-off:** Queries that need all posts by a hot user must scatter-gather across sub-shards. Fan-out reads become more complex. Better for write-heavy hot users; read-heavy hot users benefit more from Solution 1.

**In practice**, both solutions are combined: read replicas for read scaling, shard splitting for write scaling.

---

## Exercise 4

**CAP profiles for three systems.**

### Background: CAP Theorem

A distributed system can guarantee at most two of:
- **C**onsistency — every read returns the most recent write (or an error)
- **A**vailability — every request receives a response (not necessarily the latest data)
- **P**artition tolerance — the system continues operating despite network partitions

Because network partitions are unavoidable in distributed systems, the real trade-off is **CP vs AP** during a partition.

---

### (a) Cassandra
**Profile: AP (Availability + Partition tolerance)**

Default configuration uses eventual consistency (`consistency_level = ONE` or `QUORUM` with RF=3 and reads from one replica). During a network partition:
- Cassandra continues accepting reads and writes on both sides of the partition.
- Writes are reconciled after partition heals via last-write-wins (LWW) or read-repair.
- A node that missed writes during the partition may return stale data.

**Trade-off made:** Cassandra chose availability over consistency. Writes never block on partition (no 2PC). This makes it suitable for time-series, messaging, and recommendation data where occasional staleness is acceptable. You can tune toward CP with `consistency_level = ALL`, but that sacrifices availability.

---

### (b) PostgreSQL with synchronous replication
**Profile: CP (Consistency + Partition tolerance)**

With `synchronous_commit = on` and `synchronous_standby_names` set, a write is only acknowledged after at least one standby has flushed it to WAL. During a network partition that isolates the standby:
- The primary blocks or returns an error on writes (cannot confirm sync replication).
- Reads from the primary are still consistent (no stale data).
- The system sacrifices availability to guarantee that no committed transaction is lost.

**Trade-off made:** PostgreSQL chose consistency over availability. A partition that severs the sync standby causes write stalls. Suitable for financial data, inventory, and any use case where split-brain and lost updates are unacceptable.

---

### (c) DynamoDB (eventual consistency mode)
**Profile: AP (Availability + Partition tolerance)**

With `ConsistentRead=False` (the default), DynamoDB routes reads to any replica. During a partition:
- Reads may return data from a replica that hasn't yet received the latest write.
- Writes are accepted immediately (multi-AZ replication is async).
- No read will block or fail due to replication lag.

**Trade-off made:** DynamoDB eventual consistency mode prioritizes low-latency, always-available reads over strict consistency. Suitable for shopping carts, session data, and leaderboards where slightly stale reads are acceptable. DynamoDB also supports `ConsistentRead=True` for CP behavior (reads from the leader replica), at higher latency cost and reduced availability.
