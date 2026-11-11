# Solutions — Week 06 · Day 7

## Review answers

### SQL fundamentals

**Logical execution order:**
FROM → (JOINs) → WHERE → GROUP BY → HAVING → SELECT → DISTINCT → ORDER BY → LIMIT/OFFSET

This matters because: aliases defined in SELECT are not available in WHERE or HAVING (those clauses run before SELECT). Aggregate functions in HAVING can reference grouped columns, but WHERE cannot reference aggregates.

**WHERE vs HAVING:** WHERE filters individual rows before grouping; HAVING filters groups after GROUP BY. Use WHERE when possible — it reduces the row count before the aggregation, which is faster.

**When PostgreSQL skips an index:** if the table is small (few pages), a sequential scan is faster because an index lookup requires multiple I/Os (index page + heap page). Also skipped for low-selectivity predicates (e.g., `WHERE is_active = true` when 95% of rows are active). The planner uses `pg_statistics` to estimate row counts and choose.

**EXPLAIN (ANALYZE, BUFFERS) vs EXPLAIN:** plain EXPLAIN shows estimated costs only. `ANALYZE` actually executes the query and shows real row counts and timings. `BUFFERS` shows cache hits vs. disk reads — essential for diagnosing I/O bottlenecks.

### Joins and aggregations

**INNER vs LEFT vs FULL OUTER:**
- INNER: only rows with matching keys in both tables
- LEFT: all rows from left table; NULLs on right side where no match
- FULL OUTER: all rows from both tables; NULLs on either side where no match

Financial app scenario:
- INNER JOIN customers → accounts: list only customers who have accounts
- LEFT JOIN customers → accounts: list all customers, show NULL for those without accounts
- FULL OUTER: rarely used; example: reconcile two transaction sets to find records in only one

**NULL in GROUP BY:** PostgreSQL groups all NULLs together into a single bucket. `GROUP BY category_id` with multiple NULL `category_id` values produces one group for NULL.

**Window functions vs GROUP BY:** GROUP BY collapses multiple rows into one row per group; window functions compute a value for each row based on a window of related rows, leaving the row count unchanged.

### CTEs and subqueries

**MATERIALIZED vs inlined:** since PG 12, the optimizer may inline a non-recursive CTE as if it were a subquery, allowing predicate pushdown. `WITH cte AS MATERIALIZED (...)` forces evaluation once and stores the result. Use MATERIALIZED when the CTE has side effects or when you want to prevent the planner from expanding it inline (e.g., it contains a volatile function you don't want called multiple times).

**Recursive CTE termination:** the recursive part (`UNION ALL` branch) must eventually produce zero rows. This happens when the JOIN condition finds no more matching children. Always add a depth guard (`WHERE depth < N`) to handle accidental cycles.

### Constraints and normalization

**Referential integrity:** every FK value must exist in the referenced table's PK. Enforced by `FOREIGN KEY ... REFERENCES`. PostgreSQL checks on INSERT/UPDATE (new FK value must exist) and on DELETE/UPDATE of the referenced row (action depends on `ON DELETE` clause).

**1NF → 2NF → 3NF example (orders):**
1NF: atomic columns, no repeating groups.
2NF: remove partial dependencies — `product_name` depends on `product_id` alone, not on the full composite key `(order_id, product_id)`.
3NF: remove transitive dependencies — `customer_email` → `customer_name` is a non-key → non-key dependency.
Result: customers, products, orders, order_items tables (see Day 3 solution).

**Covering index / INCLUDE:** a covering index stores non-key columns in the leaf pages so PostgreSQL can answer a query entirely from the index without fetching heap pages (`Index Only Scan`, `Heap Fetches: 0`). `INCLUDE` columns are stored but not part of the sort key — you can't filter on them in the index condition; they're only there to avoid the heap fetch.

### Transactions and ACID

**Isolation level preventing phantoms:** `REPEATABLE READ` in PostgreSQL also prevents phantom reads (stricter than the SQL standard, which only requires SERIALIZABLE to prevent phantoms). `SERIALIZABLE` prevents all anomalies including serialization anomalies.

**MVCC:** each row version has `xmin` (creating transaction) and `xmax` (deleting transaction). Readers see only versions where `xmin` committed before the reader's snapshot and `xmax` not yet committed. Writers create new versions; they don't overwrite old ones. So a reader can proceed while a writer adds a new version — no lock conflict.

**Serialization failure:** PostgreSQL detects that the concurrent execution would produce a result inconsistent with any serial order. It raises `ERROR: could not serialize access due to concurrent update` (error code `40001`). The application must catch this and **retry the entire transaction** from the beginning — not just the failed statement.

### SQLAlchemy

**Identity map:** a dict inside the Session keyed by `(class, primary_key)`. When you call `session.get(Customer, 1)` twice, the second call returns the cached instance without hitting the database. Stale: after another connection commits a change to the same row, your session's cached version is outdated. Solutions: `session.refresh(obj)`, or use `expire_on_commit=True` (default) which expires all attributes after every commit.

**flush vs commit:** `flush()` writes pending changes to the database (making them visible to the current transaction via SQL) but does not commit — other connections cannot see them yet, and the changes are still rollback-able. `commit()` makes the transaction permanent, releases locks, and makes changes visible to all connections.

**pool_pre_ping=True:** before handing a connection from the pool to your code, SQLAlchemy issues a cheap test query (e.g., `SELECT 1`). If the connection is dead (server restarted, network timeout), it's discarded and a new one is created. Without this, a stale connection causes an error on the first real query.
