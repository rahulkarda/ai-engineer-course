# Phase 1 · Week 06 · Day 1 — SQL Queries, JOINs, and Aggregations

> Objective: Write fluent SELECT queries including multi-table JOINs and aggregate functions  ·  Time: ~1 hr  ·  Prereqs: [[week-05/day-7]]

## Concept

A relational database stores data in **tables** (relations): rows and columns. SQL is a declarative language — you describe *what* you want, not *how* to fetch it. PostgreSQL's query planner figures out the optimal execution path.

### Query execution order (logical, not syntactic)

```
FROM → JOIN → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT
```

This matters: you cannot reference a `SELECT`-alias in a `WHERE` clause because `WHERE` is evaluated before `SELECT`.

### Core DML statements

| Statement | Purpose |
|-----------|---------|
| `SELECT` | Read rows |
| `INSERT` | Add rows |
| `UPDATE` | Modify rows |
| `DELETE` | Remove rows |

## How it works internally

PostgreSQL parses the query into a **parse tree**, rewrites it (e.g. view expansion), then the **planner/optimizer** generates candidate execution plans and picks the cheapest one based on statistics (table size, column cardinality stored in `pg_statistics`). The **executor** runs the chosen plan node-by-node.

JOINs are implemented via three physical algorithms:
- **Nested Loop Join** — for small outer tables or indexed inner tables
- **Hash Join** — build a hash table on the smaller side, probe with the larger; good for large unsorted inputs
- **Merge Join** — requires both sides sorted; efficient when indexes provide sorted order

Run `EXPLAIN ANALYZE <query>` to see which algorithm PostgreSQL chose and actual run times.

## Worked example

```sql
-- Schema used throughout Week 6
-- customers, accounts, transactions, transaction_categories, audit_logs

-- 1. Basic SELECT with filtering and sorting
SELECT
    c.customer_id,
    c.full_name,
    c.email,
    c.created_at::date AS joined_date
FROM customers c
WHERE c.is_active = true
  AND c.created_at >= '2024-01-01'
ORDER BY c.created_at DESC
LIMIT 10;

-- 2. INSERT a new customer
INSERT INTO customers (full_name, email, phone, is_active)
VALUES ('Alice Müller', 'alice@example.com', '+49123456789', true)
RETURNING customer_id, created_at;

-- 3. UPDATE email
UPDATE customers
SET email = 'alice.mueller@example.com',
    updated_at = NOW()
WHERE customer_id = 42;

-- 4. DELETE (soft-delete pattern preferred; shown for completeness)
DELETE FROM customers
WHERE customer_id = 99
  AND is_active = false;

-- 5. INNER JOIN — customers with their accounts
SELECT
    c.full_name,
    a.account_number,
    a.account_type,
    a.balance
FROM customers c
INNER JOIN accounts a ON a.customer_id = c.customer_id
ORDER BY c.full_name;

-- 6. LEFT JOIN — all customers, even those with no accounts
SELECT
    c.full_name,
    COUNT(a.account_id) AS account_count
FROM customers c
LEFT JOIN accounts a ON a.customer_id = c.customer_id
GROUP BY c.customer_id, c.full_name
ORDER BY account_count DESC;

-- 7. Multi-table JOIN — transactions with category names
SELECT
    t.transaction_id,
    t.amount,
    t.direction,        -- 'credit' or 'debit'
    tc.name AS category,
    a.account_number,
    c.full_name AS customer
FROM transactions t
JOIN accounts a          ON a.account_id = t.account_id
JOIN customers c         ON c.customer_id = a.customer_id
LEFT JOIN transaction_categories tc ON tc.category_id = t.category_id
WHERE t.created_at >= NOW() - INTERVAL '30 days'
ORDER BY t.created_at DESC;

-- 8. Aggregations
SELECT
    tc.name                    AS category,
    COUNT(*)                   AS tx_count,
    SUM(t.amount)              AS total_amount,
    AVG(t.amount)::numeric(12,2) AS avg_amount,
    MAX(t.amount)              AS max_tx,
    MIN(t.amount)              AS min_tx
FROM transactions t
LEFT JOIN transaction_categories tc ON tc.category_id = t.category_id
WHERE t.direction = 'debit'
GROUP BY tc.name
HAVING COUNT(*) > 5
ORDER BY total_amount DESC;
```

### Reading `EXPLAIN ANALYZE` output

```sql
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT c.full_name, SUM(t.amount)
FROM customers c
JOIN accounts a ON a.customer_id = c.customer_id
JOIN transactions t ON t.account_id = a.account_id
GROUP BY c.full_name;
```

Key fields to read:
- **cost=A..B** — estimated startup..total cost (arbitrary planner units)
- **actual time=X..Y** — real ms (startup..total per loop)
- **rows=N** — actual rows returned
- **Buffers: shared hit=N read=M** — cache hits vs. disk reads

## Exercises

1. Write a query that returns the top 5 customers by total transaction volume (sum of all transaction amounts across all their accounts). Columns: `full_name`, `email`, `total_volume`.

2. Find all accounts where the balance is below $100 AND the account type is `'checking'`. Return `account_number`, `balance`, and the customer's `full_name`.

3. Write an `INSERT` that adds a new account for `customer_id = 1`, with type `'savings'`, initial balance `500.00`, and a generated `account_number` of the form `'SAV-' || nextval('account_seq')`. Use `RETURNING` to get the new `account_id`.

4. Update all transactions older than 1 year that have no category assigned (`category_id IS NULL`) — set `category_id = (SELECT category_id FROM transaction_categories WHERE name = 'Uncategorized')`.

5. Run `EXPLAIN ANALYZE` on the query from exercise 1. Identify the JOIN algorithm used and whether any sequential scans appear.

(Solutions: ../../solutions/week-06/day-1.md)

## Self-check

- [ ] I can explain the logical order of SQL clause evaluation
- [ ] I can write INNER, LEFT, and multi-table JOINs without reference
- [ ] I understand the difference between WHERE and HAVING
- [ ] I can read basic EXPLAIN ANALYZE output to identify sequential scans and join types

## Supplementary links

- [PostgreSQL Tutorial — Basic Queries](https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-select/)
- [Use The Index, Luke — Execution Plans](https://use-the-index-luke.com/sql/explain-plan)
- [PostgreSQL Docs — EXPLAIN](https://www.postgresql.org/docs/current/sql-explain.html)
