# Phase 1 · Week 06 · Day 2 — CTEs, Subqueries, and Window Functions

> Objective: Use CTEs, subqueries, and window functions to express complex analytical queries  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

## Concept

### Subqueries

A **subquery** (inner query) is a `SELECT` nested inside another statement. Three placement forms:

| Placement | Used in | Returns |
|-----------|---------|---------|
| Scalar subquery | `SELECT`, `WHERE` | Single value |
| Row subquery | `WHERE ... IN (...)` | Set of values |
| Derived table | `FROM (SELECT ...) AS alias` | Table |
| Correlated subquery | `WHERE EXISTS (SELECT ...)` | Re-runs per outer row |

Correlated subqueries reference the outer query's columns — they re-execute for every outer row, which is often slow. Prefer a JOIN or CTE.

### Common Table Expressions (CTEs)

A CTE (`WITH name AS (...)`) names a subquery for readability and reuse within the same statement. PostgreSQL materializes a CTE by default (since PG 12 the optimizer may inline it; use `WITH ... AS MATERIALIZED` or `AS NOT MATERIALIZED` to force either).

**Recursive CTEs** use `UNION ALL` to walk hierarchical data (org charts, account trees).

### Window Functions

Window functions compute a value across a **window** of rows related to the current row — without collapsing them into groups like `GROUP BY` does.

```
function(expr) OVER (
    [PARTITION BY col1, ...]
    [ORDER BY col2 ...]
    [ROWS|RANGE BETWEEN ... AND ...]
)
```

Common functions:

| Function | Meaning |
|----------|---------|
| `ROW_NUMBER()` | Sequential rank within partition, no ties |
| `RANK()` | Rank with gaps on ties |
| `DENSE_RANK()` | Rank without gaps on ties |
| `LAG(col, n)` | Value from n rows before |
| `LEAD(col, n)` | Value from n rows ahead |
| `SUM(...) OVER (...)` | Running total |
| `AVG(...) OVER (...)` | Moving average |
| `NTILE(n)` | Bucket into n equal groups |
| `FIRST_VALUE / LAST_VALUE` | First / last in window |
| `PERCENT_RANK()` | Relative rank 0.0–1.0 |

## How it works internally

Window functions run **after** `WHERE`, `GROUP BY`, and `HAVING` but **before** `ORDER BY` and `LIMIT`. PostgreSQL sorts the data into partitions, then slides the frame. For large datasets, ensure the `PARTITION BY`/`ORDER BY` columns are indexed or you'll pay for a full sort.

CTEs marked `MATERIALIZED` are evaluated once and stored as a temporary result in memory (or on disk for large results). `NOT MATERIALIZED` lets the planner inline and optimize them as if they were subqueries.

## Worked example

```sql
-- 1. Scalar subquery — transactions above the account average
SELECT
    t.transaction_id,
    t.amount,
    t.account_id,
    (SELECT AVG(t2.amount)
     FROM transactions t2
     WHERE t2.account_id = t.account_id) AS account_avg
FROM transactions t
WHERE t.amount > (
    SELECT AVG(t3.amount)
    FROM transactions t3
    WHERE t3.account_id = t.account_id
);
-- Note: the above is correlated and slow on large data.
-- Prefer the window-function version below.

-- 2. CTE replacing a correlated subquery
WITH account_avgs AS (
    SELECT
        account_id,
        AVG(amount) AS avg_amount
    FROM transactions
    GROUP BY account_id
)
SELECT
    t.transaction_id,
    t.amount,
    aa.avg_amount,
    t.amount - aa.avg_amount AS diff_from_avg
FROM transactions t
JOIN account_avgs aa ON aa.account_id = t.account_id
WHERE t.amount > aa.avg_amount
ORDER BY diff_from_avg DESC;

-- 3. Recursive CTE — account hierarchy (parent/child accounts)
WITH RECURSIVE account_tree AS (
    -- Anchor: root accounts (no parent)
    SELECT
        account_id,
        account_number,
        parent_account_id,
        0 AS depth,
        account_number::text AS path
    FROM accounts
    WHERE parent_account_id IS NULL

    UNION ALL

    -- Recursive: children
    SELECT
        a.account_id,
        a.account_number,
        a.parent_account_id,
        at.depth + 1,
        at.path || ' > ' || a.account_number
    FROM accounts a
    JOIN account_tree at ON at.account_id = a.parent_account_id
)
SELECT * FROM account_tree ORDER BY path;

-- 4. ROW_NUMBER — latest transaction per account
WITH ranked AS (
    SELECT
        t.*,
        ROW_NUMBER() OVER (
            PARTITION BY t.account_id
            ORDER BY t.created_at DESC
        ) AS rn
    FROM transactions t
)
SELECT * FROM ranked WHERE rn = 1;

-- 5. Running balance with SUM OVER
SELECT
    t.transaction_id,
    t.created_at,
    t.amount,
    t.direction,
    SUM(
        CASE WHEN t.direction = 'credit' THEN t.amount ELSE -t.amount END
    ) OVER (
        PARTITION BY t.account_id
        ORDER BY t.created_at
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_balance
FROM transactions t
WHERE t.account_id = 7
ORDER BY t.created_at;

-- 6. LAG — day-over-day transaction count change
WITH daily AS (
    SELECT
        created_at::date AS tx_date,
        COUNT(*) AS tx_count
    FROM transactions
    GROUP BY 1
)
SELECT
    tx_date,
    tx_count,
    LAG(tx_count, 1) OVER (ORDER BY tx_date) AS prev_day_count,
    tx_count - LAG(tx_count, 1) OVER (ORDER BY tx_date) AS day_delta
FROM daily
ORDER BY tx_date;

-- 7. NTILE — segment customers into spending quartiles
WITH customer_spend AS (
    SELECT
        c.customer_id,
        c.full_name,
        SUM(t.amount) AS total_spend
    FROM customers c
    JOIN accounts a ON a.customer_id = c.customer_id
    JOIN transactions t ON t.account_id = a.account_id
    WHERE t.direction = 'debit'
    GROUP BY c.customer_id, c.full_name
)
SELECT
    customer_id,
    full_name,
    total_spend,
    NTILE(4) OVER (ORDER BY total_spend) AS spend_quartile
FROM customer_spend;
```

## Exercises

1. Write a CTE `monthly_totals` that computes total credits and total debits per account per calendar month. Then query it to return accounts where debits exceeded credits in any month.

2. Use `RANK()` to find the top 3 transaction categories by total amount for each account. Return `account_id`, `category_name`, `total_amount`, `rank`. Only show rank ≤ 3.

3. Compute a 7-day rolling average transaction count per day across all accounts. Use a window frame of `ROWS BETWEEN 6 PRECEDING AND CURRENT ROW`.

4. Using a recursive CTE, list all accounts in a tree (assume a `parent_account_id` column on `accounts`). Add a `depth` column and a `path` string like `ROOT > CHILD > GRANDCHILD`.

5. Rewrite the correlated subquery in example #1 using a window function (`AVG(...) OVER (PARTITION BY account_id)`) and verify the result is identical.

(Solutions: ../../solutions/week-06/day-2.md)

## Self-check

- [ ] I can explain when a CTE is preferable over a subquery
- [ ] I can write a recursive CTE for hierarchical data
- [ ] I can apply ROW_NUMBER, RANK, LAG/LEAD, and running totals with window functions
- [ ] I understand the difference between ROWS and RANGE frames

## Supplementary links

- [PostgreSQL Docs — WITH Queries (CTEs)](https://www.postgresql.org/docs/current/queries-with.html)
- [PostgreSQL Docs — Window Functions](https://www.postgresql.org/docs/current/tutorial-window.html)
- [Mode Analytics — SQL Window Functions](https://mode.com/sql-tutorial/sql-window-functions/)
