# Solutions — Week 06 · Day 2

## Exercise 1 — Monthly CTE with debit > credit months

```sql
WITH monthly_totals AS (
    SELECT
        t.account_id,
        DATE_TRUNC('month', t.created_at)       AS month,
        SUM(CASE WHEN t.direction = 'credit' THEN t.amount ELSE 0 END) AS total_credits,
        SUM(CASE WHEN t.direction = 'debit'  THEN t.amount ELSE 0 END) AS total_debits
    FROM transactions t
    GROUP BY t.account_id, DATE_TRUNC('month', t.created_at)
)
SELECT
    account_id,
    month,
    total_credits,
    total_debits
FROM monthly_totals
WHERE total_debits > total_credits
ORDER BY account_id, month;
```

## Exercise 2 — Top 3 categories per account using RANK

```sql
WITH category_totals AS (
    SELECT
        t.account_id,
        COALESCE(tc.name, 'Uncategorized') AS category_name,
        SUM(t.amount)                       AS total_amount,
        RANK() OVER (
            PARTITION BY t.account_id
            ORDER BY SUM(t.amount) DESC
        ) AS rnk
    FROM transactions t
    LEFT JOIN transaction_categories tc ON tc.category_id = t.category_id
    GROUP BY t.account_id, tc.name
)
SELECT account_id, category_name, total_amount, rnk
FROM category_totals
WHERE rnk <= 3
ORDER BY account_id, rnk;
```

## Exercise 3 — 7-day rolling average transaction count

```sql
WITH daily_counts AS (
    SELECT
        created_at::date      AS tx_date,
        COUNT(*)              AS tx_count
    FROM transactions
    GROUP BY 1
)
SELECT
    tx_date,
    tx_count,
    AVG(tx_count) OVER (
        ORDER BY tx_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    )::numeric(10, 2) AS rolling_7d_avg
FROM daily_counts
ORDER BY tx_date;
```

## Exercise 4 — Recursive CTE for account tree

```sql
WITH RECURSIVE account_tree AS (
    -- Anchor: root accounts
    SELECT
        account_id,
        account_number,
        parent_account_id,
        0           AS depth,
        account_number::text AS path
    FROM accounts
    WHERE parent_account_id IS NULL

    UNION ALL

    SELECT
        a.account_id,
        a.account_number,
        a.parent_account_id,
        at.depth + 1,
        at.path || ' > ' || a.account_number
    FROM accounts a
    JOIN account_tree at ON at.account_id = a.parent_account_id
    WHERE at.depth < 10  -- guard against cycles
)
SELECT
    account_id,
    depth,
    path
FROM account_tree
ORDER BY path;
```

The `depth < 10` guard prevents infinite recursion if there are accidental cycles in the data.

## Exercise 5 — Window function version vs. correlated subquery

```sql
-- Correlated (original — slow, re-executes per row):
SELECT
    t.transaction_id,
    t.amount,
    t.account_id,
    (SELECT AVG(t2.amount) FROM transactions t2 WHERE t2.account_id = t.account_id) AS account_avg
FROM transactions t
WHERE t.amount > (
    SELECT AVG(t3.amount) FROM transactions t3 WHERE t3.account_id = t.account_id
);

-- Window function version (runs the AVG once per partition):
SELECT transaction_id, amount, account_id, account_avg
FROM (
    SELECT
        t.transaction_id,
        t.amount,
        t.account_id,
        AVG(t.amount) OVER (PARTITION BY t.account_id) AS account_avg
    FROM transactions t
) sub
WHERE amount > account_avg;
```

**Why they produce the same result:** both compute the average of all amounts for the same account. The window function version scans the table once; the correlated subquery re-evaluates the inner `SELECT` for every row in the outer query.

**Verifying equality:**

```sql
-- Should return 0 rows if both are equivalent
(
    SELECT transaction_id FROM (
        SELECT t.transaction_id,
               AVG(t.amount) OVER (PARTITION BY t.account_id) AS account_avg
        FROM transactions t
    ) w WHERE amount > account_avg
)
EXCEPT
(
    SELECT transaction_id FROM transactions t
    WHERE t.amount > (SELECT AVG(t2.amount) FROM transactions t2 WHERE t2.account_id = t.account_id)
);
```
