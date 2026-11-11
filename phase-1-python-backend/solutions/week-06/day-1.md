# Solutions — Week 06 · Day 1

## Exercise 1 — Top 5 customers by total transaction volume

```sql
SELECT
    c.full_name,
    c.email,
    SUM(t.amount) AS total_volume
FROM customers c
JOIN accounts a    ON a.customer_id = c.customer_id
JOIN transactions t ON t.account_id = a.account_id
GROUP BY c.customer_id, c.full_name, c.email
ORDER BY total_volume DESC
LIMIT 5;
```

## Exercise 2 — Checking accounts below $100

```sql
SELECT
    a.account_number,
    a.balance,
    c.full_name
FROM accounts a
JOIN customers c ON c.customer_id = a.customer_id
WHERE a.balance < 100
  AND a.account_type = 'checking'
ORDER BY a.balance;
```

## Exercise 3 — INSERT with sequence-generated account number

```sql
-- Assuming a sequence named account_seq exists
-- CREATE SEQUENCE IF NOT EXISTS account_seq START 1000;

INSERT INTO accounts (customer_id, account_number, account_type, balance)
VALUES (
    1,
    'SAV-' || nextval('account_seq'),
    'savings',
    500.00
)
RETURNING account_id, account_number, created_at;
```

## Exercise 4 — Categorize old uncategorized transactions

```sql
UPDATE transactions
SET category_id = (
    SELECT category_id
    FROM transaction_categories
    WHERE name = 'Uncategorized'
    LIMIT 1
)
WHERE category_id IS NULL
  AND created_at < NOW() - INTERVAL '1 year';
```

The `LIMIT 1` guards against a missing row returning NULL; in production, raise an error if the row is missing.

## Exercise 5 — EXPLAIN ANALYZE on top-volume query

```sql
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT
    c.full_name,
    c.email,
    SUM(t.amount) AS total_volume
FROM customers c
JOIN accounts a     ON a.customer_id = c.customer_id
JOIN transactions t ON t.account_id = a.account_id
GROUP BY c.customer_id, c.full_name, c.email
ORDER BY total_volume DESC
LIMIT 5;
```

**What to look for:**
- Without indexes on `accounts.customer_id` and `transactions.account_id`, you will see `Seq Scan` on both tables.
- After adding `idx_accounts_customer_id` and `idx_transactions_account_id_created`, the planner switches to `Hash Join` or `Index Scan`.
- `Buffers: shared hit=N read=M` — if `read > 0`, pages were fetched from disk; running the query twice should drop read to 0 (buffer cache warm-up).
- `actual rows` vs `estimated rows` discrepancy > 10× indicates outdated statistics — run `ANALYZE transactions`.
