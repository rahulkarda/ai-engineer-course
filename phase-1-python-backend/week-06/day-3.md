# Phase 1 · Week 06 · Day 3 — Constraints, Normalization, and Indexes

> Objective: Design a normalized schema with proper constraints and indexes that support query performance  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

## Concept

### Constraints

Constraints enforce data integrity at the database level — independent of the application.

| Constraint | Meaning |
|------------|---------|
| `PRIMARY KEY` | Unique, not null; every table should have one |
| `FOREIGN KEY` | References a PK/unique column in another table; enforces referential integrity |
| `UNIQUE` | No duplicate values in the column(s) |
| `NOT NULL` | Column must always have a value |
| `CHECK` | Boolean expression must be true for every row |
| `DEFAULT` | Value used when none is provided |
| `EXCLUDE` | No two rows can satisfy a condition simultaneously (e.g. overlapping date ranges) |

Foreign keys can have `ON DELETE` actions: `CASCADE`, `SET NULL`, `SET DEFAULT`, `RESTRICT`, `NO ACTION`.

### Normalization

Normalization reduces redundancy and update anomalies by decomposing tables.

| Normal Form | Rule |
|-------------|------|
| 1NF | Atomic columns; no repeating groups |
| 2NF | 1NF + no partial dependency on a composite key |
| 3NF | 2NF + no transitive dependency (non-key → non-key) |
| BCNF | Stricter 3NF — every determinant is a candidate key |

In practice: design to 3NF, then selectively denormalize hot read paths.

### Indexes and B-Trees

An **index** is a separate data structure that PostgreSQL maintains alongside the table to speed up lookups.

**B-Tree** (default): a balanced tree ordered by key value. Supports `=`, `<`, `>`, `BETWEEN`, `LIKE 'prefix%'`. O(log N) lookup. Most general-purpose index.

**Other types:**
- `HASH` — equality only, slightly faster than B-tree for `=`
- `GIN` — arrays, JSONB, full-text search
- `GiST` — geometric types, ranges
- `BRIN` — very large sequentially written tables (e.g. time-series); tiny but approximate

**Index selectivity**: a high-cardinality column (many distinct values) makes a selective index; PostgreSQL may skip a low-selectivity index and do a sequential scan instead.

**Covering index** (`INCLUDE`): stores extra columns in the index leaf, allowing index-only scans.

**Partial index**: indexes only rows satisfying a `WHERE` predicate — smaller and faster when queries consistently filter on that predicate.

## How it works internally

B-tree pages are 8 KB by default. PostgreSQL stores: internal pages (routing), leaf pages (actual key + heap tuple pointer `ctid`). On a lookup the planner descends the tree (typically 3–4 levels for millions of rows) and then fetches heap pages at the `ctid` addresses. If the heap pages are not in the buffer cache, those are random I/Os — this is why indexes are less useful on very wide sequential scans.

`VACUUM` and `AUTOVACUUM` reclaim dead tuple space. `ANALYZE` updates `pg_statistics` so the planner has accurate row-count estimates.

## Worked example

```sql
-- Full schema for the Financial Transaction Management System

CREATE TABLE customers (
    customer_id   BIGSERIAL PRIMARY KEY,
    full_name     TEXT        NOT NULL,
    email         TEXT        NOT NULL UNIQUE,
    phone         TEXT,
    is_active     BOOLEAN     NOT NULL DEFAULT true,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE transaction_categories (
    category_id   SERIAL PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    description   TEXT
);

CREATE TABLE accounts (
    account_id      BIGSERIAL PRIMARY KEY,
    customer_id     BIGINT      NOT NULL REFERENCES customers(customer_id) ON DELETE RESTRICT,
    account_number  TEXT        NOT NULL UNIQUE,
    account_type    TEXT        NOT NULL CHECK (account_type IN ('checking', 'savings', 'investment')),
    balance         NUMERIC(15, 2) NOT NULL DEFAULT 0.00 CHECK (balance >= 0),
    currency        CHAR(3)     NOT NULL DEFAULT 'USD',
    is_active       BOOLEAN     NOT NULL DEFAULT true,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE transactions (
    transaction_id  BIGSERIAL PRIMARY KEY,
    account_id      BIGINT         NOT NULL REFERENCES accounts(account_id) ON DELETE RESTRICT,
    category_id     INT            REFERENCES transaction_categories(category_id) ON DELETE SET NULL,
    amount          NUMERIC(15, 2) NOT NULL CHECK (amount > 0),
    direction       TEXT           NOT NULL CHECK (direction IN ('credit', 'debit')),
    description     TEXT,
    reference_id    TEXT UNIQUE,         -- idempotency key
    status          TEXT NOT NULL DEFAULT 'completed'
                         CHECK (status IN ('pending', 'completed', 'failed', 'reversed')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE audit_logs (
    log_id      BIGSERIAL PRIMARY KEY,
    table_name  TEXT        NOT NULL,
    record_id   BIGINT      NOT NULL,
    action      TEXT        NOT NULL CHECK (action IN ('INSERT', 'UPDATE', 'DELETE')),
    old_data    JSONB,
    new_data    JSONB,
    changed_by  TEXT,
    changed_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes

-- Frequently filtered: active customers
CREATE INDEX idx_customers_is_active ON customers(is_active) WHERE is_active = true;

-- Lookup by email (already unique, but explicit for clarity in EXPLAIN)
-- The UNIQUE constraint creates an index automatically.

-- Accounts per customer (FK + frequent JOIN)
CREATE INDEX idx_accounts_customer_id ON accounts(customer_id);

-- Transactions: date range queries (most common)
CREATE INDEX idx_transactions_created_at ON transactions(created_at DESC);

-- Transactions per account (FK + range queries)
CREATE INDEX idx_transactions_account_id_created ON transactions(account_id, created_at DESC);

-- Partial index: only pending transactions (small, focused)
CREATE INDEX idx_transactions_pending ON transactions(account_id, created_at)
    WHERE status = 'pending';

-- Covering index: avoid heap fetch for common report query
CREATE INDEX idx_transactions_report ON transactions(account_id, created_at DESC)
    INCLUDE (amount, direction, category_id);

-- Audit log lookups by table + record
CREATE INDEX idx_audit_table_record ON audit_logs(table_name, record_id, changed_at DESC);

-- Check constraints violation demo
-- INSERT INTO accounts (customer_id, account_number, account_type, balance)
-- VALUES (1, 'ACC-001', 'invalid_type', -100);
-- ERROR: new row for relation "accounts" violates check constraint "accounts_account_type_check"
```

### Explain index usage

```sql
-- Force planner to show index usage on transactions
EXPLAIN (ANALYZE, BUFFERS)
SELECT transaction_id, amount, direction
FROM transactions
WHERE account_id = 42
  AND created_at >= NOW() - INTERVAL '7 days'
ORDER BY created_at DESC;
-- Should use: Index Scan using idx_transactions_account_id_created
```

## Exercises

1. Add a `UNIQUE` constraint on `(customer_id, account_number)` as a composite constraint on the `accounts` table. Write both `ALTER TABLE` and inline `CONSTRAINT` forms.

2. Create a partial index on `transactions` for rows where `status = 'failed'`. Write a query that would benefit from this index and verify with `EXPLAIN`.

3. The `customers` table has `email` with a `UNIQUE` constraint. Explain what index PostgreSQL creates automatically and how it differs from an explicit `CREATE INDEX`.

4. Normalize the following denormalized table to 3NF:
   ```
   orders(order_id, customer_name, customer_email, product_id, product_name, product_price, quantity)
   ```
   Draw the resulting tables and their relationships.

5. Add a `CHECK` constraint to `transactions` that ensures `amount` does not exceed 1,000,000 for `direction = 'debit'`. Use a single `CHECK` expression.

(Solutions: ../../solutions/week-06/day-3.md)

## Self-check

- [ ] I can write a schema with PK, FK, UNIQUE, CHECK, and NOT NULL constraints
- [ ] I can normalize a table to 3NF
- [ ] I can create B-tree, partial, and covering indexes
- [ ] I can read EXPLAIN output to confirm index usage

## Supplementary links

- [Use The Index, Luke — B-Tree Basics](https://use-the-index-luke.com/sql/anatomy)
- [PostgreSQL Docs — CREATE INDEX](https://www.postgresql.org/docs/current/sql-createindex.html)
- [PostgreSQL Docs — Constraints](https://www.postgresql.org/docs/current/ddl-constraints.html)
