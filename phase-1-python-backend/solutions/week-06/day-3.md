# Solutions — Week 06 · Day 3

## Exercise 1 — Composite UNIQUE constraint

```sql
-- ALTER TABLE form (add to existing table):
ALTER TABLE accounts
    ADD CONSTRAINT uq_accounts_customer_number UNIQUE (customer_id, account_number);

-- Inline CONSTRAINT form (during CREATE TABLE):
CREATE TABLE accounts (
    account_id      BIGSERIAL PRIMARY KEY,
    customer_id     BIGINT NOT NULL REFERENCES customers(customer_id),
    account_number  TEXT   NOT NULL,
    -- ... other columns ...
    CONSTRAINT uq_accounts_customer_number UNIQUE (customer_id, account_number)
);
```

Note: the existing `UNIQUE` on `account_number` alone (from Day 3 schema) already prevents duplicates globally. The composite constraint is useful if you allow the same `account_number` string in different customers' scopes, which is uncommon but valid for multi-tenant designs.

## Exercise 2 — Partial index on failed transactions

```sql
CREATE INDEX idx_transactions_failed
    ON transactions(account_id, created_at DESC)
    WHERE status = 'failed';
```

Query that benefits from it:

```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT transaction_id, amount, created_at
FROM transactions
WHERE status = 'failed'
  AND account_id = 42
ORDER BY created_at DESC
LIMIT 20;
```

Without the index: `Seq Scan` on transactions.
With the index: `Index Scan using idx_transactions_failed` — faster because the index only contains failed rows.

## Exercise 3 — UNIQUE constraint vs explicit CREATE INDEX

When you write `UNIQUE` on a column (inline or via `ALTER TABLE ... ADD CONSTRAINT ... UNIQUE`), PostgreSQL automatically creates a B-tree index to enforce it. This index:

- Has a name like `customers_email_key` (auto-generated)
- Is owned by the constraint — you cannot drop the index directly; you must `DROP CONSTRAINT`
- Enforces uniqueness at INSERT/UPDATE time

An explicit `CREATE INDEX` on the same column:
- Does **not** enforce uniqueness (unless you write `CREATE UNIQUE INDEX`)
- Can be dropped independently of any constraint
- Allows you to name it, add `INCLUDE`, partial predicate, etc.

In practice, prefer the constraint form (`UNIQUE`) when you need enforcement; add an extra `CREATE INDEX` only if you need covering/partial behavior the constraint cannot express.

## Exercise 4 — Normalization to 3NF

**Original (violates 2NF and 3NF):**

```
orders(order_id PK, customer_name, customer_email, product_id, product_name, product_price, quantity)
```

**Violations:**
- `customer_name`, `customer_email` depend only on the implicit customer (transitive via `customer_email`)
- `product_name`, `product_price` depend only on `product_id` — partial dependency if `order_id` is the only PK

**Step 1 — 2NF:** extract `product_id`-dependent columns

```
order_items(order_id PK, product_id PK, quantity)
products(product_id PK, product_name, product_price)
```

**Step 2 — 3NF:** extract customer-dependent columns

```
customers(customer_id PK, customer_name, customer_email UNIQUE)
orders(order_id PK, customer_id FK → customers)
order_items(order_id FK → orders, product_id FK → products, quantity)   [PK: (order_id, product_id)]
products(product_id PK, product_name, product_price)
```

**SQL DDL:**

```sql
CREATE TABLE customers (
    customer_id    SERIAL PRIMARY KEY,
    customer_name  TEXT NOT NULL,
    customer_email TEXT NOT NULL UNIQUE
);

CREATE TABLE orders (
    order_id    SERIAL PRIMARY KEY,
    customer_id INT NOT NULL REFERENCES customers(customer_id)
);

CREATE TABLE products (
    product_id    SERIAL PRIMARY KEY,
    product_name  TEXT           NOT NULL,
    product_price NUMERIC(10, 2) NOT NULL CHECK (product_price >= 0)
);

CREATE TABLE order_items (
    order_id   INT NOT NULL REFERENCES orders(order_id),
    product_id INT NOT NULL REFERENCES products(product_id),
    quantity   INT NOT NULL CHECK (quantity > 0),
    PRIMARY KEY (order_id, product_id)
);
```

## Exercise 5 — CHECK constraint for large debits

```sql
ALTER TABLE transactions
    ADD CONSTRAINT ck_large_debit_limit
    CHECK (
        NOT (direction = 'debit' AND amount > 1000000)
    );
```

Equivalently:

```sql
CHECK (direction <> 'debit' OR amount <= 1000000)
```

Both are logically equivalent. The second form is idiomatic in SQL (material conditional: if direction is debit, then amount ≤ 1 000 000).

Test:
```sql
-- Should fail:
INSERT INTO transactions (account_id, amount, direction)
VALUES (1, 1000001.00, 'debit');
-- ERROR: new row for relation "transactions" violates check constraint "ck_large_debit_limit"

-- Should succeed:
INSERT INTO transactions (account_id, amount, direction)
VALUES (1, 1000001.00, 'credit');
```
