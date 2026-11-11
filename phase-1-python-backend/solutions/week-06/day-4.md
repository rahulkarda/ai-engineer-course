# Solutions — Week 06 · Day 4

## Exercise 1 — Transfer with balance and active checks

```sql
BEGIN;

-- Lock both rows in PK order to prevent deadlock
SELECT account_id, balance, is_active
FROM accounts
WHERE account_id IN (3, 7)
ORDER BY account_id
FOR UPDATE;

-- Application checks (using DO block for demonstration; real code is in Python)
DO $$
DECLARE
    src_balance NUMERIC;
    src_active  BOOLEAN;
    dst_active  BOOLEAN;
BEGIN
    SELECT balance, is_active INTO src_balance, src_active
    FROM accounts WHERE account_id = 3;

    SELECT is_active INTO dst_active
    FROM accounts WHERE account_id = 7;

    IF NOT src_active THEN
        RAISE EXCEPTION 'Source account is inactive';
    END IF;

    IF NOT dst_active THEN
        RAISE EXCEPTION 'Destination account is inactive';
    END IF;

    IF src_balance < 250 THEN
        RAISE EXCEPTION 'Insufficient funds: balance=% < 250', src_balance;
    END IF;

    UPDATE accounts SET balance = balance - 250.00 WHERE account_id = 3;
    UPDATE accounts SET balance = balance + 250.00 WHERE account_id = 7;

    INSERT INTO transactions (account_id, amount, direction, description)
    VALUES (3, 250.00, 'debit',  'Transfer to 7');

    INSERT INTO transactions (account_id, amount, direction, description)
    VALUES (7, 250.00, 'credit', 'Transfer from 3');
END $$;

COMMIT;
```

## Exercise 2 — ROLLBACK vs ROLLBACK TO SAVEPOINT

**`ROLLBACK`** aborts the entire transaction and discards all work since `BEGIN`.

**`ROLLBACK TO SAVEPOINT name`** undoes work back to the savepoint but keeps the transaction open; you can continue with other statements and eventually `COMMIT` or `ROLLBACK`.

**Scenario:**

```sql
BEGIN;

-- Charge a processing fee regardless of what happens next
INSERT INTO transactions (account_id, amount, direction, description, status)
VALUES (1, 2.50, 'debit', 'Processing fee', 'completed');

SAVEPOINT before_main_transfer;

-- Attempt the main transfer
UPDATE accounts SET balance = balance - 1000.00 WHERE account_id = 1;
-- Suppose application detects balance went negative after this:
-- ROLLBACK TO SAVEPOINT before_main_transfer;
-- The fee INSERT is preserved. COMMIT saves just the fee.

COMMIT;
```

Use `ROLLBACK TO SAVEPOINT` when part of the work (the fee) must always be recorded, even if the main operation fails.

## Exercise 3 — Preventing double-processing

```sql
SELECT transaction_id, account_id, amount
FROM transactions
WHERE status = 'pending'
ORDER BY created_at
LIMIT 1
FOR UPDATE SKIP LOCKED;
```

`FOR UPDATE` acquires a row-level lock. `SKIP LOCKED` causes the query to skip rows already locked by another concurrent worker rather than waiting. This gives each worker a distinct row to process, preventing double-processing.

After processing:

```sql
UPDATE transactions
SET status = 'completed'
WHERE transaction_id = $1;
```

## Exercise 4 — READ COMMITTED non-repeatable read

**Sequence:**

1. Transaction A (`READ COMMITTED`): `SELECT balance FROM accounts WHERE account_id = 5;` → returns 1000
2. Transaction B: `UPDATE accounts SET balance = 900 WHERE account_id = 5; COMMIT;`
3. Transaction A: `SELECT balance FROM accounts WHERE account_id = 5;` → returns **900**

Under `READ COMMITTED`, each statement within a transaction takes a fresh snapshot. So the second read in step 3 sees B's committed change — a **non-repeatable read**.

To prevent this: use **`REPEATABLE READ`** or `SERIALIZABLE`. Under `REPEATABLE READ`, transaction A takes a single snapshot at `BEGIN` and all reads within the transaction see the same data regardless of concurrent commits.

```sql
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
SELECT balance FROM accounts WHERE account_id = 5;  -- 1000
-- Even after B commits 900, this transaction still sees 1000
SELECT balance FROM accounts WHERE account_id = 5;  -- still 1000
COMMIT;
```

## Exercise 5 — Deadlock prevention via consistent lock ordering

```sql
-- Worker 1: transfer from account 3 to account 7
-- Worker 2: transfer from account 7 to account 3

-- DEADLOCK scenario (wrong order):
-- Worker 1: LOCK 3, then wait for 7
-- Worker 2: LOCK 7, then wait for 3  ← deadlock

-- CORRECT: always lock lower account_id first
-- Worker 1 and Worker 2 both lock min(3,7)=3 first, then max(3,7)=7

-- Worker 1's transaction:
BEGIN;
SELECT account_id, balance FROM accounts
WHERE account_id IN (3, 7)
ORDER BY account_id   -- 3 first, then 7
FOR UPDATE;

UPDATE accounts SET balance = balance - 500 WHERE account_id = 3;
UPDATE accounts SET balance = balance + 500 WHERE account_id = 7;
COMMIT;

-- Worker 2's transaction (same order!):
BEGIN;
SELECT account_id, balance FROM accounts
WHERE account_id IN (7, 3)
ORDER BY account_id   -- still 3 first, then 7
FOR UPDATE;

UPDATE accounts SET balance = balance - 200 WHERE account_id = 7;
UPDATE accounts SET balance = balance + 200 WHERE account_id = 3;
COMMIT;
```

**Why this prevents deadlock:** both workers acquire locks in the same order (account 3 before account 7). Worker 2 will block waiting for account 3's lock, but it will not hold any lock that Worker 1 needs — no circular dependency, no deadlock.
