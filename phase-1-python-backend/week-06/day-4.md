# Phase 1 · Week 06 · Day 4 — Transactions, ACID, and Isolation Levels

> Objective: Write correct multi-statement transactions and choose isolation levels to prevent concurrency anomalies  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

## Concept

### ACID Properties

A **transaction** is a unit of work that must satisfy ACID:

| Property | Meaning |
|----------|---------|
| **Atomicity** | All statements succeed or none do — no partial updates |
| **Consistency** | Transaction brings the DB from one valid state to another (constraints hold) |
| **Isolation** | Concurrent transactions do not see each other's intermediate state |
| **Durability** | Committed data survives crashes (written to WAL, then to disk) |

### Concurrency anomalies

| Anomaly | Description |
|---------|-------------|
| **Dirty read** | Read uncommitted data from another transaction |
| **Non-repeatable read** | Same row read twice returns different values (another tx committed between reads) |
| **Phantom read** | Same predicate query returns different row sets (another tx inserted/deleted) |
| **Serialization anomaly** | Aggregate behavior of concurrent transactions is inconsistent with any serial order |

### Isolation levels (SQL standard + PostgreSQL)

| Level | Dirty Read | Non-Repeatable | Phantom | Serialization Anomaly |
|-------|-----------|----------------|---------|----------------------|
| READ UNCOMMITTED | Possible* | Yes | Yes | Yes |
| **READ COMMITTED** (PG default) | No | Yes | Yes | Yes |
| REPEATABLE READ | No | No | No** | Yes |
| **SERIALIZABLE** | No | No | No | No |

*PostgreSQL READ UNCOMMITTED behaves like READ COMMITTED — dirty reads never happen.
**PostgreSQL's REPEATABLE READ also prevents phantoms (unlike the SQL standard).

PostgreSQL implements isolation via **MVCC** (Multi-Version Concurrency Control): each transaction sees a snapshot of the database taken at its start (or at each statement, for READ COMMITTED). Writers never block readers; readers never block writers.

### Deadlocks

Two transactions each hold a lock the other needs. PostgreSQL detects deadlocks automatically and aborts one with error code `40P01`. Application code must catch this and retry.

Lock ordering convention: always acquire locks in the same order across transactions to prevent deadlocks.

## How it works internally

PostgreSQL's WAL (Write-Ahead Log) ensures durability: changes are written to WAL before the buffer manager flushes data pages. On crash recovery, PostgreSQL replays WAL from the last checkpoint.

MVCC stores multiple row versions (`ctid` chain). Each row version has `xmin` (creating transaction ID) and `xmax` (deleting transaction ID). A transaction's snapshot determines which versions are visible — versions with `xmin` committed before the snapshot start and `xmax` not yet committed.

`VACUUM` removes dead row versions (tuples with `xmax` committed) to reclaim space.

## Worked example

```sql
-- 1. Basic transaction
BEGIN;

UPDATE accounts
SET balance = balance - 500.00,
    updated_at = NOW()
WHERE account_id = 1;

UPDATE accounts
SET balance = balance + 500.00,
    updated_at = NOW()
WHERE account_id = 2;

INSERT INTO transactions (account_id, amount, direction, description, reference_id)
VALUES (1, 500.00, 'debit',  'Transfer to ACC-002', 'REF-2024-001');

INSERT INTO transactions (account_id, amount, direction, description, reference_id)
VALUES (2, 500.00, 'credit', 'Transfer from ACC-001', 'REF-2024-002');

COMMIT;
-- If any statement fails, PostgreSQL auto-rollbacks on error.
-- Explicit ROLLBACK below:

-- 2. Explicit rollback on application error
BEGIN;

UPDATE accounts SET balance = balance - 10000.00 WHERE account_id = 1;

-- Simulate application check
DO $$
BEGIN
    IF (SELECT balance FROM accounts WHERE account_id = 1) < 0 THEN
        RAISE EXCEPTION 'Insufficient funds';
    END IF;
END
$$;

COMMIT;
-- If the RAISE EXCEPTION fires, the whole transaction is aborted.
-- Use ROLLBACK in application code after catching the error.

-- 3. SAVEPOINT — partial rollback
BEGIN;

INSERT INTO transactions (account_id, amount, direction, description)
VALUES (1, 100.00, 'debit', 'Fee charge');

SAVEPOINT before_transfer;

UPDATE accounts SET balance = balance - 5000.00 WHERE account_id = 1;
-- If this violates balance CHECK, rollback to savepoint:
ROLLBACK TO SAVEPOINT before_transfer;

-- Continue with just the fee insert
COMMIT;

-- 4. Explicit locking — SELECT FOR UPDATE
BEGIN;

-- Lock the row to prevent concurrent modifications
SELECT balance
FROM accounts
WHERE account_id = 42
FOR UPDATE;         -- acquires row-level lock

-- Safe to update; no other transaction can modify this row until COMMIT
UPDATE accounts
SET balance = balance - 200.00
WHERE account_id = 42;

COMMIT;

-- SELECT FOR UPDATE SKIP LOCKED — non-blocking queue pattern
SELECT transaction_id, account_id, amount
FROM transactions
WHERE status = 'pending'
ORDER BY created_at
LIMIT 10
FOR UPDATE SKIP LOCKED;  -- skip rows already locked by another worker

-- 5. Setting isolation level
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;

SELECT SUM(balance) FROM accounts WHERE customer_id = 5;
-- Any concurrent INSERT of a new account for customer 5 is invisible here
-- until this transaction commits.

COMMIT;

-- 6. Serializable — full serializability guarantee
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;

SELECT COUNT(*) FROM transactions WHERE account_id = 1 AND status = 'pending';
-- ... application logic ...
INSERT INTO transactions (account_id, amount, direction, status)
VALUES (1, 50.00, 'debit', 'pending');

COMMIT;
-- PostgreSQL may raise serialization_failure (40001) if it detects a conflict.
-- Application MUST retry the entire transaction.
```

### Python retry pattern for serialization failures

```python
import os
import time
import logging
import psycopg
from psycopg import errors

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 0.1  # seconds


def transfer_funds(
    account_from: int,
    account_to: int,
    amount: float,
    database_url: str | None = None,
) -> None:
    url = database_url or os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL not set")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with psycopg.connect(url) as conn:
                conn.isolation_level = "SERIALIZABLE"
                with conn.transaction():
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE accounts SET balance = balance - %s WHERE account_id = %s",
                            (amount, account_from),
                        )
                        cur.execute(
                            "UPDATE accounts SET balance = balance + %s WHERE account_id = %s",
                            (amount, account_to),
                        )
            return  # success
        except errors.SerializationFailure:
            logger.warning("Serialization failure, attempt %d/%d", attempt, MAX_RETRIES)
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_DELAY * attempt)
```

## Exercises

1. Write a transaction that transfers $250 from `account_id = 3` to `account_id = 7`. Include proper balance checks: roll back if either account has insufficient funds or is inactive.

2. Explain the difference between `ROLLBACK` and `ROLLBACK TO SAVEPOINT`. When would you use each? Give a concrete scenario with the transactions table.

3. A fintech app processes payments concurrently. Two workers both read the same pending transaction and try to mark it as `completed`. What SQL construct prevents double-processing? Write the correct query.

4. Under `READ COMMITTED`, transaction A reads a row, transaction B updates and commits it, then transaction A reads the same row again. What is the result? What isolation level would you use to prevent this?

5. Demonstrate deadlock prevention: write two stored-procedure-style transaction blocks that transfer money between accounts, using consistent lock ordering to avoid deadlocks.

(Solutions: ../../solutions/week-06/day-4.md)

## Self-check

- [ ] I can explain each ACID property with a concrete example
- [ ] I can describe all four isolation levels and which anomalies each prevents
- [ ] I know when to use SERIALIZABLE vs READ COMMITTED and the cost of each
- [ ] I can implement SELECT FOR UPDATE SKIP LOCKED for a worker queue

## Supplementary links

- [PostgreSQL Docs — Transaction Isolation](https://www.postgresql.org/docs/current/transaction-iso.html)
- [PostgreSQL Docs — Explicit Locking](https://www.postgresql.org/docs/current/explicit-locking.html)
- [Martin Kleppmann — Designing Data-Intensive Applications, Ch. 7 (excerpt)](https://dataintensive.net/)
