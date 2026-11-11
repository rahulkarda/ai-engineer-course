# Solutions — Week 06 Assessment

## Conceptual answers

**1. Logical execution order**

FROM (and JOINs) → WHERE → GROUP BY → HAVING → SELECT → DISTINCT → ORDER BY → LIMIT/OFFSET

Matters because: aliases defined in SELECT (e.g., `SUM(amount) AS total`) are not available in WHERE or HAVING — those run before SELECT. You cannot write `WHERE total > 1000`; you must repeat the expression or use a subquery/CTE.

**2. INNER JOIN vs LEFT JOIN**

INNER JOIN returns only rows with a match in both tables. LEFT JOIN returns all rows from the left table plus NULL-filled columns from the right where no match exists.

Financial app:
- INNER JOIN accounts ON accounts.customer_id = customers.customer_id → list only customers who have at least one account
- LEFT JOIN same → list all customers; customers with no accounts show NULL for account columns (useful for finding customers with no accounts)

**3. MVCC**

Each row version stores `xmin` (the transaction ID that created it) and `xmax` (the transaction ID that deleted or updated it). A reader's snapshot is taken at transaction start and includes only versions with `xmin` committed before the snapshot and `xmax` not yet committed. Writers create new row versions; they don't lock or overwrite the version a reader is looking at. Hence readers and writers never block each other — the reader sees a consistent snapshot while the writer inserts new versions in parallel.

**4. Covering index / INCLUDE**

A covering index stores additional columns in the leaf pages of the B-tree (via `INCLUDE`) so PostgreSQL can answer a query entirely from the index without fetching the actual table heap pages. This is an "Index Only Scan" (`Heap Fetches: 0`).

`INCLUDE` columns are not part of the B-tree key — they're appended at the leaf level only. You cannot filter on them in the index condition. Key columns can be used in `WHERE`, `ORDER BY`, and range scans; included columns are read-only passengers that save heap I/O.

**5. Isolation levels**

| Level | Dirty Read | Non-Repeatable | Phantom | Serialization anomaly |
|-------|-----------|----------------|---------|----------------------|
| READ UNCOMMITTED | Possible (PG: No) | Yes | Yes | Yes |
| READ COMMITTED (PG default) | No | Yes | Yes | Yes |
| REPEATABLE READ | No | No | No (PG) | Yes |
| SERIALIZABLE | No | No | No | No |

Performance trade-off: SERIALIZABLE uses predicate locks and detects conflicts at commit time, causing some transactions to fail with `serialization_failure` and requiring a full retry. Read-heavy workloads under SERIALIZABLE incur ~10–30% throughput loss vs READ COMMITTED due to the overhead of tracking predicate dependencies.

**6. flush vs commit**

`session.flush()` translates all pending ORM changes (new, dirty, deleted objects) into SQL statements and sends them to the database within the current transaction. The changes are visible to subsequent queries in the same transaction (so you can read back a generated PK), but are not yet committed — other sessions cannot see them, and the changes can still be rolled back.

`session.commit()` ends the transaction: makes all flushed changes permanent and visible to other connections, releases locks, and (with `expire_on_commit=True`) marks all loaded objects as expired so the next attribute access re-fetches from the database.

Use `flush()` without `commit()` when: you need the PK of a newly inserted row to use as a FK in another insert in the same transaction, but you are not ready to commit yet.

**7. Normal forms**

1NF: each column holds atomic (indivisible) values; no repeating groups.
2NF: 1NF + every non-key attribute depends on the *entire* primary key (no partial dependency). Violation example: `order_items(order_id, product_id, product_name)` — `product_name` depends only on `product_id`, not on the composite key. Fix: move `product_name` to a separate `products(product_id, product_name)` table.
3NF: 2NF + no transitive dependency (non-key attribute depends on another non-key attribute). Violation example: `orders(order_id, customer_id, customer_email)` — `customer_email` depends on `customer_id`, which is not the full PK. Fix: move `customer_email` to `customers(customer_id, customer_email)`.

---

## Task A — Monthly spending report (SQL)

```sql
WITH transactions_6m AS (
    SELECT
        t.account_id,
        t.amount,
        t.direction,
        DATE_TRUNC('month', t.created_at) AS month
    FROM transactions t
    WHERE t.created_at >= DATE_TRUNC('month', NOW()) - INTERVAL '5 months'
),
customer_monthly AS (
    SELECT
        c.customer_id,
        c.full_name,
        tm.month,
        SUM(CASE WHEN tm.direction = 'debit'  THEN tm.amount ELSE 0 END) AS total_debit,
        SUM(CASE WHEN tm.direction = 'credit' THEN tm.amount ELSE 0 END) AS total_credit
    FROM transactions_6m tm
    JOIN accounts a ON a.account_id = tm.account_id
    JOIN customers c ON c.customer_id = a.customer_id
    GROUP BY c.customer_id, c.full_name, tm.month
)
SELECT
    customer_id,
    full_name,
    TO_CHAR(month, 'YYYY-MM')   AS month,
    total_debit,
    total_credit,
    total_credit - total_debit  AS net,
    RANK() OVER (
        PARTITION BY month
        ORDER BY total_debit DESC
    ) AS rank_this_month
FROM customer_monthly
ORDER BY month DESC, rank_this_month ASC;
```

---

## Task B — get_account_summary implementation

```python
# src/financial_app/service.py
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

from sqlalchemy import func, case, select
from sqlalchemy.orm import Session

from .models import Account, Transaction

logger = logging.getLogger(__name__)


@dataclass
class AccountSummary:
    account_id: int
    account_number: str
    account_type: str
    current_balance: Decimal
    total_credits_30d: Decimal
    total_debits_30d: Decimal
    tx_count_30d: int
    last_transaction_date: date | None


class TransactionService:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_account_summary(self, customer_id: int) -> list[AccountSummary]:
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=30)

        stmt = (
            select(
                Account.account_id,
                Account.account_number,
                Account.account_type,
                Account.balance,
                func.coalesce(
                    func.sum(
                        case(
                            (Transaction.direction == "credit", Transaction.amount),
                            else_=Decimal("0"),
                        )
                    ).filter(Transaction.created_at >= cutoff),
                    Decimal("0"),
                ).label("total_credits_30d"),
                func.coalesce(
                    func.sum(
                        case(
                            (Transaction.direction == "debit", Transaction.amount),
                            else_=Decimal("0"),
                        )
                    ).filter(Transaction.created_at >= cutoff),
                    Decimal("0"),
                ).label("total_debits_30d"),
                func.coalesce(
                    func.count(Transaction.transaction_id).filter(Transaction.created_at >= cutoff),
                    0,
                ).label("tx_count_30d"),
                func.max(Transaction.created_at).label("last_transaction_date"),
            )
            .outerjoin(Transaction, Transaction.account_id == Account.account_id)
            .where(Account.customer_id == customer_id)
            .group_by(Account.account_id)
            .order_by(Account.account_id)
        )

        rows = self._session.execute(stmt).all()
        return [
            AccountSummary(
                account_id=r.account_id,
                account_number=r.account_number,
                account_type=r.account_type,
                current_balance=r.balance,
                total_credits_30d=r.total_credits_30d,
                total_debits_30d=r.total_debits_30d,
                tx_count_30d=r.tx_count_30d,
                last_transaction_date=r.last_transaction_date.date() if r.last_transaction_date else None,
            )
            for r in rows
        ]
```

```python
# tests/test_service.py
from decimal import Decimal
from datetime import datetime, timezone, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from financial_app.models import Base, Customer, Account, Transaction
from financial_app.service import TransactionService


@pytest.fixture(scope="module")
def engine():
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)


@pytest.fixture
def session(engine):
    with Session(engine) as s:
        yield s
        s.rollback()


def test_get_account_summary_correct_30d_totals(session):
    cust = Customer(full_name="Alice", email="alice@s.test")
    session.add(cust)
    session.flush()

    acc = Account(customer_id=cust.customer_id, account_number="S001",
                  account_type="checking", balance=Decimal("1000"))
    session.add(acc)
    session.flush()

    now = datetime.now(tz=timezone.utc)
    session.add_all([
        Transaction(account_id=acc.account_id, amount=Decimal("200"), direction="credit",
                    created_at=now - timedelta(days=5)),
        Transaction(account_id=acc.account_id, amount=Decimal("50"),  direction="debit",
                    created_at=now - timedelta(days=10)),
        # Outside 30-day window — should NOT count
        Transaction(account_id=acc.account_id, amount=Decimal("999"), direction="credit",
                    created_at=now - timedelta(days=60)),
    ])
    session.flush()

    svc = TransactionService(session)
    summaries = svc.get_account_summary(cust.customer_id)

    assert len(summaries) == 1
    s = summaries[0]
    assert s.total_credits_30d == Decimal("200")
    assert s.total_debits_30d == Decimal("50")
    assert s.tx_count_30d == 2


def test_get_account_summary_no_accounts_returns_empty(session):
    cust = Customer(full_name="Empty", email="empty@s.test")
    session.add(cust)
    session.flush()

    svc = TransactionService(session)
    result = svc.get_account_summary(cust.customer_id)
    assert result == []
```

---

## Debugging Challenge

**Problems with the original query:**

1. `EXTRACT(YEAR FROM t.created_at) = 2024` — wrapping `created_at` in a function prevents index use. PostgreSQL cannot use an index on `created_at` because it would have to evaluate the function for every row (non-sargable predicate).

2. `EXTRACT(...)` requires a sequential scan even if an index exists on `created_at`.

**Rewritten version (index-friendly):**

```sql
SELECT c.full_name, SUM(t.amount) AS total
FROM transactions t
JOIN accounts a ON a.account_id = t.account_id
JOIN customers c ON c.customer_id = a.customer_id
WHERE t.created_at >= '2024-01-01'
  AND t.created_at <  '2025-01-01'
GROUP BY c.customer_id, c.full_name
ORDER BY total DESC
LIMIT 20;
```

Now the range predicate `created_at >= ... AND created_at < ...` can use a B-tree index scan.

**Index:**

```sql
CREATE INDEX idx_transactions_account_created
    ON transactions(account_id, created_at DESC);
-- Or the covering index from the schema:
CREATE INDEX idx_transactions_report
    ON transactions(account_id, created_at DESC)
    INCLUDE (amount, direction, category_id);
```

A separate index on `transactions(created_at)` also helps:

```sql
CREATE INDEX idx_transactions_created_at ON transactions(created_at DESC);
```

**EXPLAIN comparison:**

Original: `Seq Scan on transactions (cost=0..180000 rows=50000 ...)` — scans the full table.

Rewritten: `Index Scan using idx_transactions_created_at on transactions (cost=0..850 rows=45000 ...)` — uses the index to restrict to 2024, then a Hash Join on the smaller result set.
