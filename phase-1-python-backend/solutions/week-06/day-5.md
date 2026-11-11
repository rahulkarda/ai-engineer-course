# Solutions — Week 06 · Day 5

## Exercise 1 — Account summary using SQLAlchemy Core

```python
from decimal import Decimal
from sqlalchemy import select, func, case
from sqlalchemy.orm import Session
from financial_app.models import Customer, Account, Transaction


def get_account_summary(session: Session, customer_id: int) -> list[dict]:
    stmt = (
        select(
            Account.account_id,
            Account.account_number,
            Account.account_type,
            Account.balance,
            func.sum(
                case((Transaction.direction == "credit", Transaction.amount), else_=Decimal("0"))
            ).label("total_credits"),
            func.sum(
                case((Transaction.direction == "debit", Transaction.amount), else_=Decimal("0"))
            ).label("total_debits"),
        )
        .join(Transaction, Transaction.account_id == Account.account_id, isouter=True)
        .where(Account.customer_id == customer_id)
        .group_by(Account.account_id)
        .order_by(Account.account_id)
    )

    rows = session.execute(stmt).all()
    return [
        {
            "account_id": r.account_id,
            "account_number": r.account_number,
            "account_type": r.account_type,
            "balance": r.balance,
            "total_credits": r.total_credits or Decimal("0"),
            "total_debits": r.total_debits or Decimal("0"),
        }
        for r in rows
    ]
```

## Exercise 2 — Raw psycopg transfer

```python
import os
from decimal import Decimal
import psycopg


def transfer_raw(
    from_account_id: int,
    to_account_id: int,
    amount: Decimal,
) -> None:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL not set")

    # Consistent lock order
    id_a, id_b = sorted([from_account_id, to_account_id])

    with psycopg.connect(url) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                # Lock both rows
                cur.execute(
                    "SELECT account_id, balance FROM accounts "
                    "WHERE account_id = ANY(%s) ORDER BY account_id FOR UPDATE",
                    ([id_a, id_b],),
                )
                rows = {r[0]: r[1] for r in cur.fetchall()}

                if rows.get(from_account_id, Decimal("0")) < amount:
                    raise ValueError("Insufficient funds")

                cur.execute(
                    "UPDATE accounts SET balance = balance - %s WHERE account_id = %s",
                    (amount, from_account_id),
                )
                cur.execute(
                    "UPDATE accounts SET balance = balance + %s WHERE account_id = %s",
                    (amount, to_account_id),
                )
                cur.execute(
                    "INSERT INTO transactions (account_id, amount, direction) VALUES (%s, %s, 'debit')",
                    (from_account_id, amount),
                )
                cur.execute(
                    "INSERT INTO transactions (account_id, amount, direction) VALUES (%s, %s, 'credit')",
                    (to_account_id, amount),
                )
```

**Comparison:**
- psycopg version: ~35 lines, explicit SQL strings, manual row locking logic
- SQLAlchemy ORM version: ~30 lines but also imports, model bindings; ORM handles the identity map, flush, and type coercion automatically
- Error handling is equivalent; the ORM version raises `ValueError` before hitting the DB for the balance check

## Exercise 3 — Statements SQLAlchemy emits

With `echo=True`:

```python
from sqlalchemy import create_engine
import os

engine = create_engine(os.getenv("DATABASE_URL"), echo=True)
```

`session.get(Customer, 1)` emits:

```sql
SELECT customers.customer_id AS customers_customer_id,
       customers.full_name   AS customers_full_name,
       ...
FROM customers
WHERE customers.customer_id = $1
```

Accessing `customer.accounts` (default lazy load, since `relationship` has no `lazy="joined"`):

```sql
SELECT accounts.account_id AS accounts_account_id,
       ...
FROM accounts
WHERE accounts.customer_id = $1
```

So two queries total (N+1 pattern with one customer). To eliminate the second query, use `selectinload`:

```python
from sqlalchemy.orm import selectinload

stmt = select(Customer).options(selectinload(Customer.accounts)).where(Customer.customer_id == 1)
customer = session.scalars(stmt).first()
# Now customer.accounts is already loaded — no second query
```

## Exercise 4 — AuditRepository with Pydantic validation

```python
import logging
from typing import Any
from pydantic import BaseModel
from sqlalchemy.orm import Session
from financial_app.models import AuditLog

logger = logging.getLogger(__name__)


class AuditEntry(BaseModel):
    table_name: str
    record_id: int
    action: str  # INSERT, UPDATE, DELETE
    old_data: dict[str, Any] | None = None
    new_data: dict[str, Any] | None = None
    changed_by: str | None = None


class AuditRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def log_change(self, entry: AuditEntry) -> AuditLog:
        if entry.action not in {"INSERT", "UPDATE", "DELETE"}:
            raise ValueError(f"Invalid action: {entry.action!r}")

        log = AuditLog(
            table_name=entry.table_name,
            record_id=entry.record_id,
            action=entry.action,
            old_data=entry.old_data,
            new_data=entry.new_data,
            changed_by=entry.changed_by,
        )
        self._session.add(log)
        self._session.flush()
        logger.info("Audit: %s on %s#%d", entry.action, entry.table_name, entry.record_id)
        return log
```

## Exercise 5 — pytest tests for TransactionRepository.transfer

```python
# tests/test_transaction_repo.py
from decimal import Decimal
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from financial_app.models import Base, Customer, Account
from financial_app.repository import TransactionRepository


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


@pytest.fixture
def accounts(session):
    cust = Customer(full_name="Test", email="test@t.test")
    session.add(cust)
    session.flush()

    a1 = Account(customer_id=cust.customer_id, account_number="A1",
                 account_type="checking", balance=Decimal("500"))
    a2 = Account(customer_id=cust.customer_id, account_number="A2",
                 account_type="savings", balance=Decimal("200"))
    session.add_all([a1, a2])
    session.flush()
    return a1, a2


def test_transfer_correct_balances(session, accounts):
    a1, a2 = accounts
    repo = TransactionRepository(session)
    repo.transfer(a1.account_id, a2.account_id, Decimal("100"))
    session.flush()

    session.refresh(a1)
    session.refresh(a2)
    assert a1.balance == Decimal("400")
    assert a2.balance == Decimal("300")


def test_transfer_insufficient_funds(session, accounts):
    a1, a2 = accounts
    repo = TransactionRepository(session)
    with pytest.raises(ValueError, match="Insufficient"):
        repo.transfer(a1.account_id, a2.account_id, Decimal("99999"))
```
