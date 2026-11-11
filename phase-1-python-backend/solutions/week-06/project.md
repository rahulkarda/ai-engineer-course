# Solutions — Week 06 Project

This file provides reference implementations for the key non-trivial components of the project. Learners should attempt the project before reading these.

---

## TransactionService.transfer (SERIALIZABLE isolation)

```python
# src/financial_app/service.py (transfer section)
import os
import logging
import time
from decimal import Decimal

import psycopg
from psycopg import errors as pg_errors
from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Account, Transaction

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY_S = 0.05


class TransactionService:
    def __init__(self, session: Session) -> None:
        self._session = session

    def transfer(
        self,
        from_account_id: int,
        to_account_id: int,
        amount: Decimal,
        description: str = "",
        reference_id: str | None = None,
    ) -> tuple[Transaction, Transaction]:
        """
        Atomic fund transfer. Uses SERIALIZABLE isolation on the underlying
        connection to prevent lost updates under concurrency.
        Retries up to MAX_RETRIES times on serialization failure.
        """
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return self._do_transfer(
                    from_account_id, to_account_id, amount, description, reference_id
                )
            except pg_errors.SerializationFailure:
                logger.warning(
                    "Serialization failure attempt %d/%d for transfer %s→%s",
                    attempt, MAX_RETRIES, from_account_id, to_account_id,
                )
                if attempt == MAX_RETRIES:
                    raise
                self._session.rollback()
                time.sleep(RETRY_DELAY_S * attempt)

        raise RuntimeError("Unreachable")  # pragma: no cover

    def _do_transfer(
        self,
        from_id: int,
        to_id: int,
        amount: Decimal,
        description: str,
        reference_id: str | None,
    ) -> tuple[Transaction, Transaction]:
        # Lock in consistent PK order to prevent deadlock
        ids = sorted([from_id, to_id])
        accounts = {
            a.account_id: a
            for a in self._session.scalars(
                select(Account)
                .where(Account.account_id.in_(ids))
                .with_for_update()
                .order_by(Account.account_id)
            )
        }

        src = accounts.get(from_id)
        dst = accounts.get(to_id)

        if src is None or dst is None:
            raise ValueError("One or both accounts not found")
        if not src.is_active or not dst.is_active:
            raise ValueError("Both accounts must be active")
        if src.balance < amount:
            raise ValueError(
                f"Insufficient funds: balance={src.balance} < requested={amount}"
            )

        src.balance -= amount
        dst.balance += amount

        debit = Transaction(
            account_id=from_id,
            amount=amount,
            direction="debit",
            description=description,
            reference_id=f"{reference_id}-debit" if reference_id else None,
        )
        credit = Transaction(
            account_id=to_id,
            amount=amount,
            direction="credit",
            description=description,
            reference_id=f"{reference_id}-credit" if reference_id else None,
        )
        self._session.add_all([debit, credit])
        self._session.flush()
        return debit, credit
```

---

## Transaction reversal

```python
# src/financial_app/service.py (reversal method)
def reverse_transaction(self, transaction_id: int) -> Transaction:
    """
    Reverse a completed transaction: flip the balance change and mark original as 'reversed'.
    Creates a new Transaction with direction flipped and status 'completed'.
    """
    tx = self._session.get(Transaction, transaction_id)
    if tx is None:
        raise ValueError(f"Transaction {transaction_id} not found")
    if tx.status != "completed":
        raise ValueError(f"Cannot reverse transaction with status={tx.status!r}")

    account = self._session.get(Account, tx.account_id)
    if account is None:
        raise ValueError(f"Account {tx.account_id} not found")

    # Reverse the balance change
    if tx.direction == "debit":
        account.balance += tx.amount
    else:
        if account.balance < tx.amount:
            raise ValueError("Reversal would produce negative balance")
        account.balance -= tx.amount

    tx.status = "reversed"

    reversal = Transaction(
        account_id=tx.account_id,
        amount=tx.amount,
        direction="credit" if tx.direction == "debit" else "debit",
        description=f"Reversal of tx#{transaction_id}",
        status="completed",
        category_id=tx.category_id,
    )
    self._session.add(reversal)
    self._session.flush()
    return reversal
```

---

## Filtering + pagination for transaction history

```python
# src/financial_app/repository.py (extended)
from datetime import datetime
from decimal import Decimal
from typing import Literal

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from .models import Transaction


class TransactionRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_history(
        self,
        account_id: int,
        since: datetime | None = None,
        until: datetime | None = None,
        direction: Literal["credit", "debit"] | None = None,
        status: str | None = None,
        category_id: int | None = None,
        min_amount: Decimal | None = None,
        max_amount: Decimal | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Transaction], int]:
        """Returns (items, total_count) for pagination."""
        base = (
            select(Transaction)
            .where(Transaction.account_id == account_id)
        )
        if since:
            base = base.where(Transaction.created_at >= since)
        if until:
            base = base.where(Transaction.created_at <= until)
        if direction:
            base = base.where(Transaction.direction == direction)
        if status:
            base = base.where(Transaction.status == status)
        if category_id is not None:
            base = base.where(Transaction.category_id == category_id)
        if min_amount is not None:
            base = base.where(Transaction.amount >= min_amount)
        if max_amount is not None:
            base = base.where(Transaction.amount <= max_amount)

        count_stmt = select(func.count()).select_from(base.subquery())
        total = self._session.scalar(count_stmt) or 0

        items_stmt = (
            base
            .order_by(Transaction.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        items = list(self._session.scalars(items_stmt))
        return items, total
```

---

## Monthly report query (FastAPI endpoint)

```python
# src/financial_app/main.py (report endpoint)
from sqlalchemy import text

@app.get("/reports/monthly")
def monthly_report(
    session: SessionDep,
    customer_id: int = Query(...),
    months: int = Query(default=6, ge=1, le=24),
) -> list[dict]:
    sql = text("""
        WITH transactions_window AS (
            SELECT
                t.account_id,
                t.amount,
                t.direction,
                DATE_TRUNC('month', t.created_at) AS month
            FROM transactions t
            WHERE t.created_at >= DATE_TRUNC('month', NOW()) - (:months - 1) * INTERVAL '1 month'
        ),
        customer_monthly AS (
            SELECT
                c.customer_id,
                c.full_name,
                tw.month,
                SUM(CASE WHEN tw.direction = 'debit'  THEN tw.amount ELSE 0 END) AS total_debit,
                SUM(CASE WHEN tw.direction = 'credit' THEN tw.amount ELSE 0 END) AS total_credit
            FROM transactions_window tw
            JOIN accounts a ON a.account_id = tw.account_id
            JOIN customers c ON c.customer_id = a.customer_id
            WHERE c.customer_id = :customer_id
            GROUP BY c.customer_id, c.full_name, tw.month
        )
        SELECT
            customer_id,
            full_name,
            TO_CHAR(month, 'YYYY-MM') AS month,
            total_debit,
            total_credit,
            total_credit - total_debit AS net
        FROM customer_monthly
        ORDER BY month DESC
    """)
    rows = session.execute(sql, {"customer_id": customer_id, "months": months}).mappings().all()
    return [dict(r) for r in rows]
```

---

## conftest.py for integration tests

```python
# tests/conftest.py
import os
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from financial_app.models import Base

TEST_DB = os.getenv("TEST_DATABASE_URL", "sqlite:///:memory:")


@pytest.fixture(scope="session")
def engine():
    eng = create_engine(TEST_DB, connect_args={"check_same_thread": False} if "sqlite" in TEST_DB else {})
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)


@pytest.fixture
def session(engine):
    """Each test gets an auto-rollback session."""
    with Session(engine) as sess:
        yield sess
        sess.rollback()
```

---

## Checklist: minimum passing criteria

- [ ] `alembic upgrade head` completes without errors
- [ ] `\d transactions` shows all constraints (check + FK + unique on reference_id)
- [ ] `EXPLAIN ANALYZE` on account+date query shows `Index Scan` not `Seq Scan`
- [ ] POST /transfers with insufficient funds returns HTTP 422
- [ ] POST /transfers with valid accounts updates both balances atomically
- [ ] `pytest --cov=src tests/` shows ≥ 70% coverage
- [ ] `ruff check src/` reports zero errors
