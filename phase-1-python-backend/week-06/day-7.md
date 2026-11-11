# Phase 1 · Week 06 · Day 7 (Sunday) — Connect Python App and Review

> Objective: Wire the repository layer into a FastAPI application, add filtering/pagination, test the full stack, and prepare for the assessment  ·  Time: ~3 hr  ·  Prereqs: [[day-5]], [[day-6]]

## Overview

Today ties everything together: SQLAlchemy models → repository layer → FastAPI endpoints. You'll also add Alembic-managed migration for a schema change and write integration tests.

**Deliverables:**
- FastAPI app with CRUD endpoints for customers, accounts, transactions
- Filtering and cursor/offset pagination
- Integration tests against a test database
- Assessment prep review

---

## Part 1 — FastAPI application (60 min)

### Pydantic schemas (src/financial_app/schemas.py)

```python
from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, EmailStr, Field, field_validator


class CustomerCreate(BaseModel):
    full_name: str = Field(..., min_length=1, max_length=200)
    email: EmailStr
    phone: str | None = None


class CustomerRead(BaseModel):
    customer_id: int
    full_name: str
    email: str
    phone: str | None
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class AccountCreate(BaseModel):
    customer_id: int
    account_number: str = Field(..., min_length=3, max_length=50)
    account_type: Literal["checking", "savings", "investment"]
    balance: Decimal = Field(default=Decimal("0"), ge=0)
    currency: str = Field(default="USD", min_length=3, max_length=3)

    @field_validator("balance")
    @classmethod
    def quantize_balance(cls, v: Decimal) -> Decimal:
        return v.quantize(Decimal("0.01"))


class AccountRead(BaseModel):
    account_id: int
    customer_id: int
    account_number: str
    account_type: str
    balance: Decimal
    currency: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class TransactionCreate(BaseModel):
    account_id: int
    amount: Decimal = Field(..., gt=0)
    direction: Literal["credit", "debit"]
    description: str | None = None
    reference_id: str | None = None
    category_id: int | None = None


class TransactionRead(BaseModel):
    transaction_id: int
    account_id: int
    category_id: int | None
    amount: Decimal
    direction: str
    description: str | None
    reference_id: str | None
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class TransferRequest(BaseModel):
    from_account_id: int
    to_account_id: int
    amount: Decimal = Field(..., gt=0)
    description: str = ""
    reference_id: str | None = None


class PaginatedResponse[T](BaseModel):
    items: list[T]
    total: int
    limit: int
    offset: int
```

### Main application (src/financial_app/main.py)

```python
import logging
import os
from datetime import datetime
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, status
from sqlalchemy.orm import Session

from .database import get_session
from .models import Customer, Account, Transaction
from .repository import CustomerRepository, TransactionRepository
from .schemas import (
    AccountCreate, AccountRead,
    CustomerCreate, CustomerRead,
    PaginatedResponse, TransactionCreate, TransactionRead,
    TransferRequest,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Financial Transaction Manager", version="0.1.0")

SessionDep = Annotated[Session, Depends(get_session)]


# --- Customers ---

@app.post("/customers", response_model=CustomerRead, status_code=status.HTTP_201_CREATED)
def create_customer(body: CustomerCreate, session: SessionDep) -> Customer:
    repo = CustomerRepository(session)
    existing = repo.get_by_email(body.email)
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")
    return repo.create(full_name=body.full_name, email=body.email, phone=body.phone)


@app.get("/customers/{customer_id}", response_model=CustomerRead)
def get_customer(customer_id: int, session: SessionDep) -> Customer:
    repo = CustomerRepository(session)
    customer = repo.get_by_id(customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer


@app.get("/customers", response_model=PaginatedResponse[CustomerRead])
def list_customers(
    session: SessionDep,
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> PaginatedResponse[CustomerRead]:
    repo = CustomerRepository(session)
    items = repo.list_active(limit=limit, offset=offset)
    return PaginatedResponse(items=items, total=len(items), limit=limit, offset=offset)


# --- Transactions ---

@app.get("/accounts/{account_id}/transactions", response_model=PaginatedResponse[TransactionRead])
def get_transaction_history(
    account_id: int,
    session: SessionDep,
    since: datetime | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> PaginatedResponse[TransactionRead]:
    repo = TransactionRepository(session)
    items = repo.get_history(
        account_id=account_id, since=since, limit=limit, offset=offset
    )
    return PaginatedResponse(items=items, total=len(items), limit=limit, offset=offset)


@app.post("/transfers", response_model=dict, status_code=status.HTTP_201_CREATED)
def create_transfer(body: TransferRequest, session: SessionDep) -> dict:
    repo = TransactionRepository(session)
    try:
        debit, credit = repo.transfer(
            from_account_id=body.from_account_id,
            to_account_id=body.to_account_id,
            amount=body.amount,
            description=body.description,
            reference_id=body.reference_id,
        )
        return {
            "debit_transaction_id": debit.transaction_id,
            "credit_transaction_id": credit.transaction_id,
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
```

---

## Part 2 — Integration tests (45 min)

```python
# tests/test_repository.py
import os
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from financial_app.models import Base, Customer, Account
from financial_app.repository import CustomerRepository, TransactionRepository

# Use an in-memory SQLite for unit-style tests,
# or a test Postgres via env var for integration tests.
TEST_DB_URL = os.getenv("TEST_DATABASE_URL", "sqlite:///:memory:")


@pytest.fixture(scope="module")
def engine():
    eng = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)


@pytest.fixture
def session(engine):
    with Session(engine) as s:
        yield s
        s.rollback()


@pytest.fixture
def two_accounts(session: Session):
    cust = Customer(full_name="Test User", email="test@example.test")
    session.add(cust)
    session.flush()

    acc1 = Account(
        customer_id=cust.customer_id,
        account_number="ACC-001",
        account_type="checking",
        balance=Decimal("1000.00"),
    )
    acc2 = Account(
        customer_id=cust.customer_id,
        account_number="ACC-002",
        account_type="savings",
        balance=Decimal("500.00"),
    )
    session.add_all([acc1, acc2])
    session.flush()
    return acc1, acc2


def test_customer_create_and_retrieve(session: Session) -> None:
    repo = CustomerRepository(session)
    c = repo.create(full_name="Alice", email="alice@example.test")
    session.flush()

    found = repo.get_by_id(c.customer_id)
    assert found is not None
    assert found.email == "alice@example.test"


def test_customer_email_lookup(session: Session) -> None:
    repo = CustomerRepository(session)
    repo.create(full_name="Bob", email="bob@example.test")
    session.flush()

    found = repo.get_by_email("bob@example.test")
    assert found is not None
    assert found.full_name == "Bob"

    not_found = repo.get_by_email("nobody@example.test")
    assert not_found is None


def test_transfer_updates_balances(session: Session, two_accounts) -> None:
    acc1, acc2 = two_accounts
    repo = TransactionRepository(session)

    debit, credit = repo.transfer(
        from_account_id=acc1.account_id,
        to_account_id=acc2.account_id,
        amount=Decimal("200.00"),
    )

    session.refresh(acc1)
    session.refresh(acc2)

    assert acc1.balance == Decimal("800.00")
    assert acc2.balance == Decimal("700.00")
    assert debit.direction == "debit"
    assert credit.direction == "credit"
    assert debit.amount == Decimal("200.00")


def test_transfer_insufficient_funds(session: Session, two_accounts) -> None:
    acc1, acc2 = two_accounts
    repo = TransactionRepository(session)

    with pytest.raises(ValueError, match="Insufficient funds"):
        repo.transfer(
            from_account_id=acc1.account_id,
            to_account_id=acc2.account_id,
            amount=Decimal("99999.00"),
        )


def test_transfer_history_returns_sorted(session: Session, two_accounts) -> None:
    acc1, acc2 = two_accounts
    repo = TransactionRepository(session)

    for _ in range(5):
        repo.transfer(acc1.account_id, acc2.account_id, Decimal("10.00"))
    session.flush()

    history = repo.get_history(account_id=acc1.account_id, limit=10)
    assert len(history) >= 5
    # Verify descending order
    for i in range(len(history) - 1):
        assert history[i].created_at >= history[i + 1].created_at
```

---

## Part 3 — Assessment preparation review (30 min)

Work through each topic. Close the notes and answer from memory.

### SQL fundamentals

- Logical execution order of SQL clauses
- Difference between `WHERE` and `HAVING`
- When does PostgreSQL choose a sequential scan over an index scan?
- What does `EXPLAIN (ANALYZE, BUFFERS)` tell you that `EXPLAIN` alone does not?

### Joins and aggregations

- Difference between `INNER JOIN`, `LEFT JOIN`, `FULL OUTER JOIN`
- What happens if a `GROUP BY` key is NULL?
- How do window functions differ from `GROUP BY`?

### CTEs and subqueries

- When is a CTE `MATERIALIZED` vs inlined?
- How does a recursive CTE terminate?

### Constraints and normalization

- What is referential integrity? Which SQL construct enforces it?
- Explain 1NF, 2NF, 3NF with a concrete example.
- What is a covering index and when does it allow an index-only scan?

### Transactions and ACID

- Which isolation level prevents phantom reads in PostgreSQL?
- What is MVCC and why does it allow readers and writers to not block each other?
- When does PostgreSQL raise a `serialization_failure` and what should the application do?

### SQLAlchemy

- What is the identity map and when can it return stale data?
- Difference between `session.flush()` and `session.commit()`
- What does `pool_pre_ping=True` do?

(Solutions: ../../solutions/week-06/day-7.md)

## Self-check

- [ ] I have a running FastAPI app that creates customers, accounts, and transfers
- [ ] All integration tests pass
- [ ] I can answer every review question above without notes
- [ ] I understand what each piece of EXPLAIN ANALYZE output means

## Supplementary links

- [FastAPI — Dependencies with Database Sessions](https://fastapi.tiangolo.com/tutorial/sql-databases/)
- [SQLAlchemy 2.0 — Session Basics](https://docs.sqlalchemy.org/en/20/orm/session_basics.html)
- [pytest — Fixtures](https://docs.pytest.org/en/stable/explanation/fixtures.html)
