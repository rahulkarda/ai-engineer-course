# Phase 1 · Week 06 · Day 5 — SQLAlchemy 2.0 and Python Integration

> Objective: Use SQLAlchemy 2.0 (ORM and Core) with psycopg3 to build a type-safe database layer  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

## Concept

### psycopg vs SQLAlchemy

| Tool | Layer | Use when |
|------|-------|----------|
| `psycopg` (v3) | Raw driver | Full SQL control, minimal abstraction, high performance |
| SQLAlchemy Core | SQL expression language | Programmatic SQL construction, portable, no ORM overhead |
| SQLAlchemy ORM | Object-relational mapper | Domain objects, relationships, migrations via Alembic |

SQLAlchemy 2.0 made a clean break from 1.x: `Session.execute()` returns `Result`, not a legacy `ResultProxy`. All style uses `select()`, `insert()`, etc. from `sqlalchemy` directly.

### Connection pooling

Raw psycopg opens one connection per call. Connection setup (~5–20 ms) adds up. A **connection pool** keeps a pool of open connections and checks them out on demand.

SQLAlchemy's `create_engine()` includes a built-in pool (`QueuePool` by default). Key parameters:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `pool_size` | 5 | Persistent connections |
| `max_overflow` | 10 | Extra connections above pool_size (temporary) |
| `pool_timeout` | 30 | Seconds to wait for a connection |
| `pool_pre_ping` | False | Test connection before use (detects stale) |
| `pool_recycle` | -1 | Recycle connections after N seconds |

### ORM vs raw SQL decision

Use **ORM** when: you have complex object graphs, use relationships/lazy-loading, want Alembic migrations tied to Python models.  
Use **Core / raw SQL** when: bulk inserts, complex analytical queries, maximum throughput, queries that don't map cleanly to objects.

Mix freely — SQLAlchemy supports both in the same session.

## How it works internally

SQLAlchemy Core builds a SQL expression tree (`ClauseElement` objects). When executed, the dialect (here `postgresql+psycopg`) compiles it to a parameterized SQL string and passes it to psycopg. Parameters are always passed as bind variables — never interpolated — so SQL injection is structurally prevented.

The `Session` tracks all loaded objects in an **identity map** (keyed by primary key). When you flush, it computes diffs and emits the minimum set of SQL statements. `expire_on_commit=True` (default) expires all attributes after commit, so the next access re-fetches from the DB.

## Worked example

```python
# src/financial_app/database.py
import os
import logging
from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)


def get_engine(database_url: str | None = None):
    url = database_url or os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL environment variable not set")
    return create_engine(
        url,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False,  # set True to log SQL
    )


# Module-level engine (created once)
engine = get_engine()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def get_session() -> Generator[Session, None, None]:
    """FastAPI / context-manager compatible session dependency."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
```

```python
# src/financial_app/models.py
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger, Boolean, CheckConstraint, DateTime, ForeignKey,
    Numeric, String, Text, UniqueConstraint, func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "customers"

    customer_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    full_name:   Mapped[str] = mapped_column(Text, nullable=False)
    email:       Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    phone:       Mapped[str | None] = mapped_column(Text)
    is_active:   Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at:  Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at:  Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    accounts: Mapped[list["Account"]] = relationship(back_populates="customer")

    def __repr__(self) -> str:
        return f"<Customer {self.customer_id} {self.email!r}>"


class Account(Base):
    __tablename__ = "accounts"
    __table_args__ = (
        CheckConstraint("account_type IN ('checking','savings','investment')", name="ck_account_type"),
        CheckConstraint("balance >= 0", name="ck_balance_non_negative"),
    )

    account_id:     Mapped[int]     = mapped_column(BigInteger, primary_key=True)
    customer_id:    Mapped[int]     = mapped_column(
        BigInteger, ForeignKey("customers.customer_id", ondelete="RESTRICT"), nullable=False
    )
    account_number: Mapped[str]     = mapped_column(Text, nullable=False, unique=True)
    account_type:   Mapped[str]     = mapped_column(String(20), nullable=False)
    balance:        Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False, default=Decimal("0"))
    currency:       Mapped[str]     = mapped_column(String(3), nullable=False, default="USD")
    is_active:      Mapped[bool]    = mapped_column(Boolean, nullable=False, default=True)
    created_at:     Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    customer:     Mapped["Customer"]     = relationship(back_populates="accounts")
    transactions: Mapped[list["Transaction"]] = relationship(back_populates="account")


class TransactionCategory(Base):
    __tablename__ = "transaction_categories"

    category_id: Mapped[int] = mapped_column(primary_key=True)
    name:        Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text)


class Transaction(Base):
    __tablename__ = "transactions"
    __table_args__ = (
        CheckConstraint("amount > 0", name="ck_amount_positive"),
        CheckConstraint("direction IN ('credit','debit')", name="ck_direction"),
        CheckConstraint(
            "status IN ('pending','completed','failed','reversed')",
            name="ck_status",
        ),
    )

    transaction_id: Mapped[int]     = mapped_column(BigInteger, primary_key=True)
    account_id:     Mapped[int]     = mapped_column(
        BigInteger, ForeignKey("accounts.account_id", ondelete="RESTRICT"), nullable=False
    )
    category_id:    Mapped[int | None] = mapped_column(
        ForeignKey("transaction_categories.category_id", ondelete="SET NULL")
    )
    amount:         Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)
    direction:      Mapped[str]     = mapped_column(String(10), nullable=False)
    description:    Mapped[str | None] = mapped_column(Text)
    reference_id:   Mapped[str | None] = mapped_column(Text, unique=True)
    status:         Mapped[str]     = mapped_column(String(20), nullable=False, default="completed")
    created_at:     Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    account:  Mapped["Account"]              = relationship(back_populates="transactions")
    category: Mapped["TransactionCategory | None"] = relationship()
```

```python
# src/financial_app/repository.py
from decimal import Decimal
from datetime import datetime

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from .models import Customer, Account, Transaction


class CustomerRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_by_id(self, customer_id: int) -> Customer | None:
        return self._session.get(Customer, customer_id)

    def get_by_email(self, email: str) -> Customer | None:
        stmt = select(Customer).where(Customer.email == email)
        return self._session.scalars(stmt).first()

    def list_active(self, limit: int = 100, offset: int = 0) -> list[Customer]:
        stmt = (
            select(Customer)
            .where(Customer.is_active.is_(True))
            .order_by(Customer.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(self._session.scalars(stmt))

    def create(self, full_name: str, email: str, phone: str | None = None) -> Customer:
        customer = Customer(full_name=full_name, email=email, phone=phone)
        self._session.add(customer)
        self._session.flush()  # get PK without committing
        return customer


class TransactionRepository:
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
        """Atomic fund transfer between two accounts."""
        # Lock both rows in consistent order to prevent deadlock
        ids = sorted([from_account_id, to_account_id])
        stmt = (
            select(Account)
            .where(Account.account_id.in_(ids))
            .with_for_update()
            .order_by(Account.account_id)
        )
        accounts = {a.account_id: a for a in self._session.scalars(stmt)}

        src = accounts[from_account_id]
        dst = accounts[to_account_id]

        if src.balance < amount:
            raise ValueError(f"Insufficient funds: balance={src.balance} < amount={amount}")

        src.balance -= amount
        dst.balance += amount

        debit = Transaction(
            account_id=from_account_id,
            amount=amount,
            direction="debit",
            description=description,
            reference_id=reference_id and f"{reference_id}-debit",
        )
        credit = Transaction(
            account_id=to_account_id,
            amount=amount,
            direction="credit",
            description=description,
            reference_id=reference_id and f"{reference_id}-credit",
        )
        self._session.add_all([debit, credit])
        self._session.flush()
        return debit, credit

    def get_history(
        self,
        account_id: int,
        since: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Transaction]:
        stmt = (
            select(Transaction)
            .where(Transaction.account_id == account_id)
            .order_by(Transaction.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        if since:
            stmt = stmt.where(Transaction.created_at >= since)
        return list(self._session.scalars(stmt))
```

## Exercises

1. Add a `get_account_summary` method to `CustomerRepository` that returns each account with its total credit and debit amounts. Use SQLAlchemy Core (`select`, `func.sum`, `case`) rather than the ORM.

2. Write a raw SQL version of the `transfer` method using `psycopg` directly (no SQLAlchemy). Compare the code length and error handling with the ORM version.

3. Configure the engine with `echo=True` and run a query. Capture the logged SQL and explain each statement SQLAlchemy emits for a `session.get(Customer, 1)` followed by accessing `customer.accounts`.

4. Implement an `AuditRepository.log_change` method that inserts into `audit_logs`. Accept `table_name`, `record_id`, `action`, `old_data`, `new_data` as typed parameters using Pydantic for input validation.

5. Write a pytest test for `TransactionRepository.transfer` that: (a) verifies balances update correctly, (b) verifies a `ValueError` is raised on insufficient funds, (c) uses a test database (or `unittest.mock`) to avoid hitting a real DB.

(Solutions: ../../solutions/week-06/day-5.md)

## Self-check

- [ ] I can configure a SQLAlchemy engine with connection pooling
- [ ] I can write Mapped ORM models using SQLAlchemy 2.0 style
- [ ] I understand the difference between flush, commit, and rollback in a Session
- [ ] I can mix ORM and Core queries in the same session

## Supplementary links

- [SQLAlchemy 2.0 ORM Tutorial](https://docs.sqlalchemy.org/en/20/orm/quickstart.html)
- [SQLAlchemy 2.0 — Mapped Columns](https://docs.sqlalchemy.org/en/20/orm/mapped_attributes.html)
- [psycopg 3 Documentation](https://www.psycopg.org/psycopg3/docs/)
- [Alembic Tutorial](https://alembic.sqlalchemy.org/en/latest/tutorial.html)
