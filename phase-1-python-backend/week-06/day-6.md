# Phase 1 · Week 06 · Day 6 (Saturday) — Design and Implement the PostgreSQL Schema

> Objective: Design the full Financial Transaction Management System schema from requirements, implement it with migrations, seed data, and verify with queries  ·  Time: ~4 hr  ·  Prereqs: [[day-3]], [[day-4]], [[day-5]]

## Overview

Today you build the schema from scratch using Alembic migrations — the production-correct way. No raw `CREATE TABLE` dumps. Every change is versioned and reversible.

**Deliverables:**
- `alembic/` directory with a complete initial migration
- All five tables with constraints, indexes
- Seed script (`scripts/seed.py`) populating realistic test data
- Five verification queries that exercise JOINs, aggregations, and indexes

---

## Part 1 — Project setup (45 min)

### Directory layout

```
financial_app/
├── src/
│   └── financial_app/
│       ├── __init__.py
│       ├── config.py
│       ├── database.py
│       ├── models.py
│       └── repository.py
├── alembic/
│   ├── env.py
│   └── versions/
├── scripts/
│   └── seed.py
├── tests/
│   └── test_models.py
├── alembic.ini
├── pyproject.toml
└── .env.example
```

### pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "financial-app"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "sqlalchemy>=2.0",
    "psycopg[binary]>=3.1",
    "alembic>=1.13",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-postgresql",
    "ruff",
]
```

### .env.example

```
# Copy to .env and fill in your values
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@localhost:5432/financial_app
```

### config.py

```python
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    def get_database_url(self) -> str:
        url = self.database_url or os.getenv("DATABASE_URL", "")
        if not url:
            raise ValueError("DATABASE_URL must be set in environment or .env")
        return url


settings = Settings()
```

---

## Part 2 — Alembic setup and initial migration (45 min)

### Initialize Alembic

```bash
alembic init alembic
```

### alembic/env.py (key section)

```python
import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# Import your models so Alembic can detect them
from financial_app.models import Base  # noqa: F401

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_url() -> str:
    url = os.getenv("DATABASE_URL", "")
    if not url:
        raise ValueError("DATABASE_URL not set")
    return url


def run_migrations_offline() -> None:
    url = get_url()
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    from sqlalchemy import create_engine
    connectable = create_engine(get_url(), poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

### Generate and write migration

```bash
alembic revision --autogenerate -m "initial_schema"
```

Review the generated file in `alembic/versions/`. Autogenerate catches table/column additions; manually add your index creation calls if they don't appear.

### Manual migration additions (indexes)

```python
# Inside upgrade():
op.create_index("idx_accounts_customer_id", "accounts", ["customer_id"])
op.create_index(
    "idx_transactions_account_created",
    "transactions",
    ["account_id", sa.text("created_at DESC")],
)
op.create_index(
    "idx_transactions_pending",
    "transactions",
    ["account_id", "created_at"],
    postgresql_where=sa.text("status = 'pending'"),
)
op.create_index(
    "idx_transactions_report",
    "transactions",
    ["account_id", sa.text("created_at DESC")],
    postgresql_include=["amount", "direction", "category_id"],
)

# Inside downgrade():
op.drop_index("idx_transactions_report", table_name="transactions")
op.drop_index("idx_transactions_pending", table_name="transactions")
op.drop_index("idx_transactions_account_created", table_name="transactions")
op.drop_index("idx_accounts_customer_id", table_name="accounts")
```

### Apply migration

```bash
alembic upgrade head
```

---

## Part 3 — Seed script (45 min)

```python
# scripts/seed.py
"""Populate the database with realistic test data."""
import os
import logging
import random
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from financial_app.models import (
    Customer, Account, TransactionCategory, Transaction, AuditLog
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise SystemExit("DATABASE_URL environment variable not set")

CATEGORIES = [
    "Salary", "Rent", "Groceries", "Utilities", "Entertainment",
    "Healthcare", "Transfer", "Refund", "Uncategorized",
]

ACCOUNT_TYPES = ["checking", "savings", "investment"]


def seed(session: Session) -> None:
    # Categories
    cat_objs = {
        name: TransactionCategory(name=name)
        for name in CATEGORIES
    }
    session.add_all(cat_objs.values())
    session.flush()

    customers: list[Customer] = []
    for i in range(1, 21):
        c = Customer(
            full_name=f"Customer {i:02d}",
            email=f"customer{i:02d}@example.test",
            phone=f"+1555{i:07d}",
            is_active=(i % 10 != 0),
        )
        session.add(c)
        customers.append(c)
    session.flush()

    accounts: list[Account] = []
    for c in customers:
        n_accounts = random.randint(1, 3)
        for j in range(n_accounts):
            a = Account(
                customer_id=c.customer_id,
                account_number=f"ACC-{c.customer_id:04d}-{j+1:02d}",
                account_type=random.choice(ACCOUNT_TYPES),
                balance=Decimal(str(random.uniform(100, 50000))).quantize(Decimal("0.01")),
            )
            session.add(a)
            accounts.append(a)
    session.flush()

    now = datetime.now(tz=timezone.utc)
    for _ in range(500):
        account = random.choice(accounts)
        direction = random.choice(["credit", "debit"])
        amount = Decimal(str(random.uniform(1, 2000))).quantize(Decimal("0.01"))
        created_at = now - timedelta(days=random.randint(0, 365))
        t = Transaction(
            account_id=account.account_id,
            category_id=random.choice(list(cat_objs.values())).category_id,
            amount=amount,
            direction=direction,
            description=f"Test {direction} transaction",
            status=random.choice(["completed", "completed", "completed", "pending", "failed"]),
            created_at=created_at,
        )
        session.add(t)

    session.commit()
    logger.info("Seed complete")


if __name__ == "__main__":
    engine = create_engine(DATABASE_URL)
    with Session(engine) as session:
        seed(session)
```

---

## Part 4 — Verification queries (45 min)

Run each query and confirm expected output.

```sql
-- V1: Account count per customer, with total balance
SELECT
    c.full_name,
    COUNT(a.account_id)     AS accounts,
    SUM(a.balance)          AS total_balance,
    MAX(a.balance)          AS max_balance
FROM customers c
LEFT JOIN accounts a ON a.customer_id = c.customer_id
GROUP BY c.customer_id, c.full_name
ORDER BY total_balance DESC NULLS LAST
LIMIT 10;

-- V2: Transaction volume per category (last 90 days)
SELECT
    COALESCE(tc.name, 'Uncategorized') AS category,
    COUNT(*)                           AS tx_count,
    SUM(t.amount)                      AS total_amount,
    AVG(t.amount)::numeric(12,2)       AS avg_amount
FROM transactions t
LEFT JOIN transaction_categories tc ON tc.category_id = t.category_id
WHERE t.created_at >= NOW() - INTERVAL '90 days'
GROUP BY tc.name
ORDER BY total_amount DESC;

-- V3: Running balance for account 1 (last 20 transactions)
SELECT
    t.transaction_id,
    t.created_at::date,
    t.direction,
    t.amount,
    SUM(
        CASE WHEN t.direction = 'credit' THEN t.amount ELSE -t.amount END
    ) OVER (
        PARTITION BY t.account_id
        ORDER BY t.created_at
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_balance
FROM transactions t
WHERE t.account_id = 1
ORDER BY t.created_at DESC
LIMIT 20;

-- V4: Top customer per month by transaction volume
WITH monthly AS (
    SELECT
        DATE_TRUNC('month', t.created_at) AS month,
        c.customer_id,
        c.full_name,
        SUM(t.amount) AS monthly_volume,
        RANK() OVER (
            PARTITION BY DATE_TRUNC('month', t.created_at)
            ORDER BY SUM(t.amount) DESC
        ) AS rnk
    FROM transactions t
    JOIN accounts a ON a.account_id = t.account_id
    JOIN customers c ON c.customer_id = a.customer_id
    GROUP BY 1, c.customer_id, c.full_name
)
SELECT month, full_name, monthly_volume
FROM monthly
WHERE rnk = 1
ORDER BY month DESC;

-- V5: EXPLAIN — verify index usage on account+date query
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT transaction_id, amount, direction, category_id
FROM transactions
WHERE account_id = 3
  AND created_at >= NOW() - INTERVAL '30 days'
ORDER BY created_at DESC
LIMIT 50;
-- Expected: Index Only Scan on idx_transactions_report
```

---

## Part 5 — Self-directed practice (30 min)

1. Add an `AuditLog` model (see Day 3 schema) and a SQLAlchemy event listener (`event.listen(Session, 'after_flush', ...)`) that auto-inserts audit records when accounts change.

2. Write a pytest test that: creates a customer and two accounts in a test session, runs a transfer, asserts balances, then rolls back.

3. Run `ANALYZE` and then re-run V5 — observe whether the cost estimate changes.

(Solutions: ../../solutions/week-06/day-6.md)

## Self-check

- [ ] I can initialize Alembic and write a migration with both upgrade and downgrade
- [ ] I can seed a database with Python (no raw SQL scripts)
- [ ] I can write verification queries mixing window functions and CTEs
- [ ] I can read EXPLAIN ANALYZE to confirm index-only scans

## Supplementary links

- [Alembic — Auto-generating Migrations](https://alembic.sqlalchemy.org/en/latest/autogenerate.html)
- [SQLAlchemy — ORM Events](https://docs.sqlalchemy.org/en/20/orm/events.html)
- [Use The Index, Luke — Index-Only Scans](https://use-the-index-luke.com/sql/clustering/index-only-scan-covering-index)
