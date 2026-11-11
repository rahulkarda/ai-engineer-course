# Week 07 · Day 7 — Solutions

Integration testing with pytest-asyncio and diagnosing N+1 queries.

---

## Exercise 1 — Async test database fixture + config

### `pyproject.toml` configuration

```toml
# pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"          # pytest-asyncio: treats all async tests as auto
testpaths  = ["tests"]
addopts    = "-v --tb=short"

[tool.pytest.ini_options.markers]
integration = "mark test as integration (uses real async DB)"
unit        = "mark test as unit (mocks all I/O)"
```

> **`asyncio_mode = "auto"`** — all `async def test_*` and `async def` fixtures
> are handled by pytest-asyncio without needing an explicit `@pytest.mark.asyncio`
> decorator on every test.

### Async database fixture

```python
# tests/integration/conftest.py
import os
import pytest
import pytest_asyncio
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from app.db.base import Base  # Base = declarative_base() shared by all models

# Use an isolated in-memory SQLite database per test session.
# StaticPool ensures all async calls share one connection so the schema
# persists within the session (aiosqlite creates per-connection memory DBs
# by default, which would lose the schema between queries).
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

_engine = create_async_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
    echo=False,  # set True to see SQL in test output during debugging
)

_session_factory = async_sessionmaker(
    bind=_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


@pytest_asyncio.fixture(scope="session")
async def db_engine():
    """Create schema once for the entire test session, drop at teardown."""
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield _engine

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await _engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    """
    Yield a transactional async session.

    Each test runs inside an implicit transaction that is rolled back
    at teardown — tables are not truncated, rows are not deleted.
    This keeps tests fast and isolated.
    """
    async with _session_factory() as session:
        yield session
        # Rollback ensures no state leaks between tests.
        await session.rollback()
```

---

## Exercise 2 — Full integration test: customer → transaction → retrieval

```python
# tests/integration/test_transaction_flow.py
import pytest
import pytest_asyncio
from decimal import Decimal
from collections.abc import AsyncGenerator

from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession

from app.main import app
from app.db.session import get_db


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Override get_db with the test session, then yield an AsyncClient."""

    async def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.mark.integration
async def test_full_transaction_flow(client: AsyncClient) -> None:
    """
    Integration test covering:
      1. Create a customer (POST /customers).
      2. Create a transaction for that customer (POST /transactions).
      3. Retrieve the transaction (GET /transactions/{id}).
      4. Assert all fields round-trip correctly.
    """
    # ------------------------------------------------------------------
    # Step 1: Create a customer.
    # ------------------------------------------------------------------
    customer_payload = {
        "name": "Integration Tester",
        "email": "integration@example.com",
    }
    create_customer_resp = await client.post("/customers", json=customer_payload)
    assert create_customer_resp.status_code == 201, create_customer_resp.text

    customer = create_customer_resp.json()
    customer_id: int = customer["id"]
    assert customer["email"] == customer_payload["email"]

    # ------------------------------------------------------------------
    # Step 2: Create a transaction linked to the customer.
    # ------------------------------------------------------------------
    txn_payload = {
        "customer_id": customer_id,
        "amount": "149.99",
        "currency": "USD",
        "description": "Integration test purchase",
    }
    create_txn_resp = await client.post("/transactions", json=txn_payload)
    assert create_txn_resp.status_code == 201, create_txn_resp.text

    created_txn = create_txn_resp.json()
    txn_id: int = created_txn["id"]
    assert created_txn["customer_id"] == customer_id
    assert Decimal(created_txn["amount"]) == Decimal("149.99")
    assert created_txn["currency"] == "USD"

    # ------------------------------------------------------------------
    # Step 3: Retrieve the transaction by ID.
    # ------------------------------------------------------------------
    get_txn_resp = await client.get(f"/transactions/{txn_id}")
    assert get_txn_resp.status_code == 200, get_txn_resp.text

    fetched_txn = get_txn_resp.json()

    # ------------------------------------------------------------------
    # Step 4: Assert fields match across all three representations.
    # ------------------------------------------------------------------
    assert fetched_txn["id"] == txn_id
    assert fetched_txn["customer_id"] == customer_id
    assert Decimal(fetched_txn["amount"]) == Decimal(txn_payload["amount"])
    assert fetched_txn["currency"] == txn_payload["currency"]
    assert fetched_txn["description"] == txn_payload["description"]
```

---

## Exercise 3 — Diagnosing and fixing the N+1 query

### The N+1 problem

The following code loads every customer with a separate query per customer to
fetch their transactions — that is 1 query to list customers and N additional
queries (one per customer), hence "N+1":

```python
# BAD: N+1 query pattern
# models/customer.py assumes a relationship:
#   transactions = relationship("Transaction", back_populates="customer")

from sqlalchemy import select
from sqlalchemy.orm import Session
from models.customer import Customer


def get_customers_with_transactions_bad(session: Session) -> list[dict]:
    """
    Fetches customers, then triggers a lazy load for each customer's
    transactions individually.  With 1000 customers this is 1001 SQL queries.
    """
    customers = session.execute(select(Customer)).scalars().all()

    result = []
    for customer in customers:
        # This line triggers a new SELECT per customer because SQLAlchemy
        # lazy-loads the relationship on first access.
        txn_count = len(customer.transactions)  # <- implicit SELECT here
        result.append({"id": customer.id, "transaction_count": txn_count})

    return result
```

**Why it happens:** SQLAlchemy's default relationship loading strategy is
`lazy="select"`, meaning the related collection is fetched on-demand (lazily)
when the attribute is first accessed.  Inside a loop this becomes one extra
query per parent row.

### Fix 1 — `selectinload` (recommended for async)

```python
# GOOD: selectinload — two queries total, regardless of result set size.
# Query 1: SELECT * FROM customers
# Query 2: SELECT * FROM transactions WHERE customer_id IN (1, 2, 3, ...)

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from models.customer import Customer


async def get_customers_with_transactions_selectinload(
    session: AsyncSession,
) -> list[Customer]:
    """
    Load customers and their transactions in two queries.

    selectinload issues a single IN-clause query for all related rows,
    avoiding the per-row round-trip.  It is safe with async sessions because
    it does not require an open cursor while iterating results.
    """
    result = await session.execute(
        select(Customer).options(selectinload(Customer.transactions))
    )
    return result.scalars().all()
```

### Fix 2 — `joinedload` (single query with JOIN)

```python
# GOOD: joinedload — one query with a LEFT OUTER JOIN.
# Useful when you always need the related data and the result set is small.

from sqlalchemy.orm import joinedload


async def get_customers_with_transactions_joinedload(
    session: AsyncSession,
) -> list[Customer]:
    """
    Load customers and transactions in a single LEFT OUTER JOIN query.

    Trade-off vs selectinload:
    - joinedload: one query, but result set can be large (cartesian product)
      if there are many transactions per customer.
    - selectinload: two queries, but memory footprint is predictable.

    For large datasets prefer selectinload.
    """
    result = await session.execute(
        select(Customer)
        .options(joinedload(Customer.transactions))
        .distinct()  # avoid duplicate Customer rows from the JOIN
    )
    return result.unique().scalars().all()
```

### Comparison table

| Strategy         | Queries | Memory       | Async-safe | Best for                            |
|------------------|---------|--------------|------------|-------------------------------------|
| Lazy (default)   | 1 + N   | Low per obj  | No*        | Single object, relationship rarely needed |
| `selectinload`   | 2       | Moderate     | Yes        | Lists, async handlers               |
| `joinedload`     | 1       | High (JOIN)  | Yes        | Small datasets, always-needed data  |
| `subqueryload`   | 2       | Moderate     | No*        | Sync only                           |

> *Lazy loading and subqueryload require an active session at attribute-access time,
> which breaks in async context after the session is closed.  Always prefer
> `selectinload` in async FastAPI routes.
