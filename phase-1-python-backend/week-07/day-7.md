# Phase 1 · Week 07 · Day 7 — Integration Testing, Debugging & Assessment Prep

> Objective: Write integration tests against a real test DB and fix common FastAPI bugs  ·  Time: ~3 hr  ·  Prereqs: [[week-07/day-6]]

## Concept

### Unit tests vs. integration tests

| | Unit | Integration |
|---|---|---|
| Scope | One function/class | Multiple layers end-to-end |
| DB | Mocked | Real (test) DB |
| Speed | Fast (<1 ms each) | Slower (10–100 ms, includes I/O) |
| Catches | Logic bugs | Wiring bugs, query bugs, migration drift |

Both are needed. Unit tests catch regressions fast in the inner loop. Integration tests catch bugs that only appear when layers talk to each other — N+1 queries, missing `await`, schema mismatches.

### pytest-asyncio for async tests

`pytest-asyncio` adds support for `async def test_*` functions. With `asyncio_mode = "auto"` in `pyproject.toml`, every `async def test_*` is automatically collected as an async test — no `@pytest.mark.asyncio` decorator needed.

Async fixtures use `@pytest_asyncio.fixture`. Session-scoped async fixtures require `asyncio_mode = "auto"` and `scope="session"`.

### Test database setup

Never run integration tests against the development database. The test DB:
1. Has a separate `TEST_DATABASE_URL` (different DB name, same engine type)
2. Is created fresh at the start of the test session (`create_all`)
3. Is torn down at the end (`drop_all`) or between tests (transaction rollback)

The transaction rollback pattern is fastest: each test runs inside a transaction that is rolled back at the end. No data persists between tests. No truncation needed.

### Common FastAPI bugs

**N+1 query problem**
Accessing a relationship attribute inside a loop triggers one query per row:
```python
# Bug: one SELECT per customer
for customer in customers:
    print(customer.accounts)  # lazy load fires here — but async won't allow it
```
In async SQLAlchemy, lazy loading raises `MissingGreenlet`. Fix: use `selectinload` or `joinedload` in the original query.

**Missing `await`**
```python
# Bug: result is a coroutine object, not a row
result = session.execute(select(User))   # missing await
user = result.scalar_one_or_none()       # AttributeError: coroutine has no attr
```
The symptom is often `AttributeError` on a coroutine object, not a clear "missing await" message.

**Response model leaks**
If `response_model` is omitted and the endpoint returns a SQLAlchemy object directly, Pydantic does not filter fields — private fields (`hashed_password`) may appear. Always specify `response_model`.

**Unhandled `IntegrityError`**
SQLAlchemy wraps DB constraint violations in `IntegrityError`. If not caught, FastAPI returns a `500`. Always catch and map to a 4xx response.

## How it works internally

### Transaction rollback fixture pattern

```
session-scoped engine (create_all once)
    |
function-scoped connection
    |
BEGIN SAVEPOINT              ← test starts here
    |
test body runs               ← inserts/updates inside the transaction
    |
ROLLBACK TO SAVEPOINT        ← test teardown wipes all changes
```

The DB never sees a `COMMIT` — every row inserted during the test disappears on rollback. The schema (tables) persists for the whole session.

### SQLAlchemy `echo=True` for debugging

`create_async_engine(url, echo=True)` logs every SQL statement and its parameters to stdout. Use this during debugging to see exactly what queries are executing. Turn off in production — it logs parameter values including sensitive data.

### pytest `-s` flag

By default, pytest captures stdout. `-s` disables capture so `print()` and logging output appear in the terminal. Combine with `-v` for verbose test names: `pytest -sv tests/integration/`.

## Worked example

```python
# tests/conftest.py  (full integration test setup)
import os
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from starlette.testclient import TestClient

from src.finops.main import app
from src.finops.database import get_session
from src.finops.models import Base, User, Customer, Account
from src.finops.auth import create_access_token, hash_password

TEST_DATABASE_URL: str = os.getenv(
    "TEST_DATABASE_URL", "sqlite+aiosqlite:///./test.db"
)

# Session-scoped engine — schema created once per test run
test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestAsyncSession = async_sessionmaker(test_engine, expire_on_commit=False)


@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_database():
    """Create all tables before the test session, drop after."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await test_engine.dispose()


@pytest_asyncio.fixture
async def db_session():
    """Each test gets a session that rolls back on teardown."""
    async with test_engine.connect() as conn:
        await conn.begin()
        session = AsyncSession(bind=conn, expire_on_commit=False)
        try:
            yield session
        finally:
            await session.close()
            await conn.rollback()


@pytest.fixture
def client(db_session: AsyncSession) -> TestClient:
    """Override get_session so endpoints use the rollback session."""
    async def _override():
        yield db_session

    app.dependency_overrides[get_session] = _override
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def seeded_user(db_session: AsyncSession) -> User:
    user = User(
        email="testuser@example.com",
        hashed_password=hash_password("testpassword1"),
        role="viewer",
    )
    db_session.add(user)
    await db_session.flush()  # flush to get the id without committing
    return user


@pytest_asyncio.fixture
async def seeded_customer(db_session: AsyncSession) -> Customer:
    customer = Customer(
        full_name="Test Customer",
        email="customer@example.com",
        phone="+49123456789",
    )
    db_session.add(customer)
    await db_session.flush()
    return customer


@pytest_asyncio.fixture
async def seeded_account(db_session: AsyncSession, seeded_customer: Customer) -> Account:
    account = Account(
        account_number="ACC-001",
        account_type="checking",
        balance=1000,
        customer_id=seeded_customer.id,
    )
    db_session.add(account)
    await db_session.flush()
    return account


@pytest.fixture
def auth_headers(seeded_user: User) -> dict[str, str]:
    token = create_access_token({"sub": str(seeded_user.id), "role": "viewer"})
    return {"Authorization": f"Bearer {token}"}
```

```python
# tests/integration/test_customers_integration.py
import pytest
from starlette.testclient import TestClient

from src.finops.models import Customer


def test_create_customer_persists(
    client: TestClient, auth_headers: dict
) -> None:
    """Full round-trip: POST creates, GET retrieves the same record."""
    payload = {
        "full_name": "Integration Test User",
        "email": "integration@example.com",
        "phone": "+19998887777",
    }
    create_resp = client.post("/customers", json=payload, headers=auth_headers)
    assert create_resp.status_code == 201
    customer_id = create_resp.json()["id"]

    get_resp = client.get(f"/customers/{customer_id}", headers=auth_headers)
    assert get_resp.status_code == 200
    assert get_resp.json()["email"] == payload["email"]


def test_duplicate_email_returns_409(
    client: TestClient, auth_headers: dict, seeded_customer: Customer
) -> None:
    """Seeded customer's email already in DB — should get 409."""
    payload = {
        "full_name": "Duplicate",
        "email": seeded_customer.email,  # already exists
    }
    response = client.post("/customers", json=payload, headers=auth_headers)
    assert response.status_code == 409
    assert "already registered" in response.json()["detail"]


def test_get_nonexistent_customer(client: TestClient, auth_headers: dict) -> None:
    response = client.get("/customers/999999", headers=auth_headers)
    assert response.status_code == 404
```

```python
# tests/integration/test_transactions_integration.py
from decimal import Decimal

import pytest
from starlette.testclient import TestClient

from src.finops.models import Account


def test_credit_transaction_updates_balance(
    client: TestClient,
    auth_headers: dict,
    seeded_account: Account,
    db_session,
) -> None:
    """POST /transactions credit should increase account balance."""
    initial_balance = float(seeded_account.balance)

    response = client.post(
        "/transactions",
        json={
            "amount": "200.00",
            "direction": "credit",
            "account_id": seeded_account.id,
        },
        headers=auth_headers,
    )
    assert response.status_code == 201

    # Re-fetch account to verify balance update
    import asyncio
    from sqlalchemy import select
    from src.finops.models import Account as AccountModel

    async def fetch():
        result = await db_session.execute(
            select(AccountModel).where(AccountModel.id == seeded_account.id)
        )
        return result.scalar_one()

    updated = asyncio.get_event_loop().run_until_complete(fetch())
    assert float(updated.balance) == initial_balance + 200.0


def test_transaction_nonexistent_account(
    client: TestClient, auth_headers: dict
) -> None:
    response = client.post(
        "/transactions",
        json={"amount": "10.00", "direction": "debit", "account_id": 999999},
        headers=auth_headers,
    )
    assert response.status_code == 404
```

### Debugging: logging middleware

```python
# src/finops/main.py  — add middleware for request/response logging
import time
import logging
from fastapi import Request

logger = logging.getLogger(__name__)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s → %d (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response
```

### Debugging: SQLAlchemy echo

```python
# Toggle echo on the test engine to see queries during debugging
test_engine = create_async_engine(TEST_DATABASE_URL, echo=True)

# Run tests with output visible:
# pytest -sv tests/integration/test_customers_integration.py
```

## Exercises

1. The `test_credit_transaction_updates_balance` test uses `asyncio.get_event_loop().run_until_complete()` inside a sync test — this is fragile. Rewrite it as a fully async test using `pytest-asyncio` and `AsyncClient` from `httpx` (with `ASGITransport`). Ensure the rollback fixture still works.

2. The `seeded_account` fixture creates one account with a fixed `account_number = "ACC-001"`. If two tests use this fixture in the same session and the rollback is not working correctly, the second test will fail with a unique constraint error. Verify the rollback is working by running `pytest -sv` and inspecting the SQL log. Then add a test that deliberately causes an `IntegrityError` (insert duplicate `account_number`) and assert it raises a `409`, not a `500`.

3. Implement the N+1 detection exercise: add a `GET /customers/{id}/accounts` endpoint that returns all accounts for a customer. First implement it with a plain `select(Account).where(Account.customer_id == id)`. Then instrument the test with SQLAlchemy's `event.listen` on the engine to count executed statements, and assert that fetching 1 customer with 5 accounts issues exactly 1 query (requires using `selectinload`).

(Solutions: ../solutions/week-07/day-7.md)

## Assessment prep — Week 7 checklist

Work through each item. If you cannot answer from memory, revisit the relevant day.

### JWT & Auth (Day 4)
- [ ] Explain what the JWT signature guarantees and what it does not guarantee
- [ ] Describe the `header.payload.signature` format and what each section contains
- [ ] Write `create_access_token` and `get_current_user` from memory
- [ ] Explain why `OAuth2PasswordBearer` does not validate the token itself
- [ ] Describe the two-layer `get_current_user` → `require_role` dependency chain

### Async SQLAlchemy (Day 4)
- [ ] Explain what happens if you forget `await` before `session.execute()`
- [ ] Explain why `expire_on_commit=False` is required for async sessions
- [ ] Describe the difference between `scalar_one()`, `scalar_one_or_none()`, and `scalars().all()`

### API Testing (Day 5)
- [ ] Write a `conftest.py` with `app_client`, `auth_headers`, and `admin_headers` fixtures
- [ ] Explain the difference in scope between `"session"` and `"function"` fixtures
- [ ] Write `pytest.mark.parametrize` for testing multiple invalid input cases
- [ ] Explain how FastAPI generates the OpenAPI spec from route and Pydantic annotations

### Build project (Day 6)
- [ ] Reproduce the full project directory structure from memory
- [ ] Write a `Customer` SQLAlchemy model with `Mapped` type annotations
- [ ] Write a `CustomerCreate` Pydantic schema with `EmailStr` and field constraints
- [ ] Wire a `lifespan` handler for `create_all` at startup

### Integration testing (Day 7)
- [ ] Explain the transaction rollback fixture pattern and why it is faster than truncation
- [ ] Describe three common FastAPI bugs: N+1, missing `await`, response model leak
- [ ] Use `echo=True` and `pytest -s` to debug a failing integration test

## Self-check

- [ ] I can set up a test database with session-scoped schema creation and per-test rollback
- [ ] I can write integration tests that seed data and assert full request-response flows
- [ ] I can identify and fix N+1 queries using `selectinload`
- [ ] I can debug async SQLAlchemy issues using `echo=True` and logging middleware
- [ ] I can answer every item in the assessment prep checklist without reference
