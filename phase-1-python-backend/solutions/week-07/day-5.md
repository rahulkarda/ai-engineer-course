# Week 07 · Day 5 — Solutions

API testing with pytest, TestClient / AsyncClient, and OpenAPI metadata.

---

## Exercise 1 — TestClient fixture with overridden `get_db`

Use an in-memory SQLite async engine so tests never touch the real database.

```python
# tests/conftest.py
import os
import pytest
import pytest_asyncio
from collections.abc import AsyncGenerator

from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db.session import get_db
from app.db.base import Base  # declarative Base that all models use


# ---------------------------------------------------------------------------
# In-memory SQLite engine — one engine per test session.
# StaticPool keeps a single connection so the schema persists across
# multiple async calls within the same test.
# ---------------------------------------------------------------------------
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


@pytest_asyncio.fixture(scope="session", autouse=True)
async def create_tables():
    """Create all ORM tables once per test session, drop them at teardown."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Yield a transactional async session, then roll back so each test is
    isolated without hitting disk.
    """
    async with TestingSessionLocal() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """
    AsyncClient wired to the FastAPI app with the get_db dependency
    overridden to return the test session.

    The override pattern:
        app.dependency_overrides[original_dep] = replacement_callable

    FastAPI resolves the replacement at request time, so the test session
    is injected wherever get_db was used.
    """

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        yield ac

    # Always clean up overrides so other tests are unaffected.
    app.dependency_overrides.clear()
```

**`pyproject.toml` / `pytest.ini` snippet required:**

```toml
# pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"          # pytest-asyncio: auto-detect async tests
testpaths = ["tests"]
```

---

## Exercise 2 — Tests for `POST /customers`

```python
# tests/test_customers.py
import pytest
from httpx import AsyncClient


# ---------------------------------------------------------------------------
# Happy path — 201 Created
# ---------------------------------------------------------------------------
async def test_create_customer_success(client: AsyncClient) -> None:
    payload = {
        "name": "Alice Wonderland",
        "email": "alice@example.com",
        "phone": "+1-555-0100",
    }
    response = await client.post("/customers", json=payload)
    assert response.status_code == 201
    body = response.json()
    assert body["email"] == payload["email"]
    assert "id" in body


# ---------------------------------------------------------------------------
# Duplicate email — 409 Conflict
# ---------------------------------------------------------------------------
async def test_create_customer_duplicate_email(client: AsyncClient) -> None:
    payload = {"name": "Bob", "email": "bob@example.com"}
    await client.post("/customers", json=payload)  # first insert succeeds
    response = await client.post("/customers", json=payload)  # duplicate
    assert response.status_code == 409
    assert "already exists" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Validation errors — 422 Unprocessable Entity
# Parametrize covers multiple bad payloads in a single test function.
# ---------------------------------------------------------------------------
BAD_PAYLOADS = [
    pytest.param({}, id="empty_body"),
    pytest.param({"name": "No Email"}, id="missing_email"),
    pytest.param({"name": "", "email": "valid@example.com"}, id="empty_name"),
    pytest.param({"name": "Bad", "email": "not-an-email"}, id="invalid_email_format"),
    pytest.param(
        {"name": "X" * 300, "email": "toolong@example.com"},
        id="name_too_long",
    ),
]


@pytest.mark.parametrize("payload", BAD_PAYLOADS)
async def test_create_customer_validation_error(
    client: AsyncClient,
    payload: dict,
) -> None:
    response = await client.post("/customers", json=payload)
    assert response.status_code == 422
    # Pydantic v2 error bodies always include a "detail" list.
    assert isinstance(response.json()["detail"], list)
```

---

## Exercise 3 — Authenticated test fixture

```python
# tests/conftest.py (additions)
import pytest_asyncio
from httpx import AsyncClient


@pytest_asyncio.fixture
async def auth_client(client: AsyncClient) -> AsyncClient:
    """
    Return an AsyncClient pre-loaded with a valid JWT.

    Steps:
      1. Register a test user via the API (or insert directly into DB).
      2. POST to /auth/login to receive a token.
      3. Attach the token as a default header on the client.
    """
    # 1. Create a test user.
    register_payload = {
        "username": "testuser",
        "email": "testuser@example.com",
        "password": "Str0ngP@ssw0rd!",
        "role": "user",
    }
    reg_response = await client.post("/auth/register", json=register_payload)
    assert reg_response.status_code in (200, 201), (
        f"Test user registration failed: {reg_response.text}"
    )

    # 2. Log in to obtain a JWT.
    login_payload = {
        "username": register_payload["username"],
        "password": register_payload["password"],
    }
    login_response = await client.post(
        "/auth/login",
        data=login_payload,  # OAuth2PasswordRequestForm expects form data
    )
    assert login_response.status_code == 200, (
        f"Login failed: {login_response.text}"
    )
    token = login_response.json()["access_token"]

    # 3. Attach the Bearer token as a default header.
    client.headers.update({"Authorization": f"Bearer {token}"})
    return client


# ---------------------------------------------------------------------------
# Example usage of auth_client in a test:
# ---------------------------------------------------------------------------
async def test_authenticated_endpoint(auth_client: AsyncClient) -> None:
    response = await auth_client.get("/users/me")
    assert response.status_code == 200
    assert response.json()["username"] == "testuser"
```

---

## Exercise 4 — OpenAPI metadata on a router

```python
# routers/customers.py
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated

from db.session import get_db
from models.customer import Customer
from schemas.customer import CustomerCreate, CustomerRead, CustomerUpdate

router = APIRouter(
    prefix="/customers",
    tags=["Customers"],          # groups all endpoints under "Customers" in the docs
)


@router.get(
    "",
    summary="List all customers",
    response_description="Paginated list of customers",
    responses={
        200: {"description": "Successfully retrieved customer list"},
        401: {"description": "Authentication required"},
    },
)
async def list_customers(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(20, le=100, description="Maximum records to return"),
    session: Annotated[AsyncSession, Depends(get_db)] = None,
) -> list[CustomerRead]:
    ...


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Create a new customer",
    response_description="The newly created customer record",
    responses={
        201: {"description": "Customer created successfully"},
        409: {"description": "A customer with this email already exists"},
        422: {
            "description": (
                "Validation error — check that all required fields are present "
                "and formatted correctly (e.g. valid email, non-empty name)."
            )
        },
    },
)
async def create_customer(
    payload: CustomerCreate,
    session: Annotated[AsyncSession, Depends(get_db)] = None,
) -> CustomerRead:
    ...


@router.get(
    "/{customer_id}",
    summary="Retrieve a single customer by ID",
    response_description="Customer record matching the given ID",
    responses={
        200: {"description": "Customer found"},
        404: {"description": "No customer with this ID exists"},
    },
)
async def get_customer(
    customer_id: int,
    session: Annotated[AsyncSession, Depends(get_db)] = None,
) -> CustomerRead:
    ...


@router.put(
    "/{customer_id}",
    summary="Update a customer (full replace)",
    response_description="The updated customer record",
)
async def update_customer(
    customer_id: int,
    payload: CustomerUpdate,
    session: Annotated[AsyncSession, Depends(get_db)] = None,
) -> CustomerRead:
    ...


@router.delete(
    "/{customer_id}/export",
    summary="[DEPRECATED] Export customer as CSV",
    response_description="CSV download of the customer record",
    deprecated=True,   # renders a deprecation badge in Swagger UI
    responses={
        200: {"description": "CSV file returned (use /reports/customers instead)"},
    },
)
async def export_customer_csv(customer_id: int) -> dict:
    """
    **Deprecated.** This endpoint will be removed in v2.0.
    Use `GET /reports/customers/{customer_id}` instead.
    """
    ...
```
