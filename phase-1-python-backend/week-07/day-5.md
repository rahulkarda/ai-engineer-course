# Phase 1 · Week 07 · Day 5 — API Testing & OpenAPI Documentation

> Objective: Write pytest tests for FastAPI endpoints and understand OpenAPI spec generation  ·  Time: ~1 hr  ·  Prereqs: [[week-07/day-4]]

## Concept

### TestClient vs AsyncClient

FastAPI provides two test transport modes:

| | `TestClient` (Starlette) | `AsyncClient` (httpx) |
|---|---|---|
| Sync/async | Synchronous | Async |
| Event loop | Spins up internally | Uses pytest-asyncio loop |
| Use for | Simple endpoint tests | Async test fixtures, `lifespan` hooks |
| Import from | `starlette.testclient` | `httpx` + `asgi-transport` |

`TestClient` wraps the ASGI app in a thread and runs synchronously — it works for most cases. Use `AsyncClient` when you need true async test helpers or when testing `lifespan` startup/shutdown.

```python
# Sync (simpler)
from starlette.testclient import TestClient
client = TestClient(app)
response = client.get("/health")

# Async (for async fixtures / lifespan)
import httpx
from httpx import ASGITransport

async with httpx.AsyncClient(
    transport=ASGITransport(app=app), base_url="http://test"
) as client:
    response = await client.get("/health")
```

### pytest fixtures for FastAPI

Fixtures compose. A common pattern:

```
app fixture  →  client fixture  →  authenticated_client fixture
```

Scope controls how often the fixture is set up/torn down:
- `"session"` — once per test run (DB schema creation, static config)
- `"function"` (default) — once per test (DB row insertion, token generation)

### Testing auth flows

Three cases always needed:
1. **Missing token** — expect `401`
2. **Invalid/expired token** — expect `401`
3. **Valid token, wrong role** — expect `403`

Validate the error response body shape, not just the status code.

### OpenAPI 3.x spec generation

FastAPI introspects route signatures at startup and builds an OpenAPI 3.x spec automatically:

- **Path + method** — from `@router.get("/path")`
- **Request body schema** — from Pydantic model type hint
- **Response schema** — from `response_model=`
- **Query/path/header params** — from function parameter types
- **Security scheme** — from `OAuth2PasswordBearer` registration

The spec is served at `/openapi.json`. Swagger UI at `/docs` and ReDoc at `/redoc` consume it.

### Customizing the spec

```python
@router.get(
    "/customers",
    tags=["customers"],
    summary="List all active customers",
    description="Returns a paginated list. Requires 'viewer' role or higher.",
    deprecated=False,
    response_description="Array of CustomerResponse objects",
)
```

Set `include_in_schema=False` to hide internal routes from the spec entirely.

## How it works internally

### How FastAPI generates the spec

At import time, each `@router.get(...)` call:
1. Registers the route with Starlette's router
2. Calls `get_openapi_operation_metadata()` — reads the function signature via `inspect`
3. Calls Pydantic's `model_json_schema()` on all Pydantic models in the signature
4. Merges schemas into `components/schemas`, adds `$ref` pointers in operations

On first `/openapi.json` request, the full spec dict is assembled and cached. The spec is a pure Python dict; FastAPI never writes a YAML file.

### What `parametrize` does

`pytest.mark.parametrize` expands one test function into N independent test items at collection time. Each item runs with its own setup/teardown cycle. Parametrize reduces duplication for boundary-condition testing.

```python
@pytest.mark.parametrize("amount,expected_status", [
    (100.00, 201),
    (-1.00,  422),   # negative amount — validation error
    (0.00,   422),   # zero amount
])
def test_create_transaction(client, amount, expected_status):
    ...
```

## Worked example

```python
# tests/conftest.py
import os
import pytest
from starlette.testclient import TestClient

from src.finops.main import app
from src.finops.auth import create_access_token, hash_password
from src.finops.database import AsyncSessionLocal
from src.finops.models import Base, User, Customer
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

# Override DB to use a test database
TEST_DB_URL = os.getenv("TEST_DATABASE_URL", "sqlite+aiosqlite:///./test.db")

test_engine = create_async_engine(TEST_DB_URL, echo=False)
TestAsyncSession = async_sessionmaker(test_engine, expire_on_commit=False)


@pytest.fixture(scope="session", autouse=True)
def event_loop_policy():
    # Needed for pytest-asyncio with session-scoped async fixtures
    import asyncio
    asyncio.get_event_loop_policy()


@pytest.fixture(scope="session")
def app_client():
    """One TestClient for the whole session."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def client(app_client):
    return app_client


@pytest.fixture
def viewer_token() -> str:
    return create_access_token({"sub": "1", "role": "viewer"})


@pytest.fixture
def admin_token() -> str:
    return create_access_token({"sub": "2", "role": "admin"})


@pytest.fixture
def auth_headers(viewer_token) -> dict[str, str]:
    return {"Authorization": f"Bearer {viewer_token}"}


@pytest.fixture
def admin_headers(admin_token) -> dict[str, str]:
    return {"Authorization": f"Bearer {admin_token}"}
```

```python
# tests/test_auth.py
import pytest
from starlette.testclient import TestClient


def test_login_success(client: TestClient) -> None:
    response = client.post(
        "/auth/login",
        data={"username": "alice@example.com", "password": "correct-horse"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"


def test_login_wrong_password(client: TestClient) -> None:
    response = client.post(
        "/auth/login",
        data={"username": "alice@example.com", "password": "wrong"},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect email or password"


def test_me_requires_auth(client: TestClient) -> None:
    response = client.get("/auth/me")
    assert response.status_code == 401


def test_me_returns_current_user(client: TestClient, auth_headers: dict) -> None:
    response = client.get("/auth/me", headers=auth_headers)
    assert response.status_code == 200
    body = response.json()
    assert "email" in body
    assert "hashed_password" not in body  # must never leak


def test_admin_only_forbidden_for_viewer(
    client: TestClient, auth_headers: dict
) -> None:
    response = client.get("/auth/admin-only", headers=auth_headers)
    assert response.status_code == 403


def test_admin_only_allowed_for_admin(
    client: TestClient, admin_headers: dict
) -> None:
    response = client.get("/auth/admin-only", headers=admin_headers)
    assert response.status_code == 200
```

```python
# tests/test_customers.py
import pytest
from starlette.testclient import TestClient


VALID_CUSTOMER = {
    "full_name": "Jane Doe",
    "email": "jane@example.com",
    "phone": "+49123456789",
}


def test_create_customer_success(client: TestClient, auth_headers: dict) -> None:
    response = client.post("/customers", json=VALID_CUSTOMER, headers=auth_headers)
    assert response.status_code == 201
    body = response.json()
    assert body["email"] == VALID_CUSTOMER["email"]
    assert "id" in body


def test_create_customer_unauthenticated(client: TestClient) -> None:
    response = client.post("/customers", json=VALID_CUSTOMER)
    assert response.status_code == 401


def test_create_customer_duplicate_email(
    client: TestClient, auth_headers: dict
) -> None:
    client.post("/customers", json=VALID_CUSTOMER, headers=auth_headers)
    response = client.post("/customers", json=VALID_CUSTOMER, headers=auth_headers)
    assert response.status_code == 409


@pytest.mark.parametrize(
    "payload,expected_status",
    [
        ({"email": "notanemail", "full_name": "X"}, 422),  # invalid email
        ({"email": "a@b.com"}, 422),                        # missing full_name
        ({}, 422),                                          # empty body
        ({"full_name": "X", "email": ""}, 422),            # empty email
    ],
)
def test_create_customer_validation(
    client: TestClient,
    auth_headers: dict,
    payload: dict,
    expected_status: int,
) -> None:
    response = client.post("/customers", json=payload, headers=auth_headers)
    assert response.status_code == expected_status


def test_get_customer_not_found(client: TestClient, auth_headers: dict) -> None:
    response = client.get("/customers/999999", headers=auth_headers)
    assert response.status_code == 404


def test_openapi_spec_has_customers_tag(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    paths = spec.get("paths", {})
    post_op = paths.get("/customers", {}).get("post", {})
    assert "customers" in post_op.get("tags", [])
```

### Customizing the OpenAPI spec

```python
# src/finops/main.py
from fastapi import FastAPI

app = FastAPI(
    title="FinOps API",
    version="1.0.0",
    description="Financial operations REST API — Week 7 build project",
    contact={"name": "Platform Engineering"},
    license_info={"name": "Internal"},
)

# Suppress internal health route from public docs
@app.get("/internal/ping", include_in_schema=False)
async def internal_ping() -> dict:
    return {"status": "ok"}
```

## Exercises

1. The `test_create_customer_duplicate_email` test posts the same customer twice. This works in isolation but may fail if other tests have already inserted the same email. Refactor using a `unique_email` fixture that generates a fresh email per test using `uuid4`.

2. Add a `GET /customers` endpoint (not yet in the worked example) that returns a list of customers. Write tests that: (a) assert an empty list when no customers exist, (b) assert the list length after seeding two customers, (c) verify pagination params `?skip=0&limit=10` appear in the OpenAPI spec as query parameters.

3. Instrument the test suite with `pytest-cov`. Run `pytest --cov=src/finops --cov-report=term-missing` and identify which branches in `auth.py` are not covered. Write the missing tests.

4. The `/openapi.json` endpoint is publicly accessible by default. Add a middleware or route guard that returns `403` when the request's `X-Internal-Token` header does not match a value from `os.getenv("INTERNAL_DOCS_TOKEN")`. Write a test covering both the blocked and allowed cases.

(Solutions: ../solutions/week-07/day-5.md)

## Self-check

- [ ] I can explain when to use `TestClient` vs `AsyncClient`
- [ ] I can write a `conftest.py` with app, client, and auth header fixtures at the right scope
- [ ] I can test 401, 403, 409, and 422 responses with specific assertions
- [ ] I understand how FastAPI generates the OpenAPI spec from type hints and `response_model`
- [ ] I can use `pytest.mark.parametrize` to test multiple input/output combinations
