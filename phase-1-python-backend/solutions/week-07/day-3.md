# Week 07 · Day 3 — Solutions
## Dependency Injection, Middleware, and Exception Handlers

---

### Exercise 1
**Write an async `get_db` dependency using SQLAlchemy `async_sessionmaker` that yields a session and always closes it in `finally`. Show how to use it in a route.**

```python
import os
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from fastapi import FastAPI, Depends


# ---------------------------------------------------------------------------
# Engine and session factory
# ---------------------------------------------------------------------------

# DATABASE_URL must be set in the environment, e.g.:
#   DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/banking
DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./dev.db")

engine = create_async_engine(
    DATABASE_URL,
    echo=False,       # Set to True locally for SQL logging; never in production.
    pool_pre_ping=True,  # Detect stale connections before checkout.
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Avoid lazy-load errors after commit in async context.
    autocommit=False,
    autoflush=False,
)


# ---------------------------------------------------------------------------
# Base declarative class (all ORM models inherit from this)
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session for the duration of a single request.

    The session is always closed in the `finally` block, even if the route
    handler raises an exception.  The caller is responsible for commit/rollback.

    Usage in a route:
        async def my_route(db: AsyncSession = Depends(get_db)): ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            # `async with` already calls session.close() on __aexit__,
            # but the explicit finally makes the intent crystal clear and
            # guards against future refactors that remove the context manager.
            await session.close()


# ---------------------------------------------------------------------------
# Example ORM model and route
# ---------------------------------------------------------------------------

from sqlalchemy import Integer, String, select
from sqlalchemy.orm import Mapped, mapped_column


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[str] = mapped_column(String(254), unique=True, nullable=False)


app = FastAPI()


@app.get("/customers/{customer_id}")
async def get_customer(
    customer_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Retrieve a customer by ID.

    The `db` session is injected by FastAPI's DI system; no manual
    session management is needed inside the route.
    """
    result = await db.execute(select(Customer).where(Customer.id == customer_id))
    customer = result.scalar_one_or_none()
    if customer is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Customer not found")
    return {"id": customer.id, "name": customer.name, "email": customer.email}
```

**Key points:**
- `expire_on_commit=False` prevents SQLAlchemy from expiring attributes after `commit()`, which would cause `MissingGreenlet` errors when accessing them outside the session context in async code.
- Using `async with AsyncSessionLocal() as session` is equivalent to calling `session.close()` in `finally` — both patterns are shown so students see the relationship.
- Each request gets its own session; never share sessions between requests.

---

### Exercise 2
**Write a `RequestTimingMiddleware` (`BaseHTTPMiddleware`) that adds `X-Process-Time-Ms` to every response.**

```python
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from fastapi import FastAPI

app = FastAPI()


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """Measure the wall-clock time taken to process each request and attach
    it to the response as `X-Process-Time-Ms` (milliseconds, 2 decimal places).

    This is useful for:
    - Identifying slow endpoints without a dedicated APM tool.
    - Frontend developers who need rough latency figures in browser DevTools.
    - Integration tests that assert on performance budgets.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()          # High-resolution monotonic timer
        response: Response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1_000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        return response


app.add_middleware(RequestTimingMiddleware)


# ---------------------------------------------------------------------------
# Example route for manual testing
# ---------------------------------------------------------------------------

import asyncio

@app.get("/slow")
async def slow_endpoint() -> dict:
    """Simulates a slow operation so the timing header is visible."""
    await asyncio.sleep(0.1)
    return {"message": "done"}
```

**Why `time.perf_counter()` instead of `time.time()`?**
`perf_counter()` is a monotonic, high-resolution clock that is not affected by system clock adjustments. It is the correct choice for measuring elapsed wall time within a single process.

**Test:**
```python
from fastapi.testclient import TestClient

client = TestClient(app)

def test_timing_header_present():
    response = client.get("/slow")
    assert "X-Process-Time-Ms" in response.headers
    elapsed = float(response.headers["X-Process-Time-Ms"])
    assert elapsed >= 0  # Timing is always non-negative
```

---

### Exercise 3
**Create a custom `NotFoundException` and register an exception handler returning `{"error": "not_found", "detail": str(exc)}` with status 404.**

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


app = FastAPI()


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class NotFoundException(Exception):
    """Raised when a requested resource cannot be found.

    Prefer this over FastAPI's built-in HTTPException in domain/service code
    because it decouples business logic from the HTTP layer — the same exception
    can be caught in a CLI script or test without importing FastAPI.

    Args:
        resource: Human-readable name of the resource type (e.g. "Customer").
        identifier: The ID or key that was looked up.
    """

    def __init__(self, resource: str, identifier: str | int) -> None:
        self.resource = resource
        self.identifier = identifier
        super().__init__(f"{resource} '{identifier}' not found")


# ---------------------------------------------------------------------------
# Exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(NotFoundException)
async def not_found_exception_handler(
    request: Request,
    exc: NotFoundException,
) -> JSONResponse:
    """Convert NotFoundException to a structured 404 JSON response."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "not_found",
            "detail": str(exc),          # Uses the message set in __init__
        },
    )


# ---------------------------------------------------------------------------
# Example routes
# ---------------------------------------------------------------------------

_CUSTOMERS: dict[int, dict] = {1: {"id": 1, "name": "Jane Doe"}}


@app.get("/customers/{customer_id}")
async def get_customer(customer_id: int) -> dict:
    if customer_id not in _CUSTOMERS:
        raise NotFoundException(resource="Customer", identifier=customer_id)
    return _CUSTOMERS[customer_id]


_ACCOUNTS: dict[int, dict] = {101: {"id": 101, "balance": 500.00}}


@app.get("/accounts/{account_id}")
async def get_account(account_id: int) -> dict:
    if account_id not in _ACCOUNTS:
        raise NotFoundException(resource="Account", identifier=account_id)
    return _ACCOUNTS[account_id]
```

**Test:**
```python
from fastapi.testclient import TestClient

client = TestClient(app)

def test_not_found_handler():
    response = client.get("/customers/9999")
    assert response.status_code == 404
    body = response.json()
    assert body["error"] == "not_found"
    assert "Customer" in body["detail"]
    assert "9999" in body["detail"]
```

---

### Exercise 4
**Write a `rate_limit` dependency that tracks calls per client IP in-memory and raises HTTP 429 after 100 requests per minute using `time.time()` bucketing.**

```python
import time
from collections import defaultdict
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, status


app = FastAPI()


# ---------------------------------------------------------------------------
# In-memory rate limit tracker
# ---------------------------------------------------------------------------

# Structure: { ip_address: { minute_bucket: request_count } }
# minute_bucket = int(time.time() // 60)  — a unique integer per calendar minute.
_rate_limit_store: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))

RATE_LIMIT_REQUESTS: int = 100   # Max requests allowed per window
RATE_LIMIT_WINDOW_SECONDS: int = 60  # Window size in seconds


def _current_bucket() -> int:
    """Return the current one-minute bucket as an integer.

    Dividing Unix time by 60 and flooring gives a bucket that increments
    once per minute and is the same for all requests in that same minute.
    """
    return int(time.time() // RATE_LIMIT_WINDOW_SECONDS)


def _clean_old_buckets(ip: str, current_bucket: int) -> None:
    """Remove buckets older than the current window to prevent memory leaks.

    We only need the current minute's bucket, so any key != current_bucket
    can be discarded.
    """
    old_keys = [k for k in _rate_limit_store[ip] if k != current_bucket]
    for key in old_keys:
        del _rate_limit_store[ip][key]


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------

async def rate_limit(request: Request) -> None:
    """FastAPI dependency that enforces a per-IP rate limit.

    Raises:
        HTTPException(429): When the client IP exceeds RATE_LIMIT_REQUESTS
                            within the current one-minute bucket.

    Usage:
        @app.get("/protected", dependencies=[Depends(rate_limit)])
        async def protected_route(): ...

        # Or inject it to access the request inside the route:
        async def route(request: Request, _: None = Depends(rate_limit)): ...
    """
    # Prefer X-Forwarded-For for clients behind a reverse proxy / load balancer.
    client_ip: str = request.headers.get(
        "X-Forwarded-For", request.client.host if request.client else "unknown"
    ).split(",")[0].strip()

    bucket = _current_bucket()
    _clean_old_buckets(client_ip, bucket)

    _rate_limit_store[client_ip][bucket] += 1
    current_count = _rate_limit_store[client_ip][bucket]

    if current_count > RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Rate limit exceeded: {RATE_LIMIT_REQUESTS} requests per "
                f"{RATE_LIMIT_WINDOW_SECONDS}s allowed"
            ),
            headers={"Retry-After": str(RATE_LIMIT_WINDOW_SECONDS)},
        )


# ---------------------------------------------------------------------------
# Protected routes
# ---------------------------------------------------------------------------

@app.get("/items", dependencies=[Depends(rate_limit)])
async def list_items() -> dict:
    """Rate-limited endpoint — max 100 requests per minute per IP."""
    return {"items": []}


@app.get("/status")
async def status_check() -> dict:
    """Unprotected endpoint — no rate limit."""
    return {"status": "ok"}
```

**Limitations of this in-memory approach (worth discussing in class):**

| Concern | Description | Production fix |
|---------|-------------|----------------|
| Not distributed | Each worker process has its own `_rate_limit_store`; a client can hit 100 * N workers | Use Redis + `INCR` + `EXPIRE` |
| Not thread-safe | Concurrent async calls can race on the counter | `asyncio.Lock` per IP, or Redis atomicity |
| Memory growth | `_clean_old_buckets` only runs on the next request from the same IP | Background task or TTL-aware store |
| IP spoofing | `X-Forwarded-For` can be forged by clients | Trust only the first hop; validate at the load-balancer level |

**Test:**
```python
from unittest.mock import patch
from fastapi.testclient import TestClient

client = TestClient(app)

def test_rate_limit_enforced():
    """After RATE_LIMIT_REQUESTS + 1 calls, expect a 429."""
    # Patch _current_bucket so all calls land in the same window.
    with patch("__main__._current_bucket", return_value=99999):
        # Also clear any previous state for the test IP.
        _rate_limit_store.clear()

        for i in range(RATE_LIMIT_REQUESTS):
            r = client.get("/items")
            assert r.status_code == 200, f"Request {i+1} unexpectedly rate-limited"

        # The 101st request should be rejected.
        r = client.get("/items")
        assert r.status_code == 429
        assert "Retry-After" in r.headers
```
