# Week 7 Assessment — Answer Key

## Conceptual Questions

### 1. HTTP Methods and Status Codes

**HTTP Methods:**

| Method | Safe | Idempotent | Typical Use |
|--------|------|------------|-------------|
| GET | Yes | Yes | Retrieve resource; no side effects |
| POST | No | No | Create resource or trigger action |
| PUT | No | Yes | Full replace of existing resource |
| PATCH | No | No* | Partial update of existing resource |
| DELETE | No | Yes | Remove resource |

*PATCH idempotency depends on implementation; spec does not require it.

**Status Codes:**

| Code | Name | When to Use |
|------|------|-------------|
| 200 | OK | Successful GET, PUT, PATCH |
| 201 | Created | Successful POST that created a resource; include Location header |
| 204 | No Content | Successful DELETE or action with no response body |
| 400 | Bad Request | Malformed syntax, invalid parameters client controls |
| 401 | Unauthorized | Missing or invalid authentication credentials |
| 403 | Forbidden | Authenticated but lacking permission |
| 404 | Not Found | Resource does not exist |
| 409 | Conflict | State conflict (duplicate unique key, optimistic lock failure) |
| 422 | Unprocessable Entity | Well-formed but semantically invalid (FastAPI validation errors) |
| 429 | Too Many Requests | Rate limit exceeded; include Retry-After header |
| 500 | Internal Server Error | Unhandled server-side failure |

---

### 2. Six REST Architectural Constraints

1. **Client-Server Separation** — UI and data storage concerns are decoupled. Client and server evolve independently as long as the interface contract is maintained.

2. **Statelessness** — Each request from client to server must contain all information needed to understand it. Session state is never stored on the server between requests.

3. **Cacheability** — Responses must declare themselves cacheable or non-cacheable. Caching eliminates redundant interactions and improves scalability.

4. **Uniform Interface** — The central REST constraint: resource identification via URIs, manipulation through representations, self-descriptive messages, HATEOAS (hypermedia as the engine of application state).

5. **Layered System** — Client cannot tell whether it is connected directly to the end server or an intermediary (load balancer, cache, gateway). Intermediaries can be added transparently.

6. **Code-On-Demand** (optional) — Servers can extend client functionality by transferring executable code (e.g., JavaScript). The only optional constraint.

---

### 3. Pydantic v2 Validation Mechanisms

**`field_validator`** — Validates or transforms a single field value.

```python
from pydantic import BaseModel, field_validator

class Transaction(BaseModel):
    amount: float

    @field_validator("amount", mode="before")
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("amount must be positive")
        return v
```

- `mode="before"`: runs before Pydantic's type coercion (receives raw input).
- `mode="after"` (default): runs after coercion on the typed value.

**`model_validator`** — Validates across multiple fields or transforms the entire model.

```python
from pydantic import BaseModel, model_validator
from typing import Self

class DateRange(BaseModel):
    start_date: date
    end_date: date

    @model_validator(mode="after")
    def end_after_start(self) -> Self:
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        return self
```

- `mode="before"`: receives raw dict before field parsing.
- `mode="after"`: receives the fully constructed model instance.

**`computed_field`** — Declares a read-only derived property included in serialization.

```python
from pydantic import BaseModel, computed_field

class Account(BaseModel):
    balance: float
    currency: str

    @computed_field
    @property
    def balance_display(self) -> str:
        return f"{self.balance:.2f} {self.currency}"
```

---

### 4. FastAPI Dependency Injection

`Depends()` builds a directed acyclic graph of callables resolved per request:

- Each dependency is called once per request regardless of how many route functions declare it.
- **Generator dependencies** (`yield`): code before `yield` is setup; code in `finally` after `yield` is teardown (e.g., closing a DB session).
- Dependencies can depend on other dependencies (nesting); FastAPI resolves the full graph before calling the route handler.
- Shared dependencies at the same level are cached within a single request by default (`use_cache=True`).

```python
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    ...

@router.get("/me")
async def read_me(current_user: User = Depends(get_current_user)):
    return current_user
```

---

### 5. JWT Structure and Stateless Authentication

**Structure:** three Base64url-encoded segments separated by dots:

```
base64url(header) . base64url(payload) . HMAC-SHA256(header.payload, secret)
```

- **Header**: `{"alg": "HS256", "typ": "JWT"}`
- **Payload (claims)**: `{"sub": "user_id", "exp": 1700000000, "iat": 1699996400}`
- **Signature**: cryptographic proof the token was issued by the server

**Why stateless:** The server holds only the secret key. To authenticate, it re-derives the HMAC over the received header+payload and compares to the signature. No session table lookup needed. Validity is proven cryptographically; expiry is enforced via the `exp` claim timestamp.

**Tradeoffs:** Tokens cannot be revoked before expiry without a blocklist (stateful). Short expiry + refresh token pattern mitigates this.

---

### 6. Async SQLAlchemy Pattern

```python
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
)
from sqlalchemy import select

engine = create_async_engine(os.getenv("DATABASE_URL"), echo=False)
async_session = async_sessionmaker(engine, expire_on_commit=False)

async def get_user(user_id: int) -> User | None:
    async with async_session() as session:
        result = await session.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()

async def create_user(data: dict) -> User:
    async with async_session() as session:
        user = User(**data)
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user
```

Key points:
- `create_async_engine` requires an async driver URL (e.g., `postgresql+asyncpg://...`).
- `async_sessionmaker` replaces `sessionmaker` for async sessions.
- All I/O calls are `await`ed: `execute`, `commit`, `refresh`, `flush`.
- `expire_on_commit=False` prevents lazy-load errors after commit in async context.

---

### 7. Middleware vs. Exception Handlers

**Middleware** wraps every request/response in the application:

- Runs for ALL requests regardless of route match.
- Has access to the raw `Request` before routing and the `Response` after the handler returns.
- Use cases: logging, timing, CORS, request ID injection, compression, authentication pre-checks.

```python
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response
```

**Exception Handlers** catch specific exception types raised anywhere in the stack:

- Registered per exception class; FastAPI routes to the most specific match.
- Transform the exception into a structured HTTP response.
- Do not run for requests that complete without raising.

```python
@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )
```

**Key distinction:** middleware processes every request unconditionally; exception handlers are invoked only when a matching exception propagates up.

---

### 8. FastAPI OpenAPI Generation

FastAPI auto-generates an OpenAPI 3.x schema by introspecting:

1. **Route decorators** (`@router.get`, `@router.post`, etc.) — path, method, operation ID.
2. **Type hints on path/query parameters** — generates parameter schemas with types and constraints.
3. **Pydantic models in `Body()`/`response_model`** — generates JSON Schema via `model_json_schema()`.
4. **`Depends()`** — resolves and includes parameters from dependency callables.
5. **`status_code` and `responses` kwargs** — documents response codes.
6. **Docstrings** — used as operation summary/description.

Schema served at `/openapi.json`; Swagger UI at `/docs`; ReDoc at `/redoc`.

```python
@router.post(
    "/transactions",
    response_model=TransactionRead,
    status_code=201,
    responses={409: {"model": ErrorResponse}},
    summary="Create a new transaction",
)
async def create_transaction(body: TransactionCreate, db: AsyncSession = Depends(get_db)):
    """
    Create a debit or credit transaction on an account.
    Validates that the account exists and has sufficient funds for debits.
    """
    ...
```

---

## Coding Challenges

### Challenge 1 — Rate-Limiting Dependency

Requirement: sliding-window rate limiter as a FastAPI dependency; attach `X-RateLimit-*` headers; return 429 on exceed.

```python
import time
import os
from collections import defaultdict
from fastapi import Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse

# In production use Redis. This is per-worker in-memory storage.
_rate_limit_store: dict[str, tuple[int, float]] = defaultdict(lambda: (0, 0.0))

RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))


class RateLimitExceeded(Exception):
    pass


def get_client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def rate_limit(request: Request, response: Response) -> None:
    """
    Dependency that enforces per-IP rate limiting.
    Attaches X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset headers.
    Raises HTTP 429 when the limit is exceeded.
    """
    ip = get_client_ip(request)
    now = time.time()
    count, window_start = _rate_limit_store[ip]

    if now - window_start >= RATE_LIMIT_WINDOW:
        # New window
        count = 0
        window_start = now

    count += 1
    _rate_limit_store[ip] = (count, window_start)

    reset_at = int(window_start + RATE_LIMIT_WINDOW)
    remaining = max(0, RATE_LIMIT_REQUESTS - count)

    response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT_REQUESTS)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Reset"] = str(reset_at)

    if count > RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again after the reset window.",
            headers={
                "Retry-After": str(reset_at - int(now)),
                "X-RateLimit-Limit": str(RATE_LIMIT_REQUESTS),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(reset_at),
            },
        )


# Usage on a route:
# @router.get("/items", dependencies=[Depends(rate_limit)])
# async def list_items(): ...
#
# Or injected per-router:
# router = APIRouter(dependencies=[Depends(rate_limit)])
```

---

### Challenge 2 — Cursor-Based Pagination: GET /transactions

Requirement: cursor-based pagination using base64-encoded last item ID; return `{"items": [...], "next_cursor": str | None}`.

```python
import base64
import os
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Assume Transaction ORM model and TransactionRead schema are already defined.


class PaginatedTransactions(BaseModel):
    items: list[TransactionRead]
    next_cursor: str | None


def encode_cursor(item_id: int) -> str:
    return base64.urlsafe_b64encode(str(item_id).encode()).decode()


def decode_cursor(cursor: str) -> int:
    try:
        return int(base64.urlsafe_b64decode(cursor.encode()).decode())
    except Exception as exc:
        raise ValueError(f"Invalid cursor: {cursor}") from exc


router = APIRouter(prefix="/transactions", tags=["transactions"])


@router.get("", response_model=PaginatedTransactions)
async def list_transactions(
    db: AsyncSession = Depends(get_db),
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: Annotated[str | None, Query()] = None,
) -> PaginatedTransactions:
    """
    Return paginated transactions ordered by id ascending.
    Pass `cursor` from the previous response's `next_cursor` to fetch the next page.
    """
    stmt = select(Transaction).order_by(Transaction.id.asc()).limit(limit + 1)

    if cursor is not None:
        after_id = decode_cursor(cursor)
        stmt = stmt.where(Transaction.id > after_id)

    result = await db.execute(stmt)
    rows = result.scalars().all()

    has_more = len(rows) > limit
    page = rows[:limit]

    next_cursor: str | None = None
    if has_more:
        next_cursor = encode_cursor(page[-1].id)

    return PaginatedTransactions(
        items=[TransactionRead.model_validate(t) for t in page],
        next_cursor=next_cursor,
    )
```

**Design notes:**

- Fetching `limit + 1` rows is a look-ahead trick to detect whether a next page exists without a separate COUNT query.
- Cursor encodes the last seen `id`; the next query filters `id > last_id`, which uses the primary key index efficiently regardless of dataset size.
- Unlike offset pagination, this approach is stable under inserts/deletes between pages.
- `base64.urlsafe_b64encode` keeps the cursor URL-safe without percent-encoding.
