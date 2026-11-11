# Phase 1 · Week 07 · Day 3 — Dependency Injection, Middleware & Exception Handlers

> Objective: Use FastAPI DI for shared resources, add middleware, and write clean error handlers  ·  Time: ~1 hr  ·  Prereqs: [[week-07/day-2]]

## Concept

### FastAPI Depends() — the DI system

`Depends()` is FastAPI's dependency injection mechanism. A dependency is any callable whose return value is injected into a route handler. Dependencies can be:
- Plain functions
- Async functions
- Classes (instantiated on each request)
- Generator functions (with cleanup via `yield`)

```python
from fastapi import Depends, FastAPI

def get_current_user(token: str = Header(...)) -> User:
    # validate token, return User
    ...

@app.get("/profile")
async def profile(user: User = Depends(get_current_user)) -> UserResponse:
    return user.to_response()
```

**Resolution order**: FastAPI builds a dependency graph at startup. On each request, it resolves leaf dependencies first, injects their results upward. Identical dependency callables are called once per request (cached by default); pass `use_cache=False` to `Depends()` to force re-execution.

**Yield dependencies** follow a try/finally pattern to guarantee cleanup even on exceptions:

```python
from collections.abc import AsyncGenerator

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with sessionmaker() as session:
        try:
            yield session          # handler receives this
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        # sessionmaker's __aexit__ closes the connection
```

`yield` dependencies are wrapped in an `ExitStack` internally — FastAPI calls the generator's `throw()` on exception and `close()` on completion, ensuring teardown always runs.

### Database session dependency pattern (SQLAlchemy async)

SQLAlchemy 2.0 introduced full async support. The canonical pattern:

```python
# db.py
import os
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# URL comes from environment — never hardcoded
engine = create_async_engine(
    os.getenv("DATABASE_URL"),  # e.g. postgresql+asyncpg://...
    echo=False,
    pool_size=5,
    max_overflow=10,
)

SessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # avoid lazy-load after commit
)
```

```python
# deps.py
from collections.abc import AsyncGenerator
from db import SessionLocal
from sqlalchemy.ext.asyncio import AsyncSession

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
```

Route handler:
```python
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from deps import get_session

@app.get("/users/{user_id}")
async def get_user(
    user_id: int,
    session: AsyncSession = Depends(get_session),
) -> UserResponse:
    result = await session.get(UserModel, user_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Not found")
    return UserResponse.model_validate(result)
```

### Middleware

Middleware wraps every request/response. Use it for cross-cutting concerns that belong outside business logic: logging, timing, CORS, rate limiting, request ID injection.

FastAPI inherits Starlette's middleware system. Two patterns:

**Starlette `BaseHTTPMiddleware`** — simpler, runs as a coroutine:
```python
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.1f}"
        return response
```

**Pure ASGI middleware** — more control, necessary for streaming responses. Uses `__init__` and `__call__`.

**CORS middleware** (built-in via Starlette):
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.example.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)
```

Middleware is stacked in LIFO order: the last `add_middleware` call wraps outermost. Add logging middleware last so it times the full stack.

### HTTPException vs custom exception handlers

**`HTTPException`** is the standard way to abort a request with a specific status:
```python
from fastapi import HTTPException

raise HTTPException(status_code=404, detail="Resource not found")
raise HTTPException(
    status_code=401,
    detail="Not authenticated",
    headers={"WWW-Authenticate": "Bearer"},
)
```

**Custom exception handlers** intercept non-`HTTPException` errors and domain exceptions:

```python
from fastapi import Request
from fastapi.responses import JSONResponse

class ResourceNotFoundError(Exception):
    def __init__(self, resource: str, id: int) -> None:
        self.resource = resource
        self.id = id

@app.exception_handler(ResourceNotFoundError)
async def not_found_handler(
    request: Request,
    exc: ResourceNotFoundError,
) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"error": "not_found", "resource": exc.resource, "id": exc.id},
    )
```

A global catch-all handler for unhandled exceptions (prevents leaking stack traces):

```python
@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error"},
    )
```

## How it works internally

**DI graph resolution:**
At startup, FastAPI inspects every route handler with `inspect.signature`. For each `Depends(callable)`, it recursively inspects `callable`'s signature to find its own dependencies, building a directed acyclic graph. At request time, FastAPI resolves the graph in topological order: leaves first, then injects results upward. The `use_cache=True` default means the same callable instance is called once per request scope; this is implemented with a `dict` keyed on the dependency callable itself.

Yield dependencies are tracked in a `contextlib.AsyncExitStack` per request. On normal completion, each generator receives `.asend(None)` to run post-`yield` code. On exception, each generator receives `.athrow(exc)` — this is why your `except Exception: rollback; raise` runs even on 500 errors.

**Middleware execution order:**
Starlette builds a chain of ASGI apps. `add_middleware(M)` wraps the current app: `M(current_app)`. Last added = outermost. Each middleware calls `await call_next(request)` to pass down the chain. `BaseHTTPMiddleware` buffers the response body to allow modification; pure ASGI middleware can stream without buffering.

**Exception handler dispatch:**
Starlette's `ExceptionMiddleware` (innermost) catches `HTTPException` and routes it to registered handlers. FastAPI wraps it with `ServerErrorMiddleware` (outermost non-user middleware) to catch any unhandled exception. Custom `@app.exception_handler` registrations are stored in a dict; dispatch walks the MRO of the exception class to find the most specific handler.

## Worked example

```python
# app.py
# DB session dependency + request logging middleware + global exception handler
# Run: DATABASE_URL=sqlite+aiosqlite:///./dev.db uvicorn app:app --reload

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Database setup — URL from environment only
# ---------------------------------------------------------------------------
_db_url = os.getenv("DATABASE_URL")
if not _db_url:
    raise RuntimeError("DATABASE_URL env var not set")

engine = create_async_engine(_db_url, echo=False)
SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


# ---------------------------------------------------------------------------
# Lifespan — startup/shutdown logic (replaces @app.on_event)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting up — verifying DB connection")
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    yield
    logger.info("Shutting down — disposing engine")
    await engine.dispose()


app = FastAPI(title="Demo API", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Custom domain exceptions
# ---------------------------------------------------------------------------
class NotFoundError(Exception):
    def __init__(self, resource: str, id: int | str) -> None:
        self.resource = resource
        self.id = id


class ConflictError(Exception):
    def __init__(self, detail: str) -> None:
        self.detail = detail


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------
@app.exception_handler(NotFoundError)
async def not_found_handler(request: Request, exc: NotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"error": "not_found", "resource": exc.resource, "id": exc.id},
    )


@app.exception_handler(ConflictError)
async def conflict_handler(request: Request, exc: ConflictError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"error": "conflict", "detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    # Log with traceback but never expose it to the caller
    logger.exception(
        "Unhandled exception request_id=%s method=%s path=%s",
        request.state.request_id,
        request.method,
        request.url.path,
    )
    return JSONResponse(status_code=500, content={"error": "internal_server_error"})


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Attach a request ID and log method, path, status, and duration."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        response.headers["X-Request-Id"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.1f}"

        logger.info(
            "request_id=%s method=%s path=%s status=%d duration_ms=%.1f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response


# Add CORS before logging so CORS headers are present on all responses
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

# Add logging last — outermost, times the full stack
app.add_middleware(RequestLoggingMiddleware)


# ---------------------------------------------------------------------------
# DB session dependency
# ---------------------------------------------------------------------------
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async DB session; commit on success, rollback on error."""
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health(session: AsyncSession = Depends(get_session)) -> dict[str, str]:
    """Liveness check — verifies DB connectivity."""
    await session.execute(text("SELECT 1"))
    return {"status": "ok"}


@app.get("/crash")
async def crash() -> None:
    """Deliberately raises to demonstrate the global exception handler."""
    raise RuntimeError("simulated server error")
```

Run and test:
```bash
pip install fastapi uvicorn sqlalchemy aiosqlite

DATABASE_URL=sqlite+aiosqlite:///./dev.db uvicorn app:app --reload

# Health check — observe X-Request-Id and X-Process-Time-Ms headers
curl -v http://localhost:8000/health

# Trigger global handler — returns 500 JSON, no traceback
curl http://localhost:8000/crash

# Raise NotFoundError from a route (add a test route or reuse in exercises)
```

## Exercises

1. Add a `get_current_user` dependency that reads an `Authorization: Bearer <token>` header (use `Header(alias="authorization")`). If the header is missing or the token is not the value of `os.getenv("API_TOKEN")`, raise `HTTPException(status_code=401)`. Inject it into the `GET /health` endpoint.

2. Create a `RateLimitMiddleware(BaseHTTPMiddleware)` that tracks the number of requests per IP (use `request.client.host`) within a 60-second window stored in a module-level `dict`. Return a `JSONResponse` with status 429 if an IP exceeds 100 requests in the window. Do not call `call_next` if the limit is exceeded.

3. Add a second dependency `get_admin_session` that calls `Depends(get_session)` internally (dependency chaining) but also calls `Depends(get_current_user)` and raises 403 if the user is not flagged as admin. Demonstrate that FastAPI calls `get_session` only once per request even though both `get_session` and `get_admin_session` depend on it.

4. Write `tests/test_app.py` using `TestClient` (synchronous) and `pytest-asyncio` + `httpx.AsyncClient` (async). Cover: health returns 200 with `X-Request-Id` header present, `/crash` returns 500 JSON with key `error`, injecting a missing `Authorization` header returns 401 after Exercise 1 is implemented.

(Solutions: ../solutions/week-07/day-3.md)

## Self-check

- [ ] I can explain how FastAPI resolves a dependency graph at request time
- [ ] I can explain why yield dependencies guarantee cleanup even when an exception propagates
- [ ] I can implement an async DB session dependency with commit/rollback logic without reference
- [ ] I can explain middleware execution order and why the last `add_middleware` call is outermost
- [ ] I can explain the difference between `HTTPException` and a custom exception handler and when to use each
