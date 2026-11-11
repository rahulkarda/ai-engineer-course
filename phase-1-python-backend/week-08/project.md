# Week 08 Project — Production-Grade Financial Operations Backend

## Goal

Extend the Week 7 Financial API into a production-ready service by adding:

- A clean layered architecture (API → Service → Repository)
- Role-based access control with three roles: `admin`, `analyst`, `viewer`
- Async database access via SQLAlchemy 2.0 + asyncpg
- Structured JSON logging with per-request correlation IDs
- Liveness and readiness health endpoints
- Containerisation with Docker + docker-compose
- A GitHub Actions CI pipeline
- A comprehensive pytest suite (unit + integration)

Estimated time: 4–6 hours.

---

## Tech Stack

| Component | Technology |
|---|---|
| Web framework | FastAPI (latest stable) |
| Database | PostgreSQL 15+ |
| ORM / sessions | SQLAlchemy 2.0 async (`asyncpg` driver) |
| Schema validation | Pydantic v2 |
| Authentication | JWT (HS256) via `PyJWT` |
| Testing | pytest, pytest-asyncio, httpx `AsyncClient` |
| Config | `pydantic-settings`, all secrets via `os.getenv()` |
| Logging | `python-json-logger` or stdlib `logging` with custom formatter |
| Containerisation | Docker, docker-compose v2 |
| CI | GitHub Actions |

**Required environment variables — no hardcoded values, no defaults for secrets:**

```python
# config.py — read exclusively via os.getenv()
DATABASE_URL          = os.getenv("DATABASE_URL")          # postgresql+asyncpg://...
JWT_SECRET_KEY        = os.getenv("JWT_SECRET_KEY")        # no default
JWT_ALGORITHM         = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
LOG_LEVEL             = os.getenv("LOG_LEVEL", "INFO")
ENVIRONMENT           = os.getenv("ENVIRONMENT", "development")
```

Provide `.env.example` with all names and no values.

---

## Project Structure

```
financial-ops/
├── src/
│   └── financial_ops/
│       ├── api/
│       │   ├── deps.py              # FastAPI Depends: auth, db session, role guards
│       │   └── routes/
│       │       ├── accounts.py
│       │       ├── transactions.py
│       │       └── health.py
│       ├── services/
│       │   ├── account_service.py
│       │   └── transaction_service.py
│       ├── repositories/
│       │   ├── base_repo.py         # Generic async repository base
│       │   ├── account_repo.py
│       │   └── transaction_repo.py
│       ├── models/
│       │   ├── db.py                # SQLAlchemy ORM models
│       │   └── schemas.py           # Pydantic request / response schemas
│       ├── config.py                # pydantic-settings Settings class
│       ├── logging_config.py        # JSON log formatter + correlation ID context var
│       └── main.py                  # App factory, middleware, exception handlers
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_account_service.py
│   │   └── test_transaction_service.py
│   └── integration/
│       ├── test_accounts.py
│       └── test_transactions.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── pyproject.toml
└── README.md
```

No file in `src/` should exceed 300 lines. Split if it does.

---

## Day-by-Day Implementation Plan

### Session 1 (≈ 60 min) — Project scaffold and config

1. Create the directory tree above. Use `pyproject.toml` with `[project]` and `[tool.ruff]` sections.
2. Implement `config.py` with `pydantic-settings`:

```python
# src/financial_ops/config.py
import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = os.getenv("DATABASE_URL", "")
    jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    access_token_expire_minutes: int = int(
        os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
    )
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    environment: str = os.getenv("ENVIRONMENT", "development")


settings = Settings()
```

3. Implement `logging_config.py` with JSON formatter and a `contextvars.ContextVar` for correlation ID:

```python
# src/financial_ops/logging_config.py
import logging
import contextvars
import json
from datetime import datetime, timezone

correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default="-"
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id_var.get(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)
```

4. Write `.env.example` (no values). Verify `.env` is in `.gitignore`.

---

### Session 2 (≈ 60 min) — Database models and async session

1. Implement `models/db.py` — SQLAlchemy ORM models for `Account` and `Transaction`.
2. Set up async engine and session factory:

```python
# src/financial_ops/models/db.py
import os
import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import Numeric, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import (
    AsyncAttrs,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

DATABASE_URL: str = os.getenv("DATABASE_URL", "")

engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True)

AsyncSessionFactory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    engine, expire_on_commit=False
)


class Base(AsyncAttrs, DeclarativeBase):
    pass


class Account(Base):
    __tablename__ = "accounts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    owner_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    amount: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    transaction_type: Mapped[str] = mapped_column(String(10), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    description: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now(), nullable=False
    )
```

3. Implement `api/deps.py` — `get_db_session` yield dependency:

```python
# src/financial_ops/api/deps.py
import os
from collections.abc import AsyncGenerator

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from financial_ops.models.db import AsyncSession, AsyncSessionFactory

bearer_scheme = HTTPBearer()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def _decode_token(token: str) -> dict:
    secret = os.getenv("JWT_SECRET_KEY", "")
    algorithm = os.getenv("JWT_ALGORITHM", "HS256")
    try:
        return jwt.decode(token, secret, algorithms=[algorithm])
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    return _decode_token(credentials.credentials)


def require_role(*roles: str):
    """Dependency factory. Usage: Depends(require_role('admin', 'analyst'))"""

    async def _check(
        current_user: dict = Depends(get_current_user),
    ) -> dict:
        if current_user.get("role") not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required role: {roles}",
            )
        return current_user

    return _check
```

---

### Session 3 (≈ 60 min) — Repository layer

Implement a generic base repository, then concrete repositories for `Account` and `Transaction`.

```python
# src/financial_ops/repositories/base_repo.py
import uuid
from typing import Generic, TypeVar

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_ops.models.db import Base

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    model: type[ModelT]

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, record_id: uuid.UUID) -> ModelT | None:
        result = await self._session.execute(
            select(self.model).where(self.model.id == record_id)
        )
        return result.scalar_one_or_none()

    async def create(self, instance: ModelT) -> ModelT:
        self._session.add(instance)
        await self._session.flush()
        await self._session.refresh(instance)
        return instance

    async def delete(self, instance: ModelT) -> None:
        await self._session.delete(instance)
        await self._session.flush()
```

`AccountRepository` and `TransactionRepository` inherit `BaseRepository` and add domain-specific queries (e.g. `list_by_owner`, `compute_balance`).

---

### Session 4 (≈ 60 min) — Service layer and Pydantic schemas

1. Implement `models/schemas.py` — all Pydantic v2 request/response models:

```python
# src/financial_ops/models/schemas.py
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ACCEPTED_CURRENCIES = frozenset({"USD", "EUR", "GBP", "JPY", "CHF", "SGD"})


class AccountCreate(BaseModel):
    currency: str = Field(min_length=3, max_length=3)

    @field_validator("currency")
    @classmethod
    def currency_must_be_valid(cls, v: str) -> str:
        if v.upper() not in ACCEPTED_CURRENCIES:
            raise ValueError(f"Currency must be one of {sorted(ACCEPTED_CURRENCIES)}")
        return v.upper()


class AccountResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    owner_id: uuid.UUID
    currency: str
    created_at: datetime


class TransactionCreate(BaseModel):
    account_id: uuid.UUID
    amount: Decimal = Field(gt=Decimal("0"), decimal_places=2)
    currency: str = Field(min_length=3, max_length=3)
    transaction_type: Literal["credit", "debit"]
    description: str | None = Field(default=None, max_length=255)

    @field_validator("currency")
    @classmethod
    def currency_must_be_valid(cls, v: str) -> str:
        if v.upper() not in ACCEPTED_CURRENCIES:
            raise ValueError(f"Currency must be one of {sorted(ACCEPTED_CURRENCIES)}")
        return v.upper()


class TransactionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    account_id: uuid.UUID
    amount: Decimal
    currency: str
    transaction_type: str
    status: Literal["pending", "completed", "failed", "reversed"]
    description: str | None
    created_at: datetime
    updated_at: datetime


class AccountBalance(BaseModel):
    account_id: uuid.UUID
    currency: str
    balance: Decimal
    as_of: datetime
```

2. Implement `services/account_service.py` and `services/transaction_service.py`. Services own all business logic (auth ownership checks, status-transition rules, balance computation). Repositories handle only SQL.

```python
# src/financial_ops/services/account_service.py
import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from financial_ops.models.schemas import AccountBalance, AccountCreate, AccountResponse
from financial_ops.repositories.account_repo import AccountRepository
from financial_ops.models.db import Account

logger = logging.getLogger(__name__)


class AccountService:
    def __init__(self, session: AsyncSession) -> None:
        self._repo = AccountRepository(session)

    async def create_account(
        self, payload: AccountCreate, owner_id: uuid.UUID
    ) -> AccountResponse:
        account = Account(owner_id=owner_id, currency=payload.currency)
        created = await self._repo.create(account)
        logger.info("account.created", extra={"account_id": str(created.id)})
        return AccountResponse.model_validate(created)

    async def get_balance(
        self, account_id: uuid.UUID, requester_id: uuid.UUID, role: str
    ) -> AccountBalance:
        account = await self._repo.get_by_id(account_id)
        if account is None:
            from fastapi import HTTPException, status
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Account not found")
        if role != "admin" and account.owner_id != requester_id:
            from fastapi import HTTPException, status
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
        balance = await self._repo.compute_balance(account_id)
        return AccountBalance(
            account_id=account_id,
            currency=account.currency,
            balance=balance,
            as_of=datetime.now(timezone.utc),
        )
```

---

### Session 5 (≈ 60 min) — Routes, middleware, and health endpoints

1. Implement `api/routes/health.py`:

```python
# src/financial_ops/api/routes/health.py
import logging

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from financial_ops.api.deps import get_db_session

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


@router.get("/health", summary="Liveness — is the process alive?")
async def liveness() -> dict:
    return {"status": "ok"}


@router.get("/ready", summary="Readiness — can the service handle traffic?")
async def readiness(session: AsyncSession = Depends(get_db_session)) -> dict:
    try:
        await session.execute(text("SELECT 1"))
        return {"status": "ready", "db": "ok"}
    except Exception as exc:
        logger.error("readiness.db_check_failed", exc_info=True)
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable",
        ) from exc
```

2. Implement correlation-ID middleware in `main.py`:

```python
# src/financial_ops/main.py
import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from financial_ops.api.routes import accounts, health, transactions
from financial_ops.logging_config import configure_logging, correlation_id_var
from financial_ops.config import settings

configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(title="Financial Ops API", version="1.0.0")

    @app.middleware("http")
    async def correlation_id_middleware(request: Request, call_next):
        cid = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        token = correlation_id_var.set(cid)
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = cid
        correlation_id_var.reset(token)
        return response

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("unhandled_exception", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    app.include_router(health.router)
    app.include_router(accounts.router, prefix="/accounts")
    app.include_router(transactions.router, prefix="/transactions")
    return app


app = create_app()
```

---

### Session 6 (≈ 60 min) — Tests

Write tests in `tests/conftest.py`, `tests/unit/`, and `tests/integration/`.

Key patterns:

```python
# tests/conftest.py
import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import AsyncMock

from financial_ops.main import app
from financial_ops.api.deps import get_db_session


@pytest.fixture
def mock_session():
    return AsyncMock()


@pytest.fixture
def override_db(mock_session):
    async def _override():
        yield mock_session
    app.dependency_overrides[get_db_session] = _override
    yield mock_session
    app.dependency_overrides.clear()


@pytest.fixture
async def client(override_db):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
```

Required test coverage:
- `tests/unit/` — service logic without HTTP or real DB (mock the repository).
- `tests/integration/` — every endpoint: one happy-path test, one error-path test.
- `tests/` — auth tests: expired token → 401, wrong role → 403, ownership violation → 403.

---

### Session 7 (≈ 60 min) — Docker and CI

1. `Dockerfile` (multi-stage, cache-friendly):

```dockerfile
# Dockerfile
FROM python:3.12-slim AS base
WORKDIR /app

# Layer 1: install dependencies (cached until pyproject.toml changes)
COPY pyproject.toml .
RUN pip install --no-cache-dir ".[prod]"

# Layer 2: copy source (invalidated on code changes only)
COPY src/ src/

ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "financial_ops.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

2. `docker-compose.yml`:

```yaml
# docker-compose.yml
services:
  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: financial_ops
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U $${POSTGRES_USER}"]
      interval: 5s
      timeout: 5s
      retries: 5

  api:
    build: .
    env_file: .env
    ports:
      - "8000:8000"
    depends_on:
      db:
        condition: service_healthy
```

All credentials come from `.env` — never hardcoded.

3. `.github/workflows/ci.yml`:

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: ["main", "develop"]
  pull_request:

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install ruff
      - run: ruff check src/ tests/

  test:
    runs-on: ubuntu-latest
    needs: lint
    env:
      DATABASE_URL: ${{ secrets.DATABASE_URL }}
      JWT_SECRET_KEY: ${{ secrets.JWT_SECRET_KEY }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install ".[dev]"
      - run: pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=80

  build:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - uses: docker/build-push-action@v5
        with:
          context: .
          push: false
```

---

## Acceptance Criteria Checklist

### Architecture

- [ ] Controllers (`routes/`) call services only — no SQL in route handlers.
- [ ] Services call repositories only — no SQLAlchemy imports in service files.
- [ ] Repositories call SQLAlchemy only — no business logic in repo files.
- [ ] Dependency injection via `FastAPI.Depends` for DB session and current user.
- [ ] No file in `src/` exceeds 300 lines.

### Auth and RBAC

- [ ] JWT bearer auth on all non-health endpoints.
- [ ] Three roles enforced: `admin` (full access), `analyst` (read + create transactions), `viewer` (read-only).
- [ ] `require_role("admin")` and `require_role("admin", "analyst")` both composable via `Depends`.
- [ ] Ownership check: `analyst`/`viewer` can only access their own accounts and transactions.
- [ ] Expired tokens → 401. Invalid role → 403. Ownership violation → 403.

### Data Layer

- [ ] `AsyncSession` used throughout — no sync session calls in async path operations.
- [ ] `async_sessionmaker` with `expire_on_commit=False`.
- [ ] `DATABASE_URL` read via `os.getenv()` only.
- [ ] Balance computed via SQL aggregation (`SUM`) not Python loop over fetched rows.

### API

- [ ] `POST /accounts` — create account for authenticated user.
- [ ] `GET /accounts/{id}` — get account (ownership enforced).
- [ ] `GET /accounts/{id}/balance` — compute balance (SQL aggregation).
- [ ] `POST /transactions` — create transaction (account ownership enforced).
- [ ] `GET /transactions` — list transactions (role-filtered).
- [ ] `GET /transactions/{id}` — get transaction (ownership enforced).
- [ ] `PATCH /transactions/{id}` — update status (admin only, terminal-state guard).
- [ ] `GET /health` — liveness, always 200 while process is up.
- [ ] `GET /ready` — readiness, 503 when DB is down.

### Observability

- [ ] Every log line is valid JSON with `timestamp`, `level`, `message`, `correlation_id`.
- [ ] Correlation ID injected from `X-Correlation-ID` request header (or generated if absent).
- [ ] Correlation ID echoed in `X-Correlation-ID` response header.
- [ ] Global exception handler returns 500 with `{"detail": "Internal server error"}` — no stack traces in response body.
- [ ] `ERROR` level log with full traceback on every unhandled exception.

### Testing

- [ ] `pytest` passes with zero failures.
- [ ] Line coverage ≥ 80% (`pytest --cov=src --cov-fail-under=80`).
- [ ] Unit tests mock all external dependencies (DB, JWT).
- [ ] Integration tests use `httpx.AsyncClient` with `ASGITransport` — no live server.
- [ ] DB dependency overridden via `app.dependency_overrides` in all integration tests.
- [ ] `pytest.mark.unit`, `pytest.mark.integration`, `pytest.mark.auth` markers used.
- [ ] Every endpoint has at least one happy-path and one error-path test.

### Code Quality

- [ ] `ruff check src/ tests/` passes with zero violations.
- [ ] All functions and classes are typed (no `Any` without justification).
- [ ] Pydantic models for all data crossing layer boundaries.
- [ ] `logging` used, not `print`, throughout.
- [ ] `.env.example` present with all variable names and no values.
- [ ] `.env` in `.gitignore`.

### Containerisation & CI

- [ ] `docker compose up` starts API + PostgreSQL without manual steps.
- [ ] `docker compose up` reads credentials from `.env`, never hardcoded in `docker-compose.yml`.
- [ ] `Dockerfile` copies `pyproject.toml` before source so pip install is cached.
- [ ] GitHub Actions CI: lint → test → build, each job depends on the previous.
- [ ] Secrets (`DATABASE_URL`, `JWT_SECRET_KEY`) loaded from GitHub Actions secrets — not committed.

### README

- [ ] ASCII architecture diagram showing layers.
- [ ] Environment variable table.
- [ ] `docker compose up` quickstart.
- [ ] How to run tests locally.
- [ ] Link to `/docs` for Swagger UI.

---

## ASCII Architecture Diagram

```
┌────────────────────────────────────────────────────┐
│                    HTTP Client                      │
└───────────────────────┬────────────────────────────┘
                        │ HTTP/JSON
┌───────────────────────▼────────────────────────────┐
│              FastAPI (main.py)                      │
│  middleware: correlation ID, logging, error handler │
│                                                     │
│  routes/accounts.py   routes/transactions.py        │
│  routes/health.py                                   │
│                                                     │
│  deps.py  ── get_db_session(), require_role()       │
└───────────────────────┬────────────────────────────┘
                        │ calls
┌───────────────────────▼────────────────────────────┐
│            Service Layer                            │
│  account_service.py   transaction_service.py        │
│  (business logic, auth ownership, state machines)   │
└───────────────────────┬────────────────────────────┘
                        │ calls
┌───────────────────────▼────────────────────────────┐
│           Repository Layer                          │
│  base_repo.py         account_repo.py               │
│  transaction_repo.py                                │
│  (SQL only, no business logic)                      │
└───────────────────────┬────────────────────────────┘
                        │ asyncpg
┌───────────────────────▼────────────────────────────┐
│             PostgreSQL 15                           │
└────────────────────────────────────────────────────┘
```

---

## Grading Rubric (pass ≥ 75%)

| Dimension | Weight |
|---|---|
| Independent implementation | 40% |
| Conceptual understanding | 25% |
| Debugging & problem-solving | 20% |
| Code quality & testing | 15% |

---

## Bonus Challenges

### Bonus 1 — Redis Caching Layer

Add a Redis-backed cache for `GET /accounts/{id}/balance`:

- Cache key: `balance:{account_id}`.
- TTL: 30 seconds (configurable via `CACHE_TTL_SECONDS` env var).
- Cache miss: compute from DB and write through.
- Cache invalidation: invalidate on any `POST /transactions` for that account.
- Use `redis.asyncio` — no blocking calls in async path operations.
- Add `REDIS_URL` to `.env.example`. Read via `os.getenv("REDIS_URL")`.

### Bonus 2 — WebSocket Live Transaction Feed

Add a `GET /ws/transactions` WebSocket endpoint:

- On connect, authenticate via a `token` query parameter (same JWT validation).
- On each new transaction committed for the user's accounts, push a `TransactionResponse` JSON message to all connected clients with matching `owner_id`.
- Use `asyncio.Queue` per connection; a background task feeds queues on DB commit.
- Add at least one pytest test using `httpx.AsyncClient` WebSocket support.
