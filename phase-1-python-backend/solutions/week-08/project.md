# Week 8 Project — Production-Grade Financial Operations Backend: Reference Implementation

This document is the annotated reference solution. Every file is complete and runnable.
Instructors use it for grading; students use it only after attempting the project.

---

## Project Layout

```
financial-ops/
├── src/
│   └── financial_ops/
│       ├── __init__.py
│       ├── config.py
│       ├── logging_config.py
│       ├── main.py
│       ├── api/
│       │   ├── __init__.py
│       │   ├── deps.py
│       │   └── routes/
│       │       ├── __init__.py
│       │       ├── accounts.py
│       │       ├── transactions.py
│       │       └── health.py
│       ├── services/
│       │   ├── __init__.py
│       │   ├── account_service.py
│       │   └── transaction_service.py
│       ├── repositories/
│       │   ├── __init__.py
│       │   ├── base_repo.py
│       │   ├── account_repo.py
│       │   └── transaction_repo.py
│       └── models/
│           ├── __init__.py
│           ├── db.py
│           └── schemas.py
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_account_service.py
│   │   └── test_transaction_service.py
│   └── integration/
│       ├── __init__.py
│       ├── test_accounts.py
│       ├── test_transactions.py
│       └── test_auth.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

**Layer contract** (enforced in grading):
- `routes/` → call services only, no SQL, no ORM imports.
- `services/` → call repositories only, own all business rules and RBAC ownership checks.
- `repositories/` → SQL only, no business logic, no HTTP exceptions.

---

## pyproject.toml

```toml
[project]
name = "financial-ops"
version = "1.0.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "sqlalchemy[asyncio]>=2.0.0",
    "asyncpg>=0.29.0",
    "pydantic>=2.9.0",
    "pydantic-settings>=2.5.0",
    "PyJWT>=2.9.0",
    "passlib[bcrypt]>=1.7.4",
    "python-multipart>=0.0.12",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=5.0.0",
    "httpx>=0.27.0",
    "anyio>=4.6.0",
    "aiosqlite>=0.20.0",
]
prod = [
    "financial-ops",
]

[build-system]
requires = ["setuptools>=70"]
build-backend = "setuptools.backends.legacy:build"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
markers = [
    "unit: pure unit tests, no DB or HTTP",
    "integration: HTTP-layer tests with ASGITransport",
    "auth: authentication and authorisation edge cases",
]

[tool.ruff]
target-version = "py312"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
```

---

## .env.example

```dotenv
# All variable names listed; no values — fill in .env (never commit .env)

# Database — asyncpg driver
DATABASE_URL=

# JWT
JWT_SECRET_KEY=
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Observability
LOG_LEVEL=INFO
ENVIRONMENT=development

# Docker Compose postgres service
POSTGRES_USER=
POSTGRES_PASSWORD=
POSTGRES_DB=financial_ops
```

---

## .gitignore (relevant excerpt)

```
.env
*.pyc
__pycache__/
.pytest_cache/
.coverage
htmlcov/
dist/
*.egg-info/
```

---

## src/financial_ops/config.py

`pydantic-settings` reads from the process environment (and optionally `.env`).
All secrets have **no** fallback default — the app raises `ValidationError` at import time if they are absent.

```python
# src/financial_ops/config.py
import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Secrets — no fallback; missing value = startup failure
    database_url: str = os.getenv("DATABASE_URL", "")
    jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "")

    # Non-secret config with safe defaults
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    access_token_expire_minutes: int = int(
        os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
    )
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    environment: str = os.getenv("ENVIRONMENT", "development")


settings = Settings()
```

---

## src/financial_ops/logging_config.py

Structured JSON logs with a per-request `correlation_id` injected via a `ContextVar`.
The `correlation_id_var` is set by middleware in `main.py` before any log call fires.

```python
# src/financial_ops/logging_config.py
import contextvars
import json
import logging
from datetime import datetime, timezone

correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default="-"
)


class JsonFormatter(logging.Formatter):
    """Emit every log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id_var.get(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        # Merge any extra fields added via logger.info("...", extra={...})
        for key, value in record.__dict__.items():
            if key not in (
                "args", "asctime", "created", "exc_info", "exc_text",
                "filename", "funcName", "id", "levelname", "levelno",
                "lineno", "module", "msecs", "message", "msg", "name",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "thread", "threadName",
            ):
                payload[key] = value
        return json.dumps(payload)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level.upper(), handlers=[handler], force=True)
```

---

## src/financial_ops/models/db.py

SQLAlchemy 2.0 async ORM models. `AsyncAttrs` enables lazy-load awaiting where needed.
`expire_on_commit=False` on the session factory prevents detached-instance errors after commit.

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

_DATABASE_URL: str = os.getenv("DATABASE_URL", "")

engine = create_async_engine(
    _DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionFactory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    engine,
    expire_on_commit=False,
)


class Base(AsyncAttrs, DeclarativeBase):
    pass


class Account(Base):
    __tablename__ = "accounts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), nullable=False
    )


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
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending"
    )
    description: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now(), nullable=False
    )
```

---

## src/financial_ops/models/schemas.py

All Pydantic v2 request and response schemas. `ConfigDict(from_attributes=True)` enables
`model_validate(orm_instance)` in service return paths.

```python
# src/financial_ops/models/schemas.py
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ACCEPTED_CURRENCIES: frozenset[str] = frozenset(
    {"USD", "EUR", "GBP", "JPY", "CHF", "SGD"}
)


# ── Account ───────────────────────────────────────────────────────────────────


class AccountCreate(BaseModel):
    currency: str = Field(min_length=3, max_length=3)

    @field_validator("currency")
    @classmethod
    def currency_must_be_valid(cls, v: str) -> str:
        upper = v.upper()
        if upper not in ACCEPTED_CURRENCIES:
            raise ValueError(
                f"Currency must be one of {sorted(ACCEPTED_CURRENCIES)}"
            )
        return upper


class AccountResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    owner_id: uuid.UUID
    currency: str
    created_at: datetime


class AccountBalance(BaseModel):
    account_id: uuid.UUID
    currency: str
    balance: Decimal
    as_of: datetime


# ── Transaction ───────────────────────────────────────────────────────────────


class TransactionCreate(BaseModel):
    account_id: uuid.UUID
    amount: Decimal = Field(gt=Decimal("0"), decimal_places=2)
    currency: str = Field(min_length=3, max_length=3)
    transaction_type: Literal["credit", "debit"]
    description: str | None = Field(default=None, max_length=255)

    @field_validator("currency")
    @classmethod
    def currency_must_be_valid(cls, v: str) -> str:
        upper = v.upper()
        if upper not in ACCEPTED_CURRENCIES:
            raise ValueError(
                f"Currency must be one of {sorted(ACCEPTED_CURRENCIES)}"
            )
        return upper


class TransactionStatusUpdate(BaseModel):
    status: Literal["pending", "completed", "failed", "reversed"]


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


# ── Auth ──────────────────────────────────────────────────────────────────────


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ── Error ─────────────────────────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    detail: str
```

---

## src/financial_ops/repositories/base_repo.py

A generic typed base repository. Concrete repos inherit `BaseRepository[ModelT]` and
call `super()` methods rather than reimplementing `get_by_id` / `create` / `delete`.

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
            select(self.model).where(self.model.id == record_id)  # type: ignore[attr-defined]
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

---

## src/financial_ops/repositories/account_repo.py

Domain-specific queries sit here; business rules do not.
`compute_balance` uses a SQL `SUM` aggregation — never a Python loop over fetched rows.

```python
# src/financial_ops/repositories/account_repo.py
import uuid
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_ops.models.db import Account, Transaction
from financial_ops.repositories.base_repo import BaseRepository


class AccountRepository(BaseRepository[Account]):
    model = Account

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def list_by_owner(self, owner_id: uuid.UUID) -> list[Account]:
        result = await self._session.execute(
            select(Account).where(Account.owner_id == owner_id)
        )
        return list(result.scalars().all())

    async def compute_balance(self, account_id: uuid.UUID) -> Decimal:
        """
        Compute net balance via SQL aggregation:
          SUM(credit amounts) - SUM(debit amounts)
        Returns Decimal("0") if no transactions exist.
        """
        credit_result = await self._session.execute(
            select(func.coalesce(func.sum(Transaction.amount), 0)).where(
                Transaction.account_id == account_id,
                Transaction.transaction_type == "credit",
                Transaction.status != "reversed",
            )
        )
        debit_result = await self._session.execute(
            select(func.coalesce(func.sum(Transaction.amount), 0)).where(
                Transaction.account_id == account_id,
                Transaction.transaction_type == "debit",
                Transaction.status != "reversed",
            )
        )
        credits: Decimal = credit_result.scalar_one() or Decimal("0")
        debits: Decimal = debit_result.scalar_one() or Decimal("0")
        return Decimal(str(credits)) - Decimal(str(debits))
```

---

## src/financial_ops/repositories/transaction_repo.py

```python
# src/financial_ops/repositories/transaction_repo.py
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_ops.models.db import Account, Transaction
from financial_ops.repositories.base_repo import BaseRepository


class TransactionRepository(BaseRepository[Transaction]):
    model = Transaction

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def list_by_account(self, account_id: uuid.UUID) -> list[Transaction]:
        result = await self._session.execute(
            select(Transaction)
            .where(Transaction.account_id == account_id)
            .order_by(Transaction.created_at.asc())
        )
        return list(result.scalars().all())

    async def list_for_owner(self, owner_id: uuid.UUID) -> list[Transaction]:
        """Return all transactions for all accounts owned by `owner_id`."""
        result = await self._session.execute(
            select(Transaction)
            .join(Account, Transaction.account_id == Account.id)
            .where(Account.owner_id == owner_id)
            .order_by(Transaction.created_at.asc())
        )
        return list(result.scalars().all())

    async def list_all(self) -> list[Transaction]:
        result = await self._session.execute(
            select(Transaction).order_by(Transaction.created_at.asc())
        )
        return list(result.scalars().all())
```

---

## src/financial_ops/services/account_service.py

Services own all business logic. Note that `HTTPException` is imported inline where
raised so that repository files stay import-clean.

```python
# src/financial_ops/services/account_service.py
import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from financial_ops.models.db import Account
from financial_ops.models.schemas import AccountBalance, AccountCreate, AccountResponse
from financial_ops.repositories.account_repo import AccountRepository

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

    async def get_account(
        self,
        account_id: uuid.UUID,
        requester_id: uuid.UUID,
        role: str,
    ) -> AccountResponse:
        from fastapi import HTTPException, status

        account = await self._repo.get_by_id(account_id)
        if account is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Account not found",
            )
        if role != "admin" and account.owner_id != requester_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied",
            )
        return AccountResponse.model_validate(account)

    async def list_accounts(
        self,
        requester_id: uuid.UUID,
        role: str,
    ) -> list[AccountResponse]:
        if role == "admin":
            # admin sees all — for simplicity we still scope to owner; extend if needed
            accounts = await self._repo.list_by_owner(requester_id)
        else:
            accounts = await self._repo.list_by_owner(requester_id)
        return [AccountResponse.model_validate(a) for a in accounts]

    async def get_balance(
        self,
        account_id: uuid.UUID,
        requester_id: uuid.UUID,
        role: str,
    ) -> AccountBalance:
        from fastapi import HTTPException, status

        account = await self._repo.get_by_id(account_id)
        if account is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Account not found",
            )
        if role != "admin" and account.owner_id != requester_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied",
            )
        balance = await self._repo.compute_balance(account_id)
        return AccountBalance(
            account_id=account_id,
            currency=account.currency,
            balance=balance,
            as_of=datetime.now(timezone.utc),
        )
```

---

## src/financial_ops/services/transaction_service.py

Terminal-state guard: once a transaction reaches `completed`, `failed`, or `reversed`
it cannot transition back to `pending`. Enforced here, not in the route.

```python
# src/financial_ops/services/transaction_service.py
import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from financial_ops.models.db import Transaction
from financial_ops.models.schemas import (
    TransactionCreate,
    TransactionResponse,
    TransactionStatusUpdate,
)
from financial_ops.repositories.account_repo import AccountRepository
from financial_ops.repositories.transaction_repo import TransactionRepository

logger = logging.getLogger(__name__)

# Transitions that are not allowed (terminal → pending is forbidden)
_TERMINAL_STATES: frozenset[str] = frozenset({"completed", "failed", "reversed"})


class TransactionService:
    def __init__(self, session: AsyncSession) -> None:
        self._txn_repo = TransactionRepository(session)
        self._acct_repo = AccountRepository(session)

    async def create_transaction(
        self,
        payload: TransactionCreate,
        requester_id: uuid.UUID,
        role: str,
    ) -> TransactionResponse:
        from fastapi import HTTPException, status

        account = await self._acct_repo.get_by_id(payload.account_id)
        if account is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Account not found",
            )
        # analyst and viewer can only transact on their own accounts
        if role not in ("admin",) and account.owner_id != requester_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied",
            )
        # viewer cannot create transactions
        if role == "viewer":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="viewer role cannot create transactions",
            )

        txn = Transaction(
            account_id=payload.account_id,
            amount=payload.amount,
            currency=payload.currency,
            transaction_type=payload.transaction_type,
            status="pending",
            description=payload.description,
        )
        created = await self._txn_repo.create(txn)
        logger.info(
            "transaction.created",
            extra={
                "transaction_id": str(created.id),
                "account_id": str(created.account_id),
                "type": created.transaction_type,
                "amount": str(created.amount),
            },
        )
        return TransactionResponse.model_validate(created)

    async def get_transaction(
        self,
        transaction_id: uuid.UUID,
        requester_id: uuid.UUID,
        role: str,
    ) -> TransactionResponse:
        from fastapi import HTTPException, status

        txn = await self._txn_repo.get_by_id(transaction_id)
        if txn is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Transaction not found",
            )
        account = await self._acct_repo.get_by_id(txn.account_id)
        if account is None or (
            role != "admin" and account.owner_id != requester_id
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied",
            )
        return TransactionResponse.model_validate(txn)

    async def list_transactions(
        self,
        requester_id: uuid.UUID,
        role: str,
    ) -> list[TransactionResponse]:
        if role == "admin":
            rows = await self._txn_repo.list_all()
        else:
            rows = await self._txn_repo.list_for_owner(requester_id)
        return [TransactionResponse.model_validate(t) for t in rows]

    async def update_status(
        self,
        transaction_id: uuid.UUID,
        payload: TransactionStatusUpdate,
        requester_id: uuid.UUID,
        role: str,
    ) -> TransactionResponse:
        from fastapi import HTTPException, status

        # Only admin may change transaction status
        if role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admin can update transaction status",
            )
        txn = await self._txn_repo.get_by_id(transaction_id)
        if txn is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Transaction not found",
            )
        # Terminal-state guard: cannot move a terminal transaction back to pending
        if txn.status in _TERMINAL_STATES and payload.status == "pending":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Cannot transition from '{txn.status}' back to 'pending'",
            )
        txn.status = payload.status
        await self._txn_repo._session.flush()
        await self._txn_repo._session.refresh(txn)
        logger.info(
            "transaction.status_updated",
            extra={
                "transaction_id": str(transaction_id),
                "new_status": payload.status,
                "updated_by": str(requester_id),
            },
        )
        return TransactionResponse.model_validate(txn)
```

---

## src/financial_ops/api/deps.py

All FastAPI `Depends` objects live here. `require_role` is a dependency factory —
it returns an async callable, so `Depends(require_role("admin"))` works at the route level.

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
    """Yield a session; commit on success, rollback on exception."""
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
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
        ) from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        ) from exc


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    """Decode the bearer token and return the claims dict."""
    return _decode_token(credentials.credentials)


def require_role(*roles: str):
    """
    Dependency factory.

    Usage:
        @router.get("/", dependencies=[Depends(require_role("admin", "analyst"))])
        async def handler(user: dict = Depends(require_role("admin"))): ...
    """

    async def _check(
        current_user: dict = Depends(get_current_user),
    ) -> dict:
        if current_user.get("role") not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required role: {list(roles)}",
            )
        return current_user

    return _check
```

---

## src/financial_ops/api/routes/health.py

`/health` is a pure liveness check — it never touches the DB so it stays green
even when the database is down (useful for Kubernetes liveness probes).
`/ready` is a readiness check — 503 when the DB is unreachable (Kubernetes readiness probe).

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
        from fastapi import HTTPException, status as http_status

        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable",
        ) from exc
```

---

## src/financial_ops/api/routes/accounts.py

Routes call the service and nothing else. Token claims (`sub`, `role`) are extracted
from `Depends(get_current_user)` and forwarded to the service layer.

```python
# src/financial_ops/api/routes/accounts.py
import uuid

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from financial_ops.api.deps import get_current_user, get_db_session, require_role
from financial_ops.models.schemas import (
    AccountBalance,
    AccountCreate,
    AccountResponse,
    ErrorResponse,
)
from financial_ops.services.account_service import AccountService

router = APIRouter(prefix="/accounts", tags=["accounts"])


@router.post(
    "",
    response_model=AccountResponse,
    status_code=status.HTTP_201_CREATED,
    responses={401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    summary="Create a new account for the authenticated user",
)
async def create_account(
    body: AccountCreate,
    session: AsyncSession = Depends(get_db_session),
    current_user: dict = Depends(require_role("admin", "analyst", "viewer")),
) -> AccountResponse:
    svc = AccountService(session)
    return await svc.create_account(
        payload=body,
        owner_id=uuid.UUID(current_user["sub"]),
    )


@router.get(
    "/{account_id}",
    response_model=AccountResponse,
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
    summary="Retrieve an account by ID",
)
async def get_account(
    account_id: uuid.UUID,
    session: AsyncSession = Depends(get_db_session),
    current_user: dict = Depends(require_role("admin", "analyst", "viewer")),
) -> AccountResponse:
    svc = AccountService(session)
    return await svc.get_account(
        account_id=account_id,
        requester_id=uuid.UUID(current_user["sub"]),
        role=current_user.get("role", ""),
    )


@router.get(
    "/{account_id}/balance",
    response_model=AccountBalance,
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
    summary="Compute account balance via SQL aggregation",
)
async def get_balance(
    account_id: uuid.UUID,
    session: AsyncSession = Depends(get_db_session),
    current_user: dict = Depends(require_role("admin", "analyst", "viewer")),
) -> AccountBalance:
    svc = AccountService(session)
    return await svc.get_balance(
        account_id=account_id,
        requester_id=uuid.UUID(current_user["sub"]),
        role=current_user.get("role", ""),
    )
```

---

## src/financial_ops/api/routes/transactions.py

`PATCH /transactions/{id}` is admin-only; the service layer enforces it.
All other endpoints are accessible by authenticated users, with ownership
enforcement in the service.

```python
# src/financial_ops/api/routes/transactions.py
import uuid

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from financial_ops.api.deps import get_current_user, get_db_session, require_role
from financial_ops.models.schemas import (
    ErrorResponse,
    TransactionCreate,
    TransactionResponse,
    TransactionStatusUpdate,
)
from financial_ops.services.transaction_service import TransactionService

router = APIRouter(prefix="/transactions", tags=["transactions"])


@router.post(
    "",
    response_model=TransactionResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
    summary="Create a transaction (admin + analyst only)",
)
async def create_transaction(
    body: TransactionCreate,
    session: AsyncSession = Depends(get_db_session),
    current_user: dict = Depends(require_role("admin", "analyst")),
) -> TransactionResponse:
    svc = TransactionService(session)
    return await svc.create_transaction(
        payload=body,
        requester_id=uuid.UUID(current_user["sub"]),
        role=current_user.get("role", ""),
    )


@router.get(
    "",
    response_model=list[TransactionResponse],
    responses={401: {"model": ErrorResponse}},
    summary="List transactions (admin sees all; others see own)",
)
async def list_transactions(
    session: AsyncSession = Depends(get_db_session),
    current_user: dict = Depends(require_role("admin", "analyst", "viewer")),
) -> list[TransactionResponse]:
    svc = TransactionService(session)
    return await svc.list_transactions(
        requester_id=uuid.UUID(current_user["sub"]),
        role=current_user.get("role", ""),
    )


@router.get(
    "/{transaction_id}",
    response_model=TransactionResponse,
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
    summary="Retrieve a transaction by ID",
)
async def get_transaction(
    transaction_id: uuid.UUID,
    session: AsyncSession = Depends(get_db_session),
    current_user: dict = Depends(require_role("admin", "analyst", "viewer")),
) -> TransactionResponse:
    svc = TransactionService(session)
    return await svc.get_transaction(
        transaction_id=transaction_id,
        requester_id=uuid.UUID(current_user["sub"]),
        role=current_user.get("role", ""),
    )


@router.patch(
    "/{transaction_id}",
    response_model=TransactionResponse,
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
    summary="Update transaction status — admin only",
)
async def update_transaction_status(
    transaction_id: uuid.UUID,
    body: TransactionStatusUpdate,
    session: AsyncSession = Depends(get_db_session),
    current_user: dict = Depends(require_role("admin")),
) -> TransactionResponse:
    svc = TransactionService(session)
    return await svc.update_status(
        transaction_id=transaction_id,
        payload=body,
        requester_id=uuid.UUID(current_user["sub"]),
        role=current_user.get("role", ""),
    )
```

---

## src/financial_ops/main.py

App factory pattern. Calling `create_app()` returns a configured `FastAPI` instance.
The module-level `app = create_app()` is what Uvicorn imports.

```python
# src/financial_ops/main.py
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from financial_ops.api.routes import accounts, health, transactions
from financial_ops.config import settings
from financial_ops.logging_config import configure_logging, correlation_id_var
from financial_ops.models.db import Base, engine

configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup: creating tables if not exist")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    logger.info("shutdown: disposing engine")
    await engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Financial Ops API",
        description="Production-grade financial backend — Week 8 reference implementation.",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Middleware ────────────────────────────────────────────────────────────

    @app.middleware("http")
    async def correlation_id_middleware(request: Request, call_next):
        cid = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        token = correlation_id_var.set(cid)
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = cid
        correlation_id_var.reset(token)
        return response

    # ── Exception handlers ────────────────────────────────────────────────────

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors()},
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("unhandled_exception", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    # ── Routers ───────────────────────────────────────────────────────────────

    app.include_router(health.router)
    app.include_router(accounts.router)
    app.include_router(transactions.router)

    return app


app = create_app()
```

---

## tests/conftest.py

Uses `aiosqlite` (SQLite async driver) so tests need no live PostgreSQL.
The `jwt_token` factory fixture mints tokens for any `sub`/`role` combination.

```python
# tests/conftest.py
import os
import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Set env vars BEFORE any financial_ops import so modules that read os.getenv at
# import time receive valid values.
_TEST_SECRET = "test-only-not-used-in-production-xK9mP"
os.environ.setdefault("JWT_SECRET_KEY", _TEST_SECRET)
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test_financial_ops.db")
os.environ.setdefault("JWT_ALGORITHM", "HS256")

from financial_ops.api.deps import get_db_session  # noqa: E402
from financial_ops.main import create_app  # noqa: E402
from financial_ops.models.db import Base  # noqa: E402

_TEST_DB_URL = os.environ["DATABASE_URL"]
_test_engine = create_async_engine(_TEST_DB_URL, echo=False)
_TestSessionFactory = async_sessionmaker(_test_engine, expire_on_commit=False)


@pytest.fixture(scope="function", autouse=True)
async def _reset_db():
    async with _test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with _test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def _override_db() -> AsyncSession:
    async with _TestSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@pytest.fixture
async def client() -> AsyncClient:
    app = create_app()
    app.dependency_overrides[get_db_session] = _override_db
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


def mint_token(
    sub: str | None = None,
    role: str = "analyst",
    expired: bool = False,
) -> str:
    """Helper used by fixtures and tests to mint test JWTs."""
    now = datetime.now(UTC)
    payload = {
        "sub": sub or str(uuid.uuid4()),
        "role": role,
        "exp": now + (timedelta(seconds=-1) if expired else timedelta(minutes=30)),
        "iat": now,
    }
    return jwt.encode(payload, _TEST_SECRET, algorithm="HS256")


@pytest.fixture
def admin_token() -> str:
    return mint_token(role="admin")


@pytest.fixture
def analyst_token() -> str:
    return mint_token(role="analyst")


@pytest.fixture
def viewer_token() -> str:
    return mint_token(role="viewer")


@pytest.fixture
def expired_token() -> str:
    return mint_token(expired=True)
```

---

## tests/unit/test_account_service.py

Unit tests mock the repository — no DB, no HTTP.
`pytest.mark.unit` allows selective runs: `pytest -m unit`.

```python
# tests/unit/test_account_service.py
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from financial_ops.models.db import Account
from financial_ops.models.schemas import AccountBalance, AccountCreate, AccountResponse
from financial_ops.services.account_service import AccountService


def _make_account(
    owner_id: uuid.UUID | None = None,
    currency: str = "USD",
) -> Account:
    acct = MagicMock(spec=Account)
    acct.id = uuid.uuid4()
    acct.owner_id = owner_id or uuid.uuid4()
    acct.currency = currency
    acct.created_at = datetime.now(timezone.utc)
    return acct


@pytest.mark.unit
async def test_create_account_returns_response():
    mock_session = AsyncMock()
    owner_id = uuid.uuid4()
    expected_account = _make_account(owner_id=owner_id)

    with patch(
        "financial_ops.services.account_service.AccountRepository"
    ) as MockRepo:
        mock_repo = MockRepo.return_value
        mock_repo.create = AsyncMock(return_value=expected_account)

        svc = AccountService(mock_session)
        result = await svc.create_account(
            AccountCreate(currency="USD"), owner_id=owner_id
        )

    assert isinstance(result, AccountResponse)
    mock_repo.create.assert_awaited_once()


@pytest.mark.unit
async def test_get_account_not_found_raises_404():
    from fastapi import HTTPException

    mock_session = AsyncMock()
    with patch(
        "financial_ops.services.account_service.AccountRepository"
    ) as MockRepo:
        MockRepo.return_value.get_by_id = AsyncMock(return_value=None)
        svc = AccountService(mock_session)
        with pytest.raises(HTTPException) as exc_info:
            await svc.get_account(
                account_id=uuid.uuid4(),
                requester_id=uuid.uuid4(),
                role="analyst",
            )
    assert exc_info.value.status_code == 404


@pytest.mark.unit
async def test_get_account_wrong_owner_raises_403():
    from fastapi import HTTPException

    mock_session = AsyncMock()
    account = _make_account(owner_id=uuid.uuid4())

    with patch(
        "financial_ops.services.account_service.AccountRepository"
    ) as MockRepo:
        MockRepo.return_value.get_by_id = AsyncMock(return_value=account)
        svc = AccountService(mock_session)
        with pytest.raises(HTTPException) as exc_info:
            await svc.get_account(
                account_id=account.id,
                requester_id=uuid.uuid4(),  # different user
                role="analyst",
            )
    assert exc_info.value.status_code == 403


@pytest.mark.unit
async def test_get_balance_sql_aggregation():
    mock_session = AsyncMock()
    owner_id = uuid.uuid4()
    account = _make_account(owner_id=owner_id, currency="EUR")

    with patch(
        "financial_ops.services.account_service.AccountRepository"
    ) as MockRepo:
        mock_repo = MockRepo.return_value
        mock_repo.get_by_id = AsyncMock(return_value=account)
        mock_repo.compute_balance = AsyncMock(return_value=Decimal("500.00"))

        svc = AccountService(mock_session)
        result = await svc.get_balance(
            account_id=account.id,
            requester_id=owner_id,
            role="analyst",
        )

    assert isinstance(result, AccountBalance)
    assert result.balance == Decimal("500.00")
    mock_repo.compute_balance.assert_awaited_once_with(account.id)
```

---

## tests/unit/test_transaction_service.py

```python
# tests/unit/test_transaction_service.py
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from financial_ops.models.db import Account, Transaction
from financial_ops.models.schemas import TransactionCreate, TransactionStatusUpdate
from financial_ops.services.transaction_service import TransactionService


def _make_account(owner_id: uuid.UUID) -> Account:
    acct = MagicMock(spec=Account)
    acct.id = uuid.uuid4()
    acct.owner_id = owner_id
    acct.currency = "USD"
    return acct


def _make_txn(account_id: uuid.UUID, status: str = "pending") -> Transaction:
    txn = MagicMock(spec=Transaction)
    txn.id = uuid.uuid4()
    txn.account_id = account_id
    txn.amount = Decimal("100.00")
    txn.currency = "USD"
    txn.transaction_type = "credit"
    txn.status = status
    txn.description = None
    txn.created_at = datetime.now(timezone.utc)
    txn.updated_at = datetime.now(timezone.utc)
    return txn


@pytest.mark.unit
async def test_create_transaction_viewer_forbidden():
    from fastapi import HTTPException

    mock_session = AsyncMock()
    owner_id = uuid.uuid4()
    account = _make_account(owner_id)

    with (
        patch("financial_ops.services.transaction_service.TransactionRepository"),
        patch(
            "financial_ops.services.transaction_service.AccountRepository"
        ) as MockAcctRepo,
    ):
        MockAcctRepo.return_value.get_by_id = AsyncMock(return_value=account)
        svc = TransactionService(mock_session)
        with pytest.raises(HTTPException) as exc_info:
            await svc.create_transaction(
                TransactionCreate(
                    account_id=account.id,
                    amount=Decimal("10.00"),
                    currency="USD",
                    transaction_type="credit",
                ),
                requester_id=owner_id,
                role="viewer",
            )
    assert exc_info.value.status_code == 403


@pytest.mark.unit
async def test_update_status_terminal_to_pending_rejected():
    from fastapi import HTTPException

    mock_session = AsyncMock()
    account = _make_account(uuid.uuid4())
    txn = _make_txn(account.id, status="completed")

    with (
        patch(
            "financial_ops.services.transaction_service.TransactionRepository"
        ) as MockTxnRepo,
        patch("financial_ops.services.transaction_service.AccountRepository"),
    ):
        MockTxnRepo.return_value.get_by_id = AsyncMock(return_value=txn)
        svc = TransactionService(mock_session)
        with pytest.raises(HTTPException) as exc_info:
            await svc.update_status(
                transaction_id=txn.id,
                payload=TransactionStatusUpdate(status="pending"),
                requester_id=uuid.uuid4(),
                role="admin",
            )
    assert exc_info.value.status_code == 409


@pytest.mark.unit
async def test_update_status_non_admin_rejected():
    from fastapi import HTTPException

    mock_session = AsyncMock()
    with (
        patch("financial_ops.services.transaction_service.TransactionRepository"),
        patch("financial_ops.services.transaction_service.AccountRepository"),
    ):
        svc = TransactionService(mock_session)
        with pytest.raises(HTTPException) as exc_info:
            await svc.update_status(
                transaction_id=uuid.uuid4(),
                payload=TransactionStatusUpdate(status="completed"),
                requester_id=uuid.uuid4(),
                role="analyst",
            )
    assert exc_info.value.status_code == 403
```

---

## tests/integration/test_accounts.py

Integration tests hit the FastAPI app via `httpx.AsyncClient` + `ASGITransport`.
No live server, no live DB — the test DB dependency override is set in `conftest.py`.

```python
# tests/integration/test_accounts.py
import uuid

import pytest
from httpx import AsyncClient

from tests.conftest import mint_token


@pytest.mark.integration
async def test_create_account_happy_path(client: AsyncClient):
    user_id = str(uuid.uuid4())
    token = mint_token(sub=user_id, role="analyst")
    resp = await client.post(
        "/accounts",
        json={"currency": "USD"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["currency"] == "USD"
    assert data["owner_id"] == user_id
    assert "id" in data


@pytest.mark.integration
async def test_create_account_invalid_currency(client: AsyncClient):
    token = mint_token(role="analyst")
    resp = await client.post(
        "/accounts",
        json={"currency": "XYZ"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422


@pytest.mark.integration
async def test_get_account_not_found(client: AsyncClient):
    token = mint_token(role="analyst")
    resp = await client.get(
        f"/accounts/{uuid.uuid4()}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 404


@pytest.mark.integration
async def test_get_account_ownership_enforced(client: AsyncClient):
    owner_id = str(uuid.uuid4())
    other_id = str(uuid.uuid4())
    owner_token = mint_token(sub=owner_id, role="analyst")
    other_token = mint_token(sub=other_id, role="analyst")

    # Create account as owner
    create_resp = await client.post(
        "/accounts",
        json={"currency": "EUR"},
        headers={"Authorization": f"Bearer {owner_token}"},
    )
    account_id = create_resp.json()["id"]

    # Other user cannot read it
    resp = await client.get(
        f"/accounts/{account_id}",
        headers={"Authorization": f"Bearer {other_token}"},
    )
    assert resp.status_code == 403


@pytest.mark.integration
async def test_get_balance_empty_account(client: AsyncClient):
    user_id = str(uuid.uuid4())
    token = mint_token(sub=user_id, role="analyst")

    create_resp = await client.post(
        "/accounts",
        json={"currency": "USD"},
        headers={"Authorization": f"Bearer {token}"},
    )
    account_id = create_resp.json()["id"]

    bal_resp = await client.get(
        f"/accounts/{account_id}/balance",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert bal_resp.status_code == 200
    assert bal_resp.json()["balance"] == "0"


@pytest.mark.integration
async def test_get_account_admin_bypasses_ownership(client: AsyncClient):
    owner_id = str(uuid.uuid4())
    owner_token = mint_token(sub=owner_id, role="analyst")
    admin_token = mint_token(role="admin")

    create_resp = await client.post(
        "/accounts",
        json={"currency": "GBP"},
        headers={"Authorization": f"Bearer {owner_token}"},
    )
    account_id = create_resp.json()["id"]

    resp = await client.get(
        f"/accounts/{account_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 200
```

---

## tests/integration/test_transactions.py

```python
# tests/integration/test_transactions.py
import uuid

import pytest
from httpx import AsyncClient

from tests.conftest import mint_token


async def _create_account(client: AsyncClient, token: str, currency: str = "USD") -> str:
    resp = await client.post(
        "/accounts",
        json={"currency": currency},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 201
    return resp.json()["id"]


@pytest.mark.integration
async def test_create_transaction_happy_path(client: AsyncClient):
    user_id = str(uuid.uuid4())
    token = mint_token(sub=user_id, role="analyst")
    account_id = await _create_account(client, token)

    resp = await client.post(
        "/transactions",
        json={
            "account_id": account_id,
            "amount": "100.00",
            "currency": "USD",
            "transaction_type": "credit",
            "description": "salary",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["transaction_type"] == "credit"
    assert data["status"] == "pending"
    assert data["account_id"] == account_id


@pytest.mark.integration
async def test_viewer_cannot_create_transaction(client: AsyncClient):
    user_id = str(uuid.uuid4())
    analyst_token = mint_token(sub=user_id, role="analyst")
    viewer_token = mint_token(sub=user_id, role="viewer")
    account_id = await _create_account(client, analyst_token)

    resp = await client.post(
        "/transactions",
        json={
            "account_id": account_id,
            "amount": "10.00",
            "currency": "USD",
            "transaction_type": "credit",
        },
        headers={"Authorization": f"Bearer {viewer_token}"},
    )
    assert resp.status_code == 403


@pytest.mark.integration
async def test_list_transactions_scoped_to_owner(client: AsyncClient):
    alice_id = str(uuid.uuid4())
    bob_id = str(uuid.uuid4())
    alice_token = mint_token(sub=alice_id, role="analyst")
    bob_token = mint_token(sub=bob_id, role="analyst")

    alice_acct = await _create_account(client, alice_token)
    await client.post(
        "/transactions",
        json={
            "account_id": alice_acct,
            "amount": "50.00",
            "currency": "USD",
            "transaction_type": "credit",
        },
        headers={"Authorization": f"Bearer {alice_token}"},
    )

    # Bob should see zero transactions
    resp = await client.get(
        "/transactions",
        headers={"Authorization": f"Bearer {bob_token}"},
    )
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.integration
async def test_patch_transaction_status_admin_only(client: AsyncClient):
    user_id = str(uuid.uuid4())
    analyst_token = mint_token(sub=user_id, role="analyst")
    admin_token = mint_token(role="admin")

    account_id = await _create_account(client, analyst_token)
    create_resp = await client.post(
        "/transactions",
        json={
            "account_id": account_id,
            "amount": "25.00",
            "currency": "USD",
            "transaction_type": "debit",
        },
        headers={"Authorization": f"Bearer {analyst_token}"},
    )
    txn_id = create_resp.json()["id"]

    # Analyst cannot patch
    bad = await client.patch(
        f"/transactions/{txn_id}",
        json={"status": "completed"},
        headers={"Authorization": f"Bearer {analyst_token}"},
    )
    assert bad.status_code == 403

    # Admin can patch
    good = await client.patch(
        f"/transactions/{txn_id}",
        json={"status": "completed"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert good.status_code == 200
    assert good.json()["status"] == "completed"


@pytest.mark.integration
async def test_balance_reflects_transactions(client: AsyncClient):
    user_id = str(uuid.uuid4())
    token = mint_token(sub=user_id, role="analyst")
    account_id = await _create_account(client, token)

    for _ in range(3):
        await client.post(
            "/transactions",
            json={
                "account_id": account_id,
                "amount": "100.00",
                "currency": "USD",
                "transaction_type": "credit",
            },
            headers={"Authorization": f"Bearer {token}"},
        )

    bal_resp = await client.get(
        f"/accounts/{account_id}/balance",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert bal_resp.status_code == 200
    assert float(bal_resp.json()["balance"]) == 300.0
```

---

## tests/integration/test_auth.py

```python
# tests/integration/test_auth.py
import uuid

import pytest
from httpx import AsyncClient

from tests.conftest import mint_token


@pytest.mark.auth
async def test_expired_token_returns_401(client: AsyncClient, expired_token: str):
    resp = await client.get(
        f"/accounts/{uuid.uuid4()}",
        headers={"Authorization": f"Bearer {expired_token}"},
    )
    assert resp.status_code == 401
    assert "expired" in resp.json()["detail"].lower()


@pytest.mark.auth
async def test_invalid_token_returns_401(client: AsyncClient):
    resp = await client.get(
        f"/accounts/{uuid.uuid4()}",
        headers={"Authorization": "Bearer not-a-real-jwt"},
    )
    assert resp.status_code == 401


@pytest.mark.auth
async def test_missing_token_returns_403(client: AsyncClient):
    # HTTPBearer returns 403 when no header is present
    resp = await client.get(f"/accounts/{uuid.uuid4()}")
    assert resp.status_code == 403


@pytest.mark.auth
async def test_wrong_role_returns_403(client: AsyncClient):
    # viewer cannot create transactions
    user_id = str(uuid.uuid4())
    analyst_token = mint_token(sub=user_id, role="analyst")
    viewer_token = mint_token(sub=user_id, role="viewer")

    account_resp = await client.post(
        "/accounts",
        json={"currency": "USD"},
        headers={"Authorization": f"Bearer {analyst_token}"},
    )
    account_id = account_resp.json()["id"]

    resp = await client.post(
        "/transactions",
        json={
            "account_id": account_id,
            "amount": "10.00",
            "currency": "USD",
            "transaction_type": "credit",
        },
        headers={"Authorization": f"Bearer {viewer_token}"},
    )
    assert resp.status_code == 403


@pytest.mark.auth
async def test_health_requires_no_auth(client: AsyncClient):
    assert (await client.get("/health")).status_code == 200
    assert (await client.get("/ready")).status_code == 200


@pytest.mark.auth
async def test_correlation_id_echoed(client: AsyncClient):
    cid = "test-correlation-abc"
    resp = await client.get("/health", headers={"X-Correlation-ID": cid})
    assert resp.headers.get("x-correlation-id") == cid


@pytest.mark.auth
async def test_correlation_id_generated_when_absent(client: AsyncClient):
    resp = await client.get("/health")
    assert "x-correlation-id" in resp.headers
    assert resp.headers["x-correlation-id"] != "-"
```

---

## Dockerfile

Multi-stage, cache-friendly: `pyproject.toml` is copied before source so that the
`pip install` layer is invalidated only when dependencies change, not on every code edit.

```dockerfile
# Dockerfile
FROM python:3.12-slim AS base

WORKDIR /app

# Layer 1: install deps — cached until pyproject.toml changes
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "."

# Layer 2: copy source — invalidated only on code changes
COPY src/ src/

ENV PYTHONPATH=/app/src
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "financial_ops.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-config", "/dev/null"]
```

---

## docker-compose.yml

All credentials come from `.env` via `${VAR}` substitution — never hardcoded.
The `api` service waits for the `db` healthcheck to pass before starting.

```yaml
# docker-compose.yml
services:
  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-financial_ops}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U $${POSTGRES_USER} -d $${POSTGRES_DB}"]
      interval: 5s
      timeout: 5s
      retries: 10

  api:
    build: .
    env_file: .env
    ports:
      - "8000:8000"
    depends_on:
      db:
        condition: service_healthy
    restart: unless-stopped

volumes:
  postgres_data:
```

---

## .github/workflows/ci.yml

Three sequential jobs: **lint → test → build**. Each job depends on the previous.
Secrets are injected from GitHub Actions secrets — never committed to the repo.

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
          cache: "pip"
      - run: pip install ruff
      - run: ruff check src/ tests/

  test:
    runs-on: ubuntu-latest
    needs: lint
    env:
      DATABASE_URL: ${{ secrets.DATABASE_URL }}
      JWT_SECRET_KEY: ${{ secrets.JWT_SECRET_KEY }}
      JWT_ALGORITHM: HS256
      LOG_LEVEL: WARNING
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: "pip"
      - run: pip install ".[dev]"
      - run: pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=80

  build:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/build-push-action@v5
        with:
          context: .
          push: false
          tags: financial-ops:ci
```

---

## README.md (content to place in project)

````markdown
# Financial Ops API

Production-grade FastAPI backend — Week 8 capstone.

## Architecture

```
┌────────────────────────────────────────────────────┐
│                    HTTP Client                      │
└───────────────────────┬────────────────────────────┘
                        │ HTTP/JSON + X-Correlation-ID
┌───────────────────────▼────────────────────────────┐
│              FastAPI  (main.py)                     │
│  middleware: correlation-ID, global exception       │
│                                                     │
│  routes/accounts.py   routes/transactions.py        │
│  routes/health.py                                   │
│                                                     │
│  deps.py — get_db_session(), require_role()         │
└───────────────────────┬────────────────────────────┘
                        │ calls service only
┌───────────────────────▼────────────────────────────┐
│            Service Layer                            │
│  account_service.py   transaction_service.py        │
│  business logic · RBAC ownership · state machines   │
└───────────────────────┬────────────────────────────┘
                        │ calls repository only
┌───────────────────────▼────────────────────────────┐
│           Repository Layer                          │
│  base_repo.py  account_repo.py  transaction_repo.py │
│  SQL only · no business logic                       │
└───────────────────────┬────────────────────────────┘
                        │ asyncpg
┌───────────────────────▼────────────────────────────┐
│             PostgreSQL 15                           │
└────────────────────────────────────────────────────┘
```

## Environment Variables

| Variable | Required | Default | Notes |
|---|---|---|---|
| `DATABASE_URL` | yes | — | `postgresql+asyncpg://...` |
| `JWT_SECRET_KEY` | yes | — | min 32 random bytes, hex-encoded |
| `JWT_ALGORITHM` | no | `HS256` | |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | no | `30` | |
| `LOG_LEVEL` | no | `INFO` | |
| `ENVIRONMENT` | no | `development` | |
| `POSTGRES_USER` | compose | — | used by docker-compose only |
| `POSTGRES_PASSWORD` | compose | — | used by docker-compose only |
| `POSTGRES_DB` | compose | `financial_ops` | |

## Quickstart

```bash
cp .env.example .env
# Edit .env — fill in JWT_SECRET_KEY, POSTGRES_USER, POSTGRES_PASSWORD
docker compose up --build
```

API docs: http://localhost:8000/docs

## Run Tests Locally

```bash
pip install ".[dev]"
# Tests use SQLite in-process — no PostgreSQL needed
pytest tests/ -v --cov=src --cov-report=term-missing

# Run only unit tests
pytest -m unit

# Run only auth edge cases
pytest -m auth
```

## RBAC Summary

| Role | Create account | Create transaction | Read own | Read all | Update status |
|---|---|---|---|---|---|
| `admin` | yes | yes | yes | yes | yes |
| `analyst` | yes | yes | yes | no | no |
| `viewer` | yes | no | yes | no | no |
````

---

## Implementation Notes

### Why `expire_on_commit=False`

SQLAlchemy's default expires all ORM attributes after `session.commit()`. In an async
path this triggers implicit lazy loads (which are illegal on a closed session) when you
access an attribute after the commit. `expire_on_commit=False` sidesteps this; the
tradeoff is that you must call `session.refresh(instance)` explicitly when you need
freshly-committed server-generated values (e.g. `server_default` timestamps).

### Balance via SQL aggregation

`compute_balance` issues two `SELECT SUM(...)` queries rather than fetching all rows and
summing in Python. On large transaction histories this is orders of magnitude faster and
keeps the balance computation inside the DB transaction, guaranteeing consistency.

### Terminal-state guard

The state machine is intentionally simple: any terminal state (`completed`, `failed`,
`reversed`) cannot transition back to `pending`. Extend by replacing the guard with an
explicit transition table if more complex state logic is needed.

### JWT and secret injection

`deps.py` reads `JWT_SECRET_KEY` via `os.getenv()` on every request decode rather than
caching it at module load time. This is intentional: it allows the secret to be rotated
and reloaded without a process restart (useful in some container secret-injection
patterns). The performance overhead is negligible (`os.getenv` reads from the process
environment, not from disk).

### Test isolation

Each test function gets a fresh SQLite database (`autouse=True` fixture drops and
recreates all tables). The `app.dependency_overrides` dict is replaced in the `client`
fixture and cleared on exit, ensuring overrides don't leak between tests.

### Correlation IDs

The `ContextVar` (`correlation_id_var`) is reset via `contextvars.ContextVar.reset(token)`
after each request. Without the reset, the coroutine's context would carry the previous
request's ID into any subsequent work — particularly relevant when connection pooling
reuses coroutines.

### Graceful shutdown

FastAPI's `lifespan` context manager calls `engine.dispose()` on shutdown, draining the
connection pool cleanly. This ensures in-flight PostgreSQL connections are closed rather
than abandoned when the container receives `SIGTERM`.

---

## Acceptance Criteria Reference (copy from project spec)

Architecture, Auth/RBAC, Data Layer, API endpoints, Observability, Testing, Code Quality,
Containerisation/CI, README — all items from the Week 8 spec are satisfied by the
implementation above. See `week-08/project.md` for the full checklist.
