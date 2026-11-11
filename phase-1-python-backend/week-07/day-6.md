# Phase 1 · Week 07 · Day 6 — Build Session: Financial Operations REST API (Part 1)

> Objective: Build POST /customers, GET /customers/{id}, POST /transactions with full validation and auth  ·  Time: ~4 hr  ·  Prereqs: [[week-07/day-5]]

## Concept

This is a guided build session. You scaffold a production-shaped FastAPI project from scratch, wire async SQLAlchemy models, define Pydantic schemas, and implement three authenticated endpoints. Each step is small and verifiable.

### Why this structure

```
src/finops/
  models.py       # SQLAlchemy ORM — DB layer only, no business logic
  schemas.py      # Pydantic I/O contracts — what crosses the API boundary
  routers/        # Route handlers — thin: validate, call service, return response
  auth.py         # JWT encode/decode, password hashing, dependencies
  database.py     # Engine, session factory, get_session dependency
  main.py         # App factory, router registration, lifespan
```

Separating models from schemas is critical: ORM models carry DB concerns (nullable, FK, index); schemas carry API concerns (validation, serialization, versioning). Never return an ORM object directly from an endpoint — always go through a `response_model` schema.

## How it works internally

### SQLAlchemy 2.0 async model

```
create_async_engine(url)
        |
async_sessionmaker(engine)  ← configured once at startup
        |
AsyncSession (per request)  ← yielded by get_session(), closed after response
        |
await session.execute(select(Model))
await session.commit()
await session.refresh(instance)
```

`expire_on_commit=False` in `async_sessionmaker` is required for async: without it, accessing attributes after `commit()` triggers a lazy load, which is not allowed in async mode and raises `MissingGreenlet`.

### Alembic hint

Alembic tracks schema migrations in `alembic/versions/`. For this build session, use `Base.metadata.create_all(engine)` directly in a setup script to avoid the migration setup overhead. In production, use Alembic: never `create_all` against a live database that already has data.

### Pydantic v2 `model_config`

```python
model_config = {"from_attributes": True}
```

Enables ORM mode: Pydantic reads attributes from ORM objects instead of expecting a dict. Required for `response_model` to work with SQLAlchemy instances.

## Worked example

### Step 1 — Project scaffold

```bash
mkdir -p src/finops/routers tests/unit tests/integration
touch src/__init__.py src/finops/__init__.py
touch src/finops/{models,schemas,auth,database,main}.py
touch src/finops/routers/__init__.py
touch src/finops/routers/{auth,customers,transactions}.py
touch tests/__init__.py tests/unit/__init__.py tests/integration/__init__.py
touch tests/conftest.py
touch .env.example pyproject.toml
```

`.env.example`:
```
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/finops
JWT_SECRET_KEY=<generate with: python -c "import secrets; print(secrets.token_hex(32))">
TEST_DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/finops_test
```

### Step 2 — Database setup

```python
# src/finops/database.py
import os
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

DATABASE_URL: str = os.getenv("DATABASE_URL", "")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set")

engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True)

# expire_on_commit=False is required for async — prevents lazy-load after commit
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session
```

### Step 3 — SQLAlchemy models

```python
# src/finops/models.py
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Numeric, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(50), default="viewer")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(primary_key=True)
    full_name: Mapped[str] = mapped_column(String(255))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    phone: Mapped[str | None] = mapped_column(String(50), nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    accounts: Mapped[list[Account]] = relationship(back_populates="customer")


class Account(Base):
    __tablename__ = "accounts"

    id: Mapped[int] = mapped_column(primary_key=True)
    account_number: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    account_type: Mapped[str] = mapped_column(String(50))  # checking | savings
    balance: Mapped[Decimal] = mapped_column(Numeric(18, 2), default=Decimal("0.00"))
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    customer: Mapped[Customer] = relationship(back_populates="accounts")
    transactions: Mapped[list[Transaction]] = relationship(back_populates="account")


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(primary_key=True)
    amount: Mapped[Decimal] = mapped_column(Numeric(18, 2))
    direction: Mapped[str] = mapped_column(String(10))  # credit | debit
    description: Mapped[str | None] = mapped_column(String(500), nullable=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    account: Mapped[Account] = relationship(back_populates="transactions")
```

### Step 4 — Pydantic schemas

```python
# src/finops/schemas.py
from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, EmailStr, Field, field_validator


# --- Customer ---

class CustomerCreate(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=255)
    email: EmailStr
    phone: str | None = Field(default=None, pattern=r"^\+?[1-9]\d{7,14}$")


class CustomerResponse(BaseModel):
    id: int
    full_name: str
    email: str
    phone: str | None
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# --- Transaction ---

class TransactionCreate(BaseModel):
    amount: Decimal = Field(..., gt=Decimal("0"), decimal_places=2)
    direction: Literal["credit", "debit"]
    description: str | None = Field(default=None, max_length=500)
    account_id: int

    @field_validator("amount")
    @classmethod
    def amount_precision(cls, v: Decimal) -> Decimal:
        # Enforce max 2 decimal places
        if v.as_tuple().exponent < -2:
            raise ValueError("amount must have at most 2 decimal places")
        return v


class TransactionResponse(BaseModel):
    id: int
    amount: Decimal
    direction: str
    description: str | None
    account_id: int
    created_at: datetime

    model_config = {"from_attributes": True}


# --- Auth ---

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    role: str | None = None


class UserResponse(BaseModel):
    id: int
    email: str
    role: str

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
```

### Step 5 — Auth module

```python
# src/finops/auth.py
import os
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .database import get_session
from .models import User

SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "")
if not SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY environment variable is not set")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    payload = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload["exp"] = expire
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> User:
    exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str | None = payload.get("sub")
        if not user_id:
            raise exc
    except JWTError:
        raise exc

    result = await session.execute(select(User).where(User.id == int(user_id)))
    user = result.scalar_one_or_none()
    if user is None:
        raise exc
    return user


def require_role(*roles: str):
    async def _check(user: Annotated[User, Depends(get_current_user)]) -> User:
        if user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user.role}' is not permitted",
            )
        return user
    return _check
```

### Step 6 — Routers

```python
# src/finops/routers/customers.py
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import get_current_user
from ..database import get_session
from ..models import Customer, User
from ..schemas import CustomerCreate, CustomerResponse

router = APIRouter(prefix="/customers", tags=["customers"])


@router.post(
    "",
    response_model=CustomerResponse,
    status_code=201,
    summary="Create a new customer",
)
async def create_customer(
    body: CustomerCreate,
    session: Annotated[AsyncSession, Depends(get_session)],
    _: Annotated[User, Depends(get_current_user)],
) -> Customer:
    existing = await session.execute(
        select(Customer).where(Customer.email == body.email)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered")

    customer = Customer(**body.model_dump())
    session.add(customer)
    await session.commit()
    await session.refresh(customer)
    return customer


@router.get(
    "/{customer_id}",
    response_model=CustomerResponse,
    summary="Get a customer by ID",
)
async def get_customer(
    customer_id: int,
    session: Annotated[AsyncSession, Depends(get_session)],
    _: Annotated[User, Depends(get_current_user)],
) -> Customer:
    result = await session.execute(
        select(Customer).where(Customer.id == customer_id)
    )
    customer = result.scalar_one_or_none()
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer
```

```python
# src/finops/routers/transactions.py
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import get_current_user
from ..database import get_session
from ..models import Account, Transaction, User
from ..schemas import TransactionCreate, TransactionResponse

router = APIRouter(prefix="/transactions", tags=["transactions"])


@router.post(
    "",
    response_model=TransactionResponse,
    status_code=201,
    summary="Post a new transaction",
)
async def create_transaction(
    body: TransactionCreate,
    session: Annotated[AsyncSession, Depends(get_session)],
    _: Annotated[User, Depends(get_current_user)],
) -> Transaction:
    # Verify account exists before creating transaction
    result = await session.execute(
        select(Account).where(Account.id == body.account_id)
    )
    account = result.scalar_one_or_none()
    if account is None:
        raise HTTPException(status_code=404, detail="Account not found")

    tx = Transaction(**body.model_dump())
    session.add(tx)

    # Update account balance atomically in the same transaction
    if body.direction == "credit":
        account.balance += body.amount
    else:
        account.balance -= body.amount

    await session.commit()
    await session.refresh(tx)
    return tx
```

```python
# src/finops/routers/auth.py
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)
from ..database import get_session
from ..models import User
from ..schemas import TokenResponse, UserCreate, UserResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse, status_code=201)
async def register(
    body: UserCreate,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> User:
    result = await session.execute(select(User).where(User.email == body.email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered")
    user = User(
        email=body.email,
        hashed_password=hash_password(body.password),
        role=body.role or "viewer",
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


@router.post("/login", response_model=TokenResponse)
async def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> TokenResponse:
    result = await session.execute(select(User).where(User.email == form.username))
    user = result.scalar_one_or_none()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token({"sub": str(user.id), "role": user.role})
    return TokenResponse(access_token=token, token_type="bearer")


@router.get("/me", response_model=UserResponse)
async def me(user: Annotated[User, Depends(get_current_user)]) -> User:
    return user
```

### Step 7 — App factory and lifespan

```python
# src/finops/main.py
import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI

from .database import engine
from .models import Base
from .routers import auth, customers, transactions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup: create tables (use Alembic migrations in production)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables verified")
    yield
    # Shutdown: dispose engine connection pool
    await engine.dispose()
    logger.info("Database engine disposed")


app = FastAPI(
    title="FinOps API",
    version="1.0.0",
    description="Financial operations REST API",
    lifespan=lifespan,
)

app.include_router(auth.router)
app.include_router(customers.router)
app.include_router(transactions.router)


@app.get("/health", include_in_schema=False)
async def health() -> dict:
    return {"status": "ok"}
```

### Step 8 — pyproject.toml

```toml
[project]
name = "finops"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "sqlalchemy[asyncio]>=2.0",
    "asyncpg>=0.29",
    "pydantic[email]>=2.7",
    "passlib[bcrypt]>=1.7",
    "python-jose[cryptography]>=3.3",
    "python-multipart>=0.0.9",  # required for OAuth2PasswordRequestForm
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.23",
    "httpx>=0.27",
    "aiosqlite>=0.20",
    "pytest-cov>=5",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

### Running the app

```bash
# Copy and fill in your real values
cp .env.example .env
# Edit .env — set DATABASE_URL and JWT_SECRET_KEY

uvicorn src.finops.main:app --reload --port 8000
# Docs at http://localhost:8000/docs
```

## Exercises

1. The `create_transaction` endpoint updates `account.balance` in Python. Replace this with a SQL-level atomic update using `update(Account).where(...).values(balance=Account.balance + body.amount)` and `await session.execute(...)`. Verify the result is equivalent by checking the balance before and after in a test.

2. Add a `GET /customers` endpoint with `skip: int = 0` and `limit: int = Query(default=20, le=100)` pagination. Return `list[CustomerResponse]`. The response for `limit=0` should return an empty list, not an error.

3. The `Transaction` model has no constraint preventing a debit that would push the account balance below zero. Add a `check_constraint` to the `Account` model (`balance >= 0`), handle the resulting `IntegrityError` in the transaction router, and return a `422` with `detail="Insufficient funds"`.

(Solutions: ../solutions/week-07/day-6.md)

## Self-check

- [ ] I can scaffold a FastAPI project with separate models, schemas, routers, auth, and database modules
- [ ] I can define SQLAlchemy 2.0 async models with `Mapped` type annotations
- [ ] I can write Pydantic v2 schemas with `field_validator` and `model_config`
- [ ] I understand why `expire_on_commit=False` is required for async SQLAlchemy
- [ ] I can wire a `lifespan` handler for startup/shutdown logic
- [ ] I can implement POST and GET endpoints with auth dependencies and error handling
