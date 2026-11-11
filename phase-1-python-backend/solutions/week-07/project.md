# Week 7 Project — Financial Operations REST API: Reference Implementation

## Project Structure

```
finops/
├── src/
│   └── finops/
│       ├── __init__.py
│       ├── main.py
│       ├── config.py
│       ├── database.py
│       ├── models.py
│       ├── schemas.py
│       ├── auth.py
│       ├── dependencies.py
│       └── routers/
│           ├── __init__.py
│           ├── auth.py
│           ├── customers.py
│           ├── accounts.py
│           └── transactions.py
├── tests/
│   ├── conftest.py
│   ├── test_customers.py
│   └── test_transactions.py
├── .env.example
├── pyproject.toml
└── alembic/
    ├── env.py
    └── versions/
```

---

## pyproject.toml (dependencies section)

```toml
[project]
name = "finops"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "sqlalchemy[asyncio]>=2.0.0",
    "asyncpg>=0.29.0",
    "pydantic>=2.9.0",
    "pydantic-settings>=2.5.0",
    "python-jose[cryptography]>=3.3.0",
    "passlib[bcrypt]>=1.7.4",
    "python-multipart>=0.0.12",
    "alembic>=1.13.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
    "httpx>=0.27.0",
    "anyio>=4.6.0",
    "aiosqlite>=0.20.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
target-version = "py312"
line-length = 88
select = ["E", "F", "I", "UP", "B", "SIM"]
```

---

## .env.example

```dotenv
# Application
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000

# Database — asyncpg driver required
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/finops

# JWT
JWT_SECRET_KEY=replace-with-a-random-256-bit-hex-string
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30

# Rate limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW_SECONDS=60
```

---

## src/finops/config.py

```python
import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    database_url: str = os.getenv("DATABASE_URL", "")

    jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "")
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30

    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60


settings = Settings()
```

---

## src/finops/database.py

```python
import logging
import os
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

logger = logging.getLogger(__name__)

_database_url = os.getenv("DATABASE_URL")
if not _database_url:
    raise RuntimeError("DATABASE_URL environment variable is not set")

engine = create_async_engine(
    _database_url,
    echo=False,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an AsyncSession; commit on success, rollback on exception."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
```

---

## src/finops/models.py

```python
import enum
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum,
    ForeignKey,
    Numeric,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from finops.database import Base


class TransactionType(str, enum.Enum):
    debit = "debit"
    credit = "credit"


class TransactionStatus(str, enum.Enum):
    pending = "pending"
    completed = "completed"
    failed = "failed"
    reversed = "reversed"


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(254), nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    accounts: Mapped[list["Account"]] = relationship(
        "Account", back_populates="customer", cascade="all, delete-orphan"
    )

    __table_args__ = (UniqueConstraint("email", name="uq_customers_email"),)


class Account(Base):
    __tablename__ = "accounts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    customer_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("customers.id", ondelete="CASCADE"), nullable=False
    )
    account_number: Mapped[str] = mapped_column(String(20), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="USD")
    balance: Mapped[Decimal] = mapped_column(
        Numeric(precision=18, scale=4), nullable=False, default=Decimal("0.0000")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    customer: Mapped["Customer"] = relationship("Customer", back_populates="accounts")
    transactions: Mapped[list["Transaction"]] = relationship(
        "Transaction", back_populates="account", cascade="all, delete-orphan"
    )

    __table_args__ = (UniqueConstraint("account_number", name="uq_accounts_number"),)


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False
    )
    amount: Mapped[Decimal] = mapped_column(
        Numeric(precision=18, scale=4), nullable=False
    )
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    transaction_type: Mapped[TransactionType] = mapped_column(
        Enum(TransactionType), nullable=False
    )
    status: Mapped[TransactionStatus] = mapped_column(
        Enum(TransactionStatus), nullable=False, default=TransactionStatus.pending
    )
    reference: Mapped[str | None] = mapped_column(String(255), nullable=True)
    description: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    account: Mapped["Account"] = relationship("Account", back_populates="transactions")
```

---

## src/finops/schemas.py

```python
from datetime import datetime
from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, ConfigDict, EmailStr, Field, model_validator
from typing_extensions import Self

from finops.models import TransactionStatus, TransactionType


# ── Customer ─────────────────────────────────────────────────────────────────


class CustomerCreate(BaseModel):
    email: EmailStr
    full_name: Annotated[str, Field(min_length=1, max_length=255)]
    password: Annotated[str, Field(min_length=8, max_length=128)]


class CustomerRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr
    full_name: str
    created_at: datetime


# ── Account ───────────────────────────────────────────────────────────────────


class AccountRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    customer_id: int
    account_number: str
    currency: str
    balance: Decimal
    created_at: datetime


class AccountBalance(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    account_id: int
    account_number: str
    currency: str
    balance: Decimal


# ── Transaction ───────────────────────────────────────────────────────────────


class TransactionCreate(BaseModel):
    account_id: int
    amount: Annotated[Decimal, Field(gt=Decimal("0"), decimal_places=4)]
    currency: Annotated[str, Field(min_length=3, max_length=3)]
    transaction_type: TransactionType
    reference: Annotated[str | None, Field(max_length=255)] = None
    description: Annotated[str | None, Field(max_length=512)] = None


class TransactionPatch(BaseModel):
    status: TransactionStatus | None = None
    description: Annotated[str | None, Field(max_length=512)] = None

    @model_validator(mode="after")
    def at_least_one_field(self) -> Self:
        if self.status is None and self.description is None:
            raise ValueError("at least one field must be provided for update")
        return self


class TransactionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    account_id: int
    amount: Decimal
    currency: str
    transaction_type: TransactionType
    status: TransactionStatus
    reference: str | None
    description: str | None
    created_at: datetime
    updated_at: datetime


class PaginatedTransactions(BaseModel):
    items: list[TransactionRead]
    next_cursor: str | None


# ── Auth ──────────────────────────────────────────────────────────────────────


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    sub: str
    exp: int


# ── Error ─────────────────────────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    detail: str
```

---

## src/finops/auth.py

```python
import logging
import os
from datetime import UTC, datetime, timedelta

from jose import JWTError, jwt
from passlib.context import CryptContext

logger = logging.getLogger(__name__)

_jwt_secret = os.getenv("JWT_SECRET_KEY")
if not _jwt_secret:
    raise RuntimeError("JWT_SECRET_KEY environment variable is not set")

_jwt_algorithm = os.getenv("JWT_ALGORITHM", "HS256")
_access_token_expire_minutes = int(
    os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30")
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(subject: str, expires_delta: timedelta | None = None) -> str:
    expire = datetime.now(UTC) + (
        expires_delta or timedelta(minutes=_access_token_expire_minutes)
    )
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, _jwt_secret, algorithm=_jwt_algorithm)


def decode_access_token(token: str) -> dict:
    """
    Decode and verify a JWT. Raises JWTError on invalid or expired token.
    The caller is responsible for mapping JWTError to an HTTP 401.
    """
    return jwt.decode(token, _jwt_secret, algorithms=[_jwt_algorithm])
```

---

## src/finops/dependencies.py

```python
import logging
from collections import defaultdict
from collections.abc import AsyncGenerator

from fastapi import Depends, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from finops.auth import decode_access_token
from finops.config import settings
from finops.database import get_db
from finops.models import Customer

logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# Per-worker in-memory store: {ip: (count, window_start_epoch)}
_rate_store: dict[str, tuple[int, float]] = defaultdict(lambda: (0, 0.0))


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def rate_limit(request: Request, response: Response) -> None:
    import time

    ip = _client_ip(request)
    now = time.time()
    count, window_start = _rate_store[ip]

    if now - window_start >= settings.rate_limit_window_seconds:
        count = 0
        window_start = now

    count += 1
    _rate_store[ip] = (count, window_start)

    reset_at = int(window_start + settings.rate_limit_window_seconds)
    remaining = max(0, settings.rate_limit_requests - count)

    response.headers["X-RateLimit-Limit"] = str(settings.rate_limit_requests)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Reset"] = str(reset_at)

    if count > settings.rate_limit_requests:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded.",
            headers={
                "Retry-After": str(max(0, reset_at - int(now))),
                "X-RateLimit-Limit": str(settings.rate_limit_requests),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(reset_at),
            },
        )


async def get_current_customer(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> Customer:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_access_token(token)
        customer_id_str: str | None = payload.get("sub")
        if customer_id_str is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    result = await db.execute(
        select(Customer).where(Customer.id == int(customer_id_str))
    )
    customer = result.scalar_one_or_none()
    if customer is None:
        raise credentials_exception
    return customer
```

---

## src/finops/routers/auth.py

```python
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from finops.auth import create_access_token, verify_password
from finops.database import get_db
from finops.models import Customer
from finops.schemas import TokenResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/token", response_model=TokenResponse, summary="Obtain access token")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    result = await db.execute(
        select(Customer).where(Customer.email == form_data.username)
    )
    customer = result.scalar_one_or_none()
    if customer is None or not verify_password(
        form_data.password, customer.hashed_password
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(subject=str(customer.id))
    logger.info("Issued token for customer_id=%s", customer.id)
    return TokenResponse(access_token=token)
```

---

## src/finops/routers/customers.py

```python
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from finops.auth import hash_password
from finops.database import get_db
from finops.dependencies import get_current_customer, rate_limit
from finops.models import Customer
from finops.schemas import CustomerCreate, CustomerRead, ErrorResponse

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/customers",
    tags=["customers"],
    dependencies=[Depends(rate_limit)],
)


@router.post(
    "",
    response_model=CustomerRead,
    status_code=status.HTTP_201_CREATED,
    responses={409: {"model": ErrorResponse}},
    summary="Register a new customer",
)
async def create_customer(
    body: CustomerCreate,
    db: AsyncSession = Depends(get_db),
) -> CustomerRead:
    existing = await db.execute(
        select(Customer).where(Customer.email == body.email)
    )
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Email '{body.email}' is already registered",
        )

    customer = Customer(
        email=body.email,
        full_name=body.full_name,
        hashed_password=hash_password(body.password),
    )
    db.add(customer)
    try:
        await db.flush()
        await db.refresh(customer)
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    logger.info("Created customer id=%s email=%s", customer.id, customer.email)
    return CustomerRead.model_validate(customer)


@router.get(
    "/{customer_id}",
    response_model=CustomerRead,
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
    summary="Retrieve a customer by ID",
)
async def get_customer(
    customer_id: int,
    db: AsyncSession = Depends(get_db),
    current_customer: Customer = Depends(get_current_customer),
) -> CustomerRead:
    if current_customer.id != customer_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    result = await db.execute(select(Customer).where(Customer.id == customer_id))
    customer = result.scalar_one_or_none()
    if customer is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer {customer_id} not found",
        )
    return CustomerRead.model_validate(customer)
```

---

## src/finops/routers/accounts.py

```python
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from finops.database import get_db
from finops.dependencies import get_current_customer, rate_limit
from finops.models import Account, Customer
from finops.schemas import AccountBalance, ErrorResponse

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/accounts",
    tags=["accounts"],
    dependencies=[Depends(rate_limit)],
)


@router.get(
    "/{account_id}/balance",
    response_model=AccountBalance,
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
    summary="Get account balance",
)
async def get_balance(
    account_id: int,
    db: AsyncSession = Depends(get_db),
    current_customer: Customer = Depends(get_current_customer),
) -> AccountBalance:
    result = await db.execute(select(Account).where(Account.id == account_id))
    account = result.scalar_one_or_none()

    if account is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Account {account_id} not found",
        )
    if account.customer_id != current_customer.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    logger.info(
        "Balance query account_id=%s customer_id=%s", account_id, current_customer.id
    )
    return AccountBalance(
        account_id=account.id,
        account_number=account.account_number,
        currency=account.currency,
        balance=account.balance,
    )
```

---

## src/finops/routers/transactions.py

```python
import base64
import logging
from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from finops.database import get_db
from finops.dependencies import get_current_customer, rate_limit
from finops.models import Account, Customer, Transaction, TransactionStatus
from finops.schemas import (
    ErrorResponse,
    PaginatedTransactions,
    TransactionCreate,
    TransactionPatch,
    TransactionRead,
)

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/transactions",
    tags=["transactions"],
    dependencies=[Depends(rate_limit)],
)


def _encode_cursor(item_id: int) -> str:
    return base64.urlsafe_b64encode(str(item_id).encode()).decode()


def _decode_cursor(cursor: str) -> int:
    try:
        return int(base64.urlsafe_b64decode(cursor.encode()).decode())
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid pagination cursor: {cursor}",
        ) from exc


@router.post(
    "",
    response_model=TransactionRead,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
    summary="Create a transaction",
)
async def create_transaction(
    body: TransactionCreate,
    db: AsyncSession = Depends(get_db),
    current_customer: Customer = Depends(get_current_customer),
) -> TransactionRead:
    result = await db.execute(select(Account).where(Account.id == body.account_id))
    account = result.scalar_one_or_none()

    if account is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Account {body.account_id} not found",
        )
    if account.customer_id != current_customer.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )
    if body.currency != account.currency:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Currency mismatch: account uses {account.currency}",
        )

    if body.transaction_type.value == "debit":
        if account.balance < body.amount:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Insufficient funds",
            )
        account.balance = account.balance - body.amount
    else:
        account.balance = account.balance + body.amount

    txn = Transaction(
        account_id=body.account_id,
        amount=body.amount,
        currency=body.currency,
        transaction_type=body.transaction_type,
        status=TransactionStatus.completed,
        reference=body.reference,
        description=body.description,
    )
    db.add(txn)
    await db.flush()
    await db.refresh(txn)

    logger.info(
        "Created transaction id=%s type=%s amount=%s account_id=%s",
        txn.id,
        txn.transaction_type,
        txn.amount,
        txn.account_id,
    )
    return TransactionRead.model_validate(txn)


@router.get(
    "",
    response_model=PaginatedTransactions,
    responses={400: {"model": ErrorResponse}},
    summary="List transactions (cursor-paginated)",
)
async def list_transactions(
    db: AsyncSession = Depends(get_db),
    current_customer: Customer = Depends(get_current_customer),
    account_id: int | None = Query(default=None, description="Filter by account"),
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: str | None = Query(default=None, description="Pagination cursor"),
) -> PaginatedTransactions:
    stmt = (
        select(Transaction)
        .join(Account, Transaction.account_id == Account.id)
        .where(Account.customer_id == current_customer.id)
        .order_by(Transaction.id.asc())
        .limit(limit + 1)
    )

    if account_id is not None:
        stmt = stmt.where(Transaction.account_id == account_id)

    if cursor is not None:
        after_id = _decode_cursor(cursor)
        stmt = stmt.where(Transaction.id > after_id)

    result = await db.execute(stmt)
    rows = result.scalars().all()

    has_more = len(rows) > limit
    page = rows[:limit]

    next_cursor: str | None = _encode_cursor(page[-1].id) if has_more else None

    return PaginatedTransactions(
        items=[TransactionRead.model_validate(t) for t in page],
        next_cursor=next_cursor,
    )


@router.get(
    "/{transaction_id}",
    response_model=TransactionRead,
    responses={
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
    summary="Retrieve a transaction by ID",
)
async def get_transaction(
    transaction_id: int,
    db: AsyncSession = Depends(get_db),
    current_customer: Customer = Depends(get_current_customer),
) -> TransactionRead:
    result = await db.execute(
        select(Transaction)
        .join(Account, Transaction.account_id == Account.id)
        .where(Transaction.id == transaction_id)
    )
    txn = result.scalar_one_or_none()

    if txn is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction {transaction_id} not found",
        )

    account_result = await db.execute(
        select(Account).where(Account.id == txn.account_id)
    )
    account = account_result.scalar_one_or_none()
    if account is None or account.customer_id != current_customer.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )
    return TransactionRead.model_validate(txn)


@router.patch(
    "/{transaction_id}",
    response_model=TransactionRead,
    responses={
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
    summary="Partially update a transaction",
)
async def patch_transaction(
    transaction_id: int,
    body: TransactionPatch,
    db: AsyncSession = Depends(get_db),
    current_customer: Customer = Depends(get_current_customer),
) -> TransactionRead:
    result = await db.execute(
        select(Transaction).where(Transaction.id == transaction_id)
    )
    txn = result.scalar_one_or_none()

    if txn is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction {transaction_id} not found",
        )

    account_result = await db.execute(
        select(Account).where(Account.id == txn.account_id)
    )
    account = account_result.scalar_one_or_none()
    if account is None or account.customer_id != current_customer.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    if body.status is not None:
        txn.status = body.status
    if body.description is not None:
        txn.description = body.description

    await db.flush()
    await db.refresh(txn)

    logger.info("Patched transaction id=%s", txn.id)
    return TransactionRead.model_validate(txn)
```

---

## src/finops/main.py

```python
import logging
import os
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError

from finops.database import Base, engine
from finops.routers import accounts, auth, customers, transactions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FinOps API — creating tables if needed")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    logger.info("Shutting down FinOps API")
    await engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Financial Operations API",
        description="REST API for managing customers, accounts, and transactions.",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Middleware ────────────────────────────────────────────────────────────

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
        logger.info(
            "method=%s path=%s status=%s ms=%.1f request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request_id,
        )
        return response

    # ── Exception handlers ────────────────────────────────────────────────────

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": exc.errors()},
        )

    @app.exception_handler(IntegrityError)
    async def integrity_error_handler(
        request: Request, exc: IntegrityError
    ) -> JSONResponse:
        logger.error("IntegrityError on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"detail": "Database conflict. The resource may already exist."},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("Unhandled exception on %s", request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )

    # ── Routers ───────────────────────────────────────────────────────────────

    app.include_router(auth.router)
    app.include_router(customers.router)
    app.include_router(accounts.router)
    app.include_router(transactions.router)

    @app.get("/health", tags=["ops"], include_in_schema=False)
    async def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "finops.main:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", "8000")),
        reload=os.getenv("APP_ENV", "development") == "development",
    )
```

---

## tests/conftest.py

```python
import os
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from finops.database import Base, get_db
from finops.main import create_app

# Use SQLite for tests — no external DB required
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"

# Must be set before finops.auth is imported by any module under test
os.environ.setdefault("JWT_SECRET_KEY", "test-only-key-not-used-in-production")
os.environ.setdefault("DATABASE_URL", TEST_DATABASE_URL)

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
test_session_factory = async_sessionmaker(
    test_engine, class_=AsyncSession, expire_on_commit=False
)


@pytest_asyncio.fixture(scope="function", autouse=True)
async def setup_db():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def override_get_db():
    async with test_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@pytest_asyncio.fixture
async def client():
    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest_asyncio.fixture
async def customer_token(client: AsyncClient) -> str:
    """Register a customer and return a valid Bearer token."""
    await client.post(
        "/customers",
        json={
            "email": "alice@example.com",
            "full_name": "Alice Tester",
            "password": "securepassword123",
        },
    )
    resp = await client.post(
        "/auth/token",
        data={"username": "alice@example.com", "password": "securepassword123"},
    )
    return resp.json()["access_token"]
```

---

## tests/test_customers.py

```python
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_customer_success(client: AsyncClient):
    response = await client.post(
        "/customers",
        json={
            "email": "bob@example.com",
            "full_name": "Bob Builder",
            "password": "strongpassword",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "bob@example.com"
    assert data["full_name"] == "Bob Builder"
    assert "id" in data
    assert "hashed_password" not in data


@pytest.mark.asyncio
async def test_create_customer_duplicate_email(client: AsyncClient):
    payload = {
        "email": "dup@example.com",
        "full_name": "Dup User",
        "password": "somepassword",
    }
    r1 = await client.post("/customers", json=payload)
    assert r1.status_code == 201

    r2 = await client.post("/customers", json=payload)
    assert r2.status_code == 409


@pytest.mark.asyncio
async def test_create_customer_invalid_email(client: AsyncClient):
    response = await client.post(
        "/customers",
        json={"email": "not-an-email", "full_name": "Bad User", "password": "pass1234"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_customer_short_password(client: AsyncClient):
    response = await client.post(
        "/customers",
        json={"email": "short@example.com", "full_name": "Short Pass", "password": "abc"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_customer_unauthenticated(client: AsyncClient):
    response = await client.get("/customers/1")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_get_customer_self(client: AsyncClient, customer_token: str):
    # customer_token fixture creates customer id=1
    response = await client.get(
        "/customers/1",
        headers={"Authorization": f"Bearer {customer_token}"},
    )
    assert response.status_code == 200
    assert response.json()["email"] == "alice@example.com"


@pytest.mark.asyncio
async def test_get_customer_forbidden(client: AsyncClient, customer_token: str):
    # Create a second customer
    await client.post(
        "/customers",
        json={
            "email": "carol@example.com",
            "full_name": "Carol",
            "password": "password123",
        },
    )
    # Alice (id=1) tries to access Carol (id=2)
    response = await client.get(
        "/customers/2",
        headers={"Authorization": f"Bearer {customer_token}"},
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_get_customer_not_found(client: AsyncClient, customer_token: str):
    response = await client.get(
        "/customers/9999",
        headers={"Authorization": f"Bearer {customer_token}"},
    )
    # Returns 403 (access denied for other users) or 404 depending on implementation
    assert response.status_code in (403, 404)
```

---

## tests/test_transactions.py

```python
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from finops.models import Account
from tests.conftest import test_session_factory


async def _create_account_for_customer(customer_id: int, balance: float = 1000.0) -> int:
    """Helper: insert an account directly into the test DB and return its id."""
    from decimal import Decimal

    async with test_session_factory() as session:
        account = Account(
            customer_id=customer_id,
            account_number=f"ACC-{customer_id}-001",
            currency="USD",
            balance=Decimal(str(balance)),
        )
        session.add(account)
        await session.commit()
        await session.refresh(account)
        return account.id


@pytest.mark.asyncio
async def test_create_transaction_credit(client: AsyncClient, customer_token: str):
    account_id = await _create_account_for_customer(customer_id=1)

    response = await client.post(
        "/transactions",
        json={
            "account_id": account_id,
            "amount": "200.00",
            "currency": "USD",
            "transaction_type": "credit",
            "description": "Initial deposit",
        },
        headers={"Authorization": f"Bearer {customer_token}"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["transaction_type"] == "credit"
    assert data["status"] == "completed"
    assert float(data["amount"]) == 200.0


@pytest.mark.asyncio
async def test_create_transaction_debit_insufficient_funds(
    client: AsyncClient, customer_token: str
):
    account_id = await _create_account_for_customer(customer_id=1, balance=50.0)

    response = await client.post(
        "/transactions",
        json={
            "account_id": account_id,
            "amount": "500.00",
            "currency": "USD",
            "transaction_type": "debit",
        },
        headers={"Authorization": f"Bearer {customer_token}"},
    )
    assert response.status_code == 400
    assert "Insufficient funds" in response.json()["detail"]


@pytest.mark.asyncio
async def test_create_transaction_currency_mismatch(
    client: AsyncClient, customer_token: str
):
    account_id = await _create_account_for_customer(customer_id=1)

    response = await client.post(
        "/transactions",
        json={
            "account_id": account_id,
            "amount": "10.00",
            "currency": "EUR",
            "transaction_type": "credit",
        },
        headers={"Authorization": f"Bearer {customer_token}"},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_create_transaction_account_not_found(
    client: AsyncClient, customer_token: str
):
    response = await client.post(
        "/transactions",
        json={
            "account_id": 9999,
            "amount": "10.00",
            "currency": "USD",
            "transaction_type": "credit",
        },
        headers={"Authorization": f"Bearer {customer_token}"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_list_transactions_paginated(client: AsyncClient, customer_token: str):
    account_id = await _create_account_for_customer(customer_id=1, balance=10000.0)
    headers = {"Authorization": f"Bearer {customer_token}"}

    # Create 5 transactions
    for i in range(5):
        await client.post(
            "/transactions",
            json={
                "account_id": account_id,
                "amount": "10.00",
                "currency": "USD",
                "transaction_type": "credit",
                "description": f"txn {i}",
            },
            headers=headers,
        )

    # Fetch first page of 3
    r1 = await client.get(
        "/transactions",
        params={"limit": 3},
        headers=headers,
    )
    assert r1.status_code == 200
    page1 = r1.json()
    assert len(page1["items"]) == 3
    assert page1["next_cursor"] is not None

    # Fetch second page using cursor
    r2 = await client.get(
        "/transactions",
        params={"limit": 3, "cursor": page1["next_cursor"]},
        headers=headers,
    )
    assert r2.status_code == 200
    page2 = r2.json()
    assert len(page2["items"]) == 2
    assert page2["next_cursor"] is None

    # Verify no overlap between pages
    ids_page1 = {t["id"] for t in page1["items"]}
    ids_page2 = {t["id"] for t in page2["items"]}
    assert ids_page1.isdisjoint(ids_page2)


@pytest.mark.asyncio
async def test_get_transaction_not_found(client: AsyncClient, customer_token: str):
    response = await client.get(
        "/transactions/9999",
        headers={"Authorization": f"Bearer {customer_token}"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_patch_transaction_status(client: AsyncClient, customer_token: str):
    account_id = await _create_account_for_customer(customer_id=1)
    headers = {"Authorization": f"Bearer {customer_token}"}

    create_resp = await client.post(
        "/transactions",
        json={
            "account_id": account_id,
            "amount": "50.00",
            "currency": "USD",
            "transaction_type": "credit",
        },
        headers=headers,
    )
    txn_id = create_resp.json()["id"]

    patch_resp = await client.patch(
        f"/transactions/{txn_id}",
        json={"status": "reversed"},
        headers=headers,
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["status"] == "reversed"


@pytest.mark.asyncio
async def test_patch_transaction_empty_body_rejected(
    client: AsyncClient, customer_token: str
):
    account_id = await _create_account_for_customer(customer_id=1)
    headers = {"Authorization": f"Bearer {customer_token}"}

    create_resp = await client.post(
        "/transactions",
        json={
            "account_id": account_id,
            "amount": "10.00",
            "currency": "USD",
            "transaction_type": "credit",
        },
        headers=headers,
    )
    txn_id = create_resp.json()["id"]

    patch_resp = await client.patch(
        f"/transactions/{txn_id}",
        json={},
        headers=headers,
    )
    assert patch_resp.status_code == 422
```

---

## Implementation Notes

### Balance Consistency

The `create_transaction` route updates `account.balance` in the same SQLAlchemy unit of work as inserting the `Transaction` row. Both changes commit atomically. In high-concurrency production systems, replace this with `SELECT FOR UPDATE` or an optimistic lock (`version_id` column with `mapper_args = {"version_id_col": ...}`) to prevent race conditions.

### Cursor Pagination vs. Offset

Cursor pagination maintains stable results under concurrent writes and scales to large datasets because the query uses an indexed `WHERE id > ?` condition. Offset pagination (`LIMIT n OFFSET k`) degrades as `k` grows and produces skipped or duplicated rows when rows are inserted between page fetches.

### JWT Revocation

The current implementation is stateless: tokens are valid until `exp`. To support logout or forced revocation (e.g., password change, compromised credential), maintain a Redis-backed blocklist keyed by token `jti` claim with TTL equal to the remaining token lifetime.

### Rate Limiting in Production

The in-memory `_rate_store` dict is per-worker and does not share state across multiple Uvicorn processes or pods. In production, back the rate limiter with Redis using `redis-py` async client and atomic `INCR` + `EXPIRE` operations, or delegate to an API gateway (Kong, AWS API Gateway, nginx `limit_req`).

### Secret Management

All secrets are consumed via `os.getenv()` with no fallback defaults. The application raises `RuntimeError` at startup if `JWT_SECRET_KEY` or `DATABASE_URL` are absent. In production, inject these via a secrets manager (AWS Secrets Manager, HashiCorp Vault, Kubernetes Secret) rather than plain environment variables where possible.
