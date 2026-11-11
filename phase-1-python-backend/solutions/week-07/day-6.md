# Week 07 · Day 6 — Solutions

Saturday build session: list pagination, Pydantic v2 validators, and background tasks.

---

## Exercise 1 — `GET /customers` with pagination

Return a consistent envelope so clients always know the full count alongside the page.

```python
# schemas/pagination.py
from typing import Generic, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    """Generic paginated response envelope."""

    items: list[T]
    total: int = Field(..., description="Total number of records matching the query")
    skip: int = Field(..., ge=0, description="Records skipped")
    limit: int = Field(..., ge=1, le=100, description="Page size requested")
```

```python
# routers/customers.py
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_db
from models.customer import Customer
from schemas.customer import CustomerRead
from schemas.pagination import Page

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/customers", tags=["Customers"])


@router.get("", response_model=Page[CustomerRead])
async def list_customers(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(20, le=100, description="Max records to return (hard cap: 100)"),
    session: Annotated[AsyncSession, Depends(get_db)] = None,
) -> Page[CustomerRead]:
    """
    Return a paginated list of customers.

    `skip` and `limit` implement offset-based pagination.  For cursor-based
    pagination on large tables, replace with a keyset approach in a later
    iteration.
    """
    # Count total rows matching the (unfiltered) query.
    count_result = await session.execute(select(func.count(Customer.id)))
    total: int = count_result.scalar_one()

    # Fetch the requested page.
    items_result = await session.execute(
        select(Customer).offset(skip).limit(limit).order_by(Customer.id)
    )
    customers = items_result.scalars().all()

    logger.debug("list_customers skip=%d limit=%d total=%d", skip, limit, total)

    return Page(
        items=list(customers),
        total=total,
        skip=skip,
        limit=limit,
    )
```

---

## Exercise 2 — `TransactionCreate` with Pydantic v2 validators

```python
# schemas/transaction.py
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# Supported currencies as a Literal type — adding a currency means updating
# one place and re-running tests.
SupportedCurrency = Literal["USD", "EUR", "GBP", "JPY"]


class TransactionCreate(BaseModel):
    customer_id: int = Field(..., gt=0, description="ID of the owning customer")
    amount: Decimal = Field(
        ...,
        description="Transaction amount — must be strictly positive",
    )
    currency: SupportedCurrency = Field(
        ...,
        description="ISO 4217 currency code: USD, EUR, GBP, or JPY",
    )
    description: str | None = Field(
        None,
        max_length=255,
        description="Optional human-readable description",
    )

    # ------------------------------------------------------------------
    # Field-level validator (Pydantic v2 style — replaces @validator).
    # mode="before" coerces the raw input before type conversion.
    # ------------------------------------------------------------------
    @field_validator("amount", mode="before")
    @classmethod
    def amount_must_be_positive(cls, value: object) -> object:
        """
        Reject zero and negative amounts before Decimal conversion so the
        error message is friendly rather than a raw type error.
        """
        try:
            numeric = Decimal(str(value))
        except Exception:
            # Let Pydantic produce the type error downstream.
            return value
        if numeric <= 0:
            raise ValueError(
                f"amount must be greater than 0; received {numeric}"
            )
        return value

    # ------------------------------------------------------------------
    # Alternative: validate currency explicitly if SupportedCurrency
    # Literal is not sufficient (e.g. case-insensitive input).
    # ------------------------------------------------------------------
    @field_validator("currency", mode="before")
    @classmethod
    def normalise_currency(cls, value: object) -> object:
        """Accept lowercase input and normalise to uppercase."""
        if isinstance(value, str):
            return value.upper()
        return value

    model_config = {"str_strip_whitespace": True}
```

**Verification — interactive demo:**

```python
from decimal import Decimal
from schemas.transaction import TransactionCreate
from pydantic import ValidationError

# Valid
txn = TransactionCreate(customer_id=1, amount="49.99", currency="usd")
assert txn.currency == "USD"
assert txn.amount == Decimal("49.99")

# Invalid amount
try:
    TransactionCreate(customer_id=1, amount="-10", currency="EUR")
except ValidationError as exc:
    print(exc)
# -> amount must be greater than 0; received -10

# Invalid currency
try:
    TransactionCreate(customer_id=1, amount="10", currency="BTC")
except ValidationError as exc:
    print(exc)
# -> Input should be 'USD', 'EUR', 'GBP' or 'JPY'
```

---

## Exercise 3 — Background task after `POST /transactions`

```python
# routers/transactions.py
import logging
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_db
from models.transaction import Transaction
from schemas.transaction import TransactionCreate, TransactionRead

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/transactions", tags=["Transactions"])


# ---------------------------------------------------------------------------
# Background task function — runs after the HTTP response has been sent,
# so it must not raise unhandled exceptions (they are silently swallowed by
# FastAPI; log them instead).
# ---------------------------------------------------------------------------
def log_transaction_created(transaction_id: int, customer_id: int) -> None:
    """
    Log a structured message after a transaction is persisted.

    This is intentionally lightweight: heavy processing (emails, webhooks,
    billing events) belongs in a proper task queue (Celery, ARQ, etc.).
    """
    logger.info(
        "Transaction %d created for customer %d",
        transaction_id,
        customer_id,
    )


@router.post("", status_code=status.HTTP_201_CREATED, response_model=TransactionRead)
async def create_transaction(
    payload: TransactionCreate,
    background_tasks: BackgroundTasks,          # injected by FastAPI
    session: Annotated[AsyncSession, Depends(get_db)] = None,
) -> TransactionRead:
    """
    Create a transaction and enqueue a background log task.

    The `BackgroundTasks` instance is provided automatically by FastAPI.
    `add_task` schedules the callable to run after the response is sent.
    """
    transaction = Transaction(
        customer_id=payload.customer_id,
        amount=payload.amount,
        currency=payload.currency,
        description=payload.description,
    )
    session.add(transaction)
    await session.commit()
    await session.refresh(transaction)

    # Enqueue the background task.  Arguments are passed positionally or
    # as keyword arguments after the callable.
    background_tasks.add_task(
        log_transaction_created,
        transaction_id=transaction.id,
        customer_id=transaction.customer_id,
    )

    return transaction
```

**Notes on `BackgroundTasks` vs. a task queue:**

| Concern                     | `BackgroundTasks`        | Celery / ARQ / RQ          |
|-----------------------------|--------------------------|----------------------------|
| Runs in the same process    | Yes                      | No (separate worker)       |
| Survives server restart     | No                       | Yes (broker-backed)        |
| Retry on failure            | No                       | Yes (configurable)         |
| Suitable for                | Logging, cache invalidation | Email, billing, webhooks |

Use `BackgroundTasks` only for fast, non-critical, fire-and-forget work.
