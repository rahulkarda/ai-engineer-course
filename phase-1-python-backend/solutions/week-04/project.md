# Week 4 Reference Solution — Reusable Transaction Library

## Package Structure

```
transaction_lib/
├── src/
│   └── transaction_lib/
│       ├── __init__.py
│       ├── models.py
│       ├── processor.py
│       ├── validators.py
│       ├── exceptions.py
│       └── config.py
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_processor.py
│   ├── test_validators.py
│   └── test_exceptions.py
├── pyproject.toml
└── README.md
```

---

## `src/transaction_lib/exceptions.py`

```python
"""Custom exception hierarchy for transaction_lib."""


class TransactionError(Exception):
    """Base exception for all transaction library errors."""


class ValidationError(TransactionError):
    """Raised when input data fails validation."""

    def __init__(self, message: str, field_name: str) -> None:
        super().__init__(message)
        self.field_name = field_name


class ProcessingError(TransactionError):
    """Raised when transaction processing fails."""

    def __init__(self, message: str, transaction_id: str) -> None:
        super().__init__(message)
        self.transaction_id = transaction_id


class CurrencyError(ValidationError):
    """Raised when an unsupported or malformed currency is provided."""


class AmountError(ValidationError):
    """Raised when a transaction amount is out of acceptable range."""
```

---

## `src/transaction_lib/models.py`

```python
"""Pydantic v2 domain models for transaction_lib."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import (
    UUID4,
    BaseModel,
    Field,
    computed_field,
    field_validator,
    model_validator,
)
from typing_extensions import Self


class TransactionStatus(str, Enum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


Currency = Literal["USD", "EUR", "GBP", "JPY", "CHF"]

SUPPORTED_CURRENCIES: frozenset[str] = frozenset({"USD", "EUR", "GBP", "JPY", "CHF"})


class Transaction(BaseModel):
    id: UUID4 = Field(default_factory=uuid.uuid4)
    amount: Decimal = Field(gt=Decimal("0"))
    currency: Currency
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    merchant: str = Field(min_length=1)
    status: TransactionStatus = TransactionStatus.PENDING
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("currency", mode="before")
    @classmethod
    def currency_uppercase(cls, v: str) -> str:
        return v.upper() if isinstance(v, str) else v

    @model_validator(mode="after")
    def ensure_id(self) -> Self:
        if not self.id:
            object.__setattr__(self, "id", uuid.uuid4())
        return self


class Address(BaseModel):
    street: str
    city: str
    country_code: str = Field(min_length=2, max_length=2)


class TransactionBatch(BaseModel):
    batch_id: UUID4 = Field(default_factory=uuid.uuid4)
    transactions: list[Transaction]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @computed_field  # type: ignore[misc]
    @property
    def total_amount(self) -> Decimal:
        return sum((t.amount for t in self.transactions), Decimal("0"))
```

---

## `src/transaction_lib/validators.py`

```python
"""Standalone validation functions for transaction fields."""

from __future__ import annotations

import re
from decimal import Decimal

from transaction_lib.exceptions import AmountError, CurrencyError, ValidationError

SUPPORTED_CURRENCIES: frozenset[str] = frozenset({"USD", "EUR", "GBP", "JPY", "CHF"})
MAX_AMOUNT = Decimal("1_000_000")
_MERCHANT_RE = re.compile(r"^[A-Za-z0-9 \-&'.]+$")


def validate_amount(amount: Decimal) -> Decimal:
    """Return *amount* unchanged if valid; raise AmountError otherwise."""
    if amount <= Decimal("0"):
        raise AmountError(
            f"Amount must be positive, got {amount}",
            field_name="amount",
        )
    if amount > MAX_AMOUNT:
        raise AmountError(
            f"Amount {amount} exceeds maximum {MAX_AMOUNT}",
            field_name="amount",
        )
    return amount


def validate_currency(currency: str) -> str:
    """Return normalised currency string or raise CurrencyError."""
    normalised = currency.strip().upper()
    if normalised not in SUPPORTED_CURRENCIES:
        raise CurrencyError(
            f"Unsupported currency '{currency}'. Supported: {sorted(SUPPORTED_CURRENCIES)}",
            field_name="currency",
        )
    return normalised


def validate_merchant(name: str) -> str:
    """Return *name* unchanged if valid; raise ValidationError otherwise."""
    stripped = name.strip()
    if not stripped:
        raise ValidationError("Merchant name must not be empty.", field_name="merchant")
    if not _MERCHANT_RE.match(stripped):
        raise ValidationError(
            f"Merchant name '{stripped}' contains disallowed characters.",
            field_name="merchant",
        )
    return stripped
```

---

## `src/transaction_lib/processor.py`

```python
"""High-level processing utilities and TransactionProcessor class."""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Generator, Iterable
from decimal import Decimal
from types import TracebackType
from typing import Any, Callable, TypeVar

from transaction_lib.models import Currency, Transaction, TransactionBatch, TransactionStatus

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def timer(func: F) -> F:
    """Decorator: log wall-clock execution time of *func*."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug("%s completed in %.4fs", func.__qualname__, elapsed)
        return result

    return wrapper  # type: ignore[return-value]


def logged(func: F) -> F:
    """Decorator: log function entry and exit (DEBUG level)."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.debug("Entering %s", func.__qualname__)
        result = func(*args, **kwargs)
        logger.debug("Exiting %s", func.__qualname__)
        return result

    return wrapper  # type: ignore[return-value]


def filter_transactions(
    transactions: Iterable[Transaction],
    *,
    min_amount: Decimal | None = None,
    max_amount: Decimal | None = None,
    currencies: set[str] | None = None,
    statuses: set[TransactionStatus] | None = None,
) -> Generator[Transaction, None, None]:
    """Yield transactions that match all supplied filter criteria."""
    for txn in transactions:
        if min_amount is not None and txn.amount < min_amount:
            continue
        if max_amount is not None and txn.amount > max_amount:
            continue
        if currencies is not None and txn.currency not in currencies:
            continue
        if statuses is not None and txn.status not in statuses:
            continue
        yield txn


def transform_to_summary(
    transactions: Iterable[Transaction],
) -> Generator[dict[str, Any], None, None]:
    """Yield a slim summary dict for each transaction."""
    for txn in transactions:
        yield {
            "id": str(txn.id),
            "amount": txn.amount,
            "currency": txn.currency,
            "merchant": txn.merchant,
        }


class TransactionProcessor:
    """Stateful processor with context-manager lifecycle tracking."""

    def __init__(self) -> None:
        self._processed: int = 0
        self._errors: int = 0
        self._active: bool = False

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "TransactionProcessor":
        self._active = True
        self._processed = 0
        self._errors = 0
        logger.info("TransactionProcessor session started.")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        self._active = False
        if exc_type is not None:
            logger.error(
                "TransactionProcessor session ended with error: %s", exc_val
            )
        else:
            logger.info(
                "TransactionProcessor session ended. processed=%d errors=%d",
                self._processed,
                self._errors,
            )
        # Do not suppress exceptions
        return False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def processed_count(self) -> int:
        return self._processed

    @property
    def error_count(self) -> int:
        return self._errors

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    @timer
    @logged
    def process_batch(self, batch: TransactionBatch) -> list[Transaction]:
        """Process a batch, updating internal counters.

        Returns the list of successfully processed transactions.
        Marks each processed transaction status as COMPLETED.
        """
        if not self._active:
            raise RuntimeError("process_batch must be called inside a context manager.")

        results: list[Transaction] = []
        for txn in batch.transactions:
            try:
                updated = txn.model_copy(update={"status": TransactionStatus.COMPLETED})
                results.append(updated)
                self._processed += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to process transaction %s: %s", txn.id, exc)
                self._errors += 1

        return results
```

---

## `src/transaction_lib/config.py`

```python
"""Application configuration via pydantic-settings."""

from __future__ import annotations

from decimal import Decimal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TXLIB_",
        env_file=".env",
        extra="ignore",
    )

    max_amount: Decimal = Field(default=Decimal("1_000_000"))
    supported_currencies: list[str] = Field(
        default=["USD", "EUR", "GBP", "JPY", "CHF"]
    )
    log_level: str = Field(default="INFO")
    batch_size: int = Field(default=100)
```

---

## `src/transaction_lib/__init__.py`

```python
"""transaction_lib — reusable transaction processing package."""

from transaction_lib.config import Settings
from transaction_lib.exceptions import (
    AmountError,
    CurrencyError,
    ProcessingError,
    TransactionError,
    ValidationError,
)
from transaction_lib.models import (
    Address,
    Currency,
    Transaction,
    TransactionBatch,
    TransactionStatus,
)
from transaction_lib.processor import TransactionProcessor, filter_transactions, transform_to_summary
from transaction_lib.validators import validate_amount, validate_currency, validate_merchant

__all__ = [
    "Address",
    "AmountError",
    "Currency",
    "CurrencyError",
    "ProcessingError",
    "Settings",
    "Transaction",
    "TransactionBatch",
    "TransactionError",
    "TransactionProcessor",
    "TransactionStatus",
    "ValidationError",
    "filter_transactions",
    "transform_to_summary",
    "validate_amount",
    "validate_currency",
    "validate_merchant",
]
```

---

## `tests/__init__.py`

```python
```

---

## `tests/test_models.py`

```python
"""Tests for transaction_lib.models."""

from __future__ import annotations

import uuid
from decimal import Decimal

import pytest

from transaction_lib.models import Transaction, TransactionBatch, TransactionStatus


def _make_transaction(**overrides) -> Transaction:
    defaults = {
        "amount": Decimal("99.99"),
        "currency": "USD",
        "merchant": "ACME Corp",
    }
    defaults.update(overrides)
    return Transaction(**defaults)


class TestTransaction:
    def test_valid_transaction_creation(self) -> None:
        txn = _make_transaction()
        assert txn.amount == Decimal("99.99")
        assert txn.currency == "USD"
        assert txn.merchant == "ACME Corp"
        assert txn.status == TransactionStatus.PENDING
        assert txn.id is not None

    def test_transaction_with_invalid_amount_raises(self) -> None:
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            _make_transaction(amount=Decimal("-1"))

        with pytest.raises(PydanticValidationError):
            _make_transaction(amount=Decimal("0"))

    def test_transaction_serialization_roundtrip(self) -> None:
        txn = _make_transaction()
        data = txn.model_dump()
        restored = Transaction(**data)
        assert restored == txn

    def test_currency_normalised_to_uppercase(self) -> None:
        txn = _make_transaction(currency="usd")
        assert txn.currency == "USD"


class TestTransactionBatch:
    def test_transaction_batch_total_amount(self) -> None:
        txns = [
            _make_transaction(amount=Decimal("100.00")),
            _make_transaction(amount=Decimal("200.50")),
            _make_transaction(amount=Decimal("50.25")),
        ]
        batch = TransactionBatch(transactions=txns)
        assert batch.total_amount == Decimal("350.75")

    def test_empty_batch_total_is_zero(self) -> None:
        batch = TransactionBatch(transactions=[])
        assert batch.total_amount == Decimal("0")
```

---

## `tests/test_processor.py`

```python
"""Tests for transaction_lib.processor."""

from __future__ import annotations

from decimal import Decimal

import pytest

from transaction_lib.models import Transaction, TransactionBatch, TransactionStatus
from transaction_lib.processor import TransactionProcessor, filter_transactions


def _txn(amount: str, currency: str = "USD", status: TransactionStatus = TransactionStatus.PENDING) -> Transaction:
    return Transaction(amount=Decimal(amount), currency=currency, merchant="Test Merchant", status=status)


class TestFilterTransactions:
    def test_filter_by_amount_range(self) -> None:
        txns = [_txn("10"), _txn("50"), _txn("200"), _txn("1000")]
        result = list(
            filter_transactions(txns, min_amount=Decimal("20"), max_amount=Decimal("500"))
        )
        assert len(result) == 2
        amounts = {t.amount for t in result}
        assert amounts == {Decimal("50"), Decimal("200")}

    def test_filter_by_currency(self) -> None:
        txns = [_txn("10", "USD"), _txn("20", "EUR"), _txn("30", "GBP")]
        result = list(filter_transactions(txns, currencies={"USD", "EUR"}))
        assert len(result) == 2

    def test_filter_no_criteria_returns_all(self) -> None:
        txns = [_txn("10"), _txn("20"), _txn("30")]
        result = list(filter_transactions(txns))
        assert len(result) == 3


class TestTransactionProcessor:
    def _make_batch(self, n: int = 3) -> TransactionBatch:
        return TransactionBatch(transactions=[_txn(str(i * 10 + 10)) for i in range(n)])

    def test_processor_context_manager_counts(self) -> None:
        batch = self._make_batch(3)
        with TransactionProcessor() as proc:
            proc.process_batch(batch)
        assert proc.processed_count == 3
        assert proc.error_count == 0

    def test_processor_context_manager_rollback_on_exception(self) -> None:
        batch = self._make_batch(2)
        proc = TransactionProcessor()
        try:
            with proc:
                proc.process_batch(batch)
                raise ValueError("Simulated failure")
        except ValueError:
            pass
        # counts still reflect work done before exception
        assert proc.processed_count == 2
        # processor is no longer active
        assert not proc._active

    def test_process_batch_outside_context_raises(self) -> None:
        batch = self._make_batch(1)
        proc = TransactionProcessor()
        with pytest.raises(RuntimeError, match="context manager"):
            proc.process_batch(batch)

    def test_process_batch_marks_completed(self) -> None:
        batch = self._make_batch(2)
        with TransactionProcessor() as proc:
            results = proc.process_batch(batch)
        assert all(t.status == TransactionStatus.COMPLETED for t in results)
```

---

## `tests/test_validators.py`

```python
"""Tests for transaction_lib.validators."""

from __future__ import annotations

from decimal import Decimal

import pytest

from transaction_lib.exceptions import AmountError, CurrencyError, ValidationError
from transaction_lib.validators import validate_amount, validate_currency, validate_merchant


class TestValidateAmount:
    def test_validate_amount_positive(self) -> None:
        result = validate_amount(Decimal("100.00"))
        assert result == Decimal("100.00")

    def test_validate_amount_zero_raises(self) -> None:
        with pytest.raises(AmountError) as exc_info:
            validate_amount(Decimal("0"))
        assert exc_info.value.field_name == "amount"

    def test_validate_amount_negative_raises(self) -> None:
        with pytest.raises(AmountError):
            validate_amount(Decimal("-5.00"))

    def test_validate_amount_too_large_raises(self) -> None:
        with pytest.raises(AmountError) as exc_info:
            validate_amount(Decimal("1_000_001"))
        assert exc_info.value.field_name == "amount"

    def test_validate_amount_at_max_passes(self) -> None:
        result = validate_amount(Decimal("1_000_000"))
        assert result == Decimal("1_000_000")


class TestValidateCurrency:
    def test_validate_currency_valid(self) -> None:
        assert validate_currency("USD") == "USD"
        assert validate_currency("eur") == "EUR"

    def test_validate_currency_invalid_raises(self) -> None:
        with pytest.raises(CurrencyError) as exc_info:
            validate_currency("XYZ")
        assert exc_info.value.field_name == "currency"

    def test_validate_currency_empty_raises(self) -> None:
        with pytest.raises(CurrencyError):
            validate_currency("")


class TestValidateMerchant:
    def test_validate_merchant_valid(self) -> None:
        assert validate_merchant("ACME Corp") == "ACME Corp"

    def test_validate_merchant_empty_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            validate_merchant("")
        assert exc_info.value.field_name == "merchant"

    def test_validate_merchant_special_chars_raises(self) -> None:
        with pytest.raises(ValidationError):
            validate_merchant("Evil<script>")
```

---

## `tests/test_exceptions.py`

```python
"""Tests for transaction_lib.exceptions hierarchy."""

from __future__ import annotations

import pytest

from transaction_lib.exceptions import (
    AmountError,
    CurrencyError,
    ProcessingError,
    TransactionError,
    ValidationError,
)


class TestExceptionHierarchy:
    def test_exception_hierarchy(self) -> None:
        amount_err = AmountError("bad amount", field_name="amount")
        currency_err = CurrencyError("bad currency", field_name="currency")
        validation_err = ValidationError("bad field", field_name="x")
        processing_err = ProcessingError("failed", transaction_id="txn-123")

        assert isinstance(amount_err, ValidationError)
        assert isinstance(amount_err, TransactionError)
        assert isinstance(amount_err, Exception)

        assert isinstance(currency_err, ValidationError)
        assert isinstance(currency_err, TransactionError)

        assert isinstance(validation_err, TransactionError)

        assert isinstance(processing_err, TransactionError)
        assert not isinstance(processing_err, ValidationError)

    def test_validation_error_has_field_name(self) -> None:
        err = ValidationError("field is invalid", field_name="merchant")
        assert err.field_name == "merchant"
        assert str(err) == "field is invalid"

    def test_processing_error_has_transaction_id(self) -> None:
        err = ProcessingError("processing failed", transaction_id="abc-001")
        assert err.transaction_id == "abc-001"
        assert str(err) == "processing failed"

    def test_amount_error_field_name(self) -> None:
        err = AmountError("too large", field_name="amount")
        assert err.field_name == "amount"

    def test_currency_error_field_name(self) -> None:
        err = CurrencyError("unknown", field_name="currency")
        assert err.field_name == "currency"
```

---

## `pyproject.toml`

```toml
[project]
name = "transaction-lib"
version = "0.1.0"
description = "Reusable transaction processing library — Week 4 mini-project"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov",
    "ruff>=0.4",
]

[build-system]
requires = ["setuptools>=70", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v"

[tool.coverage.run]
source = ["transaction_lib"]
```

---

## `README.md`

```markdown
# transaction_lib

Reusable transaction processing library. Week 4 mini-project.

## Install (editable)

```bash
pip install -e ".[dev]"
```

## Run tests

```bash
pytest
```

## Environment variables

All settings are prefixed `TXLIB_`. Copy `.env.example` to `.env` and adjust.

| Variable | Default | Description |
|---|---|---|
| `TXLIB_MAX_AMOUNT` | `1000000` | Maximum allowed transaction amount |
| `TXLIB_SUPPORTED_CURRENCIES` | `["USD","EUR","GBP","JPY","CHF"]` | Accepted currencies |
| `TXLIB_LOG_LEVEL` | `INFO` | Python logging level |
| `TXLIB_BATCH_SIZE` | `100` | Default processing batch size |
```

---

## `.env.example`

```dotenv
TXLIB_MAX_AMOUNT=1000000
TXLIB_SUPPORTED_CURRENCIES=["USD","EUR","GBP","JPY","CHF"]
TXLIB_LOG_LEVEL=INFO
TXLIB_BATCH_SIZE=100
```

---

## Key Design Decisions

- **`exceptions.py`** — thin hierarchy; `ValidationError` carries `field_name`, `ProcessingError` carries `transaction_id` so callers can route errors without string parsing.
- **`models.py`** — `Decimal` for money (never `float`); `computed_field` for `total_amount` keeps the batch immutable while exposing a derived value; `field_validator` runs before Pydantic's own coercion so the `Currency` literal check sees the normalised string.
- **`processor.py`** — `filter_transactions` and `transform_to_summary` are generators; memory-efficient for large datasets. `timer` + `logged` are composable and stack cleanly via `functools.wraps`.
- **`validators.py`** — standalone functions, not methods, so they can be called from CLI tools, API handlers, or Pydantic validators without importing the full model graph.
- **`config.py`** — `env_prefix="TXLIB_"` prevents collision with other packages; `extra="ignore"` avoids explosions when `.env` contains unrelated vars.
- **Tests** — no real I/O; all external state (env vars, time) either injected or not needed. `model_copy(update=...)` used in processor to honour Pydantic immutability.
