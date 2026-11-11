# Phase 1 · Week 04 · Day 7 — Pydantic v2 Fundamentals + Assessment Prep

> Objective: Build production-quality data validation and settings management with Pydantic v2, then consolidate Week 4 concepts for the assessment.  ·  Time: ~4 hrs (3 hr project + 1 hr assessment review)  ·  Prereqs: [[day-6]]

## Concept

Pydantic is the de facto standard for data validation in Python. Version 2 (released 2023) rewrote the core in Rust, delivering 5–50x speed improvements, and introduced a cleaner, more explicit API. Nearly every major Python framework — FastAPI, LangChain, LlamaIndex, Haystack — relies on Pydantic for request/response models, config, and agent state.

Pydantic's core value proposition: **define data shapes once with types and constraints; get validation, serialisation, and documentation for free**. A `BaseModel` class is both a type annotation target and a runtime validator. If the input data violates the schema, Pydantic raises a structured `ValidationError` rather than letting invalid data silently propagate.

## How it works internally

### v1 vs v2 differences

| Feature | v1 | v2 |
|---|---|---|
| Core engine | Python + Cython | Rust (pydantic-core) |
| Validators | `@validator` | `@field_validator`, `@model_validator` |
| Config | `class Config` | `model_config = ConfigDict(...)` |
| Serialisation | `.dict()`, `.json()` | `.model_dump()`, `.model_dump_json()` |
| Construction | `.parse_obj()`, `.parse_raw()` | `.model_validate()`, `.model_validate_json()` |
| Speed | baseline | 5-50x faster |
| Strict mode | limited | full strict mode via `ConfigDict(strict=True)` |

In v2, `@validator` still works (with a deprecation warning) for backward compatibility, but new code should use `@field_validator`.

### BaseModel and Field

Every model inherits from `pydantic.BaseModel`. Fields are declared as class-level annotations. Pydantic uses `Field()` to attach constraints and metadata:

```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    name: str = Field(min_length=1, max_length=200, description="Product name")
    price: float = Field(gt=0.0, description="Price in USD, must be positive")
    quantity: int = Field(default=0, ge=0)
```

Common `Field` arguments:
- `default` / `default_factory` — default value or callable
- `gt`, `ge`, `lt`, `le` — numeric comparisons (greater-than, etc.)
- `min_length`, `max_length` — for strings
- `pattern` — regex for strings
- `description` — used in JSON Schema / OpenAPI docs
- `alias` — alternative name when parsing input
- `frozen` — per-field immutability

### Validators

`@field_validator("field_name", mode="before" | "after" | "wrap" | "plain")`:
- `mode="before"`: runs before Pydantic's own type coercion; receives raw input
- `mode="after"`: runs after type coercion; receives the already-typed value
- Must be a `classmethod` decorated with `@classmethod`

`@model_validator(mode="before" | "after" | "wrap")`:
- `mode="before"`: receives raw dict before any field parsing
- `mode="after"`: receives the fully constructed model instance; `self` is available

```python
from pydantic import BaseModel, field_validator, model_validator

class Transfer(BaseModel):
    amount: float
    currency: str

    @field_validator("currency", mode="after")
    @classmethod
    def currency_must_be_upper(cls, v: str) -> str:
        if v != v.upper():
            raise ValueError("currency must be uppercase ISO-4217 code")
        return v

    @model_validator(mode="after")
    def amount_must_be_positive(self) -> "Transfer":
        if self.amount <= 0:
            raise ValueError("amount must be positive")
        return self
```

### Nested models and composition

Pydantic handles nested models automatically:

```python
class Address(BaseModel):
    street: str
    city: str
    country: str

class Customer(BaseModel):
    name: str
    address: Address   # nested model; validates recursively
```

Passing `{"name": "Alice", "address": {"street": "1 Main St", "city": "Berlin", "country": "DE"}}` will construct a `Customer` with an `Address` instance.

### Serialisation and deserialisation

```python
customer = Customer(name="Alice", address=Address(street="1 Main St", city="Berlin", country="DE"))

# To Python dict
d = customer.model_dump()
d = customer.model_dump(exclude={"address": {"street"}})
d = customer.model_dump(include={"name"})

# To JSON string
json_str = customer.model_dump_json()

# From dict
customer2 = Customer.model_validate(d)

# From JSON string
customer3 = Customer.model_validate_json(json_str)
```

### pydantic-settings BaseSettings

`pydantic-settings` extends Pydantic to read config from environment variables, `.env` files, secrets files, and other sources:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str
    debug: bool = False
    max_connections: int = 10
```

`AppSettings()` reads `DATABASE_URL`, `DEBUG`, `MAX_CONNECTIONS` from environment (case-insensitive) or `.env` file, validates and coerces them.

### Custom types and validators

```python
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

class PositiveInt:
    """Custom type that ensures a positive integer."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: object, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.int_schema(gt=0)
```

### Discriminated unions

Discriminated unions let Pydantic choose among model variants based on a literal `type` field, with better performance and clearer errors than a plain `Union`:

```python
from typing import Literal
from pydantic import BaseModel

class TextMessage(BaseModel):
    type: Literal["text"]
    content: str

class ImageMessage(BaseModel):
    type: Literal["image"]
    url: str
    width: int
    height: int

from typing import Annotated, Union
from pydantic import Field

Message = Annotated[
    Union[TextMessage, ImageMessage],
    Field(discriminator="type"),
]
```

### ValidationError

When validation fails, Pydantic raises `pydantic.ValidationError`. It contains a list of `ErrorDetails` dicts, each with `loc` (field path), `msg` (human message), `type` (error code), and `input` (the invalid value).

```python
from pydantic import ValidationError

try:
    Transfer(amount=-100, currency="usd")
except ValidationError as exc:
    for error in exc.errors():
        print(error["loc"], error["msg"])
```

## Worked example

```python
"""
day7_pydantic.py
Full demonstration: Transaction + Address models, validators, settings,
discriminated unions, error handling.
Python 3.12+ | pydantic>=2.0 | pydantic-settings>=2.0
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Currency(StrEnum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"


class TransactionStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Nested model: Address
# ---------------------------------------------------------------------------

class Address(BaseModel):
    model_config = ConfigDict(frozen=True)  # immutable after construction

    street: str = Field(min_length=1, max_length=200)
    city: str = Field(min_length=1, max_length=100)
    postal_code: str = Field(pattern=r"^[A-Z0-9\- ]{3,10}$")
    country_code: str = Field(min_length=2, max_length=2, description="ISO 3166-1 alpha-2")

    @field_validator("country_code", mode="after")
    @classmethod
    def country_must_be_upper(cls, v: str) -> str:
        return v.upper()


# ---------------------------------------------------------------------------
# Main model: Transaction
# ---------------------------------------------------------------------------

class Transaction(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,  # strip leading/trailing whitespace from strings
        use_enum_values=True,       # store enum values, not enum members
        populate_by_name=True,      # allow both field name and alias
    )

    id: str = Field(
        pattern=r"^TXN-[0-9]{8}$",
        description="Transaction ID in format TXN-XXXXXXXX",
    )
    amount: Annotated[Decimal, Field(gt=Decimal("0"), decimal_places=2)]
    currency: Currency
    status: TransactionStatus = TransactionStatus.PENDING
    sender: Address
    recipient: Address
    reference: str | None = Field(default=None, max_length=140)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("amount", mode="before")
    @classmethod
    def coerce_amount(cls, v: object) -> object:
        """Accept string representations of decimals."""
        if isinstance(v, str):
            try:
                return Decimal(v)
            except Exception as exc:
                raise ValueError(f"Cannot parse amount: {v!r}") from exc
        return v

    @field_validator("id", mode="after")
    @classmethod
    def id_must_be_upper(cls, v: str) -> str:
        return v.upper()

    @model_validator(mode="after")
    def sender_and_recipient_must_differ(self) -> Transaction:
        """Business rule: sender and recipient addresses must not be identical."""
        if self.sender == self.recipient:
            raise ValueError(
                "Sender and recipient addresses must be different"
            )
        return self


# ---------------------------------------------------------------------------
# Discriminated union: Message types
# ---------------------------------------------------------------------------

class TextMessage(BaseModel):
    type: Literal["text"]
    content: str = Field(min_length=1, max_length=4096)


class FileMessage(BaseModel):
    type: Literal["file"]
    filename: str
    size_bytes: int = Field(gt=0)
    mime_type: str


class ErrorMessage(BaseModel):
    type: Literal["error"]
    code: int = Field(ge=400, le=599)
    detail: str


Message = Annotated[
    Union[TextMessage, FileMessage, ErrorMessage],
    Field(discriminator="type"),
]


class MessageEnvelope(BaseModel):
    sender_id: str
    payload: Message


# ---------------------------------------------------------------------------
# Settings via pydantic-settings
# ---------------------------------------------------------------------------

class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="APP_",          # reads APP_DATABASE_URL, APP_DEBUG, etc.
        case_sensitive=False,
    )

    database_url: str = "postgresql://localhost:5432/dev"
    debug: bool = False
    max_connections: int = Field(default=10, ge=1, le=500)
    api_key: str = Field(default="dev-key", min_length=8)


# ---------------------------------------------------------------------------
# Demo functions
# ---------------------------------------------------------------------------

def demo_transaction() -> None:
    log.info("=== Transaction model ===")

    sender = Address(
        street="  1 Hacker Way  ",   # whitespace stripped
        city="Menlo Park",
        postal_code="94025",
        country_code="us",           # uppercased by validator
    )
    recipient = Address(
        street="10 Downing Street",
        city="London",
        postal_code="SW1A 2AA",
        country_code="GB",
    )

    txn = Transaction(
        id="txn-00000001",           # uppercased by validator
        amount="1250.50",            # string coerced to Decimal by before-validator
        currency=Currency.GBP,
        sender=sender,
        recipient=recipient,
        reference="Invoice #42",
        metadata={"source": "api", "version": "2"},
    )
    log.info("Transaction: %s", txn.model_dump())
    log.info("JSON: %s", txn.model_dump_json(indent=2))

    # Round-trip: dump -> validate
    raw = txn.model_dump()
    txn2 = Transaction.model_validate(raw)
    log.info("Round-trip equal: %s", txn == txn2)


def demo_validation_errors() -> None:
    log.info("=== ValidationError ===")

    try:
        Transaction(
            id="INVALID",
            amount=Decimal("-50.00"),
            currency="XYZ",           # not in Currency enum
            sender=Address(street="A", city="B", postal_code="12345", country_code="US"),
            recipient=Address(street="A", city="B", postal_code="12345", country_code="US"),
        )
    except ValidationError as exc:
        log.info("Caught %d validation error(s):", exc.error_count())
        for err in exc.errors():
            log.info("  loc=%s | type=%s | msg=%s", err["loc"], err["type"], err["msg"])


def demo_discriminated_union() -> None:
    log.info("=== Discriminated union ===")

    payloads: list[dict] = [
        {"type": "text", "content": "Hello from Alice"},
        {"type": "file", "filename": "report.pdf", "size_bytes": 102400, "mime_type": "application/pdf"},
        {"type": "error", "code": 422, "detail": "Unprocessable Entity"},
    ]

    for raw in payloads:
        envelope = MessageEnvelope(sender_id="user-1", payload=raw)
        log.info("Envelope: payload_type=%s | %s", type(envelope.payload).__name__, envelope.payload)


def demo_settings() -> None:
    log.info("=== Settings ===")

    # Override via environment for demo
    os.environ["APP_DEBUG"] = "true"
    os.environ["APP_MAX_CONNECTIONS"] = "25"

    settings = AppSettings()
    log.info("debug=%s | max_connections=%d", settings.debug, settings.max_connections)
    log.info("database_url=%s", settings.database_url)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_transaction()
    demo_validation_errors()
    demo_discriminated_union()
    demo_settings()
```

Install dependencies:
```bash
pip install "pydantic>=2.0" "pydantic-settings>=2.0"
```

Run:
```bash
python day7_pydantic.py
```

## Exercises

1. Extend `Address` with a `coordinates` optional field of type `tuple[float, float]` representing `(latitude, longitude)`. Add a `@field_validator` that checks latitude is in `[-90, 90]` and longitude is in `[-180, 180]`.

2. Add a `history` field to `Transaction` that holds a list of `StatusChange` models, each with `status: TransactionStatus`, `changed_at: datetime`, and `changed_by: str`. Write a `@model_validator(mode="after")` that ensures the list is ordered by `changed_at` ascending.

3. Create a `BulkTransactionRequest` model that holds a list of `Transaction` objects with a minimum of 1 and maximum of 100 items. Write a `@model_validator` that rejects requests where the total `amount` across all transactions exceeds 1,000,000.

4. Write a `retry_with_backoff` function that takes a `Callable[[], T]` and a `max_retries: int`. Each time the callable raises `ValidationError`, log the errors and retry (up to max). Return the result on success or re-raise after exhaustion.

5. Create a `BaseEventModel` with `event_id: str`, `event_type: str`, `timestamp: datetime`. Create two subclasses: `UserCreatedEvent` and `OrderPlacedEvent`, each adding their own fields. Build a discriminated union `AppEvent` on `event_type` using `Literal` types. Parse a list of raw dicts into `list[AppEvent]`.

6. Add a `custom_serialise` method to `Transaction` that calls `model_dump(mode="json")` and replaces the `Decimal` amount with a string formatted to 2 decimal places. Verify the output is JSON-serialisable with `json.dumps`.

(Solutions: ../solutions/week-04/day-7.md)

---

## Week 4 Assessment Review

This section consolidates the key concepts tested in `assessment.md`. Review each checkpoint before attempting the assessment.

### Memory Management (Day 4)

**Core model**: every value is an object on the heap; variables are name bindings (references), not storage boxes. Assignment never copies a value — it creates a new reference to the same object.

**Reference counting**:
- `sys.getrefcount(obj)` returns the count +1 (extra ref from the argument).
- The count increments on assignment, function call, container append.
- The count decrements on `del`, scope exit, container removal.
- When the count reaches 0, the object is immediately deallocated.

**Cyclic GC**:
- Cycles prevent refcount from reaching 0; the `gc` module handles them.
- Three generations (0/1/2); gen-0 collected most frequently.
- `gc.collect()` runs a collection manually; returns the number of unreachable objects collected.
- `gc.disable()` / `gc.enable()` toggle automatic collection.

**Identity vs equality**: `is` checks `id()` equality (same object); `==` checks value equality (`__eq__`). Use `is` only for `None`, `True`, `False`.

**Interning**: small ints `[-5, 256]` and many string literals are pre-allocated. `sys.intern(s)` forces string interning.

**tracemalloc**: `tracemalloc.start()`, `take_snapshot()`, `compare_to()` — find allocation hot-spots.

**weakref**: `weakref.ref(obj)` does not increment refcount; use for caches that should not keep objects alive.

---

### Modules & Imports (Day 5)

**Import pipeline**: `sys.modules` cache check → meta-path finders → loader → `exec_module` → store in `sys.modules`.

**sys.path**: list of directories searched by `PathFinder`. Script directory is prepended at runtime.

**`__init__.py`**: marks a directory as a regular package; runs on first import; use for re-exports and `__all__`.

**`__all__`**: controls what `from package import *` exports; also signals intent to tools.

**src/ layout**: places source under `src/` to prevent accidental direct imports; requires `pip install -e .` or `pythonpath = ["src"]` in `pyproject.toml`.

**Circular import strategies**: extract shared code into a third module; defer import to function body; use `TYPE_CHECKING` guard for type-only imports.

**`importlib.import_module("name")`**: dynamic import by string name.

**`pyproject.toml`**: `[project]`, `[build-system]`, `[tool.setuptools.packages.find]`, `[tool.ruff]`, `[tool.pytest.ini_options]`.

---

### Type Hints & Generics (Day 6)

**Syntax**: `list[int]` (3.9+), `int | None` (3.10+), `type Alias = ...` (3.12).

**Key typing constructs**:
- `TypeVar("T")`: generic placeholder; `bound=` restricts to a class hierarchy; constraints restrict to specific types.
- `Generic[T]`: base for generic classes.
- `Protocol`: structural subtyping — a class satisfies a Protocol without inheriting it.
- `TypedDict`: dict with typed string keys; `total=False` makes all keys optional; `Required`/`NotRequired` (3.11+) for per-key control.
- `Literal["a", "b"]`: restricts to specific values.
- `Final`: marks a variable as constant.
- `Annotated[T, metadata]`: attaches metadata; used by Pydantic and FastAPI.
- `ParamSpec`: captures a callable's parameter signature; required for correctly typed decorators.
- `TypeGuard[T]`: custom type narrowing function.

**Type narrowing**: `isinstance` guards, `assert`, `TypeGuard` functions. After a narrowing check, the type analyser knows the type is more specific inside that branch.

**Runtime vs static**: type hints do not enforce anything at runtime (except when libraries like Pydantic explicitly call `get_type_hints`). Static checking is done by `mypy`/`pyright`/`Pylance`.

---

### Pydantic v2 (Day 7)

**BaseModel**: field declarations with annotations and `Field()`. Use `model_config = ConfigDict(...)` for model-level settings.

**Field constraints**: `gt`, `ge`, `lt`, `le`, `min_length`, `max_length`, `pattern`, `description`, `alias`, `default_factory`.

**Validators**:
- `@field_validator("name", mode="before"|"after")`: must be a `@classmethod`; `before` receives raw input; `after` receives typed value.
- `@model_validator(mode="before"|"after")`: `after` receives `self` (the fully constructed model).

**Serialisation**: `model_dump()` → dict; `model_dump_json()` → JSON string; `model_validate(d)` → model; `model_validate_json(s)` → model.

**Nested models**: pass nested dicts; Pydantic recursively validates and constructs.

**Discriminated unions**: `Annotated[Union[A, B], Field(discriminator="type")]` — requires `type: Literal["a"]` / `type: Literal["b"]` fields; enables faster dispatch and clearer errors.

**ValidationError**: `exc.errors()` returns list of error dicts with `loc`, `msg`, `type`, `input`. `exc.error_count()` returns the count.

**pydantic-settings**: `BaseSettings` with `SettingsConfigDict(env_file=".env", env_prefix="APP_")` reads config from environment and `.env` files; validates with full Pydantic power.

---

### Assessment strategy

- Read each question fully before answering; many questions ask for specific output or error types.
- For memory questions: trace the refcount step by step.
- For import questions: trace through the `sys.modules` cache and finder pipeline.
- For type hint questions: check whether the question is about runtime behaviour (no enforcement) or static analysis.
- For Pydantic questions: remember v2 uses `model_dump()` not `.dict()`, and validators must be `@classmethod`.
- Write small test snippets in your REPL to verify your mental model before committing to an answer.

## Self-check

- [ ] I can define a `BaseModel` with `Field()` constraints and explain what happens at construction.
- [ ] I can write a `@field_validator(mode="before")` that coerces string input to a numeric type.
- [ ] I can write a `@model_validator(mode="after")` that validates a cross-field business rule.
- [ ] I can serialise a model to JSON and reconstruct it with `model_validate_json`.
- [ ] I can create a `BaseSettings` class that reads from environment variables with a prefix.
- [ ] I can build a discriminated union and explain how the `discriminator` field affects parsing.
- [ ] I can iterate `ValidationError.errors()` and extract `loc`, `msg`, `type` from each entry.
- [ ] I can summarise the v1 → v2 API changes for validators, config, and serialisation.

## Supplementary links

- https://docs.pydantic.dev/latest/ — Pydantic v2 official documentation
- https://docs.pydantic.dev/latest/concepts/validators/ — validators guide
- https://docs.pydantic.dev/latest/concepts/serialization/ — serialisation guide
- https://docs.pydantic.dev/latest/concepts/unions/ — discriminated unions
- https://docs.pydantic.dev/latest/migration/ — v1 to v2 migration guide
- https://docs.pydantic.dev/latest/concepts/pydantic_settings/ — pydantic-settings
- https://github.com/pydantic/pydantic — source and issue tracker
