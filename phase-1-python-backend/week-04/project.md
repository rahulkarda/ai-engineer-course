# Week 04 Project — Reusable Transaction Library

## Overview

Build a production-quality, installable Python package that processes financial transaction records. The library is a direct application of every topic from Week 4: Pydantic v2 models, generator-based pipelines, decorators, context managers, a custom exception hierarchy, and a settings layer via `pydantic-settings`.

**Learning goals:**
- Structure a real package with the `src/` layout and `pyproject.toml`.
- Apply Pydantic v2 models, validators, and settings idiomatically.
- Build a lazy, memory-efficient processing pipeline using generators.
- Use decorators and context managers as library-level reusable constructs.
- Write a complete exception hierarchy that communicates domain errors clearly.
- Achieve full pytest coverage of all public API surfaces.

---

## Package Structure

```
transaction_lib/
├── src/
│   └── transaction_lib/
│       ├── __init__.py
│       ├── models.py         # Pydantic models
│       ├── processor.py      # processing logic with generators
│       ├── validators.py     # custom validation functions
│       ├── exceptions.py     # custom exception hierarchy
│       └── config.py         # pydantic-settings config
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

## Module Requirements

### `models.py`

Define all domain models using **Pydantic v2** (`from pydantic import BaseModel, field_validator, model_validator, ConfigDict`).

**`TransactionStatus` (enum):**
- Values: `PENDING`, `APPROVED`, `REJECTED`.
- Use `str` mixin so serialisation yields the string value.

**`Transaction` (BaseModel):**
- `id: str` — non-empty UUID-format string.
- `amount: Decimal` — must be positive; max 2 decimal places.
- `currency: str` — exactly 2 uppercase alpha characters (ISO 4217 code).
- `timestamp: datetime` — timezone-aware; reject naive datetimes.
- `merchant: str` — 1–100 characters, stripped of leading/trailing whitespace.
- `status: TransactionStatus`.
- At least one `@field_validator` (e.g., normalise merchant name, validate currency format).
- At least one `@model_validator(mode="after")` (e.g., cross-field rule: rejected transactions must not have `amount > 0` without a reason field).
- `model_config = ConfigDict(frozen=True, str_strip_whitespace=True)`.

**`Address` (BaseModel):**
- `street: str`, `city: str`, `country_code: str` (2-char ISO), `postal_code: str`.
- Used as an optional nested field on an extended `MerchantTransaction` variant.

**`TransactionBatch` (BaseModel):**
- `batch_id: str`, `created_at: datetime`, `transactions: list[Transaction]`.
- A `@model_validator(mode="after")` that checks the batch is non-empty.
- A computed property `total_amount -> Decimal` that sums all transaction amounts.

**`ValidationReport` (TypedDict):**
- `valid_count: int`, `error_count: int`, `errors: list[dict[str, Any]]`.

---

### `processor.py`

Implement a `TransactionProcessor` class that orchestrates the pipeline using **generators** and **context managers**.

**Generator functions (module-level, not methods):**
- `filter_by_status(transactions, status: TransactionStatus) -> Generator[Transaction, None, None]`
- `filter_by_amount_range(transactions, min_amount: Decimal, max_amount: Decimal) -> Generator[Transaction, None, None]`
- `filter_by_currency(transactions, currency: str) -> Generator[Transaction, None, None]`
- `transform_merchant_name(transactions, fn: Callable[[str], str]) -> Generator[Transaction, None, None]` — yields new `Transaction` instances (frozen model, so use `.model_copy(update=...)`).
- Each function must accept any iterable of `Transaction`, enabling chaining without materialising lists.

**`TransactionProcessor` class:**
- `__init__(self, settings: Settings)` — inject config.
- `validate_batch(raw_records: list[dict]) -> tuple[list[Transaction], ValidationReport]` — parse each dict with `Transaction.model_validate(...)`, collect `ValidationError` exceptions into the report.
- `process(transactions: Iterable[Transaction], *filters) -> Generator[Transaction, None, None]` — apply a variadic chain of filter/transform generators lazily.
- `export_json(transactions: Iterable[Transaction], path: Path) -> int` — write JSON array, return count of records written. Must use a context manager for file I/O; log success and failure via `logging`.

**Decorators (module-level):**
- `@timed` — wraps any callable, logs elapsed wall-clock time to `logging.getLogger(__name__)`. Preserves `__name__`, `__doc__`, `__wrapped__` via `functools.wraps`.
- `@retry(max_attempts: int, exceptions: tuple[type[Exception], ...])` — parameterised decorator that retries on specified exceptions, raising the last exception after `max_attempts`.

**Context manager:**
- `batch_session(processor: TransactionProcessor, batch_id: str)` — a `@contextmanager` that logs batch start/end, yields the processor, and ensures any `ProcessingError` is caught and re-raised with the `batch_id` in the message.

---

### `validators.py`

Standalone validation functions (pure functions, no Pydantic dependency) reused by models and external callers.

- `validate_amount(amount: Decimal) -> Decimal` — raises `ValidationError` if amount is not positive or has more than 2 decimal places.
- `validate_currency_code(code: str) -> str` — raises `ValidationError` if not exactly 2 uppercase alpha characters.
- `validate_merchant_name(name: str) -> str` — raises `ValidationError` if empty or longer than 100 chars after stripping.
- `validate_iso_date(value: str) -> datetime` — parse ISO 8601 string; raise `ValidationError` if unparseable or naive.
- Each function must have full type annotations and a docstring.

---

### `exceptions.py`

Define a custom exception hierarchy rooted at `TransactionError`.

```
TransactionError(Exception)
├── ValidationError(TransactionError)
│   ├── AmountValidationError(ValidationError)
│   ├── CurrencyValidationError(ValidationError)
│   └── MerchantValidationError(ValidationError)
├── ProcessingError(TransactionError)
│   ├── BatchProcessingError(ProcessingError)
│   └── ExportError(ProcessingError)
└── ConfigurationError(TransactionError)
```

Requirements:
- Every exception class stores a `message: str` and an optional `context: dict` for structured error data.
- `__str__` on `TransactionError` must include both `message` and `context` when context is non-empty.
- `BatchProcessingError` must accept a `batch_id: str` parameter and include it in its string representation.
- `ExportError` must accept a `path: Path` parameter.

---

### `config.py`

Use **`pydantic-settings`** (`from pydantic_settings import BaseSettings`).

**`Settings` (BaseSettings):**
- `max_amount: Decimal = Decimal("1_000_000")`
- `min_amount: Decimal = Decimal("0.01")`
- `supported_currencies: list[str] = ["USD", "EUR", "GBP", "JPY"]`
- `max_batch_size: int = 10_000`
- `export_dir: Path = Path("./exports")`
- `log_level: str = "INFO"`
- `retry_max_attempts: int = 3`
- `model_config = SettingsConfigDict(env_prefix="TXN_", env_file=".env", env_file_encoding="utf-8")`

All fields must be overridable via environment variables prefixed `TXN_` (e.g., `TXN_MAX_AMOUNT=5000`).

---

## Milestone Breakdown (4 hours)

### Milestone 1 — Scaffold & models (60 min)
- Create the full directory structure.
- Write `pyproject.toml` (see template below).
- Implement `exceptions.py` (no external deps — start here).
- Implement `models.py` with all Pydantic models and pass `test_models.py`.

### Milestone 2 — Validators & config (30 min)
- Implement `validators.py` (pure functions).
- Implement `config.py` with `BaseSettings`.
- Pass `test_validators.py`.

### Milestone 3 — Processor & pipeline (75 min)
- Implement all generator functions in `processor.py`.
- Implement `TransactionProcessor.validate_batch` and `process`.
- Implement `@timed` and `@retry` decorators.
- Implement `batch_session` context manager.
- Pass `test_processor.py`.

### Milestone 4 — Export, integration & polish (15 min)
- Implement `export_json`.
- Wire `__init__.py` public API.
- Pass `test_exceptions.py`.
- Run full test suite; fix any gaps.

---

## Testing Requirements

### `test_models.py`
- Valid `Transaction` round-trips through `model_validate` and `model_dump`.
- `field_validator` rejects invalid currency codes, empty merchants, negative amounts.
- `model_validator` enforces cross-field rules.
- `TransactionBatch.total_amount` returns correct `Decimal` sum.
- `TransactionBatch` rejects empty `transactions` list.
- Frozen model: verify that direct attribute assignment raises `ValidationError`.

### `test_processor.py`
- `filter_by_status` yields only matching records; exhausts gracefully on empty input.
- `filter_by_amount_range` excludes boundary violations.
- Generator chain (status + currency + amount) produces correct subset without materialising intermediate lists (verify with `itertools.islice` to stop early).
- `validate_batch` returns correct `ValidationReport` counts for mixed valid/invalid input.
- `@timed` decorator preserves `__name__` and `__doc__`; logs elapsed time (capture with `caplog`).
- `@retry` retries the specified number of times and re-raises on exhaustion.
- `batch_session` context manager logs start/end; wraps `ProcessingError` with `batch_id`.
- `export_json` writes valid JSON; file is closed even when serialisation raises.

### `test_validators.py`
- Each validator accepts its valid domain and raises `ValidationError` on violations.
- `validate_amount` rejects zero, negative, and more than 2 decimal places.
- `validate_currency_code` rejects lowercase, numeric, wrong length.
- `validate_iso_date` rejects naive datetimes and unparseable strings.

### `test_exceptions.py`
- `TransactionError` hierarchy: `isinstance` checks confirm subclass relationships.
- `__str__` includes `context` dict when non-empty.
- `BatchProcessingError.__str__` includes `batch_id`.
- `ExportError.__str__` includes the `path`.
- Catching `TransactionError` catches all subclasses.

---

## Acceptance Criteria

- [ ] Package installs cleanly with `pip install -e .` from the `transaction_lib/` directory.
- [ ] `from transaction_lib import Transaction, TransactionProcessor, Settings` works after install.
- [ ] All 4 test files pass with `pytest` (zero failures, zero errors).
- [ ] No `print` statements anywhere in `src/`; use `logging`.
- [ ] `ruff check src/` exits 0.
- [ ] Every public function and class has a type annotation and a one-line docstring.
- [ ] Generator functions never materialise a full list internally.
- [ ] `Settings` fields are overridable via env vars (verify with `monkeypatch.setenv` in tests).
- [ ] `export_json` always closes the file (verify with a mock that raises on write).

---

## `pyproject.toml` Template

```toml
[project]
name = "transaction-lib"
version = "0.1.0"
description = "Reusable transaction processing library"
requires-python = ">=3.11"
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
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/transaction_lib"]

[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "ANN", "N"]
ignore = ["ANN101", "ANN102"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--tb=short -q"
```

---

## Extension Challenges

1. **Async pipeline** — Rewrite `process()` as an `async def` using `async for` and `asyncio.Queue` to simulate streaming input.
2. **Typed generic processor** — Make `TransactionProcessor` generic over the model type (`T = TypeVar("T", bound=BaseModel)`) so it can process any Pydantic model, not just `Transaction`.
3. **CSV export** — Add `export_csv(path: Path)` using `csv.DictWriter` inside a context manager; add a corresponding test.
4. **Memory profiler** — Use `tracemalloc` to measure peak memory of processing 100,000 records with the generator pipeline vs a list-based equivalent; write a comparison script.
5. **Pluggable filters** — Replace the variadic `*filters` API with a `FilterChain` class that supports `__add__` to compose filters, and serialize the chain to/from JSON.
