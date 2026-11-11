# Phase 1 · Week 05 · Day 6 — Saturday Lab: Add Tests, Logging & CI to the Transaction Library

> Objective: Retrofit the Week 4 transaction library with a complete test suite, structured logging, custom exceptions, env-based config, and a GitHub Actions CI pipeline.  ·  Time: ~4 hr  ·  Prereqs: [[week-04/project]], [[week-05/day-1..5]]

## Overview

This lab transforms the Week 4 transaction library from a working script into a production-quality package. You will add quality infrastructure on top of existing code rather than rewriting it — a real-world skill.

Deliverables by end of session:
- `tests/unit/` — unit tests for every public function/method.
- `tests/unit/test_invalid_inputs.py` — dedicated invalid-input tests.
- `tests/unit/test_mocked.py` — tests that mock all external calls.
- `src/exceptions.py` — custom exception hierarchy.
- `src/config.py` — pydantic-settings `Settings` class.
- Structured logging in all modules (no bare `print`).
- `.github/workflows/ci.yml` — GitHub Actions pipeline that runs tests + ruff.
- Coverage at ≥ 80% (checked with `pytest --cov`).

## Part 1 — Audit the existing code (30 min)

### Step 1.1 — Map the public API

List every public function and class method in your Week 4 transaction library. For each, answer:
- What are valid inputs? What are invalid inputs?
- Does it make any external call (file I/O, HTTP, DB, random)?
- Does it raise? What type?

Create a table in a scratch file:

```
| Function | Valid inputs | Invalid inputs | External call? | Raises? |
```

### Step 1.2 — Identify missing structure

Check whether the library already has:
- [ ] `src/exceptions.py` with a custom hierarchy
- [ ] `src/config.py` using pydantic-settings
- [ ] Any `logging` calls (vs `print`)
- [ ] `tests/` directory

Note gaps — you will fill them in Parts 2–5.

## Part 2 — Custom exceptions & config (45 min)

### Step 2.1 — exceptions.py

Create `src/exceptions.py`:

```python
"""Application exception hierarchy.

All app-level exceptions inherit from AppError so callers can
catch `AppError` to handle any application-level failure.
"""
from __future__ import annotations


class AppError(Exception):
    """Base for all transaction library errors."""


class InvalidTransactionError(AppError):
    """Transaction parameters violate business rules."""


class AccountNotFoundError(AppError):
    """No account exists for the requested identifier."""


class InsufficientFundsError(AppError):
    """Transaction would result in a negative balance."""


class DuplicateTransactionError(AppError):
    """A transaction with this ID has already been processed."""
```

Update every `raise ValueError(...)` / `raise Exception(...)` in your existing code to raise the appropriate custom exception.

### Step 2.2 — config.py

```python
# src/config.py
"""Centralised configuration — reads from env / .env file."""
from __future__ import annotations
import logging
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Application behaviour
    max_transaction_amount: int = 50_000
    default_currency: str = "GBP"
    transaction_fee_pct: float = 0.0

    # Observability
    log_level: str = "INFO"

    # External service — read from APP_API_KEY env var; never log this
    app_api_key: str = ""

    def configure_logging(self) -> None:
        level = getattr(logging, self.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)-8s %(name)-30s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


settings = Settings()
```

Create `.env.example`:
```
MAX_TRANSACTION_AMOUNT=50000
DEFAULT_CURRENCY=GBP
TRANSACTION_FEE_PCT=0.0
LOG_LEVEL=INFO
APP_API_KEY=
```

Ensure `.env` is in `.gitignore`.

## Part 3 — Structured logging (30 min)

Replace every `print(...)` in the transaction library with appropriate log calls.

Rules:
- Module-level: `log = logging.getLogger(__name__)`
- State transitions: `log.info`
- Invalid input caught and rejected: `log.warning`
- Unexpected failures: `log.error(..., exc_info=True)`
- Diagnostic detail: `log.debug`

Example before/after:
```python
# Before
print(f"Processing transaction {txn_id}")

# After
log.info("processing transaction: id=%s amount=%s", txn_id, amount)
```

Call `settings.configure_logging()` once in `main.py` before any other code runs.

## Part 4 — Test suite (90 min)

### Step 4.1 — Unit tests for happy path

Create `tests/unit/test_transactions.py`. For every public function:
- At least one test covering normal successful operation.
- Assert on the return value **and** any side effects (balance changes, state changes).

```python
# tests/unit/test_transactions.py
"""Happy-path unit tests for the transaction library."""
from __future__ import annotations
import pytest
# Import your Week 4 classes/functions here
```

### Step 4.2 — Invalid input tests

Create `tests/unit/test_invalid_inputs.py`:

```python
"""Tests that every invalid input raises the correct custom exception."""
from __future__ import annotations
import pytest
from src.exceptions import (
    InvalidTransactionError,
    InsufficientFundsError,
    AccountNotFoundError,
)

# Test each of these scenarios:
# - zero amount
# - negative amount
# - None as owner/ID
# - withdrawal exceeding balance
# - duplicate transaction ID (if your library tracks IDs)
# - amount over max_transaction_amount config limit
```

Each test should:
1. Assert the correct exception *type* is raised.
2. Use `match=` to verify the message contains a meaningful phrase.

### Step 4.3 — Mocked external dependencies

Create `tests/unit/test_mocked.py`. Identify every call to:
- File read/write
- HTTP requests
- Random number generation
- `datetime.now()` / `time.time()`

Mock each with `unittest.mock.patch` or `pytest-mock`'s `mocker` fixture. Verify both the mock is called with the right args and the function returns the expected value.

```python
# tests/unit/test_mocked.py
from __future__ import annotations
from unittest.mock import patch, MagicMock
import pytest

def test_notification_sent_on_successful_transfer(funded_accounts):
    with patch("src.transactions.send_notification") as mock_notify:
        # ... perform transfer
        mock_notify.assert_called_once()


def test_external_rate_api_called_with_correct_currency():
    with patch("src.transactions.fetch_exchange_rate") as mock_rate:
        mock_rate.return_value = 1.18
        # ... perform foreign currency transaction
        mock_rate.assert_called_with("GBP", "EUR")
```

### Step 4.4 — conftest.py

```python
# tests/conftest.py
"""Shared fixtures for the transaction library test suite."""
from __future__ import annotations
import pytest
from decimal import Decimal
# Import your domain classes here


@pytest.fixture
def empty_account():
    """Account with zero balance."""
    # return your Account / Ledger class here
    ...


@pytest.fixture
def funded_account():
    """Account seeded with £1000."""
    ...


@pytest.fixture
def funded_accounts():
    """Pair (alice, bob) with alice=£500, bob=£0."""
    ...
```

## Part 5 — GitHub Actions CI pipeline (30 min)

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: ["main", "feature/**"]
  pull_request:
    branches: ["main"]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Lint with ruff
        run: |
          ruff check src/ tests/
          ruff format --check src/ tests/

      - name: Run tests with coverage
        run: pytest tests/unit/ --cov=src --cov-report=term-missing --cov-fail-under=80

      - name: Upload coverage report
        uses: actions/upload-artifact@v4
        with:
          name: coverage-report
          path: .coverage
```

Add dev dependencies to `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.4",
    "pydantic-settings>=2.0",
]
```

## Part 6 — Coverage check (15 min)

Run locally:
```bash
pytest tests/unit/ --cov=src --cov-report=term-missing
```

Read the output:
```
Name                    Stmts   Miss  Cover
-------------------------------------------
src/exceptions.py          10      0   100%
src/transactions.py        87     14    84%
src/config.py              18      2    89%
-------------------------------------------
TOTAL                     115     16    86%
```

For any file below 80%, look at the "Miss" lines and write tests for the uncovered branches.

## Exercises

These exercises push beyond the baseline — attempt all:

1. Add a `@pytest.mark.integration` test that writes transactions to a real SQLite database (use `sqlite3` from stdlib), reads them back, and asserts they match. Keep it separate from unit tests.

2. Add a `--cov-fail-under=80` flag to the CI config and deliberately lower coverage below 80% (delete a test). Verify the pipeline fails. Restore.

3. Write a `conftest.py` session-scoped fixture that checks `settings.app_api_key` is empty (or a known placeholder) and skips the entire session with `pytest.skip` if a real key is accidentally loaded. This prevents accidental API calls in CI.

4. Configure ruff's `B` (bugbear) ruleset and fix any issues it finds in your codebase.

5. Add `pytest-xdist` to dev dependencies and run `pytest -n auto` — parallel test execution. Verify all tests still pass. If any fail, diagnose why (hint: shared mutable state between tests).

(Solutions: ../solutions/week-05/day-6.md)

## Self-check
- [ ] Every public function has at least one unit test.
- [ ] Every invalid input has a test asserting the correct custom exception.
- [ ] All external calls are mocked in unit tests.
- [ ] `logging` replaces all `print` calls.
- [ ] `.env` is git-ignored; `.env.example` is committed.
- [ ] CI pipeline passes on push to main.
- [ ] Coverage ≥ 80%.
