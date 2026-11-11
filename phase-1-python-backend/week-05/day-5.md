# Phase 1 · Week 05 · Day 5 — Clean Code: Separation of Concerns, Dependency Injection, ruff & Config

> Objective: Apply separation of concerns, basic dependency injection, automated linting with ruff, config management with pydantic-settings, and custom exceptions.  ·  Time: ~1 hr  ·  Prereqs: [[week-05/day-4]] (logging), [[week-02/day-3]] (classes)

## Concept

### Separation of concerns (SoC)

Every module/class/function should have **one reason to change**. When a class does business logic, I/O, and formatting simultaneously, a change to the API format forces you to touch business logic — fragile and hard to test.

Typical layers in a backend service:
```
Controller / API handler   ← validates HTTP input, calls service
Service / use-case layer   ← business rules, orchestrates operations
Repository / data layer    ← persistence (DB, file, external API)
Domain model               ← pure data + invariants (Pydantic, dataclasses)
```

Each layer depends only on the layer below. The service never knows about HTTP. The repo never knows about business rules.

### Dependency injection (DI)

Instead of a class constructing its own dependencies, **receive them via the constructor**. This makes the dependency visible and replaceable:

```python
# Tightly coupled — hard to test, hard to swap
class LedgerService:
    def __init__(self):
        self.repo = SqliteRepo("ledger.db")   # hardcoded dependency

# Loosely coupled — inject any compatible repo
class LedgerService:
    def __init__(self, repo: LedgerRepository) -> None:
        self.repo = repo
```

In tests you inject a mock; in production you inject the real implementation. No framework required for basic DI — just constructor parameters.

### Custom exceptions

Generic `Exception` or `ValueError` gives callers no way to distinguish error types. Custom exceptions form a contract:

```python
# exceptions.py — central exceptions module
class AppError(Exception):
    """Base for all application errors."""

class InsufficientFundsError(AppError):
    """Withdrawal exceeds available balance."""

class LedgerNotFoundError(AppError):
    """No ledger exists for the requested owner."""

class LedgerFrozenError(AppError):
    """Operation attempted on a frozen ledger."""
```

Callers can catch `AppError` for any application error, or specific subclasses for targeted handling. Never use bare `except Exception` — always catch the specific type you expect.

### ruff — fast Python linter + formatter

ruff replaces flake8 + isort + many plugins in a single Rust binary. Run it on save and in CI.

```bash
pip install ruff
ruff check src/ tests/       # lint
ruff check --fix src/ tests/ # auto-fix safe issues
ruff format src/ tests/      # format (like black)
```

Configure in `pyproject.toml`:
```toml
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
# E=pycodestyle, F=pyflakes, I=isort, UP=pyupgrade, B=bugbear, SIM=simplify
ignore = ["E501"]  # line too long — handled by formatter
```

### Config management with pydantic-settings

Never hardcode config or read `os.getenv` scattered across modules. Centralise in a settings class:

```python
# src/config.py
from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str
    log_level: str = "INFO"
    max_transfer_limit: int = 10_000
    app_api_key: str = ""   # populated from APP_API_KEY env var


# Singleton — import this everywhere
settings = Settings()
```

pydantic-settings reads from (in order): environment variables → `.env` file → field defaults. Type validation is automatic — `max_transfer_limit: int` raises a startup error if the env var is not a valid integer.

`.env.example` (committed, no real values):
```
DATABASE_URL=
LOG_LEVEL=INFO
MAX_TRANSFER_LIMIT=10000
APP_API_KEY=
```

`.env` (never committed — add to `.gitignore`):
```
DATABASE_URL=sqlite:///local.db
LOG_LEVEL=DEBUG
APP_API_KEY=
```

## How it works internally

### ruff's speed

ruff is written in Rust and parses Python to AST in parallel across files. It applies rules as visitor passes over the AST — no subprocess spawning per file. On a 100k-line codebase ruff finishes in under a second where flake8 takes 30s.

### pydantic-settings field resolution

`BaseSettings` uses a custom `__init__` that, for each field, tries (in priority order): init kwargs → environment variables → secrets directory → dotenv file → field default. The env var name defaults to the field name uppercased; `model_config` lets you set a prefix.

## Worked example

```python
# src/exceptions.py
"""Central exception hierarchy — all app exceptions derive from AppError."""
from __future__ import annotations


class AppError(Exception):
    """Base class for all application exceptions."""


class InsufficientFundsError(AppError):
    """Withdrawal amount exceeds available balance."""


class LedgerNotFoundError(AppError):
    """No ledger found for the given owner identifier."""


class LedgerFrozenError(AppError):
    """Operation attempted on a frozen (locked) ledger."""


class InvalidAmountError(AppError):
    """Amount is zero, negative, or otherwise invalid."""
```

```python
# src/config.py
"""Application configuration — single source of truth for all settings."""
from __future__ import annotations
import logging
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Required — must be set in env or .env
    database_url: str = "sqlite:///ledger.db"

    # Optional with sensible defaults
    log_level: str = "INFO"
    max_transfer_limit: int = 10_000
    app_environment: str = "development"

    # Never log this value
    app_api_key: str = ""

    def configure_logging(self) -> None:
        level = getattr(logging, self.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


settings = Settings()
```

```python
# src/repository.py
"""Repository interface + in-memory implementation."""
from __future__ import annotations
from abc import ABC, abstractmethod
from decimal import Decimal
from src.ledger import Ledger
from src.exceptions import LedgerNotFoundError


class LedgerRepository(ABC):
    """Abstract base — depends on nothing concrete."""

    @abstractmethod
    def save(self, ledger: Ledger) -> None: ...

    @abstractmethod
    def get(self, owner: str) -> Ledger: ...


class InMemoryLedgerRepository(LedgerRepository):
    """In-process store — suitable for tests and dev."""

    def __init__(self) -> None:
        self._store: dict[str, Ledger] = {}

    def save(self, ledger: Ledger) -> None:
        self._store[ledger.owner] = ledger

    def get(self, owner: str) -> Ledger:
        try:
            return self._store[owner]
        except KeyError:
            raise LedgerNotFoundError(f"no ledger found for owner={owner!r}")
```

```python
# src/service.py
"""LedgerService — business logic, no I/O concerns."""
from __future__ import annotations
import logging
from decimal import Decimal
from src.repository import LedgerRepository
from src.ledger import Ledger
from src.exceptions import InvalidAmountError
from src.config import settings

log = logging.getLogger(__name__)


class LedgerService:
    """Orchestrates ledger operations. Injected with a repository."""

    def __init__(self, repo: LedgerRepository) -> None:
        self._repo = repo

    def create_ledger(self, owner: str, initial: Decimal = Decimal("0")) -> Ledger:
        ledger = Ledger(owner, initial)
        self._repo.save(ledger)
        log.info("ledger created: owner=%s", owner)
        return ledger

    def transfer(self, from_owner: str, to_owner: str, amount: Decimal) -> None:
        if amount > settings.max_transfer_limit:
            raise InvalidAmountError(
                f"amount {amount} exceeds limit {settings.max_transfer_limit}"
            )
        source = self._repo.get(from_owner)
        dest = self._repo.get(to_owner)
        source.transfer_to(dest, amount)
        self._repo.save(source)
        self._repo.save(dest)
        log.info("transfer: from=%s to=%s amount=%s", from_owner, to_owner, amount)
```

```python
# tests/test_service.py
"""Unit tests for LedgerService using injected InMemoryLedgerRepository."""
from __future__ import annotations
import pytest
from decimal import Decimal
from src.service import LedgerService
from src.repository import InMemoryLedgerRepository
from src.exceptions import LedgerNotFoundError, InvalidAmountError


@pytest.fixture
def service() -> LedgerService:
    return LedgerService(repo=InMemoryLedgerRepository())


def test_create_and_retrieve(service):
    ledger = service.create_ledger("alice", Decimal("200"))
    assert ledger.balance == Decimal("200")


def test_transfer_between_accounts(service):
    service.create_ledger("alice", Decimal("500"))
    service.create_ledger("bob", Decimal("0"))
    service.transfer("alice", "bob", Decimal("150"))
    assert service._repo.get("alice").balance == Decimal("350")
    assert service._repo.get("bob").balance == Decimal("150")


def test_transfer_unknown_owner_raises(service):
    service.create_ledger("alice", Decimal("100"))
    with pytest.raises(LedgerNotFoundError):
        service.transfer("alice", "nobody", Decimal("10"))


def test_transfer_exceeds_limit_raises(service):
    service.create_ledger("alice", Decimal("100_000"))
    service.create_ledger("bob", Decimal("0"))
    with pytest.raises(InvalidAmountError, match="exceeds limit"):
        service.transfer("alice", "bob", Decimal("20_000"))
```

## Exercises

1. Add a `SqliteLedgerRepository` stub class that inherits from `LedgerRepository` and raises `NotImplementedError` on both methods. Write a test that confirms it raises `NotImplementedError`.

2. Run `ruff check src/ tests/` on the project. Fix every reported issue. Then add `ruff format src/ tests/` to a `Makefile` (or `pyproject.toml` scripts section) so a single command runs both lint and format.

3. Add a `transfer_limit` to `Settings` that reads from env var `TRANSFER_LIMIT`. Write a test that creates a `LedgerService` with a mocked settings object (or monkeypatches `settings.max_transfer_limit`) to verify the limit check uses the configured value.

4. Add a `LedgerFrozenError` to the exception hierarchy. Modify `Ledger` to support a `freeze()` method that sets `self._frozen = True`. Raise `LedgerFrozenError` on any subsequent deposit or withdrawal. Write tests.

5. Explain: what is the difference between `select` and `ignore` in ruff config? What does the `B` (bugbear) ruleset catch that `E` and `F` do not?

(Solutions: ../solutions/week-05/day-5.md)

## Self-check
- [ ] I can describe the four-layer architecture (controller, service, repo, domain).
- [ ] I understand DI and can inject a fake repo in tests without mocking.
- [ ] I have a single `exceptions.py` with a custom hierarchy derived from `AppError`.
- [ ] I can run `ruff check --fix` and `ruff format` and fix reported issues.
- [ ] I have a `Settings` class using pydantic-settings and `.env` is git-ignored.
- [ ] I never call `os.getenv` outside the `Settings` class.

## Supplementary links
- [ruff documentation](https://docs.astral.sh/ruff/)
- [pydantic-settings docs](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [Real Python — SOLID principles in Python](https://realpython.com/solid-principles-python/)
- [12-Factor App — Config](https://12factor.net/config)
