# Solutions — Week 05 · Day 5 (Clean Code)

## Exercise 1 — SqliteLedgerRepository stub

```python
# src/repository.py addition
class SqliteLedgerRepository(LedgerRepository):
    """Sqlite-backed repository — not yet implemented."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    def save(self, ledger: Ledger) -> None:
        raise NotImplementedError("SqliteLedgerRepository.save not implemented")

    def get(self, owner: str) -> Ledger:
        raise NotImplementedError("SqliteLedgerRepository.get not implemented")
```

```python
# tests/test_repository.py
import pytest
from decimal import Decimal
from src.repository import SqliteLedgerRepository
from src.ledger import Ledger


def test_sqlite_save_raises_not_implemented():
    repo = SqliteLedgerRepository(":memory:")
    ledger = Ledger("test", Decimal("0"))
    with pytest.raises(NotImplementedError):
        repo.save(ledger)


def test_sqlite_get_raises_not_implemented():
    repo = SqliteLedgerRepository(":memory:")
    with pytest.raises(NotImplementedError):
        repo.get("test")
```

## Exercise 2 — ruff in Makefile / pyproject.toml

```makefile
# Makefile
.PHONY: lint format check

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

check: lint format
	pytest tests/ --cov=src
```

Or in `pyproject.toml` using `[tool.ruff]` and a script runner like `just` or `tox`:

```toml
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
```

## Exercise 3 — Test that uses monkeypatched settings

```python
# tests/test_service.py addition
import pytest
from decimal import Decimal
from src.service import LedgerService
from src.repository import InMemoryLedgerRepository
from src.exceptions import InvalidAmountError


def test_transfer_respects_configured_limit(monkeypatch):
    """Service must use settings.max_transfer_limit, not a hardcoded value."""
    import src.service as service_module
    monkeypatch.setattr(service_module.settings, "max_transfer_limit", 100)

    service = LedgerService(repo=InMemoryLedgerRepository())
    service.create_ledger("alice", Decimal("5000"))
    service.create_ledger("bob", Decimal("0"))

    with pytest.raises(InvalidAmountError, match="exceeds limit"):
        service.transfer("alice", "bob", Decimal("150"))  # over the patched limit of 100

    # Amount at the limit should succeed
    service.transfer("alice", "bob", Decimal("100"))
    assert service._repo.get("bob").balance == Decimal("100")
```

`monkeypatch.setattr` is a pytest built-in that restores the original value after the test, so it does not affect other tests.

## Exercise 4 — LedgerFrozenError

```python
# src/exceptions.py addition
class LedgerFrozenError(AppError):
    """Operation attempted on a frozen ledger."""
```

```python
# src/ledger.py changes
from src.exceptions import LedgerFrozenError, InsufficientFundsError, InvalidAmountError

class Ledger:
    def __init__(self, owner: str, initial_balance: Decimal = Decimal("0")) -> None:
        ...
        self._frozen: bool = False

    def freeze(self) -> None:
        self._frozen = True
        log.info("ledger frozen: owner=%s", self.owner)

    def _check_not_frozen(self) -> None:
        if self._frozen:
            raise LedgerFrozenError(f"ledger for {self.owner!r} is frozen")

    def deposit(self, amount: Decimal) -> None:
        self._check_not_frozen()
        ...

    def withdraw(self, amount: Decimal) -> None:
        self._check_not_frozen()
        ...
```

```python
# Tests
from src.exceptions import LedgerFrozenError

def test_freeze_blocks_deposit(funded_alice):
    funded_alice.freeze()
    with pytest.raises(LedgerFrozenError):
        funded_alice.deposit(Decimal("10"))


def test_freeze_blocks_withdrawal(funded_alice):
    funded_alice.freeze()
    with pytest.raises(LedgerFrozenError):
        funded_alice.withdraw(Decimal("10"))


def test_freeze_preserves_balance(funded_alice):
    original = funded_alice.balance
    funded_alice.freeze()
    assert funded_alice.balance == original


def test_not_frozen_by_default(alice):
    # Should not raise
    alice.deposit(Decimal("10"))
    assert alice.balance == Decimal("10")
```

## Exercise 5 — ruff select vs ignore, bugbear (written answer)

`select` is a whitelist: it enables only the specified rule codes. `ignore` is a blacklist within the selected set. You typically `select` a broad set and `ignore` specific rules that don't apply to your project.

The `B` (flake8-bugbear) ruleset catches common Python gotchas that are syntactically valid but logically wrong or dangerous:
- `B006` — mutable default arguments (`def f(x=[])`) — the list is shared across all calls.
- `B007` — loop variable unused in body.
- `B008` — function calls in default arguments evaluated at definition time.
- `B023` — function definition in a loop without capturing the loop variable.

`E` (pycodestyle) catches formatting/whitespace issues; `F` (pyflakes) catches undefined names and unused imports. Neither catches the semantic issues that `B` targets.
