# Phase 1 · Week 05 · Day 3 — Fixtures, Mocking & Integration Tests

> Objective: Use pytest fixtures for shared setup, mock external dependencies with `unittest.mock`, and write integration tests that test component boundaries.  ·  Time: ~1 hr  ·  Prereqs: [[week-05/day-2]] (pytest basics)

## Concept

### Fixtures

A **fixture** is a pytest-managed setup/teardown function. Any test that declares a fixture name as a parameter automatically receives its return value:

```python
@pytest.fixture
def empty_ledger():
    return Ledger("test_user")

def test_deposit(empty_ledger):          # pytest injects the fixture
    empty_ledger.deposit(Decimal("10"))
    assert empty_ledger.balance == Decimal("10")
```

Fixtures compose — a fixture can itself declare other fixtures as parameters.

### Fixture scope

| Scope | Created | Destroyed | Use when |
|-------|---------|-----------|----------|
| `function` (default) | Before each test | After each test | Mutable objects |
| `class` | Before first test in class | After last test in class | Shared but class-scoped state |
| `module` | Before first test in file | After last test in file | Expensive but immutable setup |
| `session` | Once per test run | At the end of the run | DB connections, temp dirs |

```python
@pytest.fixture(scope="module")
def db_connection():
    conn = create_test_db()
    yield conn          # yield separates setup from teardown
    conn.close()        # teardown: runs after all tests in module
```

Use `yield` when you need teardown — everything after `yield` is the cleanup block.

### conftest.py

Fixtures in `conftest.py` are automatically available to all tests in the same directory and below — no import needed. This is the right place for shared fixtures.

```
tests/
├── conftest.py          ← shared fixtures here
├── test_ledger.py
└── integration/
    ├── conftest.py      ← integration-specific fixtures
    └── test_api.py
```

### Mocking

A **mock** replaces a real dependency (HTTP call, database, file I/O) with a controllable fake during tests. `unittest.mock.MagicMock` and `patch` are the standard tools:

```python
from unittest.mock import MagicMock, patch

# Replace an attribute of an object
ledger = Ledger("alice")
ledger._notify = MagicMock()         # replace with mock
ledger.deposit(Decimal("10"))
ledger._notify.assert_called_once()  # verify it was called

# Patch a module-level name
with patch("src.notifications.send_email") as mock_send:
    process_transaction(...)
    mock_send.assert_called_with("alice@example.com", subject="Transaction")
```

`patch` restores the original after the `with` block exits — safe and leak-free.

### Integration tests

An **integration test** exercises multiple real components together — e.g., your service layer + a real (test) database, or two classes interacting. They are slower but catch problems that unit tests miss (incorrect wiring, SQL errors, serialisation mismatches).

Separate integration tests into their own directory and mark them:

```python
# tests/integration/test_ledger_repo.py
import pytest

@pytest.mark.integration
def test_persist_and_reload():
    ...
```

Run only unit tests: `pytest -m "not integration"`
Run all: `pytest`

Configure in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
markers = ["integration: marks tests as integration tests"]
```

## How it works internally

### Fixture dependency injection

pytest's fixture system is a dependency injection framework built on top of Python introspection. When pytest collects a test function, it inspects `inspect.signature()` to find parameter names, then looks up a fixture with that name in the fixture registry. Fixtures form a DAG; pytest resolves dependencies topologically, respecting scopes (broader scope = fewer instantiations).

### patch mechanics

`unittest.mock.patch` works by temporarily replacing an attribute on a module or class object using `setattr`. The target string `"mymodule.ClassName.method"` is resolved by splitting on `.` and walking through `sys.modules`. On context manager exit, it restores the original value with another `setattr`. This is why you must patch the name **where it is used**, not where it is defined:

```python
# ledger.py imports: from notifications import send_email
# We patch the name in ledger's namespace:
patch("src.ledger.send_email")   # correct
patch("src.notifications.send_email")  # wrong — ledger already has its own reference
```

## Worked example

```python
# tests/conftest.py
"""Shared fixtures for all Week 5 tests."""
from __future__ import annotations
import pytest
from decimal import Decimal
from src.ledger import Ledger


@pytest.fixture
def alice() -> Ledger:
    """Fresh ledger for Alice with no balance."""
    return Ledger("alice")


@pytest.fixture
def funded_alice() -> Ledger:
    """Alice with £500 starting balance."""
    ledger = Ledger("alice", Decimal("500"))
    return ledger


@pytest.fixture
def funded_bob() -> Ledger:
    return Ledger("bob", Decimal("200"))
```

```python
# src/notifications.py
"""External notification stub — would send email in production."""
from __future__ import annotations
import logging

log = logging.getLogger(__name__)


def send_deposit_notification(owner: str, amount: str) -> None:
    """Send email notification. Real impl calls external API."""
    log.info("Notifying %s of deposit %s", owner, amount)
    # In production: call email/SMS API here
```

```python
# src/ledger.py  (updated to call notification)
from __future__ import annotations
from decimal import Decimal
from src.notifications import send_deposit_notification


class InsufficientFundsError(Exception):
    pass


class Ledger:
    def __init__(self, owner: str, initial_balance: Decimal = Decimal("0")) -> None:
        if initial_balance < 0:
            raise ValueError("initial_balance must be non-negative")
        self.owner = owner
        self._balance = initial_balance

    @property
    def balance(self) -> Decimal:
        return self._balance

    def deposit(self, amount: Decimal) -> None:
        if amount <= 0:
            raise ValueError(f"deposit amount must be positive, got {amount}")
        self._balance += amount
        send_deposit_notification(self.owner, str(amount))  # external call

    def withdraw(self, amount: Decimal) -> None:
        if amount <= 0:
            raise ValueError(f"withdrawal amount must be positive, got {amount}")
        if amount > self._balance:
            raise InsufficientFundsError(
                f"cannot withdraw {amount}: balance is {self._balance}"
            )
        self._balance -= amount

    def transfer_to(self, other: "Ledger", amount: Decimal) -> None:
        self.withdraw(amount)
        other.deposit(amount)
```

```python
# tests/test_ledger_mocked.py
"""Tests that mock the external notification call."""
from __future__ import annotations
import pytest
from decimal import Decimal
from unittest.mock import patch, MagicMock


def test_deposit_calls_notification(alice):
    """deposit() must call send_deposit_notification once with correct args."""
    with patch("src.ledger.send_deposit_notification") as mock_notify:
        alice.deposit(Decimal("100"))
        mock_notify.assert_called_once_with("alice", "100")


def test_deposit_does_not_call_notification_on_error(alice):
    """If deposit raises, notification must NOT fire."""
    with patch("src.ledger.send_deposit_notification") as mock_notify:
        with pytest.raises(ValueError):
            alice.deposit(Decimal("-10"))
        mock_notify.assert_not_called()


def test_transfer_notifies_recipient(funded_alice, funded_bob):
    """Transfer notifies the deposit recipient (bob), not the sender."""
    with patch("src.ledger.send_deposit_notification") as mock_notify:
        funded_alice.transfer_to(funded_bob, Decimal("50"))
        # deposit is called on bob — notification fired once with bob's name
        mock_notify.assert_called_once_with("bob", "50")


# ── Fixture teardown demo ─────────────────────────────────────────────────────

@pytest.fixture
def temp_ledger_file(tmp_path):
    """tmp_path is a built-in pytest fixture providing a per-test temp dir."""
    path = tmp_path / "ledger.json"
    yield path
    # teardown: file cleaned up automatically by tmp_path


def test_ledger_serialises(temp_ledger_file, funded_alice):
    import json
    data = {"owner": funded_alice.owner, "balance": str(funded_alice.balance)}
    temp_ledger_file.write_text(json.dumps(data))
    loaded = json.loads(temp_ledger_file.read_text())
    assert loaded["owner"] == "alice"
    assert loaded["balance"] == "500"
```

## Exercises

1. Add a `conftest.py` with a `session`-scoped fixture that prints `"--- test session start ---"` before any tests run and `"--- test session end ---"` after all tests finish (use `yield` for teardown). Run pytest and observe the output.

2. The `Ledger` class should call `send_withdrawal_notification(owner, amount)` from `src.notifications` whenever `withdraw` succeeds. Implement that function and write three mocked tests: (a) notification called on success, (b) not called if amount is invalid, (c) not called if insufficient funds.

3. Write a `@pytest.fixture` named `ledger_pair` that returns a tuple `(Ledger, Ledger)` with Alice having £1000 and Bob having £0. Use this fixture in three tests covering different transfer scenarios.

4. Using `MagicMock`, simulate a `PaymentGateway` class with a `charge(amount)` method that can be configured to either return `{"status": "ok"}` or raise `ConnectionError`. Write tests that verify your service handles both outcomes.

5. Explain: why must you patch `"src.ledger.send_deposit_notification"` rather than `"src.notifications.send_deposit_notification"`?

(Solutions: ../solutions/week-05/day-3.md)

## Self-check
- [ ] I can write shared fixtures in `conftest.py` and use them across test files.
- [ ] I can set fixture scope and explain when each scope is appropriate.
- [ ] I can use `yield` in a fixture for teardown.
- [ ] I can patch a module-level function with `unittest.mock.patch`.
- [ ] I understand "patch where it's used, not where it's defined."
- [ ] I know the difference between a unit test and an integration test.

## Supplementary links
- [pytest — Fixtures reference](https://docs.pytest.org/en/stable/reference/fixtures.html)
- [Python docs — unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
- [Real Python — Understanding pytest Fixtures](https://realpython.com/pytest-python-testing/)
