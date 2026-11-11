# Phase 1 · Week 05 · Day 2 — pytest Fundamentals & Unit Testing

> Objective: Write a full pytest test suite — test discovery, assertions, parametrize, and isolated unit tests.  ·  Time: ~1 hr  ·  Prereqs: [[week-05/day-1]] (Git), [[week-03/day-5]] (functions)

## Concept

### What is a unit test?

A **unit test** exercises one function/method in complete isolation — no database, no network, no filesystem. It calls the unit with known inputs and asserts on outputs or side effects.

Properties of a good unit test (FIRST):
- **Fast**: milliseconds, not seconds.
- **Isolated**: no shared state between tests.
- **Repeatable**: same result every run.
- **Self-verifying**: pass/fail, no human inspection needed.
- **Timely**: written alongside or before the code.

### pytest discovery

pytest finds tests by convention:
- Files named `test_*.py` or `*_test.py`.
- Functions named `test_*` inside those files.
- Classes named `Test*` whose methods are named `test_*`.

```
project/
├── src/
│   └── ledger.py
└── tests/
    └── test_ledger.py   ← pytest discovers this
```

Run: `pytest tests/` or just `pytest` (auto-discovers from current dir).

### Assertions

pytest rewrites Python's `assert` at import time to produce detailed failure messages — no special assert methods needed:

```python
assert result == expected          # equality
assert result is None              # identity
assert "error" in message          # containment
assert value > 0                   # comparison
assert isinstance(obj, MyClass)    # type check
```

When an assertion fails, pytest shows the values of sub-expressions — much more useful than `assertTrue(x == y)`.

### Testing exceptions

```python
import pytest

def test_divide_by_zero():
    with pytest.raises(ZeroDivisionError):
        divide(10, 0)

# Check the message too
def test_invalid_amount():
    with pytest.raises(ValueError, match="must be positive"):
        deposit(-50)
```

### Parametrize

`@pytest.mark.parametrize` runs one test function with multiple input/output pairs — avoids duplicated test bodies:

```python
@pytest.mark.parametrize("a,b,expected", [
    (1, 2, 3),
    (0, 0, 0),
    (-1, 1, 0),
    (100, -50, 50),
])
def test_add(a, b, expected):
    assert add(a, b) == expected
```

pytest reports each case separately — you see exactly which combination failed.

## How it works internally

### Assert rewriting

pytest uses an import hook (`_pytest.assertion.rewrite`) that transforms the AST of test modules before they execute. It replaces `assert a == b` with code that captures `a` and `b` separately and formats them into a failure message. This is why pytest assertions are so readable without any assertion library.

### Test collection

When you run `pytest`, it:
1. Collects all matching files (respecting `testpaths` in `pyproject.toml`).
2. Imports each module.
3. Walks the module for `test_*` functions and `Test*` classes.
4. Builds a test plan (items list).
5. Runs each item, capturing stdout/stderr and exceptions.
6. Reports results.

### Exit codes

| Code | Meaning |
|------|---------|
| 0    | All tests passed |
| 1    | Some tests failed |
| 2    | Interrupted by user |
| 3    | Internal error |
| 4    | Command-line usage error |
| 5    | No tests collected |

CI pipelines check exit code 0/1 to pass or fail the build.

## Worked example

```python
# src/ledger.py
"""Minimal ledger with deposit, withdraw, balance."""
from __future__ import annotations
from decimal import Decimal


class InsufficientFundsError(Exception):
    """Raised when a withdrawal exceeds the current balance."""


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
# tests/test_ledger.py
"""Unit tests for ledger.py — no external dependencies."""
from __future__ import annotations
import pytest
from decimal import Decimal
from src.ledger import Ledger, InsufficientFundsError


# ── Basic behaviour ────────────────────────────────────────────────────────────

def test_initial_balance_zero():
    ledger = Ledger("alice")
    assert ledger.balance == Decimal("0")


def test_initial_balance_custom():
    ledger = Ledger("bob", Decimal("100"))
    assert ledger.balance == Decimal("100")


def test_deposit_increases_balance():
    ledger = Ledger("alice")
    ledger.deposit(Decimal("50"))
    assert ledger.balance == Decimal("50")


def test_withdraw_decreases_balance():
    ledger = Ledger("alice", Decimal("100"))
    ledger.withdraw(Decimal("30"))
    assert ledger.balance == Decimal("70")


def test_transfer_moves_funds():
    alice = Ledger("alice", Decimal("200"))
    bob = Ledger("bob", Decimal("0"))
    alice.transfer_to(bob, Decimal("75"))
    assert alice.balance == Decimal("125")
    assert bob.balance == Decimal("75")


# ── Invalid inputs ─────────────────────────────────────────────────────────────

def test_negative_initial_balance_raises():
    with pytest.raises(ValueError, match="non-negative"):
        Ledger("alice", Decimal("-1"))


def test_deposit_zero_raises():
    ledger = Ledger("alice")
    with pytest.raises(ValueError, match="positive"):
        ledger.deposit(Decimal("0"))


def test_deposit_negative_raises():
    ledger = Ledger("alice")
    with pytest.raises(ValueError, match="positive"):
        ledger.deposit(Decimal("-10"))


def test_withdraw_more_than_balance_raises():
    ledger = Ledger("alice", Decimal("50"))
    with pytest.raises(InsufficientFundsError):
        ledger.withdraw(Decimal("100"))


# ── Parametrized deposit amounts ──────────────────────────────────────────────

@pytest.mark.parametrize("amount,expected_balance", [
    (Decimal("1"), Decimal("1")),
    (Decimal("0.01"), Decimal("0.01")),
    (Decimal("9999.99"), Decimal("9999.99")),
    (Decimal("100"), Decimal("100")),
])
def test_deposit_various_amounts(amount: Decimal, expected_balance: Decimal):
    ledger = Ledger("test")
    ledger.deposit(amount)
    assert ledger.balance == expected_balance


# ── Parametrized invalid withdrawals ─────────────────────────────────────────

@pytest.mark.parametrize("amount", [
    Decimal("0"),
    Decimal("-1"),
    Decimal("-100"),
])
def test_withdraw_invalid_amounts_raise(amount: Decimal):
    ledger = Ledger("test", Decimal("100"))
    with pytest.raises(ValueError, match="positive"):
        ledger.withdraw(amount)
```

Run with: `pytest tests/test_ledger.py -v`

## Exercises

1. Add a `Ledger.statement()` method that returns a list of `(type, amount)` tuples recording every deposit and withdrawal. Write at least four unit tests for it: correct recording, ordering, empty ledger, mixed operations.

2. Write a parametrized test for `transfer_to` covering: successful transfer, transfer of full balance, zero-amount transfer (should raise), transfer larger than balance (should raise).

3. Add a `Ledger.freeze()` method that prevents further transactions. Write tests confirming that deposit/withdraw raise `RuntimeError("ledger is frozen")` after freezing, and that the balance is unchanged.

4. Run `pytest -v --tb=short` and explain each column in the output. What does the `-v` flag change?

5. pytest exit codes matter for CI. Write a tiny shell command (one line) that runs pytest and echoes `"CI passed"` only if all tests pass. Hint: `&&`.

(Solutions: ../solutions/week-05/day-2.md)

## Self-check
- [ ] I can write a test file pytest discovers automatically.
- [ ] I use plain `assert` and understand why pytest's output is informative.
- [ ] I can test that a function raises a specific exception with a specific message.
- [ ] I can parametrize a test to cover multiple input/output pairs in one function.
- [ ] I understand what "unit isolation" means and why it matters.

## Supplementary links
- [pytest documentation](https://docs.pytest.org/)
- [pytest — Good Integration Practices](https://docs.pytest.org/en/stable/explanation/goodpractices.html)
- [Real Python — Getting Started With Testing in Python](https://realpython.com/python-testing/)
