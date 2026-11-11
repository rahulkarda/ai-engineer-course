# Phase 1 · Week 05 · Day 7 — Debug Intentionally-Broken Code

> Objective: Practice the full debugging loop — read tracebacks, use the VS Code debugger, add targeted logging, and fix bugs in code you did not write.  ·  Time: ~3 hr  ·  Prereqs: [[week-05/day-4]] (logging/debugging), [[week-05/day-6]] (lab)

## Overview

This session provides intentionally broken Python code. Your job is to diagnose each bug using the systematic method from Day 4 and fix it — without peeking at the solution first.

For each bug:
1. Read the traceback or observe the wrong output.
2. Form a hypothesis.
3. Set a breakpoint (or add a targeted log) to verify.
4. Fix the root cause — not just the symptom.
5. Write a regression test that would catch the bug if reintroduced.

## Broken module: `buggy_ledger.py`

Copy this file into your project as `debugging/buggy_ledger.py`. Do **not** edit it — create a fixed copy at `debugging/fixed_ledger.py`.

```python
# debugging/buggy_ledger.py
# This file contains FIVE intentional bugs. Find and fix them all.
from __future__ import annotations
from decimal import Decimal
import logging

log = logging.getLogger(__name__)


class InsufficientFundsError(Exception):
    pass


# ── Bug hunt ──────────────────────────────────────────────────────────────────

class Account:
    """Simple bank account. Contains bugs — find them."""

    def __init__(self, owner: str, balance: Decimal = Decimal(0)) -> None:
        self.owner = owner
        self.balance = balance          # Bug 1: public mutable attribute
        self._transactions: list[dict] = []

    def deposit(self, amount: Decimal) -> None:
        if amount < 0:                  # Bug 2: wrong comparison
            raise ValueError("amount must be positive")
        self.balance += amount
        self._transactions.append({"type": "deposit", "amount": amount})
        log.info("deposited %s", amount)

    def withdraw(self, amount: Decimal) -> None:
        if amount <= 0:
            raise ValueError("amount must be positive")
        if amount > self.balance:
            raise InsufficientFundsError("insufficient funds")
        self.balance -= amount
        self._transactions.append({"type": "withdraw", "amount": amount})

    def transfer_to(self, other: "Account", amount: Decimal) -> None:
        self.withdraw(amount)
        other.deposit(amount)

    def statement(self) -> list[dict]:
        # Bug 3: returns a reference to internal list — caller can mutate it
        return self._transactions

    def total_deposits(self) -> Decimal:
        # Bug 4: sums ALL transactions, not just deposits
        return sum(t["amount"] for t in self._transactions)

    def net_balance_change(self) -> Decimal:
        # Bug 5: off by one — skips the first transaction
        total = Decimal(0)
        for i in range(1, len(self._transactions)):
            t = self._transactions[i]
            if t["type"] == "deposit":
                total += t["amount"]
            else:
                total -= t["amount"]
        return total
```

## Bug descriptions and debugging tasks

### Bug 1 — Public mutable attribute

**Symptom**: A caller can do `account.balance = Decimal("9999999")` with no checks.

**Task**: Find the line. Explain why direct attribute access is a design flaw. Fix by making `balance` a property backed by a private `_balance` attribute. Write a test that confirms `account.balance = ...` now raises `AttributeError`.

**Debugging technique**: Static analysis — ruff or a code review is sufficient. No runtime debugging needed.

---

### Bug 2 — Wrong comparison operator

**Symptom**: `account.deposit(Decimal("0"))` succeeds silently and adds a zero-amount deposit to the history.

**Task**:
1. Run this:
   ```python
   acc = Account("test")
   acc.deposit(Decimal("0"))
   print(acc.balance)          # 0 — looks fine
   print(acc.statement())       # [{"type": "deposit", "amount": 0}] — wrong!
   ```
2. Set a breakpoint on the `if amount < 0:` line. Step through with `amount = Decimal("0")`. Does the condition evaluate to `True` or `False`? Why?
3. Fix the condition. Write a parametrized test for `amount in [0, -1, -100]` that all raise `ValueError`.

---

### Bug 3 — Returning internal mutable state

**Symptom**: Modifying the list returned by `statement()` corrupts the account's history.

**Task**:
1. Run this:
   ```python
   acc = Account("test")
   acc.deposit(Decimal("50"))
   history = acc.statement()
   history.clear()                     # caller clears the list
   print(acc.statement())              # [] — history destroyed!
   ```
2. Identify why this happens (hint: Python list references).
3. Fix `statement()` to return a copy. Write a test that modifies the returned list and verifies the account's internal state is unchanged.

---

### Bug 4 — Wrong aggregate logic

**Symptom**: `total_deposits()` includes withdrawal amounts in the sum.

**Task**:
1. Run this:
   ```python
   acc = Account("test", Decimal("100"))
   acc.deposit(Decimal("50"))
   acc.withdraw(Decimal("30"))
   acc.deposit(Decimal("20"))
   print(acc.total_deposits())   # Should be 70 (50+20), prints 100 (50+30+20)
   ```
2. Add a debug log just inside the generator to print each `t["type"]`. Run and observe.
3. Fix the generator to filter only `deposit` type. Write parametrized tests.

---

### Bug 5 — Off-by-one error

**Symptom**: `net_balance_change()` ignores the first transaction, producing wrong totals.

**Task**:
1. Run this:
   ```python
   acc = Account("test")
   acc.deposit(Decimal("100"))    # transaction index 0 — SKIPPED
   acc.deposit(Decimal("50"))
   acc.withdraw(Decimal("30"))
   # Expected net change: +100 +50 -30 = +120
   # Actual: +50 -30 = +20 (first deposit skipped)
   print(acc.net_balance_change())
   ```
2. Set a breakpoint inside the `for` loop. Inspect `i` and `self._transactions` on the first iteration. What index is missing?
3. Fix the range. Write a test that deposits then withdraws and asserts the correct net change.

## Writing regression tests

For each bug you fixed, write a regression test in `tests/unit/test_regression.py`:

```python
# tests/unit/test_regression.py
"""
Regression tests — one per bug found.
Each test is named with the bug it catches so future failures are self-documenting.
"""
import pytest
from decimal import Decimal
from debugging.fixed_ledger import Account


def test_bug1_balance_attribute_is_readonly():
    """Balance must not be directly writable."""
    acc = Account("test", Decimal("100"))
    with pytest.raises(AttributeError):
        acc.balance = Decimal("9999999")


def test_bug2_deposit_zero_raises():
    """Zero deposit must raise ValueError."""
    acc = Account("test")
    with pytest.raises(ValueError, match="positive"):
        acc.deposit(Decimal("0"))


def test_bug3_statement_returns_copy():
    """Mutating the returned statement must not affect internal history."""
    acc = Account("test")
    acc.deposit(Decimal("50"))
    history = acc.statement()
    history.clear()
    assert len(acc.statement()) == 1   # internal history intact


def test_bug4_total_deposits_excludes_withdrawals():
    """total_deposits() must count only deposits."""
    acc = Account("test", Decimal("100"))
    acc.deposit(Decimal("50"))
    acc.withdraw(Decimal("30"))
    acc.deposit(Decimal("20"))
    assert acc.total_deposits() == Decimal("70")


def test_bug5_net_balance_change_includes_first_transaction():
    """net_balance_change() must include the very first transaction."""
    acc = Account("test")
    acc.deposit(Decimal("100"))
    acc.deposit(Decimal("50"))
    acc.withdraw(Decimal("30"))
    assert acc.net_balance_change() == Decimal("120")
```

## Bonus challenge

The module below has a subtle concurrency bug. Without running it, read the code and explain in writing what the bug is and how it would manifest under concurrent access:

```python
import threading
from decimal import Decimal

class ThreadedAccount:
    def __init__(self):
        self._balance = Decimal("0")

    def deposit(self, amount: Decimal) -> None:
        current = self._balance        # read
        # (context switch can happen here)
        self._balance = current + amount  # write

    def withdraw(self, amount: Decimal) -> None:
        if amount > self._balance:
            raise ValueError("insufficient")
        current = self._balance
        self._balance = current - amount
```

(No code change needed — written explanation only.)

## Self-check
- [ ] I read each traceback bottom-up and identified the root-cause frame.
- [ ] I used at least two different debugging techniques (breakpoint, log, static analysis).
- [ ] I fixed the root cause, not just silenced the symptom.
- [ ] Every bug has a regression test in `test_regression.py`.
- [ ] I can explain the concurrency bug in `ThreadedAccount` in writing.

(Solutions: ../solutions/week-05/day-7.md)
