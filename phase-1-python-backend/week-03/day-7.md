# Phase 1 · Week 03 · Day 7 — Sunday: Refactor, Test & Assessment Prep

> Objective: Refactor the Day 6 banking application with MRO / multiple inheritance patterns, finalise test coverage, and prepare for the Week 3 assessment.  ·  Time: ~3 hr project + 1 hr assessment prep  ·  Prereqs: [[day-6]]

## Overview

Three streams today:

1. **Refactor (1.5 hr)** — apply MRO / multiple inheritance, mixins, and Protocol enhancements.
2. **Test hardening (1 hr)** — edge cases, parametrize, coverage report.
3. **Assessment prep (0.5 hr)** — self-quiz on the week's concepts.

---

## Part 1 — Refactor: MRO, Mixins & Protocols (1.5 hr)

### Multiple inheritance & MRO review

Python resolves method lookup with C3 linearisation. For:

```python
class A: ...
class B(A): ...
class C(A): ...
class D(B, C): ...
```

`D.__mro__` is `(D, B, C, A, object)`. Each class appears exactly once; the order is left-to-right, depth-first with de-duplication.

**Mixins** are thin classes that add behaviour without forming part of the main hierarchy. Convention: name them `XxxMixin`.

### Refactoring tasks

**Task 1 — Add a `LoggingMixin`**

```python
class LoggingMixin:
    """Mixin that adds structured logging of balance changes."""

    def _log_transaction(self, type_: str, amount: float, balance_after: float) -> None:
        logger.info(
            "%-12s | owner=%-15s | amount=%8.2f | balance=%10.2f",
            type_.upper(),
            getattr(self, "owner_name", "unknown"),
            amount,
            balance_after,
        )
```

Apply it: `class SavingsAccount(LoggingMixin, Account)`. Call `self._log_transaction(...)` inside `deposit` and `withdraw`. Verify `__mro__` reflects the order.

**Task 2 — Add an `AuditMixin`**

```python
class AuditMixin:
    """Records every method call on deposit/withdraw to an external audit log (list)."""
    _audit_log: list[str]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        cls._audit_log = []
```

Override `deposit` and `withdraw` in the mixin to append `f"{datetime.utcnow().isoformat()} {self.__class__.__name__}.{method}({amount})"` before calling `super()`. Apply to `CurrentAccount`. Write a test that checks `CurrentAccount._audit_log` is populated after a withdrawal.

**Task 3 — MRO diamond check**

Create a `PremiumAccount` that inherits from both `SavingsAccount` and an `AuditMixin`:

```python
class PremiumAccount(AuditMixin, SavingsAccount):
    """High-interest savings with full audit trail."""
    RATE = 0.05
```

Print `PremiumAccount.__mro__`. Confirm `deposit` calls `AuditMixin.deposit` → `SavingsAccount.deposit` → `Account.deposit` in order via `super()` chain. Add a test.

**Task 4 — Enhance Protocols**

Add a `Reportable` Protocol:

```python
@runtime_checkable
class Reportable(Protocol):
    def generate_statement(self) -> str: ...
    def get_history(self) -> list[Transaction]: ...
```

Write `export_to_csv(account: Reportable, path: Path) -> None` that writes transaction history to a CSV file using `csv.writer`. Test it with a temp file.

---

## Part 2 — Test hardening (1 hr)

### Coverage target: ≥ 85% on `banking/models.py`

Install: `pip install pytest-cov`

Run: `pytest tests/ --cov=banking --cov-report=term-missing`

### Required edge-case tests (add to `tests/test_banking.py`)

```python
# Parametrized deposit amounts
@pytest.mark.parametrize("amount", [0.01, 1.0, 999_999.99])
def test_savings_deposit_valid_amounts(amount: float) -> None: ...

# Zero amount must always fail
@pytest.mark.parametrize("method", ["deposit", "withdraw"])
def test_zero_amount_raises(method: str) -> None: ...

# Negative amount must always fail
def test_negative_deposit_raises() -> None: ...

# Minimum balance exactly met
def test_savings_withdraw_to_minimum_exactly() -> None: ...

# Overdraft limit edge cases
def test_current_withdraw_exactly_at_limit() -> None: ...
def test_current_withdraw_one_cent_over_limit_raises() -> None: ...

# Transaction history integrity
def test_transaction_history_immutable_copy() -> None:
    """Mutating the returned list must not affect internal history."""
    ...

# Bank monthly update hits every account
def test_bank_monthly_update_all_accounts_updated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use monkeypatch to track apply_monthly_update calls."""
    ...

# Customer email validation
@pytest.mark.parametrize("email", ["notanemail", "", "no_at_sign"])
def test_invalid_emails_raise(email: str) -> None: ...
```

### Fixtures to add to `conftest.py`

```python
import pytest
from banking.models import Bank, Customer, SavingsAccount, CurrentAccount

@pytest.fixture()
def savings_account() -> SavingsAccount:
    return SavingsAccount("Alice", initial_balance=1000.0)

@pytest.fixture()
def current_account() -> CurrentAccount:
    return CurrentAccount("Bob", initial_balance=500.0)

@pytest.fixture()
def customer_with_accounts(
    savings_account: SavingsAccount, current_account: CurrentAccount
) -> Customer:
    c = Customer("Alice", "alice@example.com")
    c.add_account(savings_account)
    c.add_account(current_account)
    return c

@pytest.fixture()
def bank(customer_with_accounts: Customer) -> Bank:
    b = Bank("Test Bank")
    b.register_customer(customer_with_accounts)
    return b
```

---

## Part 3 — Assessment prep (0.5 hr)

Work through these self-quiz questions without looking at notes. Check your answers against Day 1–5 material.

1. What does `__new__` do vs `__init__`? When would you override `__new__`?
2. A class method returns `cls(...)` — why is this better than `ClassName(...)` for factory methods in an inheritance hierarchy?
3. You have `class A: x = 1` and `a = A(); a.x = 2`. What does `A.x` equal? What does `a.x` equal? What's in `a.__dict__`?
4. Explain the Liskov Substitution Principle. Does `CurrentAccount.withdraw` satisfy it compared to `Account.withdraw`?
5. What is the difference between `@property` and a plain attribute? When does the extra syntax pay off?
6. A colleague defines `class Mixin(Account)`. Why is this wrong, and how should they define it?
7. You call `sorted(accounts)` on a list of `Account` objects. What dunder methods must `Account` implement?
8. `@dataclass(frozen=True)` — what changes, and what breaks if you try `transaction.amount = 99`?
9. Why must `__hash__` be re-implemented when `__eq__` is defined?
10. Sketch the MRO for `class D(B, C)` where `B(A)` and `C(A)`.

(Detailed answers are in the Week 3 Assessment answer key: `../solutions/week-03/assessment.md`)

---

## Exercises

1. Implement all four refactoring tasks above. Run the full test suite — all tests must pass before moving to assessment.

2. Reach ≥ 85% coverage. Identify which branches are uncovered and add tests for them.

3. Add `__slots__` to `Transaction` (frozen dataclass already has it implicitly, but add it to `Account` subclasses manually). Measure memory with `sys.getsizeof` before and after. Is the saving significant for 1,000 instances?

4. Write a `conftest.py` with the fixtures above. Refactor at least three existing tests to use them.

(Solutions: ../solutions/week-03/day-7.md)

## Self-check

- [ ] I can trace the MRO for a 3-class diamond hierarchy.
- [ ] I know why `super()` in a mixin always calls the next class in MRO, not a fixed parent.
- [ ] My test suite reaches ≥ 85% line coverage.
- [ ] I can answer the assessment prep questions without notes.
- [ ] I feel ready to build the full Banking Application from spec in the assessment.

## Supplementary links

- [Python docs — `__slots__`](https://docs.python.org/3/reference/datamodel.html#slots)
- [Python docs — MRO details](https://docs.python.org/3/library/stdtypes.html#class.__mro__)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
