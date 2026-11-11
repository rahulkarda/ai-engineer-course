# Week 03 Assessment — OOP & Python Object Model

## Conceptual (interview-style)

Answer in your own words — no notes.

1. Explain the difference between a **class** and an **instance**. Where are instance attributes stored at runtime?

2. What is the purpose of `__new__` vs `__init__`? Give one scenario where overriding `__new__` is appropriate.

3. A class defines `class_var = []` as a class attribute. A new instance appends to it via `self.class_var.append(x)`. What happens, and why is this a common bug?

4. Compare `@classmethod` and `@staticmethod`. Give one legitimate use case for each.

5. What does Python's name-mangling (`__attr`) actually do? Is it true access control? Why or why not?

6. Define the **Liskov Substitution Principle**. Does a `CurrentAccount.withdraw` that allows overdrafts violate it relative to `Account.withdraw`? Justify your answer.

7. What is the difference between an **ABC** and a **Protocol**? When would you prefer each?

8. Explain `@total_ordering`. What is the minimum set of comparison dunders you must implement?

9. `@dataclass(frozen=True)` — what does it do, and how does it affect `__hash__`?

10. Sketch the MRO for:
    ```python
    class A: ...
    class B(A): ...
    class C(A): ...
    class D(B, C): ...
    ```
    In what order are methods resolved? Why does C3 choose this order?

---

## Coding challenge

**Task: Extend the Banking Application**

You have been given the following stub. Implement it to specification.

```python
# challenge.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal
import uuid


class InsufficientFundsError(ValueError):
    """Raised when a withdrawal would breach balance or overdraft constraints."""


@dataclass(frozen=True)
class Transaction:
    amount: float
    type: Literal["deposit", "withdrawal", "interest", "fee"]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        # TODO: validate amount > 0
        ...


class Account(ABC):
    """Abstract base — do NOT modify the interface."""

    def __init__(self, owner: str, balance: float = 0.0) -> None:
        # TODO: validate balance >= 0; store owner, _balance, _history, generate account_id
        ...

    @property
    def balance(self) -> float:
        # TODO
        ...

    @property
    def account_id(self) -> str:
        # TODO
        ...

    @abstractmethod
    def deposit(self, amount: float) -> None: ...

    @abstractmethod
    def withdraw(self, amount: float) -> None: ...

    @abstractmethod
    def apply_monthly_update(self) -> None: ...

    def get_history(self) -> list[Transaction]:
        # TODO: return defensive copy
        ...

    def __repr__(self) -> str: ...
    def __len__(self) -> int: ...  # number of transactions
    def __eq__(self, other: object) -> bool: ...  # compare by account_id


class SavingsAccount(Account):
    """
    - interest_rate: float (default 0.02); validated 0 < rate < 1
    - minimum_balance: float (default 100.0)
    - deposit: add to balance; record Transaction("deposit")
    - withdraw: raise InsufficientFundsError if balance - amount < minimum_balance
    - apply_monthly_update: add interest; record Transaction("interest")
    """
    ...


class CurrentAccount(Account):
    """
    - overdraft_limit: property with setter; 0 <= limit <= 10_000
    - monthly_fee: float (default 5.0)
    - deposit: add, record
    - withdraw: allow to -(overdraft_limit); raise InsufficientFundsError beyond
    - apply_monthly_update: deduct monthly_fee; record Transaction("fee")
    """
    ...
```

**Acceptance criteria**

- [ ] `Transaction.__post_init__` raises `ValueError` on non-positive amount.
- [ ] `Account` cannot be instantiated directly (`TypeError`).
- [ ] `SavingsAccount.withdraw(amount)` raises `InsufficientFundsError` when `balance - amount < minimum_balance`.
- [ ] `CurrentAccount.withdraw(amount)` succeeds when `balance - amount >= -overdraft_limit`.
- [ ] `CurrentAccount.withdraw(amount)` raises `InsufficientFundsError` when it would exceed the overdraft.
- [ ] `apply_monthly_update` on `SavingsAccount` increases balance by `balance * interest_rate`.
- [ ] `apply_monthly_update` on `CurrentAccount` decreases balance by `monthly_fee`.
- [ ] `get_history()` returns a copy — mutating the returned list does not affect the account.
- [ ] `len(account)` returns number of transactions.
- [ ] `account1 == account2` is `True` iff they share the same `account_id`.

**Write tests** in `tests/test_challenge.py` that verify all acceptance criteria above.

---

## Rubric (pass ≥ 75%)

| Dimension | Weight | Criteria |
|---|---|---|
| Independent implementation | 40% | All acceptance criteria pass with no external aid |
| Conceptual understanding | 25% | Conceptual answers are accurate and specific |
| Debugging & problem-solving | 20% | Tests catch real edge cases; MRO / ABC errors are diagnosed correctly |
| Code quality & testing | 15% | Type hints throughout; meaningful test names; no bare `except`; `logging` not `print` |

**Pass gate: 75%**

Answer key: ../solutions/week-03/assessment.md
