# Solutions — Week 03 Assessment

## Conceptual answers

**1. Class vs instance; `__dict__`**  
A class is the blueprint stored once in the module namespace. An instance is a concrete object created by calling the class. Instance attributes live in the instance's `__dict__`. The class's methods and class attributes live in the class's `__dict__`. Python looks up attributes instance-first, then class, then MRO chain.

**2. `__new__` vs `__init__`**  
`__new__` allocates and returns the object (called first). `__init__` receives the already-allocated object and initialises it. Override `__new__` when subclassing immutable types (`str`, `int`, `tuple`) because by the time `__init__` runs the value is already fixed.

**3. Mutable class attribute bug**  
`class A: x = []` — `A.x` is one list shared by all instances. `a.x.append(1)` mutates that shared list; it does NOT create an instance attribute. `A.x` and `a.x` both show `[1]`. Fix: `def __init__(self): self.x = []`.

**4. `@classmethod` vs `@staticmethod`**  
`@classmethod` — alternate constructors (`Account.from_dict(data)`); `cls` is the class so subclasses get a correct factory.  
`@staticmethod` — pure utilities with no instance/class state: `Account.is_valid_account_number("1234567890")`.

**5. Name mangling**  
`self.__attr` inside class `Foo` is stored as `self._Foo__attr`. Accessible but ugly. Prevents *accidental* overrides in subclasses; not true access control.

**6. LSP — `CurrentAccount.withdraw`**  
LSP: subtypes must honour the parent's contract. `Account.withdraw` contract: "withdraws amount if sufficient funds". `CurrentAccount.withdraw` weakens the precondition (allows going below 0 up to overdraft) — LSP says weakening preconditions is acceptable. So it does NOT violate LSP, provided the new invariant is documented.

**7. ABC vs Protocol**  
ABCs: explicit inheritance, `TypeError` at instantiation if abstract methods unimplemented. Good for internal hierarchies you control.  
Protocols: structural, checked by mypy/pyright only, no inheritance. Good for function parameter type hints: "I accept anything with a `notify(owner, message)` method."

**8. `@total_ordering`**  
Define `__eq__` + one of `__lt__`, `__le__`, `__gt__`, `__ge__`. The decorator fills in the rest. Minimum: `__eq__` + `__lt__`.

**9. `frozen=True` and `__hash__`**  
`frozen=True` makes all fields read-only (raises `FrozenInstanceError` on mutation) and auto-generates `__hash__` from all fields. Without `frozen`, defining `__eq__` sets `__hash__ = None` (unhashable) to prevent set/dict key inconsistency.

**10. MRO for D(B, C), B(A), C(A)**  
`D → B → C → A → object`. C3 takes classes left-to-right, depth-first, but defers any class that still appears as a parent later in the merge list, avoiding the inconsistency that naïve DFS produces.

---

## Coding challenge — complete solution

```python
# challenge.py
from __future__ import annotations
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal


class InsufficientFundsError(ValueError):
    pass


@dataclass(frozen=True)
class Transaction:
    amount: float
    type: Literal["deposit", "withdrawal", "interest", "fee"]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if self.amount <= 0:
            raise ValueError(f"Transaction amount must be positive, got {self.amount}")


class Account(ABC):
    def __init__(self, owner: str, balance: float = 0.0) -> None:
        if balance < 0:
            raise ValueError("Balance cannot be negative")
        self.owner = owner
        self._account_id: str = str(uuid.uuid4())[:8]
        self._balance: float = balance
        self._history: list[Transaction] = []

    @property
    def balance(self) -> float:
        return self._balance

    @property
    def account_id(self) -> str:
        return self._account_id

    @abstractmethod
    def deposit(self, amount: float) -> None: ...

    @abstractmethod
    def withdraw(self, amount: float) -> None: ...

    @abstractmethod
    def apply_monthly_update(self) -> None: ...

    def get_history(self) -> list[Transaction]:
        return list(self._history)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(owner={self.owner!r}, balance={self._balance:.2f})"

    def __len__(self) -> int:
        return len(self._history)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Account):
            return NotImplemented
        return self._account_id == other._account_id


class SavingsAccount(Account):
    def __init__(
        self,
        owner: str,
        balance: float = 0.0,
        interest_rate: float = 0.02,
        minimum_balance: float = 100.0,
    ) -> None:
        if not (0 < interest_rate < 1):
            raise ValueError(f"Invalid rate: {interest_rate}")
        super().__init__(owner, balance)
        self.interest_rate = interest_rate
        self.minimum_balance = minimum_balance

    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Amount must be positive")
        self._balance += amount
        self._history.append(Transaction(amount=amount, type="deposit"))

    def withdraw(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Amount must be positive")
        if self._balance - amount < self.minimum_balance:
            raise InsufficientFundsError(
                f"Withdrawal would breach minimum balance of {self.minimum_balance}"
            )
        self._balance -= amount
        self._history.append(Transaction(amount=amount, type="withdrawal"))

    def apply_monthly_update(self) -> None:
        interest = self._balance * self.interest_rate
        self._balance += interest
        self._history.append(Transaction(amount=interest, type="interest"))


class CurrentAccount(Account):
    def __init__(
        self,
        owner: str,
        balance: float = 0.0,
        overdraft_limit: float = 500.0,
        monthly_fee: float = 5.0,
    ) -> None:
        super().__init__(owner, balance)
        self._overdraft_limit = 0.0
        self.overdraft_limit = overdraft_limit
        self.monthly_fee = monthly_fee

    @property
    def overdraft_limit(self) -> float:
        return self._overdraft_limit

    @overdraft_limit.setter
    def overdraft_limit(self, value: float) -> None:
        if not (0 <= value <= 10_000):
            raise ValueError(f"Overdraft limit must be 0–10,000, got {value}")
        self._overdraft_limit = value

    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Amount must be positive")
        self._balance += amount
        self._history.append(Transaction(amount=amount, type="deposit"))

    def withdraw(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Amount must be positive")
        if amount > self._balance + self._overdraft_limit:
            raise InsufficientFundsError("Exceeds overdraft limit")
        self._balance -= amount
        self._history.append(Transaction(amount=amount, type="withdrawal"))

    def apply_monthly_update(self) -> None:
        self._balance -= self.monthly_fee
        self._history.append(Transaction(amount=self.monthly_fee, type="fee"))
```

```python
# tests/test_challenge.py
import pytest
from challenge import Account, CurrentAccount, InsufficientFundsError, SavingsAccount, Transaction


def test_transaction_post_init_rejects_zero() -> None:
    with pytest.raises(ValueError):
        Transaction(amount=0, type="deposit")


def test_account_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        Account("Alice")  # type: ignore


def test_savings_withdraw_below_minimum_raises() -> None:
    acc = SavingsAccount("Alice", 200.0, minimum_balance=100.0)
    with pytest.raises(InsufficientFundsError):
        acc.withdraw(150.0)  # would leave 50 < 100


def test_current_withdraw_within_overdraft() -> None:
    acc = CurrentAccount("Bob", 100.0, overdraft_limit=200.0)
    acc.withdraw(250.0)
    assert acc.balance == pytest.approx(-150.0)


def test_current_withdraw_exceeds_overdraft_raises() -> None:
    acc = CurrentAccount("Bob", 100.0, overdraft_limit=200.0)
    with pytest.raises(InsufficientFundsError):
        acc.withdraw(400.0)


def test_savings_monthly_interest() -> None:
    acc = SavingsAccount("Alice", 1000.0, interest_rate=0.02)
    acc.apply_monthly_update()
    assert acc.balance == pytest.approx(1020.0)


def test_current_monthly_fee() -> None:
    acc = CurrentAccount("Bob", 100.0, monthly_fee=5.0)
    acc.apply_monthly_update()
    assert acc.balance == pytest.approx(95.0)


def test_get_history_returns_copy() -> None:
    acc = SavingsAccount("Alice", 500.0)
    acc.deposit(100.0)
    history = acc.get_history()
    history.clear()
    assert len(acc.get_history()) > 0  # internal list unchanged


def test_len_returns_transaction_count() -> None:
    acc = SavingsAccount("Alice", 500.0)
    acc.deposit(100.0)
    acc.deposit(50.0)
    assert len(acc) == 2


def test_eq_by_account_id() -> None:
    acc1 = SavingsAccount("Alice", 100.0)
    acc2 = SavingsAccount("Alice", 100.0)
    assert acc1 != acc2  # different UUIDs
    # same id check
    class FakeAccount(CurrentAccount):
        pass
    fa = FakeAccount("Bob", 0.0)
    fa._account_id = acc1.account_id  # type: ignore
    assert acc1 == fa
```
