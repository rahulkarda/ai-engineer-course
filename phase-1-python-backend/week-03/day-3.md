# Phase 1 · Week 03 · Day 3 — Inheritance, Polymorphism & Composition

> Objective: Use inheritance to specialise classes, override methods for polymorphism, and understand when to prefer composition.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

## Concept

### Inheritance

A **subclass** extends a **parent class** (superclass), inheriting all its methods and attributes.

```python
class SavingsAccount(Account):  # SavingsAccount IS-A Account
    ...
```

Use inheritance when the relationship is genuinely "is-a" and behaviour is shared.

### Method overriding

A subclass redefines a method from the parent. Python calls the most-derived version first. Call `super()` to delegate to the parent's implementation.

### Polymorphism

Different classes respond to the same method call differently — callers don't need to know the concrete type:

```python
accounts: list[Account] = [SavingsAccount(...), CurrentAccount(...)]
for acc in accounts:
    acc.apply_monthly_update()  # each type does the right thing
```

### Composition vs Inheritance

**Inheritance** ("is-a"): `SavingsAccount` is an `Account`.  
**Composition** ("has-a"): `Customer` has a `list[Account]`.

Prefer composition when:
- The relationship is not truly "is-a".
- You need to combine behaviours from multiple unrelated classes.
- You want to swap the contained object at runtime.

Rule of thumb: **favour composition over inheritance** (GoF). Use inheritance for genuine specialisation only.

## How it works internally

Python's **MRO** (Method Resolution Order) determines which class's method is called. It uses the **C3 linearisation** algorithm. Inspect it:

```python
print(SavingsAccount.__mro__)
# (<class 'SavingsAccount'>, <class 'Account'>, <class 'object'>)
```

`super()` in Python 3 is `super(CurrentClass, self)` — it returns a proxy that starts searching from the *next* class in the MRO, not necessarily the direct parent. This matters in multiple inheritance.

## Worked example

```python
# accounts.py
from __future__ import annotations
import logging
from typing import ClassVar

logger = logging.getLogger(__name__)


class Account:
    """Base account."""

    def __init__(self, owner: str, balance: float = 0.0) -> None:
        if balance < 0:
            raise ValueError("Balance cannot be negative")
        self.owner = owner
        self._balance: float = balance

    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Amount must be positive")
        self._balance += amount

    def withdraw(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Amount must be positive")
        if amount > self._balance:
            raise ValueError("Insufficient funds")
        self._balance -= amount

    def get_balance(self) -> float:
        return self._balance

    def apply_monthly_update(self) -> None:
        """Override in subclasses for type-specific monthly logic."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(owner={self.owner!r}, balance={self._balance:.2f})"


class SavingsAccount(Account):
    """Earns monthly interest; no overdraft."""

    INTEREST_RATE: ClassVar[float] = 0.02

    def __init__(self, owner: str, balance: float = 0.0) -> None:
        super().__init__(owner, balance)  # delegate to Account.__init__
        self.interest_earned: float = 0.0

    def apply_monthly_update(self) -> None:
        interest = self._balance * self.INTEREST_RATE
        self._balance += interest
        self.interest_earned += interest
        logger.info("%s earned interest: %.2f", self.owner, interest)

    def withdraw(self, amount: float) -> None:
        """Override: savings accounts do not allow overdraft (parent already blocks it)."""
        super().withdraw(amount)  # reuse parent logic — already correct here
        logger.info("%s withdrew %.2f from savings", self.owner, amount)


class CurrentAccount(Account):
    """Allows overdraft up to a limit; charges monthly fee."""

    MONTHLY_FEE: ClassVar[float] = 5.0

    def __init__(self, owner: str, balance: float = 0.0, overdraft_limit: float = 500.0) -> None:
        super().__init__(owner, balance)
        self.overdraft_limit = overdraft_limit

    def withdraw(self, amount: float) -> None:
        """Override: allow overdraft up to limit."""
        if amount <= 0:
            raise ValueError("Amount must be positive")
        if amount > self._balance + self.overdraft_limit:
            raise ValueError(f"Exceeds overdraft limit of {self.overdraft_limit:.2f}")
        self._balance -= amount

    def apply_monthly_update(self) -> None:
        self._balance -= self.MONTHLY_FEE
        logger.info("%s charged monthly fee %.2f", self.owner, self.MONTHLY_FEE)


# ── Composition example ───────────────────────────────────────────────────────

class Customer:
    """HAS-A list of accounts (composition, not inheritance)."""

    def __init__(self, name: str, email: str) -> None:
        self.name = name
        self.email = email
        self._accounts: list[Account] = []

    def add_account(self, account: Account) -> None:
        self._accounts.append(account)

    def total_balance(self) -> float:
        return sum(a.get_balance() for a in self._accounts)

    def apply_monthly_update_all(self) -> None:
        for acc in self._accounts:
            acc.apply_monthly_update()  # polymorphism — each type does the right thing

    def __repr__(self) -> str:
        return f"Customer(name={self.name!r}, accounts={len(self._accounts)})"


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    alice = Customer("Alice", "alice@example.com")
    alice.add_account(SavingsAccount("Alice", 1000.0))
    alice.add_account(CurrentAccount("Alice", 200.0, overdraft_limit=300.0))

    print(alice)
    print("Total:", alice.total_balance())

    alice.apply_monthly_update_all()
    print("After monthly update:", alice.total_balance())

    # Polymorphism: process a mixed list
    accounts: list[Account] = [SavingsAccount("Bob", 500), CurrentAccount("Carol", 100)]
    for acc in accounts:
        acc.apply_monthly_update()
        print(acc)


if __name__ == "__main__":
    main()
```

## Exercises

1. Add a `PremiumSavingsAccount(SavingsAccount)` that overrides `INTEREST_RATE = 0.05`. Confirm `apply_monthly_update` applies the higher rate.

2. The `CurrentAccount` currently allows negative balances. Add a `get_balance()` override that shows the overdraft as a negative number and a separate `is_overdrawn() -> bool` method.

3. Create a `Transaction` class (composition) with fields: `amount: float`, `type: str` (`"deposit"` or `"withdrawal"`), `timestamp: str`. Modify `Account.deposit` and `Account.withdraw` to record transactions in `self._history: list[Transaction]`. Add `get_history() -> list[Transaction]`.

4. Write a function `process_accounts(accounts: list[Account]) -> dict[str, float]` that returns a mapping of `owner → balance`. Verify it works with a mixed list.

5. Explain: when should you use inheritance vs composition for a `LoanAccount` that needs repayment logic?

(Solutions: ../solutions/week-03/day-3.md)

## Self-check

- [ ] I can create a subclass and call `super().__init__()` correctly.
- [ ] I understand method overriding and when to call `super()`.
- [ ] I can explain polymorphism with a concrete example.
- [ ] I know the C3 MRO rule and can predict `__mro__` for simple hierarchies.
- [ ] I can articulate "favour composition over inheritance" with an example.

## Supplementary links

- [Python docs — super()](https://docs.python.org/3/library/functions.html#super)
- [Python MRO — C3 linearisation](https://www.python.org/download/releases/2.3/mro/)
- [Real Python — Inheritance and composition](https://realpython.com/inheritance-composition-python/)
