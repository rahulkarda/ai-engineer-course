# Phase 1 · Week 03 · Day 2 — Instance, Class & Static Methods + Encapsulation

> Objective: Distinguish the three method types, apply name-mangling for encapsulation, and introduce abstraction via public interfaces.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

## Concept

### Three kinds of methods

| Type | Decorator | First param | Receives |
|---|---|---|---|
| Instance method | (none) | `self` | the instance |
| Class method | `@classmethod` | `cls` | the class itself |
| Static method | `@staticmethod` | (none) | neither |

**Instance methods** are the norm — they read/write instance state.

**Class methods** are factories or alternate constructors — they create or configure instances without needing an existing one.

**Static methods** are utility functions logically grouped with the class but not needing instance or class state.

### Encapsulation

Hide internal implementation; expose only what callers should use.

Python uses **name convention** rather than compiler enforcement:
- `_name` → "internal, use at your own risk"
- `__name` → name-mangled to `_ClassName__name`, hard to access accidentally

**Abstraction**: expose a simple, intention-revealing interface; hide the "how".

## How it works internally

`@classmethod` wraps the function in a descriptor that passes the class (not the instance) as the first argument. Useful because subclasses inherit it and `cls` becomes the subclass — enabling polymorphic factories.

`@staticmethod` wraps the function so no implicit first argument is injected. Calling `MyClass.static_fn()` is exactly like calling a plain module function.

Name mangling: `self.__balance` inside class `Account` is stored as `self._Account__balance`. This prevents accidental overrides in subclasses that define the same attribute name, not true private enforcement.

## Worked example

```python
# account.py
from __future__ import annotations
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class Account:
    """Bank account with encapsulated balance and factory constructors."""

    _INTEREST_RATE: float = 0.03  # class attribute — shared by all instances

    def __init__(self, owner: str, balance: float = 0.0) -> None:
        if balance < 0:
            raise ValueError("Balance cannot be negative")
        self.owner = owner
        self.__balance: float = balance   # name-mangled; not a true secret
        self._created_at: str = datetime.utcnow().isoformat()  # internal

    # ── Instance methods ─────────────────────────────────────────────────
    def deposit(self, amount: float) -> None:
        self._validate_positive(amount, "deposit")
        self.__balance += amount
        logger.info("%s deposited %.2f", self.owner, amount)

    def withdraw(self, amount: float) -> None:
        self._validate_positive(amount, "withdrawal")
        if amount > self.__balance:
            raise ValueError("Insufficient funds")
        self.__balance -= amount

    def get_balance(self) -> float:
        """Public accessor — the intended interface for reading balance."""
        return self.__balance

    # ── Class methods ────────────────────────────────────────────────────
    @classmethod
    def from_dict(cls, data: dict[str, float | str]) -> "Account":
        """Alternate constructor — create Account from a plain dict."""
        return cls(owner=str(data["owner"]), balance=float(data.get("balance", 0)))

    @classmethod
    def set_interest_rate(cls, rate: float) -> None:
        if not 0 < rate < 1:
            raise ValueError("Rate must be between 0 and 1")
        cls._INTEREST_RATE = rate

    # ── Static methods ───────────────────────────────────────────────────
    @staticmethod
    def _validate_positive(amount: float, label: str) -> None:
        if amount <= 0:
            raise ValueError(f"{label} amount must be positive, got {amount}")

    @staticmethod
    def format_currency(amount: float) -> str:
        return f"${amount:,.2f}"

    def __repr__(self) -> str:
        return f"Account(owner={self.owner!r}, balance={self.format_currency(self.__balance)})"


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    acc = Account("Alice", 1000.0)
    acc.deposit(500.0)
    acc.withdraw(200.0)
    print(acc)                              # Account(owner='Alice', balance=$1,300.00)

    # Class method factory
    acc2 = Account.from_dict({"owner": "Bob", "balance": 250.0})
    print(acc2)

    # Static utility
    print(Account.format_currency(99999.5))  # $99,999.50

    # Class-level state change
    Account.set_interest_rate(0.05)
    print(Account._INTEREST_RATE)            # 0.05

    # Name mangling demo
    print(acc.__dict__)  # shows _Account__balance, not __balance


if __name__ == "__main__":
    main()
```

## Exercises

1. Add a `apply_interest() -> None` instance method that multiplies the current balance by `1 + _INTEREST_RATE`. Write a test confirming the math.

2. Add a `from_csv_line(cls, line: str) -> Account` class method that parses `"Alice,1000.00"` format.

3. Add a `is_valid_account_number(number: str) -> bool` static method (an account number is valid if it is exactly 10 digits). Write two tests: one valid, one invalid number.

4. Try accessing `acc._Account__balance` directly in the REPL. Does it work? What does this tell you about Python's approach to privacy?

5. Refactor exercise 2 from Day 1: make `Customer.from_name_email(cls, name: str, email: str)` a class method and verify it still works.

(Solutions: ../solutions/week-03/day-2.md)

## Self-check

- [ ] I know when to use `@classmethod` vs `@staticmethod` vs plain instance method.
- [ ] I can explain what `_` and `__` prefixes mean in practice.
- [ ] I understand name mangling and can demonstrate it at the REPL.
- [ ] I can write a factory class method that returns a fully initialised instance.
- [ ] I can articulate the difference between encapsulation and access control.

## Supplementary links

- [Python docs — classmethod](https://docs.python.org/3/library/functions.html#classmethod)
- [Python docs — staticmethod](https://docs.python.org/3/library/functions.html#staticmethod)
- [Real Python — OOP method types](https://realpython.com/instance-class-and-static-methods-demystified/)
