# Phase 1 · Week 03 · Day 1 — Classes, Objects & Constructors

> Objective: Build and instantiate Python classes with instance attributes using `__init__` and understand the object model.  ·  Time: ~1 hr  ·  Prereqs: [[week-02/day-7]]

## Concept

A **class** is a blueprint; an **object** (instance) is a concrete value made from that blueprint. Python classes group data (attributes) and behaviour (methods) into one unit.

Key vocabulary:
- **Class**: defined with `class Name:`, lives in the module namespace.
- **Instance**: created by calling the class like a function — `obj = MyClass()`.
- **Constructor**: `__init__` — runs once per instance to initialise attributes.
- **`self`**: the first parameter of every instance method; Python passes the instance automatically.

```python
class BankAccount:
    def __init__(self, owner: str, balance: float = 0.0) -> None:
        self.owner = owner        # instance attribute
        self.balance = balance    # instance attribute
```

Every call to `BankAccount("Alice")` creates a fresh, independent object with its own `owner` and `balance`.

## How it works internally

When Python executes `account = BankAccount("Alice")`:

1. `BankAccount.__new__(BankAccount)` allocates memory for the object and returns an empty instance.
2. `BankAccount.__init__(instance, "Alice")` is called on that new instance — `self` is the freshly allocated object.
3. The name `account` is bound to that object in the local namespace.

Instance attributes live in the instance's `__dict__`:

```python
account = BankAccount("Alice", 100.0)
print(account.__dict__)  # {'owner': 'Alice', 'balance': 100.0}
```

Python looks up attributes using the **MRO** (Method Resolution Order): instance `__dict__` → class `__dict__` → base classes. If an attribute isn't found anywhere, `AttributeError` is raised.

## Worked example

```python
# bank_account.py
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


class BankAccount:
    """Simple bank account with owner and balance."""

    def __init__(self, owner: str, balance: float = 0.0) -> None:
        if balance < 0:
            raise ValueError(f"Initial balance cannot be negative: {balance}")
        self.owner: str = owner
        self.balance: float = balance
        logger.debug("Created account for %s with balance %.2f", owner, balance)

    def deposit(self, amount: float) -> None:
        """Add amount to balance."""
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self.balance += amount

    def __repr__(self) -> str:
        return f"BankAccount(owner={self.owner!r}, balance={self.balance:.2f})"


def main() -> None:
    logging.basicConfig(level=logging.DEBUG)
    acc = BankAccount("Alice", 500.0)
    print(acc)               # BankAccount(owner='Alice', balance=500.00)
    acc.deposit(200.0)
    print(acc.balance)       # 700.0

    # Two independent instances
    acc2 = BankAccount("Bob")
    print(acc2.balance)      # 0.0
    print(acc is acc2)       # False


if __name__ == "__main__":
    main()
```

Run it:

```
python bank_account.py
```

Notice:
- `acc` and `acc2` are distinct objects; modifying one does not affect the other.
- `__repr__` gives a useful developer representation.
- Validation in `__init__` keeps the object in a valid state from the start.

## Exercises

1. Add a `withdraw(amount: float)` method to `BankAccount`. Raise `ValueError` if `amount > self.balance` or `amount <= 0`. Write a short script that deposits, then withdraws, and prints the final balance.

2. Create a `Customer` class with attributes: `name: str`, `email: str`, and `accounts: list[BankAccount]` (default empty list). Add an `add_account(account: BankAccount) -> None` method. Instantiate one customer, add two accounts, and print the customer's account count.

3. What does `account.__class__` return? What about `type(account)`? Are they the same? Try it in the REPL and explain.

4. Write a pytest test file `tests/test_bank_account.py` that tests:
   - Creating a valid `BankAccount` succeeds.
   - Creating with `balance=-50` raises `ValueError`.
   - `deposit(100)` increments `balance` by 100.

(Solutions: ../solutions/week-03/day-1.md)

## Self-check

- [ ] I can explain the difference between a class and an instance.
- [ ] I can write `__init__` with typed parameters and default values.
- [ ] I know where instance attributes are stored (`__dict__`).
- [ ] I can add validation in `__init__` that raises meaningful errors.
- [ ] I understand why `self` must be the first parameter.

## Supplementary links

- [Python Data Model — Objects, values, types](https://docs.python.org/3/reference/datamodel.html)
- [Real Python — OOP in Python 3](https://realpython.com/python3-object-oriented-programming/)
