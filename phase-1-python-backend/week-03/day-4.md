# Phase 1 · Week 03 · Day 4 — Dunder Methods, @property & Dataclasses

> Objective: Implement dunder methods for operator overloading and string representation; use `@property` for controlled attribute access; simplify data classes with `dataclasses`.  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

## Concept

### Dunder (magic) methods

Methods named `__like_this__` hook into Python's built-in protocols. They are called automatically by operators, built-in functions, and language constructs — you never call them directly.

| Dunder | Called by |
|---|---|
| `__repr__` | `repr(obj)`, REPL display |
| `__str__` | `str(obj)`, `print(obj)` |
| `__eq__` | `a == b` |
| `__lt__`, `__le__`, … | `a < b`, comparison operators |
| `__add__` | `a + b` |
| `__len__` | `len(obj)` |
| `__contains__` | `x in obj` |
| `__iter__` | `for x in obj` |
| `__enter__`, `__exit__` | `with obj:` context manager |

### @property

`@property` turns a method into a read-only attribute. Add a setter with `@name.setter`. This enforces validation without changing the caller's syntax.

```python
account.balance = -50  # calls the setter → raises ValueError
print(account.balance)  # calls the getter → returns float
```

### dataclasses

`@dataclass` auto-generates `__init__`, `__repr__`, and `__eq__` from annotated fields. Use it for pure data containers. Use `field(default_factory=...)` for mutable defaults.

```python
from dataclasses import dataclass, field

@dataclass
class Transaction:
    amount: float
    type: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
```

## How it works internally

`@property` uses Python's **descriptor protocol**. When Python looks up `account.balance`:
1. It finds `balance` in the class `__dict__` — it's a descriptor (has `__get__`).
2. The descriptor's `__get__` is called, which runs your getter function.

`@dataclass` is a **class decorator** that inspects `__annotations__` and injects `__init__`, `__repr__`, `__eq__` (and optionally `__hash__`, `__lt__`, etc.) at class-creation time. It's code generation at import time — no runtime overhead.

`__repr__` should return a string that, when `eval()`-ed, recreates the object (or is at least unambiguous). `__str__` is the human-friendly version. If only `__repr__` is defined, `str()` falls back to it.

## Worked example

```python
# models.py
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime
from functools import total_ordering

logger = logging.getLogger(__name__)


@dataclass
class Transaction:
    """Immutable record of a single account operation."""
    amount: float
    type: str  # "deposit" | "withdrawal"
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def __post_init__(self) -> None:
        if self.amount <= 0:
            raise ValueError("Transaction amount must be positive")
        if self.type not in ("deposit", "withdrawal"):
            raise ValueError(f"Invalid transaction type: {self.type}")


@total_ordering  # fills in __le__, __gt__, __ge__ from __eq__ + __lt__
class Account:
    """Account with property-based balance, dunder operators, and transaction history."""

    def __init__(self, owner: str, balance: float = 0.0) -> None:
        self.owner = owner
        self._balance = 0.0          # backing store for property
        self._history: list[Transaction] = []
        if balance > 0:
            self._apply_deposit(balance)  # record initial deposit

    # ── Properties ────────────────────────────────────────────────────────
    @property
    def balance(self) -> float:
        """Current balance (read-only via property)."""
        return self._balance

    # No setter — balance changes only via deposit/withdraw methods.

    # ── Business methods ──────────────────────────────────────────────────
    def deposit(self, amount: float) -> "Account":
        """Return self to allow chaining: account.deposit(100).deposit(50)."""
        self._apply_deposit(amount)
        return self

    def withdraw(self, amount: float) -> "Account":
        if amount <= 0:
            raise ValueError("Withdrawal must be positive")
        if amount > self._balance:
            raise ValueError("Insufficient funds")
        self._balance -= amount
        self._history.append(Transaction(amount=amount, type="withdrawal"))
        return self

    def _apply_deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Deposit must be positive")
        self._balance += amount
        self._history.append(Transaction(amount=amount, type="deposit"))

    def get_history(self) -> list[Transaction]:
        return list(self._history)  # defensive copy

    # ── Dunder methods ────────────────────────────────────────────────────
    def __repr__(self) -> str:
        return f"Account(owner={self.owner!r}, balance={self._balance:.2f})"

    def __str__(self) -> str:
        return f"{self.owner}'s account: ${self._balance:,.2f}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Account):
            return NotImplemented
        return self.owner == other.owner and self._balance == other._balance

    def __lt__(self, other: "Account") -> bool:
        if not isinstance(other, Account):
            return NotImplemented
        return self._balance < other._balance

    def __len__(self) -> int:
        """Number of transactions."""
        return len(self._history)

    def __contains__(self, amount: float) -> bool:
        """Check if a specific amount appears in history."""
        return any(t.amount == amount for t in self._history)

    def __add__(self, other: "Account") -> "Account":
        """Merge two accounts — combined balance under first owner."""
        merged = Account(self.owner, self._balance + other._balance)
        return merged


def main() -> None:
    a = Account("Alice", 1000.0)
    b = Account("Bob", 500.0)

    a.deposit(200.0).deposit(100.0)  # method chaining
    print(a)            # Alice's account: $1,300.00
    print(repr(a))      # Account(owner='Alice', balance=1300.00)
    print(len(a))       # 3 transactions
    print(300.0 in a)   # True (200+100 … but check: 200 is in history → True)

    print(a > b)        # True  (uses __lt__ via @total_ordering)
    merged = a + b
    print(merged)

    for txn in a.get_history():
        print(txn)


if __name__ == "__main__":
    main()
```

## Exercises

1. Add `__hash__` to `Account` — what constraint does Python enforce when you define `__eq__`? Make `Account` hashable using `(owner, balance)` as the key.

2. Implement `__iter__` on `Account` that yields `Transaction` objects. Write a for-loop that prints each transaction.

3. Convert `Transaction` to a frozen dataclass (`frozen=True`). What changes? What breaks? Fix it.

4. Add a `@property` setter for `overdraft_limit` on `CurrentAccount` (from Day 3) that validates the limit is between 0 and 10,000.

5. Write a pytest test verifying:
   - `account1 == account2` when owner and balance match.
   - `account1 < account2` when balance is lower.
   - `100.0 in account` after a deposit of 100.0.

(Solutions: ../solutions/week-03/day-4.md)

## Self-check

- [ ] I can implement `__repr__`, `__str__`, `__eq__`, and `__lt__` correctly.
- [ ] I know why `__hash__` must be considered when `__eq__` is defined.
- [ ] I can use `@property` with a getter and explain why it beats a plain attribute for validated state.
- [ ] I can write a `@dataclass` with `field(default_factory=...)` and `__post_init__`.
- [ ] I know `@total_ordering` and the minimum dunders required to use it.

## Supplementary links

- [Python docs — data model](https://docs.python.org/3/reference/datamodel.html#special-method-names)
- [Python docs — dataclasses](https://docs.python.org/3/library/dataclasses.html)
- [Real Python — @property](https://realpython.com/python-property/)
