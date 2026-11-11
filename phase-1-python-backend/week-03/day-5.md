# Phase 1 · Week 03 · Day 5 — Abstract Base Classes, Protocols & SOLID Principles

> Objective: Enforce interface contracts with `ABC` and `Protocol`; apply SOLID principles to evaluate and improve class designs.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

## Concept

### Abstract Base Classes (ABC)

`abc.ABC` + `@abstractmethod` defines a contract: subclasses **must** implement decorated methods or instantiation raises `TypeError`. This makes the interface explicit and enforced at runtime.

```python
from abc import ABC, abstractmethod

class PaymentProcessor(ABC):
    @abstractmethod
    def process(self, amount: float) -> bool: ...
```

### Protocols (structural subtyping)

`typing.Protocol` defines duck-typing contracts checked by the **type checker** (mypy/pyright), not at runtime. A class satisfies a Protocol if it has the right methods — no inheritance needed.

```python
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> str: ...
```

**ABC vs Protocol**:
- ABC = explicit inheritance required; runtime enforcement.
- Protocol = structural — any class with matching methods qualifies; no inheritance.
- Use ABCs for internal hierarchies you control. Use Protocols for function parameters that accept "anything with a `process` method".

### SOLID principles

| Letter | Principle | One line |
|---|---|---|
| S | Single Responsibility | One class = one reason to change |
| O | Open/Closed | Open for extension, closed for modification |
| L | Liskov Substitution | Subtypes must be usable wherever the base type is expected |
| I | Interface Segregation | Prefer many small interfaces over one large one |
| D | Dependency Inversion | Depend on abstractions, not concretions |

## How it works internally

`ABCMeta` is the metaclass for `ABC`. When a class with `ABCMeta` is instantiated, the metaclass checks `__abstractmethods__` — a frozenset of unimplemented abstract methods. If non-empty, it raises `TypeError`.

`Protocol` is handled entirely by static type checkers. At runtime, `isinstance(obj, MyProtocol)` only works if `MyProtocol` is decorated with `@runtime_checkable`. Without that decorator, Protocols exist purely for type-checker satisfaction.

## Worked example

```python
# interfaces.py
from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ── Abstract Base Class ──────────────────────────────────────────────────────

class BaseAccount(ABC):
    """Contract every account type must fulfil."""

    def __init__(self, owner: str, balance: float = 0.0) -> None:
        if balance < 0:
            raise ValueError("Balance cannot be negative")
        self.owner = owner
        self._balance = balance

    @abstractmethod
    def deposit(self, amount: float) -> None: ...

    @abstractmethod
    def withdraw(self, amount: float) -> None: ...

    @abstractmethod
    def apply_monthly_update(self) -> None: ...

    # Concrete method available to all subclasses
    def get_balance(self) -> float:
        return self._balance

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(owner={self.owner!r}, balance={self._balance:.2f})"


# ── Protocol (structural duck-typing) ────────────────────────────────────────

@runtime_checkable
class Statementable(Protocol):
    """Anything that can generate a text statement."""
    def generate_statement(self) -> str: ...


# ── Concrete implementations ─────────────────────────────────────────────────

@dataclass
class Transaction:
    amount: float
    type: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class SavingsAccount(BaseAccount):
    RATE = 0.02

    def __init__(self, owner: str, balance: float = 0.0) -> None:
        super().__init__(owner, balance)
        self._history: list[Transaction] = []

    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Amount must be positive")
        self._balance += amount
        self._history.append(Transaction(amount, "deposit"))

    def withdraw(self, amount: float) -> None:
        if amount <= 0 or amount > self._balance:
            raise ValueError("Invalid withdrawal")
        self._balance -= amount
        self._history.append(Transaction(amount, "withdrawal"))

    def apply_monthly_update(self) -> None:
        interest = self._balance * self.RATE
        self._balance += interest
        logger.info("Interest applied: %.2f", interest)

    def generate_statement(self) -> str:
        """Satisfies Statementable Protocol."""
        lines = [f"Statement for {self.owner}"]
        for t in self._history:
            lines.append(f"  {t.type:>12}  ${t.amount:>10,.2f}  {t.timestamp}")
        lines.append(f"  {'Balance':>12}  ${self._balance:>10,.2f}")
        return "\n".join(lines)


# ── SOLID: Dependency Inversion ──────────────────────────────────────────────

class NotificationService(Protocol):
    """D principle: Bank depends on this abstraction, not a concrete emailer."""
    def notify(self, owner: str, message: str) -> None: ...


class EmailNotifier:
    """Concrete implementation — swappable."""
    def notify(self, owner: str, message: str) -> None:
        logger.info("EMAIL → %s: %s", owner, message)


class SmsNotifier:
    """Another implementation — same Protocol."""
    def notify(self, owner: str, message: str) -> None:
        logger.info("SMS → %s: %s", owner, message)


class Bank:
    """
    S — Bank manages accounts, delegates notifications (not its job).
    O — Adding a new account type requires no change to Bank.
    D — Bank depends on BaseAccount and NotificationService abstractions.
    """

    def __init__(self, notifier: NotificationService) -> None:
        self._accounts: dict[str, BaseAccount] = {}
        self._notifier = notifier

    def open_account(self, account: BaseAccount) -> None:
        self._accounts[account.owner] = account
        self._notifier.notify(account.owner, "Account opened")

    def process_deposit(self, owner: str, amount: float) -> None:
        self._accounts[owner].deposit(amount)
        self._notifier.notify(owner, f"Deposited ${amount:.2f}")

    def print_statement(self, owner: str) -> None:
        acc = self._accounts[owner]
        # I principle — only call generate_statement if the account supports it
        if isinstance(acc, Statementable):
            print(acc.generate_statement())
        else:
            print(f"No statement available for {owner}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    bank = Bank(notifier=EmailNotifier())   # inject dependency
    acc = SavingsAccount("Alice", 500.0)
    bank.open_account(acc)
    bank.process_deposit("Alice", 300.0)
    bank.print_statement("Alice")

    # Swap to SMS with zero Bank changes — Open/Closed in action
    bank2 = Bank(notifier=SmsNotifier())
    acc2 = SavingsAccount("Bob", 200.0)
    bank2.open_account(acc2)


if __name__ == "__main__":
    main()
```

## Exercises

1. Create a `CurrentAccount(BaseAccount)` that implements all abstract methods (overdraft allowed). Verify `Bank.open_account` works with it unchanged.

2. Try instantiating `BaseAccount("Alice")` directly. What error do you get? Why?

3. Add a `@runtime_checkable` Protocol `Interestable` with `apply_monthly_update(self) -> None`. Write a function `apply_all(accounts: list[BaseAccount]) -> None` that checks `isinstance(acc, Interestable)` before calling the method.

4. Identify an SRP violation: if `SavingsAccount.generate_statement` also sent an email, which principle would that break? Refactor it to satisfy SRP.

5. Write a pytest test that uses a `MockNotifier` (a simple class with a `notify` method that records calls) injected into `Bank`, then verifies `notify` was called after `open_account`.

(Solutions: ../solutions/week-03/day-5.md)

## Self-check

- [ ] I can write an ABC and explain `TypeError` when a concrete subclass is missing an abstract method.
- [ ] I know the difference between ABC (runtime) and Protocol (static typing) enforcement.
- [ ] I can state each SOLID principle in one sentence and give a banking example.
- [ ] I can implement Dependency Inversion by accepting a Protocol parameter instead of a concrete class.
- [ ] I understand `@runtime_checkable` and its limitations.

## Supplementary links

- [Python docs — abc module](https://docs.python.org/3/library/abc.html)
- [Python docs — typing.Protocol](https://docs.python.org/3/library/typing.html#typing.Protocol)
- [Real Python — ABCs](https://realpython.com/python-interface/)
- [SOLID principles — Rian van der Merwe](https://www.digitalocean.com/community/conceptual-articles/s-o-l-i-d-the-first-five-principles-of-object-oriented-design)
