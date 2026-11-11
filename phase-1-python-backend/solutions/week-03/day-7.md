# Solutions — Week 03 · Day 7

## Task 1 — `LoggingMixin`

```python
import logging
logger = logging.getLogger(__name__)

class LoggingMixin:
    def _log_transaction(self, type_: str, amount: float, balance_after: float) -> None:
        logger.info(
            "%-12s | owner=%-15s | amount=%8.2f | balance=%10.2f",
            type_.upper(),
            getattr(self, "owner_name", "unknown"),
            amount,
            balance_after,
        )


class SavingsAccount(LoggingMixin, Account):
    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Deposit must be positive")
        self._balance += amount
        self._record(Transaction(amount=amount, type="deposit"))
        self._log_transaction("deposit", amount, self._balance)
```

MRO: `(SavingsAccount, LoggingMixin, Account, object)` — `LoggingMixin` comes before `Account` so its methods are found first if they share names.

## Task 2 — `AuditMixin`

```python
from datetime import datetime

class AuditMixin:
    _audit_log: list[str]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        cls._audit_log = []

    def deposit(self, amount: float) -> None:
        self._audit_log.append(
            f"{datetime.utcnow().isoformat()} {self.__class__.__name__}.deposit({amount})"
        )
        super().deposit(amount)  # type: ignore[misc]

    def withdraw(self, amount: float) -> None:
        self._audit_log.append(
            f"{datetime.utcnow().isoformat()} {self.__class__.__name__}.withdraw({amount})"
        )
        super().withdraw(amount)  # type: ignore[misc]


class CurrentAccount(AuditMixin, Account):
    ...  # existing implementation


def test_audit_log_populated() -> None:
    acc = CurrentAccount("Bob", 500.0)
    acc.withdraw(100.0)
    assert len(CurrentAccount._audit_log) == 1
    assert "withdraw" in CurrentAccount._audit_log[0]
```

## Task 3 — `PremiumAccount` MRO

```python
class PremiumAccount(AuditMixin, SavingsAccount):
    RATE = 0.05

print(PremiumAccount.__mro__)
# (PremiumAccount, AuditMixin, SavingsAccount, LoggingMixin, Account, object)
```

When `acc.deposit(100)` is called:
1. `AuditMixin.deposit` runs → records audit entry → calls `super().deposit(100)`
2. `SavingsAccount.deposit` runs → updates balance → calls `self._log_transaction(...)` from LoggingMixin
3. `Account` base methods are inherited

```python
def test_premium_deposit_audit_and_interest() -> None:
    acc = PremiumAccount("Dave", 1000.0)
    PremiumAccount._audit_log.clear()
    acc.deposit(200.0)
    assert acc.balance == pytest.approx(1200.0)
    assert any("deposit" in entry for entry in PremiumAccount._audit_log)
```

## Task 4 — `Reportable` Protocol + CSV export

```python
import csv
from pathlib import Path
from typing import Protocol, runtime_checkable
from banking.models import Transaction

@runtime_checkable
class Reportable(Protocol):
    def generate_statement(self) -> str: ...
    def get_history(self) -> list[Transaction]: ...


def export_to_csv(account: Reportable, path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "type", "amount", "description"])
        for txn in account.get_history():
            writer.writerow([txn.timestamp.isoformat(), txn.type, txn.amount, txn.description])


# Test with tmp_path fixture
def test_export_csv(tmp_path: Path, savings_account: SavingsAccount) -> None:
    savings_account.deposit(100.0)
    out = tmp_path / "statement.csv"
    export_to_csv(savings_account, out)
    lines = out.read_text().splitlines()
    assert lines[0] == "timestamp,type,amount,description"
    assert len(lines) >= 2  # header + at least one row
```

## Assessment prep answers

1. `__new__` allocates the object; `__init__` initialises it. Override `__new__` for singletons or immutable types (`str`, `tuple` subclasses) where you need to return a different instance or the same cached one.

2. Class attributes are shared. `self.class_var.append(x)` mutates the *shared* list (no new instance attribute is created). Fix: use `field(default_factory=list)` or initialise in `__init__`.

3. `A.x == 1` (unchanged). `a.x == 2` (instance attribute shadows class attribute). `a.__dict__ == {'x': 2}`.

4. `@classmethod` use case: factory method. `@staticmethod` use case: pure utility with no instance/class state needed.

5. `__attr` → `_ClassName__attr` at compile time. Not true access control — still reachable with `_ClassName__attr`. Purpose: avoid accidental name collisions in subclasses.

6. LSP: subtypes must be substitutable for the base type. `CurrentAccount.withdraw` relaxes the precondition (no longer raises on negative balance), which is actually LSP-compliant — postconditions must not be strengthened, but preconditions can be weakened. Whether it's a violation depends on whether the caller's expected contract includes "never negative balance".

7. ABCs require explicit inheritance + enforce via `TypeError` at runtime. Protocols are structural (duck typing); a class satisfies one just by having the right methods. Use ABCs for your own hierarchies; Protocols for function parameters.

8. `@total_ordering` requires `__eq__` + one of `__lt__`, `__le__`, `__gt__`, `__ge__`. It synthesises the rest.

9. `frozen=True` makes all fields immutable and auto-generates `__hash__`. Without `frozen=True`, `@dataclass` with `eq=True` sets `__hash__ = None` (unhashable by default).

10. MRO for D(B, C), B(A), C(A): `D → B → C → A → object`. C3 ensures each class appears once, left-to-right, with parents after all children that reference them.
