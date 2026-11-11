# Solutions — Week 03 · Day 3

## Exercise 1 — `PremiumSavingsAccount`

```python
class PremiumSavingsAccount(SavingsAccount):
    INTEREST_RATE: ClassVar[float] = 0.05

# Test
def test_premium_interest_rate() -> None:
    acc = PremiumSavingsAccount("Dave", 1000.0)
    acc.apply_monthly_update()
    assert acc.get_balance() == pytest.approx(1050.0)
```

## Exercise 2 — `CurrentAccount.get_balance` & `is_overdrawn`

```python
class CurrentAccount(Account):
    ...
    def get_balance(self) -> float:
        """Returns negative value when overdrawn."""
        return self._balance  # parent already stores it; no override needed in base

    def is_overdrawn(self) -> bool:
        return self._balance < 0
```

## Exercise 3 — `Transaction` and history

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Transaction:
    amount: float
    type: str  # "deposit" | "withdrawal"
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


class Account:
    def __init__(self, owner: str, balance: float = 0.0) -> None:
        ...
        self._history: list[Transaction] = []

    def deposit(self, amount: float) -> None:
        ...
        self._history.append(Transaction(amount=amount, type="deposit"))

    def withdraw(self, amount: float) -> None:
        ...
        self._history.append(Transaction(amount=amount, type="withdrawal"))

    def get_history(self) -> list[Transaction]:
        return list(self._history)  # defensive copy
```

## Exercise 4 — `process_accounts`

```python
def process_accounts(accounts: list[Account]) -> dict[str, float]:
    return {acc.owner: acc.get_balance() for acc in accounts}

# Test
def test_process_accounts() -> None:
    accounts: list[Account] = [
        SavingsAccount("Alice", 1000.0),
        CurrentAccount("Bob", 500.0),
    ]
    result = process_accounts(accounts)
    assert result == {"Alice": 1000.0, "Bob": 500.0}
```

## Exercise 5 — Inheritance vs Composition for `LoanAccount`

**Use composition.** A loan is not a type of bank account in the same hierarchy — it has repayment schedules, interest accrual on debt, and different invariants (balance is what's owed, not what's held). Embedding that in `Account`'s hierarchy via inheritance would either violate LSP (withdrawals make no sense) or require awkward overrides.

Better: `LoanAccount` has-a `Account` (or references the borrower's customer record) and owns its own repayment logic.
