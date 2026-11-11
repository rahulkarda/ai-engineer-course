# Solutions — Week 03 · Day 6

This solution provides complete, working implementations for the Day 6 Banking Domain Model spec.

## `banking/exceptions.py`

```python
class InsufficientFundsError(ValueError):
    """Raised when a withdrawal would breach balance or overdraft constraints."""

class DuplicateCustomerError(ValueError):
    """Raised when registering a customer who is already registered."""

class AccountNotFoundError(KeyError):
    """Raised when looking up an account_id that does not exist."""
```

## `banking/models.py`

```python
from __future__ import annotations
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from banking.exceptions import AccountNotFoundError, InsufficientFundsError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Transaction:
    amount: float
    type: Literal["deposit", "withdrawal", "interest", "fee"]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    description: str = ""

    def __post_init__(self) -> None:
        if self.amount <= 0:
            raise ValueError(f"Transaction amount must be positive, got {self.amount}")
        if self.type not in ("deposit", "withdrawal", "interest", "fee"):
            raise ValueError(f"Invalid type: {self.type}")


class Account(ABC):
    def __init__(self, owner_name: str, initial_balance: float = 0.0) -> None:
        if initial_balance < 0:
            raise ValueError("Initial balance cannot be negative")
        self.owner_name = owner_name
        self._account_id: str = str(uuid.uuid4())[:8]
        self._balance: float = 0.0
        self._history: list[Transaction] = []
        if initial_balance > 0:
            self._record(Transaction(amount=initial_balance, type="deposit", description="initial"))
            self._balance = initial_balance

    @property
    def balance(self) -> float:
        return self._balance

    @property
    def account_id(self) -> str:
        return self._account_id

    def _record(self, txn: Transaction) -> None:
        self._history.append(txn)

    @abstractmethod
    def deposit(self, amount: float) -> None: ...

    @abstractmethod
    def withdraw(self, amount: float) -> None: ...

    @abstractmethod
    def apply_monthly_update(self) -> None: ...

    def get_history(self) -> list[Transaction]:
        return list(self._history)

    def get_statement(self) -> str:
        lines = [f"Account [{self._account_id}] — {self.owner_name}"]
        for t in self._history:
            sign = "+" if t.type in ("deposit", "interest") else "-"
            lines.append(f"  {t.timestamp.isoformat()[:19]}  {t.type:>12}  {sign}${t.amount:>10,.2f}")
        lines.append(f"  {'Balance':>38}   ${self._balance:>10,.2f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(owner={self.owner_name!r}, balance={self._balance:.2f}, id={self._account_id})"

    def __str__(self) -> str:
        return f"[{self._account_id}] {self.__class__.__name__:<20} ${self._balance:>10,.2f}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Account):
            return NotImplemented
        return self._account_id == other._account_id

    def __len__(self) -> int:
        return len(self._history)


class SavingsAccount(Account):
    def __init__(
        self,
        owner_name: str,
        initial_balance: float = 0.0,
        interest_rate: float = 0.02,
        minimum_balance: float = 100.0,
    ) -> None:
        if not (0 < interest_rate < 1):
            raise ValueError(f"Interest rate must be between 0 and 1, got {interest_rate}")
        self._interest_rate = interest_rate
        self.minimum_balance = minimum_balance
        super().__init__(owner_name, initial_balance)

    @property
    def interest_rate(self) -> float:
        return self._interest_rate

    @interest_rate.setter
    def interest_rate(self, value: float) -> None:
        if not (0 < value < 1):
            raise ValueError(f"Interest rate must be between 0 and 1, got {value}")
        self._interest_rate = value

    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Deposit must be positive")
        self._balance += amount
        self._record(Transaction(amount=amount, type="deposit"))
        logger.info("%s deposited %.2f", self.owner_name, amount)

    def withdraw(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Withdrawal must be positive")
        if self._balance - amount < self.minimum_balance:
            raise InsufficientFundsError(
                f"Withdrawal of {amount:.2f} would breach minimum balance {self.minimum_balance:.2f}"
            )
        self._balance -= amount
        self._record(Transaction(amount=amount, type="withdrawal"))

    def apply_monthly_update(self) -> None:
        interest = self._balance * self._interest_rate
        self._balance += interest
        self._record(Transaction(amount=interest, type="interest", description="monthly interest"))
        logger.info("%s interest: %.4f", self.owner_name, interest)


class CurrentAccount(Account):
    def __init__(
        self,
        owner_name: str,
        initial_balance: float = 0.0,
        overdraft_limit: float = 500.0,
        monthly_fee: float = 5.0,
    ) -> None:
        self._overdraft_limit = 0.0  # set before super().__init__ to avoid property guard issues
        super().__init__(owner_name, initial_balance)
        self.overdraft_limit = overdraft_limit  # use setter for validation
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
            raise ValueError("Deposit must be positive")
        self._balance += amount
        self._record(Transaction(amount=amount, type="deposit"))

    def withdraw(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Withdrawal must be positive")
        if amount > self._balance + self._overdraft_limit:
            raise InsufficientFundsError(
                f"Withdrawal of {amount:.2f} exceeds available funds "
                f"(balance {self._balance:.2f} + overdraft {self._overdraft_limit:.2f})"
            )
        self._balance -= amount
        self._record(Transaction(amount=amount, type="withdrawal"))

    def apply_monthly_update(self) -> None:
        self._balance -= self.monthly_fee
        self._record(Transaction(amount=self.monthly_fee, type="fee", description="monthly fee"))
```

## `banking/customer.py`

```python
from __future__ import annotations
import uuid
from dataclasses import dataclass, field

from banking.exceptions import AccountNotFoundError, DuplicateAccountError
from banking.models import Account


@dataclass
class Customer:
    name: str
    email: str
    customer_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __post_init__(self) -> None:
        if "@" not in self.email:
            raise ValueError(f"Invalid email address: {self.email!r}")
        self._accounts: list[Account] = []

    def add_account(self, account: Account) -> None:
        if any(a.account_id == account.account_id for a in self._accounts):
            raise DuplicateAccountError(f"Account {account.account_id} already exists")
        self._accounts.append(account)

    def remove_account(self, account_id: str) -> None:
        for i, acc in enumerate(self._accounts):
            if acc.account_id == account_id:
                self._accounts.pop(i)
                return
        raise AccountNotFoundError(f"Account {account_id!r} not found")

    def get_account(self, account_id: str) -> Account | None:
        return next((a for a in self._accounts if a.account_id == account_id), None)

    def total_balance(self) -> float:
        return sum(a.balance for a in self._accounts)

    def all_accounts(self) -> list[Account]:
        return list(self._accounts)

    def __repr__(self) -> str:
        return f"Customer(name={self.name!r}, accounts={len(self._accounts)})"

    def __len__(self) -> int:
        return len(self._accounts)

    def __iter__(self):
        return iter(self._accounts)
```

## `banking/bank.py`

```python
from __future__ import annotations
import logging

from banking.customer import Customer
from banking.exceptions import CustomerNotFoundError, DuplicateCustomerError
from banking.models import Account

logger = logging.getLogger(__name__)


class Bank:
    def __init__(self, name: str) -> None:
        self.name = name
        self._customers: dict[str, Customer] = {}

    def register_customer(self, customer: Customer) -> None:
        if customer.customer_id in self._customers:
            raise DuplicateCustomerError(f"Customer {customer.customer_id} already registered")
        self._customers[customer.customer_id] = customer
        logger.info("Registered customer: %s", customer.name)

    def get_customer(self, customer_id: str) -> Customer:
        try:
            return self._customers[customer_id]
        except KeyError:
            raise CustomerNotFoundError(f"Customer {customer_id!r} not found")

    def open_account(self, customer_id: str, account: Account) -> None:
        self.get_customer(customer_id).add_account(account)

    def close_account(self, customer_id: str, account_id: str) -> None:
        self.get_customer(customer_id).remove_account(account_id)

    def total_assets(self) -> float:
        return sum(
            acc.balance
            for customer in self._customers.values()
            for acc in customer
        )

    def monthly_update(self) -> None:
        for customer in self._customers.values():
            for acc in customer:
                acc.apply_monthly_update()

    def generate_bank_report(self) -> str:
        total = self.total_assets()
        n_customers = len(self._customers)
        n_accounts = sum(len(c) for c in self._customers.values())
        width = 48
        lines = [
            f"╔══ {self.name} {'═' * (width - len(self.name) - 4)}╗",
            f"║ Customers: {n_customers}   Accounts: {n_accounts}   Assets: ${total:,.2f}",
            f"╚{'═' * width}╝",
            "",
        ]
        for customer in self._customers.values():
            lines.append(
                f"{customer.name} ({customer.email}) [cust: {customer.customer_id}]"
            )
            for acc in customer:
                lines.append(f"  {acc}")
            lines.append(f"  Subtotal: ${customer.total_balance():,.2f}")
            lines.append("")
        return "\n".join(lines)
```

## Tests (`tests/test_banking.py`) — key examples

```python
import pytest
from banking.exceptions import InsufficientFundsError
from banking.models import SavingsAccount, CurrentAccount
from banking.customer import Customer
from banking.bank import Bank


@pytest.fixture()
def savings():
    return SavingsAccount("Alice", initial_balance=1000.0)

@pytest.fixture()
def current():
    return CurrentAccount("Bob", initial_balance=500.0, overdraft_limit=200.0)


def test_savings_deposit_increases_balance(savings: SavingsAccount) -> None:
    savings.deposit(200.0)
    assert savings.balance == pytest.approx(1200.0)

def test_savings_withdraw_respects_minimum_balance(savings: SavingsAccount) -> None:
    with pytest.raises(InsufficientFundsError):
        savings.withdraw(950.0)  # would leave 50 < 100 minimum

def test_savings_monthly_interest_applied(savings: SavingsAccount) -> None:
    savings.apply_monthly_update()
    assert savings.balance == pytest.approx(1020.0)

def test_current_overdraft_allowed_within_limit(current: CurrentAccount) -> None:
    current.withdraw(600.0)  # 500 balance + 200 overdraft = 700 max
    assert current.balance == pytest.approx(-100.0)

def test_current_overdraft_beyond_limit_raises(current: CurrentAccount) -> None:
    with pytest.raises(InsufficientFundsError):
        current.withdraw(800.0)  # exceeds 500 + 200

def test_current_monthly_fee_deducted(current: CurrentAccount) -> None:
    current.apply_monthly_update()
    assert current.balance == pytest.approx(495.0)

def test_customer_total_balance() -> None:
    c = Customer("Alice", "alice@example.com")
    c.add_account(SavingsAccount("Alice", 1000.0))
    c.add_account(CurrentAccount("Alice", 500.0))
    assert c.total_balance() == pytest.approx(1500.0)

def test_bank_total_assets() -> None:
    bank = Bank("Test")
    c = Customer("Alice", "alice@example.com")
    c.add_account(SavingsAccount("Alice", 1000.0))
    bank.register_customer(c)
    assert bank.total_assets() == pytest.approx(1000.0)

def test_transaction_history_records_correctly(savings: SavingsAccount) -> None:
    savings.deposit(100.0)
    history = savings.get_history()
    txn = next(t for t in history if t.type == "deposit" and t.amount == 100.0)
    assert txn is not None

def test_duplicate_customer_raises() -> None:
    bank = Bank("Test")
    c = Customer("Alice", "alice@example.com")
    bank.register_customer(c)
    from banking.exceptions import DuplicateCustomerError
    with pytest.raises(DuplicateCustomerError):
        bank.register_customer(c)

def test_invalid_email_raises() -> None:
    with pytest.raises(ValueError):
        Customer("Alice", "notanemail")
```
