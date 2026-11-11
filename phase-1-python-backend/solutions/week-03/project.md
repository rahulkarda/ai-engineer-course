# Solutions — Week 03 Project — Banking Application

Full reference implementation. All paths relative to `banking_app/`.

## `banking/exceptions.py`

```python
class BankingError(Exception):
    """Base exception for all banking domain errors."""


class InsufficientFundsError(BankingError, ValueError):
    """Raised when a withdrawal would breach balance or overdraft constraints."""


class DuplicateCustomerError(BankingError, ValueError):
    """Raised when registering a customer who is already registered."""


class DuplicateAccountError(BankingError, ValueError):
    """Raised when adding an account that already exists on a customer."""


class AccountNotFoundError(BankingError, KeyError):
    """Raised when looking up an account_id that does not exist."""


class CustomerNotFoundError(BankingError, KeyError):
    """Raised when looking up a customer_id that does not exist."""


class InvalidAmountError(BankingError, ValueError):
    """Raised when a transaction amount is not strictly positive."""
```

## `banking/models.py`

```python
from __future__ import annotations
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator, Literal

from banking.exceptions import AccountNotFoundError, InsufficientFundsError, InvalidAmountError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Transaction:
    amount: float
    type: Literal["deposit", "withdrawal", "interest", "fee", "transfer"]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    description: str = ""

    def __post_init__(self) -> None:
        if self.amount <= 0:
            raise InvalidAmountError(f"Transaction amount must be positive, got {self.amount}")
        valid_types = ("deposit", "withdrawal", "interest", "fee", "transfer")
        if self.type not in valid_types:
            raise ValueError(f"Invalid transaction type: {self.type!r}")


class Account(ABC):
    def __init__(self, owner_name: str, initial_balance: float = 0.0) -> None:
        if initial_balance < 0:
            raise ValueError("Initial balance cannot be negative")
        self.owner_name = owner_name
        self._account_id: str = str(uuid.uuid4())[:8]
        self._balance: float = 0.0
        self._history: list[Transaction] = []
        if initial_balance > 0:
            txn = Transaction(amount=initial_balance, type="deposit", description="opening balance")
            self._history.append(txn)
            self._balance = initial_balance

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

    def get_statement(self) -> str:
        lines = [f"Account [{self._account_id}] — {self.owner_name}"]
        lines.append(f"  {'Date':^19}  {'Type':>12}  {'Amount':>12}")
        lines.append(f"  {'-'*19}  {'-'*12}  {'-'*12}")
        for t in self._history:
            sign = "+" if t.type in ("deposit", "interest") else "-"
            lines.append(
                f"  {t.timestamp.isoformat()[:19]}  {t.type:>12}  {sign}${t.amount:>10,.2f}"
            )
        lines.append(f"  {'':19}  {'Balance':>12}  ${self._balance:>10,.2f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"owner={self.owner_name!r}, "
            f"balance={self._balance:.2f}, "
            f"id={self._account_id!r})"
        )

    def __str__(self) -> str:
        return f"[{self._account_id}] {self.__class__.__name__:<22} ${self._balance:>10,.2f}  ({len(self._history)} txns)"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Account):
            return NotImplemented
        return self._account_id == other._account_id

    def __hash__(self) -> int:
        return hash(self._account_id)

    def __len__(self) -> int:
        return len(self._history)

    def __iter__(self) -> Iterator[Transaction]:
        return iter(self._history)


class SavingsAccount(Account):
    def __init__(
        self,
        owner_name: str,
        initial_balance: float = 0.0,
        interest_rate: float = 0.02,
        minimum_balance: float = 100.0,
    ) -> None:
        if not (0 < interest_rate < 1):
            raise ValueError(f"Interest rate must be (0, 1), got {interest_rate}")
        self._interest_rate = interest_rate
        self.minimum_balance = minimum_balance
        super().__init__(owner_name, initial_balance)

    @property
    def interest_rate(self) -> float:
        return self._interest_rate

    @interest_rate.setter
    def interest_rate(self, value: float) -> None:
        if not (0 < value < 1):
            raise ValueError(f"Interest rate must be (0, 1), got {value}")
        self._interest_rate = value

    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise InvalidAmountError("Deposit must be positive")
        self._balance += amount
        self._history.append(Transaction(amount=amount, type="deposit"))
        logger.info("Savings deposit: %s %.2f (balance: %.2f)", self.owner_name, amount, self._balance)

    def withdraw(self, amount: float) -> None:
        if amount <= 0:
            raise InvalidAmountError("Withdrawal must be positive")
        if self._balance - amount < self.minimum_balance:
            raise InsufficientFundsError(
                f"Withdrawal of {amount:.2f} would leave {self._balance - amount:.2f}, "
                f"below minimum {self.minimum_balance:.2f}"
            )
        self._balance -= amount
        self._history.append(Transaction(amount=amount, type="withdrawal"))

    def apply_monthly_update(self) -> None:
        interest = self._balance * self._interest_rate
        self._balance += interest
        self._history.append(
            Transaction(amount=interest, type="interest", description="monthly interest")
        )
        logger.info("Interest applied to %s: %.4f", self.owner_name, interest)


class CurrentAccount(Account):
    def __init__(
        self,
        owner_name: str,
        initial_balance: float = 0.0,
        overdraft_limit: float = 500.0,
        monthly_fee: float = 5.0,
    ) -> None:
        self._overdraft_limit = 0.0
        super().__init__(owner_name, initial_balance)
        self.overdraft_limit = overdraft_limit
        self.monthly_fee = monthly_fee

    @property
    def overdraft_limit(self) -> float:
        return self._overdraft_limit

    @overdraft_limit.setter
    def overdraft_limit(self, value: float) -> None:
        if not (0 <= value <= 10_000):
            raise ValueError(f"Overdraft limit must be [0, 10000], got {value}")
        self._overdraft_limit = value

    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise InvalidAmountError("Deposit must be positive")
        self._balance += amount
        self._history.append(Transaction(amount=amount, type="deposit"))

    def withdraw(self, amount: float) -> None:
        if amount <= 0:
            raise InvalidAmountError("Withdrawal must be positive")
        available = self._balance + self._overdraft_limit
        if amount > available:
            raise InsufficientFundsError(
                f"Withdrawal of {amount:.2f} exceeds available {available:.2f}"
            )
        self._balance -= amount
        self._history.append(Transaction(amount=amount, type="withdrawal"))

    def apply_monthly_update(self) -> None:
        self._balance -= self.monthly_fee
        self._history.append(
            Transaction(amount=self.monthly_fee, type="fee", description="monthly fee")
        )
```

## `banking/customer.py`

```python
from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from typing import Iterator

from banking.exceptions import AccountNotFoundError, DuplicateAccountError
from banking.models import Account


@dataclass
class Customer:
    name: str
    email: str
    customer_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __post_init__(self) -> None:
        if "@" not in self.email or self.email.startswith("@") or self.email.endswith("@"):
            raise ValueError(f"Invalid email address: {self.email!r}")
        self._accounts: list[Account] = []

    def add_account(self, account: Account) -> None:
        if any(a.account_id == account.account_id for a in self._accounts):
            raise DuplicateAccountError(f"Account {account.account_id!r} already exists")
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
        return f"Customer(name={self.name!r}, email={self.email!r}, accounts={len(self._accounts)})"

    def __len__(self) -> int:
        return len(self._accounts)

    def __iter__(self) -> Iterator[Account]:
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
            raise DuplicateCustomerError(
                f"Customer {customer.customer_id!r} already registered"
            )
        self._customers[customer.customer_id] = customer
        logger.info("Registered: %s [%s]", customer.name, customer.customer_id)

    def unregister_customer(self, customer_id: str) -> None:
        if customer_id not in self._customers:
            raise CustomerNotFoundError(f"Customer {customer_id!r} not found")
        del self._customers[customer_id]

    def get_customer(self, customer_id: str) -> Customer:
        try:
            return self._customers[customer_id]
        except KeyError:
            raise CustomerNotFoundError(f"Customer {customer_id!r} not found")

    def open_account(self, customer_id: str, account: Account) -> None:
        self.get_customer(customer_id).add_account(account)

    def close_account(self, customer_id: str, account_id: str) -> None:
        self.get_customer(customer_id).remove_account(account_id)

    def transfer(self, from_account: Account, to_account: Account, amount: float) -> None:
        from_account.withdraw(amount)
        try:
            to_account.deposit(amount)
        except Exception:
            from_account.deposit(amount)  # rollback
            raise

    def total_assets(self) -> float:
        return sum(acc.balance for customer in self._customers.values() for acc in customer)

    def monthly_update(self) -> None:
        for customer in self._customers.values():
            for acc in customer:
                acc.apply_monthly_update()

    def customer_count(self) -> int:
        return len(self._customers)

    def account_count(self) -> int:
        return sum(len(c) for c in self._customers.values())

    def generate_bank_report(self) -> str:
        total = self.total_assets()
        n_c = self.customer_count()
        n_a = self.account_count()
        width = 46
        lines = [
            f"╔══ {self.name} {'═' * max(0, width - len(self.name) - 4)}╗",
            f"║ Customers: {n_c}   Accounts: {n_a}   Assets: ${total:,.2f}",
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

## `banking/reports.py`

```python
from __future__ import annotations
import csv
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from banking.bank import Bank
    from banking.models import Account


class ReportGenerator:
    @staticmethod
    def text_report(bank: "Bank") -> str:
        return bank.generate_bank_report()

    @staticmethod
    def export_csv(accounts: list["Account"], path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "type", "amount", "description"])
            for account in accounts:
                for txn in account.get_history():
                    writer.writerow([
                        txn.timestamp.isoformat(),
                        txn.type,
                        f"{txn.amount:.2f}",
                        txn.description,
                    ])
```

## Key tests overview

See `day-6.md` solution for the 12 required model tests. Additional project-level tests:

```python
# tests/test_bank.py
def test_transfer_success(savings_account, current_account) -> None:
    bank = Bank("Test")
    bank.transfer(savings_account, current_account, 100.0)
    assert savings_account.balance == pytest.approx(900.0)
    assert current_account.balance == pytest.approx(600.0)

def test_transfer_rollback_on_deposit_failure(savings_account, monkeypatch) -> None:
    bad_account = SavingsAccount("Bad", 0.0)
    original = bad_account.deposit
    def failing_deposit(amount: float) -> None:
        raise RuntimeError("deposit exploded")
    monkeypatch.setattr(bad_account, "deposit", failing_deposit)
    with pytest.raises(RuntimeError):
        bank = Bank("Test")
        bank.transfer(savings_account, bad_account, 100.0)
    assert savings_account.balance == pytest.approx(1000.0)  # rolled back

# tests/test_reports.py
def test_export_csv_writes_correct_headers(tmp_path, savings_account) -> None:
    savings_account.deposit(100.0)
    out = tmp_path / "report.csv"
    ReportGenerator.export_csv([savings_account], out)
    lines = out.read_text().splitlines()
    assert lines[0] == "timestamp,type,amount,description"

def test_export_csv_row_count(tmp_path, savings_account) -> None:
    savings_account.deposit(50.0)
    savings_account.deposit(75.0)
    out = tmp_path / "report.csv"
    ReportGenerator.export_csv([savings_account], out)
    lines = out.read_text().splitlines()
    assert len(lines) >= 3  # header + opening balance + 2 deposits
```
