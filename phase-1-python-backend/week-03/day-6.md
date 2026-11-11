# Phase 1 · Week 03 · Day 6 — Saturday Deep Practice: Banking Domain Model

> Objective: Build the full Banking Application domain model from scratch — `Account`, `SavingsAccount`, `CurrentAccount`, `Customer`, `Transaction`, `Bank` — applying all week concepts.  ·  Time: ~4 hr  ·  Prereqs: [[day-1]] through [[day-5]]

## Overview

Today is a 4-hour build session. You will implement a production-quality (for learning purposes) banking domain from a specification, using:

- ABCs + Protocols for contracts
- Dataclasses for value objects
- Properties for validated state
- Dunder methods for natural Python idioms
- Composition for `Customer → accounts` and `Bank → customers`
- Inheritance for `SavingsAccount`, `CurrentAccount`
- SOLID throughout

Work through the specification sequentially. Each part builds on the previous.

---

## Specification

### Domain entities

**`Transaction`** (dataclass, frozen)
- `amount: float` — must be positive
- `type: Literal["deposit", "withdrawal", "interest", "fee"]`
- `timestamp: datetime` — default now (UTC)
- `description: str` — default `""`
- `__post_init__` validates amount > 0

**`Account`** (ABC)
- `account_id: str` — auto-generated UUID prefix (first 8 chars)
- `owner_name: str`
- `_balance: float` — backing store for `balance` property
- `_history: list[Transaction]` — internal
- `balance` property (read-only)
- `deposit(amount)` — abstract
- `withdraw(amount)` — abstract
- `apply_monthly_update()` — abstract
- `get_history() -> list[Transaction]` — concrete, returns a copy
- `get_statement() -> str` — concrete: formatted multi-line string
- `__repr__`, `__str__`, `__eq__` (by `account_id`), `__len__` (transaction count)

**`SavingsAccount(Account)`**
- `interest_rate: float` (default `0.02`; must be 0 < rate < 1)
- `minimum_balance: float` (default `100.0`)
- `deposit`: add to balance, record transaction
- `withdraw`: validate minimum balance will be maintained; record transaction
- `apply_monthly_update`: apply interest; record interest transaction

**`CurrentAccount(Account)`**
- `overdraft_limit: float` property with setter (0 ≤ limit ≤ 10_000)
- `monthly_fee: float` (default `5.0`)
- `deposit`: add, record
- `withdraw`: allow to −overdraft_limit; record
- `apply_monthly_update`: deduct monthly fee; record

**`Customer`** (dataclass with `__post_init__`)
- `name: str`
- `email: str` — validate contains `@`
- `customer_id: str` — auto UUID prefix
- `_accounts: list[Account]` — private, not in dataclass fields
- `add_account(account)`, `remove_account(account_id: str)`
- `get_account(account_id: str) -> Account | None`
- `total_balance() -> float`
- `all_accounts() -> list[Account]`
- `__repr__`, `__len__` (account count)

**`Bank`**
- `name: str`
- `_customers: dict[str, Customer]` — keyed by customer_id
- `register_customer(customer: Customer)` — raises if already registered
- `get_customer(customer_id: str) -> Customer`
- `open_account(customer_id: str, account: Account)` — adds to customer
- `close_account(customer_id: str, account_id: str)` — removes from customer
- `total_assets() -> float` — sum of all account balances
- `monthly_update()` — calls `apply_monthly_update()` on every account
- `generate_bank_report() -> str` — formatted summary

### Validation rules
- Negative or zero amounts → `ValueError`
- Withdrawal below minimum balance (savings) → `InsufficientFundsError(ValueError)`
- Withdrawal beyond overdraft (current) → `InsufficientFundsError`
- Invalid email → `ValueError`
- Duplicate customer registration → `DuplicateCustomerError(ValueError)`
- Account not found → `AccountNotFoundError(KeyError)`

### Required pytest tests (write these too)
All in `tests/test_banking.py`:

- `test_savings_deposit_increases_balance`
- `test_savings_withdraw_respects_minimum_balance`
- `test_savings_monthly_interest_applied`
- `test_current_overdraft_allowed_within_limit`
- `test_current_overdraft_beyond_limit_raises`
- `test_current_monthly_fee_deducted`
- `test_customer_total_balance_sums_all_accounts`
- `test_bank_total_assets`
- `test_bank_monthly_update_calls_all_accounts`
- `test_transaction_history_records_correctly`
- `test_duplicate_customer_raises`
- `test_invalid_email_raises`

---

## Suggested build order

**Hour 1 (0:00–1:00)** — Exceptions, Transaction, Account ABC  
**Hour 2 (1:00–2:00)** — SavingsAccount, CurrentAccount, unit tests for both  
**Hour 3 (2:00–3:00)** — Customer, Bank  
**Hour 4 (3:00–4:00)** — Integration tests, `generate_bank_report`, refine statement format

---

## Worked skeleton (starter code — fill in the `...` bodies)

```python
# banking/exceptions.py
class InsufficientFundsError(ValueError): ...
class DuplicateCustomerError(ValueError): ...
class AccountNotFoundError(KeyError): ...
```

```python
# banking/models.py
from __future__ import annotations
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from banking.exceptions import (
    AccountNotFoundError,
    DuplicateCustomerError,
    InsufficientFundsError,
)


@dataclass(frozen=True)
class Transaction:
    amount: float
    type: Literal["deposit", "withdrawal", "interest", "fee"]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    description: str = ""

    def __post_init__(self) -> None:
        ...  # validate amount > 0


class Account(ABC):
    def __init__(self, owner_name: str, initial_balance: float = 0.0) -> None:
        ...

    @property
    def balance(self) -> float:
        ...

    @abstractmethod
    def deposit(self, amount: float) -> None: ...

    @abstractmethod
    def withdraw(self, amount: float) -> None: ...

    @abstractmethod
    def apply_monthly_update(self) -> None: ...

    def get_history(self) -> list[Transaction]:
        ...

    def get_statement(self) -> str:
        ...


class SavingsAccount(Account):
    def __init__(
        self,
        owner_name: str,
        initial_balance: float = 0.0,
        interest_rate: float = 0.02,
        minimum_balance: float = 100.0,
    ) -> None:
        ...

    def deposit(self, amount: float) -> None: ...
    def withdraw(self, amount: float) -> None: ...
    def apply_monthly_update(self) -> None: ...


class CurrentAccount(Account):
    def __init__(
        self,
        owner_name: str,
        initial_balance: float = 0.0,
        overdraft_limit: float = 500.0,
        monthly_fee: float = 5.0,
    ) -> None:
        ...

    @property
    def overdraft_limit(self) -> float: ...

    @overdraft_limit.setter
    def overdraft_limit(self, value: float) -> None: ...

    def deposit(self, amount: float) -> None: ...
    def withdraw(self, amount: float) -> None: ...
    def apply_monthly_update(self) -> None: ...


@dataclass
class Customer:
    name: str
    email: str
    customer_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __post_init__(self) -> None:
        ...

    def add_account(self, account: Account) -> None: ...
    def remove_account(self, account_id: str) -> None: ...
    def get_account(self, account_id: str) -> Account | None: ...
    def total_balance(self) -> float: ...
    def all_accounts(self) -> list[Account]: ...


class Bank:
    def __init__(self, name: str) -> None:
        ...

    def register_customer(self, customer: Customer) -> None: ...
    def get_customer(self, customer_id: str) -> Customer: ...
    def open_account(self, customer_id: str, account: Account) -> None: ...
    def close_account(self, customer_id: str, account_id: str) -> None: ...
    def total_assets(self) -> float: ...
    def monthly_update(self) -> None: ...
    def generate_bank_report(self) -> str: ...
```

---

## Exercises

1. Implement everything in the skeleton. Run `pytest tests/test_banking.py -v` until all 12 tests pass.

2. Add a `transfer(from_account: Account, to_account: Account, amount: float) -> None` function at module level. Write two tests.

3. Implement `__iter__` on `Customer` to yield their accounts. Use it in `Bank.monthly_update` instead of calling `all_accounts()`.

4. Generate a bank report string that looks like:

```
═══ First National Bank ═══
Customers: 3
Total assets: $12,450.00

Alice (alice@example.com) — 2 accounts, $5,000.00
  [abc12345] SavingsAccount   $3,000.00
  [def67890] CurrentAccount   $2,000.00

Bob (bob@example.com) — 1 account, $7,450.00
  [ghi11223] SavingsAccount   $7,450.00
```

5. Add type annotations everywhere and run `mypy banking/` (install: `pip install mypy`). Fix all errors.

(Solutions: ../solutions/week-03/day-6.md)

## Self-check

- [ ] I implemented all entities to spec with no hints.
- [ ] All 12 required tests pass.
- [ ] My code has no god files (no file > 300 lines; split if needed).
- [ ] Every public method has a type annotation.
- [ ] I applied at least three SOLID principles consciously.
