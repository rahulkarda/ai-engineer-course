# Week 03 Project — Banking Application

## Goal

Build a complete, production-quality (for learning purposes) Banking Application that demonstrates every OOP concept from the week in a coherent, working system.

---

## Deliverables

```
banking_app/
├── banking/
│   ├── __init__.py
│   ├── exceptions.py
│   ├── models.py          # Transaction, Account ABC, SavingsAccount, CurrentAccount
│   ├── customer.py        # Customer dataclass
│   ├── bank.py            # Bank class
│   └── reports.py         # ReportGenerator, export helpers
├── tests/
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_customer.py
│   ├── test_bank.py
│   └── test_reports.py
├── pyproject.toml
└── README.md
```

---

## Full specification

### `banking/exceptions.py`

```python
class BankingError(Exception): ...
class InsufficientFundsError(BankingError, ValueError): ...
class DuplicateCustomerError(BankingError, ValueError): ...
class DuplicateAccountError(BankingError, ValueError): ...
class AccountNotFoundError(BankingError, KeyError): ...
class CustomerNotFoundError(BankingError, KeyError): ...
class InvalidAmountError(BankingError, ValueError): ...
```

### `banking/models.py`

**`Transaction`** — frozen dataclass
- Fields: `amount: float`, `type: Literal["deposit","withdrawal","interest","fee","transfer"]`, `timestamp: datetime` (UTC default), `description: str` ("")
- `__post_init__`: raise `InvalidAmountError` if `amount <= 0`

**`Account`** — ABC
- Constructor: `owner_name: str`, `initial_balance: float = 0.0`
- `account_id: str` — first 8 chars of a UUID4, generated on construction
- `_balance: float` — backing store
- `_history: list[Transaction]` — private
- Properties: `balance` (read-only), `account_id` (read-only)
- Abstract: `deposit(amount)`, `withdraw(amount)`, `apply_monthly_update()`
- Concrete: `get_history() -> list[Transaction]` (copy), `get_statement() -> str`
- Dunders: `__repr__`, `__str__`, `__eq__` (by account_id), `__hash__` (by account_id), `__len__` (transaction count), `__iter__` (yields Transaction)

**`SavingsAccount(Account)`**
- `interest_rate: float` property; setter validates `0 < rate < 1`
- `minimum_balance: float = 100.0`
- `deposit`: validates positive; records Transaction
- `withdraw`: validates positive; `balance - amount >= minimum_balance` else `InsufficientFundsError`; records Transaction
- `apply_monthly_update`: interest = balance × rate; balance += interest; records Transaction("interest")

**`CurrentAccount(Account)`**
- `overdraft_limit: float` property; setter validates `0 <= limit <= 10_000`
- `monthly_fee: float = 5.0`
- `deposit`: validates; records
- `withdraw`: validates positive; `balance - amount >= -overdraft_limit` else `InsufficientFundsError`; records
- `apply_monthly_update`: balance -= monthly_fee; records Transaction("fee")

### `banking/customer.py`

**`Customer`** — dataclass
- Fields: `name: str`, `email: str`, `customer_id: str` (UUID prefix, not in `__init__` args — use `field(default_factory=...)`)
- `__post_init__`: validate `@` in email; raise `ValueError` otherwise
- Private `_accounts: list[Account]` — not a dataclass field; initialised in `__post_init__`
- `add_account(account: Account)`: raise `DuplicateAccountError` if already present (by `account_id`)
- `remove_account(account_id: str)`: raise `AccountNotFoundError` if not found
- `get_account(account_id: str) -> Account | None`
- `total_balance() -> float`
- `all_accounts() -> list[Account]`
- `__repr__`, `__len__` (account count), `__iter__` (yields Account)

### `banking/bank.py`

**`Bank`**
- `name: str`
- `_customers: dict[str, Customer]` — keyed by customer_id
- `register_customer(customer: Customer)`: raise `DuplicateCustomerError` if exists
- `unregister_customer(customer_id: str)`: raise `CustomerNotFoundError` if missing
- `get_customer(customer_id: str) -> Customer`: raise `CustomerNotFoundError` if missing
- `open_account(customer_id: str, account: Account)`: calls `customer.add_account`
- `close_account(customer_id: str, account_id: str)`: calls `customer.remove_account`
- `transfer(from_account: Account, to_account: Account, amount: float)`: atomic — withdraw then deposit; if deposit fails, re-deposit to from_account (rollback)
- `total_assets() -> float`: sum all account balances
- `monthly_update()`: calls `apply_monthly_update()` on every account of every customer
- `generate_bank_report() -> str`: see format below
- `customer_count() -> int`, `account_count() -> int`

### `banking/reports.py`

**`ReportGenerator`**
- Static method `text_report(bank: Bank) -> str` — formatted text
- Static method `export_csv(accounts: list[Account], path: Path) -> None` — CSV: `timestamp,type,amount,description`

**Bank report format**:
```
╔══ PyBank ═══════════════════════════════╗
║ Customers: 2   Accounts: 3   Assets: $12,450.00
╚═════════════════════════════════════════╝

Alice Smith (alice@example.com) [cust: ab12cd34]
  [acc12345] SavingsAccount    $3,000.00   (3 txns)
  [acc67890] CurrentAccount    $2,000.00   (1 txns)
  Subtotal: $5,000.00

Bob Jones (bob@example.com) [cust: ef56gh78]
  [accaabb1] SavingsAccount    $7,450.00   (5 txns)
  Subtotal: $7,450.00
```

---

## Required test coverage

Minimum 90% line coverage across `banking/`. Each test module mirrors a source module.

**`tests/conftest.py`** — fixtures: `savings_account`, `current_account`, `customer`, `bank`

**`tests/test_models.py`** — at least 20 tests:
- Happy path: deposit, withdraw, interest, fee, history integrity
- Edge cases: exact minimum balance, exact overdraft limit, one cent over limit
- Error cases: zero amount, negative amount, overdraft exceeded, below minimum
- Dunder: `__len__`, `__eq__`, `__iter__`, `__repr__`

**`tests/test_customer.py`** — at least 10 tests:
- add/remove/get account
- `total_balance` accuracy
- duplicate account raises
- invalid email raises
- `__iter__` and `__len__`

**`tests/test_bank.py`** — at least 15 tests:
- register/unregister customer
- open/close account
- transfer success, transfer rollback on second failure (use `monkeypatch`)
- `monthly_update` calls every account (verify via `monkeypatch`)
- `total_assets` sum
- duplicate customer raises
- not-found errors

**`tests/test_reports.py`** — at least 5 tests:
- `text_report` contains expected strings
- `export_csv` writes correct headers and rows (use `tmp_path`)

---

## Grading

| Criterion | Points |
|---|---|
| All domain classes implemented to spec | 30 |
| All error types raised in correct situations | 15 |
| Test suite passes at ≥ 90% coverage | 20 |
| Type annotations on every public API | 10 |
| SOLID principles applied (evidence in code) | 10 |
| Bank report formatted correctly | 10 |
| Transfer rollback implemented and tested | 5 |
| **Total** | **100** |

**Pass gate: 75 / 100**

Reference solution: `../solutions/week-03/project.md`
