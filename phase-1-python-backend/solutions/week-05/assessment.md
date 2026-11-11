# Solutions — Week 05 Assessment

## Part 1 — Conceptual answers

**1. Unit vs integration test**

A unit test exercises one function or class in complete isolation — all external dependencies (DB, HTTP, file) are mocked. An integration test exercises multiple real components together. A unit test cannot catch a mismatch between what a service passes to a repository and what the repository's SQL actually expects — only an integration test that runs real SQL will reveal that.

**2. @pytest.mark.parametrize**

`@pytest.mark.parametrize("a,b,expected", [...])` generates one test case per tuple in the list, running the test function with each set of inputs. Prefer it over three separate functions when the test logic is identical and only the data varies — it reduces duplication and makes it easy to add new cases without copy-pasting code.

**3. Mocking requests.get**

```python
from unittest.mock import patch, MagicMock

def test_fetch_data():
    mock_response = MagicMock()
    mock_response.json.return_value = {"rate": 1.2}
    with patch("src.rates.requests.get", return_value=mock_response) as mock_get:
        result = fetch_rate("GBP", "EUR")
        mock_get.assert_called_once()
        assert result == 1.2
```

You patch `src.rates.requests.get` (the name as it exists in the module under test) because `from src.rates import fetch_rate` has already bound `requests` in the `src.rates` namespace. Patching `requests.get` directly would replace it in the `requests` module, but `src.rates` already holds its own reference — the patch would have no effect on that reference.

**4. git merge vs rebase**

`git merge` creates a merge commit joining two branch tips — it preserves the exact order events happened. `git rebase` replays commits on top of another branch, producing a linear history. Use rebase for local feature branches before opening a PR (keeps main's history clean); use merge when integrating completed features or hotfixes into `main` (preserves explicit merge boundaries). Never rebase commits that others have pulled: rebase rewrites SHAs, so the remote and the teammate's local copy would diverge, causing a confusing split history.

**5. Fixture scope and shared mutable state**

Scope controls how many times pytest instantiates a fixture: `function` (default) = once per test; `session` = once per run. If you use `scope="session"` for a fixture that returns a mutable object (like a list or a ledger instance), all tests share the same object. A test that deposits into the ledger changes the balance for every subsequent test — order-dependent failures appear. Use `scope="session"` only for immutable or idempotent resources (DB connections, read-only config).

**6. Hardcoded API key risks**

Risk 1 — **version control exposure**: the key is committed to Git history. Even if deleted later, the history is permanent and the key must be rotated.

Risk 2 — **environment bleed**: the same key is used in dev, staging, and prod. A dev environment compromise exposes production credentials.

Risk 3 — **no rotation path**: changing the key requires a code change and redeployment. With `os.getenv` / pydantic-settings, rotation is an env var swap with no code change.

**7. Python logging hierarchy**

Logger `src.ledger.accounts` is a child of `src.ledger`, which is a child of `src`, which is a child of the root. Setting its level to `WARNING` means only `WARNING`, `ERROR`, and `CRITICAL` records pass the level check on that logger. `DEBUG` and `INFO` records are suppressed at the source — they are not forwarded to the root logger (they never become `LogRecord` objects that propagate). The root logger is configured independently and handles records from all other loggers that propagate up to it.

**8. Separation of concerns diagram**

```
HTTP handler (controller)     ← HTTP request parsing, response serialisation
     ↓
Service / use-case layer      ← business rule validation
     ↓
Repository / data layer       ← SQL query
     ↓
Domain model (Pydantic)       ← Pydantic model definition
```

- HTTP request parsing → controller
- SQL query → repository
- Business rule validation → service
- Pydantic model definition → domain model

---

## Part 2 — TransactionLog implementation

```python
# src/transaction_log.py
"""TransactionLog — tracks deposits and withdrawals with unique IDs."""
from __future__ import annotations
import logging
from decimal import Decimal
from src.exceptions import InvalidAmountError, InsufficientFundsError
from src.config import settings

log = logging.getLogger(__name__)

_VALID_FILTERS = {"deposit", "withdraw", None}


class TransactionLog:
    def __init__(self) -> None:
        self._counter: int = 0
        self._balance: Decimal = Decimal("0")
        self._records: list[dict] = []

    def _next_id(self) -> str:
        self._counter += 1
        return f"TXN-{self._counter:04d}"

    def deposit(self, amount: Decimal, ref: str) -> str:
        log.debug("deposit attempt: amount=%s ref=%s", amount, ref)
        if amount <= 0:
            log.warning("rejected deposit: amount=%s", amount)
            raise InvalidAmountError(f"deposit amount must be positive, got {amount}")
        txn_id = self._next_id()
        self._balance += amount
        self._records.append({"id": txn_id, "type": "deposit", "amount": amount, "ref": ref})
        log.info("deposit: id=%s amount=%s ref=%s balance=%s", txn_id, amount, ref, self._balance)
        return txn_id

    def withdraw(self, amount: Decimal, ref: str) -> str:
        log.debug("withdraw attempt: amount=%s ref=%s", amount, ref)
        if amount <= 0:
            log.warning("rejected withdrawal: amount=%s", amount)
            raise InvalidAmountError(f"withdrawal amount must be positive, got {amount}")
        if amount > self._balance:
            log.error("insufficient funds: requested=%s balance=%s", amount, self._balance)
            raise InsufficientFundsError(
                f"cannot withdraw {amount}: balance is {self._balance}"
            )
        txn_id = self._next_id()
        self._balance -= amount
        self._records.append({"id": txn_id, "type": "withdraw", "amount": amount, "ref": ref})
        log.info("withdrawal: id=%s amount=%s ref=%s balance=%s", txn_id, amount, ref, self._balance)
        return txn_id

    def balance(self) -> Decimal:
        return self._balance

    def history(self, type_filter: str | None = None) -> list[dict]:
        if type_filter not in _VALID_FILTERS:
            raise ValueError(f"type_filter must be 'deposit', 'withdraw', or None; got {type_filter!r}")
        if type_filter is None:
            return list(self._records)
        return [r for r in self._records if r["type"] == type_filter]

    def summary(self) -> dict:
        deposits = sum(r["amount"] for r in self._records if r["type"] == "deposit")
        withdrawals = sum(r["amount"] for r in self._records if r["type"] == "withdraw")
        return {
            "deposits": deposits,
            "withdrawals": withdrawals,
            "net": deposits - withdrawals,
        }
```

```python
# tests/unit/test_transaction_log.py
"""Happy-path and edge-case tests for TransactionLog."""
from __future__ import annotations
import pytest
from decimal import Decimal
from src.transaction_log import TransactionLog
from src.exceptions import InvalidAmountError, InsufficientFundsError


@pytest.fixture
def log_obj():
    return TransactionLog()


def test_ids_increment(log_obj):
    id1 = log_obj.deposit(Decimal("10"), ref="a")
    id2 = log_obj.deposit(Decimal("20"), ref="b")
    assert id1 == "TXN-0001"
    assert id2 == "TXN-0002"


def test_deposit_updates_balance(log_obj):
    log_obj.deposit(Decimal("100"), ref="pay")
    assert log_obj.balance() == Decimal("100")


def test_withdraw_updates_balance(log_obj):
    log_obj.deposit(Decimal("100"), ref="pay")
    log_obj.withdraw(Decimal("40"), ref="rent")
    assert log_obj.balance() == Decimal("60")


def test_history_all(log_obj):
    log_obj.deposit(Decimal("50"), ref="d1")
    log_obj.withdraw(Decimal("10"), ref="w1")
    assert len(log_obj.history()) == 2


def test_history_deposit_filter(log_obj):
    log_obj.deposit(Decimal("50"), ref="d1")
    log_obj.withdraw(Decimal("10"), ref="w1")
    log_obj.deposit(Decimal("20"), ref="d2")
    deposits = log_obj.history("deposit")
    assert len(deposits) == 2
    assert all(r["type"] == "deposit" for r in deposits)


def test_history_withdraw_filter(log_obj):
    log_obj.deposit(Decimal("50"), ref="d1")
    log_obj.withdraw(Decimal("10"), ref="w1")
    withdrawals = log_obj.history("withdraw")
    assert len(withdrawals) == 1


def test_summary(log_obj):
    log_obj.deposit(Decimal("100"), ref="a")
    log_obj.deposit(Decimal("50"), ref="b")
    log_obj.withdraw(Decimal("30"), ref="c")
    s = log_obj.summary()
    assert s["deposits"] == Decimal("150")
    assert s["withdrawals"] == Decimal("30")
    assert s["net"] == Decimal("120")


def test_history_returns_copy(log_obj):
    log_obj.deposit(Decimal("50"), ref="x")
    h = log_obj.history()
    h.clear()
    assert len(log_obj.history()) == 1


@pytest.mark.parametrize("amount,ref", [
    (Decimal("0.01"), "tiny"),
    (Decimal("1"), "one"),
    (Decimal("9999.99"), "large"),
    (Decimal("50000"), "max"),
])
def test_deposit_various_amounts(log_obj, amount, ref):
    txn_id = log_obj.deposit(amount, ref=ref)
    assert log_obj.balance() == amount
    assert txn_id.startswith("TXN-")
```

```python
# tests/unit/test_transaction_log_errors.py
"""Error-path tests for TransactionLog."""
from __future__ import annotations
import pytest
from decimal import Decimal
from src.transaction_log import TransactionLog
from src.exceptions import InvalidAmountError, InsufficientFundsError


@pytest.fixture
def log_obj():
    return TransactionLog()


@pytest.mark.parametrize("amount", [Decimal("0"), Decimal("-1"), Decimal("-0.01")])
def test_deposit_invalid_amount(log_obj, amount):
    with pytest.raises(InvalidAmountError):
        log_obj.deposit(amount, ref="bad")


@pytest.mark.parametrize("amount", [Decimal("0"), Decimal("-5")])
def test_withdraw_invalid_amount(log_obj, amount):
    log_obj.deposit(Decimal("100"), ref="seed")
    with pytest.raises(InvalidAmountError):
        log_obj.withdraw(amount, ref="bad")


def test_withdraw_insufficient_funds(log_obj):
    log_obj.deposit(Decimal("50"), ref="seed")
    with pytest.raises(InsufficientFundsError):
        log_obj.withdraw(Decimal("100"), ref="big")


def test_history_invalid_filter(log_obj):
    with pytest.raises(ValueError, match="type_filter"):
        log_obj.history("invalid_type")


def test_withdraw_from_empty_account(log_obj):
    with pytest.raises(InsufficientFundsError):
        log_obj.withdraw(Decimal("1"), ref="empty")
```

---

## Part 3 — batch_transfer bug analysis

**Bug A** — `result = accounts` creates an alias, not a copy. The caller's `accounts` dict is mutated in-place.

Fix: `result = dict(accounts)`

**Bug B** — `if result[from_acc] < 0` checks if balance is already negative, not if the withdrawal would cause it.

Fix: `if result[from_acc] < amount`

**Bug C** — `result[to_acc] -= amount` subtracts from the recipient instead of adding.

Fix: `result[to_acc] += amount`

Regression tests:
```python
def test_bug_a_does_not_mutate_input():
    accounts = {"alice": Decimal("100"), "bob": Decimal("0")}
    original = dict(accounts)
    batch_transfer(accounts, [("alice", "bob", Decimal("50"))])
    assert accounts == original   # input unchanged


def test_bug_b_insufficient_funds_raises():
    accounts = {"alice": Decimal("10"), "bob": Decimal("0")}
    with pytest.raises(ValueError, match="insufficient"):
        batch_transfer(accounts, [("alice", "bob", Decimal("50"))])


def test_bug_c_recipient_balance_increases():
    accounts = {"alice": Decimal("100"), "bob": Decimal("0")}
    result = batch_transfer(accounts, [("alice", "bob", Decimal("40"))])
    assert result["bob"] == Decimal("40")
```

---

## Part 4 — Code quality rewrite

Issues in the original:
1. `import requests` inside function body — re-imported on every call.
2. Hardcoded API key `"abc123"` as a default argument — secret in source.
3. Bare `except:` catches everything including `SystemExit`, `KeyboardInterrupt`.
4. `print(...)` instead of `logging`.
5. No type hints.
6. Function name `do_stuff` is meaningless.
7. `float(x)` — loses precision for financial amounts; should stay `Decimal` or use the value as-is.
8. Returns `0` on error silently — caller cannot tell it failed.
9. No separation of concern — HTTP call and business logic in one function.

```python
# src/rates.py — clean version
"""Exchange rate fetcher — separated from business logic."""
from __future__ import annotations
import logging
import requests
from src.config import settings
from src.exceptions import AppError

log = logging.getLogger(__name__)


class RateFetchError(AppError):
    """Failed to retrieve exchange rate from upstream."""


def fetch_exchange_rate(from_currency: str, to_currency: str) -> float:
    """Fetch the exchange rate from_currency → to_currency.

    Raises RateFetchError on any network or API error.
    """
    log.debug("fetching rate: %s -> %s", from_currency, to_currency)
    try:
        response = requests.get(
            "https://api.example.com/rate",
            params={"from": from_currency, "to": to_currency},
            headers={"X-Key": settings.app_api_key},
            timeout=5,
        )
        response.raise_for_status()
        rate: float = response.json()["rate"]
        log.info("rate fetched: %s->%s rate=%s", from_currency, to_currency, rate)
        return rate
    except (requests.RequestException, KeyError) as exc:
        log.error("rate fetch failed: %s", exc, exc_info=True)
        raise RateFetchError(f"could not fetch rate {from_currency}->{to_currency}") from exc
```
