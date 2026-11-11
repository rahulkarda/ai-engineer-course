# Solutions — Week 05 · Day 3 (Fixtures, Mocking, Integration)

## Exercise 1 — Session-scoped fixture with output

```python
# tests/conftest.py
import pytest

@pytest.fixture(scope="session", autouse=True)
def session_banner():
    print("\n--- test session start ---")
    yield
    print("\n--- test session end ---")
```

`autouse=True` applies the fixture to all tests without them declaring it. The `\n` ensures the banner appears on its own line in terminal output. Run `pytest -s` to see print output (by default pytest captures it).

## Exercise 2 — Withdrawal notification mocked tests

```python
# src/notifications.py addition
def send_withdrawal_notification(owner: str, amount: str) -> None:
    import logging
    logging.getLogger(__name__).info("Notifying %s of withdrawal %s", owner, amount)
```

```python
# src/ledger.py — withdraw updated
from src.notifications import send_withdrawal_notification

def withdraw(self, amount: Decimal) -> None:
    if amount <= 0:
        raise ValueError(f"withdrawal amount must be positive, got {amount}")
    if amount > self._balance:
        raise InsufficientFundsError(...)
    self._balance -= amount
    send_withdrawal_notification(self.owner, str(amount))
```

```python
# tests — three mocked withdrawal notification tests
from unittest.mock import patch
import pytest
from decimal import Decimal
from src.ledger import Ledger, InsufficientFundsError


def test_withdraw_calls_notification(funded_alice):
    with patch("src.ledger.send_withdrawal_notification") as mock_notify:
        funded_alice.withdraw(Decimal("50"))
        mock_notify.assert_called_once_with("alice", "50")


def test_withdraw_invalid_amount_no_notification(funded_alice):
    with patch("src.ledger.send_withdrawal_notification") as mock_notify:
        with pytest.raises(ValueError):
            funded_alice.withdraw(Decimal("-10"))
        mock_notify.assert_not_called()


def test_withdraw_insufficient_funds_no_notification(alice):
    with patch("src.ledger.send_withdrawal_notification") as mock_notify:
        with pytest.raises(InsufficientFundsError):
            alice.withdraw(Decimal("100"))   # alice has 0 balance
        mock_notify.assert_not_called()
```

## Exercise 3 — ledger_pair fixture

```python
# tests/conftest.py addition
@pytest.fixture
def ledger_pair():
    alice = Ledger("alice", Decimal("1000"))
    bob = Ledger("bob", Decimal("0"))
    return alice, bob


# Tests using ledger_pair
def test_transfer_full_balance(ledger_pair):
    alice, bob = ledger_pair
    alice.transfer_to(bob, Decimal("1000"))
    assert alice.balance == Decimal("0")
    assert bob.balance == Decimal("1000")


def test_transfer_partial(ledger_pair):
    alice, bob = ledger_pair
    alice.transfer_to(bob, Decimal("250"))
    assert alice.balance == Decimal("750")
    assert bob.balance == Decimal("250")


def test_transfer_insufficient_raises(ledger_pair):
    alice, bob = ledger_pair
    with pytest.raises(InsufficientFundsError):
        alice.transfer_to(bob, Decimal("2000"))
```

## Exercise 4 — MagicMock PaymentGateway

```python
from unittest.mock import MagicMock
import pytest


class PaymentService:
    def __init__(self, gateway) -> None:
        self._gateway = gateway

    def charge(self, amount):
        try:
            result = self._gateway.charge(amount)
            return result["status"] == "ok"
        except ConnectionError:
            return False


def test_charge_success():
    gateway = MagicMock()
    gateway.charge.return_value = {"status": "ok"}
    service = PaymentService(gateway)
    assert service.charge(100) is True
    gateway.charge.assert_called_once_with(100)


def test_charge_connection_error():
    gateway = MagicMock()
    gateway.charge.side_effect = ConnectionError("timeout")
    service = PaymentService(gateway)
    assert service.charge(100) is False
```

## Exercise 5 — Patch where it's used (written answer)

When `ledger.py` executes `from src.notifications import send_deposit_notification`, Python resolves the name and binds `ledger.send_deposit_notification` to the original function object. Patching `src.notifications.send_deposit_notification` replaces the attribute on the `notifications` module, but `ledger.py`'s local binding already points to the original function — it is unaffected.

Patching `src.ledger.send_deposit_notification` replaces the name inside `ledger.py`'s namespace, which is the reference actually called at runtime. The rule is: **patch the name as seen by the code under test**, not where the object was originally defined.
