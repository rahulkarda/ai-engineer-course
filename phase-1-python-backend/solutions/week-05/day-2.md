# Solutions — Week 05 · Day 2 (pytest)

## Exercise 1 — Ledger.statement() with tests

```python
# src/ledger.py addition
def statement(self) -> list[tuple[str, Decimal]]:
    """Return a copy of the transaction history as (type, amount) tuples."""
    return list(self._history)

# Modified __init__:
def __init__(self, owner: str, initial_balance: Decimal = Decimal("0")) -> None:
    if initial_balance < 0:
        raise ValueError("initial_balance must be non-negative")
    self.owner = owner
    self._balance = initial_balance
    self._history: list[tuple[str, Decimal]] = []

# Modified deposit/withdraw to record:
def deposit(self, amount: Decimal) -> None:
    if amount <= 0:
        raise ValueError(f"deposit amount must be positive, got {amount}")
    self._balance += amount
    self._history.append(("deposit", amount))

def withdraw(self, amount: Decimal) -> None:
    if amount <= 0:
        raise ValueError(f"withdrawal amount must be positive, got {amount}")
    if amount > self._balance:
        raise InsufficientFundsError(
            f"cannot withdraw {amount}: balance is {self._balance}"
        )
    self._balance -= amount
    self._history.append(("withdrawal", amount))
```

```python
# tests — statement() tests
def test_statement_empty_on_new_ledger():
    ledger = Ledger("test")
    assert ledger.statement() == []


def test_statement_records_deposit():
    ledger = Ledger("test")
    ledger.deposit(Decimal("50"))
    assert ledger.statement() == [("deposit", Decimal("50"))]


def test_statement_records_in_order():
    ledger = Ledger("test")
    ledger.deposit(Decimal("100"))
    ledger.withdraw(Decimal("30"))
    ledger.deposit(Decimal("20"))
    assert ledger.statement() == [
        ("deposit", Decimal("100")),
        ("withdrawal", Decimal("30")),
        ("deposit", Decimal("20")),
    ]


def test_statement_returns_copy():
    ledger = Ledger("test")
    ledger.deposit(Decimal("50"))
    s = ledger.statement()
    s.clear()
    assert len(ledger.statement()) == 1  # original unaffected
```

## Exercise 2 — Parametrized transfer_to tests

```python
@pytest.mark.parametrize("alice_start,bob_start,amount,alice_end,bob_end,exc", [
    (Decimal("200"), Decimal("0"), Decimal("75"), Decimal("125"), Decimal("75"), None),
    (Decimal("100"), Decimal("0"), Decimal("100"), Decimal("0"), Decimal("100"), None),
    (Decimal("100"), Decimal("0"), Decimal("0"), None, None, ValueError),
    (Decimal("50"), Decimal("0"), Decimal("100"), None, None, InsufficientFundsError),
])
def test_transfer_to_cases(alice_start, bob_start, amount, alice_end, bob_end, exc):
    alice = Ledger("alice", alice_start)
    bob = Ledger("bob", bob_start)
    if exc:
        with pytest.raises(exc):
            alice.transfer_to(bob, amount)
    else:
        alice.transfer_to(bob, amount)
        assert alice.balance == alice_end
        assert bob.balance == bob_end
```

## Exercise 3 — freeze() tests

```python
# src/ledger.py addition
def freeze(self) -> None:
    self._frozen = True

# Modified deposit/withdraw preamble:
def deposit(self, amount: Decimal) -> None:
    if getattr(self, "_frozen", False):
        raise RuntimeError("ledger is frozen")
    ...

def withdraw(self, amount: Decimal) -> None:
    if getattr(self, "_frozen", False):
        raise RuntimeError("ledger is frozen")
    ...
```

```python
def test_frozen_ledger_rejects_deposit():
    ledger = Ledger("test", Decimal("100"))
    ledger.freeze()
    with pytest.raises(RuntimeError, match="frozen"):
        ledger.deposit(Decimal("10"))


def test_frozen_ledger_rejects_withdrawal():
    ledger = Ledger("test", Decimal("100"))
    ledger.freeze()
    with pytest.raises(RuntimeError, match="frozen"):
        ledger.withdraw(Decimal("10"))


def test_freeze_preserves_balance():
    ledger = Ledger("test", Decimal("100"))
    ledger.freeze()
    assert ledger.balance == Decimal("100")
```

## Exercise 4 — pytest -v output explanation

Running `pytest tests/test_ledger.py -v` produces lines like:

```
tests/test_ledger.py::test_initial_balance_zero PASSED        [ 10%]
tests/test_ledger.py::test_deposit_various_amounts[1-1] PASSED [ 20%]
tests/test_ledger.py::test_withdraw_more_than_balance_raises FAILED [ 30%]
```

Columns:
- `tests/test_ledger.py::test_name` — full test node ID (file::function).
- `PASSED / FAILED / ERROR / SKIPPED` — result.
- `[NN%]` — progress through the test plan.

The `-v` flag enables verbose mode — without it, pytest shows only a dot per test. `--tb=short` limits the traceback to a concise form on failures.

## Exercise 5 — Shell one-liner for CI

```bash
pytest tests/ && echo "CI passed"
```

`&&` runs the right side only if the left side exits with code 0. If any test fails, pytest exits with code 1, the `&&` short-circuits, and `"CI passed"` is never printed.
