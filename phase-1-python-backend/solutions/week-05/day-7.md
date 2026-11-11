# Solutions — Week 05 · Day 7 (Debugging)

## Bug 1 — Public mutable attribute → make balance a property

```python
# Fixed Account class — balance is read-only property
class Account:
    def __init__(self, owner: str, balance: Decimal = Decimal(0)) -> None:
        self.owner = owner
        self._balance = balance        # private; property exposes read-only view
        self._transactions: list[dict] = []

    @property
    def balance(self) -> Decimal:
        return self._balance

    # No setter defined — attempts to assign raise AttributeError
```

Regression test:
```python
def test_bug1_balance_attribute_is_readonly():
    acc = Account("test", Decimal("100"))
    with pytest.raises(AttributeError):
        acc.balance = Decimal("9999999")
```

## Bug 2 — `amount < 0` should be `amount <= 0`

The condition `if amount < 0` allows zero through. Zero deposits are meaningless and corrupt the transaction history.

```python
# Fixed
def deposit(self, amount: Decimal) -> None:
    if amount <= 0:                 # was: amount < 0
        raise ValueError("amount must be positive")
    self._balance += amount
    self._transactions.append({"type": "deposit", "amount": amount})
```

Regression tests:
```python
@pytest.mark.parametrize("amount", [Decimal("0"), Decimal("-1"), Decimal("-100")])
def test_bug2_deposit_zero_or_negative_raises(amount):
    acc = Account("test")
    with pytest.raises(ValueError, match="positive"):
        acc.deposit(amount)
```

## Bug 3 — statement() returns reference; must return copy

Python list assignment creates an alias, not a copy. `return self._transactions` lets callers mutate the internal list. Fix: return `list(self._transactions)` (shallow copy sufficient for dicts).

```python
def statement(self) -> list[dict]:
    return list(self._transactions)  # was: return self._transactions
```

Regression test:
```python
def test_bug3_statement_returns_copy():
    acc = Account("test")
    acc.deposit(Decimal("50"))
    history = acc.statement()
    history.clear()
    assert len(acc.statement()) == 1   # internal state unaffected
```

## Bug 4 — total_deposits() sums all types; must filter

```python
def total_deposits(self) -> Decimal:
    # was: sum(t["amount"] for t in self._transactions)
    return sum(
        t["amount"]
        for t in self._transactions
        if t["type"] == "deposit"   # added filter
    )
```

Regression test:
```python
def test_bug4_total_deposits_excludes_withdrawals():
    acc = Account("test", Decimal("100"))
    acc.deposit(Decimal("50"))
    acc.withdraw(Decimal("30"))
    acc.deposit(Decimal("20"))
    assert acc.total_deposits() == Decimal("70")  # 50 + 20, not 100
```

## Bug 5 — Off-by-one: range(1, ...) skips index 0

```python
def net_balance_change(self) -> Decimal:
    total = Decimal(0)
    # was: for i in range(1, len(self._transactions)):
    for i in range(len(self._transactions)):  # start at 0
        t = self._transactions[i]
        if t["type"] == "deposit":
            total += t["amount"]
        else:
            total -= t["amount"]
    return total
```

Or more Pythonically:
```python
def net_balance_change(self) -> Decimal:
    total = Decimal(0)
    for t in self._transactions:
        if t["type"] == "deposit":
            total += t["amount"]
        else:
            total -= t["amount"]
    return total
```

Regression test:
```python
def test_bug5_net_balance_includes_first_transaction():
    acc = Account("test")
    acc.deposit(Decimal("100"))   # transaction index 0 — previously skipped
    acc.deposit(Decimal("50"))
    acc.withdraw(Decimal("30"))
    assert acc.net_balance_change() == Decimal("120")
```

## Bonus — ThreadedAccount concurrency bug (written explanation)

`ThreadedAccount` has a **check-then-act** race condition in `withdraw`, and a **read-modify-write** race in both `deposit` and `withdraw`.

In `deposit`:
```python
current = self._balance        # Thread A reads 0
                               # Thread B also reads 0 (context switch here)
self._balance = current + amount  # Thread A writes 100
                               # Thread B also writes 100 — A's update lost!
```
Two concurrent `deposit(100)` calls can result in a balance of 100 instead of 200.

In `withdraw`, the check `if amount > self._balance` and the subsequent `self._balance = current - amount` are not atomic. Between the check and the write, another thread can reduce the balance, making the final write go negative:
```
Thread A: checks balance 100 >= 80 → OK
Thread B: checks balance 100 >= 70 → OK
Thread A: writes 100 - 80 = 20
Thread B: writes 100 - 70 = 30  (using stale `current`)
# Actual balance should be -50, but it shows 30
```

Fix: use `threading.Lock` to make each operation atomic:
```python
import threading

class ThreadedAccount:
    def __init__(self):
        self._balance = Decimal("0")
        self._lock = threading.Lock()

    def deposit(self, amount: Decimal) -> None:
        with self._lock:
            self._balance += amount

    def withdraw(self, amount: Decimal) -> None:
        with self._lock:
            if amount > self._balance:
                raise ValueError("insufficient")
            self._balance -= amount
```
