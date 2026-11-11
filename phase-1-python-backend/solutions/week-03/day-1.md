# Solutions — Week 03 · Day 1

## Exercise 1 — `withdraw` method

```python
def withdraw(self, amount: float) -> None:
    if amount <= 0:
        raise ValueError(f"Withdrawal amount must be positive, got {amount}")
    if amount > self.balance:
        raise ValueError(f"Insufficient funds: balance {self.balance:.2f} < {amount:.2f}")
    self.balance -= amount

# Demo script
if __name__ == "__main__":
    acc = BankAccount("Alice", 500.0)
    acc.deposit(200.0)
    acc.withdraw(100.0)
    print(acc.balance)  # 600.0
```

## Exercise 2 — `Customer` class

```python
class Customer:
    def __init__(self, name: str, email: str) -> None:
        self.name = name
        self.email = email
        self.accounts: list[BankAccount] = []

    def add_account(self, account: BankAccount) -> None:
        self.accounts.append(account)

    def __repr__(self) -> str:
        return f"Customer(name={self.name!r}, accounts={len(self.accounts)})"

# Demo
alice = Customer("Alice", "alice@example.com")
alice.add_account(BankAccount("Alice", 100.0))
alice.add_account(BankAccount("Alice", 200.0))
print(len(alice.accounts))  # 2
```

## Exercise 3 — `__class__` vs `type()`

```python
acc = BankAccount("Alice")
print(acc.__class__)   # <class '__main__.BankAccount'>
print(type(acc))       # <class '__main__.BankAccount'>
print(acc.__class__ is type(acc))  # True
```

`__class__` is an instance attribute that points to the class; `type()` is a built-in that returns the same object. They are identical for normal classes. The difference matters only with `__class__` assignment or metaclass tricks.

## Exercise 4 — pytest tests

```python
# tests/test_bank_account.py
import pytest
from bank_account import BankAccount


def test_create_valid_account() -> None:
    acc = BankAccount("Alice", 100.0)
    assert acc.owner == "Alice"
    assert acc.balance == 100.0


def test_create_negative_balance_raises() -> None:
    with pytest.raises(ValueError, match="negative"):
        BankAccount("Alice", -50.0)


def test_deposit_increases_balance() -> None:
    acc = BankAccount("Alice", 0.0)
    acc.deposit(100.0)
    assert acc.balance == 100.0
```
