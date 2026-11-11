# Solutions — Week 03 · Day 2

## Exercise 1 — `apply_interest`

```python
def apply_interest(self) -> None:
    interest = self.__balance * self._INTEREST_RATE
    self.__balance += interest
    logger.info("%s interest applied: %.2f", self.owner, interest)

# Test
def test_apply_interest() -> None:
    acc = Account("Alice", 1000.0)
    Account.set_interest_rate(0.03)
    acc.apply_interest()
    assert acc.get_balance() == pytest.approx(1030.0)
```

## Exercise 2 — `from_csv_line`

```python
@classmethod
def from_csv_line(cls, line: str) -> "Account":
    """Parse 'Alice,1000.00' format."""
    parts = line.strip().split(",")
    if len(parts) != 2:
        raise ValueError(f"Expected 'owner,balance', got: {line!r}")
    return cls(owner=parts[0].strip(), balance=float(parts[1].strip()))

# Demo
acc = Account.from_csv_line("Alice, 1000.00")
assert acc.owner == "Alice"
assert acc.get_balance() == 1000.0
```

## Exercise 3 — `is_valid_account_number`

```python
@staticmethod
def is_valid_account_number(number: str) -> bool:
    return len(number) == 10 and number.isdigit()

# Tests
def test_valid_account_number() -> None:
    assert Account.is_valid_account_number("1234567890") is True

def test_invalid_account_number() -> None:
    assert Account.is_valid_account_number("123abc") is False
    assert Account.is_valid_account_number("12345") is False
```

## Exercise 4 — Name mangling access

```python
acc = Account("Alice", 100.0)
print(acc._Account__balance)  # 100.0 — accessible but ugly
```

This works because Python rewrites `__balance` to `_Account__balance` in the class body. It's **convention-based privacy** — not enforced by the interpreter. The goal is to prevent *accidental* collision in subclasses, not to hide secrets.

## Exercise 5 — `Customer.from_name_email`

```python
class Customer:
    def __init__(self, name: str, email: str) -> None:
        self.name = name
        self.email = email
        self.accounts: list[Account] = []

    @classmethod
    def from_name_email(cls, name: str, email: str) -> "Customer":
        return cls(name=name, email=email)

# Verify
c = Customer.from_name_email("Alice", "alice@example.com")
assert c.name == "Alice"
```
