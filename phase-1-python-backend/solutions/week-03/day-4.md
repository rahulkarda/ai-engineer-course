# Solutions — Week 03 · Day 4

## Exercise 1 — `__hash__`

When you define `__eq__`, Python sets `__hash__ = None` to prevent inconsistent hashing. Objects used in sets or as dict keys must satisfy: `a == b → hash(a) == hash(b)`.

```python
def __hash__(self) -> int:
    # Use immutable, identity-forming fields
    return hash((self.owner, round(self._balance, 2)))
```

Note: since balance can change, a mutable object in a set is risky. Consider making `account_id` the hash key if it's immutable after creation.

## Exercise 2 — `__iter__`

```python
def __iter__(self):
    return iter(self._history)

# Usage
for txn in account:
    print(txn)
```

## Exercise 3 — Frozen dataclass

```python
@dataclass(frozen=True)
class Transaction:
    amount: float
    type: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
```

**What changes**: `__hash__` is auto-generated (frozen dataclasses are hashable). `__setattr__` and `__delattr__` raise `FrozenInstanceError`.

**What breaks**: any code that does `txn.amount = 99` raises `dataclasses.FrozenInstanceError`. Fix by constructing a new `Transaction` instead of mutating.

```python
# Instead of txn.amount = 99:
corrected = Transaction(amount=99, type=txn.type, timestamp=txn.timestamp)
```

## Exercise 4 — `overdraft_limit` setter

```python
@property
def overdraft_limit(self) -> float:
    return self._overdraft_limit

@overdraft_limit.setter
def overdraft_limit(self, value: float) -> None:
    if not (0 <= value <= 10_000):
        raise ValueError(f"Overdraft limit must be 0–10,000, got {value}")
    self._overdraft_limit = value
```

## Exercise 5 — pytest tests

```python
def test_accounts_equal_when_same_owner_and_balance() -> None:
    a1 = Account("Alice", 100.0)
    a2 = Account("Alice", 100.0)
    assert a1 == a2

def test_lower_balance_is_less_than() -> None:
    a1 = Account("Alice", 100.0)
    a2 = Account("Bob", 200.0)
    assert a1 < a2

def test_in_operator_after_deposit() -> None:
    acc = Account("Alice")
    acc.deposit(100.0)
    assert 100.0 in acc
```
