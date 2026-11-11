# Solutions — Week 03 · Day 5

## Exercise 1 — `CurrentAccount(BaseAccount)`

```python
class CurrentAccount(BaseAccount):
    MONTHLY_FEE: ClassVar[float] = 5.0

    def __init__(self, owner: str, balance: float = 0.0, overdraft_limit: float = 500.0) -> None:
        super().__init__(owner, balance)
        self._overdraft_limit = overdraft_limit

    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Amount must be positive")
        self._balance += amount

    def withdraw(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Amount must be positive")
        if amount > self._balance + self._overdraft_limit:
            raise ValueError("Exceeds overdraft limit")
        self._balance -= amount

    def apply_monthly_update(self) -> None:
        self._balance -= self.MONTHLY_FEE
```

## Exercise 2 — Instantiating BaseAccount directly

```python
try:
    acc = BaseAccount("Alice")
except TypeError as e:
    print(e)
# TypeError: Can't instantiate abstract class BaseAccount
# with abstract methods deposit, withdraw, apply_monthly_update
```

ABCMeta checks `cls.__abstractmethods__` at `__call__` time and raises `TypeError` if any are unimplemented.

## Exercise 3 — `Interestable` Protocol

```python
@runtime_checkable
class Interestable(Protocol):
    def apply_monthly_update(self) -> None: ...

def apply_all(accounts: list[BaseAccount]) -> None:
    for acc in accounts:
        if isinstance(acc, Interestable):
            acc.apply_monthly_update()
```

## Exercise 4 — SRP violation fix

If `generate_statement` also sent an email, it would violate **SRP** (and arguably **SRP + OCP**): the method now has two reasons to change — statement format AND email configuration. Fix:

```python
# In SavingsAccount — only format the statement
def generate_statement(self) -> str:
    ...  # pure formatting, no I/O

# In a dedicated service
class StatementMailer:
    def __init__(self, notifier: NotificationService) -> None:
        self._notifier = notifier

    def mail_statement(self, account: Statementable) -> None:
        statement = account.generate_statement()
        self._notifier.notify(account.owner, statement)
```

## Exercise 5 — MockNotifier test

```python
class MockNotifier:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def notify(self, owner: str, message: str) -> None:
        self.calls.append((owner, message))


def test_bank_notifies_on_open_account() -> None:
    mock = MockNotifier()
    bank = Bank(notifier=mock)
    acc = SavingsAccount("Alice", 500.0)
    bank.open_account(acc)
    assert len(mock.calls) == 1
    owner, message = mock.calls[0]
    assert owner == "Alice"
    assert "opened" in message.lower()
```
