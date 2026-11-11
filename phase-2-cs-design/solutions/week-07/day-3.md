# Solutions — Week 07 · Day 3

## Exercise 1 — NewAlertAdapter

```python
from typing import Protocol


class NewAlertLibrary:
    """Third-party library we cannot modify."""

    def push(self, topic: str, payload: dict) -> None:
        import logging
        logging.getLogger(__name__).info("[NEW_LIB] topic=%s payload=%s", topic, payload)


class AlertNotifier(Protocol):
    def notify(self, channel: str, message: str) -> None: ...


class NewAlertAdapter:
    """Adapts NewAlertLibrary.push() to AlertNotifier.notify()."""

    def __init__(self, lib: NewAlertLibrary) -> None:
        self._lib = lib

    def notify(self, channel: str, message: str) -> None:
        self._lib.push(topic=channel, payload={"text": message})
```

---

## Exercise 2 — RetryDecorator

```python
import time
import logging

log = logging.getLogger(__name__)


class RetryDecorator:
    """Wraps any AlertNotifier with retry logic."""

    def __init__(
        self,
        wrapped: AlertNotifier,
        max_retries: int = 3,
        delay_seconds: float = 0.1,
    ) -> None:
        self._wrapped = wrapped
        self._max_retries = max_retries
        self._delay = delay_seconds

    def notify(self, channel: str, message: str) -> None:
        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                self._wrapped.notify(channel, message)
                return
            except Exception as exc:
                last_exc = exc
                log.warning("Attempt %d/%d failed: %s", attempt, self._max_retries, exc)
                if attempt < self._max_retries:
                    time.sleep(self._delay)
        raise last_exc  # type: ignore[misc]
```

Test:
```python
import pytest
from unittest.mock import MagicMock


def test_retry_succeeds_on_third_attempt():
    mock = MagicMock(spec=["notify"])
    mock.notify.side_effect = [RuntimeError("fail"), RuntimeError("fail"), None]
    retrying = RetryDecorator(mock, max_retries=3, delay_seconds=0)
    retrying.notify("ops", "alert")
    assert mock.notify.call_count == 3


def test_retry_raises_after_max():
    mock = MagicMock(spec=["notify"])
    mock.notify.side_effect = RuntimeError("always fails")
    retrying = RetryDecorator(mock, max_retries=2, delay_seconds=0)
    with pytest.raises(RuntimeError):
        retrying.notify("ops", "alert")
    assert mock.notify.call_count == 2
```

---

## Exercise 3 — SqliteProductRepository

```python
import sqlite3
from uuid import UUID
from pathlib import Path
from day3_structural import Product, ProductRepository
import logging

log = logging.getLogger(__name__)


class SqliteProductRepository(ProductRepository):
    """SQLite-backed repository."""

    def __init__(self, db_path: Path = Path(":memory:")) -> None:
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS products "
            "(id TEXT PRIMARY KEY, name TEXT, price REAL)"
        )
        self._conn.commit()

    def add(self, product: Product) -> None:
        self._conn.execute(
            "INSERT INTO products VALUES (?, ?, ?)",
            (str(product.product_id), product.name, product.price),
        )
        self._conn.commit()
        log.info("Stored product %s", product.name)

    def get(self, product_id: UUID) -> Product | None:
        row = self._conn.execute(
            "SELECT id, name, price FROM products WHERE id = ?",
            (str(product_id),),
        ).fetchone()
        if row is None:
            return None
        pid, name, price = row
        return Product(name=name, price=price, product_id=UUID(pid))

    def list_all(self) -> list[Product]:
        rows = self._conn.execute("SELECT id, name, price FROM products").fetchall()
        return [Product(name=n, price=p, product_id=UUID(i)) for i, n, p in rows]

    def remove(self, product_id: UUID) -> bool:
        cur = self._conn.execute(
            "DELETE FROM products WHERE id = ?", (str(product_id),)
        )
        self._conn.commit()
        return cur.rowcount > 0
```

---

## Exercise 4 — Decorator vs. inheritance explanation

Inheritance approach creates a class for every combination:
- `LoggingEmailNotifier`
- `TimingEmailNotifier`
- `RetryEmailNotifier`
- `LoggingTimingEmailNotifier`
- `LoggingRetryEmailNotifier`
- `TimingRetryEmailNotifier`
- `LoggingTimingRetryEmailNotifier`

With 3 behaviours and 1 base: **7 subclasses**. With 4 behaviours: **15**. With n: **2ⁿ − 1**.

Decorator composes independently at runtime. 3 decorators + 1 base = **4 classes total**, combined in any order. Adding a `RateLimitDecorator` requires one new class, not doubling the hierarchy.

---

## Exercise 5 — pytest test for ProductService

```python
import pytest
from day3_structural import InMemoryProductRepository, ProductService


@pytest.fixture
def service() -> ProductService:
    return ProductService(InMemoryProductRepository())


def test_get_expensive_returns_only_above_threshold(service):
    service.add_product("Cheap", 5.0)
    service.add_product("Mid", 50.0)
    service.add_product("Expensive", 200.0)

    result = service.get_expensive(49.99)
    names = {p.name for p in result}
    assert names == {"Mid", "Expensive"}


def test_get_expensive_empty_when_all_below(service):
    service.add_product("Cheap", 1.0)
    assert service.get_expensive(100.0) == []


def test_negative_price_raises(service):
    with pytest.raises(ValueError):
        service.add_product("Bad", -1.0)
```
