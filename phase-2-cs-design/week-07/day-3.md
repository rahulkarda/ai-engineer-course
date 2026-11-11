# Phase 2 · Week 07 · Day 3 — Structural Patterns: Adapter, Decorator & Repository

> Objective: Implement Adapter, Decorator, and Repository patterns; know when each solves a real integration or abstraction problem.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

---

## Concept

Structural patterns deal with **how objects and classes are composed** to form larger structures.

### Adapter

Converts the interface of a class into another interface clients expect. Like a power-plug adapter — the existing device (adaptee) and the target interface don't match; the adapter bridges them.

**Use when:** You want to use an existing class but its interface doesn't match what you need, especially when integrating third-party libraries.

### Decorator

Attaches additional responsibilities to an object dynamically. Decorators provide a flexible alternative to subclassing for extending functionality.

**Use when:** You need to add behaviours to individual objects without affecting others, and subclassing would cause a combinatorial explosion of classes.

**Note:** Python's `@decorator` syntax is related but not identical to the GoF pattern — Python decorators wrap *functions*, the GoF pattern wraps *objects*.

### Repository

Mediates between the domain and data-mapping layers using a collection-like interface for accessing domain objects. The rest of the application sees an in-memory collection; persistence details are hidden.

**Use when:** You want to decouple domain logic from database/ORM specifics, making the domain testable with simple in-memory fakes.

---

## How it works internally

**Adapter composition**: Python adapters typically use composition (hold a reference to the adaptee) rather than inheritance, avoiding multiple-inheritance complexity.

**Decorator chain**: Each decorator holds a reference to the next component (wrapped object). Calling `operation()` on the outermost decorator triggers a chain down to the concrete component. The key invariant: every link implements the same interface.

**Repository and the Unit of Work**: Production repositories often pair with a *Unit of Work* (tracks changes, commits atomically). For this week we focus on the repository interface; UoW appears in advanced patterns.

---

## Worked example

```python
"""
day3_structural.py
Adapter, GoF Decorator, and Repository patterns — logging/metrics domain.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID, uuid4

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─── Adapter ─────────────────────────────────────────────────────────────────
# Third-party library uses send_message(); our system expects notify().

class LegacyAlertSystem:
    """Existing third-party class we cannot modify."""

    def send_message(self, recipient: str, text: str) -> None:
        log.info("[LEGACY] -> %s: %s", recipient, text)


class AlertNotifier(Protocol):
    """The interface our application expects."""

    def notify(self, channel: str, message: str) -> None: ...


class LegacyAlertAdapter:
    """Adapter: wraps LegacyAlertSystem to satisfy AlertNotifier."""

    def __init__(self, legacy: LegacyAlertSystem) -> None:
        self._legacy = legacy

    def notify(self, channel: str, message: str) -> None:
        # Translate: channel → recipient, message → text
        self._legacy.send_message(recipient=channel, text=message)


# ─── GoF Decorator ───────────────────────────────────────────────────────────
# Add logging and timing to any AlertNotifier without subclassing.

class LoggingDecorator:
    """Adds structured logging around any AlertNotifier."""

    def __init__(self, wrapped: AlertNotifier) -> None:
        self._wrapped = wrapped

    def notify(self, channel: str, message: str) -> None:
        log.info("[LOG] Sending to %s", channel)
        self._wrapped.notify(channel, message)
        log.info("[LOG] Sent OK")


class TimingDecorator:
    """Measures and logs notification latency."""

    def __init__(self, wrapped: AlertNotifier) -> None:
        self._wrapped = wrapped

    def notify(self, channel: str, message: str) -> None:
        start = time.perf_counter()
        self._wrapped.notify(channel, message)
        elapsed_ms = (time.perf_counter() - start) * 1000
        log.info("[TIMING] %.2f ms", elapsed_ms)


# Stack decorators: TimingDecorator → LoggingDecorator → LegacyAlertAdapter
# Each call bubbles through the chain.


# ─── Repository ──────────────────────────────────────────────────────────────

@dataclass
class Product:
    name: str
    price: float
    product_id: UUID = field(default_factory=uuid4)


class ProductRepository(ABC):
    """Abstract repository: collection-like interface for Products."""

    @abstractmethod
    def add(self, product: Product) -> None: ...

    @abstractmethod
    def get(self, product_id: UUID) -> Product | None: ...

    @abstractmethod
    def list_all(self) -> list[Product]: ...

    @abstractmethod
    def remove(self, product_id: UUID) -> bool: ...


class InMemoryProductRepository(ProductRepository):
    """In-memory implementation — ideal for tests and prototypes."""

    def __init__(self) -> None:
        self._store: dict[UUID, Product] = {}

    def add(self, product: Product) -> None:
        self._store[product.product_id] = product
        log.info("Added product %s (%s)", product.name, product.product_id)

    def get(self, product_id: UUID) -> Product | None:
        return self._store.get(product_id)

    def list_all(self) -> list[Product]:
        return list(self._store.values())

    def remove(self, product_id: UUID) -> bool:
        if product_id in self._store:
            del self._store[product_id]
            return True
        return False


class ProductService:
    """Domain service — depends only on the abstract repository."""

    def __init__(self, repo: ProductRepository) -> None:
        self._repo = repo

    def add_product(self, name: str, price: float) -> Product:
        if price < 0:
            raise ValueError("Price cannot be negative")
        p = Product(name=name, price=price)
        self._repo.add(p)
        return p

    def get_expensive(self, threshold: float) -> list[Product]:
        return [p for p in self._repo.list_all() if p.price >= threshold]


# ─── Demo ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Adapter
    legacy = LegacyAlertSystem()
    adapter: AlertNotifier = LegacyAlertAdapter(legacy)

    # Decorator chain
    notifier: AlertNotifier = TimingDecorator(LoggingDecorator(adapter))
    notifier.notify("ops-channel", "Disk usage > 90%")

    # Repository
    repo = InMemoryProductRepository()
    service = ProductService(repo)

    service.add_product("Widget", 9.99)
    service.add_product("Gadget", 49.99)
    service.add_product("Doohickey", 4.99)

    expensive = service.get_expensive(10.0)
    log.info("Expensive products: %s", [p.name for p in expensive])
```

---

## Exercises

1. **Adapter for a new source.** A new alerting library has `push(topic: str, payload: dict)`. Write `NewAlertAdapter` that adapts it to `AlertNotifier`. Do not modify the library or `AlertNotifier`.

2. **Decorator — retry.** Write a `RetryDecorator` that wraps any `AlertNotifier`. If `notify()` raises an exception, retry up to `max_retries` times (default 3) with a 0.1s delay between attempts, then re-raise on final failure.

3. **Repository — SQLite backend.** Write a `SqliteProductRepository` that implements `ProductRepository` using the standard-library `sqlite3` module. Show `add` and `get`. (`InMemoryProductRepository` should remain fully functional for tests.)

4. **Decorator vs. inheritance.** Explain why using Decorator to add logging + timing + retry to a notifier is better than creating `LoggingEmailNotifier`, `TimingEmailNotifier`, `LoggingTimingEmailNotifier`, etc.

5. **Repository in tests.** Write a `pytest` test for `ProductService.get_expensive` using `InMemoryProductRepository`. No real database needed.

(Solutions: ../solutions/week-07/day-3.md)

---

## Self-check

- [ ] I can implement Adapter using composition (not inheritance).
- [ ] I can chain multiple GoF Decorators and explain the call flow.
- [ ] I can define an abstract Repository and swap implementations.
- [ ] I can test domain services by injecting an in-memory repository.

---

## Supplementary links

- *Design Patterns* — GoF, Chapter 4 (Structural Patterns)
- [Python Patterns — Decorator](https://python-patterns.guide/gang-of-four/decorator-pattern/)
- [Repository Pattern](https://martinfowler.com/eaaCatalog/repository.html) — Fowler's PoEAA catalog
- [Python `Protocol` for structural subtyping](https://docs.python.org/3/library/typing.html#typing.Protocol)
