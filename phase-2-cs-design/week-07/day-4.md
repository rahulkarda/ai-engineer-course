# Phase 2 · Week 07 · Day 4 — Behavioral Patterns: Strategy & Observer

> Objective: Implement Strategy and Observer; distinguish them from each other and from simpler alternatives like callbacks or conditionals.  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

---

## Concept

Behavioral patterns deal with **how objects communicate and distribute responsibility**.

### Strategy

Defines a family of algorithms, encapsulates each one, and makes them interchangeable. Strategy lets the algorithm vary independently from the clients that use it.

**Use when:**
- Multiple variants of an algorithm exist and you want to switch them at runtime.
- A class has many conditional branches that select behaviour — each branch is a strategy.
- You want to eliminate `if algorithm == "X": ...` chains.

**Core structure:** Context holds a reference to a Strategy interface. Client injects the concrete strategy.

### Observer (Publish-Subscribe)

Defines a one-to-many dependency so that when one object (Subject/Publisher) changes state, all its dependents (Observers/Subscribers) are notified automatically.

**Use when:**
- A change in one object requires changing an unknown number of others.
- Objects should be able to notify other objects without making assumptions about those objects.
- You're implementing event systems, data-binding, or reactive pipelines.

**Relationship to modern patterns:** Python's `asyncio` events, LangGraph's node callbacks, and message-broker patterns (Kafka topics) are all Observer at scale.

---

## How it works internally

**Strategy in Python**: Functions are first-class, so a strategy can be a `Callable[[args], result]` rather than a full class. Use classes when the strategy needs state (e.g., memoization cache). Use callables for stateless algorithms.

**Observer and memory leaks**: If observers hold strong references to subjects (or vice versa), objects can't be garbage-collected. Python's `weakref` module addresses this. Event systems like `blinker` use weak references by default.

**Thread safety in Observer**: Notifying observers while the subscriber list is being modified causes bugs. Snapshot the list before iteration, or use a queue-based dispatch.

---

## Worked example

```python
"""
day4_behavioral.py
Strategy and Observer — e-commerce pricing and order-event system.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─── Strategy ────────────────────────────────────────────────────────────────

@dataclass
class CartItem:
    name: str
    price: float
    quantity: int


class DiscountStrategy(ABC):
    """Abstract strategy for computing a discount multiplier."""

    @abstractmethod
    def apply(self, subtotal: float) -> float:
        """Return discounted total."""
        ...


class NoDiscount(DiscountStrategy):
    def apply(self, subtotal: float) -> float:
        return subtotal


class PercentageDiscount(DiscountStrategy):
    def __init__(self, percent: float) -> None:
        if not (0 < percent < 100):
            raise ValueError("Percent must be in (0, 100)")
        self._multiplier = 1 - percent / 100

    def apply(self, subtotal: float) -> float:
        return subtotal * self._multiplier


class BuyTwoGetOneFree(DiscountStrategy):
    """For every 3 units, charge 2."""

    def apply(self, subtotal: float) -> float:
        # Simplified: discount is 1/3 of subtotal
        return subtotal * (2 / 3)


class ShoppingCart:
    """Context: uses a discount strategy, replaceable at runtime."""

    def __init__(self, strategy: DiscountStrategy | None = None) -> None:
        self._items: list[CartItem] = []
        self._strategy: DiscountStrategy = strategy or NoDiscount()

    def add(self, item: CartItem) -> None:
        self._items.append(item)

    def set_discount(self, strategy: DiscountStrategy) -> None:
        """Swap strategy at runtime — e.g., apply a promo code."""
        self._strategy = strategy

    def total(self) -> float:
        subtotal = sum(i.price * i.quantity for i in self._items)
        return self._strategy.apply(subtotal)


# ─── Functional variant ──────────────────────────────────────────────────────
# For stateless algorithms, a Callable is simpler than a class.

DiscountFn = Callable[[float], float]


def flat_rate_discount(amount: float) -> DiscountFn:
    """Return a strategy function that subtracts a flat amount."""
    def _apply(subtotal: float) -> float:
        return max(0.0, subtotal - amount)
    return _apply


# ─── Observer ────────────────────────────────────────────────────────────────

@dataclass
class OrderEvent:
    event_type: str   # "placed" | "shipped" | "cancelled"
    order_id: str
    payload: dict = field(default_factory=dict)


class OrderObserver(ABC):
    @abstractmethod
    def on_event(self, event: OrderEvent) -> None: ...


class EmailConfirmationObserver(OrderObserver):
    def on_event(self, event: OrderEvent) -> None:
        if event.event_type == "placed":
            log.info("[EMAIL] Confirmation for order %s", event.order_id)


class InventoryObserver(OrderObserver):
    def on_event(self, event: OrderEvent) -> None:
        if event.event_type in ("placed", "cancelled"):
            log.info("[INVENTORY] Adjusting stock for order %s (%s)",
                     event.order_id, event.event_type)


class FraudDetectionObserver(OrderObserver):
    def on_event(self, event: OrderEvent) -> None:
        amount = event.payload.get("amount", 0)
        if amount > 10_000:
            log.warning("[FRAUD] High-value order %s: %.2f", event.order_id, amount)


class OrderEventBus:
    """Subject: manages observers and fires events."""

    def __init__(self) -> None:
        self._observers: list[OrderObserver] = []

    def subscribe(self, observer: OrderObserver) -> None:
        self._observers.append(observer)

    def unsubscribe(self, observer: OrderObserver) -> None:
        self._observers.remove(observer)

    def publish(self, event: OrderEvent) -> None:
        log.info("[BUS] Publishing %s for order %s", event.event_type, event.order_id)
        for observer in list(self._observers):  # snapshot to allow unsubscribe during notify
            observer.on_event(event)


# ─── Demo ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Strategy
    cart = ShoppingCart()
    cart.add(CartItem("Widget", 10.0, 3))
    cart.add(CartItem("Gadget", 50.0, 1))
    log.info("No discount: %.2f", cart.total())           # 80.00

    cart.set_discount(PercentageDiscount(20))
    log.info("20%% discount: %.2f", cart.total())         # 64.00

    cart.set_discount(BuyTwoGetOneFree())
    log.info("B2G1 discount: %.2f", cart.total())         # ~53.33

    # Functional strategy
    flat_five = flat_rate_discount(5.0)
    log.info("Flat $5 off: %.2f", flat_five(80.0))        # 75.00

    # Observer
    bus = OrderEventBus()
    bus.subscribe(EmailConfirmationObserver())
    bus.subscribe(InventoryObserver())
    bus.subscribe(FraudDetectionObserver())

    bus.publish(OrderEvent("placed", "ORD-001", {"amount": 80.0}))
    bus.publish(OrderEvent("placed", "ORD-002", {"amount": 15_000.0}))
    bus.publish(OrderEvent("shipped", "ORD-001"))
    bus.publish(OrderEvent("cancelled", "ORD-001"))
```

---

## Exercises

1. **New strategy.** Implement a `TieredDiscount` strategy: 0% off below $50, 10% off $50–$100, 20% off above $100. Add it to the cart demo.

2. **Functional vs. class strategy.** When would you prefer a class-based `DiscountStrategy` over a `DiscountFn` callable? Give two concrete reasons.

3. **Observer filter.** Add a `channel` parameter to `subscribe()` so observers only receive events of a specific type. Rewrite `publish()` accordingly.

4. **Observer weak reference.** Explain what happens to an `OrderObserver` instance held only by the event bus (no other references) if Python's GC runs. How would you fix this with `weakref.WeakSet`?

5. **Strategy + Observer together.** Design a `PricingEngine` class that: (a) holds a list of `DiscountStrategy` objects applied in order, and (b) publishes a `PricingEvent` on the bus whenever the final price is computed. Sketch the class (no implementation required).

(Solutions: ../solutions/week-07/day-4.md)

---

## Self-check

- [ ] I can implement Strategy using both an ABC and a Callable.
- [ ] I can wire an Observer event bus and subscribe/unsubscribe observers.
- [ ] I can explain the difference between Strategy (algorithm swap) and Observer (event notification).
- [ ] I understand the weak-reference concern in long-lived event buses.

---

## Supplementary links

- *Design Patterns* — GoF, Chapter 5.9 (Strategy), 5.7 (Observer)
- [Python Patterns — Strategy](https://python-patterns.guide/gang-of-four/strategy/)
- [Observer vs. Event Bus](https://refactoring.guru/design-patterns/observer) — Refactoring.guru
- [Python `weakref` docs](https://docs.python.org/3/library/weakref.html)
