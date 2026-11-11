# Solutions — Week 07 · Day 4

## Exercise 1 — TieredDiscount strategy

```python
from day4_behavioral import DiscountStrategy, ShoppingCart, CartItem
import logging

log = logging.getLogger(__name__)


class TieredDiscount(DiscountStrategy):
    """0% < $50  |  10% $50–$100  |  20% > $100"""

    def apply(self, subtotal: float) -> float:
        if subtotal < 50:
            return subtotal
        elif subtotal <= 100:
            return subtotal * 0.90
        else:
            return subtotal * 0.80


# Demo
cart = ShoppingCart(TieredDiscount())
cart.add(CartItem("Widget", 10.0, 1))   # $10 → no discount
log.info("Tier (<50): %.2f", cart.total())   # 10.00

cart2 = ShoppingCart(TieredDiscount())
cart2.add(CartItem("Gadget", 80.0, 1))  # $80 → 10% off
log.info("Tier (50-100): %.2f", cart2.total())  # 72.00

cart3 = ShoppingCart(TieredDiscount())
cart3.add(CartItem("Bigitem", 150.0, 1))  # $150 → 20% off
log.info("Tier (>100): %.2f", cart3.total())  # 120.00
```

---

## Exercise 2 — Class-based vs. functional strategy

Prefer a **class-based strategy** when:

1. **The strategy has state.** A `LoyaltyDiscount` that tracks how many orders a customer has placed needs instance variables. A plain callable would need a closure or a module-level mutable, both of which are harder to test and reset.

2. **The strategy needs configuration injected at construction time.** `PercentageDiscount(20)` stores `self._multiplier`. A callable could use a closure, but a class makes the configuration visible, inspectable, and overridable via subclassing.

Use a **callable** when the algorithm is truly stateless and takes no parameters beyond the inputs (e.g., `lambda subtotal: subtotal * 0.9`).

---

## Exercise 3 — Observer filter

```python
from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import TypeAlias
import logging

log = logging.getLogger(__name__)

EventType: TypeAlias = str


@dataclass
class OrderEvent:
    event_type: EventType
    order_id: str
    payload: dict = field(default_factory=dict)


class OrderObserver(ABC):
    @abstractmethod
    def on_event(self, event: OrderEvent) -> None: ...


class FilteringEventBus:
    """Observers can subscribe to specific event types."""

    def __init__(self) -> None:
        self._subs: dict[EventType | None, list[OrderObserver]] = {}

    def subscribe(
        self,
        observer: OrderObserver,
        channel: EventType | None = None,  # None = subscribe to all
    ) -> None:
        self._subs.setdefault(channel, []).append(observer)

    def publish(self, event: OrderEvent) -> None:
        # Notify observers subscribed to this specific type + those subscribed to all
        targets = (
            self._subs.get(event.event_type, []) +
            self._subs.get(None, [])
        )
        for observer in list(targets):
            observer.on_event(event)
```

---

## Exercise 4 — Weak reference explanation

**Problem:** The event bus holds a strong reference to every observer. If a subscriber goes out of scope (no other reference), the GC cannot collect it because the bus still points to it. Over time, the bus accumulates dead observers that fire callbacks on deleted objects — or worse, keep large objects alive indefinitely (memory leak).

**Fix with `weakref.WeakSet`:**

```python
import weakref
from day4_behavioral import OrderObserver, OrderEvent
import logging

log = logging.getLogger(__name__)


class WeakEventBus:
    def __init__(self) -> None:
        self._observers: weakref.WeakSet[OrderObserver] = weakref.WeakSet()

    def subscribe(self, observer: OrderObserver) -> None:
        self._observers.add(observer)

    def publish(self, event: OrderEvent) -> None:
        for obs in list(self._observers):  # list() snapshot; dead refs auto-removed
            obs.on_event(event)
```

`WeakSet` stores weak references; when the last strong reference to an observer disappears, it is automatically removed from the set on the next iteration.

**Caveat:** The caller must keep a strong reference to the observer for as long as it should receive events. This is often handled by storing observers in a list on the subscribing object.

---

## Exercise 5 — PricingEngine sketch

```python
from __future__ import annotations
from day4_behavioral import DiscountStrategy, OrderEvent, OrderEventBus
from dataclasses import dataclass


@dataclass
class PricingEvent:
    original: float
    final: float
    applied_strategies: list[str]


class PricingEngine:
    """
    Holds a pipeline of DiscountStrategy objects and publishes
    a PricingEvent after computing the final price.

    Patterns: Strategy (each discount), Observer (event bus notification).
    """

    def __init__(
        self,
        strategies: list[DiscountStrategy],
        event_bus: OrderEventBus,
    ) -> None:
        self._strategies = strategies
        self._bus = event_bus

    def compute(self, order_id: str, subtotal: float) -> float:
        price = subtotal
        names: list[str] = []
        for strategy in self._strategies:
            price = strategy.apply(price)
            names.append(type(strategy).__name__)
        event = OrderEvent(
            event_type="price_computed",
            order_id=order_id,
            payload={"original": subtotal, "final": price, "strategies": names},
        )
        self._bus.publish(event)
        return price
```
