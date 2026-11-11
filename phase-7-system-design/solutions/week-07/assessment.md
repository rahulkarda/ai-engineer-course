# Answer Key — Week 07 Assessment

---

## Part 1: Conceptual Answers

**1. (3 pts) The six steps of the LLD method:**

| Step | Name | Purpose |
|---|---|---|
| 1 | Clarify requirements | Confirm functional scope; identify what is and is not in scope. Avoids over-engineering. |
| 2 | Identify entities and relationships | Extract nouns from requirements; determine 1-to-1, 1-to-many, many-to-many associations. |
| 3 | Define interfaces (ABCs) | State what each component *must do* before deciding *how* it does it. Enables DI and testability. |
| 4 | Design the data model | Choose value objects, entities, and aggregate roots. Decide what state each class owns. |
| 5 | Implement core logic | Fill in the concrete classes; keep each class under one responsibility. |
| 6 | Identify concurrency and edge cases | Review every public method for thread safety, error paths, and invariants. Add locks, validation. |

Each step is a gate: output of one step is the input to the next. Skipping step 3 (interfaces) leads to tightly coupled classes that cannot be tested independently.

---

**2. (3 pts) Open/Closed violation in `AreaCalculator`:**

Violation: `compute_area` uses a chain of `if/elif` on `shape["type"]`. Adding a new shape (e.g., `"ellipse"`) requires editing the body of `compute_area`, violating OCP — the method is not closed to modification.

Fix: define an abstract `Shape` interface with a `compute_area()` method. Each shape is a concrete class. `AreaCalculator.compute_area` accepts a `Shape` and delegates:

```python
from abc import ABC, abstractmethod
import math


class Shape(ABC):
    @abstractmethod
    def compute_area(self) -> float: ...


class Circle(Shape):
    def __init__(self, radius: float) -> None:
        self.radius = radius

    def compute_area(self) -> float:
        return math.pi * self.radius ** 2


class Rectangle(Shape):
    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height

    def compute_area(self) -> float:
        return self.width * self.height


class Triangle(Shape):
    def __init__(self, base: float, height: float) -> None:
        self.base = base
        self.height = height

    def compute_area(self) -> float:
        return 0.5 * self.base * self.height


class AreaCalculator:
    def compute_area(self, shape: Shape) -> float:
        return shape.compute_area()
```

`AreaCalculator.compute_area` is now closed to modification. Adding `Ellipse` requires only a new class, not a change to `AreaCalculator`.

---

**3. (3 pts) LSP violation analysis for `ReadOnlyStack`:**

The reviewer is **correct**. LSP states: if `S` is a subtype of `T`, then objects of type `T` may be replaced by objects of type `S` without altering the correctness of the program.

`Stack` declares a `push` method that callers depend on to add items. `ReadOnlyStack` inherits `Stack` but raises `NotImplementedError` in `push`. Any code that holds a `Stack` reference and calls `push` will receive an unexpected exception when handed a `ReadOnlyStack`. The precondition "push succeeds" is violated — a strengthened precondition (you must not call push) is imposed on callers, which LSP forbids.

Correct fix: do not make `ReadOnlyStack` a subtype of `Stack`. Use a separate `ReadableStack` interface that does not declare `push`. `ReadOnlyStack` implements `ReadableStack`; `Stack` also implements `ReadableStack` but additionally exposes `push`.

---

**4. (3 pts) Race condition in TokenBucketRateLimiter:**

**Specific race condition (check-then-act without a lock):**

Thread A calls `allow_request()`, reads `self.tokens == 1`, passes the `if self.tokens >= 1` check. Before Thread A decrements, Thread B also reads `self.tokens == 1`, also passes the check. Both threads decrement: `self.tokens` goes to `-1`. Both threads returned `True` — two requests allowed when only one token existed.

**Observed symptoms:** Under concurrent load, the rate limiter allows more requests per second than the configured capacity. The token counter goes negative, which can cause further logic errors downstream.

**Minimal fix:** hold the lock for the entire refill + check + decrement sequence:

```python
def allow_request(self) -> bool:
    with self._lock:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False
```

The lock must wrap `_refill` too, not just the decrement, because `_refill` reads and writes `self.tokens` and `self._last_refill`.

---

**5. (3 pts) Class diagram analysis:**

**a) Design pattern:**

This is the **Strategy pattern** (also sometimes called the **Bridge** pattern in this variant). `Notification` holds a reference to the `NotificationSender` interface and delegates the send operation to it. The concrete sender (`SmtpSender`, `SlackSender`) is injected at construction time.

More precisely this is **Dependency Injection via interface** — the `Notification` depends on the abstraction `NotificationSender`, not on any concrete class.

**b) Adding a `LoggingSender`:**

Use the **Decorator pattern**. `LoggingSender` wraps another `NotificationSender` and logs before forwarding:

```python
import logging
from notifications.interfaces import NotificationSender

logger = logging.getLogger(__name__)


class LoggingSender(NotificationSender):
    def __init__(self, wrapped: NotificationSender) -> None:
        self._wrapped = wrapped

    def send(self, msg: str) -> None:
        logger.info("Sending notification: %s", msg)
        self._wrapped.send(msg)
        logger.info("Notification sent successfully")
```

Constructor signature: `def __init__(self, wrapped: NotificationSender) -> None`.

**c) DIP analysis:**

`Notification` does **not** depend on `SmtpSender`. It depends on `NotificationSender` (the interface). This satisfies the Dependency Inversion Principle: the high-level class (`Notification`) depends on an abstraction, not a concretion. `SmtpSender` depends on the same abstraction (by implementing it). Both high-level and low-level modules depend on the abstraction, not on each other.

---

**6. (3 pts) Singleton — Lock vs. module-level initialization:**

**Lock-based approach:**

```python
import threading

_instance = None
_lock = threading.Lock()

class AppLogger:
    @classmethod
    def get_instance(cls) -> "AppLogger":
        global _instance
        with _lock:
            if _instance is None:
                _instance = cls()
        return _instance
```

Appropriate when the singleton must be created lazily (construction is expensive and you do not want it at import time), or when the singleton depends on runtime configuration that is not available at module load.

**Module-level approach:**

```python
# module: app_logger.py
class AppLogger: ...

_instance = AppLogger()

def get_instance() -> AppLogger:
    return _instance
```

Appropriate when construction is cheap and safe at import time. Python's import lock guarantees that module-level code executes exactly once per interpreter, so the singleton is thread-safe by construction.

**Main risk of lock-based approach the module-level approach avoids:** double-checked locking bugs. If the lock is accidentally dropped (e.g., refactored away), or if the lock granularity is wrong, two threads can both observe `_instance is None` and both proceed to construct. The module-level approach has no such bug surface — the interpreter guarantees single execution.

---

**7. (3 pts) Vending machine — State vs. Strategy:**

**a) Use the State pattern.**

The vending machine's behavior changes depending on its current state — the same input (coin inserted) means something different in `IDLE` vs. `ITEM_SELECTED`. The State pattern models this by encoding behavior in state objects rather than in a chain of `if/elif` in the context class. Strategy is for selecting among interchangeable algorithms when the context itself does not change — it does not model state transitions.

**b) Python class structure:**

```python
from abc import ABC, abstractmethod


class VendingMachineState(ABC):
    @abstractmethod
    def select_item(self, machine: "VendingMachine", item_id: str) -> None: ...

    @abstractmethod
    def insert_coin(self, machine: "VendingMachine", amount: float) -> None: ...

    @abstractmethod
    def cancel(self, machine: "VendingMachine") -> None: ...

    @abstractmethod
    def dispense(self, machine: "VendingMachine") -> None: ...


class IdleState(VendingMachineState): ...
class ItemSelectedState(VendingMachineState): ...
class PaymentPendingState(VendingMachineState): ...
class DispensingState(VendingMachineState): ...


class VendingMachine:
    def __init__(self) -> None:
        self._state: VendingMachineState = IdleState()

    def set_state(self, state: VendingMachineState) -> None:
        self._state = state

    def select_item(self, item_id: str) -> None:
        self._state.select_item(self, item_id)

    def insert_coin(self, amount: float) -> None:
        self._state.insert_coin(self, amount)

    def cancel(self) -> None:
        self._state.cancel(self)

    def dispense(self) -> None:
        self._state.dispense(self)
```

**c) Invalid input in a given state:**

Each state class implements all methods. For inputs that are invalid in that state, the method raises an `InvalidOperationError` or silently no-ops (depending on design). Example:

```python
class IdleState(VendingMachineState):
    def insert_coin(self, machine: VendingMachine, amount: float) -> None:
        raise InvalidOperationError("Select an item before inserting coins")
```

This is cleaner than a monolithic `if self.state == IDLE: raise ...` block in the context class — each invalid transition is co-located with the state that rejects it, and adding a new state does not require modifying existing states.

---

**8. (4 pts) OCP for OrderProcessor discount system:**

**Abstract interface:**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    name: str
    price: float
    quantity: int = 1


class DiscountStrategy(ABC):
    @abstractmethod
    def apply(self, items: list[Item]) -> float:
        """Return the discount amount (positive number to subtract from total)."""
        ...
```

**One concrete implementation:**

```python
class PercentageDiscount(DiscountStrategy):
    def __init__(self, percent: float) -> None:
        if not 0.0 <= percent <= 100.0:
            raise ValueError(f"Percent must be 0-100, got {percent}")
        self._percent = percent

    def apply(self, items: list[Item]) -> float:
        subtotal = sum(item.price * item.quantity for item in items)
        return subtotal * (self._percent / 100.0)


class FlatDiscount(DiscountStrategy):
    def __init__(self, amount: float) -> None:
        self._amount = amount

    def apply(self, items: list[Item]) -> float:
        return self._amount


class BuyOneGetOneDiscount(DiscountStrategy):
    """Every second unit of each item is free."""

    def apply(self, items: list[Item]) -> float:
        discount = 0.0
        for item in items:
            free_units = item.quantity // 2
            discount += free_units * item.price
        return discount
```

**`OrderProcessor` — does not change:**

```python
class OrderProcessor:
    def __init__(self, discount: DiscountStrategy | None = None) -> None:
        self._discount = discount

    def calculate_total(self, items: list[Item]) -> float:
        subtotal = sum(item.price * item.quantity for item in items)
        discount = self._discount.apply(items) if self._discount else 0.0
        return max(0.0, subtotal - discount)
```

`OrderProcessor.calculate_total` does not change when a new discount type is added. The new type is a new class implementing `DiscountStrategy`, injected at construction. The class is open for extension (new discount strategies) and closed for modification (existing code unchanged).

---

## Part 2: Parking Lot Design Solution

### Deliverable 1 — Class diagram

```
Interfaces (ABCs)
─────────────────
+─────────────────────+     +──────────────────────+     +──────────────────────+
│  SpotAllocator      │     │   FeeCalculator       │     │  PaymentProcessor    │
│  (ABC)              │     │   (ABC)               │     │  (ABC)               │
+─────────────────────+     +──────────────────────+     +──────────────────────+
│ +find_spot(         │     │ +compute_fee(         │     │ +process_payment(    │
│   vehicle: VehicleType│   │   ticket: Ticket,     │     │   amount: float,     │
│  ) -> ParkingSpot?  │     │   exit: datetime      │     │   ticket_id: str     │
│ +release_spot(      │     │  ) -> float           │     │  ) -> bool           │
│   spot: ParkingSpot │     +──────────────────────+     +──────────────────────+
│  ) -> None          │              ^                             ^
│ +available_count(   │              |                             |
│   floor: int        │    +─────────────────────+    +────────────────────────+
│  ) -> int           │    │ HourlyFeeCalculator  │    │ InMemoryPaymentProc    │
+─────────────────────+    +─────────────────────+    +────────────────────────+
         ^
         |
+─────────────────────+
│ DefaultSpotAllocator│
+─────────────────────+

+──────────────────────+
│  Clock (ABC)         │
│ +now() -> datetime   │
+──────────────────────+
         ^
         |
+──────────────────────+
│  SystemClock         │
+──────────────────────+

Value objects / Entities
────────────────────────
+──────────────────────+   1    *   +──────────────────────+
│  ParkingLot          │───────────>│  ParkingFloor         │
│ -allocator           │           │ -floor_num: int        │
│ -fee_calculator      │           │ -spots: list[ParkingSpot]│
│ -payment_processor   │           +──────────────────────+
│ -clock               │                     | 1
│ -active_tickets      │                     | *
│ -gates: list[Gate]   │           +──────────────────────+
│                      │           │  ParkingSpot          │
│ +park(...)           │           │ -floor: int           │
│ +exit_vehicle(...)   │           │ -index: int           │
│ +available_spots()   │           │ -spot_type: SpotType  │
+──────────────────────+           │ -is_occupied: bool    │
         |                         +──────────────────────+
         | 1
         | *
+──────────────────────+
│  Gate                 │
│ -gate_id: int         │
│ -gate_type: GateType  │
│ -semaphore: Semaphore │
│ +enter(vehicle_type)  │
│ +exit(ticket_id)      │
+──────────────────────+

+──────────────────────+
│  Ticket               │
│ -ticket_id: str       │
│ -plate: str           │
│ -vehicle_type         │
│ -spot: ParkingSpot    │
│ -entry_time: datetime │
+──────────────────────+

Enums
─────
VehicleType: MOTORCYCLE | CAR | TRUCK
SpotType:    MOTORCYCLE | STANDARD | OVERSIZED
GateType:    ENTRY | EXIT

Dependency arrows (→ = depends on):
  ParkingLot → SpotAllocator (interface)
  ParkingLot → FeeCalculator (interface)
  ParkingLot → PaymentProcessor (interface)
  ParkingLot → Clock (interface)
  Gate → ParkingLot
```

---

### Deliverable 2 — Python implementation

```python
"""
parking_lot/models.py — Value objects and enums.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from enum import Enum, auto


class VehicleType(str, Enum):
    MOTORCYCLE = "motorcycle"
    CAR = "car"
    TRUCK = "truck"


class SpotType(str, Enum):
    MOTORCYCLE = "motorcycle"
    STANDARD = "standard"
    OVERSIZED = "oversized"


class GateType(str, Enum):
    ENTRY = "entry"
    EXIT = "exit"


# Map vehicle type to the spot types it is allowed to use.
VEHICLE_SPOT_COMPATIBILITY: dict[VehicleType, set[SpotType]] = {
    VehicleType.MOTORCYCLE: {SpotType.MOTORCYCLE},
    VehicleType.CAR: {SpotType.STANDARD},
    VehicleType.TRUCK: {SpotType.OVERSIZED},
}


@dataclass(frozen=True)
class ParkingSpot:
    floor: int
    index: int
    spot_type: SpotType
    is_occupied: bool = False

    def occupy(self) -> "ParkingSpot":
        return ParkingSpot(
            floor=self.floor,
            index=self.index,
            spot_type=self.spot_type,
            is_occupied=True,
        )

    def vacate(self) -> "ParkingSpot":
        return ParkingSpot(
            floor=self.floor,
            index=self.index,
            spot_type=self.spot_type,
            is_occupied=False,
        )


@dataclass
class Ticket:
    ticket_id: str
    plate: str
    vehicle_type: VehicleType
    spot: ParkingSpot
    entry_time: datetime.datetime
```

```python
"""
parking_lot/interfaces.py — Abstract interfaces.
"""

from __future__ import annotations

import datetime
from abc import ABC, abstractmethod
from typing import Optional

from .models import ParkingSpot, SpotType, Ticket, VehicleType


class SpotAllocator(ABC):
    @abstractmethod
    def find_spot(self, vehicle_type: VehicleType) -> Optional[ParkingSpot]:
        """Find and reserve a spot. Returns None if lot is full."""
        ...

    @abstractmethod
    def release_spot(self, spot: ParkingSpot) -> None:
        """Mark the spot as available."""
        ...

    @abstractmethod
    def available_count(self, floor: int, spot_type: SpotType) -> int:
        """Count free spots on a floor for a given spot type."""
        ...


class FeeCalculator(ABC):
    @abstractmethod
    def compute_fee(self, ticket: Ticket, exit_time: datetime.datetime) -> float:
        ...


class PaymentProcessor(ABC):
    @abstractmethod
    def process_payment(self, amount: float, ticket_id: str) -> bool:
        """Return True if payment succeeded."""
        ...


class Clock(ABC):
    @abstractmethod
    def now(self) -> datetime.datetime:
        ...
```

```python
"""
parking_lot/allocator.py — DefaultSpotAllocator implementation.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

from .interfaces import SpotAllocator
from .models import (
    VEHICLE_SPOT_COMPATIBILITY,
    ParkingSpot,
    SpotType,
    VehicleType,
)

logger = logging.getLogger(__name__)


@dataclass
class FloorConfig:
    floor: int
    spot_configs: list[tuple[int, SpotType]]  # (count, spot_type) pairs


class DefaultSpotAllocator(SpotAllocator):
    """
    First-fit allocator. Floor and spot-type constraints come from FloorConfig
    objects passed at construction — no hardcoded conditionals.

    Thread-safe via a per-floor RLock.
    """

    def __init__(self, floors: list[FloorConfig]) -> None:
        # spots[floor_num][index] = ParkingSpot (mutable replacement pattern)
        self._spots: dict[int, list[ParkingSpot]] = {}
        self._locks: dict[int, threading.RLock] = {}

        for floor_cfg in floors:
            spot_list: list[ParkingSpot] = []
            idx = 0
            for count, stype in floor_cfg.spot_configs:
                for _ in range(count):
                    spot_list.append(ParkingSpot(
                        floor=floor_cfg.floor,
                        index=idx,
                        spot_type=stype,
                        is_occupied=False,
                    ))
                    idx += 1
            self._spots[floor_cfg.floor] = spot_list
            self._locks[floor_cfg.floor] = threading.RLock()

    def find_spot(self, vehicle_type: VehicleType) -> Optional[ParkingSpot]:
        allowed_types = VEHICLE_SPOT_COMPATIBILITY[vehicle_type]
        for floor_num, spots in self._spots.items():
            with self._locks[floor_num]:
                for i, spot in enumerate(spots):
                    if spot.spot_type in allowed_types and not spot.is_occupied:
                        occupied = spot.occupy()
                        spots[i] = occupied
                        logger.debug("Reserved %s", occupied)
                        return occupied
        return None

    def release_spot(self, spot: ParkingSpot) -> None:
        with self._locks[spot.floor]:
            spots = self._spots[spot.floor]
            spots[spot.index] = spots[spot.index].vacate()
            logger.debug("Released floor=%d index=%d", spot.floor, spot.index)

    def available_count(self, floor: int, spot_type: SpotType) -> int:
        with self._locks[floor]:
            return sum(
                1 for s in self._spots[floor]
                if s.spot_type == spot_type and not s.is_occupied
            )
```

```python
"""
parking_lot/pricing.py — HourlyFeeCalculator implementation.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass

from .interfaces import FeeCalculator
from .models import Ticket, VehicleType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HourlyRate:
    vehicle_type: VehicleType
    rate_per_hour: float
    minimum_fee: float


class HourlyFeeCalculator(FeeCalculator):
    """
    Fee = max(hourly_rate * hours, minimum_fee).
    Rates are injected at construction — no hardcoded pricing.
    """

    def __init__(self, rates: list[HourlyRate]) -> None:
        self._rates: dict[VehicleType, HourlyRate] = {
            r.vehicle_type: r for r in rates
        }

    def compute_fee(self, ticket: Ticket, exit_time: datetime.datetime) -> float:
        rate_cfg = self._rates[ticket.vehicle_type]
        duration_hours = (
            exit_time - ticket.entry_time
        ).total_seconds() / 3600.0
        fee = rate_cfg.rate_per_hour * duration_hours
        result = max(fee, rate_cfg.minimum_fee)
        logger.debug(
            "Fee for %s: %.2f (%.2f hrs at %.2f/hr, min=%.2f)",
            ticket.ticket_id,
            result,
            duration_hours,
            rate_cfg.rate_per_hour,
            rate_cfg.minimum_fee,
        )
        return result
```

```python
"""
parking_lot/payment.py — InMemoryPaymentProcessor (always approves in tests).
"""

from __future__ import annotations

import logging

from .interfaces import PaymentProcessor

logger = logging.getLogger(__name__)


class InMemoryPaymentProcessor(PaymentProcessor):
    """
    Simulates successful payment processing in memory.
    In production this would call a real payment gateway.
    """

    def __init__(self) -> None:
        self._processed: dict[str, float] = {}

    def process_payment(self, amount: float, ticket_id: str) -> bool:
        self._processed[ticket_id] = amount
        logger.info("Payment of %.2f processed for ticket %s", amount, ticket_id)
        return True

    @property
    def total_collected(self) -> float:
        return sum(self._processed.values())
```

```python
"""
parking_lot/clock.py — Clock implementations.
"""

from __future__ import annotations

import datetime

from .interfaces import Clock


class SystemClock(Clock):
    def now(self) -> datetime.datetime:
        return datetime.datetime.now(datetime.timezone.utc)


class FixedClock(Clock):
    """Deterministic clock for tests. Advances only when explicitly told to."""

    def __init__(self, ts: datetime.datetime | None = None) -> None:
        self._ts = ts or datetime.datetime(
            2025, 1, 1, 9, 0, 0, tzinfo=datetime.timezone.utc
        )

    def advance(self, seconds: float) -> None:
        self._ts += datetime.timedelta(seconds=seconds)

    def now(self) -> datetime.datetime:
        return self._ts
```

```python
"""
parking_lot/lot.py — ParkingLot orchestrator.

Design decisions:
  - ParkingLot depends only on interfaces (SpotAllocator, FeeCalculator,
    PaymentProcessor, Clock). It orchestrates without knowing implementation details.
  - Thread safety: a single RLock guards active_tickets and the ticket counter.
    Spot-level locking is delegated to DefaultSpotAllocator, which uses per-floor locks.
    This avoids a global lock that would serialize all gate operations.
  - Tradeoff: per-floor locking in the allocator allows floor 1 and floor 2 operations
    to proceed concurrently. The ParkingLot lock only serializes ticket bookkeeping,
    which is O(1) and very brief.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from .interfaces import Clock, FeeCalculator, PaymentProcessor, SpotAllocator
from .models import SpotType, Ticket, VehicleType

logger = logging.getLogger(__name__)


class ParkingLot:
    """
    Orchestrates vehicle entry and exit.
    All business logic is delegated to injected collaborators.
    Thread-safe via RLock on ticket registry.
    """

    def __init__(
        self,
        allocator: SpotAllocator,
        fee_calculator: FeeCalculator,
        payment_processor: PaymentProcessor,
        clock: Clock,
    ) -> None:
        self._allocator = allocator
        self._fee_calculator = fee_calculator
        self._payment_processor = payment_processor
        self._clock = clock
        self._active_tickets: dict[str, Ticket] = {}
        self._ticket_counter: int = 0
        self._lock = threading.RLock()
        self._revenue: float = 0.0

    def _next_ticket_id(self, spot_floor: int, spot_index: int) -> str:
        self._ticket_counter += 1
        return f"T{self._ticket_counter:06d}-F{spot_floor}-S{spot_index:03d}"

    def park(self, vehicle_type: VehicleType, plate: str) -> Optional[str]:
        """
        Attempt to park a vehicle.
        Returns ticket_id on success, None if no spot is available.
        Thread-safe.
        """
        spot = self._allocator.find_spot(vehicle_type)
        if spot is None:
            logger.warning("No spot available for %s (%s)", plate, vehicle_type.value)
            return None

        with self._lock:
            ticket_id = self._next_ticket_id(spot.floor, spot.index)
            ticket = Ticket(
                ticket_id=ticket_id,
                plate=plate,
                vehicle_type=vehicle_type,
                spot=spot,
                entry_time=self._clock.now(),
            )
            self._active_tickets[ticket_id] = ticket

        logger.info("Parked %s (%s) → ticket=%s floor=%d spot=%d",
                    plate, vehicle_type.value, ticket_id, spot.floor, spot.index)
        return ticket_id

    def exit_vehicle(self, ticket_id: str) -> float:
        """
        Process vehicle exit: compute fee, collect payment, release spot.
        Returns the fee charged.
        Raises KeyError if ticket_id is unknown.
        Raises RuntimeError if payment fails.
        Thread-safe.
        """
        with self._lock:
            if ticket_id not in self._active_tickets:
                raise KeyError(f"Unknown ticket: {ticket_id!r}")
            ticket = self._active_tickets.pop(ticket_id)

        exit_time = self._clock.now()
        fee = self._fee_calculator.compute_fee(ticket, exit_time)

        if not self._payment_processor.process_payment(fee, ticket_id):
            # Re-insert ticket — payment failed, vehicle has not left.
            with self._lock:
                self._active_tickets[ticket_id] = ticket
            raise RuntimeError(f"Payment failed for ticket {ticket_id!r}")

        self._allocator.release_spot(ticket.spot)

        with self._lock:
            self._revenue += fee

        logger.info("Exit ticket=%s plate=%s fee=%.2f", ticket_id, ticket.plate, fee)
        return fee

    def available_spots(self) -> dict[int, dict[SpotType, int]]:
        """
        Return {floor_num: {SpotType: count}} for all floors and spot types.
        Builds snapshot — no lock needed (allocator is independently thread-safe).
        """
        from .allocator import DefaultSpotAllocator  # avoid circular at module level

        result: dict[int, dict[SpotType, int]] = {}
        if isinstance(self._allocator, DefaultSpotAllocator):
            floors = list(self._allocator._spots.keys())
            for floor_num in floors:
                result[floor_num] = {}
                for stype in SpotType:
                    result[floor_num][stype] = self._allocator.available_count(
                        floor_num, stype
                    )
        return result

    @property
    def revenue(self) -> float:
        with self._lock:
            return self._revenue
```

---

### Deliverable 3 — Concurrency explanation

`ParkingLot.park` and `exit_vehicle` can be called simultaneously from up to 20 gate threads. The approach is **layered locking at two granularities**:

1. **Per-floor RLock in `DefaultSpotAllocator`**: `find_spot` and `release_spot` acquire only the lock for the floor being searched. Entry gates competing for spots on different floors do not block each other. This allows most concurrent park/exit operations to proceed in parallel, since in a 5-floor lot only 1 in 5 operations contend on the same lock.

2. **Single RLock in `ParkingLot`** for `_active_tickets` and `_ticket_counter`: this is an O(1) dict operation held for microseconds. Even with 20 concurrent threads this lock is never held long enough to become a bottleneck.

**Tradeoff**: A single global lock on the entire `ParkingLot` (including spot search) would be simpler to reason about but would serialize all 20 gates — at peak, throughput would be limited to one park/exit at a time. Per-floor locking allows up to `num_floors` concurrent spot allocations. Per-spot locking would be the finest granularity (allowing N concurrent allocations for N spots) but adds significant overhead for lock creation and the risk of deadlock if multiple spots are acquired in one transaction.

The chosen approach (per-floor allocator lock + single ticket registry lock) is the correct tradeoff for the stated scale: 20 concurrent gates on 10 floors means at most 2 gates per floor on average, so contention is low and the implementation is simple.

---

## Part 3: Rate Limiter Bug Fixes

### Bug analysis

**Bug A — `allow_request` returns wrong value after decrement (line: `return self.tokens >= 0`)**

Root cause: after `self.tokens -= 1`, if `tokens` was exactly `1.0`, it becomes `0.0`. The guard checks `>= 1` before decrementing, but the return expression `self.tokens >= 0` returns `True` even when tokens is `0` — meaning the function returns `True` again for a request it should have already consumed. This is an off-by-one logic error.

Symptom: callers see `True` returned even when the budget should be exhausted.

**Bug B — `_refill` does not cap tokens at capacity (line: `self.tokens += elapsed * self.rate`)**

Root cause: if the service is idle for a long time (e.g., 1000 seconds) and `rate=100`, `tokens` grows to `100,000` — far above `capacity`. All subsequent requests are allowed through in a burst until tokens drain, effectively defeating the rate limiter.

Symptom: after a long quiet period, a flood of requests are all allowed through.

**Bug C — `_refill` is called outside the lock (line: `self._refill()` before `with self._lock`)**

Root cause: `_refill` reads and writes `self.tokens` and `self._last_refill` without holding the lock. Two threads can both call `_refill` concurrently, both computing `elapsed` from the same `_last_refill`, both adding `elapsed * rate` — double-counting the refill. Then both enter `with self._lock`, each decrementing `tokens`, potentially going negative.

Symptom: under load, more requests are allowed than the bucket permits.

**Additional type bug (Bug C from the assessment — type checker complaint):**

`self.tokens = capacity` where `capacity: int` — `tokens` is assigned an `int` but `self.tokens += elapsed * self.rate` assigns a `float`. The type checker infers `tokens: int` from the constructor assignment and flags the mixed arithmetic. Fix: annotate `tokens` as `float` and initialize `self.tokens: float = float(capacity)`.

---

### Corrected implementation

```python
"""
rate_limiter.py — Corrected TokenBucketRateLimiter.
"""

from __future__ import annotations

import threading
import time


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter.
    Refills at `rate` tokens per second up to `capacity` tokens.
    Each call to allow_request() consumes one token.
    Returns True if the request is allowed, False if rate-limited.

    Thread-safe: all state mutations hold self._lock.
    """

    def __init__(self, capacity: int, rate: float) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if rate <= 0.0:
            raise ValueError(f"rate must be positive, got {rate}")
        self.capacity: int = capacity
        self.rate: float = rate
        self.tokens: float = float(capacity)   # Fix C: explicitly float
        self._last_refill: float = time.monotonic()
        self._lock: threading.Lock = threading.Lock()

    def _refill(self) -> None:
        """Must be called with self._lock held."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now
        self.tokens = min(                     # Fix B: cap at capacity
            float(self.capacity),
            self.tokens + elapsed * self.rate,
        )

    def allow_request(self) -> bool:
        with self._lock:                       # Fix A: lock wraps entire operation
            self._refill()                     # Fix A+C: refill inside lock
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True                    # Fix A: correct return logic
            return False
```

Key changes:

| Bug | Fix | Line |
|---|---|---|
| A — wrong return value | Return `True` only inside the `if tokens >= 1` branch; return `False` after | `allow_request` |
| A — race condition | Move `_refill()` inside `with self._lock` | `allow_request` |
| B — no cap on refill | `self.tokens = min(capacity, ...)` | `_refill` |
| C — type annotation | Declare `self.tokens: float = float(capacity)` | `__init__` |

---

## Part 4: Singleton Logger Fixes

### Issue 1 — Thread safety: double instantiation scenario

**Scenario:** Thread A and Thread B both call `AppLogger()` for the first time simultaneously.

1. Thread A checks `_instance is None` → True.
2. Thread B context-switches in, checks `_instance is None` → still True (A has not yet assigned).
3. Thread B calls `super().__new__(cls)` and assigns to `_instance`.
4. Thread A resumes, also calls `super().__new__(cls)` — creates a second object. The second assignment clobbers the first.

Result: two `AppLogger` objects exist momentarily; whichever `_instance` points to last wins, but any code that captured the first reference now holds a stale object.

**Fix:**

```python
import logging
import threading

_instance: "AppLogger | None" = None
_lock: threading.Lock = threading.Lock()


class AppLogger:
    def __new__(cls) -> "AppLogger":
        global _instance
        if _instance is None:                  # fast path (no lock)
            with _lock:
                if _instance is None:          # double-check inside lock
                    _instance = super().__new__(cls)
                    _instance.logger = logging.getLogger("app")
                    _instance.logger.setLevel(logging.DEBUG)
        return _instance
```

The double-checked locking pattern avoids acquiring the lock on every call (expensive under high concurrency) while still protecting the critical section.

---

### Issue 2 — Missing interface: problem for testing

Without an ABC, any class that depends on `AppLogger` (e.g., a service that calls `get_logger().log(...)`) cannot be tested with a fake logger — it must either use the real `AppLogger` (which writes to the real log file/handler) or monkeypatch.

**`Logger` ABC:**

```python
import logging
from abc import ABC, abstractmethod
from typing import Union


class Logger(ABC):
    @abstractmethod
    def log(self, message: str) -> None:
        """Log a message at INFO level."""
        ...

    @abstractmethod
    def set_level(self, level: Union[int, str]) -> None:
        """Set the logging level. Accepts int (logging.DEBUG) or str ('DEBUG')."""
        ...


class AppLogger(Logger):
    ...
```

In tests, inject a `FakeLogger(Logger)` that stores messages in a list rather than writing to disk.

---

### Issue 3 — Missing type hints

```python
import logging
from typing import Union


class AppLogger:

    def log(self, message: str) -> None:
        self.logger.info(message)

    def set_level(self, level: Union[int, str]) -> None:
        self.logger.setLevel(level)
```

`Union[int, str]` matches `logging.setLevel`'s accepted types. In Python 3.10+, this can be written `int | str`.

---

### Issue 4 — `get_logger()` calling `AppLogger()` each time

**Subtle bug scenario:** A caller stores a reference:

```python
logger_a = get_logger()
logger_a.set_level(logging.WARNING)

# Later, in test teardown:
logger_b = get_logger()
# logger_b IS logger_a due to __new__, but the intent is hidden.
# If __new__ is ever changed to not return the singleton (e.g., during refactoring),
# set_level on logger_a would not affect logger_b.
```

The deeper issue: `AppLogger()` relies on `__new__` to enforce the singleton, but calling the constructor every time is semantically misleading — it looks like object creation. If `__new__` is ever refactored away (e.g., the class is replaced with a dataclass), callers silently get new instances.

**Cleaner pattern:** use a module-level instance directly, or expose a class method:

```python
import logging
import threading
from typing import Union

_lock: threading.Lock = threading.Lock()
_logger_instance: "AppLogger | None" = None


class AppLogger:
    def __init__(self) -> None:
        self._logger = logging.getLogger("app")
        self._logger.setLevel(logging.DEBUG)

    def log(self, message: str) -> None:
        self._logger.info(message)

    def set_level(self, level: Union[int, str]) -> None:
        self._logger.setLevel(level)

    @classmethod
    def get_instance(cls) -> "AppLogger":
        global _logger_instance
        if _logger_instance is None:
            with _lock:
                if _logger_instance is None:
                    _logger_instance = cls()
        return _logger_instance


def get_logger() -> AppLogger:
    return AppLogger.get_instance()
```

Now `get_logger()` explicitly communicates that it returns the shared instance. Construction logic is in `__init__` (normal Python), not `__new__` (unusual Python). The singleton enforcement is in a clearly named `get_instance` class method.
