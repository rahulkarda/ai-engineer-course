# Solutions — Phase 7 · Week 07 · Day 3

## Exercise 1 — Identify SOLID Violations

### The Code Under Review

```python
class ReportService:
    def generate(self, data: list[dict]) -> str:
        # Format
        lines = [f"{d['name']}: {d['value']}" for d in data]
        report = "\n".join(lines)
        # Save
        with open("report.txt", "w") as f:
            f.write(report)
        # Email
        import smtplib
        smtp = smtplib.SMTP("mail.example.com")
        smtp.sendmail("from@x.com", "to@x.com", report)
        return report

class PDFReport(ReportService):
    def generate(self, data: list[dict]) -> str:
        # Generates PDF but never saves or emails
        return "<pdf bytes>"
```

### Violations Found

#### Violation 1 — SRP (Single Responsibility Principle)

**Where:** `ReportService.generate` does three completely unrelated things in one method: formatting data into a string, persisting to a file, and sending an email. Each of those is a separate reason to change:
- The formatting team changes the template → edit `generate`.
- Ops moves from file to S3 → edit `generate`.
- The email team changes SMTP server → edit `generate`.

**Fix:** extract three collaborators with single responsibilities, inject them.

```python
import logging
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)


class ReportFormatter(ABC):
    @abstractmethod
    def format(self, data: list[dict]) -> str: ...


class ReportPersister(ABC):
    @abstractmethod
    def save(self, content: str) -> None: ...


class ReportNotifier(ABC):
    @abstractmethod
    def send(self, content: str) -> None: ...


class TextFormatter(ReportFormatter):
    def format(self, data: list[dict]) -> str:
        return "\n".join(f"{d['name']}: {d['value']}" for d in data)


class FileReportPersister(ReportPersister):
    def __init__(self, path: str = "report.txt") -> None:
        self._path = path

    def save(self, content: str) -> None:
        with open(self._path, "w", encoding="utf-8") as fh:
            fh.write(content)
        log.info("Report saved to %s", self._path)


class ReportService:
    def __init__(
        self,
        formatter: ReportFormatter,
        persister: ReportPersister,
        notifier: ReportNotifier,
    ) -> None:
        self._formatter = formatter
        self._persister = persister
        self._notifier = notifier

    def generate(self, data: list[dict]) -> str:
        report = self._formatter.format(data)
        self._persister.save(report)
        self._notifier.send(report)
        return report
```

Now `ReportService` has exactly one reason to change: how the pipeline is orchestrated.

---

#### Violation 2 — OCP (Open/Closed Principle)

**Where:** `ReportService.generate` hardcodes the file path `"report.txt"` and the SMTP host `"mail.example.com"`. To support a new persistence target (S3, database) or a new notification channel (Slack, SMS), you must modify `generate` directly.

**Fix:** Already addressed by the SRP fix — the formatter/persister/notifier are strategies (interfaces). Adding `S3ReportPersister` or `SlackNotifier` requires no change to `ReportService`.

```python
class S3ReportPersister(ReportPersister):
    """New persistence target — no change to ReportService needed."""
    def __init__(self, bucket: str, key: str) -> None:
        self._bucket = bucket
        self._key = key

    def save(self, content: str) -> None:
        # boto3.client('s3').put_object(...)
        log.info("Report uploaded to s3://%s/%s", self._bucket, self._key)
```

---

#### Violation 3 — LSP (Liskov Substitution Principle)

**Where:** `PDFReport` extends `ReportService` but its `generate` implementation silently skips saving and emailing — postconditions that callers of `ReportService.generate` expect to hold. Any code that calls `generate` on a `ReportService` reference and then checks the file system or email will break when handed a `PDFReport` instance.

```python
# Caller contract assumed by callers of ReportService:
service: ReportService = get_report_service()
service.generate(data)
# Caller assumes: file was saved, email was sent.
# PDFReport breaks both assumptions silently.
```

**Fix 1 (preferred):** Do not use inheritance. `PDFReport` is not a `ReportService` — it is a different kind of report generator. Use a shared interface instead of a base class.

```python
class ReportGenerator(ABC):
    @abstractmethod
    def generate(self, data: list[dict]) -> str: ...


class TextReportGenerator(ReportGenerator):
    def __init__(
        self,
        formatter: ReportFormatter,
        persister: ReportPersister,
        notifier: ReportNotifier,
    ) -> None:
        self._formatter = formatter
        self._persister = persister
        self._notifier = notifier

    def generate(self, data: list[dict]) -> str:
        report = self._formatter.format(data)
        self._persister.save(report)
        self._notifier.send(report)
        return report


class PDFReportGenerator(ReportGenerator):
    """PDF variant — no save/email in this generator's contract."""
    def generate(self, data: list[dict]) -> str:
        log.info("Generating PDF report for %d rows", len(data))
        return "<pdf bytes>"
```

Now callers that call `generate` on a `ReportGenerator` can substitute either implementation without surprise: both fulfill the contract of "return a formatted report string."

**Fix 2 (if inheritance is required):** make saving and emailing optional behaviors controlled by constructor flags, and document the narrowed contract clearly. Fix 1 is architecturally cleaner.

---

### Summary Table

| # | Violation | Location | Principle | Fix |
|---|-----------|----------|-----------|-----|
| 1 | `generate` does formatting + file I/O + SMTP in one method | `ReportService.generate` | SRP | Extract `ReportFormatter`, `ReportPersister`, `ReportNotifier`; inject via constructor |
| 2 | Hardcoded file path and SMTP host; new destinations require editing existing code | `ReportService.generate` | OCP | Strategy pattern (already solved by fix 1) |
| 3 | `PDFReport` silently skips save + email, breaking caller postconditions | `PDFReport.generate` | LSP | Replace inheritance with a shared `ReportGenerator` interface |

---

## Exercise 2 — Pattern Identification Drill

### 1. "The checkout process can use PayPal, Stripe, or bank transfer interchangeably."

**Pattern: Strategy**

The checkout's payment step is the algorithm that varies. Define a `PaymentStrategy` interface with a `charge(amount)` method; `PayPalStrategy`, `StripeStrategy`, and `BankTransferStrategy` each implement it. The `CheckoutService` holds a `PaymentStrategy` reference and calls `charge` — swapping strategies requires no change to `CheckoutService`.

Clue words: *"interchangeably"*, *"can use X or Y or Z"*.

```python
from abc import ABC, abstractmethod

class PaymentStrategy(ABC):
    @abstractmethod
    def charge(self, amount: float, reference: str) -> str: ...

class StripeStrategy(PaymentStrategy):
    def charge(self, amount: float, reference: str) -> str:
        # call Stripe SDK
        return f"stripe_txn_{reference}"

class CheckoutService:
    def __init__(self, payment: PaymentStrategy) -> None:
        self._payment = payment

    def process(self, amount: float, order_id: str) -> str:
        return self._payment.charge(amount, order_id)
```

---

### 2. "When an order status changes, the warehouse, customer, and analytics system all need to know."

**Pattern: Observer**

Three independent consumers react to one event (order status change). The `Order` is the Subject; `WarehouseObserver`, `CustomerNotifier`, and `AnalyticsCollector` are Observers. When `order.set_status(SHIPPED)` is called, it notifies all registered observers. Adding a fourth observer (e.g., `FraudDetector`) requires no change to `Order`.

Clue words: *"all need to know"*, *"when X changes"*, multiple distinct consumers.

```python
from abc import ABC, abstractmethod
from enum import Enum, auto

class OrderStatus(Enum):
    PENDING = auto()
    SHIPPED = auto()
    DELIVERED = auto()

class OrderObserver(ABC):
    @abstractmethod
    def on_status_change(self, order_id: str, new_status: OrderStatus) -> None: ...

class Order:
    def __init__(self, order_id: str) -> None:
        self.order_id = order_id
        self._status = OrderStatus.PENDING
        self._observers: list[OrderObserver] = []

    def subscribe(self, observer: OrderObserver) -> None:
        self._observers.append(observer)

    def set_status(self, new_status: OrderStatus) -> None:
        self._status = new_status
        for obs in self._observers:
            obs.on_status_change(self.order_id, new_status)
```

---

### 3. "The application has one database connection pool shared across all services."

**Pattern: Singleton**

A connection pool is a shared, stateful resource. Multiple instances would open redundant connections and waste DB server resources. The Singleton ensures exactly one pool exists, and all services acquire connections from the same pool.

Clue words: *"one … shared across all"*, "connection pool", "global config".

```python
import logging
from typing import Optional

log = logging.getLogger(__name__)

class ConnectionPool:
    _instance: Optional[ConnectionPool] = None

    def __init__(self, dsn: str, max_connections: int) -> None:
        self._dsn = dsn
        self._max = max_connections
        log.info("ConnectionPool created: dsn=%s max=%d", dsn, max_connections)

    @classmethod
    def get_instance(cls, dsn: str = "", max_connections: int = 10) -> ConnectionPool:
        if cls._instance is None:
            cls._instance = cls(dsn, max_connections)
        return cls._instance

    def acquire(self) -> object:
        log.debug("Acquiring connection from pool")
        return object()  # placeholder

    def release(self, conn: object) -> None:
        log.debug("Releasing connection to pool")
```

---

### 4. "Users can undo up to 10 text edits."

**Pattern: Command**

Each text edit is encapsulated as a `Command` object with `execute()` and `undo()` methods. The editor maintains a bounded history stack. `ctrl+z` calls `history.undo()`, which pops the last command and calls its `undo()`. This works for any edit type without a giant `if/elif` in the undo handler.

Clue words: *"undo"*, *"last N actions"*, *"history"*, *"audit log"* (Command enables replay too).

```python
from abc import ABC, abstractmethod
from collections import deque

class TextCommand(ABC):
    @abstractmethod
    def execute(self) -> None: ...

    @abstractmethod
    def undo(self) -> None: ...

class InsertTextCommand(TextCommand):
    def __init__(self, document: list[str], position: int, text: str) -> None:
        self._doc = document
        self._pos = position
        self._text = text

    def execute(self) -> None:
        self._doc.insert(self._pos, self._text)

    def undo(self) -> None:
        if self._pos < len(self._doc) and self._doc[self._pos] == self._text:
            self._doc.pop(self._pos)

class CommandHistory:
    def __init__(self, max_size: int = 10) -> None:
        self._stack: deque[TextCommand] = deque(maxlen=max_size)

    def execute(self, cmd: TextCommand) -> None:
        cmd.execute()
        self._stack.append(cmd)

    def undo(self) -> None:
        if self._stack:
            self._stack.pop().undo()
```

---

### 5. "The system creates notification objects (email, SMS, push) based on user preferences."

**Pattern: Factory Method**

The caller (`NotificationService`) needs to create a `Notification` object but should not know whether it is email, SMS, or push — that decision depends on user preferences stored at runtime. A `NotificationFactory.create(user_preferences)` centralizes the creation logic. New notification types add a new class and a new branch in the factory only — no other callers change.

Clue words: *"creates … based on"*, *"type"*, *"preferences"*, multiple interchangeable product types.

```python
from abc import ABC, abstractmethod

class Notification(ABC):
    @abstractmethod
    def send(self, message: str) -> None: ...

class EmailNotification(Notification):
    def send(self, message: str) -> None:
        log.info("[EMAIL] %s", message)

class SMSNotification(Notification):
    def send(self, message: str) -> None:
        log.info("[SMS] %s", message)

class PushNotification(Notification):
    def send(self, message: str) -> None:
        log.info("[PUSH] %s", message)

class NotificationFactory:
    _registry: dict[str, type[Notification]] = {
        "email": EmailNotification,
        "sms": SMSNotification,
        "push": PushNotification,
    }

    @classmethod
    def create(cls, channel: str) -> Notification:
        klass = cls._registry.get(channel.lower())
        if klass is None:
            raise ValueError(f"Unknown notification channel: {channel!r}")
        return klass()
```

---

## Exercise 3 — 25-Minute Timed Design: Ride-Sharing Dispatch System

### Time allocation used

- 0:00–2:00 — clarification
- 2:00–5:00 — entities + relationships
- 5:00–10:00 — interfaces (Steps 4-5)
- 10:00–25:00 — implementation of key classes

---

### Step 1 — Clarify

- **Actors**: Rider (requests a ride), Driver (accepts/completes), Dispatcher (algorithm).
- **Pricing**: multiple strategies (surge, flat, per-km). Swappable at runtime.
- **Matching**: pluggable algorithm (nearest, batch, priority queue). Swappable.
- **Notifications**: driver and rider both notified on match, ride start, ride end.
- **Logging**: all ride events logged to a single ride log (Singleton).
- **Constraints**: a driver can only carry 1 rider at a time; matching is synchronous for MVP.
- **Out of scope**: payment processing, route computation, driver ratings.

### Step 2 — Entities

```
Rider            — person requesting a ride
Driver           — registered driver with a vehicle
Location         — (lat, lng) value object
Ride             — the active trip: rider, driver, route, price
RideStatus       — enum: REQUESTED / MATCHED / IN_PROGRESS / COMPLETED / CANCELLED
PricingStrategy  — interface for fare calculation
MatchingStrategy — interface for driver selection
RideEventLog     — Singleton; append-only log of all ride events
RideEvent        — a timestamped event record
RideMatchingService — orchestrator
```

### Step 3 — Relationships

```
RideMatchingService 1 ---- 1 PricingStrategy
RideMatchingService 1 ---- 1 MatchingStrategy
RideMatchingService 1 ---- 1 RideEventLog    (Singleton)
RideMatchingService 1 ----< Driver
Ride               1 ---- 1 Rider
Ride               1 ---- 1 Driver
Ride               1 ---- 1 PricingStrategy  (locked at match time)
Driver             1 ---- 0..1 Ride           (active ride)
```

### Step 4 — Interfaces (10 minutes)

```
+--------------------------------+
|  <<interface>>                 |
|  PricingStrategy               |
+--------------------------------+
| + calculate_fare(              |
|     ride: Ride                 |
|   ) -> float                   |
+--------------------------------+

+--------------------------------+
|  <<interface>>                 |
|  MatchingStrategy              |
+--------------------------------+
| + find_driver(                 |
|     rider: Rider,              |
|     drivers: list[Driver]      |
|   ) -> Driver | None           |
+--------------------------------+

+--------------------------------+
|  <<interface>>                 |
|  RideObserver                  |
+--------------------------------+
| + on_ride_event(               |
|     event: RideEvent           |
|   ) -> None                    |
+--------------------------------+
```

### Step 5 — Patterns Applied

| Problem | Pattern | Why |
|---------|---------|-----|
| Multiple fare algorithms | Strategy (`PricingStrategy`) | "Multiple pricing strategies" is textbook Strategy |
| Multiple matching algorithms | Strategy (`MatchingStrategy`) | Same — algorithms are interchangeable |
| Notify driver and rider | Observer (`RideObserver`) | Two distinct recipients react to same events |
| Single ride event log | Singleton (`RideEventLog`) | "Log all ride events" implies one canonical store |
| Create ride objects | Factory Method | `RideFactory.create` decouples ride instantiation |

### Step 6 — Implementation

```python
"""
ride_sharing.py — LLD: Ride-Sharing Dispatch System
Python 3.12+ | pydantic v2 | logging
Patterns: Strategy (pricing + matching), Observer (notifications), Singleton (event log)
"""
from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum, auto
from typing import Optional

from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RideStatus(Enum):
    REQUESTED = auto()
    MATCHED = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    CANCELLED = auto()


class EventType(Enum):
    RIDE_REQUESTED = auto()
    DRIVER_MATCHED = auto()
    RIDE_STARTED = auto()
    RIDE_COMPLETED = auto()
    RIDE_CANCELLED = auto()


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

class Location(BaseModel):
    lat: float
    lng: float

    model_config = {"frozen": True}

    def distance_to(self, other: Location) -> float:
        """Euclidean distance — sufficient for interview; use Haversine in production."""
        return ((self.lat - other.lat) ** 2 + (self.lng - other.lng) ** 2) ** 0.5


class RideEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    event_type: EventType
    ride_id: str
    payload: dict[str, str] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------

class Rider(BaseModel):
    rider_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    current_location: Location


class Driver(BaseModel):
    driver_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    current_location: Location
    is_available: bool = True
    active_ride_id: Optional[str] = None

    def assign_ride(self, ride_id: str) -> None:
        self.is_available = False
        self.active_ride_id = ride_id
        log.info("Driver %s assigned to ride %s", self.driver_id, ride_id)

    def complete_ride(self) -> None:
        self.is_available = True
        self.active_ride_id = None
        log.info("Driver %s now available", self.driver_id)


class Ride(BaseModel):
    ride_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:10])
    rider_id: str
    driver_id: Optional[str] = None
    pickup: Location
    dropoff: Location
    status: RideStatus = RideStatus.REQUESTED
    estimated_fare: float = 0.0
    final_fare: Optional[float] = None
    requested_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def distance_km(self) -> float:
        """Straight-line distance as a proxy for trip distance."""
        return self.pickup.distance_to(self.dropoff) * 111.0  # 1 degree ~ 111 km

    def duration_minutes(self) -> float:
        if self.completed_at is None:
            return 0.0
        return (self.completed_at - self.requested_at).total_seconds() / 60


# ---------------------------------------------------------------------------
# Strategy: Pricing
# ---------------------------------------------------------------------------

class PricingStrategy(ABC):
    @abstractmethod
    def calculate_fare(self, ride: Ride) -> float: ...


class PerKmPricing(PricingStrategy):
    """$1.50 base + $2.00 per km."""
    BASE = 1.50
    RATE_PER_KM = 2.00

    def calculate_fare(self, ride: Ride) -> float:
        return round(self.BASE + ride.distance_km() * self.RATE_PER_KM, 2)


class FlatRatePricing(PricingStrategy):
    """Fixed fare regardless of distance."""
    FLAT_RATE = 15.00

    def calculate_fare(self, ride: Ride) -> float:
        return self.FLAT_RATE


class SurgePricing(PricingStrategy):
    """Per-km with a surge multiplier (e.g., 1.5x–3.0x)."""

    def __init__(self, base_strategy: PricingStrategy, multiplier: float) -> None:
        self._base = base_strategy
        self._multiplier = multiplier

    def calculate_fare(self, ride: Ride) -> float:
        base_fare = self._base.calculate_fare(ride)
        return round(base_fare * self._multiplier, 2)


# ---------------------------------------------------------------------------
# Strategy: Matching
# ---------------------------------------------------------------------------

class MatchingStrategy(ABC):
    @abstractmethod
    def find_driver(
        self, rider: Rider, drivers: list[Driver]
    ) -> Optional[Driver]: ...


class NearestDriverMatcher(MatchingStrategy):
    """Assigns the idle driver closest to the rider's pickup location."""

    def find_driver(
        self, rider: Rider, drivers: list[Driver]
    ) -> Optional[Driver]:
        available = [d for d in drivers if d.is_available]
        if not available:
            log.warning("No available drivers for rider %s", rider.rider_id)
            return None
        best = min(
            available,
            key=lambda d: d.current_location.distance_to(rider.current_location),
        )
        log.info(
            "NearestMatcher: driver %s selected (dist=%.4f)",
            best.driver_id,
            best.current_location.distance_to(rider.current_location),
        )
        return best


class PriorityQueueMatcher(MatchingStrategy):
    """
    Assigns driver with the longest idle time (fairness for drivers).
    Simplified: picks driver with the lowest driver_id lexicographically
    as a stand-in for idle-time ordering.
    """

    def find_driver(
        self, rider: Rider, drivers: list[Driver]
    ) -> Optional[Driver]:
        available = sorted(
            [d for d in drivers if d.is_available],
            key=lambda d: d.driver_id,
        )
        return available[0] if available else None


# ---------------------------------------------------------------------------
# Observer: Ride notifications
# ---------------------------------------------------------------------------

class RideObserver(ABC):
    @abstractmethod
    def on_ride_event(self, event: RideEvent) -> None: ...


class RiderNotifier(RideObserver):
    """Sends push notifications to the rider."""

    def on_ride_event(self, event: RideEvent) -> None:
        log.info(
            "[RIDER NOTIFICATION] ride=%s type=%s payload=%s",
            event.ride_id, event.event_type.name, event.payload,
        )


class DriverNotifier(RideObserver):
    """Sends push notifications to the driver."""

    def on_ride_event(self, event: RideEvent) -> None:
        log.info(
            "[DRIVER NOTIFICATION] ride=%s type=%s payload=%s",
            event.ride_id, event.event_type.name, event.payload,
        )


# ---------------------------------------------------------------------------
# Singleton: RideEventLog
# ---------------------------------------------------------------------------

class RideEventLog:
    """
    Singleton — single, authoritative log of all ride events.
    Also acts as an Observable: dispatches events to registered observers.
    """
    _instance: Optional[RideEventLog] = None

    def __init__(self) -> None:
        self._events: list[RideEvent] = []
        self._observers: list[RideObserver] = []
        log.info("RideEventLog singleton created")

    @classmethod
    def get_instance(cls) -> RideEventLog:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def subscribe(self, observer: RideObserver) -> None:
        self._observers.append(observer)

    def record(self, event: RideEvent) -> None:
        self._events.append(event)
        log.info(
            "RideEventLog: %s [ride=%s]", event.event_type.name, event.ride_id
        )
        for obs in self._observers:
            try:
                obs.on_ride_event(event)
            except Exception as exc:
                log.error("Observer %r failed: %s", obs, exc)

    def events_for_ride(self, ride_id: str) -> list[RideEvent]:
        return [e for e in self._events if e.ride_id == ride_id]

    def all_events(self) -> list[RideEvent]:
        return list(self._events)


# ---------------------------------------------------------------------------
# RideMatchingService — orchestrates everything
# ---------------------------------------------------------------------------

class RideMatchingService:
    def __init__(
        self,
        pricing: PricingStrategy,
        matcher: MatchingStrategy,
        event_log: RideEventLog,
    ) -> None:
        self._pricing = pricing
        self._matcher = matcher
        self._event_log = event_log
        self._drivers: dict[str, Driver] = {}
        self._rides: dict[str, Ride] = {}

    # ------------------------------------------------------------------
    # Driver management
    # ------------------------------------------------------------------

    def register_driver(self, driver: Driver) -> None:
        self._drivers[driver.driver_id] = driver
        log.info("Driver registered: %s", driver.driver_id)

    # ------------------------------------------------------------------
    # Ride lifecycle
    # ------------------------------------------------------------------

    def request_ride(self, rider: Rider, dropoff: Location) -> Optional[Ride]:
        ride = Ride(
            rider_id=rider.rider_id,
            pickup=rider.current_location,
            dropoff=dropoff,
        )
        self._rides[ride.ride_id] = ride

        self._event_log.record(RideEvent(
            event_type=EventType.RIDE_REQUESTED,
            ride_id=ride.ride_id,
            payload={"rider_id": rider.rider_id},
        ))

        driver = self._matcher.find_driver(rider, list(self._drivers.values()))
        if driver is None:
            ride.status = RideStatus.CANCELLED
            self._event_log.record(RideEvent(
                event_type=EventType.RIDE_CANCELLED,
                ride_id=ride.ride_id,
                payload={"reason": "no_driver_available"},
            ))
            return None

        ride.driver_id = driver.driver_id
        ride.status = RideStatus.MATCHED
        ride.estimated_fare = self._pricing.calculate_fare(ride)
        driver.assign_ride(ride.ride_id)

        self._event_log.record(RideEvent(
            event_type=EventType.DRIVER_MATCHED,
            ride_id=ride.ride_id,
            payload={
                "driver_id": driver.driver_id,
                "estimated_fare": str(ride.estimated_fare),
            },
        ))

        log.info(
            "Ride %s matched: rider=%s driver=%s est_fare=$%.2f",
            ride.ride_id, rider.rider_id, driver.driver_id, ride.estimated_fare,
        )
        return ride

    def start_ride(self, ride_id: str) -> bool:
        ride = self._rides.get(ride_id)
        if ride is None or ride.status != RideStatus.MATCHED:
            log.error("Cannot start ride %s (status=%s)", ride_id,
                      ride.status.name if ride else "NOT_FOUND")
            return False
        ride.status = RideStatus.IN_PROGRESS
        self._event_log.record(RideEvent(
            event_type=EventType.RIDE_STARTED,
            ride_id=ride_id,
            payload={"driver_id": ride.driver_id or ""},
        ))
        return True

    def complete_ride(self, ride_id: str) -> Optional[float]:
        ride = self._rides.get(ride_id)
        if ride is None or ride.status != RideStatus.IN_PROGRESS:
            log.error("Cannot complete ride %s", ride_id)
            return None

        ride.completed_at = datetime.now()
        ride.status = RideStatus.COMPLETED
        ride.final_fare = self._pricing.calculate_fare(ride)

        driver = self._drivers.get(ride.driver_id or "")
        if driver:
            driver.complete_ride()

        self._event_log.record(RideEvent(
            event_type=EventType.RIDE_COMPLETED,
            ride_id=ride_id,
            payload={"final_fare": str(ride.final_fare)},
        ))
        log.info("Ride %s completed. Final fare: $%.2f", ride_id, ride.final_fare)
        return ride.final_fare


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    RideEventLog.reset()
    event_log = RideEventLog.get_instance()

    # Register observers
    event_log.subscribe(RiderNotifier())
    event_log.subscribe(DriverNotifier())

    # Wire up service
    service = RideMatchingService(
        pricing=SurgePricing(PerKmPricing(), multiplier=1.5),
        matcher=NearestDriverMatcher(),
        event_log=event_log,
    )

    # Register drivers
    d1 = Driver(name="Alice (driver)", current_location=Location(lat=37.78, lng=-122.41))
    d2 = Driver(name="Bob (driver)", current_location=Location(lat=37.80, lng=-122.43))
    service.register_driver(d1)
    service.register_driver(d2)

    # Rider requests a ride
    rider = Rider(name="Carol", current_location=Location(lat=37.79, lng=-122.42))
    dropoff = Location(lat=37.75, lng=-122.39)

    ride = service.request_ride(rider, dropoff)
    if ride:
        log.info("Ride created: %s est=$%.2f", ride.ride_id, ride.estimated_fare)
        service.start_ride(ride.ride_id)
        final = service.complete_ride(ride.ride_id)
        log.info("Final fare: $%.2f", final or 0)

    log.info("--- Event log for ride ---")
    if ride:
        for event in event_log.events_for_ride(ride.ride_id):
            log.info("  %s @ %s: %s", event.event_type.name,
                     event.timestamp.strftime("%H:%M:%S"), event.payload)


if __name__ == "__main__":
    main()
```

### Design Review — SOLID check

| Principle | Status |
|-----------|--------|
| SRP | Each class has one job: `RideMatchingService` orchestrates; `RideEventLog` stores events; `NearestDriverMatcher` picks drivers. |
| OCP | Adding `SurgePricing` required no change to `RideMatchingService`. Adding `PriorityQueueMatcher` required no change. |
| LSP | `SurgePricing` composes `PerKmPricing` rather than overriding in a breaking way. Any `PricingStrategy` is substitutable. |
| ISP | `PricingStrategy`, `MatchingStrategy`, and `RideObserver` are all narrow single-method interfaces — no client is forced to depend on unused methods. |
| DIP | `RideMatchingService` depends on `PricingStrategy`, `MatchingStrategy`, and `RideEventLog` abstractions — not on `PerKmPricing` or `NearestDriverMatcher` directly. All injected via constructor. |

### Pattern composition summary

```
RideMatchingService
    ├── PricingStrategy  (Strategy)
    │       └── SurgePricing -> PerKmPricing (Decorator-like composition)
    ├── MatchingStrategy (Strategy)
    │       └── NearestDriverMatcher
    └── RideEventLog     (Singleton + Observable)
            ├── RiderNotifier  (Observer)
            └── DriverNotifier (Observer)
```

Patterns do not conflict — Strategy governs the *algorithm* choice, Observer governs *side-effects after an event*, and Singleton governs *instance cardinality* of the log. Each pattern solves a distinct problem.
