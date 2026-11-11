# Week 07 Assessment — LLD & OOD Interviews

> Pass gate: 75%  ·  Time allowed: 90 minutes  ·  Open notes: No

---

## Part 1: Conceptual (25 points)

Answer each question in 2–5 sentences. Precision beats length.

**1. (3 pts) Describe the six steps of the LLD method as practiced this week. What is the purpose of each step?**

**2. (3 pts) The following class violates the Open/Closed Principle. Identify the violation and describe the exact change that would fix it without altering the existing method.**

```python
class AreaCalculator:
    def compute_area(self, shape: dict) -> float:
        if shape["type"] == "circle":
            return 3.14159 * shape["radius"] ** 2
        elif shape["type"] == "rectangle":
            return shape["width"] * shape["height"]
        elif shape["type"] == "triangle":
            return 0.5 * shape["base"] * shape["height"]
        raise ValueError(f"Unknown shape: {shape['type']}")
```

**3. (3 pts) A code review flags this as a Liskov Substitution Principle violation. Is the reviewer correct? Justify your answer with reference to LSP's formal definition.**

```python
class ReadOnlyStack(Stack):
    def push(self, item: int) -> None:
        raise NotImplementedError("ReadOnlyStack does not support push")
```

**4. (3 pts) You have a `TokenBucketRateLimiter` class used from 20 concurrent threads. The class has an `allow_request()` method that checks `self.tokens` and decrements it if positive. No locks are used. Describe the specific race condition, what symptoms a caller would observe, and the minimal fix.**

**5. (3 pts) Read the following class diagram fragment and answer the questions below.**

```
+------------------+         +-------------------+
|   Notification   |-------->|  NotificationSender|  (interface)
+------------------+    1    +-------------------+
| -message: str    |         | +send(msg: str)   |
| -recipient: str  |         +-------------------+
| +dispatch() void |                 ^
+------------------+                 |
                            +--------+--------+
                            |                 |
                   +--------+---+   +---------+--+
                   | SmtpSender |   | SlackSender|
                   +------------+   +------------+
```

a) What design pattern does this structure implement?
b) If a `LoggingSender` is needed that wraps another sender and logs all messages before forwarding them, which pattern would you use to add it, and what would its constructor signature look like in Python?
c) Does `Notification` depend on `SmtpSender`? Explain in terms of the Dependency Inversion Principle.

**6. (3 pts) Compare thread-safe Singleton implementations using a class-level `threading.Lock` versus using Python's module-level initialization. When is each approach appropriate? What is the main risk of the lock-based approach that the module-level approach avoids?**

**7. (3 pts) You need to model a vending machine that dispenses items. The machine moves through states: `IDLE` → `ITEM_SELECTED` → `PAYMENT_PENDING` → `DISPENSING` → `IDLE`. A user pressing "cancel" returns the machine to `IDLE` from any state except `DISPENSING`.**

a) Should you use the State pattern or the Strategy pattern? Justify.
b) Sketch the Python class structure (class names and key method signatures only, no implementation).
c) What happens if invalid input is received in a given state (e.g., a coin is inserted while in `IDLE`)? How does the State pattern handle this cleanly?

**8. (4 pts) You are adding a discount feature to an e-commerce checkout. Currently:**

```python
class OrderProcessor:
    def calculate_total(self, items: list[Item]) -> float:
        return sum(item.price for item in items)
```

Three discount types are planned: percentage discount, flat discount, and buy-one-get-one. More types will be added quarterly. Apply OCP to extend this system. Write the abstract interface and one concrete implementation. Show how `OrderProcessor.calculate_total` changes (or does not change) as a result.

---

## Part 2: Design Challenge (40 points)

### Problem: Parking Lot System

**Specification:**

Design a parking lot system for a multi-level parking structure. The system must handle vehicle entry and exit through automated gates, compute fees, and provide occupancy information.

**Functional requirements:**

- The parking lot has multiple floors. Each floor has a fixed number of spots.
- Vehicle types: `Motorcycle`, `Car`, `Truck`. Each type has different spot requirements and pricing.
  - Motorcycles may park in motorcycle spots only (floor 3 in the reference building).
  - Cars may park in any standard spot.
  - Trucks require oversized spots (floor 1 only in the reference building, limited capacity).
- Entry flow: vehicle arrives at an entry gate → system finds a spot → issues a ticket with entry timestamp → raises the gate barrier.
- Exit flow: vehicle presents ticket at exit gate → system computes fee → collects payment → releases the spot → raises the gate barrier.
- Payment: fee is computed as: `max(hourly_rate * hours_parked, minimum_fee)`. Rates are per vehicle type and configurable.
- Occupancy query: the system must report available spots per floor and per vehicle type at any time.
- Multiple entry and exit gates operate concurrently.

**Constraints:**

| Constraint | Value |
|---|---|
| Max floors | 10 |
| Max spots per floor | 200 |
| Max concurrent gate operations | 20 |
| Ticket ID format | Unique string, contains floor and spot index |
| Clock dependency | Must be injectable (for testability) |
| No external dependencies | In-memory only; no DB, no network |

**Deliverable 1 — Class diagram (draw or describe):**

Produce a complete class diagram for this system. It must include:
- All entity classes with attributes and types.
- All interfaces (abstract classes or Protocols) with method signatures.
- All associations, with multiplicity where relevant (1-to-1, 1-to-many, etc.).
- Indication of which classes depend on interfaces rather than concretions.

**Deliverable 2 — Python implementation:**

Implement the following classes in Python 3.12+ with full type hints. No external libraries beyond the standard library and Pydantic v2.

Required classes and key methods (you may add helpers):

```python
# Enums and value objects
class VehicleType(str, Enum): ...
class SpotType(str, Enum): ...   # MOTORCYCLE, STANDARD, OVERSIZED

# Interfaces
class SpotAllocator(ABC):
    def find_spot(self, vehicle_type: VehicleType) -> Optional["ParkingSpot"]: ...
    def release_spot(self, spot: "ParkingSpot") -> None: ...

class FeeCalculator(ABC):
    def compute_fee(self, ticket: "Ticket", exit_time: datetime) -> float: ...

class PaymentProcessor(ABC):
    def process_payment(self, amount: float, ticket_id: str) -> bool: ...

class Clock(ABC):
    def now(self) -> datetime: ...

# Core domain classes
class ParkingSpot:
    floor: int
    index: int
    spot_type: SpotType
    is_occupied: bool

class Ticket:
    ticket_id: str
    plate: str
    vehicle_type: VehicleType
    spot: ParkingSpot
    entry_time: datetime

class ParkingLot:
    def park(self, vehicle_type: VehicleType, plate: str) -> Optional[str]: ...
    def exit_vehicle(self, ticket_id: str) -> float: ...
    def available_spots(self) -> dict[int, dict[SpotType, int]]: ...
```

**Deliverable 3 — Concurrency:**

Explain in 3–5 sentences how you would make `ParkingLot.park` and `ParkingLot.exit_vehicle` safe for concurrent access from 20 simultaneous gate threads. Specify the lock granularity (one global lock vs. per-floor lock vs. per-spot lock) and justify your choice. What is the tradeoff between lock granularity and throughput?

**Scoring:**

| Sub-deliverable | Points |
|---|---|
| Class diagram — completeness and correctness | 10 |
| Python implementation — correctness, type hints, DI | 20 |
| Concurrency explanation — accuracy and tradeoff analysis | 10 |
| **Total** | **40** |

---

## Part 3: Debugging & Problem-Solving (20 points)

**Scenario:**

The following Python class implements a token bucket rate limiter. It is used in production to limit API calls to 100 requests per second. Three engineers have filed bugs:

- Engineer A: "Under load, callers sometimes get allowed through even after the token count should be 0."
- Engineer B: "After a long idle period, the bucket fills up to way more than the max capacity."
- Engineer C: "The type checker complains about the `tokens` field but I cannot see why."

Read the code, identify all three bugs, explain the root cause of each, and provide the corrected version.

```python
from __future__ import annotations

import threading
import time


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter.
    Refills at `rate` tokens per second up to `capacity` tokens.
    Each call to allow_request() consumes one token.
    Returns True if the request is allowed, False if rate-limited.
    """

    def __init__(self, capacity: int, rate: float) -> None:
        self.capacity = capacity
        self.rate = rate
        self.tokens = capacity       # Bug C is here
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now
        self.tokens += elapsed * self.rate    # Bug B is here — no cap applied

    def allow_request(self) -> bool:
        self._refill()                         # Bug A is here — lock not held during refill
        with self._lock:
            if self.tokens >= 1:
                self.tokens -= 1
            return self.tokens >= 0            # Bug A continued — wrong return logic
```

Your answer must:

1. State the exact line(s) where each bug manifests.
2. Explain the failure scenario that triggers each bug (what inputs/timing cause it).
3. Provide the fully corrected class.

---

## Part 4: Code Quality (15 points)

Review the following Singleton logger implementation. Four specific quality issues are present.

```python
import logging


_instance = None      # module-level mutable global


class AppLogger:

    def __new__(cls):
        global _instance
        if _instance is None:
            _instance = super().__new__(cls)
            _instance.logger = logging.getLogger("app")
            _instance.logger.setLevel(logging.DEBUG)
        return _instance

    def log(self, message):                  # missing type hints
        self.logger.info(message)

    def set_level(self, level):              # missing type hints
        self.logger.setLevel(level)


def get_logger():
    return AppLogger()                       # creates new AppLogger() every call
```

Answer each question:

1. **(4 pts)** This implementation is not thread-safe. Under what exact concurrent execution scenario does a second instance get created? Provide a fix using `threading.Lock`.

2. **(3 pts)** The class exposes no interface. Why is this a problem for testing? Write a `Logger` ABC that `AppLogger` should implement, with appropriate method signatures and type hints.

3. **(4 pts)** Add full type hints to `log` and `set_level`. The `level` parameter should accept both `int` and `str` (as `logging.setLevel` does). Show the corrected method signatures.

4. **(4 pts)** `get_logger()` calls `AppLogger()` each time. Because of `__new__`, this appears to work — but identify one scenario where this causes a subtle behavioral bug, and propose a cleaner pattern.

---

## Rubric

| Dimension | Weight | Points |
|---|---|---|
| Independent implementation (design challenge) | 40% | 40 |
| Conceptual understanding | 25% | 25 |
| Debugging & problem-solving | 20% | 20 |
| Code quality & testing | 15% | 15 |
| **Total** | **100%** | **100** |

**Pass gate: 75 / 100**

---

Answer key: ../solutions/week-07/assessment.md
