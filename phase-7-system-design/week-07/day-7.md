# Phase 7 · Week 07 · Day 7 — Project Day: Polish + Reflect

> Objective: Finalize the week's LLD project, refactor toward clean testable OO design, and run a structured mock-interview debrief.  ·  Time: ~3 hr  ·  Prereqs: [[day-1]]..[[day-6]]

---

## Concept

Day 7 is not a new topic — it is the integration and reflection session. Three things happen:

1. **Polish** — finish the week project, fix rough edges, ensure all tests pass.
2. **Reflect** — review the LLD method you practiced all week and identify where you struggled.
3. **Debrief** — run a mini mock-interview debrief on one of the problems you designed.

Before polishing any code, it is worth understanding what "clean OO code" actually means in a testable sense.

### What makes OO code testable

A class is hard to test when it has any of these properties:

- It creates its own dependencies (e.g., instantiates a database client in `__init__`).
- It reads from or writes to global or module-level state.
- It does too many things — violates Single Responsibility.
- It relies on concrete implementations rather than abstractions.

The solution to all four problems is the same: **dependency injection through abstract interfaces**.

Dependency injection means: receive your collaborators from the outside rather than constructing them yourself. Abstract interfaces (Python's `ABC` or `Protocol`) mean: depend on what a collaborator *does*, not on what it *is*.

### The three laws of testable OO code

**Law 1 — Depend on abstractions, not concretions.**

Every dependency that crosses a process boundary (I/O, clock, randomness, network) must be injected as an interface. The class constructor or factory accepts the interface. Tests inject a fake or mock. Production code injects the real implementation.

**Law 2 — No global state.**

Global variables, module-level singletons, and class-level mutable state make tests order-dependent and non-deterministic. Every piece of state that changes during a test must either live in a local variable or be scoped to an object you control.

**Law 3 — One reason to change per class.**

If adding a new payment provider requires modifying the `ParkingLot` class, that class has too many responsibilities. If adding a new vehicle type requires modifying the pricing logic, the pricing logic is in the wrong place. Each class should have exactly one reason to change.

---

## How it works internally

### The God class problem

A God class is a class that knows too much and does too much. It accumulates responsibilities over time as features are added without discipline. Signs:

- More than ~200–300 lines.
- More than one layer of abstraction inside a single class (orchestration code mixed with parsing, mixed with persistence).
- More than five or six instance variables.
- Test file is longer than the implementation and still does not cover everything.
- Every new feature requires modifying the same class.

### The refactoring pattern

Refactoring a God class follows a predictable pattern:

1. **Identify responsibility clusters.** Group the methods by what they operate on. Each cluster is a candidate class.
2. **Extract interfaces first.** Define what each new class *must do* as an ABC or Protocol before writing the implementation.
3. **Inject dependencies.** Replace direct construction of collaborators with injection points.
4. **Move methods to their natural owner.** A method that only touches `Vehicle` data belongs in or near `Vehicle`, not in `ParkingLot`.
5. **Write tests for each extracted class independently.** The point of the refactor is testability; verify it by writing tests that do not require instantiating the entire system.

### Python tools for clean OO design

**`abc.ABC` and `abc.abstractmethod`** — define interfaces with mandatory method signatures. A class that inherits from an ABC and does not implement all abstract methods raises `TypeError` at instantiation time, not at call time.

**`typing.Protocol`** — structural subtyping. A class satisfies a Protocol without explicitly inheriting from it, as long as it implements the required methods. Prefer Protocol for things you do not own (e.g., a third-party payment client that happens to have a `charge` method).

**`dataclasses.dataclass` and `pydantic.BaseModel`** — value objects with no behavior. Use these for data that crosses boundaries: requests, responses, domain events. Never put business logic inside a Pydantic model.

---

## Worked example

### Before: a God class

```python
"""
Before refactoring: a single ParkingManager class
that handles everything — a typical God class.
"""

from __future__ import annotations

import datetime
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ParkingManager:
    """Manages the entire parking lot. Too many responsibilities."""

    def __init__(self) -> None:
        self.floors: dict[int, list[Optional[str]]] = {
            1: [None] * 50,
            2: [None] * 50,
            3: [None] * 30,
        }
        self.active_tickets: dict[str, dict] = {}
        self.revenue: float = 0.0
        self.rates: dict[str, float] = {
            "motorcycle": 1.0,
            "car": 2.0,
            "truck": 4.0,
        }

    def park(self, vehicle_type: str, plate: str) -> Optional[str]:
        """Find a spot and issue a ticket. Handles all vehicle types inline."""
        # motorcycle goes to floor 3 (compacts only)
        # cars go to floors 1 or 2
        # trucks only on floor 1 (clearance)
        for floor_num, spots in self.floors.items():
            if vehicle_type == "truck" and floor_num != 1:
                continue
            if vehicle_type == "motorcycle" and floor_num != 3:
                continue
            for idx, spot in enumerate(spots):
                if spot is None:
                    self.floors[floor_num][idx] = plate
                    ticket_id = f"{floor_num}-{idx}-{plate}"
                    self.active_tickets[ticket_id] = {
                        "plate": plate,
                        "vehicle_type": vehicle_type,
                        "floor": floor_num,
                        "spot": idx,
                        "entry_time": datetime.datetime.now(datetime.timezone.utc),
                    }
                    logger.info("Parked %s at %s", plate, ticket_id)
                    return ticket_id
        return None

    def exit_vehicle(self, ticket_id: str) -> float:
        """Calculate fee and free the spot."""
        if ticket_id not in self.active_tickets:
            raise ValueError(f"Unknown ticket: {ticket_id}")
        ticket = self.active_tickets.pop(ticket_id)
        duration = (
            datetime.datetime.now(datetime.timezone.utc) - ticket["entry_time"]
        ).total_seconds() / 3600.0
        rate = self.rates[ticket["vehicle_type"]]
        fee = max(rate, rate * duration)
        self.floors[ticket["floor"]][ticket["spot"]] = None
        self.revenue += fee
        logger.info("Exit %s, fee=%.2f, total_revenue=%.2f", ticket_id, fee, self.revenue)
        return fee

    def available_spots(self) -> dict[int, int]:
        """Count free spots per floor."""
        return {
            floor_num: sum(1 for s in spots if s is None)
            for floor_num, spots in self.floors.items()
        }
```

Problems with this class:

- `park` encodes floor-assignment rules inline. Adding a new vehicle type requires editing `park`.
- `exit_vehicle` computes fees inline. Changing pricing requires editing `exit_vehicle`.
- `datetime.datetime.now()` is called directly — impossible to test time-dependent behavior.
- All state is mutable on `self` — no clear boundary between what a test can reset vs. what leaks between tests.
- No interfaces — impossible to inject fakes.

### After: extracted, injected, testable

```python
"""
After refactoring: separated responsibilities with
dependency injection and abstract interfaces.

File layout (one class per file in production):
  models.py       - VehicleType, ParkingSpot, Ticket (value objects)
  interfaces.py   - SpotAllocator, FeeCalculator, Clock (ABCs)
  allocator.py    - DefaultSpotAllocator
  pricing.py      - HourlyFeeCalculator
  lot.py          - ParkingLot (orchestrator only)
"""

from __future__ import annotations

import datetime
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Value objects (data only, no business logic)
# ---------------------------------------------------------------------------

class VehicleType(str, Enum):
    MOTORCYCLE = "motorcycle"
    CAR = "car"
    TRUCK = "truck"


@dataclass(frozen=True)
class ParkingSpot:
    floor: int
    index: int

    def __str__(self) -> str:
        return f"floor={self.floor} spot={self.index}"


@dataclass
class Ticket:
    ticket_id: str
    plate: str
    vehicle_type: VehicleType
    spot: ParkingSpot
    entry_time: datetime.datetime


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------

class SpotAllocator(ABC):
    """Finds a free spot for a given vehicle type."""

    @abstractmethod
    def find_spot(self, vehicle_type: VehicleType) -> Optional[ParkingSpot]:
        """Return the first available spot, or None if the lot is full."""
        ...

    @abstractmethod
    def release_spot(self, spot: ParkingSpot) -> None:
        """Mark a spot as available."""
        ...

    @abstractmethod
    def available_count(self, floor: int) -> int:
        """Return number of free spots on a floor."""
        ...


class FeeCalculator(ABC):
    """Computes the parking fee given a ticket and exit time."""

    @abstractmethod
    def compute_fee(
        self,
        ticket: Ticket,
        exit_time: datetime.datetime,
    ) -> float:
        """Return the fee in USD (or local currency unit)."""
        ...


class Clock(ABC):
    """Provides the current time. Abstract so tests can inject a fixed clock."""

    @abstractmethod
    def now(self) -> datetime.datetime:
        ...


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------

class SystemClock(Clock):
    def now(self) -> datetime.datetime:
        return datetime.datetime.now(datetime.timezone.utc)


@dataclass
class FloorConfig:
    floor: int
    capacity: int
    allowed_types: set[VehicleType]


class DefaultSpotAllocator(SpotAllocator):
    """
    Simple first-fit allocator. Floor constraints come from configuration,
    not from hardcoded conditionals.
    """

    def __init__(self, floors: list[FloorConfig]) -> None:
        # spots[floor][index] = plate or None
        self._spots: dict[int, list[Optional[str]]] = {
            f.floor: [None] * f.capacity for f in floors
        }
        self._allowed: dict[int, set[VehicleType]] = {
            f.floor: f.allowed_types for f in floors
        }

    def find_spot(self, vehicle_type: VehicleType) -> Optional[ParkingSpot]:
        for floor_num, spots in self._spots.items():
            if vehicle_type not in self._allowed[floor_num]:
                continue
            for idx, occupant in enumerate(spots):
                if occupant is None:
                    spots[idx] = "__reserved__"
                    return ParkingSpot(floor=floor_num, index=idx)
        return None

    def release_spot(self, spot: ParkingSpot) -> None:
        self._spots[spot.floor][spot.index] = None

    def occupy_spot(self, spot: ParkingSpot, plate: str) -> None:
        self._spots[spot.floor][spot.index] = plate

    def available_count(self, floor: int) -> int:
        return sum(1 for s in self._spots[floor] if s is None)


@dataclass
class HourlyRate:
    vehicle_type: VehicleType
    rate_per_hour: float
    minimum_fee: float


class HourlyFeeCalculator(FeeCalculator):
    """
    Charges by the hour with a per-vehicle-type rate.
    Rates come from configuration — no hardcoded constants.
    """

    def __init__(self, rates: list[HourlyRate]) -> None:
        self._rates: dict[VehicleType, HourlyRate] = {
            r.vehicle_type: r for r in rates
        }

    def compute_fee(
        self,
        ticket: Ticket,
        exit_time: datetime.datetime,
    ) -> float:
        rate_config = self._rates[ticket.vehicle_type]
        duration_hours = (
            exit_time - ticket.entry_time
        ).total_seconds() / 3600.0
        fee = rate_config.rate_per_hour * duration_hours
        return max(fee, rate_config.minimum_fee)


# ---------------------------------------------------------------------------
# Orchestrator: ParkingLot depends only on interfaces
# ---------------------------------------------------------------------------

class ParkingLot:
    """
    Orchestrates spot allocation and fee calculation.
    Has no knowledge of floor layouts or pricing formulas.
    All collaborators are injected.
    """

    def __init__(
        self,
        allocator: SpotAllocator,
        fee_calculator: FeeCalculator,
        clock: Clock,
    ) -> None:
        self._allocator = allocator
        self._fee_calculator = fee_calculator
        self._clock = clock
        self._active_tickets: dict[str, Ticket] = {}
        self._revenue: float = 0.0
        self._ticket_counter: int = 0

    def _generate_ticket_id(self, spot: ParkingSpot) -> str:
        self._ticket_counter += 1
        return f"T{self._ticket_counter:05d}-F{spot.floor}-S{spot.index}"

    def park(self, vehicle_type: VehicleType, plate: str) -> Optional[str]:
        """
        Attempt to park a vehicle. Returns ticket_id on success, None if full.
        """
        spot = self._allocator.find_spot(vehicle_type)
        if spot is None:
            logger.warning("No spot available for %s (%s)", plate, vehicle_type)
            return None

        if isinstance(self._allocator, DefaultSpotAllocator):
            self._allocator.occupy_spot(spot, plate)

        ticket_id = self._generate_ticket_id(spot)
        ticket = Ticket(
            ticket_id=ticket_id,
            plate=plate,
            vehicle_type=vehicle_type,
            spot=spot,
            entry_time=self._clock.now(),
        )
        self._active_tickets[ticket_id] = ticket
        logger.info("Parked %s at %s (ticket=%s)", plate, spot, ticket_id)
        return ticket_id

    def exit_vehicle(self, ticket_id: str) -> float:
        """
        Process exit: compute fee, release spot, accumulate revenue.
        Returns the fee charged.
        """
        if ticket_id not in self._active_tickets:
            raise ValueError(f"Unknown ticket: {ticket_id!r}")

        ticket = self._active_tickets.pop(ticket_id)
        exit_time = self._clock.now()
        fee = self._fee_calculator.compute_fee(ticket, exit_time)
        self._allocator.release_spot(ticket.spot)
        self._revenue += fee
        logger.info(
            "Exit %s plate=%s fee=%.2f revenue=%.2f",
            ticket_id, ticket.plate, fee, self._revenue,
        )
        return fee

    @property
    def revenue(self) -> float:
        return self._revenue
```

```python
"""
tests/test_parking_lot.py

Unit tests for the refactored ParkingLot.
Each test is independent — no shared mutable state.
"""

from __future__ import annotations

import datetime

import pytest

from parking.interfaces import Clock, FeeCalculator
from parking.lot import ParkingLot, Ticket, VehicleType
from parking.allocator import DefaultSpotAllocator, FloorConfig
from parking.pricing import HourlyFeeCalculator, HourlyRate


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class FixedClock(Clock):
    """Clock that returns a fixed, controllable time."""

    def __init__(self, ts: datetime.datetime) -> None:
        self._ts = ts

    def advance(self, seconds: float) -> None:
        self._ts += datetime.timedelta(seconds=seconds)

    def now(self) -> datetime.datetime:
        return self._ts


class FakeFeeCalculator(FeeCalculator):
    """Always returns a fixed fee regardless of duration."""

    def __init__(self, fee: float) -> None:
        self._fee = fee

    def compute_fee(self, ticket: Ticket, exit_time: datetime.datetime) -> float:
        return self._fee


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BASE_TIME = datetime.datetime(2025, 1, 1, 9, 0, 0, tzinfo=datetime.timezone.utc)


@pytest.fixture
def clock() -> FixedClock:
    return FixedClock(BASE_TIME)


@pytest.fixture
def allocator() -> DefaultSpotAllocator:
    floors = [
        FloorConfig(floor=1, capacity=2, allowed_types={VehicleType.CAR, VehicleType.TRUCK}),
        FloorConfig(floor=2, capacity=2, allowed_types={VehicleType.CAR}),
        FloorConfig(floor=3, capacity=1, allowed_types={VehicleType.MOTORCYCLE}),
    ]
    return DefaultSpotAllocator(floors)


@pytest.fixture
def fee_calc() -> HourlyFeeCalculator:
    rates = [
        HourlyRate(VehicleType.MOTORCYCLE, rate_per_hour=1.0, minimum_fee=1.0),
        HourlyRate(VehicleType.CAR, rate_per_hour=2.0, minimum_fee=2.0),
        HourlyRate(VehicleType.TRUCK, rate_per_hour=4.0, minimum_fee=4.0),
    ]
    return HourlyFeeCalculator(rates)


@pytest.fixture
def lot(allocator: DefaultSpotAllocator, fee_calc: HourlyFeeCalculator, clock: FixedClock) -> ParkingLot:
    return ParkingLot(allocator=allocator, fee_calculator=fee_calc, clock=clock)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPark:
    def test_park_returns_ticket_id(self, lot: ParkingLot) -> None:
        ticket_id = lot.park(VehicleType.CAR, "ABC123")
        assert ticket_id is not None

    def test_park_motorcycle_on_floor3(
        self, lot: ParkingLot, allocator: DefaultSpotAllocator
    ) -> None:
        ticket_id = lot.park(VehicleType.MOTORCYCLE, "MOTO1")
        assert ticket_id is not None
        assert "F3" in ticket_id

    def test_park_returns_none_when_full(self, lot: ParkingLot) -> None:
        lot.park(VehicleType.MOTORCYCLE, "M1")
        result = lot.park(VehicleType.MOTORCYCLE, "M2")
        assert result is None

    def test_park_different_vehicle_types_independently(self, lot: ParkingLot) -> None:
        t1 = lot.park(VehicleType.CAR, "CAR1")
        t2 = lot.park(VehicleType.MOTORCYCLE, "MOTO1")
        t3 = lot.park(VehicleType.TRUCK, "TRUCK1")
        assert all(t is not None for t in [t1, t2, t3])


class TestExit:
    def test_exit_returns_minimum_fee(
        self, lot: ParkingLot, clock: FixedClock
    ) -> None:
        ticket_id = lot.park(VehicleType.CAR, "CAR1")
        assert ticket_id is not None
        # Exit immediately — minimum fee applies
        fee = lot.exit_vehicle(ticket_id)
        assert fee == pytest.approx(2.0)

    def test_exit_charges_by_hour(
        self, lot: ParkingLot, clock: FixedClock
    ) -> None:
        ticket_id = lot.park(VehicleType.CAR, "CAR1")
        assert ticket_id is not None
        clock.advance(3600 * 3)  # 3 hours
        fee = lot.exit_vehicle(ticket_id)
        assert fee == pytest.approx(6.0)

    def test_exit_releases_spot_for_reuse(
        self, lot: ParkingLot, allocator: DefaultSpotAllocator
    ) -> None:
        # Floor 3 has 1 motorcycle spot
        t1 = lot.park(VehicleType.MOTORCYCLE, "MOTO1")
        assert t1 is not None
        lot.exit_vehicle(t1)
        t2 = lot.park(VehicleType.MOTORCYCLE, "MOTO2")
        assert t2 is not None

    def test_exit_unknown_ticket_raises(self, lot: ParkingLot) -> None:
        with pytest.raises(ValueError, match="Unknown ticket"):
            lot.exit_vehicle("NONEXISTENT")


class TestRevenue:
    def test_revenue_accumulates_across_exits(
        self, lot: ParkingLot, clock: FixedClock
    ) -> None:
        t1 = lot.park(VehicleType.CAR, "CAR1")
        t2 = lot.park(VehicleType.CAR, "CAR2")
        assert t1 is not None and t2 is not None
        lot.exit_vehicle(t1)
        lot.exit_vehicle(t2)
        assert lot.revenue == pytest.approx(4.0)  # 2 x minimum $2
```

### Key observations from the refactor

| Concern | Before (God class) | After (clean OO) |
|---|---|---|
| Floor constraints | Hardcoded `if` in `park` | `FloorConfig` list passed to `DefaultSpotAllocator` |
| Pricing logic | Hardcoded rates in `exit_vehicle` | `HourlyRate` list passed to `HourlyFeeCalculator` |
| Current time | `datetime.datetime.now()` inline | Injected `Clock` interface |
| Testing | Must instantiate full system | Each class testable independently |
| Adding a vehicle type | Modify `park` and `exit_vehicle` | Add a `FloorConfig` entry and an `HourlyRate` entry |

---

## Exercises

### Exercise 1 — Refactor a messy notification class

The following class manages notifications. It has at least five SRP/DI violations. Identify each violation, then refactor the class using the patterns from the worked example. Write unit tests for your refactored version.

```python
"""
BEFORE: Messy NotificationManager.
Do NOT run this code in production — it has real design problems.
Identify the violations, refactor, and write tests.
"""

from __future__ import annotations

import smtplib
import time
import json
import os
from typing import Optional


SMTP_HOST = "smtp.company.com"   # hardcoded
SMTP_PORT = 587
MAX_RETRIES = 3


class NotificationManager:

    sent_count = 0        # class-level counter — shared across all instances
    last_sent_time = 0.0  # class-level — shared across all instances

    def __init__(self) -> None:
        self.conn = smtplib.SMTP(SMTP_HOST, SMTP_PORT)  # connects in constructor
        self.template_dir = "/etc/notif/templates"      # hardcoded path

    def send(self, user_id: str, message_type: str, context: dict) -> bool:
        # Load template from disk inline
        template_path = os.path.join(self.template_dir, f"{message_type}.json")
        with open(template_path) as f:
            template = json.load(f)

        body = template["body"].format(**context)
        subject = template["subject"].format(**context)

        # Rate limit: 1 message per second
        now = time.time()
        if now - NotificationManager.last_sent_time < 1.0:
            time.sleep(1.0 - (now - NotificationManager.last_sent_time))

        # Lookup email from user_id — hits an external service inline
        email = self._get_email(user_id)
        if email is None:
            return False

        for attempt in range(MAX_RETRIES):
            try:
                self.conn.sendmail("noreply@company.com", email, f"Subject: {subject}\n\n{body}")
                NotificationManager.sent_count += 1
                NotificationManager.last_sent_time = time.time()
                return True
            except Exception:
                if attempt == MAX_RETRIES - 1:
                    return False
        return False

    def _get_email(self, user_id: str) -> Optional[str]:
        import requests  # local import — hides dependency
        resp = requests.get(f"https://users.internal/api/{user_id}/email")
        if resp.status_code == 200:
            return resp.json()["email"]
        return None

    def report(self) -> None:
        print(f"Total sent: {NotificationManager.sent_count}")   # print, not logging
```

Your refactored version must:

- Define at least three ABC interfaces (e.g., `EmailSender`, `TemplateLoader`, `UserEmailLookup`).
- Remove all class-level mutable state.
- Remove all direct I/O from `__init__` and from business logic methods.
- Be fully unit-testable without any network or filesystem access.
- Have tests that cover: successful send, rate-limiting behavior, user not found, SMTP failure, and retry logic.

(Solutions: ../solutions/week-07/day-7.md)

### Exercise 2 — Write missing unit tests for the Elevator System

Your week project includes an Elevator System. Without looking at the solution, write unit tests for the following scenarios using only the class interfaces (no implementation details):

1. A request for floor 5 from the lobby (floor 0) is added to the scheduler queue.
2. An elevator traveling UP passes floor 3 — a waiting request for floor 3 is picked up.
3. An elevator at capacity refuses a new boarding request.
4. The SCAN algorithm resolves a set of floor requests in correct order when traveling UP.
5. Calling `open_door()` on an elevator already in `DOOR_OPEN` state does not trigger a second transition.
6. Closing the door while a passenger is boarding raises the appropriate exception.
7. After all requests are served, the elevator returns to `IDLE` state.
8. An elevator at floor 10 (top floor) reverses direction correctly when no higher requests exist.

Write these as `pytest` test stubs with clear docstrings. You do not need a passing implementation — write the test structure and assert statements as if the implementation exists.

(Solutions: ../solutions/week-07/day-7.md)

---

## Self-check

- [ ] I can name the three laws of testable OO code without looking.
- [ ] I can explain, in one sentence each, what ABC and Protocol are and when to prefer one over the other.
- [ ] I can identify at least five violations in the `NotificationManager` class above.
- [ ] My week project compiles and all tests pass (`pytest -v`).
- [ ] I can walk through the SCAN scheduling algorithm on a whiteboard for an arbitrary set of floor requests.
- [ ] I can explain why injecting a `Clock` interface makes time-dependent tests deterministic.
- [ ] My project classes average under 150 lines each (split if not).
- [ ] I ran `ruff check` on all project files with zero violations.

---

## Supplementary

- [Dependency Injection in Python — real-world guide](https://python-dependency-injector.ets-labs.org/introduction/di_in_python.html)
- [Python ABCs official docs](https://docs.python.org/3/library/abc.html)
- [typing.Protocol — PEP 544](https://peps.python.org/pep-0544/)
- [Refactoring Guru — God Object anti-pattern](https://refactoring.guru/antipatterns/god-object)
- [SOLID Principles in Python — Real Python](https://realpython.com/solid-principles-python/)
- [pytest fixtures and dependency injection](https://docs.pytest.org/en/stable/explanation/fixtures.html)
