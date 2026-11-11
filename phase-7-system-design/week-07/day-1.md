# Phase 7 · Week 07 · Day 1 — The LLD Interview Method

> Objective: Master the 6-step LLD interview method and apply it end-to-end on a parking lot system.  ·  Time: ~1 hr  ·  Prereqs: none

## Concept

**LLD vs HLD**

High-level design (HLD) answers: "what services exist, how do they communicate, how do they scale?" — boxes and arrows at the system boundary.

Low-level design (LLD) answers: "what classes exist, what are their responsibilities, how do they interact?" — the code skeleton that a team could actually implement.

In an interview, LLD tests whether you can decompose a problem, assign responsibilities clearly, and produce an API that a teammate can code against.

**The 6-Step Method**

```
Step 1 — CLARIFY     : nail scope, actors, constraints
Step 2 — ENTITIES    : list the core nouns
Step 3 — RELATIONSHIPS: cardinality + ownership between entities
Step 4 — INTERFACES  : define public contracts (methods + signatures)
Step 5 — PATTERNS    : pick structural/behavioural patterns that fit
Step 6 — CODE        : implement key classes, defer boilerplate
```

Work the steps in order. Steps 1-3 are whiteboard talk; Steps 4-6 are code. In a 45-min interview: Steps 1-3 should take ~10 min, Step 4 ~10 min, Steps 5-6 ~20 min.

## How it works internally

**Why clarify first?**
Requirements are underspecified by design. Interviewers inject ambiguity to see if you ask the right questions. Diving to code without clarifying wastes time and signals poor engineering instinct.

Key clarification axes:
- Scale: how many concurrent users / objects?
- Features: MVP vs full product? What is explicitly out of scope?
- Actors: who are the users, what do they need to do?
- Constraints: real-time? persistence? external integrations?

**Why entities before interfaces?**
Entities are the stable nouns. Interfaces change as you discover more behavior. Getting entities wrong early means refactoring everything; getting entities right means interfaces fall out naturally.

**Why patterns after entities/relationships, not before?**
Patterns are solutions to known problems. You cannot pick a pattern until you know what problem you have. Applying a pattern prematurely bloats the design.

**Cardinality notation used in this course**

```
1 ----< many    (one-to-many)
>----< many-to-many
1 ---- 1        (one-to-one)
```

## Worked example

**Problem: Parking Lot System**

### Step 1 — Clarify

Questions to ask (and assumed answers):
- Multiple floors? Yes, multi-storey.
- Vehicle types? Cars, motorbikes, trucks. Different spot sizes.
- Payment? Hourly rate, pay on exit.
- Concurrent entries? Yes, multiple gates.
- Reservations? No, walk-in only.
- Persistence? In-memory for interview, note DB could be added.

### Step 2 — Entities

```
ParkingLot       — the overall facility
Floor            — one level of the lot
ParkingSpot      — a single space (has a type: SMALL/MEDIUM/LARGE)
Vehicle          — a car/motorbike/truck entering the lot
Ticket           — issued on entry, closed on exit
Payment          — record of a completed transaction
Gate             — entry or exit point
```

### Step 3 — Relationships

```
ParkingLot  1 ----< Floor
Floor       1 ----< ParkingSpot
ParkingSpot 1 ---- 1 Vehicle   (when occupied)
Ticket      1 ---- 1 Vehicle
Ticket      1 ---- 1 ParkingSpot
Ticket      1 ---- 1 Payment   (after exit)
ParkingLot  1 ----< Gate
```

### Step 4 — Interfaces (ASCII class diagram)

```
+---------------------+
|   <<interface>>     |
|   Parkable          |
+---------------------+
| + get_type() -> VehicleType
| + get_plate() -> str
+---------------------+
         ^
         |
+---------------------+
|   Vehicle           |
+---------------------+
| - plate: str        |
| - type: VehicleType |
+---------------------+

+---------------------+
|   <<interface>>     |
|   SpotFinder        |
+---------------------+
| + find_spot(v: Vehicle) -> ParkingSpot | None
+---------------------+

+---------------------+
|   <<interface>>     |
|   PricingStrategy   |
+---------------------+
| + calculate(ticket: Ticket) -> float
+---------------------+

+---------------------+          +---------------------+
|   ParkingLot        |1       * |   Floor             |
+---------------------+----------+---------------------+
| - floors: list[Floor]         | - spots: list[Spot]  |
| - gates: list[Gate]           | - number: int        |
| + park(v: Vehicle) -> Ticket  +---------------------+
| + exit(t: Ticket) -> Payment  |
+---------------------+
```

### Step 5 — Patterns

| Problem | Pattern |
|---------|---------|
| Different pricing algorithms (hourly, flat, weekend) | Strategy |
| Finding available spot across floors | Strategy (SpotFinder) |
| Only one ParkingLot instance | Singleton |
| Creating Vehicle from type string | Factory |
| Notifying display boards when spot state changes | Observer |

### Step 6 — Code

```python
"""
parking_lot.py — LLD Interview: Parking Lot System
Python 3.12+ | pydantic v2 | logging
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

class VehicleType(Enum):
    MOTORBIKE = auto()
    CAR = auto()
    TRUCK = auto()


class SpotSize(Enum):
    SMALL = auto()   # motorbike
    MEDIUM = auto()  # car
    LARGE = auto()   # truck


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------

class Vehicle(BaseModel):
    plate: str
    vehicle_type: VehicleType

    def required_spot_size(self) -> SpotSize:
        mapping = {
            VehicleType.MOTORBIKE: SpotSize.SMALL,
            VehicleType.CAR: SpotSize.MEDIUM,
            VehicleType.TRUCK: SpotSize.LARGE,
        }
        return mapping[self.vehicle_type]


class ParkingSpot(BaseModel):
    spot_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    floor_number: int
    size: SpotSize
    is_occupied: bool = False
    current_plate: Optional[str] = None

    def can_fit(self, vehicle: Vehicle) -> bool:
        return not self.is_occupied and self.size == vehicle.required_spot_size()

    def occupy(self, vehicle: Vehicle) -> None:
        self.is_occupied = True
        self.current_plate = vehicle.plate
        log.info("Spot %s occupied by %s", self.spot_id, vehicle.plate)

    def vacate(self) -> None:
        log.info("Spot %s vacated (was: %s)", self.spot_id, self.current_plate)
        self.is_occupied = False
        self.current_plate = None


class Ticket(BaseModel):
    ticket_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    vehicle: Vehicle
    spot: ParkingSpot
    entry_time: datetime = Field(default_factory=datetime.now)
    exit_time: Optional[datetime] = None

    def close(self) -> None:
        self.exit_time = datetime.now()

    def duration_hours(self) -> float:
        if self.exit_time is None:
            raise ValueError("Ticket not closed yet")
        delta = self.exit_time - self.entry_time
        return max(delta.total_seconds() / 3600, 0.0)


class Payment(BaseModel):
    payment_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    ticket_id: str
    amount: float
    paid_at: datetime = Field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Strategy: Pricing
# ---------------------------------------------------------------------------

class PricingStrategy(ABC):
    @abstractmethod
    def calculate(self, ticket: Ticket) -> float:
        ...


class HourlyPricing(PricingStrategy):
    """
    Rate per hour varies by vehicle type.
    Minimum 1 hour charged.
    """
    RATES: dict[VehicleType, float] = {
        VehicleType.MOTORBIKE: 1.0,
        VehicleType.CAR: 2.0,
        VehicleType.TRUCK: 4.0,
    }

    def calculate(self, ticket: Ticket) -> float:
        hours = max(ticket.duration_hours(), 1.0)
        rate = self.RATES[ticket.vehicle.vehicle_type]
        return round(hours * rate, 2)


class FlatDayRatePricing(PricingStrategy):
    """Flat rate per calendar day regardless of duration."""
    DAY_RATE = 10.0

    def calculate(self, ticket: Ticket) -> float:
        hours = ticket.duration_hours()
        days = max(int(hours / 24) + 1, 1)
        return round(days * self.DAY_RATE, 2)


# ---------------------------------------------------------------------------
# Strategy: Spot Finder
# ---------------------------------------------------------------------------

class SpotFinder(ABC):
    @abstractmethod
    def find_spot(
        self, vehicle: Vehicle, floors: list[Floor]
    ) -> Optional[ParkingSpot]:
        ...


class FirstAvailableFinder(SpotFinder):
    """Scan floors in order, return first matching spot."""

    def find_spot(
        self, vehicle: Vehicle, floors: list[Floor]
    ) -> Optional[ParkingSpot]:
        for floor in floors:
            for spot in floor.spots:
                if spot.can_fit(vehicle):
                    return spot
        return None


# ---------------------------------------------------------------------------
# Floor
# ---------------------------------------------------------------------------

class Floor:
    def __init__(self, number: int, spot_counts: dict[SpotSize, int]) -> None:
        self.number = number
        self.spots: list[ParkingSpot] = []
        for size, count in spot_counts.items():
            for _ in range(count):
                self.spots.append(ParkingSpot(floor_number=number, size=size))
        log.info(
            "Floor %d created: %d spots total",
            number,
            len(self.spots),
        )

    def available_count(self, size: SpotSize) -> int:
        return sum(1 for s in self.spots if s.size == size and not s.is_occupied)


# ---------------------------------------------------------------------------
# ParkingLot — Singleton
# ---------------------------------------------------------------------------

class ParkingLot:
    _instance: Optional[ParkingLot] = None

    def __init__(
        self,
        name: str,
        floors: list[Floor],
        pricing: PricingStrategy,
        finder: SpotFinder,
    ) -> None:
        self.name = name
        self.floors = floors
        self.pricing = pricing
        self.finder = finder
        self._active_tickets: dict[str, Ticket] = {}  # ticket_id -> Ticket

    @classmethod
    def get_instance(
        cls,
        name: str,
        floors: list[Floor],
        pricing: PricingStrategy,
        finder: SpotFinder,
    ) -> ParkingLot:
        if cls._instance is None:
            cls._instance = cls(name, floors, pricing, finder)
            log.info("ParkingLot '%s' created (singleton)", name)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Testing helper — reset singleton."""
        cls._instance = None

    def park(self, vehicle: Vehicle) -> Optional[Ticket]:
        spot = self.finder.find_spot(vehicle, self.floors)
        if spot is None:
            log.warning("No spot available for %s (%s)", vehicle.plate, vehicle.vehicle_type)
            return None
        spot.occupy(vehicle)
        ticket = Ticket(vehicle=vehicle, spot=spot)
        self._active_tickets[ticket.ticket_id] = ticket
        log.info("Issued ticket %s for %s", ticket.ticket_id, vehicle.plate)
        return ticket

    def exit(self, ticket_id: str) -> Optional[Payment]:
        ticket = self._active_tickets.pop(ticket_id, None)
        if ticket is None:
            log.error("Unknown ticket %s", ticket_id)
            return None
        ticket.close()
        ticket.spot.vacate()
        amount = self.pricing.calculate(ticket)
        payment = Payment(ticket_id=ticket.ticket_id, amount=amount)
        log.info(
            "Payment %s: $%.2f for ticket %s", payment.payment_id, amount, ticket_id
        )
        return payment

    def availability_summary(self) -> dict[str, int]:
        summary: dict[str, int] = {}
        for size in SpotSize:
            total = sum(f.available_count(size) for f in self.floors)
            summary[size.name] = total
        return summary


# ---------------------------------------------------------------------------
# Factory: Vehicle creation
# ---------------------------------------------------------------------------

class VehicleFactory:
    _TYPE_MAP: dict[str, VehicleType] = {
        "motorbike": VehicleType.MOTORBIKE,
        "car": VehicleType.CAR,
        "truck": VehicleType.TRUCK,
    }

    @classmethod
    def create(cls, plate: str, type_str: str) -> Vehicle:
        vtype = cls._TYPE_MAP.get(type_str.lower())
        if vtype is None:
            raise ValueError(f"Unknown vehicle type: {type_str!r}")
        return Vehicle(plate=plate, vehicle_type=vtype)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    floors = [
        Floor(
            number=1,
            spot_counts={
                SpotSize.SMALL: 5,
                SpotSize.MEDIUM: 10,
                SpotSize.LARGE: 3,
            },
        ),
        Floor(
            number=2,
            spot_counts={
                SpotSize.SMALL: 5,
                SpotSize.MEDIUM: 10,
                SpotSize.LARGE: 2,
            },
        ),
    ]

    lot = ParkingLot.get_instance(
        name="Central Parking",
        floors=floors,
        pricing=HourlyPricing(),
        finder=FirstAvailableFinder(),
    )

    car = VehicleFactory.create("ABC-123", "car")
    bike = VehicleFactory.create("XYZ-999", "motorbike")

    t1 = lot.park(car)
    t2 = lot.park(bike)

    log.info("Availability: %s", lot.availability_summary())

    if t1:
        payment = lot.exit(t1.ticket_id)
        log.info("Car payment: $%.2f", payment.amount if payment else 0)

    if t2:
        payment2 = lot.exit(t2.ticket_id)
        log.info("Bike payment: $%.2f", payment2.amount if payment2 else 0)


if __name__ == "__main__":
    main()
```

## Exercises

1. **Elevator system** — Apply all 6 steps to design an elevator controller for a 20-floor building with 4 elevators. Write the entity list, relationship diagram, key interfaces (at minimum: `ElevatorController`, `SchedulingStrategy`), and implement the `NearestElevatorScheduler` class that picks the closest idle elevator to a floor request.

2. **Library management** — A library has books (multiple copies), members, and librarians. Members borrow and return books. Overdue fines accumulate at $0.10/day. Apply all 6 steps. Implement `Library.borrow(member_id, isbn) -> BorrowRecord` and `Library.return_book(record_id) -> Fine | None`. Include a `ReservationQueue` for books currently borrowed by others.

3. **Time pressure drill** — Set a 20-minute timer. Design a movie ticket booking system (cinema, halls, seats, shows, bookings) completing Steps 1-4 only. Do not write any implementation code — focus on producing a clean interface specification. Review: did you cover all actor flows? Any missing edge cases?

(Solutions: ../solutions/week-07/day-1.md)

## Self-check

- [ ] I can recite the 6 steps and explain why the order matters
- [ ] I can distinguish LLD from HLD with a concrete example each
- [ ] I can identify entities and relationships for an unfamiliar domain in under 5 minutes
- [ ] I can draw an ASCII class diagram showing cardinality
- [ ] I can implement the parking lot from memory, choosing appropriate patterns
- [ ] I can explain why ParkingLot is a Singleton and PricingStrategy uses Strategy pattern

## Supplementary

- [Refactoring Guru — Design Patterns](https://refactoring.guru/design-patterns)
- [OOD Cheat Sheet — Grokking the OO Design Interview](https://www.educative.io/courses/grokking-the-object-oriented-design-interview)
- [Python ABCs and Protocols](https://docs.python.org/3/library/abc.html)
- [Pydantic v2 Models](https://docs.pydantic.dev/latest/concepts/models/)
