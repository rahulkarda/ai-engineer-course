# Phase 2 · Week 07 · Day 6 — LLD a System: 4-Hour Design Session (Parking Lot)

> Objective: Apply the full OOD process to design a parking-lot management system from scratch — requirements to implementation.  ·  Time: ~4 hr  ·  Prereqs: [[day-1]] through [[day-5]]

---

## Overview

This is a structured 4-hour design session. Work through the phases in order. Each phase has a time box — resist the temptation to skip ahead.

| Phase | Time | Activity |
|-------|------|----------|
| 1 | 30 min | Requirements & use-case analysis |
| 2 | 30 min | Class identification & UML |
| 3 | 60 min | Core implementation skeleton |
| 4 | 60 min | Apply patterns + full implementation |
| 5 | 30 min | Test suite |
| 6 | 30 min | Review & refactor |

---

## Phase 1 — Requirements & Use-Case Analysis (30 min)

### Problem statement

Design a multi-level parking-lot system. The system must:

- Support multiple levels, each with multiple parking spots.
- Handle three vehicle types: Motorcycle, Car, Truck.
- Spot sizes: Small (motorcycles only), Medium (motorcycle or car), Large (any).
- Park a vehicle: find the nearest available appropriate spot, mark it occupied, return a ticket.
- Free a vehicle: accept a ticket, mark spot free, compute and return a fee.
- Show occupancy: level and overall stats.

### Constraints

- No external database for now (in-memory is fine).
- Concurrent access not required (single-threaded).
- Fee model: flat rate per hour, configurable per vehicle type.

### Actors & use cases

**Actor: Driver**
- Park vehicle → get ticket
- Pay & exit → return ticket, get receipt

**Actor: Operator**
- View current occupancy
- Configure fee rates

### Questions you should answer before designing

1. Can a motorcycle park in a Large spot if no Small/Medium is available?
2. Is a ticket tied to a spot or to a vehicle?
3. How is time tracked — wall clock or injected clock (for tests)?
4. What happens when the lot is full?

Write your answers before moving on. (Suggested answers in solutions.)

---

## Phase 2 — Class Identification & UML (30 min)

### Noun extraction (candidate classes)

From the requirements, underline nouns:

> A **parking lot** has multiple **levels**. Each **level** has multiple **spots**. A **spot** has a **size**. A **vehicle** has a **type** and a **plate**. A **ticket** records the **spot**, **vehicle**, **entry time**. A **fee calculator** computes the **charge**.

Candidate classes: `ParkingLot`, `Level`, `ParkingSpot`, `Vehicle` (+ subtypes), `Ticket`, `FeeCalculator`.

### Class diagram (sketch before coding)

```
ParkingLot
  ├── levels: list[Level]        (composition)
  └── fee_calculator: FeeCalculator  (dependency)

Level
  ├── level_id: int
  └── spots: list[ParkingSpot]  (composition)

ParkingSpot
  ├── spot_id: str
  ├── size: SpotSize
  ├── is_occupied: bool
  └── vehicle: Vehicle | None

Vehicle  (abstract)
  ├── plate: str
  └── type: VehicleType
  Motorcycle, Car, Truck  (concrete)

Ticket
  ├── ticket_id: UUID
  ├── vehicle: Vehicle
  ├── spot: ParkingSpot
  └── entry_time: datetime

FeeCalculator  (abstract)
  └── HourlyFeeCalculator  (concrete)
```

### Key design decisions

1. **Vehicle fitting**: `SpotSize.can_fit(vehicle_type)` — encapsulates sizing rules.
2. **Spot selection**: `Level.find_spot(vehicle)` — Strategy pattern for spot-selection algorithm.
3. **Fee calculation**: injected `FeeCalculator` — open/closed, testable with a stub.
4. **Clock**: injected `Callable[[], datetime]` — replaces `datetime.now()` for deterministic tests.

---

## Phase 3 — Core Implementation Skeleton (60 min)

```python
"""
parking_lot/models.py — domain models (no logic, pure data)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum, auto
from uuid import UUID, uuid4


class VehicleType(StrEnum):
    MOTORCYCLE = auto()
    CAR = auto()
    TRUCK = auto()


class SpotSize(StrEnum):
    SMALL = auto()
    MEDIUM = auto()
    LARGE = auto()

    def can_fit(self, vehicle_type: VehicleType) -> bool:
        """Return True if this spot size can accommodate the vehicle."""
        rules: dict[SpotSize, set[VehicleType]] = {
            SpotSize.SMALL:  {VehicleType.MOTORCYCLE},
            SpotSize.MEDIUM: {VehicleType.MOTORCYCLE, VehicleType.CAR},
            SpotSize.LARGE:  {VehicleType.MOTORCYCLE, VehicleType.CAR, VehicleType.TRUCK},
        }
        return vehicle_type in rules[self]


@dataclass
class Vehicle:
    plate: str
    vehicle_type: VehicleType


@dataclass
class ParkingSpot:
    spot_id: str
    size: SpotSize
    level_id: int
    vehicle: Vehicle | None = None

    @property
    def is_occupied(self) -> bool:
        return self.vehicle is not None

    def park(self, vehicle: Vehicle) -> None:
        if self.is_occupied:
            raise RuntimeError(f"Spot {self.spot_id} already occupied")
        if not self.size.can_fit(vehicle.vehicle_type):
            raise ValueError(f"{vehicle.vehicle_type} does not fit in {self.size} spot")
        self.vehicle = vehicle

    def vacate(self) -> Vehicle:
        if not self.is_occupied:
            raise RuntimeError(f"Spot {self.spot_id} is empty")
        v = self.vehicle
        self.vehicle = None
        return v  # type: ignore[return-value]


@dataclass
class Ticket:
    ticket_id: UUID
    vehicle: Vehicle
    spot: ParkingSpot
    entry_time: datetime
```

```python
"""
parking_lot/fee.py — fee calculation (Strategy pattern)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime
from .models import VehicleType


class FeeCalculator(ABC):
    @abstractmethod
    def compute(self, vehicle_type: VehicleType, entry: datetime, exit: datetime) -> float:
        """Return fee in dollars."""
        ...


class HourlyFeeCalculator(FeeCalculator):
    """Flat hourly rate, configurable per vehicle type."""

    DEFAULT_RATES: dict[VehicleType, float] = {
        VehicleType.MOTORCYCLE: 1.0,
        VehicleType.CAR:        2.0,
        VehicleType.TRUCK:      3.5,
    }

    def __init__(self, rates: dict[VehicleType, float] | None = None) -> None:
        self._rates = rates or self.DEFAULT_RATES.copy()

    def compute(self, vehicle_type: VehicleType, entry: datetime, exit: datetime) -> float:
        elapsed_hours = max((exit - entry).total_seconds() / 3600, 1/6)  # min 10 min
        return round(elapsed_hours * self._rates[vehicle_type], 2)
```

---

## Phase 4 — Full Implementation with Patterns (60 min)

```python
"""
parking_lot/lot.py — ParkingLot, Level, and spot-selection strategy
"""
from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable
from uuid import uuid4

from .fee import FeeCalculator, HourlyFeeCalculator
from .models import ParkingSpot, SpotSize, Ticket, Vehicle, VehicleType

log = logging.getLogger(__name__)


# ─── Spot selection strategy ────────────────────────────────────────────────

class SpotSelectionStrategy(ABC):
    """Strategy: how to pick a spot from available candidates."""

    @abstractmethod
    def select(self, candidates: list[ParkingSpot]) -> ParkingSpot | None: ...


class FirstAvailableStrategy(SpotSelectionStrategy):
    """Pick the first available spot (list order = spot ID order)."""

    def select(self, candidates: list[ParkingSpot]) -> ParkingSpot | None:
        return candidates[0] if candidates else None


class SmallestFitStrategy(SpotSelectionStrategy):
    """Pick the smallest spot that fits (minimise waste)."""

    SIZE_ORDER = [SpotSize.SMALL, SpotSize.MEDIUM, SpotSize.LARGE]

    def select(self, candidates: list[ParkingSpot]) -> ParkingSpot | None:
        if not candidates:
            return None
        return min(candidates, key=lambda s: self.SIZE_ORDER.index(s.size))


# ─── Level ──────────────────────────────────────────────────────────────────

class Level:
    def __init__(self, level_id: int, spots: list[ParkingSpot]) -> None:
        self._level_id = level_id
        self._spots = spots

    @property
    def level_id(self) -> int:
        return self._level_id

    def available_spots(self, vehicle: Vehicle) -> list[ParkingSpot]:
        return [
            s for s in self._spots
            if not s.is_occupied and s.size.can_fit(vehicle.vehicle_type)
        ]

    def occupancy(self) -> dict[str, int]:
        total = len(self._spots)
        occupied = sum(1 for s in self._spots if s.is_occupied)
        return {"total": total, "occupied": occupied, "available": total - occupied}


# ─── Parking Lot ─────────────────────────────────────────────────────────────

class ParkingLot:
    def __init__(
        self,
        levels: list[Level],
        fee_calculator: FeeCalculator | None = None,
        strategy: SpotSelectionStrategy | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._levels = levels
        self._fee_calc = fee_calculator or HourlyFeeCalculator()
        self._strategy = strategy or SmallestFitStrategy()
        self._clock = clock or datetime.now
        self._tickets: dict[str, Ticket] = {}   # ticket_id str -> Ticket

    def park(self, vehicle: Vehicle) -> Ticket:
        """Find a spot and issue a ticket. Raises RuntimeError if full."""
        for level in self._levels:
            candidates = level.available_spots(vehicle)
            spot = self._strategy.select(candidates)
            if spot is not None:
                spot.park(vehicle)
                ticket = Ticket(
                    ticket_id=uuid4(),
                    vehicle=vehicle,
                    spot=spot,
                    entry_time=self._clock(),
                )
                self._tickets[str(ticket.ticket_id)] = ticket
                log.info("Parked %s at spot %s (level %d)",
                         vehicle.plate, spot.spot_id, spot.level_id)
                return ticket
        raise RuntimeError(f"Parking lot is full — cannot park {vehicle.plate}")

    def exit(self, ticket_id: str) -> float:
        """Free the spot and return the fee. Raises KeyError if unknown ticket."""
        ticket = self._tickets.pop(ticket_id, None)
        if ticket is None:
            raise KeyError(f"Unknown ticket: {ticket_id}")
        exit_time = self._clock()
        ticket.spot.vacate()
        fee = self._fee_calc.compute(ticket.vehicle.vehicle_type, ticket.entry_time, exit_time)
        log.info("Vehicle %s exited. Fee: $%.2f", ticket.vehicle.plate, fee)
        return fee

    def occupancy(self) -> dict[str, object]:
        """Return occupancy stats per level and overall."""
        levels_stats = {
            f"level_{lvl.level_id}": lvl.occupancy()
            for lvl in self._levels
        }
        totals = {"total": 0, "occupied": 0, "available": 0}
        for stats in levels_stats.values():
            for k in totals:
                totals[k] += stats[k]  # type: ignore[operator]
        return {"levels": levels_stats, "totals": totals}


# ─── Factory helper ──────────────────────────────────────────────────────────

def build_lot(levels: int = 3, spots_per_level: int = 10) -> ParkingLot:
    """Build a standard lot with mixed spot sizes (40% S, 40% M, 20% L)."""
    sizes = (
        [SpotSize.SMALL]  * int(spots_per_level * 0.4) +
        [SpotSize.MEDIUM] * int(spots_per_level * 0.4) +
        [SpotSize.LARGE]  * (spots_per_level - int(spots_per_level * 0.8))
    )
    built_levels = []
    for lvl_id in range(levels):
        spots = [
            ParkingSpot(spot_id=f"L{lvl_id}-{i:02d}", size=size, level_id=lvl_id)
            for i, size in enumerate(sizes)
        ]
        built_levels.append(Level(lvl_id, spots))
    return ParkingLot(built_levels)
```

---

## Phase 5 — Test Suite (30 min)

```python
"""
tests/test_parking_lot.py — pytest suite for the parking-lot system
"""
from __future__ import annotations
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from parking_lot.fee import HourlyFeeCalculator
from parking_lot.lot import FirstAvailableStrategy, Level, ParkingLot, SmallestFitStrategy
from parking_lot.models import ParkingSpot, SpotSize, Vehicle, VehicleType


# ─── Fixtures ────────────────────────────────────────────────────────────────

ENTRY = datetime(2024, 1, 1, 9, 0, 0)
EXIT_1H = ENTRY + timedelta(hours=1)
EXIT_2H = ENTRY + timedelta(hours=2)


def car(plate: str = "ABC123") -> Vehicle:
    return Vehicle(plate=plate, vehicle_type=VehicleType.CAR)


def motorcycle(plate: str = "MOTO1") -> Vehicle:
    return Vehicle(plate=plate, vehicle_type=VehicleType.MOTORCYCLE)


def truck(plate: str = "TRCK1") -> Vehicle:
    return Vehicle(plate=plate, vehicle_type=VehicleType.TRUCK)


def small_lot(clock_fn=None) -> ParkingLot:
    """1 level, 1 small, 1 medium, 1 large spot."""
    spots = [
        ParkingSpot("S-00", SpotSize.SMALL,  0),
        ParkingSpot("S-01", SpotSize.MEDIUM, 0),
        ParkingSpot("S-02", SpotSize.LARGE,  0),
    ]
    level = Level(0, spots)
    return ParkingLot([level], clock=clock_fn)


# ─── Unit: SpotSize.can_fit ──────────────────────────────────────────────────

class TestSpotSizeFit:
    def test_small_fits_motorcycle(self):
        assert SpotSize.SMALL.can_fit(VehicleType.MOTORCYCLE)

    def test_small_rejects_car(self):
        assert not SpotSize.SMALL.can_fit(VehicleType.CAR)

    def test_medium_fits_car(self):
        assert SpotSize.MEDIUM.can_fit(VehicleType.CAR)

    def test_medium_rejects_truck(self):
        assert not SpotSize.MEDIUM.can_fit(VehicleType.TRUCK)

    def test_large_fits_truck(self):
        assert SpotSize.LARGE.can_fit(VehicleType.TRUCK)


# ─── Unit: HourlyFeeCalculator ───────────────────────────────────────────────

class TestFeeCalculator:
    def test_car_one_hour(self):
        calc = HourlyFeeCalculator()
        fee = calc.compute(VehicleType.CAR, ENTRY, EXIT_1H)
        assert fee == 2.0

    def test_truck_two_hours(self):
        calc = HourlyFeeCalculator()
        fee = calc.compute(VehicleType.TRUCK, ENTRY, EXIT_2H)
        assert fee == 7.0

    def test_minimum_fee(self):
        calc = HourlyFeeCalculator()
        # 1 minute stay — minimum 10-minute billing
        exit_min = ENTRY + timedelta(minutes=1)
        fee = calc.compute(VehicleType.CAR, ENTRY, exit_min)
        assert fee > 0


# ─── Integration: ParkingLot ─────────────────────────────────────────────────

class TestParkingLot:
    def _fixed_clock(self, t: datetime):
        return lambda: t

    def test_park_and_exit(self):
        calls = [ENTRY, EXIT_1H]
        idx = 0

        def clock():
            nonlocal idx
            val = calls[idx]
            idx += 1
            return val

        lot = small_lot(clock_fn=clock)
        ticket = lot.park(car())
        fee = lot.exit(str(ticket.ticket_id))
        assert fee == 2.0  # CAR, 1h

    def test_lot_full_raises(self):
        lot = small_lot()
        # Park 3 trucks — only 1 large spot, so 2nd truck should fail
        lot.park(truck("T1"))
        with pytest.raises(RuntimeError, match="full"):
            lot.park(truck("T2"))

    def test_motorcycle_uses_small_spot_first(self):
        lot = small_lot()
        ticket = lot.park(motorcycle())
        assert ticket.spot.size == SpotSize.SMALL  # SmallestFitStrategy

    def test_car_skips_small_spot(self):
        lot = small_lot()
        ticket = lot.park(car())
        assert ticket.spot.size == SpotSize.MEDIUM

    def test_unknown_ticket_raises(self):
        lot = small_lot()
        with pytest.raises(KeyError):
            lot.exit("nonexistent")

    def test_occupancy_after_parking(self):
        lot = small_lot()
        lot.park(car())
        stats = lot.occupancy()
        assert stats["totals"]["occupied"] == 1
        assert stats["totals"]["available"] == 2

    def test_spot_freed_after_exit(self):
        calls = [ENTRY, EXIT_1H]
        idx = 0
        def clock():
            nonlocal idx; v = calls[idx]; idx += 1; return v
        lot = small_lot(clock_fn=clock)
        ticket = lot.park(car())
        lot.exit(str(ticket.ticket_id))
        stats = lot.occupancy()
        assert stats["totals"]["occupied"] == 0
```

---

## Phase 6 — Review & Refactor (30 min)

Work through these refactoring prompts:

1. **Extensibility check.** If you added an `Electric` vehicle type needing a charging spot, which classes change? Should be zero existing classes if OCP holds.

2. **Coupling audit.** Does `ParkingLot` know about `SpotSize` directly? It shouldn't — that rule lives in `SpotSize.can_fit`. Verify.

3. **Naming pass.** Rename any method that starts with "get_" if it returns a computed value — prefer noun (`.occupancy()`, not `.get_occupancy()`).

4. **Missing edge case.** What happens if a vehicle tries to park with the same plate twice? Should the system prevent this? Add a check in `ParkingLot.park()` if you decide it should.

5. **Error handling.** Replace bare `RuntimeError` with domain-specific exceptions (`ParkingLotFullError`, `SpotAlreadyOccupiedError`). Add them to `models.py`.

---

## Self-check

- [ ] I completed all six phases within the time boxes.
- [ ] My class diagram matched (or was close to) the implementation.
- [ ] I applied at least three of: Factory, Strategy, Repository pattern, DI.
- [ ] All unit and integration tests pass.
- [ ] I identified at least two refactoring improvements in Phase 6.

---

## Supplementary links

- [LLD Interview: Parking Lot](https://www.educative.io/courses/grokking-the-object-oriented-design-interview/parking-lot) — Educative (reference only)
- *Clean Code* — Robert C. Martin, Chapter 3 (functions / single responsibility)
- [Python `datetime` and testing](https://docs.python.org/3/library/datetime.html) — inject the clock, don't call `datetime.now()` directly
