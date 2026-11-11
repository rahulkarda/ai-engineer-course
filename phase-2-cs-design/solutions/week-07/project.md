# Solutions — Week 07 Project (Parking Lot)

The project is fully implemented across the lesson files. This solution file consolidates the complete, runnable package and notes on common mistakes.

---

## Complete `parking_lot/__init__.py`

```python
"""Parking Lot Management System."""
from .models import (
    Vehicle,
    VehicleType,
    ParkingSpot,
    SpotSize,
    Ticket,
    ParkingError,
    ParkingLotFullError,
    SpotAlreadyOccupiedError,
    SpotVehicleMismatchError,
    UnknownTicketError,
    DuplicatePlateError,
)
from .lot import ParkingLot, Level, build_lot, build_lot_from_settings
from .fee import FeeCalculator, HourlyFeeCalculator

__all__ = [
    "Vehicle", "VehicleType", "ParkingSpot", "SpotSize", "Ticket",
    "ParkingError", "ParkingLotFullError", "SpotAlreadyOccupiedError",
    "SpotVehicleMismatchError", "UnknownTicketError", "DuplicatePlateError",
    "ParkingLot", "Level", "build_lot", "build_lot_from_settings",
    "FeeCalculator", "HourlyFeeCalculator",
]
```

---

## Complete `parking_lot/models.py`

```python
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
        rules: dict["SpotSize", set[VehicleType]] = {
            SpotSize.SMALL:  {VehicleType.MOTORCYCLE},
            SpotSize.MEDIUM: {VehicleType.MOTORCYCLE, VehicleType.CAR},
            SpotSize.LARGE:  {VehicleType.MOTORCYCLE, VehicleType.CAR, VehicleType.TRUCK},
        }
        return vehicle_type in rules[self]


class ParkingError(Exception):
    """Base for all parking-lot domain errors."""


class ParkingLotFullError(ParkingError):
    def __init__(self, plate: str) -> None:
        super().__init__(f"Lot is full — cannot park {plate}")


class SpotAlreadyOccupiedError(ParkingError):
    def __init__(self, spot_id: str) -> None:
        super().__init__(f"Spot {spot_id} is already occupied")


class SpotVehicleMismatchError(ParkingError):
    def __init__(self, vehicle_type: str, spot_size: str) -> None:
        super().__init__(f"{vehicle_type} does not fit in {spot_size} spot")


class UnknownTicketError(ParkingError):
    def __init__(self, ticket_id: str) -> None:
        super().__init__(f"Unknown ticket: {ticket_id}")


class DuplicatePlateError(ParkingError):
    def __init__(self, plate: str) -> None:
        super().__init__(f"Vehicle {plate} is already parked")


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
            raise SpotAlreadyOccupiedError(self.spot_id)
        if not self.size.can_fit(vehicle.vehicle_type):
            raise SpotVehicleMismatchError(str(vehicle.vehicle_type), str(self.size))
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

---

## Common mistake corrections

**Mistake:** Using `datetime.now()` directly in `ParkingLot.park()`.
**Fix:** `self._clock = clock or datetime.now` then `entry_time=self._clock()`.

**Mistake:** `class ParkingLot: _tickets = {}` (class-level dict).
**Fix:** `self._tickets: dict[str, Ticket] = {}` in `__init__`.

**Mistake:** `for level in self._levels: for spot in level._spots:` — accessing private `_spots`.
**Fix:** `Level.available_spots(vehicle)` public method; `ParkingLot` never touches spot internals.

**Mistake:** `if vehicle.vehicle_type == VehicleType.CAR and spot.size == SpotSize.SMALL: continue` in `ParkingLot.park()`.
**Fix:** This logic belongs only in `SpotSize.can_fit`. `ParkingLot` calls `level.available_spots(vehicle)` which calls `can_fit` internally.

---

## Running the full test suite

```bash
pip install -e ".[dev]"
pytest tests/ -v --cov=parking_lot --cov-report=term-missing
```

Expected: all tests green, coverage ≥ 85%.

---

## Manual CLI demo

```bash
# Park three vehicles
python -m parking_lot.cli park --plate MOTO1 --type motorcycle
python -m parking_lot.cli park --plate CAR01 --type car
python -m parking_lot.cli park --plate TRCK1 --type truck

# Check status
python -m parking_lot.cli status

# Exit (replace TICKET_ID with actual UUID from park output)
python -m parking_lot.cli exit --plate CAR01
```
