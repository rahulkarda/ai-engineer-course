# Phase 2 · Week 07 · Day 7 — Week Project: Parking-Lot System (Full Build & Polish)

> Objective: Deliver a production-quality Python parking-lot implementation with complete tests, packaging, and documentation.  ·  Time: ~3 hr project + see assessment.md  ·  Prereqs: [[day-6]]

---

## Overview

Day 7 takes the design from Day 6 and makes it production-ready: proper package layout, full test coverage, error hierarchy, configuration via environment variables, and a CLI entry point.

---

## Package layout

```
parking_lot/
├── __init__.py
├── models.py          # domain entities (Vehicle, ParkingSpot, Ticket, exceptions)
├── fee.py             # FeeCalculator ABC + HourlyFeeCalculator
├── lot.py             # Level, ParkingLot, SpotSelectionStrategy
├── config.py          # Settings via pydantic-settings
└── cli.py             # argparse CLI for manual testing
tests/
├── __init__.py
├── test_models.py
├── test_fee.py
└── test_lot.py
pyproject.toml
```

---

## Step 1 — Domain exceptions (`models.py` addition)

```python
# Add to parking_lot/models.py

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
```

Update `ParkingSpot.park()`, `ParkingSpot.vacate()`, and `ParkingLot.park()` / `ParkingLot.exit()` to raise these instead of bare exceptions.

---

## Step 2 — Configuration (`config.py`)

```python
"""
parking_lot/config.py
Environment-driven configuration using pydantic-settings.
"""
from __future__ import annotations

import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class ParkingLotSettings(BaseSettings):
    """All config from environment; never hardcode values."""

    num_levels: int = Field(default=3, ge=1, le=20)
    spots_per_level: int = Field(default=20, ge=1, le=200)

    # Fee rates (USD per hour) — override via env vars
    motorcycle_rate: float = Field(default=1.0, gt=0)
    car_rate: float = Field(default=2.0, gt=0)
    truck_rate: float = Field(default=3.5, gt=0)

    model_config = {"env_prefix": "PARKING_"}


@lru_cache(maxsize=1)
def get_settings() -> ParkingLotSettings:
    return ParkingLotSettings()
```

Usage:
```bash
PARKING_CAR_RATE=2.50 PARKING_NUM_LEVELS=5 python -m parking_lot.cli park --plate ABC123 --type car
```

---

## Step 3 — Update `lot.py` to use settings

```python
# parking_lot/lot.py  (updated factory helper)

from .config import get_settings
from .models import VehicleType

def build_lot_from_settings() -> ParkingLot:
    """Build a ParkingLot from environment configuration."""
    cfg = get_settings()
    rates = {
        VehicleType.MOTORCYCLE: cfg.motorcycle_rate,
        VehicleType.CAR:        cfg.car_rate,
        VehicleType.TRUCK:      cfg.truck_rate,
    }
    return build_lot(
        levels=cfg.num_levels,
        spots_per_level=cfg.spots_per_level,
        fee_calculator=HourlyFeeCalculator(rates),
    )
```

---

## Step 4 — CLI entry point (`cli.py`)

```python
"""
parking_lot/cli.py
Simple argparse CLI for manual interaction with the parking lot.
Run: python -m parking_lot.cli --help
"""
from __future__ import annotations

import argparse
import logging
import sys

from .lot import build_lot_from_settings
from .models import Vehicle, VehicleType

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Module-level lot (demo only — not for production multi-process)
_LOT = build_lot_from_settings()
_ACTIVE_TICKETS: dict[str, str] = {}  # plate -> ticket_id


def cmd_park(args: argparse.Namespace) -> None:
    try:
        vtype = VehicleType(args.type.lower())
    except ValueError:
        log.error("Unknown vehicle type: %s", args.type)
        sys.exit(1)
    vehicle = Vehicle(plate=args.plate, vehicle_type=vtype)
    try:
        ticket = _LOT.park(vehicle)
        print(f"Parked. Ticket ID: {ticket.ticket_id}")
        _ACTIVE_TICKETS[args.plate] = str(ticket.ticket_id)
    except Exception as exc:
        log.error("Park failed: %s", exc)
        sys.exit(1)


def cmd_exit(args: argparse.Namespace) -> None:
    ticket_id = _ACTIVE_TICKETS.pop(args.plate, None)
    if not ticket_id:
        log.error("No active ticket for plate %s", args.plate)
        sys.exit(1)
    try:
        fee = _LOT.exit(ticket_id)
        print(f"Fee: ${fee:.2f}")
    except Exception as exc:
        log.error("Exit failed: %s", exc)
        sys.exit(1)


def cmd_status(_args: argparse.Namespace) -> None:
    occ = _LOT.occupancy()
    print("Occupancy:")
    for level, stats in occ["levels"].items():
        print(f"  {level}: {stats['occupied']}/{stats['total']} occupied")
    totals = occ["totals"]
    print(f"  TOTAL: {totals['occupied']}/{totals['total']} occupied")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parking Lot CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_park = sub.add_parser("park", help="Park a vehicle")
    p_park.add_argument("--plate", required=True)
    p_park.add_argument("--type", required=True, choices=["motorcycle", "car", "truck"])
    p_park.set_defaults(func=cmd_park)

    p_exit = sub.add_parser("exit", help="Exit and pay")
    p_exit.add_argument("--plate", required=True)
    p_exit.set_defaults(func=cmd_exit)

    p_status = sub.add_parser("status", help="Show occupancy")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

---

## Step 5 — `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=70", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "parking-lot"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["pydantic>=2.0", "pydantic-settings>=2.0"]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=5.0", "ruff>=0.4"]

[project.scripts]
parking-lot = "parking_lot.cli:main"

[tool.ruff]
line-length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
```

---

## Step 6 — Stretch goals (if time allows)

1. **Concurrent access.** Add a `threading.Lock` per spot, acquired in `park()` and `exit()`. Write a test that parks 10 vehicles from 10 threads and asserts no double-parking.

2. **Persistence.** Implement `SqliteParkingSpotRepository` that stores spot state in a local SQLite file. The lot loads state on init and saves on every `park()` / `exit()`.

3. **Async API.** Wrap `ParkingLot` in a FastAPI app with endpoints `POST /park`, `POST /exit/{ticket_id}`, `GET /status`. Use `Annotated[ParkingLot, Depends(get_lot)]` for injection.

4. **Observability.** Add an Observer event bus (from Day 4) that fires `VehicleParked` / `VehicleExited` events. Wire a `MetricsObserver` that counts events and exposes them via `/metrics`.

---

## Self-check

- [ ] All domain exceptions extend `ParkingError`.
- [ ] Config is loaded from environment — no hardcoded values.
- [ ] `pytest tests/ -v` passes with ≥ 90% line coverage.
- [ ] CLI responds correctly to `park`, `exit`, `status` subcommands.
- [ ] At least one stretch goal is complete.

---

## Supplementary links

- [pydantic-settings docs](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [pytest-cov](https://pytest-cov.readthedocs.io/) — measure coverage
- [Python `argparse` docs](https://docs.python.org/3/library/argparse.html)
