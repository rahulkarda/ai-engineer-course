# Solutions — Week 07 · Day 7

Day 7 is a build session. The full reference implementation is the complete set of files described in the lesson. Below are notes on common mistakes and the stretch-goal implementations.

---

## Common mistakes to check

1. **`datetime.now()` called inside `__init__`** — will freeze time at construction. Always inject the clock.
2. **`ParkingLot` knows `SpotSize` directly** — sizing logic should live only in `SpotSize.can_fit`.
3. **Mutable class variable** — `class X: tickets = {}` is shared across all instances. Always use `self._tickets = {}` in `__init__`.
4. **Missing `__init__.py`** — package won't import without it.
5. **`get_settings()` called at import time** — wrap in `@lru_cache` and call lazily.

---

## Stretch goal 1 — Thread-safe spot locking

```python
# In models.py
import threading
from dataclasses import dataclass, field

@dataclass
class ParkingSpot:
    spot_id: str
    size: "SpotSize"
    level_id: int
    vehicle: "Vehicle | None" = None
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def park(self, vehicle: "Vehicle") -> None:
        with self._lock:
            if self.vehicle is not None:
                raise SpotAlreadyOccupiedError(self.spot_id)
            if not self.size.can_fit(vehicle.vehicle_type):
                raise SpotVehicleMismatchError(vehicle.vehicle_type, self.size)
            self.vehicle = vehicle

    def vacate(self) -> "Vehicle":
        with self._lock:
            if self.vehicle is None:
                raise RuntimeError(f"Spot {self.spot_id} is empty")
            v = self.vehicle
            self.vehicle = None
            return v
```

Concurrent test:
```python
import threading
import pytest
from parking_lot.lot import build_lot
from parking_lot.models import Vehicle, VehicleType


def test_concurrent_parking():
    lot = build_lot(levels=1, spots_per_level=10)
    errors: list[Exception] = []
    tickets = []
    lock = threading.Lock()

    def park_vehicle(plate: str) -> None:
        try:
            t = lot.park(Vehicle(plate=plate, vehicle_type=VehicleType.CAR))
            with lock:
                tickets.append(t)
        except Exception as exc:
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=park_vehicle, args=(f"P{i:03d}",)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Some threads might fail (lot full) but no spot should be double-parked
    spot_ids = [t.spot.spot_id for t in tickets]
    assert len(spot_ids) == len(set(spot_ids)), "No spot parked twice"
```

---

## Stretch goal 3 — FastAPI wrapper (sketch)

```python
# parking_lot/api.py
from __future__ import annotations
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from .lot import ParkingLot, build_lot_from_settings
from .models import Vehicle, VehicleType

app = FastAPI(title="Parking Lot API")

_LOT: ParkingLot | None = None


def get_lot() -> ParkingLot:
    global _LOT
    if _LOT is None:
        _LOT = build_lot_from_settings()
    return _LOT


class ParkRequest(BaseModel):
    plate: str
    vehicle_type: VehicleType


@app.post("/park")
def park(req: ParkRequest, lot: ParkingLot = Depends(get_lot)):
    try:
        ticket = lot.park(Vehicle(plate=req.plate, vehicle_type=req.vehicle_type))
        return {"ticket_id": str(ticket.ticket_id)}
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/exit/{ticket_id}")
def exit_lot(ticket_id: str, lot: ParkingLot = Depends(get_lot)):
    try:
        fee = lot.exit(ticket_id)
        return {"fee": fee}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/status")
def status(lot: ParkingLot = Depends(get_lot)):
    return lot.occupancy()
```

Run with: `uvicorn parking_lot.api:app --reload`
