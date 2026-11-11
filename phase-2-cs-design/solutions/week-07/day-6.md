# Solutions — Week 07 · Day 6

## Phase 1 answers — Requirements questions

**Q1: Can a motorcycle park in a Large spot if no Small/Medium is available?**
Yes. `SpotSize.can_fit` returns `True` for Large + Motorcycle. `SmallestFitStrategy` prefers the smallest fitting spot, so Large is used only when no smaller option exists.

**Q2: Is a ticket tied to a spot or to a vehicle?**
A ticket is tied to both — it records the spot assigned and the vehicle that parked. The ticket is the contract between the lot and the driver.

**Q3: How is time tracked?**
The clock is injected as `Callable[[], datetime]` defaulting to `datetime.now`. Tests pass a lambda that returns a fixed or advancing datetime, giving fully deterministic time-based tests.

**Q4: What happens when the lot is full?**
`ParkingLot.park()` raises `ParkingLotFullError` after exhausting all levels. The caller (CLI or API) handles this gracefully — it is a domain exception, not a programming error.

---

## Phase 2 — Additional design notes

**Why `SpotSize.can_fit` on the model, not on `Level`?**
The sizing rule is a property of the spot itself, not of the level. Keeping it on `SpotSize` means no scattered `if size == SMALL and vehicle_type == MOTORCYCLE` conditionals across the codebase — there is exactly one place to change if the rules change.

**Why inject the strategy into `ParkingLot`, not `Level`?**
A single strategy governs the entire lot (across levels). Injecting at the lot level allows global policies like "always use the deepest available spot first" or "prefer the most-full level."

---

## Phase 6 refactoring answers

**1. Electric vehicle type — what changes?**
- Add `VehicleType.ELECTRIC` to the enum.
- Add `SpotSize.CHARGING` (or extend existing sizes with a `has_charger: bool` flag).
- No other existing classes change if `SpotSize.can_fit` is the only sizing authority. OCP holds.

**2. Coupling audit.**
`ParkingLot` calls `level.available_spots(vehicle)` — it never inspects `SpotSize` directly. `Level.available_spots` filters by `spot.size.can_fit(vehicle.vehicle_type)`. `ParkingLot` is decoupled from sizing rules. ✓

**3. Naming pass — no changes needed.**
`lot.occupancy()`, `lot.park()`, `lot.exit()` are all already verb (action) or noun (query) forms without `get_` prefix.

**4. Duplicate plate prevention.**
```python
# In ParkingLot.park():
active_plates = {t.vehicle.plate for t in self._tickets.values()}
if vehicle.plate in active_plates:
    raise DuplicatePlateError(vehicle.plate)
```

**5. Domain exceptions (already added in Day 7):**
```python
class ParkingError(Exception): ...
class ParkingLotFullError(ParkingError): ...
class SpotAlreadyOccupiedError(ParkingError): ...
class SpotVehicleMismatchError(ParkingError): ...
class UnknownTicketError(ParkingError): ...
class DuplicatePlateError(ParkingError): ...
```
