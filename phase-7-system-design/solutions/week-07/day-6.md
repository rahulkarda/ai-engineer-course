# Solutions — Phase 7 · Week 07 · Day 6 (4-Hour Practice)

## Problem A: Elevator System — Full Solution

```python
"""
elevator_system.py — Day 6 Block A full solution.

Implements a multi-elevator building controller with:
  - SCAN (look-ahead) scheduling algorithm
  - Per-car threading.Lock for concurrent hall calls
  - Door state machine (CLOSED / OPENING / OPEN / CLOSING)
  - step()-based simulation (no real-time loop required)

Python 3.12+.  External dependency: sortedcontainers.
    pip install sortedcontainers
"""

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum, auto

from sortedcontainers import SortedList  # type: ignore[import-untyped]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Direction(Enum):
    UP = auto()
    DOWN = auto()
    IDLE = auto()


class DoorState(Enum):
    CLOSED = auto()
    OPENING = auto()
    OPEN = auto()
    CLOSING = auto()


class RequestType(Enum):
    INTERNAL = auto()   # car call: passenger pressed floor button inside car
    EXTERNAL = auto()   # hall call: passenger pressed UP/DOWN on a floor


# ---------------------------------------------------------------------------
# FloorRequest dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FloorRequest:
    floor: int
    request_type: RequestType
    direction: Direction = Direction.IDLE   # relevant for EXTERNAL requests


# ---------------------------------------------------------------------------
# Door
# ---------------------------------------------------------------------------


class Door:
    """
    Four-state door controller.

    Tick-based: each call to tick() advances one simulation step.

    CLOSED  --(open tick)-->  OPENING
    OPENING --(open tick)-->  OPEN
    OPEN    --(close tick)--> CLOSING
    CLOSING --(close tick)--> CLOSED

    obstruct() re-opens a CLOSING door.
    """

    def __init__(self) -> None:
        self._state: DoorState = DoorState.CLOSED

    @property
    def state(self) -> DoorState:
        return self._state

    def is_closed(self) -> bool:
        return self._state is DoorState.CLOSED

    def is_open(self) -> bool:
        return self._state is DoorState.OPEN

    def start_opening(self) -> None:
        """Transition CLOSED -> OPENING."""
        if self._state is not DoorState.CLOSED:
            log.warning("start_opening called from state %s", self._state.name)
            return
        self._state = DoorState.OPENING
        log.debug("door: CLOSED -> OPENING")

    def tick(self) -> None:
        """Advance one simulation step."""
        if self._state is DoorState.OPENING:
            self._state = DoorState.OPEN
            log.debug("door: OPENING -> OPEN")
        elif self._state is DoorState.CLOSING:
            self._state = DoorState.CLOSED
            log.debug("door: CLOSING -> CLOSED")
        # OPEN and CLOSED are stable; call start_closing() / start_opening() explicitly

    def start_closing(self) -> None:
        """Transition OPEN -> CLOSING."""
        if self._state is not DoorState.OPEN:
            log.warning("start_closing called from state %s", self._state.name)
            return
        self._state = DoorState.CLOSING
        log.debug("door: OPEN -> CLOSING")

    def obstruct(self) -> None:
        """Obstruction: reverse CLOSING -> OPENING."""
        if self._state is DoorState.CLOSING:
            self._state = DoorState.OPENING
            log.debug("door: obstruction — CLOSING -> OPENING")


# ---------------------------------------------------------------------------
# ElevatorCar
# ---------------------------------------------------------------------------


class ElevatorCar:
    """
    Single elevator car.

    Stop queues:
      _up_stops   — SortedList ascending; serviced when direction is UP
      _down_stops — SortedList descending; serviced when direction is DOWN

    SCAN algorithm:
      - While travelling UP, service all up_stops in ascending order.
      - When up_stops is empty, switch to DOWN and service down_stops in
        descending order.  Symmetric for DOWN.
      - If both queues are empty, become IDLE.

    Thread safety: acquire self.lock before any mutation.
    """

    def __init__(self, car_id: int, top_floor: int) -> None:
        self.id: int = car_id
        self.top_floor: int = top_floor
        self.current_floor: int = 0
        self.direction: Direction = Direction.IDLE
        self.door: Door = Door()

        # Two separate sorted structures for SCAN
        self._up_stops: SortedList[int] = SortedList()
        self._down_stops: SortedList[int] = SortedList()

        self.lock: threading.Lock = threading.Lock()

    # ---- stop management ----

    def add_stop(self, floor: int) -> None:
        """
        Add floor to the appropriate stop queue.

        If the car is IDLE, the floor determines its initial direction.
        If the car is moving and the floor is in the current direction,
        it goes to the current-direction queue; otherwise to the opposite queue.
        """
        with self.lock:
            self._add_stop_unsafe(floor)

    def _add_stop_unsafe(self, floor: int) -> None:
        """Add stop without acquiring lock.  Caller must hold self.lock."""
        if floor < 0 or floor > self.top_floor:
            raise ValueError(f"Floor {floor} out of range [0, {self.top_floor}]")

        if self.direction is Direction.UP or (
            self.direction is Direction.IDLE and floor >= self.current_floor
        ):
            if floor not in self._up_stops:
                self._up_stops.add(floor)
                log.debug("car %d: added up stop %d", self.id, floor)
        else:
            if floor not in self._down_stops:
                self._down_stops.add(floor)
                log.debug("car %d: added down stop %d", self.id, floor)

    def next_stop(self) -> int | None:
        """Return the next floor to visit without mutating state.  Thread-safe."""
        with self.lock:
            return self._next_stop_unsafe()

    def _next_stop_unsafe(self) -> int | None:
        """Next stop without acquiring lock.  Caller must hold self.lock."""
        if self.direction is Direction.UP and self._up_stops:
            return self._up_stops[0]
        if self.direction is Direction.DOWN and self._down_stops:
            return self._down_stops[-1]
        # IDLE or direction exhausted: check the other queue
        if self._up_stops:
            return self._up_stops[0]
        if self._down_stops:
            return self._down_stops[-1]
        return None

    # ---- simulation tick ----

    def step(self) -> None:
        """
        Advance simulation by one tick.

        Tick semantics (in order of priority):
        1. If door is OPENING or CLOSING, tick the door state machine.
        2. If door is OPEN, start closing.
        3. If IDLE and there is a pending stop, start moving.
        4. If MOVING_UP / MOVING_DOWN, move one floor; open door if it is a stop.
        """
        with self.lock:
            self._step_unsafe()

    def _step_unsafe(self) -> None:  # noqa: C901
        """Step without acquiring lock.  Caller must hold self.lock."""
        # --- door transitions take priority ---
        if self.door.state in (DoorState.OPENING, DoorState.CLOSING):
            self.door.tick()
            # After closing completes, decide whether to continue or go idle
            if self.door.is_closed():
                self._choose_direction()
            return

        if self.door.is_open():
            self.door.start_closing()
            return

        # --- door is CLOSED from here ---

        if self.direction is Direction.IDLE:
            self._choose_direction()
            return

        # Move one floor in current direction
        if self.direction is Direction.UP:
            self.current_floor += 1
            log.debug("car %d: moved UP -> floor %d", self.id, self.current_floor)
            if self.current_floor in self._up_stops:
                self._up_stops.remove(self.current_floor)
                log.debug("car %d: serving up stop at floor %d", self.id, self.current_floor)
                self.door.start_opening()
                # Door opening is 1 tick, so tick it now to reach OPEN
                self.door.tick()

        elif self.direction is Direction.DOWN:
            self.current_floor -= 1
            log.debug("car %d: moved DOWN -> floor %d", self.id, self.current_floor)
            if self.current_floor in self._down_stops:
                self._down_stops.remove(self.current_floor)
                log.debug("car %d: serving down stop at floor %d", self.id, self.current_floor)
                self.door.start_opening()
                self.door.tick()

        # After moving, decide if we should keep going or switch/idle
        if self.door.is_closed():
            self._choose_direction()

    def _choose_direction(self) -> None:
        """Update direction based on pending stops.  Caller must hold self.lock."""
        if self.direction is Direction.UP:
            if self._up_stops:
                pass  # keep going UP
            elif self._down_stops:
                self.direction = Direction.DOWN
                log.debug("car %d: reversing to DOWN", self.id)
            else:
                self.direction = Direction.IDLE
                log.debug("car %d: no more stops, IDLE at floor %d", self.id, self.current_floor)
        elif self.direction is Direction.DOWN:
            if self._down_stops:
                pass  # keep going DOWN
            elif self._up_stops:
                self.direction = Direction.UP
                log.debug("car %d: reversing to UP", self.id)
            else:
                self.direction = Direction.IDLE
                log.debug("car %d: no more stops, IDLE at floor %d", self.id, self.current_floor)
        else:  # IDLE
            nxt = self._next_stop_unsafe()
            if nxt is not None:
                self.direction = Direction.UP if nxt >= self.current_floor else Direction.DOWN
                log.debug(
                    "car %d: leaving IDLE, heading %s toward floor %d",
                    self.id, self.direction.name, nxt,
                )

    def pending_stops(self) -> list[int]:
        """Return sorted list of all pending stops (for inspection / tests)."""
        with self.lock:
            return sorted(set(list(self._up_stops) + list(self._down_stops)))

    def is_idle(self) -> bool:
        with self.lock:
            return self.direction is Direction.IDLE and not self._up_stops and not self._down_stops


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class Scheduler:
    """
    Assigns hall calls to elevator cars.

    Algorithm: nearest-car with IDLE-car preference.

    Cost function:
      idle car   -> abs(car.current_floor - request_floor)
      moving car -> abs(car.current_floor - request_floor) + penalty

    The penalty is half the building height, making IDLE cars always
    preferred over moving cars at the same distance.
    """

    def __init__(self, cars: list[ElevatorCar], num_floors: int) -> None:
        self._cars: list[ElevatorCar] = cars
        self._penalty: int = max(1, num_floors // 2)

    def assign(self, floor: int, direction: Direction) -> ElevatorCar:  # noqa: ARG002
        """
        Assign hall call to the best car.

        direction is accepted for interface compatibility; full SCAN would
        use it to prefer a car already heading in the right direction.
        """
        best = min(self._cars, key=lambda c: self._cost(c, floor))
        log.debug(
            "scheduler: assigned floor=%d to car %d (floor %d, direction %s)",
            floor, best.id, best.current_floor, best.direction.name,
        )
        return best

    def _cost(self, car: ElevatorCar, target_floor: int) -> int:
        distance = abs(car.current_floor - target_floor)
        with car.lock:
            if car.direction is Direction.IDLE:
                return distance
        return distance + self._penalty


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


class Building:
    """
    Top-level facade: holds cars + scheduler, exposes hall_call / car_call / step_all.

    Thread-safe: concurrent hall_call() from multiple threads is safe
    because Scheduler.assign() is read-only on cars, and add_stop() on the
    chosen car acquires that car's lock.
    """

    def __init__(self, num_floors: int, num_cars: int) -> None:
        if num_floors < 2:
            raise ValueError("num_floors must be >= 2")
        if num_cars < 1:
            raise ValueError("num_cars must be >= 1")
        self.num_floors: int = num_floors
        self._cars: list[ElevatorCar] = [
            ElevatorCar(car_id=i, top_floor=num_floors - 1) for i in range(num_cars)
        ]
        self._scheduler: Scheduler = Scheduler(self._cars, num_floors)

    def hall_call(self, floor: int, direction: Direction) -> None:
        """Passenger on `floor` pressed the `direction` button."""
        if floor < 0 or floor >= self.num_floors:
            raise ValueError(f"Floor {floor} out of range")
        car = self._scheduler.assign(floor, direction)
        car.add_stop(floor)
        log.info("hall_call: floor=%d dir=%s assigned to car %d", floor, direction.name, car.id)

    def car_call(self, car_id: int, floor: int) -> None:
        """Passenger inside car `car_id` pressed floor button."""
        if car_id < 0 or car_id >= len(self._cars):
            raise ValueError(f"car_id {car_id} out of range")
        if floor < 0 or floor >= self.num_floors:
            raise ValueError(f"Floor {floor} out of range")
        self._cars[car_id].add_stop(floor)
        log.info("car_call: car=%d floor=%d", car_id, floor)

    def step_all(self) -> None:
        """Advance every car by one simulation tick."""
        for car in self._cars:
            car.step()

    def car(self, car_id: int) -> ElevatorCar:
        return self._cars[car_id]
```

---

## Problem A: Test Suite

```python
"""
tests/test_elevator.py — Day 6 Block A test suite.

Run with:  python -m pytest tests/test_elevator.py -v

Requires sortedcontainers:  pip install sortedcontainers
"""

import threading
import pytest

from elevator_system import (
    Building,
    Direction,
    Door,
    DoorState,
    ElevatorCar,
    Scheduler,
)


# ---------------------------------------------------------------------------
# Door tests
# ---------------------------------------------------------------------------


def test_door_initial_state_is_closed():
    d = Door()
    assert d.state is DoorState.CLOSED
    assert d.is_closed()


def test_door_opens_over_two_ticks():
    d = Door()
    d.start_opening()
    assert d.state is DoorState.OPENING
    d.tick()
    assert d.state is DoorState.OPEN
    assert d.is_open()


def test_door_closes_over_two_ticks():
    d = Door()
    d.start_opening()
    d.tick()                # -> OPEN
    d.start_closing()
    assert d.state is DoorState.CLOSING
    d.tick()
    assert d.state is DoorState.CLOSED


def test_door_obstruction_reverses_closing():
    d = Door()
    d.start_opening()
    d.tick()                # -> OPEN
    d.start_closing()       # -> CLOSING
    d.obstruct()            # -> OPENING
    assert d.state is DoorState.OPENING


def test_door_obstruct_has_no_effect_when_open():
    """Obstructing an already-OPEN door has no effect (it stays OPEN)."""
    d = Door()
    d.start_opening()
    d.tick()                # -> OPEN
    # obstruct() is only meaningful in CLOSING; when OPEN it is a no-op
    # The Door.obstruct() only acts on CLOSING, so state should remain OPEN
    d.obstruct()
    assert d.state is DoorState.OPEN


# ---------------------------------------------------------------------------
# ElevatorCar unit tests
# ---------------------------------------------------------------------------


def test_new_car_is_idle_at_floor_zero():
    car = ElevatorCar(car_id=0, top_floor=9)
    assert car.current_floor == 0
    assert car.direction is Direction.IDLE


def test_car_moves_up_after_stop_added_above():
    car = ElevatorCar(car_id=0, top_floor=9)
    car.add_stop(3)
    car.step()  # IDLE -> UP, move to floor 1
    assert car.current_floor == 1
    assert car.direction is Direction.UP


def test_car_opens_door_when_reaching_stop():
    car = ElevatorCar(car_id=0, top_floor=9)
    car.add_stop(1)
    car.step()  # move to floor 1, open door
    assert car.current_floor == 1
    assert car.door.is_open()


def test_car_continues_after_door_closes():
    car = ElevatorCar(car_id=0, top_floor=9)
    car.add_stop(1)
    car.add_stop(3)
    car.step()  # -> floor 1, door OPEN
    car.step()  # door CLOSING
    car.step()  # door CLOSED, resume
    car.step()  # -> floor 2
    assert car.current_floor == 2
    assert car.direction is Direction.UP


def test_car_becomes_idle_when_no_more_stops():
    car = ElevatorCar(car_id=0, top_floor=9)
    car.add_stop(1)
    car.step()  # -> floor 1, door OPEN
    car.step()  # door CLOSING
    car.step()  # door CLOSED, no more stops -> IDLE
    assert car.direction is Direction.IDLE
    assert car.is_idle()


def test_car_reverses_for_down_stop():
    car = ElevatorCar(car_id=0, top_floor=9)
    # Travel to floor 5
    car.add_stop(5)
    for _ in range(10):  # enough steps to arrive
        car.step()
        if car.current_floor == 5 and car.door.is_closed() and car.direction is Direction.IDLE:
            break

    assert car.current_floor == 5

    # Now request a floor below current
    car.add_stop(2)
    car.step()  # should start moving DOWN
    assert car.direction is Direction.DOWN


def test_car_services_multiple_stops_in_order():
    """Car travelling UP should service stops 2, 4, 6 in that order."""
    car = ElevatorCar(car_id=0, top_floor=9)
    car.add_stop(6)
    car.add_stop(2)
    car.add_stop(4)

    visited: list[int] = []

    for _ in range(50):
        prev_floor = car.current_floor
        prev_open = car.door.is_open()
        car.step()
        if car.door.is_open() and not prev_open:
            visited.append(car.current_floor)
        if car.is_idle():
            break

    assert visited == [2, 4, 6], f"Expected [2, 4, 6], got {visited}"


@pytest.mark.parametrize("floors,expected_count", [
    ([1, 3, 5], 3),
    ([5], 1),
    ([0, 9], 2),
])
def test_car_serves_all_stops(floors: list[int], expected_count: int):
    car = ElevatorCar(car_id=0, top_floor=9)
    for f in floors:
        car.add_stop(f)

    for _ in range(200):
        car.step()
        if car.is_idle():
            break

    assert car.is_idle()
    assert car.pending_stops() == []


# ---------------------------------------------------------------------------
# Scheduler unit tests
# ---------------------------------------------------------------------------


def test_scheduler_assigns_nearest_idle_car():
    car_a = ElevatorCar(car_id=0, top_floor=9)   # floor 0, IDLE
    car_b = ElevatorCar(car_id=1, top_floor=9)   # floor 0, IDLE
    car_b.current_floor = 5
    scheduler = Scheduler([car_a, car_b], num_floors=10)
    assigned = scheduler.assign(floor=4, direction=Direction.UP)
    assert assigned is car_b  # car_b at 5 is closer to 4 than car_a at 0


def test_scheduler_prefers_idle_over_moving():
    car_a = ElevatorCar(car_id=0, top_floor=9)   # floor 3, IDLE
    car_b = ElevatorCar(car_id=1, top_floor=9)   # floor 3, MOVING_UP
    car_a.current_floor = 3
    car_b.current_floor = 3
    car_b.direction = Direction.UP   # simulate moving
    # Both are distance 0 from floor 3; car_a (IDLE) should win
    scheduler = Scheduler([car_a, car_b], num_floors=10)
    assigned = scheduler.assign(floor=3, direction=Direction.UP)
    assert assigned is car_a


# ---------------------------------------------------------------------------
# Building integration tests
# ---------------------------------------------------------------------------


def test_full_passenger_journey_single_car():
    building = Building(num_floors=10, num_cars=1)
    building.hall_call(floor=5, direction=Direction.UP)

    for _ in range(30):
        building.step_all()
        if building.car(0).current_floor == 5:
            break

    assert building.car(0).current_floor == 5


def test_building_rejects_invalid_floor():
    building = Building(num_floors=5, num_cars=1)
    with pytest.raises(ValueError):
        building.hall_call(floor=10, direction=Direction.UP)


# ---------------------------------------------------------------------------
# Concurrency tests
# ---------------------------------------------------------------------------


def test_concurrent_hall_calls_no_exception():
    building = Building(num_floors=20, num_cars=3)
    errors: list[Exception] = []

    def issue_calls() -> None:
        try:
            for floor in range(10):
                building.hall_call(floor, Direction.UP)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=issue_calls) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Concurrent calls raised: {errors}"


def test_concurrent_calls_and_steps_no_exception():
    """Mix of hall_call and step_all from concurrent threads."""
    building = Building(num_floors=15, num_cars=2)
    errors: list[Exception] = []

    def caller() -> None:
        try:
            for i in range(5):
                building.hall_call(i % 15, Direction.UP)
        except Exception as exc:
            errors.append(exc)

    def stepper() -> None:
        try:
            for _ in range(20):
                building.step_all()
        except Exception as exc:
            errors.append(exc)

    threads = (
        [threading.Thread(target=caller) for _ in range(5)]
        + [threading.Thread(target=stepper) for _ in range(3)]
    )
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Errors in concurrent test: {errors}"
```

---

## Problem B: In-Memory KV Store — Full Solution

```python
"""
kv_store.py — Day 6 Block B full solution.

Thread-safe in-memory key-value store with:
  - Optional per-key TTL
  - Lazy expiry on get() / keys()
  - Background eviction daemon thread
  - Namespace view with prefix isolation
  - threading.RLock for safe re-entrant locking
  - LRU eviction using collections.OrderedDict

Python 3.12+.  No third-party dependencies.
"""

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Generic, TypeVar

log = logging.getLogger(__name__)

V = TypeVar("V")

# Sentinel that means "evict by LRU when over capacity"
_NO_CAPACITY = -1


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


@dataclass
class KVEntry(Generic[V]):
    """
    A stored value with an optional expiry timestamp.

    expires_at: monotonic timestamp (from time.monotonic()) or None for no expiry.
    """

    value: V
    expires_at: float | None = field(default=None)

    def is_expired(self) -> bool:
        """Return True if the entry has a TTL that has elapsed."""
        if self.expires_at is None:
            return False
        return time.monotonic() >= self.expires_at


# ---------------------------------------------------------------------------
# KVStore
# ---------------------------------------------------------------------------


class KVStore(Generic[V]):
    """
    Thread-safe in-memory key-value store with TTL and background eviction.

    Parameters
    ----------
    eviction_interval : float
        How often the background thread sweeps for expired keys, in seconds.
    max_size : int | None
        If set, LRU eviction triggers when the store reaches this size.
        Defaults to None (unlimited).

    Thread safety
    -------------
    All public methods acquire self._lock (an RLock).
    RLock is used (instead of Lock) because get() may call delete() during
    lazy expiry, and _evict_loop also calls _delete_expired() — both paths
    must be able to re-acquire the same lock on the same thread.

    LRU eviction
    ------------
    _data is a collections.OrderedDict; on each get() the accessed key is
    moved to the end (most-recently-used).  On capacity overflow, the item
    at the front (least-recently-used) is evicted.
    """

    def __init__(
        self,
        eviction_interval: float = 5.0,
        max_size: int | None = None,
    ) -> None:
        self._data: OrderedDict[str, KVEntry[V]] = OrderedDict()
        self._lock: threading.RLock = threading.RLock()
        self._stop_event: threading.Event = threading.Event()
        self._eviction_interval = eviction_interval
        self._max_size = max_size

        self._eviction_thread = threading.Thread(
            target=self._evict_loop,
            name="kv-eviction",
            daemon=True,
        )
        self._eviction_thread.start()

    # ---- public API ----

    def put(self, key: str, value: V, ttl_seconds: float | None = None) -> None:
        """
        Insert or update key.  Optional TTL in seconds.

        If max_size is set and the store is at capacity after insertion,
        the LRU entry is evicted.
        """
        expires_at: float | None = (
            time.monotonic() + ttl_seconds if ttl_seconds is not None else None
        )
        entry: KVEntry[V] = KVEntry(value=value, expires_at=expires_at)
        with self._lock:
            if key in self._data:
                del self._data[key]       # remove to re-insert at end (LRU update)
            self._data[key] = entry
            log.debug("put key=%r ttl=%s", key, ttl_seconds)
            self._enforce_capacity()

    def get(self, key: str) -> V | None:
        """
        Return value or None.

        Lazily evicts the entry if it has expired.
        Moves the entry to MRU position (end of OrderedDict) on hit.
        """
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            if entry.is_expired():
                del self._data[key]
                log.debug("lazy eviction: key=%r expired", key)
                return None
            # LRU: move to end
            self._data.move_to_end(key)
            return entry.value

    def delete(self, key: str) -> bool:
        """Remove key.  Return True if it existed (even if expired)."""
        with self._lock:
            if key in self._data:
                del self._data[key]
                log.debug("delete key=%r", key)
                return True
            return False

    def exists(self, key: str) -> bool:
        """Return True if key exists and has not expired."""
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return False
            if entry.is_expired():
                del self._data[key]
                return False
            return True

    def keys(self) -> list[str]:
        """Snapshot of all non-expired keys at call time."""
        with self._lock:
            expired = [k for k, e in self._data.items() if e.is_expired()]
            for k in expired:
                del self._data[k]
            return list(self._data.keys())

    def size(self) -> int:
        """Number of non-expired entries."""
        return len(self.keys())

    def stop(self) -> None:
        """Signal the background eviction thread to stop."""
        self._stop_event.set()

    def namespace(self, prefix: str) -> "Namespace[V]":
        """Return a namespaced view of this store."""
        return Namespace(store=self, prefix=prefix)

    # ---- internal ----

    def _evict_loop(self) -> None:
        """Daemon: periodically delete expired entries."""
        while not self._stop_event.wait(timeout=self._eviction_interval):
            with self._lock:
                n = self._delete_expired()
            if n:
                log.debug("background eviction: removed %d expired keys", n)

    def _delete_expired(self) -> int:
        """
        Delete all expired entries.  Caller MUST hold self._lock.

        Returns the count of evicted entries.
        """
        expired_keys = [k for k, e in self._data.items() if e.is_expired()]
        for k in expired_keys:
            del self._data[k]
        return len(expired_keys)

    def _enforce_capacity(self) -> None:
        """
        Evict the LRU entry if max_size is set and exceeded.

        Caller MUST hold self._lock.
        """
        if self._max_size is None or self._max_size <= 0:
            return
        while len(self._data) > self._max_size:
            lru_key, _ = self._data.popitem(last=False)
            log.debug("LRU eviction: key=%r", lru_key)


# ---------------------------------------------------------------------------
# Namespace
# ---------------------------------------------------------------------------


class Namespace(Generic[V]):
    """
    A namespaced view of a KVStore.

    All keys are stored as f"{prefix}:{key}" in the parent store.
    This class never copies data; every operation delegates to the parent.

    Usage
    -----
    ns = store.namespace("users")
    ns.put("alice", {...})    # stored as "users:alice"
    ns.get("alice")           # fetches "users:alice"
    ns.keys()                 # returns ["alice"] (prefix stripped)
    """

    def __init__(self, store: KVStore[V], prefix: str) -> None:
        if not prefix:
            raise ValueError("Namespace prefix must be non-empty")
        self._store: KVStore[V] = store
        self._prefix: str = prefix

    def _ns(self, key: str) -> str:
        return f"{self._prefix}:{key}"

    def put(self, key: str, value: V, ttl_seconds: float | None = None) -> None:
        self._store.put(self._ns(key), value, ttl_seconds=ttl_seconds)

    def get(self, key: str) -> V | None:
        return self._store.get(self._ns(key))

    def delete(self, key: str) -> bool:
        return self._store.delete(self._ns(key))

    def exists(self, key: str) -> bool:
        return self._store.exists(self._ns(key))

    def keys(self) -> list[str]:
        """Return all keys in this namespace, with the prefix stripped."""
        prefix_with_sep = f"{self._prefix}:"
        return [
            k[len(prefix_with_sep):]
            for k in self._store.keys()
            if k.startswith(prefix_with_sep)
        ]

    def size(self) -> int:
        return len(self.keys())
```

---

## Problem B: Test Suite

```python
"""
tests/test_kv_store.py — Day 6 Block B test suite.

Run with:  python -m pytest tests/test_kv_store.py -v

Covers:
  - KVEntry expiry logic
  - KVStore basic CRUD
  - TTL / lazy eviction
  - Background eviction daemon
  - Thread safety (concurrent readers + writers)
  - Namespace isolation
  - LRU eviction (max_size)
"""

import threading
import time

import pytest

from kv_store import KVEntry, KVStore, Namespace


# ---------------------------------------------------------------------------
# KVEntry tests
# ---------------------------------------------------------------------------


def test_entry_without_ttl_never_expires():
    e: KVEntry[int] = KVEntry(value=42)
    assert not e.is_expired()


def test_entry_with_future_expiry_not_expired():
    e: KVEntry[int] = KVEntry(value=42, expires_at=time.monotonic() + 100)
    assert not e.is_expired()


def test_entry_with_past_expiry_is_expired():
    e: KVEntry[int] = KVEntry(value=42, expires_at=time.monotonic() - 1)
    assert e.is_expired()


# ---------------------------------------------------------------------------
# KVStore basic CRUD tests
# ---------------------------------------------------------------------------


def test_put_and_get():
    store: KVStore[str] = KVStore()
    store.put("name", "alice")
    assert store.get("name") == "alice"
    store.stop()


def test_get_missing_returns_none():
    store: KVStore[str] = KVStore()
    assert store.get("missing") is None
    store.stop()


def test_delete_existing_key_returns_true():
    store: KVStore[int] = KVStore()
    store.put("x", 1)
    assert store.delete("x") is True
    assert store.get("x") is None
    store.stop()


def test_delete_missing_key_returns_false():
    store: KVStore[int] = KVStore()
    assert store.delete("nonexistent") is False
    store.stop()


def test_keys_returns_all_live_keys():
    store: KVStore[int] = KVStore()
    store.put("a", 1)
    store.put("b", 2)
    store.put("c", 3)
    assert sorted(store.keys()) == ["a", "b", "c"]
    store.stop()


def test_put_overwrites_existing_key():
    store: KVStore[int] = KVStore()
    store.put("n", 1)
    store.put("n", 2)
    assert store.get("n") == 2
    store.stop()


def test_exists_returns_true_for_live_key():
    store: KVStore[str] = KVStore()
    store.put("k", "v")
    assert store.exists("k") is True
    store.stop()


def test_exists_returns_false_for_missing_key():
    store: KVStore[str] = KVStore()
    assert store.exists("absent") is False
    store.stop()


def test_size_reflects_live_entries():
    store: KVStore[int] = KVStore()
    assert store.size() == 0
    store.put("a", 1)
    store.put("b", 2)
    assert store.size() == 2
    store.delete("a")
    assert store.size() == 1
    store.stop()


# ---------------------------------------------------------------------------
# TTL / lazy eviction tests
# ---------------------------------------------------------------------------


def test_get_returns_none_after_ttl_expires():
    store: KVStore[str] = KVStore()
    store.put("tmp", "value", ttl_seconds=0.05)
    assert store.get("tmp") == "value"
    time.sleep(0.1)
    assert store.get("tmp") is None
    store.stop()


def test_keys_excludes_expired_entries():
    store: KVStore[str] = KVStore()
    store.put("persistent", "keep")
    store.put("ephemeral", "gone", ttl_seconds=0.05)
    time.sleep(0.1)
    keys = store.keys()
    assert "ephemeral" not in keys
    assert "persistent" in keys
    store.stop()


def test_exists_returns_false_for_expired_key():
    store: KVStore[str] = KVStore()
    store.put("k", "v", ttl_seconds=0.05)
    time.sleep(0.1)
    assert store.exists("k") is False
    store.stop()


def test_background_eviction_clears_expired_keys():
    store: KVStore[str] = KVStore(eviction_interval=0.1)
    for i in range(5):
        store.put(f"k{i}", "v", ttl_seconds=0.05)
    # Wait for at least one eviction cycle after all keys have expired
    time.sleep(0.5)
    assert store.size() == 0, f"Expected 0, got {store.size()}"
    store.stop()


# ---------------------------------------------------------------------------
# Thread safety tests
# ---------------------------------------------------------------------------


def test_concurrent_puts_no_exception():
    store: KVStore[int] = KVStore()
    errors: list[Exception] = []

    def writer(tid: int) -> None:
        try:
            for i in range(200):
                store.put(f"key-{tid}-{i}", i)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(5)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert errors == []
    store.stop()


def test_concurrent_reads_and_writes_consistent():
    store: KVStore[int] = KVStore()
    errors: list[str] = []

    def writer() -> None:
        for i in range(100):
            store.put(f"shared-{i % 10}", i)

    def reader() -> None:
        for i in range(100):
            val = store.get(f"shared-{i % 10}")
            if val is not None and not isinstance(val, int):
                errors.append(f"unexpected type: {type(val)}")

    threads = (
        [threading.Thread(target=writer) for _ in range(3)]
        + [threading.Thread(target=reader) for _ in range(3)]
    )
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert errors == []
    store.stop()


def test_concurrent_ttl_expiry_no_exception():
    """Multiple threads getting an expiring key concurrently should not raise."""
    store: KVStore[str] = KVStore()
    store.put("expiring", "v", ttl_seconds=0.05)
    errors: list[Exception] = []

    def reader() -> None:
        try:
            for _ in range(50):
                store.get("expiring")
                time.sleep(0.001)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(10)]
    time.sleep(0.06)  # let key expire before threads start racing
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert errors == []
    store.stop()


# ---------------------------------------------------------------------------
# Namespace tests
# ---------------------------------------------------------------------------


def test_namespace_put_and_get():
    store: KVStore[str] = KVStore()
    ns = store.namespace("users")
    ns.put("alice", "data")
    assert ns.get("alice") == "data"
    # Verify stored with prefix in parent store
    assert store.get("users:alice") == "data"
    store.stop()


def test_namespace_keys_strips_prefix():
    store: KVStore[str] = KVStore()
    ns = store.namespace("cfg")
    ns.put("timeout", "30")
    ns.put("retries", "3")
    assert sorted(ns.keys()) == ["retries", "timeout"]
    store.stop()


def test_namespace_does_not_leak_across_namespaces():
    store: KVStore[str] = KVStore()
    ns_a = store.namespace("a")
    ns_b = store.namespace("b")
    ns_a.put("x", "alpha")
    ns_b.put("x", "beta")
    assert ns_a.get("x") == "alpha"
    assert ns_b.get("x") == "beta"
    store.stop()


def test_namespace_delete():
    store: KVStore[str] = KVStore()
    ns = store.namespace("session")
    ns.put("token", "abc")
    assert ns.delete("token") is True
    assert ns.get("token") is None
    store.stop()


def test_namespace_keys_excludes_other_namespaces():
    store: KVStore[int] = KVStore()
    ns_users = store.namespace("users")
    ns_orders = store.namespace("orders")
    ns_users.put("alice", 1)
    ns_orders.put("o1", 100)
    assert ns_users.keys() == ["alice"]
    assert ns_orders.keys() == ["o1"]
    store.stop()


def test_namespace_ttl_passthrough():
    store: KVStore[str] = KVStore()
    ns = store.namespace("temp")
    ns.put("k", "v", ttl_seconds=0.05)
    assert ns.get("k") == "v"
    time.sleep(0.1)
    assert ns.get("k") is None
    store.stop()


# ---------------------------------------------------------------------------
# LRU eviction tests (max_size)
# ---------------------------------------------------------------------------


def test_lru_eviction_on_overflow():
    store: KVStore[int] = KVStore(max_size=3)
    store.put("a", 1)
    store.put("b", 2)
    store.put("c", 3)
    # Access "a" to make it MRU; LRU is now "b"
    store.get("a")
    # Adding "d" should evict "b" (LRU)
    store.put("d", 4)
    assert store.get("b") is None, "b should have been evicted"
    assert store.get("a") == 1
    assert store.get("c") == 3
    assert store.get("d") == 4
    store.stop()


def test_lru_store_never_exceeds_max_size():
    store: KVStore[int] = KVStore(max_size=5)
    for i in range(20):
        store.put(f"k{i}", i)
    assert store.size() <= 5
    store.stop()


@pytest.mark.parametrize("cap,inserts,expected_size", [
    (2, 5, 2),
    (10, 3, 3),
    (1, 4, 1),
])
def test_lru_parametrized(cap: int, inserts: int, expected_size: int):
    store: KVStore[int] = KVStore(max_size=cap)
    for i in range(inserts):
        store.put(f"k{i}", i)
    assert store.size() == expected_size
    store.stop()
```

---

## Wrap-up answers

**Q1 — What breaks if two threads call `car.step()` without the lock?**

`_up_stops.remove(floor)` and `current_floor += 1` are each multiple bytecode operations.
Two threads could both observe the same floor in `_up_stops`, both remove it, and both open
the door — one thread's remove would raise `ValueError` (item not in SortedList).
Also, `current_floor` could over-increment past the stop, causing the stop to be missed.

**Q2 — Upgrading from nearest-car to SCAN:**

SCAN needs to know:
- `car.current_floor`
- `car.direction`
- Whether the car is already heading toward the request floor in the same direction

Assign a car to a hall call at `(floor, direction)` only if:
- The car is IDLE, OR
- The car is moving TOWARD `floor` AND its direction matches the request direction.

If no car satisfies that, fall back to nearest-car.

**Q3 — Maximum TTL overshoot:**

A key expires at time T.  The eviction loop fires every N seconds.
Maximum survival past TTL = N seconds (if it expires just after a sweep).
Reduce it by lowering `eviction_interval` or by using lazy eviction alone (no background).
In practice, lazy eviction (on `get`) covers most cases; background is for keys never read.

**Q4 — LRU eviction data structure:**

Replace `dict[str, KVEntry]` with `collections.OrderedDict[str, KVEntry]`.
On each `get()` hit, call `move_to_end(key)`.
When capacity is exceeded after `put()`, call `popitem(last=False)` to evict the LRU item.
(This is exactly what the solution above does.)

**Q5 — Scaling to 100 concurrent threads:**

- **ElevatorCar:** Per-car locks are already fine; no global contention.
- **KVStore:** Single `RLock` becomes a bottleneck.  Partition the store into N shards,
  each with its own lock.  Key is assigned to shard `hash(key) % N`.
  Alternatively, switch to lock-free structures or `concurrent.futures` for read-heavy loads.
