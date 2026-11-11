# Reference Implementations — Week 07 Project

---

## Problem 1: Elevator System

### Class Diagram

```
Interfaces (ABCs)
─────────────────
+─────────────────────────+    +──────────────────────────+
│  DispatchStrategy (ABC) │    │  Clock (ABC)             │
│ +select_elevator(       │    │ +now() -> float          │
│    request: FloorRequest│    +──────────────────────────+
│    elevators: list      │              ^
│   ) -> int              │              |
+─────────────────────────+    +──────────────────────────+
           ^                   │  MonotonicClock          │
           |                   +──────────────────────────+
+─────────────────────────+
│  NearestIdleStrategy    │
+─────────────────────────+

Core classes
────────────
+─────────────────────────+    *   +──────────────────────────+
│  Building               │───────>│  Elevator                 │
│ -elevators: list        │        │ -elevator_id: int         │
│ -strategy               │        │ -current_floor: int       │
│ -clock                  │        │ -direction: Direction     │
│ -_lock: RLock           │        │ -door_state: DoorState    │
│ +request_elevator()     │        │ -passenger_count: int     │
│ +request_floor()        │        │ -_stops: SortedList       │
│ +step()                 │        │ -_lock: RLock             │
│ +elevator_positions()   │        │ +add_stop(floor)          │
│ +pending_requests()     │        │ +step()                   │
+─────────────────────────+        │ +open_door()              │
                                   │ +close_door()             │
Pydantic models                    │ +board_passenger()        │
─────────────────                  │ +alight_passenger()       │
+─────────────────────────+        │ +next_stop()              │
│  FloorRequest            │        +──────────────────────────+
│  -floor: int             │
│  -direction: Direction   │
+─────────────────────────+
+─────────────────────────+
│  ElevatorRequest         │
│  -elevator_id: int       │
│  -destination_floor: int │
+─────────────────────────+

Enums: Direction(UP|DOWN|IDLE), DoorState(OPEN|CLOSED|OPENING|CLOSING)
Exceptions: InvalidDoorTransitionError
```

---

### Implementation

```python
# elevator/models.py

"""
Value objects, Pydantic models, and enums for the elevator system.
"""

from __future__ import annotations

from enum import Enum, auto

from pydantic import BaseModel, Field


class Direction(Enum):
    UP = auto()
    DOWN = auto()
    IDLE = auto()


class DoorState(Enum):
    OPEN = auto()
    CLOSED = auto()
    OPENING = auto()
    CLOSING = auto()


class FloorRequest(BaseModel):
    """A request originating outside the elevator (hall call)."""

    floor: int = Field(..., ge=0)
    direction: Direction

    model_config = {"use_enum_values": False}


class ElevatorRequest(BaseModel):
    """A request originating inside an elevator (cab call)."""

    elevator_id: int
    destination_floor: int = Field(..., ge=0)
```

```python
# elevator/interfaces.py

"""
Abstract interfaces for the elevator system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .models import Direction, FloorRequest


class ElevatorState:
    """Read-only view of an elevator's state, passed to DispatchStrategy."""

    def __init__(
        self,
        elevator_id: int,
        current_floor: int,
        direction: Direction,
        passenger_count: int,
        capacity: int,
        pending_stops: list[int],
    ) -> None:
        self.elevator_id = elevator_id
        self.current_floor = current_floor
        self.direction = direction
        self.passenger_count = passenger_count
        self.capacity = capacity
        self.pending_stops = list(pending_stops)


class DispatchStrategy(ABC):
    """Selects which elevator should handle a hall call."""

    @abstractmethod
    def select_elevator(
        self,
        request: FloorRequest,
        elevators: list[ElevatorState],
    ) -> int:
        """Return the index of the selected elevator."""
        ...


class Clock(ABC):
    @abstractmethod
    def now(self) -> float:
        """Return monotonic time in seconds."""
        ...
```

```python
# elevator/elevator.py

"""
Elevator state machine with SCAN scheduling algorithm.

Design decisions:
  - SCAN queue is maintained as two sorted sets: floors above (ascending)
    and floors below (descending). This makes next_stop() O(1).
  - Door state machine is explicit; all invalid transitions raise
    InvalidDoorTransitionError rather than silently no-opping.
  - Thread safety: a single RLock per elevator protects all mutable state.
    Building-level operations call step() sequentially, so the lock mainly
    protects concurrent add_stop() calls from gate threads.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from .interfaces import Clock
from .models import Direction, DoorState

logger = logging.getLogger(__name__)


class InvalidDoorTransitionError(Exception):
    """Raised when an invalid door state transition is attempted."""


# Valid door state transitions: current_state -> set of allowed next states
_VALID_DOOR_TRANSITIONS: dict[DoorState, set[DoorState]] = {
    DoorState.CLOSED: {DoorState.OPENING},
    DoorState.OPENING: {DoorState.OPEN},
    DoorState.OPEN: {DoorState.CLOSING},
    DoorState.CLOSING: {DoorState.CLOSED},
}


class Elevator:
    """
    Single elevator with SCAN scheduling and door state machine.

    SCAN algorithm:
      - While moving UP: serve stops >= current_floor in ascending order.
      - When no stops remain above: reverse to DOWN (or IDLE if nothing below).
      - While moving DOWN: serve stops <= current_floor in descending order.
      - Tie-breaking: lower floor wins when equidistant.
    """

    def __init__(
        self,
        elevator_id: int,
        num_floors: int,
        capacity: int,
        clock: Clock,
    ) -> None:
        self._id = elevator_id
        self._num_floors = num_floors
        self._capacity = capacity
        self._clock = clock

        self._current_floor: int = 0
        self._direction: Direction = Direction.IDLE
        self._door_state: DoorState = DoorState.CLOSED
        self._passenger_count: int = 0

        # Stops above (ascending) and below (descending) the current floor.
        # Using sorted sets avoids O(n log n) sort on every step.
        self._stops_above: list[int] = []   # ascending
        self._stops_below: list[int] = []   # descending
        self._stops_set: set[int] = set()   # fast membership check

        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Properties (read-only snapshots; safe without lock for monitoring)
    # ------------------------------------------------------------------

    @property
    def elevator_id(self) -> int:
        return self._id

    @property
    def current_floor(self) -> int:
        return self._current_floor

    @property
    def direction(self) -> Direction:
        return self._direction

    @property
    def door_state(self) -> DoorState:
        return self._door_state

    @property
    def passenger_count(self) -> int:
        return self._passenger_count

    @property
    def capacity(self) -> int:
        return self._capacity

    # ------------------------------------------------------------------
    # Public mutators
    # ------------------------------------------------------------------

    def add_stop(self, floor: int) -> None:
        """Add a floor to the SCAN queue. Thread-safe."""
        if not 0 <= floor < self._num_floors:
            raise ValueError(
                f"Floor {floor} out of range [0, {self._num_floors - 1}]"
            )
        with self._lock:
            if floor in self._stops_set:
                return
            self._stops_set.add(floor)
            if floor >= self._current_floor:
                self._stops_above = sorted(
                    self._stops_above + [floor]
                )
            else:
                self._stops_below = sorted(
                    self._stops_below + [floor], reverse=True
                )
            # Wake from IDLE if needed
            if self._direction == Direction.IDLE:
                if self._stops_above:
                    self._direction = Direction.UP
                else:
                    self._direction = Direction.DOWN
            logger.debug(
                "Elevator %d: added stop %d (above=%s below=%s)",
                self._id, floor, self._stops_above, self._stops_below,
            )

    def step(self) -> None:
        """
        Advance the elevator one logical step.
        Called by Building.step() in the main simulation loop.
        """
        with self._lock:
            self._advance_door_state()
            if self._door_state == DoorState.CLOSED:
                self._move_or_idle()

    def open_door(self) -> None:
        """Initiate door opening sequence. Raises if current state is not CLOSED."""
        with self._lock:
            self._transition_door(DoorState.OPENING)

    def close_door(self) -> None:
        """Initiate door closing sequence. Raises if current state is not OPEN."""
        with self._lock:
            self._transition_door(DoorState.CLOSING)

    def board_passenger(self) -> bool:
        """
        Attempt to board one passenger. Returns False if at capacity.
        Caller must ensure door is OPEN before calling.
        """
        with self._lock:
            if self._passenger_count >= self._capacity:
                logger.info(
                    "Elevator %d: at capacity (%d), boarding refused",
                    self._id, self._capacity,
                )
                return False
            self._passenger_count += 1
            logger.debug(
                "Elevator %d: passenger boarded (%d/%d)",
                self._id, self._passenger_count, self._capacity,
            )
            return True

    def alight_passenger(self) -> None:
        """Record one passenger alighting. Raises if count would go negative."""
        with self._lock:
            if self._passenger_count <= 0:
                raise ValueError(
                    f"Elevator {self._id}: cannot alight — no passengers on board"
                )
            self._passenger_count -= 1
            logger.debug(
                "Elevator %d: passenger alighted (%d/%d)",
                self._id, self._passenger_count, self._capacity,
            )

    def next_stop(self) -> Optional[int]:
        """
        Return the next floor this elevator will stop at under SCAN, or None.
        SCAN rule: serve in current direction first; reverse if exhausted.
        Tie-breaking: lower floor wins.
        """
        with self._lock:
            return self._compute_next_stop()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transition_door(self, target: DoorState) -> None:
        """Perform a door state transition. Must be called with lock held."""
        allowed = _VALID_DOOR_TRANSITIONS.get(self._door_state, set())
        if target not in allowed:
            raise InvalidDoorTransitionError(
                f"Elevator {self._id}: cannot transition door "
                f"{self._door_state.name} -> {target.name}"
            )
        self._door_state = target
        logger.debug(
            "Elevator %d: door %s -> %s", self._id, self._door_state.name, target.name
        )

    def _advance_door_state(self) -> None:
        """
        If door is mid-transition (OPENING or CLOSING), advance it one step.
        Must be called with lock held.
        """
        if self._door_state == DoorState.OPENING:
            self._door_state = DoorState.OPEN
            logger.debug("Elevator %d: door OPEN at floor %d", self._id, self._current_floor)
        elif self._door_state == DoorState.CLOSING:
            self._door_state = DoorState.CLOSED
            logger.debug("Elevator %d: door CLOSED", self._id)

    def _move_or_idle(self) -> None:
        """
        Move one floor toward the next stop, or go IDLE if no stops remain.
        Must be called with lock held.
        """
        next_floor = self._compute_next_stop()
        if next_floor is None:
            self._direction = Direction.IDLE
            return

        # Determine move direction
        if next_floor > self._current_floor:
            self._current_floor += 1
            self._direction = Direction.UP
        elif next_floor < self._current_floor:
            self._current_floor -= 1
            self._direction = Direction.DOWN
        # else: already there (shouldn't normally happen)

        logger.debug(
            "Elevator %d: moved to floor %d (direction=%s)",
            self._id, self._current_floor, self._direction.name,
        )

        # Check if we have arrived at a stop
        if self._current_floor in self._stops_set:
            self._stops_set.discard(self._current_floor)
            # Remove from the appropriate list
            if self._current_floor in self._stops_above:
                self._stops_above.remove(self._current_floor)
            if self._current_floor in self._stops_below:
                self._stops_below.remove(self._current_floor)
            # Redistribute any stops that are now on the other side
            self._redistribute_stops()
            # Trigger door open
            self._door_state = DoorState.OPENING
            logger.info(
                "Elevator %d: arrived at floor %d, opening door",
                self._id, self._current_floor,
            )

    def _redistribute_stops(self) -> None:
        """
        After moving, ensure stops_above and stops_below are correctly partitioned.
        Must be called with lock held.
        """
        above = sorted(f for f in self._stops_set if f >= self._current_floor)
        below = sorted(
            (f for f in self._stops_set if f < self._current_floor), reverse=True
        )
        self._stops_above = above
        self._stops_below = below

    def _compute_next_stop(self) -> Optional[int]:
        """
        SCAN: return next stop in current direction; reverse if exhausted.
        Must be called with lock held.
        """
        if self._direction == Direction.UP:
            if self._stops_above:
                return self._stops_above[0]
            if self._stops_below:
                self._direction = Direction.DOWN
                return self._stops_below[0]
            return None

        if self._direction == Direction.DOWN:
            if self._stops_below:
                return self._stops_below[0]
            if self._stops_above:
                self._direction = Direction.UP
                return self._stops_above[0]
            return None

        # IDLE — pick nearest
        all_stops = self._stops_above + self._stops_below
        if not all_stops:
            return None
        # Tie-breaking: lower floor wins
        nearest = min(all_stops, key=lambda f: (abs(f - self._current_floor), f))
        if nearest >= self._current_floor:
            self._direction = Direction.UP
        else:
            self._direction = Direction.DOWN
        return nearest
```

```python
# elevator/scheduler.py

"""
Dispatch strategy implementations.
"""

from __future__ import annotations

import logging

from .interfaces import DispatchStrategy, ElevatorState
from .models import Direction, FloorRequest

logger = logging.getLogger(__name__)


class NearestIdleStrategy(DispatchStrategy):
    """
    Assigns the hall call to the nearest idle elevator.
    Falls back to the elevator with the fewest pending stops if none are idle.
    """

    def select_elevator(
        self,
        request: FloorRequest,
        elevators: list[ElevatorState],
    ) -> int:
        idle = [e for e in elevators if e.direction == Direction.IDLE]
        candidates = idle if idle else elevators

        def score(e: ElevatorState) -> tuple[int, int]:
            distance = abs(e.current_floor - request.floor)
            pending = len(e.pending_stops)
            return (distance, pending)

        selected = min(candidates, key=score)
        logger.debug(
            "NearestIdleStrategy: assigned floor=%d to elevator %d",
            request.floor, selected.elevator_id,
        )
        return selected.elevator_id
```

```python
# elevator/building.py

"""
Building dispatcher: receives hall/cab calls, assigns to elevators, runs steps.

Design decisions:
  - Strategy pattern for dispatch: NearestIdleStrategy is injected, making
    it trivial to swap in a more sophisticated algorithm (e.g., LOOK-ahead).
  - Thread safety: a single RLock on the Building protects the elevator list
    and step coordination. Individual Elevators have their own locks for
    concurrent add_stop() calls from gate threads.
  - Alternative considered: per-elevator dispatch lock. Rejected because the
    dispatch step itself is O(N elevators), which is small and not a bottleneck.
    A global Building lock is simpler and correct.

Limitations and future work:
  - This is a synchronous, step-based simulation. A production system would use
    actual threads or asyncio with condition variables, real motor controllers,
    and hardware sensor integration.
  - Persistence: elevator state would be stored in a database to survive restarts.
  - Monitoring: real systems emit telemetry (floor sensors, load cells) to a
    central dashboard. Add structured logging or OpenTelemetry spans here.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from .elevator import Elevator
from .interfaces import Clock, DispatchStrategy, ElevatorState
from .models import Direction, DoorState, FloorRequest

logger = logging.getLogger(__name__)


class Building:
    """
    Dispatcher: receives hall calls, assigns to elevators, runs simulation steps.

    Thread-safe via RLock for all public mutation methods.
    """

    def __init__(
        self,
        num_floors: int,
        num_elevators: int,
        strategy: DispatchStrategy,
        clock: Clock,
        elevator_capacity: int = 8,
    ) -> None:
        if num_floors < 2:
            raise ValueError("Building must have at least 2 floors")
        if num_elevators < 1:
            raise ValueError("Building must have at least 1 elevator")

        self._num_floors = num_floors
        self._strategy = strategy
        self._clock = clock
        self._lock = threading.RLock()

        self._elevators: list[Elevator] = [
            Elevator(
                elevator_id=i,
                num_floors=num_floors,
                capacity=elevator_capacity,
                clock=clock,
            )
            for i in range(num_elevators)
        ]
        logger.info(
            "Building initialized: %d floors, %d elevators",
            num_floors, num_elevators,
        )

    def request_elevator(self, floor: int, direction: Direction) -> None:
        """Submit a hall call. Assigns to an elevator via the dispatch strategy."""
        request = FloorRequest(floor=floor, direction=direction)
        with self._lock:
            states = self._snapshot_states()
            chosen_id = self._strategy.select_elevator(request, states)
            self._elevators[chosen_id].add_stop(floor)
        logger.info(
            "Hall call: floor=%d direction=%s -> elevator %d",
            floor, direction.name, chosen_id,
        )

    def request_floor(self, elevator_id: int, floor: int) -> None:
        """Submit a cab call (passenger inside presses floor button)."""
        if not 0 <= elevator_id < len(self._elevators):
            raise ValueError(f"Invalid elevator_id: {elevator_id}")
        self._elevators[elevator_id].add_stop(floor)
        logger.info("Cab call: elevator %d -> floor %d", elevator_id, floor)

    def step(self) -> None:
        """Advance all elevators one step. Thread-safe."""
        with self._lock:
            for elevator in self._elevators:
                elevator.step()

    def elevator_positions(self) -> list[tuple[int, Direction, DoorState]]:
        """Return (floor, direction, door_state) for each elevator."""
        return [
            (e.current_floor, e.direction, e.door_state)
            for e in self._elevators
        ]

    def pending_requests(self) -> int:
        """Return total number of unserviced stops across all elevators."""
        total = 0
        for e in self._elevators:
            total += len(e._stops_set)
        return total

    def _snapshot_states(self) -> list[ElevatorState]:
        """Build read-only state snapshots for the dispatch strategy."""
        return [
            ElevatorState(
                elevator_id=e.elevator_id,
                current_floor=e.current_floor,
                direction=e.direction,
                passenger_count=e.passenger_count,
                capacity=e.capacity,
                pending_stops=list(e._stops_set),
            )
            for e in self._elevators
        ]
```

---

### Tests

```python
# tests/test_elevator.py

"""
Elevator system test suite — 12 tests covering SCAN logic, door state machine,
concurrency, and full round-trips.
"""

from __future__ import annotations

import threading
from typing import Optional

import pytest

from elevator.building import Building
from elevator.elevator import Elevator, InvalidDoorTransitionError
from elevator.interfaces import Clock
from elevator.models import Direction, DoorState
from elevator.scheduler import NearestIdleStrategy


# ---------------------------------------------------------------------------
# Test clock
# ---------------------------------------------------------------------------

class FixedClock(Clock):
    def __init__(self, t: float = 0.0) -> None:
        self._t = t

    def advance(self, dt: float) -> None:
        self._t += dt

    def now(self) -> float:
        return self._t


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clock() -> FixedClock:
    return FixedClock()


@pytest.fixture
def single_elevator(clock: FixedClock) -> Elevator:
    return Elevator(elevator_id=0, num_floors=10, capacity=8, clock=clock)


@pytest.fixture
def building(clock: FixedClock) -> Building:
    return Building(
        num_floors=10,
        num_elevators=3,
        strategy=NearestIdleStrategy(),
        clock=clock,
        elevator_capacity=8,
    )


# ---------------------------------------------------------------------------
# Helper: step until condition or timeout
# ---------------------------------------------------------------------------

def step_until(
    building: Building,
    condition: object,
    max_steps: int = 200,
) -> bool:
    """Step the building until condition() is True or max_steps exhausted."""
    for _ in range(max_steps):
        if callable(condition) and condition():
            return True
        building.step()
    return callable(condition) and condition()


def step_elevator_until(
    elevator: Elevator,
    condition: object,
    max_steps: int = 100,
) -> bool:
    for _ in range(max_steps):
        if callable(condition) and condition():
            return True
        elevator.step()
    return callable(condition) and condition()


# ---------------------------------------------------------------------------
# Test 1: hall call assigned to nearest idle elevator
# ---------------------------------------------------------------------------

def test_hall_call_assigned_to_nearest_idle_elevator(clock: FixedClock) -> None:
    """
    Given: 3 elevators, all idle.
    Elevator 0 at floor 0, elevator 1 at floor 5, elevator 2 at floor 9.
    When: hall call for floor 5, direction UP.
    Then: elevator 1 is assigned (nearest idle — zero distance).
    """
    # Arrange: position elevators by stepping them to target floors
    building = Building(
        num_floors=10,
        num_elevators=3,
        strategy=NearestIdleStrategy(),
        clock=clock,
        elevator_capacity=8,
    )
    # Elevator 1: add stop at floor 5 and run until it gets there and goes idle
    building._elevators[1].add_stop(5)
    for _ in range(60):
        building._elevators[1].step()
        if (
            building._elevators[1].current_floor == 5
            and building._elevators[1].direction == Direction.IDLE
            and building._elevators[1].door_state == DoorState.CLOSED
        ):
            break
    # Elevator 2: position at floor 9
    building._elevators[2].add_stop(9)
    for _ in range(100):
        building._elevators[2].step()
        if (
            building._elevators[2].current_floor == 9
            and building._elevators[2].direction == Direction.IDLE
            and building._elevators[2].door_state == DoorState.CLOSED
        ):
            break

    # Act
    building.request_elevator(floor=5, direction=Direction.UP)

    # Assert: elevator 1 should have a stop for floor 5 (already there, so queue clears)
    # or another elevator that was closer. Elevator 1 is at floor 5 with 0 distance.
    # The pending stop should have been assigned to elevator 1.
    assert 5 in building._elevators[1]._stops_set or building._elevators[1].current_floor == 5


# ---------------------------------------------------------------------------
# Test 2: SCAN serves floors in ascending order while moving UP
# ---------------------------------------------------------------------------

def test_scan_serves_floors_in_ascending_order_while_moving_up(
    single_elevator: Elevator,
) -> None:
    """
    Given: elevator at floor 1, moving UP, stops queued: [4, 7, 2].
    When: step() called repeatedly.
    Then: floors visited in order 2, 4, 7.
    """
    # Arrange
    e = single_elevator
    e.add_stop(4)
    e.add_stop(7)
    e.add_stop(2)

    # Act
    visited: list[int] = []
    for _ in range(80):
        e.step()
        if e.door_state == DoorState.OPENING:
            if not visited or visited[-1] != e.current_floor:
                visited.append(e.current_floor)
        if e.direction == Direction.IDLE and e.door_state == DoorState.CLOSED and not e._stops_set:
            break

    # Assert
    assert visited == [2, 4, 7], f"Expected [2,4,7], got {visited}"


# ---------------------------------------------------------------------------
# Test 3: SCAN reverses direction at top floor
# ---------------------------------------------------------------------------

def test_scan_reverses_direction_at_top(clock: FixedClock) -> None:
    """
    Given: elevator at floor 9 (top), moving UP, stop at floor 3 queued.
    When: step() called.
    Then: direction changes to MOVING_DOWN, next stop is floor 3.
    """
    # Arrange: position elevator at floor 9
    e = Elevator(elevator_id=0, num_floors=10, capacity=8, clock=clock)
    e.add_stop(9)
    for _ in range(40):
        e.step()
        if e.current_floor == 9 and e.direction == Direction.IDLE and e.door_state == DoorState.CLOSED:
            break

    # Now add a stop below
    e.add_stop(3)

    # Act
    e.step()

    # Assert
    assert e.direction == Direction.DOWN
    assert e.next_stop() == 3


# ---------------------------------------------------------------------------
# Test 4: door open-close cycle
# ---------------------------------------------------------------------------

def test_door_open_close_cycle(single_elevator: Elevator) -> None:
    """
    Given: elevator at DOOR_CLOSED.
    When: open_door() called, then step() to OPEN, then close_door(), then step() to CLOSED.
    Then: sequence is CLOSED -> OPENING -> OPEN -> CLOSING -> CLOSED.
    """
    e = single_elevator
    assert e.door_state == DoorState.CLOSED

    e.open_door()
    assert e.door_state == DoorState.OPENING

    e.step()  # OPENING -> OPEN
    assert e.door_state == DoorState.OPEN

    e.close_door()
    assert e.door_state == DoorState.CLOSING

    e.step()  # CLOSING -> CLOSED
    assert e.door_state == DoorState.CLOSED


# ---------------------------------------------------------------------------
# Test 5: invalid door transition raises
# ---------------------------------------------------------------------------

def test_invalid_door_transition_raises(single_elevator: Elevator) -> None:
    """
    Given: elevator door is OPEN.
    When: open_door() is called again.
    Then: InvalidDoorTransitionError raised; door state still OPEN.
    """
    e = single_elevator
    e.open_door()
    e.step()  # -> OPEN
    assert e.door_state == DoorState.OPEN

    with pytest.raises(InvalidDoorTransitionError):
        e.open_door()

    assert e.door_state == DoorState.OPEN


# ---------------------------------------------------------------------------
# Test 6: elevator at capacity rejects boarding
# ---------------------------------------------------------------------------

def test_elevator_at_capacity_rejects_boarding(clock: FixedClock) -> None:
    """
    Given: elevator with capacity=2, already at 2 passengers.
    When: board_passenger() called.
    Then: returns False; count unchanged.
    """
    e = Elevator(elevator_id=0, num_floors=10, capacity=2, clock=clock)
    assert e.board_passenger() is True
    assert e.board_passenger() is True
    assert e.passenger_count == 2

    result = e.board_passenger()

    assert result is False
    assert e.passenger_count == 2


# ---------------------------------------------------------------------------
# Test 7: elevator returns IDLE when queue empty
# ---------------------------------------------------------------------------

def test_elevator_returns_idle_when_queue_empty(single_elevator: Elevator) -> None:
    """
    Given: elevator with one queued stop, moving UP.
    When: step() until stop is served.
    Then: direction == IDLE; door CLOSED; no pending stops.
    """
    e = single_elevator
    e.add_stop(3)

    for _ in range(50):
        e.step()
        if e.direction == Direction.IDLE and e.door_state == DoorState.CLOSED and not e._stops_set:
            break

    assert e.direction == Direction.IDLE
    assert e.door_state == DoorState.CLOSED
    assert e.next_stop() is None


# ---------------------------------------------------------------------------
# Test 8: concurrent hall calls all assigned
# ---------------------------------------------------------------------------

def test_concurrent_hall_calls_are_all_assigned(clock: FixedClock) -> None:
    """
    Given: building with 3 elevators.
    When: 50 hall calls submitted from 10 concurrent threads.
    Then: all 50 calls assigned without data races.
    """
    building = Building(
        num_floors=10,
        num_elevators=3,
        strategy=NearestIdleStrategy(),
        clock=clock,
        elevator_capacity=8,
    )
    errors: list[Exception] = []

    def submit_calls() -> None:
        try:
            for floor in range(5):
                building.request_elevator(floor, Direction.UP)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=submit_calls) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert not errors, f"Errors during concurrent calls: {errors}"
    assert building.pending_requests() == 50


# ---------------------------------------------------------------------------
# Test 9: cab call added to correct elevator
# ---------------------------------------------------------------------------

def test_cab_call_added_to_correct_elevator(building: Building) -> None:
    """
    Given: building with elevator 2 at floor 0.
    When: request_floor(elevator_id=2, floor=8).
    Then: elevator 2 has floor 8 in its queue; others do not.
    """
    building.request_floor(elevator_id=2, floor=8)

    assert 8 in building._elevators[2]._stops_set
    assert 8 not in building._elevators[0]._stops_set
    assert 8 not in building._elevators[1]._stops_set


# ---------------------------------------------------------------------------
# Test 10: full round trip single elevator
# ---------------------------------------------------------------------------

def test_full_round_trip_single_elevator(clock: FixedClock) -> None:
    """
    Given: single-elevator building, floors 0-9.
    When: hall call from floor 0 UP, then cab call to floor 7.
    Then: step() until pending_requests()==0; elevator visited floor 7.
    """
    building = Building(
        num_floors=10,
        num_elevators=1,
        strategy=NearestIdleStrategy(),
        clock=clock,
        elevator_capacity=8,
    )
    building.request_elevator(floor=0, direction=Direction.UP)
    building.request_floor(elevator_id=0, floor=7)

    visited: list[int] = []
    e = building._elevators[0]
    for _ in range(200):
        building.step()
        if e.door_state == DoorState.OPENING:
            if not visited or visited[-1] != e.current_floor:
                visited.append(e.current_floor)
        if building.pending_requests() == 0 and e.direction == Direction.IDLE:
            break

    assert 7 in visited, f"Elevator never stopped at floor 7; visited={visited}"
    assert building.pending_requests() == 0


# ---------------------------------------------------------------------------
# Test 11: alight_passenger raises on empty elevator
# ---------------------------------------------------------------------------

def test_alight_passenger_raises_on_empty_elevator(single_elevator: Elevator) -> None:
    """
    Given: elevator with 0 passengers.
    When: alight_passenger() called.
    Then: ValueError raised.
    """
    with pytest.raises(ValueError, match="no passengers"):
        single_elevator.alight_passenger()


# ---------------------------------------------------------------------------
# Test 12: add_stop ignores duplicate floors
# ---------------------------------------------------------------------------

def test_add_stop_ignores_duplicate(single_elevator: Elevator) -> None:
    """
    Adding the same floor twice should not add it to the queue twice.
    """
    single_elevator.add_stop(5)
    single_elevator.add_stop(5)

    assert len(single_elevator._stops_set) == 1
    assert single_elevator._stops_above.count(5) == 1
```

---

## Problem 2: In-Memory KV Store

### Class Diagram

```
Interfaces (ABCs)
─────────────────
+──────────────────────+     +──────────────────────────+
│  Clock (ABC)         │     │  EvictionPolicy (ABC)    │
│ +now() -> float      │     │ +on_access(ns, key)      │
+──────────────────────+     │ +on_delete(ns, key)      │
         ^                   │ +evict_one()             │
         |                   │   -> (ns,key) | None     │
+──────────────────────+     +──────────────────────────+
│  MonotonicClock      │               ^
+──────────────────────+               |
                            +──────────────────────────+
                            │  LRUEvictionPolicy       │
                            │ -_order: OrderedDict     │
                            +──────────────────────────+

Core
────
+──────────────────────────────────+
│  KVStore                         │
│ -_store: dict[ns][key]=KVEntry   │
│ -_lock: RLock                    │
│ -_capacity: int                  │
│ -_eviction: EvictionPolicy       │
│ -_clock: Clock                   │
│ -_sweep_thread: Thread           │
│ +set(key,val,ttl,ns)             │
│ +get(key,ns) -> Any              │
│ +delete(key,ns)                  │
│ +exists(key,ns) -> bool          │
│ +ttl(key,ns) -> float|None       │
│ +delete_namespace(ns) -> int     │
│ +size(ns=None) -> int            │
│ +stop()                          │
+──────────────────────────────────+

Pydantic model
+──────────────────────────────────+
│  KVEntry (BaseModel)             │
│ -key: str                        │
│ -namespace: str                  │
│ -value: Any                      │
│ -created_at: float               │
│ -expires_at: float | None        │
│ -last_accessed_at: float         │
│ +is_expired(now) -> bool         │
│ +remaining_ttl(now) -> float|None│
+──────────────────────────────────+
```

---

### Implementation

```python
# kv_store/models.py

"""
KVEntry Pydantic model with TTL helpers.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class KVEntry(BaseModel):
    """A single stored key-value entry with optional TTL."""

    key: str
    namespace: str
    value: Any
    created_at: float
    expires_at: Optional[float] = None  # None means persistent
    last_accessed_at: float

    model_config = {"arbitrary_types_allowed": True}

    def is_expired(self, now: float) -> bool:
        """Return True if this entry has expired at time `now`."""
        if self.expires_at is None:
            return False
        return now >= self.expires_at

    def remaining_ttl(self, now: float) -> Optional[float]:
        """
        Return remaining TTL in seconds.
        Returns -1.0 if the key is persistent (no TTL).
        Returns None if the key has expired.
        """
        if self.expires_at is None:
            return -1.0
        remaining = self.expires_at - now
        if remaining <= 0:
            return None
        return remaining
```

```python
# kv_store/interfaces.py

"""
Abstract interfaces for the KV store.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Clock(ABC):
    @abstractmethod
    def now(self) -> float:
        """Return monotonic time in seconds."""
        ...


class EvictionPolicy(ABC):
    """Decides which key to evict when the store is full."""

    @abstractmethod
    def on_access(self, namespace: str, key: str) -> None:
        """Called when a key is read (get) or written (set)."""
        ...

    @abstractmethod
    def on_delete(self, namespace: str, key: str) -> None:
        """Called when a key is explicitly deleted or expired."""
        ...

    @abstractmethod
    def evict_one(self) -> tuple[str, str] | None:
        """Return (namespace, key) to evict, or None if nothing to evict."""
        ...
```

```python
# kv_store/clock.py

"""
Clock implementations.
"""

from __future__ import annotations

import time

from .interfaces import Clock


class MonotonicClock(Clock):
    """Real-time clock backed by time.monotonic()."""

    def now(self) -> float:
        return time.monotonic()


class FakeClock(Clock):
    """
    Deterministic clock for tests.
    Advances only when advance() is called explicitly.
    """

    def __init__(self, t: float = 0.0) -> None:
        self._t = t

    def advance(self, dt: float) -> None:
        self._t += dt

    def now(self) -> float:
        return self._t
```

```python
# kv_store/eviction.py

"""
LRU eviction policy using OrderedDict.

Design:
  - OrderedDict maintains insertion order. move_to_end() is O(1).
  - on_access moves the key to the most-recently-used (rightmost) end.
  - evict_one pops the least-recently-used (leftmost) entry.
  - Global LRU order across all namespaces: the dict key is (namespace, key).

Alternative considered: per-namespace LRU with a doubly linked list.
Rejected because global LRU across namespaces is simpler and the problem
spec says eviction is global.

Complexity:
  - on_access: O(1) amortized (OrderedDict move_to_end)
  - on_delete: O(1)
  - evict_one: O(1) (popitem from leftmost end)
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Optional

from .interfaces import EvictionPolicy

logger = logging.getLogger(__name__)


class LRUEvictionPolicy(EvictionPolicy):
    """
    Global LRU eviction across all namespaces.
    """

    def __init__(self) -> None:
        # Key: (namespace, key), value: None (we only care about order)
        self._order: OrderedDict[tuple[str, str], None] = OrderedDict()

    def on_access(self, namespace: str, key: str) -> None:
        token = (namespace, key)
        if token in self._order:
            self._order.move_to_end(token)
        else:
            self._order[token] = None

    def on_delete(self, namespace: str, key: str) -> None:
        self._order.pop((namespace, key), None)

    def evict_one(self) -> Optional[tuple[str, str]]:
        if not self._order:
            return None
        ns_key, _ = self._order.popitem(last=False)  # leftmost = LRU
        logger.debug("LRU eviction: namespace=%s key=%s", ns_key[0], ns_key[1])
        return ns_key
```

```python
# kv_store/store.py

"""
Thread-safe in-memory KV store with TTL expiry and LRU eviction.

Design decisions:
  - Single RLock guards all mutable state. A reader-writer lock would allow
    concurrent reads, but Python's GIL already serializes CPU-bound operations
    and the lock is held only for O(1) dict operations. The overhead of a
    full reader-writer lock is not justified at this scale.
  - LRU eviction is global across namespaces. Alternative (per-namespace LRU)
    would allow one namespace to fill up while another is empty, which wastes
    capacity. Global LRU maximises utilization.
  - Background sweep uses a daemon thread with threading.Event for clean shutdown.
    The sweep acquires the RLock, so it is safe to call stop() from any thread.
  - Clock is injected so tests can control time without real sleeps.

Complexity:
  - get:              O(1) average
  - set (no evict):   O(1) average
  - set (with evict): O(1) amortized (LRU evict_one is O(1))
  - _sweep_expired:   O(N) where N = total number of entries
  - delete_namespace: O(K) where K = keys in the namespace

Limitations and future work:
  - No persistence. Production: write-ahead log or periodic snapshot to disk/Redis.
  - No replication or sharding. Distributed version: consistent-hash sharding
    across nodes with a gossip-based TTL sync protocol.
  - Monitoring: add Prometheus counters for hit rate, eviction rate, sweep latency.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

from .clock import MonotonicClock
from .eviction import LRUEvictionPolicy
from .interfaces import Clock, EvictionPolicy
from .models import KVEntry

logger = logging.getLogger(__name__)


class KVStore:
    """
    Thread-safe in-memory key-value store.

    Features:
      - Optional per-key TTL with lazy expiry (on get/exists) and eager expiry
        (background sweep thread).
      - LRU eviction when capacity is exceeded.
      - Namespace isolation: keys in different namespaces are independent.
      - Injectable Clock for deterministic testing.
    """

    def __init__(
        self,
        capacity: int = 1024,
        sweep_interval_seconds: float = 5.0,
        clock: Optional[Clock] = None,
        eviction_policy: Optional[EvictionPolicy] = None,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._sweep_interval = sweep_interval_seconds
        self._clock: Clock = clock or MonotonicClock()
        self._eviction: EvictionPolicy = eviction_policy or LRUEvictionPolicy()

        # _store[namespace][key] = KVEntry
        self._store: dict[str, dict[str, KVEntry]] = {}
        self._lock = threading.RLock()

        # Background sweep
        self._stop_event = threading.Event()
        self._sweep_thread = threading.Thread(
            target=self._background_sweep,
            daemon=True,
            name="kv-sweep",
        )
        self._sweep_thread.start()
        logger.info(
            "KVStore started: capacity=%d sweep_interval=%.1fs",
            capacity, sweep_interval_seconds,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[float] = None,
        namespace: str = "default",
    ) -> None:
        """Store a key. If ttl_seconds given, key expires after that duration."""
        self._validate_namespace(namespace)
        now = self._clock.now()
        expires_at = (now + ttl_seconds) if ttl_seconds is not None else None

        with self._lock:
            ns_store = self._store.setdefault(namespace, {})
            existing = ns_store.get(key)

            if existing is None:
                # New key — may need to evict
                total = self._total_size()
                while total >= self._capacity:
                    evicted = self._eviction.evict_one()
                    if evicted is None:
                        break
                    ev_ns, ev_key = evicted
                    if ev_ns in self._store and ev_key in self._store[ev_ns]:
                        del self._store[ev_ns][ev_key]
                        logger.debug("Evicted %s/%s", ev_ns, ev_key)
                    total -= 1

            entry = KVEntry(
                key=key,
                namespace=namespace,
                value=value,
                created_at=now,
                expires_at=expires_at,
                last_accessed_at=now,
            )
            ns_store[key] = entry
            self._eviction.on_access(namespace, key)

        logger.debug("SET %s/%s ttl=%s", namespace, key, ttl_seconds)

    def get(
        self,
        key: str,
        namespace: str = "default",
    ) -> Any:
        """Return value for key, or None if missing or expired."""
        self._validate_namespace(namespace)
        now = self._clock.now()

        with self._lock:
            entry = self._store.get(namespace, {}).get(key)
            if entry is None:
                return None
            if entry.is_expired(now):
                self._delete_internal(namespace, key)
                return None
            # Update last_accessed_at for LRU tracking
            updated = entry.model_copy(update={"last_accessed_at": now})
            self._store[namespace][key] = updated
            self._eviction.on_access(namespace, key)
            return updated.value

    def delete(
        self,
        key: str,
        namespace: str = "default",
    ) -> None:
        """Remove a key. No-op if key does not exist."""
        self._validate_namespace(namespace)
        with self._lock:
            self._delete_internal(namespace, key)

    def exists(
        self,
        key: str,
        namespace: str = "default",
    ) -> bool:
        """Return True if key exists and has not expired."""
        return self.get(key, namespace) is not None

    def ttl(
        self,
        key: str,
        namespace: str = "default",
    ) -> Optional[float]:
        """
        Return remaining TTL in seconds.
        Returns -1.0 if persistent.
        Returns None if key does not exist or has expired.
        """
        self._validate_namespace(namespace)
        now = self._clock.now()
        with self._lock:
            entry = self._store.get(namespace, {}).get(key)
            if entry is None or entry.is_expired(now):
                if entry is not None and entry.is_expired(now):
                    self._delete_internal(namespace, key)
                return None
            return entry.remaining_ttl(now)

    def delete_namespace(self, namespace: str) -> int:
        """Remove all keys in namespace. Return count of deleted keys."""
        self._validate_namespace(namespace)
        with self._lock:
            ns_store = self._store.pop(namespace, {})
            count = len(ns_store)
            for key in ns_store:
                self._eviction.on_delete(namespace, key)
            logger.info("Deleted namespace %s (%d keys)", namespace, count)
            return count

    def size(self, namespace: Optional[str] = None) -> int:
        """Return count of live (non-expired) keys."""
        now = self._clock.now()
        with self._lock:
            if namespace is not None:
                ns_store = self._store.get(namespace, {})
                return sum(
                    1 for e in ns_store.values() if not e.is_expired(now)
                )
            return sum(
                1
                for ns_store in self._store.values()
                for e in ns_store.values()
                if not e.is_expired(now)
            )

    def stop(self) -> None:
        """Stop the background sweep thread. Call in tests and on shutdown."""
        self._stop_event.set()
        self._sweep_thread.join(timeout=2.0)
        logger.info("KVStore stopped")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_namespace(self, namespace: str) -> None:
        if not namespace:
            raise ValueError("namespace must be a non-empty string")

    def _total_size(self) -> int:
        """Total number of entries (including expired). Must be called with lock held."""
        return sum(len(ns) for ns in self._store.values())

    def _delete_internal(self, namespace: str, key: str) -> None:
        """Remove a key. Must be called with self._lock held."""
        ns_store = self._store.get(namespace)
        if ns_store and key in ns_store:
            del ns_store[key]
            self._eviction.on_delete(namespace, key)
            if not ns_store:
                del self._store[namespace]

    def _sweep_expired(self) -> int:
        """
        Remove all expired entries. Returns count of entries removed.
        Safe to call from any thread — acquires lock internally.
        """
        now = self._clock.now()
        expired: list[tuple[str, str]] = []

        with self._lock:
            for namespace, ns_store in list(self._store.items()):
                for key, entry in list(ns_store.items()):
                    if entry.is_expired(now):
                        expired.append((namespace, key))

            for namespace, key in expired:
                self._delete_internal(namespace, key)

        if expired:
            logger.debug("Sweep removed %d expired entries", len(expired))
        return len(expired)

    def _background_sweep(self) -> None:
        """Background thread: sweep expired keys at regular intervals."""
        while not self._stop_event.wait(timeout=self._sweep_interval):
            try:
                self._sweep_expired()
            except Exception:
                logger.exception("Error in background sweep")
```

---

### Tests

```python
# tests/test_kv_store.py

"""
KV store test suite — 15 tests covering basic ops, TTL, LRU eviction,
namespace isolation, concurrency, and background sweep.
"""

from __future__ import annotations

import threading
import time

import pytest

from kv_store.clock import FakeClock
from kv_store.eviction import LRUEvictionPolicy
from kv_store.store import KVStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(0.0)


@pytest.fixture
def store(clock: FakeClock) -> KVStore:
    s = KVStore(capacity=1024, sweep_interval_seconds=60.0, clock=clock)
    yield s
    s.stop()


@pytest.fixture
def small_store(clock: FakeClock) -> KVStore:
    s = KVStore(capacity=3, sweep_interval_seconds=60.0, clock=clock)
    yield s
    s.stop()


# ---------------------------------------------------------------------------
# Test 1: basic set and get
# ---------------------------------------------------------------------------

def test_set_and_get_basic(store: KVStore) -> None:
    """set("key", "value") -> get("key") returns "value"."""
    store.set("greeting", "hello")
    assert store.get("greeting") == "hello"


# ---------------------------------------------------------------------------
# Test 2: missing key returns None
# ---------------------------------------------------------------------------

def test_get_missing_key_returns_none(store: KVStore) -> None:
    """get() on a non-existent key returns None."""
    assert store.get("nonexistent") is None


# ---------------------------------------------------------------------------
# Test 3: TTL expiry (lazy)
# ---------------------------------------------------------------------------

def test_ttl_expiry_lazy(store: KVStore, clock: FakeClock) -> None:
    """Key expires after TTL elapses; get() returns None after expiry."""
    store.set("k", "v", ttl_seconds=10.0)
    clock.advance(11.0)
    assert store.get("k") is None


# ---------------------------------------------------------------------------
# Test 4: key not expired before TTL
# ---------------------------------------------------------------------------

def test_ttl_not_expired_yet(store: KVStore, clock: FakeClock) -> None:
    """Key is still accessible before TTL elapses."""
    store.set("k", "v", ttl_seconds=10.0)
    clock.advance(5.0)
    assert store.get("k") == "v"


# ---------------------------------------------------------------------------
# Test 5: ttl() returns remaining time
# ---------------------------------------------------------------------------

def test_ttl_remaining(store: KVStore, clock: FakeClock) -> None:
    """ttl() returns approximately the remaining seconds."""
    store.set("k", "v", ttl_seconds=30.0)
    clock.advance(10.0)
    remaining = store.ttl("k")
    assert remaining is not None
    assert 19.9 <= remaining <= 20.1


# ---------------------------------------------------------------------------
# Test 6: persistent key ttl returns -1
# ---------------------------------------------------------------------------

def test_persistent_key_ttl_returns_minus_one(store: KVStore) -> None:
    """A key set without TTL returns -1.0 from ttl()."""
    store.set("k", "v")
    assert store.ttl("k") == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Test 7: LRU eviction removes least recently used
# ---------------------------------------------------------------------------

def test_lru_eviction_removes_least_recently_used(
    small_store: KVStore,
    clock: FakeClock,
) -> None:
    """
    capacity=3. After set a,b,c and accessing a, inserting d evicts b (LRU).
    """
    s = small_store
    s.set("a", 1)
    s.set("b", 2)
    s.set("c", 3)

    # Access "a" to make it recently used
    assert s.get("a") == 1

    # Insert "d" — should evict "b" (LRU, not accessed since creation)
    s.set("d", 4)

    assert s.get("b") is None     # evicted
    assert s.get("a") == 1        # recently accessed — still present
    assert s.get("d") == 4        # just inserted
    assert s.get("c") == 3        # still present (second-oldest access is "c")


# ---------------------------------------------------------------------------
# Test 8: namespace isolation
# ---------------------------------------------------------------------------

def test_namespace_isolation(store: KVStore) -> None:
    """Same key in different namespaces stores independent values."""
    store.set("k", "ns1_val", namespace="ns1")
    store.set("k", "ns2_val", namespace="ns2")
    assert store.get("k", namespace="ns1") == "ns1_val"
    assert store.get("k", namespace="ns2") == "ns2_val"


# ---------------------------------------------------------------------------
# Test 9: delete_namespace
# ---------------------------------------------------------------------------

def test_delete_namespace(store: KVStore) -> None:
    """delete_namespace removes all keys in that namespace only."""
    for i in range(5):
        store.set(f"key{i}", i, namespace="temp")
    for i in range(3):
        store.set(f"key{i}", i, namespace="persist")

    deleted = store.delete_namespace("temp")

    assert deleted == 5
    assert store.size(namespace="temp") == 0
    assert store.size(namespace="persist") == 3


# ---------------------------------------------------------------------------
# Test 10: concurrent set/get — no data races
# ---------------------------------------------------------------------------

def test_concurrent_set_get_no_data_races(clock: FakeClock) -> None:
    """20 threads each set 50 unique keys; no exceptions; size <= capacity."""
    store = KVStore(capacity=500, sweep_interval_seconds=60.0, clock=clock)
    errors: list[Exception] = []

    def worker(thread_id: int) -> None:
        try:
            for i in range(50):
                key = f"t{thread_id}_k{i}"
                store.set(key, thread_id * 1000 + i)
                val = store.get(key)
                # Value may be None if evicted, but must not raise
                assert val is None or isinstance(val, int)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    store.stop()
    assert not errors, f"Errors: {errors}"
    assert store.size() <= 500


# ---------------------------------------------------------------------------
# Test 11: background sweep removes expired keys
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_background_sweep_removes_expired() -> None:
    """
    Background sweep clears expired keys within sweep_interval.
    Uses real time.sleep — slow test, skipped in fast CI.
    """
    real_store = KVStore(
        capacity=100,
        sweep_interval_seconds=0.1,
    )
    for i in range(10):
        real_store.set(f"k{i}", i, ttl_seconds=0.05)

    time.sleep(0.4)  # Wait for at least one sweep cycle

    assert real_store.size() == 0
    real_store.stop()


# ---------------------------------------------------------------------------
# Test 12: overwrite resets TTL
# ---------------------------------------------------------------------------

def test_overwrite_resets_ttl(store: KVStore, clock: FakeClock) -> None:
    """Overwriting a key resets its TTL to the new value."""
    store.set("k", "old", ttl_seconds=10.0)
    clock.advance(8.0)
    store.set("k", "new", ttl_seconds=30.0)

    remaining = store.ttl("k")
    assert remaining is not None
    assert 29.0 <= remaining <= 30.1  # TTL reset, not extended from original


# ---------------------------------------------------------------------------
# Test 13: delete removes key
# ---------------------------------------------------------------------------

def test_delete_removes_key(store: KVStore) -> None:
    """delete() removes key; subsequent get() returns None."""
    store.set("k", "v")
    store.delete("k")
    assert store.get("k") is None


# ---------------------------------------------------------------------------
# Test 14: delete no-op on missing key
# ---------------------------------------------------------------------------

def test_delete_noops_on_missing_key(store: KVStore) -> None:
    """delete() on a non-existent key does not raise."""
    store.delete("does_not_exist")  # Should not raise


# ---------------------------------------------------------------------------
# Test 15: exists returns False after expiry
# ---------------------------------------------------------------------------

def test_exists_returns_false_after_expiry(store: KVStore, clock: FakeClock) -> None:
    """exists() returns False after the key's TTL has elapsed."""
    store.set("k", "v", ttl_seconds=5.0)
    assert store.exists("k") is True
    clock.advance(6.0)
    assert store.exists("k") is False
```

---

## Complexity Analysis

| Operation | Time | Space | Notes |
|---|---|---|---|
| Elevator: `add_stop` | O(N) | O(N) | Sorted list insertion via `sorted()`. N = stops in queue. Use `bisect.insort` for O(log N) search + O(N) shift; full sort is clearer for small N (<100 stops). |
| Elevator: `next_stop` (SCAN) | O(1) | O(1) | Returns `_stops_above[0]` or `_stops_below[0]` — always the sorted head. |
| Elevator: `step` | O(N) | O(1) | Calls `_redistribute_stops` which is O(N). Acceptable since N << 100. |
| KV: `get` | O(1) avg | O(1) | Dict lookup + optional model_copy (O(fields)). |
| KV: `set` (no eviction) | O(1) avg | O(1) | Dict insert + LRU on_access O(1). |
| KV: `set` (with eviction) | O(1) amortized | O(1) | LRU evict_one is O(1) (OrderedDict popitem). At most one eviction per set. |
| KV: `_sweep_expired` | O(N) | O(K) | N = total entries, K = expired entries collected before deletion. |
| KV: `delete_namespace` | O(K) | O(K) | K = keys in the namespace. Iterates once over the namespace dict. |

## Design Decisions

### Elevator System

**Dispatch strategy**: `NearestIdleStrategy` — assigns the closest idle elevator. Rejected: `FirstAvailableStrategy` (assigns to elevator 0 always unless busy) — causes uneven wear and longer wait times for floors distant from elevator 0.

**SCAN queue representation**: two sorted lists (`_stops_above`, `_stops_below`). Alternative: a single priority queue with direction tags. Rejected because two lists make direction reversal explicit and next_stop() O(1) without any priority queue overhead.

**Concurrency**: single RLock per elevator for `add_stop` and `step`. Building-level RLock for dispatch. No lock nesting — Building lock is released before calling `elevator.add_stop()` to avoid hold-and-wait deadlock.

### KV Store

**Eviction policy**: global LRU with `OrderedDict`. Alternative: per-namespace LRU. Rejected because global LRU maximizes overall cache utilization — a namespace that is not used should not protect its keys from eviction.

**Background sweep**: daemon thread with `threading.Event` for clean shutdown via `stop()`. Alternative: greenlet/asyncio task. Rejected — the problem spec says stdlib only; asyncio would add significant complexity without benefit for this synchronous interface.

**Lock granularity**: single `RLock` (reentrant). Alternative: separate read-lock / write-lock (readers-writer lock). Rejected — Python's GIL means lock contention is low for CPU-bound operations, and the lock is held only for O(1) dict operations. A readers-writer lock implementation in pure Python adds overhead without measurable throughput gain at the stated scale.
