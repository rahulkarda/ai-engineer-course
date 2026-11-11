# Week 07 Project — LLD Implementations: Elevator System & In-Memory KV Store

> Deliverable: Two complete LLD solutions — class diagram + working Python 3.12+ implementation + passing pytest suite for each.  ·  Time: ~4 hr  ·  Difficulty: Advanced

---

## Project Brief

This week's project applies everything from the LLD method sessions. You will implement two classic LLD problems from scratch: an Elevator System and an In-Memory Key-Value Store. Both are common in senior and staff engineer interviews at FAANG and enterprise tech companies.

For each problem you must produce:

1. A class diagram (drawn or described in text) showing all classes, interfaces, and relationships.
2. A working Python implementation with full type hints, `logging` (not `print`), and Pydantic v2 where appropriate for data models crossing boundaries.
3. A pytest test suite covering the required test cases listed below.

Both solutions must be self-contained — no external services, no network calls, no database. All state lives in memory.

**Project layout:**

```
week-07-project/
  elevator/
    __init__.py
    models.py        # enums, value objects, Pydantic models
    interfaces.py    # ABCs
    scheduler.py     # SCAN scheduling algorithm
    elevator.py      # Elevator state machine
    building.py      # Building / dispatcher orchestrator
  kv_store/
    __init__.py
    models.py        # TTL entry, namespace models
    interfaces.py    # ABCs
    store.py         # Main KV store with LRU eviction
    eviction.py      # LRU policy implementation
  tests/
    test_elevator.py
    test_kv_store.py
  pyproject.toml
```

---

## Problem 1: Elevator System

### Requirements Specification

**Functional requirements:**

- A building has `N` elevators (configurable, default 3) and `F` floors (configurable, default 10).
- Two types of requests:
  - **External request**: a person on floor `f` presses UP or DOWN.
  - **Internal request**: a passenger inside elevator `e` presses a destination floor button.
- Each elevator has:
  - A direction state: `MOVING_UP`, `MOVING_DOWN`, or `IDLE`.
  - A door state: `DOOR_OPEN`, `DOOR_CLOSED`, or `DOOR_OPENING`, `DOOR_CLOSING`.
  - A current floor (integer).
  - A capacity limit (max passengers, configurable, default 8).
  - A queue of pending floor stops.
- The dispatcher assigns external requests to the most appropriate elevator using a configurable selection strategy.
- Each elevator uses the SCAN algorithm (also called the elevator algorithm or look algorithm) to process its queue:
  - While moving UP: serve all stops at or above current floor in ascending order.
  - When no more stops above: reverse to MOVING_DOWN (or go IDLE if no requests at all).
  - While moving DOWN: serve all stops at or below current floor in descending order.
- Door state machine:
  - `DOOR_CLOSED` → `DOOR_OPENING` (when arriving at a requested floor)
  - `DOOR_OPENING` → `DOOR_OPEN` (after open delay, configurable)
  - `DOOR_OPEN` → `DOOR_CLOSING` (after dwell time, configurable)
  - `DOOR_CLOSING` → `DOOR_CLOSED` (after close delay)
  - Invalid transitions raise `InvalidDoorTransitionError`.
- Capacity: if an elevator is at capacity, it skips boarding new passengers at a floor stop (door still opens for alighting).

**Non-functional requirements:**

- Thread-safe: the dispatcher and each elevator's queue must be safe for concurrent access.
- No busy-waiting in the scheduler — use condition variables or queues.
- Deterministic in tests: inject a clock and a move-step function so tests do not rely on `time.sleep`.

### Constraints Table

| Constraint | Value |
|---|---|
| Min floors | 2 |
| Max floors | 100 |
| Min elevators | 1 |
| Max elevators | 16 |
| Default capacity per elevator | 8 passengers |
| SCAN: tie-breaking rule | Lower floor wins when two floors equidistant |
| Door transition: invalid state raises | `InvalidDoorTransitionError` |
| Clock | Injectable via `Clock` ABC |
| Scheduling strategy | Injectable via `DispatchStrategy` ABC |

### Required Python Classes

Implement all of the following. You may add helper classes, but do not remove or rename these.

```python
# models.py

from enum import Enum, auto

class Direction(Enum):
    UP = auto()
    DOWN = auto()
    IDLE = auto()

class DoorState(Enum):
    OPEN = auto()
    CLOSED = auto()
    OPENING = auto()
    CLOSING = auto()

from pydantic import BaseModel, Field

class FloorRequest(BaseModel):
    """A request originating outside the elevator (hall call)."""
    floor: int = Field(..., ge=0)
    direction: Direction

class ElevatorRequest(BaseModel):
    """A request originating inside an elevator (cab call)."""
    elevator_id: int
    destination_floor: int = Field(..., ge=0)
```

```python
# interfaces.py

from abc import ABC, abstractmethod
from .models import FloorRequest, Direction

class DispatchStrategy(ABC):
    """Selects which elevator should handle a hall call."""

    @abstractmethod
    def select_elevator(
        self,
        request: FloorRequest,
        elevators: list["ElevatorState"],  # read-only view
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
# elevator.py

class InvalidDoorTransitionError(Exception):
    """Raised when an invalid door state transition is attempted."""

class Elevator:
    """
    Single elevator with state machine and SCAN queue.

    Key methods (implement all):
    """

    def __init__(
        self,
        elevator_id: int,
        num_floors: int,
        capacity: int,
        clock: Clock,
    ) -> None: ...

    @property
    def current_floor(self) -> int: ...

    @property
    def direction(self) -> Direction: ...

    @property
    def door_state(self) -> DoorState: ...

    @property
    def passenger_count(self) -> int: ...

    def add_stop(self, floor: int) -> None:
        """Add a floor to this elevator's SCAN queue (thread-safe)."""
        ...

    def step(self) -> None:
        """
        Advance the elevator one logical step:
        - If door is OPENING/CLOSING, advance door state.
        - If DOOR_CLOSED and queue is non-empty, move one floor toward next stop.
        - If arriving at a stop, trigger door open sequence.
        - Update direction when reversing.
        """
        ...

    def open_door(self) -> None:
        """Initiate door opening. Raises InvalidDoorTransitionError if not DOOR_CLOSED."""
        ...

    def close_door(self) -> None:
        """Initiate door closing. Raises InvalidDoorTransitionError if not DOOR_OPEN."""
        ...

    def board_passenger(self) -> bool:
        """Attempt to board one passenger. Returns False if at capacity."""
        ...

    def alight_passenger(self) -> None:
        """Record one passenger alighting. Raises if count would go negative."""
        ...

    def next_stop(self) -> int | None:
        """Return the next floor this elevator will stop at under SCAN, or None."""
        ...
```

```python
# building.py

class Building:
    """
    Dispatcher: receives hall calls, assigns to elevators, runs simulation steps.

    Key methods:
    """

    def __init__(
        self,
        num_floors: int,
        num_elevators: int,
        strategy: DispatchStrategy,
        clock: Clock,
        elevator_capacity: int = 8,
    ) -> None: ...

    def request_elevator(self, floor: int, direction: Direction) -> None:
        """Submit a hall call. Assigns to an elevator via the dispatch strategy."""
        ...

    def request_floor(self, elevator_id: int, floor: int) -> None:
        """Submit a cab call (passenger inside presses floor button)."""
        ...

    def step(self) -> None:
        """Advance all elevators one step. Thread-safe."""
        ...

    def elevator_positions(self) -> list[tuple[int, Direction, DoorState]]:
        """Return (floor, direction, door_state) for each elevator."""
        ...

    def pending_requests(self) -> int:
        """Return total number of unserviced stops across all elevators."""
        ...
```

### Required Test Cases

All of the following must pass. Write them in `tests/test_elevator.py`.

```python
# test_elevator.py — required test stubs

def test_hall_call_assigned_to_nearest_idle_elevator(): ...
    """
    Given: 3 elevators, all idle. Elevator 0 at floor 0, elevator 1 at floor 5, elevator 2 at floor 9.
    When: hall call for floor 5, direction UP.
    Then: elevator 1 is assigned (nearest idle).
    """

def test_scan_serves_floors_in_ascending_order_while_moving_up(): ...
    """
    Given: elevator at floor 1, moving UP, stops queued: [4, 7, 2].
    When: step() called repeatedly.
    Then: floors are visited in order 2, 4, 7.
    """

def test_scan_reverses_direction_at_top(): ...
    """
    Given: elevator at floor 9 (top), moving UP, stop at floor 3 queued.
    When: step() called.
    Then: direction changes to MOVING_DOWN, next stop is floor 3.
    """

def test_door_open_close_cycle(): ...
    """
    Given: elevator at DOOR_CLOSED.
    When: open_door() called, then close_door() called.
    Then: sequence is CLOSED → OPENING → OPEN → CLOSING → CLOSED.
    """

def test_invalid_door_transition_raises(): ...
    """
    Given: elevator door is DOOR_OPEN.
    When: open_door() is called again.
    Then: InvalidDoorTransitionError is raised.
    """

def test_elevator_at_capacity_rejects_boarding(): ...
    """
    Given: elevator at capacity (8 passengers).
    When: board_passenger() called.
    Then: returns False, passenger count unchanged.
    """

def test_elevator_returns_idle_when_queue_empty(): ...
    """
    Given: elevator with one queued stop, moving UP.
    When: step() called until stop is reached and served.
    Then: elevator returns to IDLE with no pending stops.
    """

def test_concurrent_hall_calls_are_all_assigned(): ...
    """
    Given: building with 3 elevators.
    When: 50 hall calls submitted from 10 concurrent threads.
    Then: all 50 calls are assigned without data races (no lost updates, no assertion errors).
    """

def test_cab_call_added_to_correct_elevator(): ...
    """
    Given: building with elevator 2 at floor 3.
    When: request_floor(elevator_id=2, floor=8) called.
    Then: elevator 2 has floor 8 in its queue; other elevators do not.
    """

def test_full_round_trip_single_elevator(): ...
    """
    Given: single-elevator building, floors 0–9.
    When: hall call from floor 0 UP, then cab call to floor 7.
    Then: step() until pending_requests()==0; elevator visited floor 7.
    """
```

---

## Problem 2: In-Memory KV Store

### Requirements Specification

**Functional requirements:**

- `set(key, value, ttl_seconds=None)`: Store a key-value pair. If `ttl_seconds` is provided, the key expires after that many seconds. Overwriting an existing key resets its TTL.
- `get(key)`: Return the value for a key. Return `None` if the key does not exist or has expired. Accessing an existing key counts as a use for LRU purposes.
- `delete(key)`: Remove a key. No-op if key does not exist.
- `exists(key)`: Return `True` if the key exists and has not expired.
- `ttl(key)`: Return remaining TTL in seconds as a float. Return `-1` if the key has no TTL (persistent). Return `None` if the key does not exist or has expired.
- **LRU eviction**: The store has a configurable maximum capacity (number of keys). When the store is full and a new key is inserted, evict the least-recently-used key. A key is "used" when it is created, accessed via `get`, or overwritten via `set`.
- **Namespace support**: All operations accept an optional `namespace: str` parameter (default `"default"`). Keys in different namespaces are independent. `delete_namespace(namespace)` removes all keys in a namespace.
- **Background TTL expiry**: Expired keys are cleaned up lazily (on `get` or `exists`) AND eagerly (by a background sweep thread that runs every `sweep_interval_seconds`). The sweep must not block reads/writes.

**Non-functional requirements:**

- Thread-safe: all public methods must be safe for concurrent access.
- No external dependencies. Use only stdlib and Pydantic v2.
- The clock must be injectable for deterministic tests (no `time.time()` calls in production code paths that are not routed through the `Clock` interface).
- Memory-efficient: do not store expired entries beyond the next sweep cycle.

### Constraints Table

| Constraint | Value |
|---|---|
| Max capacity | Configurable; default 1024 keys across all namespaces |
| TTL precision | Float seconds; granularity is test-clock resolution |
| LRU eviction | Evicts across all namespaces (global LRU order) |
| Namespace naming | Any non-empty string; validated at input |
| Background sweep interval | Configurable; default 5.0 seconds |
| Thread safety | All public methods; use `threading.RLock` |
| Clock | Injectable via `Clock` ABC (same interface as elevator problem) |
| Key type | `str` only |
| Value type | `Any` (no serialization required) |

### Required Python Classes

```python
# models.py

from pydantic import BaseModel, Field
from typing import Any, Optional
import datetime

class KVEntry(BaseModel):
    """A single stored entry."""
    key: str
    namespace: str
    value: Any
    created_at: float   # monotonic timestamp
    expires_at: Optional[float] = None   # None means no expiry
    last_accessed_at: float

    model_config = {"arbitrary_types_allowed": True}

    def is_expired(self, now: float) -> bool:
        """Return True if this entry has expired at time `now`."""
        ...

    def remaining_ttl(self, now: float) -> Optional[float]:
        """
        Return remaining TTL in seconds, -1 if persistent, None if expired.
        """
        ...
```

```python
# interfaces.py

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
# eviction.py

class LRUEvictionPolicy(EvictionPolicy):
    """
    Tracks global LRU order across all namespaces using an OrderedDict.
    on_access moves (namespace, key) to the most-recently-used end.
    evict_one returns the least-recently-used (namespace, key).
    """

    def __init__(self) -> None: ...
    def on_access(self, namespace: str, key: str) -> None: ...
    def on_delete(self, namespace: str, key: str) -> None: ...
    def evict_one(self) -> tuple[str, str] | None: ...
```

```python
# store.py

import logging
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

class KVStore:
    """
    Thread-safe in-memory key-value store with TTL expiry and LRU eviction.
    """

    def __init__(
        self,
        capacity: int = 1024,
        sweep_interval_seconds: float = 5.0,
        clock: Clock | None = None,
        eviction_policy: EvictionPolicy | None = None,
    ) -> None: ...

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: float | None = None,
        namespace: str = "default",
    ) -> None: ...

    def get(
        self,
        key: str,
        namespace: str = "default",
    ) -> Any: ...

    def delete(
        self,
        key: str,
        namespace: str = "default",
    ) -> None: ...

    def exists(
        self,
        key: str,
        namespace: str = "default",
    ) -> bool: ...

    def ttl(
        self,
        key: str,
        namespace: str = "default",
    ) -> float | None: ...

    def delete_namespace(self, namespace: str) -> int:
        """Remove all keys in namespace. Return count of deleted keys."""
        ...

    def size(self, namespace: str | None = None) -> int:
        """
        Return number of live keys. If namespace given, count only that namespace.
        """
        ...

    def _sweep_expired(self) -> int:
        """
        Remove all expired entries. Called by background thread and lazily.
        Returns number of entries removed.
        """
        ...

    def stop(self) -> None:
        """Stop the background sweep thread. Call in tests and on shutdown."""
        ...
```

### Required Test Cases

All of the following must pass. Write them in `tests/test_kv_store.py`.

```python
def test_set_and_get_basic(): ...
    """
    When: set("key", "value") then get("key").
    Then: returns "value".
    """

def test_get_missing_key_returns_none(): ...
    """
    When: get("nonexistent").
    Then: returns None.
    """

def test_ttl_expiry_lazy(): ...
    """
    Given: clock at T=0. set("k", "v", ttl_seconds=10).
    When: clock advanced to T=11, get("k") called.
    Then: returns None (key expired).
    """

def test_ttl_not_expired_yet(): ...
    """
    Given: clock at T=0. set("k", "v", ttl_seconds=10).
    When: clock advanced to T=5, get("k") called.
    Then: returns "v".
    """

def test_ttl_remaining(): ...
    """
    Given: clock at T=0. set("k", "v", ttl_seconds=30).
    When: clock advanced to T=10, ttl("k") called.
    Then: returns approximately 20.0.
    """

def test_persistent_key_ttl_returns_minus_one(): ...
    """
    Given: set("k", "v") with no TTL.
    When: ttl("k") called.
    Then: returns -1.
    """

def test_lru_eviction_removes_least_recently_used(): ...
    """
    Given: capacity=3. set("a", 1), set("b", 2), set("c", 3).
    When: get("a") called (marks "a" as recently used), then set("d", 4).
    Then: "b" was evicted (LRU); get("b") returns None; get("a") returns 1; get("d") returns 4.
    """

def test_namespace_isolation(): ...
    """
    Given: set("k", "ns1_val", namespace="ns1") and set("k", "ns2_val", namespace="ns2").
    When: get("k", namespace="ns1") and get("k", namespace="ns2").
    Then: "ns1_val" and "ns2_val" returned independently.
    """

def test_delete_namespace(): ...
    """
    Given: 5 keys in namespace "temp", 3 keys in namespace "persist".
    When: delete_namespace("temp").
    Then: size(namespace="temp")==0; size(namespace="persist")==3.
    """

def test_concurrent_set_get_no_data_races(): ...
    """
    Given: store with capacity=500.
    When: 20 threads each set 50 unique keys concurrently.
    Then: no exceptions raised; size() <= 500; all remaining keys have valid values.
    """

def test_background_sweep_removes_expired(): ...
    """
    Given: sweep_interval=0.1s. set 10 keys with TTL=0.05s.
    When: real time.sleep(0.3) elapsed.
    Then: size() == 0 (sweep cleared all expired keys).
    Note: this test uses real time — mark with @pytest.mark.slow.
    """

def test_overwrite_resets_ttl(): ...
    """
    Given: clock T=0. set("k", "old", ttl_seconds=10).
    When: clock advanced to T=8. set("k", "new", ttl_seconds=30).
    Then: ttl("k") approximately 30.0 (TTL reset, not extended from original).
    """
```

---

## Required Sections in Your Submission

Both solutions must include the following sections as comments or docstrings:

### 1. Design decisions

For each implementation, add a module-level docstring in `building.py` / `store.py` with:

- Why you chose the dispatch/eviction strategy you did.
- One alternative you considered and why you rejected it.
- The concurrency approach and the lock granularity chosen.

### 2. Complexity analysis

Add a `COMPLEXITY.md` file (or a module docstring) covering:

| Operation | Time complexity | Space complexity | Notes |
|---|---|---|---|
| Elevator: `add_stop` | | | |
| Elevator: `next_stop` (SCAN) | | | |
| KV: `get` | | | |
| KV: `set` (with eviction) | | | |
| KV: `_sweep_expired` | | | |
| KV: `delete_namespace` | | | |

### 3. Limitations and future work

One paragraph per problem: what would you change if this needed to be production-ready (persistence, distributed operation, monitoring)?

---

## Evaluation Criteria

| Criterion | Points |
|---|---|
| Elevator: class diagram complete and correct | 5 |
| Elevator: implementation correct (all 10 tests pass) | 20 |
| Elevator: SCAN algorithm correctness | 10 |
| Elevator: concurrency (thread-safe, no busy-wait) | 5 |
| KV Store: class diagram complete and correct | 5 |
| KV Store: implementation correct (all 12 tests pass) | 20 |
| KV Store: LRU eviction correct | 5 |
| KV Store: TTL and background sweep correct | 5 |
| Code quality: type hints, logging, no print | 5 |
| Design decisions section present and substantive | 5 |
| Complexity analysis complete and accurate | 5 |
| Pydantic v2 used for data models where specified | 5 |
| **Total** | **100** |

**Pass gate: 75 / 100**

---

## Submission

Save your work under:

```
phase-7-system-design/week-07-project/
  elevator/
  kv_store/
  tests/
  COMPLEXITY.md   (or inline docstrings)
  pyproject.toml
```

Run the full test suite before submitting:

```bash
cd phase-7-system-design/week-07-project
python -m pytest tests/ -v --tb=short
```

All tests must pass with exit code 0. If any test is marked `@pytest.mark.slow` and skipped in CI, document why in your submission notes.
