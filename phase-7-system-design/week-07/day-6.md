# Phase 7 · Week 07 · Day 6 — 4-Hour LLD Deep Practice

> Objective: Apply the full LLD method end-to-end on two classic problems under timed conditions  ·  Time: ~4 hr  ·  Prereqs: [[day-1]], [[day-2]], [[day-3]], [[day-4]], [[day-5]]

---

## Session structure

| Block | Duration | Problem |
|---|---|---|
| Warm-up | 10 min | Read both specs; pick your design approach |
| Block A | 110 min | Elevator System — full implementation + tests |
| Break | 10 min | Step away from the screen |
| Block B | 110 min | In-memory KV Store — full implementation + tests |
| Wrap-up | 20 min | Self-review: what would you improve? |

**Rules for today:**
- No looking at Day 5 solutions while you implement.
- Write tests first (TDD: red -> green -> refactor).
- Commit after each class is complete (`git commit` or save checkpoint).
- If stuck > 15 min: read the hint for that section only.

---

## Problem A — Elevator System (110 min)

### Full specification

You are building the backend for a multi-elevator building controller.

**Functional requirements:**
1. The building has a configurable number of floors and elevator cars.
2. A passenger can press UP or DOWN on any floor (hall call).
3. A passenger inside a car can press any destination floor (car call).
4. The scheduler assigns a hall call to the most suitable car.
5. Each car moves one floor at a time in its current direction.
6. A car stops at a floor if it has that floor in its stop queue.
7. Doors open when the car stops, then close after a configurable delay.
8. The system should not deadlock or lose requests.

**Non-functional requirements:**
- Thread-safe: multiple entry-point threads may issue hall calls concurrently.
- Unit-testable: each class testable in isolation.
- No real-time simulation required: the `step()` method advances the simulation by one tick.

**Out of scope for this session:**
- Real-time threading loop (use `step()` instead).
- Persistence.
- Emergency stop / fire service mode.

**Entities to model (minimum):**

| Entity | Responsibility |
|---|---|
| `Direction` | Enum: UP, DOWN |
| `ElevatorState` | Enum: IDLE, MOVING_UP, MOVING_DOWN, DOOR_OPEN |
| `ElevatorCar` | Tracks position, state, and stop queue; advances on `step()` |
| `Scheduler` | Assigns hall calls to cars; SCAN or nearest-car algorithm |
| `Building` | Holds cars + scheduler; exposes `hall_call()` and `car_call()` |

---

### Step-by-step approach guide

Work through these in order.  Do NOT skip a step.

**Step 1 — Enumerate states and transitions (10 min)**

Draw the state machine for one elevator car on paper.

- What states exist?
- Which events cause transitions?  (`floor_reached`, `door_timer_expired`, `door_obstruction`)
- What invariant must always hold?  (e.g. "car never moves when state is DOOR_OPEN")

Write these down before touching Python.

**Step 2 — Design the stop queue (15 min)**

An elevator car needs to know: "given I am at floor F going UP, what is my next stop?"

- What data structure holds pending stops?
- How do you insert a new stop efficiently?
- How does the car decide to reverse direction?
- Edge case: what if the queue is empty?

Write `next_stop() -> int | None` and `add_stop(floor: int)` on paper first.

**Step 3 — Implement `ElevatorCar` (25 min)**

Start here.  Get `ElevatorCar` working in isolation before adding `Scheduler` or `Building`.

Write tests first:
```python
def test_car_starts_idle():
    ...

def test_car_adds_stop_and_moves():
    ...

def test_car_reverses_direction_when_stops_exhausted():
    ...

def test_door_opens_when_car_arrives():
    ...
```

**Step 4 — Implement `Scheduler` (20 min)**

The simplest correct scheduler: nearest-car.
- Assign the hall call to the car with the smallest `abs(car.current_floor - request_floor)`.
- Tie-break: prefer IDLE cars.

Test it with 2 cars and 3 simultaneous hall calls.

**Step 5 — Wire `Building` (15 min)**

`Building` is mostly delegation.  Keep it thin.

`hall_call(floor, direction)` -> `scheduler.assign(...)` -> `car.add_stop(floor)`
`car_call(car_id, floor)` -> `car.add_stop(floor)`

Write an integration test: simulate a passenger journey end-to-end using `step()`.

**Step 6 — Add thread safety (15 min)**

Identify which state is shared:
- Each car's stop queue and position.
- The scheduler's car list.

Add a `threading.Lock` per car.  Use `with car.lock:` in `add_stop` and `step`.

Write a concurrent stress test: 10 threads each issue 20 hall calls simultaneously.
Assert no exception and final state is consistent (all requests eventually processed).

**Step 7 — Refactor and self-review (10 min)**

- Are there any god functions (> 20 lines)?  Split them.
- Is state mutation happening outside the class that owns it?  Move it.
- Would you add a `DoorController` state machine?  Note it as a TODO.

---

### Starter skeleton

```python
"""
elevator_system.py — starter skeleton for Day 6 Block A.

Fill in every TODO.  Do NOT look at solutions until Block A time is up.
"""

import logging
import threading
from enum import Enum, auto

log = logging.getLogger(__name__)


class Direction(Enum):
    UP = auto()
    DOWN = auto()


class ElevatorState(Enum):
    IDLE = auto()
    MOVING_UP = auto()
    MOVING_DOWN = auto()
    DOOR_OPEN = auto()


class ElevatorCar:
    def __init__(self, car_id: int, top_floor: int) -> None:
        self.id = car_id
        self.current_floor: int = 0
        self.state: ElevatorState = ElevatorState.IDLE
        self._top_floor = top_floor
        self.lock = threading.Lock()
        # TODO: initialise stop queues (up and down)

    def add_stop(self, floor: int) -> None:
        """Add a floor to the appropriate stop queue."""
        # TODO
        raise NotImplementedError

    def next_stop(self) -> int | None:
        """Return the next floor to visit, or None if idle."""
        # TODO
        raise NotImplementedError

    def step(self) -> None:
        """
        Advance simulation by one tick.

        Tick logic:
        - If DOOR_OPEN: close doors (transition to IDLE or continue moving).
        - If MOVING_UP / MOVING_DOWN: move one floor, check if it is a stop.
        - If IDLE: check if there is a pending stop; start moving if so.
        """
        # TODO
        raise NotImplementedError


class Scheduler:
    def __init__(self, cars: list[ElevatorCar]) -> None:
        self._cars = cars

    def assign(self, floor: int, direction: Direction) -> ElevatorCar:
        """
        Assign a hall call to the best car.

        Implement nearest-car.  Prefer IDLE cars on tie.
        """
        # TODO
        raise NotImplementedError


class Building:
    def __init__(self, num_floors: int, num_cars: int) -> None:
        self.num_floors = num_floors
        self._cars: list[ElevatorCar] = [
            ElevatorCar(car_id=i, top_floor=num_floors - 1) for i in range(num_cars)
        ]
        self._scheduler = Scheduler(self._cars)

    def hall_call(self, floor: int, direction: Direction) -> None:
        """A passenger on `floor` pressed the `direction` button."""
        # TODO
        raise NotImplementedError

    def car_call(self, car_id: int, floor: int) -> None:
        """A passenger inside car `car_id` pressed floor button `floor`."""
        # TODO
        raise NotImplementedError

    def step_all(self) -> None:
        """Advance every car by one tick."""
        for car in self._cars:
            car.step()
```

---

### Test cases to pass (TDD)

Write these tests in `tests/test_elevator.py` before implementing.

```python
"""
tests/test_elevator.py

Run with: python -m pytest tests/test_elevator.py -v
All tests must be green before moving to Block B.
"""

import threading
import pytest
from elevator_system import Building, Direction, ElevatorState, ElevatorCar


# --- ElevatorCar unit tests ---

def test_new_car_is_idle_at_floor_zero():
    car = ElevatorCar(car_id=0, top_floor=9)
    assert car.current_floor == 0
    assert car.state == ElevatorState.IDLE


def test_car_moves_up_after_stop_added_above():
    car = ElevatorCar(car_id=0, top_floor=9)
    car.add_stop(3)
    car.step()
    assert car.current_floor == 1
    assert car.state == ElevatorState.MOVING_UP


def test_car_opens_door_when_reaching_stop():
    car = ElevatorCar(car_id=0, top_floor=9)
    car.add_stop(1)
    car.step()  # move to floor 1
    assert car.current_floor == 1
    assert car.state == ElevatorState.DOOR_OPEN


def test_car_continues_after_door_closes():
    car = ElevatorCar(car_id=0, top_floor=9)
    car.add_stop(1)
    car.add_stop(3)
    car.step()  # -> floor 1, DOOR_OPEN
    car.step()  # -> close door, resume moving
    car.step()  # -> floor 2, MOVING_UP
    assert car.current_floor == 2


def test_car_becomes_idle_when_no_more_stops():
    car = ElevatorCar(car_id=0, top_floor=9)
    car.add_stop(1)
    car.step()  # -> floor 1, DOOR_OPEN
    car.step()  # -> close door, IDLE (no more stops)
    assert car.state == ElevatorState.IDLE


def test_car_reverses_for_down_stop():
    car = ElevatorCar(car_id=0, top_floor=9)
    car.add_stop(5)   # up
    for _ in range(5):
        car.step()    # travel to 5, open door
    car.step()        # close door
    car.add_stop(2)   # down request
    car.step()        # start moving down
    assert car.state == ElevatorState.MOVING_DOWN


# --- Scheduler unit tests ---

def test_scheduler_assigns_nearest_car():
    from elevator_system import Scheduler
    car_a = ElevatorCar(car_id=0, top_floor=9)  # floor 0
    car_b = ElevatorCar(car_id=1, top_floor=9)  # floor 0
    # Manually position car_b at floor 5 (simulate prior movement)
    car_b.current_floor = 5
    scheduler = Scheduler([car_a, car_b])
    assigned = scheduler.assign(floor=4, direction=Direction.UP)
    assert assigned is car_b  # car_b at floor 5 is nearer to floor 4 than car_a at floor 0


# --- Building integration test ---

def test_full_passenger_journey():
    building = Building(num_floors=10, num_cars=1)
    building.hall_call(floor=5, direction=Direction.UP)
    # Step until car reaches floor 5 (max 20 steps)
    for _ in range(20):
        building.step_all()
        if building._cars[0].current_floor == 5:
            break
    assert building._cars[0].current_floor == 5


# --- Concurrency test ---

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
```

---

### Hints (read only if stuck > 15 min on a specific step)

**Hint A1 — stop queue structure:**
Use two `SortedList` objects from `sortedcontainers`: one for up-stops, one for down-stops.
When going UP, always service the smallest up-stop first.
When going DOWN, always service the largest down-stop first.
When the current-direction queue is empty, check the other queue and reverse.

**Hint A2 — step() structure:**
```
if state == DOOR_OPEN:
    state = IDLE (or MOVING if more stops)
elif state == MOVING_UP:
    current_floor += 1
    if current_floor in up_stops:
        up_stops.remove(current_floor)
        state = DOOR_OPEN
elif state == MOVING_DOWN:
    current_floor -= 1
    if current_floor in down_stops:
        down_stops.remove(current_floor)
        state = DOOR_OPEN
elif state == IDLE:
    nxt = next_stop()
    if nxt is not None:
        state = MOVING_UP if nxt > current_floor else MOVING_DOWN
```

**Hint A3 — thread safety:**
Acquire `car.lock` at the start of both `add_stop()` and `step()`.
Do NOT hold the lock across both methods at the same time from the outside — that causes
deadlock if `step_all()` holds a car lock while another thread calls `hall_call()`.

---

## Break — 10 minutes

Stand up.  Review your implementation.  Note any rough edges.

---

## Problem B — In-Memory Key-Value Store (110 min)

### Full specification

Build a production-quality, thread-safe in-memory key-value store.

**Functional requirements:**
1. `put(key, value)` — insert or update.
2. `get(key) -> value | None` — fetch; return `None` for missing key.
3. `delete(key) -> bool` — remove; return `True` if key existed.
4. `keys() -> list[str]` — snapshot of all keys at call time.
5. TTL support: `put(key, value, ttl_seconds=N)` — key expires after N seconds.
6. Lazy expiry: expired keys are removed on the next `get()` / `keys()` call that touches them.
7. Background eviction: a daemon thread sweeps for expired keys every `eviction_interval` seconds.
8. Namespace support: keys are namespaced as `"namespace:key"` strings; `get_namespace(ns)`
   returns a view that prepends the namespace automatically.

**Non-functional requirements:**
- Thread-safe for all public methods.
- Typed (Python 3.12+, full type hints).
- `Namespace` view must NOT copy data; it must delegate to the parent store.

**Out of scope:**
- Persistence to disk.
- Eviction policies (LRU / LFU).
- Replication.

**Entities to model:**

| Entity | Responsibility |
|---|---|
| `Entry` | Holds value + optional expiry timestamp |
| `KVStore` | Core store: put/get/delete/keys + TTL + background eviction |
| `Namespace` | Thin wrapper that prefixes all keys |

---

### Step-by-step approach guide

**Step 1 — Model `Entry` (10 min)**

An entry holds a value and an optional expiry timestamp (`float`, from `time.monotonic()`).

- When is `Entry.is_expired()` true?
- If `ttl_seconds` is `None`, the entry never expires.
- Write `Entry` as a `dataclass`.

**Step 2 — Core store without TTL or threading (20 min)**

Implement `put`, `get`, `delete`, `keys` using a plain `dict[str, Entry]`.
Write tests for each method.  Get all four green before adding complexity.

**Step 3 — Add TTL (20 min)**

- `put` accepts optional `ttl_seconds: float | None = None`.
- `get` calls `entry.is_expired()` and deletes + returns `None` if expired (lazy eviction).
- `keys` filters out expired entries.
- Write tests: put with ttl=0.1, sleep 0.2, assert get returns None.

**Step 4 — Add thread safety (15 min)**

Add a `threading.RLock` to the store.  Every method that reads or writes `_data` acquires it.
Use `RLock` (not `Lock`) because the background eviction thread calls `delete()` internally,
and `delete()` is also called from `get()` during lazy expiry — both paths acquire the same lock.

Write a concurrent stress test: 5 writer threads, 5 reader threads, 200 ops each.
Assert no exception; final store size is internally consistent.

**Step 5 — Background eviction (20 min)**

Start a daemon thread in `__init__` that:
1. Sleeps for `eviction_interval` seconds.
2. Acquires the lock.
3. Deletes all expired entries.
4. Releases the lock.
5. Loops.

Use `threading.Event` to allow clean shutdown: `store.stop()` sets the event; the daemon
loop checks it and exits.

Test: put 5 keys with ttl=0.1; sleep 0.2 + eviction_interval; assert keys() is empty.
Note: this test has a timing dependency — account for CI slowness with generous sleep.

**Step 6 — Implement `Namespace` (15 min)**

```python
ns = store.namespace("users")
ns.put("alice", {"age": 30})     # stored as "users:alice"
ns.get("alice")                  # fetches "users:alice"
```

`Namespace` holds a reference to the parent `KVStore` and a prefix string.
`put`, `get`, `delete` all call `self._store.put(f"{self._prefix}:{key}", ...)`.
`keys()` returns keys with the prefix stripped.

**Step 7 — Self-review (10 min)**

- Is `Entry` immutable?  Should it be?
- What happens if two threads call `get()` on an expired key simultaneously?
  (Both check `is_expired()`, both call `delete()` — is that safe with RLock?)
- Would you change the design if TTL precision needed to be millisecond-accurate?

---

### Starter skeleton

```python
"""
kv_store.py — starter skeleton for Day 6 Block B.

Fill in every TODO.  Tests are in tests/test_kv_store.py.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Generic, TypeVar

log = logging.getLogger(__name__)

V = TypeVar("V")


@dataclass
class Entry(Generic[V]):
    value: V
    expires_at: float | None = field(default=None)  # monotonic timestamp or None

    def is_expired(self) -> bool:
        """Return True if this entry has a TTL that has elapsed."""
        # TODO
        raise NotImplementedError


class KVStore(Generic[V]):
    """
    Thread-safe in-memory key-value store with TTL and background eviction.
    """

    def __init__(self, eviction_interval: float = 5.0) -> None:
        self._data: dict[str, Entry[V]] = {}
        self._lock: threading.RLock = threading.RLock()
        self._stop_event: threading.Event = threading.Event()
        self._eviction_interval = eviction_interval
        self._eviction_thread = threading.Thread(
            target=self._evict_loop,
            name="kv-eviction",
            daemon=True,
        )
        self._eviction_thread.start()

    # --- public API ---

    def put(self, key: str, value: V, ttl_seconds: float | None = None) -> None:
        """Insert or update key.  Optional TTL in seconds."""
        # TODO
        raise NotImplementedError

    def get(self, key: str) -> V | None:
        """
        Return value or None.
        Lazily evict the entry if it has expired.
        """
        # TODO
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        """Remove key.  Return True if it existed (and was not already expired)."""
        # TODO
        raise NotImplementedError

    def keys(self) -> list[str]:
        """Snapshot of non-expired keys."""
        # TODO
        raise NotImplementedError

    def size(self) -> int:
        """Number of non-expired keys."""
        return len(self.keys())

    def stop(self) -> None:
        """Signal the background eviction thread to stop."""
        self._stop_event.set()

    def namespace(self, prefix: str) -> "Namespace[V]":
        return Namespace(store=self, prefix=prefix)

    # --- internal ---

    def _evict_loop(self) -> None:
        """Background daemon: remove expired entries periodically."""
        while not self._stop_event.wait(timeout=self._eviction_interval):
            # TODO: acquire lock, delete expired entries, release lock
            pass

    def _delete_expired(self) -> int:
        """
        Delete all expired entries.  Caller MUST hold self._lock.
        Return the number of evicted entries.
        """
        # TODO
        raise NotImplementedError


class Namespace(Generic[V]):
    """
    A namespaced view of a KVStore.

    All keys are stored as f"{prefix}:{key}" in the parent store.
    This class never copies data; it delegates every operation.
    """

    def __init__(self, store: KVStore[V], prefix: str) -> None:
        self._store = store
        self._prefix = prefix

    def _ns(self, key: str) -> str:
        return f"{self._prefix}:{key}"

    def put(self, key: str, value: V, ttl_seconds: float | None = None) -> None:
        # TODO
        raise NotImplementedError

    def get(self, key: str) -> V | None:
        # TODO
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        # TODO
        raise NotImplementedError

    def keys(self) -> list[str]:
        """Return keys within this namespace, with the prefix stripped."""
        # TODO
        raise NotImplementedError
```

---

### Test cases to pass (TDD)

```python
"""
tests/test_kv_store.py

Run with: python -m pytest tests/test_kv_store.py -v
All tests must be green before finishing Block B.
"""

import threading
import time
import pytest
from kv_store import Entry, KVStore, Namespace


# --- Entry tests ---

def test_entry_without_ttl_never_expires():
    e: Entry[int] = Entry(value=42)
    assert not e.is_expired()


def test_entry_with_future_expiry_not_expired():
    e: Entry[int] = Entry(value=42, expires_at=time.monotonic() + 100)
    assert not e.is_expired()


def test_entry_with_past_expiry_is_expired():
    e: Entry[int] = Entry(value=42, expires_at=time.monotonic() - 1)
    assert e.is_expired()


# --- KVStore basic tests ---

def test_put_and_get():
    store: KVStore[str] = KVStore()
    store.put("name", "alice")
    assert store.get("name") == "alice"
    store.stop()


def test_get_missing_returns_none():
    store: KVStore[str] = KVStore()
    assert store.get("missing") is None
    store.stop()


def test_delete_returns_true_for_existing_key():
    store: KVStore[int] = KVStore()
    store.put("x", 1)
    assert store.delete("x") is True
    assert store.get("x") is None
    store.stop()


def test_delete_returns_false_for_missing_key():
    store: KVStore[int] = KVStore()
    assert store.delete("nonexistent") is False
    store.stop()


def test_keys_returns_all_keys():
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


# --- TTL tests ---

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
    assert "ephemeral" not in store.keys()
    assert "persistent" in store.keys()
    store.stop()


def test_background_eviction_clears_expired_keys():
    store: KVStore[str] = KVStore(eviction_interval=0.1)
    for i in range(5):
        store.put(f"k{i}", "v", ttl_seconds=0.05)
    time.sleep(0.5)  # wait for eviction cycle
    assert store.size() == 0
    store.stop()


# --- Thread safety tests ---

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


# --- Namespace tests ---

def test_namespace_put_and_get():
    store: KVStore[str] = KVStore()
    ns = store.namespace("users")
    ns.put("alice", "data")
    assert ns.get("alice") == "data"
    # Verify it's stored with prefix in the parent store
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
```

---

### Hints (read only if stuck > 15 min on a specific step)

**Hint B1 — Entry.is_expired():**
```python
def is_expired(self) -> bool:
    if self.expires_at is None:
        return False
    return time.monotonic() >= self.expires_at
```

**Hint B2 — TTL in put():**
```python
expires_at = time.monotonic() + ttl_seconds if ttl_seconds is not None else None
```

**Hint B3 — Why RLock, not Lock?**
`get()` may call `self.delete()` (lazy eviction); `_evict_loop` also calls `self.delete()`.
If you use a plain `Lock`, calling `delete()` from inside `get()` while the lock is held
causes a deadlock.  `RLock` allows the same thread to re-acquire safely.

**Hint B4 — `_delete_expired` without creating a new dict:**
```python
expired_keys = [k for k, v in self._data.items() if v.is_expired()]
for k in expired_keys:
    del self._data[k]
return len(expired_keys)
```

**Hint B5 — Namespace.keys() implementation:**
```python
prefix = f"{self._prefix}:"
return [k[len(prefix):] for k in self._store.keys() if k.startswith(prefix)]
```

---

## Wrap-up self-review (20 min)

Answer these questions in your notes (or a comment in the file):

1. **Elevator:** What would break if two threads called `car.step()` simultaneously without the lock?

2. **Elevator:** How would you upgrade the nearest-car scheduler to the SCAN algorithm?
   What data does the scheduler need from the car to make the decision?

3. **KV Store:** The background eviction thread calls `_delete_expired()` every N seconds.
   What is the maximum time a key can survive past its TTL?  How would you reduce it?

4. **KV Store:** You are asked to add LRU eviction (evict least-recently-used key when
   store size exceeds a cap).  What data structure would you add to `Entry` or `KVStore`?

5. **Both systems:** Which design decisions would you change if you had to scale to 100
   concurrent threads instead of 10?

---

## Self-check

- [ ] All 15 elevator tests pass
- [ ] All 17 KV store tests pass
- [ ] I implemented both without reading the solutions
- [ ] I can explain the SCAN scheduler algorithm and why it beats nearest-car under load
- [ ] I understand why RLock is needed in KVStore but not in ElevatorCar
- [ ] I can describe lazy vs background eviction trade-offs
- [ ] I identified at least one design improvement during the wrap-up

---

## Supplementary

- [sortedcontainers docs](https://grantjenks.com/docs/sortedcontainers/)
- [Python threading.RLock docs](https://docs.python.org/3/library/threading.html#rlock-objects)
- [Python threading.Event docs](https://docs.python.org/3/library/threading.html#event-objects)
- [SCAN disk scheduling algorithm (applies to elevators)](https://en.wikipedia.org/wiki/Elevator_algorithm)
- [Grokking the Object-Oriented Design Interview](https://www.educative.io/courses/grokking-the-object-oriented-design-interview)
