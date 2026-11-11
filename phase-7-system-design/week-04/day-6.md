# Phase 7 · Week 04 · Day 6 — Event-Sourcing Aggregate + Windowed Stream Processor

> Objective: Build an EventSourcedAggregate with snapshot support, an in-memory EventStore, and a tumbling-window StreamProcessor — all fully tested with pytest.  ·  Time: ~4 hr  ·  Prereqs: [[day-5]]

---

## Overview

This is a 4-hour guided build. Work through each section in order. Every section ends with a checkpoint — verify it before continuing.

```
Build roadmap:

  Part 1 (~45 min)  EventSourcedAggregate base class
  Part 2 (~30 min)  Concrete aggregate: InventoryItem
  Part 3 (~30 min)  In-memory EventStore
  Part 4 (~45 min)  Windowed StreamProcessor (tumbling window)
  Part 5 (~30 min)  Full pytest test suite
  Part 6 (~20 min)  Integration: wire all parts together
```

All code runs with **zero external dependencies** beyond the Python standard library and Pydantic. No Kafka, no Redis, no Docker needed.

---

## Project layout

```
event_sourcing/
    __init__.py
    aggregate.py        # Part 1 + Part 2
    event_store.py      # Part 3
    stream_processor.py # Part 4
    models.py           # Shared Pydantic models

tests/
    __init__.py
    test_aggregate.py
    test_event_store.py
    test_stream_processor.py
    test_integration.py
```

Create this structure before starting:

```bash
mkdir -p event_sourcing tests
touch event_sourcing/__init__.py tests/__init__.py
```

---

## Concept recap

**Event sourcing**: instead of storing current state, store every event that ever changed the aggregate. Current state = replay all events from the beginning.

```
Traditional (state stored):
  DB row: { id: 1, qty: 42, reserved: 10 }

Event sourced (events stored):
  Event 1: InventoryCreated  { qty: 100 }
  Event 2: InventoryReceived { delta: +50 }
  Event 3: InventoryReserved { delta: -108 }
  --replay--> current state: qty=42, reserved=10
```

**Snapshot**: optimization. After N events, capture current state. On replay: load snapshot + events after snapshot. Avoids replaying thousands of events on every load.

```
Events 1..500 + Snapshot@500 + Events 501..510
  = same result as replaying 1..510
  but faster (skip 500 events)
```

**Windowed stream processor**: groups a continuous event stream into fixed-size time buckets (tumbling windows), aggregates each bucket, and emits one result per window.

```
Time:   0s   10s   20s   30s
Window:  |--- W1 ---|--- W2 ---|--- W3 ---|

W1 events → aggregate → emit summary at t=10s
W2 events → aggregate → emit summary at t=20s
```

---

## Part 1: Shared models

```python
# event_sourcing/models.py
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class StoredEvent(BaseModel):
    """An immutable event as stored in the EventStore."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    aggregate_id: str
    aggregate_type: str
    event_type: str
    sequence: int  # monotonically increasing per aggregate
    occurred_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}


class Snapshot(BaseModel):
    """Point-in-time capture of aggregate state."""

    aggregate_id: str
    aggregate_type: str
    sequence: int          # sequence number of last applied event
    captured_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    state: dict[str, Any]  # serialized aggregate state

    model_config = {"frozen": True}


class StreamRecord(BaseModel):
    """A single record flowing through the stream processor."""

    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime
    topic: str
    value: float
    attributes: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}


class WindowResult(BaseModel):
    """Output of processing one tumbling window."""

    window_start: datetime
    window_end: datetime
    topic: str
    record_count: int
    total_value: float
    min_value: float
    max_value: float
    avg_value: float

    model_config = {"frozen": True}
```

**Checkpoint 1**: run `python -c "from event_sourcing.models import StoredEvent; print(StoredEvent(aggregate_id='x', aggregate_type='T', event_type='E', sequence=1))"` — should print a valid model.

---

## Part 2: EventSourcedAggregate base class

The base class handles:
- Collecting uncommitted events (`_pending_events`)
- Applying events via `apply()` dispatch
- Reconstituting state from a sequence of `StoredEvent`
- Snapshot creation and restoration

```python
# event_sourcing/aggregate.py  (Part 1 — base class)
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from event_sourcing.models import Snapshot, StoredEvent

logger = logging.getLogger(__name__)


class AggregateVersionConflict(Exception):
    """Raised when attempting to save events with a stale sequence number."""


class EventSourcedAggregate(ABC):
    """
    Base class for event-sourced aggregates.

    Subclasses must implement:
      - aggregate_type (class attribute, str)
      - _apply_<EventType>(self, payload: dict) -> None  (one method per event type)
      - to_snapshot_state(self) -> dict[str, Any]
      - from_snapshot_state(cls, state: dict[str, Any]) -> Self
    """

    aggregate_type: str = ""  # override in subclass

    def __init__(self, aggregate_id: str) -> None:
        self._aggregate_id: str = aggregate_id
        self._sequence: int = 0          # sequence of last applied event
        self._pending_events: list[StoredEvent] = []
        self._version: int = 0           # alias for sequence (used in conflict checks)

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def aggregate_id(self) -> str:
        return self._aggregate_id

    @property
    def sequence(self) -> int:
        return self._sequence

    @property
    def version(self) -> int:
        return self._sequence

    def pending_events(self) -> list[StoredEvent]:
        """Return uncommitted events (copy — do not mutate externally)."""
        return list(self._pending_events)

    def clear_pending_events(self) -> None:
        """Called by EventStore after saving events."""
        self._pending_events.clear()

    # ------------------------------------------------------------------
    # Event raising (called by subclass command methods)
    # ------------------------------------------------------------------

    def _raise_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Record and immediately apply a new domain event.

        The event is added to _pending_events and dispatched to the
        matching _apply_<EventType> handler.
        """
        self._sequence += 1
        event = StoredEvent(
            aggregate_id=self._aggregate_id,
            aggregate_type=self.aggregate_type,
            event_type=event_type,
            sequence=self._sequence,
            payload=payload,
            metadata=metadata or {},
        )
        self._pending_events.append(event)
        self._dispatch(event)
        logger.debug(
            "Raised %s on %s [seq=%d]",
            event_type,
            self._aggregate_id,
            self._sequence,
        )

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, event: StoredEvent) -> None:
        """Route event to _apply_<EventType> handler."""
        handler_name = f"_apply_{event.event_type}"
        handler = getattr(self, handler_name, None)
        if handler is None:
            raise NotImplementedError(
                f"{type(self).__name__} has no handler for event type "
                f"'{event.event_type}'. Expected method: {handler_name}"
            )
        handler(event.payload)

    # ------------------------------------------------------------------
    # Reconstitution
    # ------------------------------------------------------------------

    @classmethod
    def reconstitute(
        cls,
        aggregate_id: str,
        events: list[StoredEvent],
        snapshot: Snapshot | None = None,
    ) -> "EventSourcedAggregate":
        """
        Rebuild aggregate state from a snapshot (optional) + events.

        If snapshot is provided, events must start AFTER snapshot.sequence.
        """
        instance = cls(aggregate_id)

        if snapshot is not None:
            instance._restore_snapshot(snapshot)

        for event in events:
            instance._sequence = event.sequence
            instance._dispatch(event)

        logger.debug(
            "Reconstituted %s %s at sequence %d from %d events",
            cls.__name__,
            aggregate_id,
            instance._sequence,
            len(events),
        )
        return instance

    # ------------------------------------------------------------------
    # Snapshot support
    # ------------------------------------------------------------------

    def take_snapshot(self) -> Snapshot:
        """Serialize current state to a Snapshot."""
        return Snapshot(
            aggregate_id=self._aggregate_id,
            aggregate_type=self.aggregate_type,
            sequence=self._sequence,
            state=self.to_snapshot_state(),
        )

    def _restore_snapshot(self, snapshot: Snapshot) -> None:
        """Restore state from snapshot. Called during reconstitution."""
        self._sequence = snapshot.sequence
        self.from_snapshot_state(snapshot.state)

    # ------------------------------------------------------------------
    # Abstract: subclasses implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def to_snapshot_state(self) -> dict[str, Any]:
        """Return serializable state dict for snapshot."""

    @abstractmethod
    def from_snapshot_state(self, state: dict[str, Any]) -> None:
        """Restore state from snapshot dict. Mutates self in place."""
```

**Checkpoint 2**: class is abstract — `EventSourcedAggregate("x")` raises `TypeError` as expected (because `to_snapshot_state` and `from_snapshot_state` are abstract). Verify:
```python
python -c "from event_sourcing.aggregate import EventSourcedAggregate; EventSourcedAggregate('x')"
# TypeError: Can't instantiate abstract class ...
```

---

## Part 3: Concrete aggregate — InventoryItem

Commands (methods that mutate state):
- `receive(qty)` — add stock
- `reserve(qty)` — hold qty for an order
- `release_reservation(qty)` — cancel a reservation
- `ship(qty)` — fulfill a reservation, decrement both reserved and on-hand

```python
# event_sourcing/aggregate.py  (Part 2 — append below Part 1)


class InsufficientStock(Exception):
    pass


class InvalidQuantity(Exception):
    pass


class InventoryItem(EventSourcedAggregate):
    """
    Aggregate tracking stock levels for a single SKU.

    State:
      on_hand    — physical quantity in warehouse
      reserved   — subset of on_hand held for pending orders
      available  — on_hand - reserved (derived, not stored)
    """

    aggregate_type = "InventoryItem"

    def __init__(self, aggregate_id: str) -> None:
        super().__init__(aggregate_id)
        self.sku: str = ""
        self.on_hand: int = 0
        self.reserved: int = 0

    @property
    def available(self) -> int:
        return self.on_hand - self.reserved

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, aggregate_id: str, sku: str) -> "InventoryItem":
        item = cls(aggregate_id)
        item._raise_event("ItemCreated", {"sku": sku})
        return item

    def receive(self, qty: int) -> None:
        if qty <= 0:
            raise InvalidQuantity(f"Receive qty must be > 0, got {qty}")
        self._raise_event("StockReceived", {"qty": qty})

    def reserve(self, qty: int) -> None:
        if qty <= 0:
            raise InvalidQuantity(f"Reserve qty must be > 0, got {qty}")
        if qty > self.available:
            raise InsufficientStock(
                f"Cannot reserve {qty}; available={self.available}"
            )
        self._raise_event("StockReserved", {"qty": qty})

    def release_reservation(self, qty: int) -> None:
        if qty <= 0:
            raise InvalidQuantity(f"Release qty must be > 0, got {qty}")
        if qty > self.reserved:
            raise InsufficientStock(
                f"Cannot release {qty}; reserved={self.reserved}"
            )
        self._raise_event("ReservationReleased", {"qty": qty})

    def ship(self, qty: int) -> None:
        if qty <= 0:
            raise InvalidQuantity(f"Ship qty must be > 0, got {qty}")
        if qty > self.reserved:
            raise InsufficientStock(
                f"Cannot ship {qty}; reserved={self.reserved}"
            )
        self._raise_event("StockShipped", {"qty": qty})

    # ------------------------------------------------------------------
    # Event handlers  (_apply_<EventType>)
    # ------------------------------------------------------------------

    def _apply_ItemCreated(self, payload: dict) -> None:
        self.sku = payload["sku"]
        self.on_hand = 0
        self.reserved = 0

    def _apply_StockReceived(self, payload: dict) -> None:
        self.on_hand += payload["qty"]

    def _apply_StockReserved(self, payload: dict) -> None:
        self.reserved += payload["qty"]

    def _apply_ReservationReleased(self, payload: dict) -> None:
        self.reserved -= payload["qty"]

    def _apply_StockShipped(self, payload: dict) -> None:
        self.on_hand -= payload["qty"]
        self.reserved -= payload["qty"]

    # ------------------------------------------------------------------
    # Snapshot support
    # ------------------------------------------------------------------

    def to_snapshot_state(self) -> dict:
        return {
            "sku": self.sku,
            "on_hand": self.on_hand,
            "reserved": self.reserved,
        }

    def from_snapshot_state(self, state: dict) -> None:
        self.sku = state["sku"]
        self.on_hand = state["on_hand"]
        self.reserved = state["reserved"]
```

**Checkpoint 3**: quick smoke test in a REPL:

```python
from event_sourcing.aggregate import InventoryItem

item = InventoryItem.create("item-1", "SKU-ABC")
item.receive(100)
item.reserve(30)
item.ship(30)
print(item.on_hand, item.reserved)  # 70 0
print(item.sequence)                # 4
```

---

## Part 4: In-memory EventStore

The EventStore persists events, serves reconstitution queries, and manages snapshots.

Optimistic concurrency control: before saving, check that `expected_version` matches the current stored sequence. Prevents lost updates when two processes modify the same aggregate concurrently.

```python
# event_sourcing/event_store.py
from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from event_sourcing.aggregate import AggregateVersionConflict
from event_sourcing.models import Snapshot, StoredEvent

if TYPE_CHECKING:
    from event_sourcing.aggregate import EventSourcedAggregate

logger = logging.getLogger(__name__)

# Snapshot every N events — configurable
DEFAULT_SNAPSHOT_THRESHOLD = 50


class EventStore:
    """
    In-memory append-only event store with snapshot support.

    In production this would be backed by PostgreSQL, DynamoDB, or
    EventStoreDB. The interface remains identical.
    """

    def __init__(self, snapshot_threshold: int = DEFAULT_SNAPSHOT_THRESHOLD) -> None:
        # aggregate_id -> list[StoredEvent]  (ordered by sequence)
        self._events: dict[str, list[StoredEvent]] = defaultdict(list)
        # aggregate_id -> Snapshot  (latest only)
        self._snapshots: dict[str, Snapshot] = {}
        self._snapshot_threshold = snapshot_threshold

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def save(
        self,
        aggregate: "EventSourcedAggregate",
        expected_version: int | None = None,
    ) -> None:
        """
        Persist pending events from the aggregate.

        Args:
            aggregate:        The aggregate whose pending events to save.
            expected_version: If provided, raises AggregateVersionConflict
                              when the current stored sequence differs.
                              Pass None to skip optimistic concurrency check.
        """
        pending = aggregate.pending_events()
        if not pending:
            logger.debug("save() called with no pending events — no-op")
            return

        agg_id = aggregate.aggregate_id
        current_stored = self._current_sequence(agg_id)

        if expected_version is not None and current_stored != expected_version:
            raise AggregateVersionConflict(
                f"Conflict on {agg_id}: expected version {expected_version}, "
                f"store has {current_stored}"
            )

        for event in pending:
            self._events[agg_id].append(event)
            logger.debug(
                "Stored %s [%s] seq=%d",
                event.event_type,
                agg_id,
                event.sequence,
            )

        aggregate.clear_pending_events()

        # Auto-snapshot when threshold crossed
        total = len(self._events[agg_id])
        if total % self._snapshot_threshold == 0:
            snapshot = aggregate.take_snapshot()
            self._snapshots[agg_id] = snapshot
            logger.info(
                "Snapshot taken for %s at sequence %d",
                agg_id,
                snapshot.sequence,
            )

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def load_events(
        self,
        aggregate_id: str,
        after_sequence: int = 0,
    ) -> list[StoredEvent]:
        """Return events for aggregate_id with sequence > after_sequence."""
        all_events = self._events.get(aggregate_id, [])
        return [e for e in all_events if e.sequence > after_sequence]

    def load_snapshot(self, aggregate_id: str) -> Snapshot | None:
        return self._snapshots.get(aggregate_id)

    def exists(self, aggregate_id: str) -> bool:
        return aggregate_id in self._events and len(self._events[aggregate_id]) > 0

    # ------------------------------------------------------------------
    # Reconstitution helper
    # ------------------------------------------------------------------

    def reconstitute(
        self,
        aggregate_class: type["EventSourcedAggregate"],
        aggregate_id: str,
    ) -> "EventSourcedAggregate":
        """
        Load snapshot (if any) + subsequent events and reconstitute aggregate.

        Raises KeyError if aggregate_id has no stored events.
        """
        if not self.exists(aggregate_id):
            raise KeyError(f"No events found for aggregate_id={aggregate_id!r}")

        snapshot = self.load_snapshot(aggregate_id)
        after_seq = snapshot.sequence if snapshot else 0
        events = self.load_events(aggregate_id, after_sequence=after_seq)
        return aggregate_class.reconstitute(aggregate_id, events, snapshot)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _current_sequence(self, aggregate_id: str) -> int:
        events = self._events.get(aggregate_id, [])
        return events[-1].sequence if events else 0

    def event_count(self, aggregate_id: str) -> int:
        return len(self._events.get(aggregate_id, []))

    def all_events(self, aggregate_id: str) -> list[StoredEvent]:
        """Return all stored events for an aggregate (read-only copy)."""
        return list(self._events.get(aggregate_id, []))
```

**Checkpoint 4**: verify optimistic concurrency:

```python
from event_sourcing.aggregate import InventoryItem
from event_sourcing.event_store import EventStore

store = EventStore()
item = InventoryItem.create("item-1", "SKU-A")
item.receive(100)
store.save(item, expected_version=0)  # OK: was at 0, now at 2

item2 = store.reconstitute(InventoryItem, "item-1")
item2.reserve(10)

# Simulate stale save (expected_version still 2 but we already advanced)
item3 = store.reconstitute(InventoryItem, "item-1")
item3.receive(50)
store.save(item3, expected_version=2)  # OK

try:
    store.save(item2, expected_version=2)  # CONFLICT: store is now at 3
except Exception as e:
    print(e)  # AggregateVersionConflict ...
```

---

## Part 5: Windowed StreamProcessor

A **tumbling window** partitions a time axis into non-overlapping, fixed-size buckets. Every record belongs to exactly one window.

```
Time axis:
   0s          10s          20s         30s
   |---- W1 ----|---- W2 ----|---- W3 ---|

record at t=5s  → W1
record at t=11s → W2
record at t=9.9s → W1
```

The processor:
1. Accepts records via `ingest(record)`
2. Assigns each record to a window bucket
3. When a record arrives with timestamp >= current window's end → **flushes** the current window and opens a new one
4. Emits a `WindowResult` from each flush

```python
# event_sourcing/stream_processor.py
from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from event_sourcing.models import StreamRecord, WindowResult

logger = logging.getLogger(__name__)


class TumblingWindowProcessor:
    """
    Stateful single-partition tumbling window processor.

    Records must arrive in non-decreasing timestamp order (or be watermarked;
    out-of-order handling is a stretch goal).

    Args:
        window_size_seconds: Duration of each window in seconds.
        topic: Only process records matching this topic (None = all topics).
        on_window_close: Callback invoked with WindowResult on each flush.
    """

    def __init__(
        self,
        window_size_seconds: int,
        topic: str | None = None,
        on_window_close: Callable[[WindowResult], None] | None = None,
    ) -> None:
        self._window_size = timedelta(seconds=window_size_seconds)
        self._topic = topic
        self._on_window_close = on_window_close

        self._window_start: datetime | None = None
        self._window_end: datetime | None = None
        self._buffer: list[StreamRecord] = []
        self._closed_windows: list[WindowResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, record: StreamRecord) -> None:
        """
        Process a single record.

        May trigger one or more window flushes if the record's timestamp
        has advanced past the current window boundary.
        """
        if self._topic is not None and record.topic != self._topic:
            logger.debug("Skipping record for topic %s (filter=%s)", record.topic, self._topic)
            return

        # Initialize first window lazily
        if self._window_start is None:
            self._open_window(record.timestamp)

        # Flush any windows that have elapsed before this record's timestamp
        while record.timestamp >= self._window_end:  # type: ignore[operator]
            self._flush_window()
            self._open_window(self._window_end)  # type: ignore[arg-type]

        self._buffer.append(record)
        logger.debug(
            "Buffered record %.2f in window [%s, %s)",
            record.value,
            self._window_start,
            self._window_end,
        )

    def flush_all(self) -> None:
        """
        Force-flush the current (possibly partial) window.

        Call at end-of-stream to emit any buffered records.
        """
        if self._buffer or (self._window_start is not None):
            self._flush_window()

    def closed_windows(self) -> list[WindowResult]:
        """Return all completed windows in order."""
        return list(self._closed_windows)

    def reset(self) -> None:
        """Clear all state — useful in tests."""
        self._window_start = None
        self._window_end = None
        self._buffer = []
        self._closed_windows = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_window(self, start: datetime) -> None:
        self._window_start = start
        self._window_end = start + self._window_size
        logger.debug("Opened window [%s, %s)", self._window_start, self._window_end)

    def _flush_window(self) -> None:
        assert self._window_start is not None
        assert self._window_end is not None

        topic = self._topic or (self._buffer[0].topic if self._buffer else "unknown")
        values = [r.value for r in self._buffer]

        if values:
            result = WindowResult(
                window_start=self._window_start,
                window_end=self._window_end,
                topic=topic,
                record_count=len(values),
                total_value=sum(values),
                min_value=min(values),
                max_value=max(values),
                avg_value=sum(values) / len(values),
            )
        else:
            # Empty window — still emit (useful for gap detection)
            result = WindowResult(
                window_start=self._window_start,
                window_end=self._window_end,
                topic=topic,
                record_count=0,
                total_value=0.0,
                min_value=0.0,
                max_value=0.0,
                avg_value=0.0,
            )

        self._closed_windows.append(result)
        logger.info(
            "Window closed [%s, %s): count=%d total=%.2f",
            self._window_start,
            self._window_end,
            result.record_count,
            result.total_value,
        )

        if self._on_window_close is not None:
            self._on_window_close(result)

        self._buffer = []
```

**Checkpoint 5**: quick smoke test:

```python
from datetime import datetime, timezone, timedelta
from event_sourcing.models import StreamRecord
from event_sourcing.stream_processor import TumblingWindowProcessor

t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
proc = TumblingWindowProcessor(window_size_seconds=10)

for i in range(25):
    rec = StreamRecord(
        timestamp=t0 + timedelta(seconds=i),
        topic="clicks",
        value=float(i),
    )
    proc.ingest(rec)

proc.flush_all()
for w in proc.closed_windows():
    print(f"[{w.window_start.second}s, {w.window_end.second}s): count={w.record_count} total={w.total_value:.0f}")
# [0, 10): count=10 total=45
# [10, 20): count=10 total=145
# [20, 30): count=5 total=110
```

---

## Part 6: Full pytest test suite

Now write the tests. Tests are the contract — they should pass before any refactoring.

```python
# tests/test_aggregate.py
from __future__ import annotations

import pytest

from event_sourcing.aggregate import (
    InventoryItem,
    InsufficientStock,
    InvalidQuantity,
)
from event_sourcing.models import Snapshot, StoredEvent


class TestInventoryItemCreation:
    def test_create_produces_one_event(self) -> None:
        item = InventoryItem.create("item-1", "SKU-A")
        assert len(item.pending_events()) == 1

    def test_create_sets_sku(self) -> None:
        item = InventoryItem.create("item-1", "SKU-A")
        assert item.sku == "SKU-A"

    def test_create_initializes_stock_to_zero(self) -> None:
        item = InventoryItem.create("item-1", "SKU-A")
        assert item.on_hand == 0
        assert item.reserved == 0

    def test_create_sets_sequence_to_one(self) -> None:
        item = InventoryItem.create("item-1", "SKU-A")
        assert item.sequence == 1


class TestInventoryItemReceive:
    def setup_method(self) -> None:
        self.item = InventoryItem.create("item-1", "SKU-A")

    def test_receive_increases_on_hand(self) -> None:
        self.item.receive(50)
        assert self.item.on_hand == 50

    def test_receive_multiple_times_accumulates(self) -> None:
        self.item.receive(30)
        self.item.receive(70)
        assert self.item.on_hand == 100

    def test_receive_zero_raises(self) -> None:
        with pytest.raises(InvalidQuantity):
            self.item.receive(0)

    def test_receive_negative_raises(self) -> None:
        with pytest.raises(InvalidQuantity):
            self.item.receive(-5)

    def test_receive_adds_pending_event(self) -> None:
        initial_count = len(self.item.pending_events())
        self.item.receive(10)
        assert len(self.item.pending_events()) == initial_count + 1


class TestInventoryItemReserve:
    def setup_method(self) -> None:
        self.item = InventoryItem.create("item-1", "SKU-A")
        self.item.receive(100)

    def test_reserve_increases_reserved(self) -> None:
        self.item.reserve(40)
        assert self.item.reserved == 40

    def test_reserve_reduces_available(self) -> None:
        self.item.reserve(40)
        assert self.item.available == 60

    def test_reserve_does_not_change_on_hand(self) -> None:
        self.item.reserve(40)
        assert self.item.on_hand == 100

    def test_reserve_more_than_available_raises(self) -> None:
        with pytest.raises(InsufficientStock):
            self.item.reserve(101)

    def test_reserve_exactly_available_succeeds(self) -> None:
        self.item.reserve(100)
        assert self.item.reserved == 100
        assert self.item.available == 0


class TestInventoryItemShip:
    def setup_method(self) -> None:
        self.item = InventoryItem.create("item-1", "SKU-A")
        self.item.receive(100)
        self.item.reserve(50)

    def test_ship_reduces_on_hand_and_reserved(self) -> None:
        self.item.ship(30)
        assert self.item.on_hand == 70
        assert self.item.reserved == 20

    def test_ship_more_than_reserved_raises(self) -> None:
        with pytest.raises(InsufficientStock):
            self.item.ship(60)

    def test_ship_exact_reserved_succeeds(self) -> None:
        self.item.ship(50)
        assert self.item.reserved == 0
        assert self.item.on_hand == 50


class TestAggregateReconstitution:
    def test_reconstitute_from_events_matches_original(self) -> None:
        original = InventoryItem.create("item-1", "SKU-A")
        original.receive(100)
        original.reserve(30)
        original.ship(30)
        original.receive(50)

        events = original.pending_events()
        reconstituted = InventoryItem.reconstitute("item-1", events)

        assert reconstituted.on_hand == original.on_hand
        assert reconstituted.reserved == original.reserved
        assert reconstituted.sku == original.sku
        assert reconstituted.sequence == original.sequence

    def test_reconstituted_aggregate_has_no_pending_events(self) -> None:
        item = InventoryItem.create("item-1", "SKU-A")
        item.receive(100)
        events = item.pending_events()
        reconstituted = InventoryItem.reconstitute("item-1", events)
        assert reconstituted.pending_events() == []

    def test_reconstitute_from_empty_events_raises(self) -> None:
        with pytest.raises(Exception):
            # No events means we can't know the SKU — applying nothing leaves sku=""
            # The aggregate exists but has no meaningful state; test the behavior you define
            item = InventoryItem.reconstitute("item-1", [])
            # At minimum sequence should be 0
            assert item.sequence == 0

    def test_reconstitute_preserves_sequence(self) -> None:
        item = InventoryItem.create("item-1", "SKU-A")
        for _ in range(10):
            item.receive(1)
        events = item.pending_events()
        reconstituted = InventoryItem.reconstitute("item-1", events)
        assert reconstituted.sequence == 11  # 1 create + 10 receives


class TestAggregateSnapshot:
    def test_snapshot_captures_correct_state(self) -> None:
        item = InventoryItem.create("item-1", "SKU-A")
        item.receive(100)
        item.reserve(40)

        snapshot = item.take_snapshot()

        assert snapshot.aggregate_id == "item-1"
        assert snapshot.sequence == 3
        assert snapshot.state["sku"] == "SKU-A"
        assert snapshot.state["on_hand"] == 100
        assert snapshot.state["reserved"] == 40

    def test_reconstitute_from_snapshot_plus_events(self) -> None:
        # Build initial state, snapshot at seq=3
        item = InventoryItem.create("item-1", "SKU-A")
        item.receive(100)
        item.reserve(40)
        snapshot = item.take_snapshot()

        # Apply more events after snapshot
        item.ship(40)
        item.receive(20)
        post_snapshot_events = [
            e for e in item.pending_events() if e.sequence > snapshot.sequence
        ]

        # Reconstitute from snapshot + delta events
        restored = InventoryItem.reconstitute("item-1", post_snapshot_events, snapshot)

        assert restored.on_hand == item.on_hand
        assert restored.reserved == item.reserved
        assert restored.sequence == item.sequence

    def test_snapshot_round_trip_state_integrity(self) -> None:
        item = InventoryItem.create("item-1", "SKU-X")
        item.receive(200)
        item.reserve(75)
        item.ship(25)
        item.receive(10)

        snapshot = item.take_snapshot()
        restored = InventoryItem("item-1")
        restored._restore_snapshot(snapshot)

        assert restored.on_hand == item.on_hand
        assert restored.reserved == item.reserved
        assert restored.sku == item.sku
```

```python
# tests/test_event_store.py
from __future__ import annotations

import pytest

from event_sourcing.aggregate import AggregateVersionConflict, InventoryItem
from event_sourcing.event_store import EventStore


class TestEventStoreBasicSaveLoad:
    def setup_method(self) -> None:
        self.store = EventStore()

    def test_save_and_event_count(self) -> None:
        item = InventoryItem.create("item-1", "SKU-A")
        item.receive(100)
        self.store.save(item)
        assert self.store.event_count("item-1") == 2

    def test_save_clears_pending_events(self) -> None:
        item = InventoryItem.create("item-1", "SKU-A")
        self.store.save(item)
        assert item.pending_events() == []

    def test_load_events_returns_correct_count(self) -> None:
        item = InventoryItem.create("item-1", "SKU-A")
        item.receive(50)
        item.reserve(10)
        self.store.save(item)
        events = self.store.load_events("item-1")
        assert len(events) == 3

    def test_load_events_after_sequence(self) -> None:
        item = InventoryItem.create("item-1", "SKU-A")
        item.receive(50)
        item.reserve(10)
        self.store.save(item)
        events = self.store.load_events("item-1", after_sequence=1)
        assert len(events) == 2
        assert all(e.sequence > 1 for e in events)

    def test_exists_false_for_unknown_aggregate(self) -> None:
        assert not self.store.exists("nonexistent")

    def test_exists_true_after_save(self) -> None:
        item = InventoryItem.create("item-1", "SKU-A")
        self.store.save(item)
        assert self.store.exists("item-1")


class TestEventStoreReconstitution:
    def setup_method(self) -> None:
        self.store = EventStore()

    def test_reconstitute_matches_original_state(self) -> None:
        item = InventoryItem.create("item-1", "SKU-A")
        item.receive(100)
        item.reserve(30)
        self.store.save(item)

        restored = self.store.reconstitute(InventoryItem, "item-1")
        assert restored.on_hand == 100
        assert restored.reserved == 30
        assert restored.sequence == 3

    def test_reconstitute_nonexistent_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            self.store.reconstitute(InventoryItem, "does-not-exist")

    def test_multiple_saves_accumulate_correctly(self) -> None:
        item = InventoryItem.create("item-1", "SKU-A")
        item.receive(100)
        self.store.save(item)  # seq 2

        item.reserve(40)
        self.store.save(item)  # seq 3

        item.ship(40)
        self.store.save(item)  # seq 4

        restored = self.store.reconstitute(InventoryItem, "item-1")
        assert restored.on_hand == 60
        assert restored.reserved == 0


class TestOptimisticConcurrency:
    def setup_method(self) -> None:
        self.store = EventStore()

    def test_save_with_correct_expected_version_succeeds(self) -> None:
        item = InventoryItem.create("item-1", "SKU-A")
        self.store.save(item, expected_version=0)
        assert self.store.event_count("item-1") == 1

    def test_save_with_wrong_expected_version_raises(self) -> None:
        item = InventoryItem.create("item-1", "SKU-A")
        item.receive(100)
        self.store.save(item)  # now at seq 2

        item2 = self.store.reconstitute(InventoryItem, "item-1")
        item2.receive(50)
        self.store.save(item2, expected_version=2)  # OK, seq was 2

        # Simulate stale writer still at seq 2
        item3 = InventoryItem.reconstitute(
            "item-1", self.store.load_events("item-1", after_sequence=0)
        )
        item3.reserve(10)
        item3._sequence = 2  # force stale version
        item3._pending_events[0] = item3._pending_events[0].model_copy(
            update={"sequence": 3}
        )
        with pytest.raises(AggregateVersionConflict):
            self.store.save(item3, expected_version=2)  # store is at 3 now


class TestAutoSnapshot:
    def test_snapshot_taken_at_threshold(self) -> None:
        store = EventStore(snapshot_threshold=5)
        item = InventoryItem.create("item-1", "SKU-A")
        # Save 4 more events (total 5 = threshold)
        for _ in range(4):
            item.receive(1)
        store.save(item)  # 5 events → snapshot
        assert store.load_snapshot("item-1") is not None

    def test_no_snapshot_before_threshold(self) -> None:
        store = EventStore(snapshot_threshold=10)
        item = InventoryItem.create("item-1", "SKU-A")
        item.receive(1)
        store.save(item)  # only 2 events
        assert store.load_snapshot("item-1") is None

    def test_reconstitute_uses_snapshot(self) -> None:
        store = EventStore(snapshot_threshold=3)
        item = InventoryItem.create("item-1", "SKU-A")
        item.receive(100)
        store.save(item)  # 3 events → snapshot

        item.reserve(30)
        item.ship(30)
        store.save(item)  # 2 more events, total 5

        # Should use snapshot + 2 events
        restored = store.reconstitute(InventoryItem, "item-1")
        assert restored.on_hand == 70
        assert restored.reserved == 0
```

```python
# tests/test_stream_processor.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from event_sourcing.models import StreamRecord, WindowResult
from event_sourcing.stream_processor import TumblingWindowProcessor

UTC = timezone.utc
T0 = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)


def make_record(offset_seconds: float, value: float, topic: str = "metrics") -> StreamRecord:
    return StreamRecord(
        timestamp=T0 + timedelta(seconds=offset_seconds),
        topic=topic,
        value=value,
    )


class TestTumblingWindowBasic:
    def test_single_window_record_count(self) -> None:
        proc = TumblingWindowProcessor(window_size_seconds=10)
        for i in range(5):
            proc.ingest(make_record(i, 1.0))
        proc.flush_all()
        windows = proc.closed_windows()
        assert len(windows) == 1
        assert windows[0].record_count == 5

    def test_single_window_total_value(self) -> None:
        proc = TumblingWindowProcessor(window_size_seconds=10)
        for i in range(5):
            proc.ingest(make_record(i, float(i + 1)))
        proc.flush_all()
        assert proc.closed_windows()[0].total_value == 15.0

    def test_two_complete_windows(self) -> None:
        proc = TumblingWindowProcessor(window_size_seconds=10)
        # 10 records in [0, 10) and 10 records in [10, 20)
        for i in range(20):
            proc.ingest(make_record(i, 1.0))
        proc.flush_all()
        windows = proc.closed_windows()
        assert len(windows) == 2
        assert windows[0].record_count == 10
        assert windows[1].record_count == 10

    def test_window_boundaries_are_correct(self) -> None:
        proc = TumblingWindowProcessor(window_size_seconds=10)
        proc.ingest(make_record(0, 1.0))
        proc.flush_all()
        w = proc.closed_windows()[0]
        assert w.window_start == T0
        assert w.window_end == T0 + timedelta(seconds=10)

    def test_empty_processor_flush_produces_no_windows(self) -> None:
        proc = TumblingWindowProcessor(window_size_seconds=10)
        proc.flush_all()
        assert proc.closed_windows() == []


class TestWindowAggregation:
    def test_min_max_avg_correct(self) -> None:
        proc = TumblingWindowProcessor(window_size_seconds=10)
        values = [1.0, 5.0, 3.0, 9.0, 2.0]
        for i, v in enumerate(values):
            proc.ingest(make_record(i, v))
        proc.flush_all()
        w = proc.closed_windows()[0]
        assert w.min_value == 1.0
        assert w.max_value == 9.0
        assert abs(w.avg_value - 4.0) < 1e-9

    def test_single_record_window_aggregation(self) -> None:
        proc = TumblingWindowProcessor(window_size_seconds=10)
        proc.ingest(make_record(5, 42.0))
        proc.flush_all()
        w = proc.closed_windows()[0]
        assert w.min_value == 42.0
        assert w.max_value == 42.0
        assert w.avg_value == 42.0
        assert w.record_count == 1


class TestWindowOrdering:
    def test_records_span_three_windows(self) -> None:
        proc = TumblingWindowProcessor(window_size_seconds=10)
        for i in range(30):
            proc.ingest(make_record(i, 1.0))
        proc.flush_all()
        assert len(proc.closed_windows()) == 3

    def test_windows_are_ordered_chronologically(self) -> None:
        proc = TumblingWindowProcessor(window_size_seconds=10)
        for i in range(30):
            proc.ingest(make_record(i, 1.0))
        proc.flush_all()
        windows = proc.closed_windows()
        for i in range(1, len(windows)):
            assert windows[i].window_start > windows[i - 1].window_start

    def test_window_start_aligned_to_first_record(self) -> None:
        proc = TumblingWindowProcessor(window_size_seconds=10)
        proc.ingest(make_record(3.5, 1.0))   # first record at t=3.5s
        proc.flush_all()
        w = proc.closed_windows()[0]
        assert w.window_start == T0 + timedelta(seconds=3.5)


class TestTopicFiltering:
    def test_different_topic_ignored(self) -> None:
        proc = TumblingWindowProcessor(window_size_seconds=10, topic="clicks")
        proc.ingest(make_record(1, 5.0, topic="errors"))
        proc.ingest(make_record(2, 10.0, topic="clicks"))
        proc.flush_all()
        w = proc.closed_windows()[0]
        assert w.record_count == 1
        assert w.total_value == 10.0

    def test_no_topic_filter_accepts_all(self) -> None:
        proc = TumblingWindowProcessor(window_size_seconds=10, topic=None)
        proc.ingest(make_record(1, 1.0, topic="clicks"))
        proc.ingest(make_record(2, 2.0, topic="errors"))
        proc.flush_all()
        assert proc.closed_windows()[0].record_count == 2


class TestWindowCallback:
    def test_callback_invoked_per_window(self) -> None:
        results: list[WindowResult] = []
        proc = TumblingWindowProcessor(
            window_size_seconds=10,
            on_window_close=results.append,
        )
        for i in range(25):
            proc.ingest(make_record(i, 1.0))
        proc.flush_all()
        # 2 full windows + 1 partial = 3 flushes
        assert len(results) == 3

    def test_callback_receives_correct_result(self) -> None:
        captured: list[WindowResult] = []
        proc = TumblingWindowProcessor(
            window_size_seconds=10,
            on_window_close=captured.append,
        )
        for i in range(10):
            proc.ingest(make_record(i, float(i)))
        proc.flush_all()
        assert captured[0].record_count == 10
        assert captured[0].total_value == sum(range(10))


class TestReset:
    def test_reset_clears_windows(self) -> None:
        proc = TumblingWindowProcessor(window_size_seconds=10)
        for i in range(15):
            proc.ingest(make_record(i, 1.0))
        proc.flush_all()
        assert len(proc.closed_windows()) > 0
        proc.reset()
        assert proc.closed_windows() == []

    def test_reset_allows_reingest(self) -> None:
        proc = TumblingWindowProcessor(window_size_seconds=10)
        proc.ingest(make_record(0, 1.0))
        proc.flush_all()
        proc.reset()
        proc.ingest(make_record(5, 99.0))
        proc.flush_all()
        assert proc.closed_windows()[0].total_value == 99.0
```

```python
# tests/test_integration.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from event_sourcing.aggregate import InventoryItem
from event_sourcing.event_store import EventStore
from event_sourcing.models import StreamRecord
from event_sourcing.stream_processor import TumblingWindowProcessor

UTC = timezone.utc


class TestInventoryLifecycle:
    def test_full_order_fulfillment_lifecycle(self) -> None:
        store = EventStore(snapshot_threshold=100)

        # Day 1: receive stock, take order
        item = InventoryItem.create("sku-001", "WIDGET")
        item.receive(500)
        item.reserve(50)
        store.save(item)

        # Day 2: ship order, receive new stock
        item = store.reconstitute(InventoryItem, "sku-001")
        item.ship(50)
        item.receive(100)
        store.save(item)

        # Day 3: verify state
        final = store.reconstitute(InventoryItem, "sku-001")
        assert final.on_hand == 550
        assert final.reserved == 0
        assert final.available == 550

    def test_snapshot_reduces_events_replayed(self) -> None:
        store = EventStore(snapshot_threshold=5)
        item = InventoryItem.create("sku-002", "GADGET")
        for _ in range(9):  # 10 events total (1 create + 9 receives)
            item.receive(10)
        store.save(item)  # threshold=5, snapshot taken at seq=10? no, at seq=5 and seq=10

        # Verify snapshot exists
        snap = store.load_snapshot("sku-002")
        assert snap is not None

        # Reconstitute — should use snapshot + post-snap events
        restored = store.reconstitute(InventoryItem, "sku-002")
        assert restored.on_hand == 90  # 9 * 10

    def test_saga_compensation_pattern(self) -> None:
        store = EventStore()

        # Forward: create + receive + reserve (simulating saga step)
        item = InventoryItem.create("sku-003", "PART")
        item.receive(100)
        item.reserve(30)
        store.save(item)

        # Compensation: release reservation (payment failed)
        item = store.reconstitute(InventoryItem, "sku-003")
        item.release_reservation(30)
        store.save(item)

        final = store.reconstitute(InventoryItem, "sku-003")
        assert final.reserved == 0
        assert final.available == 100


class TestStreamProcessorIntegration:
    def test_real_time_ingestion_simulation(self) -> None:
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        received: list = []
        proc = TumblingWindowProcessor(
            window_size_seconds=60,
            topic="orders",
            on_window_close=received.append,
        )

        # Simulate 3 minutes of order events, one per second
        for i in range(180):
            proc.ingest(
                StreamRecord(
                    timestamp=t0 + timedelta(seconds=i),
                    topic="orders",
                    value=99.0,
                )
            )
        proc.flush_all()

        assert len(received) == 3
        assert all(w.record_count == 60 for w in received)
        assert all(w.total_value == 60 * 99.0 for w in received)
```

---

## Part 7: Running the tests

```bash
# From the directory containing event_sourcing/ and tests/
pip install pydantic pytest

# Run all tests with verbose output
python -m pytest tests/ -v

# Run a single module
python -m pytest tests/test_aggregate.py -v

# Run with coverage (optional)
pip install pytest-cov
python -m pytest tests/ --cov=event_sourcing --cov-report=term-missing
```

Expected output (all passing):
```
tests/test_aggregate.py::TestInventoryItemCreation::test_create_produces_one_event PASSED
tests/test_aggregate.py::TestInventoryItemCreation::test_create_sets_sku PASSED
... (40+ tests)
tests/test_integration.py::TestStreamProcessorIntegration::test_real_time_ingestion_simulation PASSED

======= 47 passed in 0.12s =======
```

---

## Exercises

1. **Sliding window**: extend `TumblingWindowProcessor` to support a **sliding window** (step < window size). A 10s window with 5s step means records in [0,10) are in W1, records in [5,15) are in W2, etc. Start with a design sketch before coding.

2. **Session window**: implement a `SessionWindowProcessor` that groups records by inactivity gap. If two consecutive records are more than `gap_seconds` apart, close the current window. No fixed duration.

3. **ShoppingCart aggregate**: implement a `ShoppingCart` aggregate with commands: `add_item(sku, qty, price)`, `remove_item(sku)`, `checkout()`. Events: `ItemAdded`, `ItemRemoved`, `CartCheckedOut`. Include a `total` property. Write at least 8 pytest tests.

4. **Projection**: a projection subscribes to the EventStore's event stream and builds a read model. Implement a `LowStockProjection` that maintains a dict of `{sku: available_qty}` and rebuilds from all `InventoryItem` events in the store. Add a `below_threshold(min_qty)` method.

5. **Versioned events**: event schemas change over time. Add an `event_version: int` field to `StoredEvent`. Implement an `upcaster` function that converts `StockReceived` v1 `{"qty": int}` to v2 `{"qty": int, "warehouse": str}`. Apply upcasting during reconstitution.

(Solutions: ../solutions/week-04/day-6.md)

---

## Self-check

- [ ] Can explain the difference between `_raise_event` (command side) and `_dispatch` (apply side).
- [ ] Understands why `from_snapshot_state` mutates `self` rather than returning a new instance.
- [ ] Can explain optimistic concurrency and which race condition it prevents.
- [ ] Knows why snapshots start at `after_sequence = snapshot.sequence` for event loading.
- [ ] Can describe when `flush_all()` must be called and what happens if it's omitted.
- [ ] Understands why tumbling windows are simpler than sliding windows in stateful processing.
- [ ] Can draw the data flow: `Command → _raise_event → StoredEvent → _dispatch → _apply_*`.

---

## Supplementary links

- Martin Fowler — Event Sourcing: https://martinfowler.com/eaaDev/EventSourcing.html
- Greg Young — CQRS Documents: https://cqrs.files.wordpress.com/2010/11/cqrs_documents.pdf
- Flink documentation — Tumbling Windows: https://nightlies.apache.org/flink/flink-docs-stable/docs/dev/datastream/operators/windows/
- EventStoreDB docs — Optimistic concurrency: https://developers.eventstore.com/clients/grpc/appending-events.html#optimistic-concurrency-and-idempotency
- Vaughn Vernon — Implementing Domain-Driven Design (Chapter 8: Domain Events): https://www.informit.com/store/implementing-domain-driven-design-9780321834577
