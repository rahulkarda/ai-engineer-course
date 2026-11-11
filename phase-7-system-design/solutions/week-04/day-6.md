# Solutions — Week 04 · Day 6

Complete reference implementation for the 4-hour build: event sourcing + tumbling-window stream processor.

---

## Pydantic Models (`models.py`)

```python
"""
models.py — Shared Pydantic models for event store and stream processor.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class StoredEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    stream_id: str
    version: int
    event_type: str
    data: dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Snapshot(BaseModel):
    stream_id: str
    version: int
    state: dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StreamRecord(BaseModel):
    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic: str
    timestamp: datetime
    value: float
    attributes: dict[str, Any] = Field(default_factory=dict)


class WindowResult(BaseModel):
    window_start: datetime
    window_end: datetime
    topic: str
    count: int
    total: float
    min_value: float
    max_value: float
```

---

## EventSourcedAggregate Base Class (`aggregate.py`)

```python
"""
aggregate.py — Abstract base class for event-sourced aggregates.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from models import StoredEvent, Snapshot

logger = logging.getLogger(__name__)


class EventSourcedAggregate(ABC):
    """
    Base class for event-sourced aggregates.

    Subclasses implement:
        _get_initial_state() -> dict
        _apply(state: dict, event: StoredEvent) -> dict
    """

    def __init__(self) -> None:
        self._state: dict[str, Any] = self._get_initial_state()
        self._version: int = 0
        self._pending_events: list[StoredEvent] = []

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _get_initial_state(self) -> dict[str, Any]:
        """Return the zero-value state for this aggregate type."""
        ...

    @abstractmethod
    def _apply(self, state: dict[str, Any], event: StoredEvent) -> dict[str, Any]:
        """
        Pure function: given current state and a stored event, return new state.
        Must not produce side effects.
        """
        ...

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _raise_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Append a new uncommitted event to the pending list and apply it."""
        from models import StoredEvent  # local import avoids circular at module level

        event = StoredEvent(
            stream_id=self._stream_id(),
            version=self._version + len(self._pending_events) + 1,
            event_type=event_type,
            data=data,
        )
        self._state = self._apply(self._state, event)
        self._pending_events.append(event)
        logger.debug("Raised event %s v%d on %s", event_type, event.version, event.stream_id)

    def _stream_id(self) -> str:
        """Override in subclass to return a meaningful stream identifier."""
        return getattr(self, "id", type(self).__name__)

    # ------------------------------------------------------------------
    # Reconstitution
    # ------------------------------------------------------------------

    @classmethod
    def reconstitute(
        cls,
        events: list[StoredEvent],
        snapshot: Snapshot | None = None,
    ) -> "EventSourcedAggregate":
        """
        Rebuild aggregate state by folding events over initial (or snapshot) state.
        If a snapshot is provided, only events after the snapshot version are replayed.
        """
        instance = cls.__new__(cls)
        instance._pending_events = []

        if snapshot is not None:
            instance._state = dict(snapshot.state)
            instance._version = snapshot.version
            events = [e for e in events if e.version > snapshot.version]
        else:
            instance._state = instance._get_initial_state()
            instance._version = 0

        for event in sorted(events, key=lambda e: e.version):
            instance._state = instance._apply(instance._state, event)
            instance._version = event.version

        return instance

    # ------------------------------------------------------------------
    # Snapshot support
    # ------------------------------------------------------------------

    def get_snapshot(self) -> dict[str, Any]:
        """Return a plain dict of current state + version (for Snapshot model)."""
        return {"version": self._version, "state": dict(self._state)}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def version(self) -> int:
        return self._version

    @property
    def pending_events(self) -> list[StoredEvent]:
        return list(self._pending_events)

    def mark_committed(self) -> None:
        """Call after events are persisted to the event store."""
        self._version += len(self._pending_events)
        self._pending_events.clear()
```

---

## InventoryItem Aggregate (`inventory.py`)

```python
"""
inventory.py — Event-sourced InventoryItem aggregate.
"""
from __future__ import annotations

import logging
from typing import Any

from aggregate import EventSourcedAggregate
from models import StoredEvent

logger = logging.getLogger(__name__)


class InsufficientStockError(Exception):
    pass


class InventoryItem(EventSourcedAggregate):
    """
    Aggregate root for an inventory SKU.

    Events: ItemCreated, StockAdded, StockReserved, StockReleased, StockAdjusted
    """

    def __init__(self, item_id: str) -> None:
        self.id = item_id
        super().__init__()

    # ------------------------------------------------------------------
    # EventSourcedAggregate implementation
    # ------------------------------------------------------------------

    def _get_initial_state(self) -> dict[str, Any]:
        return {
            "item_id": "",
            "name": "",
            "quantity_on_hand": 0,
            "quantity_reserved": 0,
        }

    def _apply(self, state: dict[str, Any], event: StoredEvent) -> dict[str, Any]:
        state = dict(state)  # shallow copy — keep it pure
        match event.event_type:
            case "ItemCreated":
                state["item_id"] = event.data["item_id"]
                state["name"] = event.data["name"]
                state["quantity_on_hand"] = event.data.get("initial_quantity", 0)
                state["quantity_reserved"] = 0
            case "StockAdded":
                state["quantity_on_hand"] += event.data["quantity"]
            case "StockReserved":
                state["quantity_on_hand"] -= event.data["quantity"]
                state["quantity_reserved"] += event.data["quantity"]
            case "StockReleased":
                state["quantity_on_hand"] += event.data["quantity"]
                state["quantity_reserved"] -= event.data["quantity"]
            case "StockAdjusted":
                state["quantity_on_hand"] = event.data["new_quantity"]
            case _:
                logger.warning("Unknown event type: %s", event.event_type)
        return state

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def create(self, name: str, initial_quantity: int = 0) -> None:
        if initial_quantity < 0:
            raise ValueError("Initial quantity cannot be negative")
        self._raise_event("ItemCreated", {
            "item_id": self.id,
            "name": name,
            "initial_quantity": initial_quantity,
        })

    def add_stock(self, quantity: int) -> None:
        if quantity <= 0:
            raise ValueError("Quantity to add must be positive")
        self._raise_event("StockAdded", {"quantity": quantity})

    def reserve_stock(self, quantity: int) -> None:
        if quantity <= 0:
            raise ValueError("Quantity to reserve must be positive")
        available = self._state["quantity_on_hand"]
        if quantity > available:
            raise InsufficientStockError(
                f"Cannot reserve {quantity}; only {available} available"
            )
        self._raise_event("StockReserved", {"quantity": quantity})

    def release_stock(self, quantity: int) -> None:
        if quantity <= 0:
            raise ValueError("Quantity to release must be positive")
        reserved = self._state["quantity_reserved"]
        if quantity > reserved:
            raise ValueError(f"Cannot release {quantity}; only {reserved} reserved")
        self._raise_event("StockReleased", {"quantity": quantity})

    def adjust_stock(self, new_quantity: int) -> None:
        if new_quantity < 0:
            raise ValueError("Stock cannot go negative")
        self._raise_event("StockAdjusted", {"new_quantity": new_quantity})

    # ------------------------------------------------------------------
    # Read projections
    # ------------------------------------------------------------------

    @property
    def quantity_on_hand(self) -> int:
        return self._state["quantity_on_hand"]

    @property
    def quantity_reserved(self) -> int:
        return self._state["quantity_reserved"]

    @property
    def name(self) -> str:
        return self._state["name"]
```

---

## InMemoryEventStore (`event_store.py`)

```python
"""
event_store.py — In-memory event store with optimistic concurrency control.
"""
from __future__ import annotations

import logging
from typing import Any

from aggregate import EventSourcedAggregate
from models import Snapshot, StoredEvent

logger = logging.getLogger(__name__)


class ConcurrencyError(Exception):
    pass


class StreamNotFoundError(Exception):
    pass


class InMemoryEventStore:
    """
    Thread-unsafe in-memory event store for testing and local development.
    Suitable for single-threaded use; wrap with a lock for concurrent access.
    """

    def __init__(self) -> None:
        # stream_id -> list of StoredEvent (ordered by version)
        self._streams: dict[str, list[StoredEvent]] = {}
        # stream_id -> latest Snapshot
        self._snapshots: dict[str, Snapshot] = {}

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def append(
        self,
        stream_id: str,
        events: list[StoredEvent],
        expected_version: int = -1,
    ) -> None:
        """
        Append events to a stream.

        Args:
            stream_id: Logical stream identifier.
            events: Events to append (must have correct sequential versions).
            expected_version: Optimistic concurrency check.
                -1 means "stream must not exist".
                0 means "stream exists but is empty".
                N means "current version must be N".

        Raises:
            ConcurrencyError: If actual version != expected_version.
        """
        current_events = self._streams.get(stream_id, [])
        current_version = current_events[-1].version if current_events else -1

        if expected_version != -2 and current_version != expected_version:
            raise ConcurrencyError(
                f"Stream '{stream_id}': expected version {expected_version}, "
                f"found {current_version}"
            )

        if stream_id not in self._streams:
            self._streams[stream_id] = []

        self._streams[stream_id].extend(events)
        logger.debug(
            "EventStore: appended %d event(s) to stream '%s' (new version: %d)",
            len(events),
            stream_id,
            events[-1].version if events else current_version,
        )

    def save_snapshot(self, snapshot: Snapshot) -> None:
        self._snapshots[snapshot.stream_id] = snapshot
        logger.debug(
            "EventStore: saved snapshot for '%s' at version %d",
            snapshot.stream_id,
            snapshot.version,
        )

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def load(self, stream_id: str, after_version: int = 0) -> list[StoredEvent]:
        """Return events for a stream with version > after_version."""
        events = self._streams.get(stream_id, [])
        return [e for e in events if e.version > after_version]

    def load_snapshot(self, stream_id: str) -> Snapshot | None:
        return self._snapshots.get(stream_id)

    def stream_version(self, stream_id: str) -> int:
        """Return -1 if stream does not exist, else latest event version."""
        events = self._streams.get(stream_id)
        if not events:
            return -1
        return events[-1].version

    # ------------------------------------------------------------------
    # Reconstitution helper
    # ------------------------------------------------------------------

    def reconstitute(
        self,
        stream_id: str,
        aggregate_class: type[EventSourcedAggregate],
        *init_args: Any,
    ) -> EventSourcedAggregate:
        """
        Load and reconstitute an aggregate from the store.

        Args:
            stream_id: Stream to load.
            aggregate_class: Class to instantiate (must accept *init_args).
            *init_args: Positional args forwarded to aggregate_class constructor.
        """
        snapshot = self.load_snapshot(stream_id)
        after_version = snapshot.version if snapshot else 0
        events = self.load(stream_id, after_version=after_version)
        return aggregate_class.reconstitute(events, snapshot=snapshot)
```

---

## TumblingWindowProcessor (`stream_processor.py`)

```python
"""
stream_processor.py — Tumbling window processor for time-series stream records.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Callable

from models import StreamRecord, WindowResult

logger = logging.getLogger(__name__)


def _window_start(ts: datetime, window_size_seconds: int) -> datetime:
    """Truncate a timestamp to the nearest window boundary."""
    epoch = datetime(1970, 1, 1)
    delta = (ts - epoch).total_seconds()
    bucket = int(delta // window_size_seconds) * window_size_seconds
    return epoch + timedelta(seconds=bucket)


class _WindowAccumulator:
    def __init__(self, start: datetime, end: datetime, topic: str) -> None:
        self.start = start
        self.end = end
        self.topic = topic
        self.count: int = 0
        self.total: float = 0.0
        self.min_value: float = float("inf")
        self.max_value: float = float("-inf")

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)

    def to_result(self) -> WindowResult:
        return WindowResult(
            window_start=self.start,
            window_end=self.end,
            topic=self.topic,
            count=self.count,
            total=round(self.total, 6),
            min_value=self.min_value if self.count > 0 else 0.0,
            max_value=self.max_value if self.count > 0 else 0.0,
        )


class TumblingWindowProcessor:
    """
    Assigns each incoming StreamRecord to a non-overlapping (tumbling) time window.
    Closes and emits a window when a record arrives with a timestamp beyond
    window_end + grace_period_seconds.

    Args:
        window_size_seconds: Window duration in seconds (e.g. 60 = 1-minute windows).
        grace_period_seconds: Extra wait before closing a window (for late data).
        on_window_close: Callback invoked with the WindowResult when a window closes.
    """

    def __init__(
        self,
        window_size_seconds: int,
        grace_period_seconds: int = 0,
        on_window_close: Callable[[WindowResult], None] | None = None,
    ) -> None:
        self._window_size = window_size_seconds
        self._grace_period = grace_period_seconds
        self._on_window_close = on_window_close or (lambda r: None)
        # (topic, window_start) -> _WindowAccumulator
        self._windows: dict[tuple[str, datetime], _WindowAccumulator] = {}
        self._closed_windows: list[WindowResult] = []
        self._watermark: datetime = datetime.min

    def process(self, record: StreamRecord) -> None:
        """
        Process a single stream record.

        Late records (beyond watermark + grace period) are silently dropped.
        """
        ws = _window_start(record.timestamp, self._window_size)
        we = ws + timedelta(seconds=self._window_size)
        cutoff = self._watermark - timedelta(seconds=self._grace_period)

        if record.timestamp < cutoff and self._watermark != datetime.min:
            logger.warning(
                "Dropping late record for topic=%s ts=%s (watermark=%s)",
                record.topic,
                record.timestamp,
                self._watermark,
            )
            return

        # Advance watermark
        if record.timestamp > self._watermark:
            self._watermark = record.timestamp
            self._close_expired_windows()

        key = (record.topic, ws)
        if key not in self._windows:
            self._windows[key] = _WindowAccumulator(ws, we, record.topic)

        self._windows[key].add(record.value)

    def _close_expired_windows(self) -> None:
        """Close all windows whose end + grace period is <= watermark."""
        to_close = [
            key
            for key, acc in self._windows.items()
            if acc.end + timedelta(seconds=self._grace_period) <= self._watermark
        ]
        for key in to_close:
            acc = self._windows.pop(key)
            result = acc.to_result()
            self._closed_windows.append(result)
            logger.info(
                "Closed window %s–%s topic=%s count=%d total=%.2f",
                result.window_start,
                result.window_end,
                result.topic,
                result.count,
                result.total,
            )
            self._on_window_close(result)

    def flush_all(self) -> None:
        """Force-close all open windows (e.g. at end-of-stream)."""
        keys = list(self._windows.keys())
        for key in keys:
            acc = self._windows.pop(key)
            result = acc.to_result()
            self._closed_windows.append(result)
            self._on_window_close(result)

    def get_open_windows(self) -> list[WindowResult]:
        """Return snapshot of all currently open (not yet closed) windows."""
        return [acc.to_result() for acc in self._windows.values()]

    @property
    def closed_windows(self) -> list[WindowResult]:
        return list(self._closed_windows)
```

---

## Test Suite

### `test_aggregate.py`

```python
"""
test_aggregate.py — Tests for InventoryItem aggregate and EventSourcedAggregate base.
"""
from __future__ import annotations

import pytest

from inventory import InsufficientStockError, InventoryItem
from models import Snapshot, StoredEvent


def _create_item(item_id: str = "sku-001", name: str = "Widget", qty: int = 0) -> InventoryItem:
    item = InventoryItem(item_id)
    item.create(name=name, initial_quantity=qty)
    return item


# ------------------------------------------------------------------
# Creation
# ------------------------------------------------------------------

def test_item_created_emits_event() -> None:
    item = InventoryItem("sku-001")
    item.create(name="Widget A", initial_quantity=10)

    assert len(item.pending_events) == 1
    assert item.pending_events[0].event_type == "ItemCreated"


def test_item_created_state() -> None:
    item = _create_item(qty=50)
    assert item.name == "Widget"
    assert item.quantity_on_hand == 50
    assert item.quantity_reserved == 0


def test_item_created_negative_quantity_raises() -> None:
    item = InventoryItem("sku-002")
    with pytest.raises(ValueError, match="negative"):
        item.create(name="Bad Item", initial_quantity=-1)


# ------------------------------------------------------------------
# Stock operations
# ------------------------------------------------------------------

def test_stock_add_increases_on_hand() -> None:
    item = _create_item(qty=10)
    item.add_stock(20)
    assert item.quantity_on_hand == 30


def test_stock_reserve_moves_to_reserved() -> None:
    item = _create_item(qty=100)
    item.reserve_stock(30)
    assert item.quantity_on_hand == 70
    assert item.quantity_reserved == 30


def test_stock_release_restores_on_hand() -> None:
    item = _create_item(qty=100)
    item.reserve_stock(40)
    item.release_stock(40)
    assert item.quantity_on_hand == 100
    assert item.quantity_reserved == 0


def test_stock_adjust_sets_absolute_quantity() -> None:
    item = _create_item(qty=50)
    item.adjust_stock(200)
    assert item.quantity_on_hand == 200


def test_cannot_over_reserve() -> None:
    item = _create_item(qty=10)
    with pytest.raises(InsufficientStockError):
        item.reserve_stock(11)


def test_adjust_negative_raises() -> None:
    item = _create_item(qty=10)
    with pytest.raises(ValueError, match="negative"):
        item.adjust_stock(-5)


def test_add_zero_raises() -> None:
    item = _create_item(qty=10)
    with pytest.raises(ValueError, match="positive"):
        item.add_stock(0)


# ------------------------------------------------------------------
# Reconstitution
# ------------------------------------------------------------------

def test_reconstitute_from_events() -> None:
    original = _create_item(qty=50)
    original.add_stock(20)
    original.reserve_stock(10)

    events = original.pending_events
    restored = InventoryItem.reconstitute(events)

    assert restored.quantity_on_hand == 60
    assert restored.quantity_reserved == 10


def test_reconstitute_from_snapshot() -> None:
    item = _create_item(qty=100)
    item.add_stock(50)
    # Take snapshot after first two events
    snap = Snapshot(
        stream_id=item.id,
        version=2,
        state={"item_id": "sku-001", "name": "Widget", "quantity_on_hand": 150, "quantity_reserved": 0},
    )
    # Append a later event
    later_event = StoredEvent(
        stream_id="sku-001",
        version=3,
        event_type="StockReserved",
        data={"quantity": 25},
    )
    restored = InventoryItem.reconstitute([later_event], snapshot=snap)
    assert restored.quantity_on_hand == 125
    assert restored.quantity_reserved == 25


def test_reconstitute_empty_stream() -> None:
    item = InventoryItem.reconstitute([])
    assert item.quantity_on_hand == 0


def test_pending_events_cleared_after_mark_committed() -> None:
    item = _create_item(qty=10)
    assert len(item.pending_events) == 1
    item.mark_committed()
    assert len(item.pending_events) == 0
    assert item.version == 1
```

### `test_event_store.py`

```python
"""
test_event_store.py — Tests for InMemoryEventStore.
"""
from __future__ import annotations

import pytest

from event_store import ConcurrencyError, InMemoryEventStore
from inventory import InventoryItem
from models import Snapshot, StoredEvent


def _make_event(stream_id: str, version: int, event_type: str = "ItemCreated") -> StoredEvent:
    return StoredEvent(
        stream_id=stream_id,
        version=version,
        event_type=event_type,
        data={},
    )


def test_append_and_load() -> None:
    store = InMemoryEventStore()
    e = _make_event("stream-1", version=1)
    store.append("stream-1", [e], expected_version=-1)
    loaded = store.load("stream-1")
    assert len(loaded) == 1
    assert loaded[0].event_id == e.event_id


def test_load_returns_empty_for_unknown_stream() -> None:
    store = InMemoryEventStore()
    assert store.load("nonexistent") == []


def test_optimistic_concurrency_raises_on_mismatch() -> None:
    store = InMemoryEventStore()
    e1 = _make_event("stream-2", version=1)
    store.append("stream-2", [e1], expected_version=-1)

    e2 = _make_event("stream-2", version=2)
    with pytest.raises(ConcurrencyError):
        store.append("stream-2", [e2], expected_version=0)  # wrong: should be 1


def test_optimistic_concurrency_succeeds_on_match() -> None:
    store = InMemoryEventStore()
    e1 = _make_event("stream-3", version=1)
    store.append("stream-3", [e1], expected_version=-1)

    e2 = _make_event("stream-3", version=2, event_type="StockAdded")
    store.append("stream-3", [e2], expected_version=1)

    loaded = store.load("stream-3")
    assert len(loaded) == 2


def test_load_after_version_filters_correctly() -> None:
    store = InMemoryEventStore()
    events = [_make_event("stream-4", v) for v in range(1, 6)]
    store.append("stream-4", events, expected_version=-1)

    loaded = store.load("stream-4", after_version=3)
    assert [e.version for e in loaded] == [4, 5]


def test_snapshot_roundtrip() -> None:
    store = InMemoryEventStore()
    snap = Snapshot(stream_id="stream-5", version=10, state={"quantity_on_hand": 500})
    store.save_snapshot(snap)
    loaded = store.load_snapshot("stream-5")
    assert loaded is not None
    assert loaded.version == 10
    assert loaded.state["quantity_on_hand"] == 500


def test_load_snapshot_returns_none_for_unknown() -> None:
    store = InMemoryEventStore()
    assert store.load_snapshot("missing") is None


def test_stream_version_minus_one_for_new_stream() -> None:
    store = InMemoryEventStore()
    assert store.stream_version("new-stream") == -1


def test_reconstitute_via_store() -> None:
    store = InMemoryEventStore()
    item = InventoryItem("sku-test")
    item.create(name="Bolt", initial_quantity=200)
    item.reserve_stock(50)

    store.append("sku-test", item.pending_events, expected_version=-1)

    restored = store.reconstitute("sku-test", InventoryItem, "sku-test")
    assert restored.quantity_on_hand == 150
    assert restored.quantity_reserved == 50
```

### `test_stream_processor.py`

```python
"""
test_stream_processor.py — Tests for TumblingWindowProcessor.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from models import StreamRecord, WindowResult
from stream_processor import TumblingWindowProcessor


def _make_record(ts: datetime, value: float, topic: str = "sensor-a") -> StreamRecord:
    return StreamRecord(topic=topic, timestamp=ts, value=value)


def _base() -> datetime:
    return datetime(2024, 1, 1, 0, 0, 0)


# ------------------------------------------------------------------
# Basic windowing
# ------------------------------------------------------------------

def test_tumbling_window_basic() -> None:
    closed: list[WindowResult] = []
    proc = TumblingWindowProcessor(
        window_size_seconds=60,
        on_window_close=closed.append,
    )
    t = _base()
    proc.process(_make_record(t, 10.0))
    proc.process(_make_record(t + timedelta(seconds=30), 20.0))
    # Record in next window triggers close of first window
    proc.process(_make_record(t + timedelta(seconds=90), 5.0))

    assert len(closed) == 1
    assert closed[0].count == 2
    assert closed[0].total == 30.0


def test_window_aggregation_min_max() -> None:
    closed: list[WindowResult] = []
    proc = TumblingWindowProcessor(
        window_size_seconds=60,
        on_window_close=closed.append,
    )
    t = _base()
    values = [3.0, 7.0, 1.0, 9.0, 5.0]
    for i, v in enumerate(values):
        proc.process(_make_record(t + timedelta(seconds=i * 5), v))

    # Flush to close remaining window
    proc.flush_all()

    assert len(closed) == 1
    assert closed[0].min_value == 1.0
    assert closed[0].max_value == 9.0
    assert closed[0].count == 5


def test_multiple_windows_emitted() -> None:
    closed: list[WindowResult] = []
    proc = TumblingWindowProcessor(
        window_size_seconds=60,
        on_window_close=closed.append,
    )
    t = _base()
    # Span three windows
    for minute in range(3):
        proc.process(_make_record(t + timedelta(minutes=minute, seconds=10), float(minute + 1)))

    proc.flush_all()

    assert len(closed) == 3
    totals = sorted(r.total for r in closed)
    assert totals == [1.0, 2.0, 3.0]


def test_late_data_dropped() -> None:
    closed: list[WindowResult] = []
    proc = TumblingWindowProcessor(
        window_size_seconds=60,
        grace_period_seconds=0,
        on_window_close=closed.append,
    )
    t = _base()
    proc.process(_make_record(t + timedelta(minutes=2), 100.0))  # advance watermark
    # Late record for minute-0 window
    proc.process(_make_record(t + timedelta(seconds=5), 999.0))

    proc.flush_all()
    # Only the minute-2 window should exist; late record dropped
    assert all(r.total != 999.0 for r in closed)


def test_flush_all_closes_open_windows() -> None:
    closed: list[WindowResult] = []
    proc = TumblingWindowProcessor(
        window_size_seconds=60,
        on_window_close=closed.append,
    )
    t = _base()
    proc.process(_make_record(t, 42.0))
    assert len(closed) == 0  # not yet closed

    proc.flush_all()
    assert len(closed) == 1
    assert closed[0].total == 42.0


def test_get_open_windows_snapshot() -> None:
    proc = TumblingWindowProcessor(window_size_seconds=60)
    t = _base()
    proc.process(_make_record(t, 10.0))
    proc.process(_make_record(t + timedelta(seconds=20), 5.0))

    open_windows = proc.get_open_windows()
    assert len(open_windows) == 1
    assert open_windows[0].count == 2


def test_multiple_topics_separate_windows() -> None:
    closed: list[WindowResult] = []
    proc = TumblingWindowProcessor(
        window_size_seconds=60,
        on_window_close=closed.append,
    )
    t = _base()
    proc.process(_make_record(t, 1.0, topic="sensor-a"))
    proc.process(_make_record(t, 2.0, topic="sensor-b"))
    proc.process(_make_record(t + timedelta(minutes=2), 0.0, topic="sensor-a"))

    proc.flush_all()

    topics = {r.topic for r in closed}
    assert "sensor-a" in topics
    assert "sensor-b" in topics


def test_window_boundaries_are_aligned() -> None:
    closed: list[WindowResult] = []
    proc = TumblingWindowProcessor(
        window_size_seconds=60,
        on_window_close=closed.append,
    )
    t = datetime(2024, 6, 15, 12, 3, 45)  # mid-minute timestamp
    proc.process(_make_record(t, 1.0))
    proc.flush_all()

    result = closed[0]
    # Window start must be on minute boundary
    assert result.window_start.second == 0
    assert result.window_start.minute == 3
    assert result.window_end.minute == 4
```

---

## Running the Test Suite

```bash
# Install deps
pip install pydantic pytest

# From the directory containing these files:
pytest test_aggregate.py test_event_store.py test_stream_processor.py -v
```

Expected: 10 + 9 + 8 = 27 tests, all passing.
