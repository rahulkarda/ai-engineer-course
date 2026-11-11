# Solutions — Week 04 · Day 4
# Stream processing — windowing, exactly-once

---

## Exercise 1

**Implement a `TumblingWindowProcessor` that groups a stream of `(timestamp, value)` tuples into fixed-size time windows and computes sum and count per window.**

```python
"""
tumbling_window.py

TumblingWindowProcessor: groups a stream of (timestamp, value) tuples into
non-overlapping, fixed-duration time windows. Computes sum and count per window.

Window semantics:
  - A window covers [start, start + window_size).
  - Window start = floor(event_timestamp / window_size) * window_size.
  - All events with the same floor value belong to the same window.
  - Windows are identified by their start time (seconds since epoch).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class WindowKey:
    window_start: float  # seconds since epoch
    window_end: float    # exclusive upper bound

    def __repr__(self) -> str:
        start = datetime.fromtimestamp(self.window_start, tz=timezone.utc).isoformat()
        end = datetime.fromtimestamp(self.window_end, tz=timezone.utc).isoformat()
        return f"Window[{start}, {end})"


@dataclass
class WindowResult:
    key: WindowKey
    count: int
    total: float
    values: list[float] = field(default_factory=list, repr=False)

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0


class TumblingWindowProcessor:
    """
    Groups a stream of (timestamp, value) tuples into tumbling windows.

    Args:
        window_size_seconds: Duration of each window in seconds.
    """

    def __init__(self, window_size_seconds: float) -> None:
        if window_size_seconds <= 0:
            raise ValueError("window_size_seconds must be positive")
        self._window_size = window_size_seconds
        # window_start -> (sum, count, [values])
        self._windows: dict[float, tuple[float, int, list[float]]] = defaultdict(
            lambda: (0.0, 0, [])
        )

    def _window_start(self, timestamp: float) -> float:
        import math
        return math.floor(timestamp / self._window_size) * self._window_size

    def add(self, timestamp: float, value: float) -> None:
        """Add a single event to its window."""
        ws = self._window_start(timestamp)
        total, count, values = self._windows[ws]
        self._windows[ws] = (total + value, count + 1, values + [value])
        logger.debug(
            "Added value=%.2f to window_start=%.0f (count=%d total=%.2f)",
            value,
            ws,
            count + 1,
            total + value,
        )

    def add_batch(self, events: list[tuple[float, float]]) -> None:
        """Add a batch of (timestamp, value) tuples."""
        for ts, val in events:
            self.add(ts, val)

    def results(self) -> list[WindowResult]:
        """Return a list of WindowResult, sorted by window start time."""
        out: list[WindowResult] = []
        for ws in sorted(self._windows):
            total, count, values = self._windows[ws]
            key = WindowKey(
                window_start=ws,
                window_end=ws + self._window_size,
            )
            out.append(WindowResult(key=key, count=count, total=total, values=values))
        return out

    def get_window(self, timestamp: float) -> WindowResult | None:
        """Return the result for the window that contains `timestamp`."""
        ws = self._window_start(timestamp)
        if ws not in self._windows:
            return None
        total, count, values = self._windows[ws]
        return WindowResult(
            key=WindowKey(ws, ws + self._window_size),
            count=count,
            total=total,
            values=values,
        )

    def close_window(self, timestamp: float) -> WindowResult | None:
        """
        Close and remove the window for `timestamp`, returning its result.
        Useful for emitting windows downstream once they are complete.
        """
        ws = self._window_start(timestamp)
        if ws not in self._windows:
            return None
        total, count, values = self._windows.pop(ws)
        result = WindowResult(
            key=WindowKey(ws, ws + self._window_size),
            count=count,
            total=total,
            values=values,
        )
        logger.info("Closed window %s: count=%d total=%.2f", result.key, count, total)
        return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_tumbling_window_groups_events_correctly() -> None:
    proc = TumblingWindowProcessor(window_size_seconds=60.0)

    # Window 1: timestamps 0-59
    proc.add(0.0, 10.0)
    proc.add(30.0, 20.0)
    proc.add(59.9, 5.0)

    # Window 2: timestamps 60-119
    proc.add(60.0, 100.0)
    proc.add(90.0, 50.0)

    results = proc.results()
    assert len(results) == 2

    w1, w2 = results
    assert w1.count == 3
    assert w1.total == 35.0
    assert w2.count == 2
    assert w2.total == 150.0


def test_tumbling_window_single_event_per_window() -> None:
    proc = TumblingWindowProcessor(window_size_seconds=10.0)
    proc.add(5.0, 1.0)
    proc.add(15.0, 2.0)
    proc.add(25.0, 3.0)

    results = proc.results()
    assert len(results) == 3
    totals = [r.total for r in results]
    assert totals == [1.0, 2.0, 3.0]


def test_tumbling_window_all_events_same_window() -> None:
    proc = TumblingWindowProcessor(window_size_seconds=100.0)
    for i in range(10):
        proc.add(float(i), float(i))

    results = proc.results()
    assert len(results) == 1
    assert results[0].count == 10
    assert results[0].total == sum(range(10))


def test_tumbling_window_mean() -> None:
    proc = TumblingWindowProcessor(window_size_seconds=60.0)
    proc.add(0.0, 10.0)
    proc.add(10.0, 30.0)
    result = proc.results()[0]
    assert result.mean == 20.0


def test_close_window_removes_it() -> None:
    proc = TumblingWindowProcessor(window_size_seconds=60.0)
    proc.add(0.0, 42.0)
    result = proc.close_window(0.0)
    assert result is not None
    assert result.total == 42.0
    # Window should no longer exist
    assert proc.get_window(0.0) is None
```

---

## Exercise 2

**Add late-data handling: events arriving up to `grace_period` seconds after window close are still included. Events beyond grace_period are dropped with a log warning.**

```python
"""
tumbling_window_grace.py

Extends TumblingWindowProcessor with late-data handling.

When an event arrives, the processor determines the window it belongs to.
If that window has already been "emitted" (closed) but within the grace period,
the event is still accepted and the window result is updated.
If the window was closed more than grace_period seconds ago (measured in
stream time, i.e., the maximum event timestamp seen), the event is dropped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from collections import defaultdict
import math

from tumbling_window import TumblingWindowProcessor, WindowKey, WindowResult

logger = logging.getLogger(__name__)


class GracefulTumblingWindowProcessor(TumblingWindowProcessor):
    """
    TumblingWindowProcessor with a grace period for late-arriving data.

    Tracks the high-water mark (maximum event timestamp seen). A window
    is considered "closed" when the high-water mark passes its end time.
    Events for a closed window are accepted if they arrive within
    `grace_period` seconds of the window's end; otherwise they are dropped.

    Args:
        window_size_seconds: Width of each tumbling window.
        grace_period: Extra seconds after window close during which late
                      events are still accepted.
    """

    def __init__(
        self, window_size_seconds: float, grace_period: float = 0.0
    ) -> None:
        super().__init__(window_size_seconds)
        self._grace_period = grace_period
        self._watermark: float = float("-inf")
        self._dropped: int = 0

    def _is_late(self, window_end: float) -> bool:
        """True if the window is beyond the grace period."""
        return self._watermark > window_end + self._grace_period

    def add(self, timestamp: float, value: float) -> None:
        # Advance watermark
        if timestamp > self._watermark:
            self._watermark = timestamp

        ws = self._window_start(timestamp)
        window_end = ws + self._window_size

        if self._is_late(window_end):
            logger.warning(
                "Dropped late event: timestamp=%.2f window_end=%.2f watermark=%.2f grace=%.2f",
                timestamp,
                window_end,
                self._watermark,
                self._grace_period,
            )
            self._dropped += 1
            return

        super().add(timestamp, value)

        if self._watermark > window_end:
            logger.debug(
                "Accepted late event within grace period: "
                "timestamp=%.2f window_end=%.2f watermark=%.2f",
                timestamp,
                window_end,
                self._watermark,
            )

    @property
    def watermark(self) -> float:
        return self._watermark

    @property
    def dropped_count(self) -> int:
        return self._dropped


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

import pytest


def test_in_order_events_all_accepted() -> None:
    proc = GracefulTumblingWindowProcessor(
        window_size_seconds=60.0, grace_period=10.0
    )
    proc.add(0.0, 1.0)
    proc.add(30.0, 2.0)
    proc.add(60.0, 3.0)  # first event in next window, closes first window

    assert proc.dropped_count == 0
    results = proc.results()
    assert len(results) == 2


def test_late_event_within_grace_period_accepted() -> None:
    proc = GracefulTumblingWindowProcessor(
        window_size_seconds=60.0, grace_period=10.0
    )
    # Window [0, 60)
    proc.add(30.0, 5.0)

    # Advance watermark past window end (t=65)
    proc.add(65.0, 100.0)  # window [60, 120)

    # Late event for window [0,60) arriving at ts=55 (within grace: 65 < 60+10)
    proc.add(55.0, 50.0)

    assert proc.dropped_count == 0

    w0 = proc.get_window(30.0)
    assert w0 is not None
    assert w0.count == 2
    assert w0.total == 55.0


def test_late_event_beyond_grace_period_dropped() -> None:
    proc = GracefulTumblingWindowProcessor(
        window_size_seconds=60.0, grace_period=5.0
    )
    proc.add(30.0, 10.0)
    # Advance watermark to 70 (window [0,60) closed, grace expires at 60+5=65)
    proc.add(70.0, 100.0)

    # Late event for window [0,60) — watermark=70 > 60+5=65 => drop
    proc.add(45.0, 999.0)

    assert proc.dropped_count == 1

    # Window [0,60) should still have only the first event
    w0 = proc.get_window(30.0)
    assert w0 is not None
    assert w0.count == 1
    assert w0.total == 10.0


def test_zero_grace_period_no_late_events_accepted() -> None:
    proc = GracefulTumblingWindowProcessor(
        window_size_seconds=60.0, grace_period=0.0
    )
    proc.add(30.0, 1.0)
    proc.add(90.0, 1.0)  # advances watermark to 90, closes [0,60)
    proc.add(10.0, 999.0)  # late for [0,60); watermark=90 > 60+0=60 => drop

    assert proc.dropped_count == 1
```

---

## Exercise 3

**Implement a deduplication buffer: before processing, check a `seen_ids` set. If duplicate, skip. Show this provides at-least-once-becomes-effectively-once semantics.**

```python
"""
deduplication.py

Deduplication buffer for stream processing.

At-least-once delivery guarantees that every message is delivered at least
once but may be delivered more than once (e.g., after a broker restart or
consumer crash). By tracking seen event IDs in a buffer, we convert
at-least-once delivery into effectively-once processing at the application layer.

This is NOT exactly-once in the distributed sense (it is not atomic), but it
eliminates duplicates that are detectable via a stable event ID.

Bounded buffer implementation: once the buffer exceeds `max_size` entries,
the oldest half is evicted (LRU-lite). This prevents unbounded memory growth
while still catching duplicates that arrive within the buffer window.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class StreamEvent:
    event_id: str
    timestamp: float
    value: float


ProcessorFn = Callable[[StreamEvent], None]


class DeduplicationBuffer:
    """
    Tracks seen event IDs to skip duplicates.

    Uses an OrderedDict as a bounded FIFO cache. When the buffer is full,
    the oldest entries are evicted to make room for new IDs.

    Args:
        max_size: Maximum number of event IDs to remember.
                  Older IDs are evicted when the buffer is full.
    """

    def __init__(self, max_size: int = 10_000) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._max_size = max_size
        self._duplicate_count: int = 0
        self._processed_count: int = 0

    def process(self, event: StreamEvent, handler: ProcessorFn) -> bool:
        """
        Process event if not seen before.

        Returns:
            True if the event was processed (first time seen).
            False if the event was a duplicate and skipped.
        """
        if event.event_id in self._seen:
            logger.debug("Duplicate skipped: event_id=%s", event.event_id)
            self._duplicate_count += 1
            return False

        # Record as seen before processing to handle handler exceptions
        self._seen[event.event_id] = None
        if len(self._seen) > self._max_size:
            self._evict()

        handler(event)
        self._processed_count += 1
        return True

    def _evict(self) -> None:
        """Remove oldest half of the buffer."""
        evict_count = self._max_size // 2
        for _ in range(evict_count):
            if self._seen:
                self._seen.popitem(last=False)
        logger.debug("Evicted %d entries from dedup buffer", evict_count)

    @property
    def duplicate_count(self) -> int:
        return self._duplicate_count

    @property
    def processed_count(self) -> int:
        return self._processed_count

    def __len__(self) -> int:
        return len(self._seen)


# ---------------------------------------------------------------------------
# Demonstration: integrating deduplication with TumblingWindowProcessor
# ---------------------------------------------------------------------------


from tumbling_window import TumblingWindowProcessor


class DeduplicatingWindowProcessor:
    """
    Wraps TumblingWindowProcessor with a deduplication buffer.
    Demonstrates at-least-once -> effectively-once semantics.
    """

    def __init__(
        self,
        window_size_seconds: float,
        dedup_buffer_size: int = 10_000,
    ) -> None:
        self._window = TumblingWindowProcessor(window_size_seconds)
        self._dedup = DeduplicationBuffer(max_size=dedup_buffer_size)

    def process(self, event: StreamEvent) -> bool:
        return self._dedup.process(
            event,
            handler=lambda e: self._window.add(e.timestamp, e.value),
        )

    @property
    def results(self):
        return self._window.results()

    @property
    def duplicate_count(self) -> int:
        return self._dedup.duplicate_count

    @property
    def processed_count(self) -> int:
        return self._dedup.processed_count


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

import pytest


def test_dedup_buffer_processes_unique_events() -> None:
    buf = DeduplicationBuffer()
    results: list[float] = []

    def handler(event: StreamEvent) -> None:
        results.append(event.value)

    events = [StreamEvent(f"id-{i}", float(i), float(i * 10)) for i in range(5)]
    for e in events:
        buf.process(e, handler)

    assert results == [0.0, 10.0, 20.0, 30.0, 40.0]
    assert buf.processed_count == 5
    assert buf.duplicate_count == 0


def test_dedup_buffer_skips_duplicates() -> None:
    buf = DeduplicationBuffer()
    processed: list[str] = []

    def handler(event: StreamEvent) -> None:
        processed.append(event.event_id)

    e = StreamEvent("evt-abc", 1.0, 42.0)
    buf.process(e, handler)
    buf.process(e, handler)  # duplicate
    buf.process(e, handler)  # duplicate again

    assert processed == ["evt-abc"]
    assert buf.processed_count == 1
    assert buf.duplicate_count == 2


def test_at_least_once_becomes_effectively_once() -> None:
    """
    Simulate a broker replaying events after a crash (at-least-once).
    Deduplication ensures each event is processed exactly once.
    """
    proc = DeduplicatingWindowProcessor(window_size_seconds=60.0)

    # Original stream
    events = [
        StreamEvent("e1", 10.0, 100.0),
        StreamEvent("e2", 20.0, 200.0),
        StreamEvent("e3", 30.0, 300.0),
    ]

    # Simulate at-least-once: broker replays events e2 and e3 after crash
    replayed_events = [
        StreamEvent("e1", 10.0, 100.0),
        StreamEvent("e2", 20.0, 200.0),
        StreamEvent("e3", 30.0, 300.0),
        StreamEvent("e2", 20.0, 200.0),  # replay: duplicate
        StreamEvent("e3", 30.0, 300.0),  # replay: duplicate
    ]

    for event in replayed_events:
        proc.process(event)

    # Despite 5 deliveries, only 3 unique events were processed
    assert proc.processed_count == 3
    assert proc.duplicate_count == 2

    # Window result is correct (not inflated by duplicates)
    results = proc.results
    assert len(results) == 1
    assert results[0].count == 3
    assert results[0].total == 600.0


def test_dedup_buffer_eviction_allows_new_ids() -> None:
    """Buffer eviction must not cause current IDs to be forgotten."""
    buf = DeduplicationBuffer(max_size=4)
    counts: dict[str, int] = {}

    def handler(event: StreamEvent) -> None:
        counts[event.event_id] = counts.get(event.event_id, 0) + 1

    # Fill buffer to max
    for i in range(4):
        buf.process(StreamEvent(f"id-{i}", 0.0, 0.0), handler)

    # One more triggers eviction of oldest 2 (id-0, id-1)
    buf.process(StreamEvent("id-4", 0.0, 0.0), handler)

    # id-4 is new, must be processed
    assert counts.get("id-4", 0) == 1

    # id-2 and id-3 are still in buffer (not evicted), so duplicates are caught
    buf.process(StreamEvent("id-2", 0.0, 0.0), handler)
    assert counts.get("id-2", 0) == 1  # not double-processed


def test_dedup_buffer_with_all_unique_events_no_duplicates_reported() -> None:
    buf = DeduplicationBuffer(max_size=100)
    processed: list[str] = []

    def handler(event: StreamEvent) -> None:
        processed.append(event.event_id)

    for i in range(50):
        buf.process(StreamEvent(f"unique-{i}", float(i), float(i)), handler)

    assert len(processed) == 50
    assert buf.duplicate_count == 0
```

---

## Discussion notes

- **Tumbling vs. sliding vs. session windows**:
  - *Tumbling*: fixed-size, non-overlapping (implemented here). Every event belongs to exactly one window.
  - *Sliding*: fixed-size, overlapping by a hop interval. One event may contribute to multiple windows. More expensive: O(window_size/hop) space per event.
  - *Session*: dynamic size, closed after an inactivity gap. Suitable for user session analytics. Complex to implement in a distributed setting.

- **Watermark strategy**: The watermark (maximum observed timestamp) is used to decide when a window is "complete enough" to emit. In production systems (Flink, Spark Structured Streaming), watermarks are set heuristically: `watermark = max_event_time - allowed_lateness`. This is exactly what `GracefulTumblingWindowProcessor` implements.

- **Grace period vs. unbounded late data**: Grace periods bound the amount of late data memory. Without a bound, you must keep all windows open indefinitely. A typical grace period is 1-5 minutes for near-real-time systems; hours are possible but require persistent window state (e.g., RocksDB in Flink).

- **Exactly-once vs. effectively-once**:
  - *Exactly-once* in the distributed sense requires atomic writes to both the state store and the output sink — achieved in Flink/Kafka with two-phase commit.
  - *Effectively-once* (shown here) uses application-level deduplication. It is weaker: if the handler fails after marking the event as seen but before completing, the event is lost, not retried. In practice, this is acceptable for analytics workloads.
  - For financial systems (billing, inventory), use Kafka's transactional API for true end-to-end exactly-once.

- **Deduplication buffer size**: The buffer must span the maximum redelivery window of your broker. If a Kafka consumer can receive a replay of up to 10 000 messages after a restart, the buffer must be at least that large. Eviction of old IDs is safe only when the broker's redelivery window has passed.

- **Event ID design**: Stable, content-derived IDs (e.g., hash of `(producer_id, sequence_number)`) are better than random UUIDs for deduplication because two retries of the same logical event will produce the same ID. Random UUIDs per delivery mean duplicates slip through.

- **Window state persistence**: The in-memory processors shown here lose state on restart. In production, periodically checkpoint window state to durable storage (Redis, RocksDB, S3) so recovery replays only the delta since the last checkpoint.

---
