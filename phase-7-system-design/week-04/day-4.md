# Phase 7 · Week 04 · Day 4 — Stream Processing: Windowing, Watermarks, and Exactly-Once Semantics

> Objective: Build a stream processor with tumbling, sliding, and session windows, implement watermark-based late-data handling, and understand exactly-once semantics via idempotent consumers and transactional producers.  ·  Time: ~1 hr  ·  Prereqs: [[week-04/day-1]], [[week-04/day-3]]

---

## Concept

Stream processing transforms an unbounded sequence of events in near-real-time. Unlike batch processing (bounded dataset, run once), stream processing operates on an infinite, continuous flow of data with latency goals in the milliseconds-to-seconds range.

### Stateless vs Stateful Processing

**Stateless** — each event is processed independently. No memory of previous events.

```
Examples:
  Filter: keep only events where amount > 100
  Map: enrich event with a lookup result
  Route: forward to different downstream topics based on content
```

**Stateful** — computation depends on a window of events accumulated over time.

```
Examples:
  Count: how many orders arrived in the last 5 minutes?
  Sum: total revenue per merchant per hour
  Join: pair a click event with a purchase event from the same user
  Anomaly: flag if the last 10 events from IP X all fail auth
```

Stateful processing requires **windows** and a mechanism to handle time.

### Two Notions of Time

| Time Type | Definition | Use For |
|---|---|---|
| Event time | When the event actually occurred (in the payload) | Accurate analytics |
| Processing time | When the event arrives at the processor | Simpler, but skewed by network/retries |

Skew: an event timestamped 09:00:00 may arrive at the processor at 09:00:45. If you window by processing time, it falls in the 09:00-09:01 window but should logically be in 09:00-09:01 too — harmless here. But if it arrives at 09:01:30 due to network delay, processing-time windowing puts it in the wrong minute. Event-time windowing handles this correctly by using the timestamp embedded in the event.

### Window Types

**Tumbling window** — fixed-size, non-overlapping buckets. Every event belongs to exactly one window.

```
Events: e1(t=0) e2(t=3) e3(t=6) e4(t=9) e5(t=11)
Window size: 5s

Window [0,5):  e1, e2          -> aggregate -> result
Window [5,10): e3, e4          -> aggregate -> result
Window [10,15):e5               -> aggregate -> result
```

**Sliding window** — fixed size, moves by a step < size. Events may belong to multiple windows.

```
Window size: 10s, step: 5s

Window [0,10):  e1, e2, e3, e4
Window [5,15):  e3, e4, e5
Window [10,20): e5
```

**Session window** — dynamically sized. A session starts on the first event and extends as long as events arrive within a gap timeout. Closes when no event arrives for `gap` seconds.

```
gap = 3s
Events: e1(t=0) e2(t=1) e3(t=2)   e4(t=10) e5(t=11)

Session 1: [0,2] -> {e1, e2, e3}   (gap of 8s follows -> close)
Session 2: [10,11] -> {e4, e5}
```

### Watermarks

A **watermark** is a signal that says "all events with timestamp <= W have arrived." It allows the processor to close windows confidently without waiting forever.

The watermark advances based on the maximum event timestamp seen, minus an allowed lateness:

```
watermark = max_event_timestamp_seen - allowed_lateness
```

When `watermark > window_end`, that window is closed and its result emitted.

**Late data** — events that arrive after the watermark has passed their window boundary. Handling options:

1. **Drop** — simplest; accept small inaccuracy
2. **Re-trigger** — update the window result (produces corrections)
3. **Side output** — route to a separate late-data stream for reconciliation

### Exactly-Once Semantics

Recall from Day 1:
- At-most-once: offset committed before processing. May lose data.
- At-least-once: offset committed after processing. May re-process. Requires idempotency.
- Exactly-once: each input event affects output exactly once, even on failure.

**Achieving exactly-once conceptually:**

1. **Idempotent consumer**: processing the same input event N times produces the same output as processing it once. Use deterministic output keys and upsert semantics.

2. **Transactional producer** (e.g., Kafka transactions): the write to the output topic and the commit of the input offset happen atomically. Either both succeed or neither does.

```
begin_transaction()
  process(event)
  write(output_topic, result)    <- atomic
  commit_offset(input_topic, offset)  <- atomic
commit_transaction()
--- or ---
abort_transaction()   <- on failure, neither write nor commit persists
```

3. **Two-phase commit** between state store and broker: rarely used in practice due to complexity.

Most production systems use at-least-once + idempotency because true exactly-once adds coordination overhead. Kafka Streams and Flink support exactly-once as a configuration option.

---

## How It Works Internally

### Window State Storage

```
State backend (in-memory or RocksDB):
  key: (window_start, window_end, group_key)
  value: accumulated WindowState { count, sum, ... }

On each event:
  1. Determine which window(s) it belongs to
  2. Load WindowState for that window key
  3. Update WindowState
  4. Store updated WindowState
  5. Check if watermark has passed window_end -> emit if yes
```

### Watermark Propagation

In a distributed system, each partition tracks its own watermark. The global watermark is the minimum across all partitions (the slowest partition determines how far all windows can advance — prevents closing a window while lagging partitions still have earlier events).

```
Partition 0: max_ts = 1000 -> watermark = 970
Partition 1: max_ts = 800  -> watermark = 770
Partition 2: max_ts = 950  -> watermark = 920

Global watermark = min(970, 770, 920) = 770

A window ending at 900 cannot be closed yet (watermark 770 < 900).
```

### Idempotency via Deduplication Key

```
Idempotent write:
  output_key = deterministic_function(input_event_id, window_id)
  upsert(output_store, key=output_key, value=result)
  -- upsert: if key exists, overwrite; if not, insert
  -- Running twice with same input produces same key -> same value -> idempotent
```

---

## Worked Example

```python
"""
stream_processor.py — In-memory stream processor with:

- Tumbling windows
- Sliding windows
- Session windows
- Watermark-based late-data handling
- Exactly-once simulation via idempotent upserts

Run:
    python stream_processor.py

No external dependencies beyond Python 3.12 stdlib.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("stream_processor")


# ---------------------------------------------------------------------------
# Input event
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StreamEvent:
    """An event with an event-time timestamp."""
    event_id: str
    event_time: float    # epoch seconds — the time the event OCCURRED
    key: str             # e.g., user_id, merchant_id — groups events
    value: float         # metric to aggregate (e.g., transaction amount)
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, key: str, event_time: float, value: float) -> "StreamEvent":
        return cls(
            event_id=str(uuid.uuid4()),
            event_time=event_time,
            key=key,
            value=value,
        )


# ---------------------------------------------------------------------------
# Window result — output of aggregation
# ---------------------------------------------------------------------------

@dataclass
class WindowResult:
    window_id: str
    window_type: str
    key: str
    window_start: float
    window_end: float
    count: int
    sum_value: float
    min_value: float
    max_value: float
    is_final: bool = True     # False if result may be updated by late data

    @property
    def average(self) -> float:
        return self.sum_value / self.count if self.count else 0.0


# ---------------------------------------------------------------------------
# Window accumulator — mutable state during window lifetime
# ---------------------------------------------------------------------------

@dataclass
class WindowAccumulator:
    window_start: float
    window_end: float
    key: str
    events: list[StreamEvent] = field(default_factory=list)

    def add(self, event: StreamEvent) -> None:
        self.events.append(event)

    def to_result(self, window_type: str, is_final: bool = True) -> WindowResult:
        values = [e.value for e in self.events]
        window_id = f"{window_type}:{self.key}:{self.window_start:.0f}-{self.window_end:.0f}"
        return WindowResult(
            window_id=window_id,
            window_type=window_type,
            key=self.key,
            window_start=self.window_start,
            window_end=self.window_end,
            count=len(self.events),
            sum_value=sum(values),
            min_value=min(values) if values else 0.0,
            max_value=max(values) if values else 0.0,
            is_final=is_final,
        )

    @property
    def is_empty(self) -> bool:
        return len(self.events) == 0


# ---------------------------------------------------------------------------
# Watermark tracker
# ---------------------------------------------------------------------------

class WatermarkTracker:
    """
    Tracks the current watermark across all observed event times.
    watermark = max_event_time_seen - allowed_lateness
    """

    def __init__(self, allowed_lateness: float = 5.0) -> None:
        self.allowed_lateness = allowed_lateness
        self._max_event_time: float = 0.0

    def update(self, event_time: float) -> float:
        """Update watermark based on a new event time. Returns new watermark."""
        if event_time > self._max_event_time:
            self._max_event_time = event_time
        return self.current

    @property
    def current(self) -> float:
        return max(0.0, self._max_event_time - self.allowed_lateness)


# ---------------------------------------------------------------------------
# Tumbling window processor
# ---------------------------------------------------------------------------

class TumblingWindowProcessor:
    """
    Emits one result per (key, window) when watermark advances past window_end.
    Windows are non-overlapping buckets of fixed duration.
    """

    def __init__(
        self,
        window_size: float,
        watermark_tracker: WatermarkTracker,
        on_result: Callable[[WindowResult], None],
    ) -> None:
        self.window_size = window_size
        self._tracker = watermark_tracker
        self._on_result = on_result
        # (key, window_start) -> WindowAccumulator
        self._windows: dict[tuple[str, float], WindowAccumulator] = {}
        self._emitted: set[tuple[str, float]] = set()
        self._log = logging.getLogger("tumbling")

    def _window_start_for(self, event_time: float) -> float:
        return (event_time // self.window_size) * self.window_size

    def process(self, event: StreamEvent) -> None:
        wm = self._tracker.update(event.event_time)

        w_start = self._window_start_for(event.event_time)
        w_end = w_start + self.window_size

        key = (event.key, w_start)
        if key not in self._windows:
            self._windows[key] = WindowAccumulator(w_start, w_end, event.key)

        if (event.key, w_start) in self._emitted:
            self._log.debug(
                "Late event for closed window [%.0f, %.0f] key=%s — dropped",
                w_start, w_end, event.key,
            )
            return

        self._windows[key].add(event)
        self._emit_ready_windows(wm)

    def _emit_ready_windows(self, watermark: float) -> None:
        to_emit = [
            k for k, acc in self._windows.items()
            if acc.window_end <= watermark and k not in self._emitted
        ]
        for k in to_emit:
            acc = self._windows[k]
            if not acc.is_empty:
                result = acc.to_result("tumbling")
                self._on_result(result)
                self._log.debug(
                    "Emitted tumbling window [%.0f,%.0f) key=%s count=%d sum=%.2f",
                    acc.window_start, acc.window_end, acc.key,
                    result.count, result.sum_value,
                )
            self._emitted.add(k)

    def flush(self) -> None:
        """Force-emit all open windows (end of stream)."""
        for k, acc in self._windows.items():
            if k not in self._emitted and not acc.is_empty:
                result = acc.to_result("tumbling", is_final=True)
                self._on_result(result)
                self._emitted.add(k)


# ---------------------------------------------------------------------------
# Sliding window processor
# ---------------------------------------------------------------------------

class SlidingWindowProcessor:
    """
    Each event may belong to multiple overlapping windows.
    window_size / step gives the overlap factor.
    """

    def __init__(
        self,
        window_size: float,
        step: float,
        watermark_tracker: WatermarkTracker,
        on_result: Callable[[WindowResult], None],
    ) -> None:
        self.window_size = window_size
        self.step = step
        self._tracker = watermark_tracker
        self._on_result = on_result
        self._windows: dict[tuple[str, float], WindowAccumulator] = {}
        self._emitted: set[tuple[str, float]] = set()
        self._log = logging.getLogger("sliding")

    def _windows_for(self, event_time: float) -> list[tuple[float, float]]:
        """Return all (start, end) windows that contain event_time."""
        import math
        # The first window whose start <= event_time
        first_start = math.floor((event_time - self.window_size) / self.step + 1) * self.step
        windows = []
        start = first_start
        while start <= event_time:
            end = start + self.window_size
            if end > event_time:
                windows.append((start, end))
            start += self.step
        return windows

    def process(self, event: StreamEvent) -> None:
        wm = self._tracker.update(event.event_time)
        for w_start, w_end in self._windows_for(event.event_time):
            k = (event.key, w_start)
            if k in self._emitted:
                continue
            if k not in self._windows:
                self._windows[k] = WindowAccumulator(w_start, w_end, event.key)
            self._windows[k].add(event)
        self._emit_ready_windows(wm)

    def _emit_ready_windows(self, watermark: float) -> None:
        to_emit = [
            k for k, acc in self._windows.items()
            if acc.window_end <= watermark and k not in self._emitted
        ]
        for k in to_emit:
            acc = self._windows[k]
            if not acc.is_empty:
                result = acc.to_result("sliding")
                self._on_result(result)
            self._emitted.add(k)

    def flush(self) -> None:
        for k, acc in self._windows.items():
            if k not in self._emitted and not acc.is_empty:
                result = acc.to_result("sliding", is_final=True)
                self._on_result(result)
                self._emitted.add(k)


# ---------------------------------------------------------------------------
# Session window processor
# ---------------------------------------------------------------------------

class SessionWindowProcessor:
    """
    Session: contiguous run of events where consecutive events are within gap seconds.
    A session closes when no event arrives within `gap` seconds (by event time).
    Watermark drives session closure.
    """

    def __init__(
        self,
        gap: float,
        watermark_tracker: WatermarkTracker,
        on_result: Callable[[WindowResult], None],
    ) -> None:
        self.gap = gap
        self._tracker = watermark_tracker
        self._on_result = on_result
        # key -> open WindowAccumulator (session)
        self._open_sessions: dict[str, WindowAccumulator] = {}
        self._log = logging.getLogger("session")

    def process(self, event: StreamEvent) -> None:
        wm = self._tracker.update(event.event_time)

        if event.key in self._open_sessions:
            session = self._open_sessions[event.key]
            if event.event_time <= session.window_end + self.gap:
                # Extend session
                session.add(event)
                session.window_end = max(session.window_end, event.event_time)
            else:
                # Gap exceeded — close old session, start new one
                if not session.is_empty:
                    self._on_result(session.to_result("session"))
                    self._log.debug(
                        "Session closed for key=%s [%.0f,%.0f) count=%d",
                        event.key, session.window_start, session.window_end, len(session.events),
                    )
                new_session = WindowAccumulator(
                    window_start=event.event_time,
                    window_end=event.event_time,
                    key=event.key,
                )
                new_session.add(event)
                self._open_sessions[event.key] = new_session
        else:
            session = WindowAccumulator(
                window_start=event.event_time,
                window_end=event.event_time,
                key=event.key,
            )
            session.add(event)
            self._open_sessions[event.key] = session

        # Close sessions whose window_end is far enough behind the watermark
        self._close_stale_sessions(wm)

    def _close_stale_sessions(self, watermark: float) -> None:
        to_close = [
            key for key, sess in self._open_sessions.items()
            if sess.window_end + self.gap <= watermark
        ]
        for key in to_close:
            sess = self._open_sessions.pop(key)
            if not sess.is_empty:
                self._on_result(sess.to_result("session"))

    def flush(self) -> None:
        for key, sess in list(self._open_sessions.items()):
            if not sess.is_empty:
                self._on_result(sess.to_result("session"))
        self._open_sessions.clear()


# ---------------------------------------------------------------------------
# Idempotent result store — simulates exactly-once output
# ---------------------------------------------------------------------------

class IdempotentResultStore:
    """
    Results are stored by window_id.
    Writing the same window_id twice overwrites (upsert = idempotent).
    Simulates the exactly-once write side: if the processor crashes and replays
    the same input events, the same window_ids are computed and the upsert
    produces no additional rows.
    """

    def __init__(self) -> None:
        self._store: dict[str, WindowResult] = {}
        self._write_count: int = 0
        self._log = logging.getLogger("result_store")

    def upsert(self, result: WindowResult) -> None:
        existed = result.window_id in self._store
        self._store[result.window_id] = result
        self._write_count += 1
        if existed:
            self._log.debug("Idempotent overwrite: %s", result.window_id)
        else:
            self._log.debug("New result: %s", result.window_id)

    def get(self, window_id: str) -> WindowResult | None:
        return self._store.get(window_id)

    def all_results(self) -> list[WindowResult]:
        return list(self._store.values())

    @property
    def unique_windows(self) -> int:
        return len(self._store)

    @property
    def total_writes(self) -> int:
        return self._write_count


# ---------------------------------------------------------------------------
# Demo: tumbling windows
# ---------------------------------------------------------------------------

def demo_tumbling(result_store: IdempotentResultStore) -> None:
    logger.info("=== Tumbling Window Demo ===")

    tracker = WatermarkTracker(allowed_lateness=2.0)
    processor = TumblingWindowProcessor(
        window_size=10.0,
        watermark_tracker=tracker,
        on_result=result_store.upsert,
    )

    # Events in two time buckets: [0,10) and [10,20)
    events = [
        StreamEvent.create("merchant-A", 1.0, 100.0),
        StreamEvent.create("merchant-A", 3.0, 200.0),
        StreamEvent.create("merchant-A", 8.0, 50.0),
        StreamEvent.create("merchant-A", 12.0, 300.0),   # new window
        StreamEvent.create("merchant-A", 15.0, 150.0),
        # This event is late (falls in [0,10) but watermark will pass it)
        StreamEvent.create("merchant-A", 25.0, 999.0),   # advance watermark
    ]

    for event in events:
        processor.process(event)

    processor.flush()

    results = [r for r in result_store.all_results() if r.key == "merchant-A"]
    for r in sorted(results, key=lambda x: x.window_start):
        logger.info(
            "Tumbling [%.0f,%.0f): count=%d sum=%.2f avg=%.2f",
            r.window_start, r.window_end, r.count, r.sum_value, r.average,
        )


# ---------------------------------------------------------------------------
# Demo: sliding windows
# ---------------------------------------------------------------------------

def demo_sliding(result_store: IdempotentResultStore) -> None:
    logger.info("=== Sliding Window Demo ===")

    tracker = WatermarkTracker(allowed_lateness=1.0)
    processor = SlidingWindowProcessor(
        window_size=10.0,
        step=5.0,
        watermark_tracker=tracker,
        on_result=result_store.upsert,
    )

    for t, v in [(1, 10), (4, 20), (7, 30), (11, 40), (14, 50), (25, 0)]:
        processor.process(StreamEvent.create("user-X", float(t), float(v)))

    processor.flush()

    results = [r for r in result_store.all_results() if r.key == "user-X" and r.window_type == "sliding"]
    logger.info("Sliding window results: %d windows emitted", len(results))
    for r in sorted(results, key=lambda x: x.window_start):
        logger.info(
            "  Sliding [%.0f,%.0f): count=%d sum=%.2f",
            r.window_start, r.window_end, r.count, r.sum_value,
        )


# ---------------------------------------------------------------------------
# Demo: session windows
# ---------------------------------------------------------------------------

def demo_session(result_store: IdempotentResultStore) -> None:
    logger.info("=== Session Window Demo ===")

    tracker = WatermarkTracker(allowed_lateness=2.0)
    processor = SessionWindowProcessor(
        gap=5.0,
        watermark_tracker=tracker,
        on_result=result_store.upsert,
    )

    # Session 1: events at t=0,2,4 — gap after t=4
    # Session 2: events at t=15,17 — gap after t=17
    # Watermark advance event at t=30
    events_by_time = [
        (0, 1.0), (2, 2.0), (4, 3.0),  # session 1
        (15, 5.0), (17, 6.0),           # session 2
        (30, 0.0),                       # advance watermark only
    ]

    for t, v in events_by_time:
        processor.process(StreamEvent.create("session-user", float(t), v))

    processor.flush()

    results = [r for r in result_store.all_results() if r.key == "session-user"]
    logger.info("Session results: %d sessions", len(results))
    for r in sorted(results, key=lambda x: x.window_start):
        logger.info(
            "  Session [%.0f,%.0f): count=%d sum=%.2f",
            r.window_start, r.window_end, r.count, r.sum_value,
        )

    assert len(results) >= 2, f"Expected at least 2 sessions, got {len(results)}"


# ---------------------------------------------------------------------------
# Demo: exactly-once via idempotent replay
# ---------------------------------------------------------------------------

def demo_exactly_once(result_store: IdempotentResultStore) -> None:
    logger.info("=== Exactly-Once Idempotency Demo ===")

    tracker = WatermarkTracker(allowed_lateness=1.0)
    processor = TumblingWindowProcessor(
        window_size=10.0,
        watermark_tracker=tracker,
        on_result=result_store.upsert,
    )

    events = [
        StreamEvent.create("idem-key", 1.0, 100.0),
        StreamEvent.create("idem-key", 3.0, 200.0),
        StreamEvent.create("idem-key", 20.0, 0.0),  # advance watermark
    ]

    writes_before = result_store.total_writes
    unique_before = result_store.unique_windows

    # First processing pass
    for e in events:
        processor.process(e)
    processor.flush()

    writes_after_first = result_store.total_writes
    unique_after_first = result_store.unique_windows

    # Simulate crash + replay: process same events again
    # A new processor instance starts from the same input events (at-least-once re-delivery)
    tracker2 = WatermarkTracker(allowed_lateness=1.0)
    processor2 = TumblingWindowProcessor(
        window_size=10.0,
        watermark_tracker=tracker2,
        on_result=result_store.upsert,  # same idempotent store
    )
    for e in events:
        processor2.process(e)
    processor2.flush()

    writes_after_replay = result_store.total_writes
    unique_after_replay = result_store.unique_windows

    logger.info(
        "Unique windows after first pass: %d, after replay: %d (must be same)",
        unique_after_first - unique_before,
        unique_after_replay - unique_before,
    )
    assert (unique_after_first - unique_before) == (unique_after_replay - unique_before), \
        "Idempotency violated: replay created new windows"
    logger.info(
        "Total writes: %d (more than unique windows is fine — upserts, not appends)",
        writes_after_replay - writes_before,
    )


# ---------------------------------------------------------------------------
# Demo: late data handling
# ---------------------------------------------------------------------------

def demo_late_data() -> None:
    logger.info("=== Late Data Handling Demo ===")

    dropped_late: list[StreamEvent] = []

    def on_result(r: WindowResult) -> None:
        logger.info(
            "Emitting %s window [%.0f,%.0f): count=%d sum=%.2f",
            r.window_type, r.window_start, r.window_end, r.count, r.sum_value,
        )

    tracker = WatermarkTracker(allowed_lateness=3.0)
    processor = TumblingWindowProcessor(
        window_size=10.0,
        watermark_tracker=tracker,
        on_result=on_result,
    )

    # Normal events in [0,10)
    processor.process(StreamEvent.create("key", 1.0, 10.0))
    processor.process(StreamEvent.create("key", 5.0, 20.0))

    # Advance watermark well past [0,10) — window closes
    processor.process(StreamEvent.create("key", 20.0, 0.0))

    # Late event arrives for closed window [0,10) — will be dropped
    late_event = StreamEvent.create("key", 2.0, 999.0)
    processor.process(late_event)
    # If dropped, the window result should NOT include 999.0

    processor.flush()

    logger.info(
        "Late event (value=%.0f, t=%.0f) was dropped because its window "
        "already closed (watermark past window_end + lateness)",
        late_event.value, late_event.event_time,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    result_store = IdempotentResultStore()

    demo_tumbling(result_store)
    print()

    demo_sliding(result_store)
    print()

    demo_session(result_store)
    print()

    demo_exactly_once(result_store)
    print()

    demo_late_data()
    print()

    logger.info(
        "Done. Total unique window results: %d. Total writes to store: %d.",
        result_store.unique_windows,
        result_store.total_writes,
    )


if __name__ == "__main__":
    main()
```

---

## Exercises

1. **Watermark visualization.** Run the tumbling window demo with events deliberately out of order (e.g., event at t=5 arrives after event at t=14). Log the watermark value at each step. Verify that the window [0,10) remains open until the watermark crosses 10 + allowed_lateness, and then closes.

2. **Late-data re-trigger.** Modify `TumblingWindowProcessor` to support a `late_data_mode` parameter: `"drop"` (current behavior) or `"retrigger"`. In retrigger mode, a late event causes the processor to re-emit an updated result with `is_final=False`, followed by the corrected result. Write a test asserting the sum changes when a late event arrives in retrigger mode.

3. **Multi-key tumbling.** Run the tumbling window demo with 3 keys (e.g., merchant-A, merchant-B, merchant-C) each emitting events at slightly different rates. Verify that windows for each key are tracked independently and results are emitted per-key.

4. **Global watermark.** Implement a `GlobalWatermarkTracker` that takes N per-partition `WatermarkTracker` instances. The global watermark is the minimum across all partitions. Wire three partition trackers together and verify that a lagging partition prevents windows from closing until it catches up.

5. **Async stream processor.** Rewrite the processor entry point as `async def process_stream(queue: asyncio.Queue[StreamEvent | None]) -> None` that drains an `asyncio.Queue`. A `None` sentinel signals end-of-stream (triggers `flush`). Run a producer coroutine that enqueues 50 events with random delays and verify all windows close correctly.

6. **Benchmark window state memory.** Instrument `TumblingWindowProcessor` to track the number of open (not-yet-emitted) accumulators at any point. Publish 10,000 events spanning 100 time windows. Log the peak open-window count and confirm it is bounded by the watermark lag (events within `allowed_lateness` of the latest event time).

*(Solutions: ../solutions/week-04/day-4.md)*

---

## Self-check

- [ ] I can explain the difference between event time and processing time and when each matters
- [ ] I can implement a tumbling window that uses watermarks to decide when to emit results
- [ ] I can describe what a session window is and name an example use case (e.g., user activity analysis)
- [ ] I can explain why exactly-once semantics require both an idempotent consumer AND a transactional producer
- [ ] I can trace what happens to a late event under each late-data mode (drop, retrigger, side output)
- [ ] I can explain why the global watermark is the minimum across all partitions

---

## Supplementary Reading

- [The Dataflow Model](https://research.google/pubs/the-dataflow-model-a-practical-approach-to-balancing-correctness-latency-and-cost-in-massive-scale-unbounded-out-of-order-data-processing/) — Akidau et al., Google (foundational paper behind Flink, Beam)
- [Streaming 101 — Tyler Akidau](https://www.oreilly.com/radar/the-world-beyond-batch-streaming-101/) — conceptual overview of event time, watermarks, windows
- [Streaming 102 — Tyler Akidau](https://www.oreilly.com/radar/the-world-beyond-batch-streaming-102/) — triggers, late data, accumulators
- [Apache Flink: Timely Stream Processing](https://nightlies.apache.org/flink/flink-docs-stable/docs/concepts/time/)
- [Kafka: Exactly-once semantics](https://www.confluent.io/blog/exactly-once-semantics-are-possible-heres-how-apache-kafka-does-it/) — Confluent engineering blog
- [Martin Kleppmann: Designing Data-Intensive Applications, ch. 11 (Stream Processing)](https://dataintensive.net/) — chapters 10-11 cover this week's full arc
