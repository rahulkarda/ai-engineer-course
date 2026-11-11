# Assessment Answer Key — Week 04

## Conceptual Answers

**1. Event sourcing vs CRUD — explain the difference and tradeoffs.**

Model answer: In CRUD, the database stores only the current state of a record; past states are overwritten and lost. In event sourcing, every state change is recorded as an immutable append-only event in an event log. Current state is derived by replaying events from the beginning (or from a snapshot). Key consequences: (a) full audit trail — you can answer "what was the state of order X at 3pm last Tuesday?" without additional tooling; (b) temporal queries and projections are trivially derived by replaying events up to a point in time; (c) storage grows indefinitely (mitigated by snapshots + log compaction); (d) eventual consistency between the write model (event store) and read models (projections). Choose event sourcing when auditability, historical replay, or event-driven integration are first-class requirements. Choose CRUD when storage cost, simplicity, and team familiarity with relational models outweigh those benefits.

---

**2. CAP theorem applied to event-driven systems — which guarantees do they typically provide?**

Model answer: Event-driven systems with Kafka-like brokers are typically AP — they favour Availability and Partition Tolerance over strong Consistency. Under a network partition, a broker replica may lag, meaning a consumer reads stale data. Eventual consistency is the norm: command side (event store) is consistent on write, but read models (projections) lag behind until events are processed. To achieve stronger consistency guarantees, you need: (a) synchronous projection updates in the same transaction as the event store write (at the cost of availability), or (b) read-your-writes via the event store directly (bypassing the projection entirely). CAP cannot be avoided — AP systems accept that read models may be briefly stale, and design consumers to be idempotent rather than transactionally correct.

---

**3. Exactly-once semantics — how do you achieve them?**

Model answer: Exactly-once requires cooperation at two levels. (1) Producer side: enable idempotent producer (`enable.idempotence=true`) so that retried sends are deduplicated by the broker using sequence numbers. Use transactional producers to atomically commit across multiple partitions/topics. (2) Consumer side: the consumer must be idempotent — processing the same message twice must produce the same outcome. Common strategies: (a) maintain a `seen_ids` set or deduplication store and skip messages whose ID is already present; (b) use conditional writes (optimistic concurrency) so a duplicate write fails silently; (c) write the processed result and the committed offset atomically in one database transaction (Kafka Streams exactly-once pattern). The combination of transactional producer + idempotent consumer gives end-to-end exactly-once. The trade-off is reduced throughput and added latency; for analytics workloads, at-least-once + idempotent aggregation is often preferred.

---

**4. Choreography vs orchestration — when to use each.**

Model answer: In a choreography-based saga, each service reacts to events and emits new events; there is no central coordinator. Services are decoupled and independently deployable; adding a new participant just means subscribing to an event. The downside is observability — the overall saga state exists only in the distributed event log, making failures hard to trace without distributed tracing tooling. In orchestration, a central coordinator (saga orchestrator, Temporal workflow, LangGraph subgraph) drives each step explicitly and manages compensation. The saga state is explicit and queryable; failures are easier to detect and retry. The cost is tight coupling to the orchestrator and a single coordination bottleneck. Choose choreography for simple, stable flows with few participants (< 4). Choose orchestration for long-running, multi-step flows with complex compensation logic or when observability is a hard requirement.

---

**5. Window strategies — describe tumbling, sliding, and session windows.**

Model answer: (1) Tumbling windows are fixed-size, non-overlapping intervals (e.g. "every 60 seconds"). Each record belongs to exactly one window. Use for per-minute metrics, hourly billing, and regular batch aggregations. (2) Sliding windows have a fixed size but advance by a step smaller than the window size, creating overlapping windows (e.g. "last 5 minutes, computed every 1 minute"). Records can belong to multiple windows simultaneously. Use for rolling averages, moving statistics, and anomaly detection over recent activity. (3) Session windows are dynamically sized based on activity gaps — a session window opens on the first event and closes after a configurable inactivity timeout. Records from the same session are grouped regardless of absolute time. Use for user session analytics, clickstream grouping, and IoT burst detection.

---

**6. Backpressure — what is it and how is it implemented?**

Model answer: Backpressure is a flow-control mechanism that slows a producer when a downstream consumer cannot keep up, preventing unbounded queue growth and OOM errors. Mechanisms: (a) Bounded queues — `asyncio.Queue(maxsize=N)` blocks `put()` when full, propagating slowdown back to the producer naturally. (b) Async semaphores — `asyncio.Semaphore(N)` limits the number of concurrent in-flight requests from a producer, so it self-throttles. (c) Pull-based polling — the consumer requests the next batch only when ready; the producer never pushes unsolicited. This is Kafka's native model (`consumer.poll()`). (d) Credits/tokens — the producer is given a quota of credits; each sent message consumes a credit; new credits arrive as the consumer acknowledges processing (used in gRPC flow control). In Python async code, the simplest approach is a bounded `asyncio.Queue` with producers calling `await queue.put()` and consumers calling `await queue.get()`.

---

**7. Snapshot strategy for event sourcing — when and how.**

Model answer: As an event stream grows, reconstituting state from the beginning of the log becomes slow. A snapshot captures the aggregate state at a specific version, so only events after the snapshot version need to be replayed. Strategy: take a snapshot every N events (typical values: 50–500 depending on state size and read latency requirements). On reconstitution: (1) load the latest snapshot for the stream; (2) load all events with `version > snapshot.version`; (3) fold those events over the snapshot state. Snapshot storage is separate from the event log (a simple key-value store: `stream_id -> Snapshot`). The event log is never truncated by snapshotting — snapshots are a read optimisation only. For aggregates with small state and infrequent reads, snapshots may not be worth the complexity.

---

## Coding Challenge A — Reference Solution

**Task:** Implement an in-memory `Counter` aggregate with event sourcing and snapshot support, including pytest tests.

```python
"""
counter_aggregate.py

In-memory Counter aggregate demonstrating event sourcing with snapshot support.
"""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

class Counter:
    """Simple event-sourced counter aggregate."""

    def __init__(self, counter_id: str) -> None:
        self.id = counter_id
        self._count: int = 0
        self._version: int = 0
        self._pending: list[StoredEvent] = []

    # ------ commands ------

    def increment(self, amount: int = 1) -> None:
        if amount <= 0:
            raise ValueError("Increment amount must be positive")
        self._apply_and_stage(StoredEvent(
            stream_id=self.id,
            version=self._version + len(self._pending) + 1,
            event_type="CounterIncremented",
            data={"amount": amount},
        ))

    def decrement(self, amount: int = 1) -> None:
        if amount <= 0:
            raise ValueError("Decrement amount must be positive")
        if self._count - amount < 0:
            raise ValueError("Counter cannot go below zero")
        self._apply_and_stage(StoredEvent(
            stream_id=self.id,
            version=self._version + len(self._pending) + 1,
            event_type="CounterDecremented",
            data={"amount": amount},
        ))

    def reset(self) -> None:
        self._apply_and_stage(StoredEvent(
            stream_id=self.id,
            version=self._version + len(self._pending) + 1,
            event_type="CounterReset",
            data={},
        ))

    # ------ state ------

    @property
    def count(self) -> int:
        return self._count

    @property
    def version(self) -> int:
        return self._version

    @property
    def pending_events(self) -> list[StoredEvent]:
        return list(self._pending)

    # ------ reconstitution ------

    @classmethod
    def reconstitute(
        cls,
        counter_id: str,
        events: list[StoredEvent],
        snapshot: Snapshot | None = None,
    ) -> "Counter":
        instance = cls.__new__(cls)
        instance.id = counter_id
        instance._pending = []

        if snapshot:
            instance._count = snapshot.state["count"]
            instance._version = snapshot.version
            events = [e for e in events if e.version > snapshot.version]
        else:
            instance._count = 0
            instance._version = 0

        for event in sorted(events, key=lambda e: e.version):
            instance._apply(event)
            instance._version = event.version

        return instance

    def get_snapshot(self) -> Snapshot:
        return Snapshot(
            stream_id=self.id,
            version=self._version,
            state={"count": self._count},
        )

    def mark_committed(self) -> None:
        self._version += len(self._pending)
        self._pending.clear()

    # ------ internal ------

    def _apply(self, event: StoredEvent) -> None:
        match event.event_type:
            case "CounterIncremented":
                self._count += event.data["amount"]
            case "CounterDecremented":
                self._count -= event.data["amount"]
            case "CounterReset":
                self._count = 0

    def _apply_and_stage(self, event: StoredEvent) -> None:
        self._apply(event)
        self._pending.append(event)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

import pytest


def test_counter_increment() -> None:
    c = Counter("c-1")
    c.increment(5)
    assert c.count == 5


def test_counter_decrement() -> None:
    c = Counter("c-1")
    c.increment(10)
    c.decrement(3)
    assert c.count == 7


def test_counter_reset() -> None:
    c = Counter("c-1")
    c.increment(100)
    c.reset()
    assert c.count == 0


def test_counter_below_zero_raises() -> None:
    c = Counter("c-1")
    c.increment(5)
    with pytest.raises(ValueError):
        c.decrement(10)


def test_pending_events_emitted() -> None:
    c = Counter("c-1")
    c.increment(1)
    c.increment(2)
    assert len(c.pending_events) == 2
    assert c.pending_events[0].event_type == "CounterIncremented"


def test_reconstitute_from_events() -> None:
    c = Counter("c-1")
    c.increment(10)
    c.decrement(3)
    events = c.pending_events

    restored = Counter.reconstitute("c-1", events)
    assert restored.count == 7
    assert restored.version == 2


def test_reconstitute_from_snapshot() -> None:
    snap = Snapshot(stream_id="c-1", version=5, state={"count": 50})
    later_event = StoredEvent(
        stream_id="c-1", version=6,
        event_type="CounterIncremented", data={"amount": 10}
    )
    restored = Counter.reconstitute("c-1", [later_event], snapshot=snap)
    assert restored.count == 60
    assert restored.version == 6


def test_snapshot_roundtrip() -> None:
    c = Counter("c-2")
    c.increment(42)
    snap = c.get_snapshot()
    assert snap.state["count"] == 42
    assert snap.version == 0  # not yet committed


def test_mark_committed_clears_pending() -> None:
    c = Counter("c-3")
    c.increment(1)
    c.mark_committed()
    assert len(c.pending_events) == 0
    assert c.version == 1


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
```

---

## Coding Challenge B — Reference Solution

**Task:** Implement `TumblingWindowProcessor` for a `temperature` topic and verify aggregation over multiple windows.

```python
"""
challenge_b_window.py
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Callable

import pytest
from pydantic import BaseModel, Field
import uuid


class StreamRecord(BaseModel):
    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic: str
    timestamp: datetime
    value: float


class WindowResult(BaseModel):
    window_start: datetime
    window_end: datetime
    topic: str
    count: int
    total: float
    min_value: float
    max_value: float


class _Acc:
    def __init__(self, start: datetime, end: datetime, topic: str) -> None:
        self.start, self.end, self.topic = start, end, topic
        self.count = 0
        self.total = 0.0
        self.min_v = float("inf")
        self.max_v = float("-inf")

    def add(self, v: float) -> None:
        self.count += 1
        self.total += v
        self.min_v = min(self.min_v, v)
        self.max_v = max(self.max_v, v)

    def result(self) -> WindowResult:
        return WindowResult(
            window_start=self.start, window_end=self.end, topic=self.topic,
            count=self.count, total=round(self.total, 6),
            min_value=self.min_v if self.count else 0.0,
            max_value=self.max_v if self.count else 0.0,
        )


class TumblingWindowProcessor:
    def __init__(
        self,
        window_size_seconds: int,
        on_window_close: Callable[[WindowResult], None] | None = None,
    ) -> None:
        self._size = window_size_seconds
        self._cb = on_window_close or (lambda _: None)
        self._windows: dict[tuple[str, datetime], _Acc] = {}
        self._closed: list[WindowResult] = []
        self._watermark = datetime.min

    def _ws(self, ts: datetime) -> datetime:
        epoch = datetime(1970, 1, 1)
        secs = int((ts - epoch).total_seconds() // self._size) * self._size
        return epoch + timedelta(seconds=secs)

    def process(self, r: StreamRecord) -> None:
        ws = self._ws(r.timestamp)
        we = ws + timedelta(seconds=self._size)
        if r.timestamp > self._watermark:
            self._watermark = r.timestamp
            for k, acc in list(self._windows.items()):
                if acc.end <= self._watermark:
                    res = self._windows.pop(k).result()
                    self._closed.append(res)
                    self._cb(res)
        elif r.timestamp < self._watermark - timedelta(seconds=self._size):
            return
        key = (r.topic, ws)
        if key not in self._windows:
            self._windows[key] = _Acc(ws, we, r.topic)
        self._windows[key].add(r.value)

    def flush_all(self) -> None:
        for acc in list(self._windows.values()):
            res = acc.result()
            self._closed.append(res)
            self._cb(res)
        self._windows.clear()

    @property
    def closed_windows(self) -> list[WindowResult]:
        return list(self._closed)


# --- tests ---

def _rec(ts: datetime, v: float) -> StreamRecord:
    return StreamRecord(topic="temperature", timestamp=ts, value=v)


def test_single_window_aggregation() -> None:
    proc = TumblingWindowProcessor(window_size_seconds=60)
    t = datetime(2024, 1, 1, 12, 0, 0)
    for v in [20.0, 22.0, 18.0, 25.0]:
        proc.process(_rec(t, v))
        t += timedelta(seconds=10)
    proc.flush_all()

    assert len(proc.closed_windows) == 1
    w = proc.closed_windows[0]
    assert w.count == 4
    assert w.min_value == 18.0
    assert w.max_value == 25.0


def test_two_windows_produce_two_results() -> None:
    closed: list[WindowResult] = []
    proc = TumblingWindowProcessor(60, on_window_close=closed.append)
    t0 = datetime(2024, 1, 1, 12, 0, 0)
    proc.process(_rec(t0 + timedelta(seconds=10), 10.0))
    proc.process(_rec(t0 + timedelta(seconds=90), 20.0))
    proc.flush_all()
    assert len(closed) == 2
    assert {r.total for r in closed} == {10.0, 20.0}
```

---

## Coding Challenge C — Reference Solution

**Task:** Implement a complete choreography saga for `Order -> Payment -> Inventory` with compensation.

See Day 5, Exercise 1 for the full reference implementation. The assessment version must additionally demonstrate:

1. All three services registered before any event is published.
2. Compensation path verified via assertion (not just print).
3. At least one test using `pytest.mark.parametrize` over happy/failure paths.

```python
"""
challenge_c_saga.py
"""
# (Re-use choreography_saga.py from Day 5)
# Additional parametrize test:

import pytest
from choreography_saga import (
    EventBus, InventoryReserved, OrderCancelled, OrderPlaced,
    PaymentService, InventoryService, OrderService
)


@pytest.mark.parametrize("should_fail,expected_type", [
    (False, InventoryReserved),
    (True, OrderCancelled),
])
def test_saga_outcomes(should_fail: bool, expected_type: type) -> None:
    bus = EventBus()
    results = []
    bus.subscribe(InventoryReserved, results.append)
    bus.subscribe(OrderCancelled, results.append)

    fail_ids = {"ord-fail"} if should_fail else set()
    PaymentService(bus, fail_order_ids=fail_ids)
    InventoryService(bus)
    OrderService(bus)

    order_id = "ord-fail" if should_fail else "ord-ok"
    bus.publish(OrderPlaced(
        order_id=order_id,
        customer_id="cust-1",
        amount=50.0,
        items=[],
    ))

    assert len(results) == 1
    assert isinstance(results[0], expected_type)
    assert results[0].order_id == order_id
```

---

## Grading Notes

**75% threshold:** Student correctly implements the core data structures (StoredEvent, aggregate apply loop) and at least the happy-path tests pass. Compensation path may be missing or incomplete. Window aggregation functional but edge cases (late data, flush_all) not tested.

**90% threshold:** All three coding challenges pass with correct assertions. Idempotency test demonstrates `call_count == 1`. Reconstitution from snapshot works. Concurrency error raised on version mismatch. Window boundaries are aligned to epoch (not just relative to first record).

**100% threshold:** All of the above, plus: (a) `hashlib.sha256` used for partition assignment (not Python `hash()`), with explanation of why; (b) choreography compensation path covered by parametrize test; (c) conceptual answers address trade-offs explicitly (not just one-sided descriptions); (d) capacity math shows intermediate steps and explains replication overhead; (e) interview answer (Q3, exactly-once) correctly distinguishes producer-side dedup from consumer-side idempotency.
