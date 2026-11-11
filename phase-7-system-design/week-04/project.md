# Week 04 Project — Event-Driven Analytics Ingestion System

## Overview

Design and implement a high-throughput analytics ingestion system for an e-commerce platform. The system must ingest clickstream, order, and inventory events, process them through windowed stream aggregation, and maintain durable aggregate state via event sourcing — all without a real broker (in-memory simulation).

This project integrates every concept from the week:
- Event-sourced aggregates (Day 6)
- Windowed stream processing (Day 6)
- Saga choreography with compensation (Day 5)
- Idempotency and deduplication (Day 5)
- Capacity estimation and failure mode analysis (Day 7)

Estimated time: 6–8 hours.

---

## System architecture

```
                    +------------------+
                    |  EventGenerator  |  (simulates producers)
                    +------------------+
                           |
               +-----------+-----------+
               |           |           |
               v           v           v
          Clickstream    Orders    Inventory
           Topic         Topic      Topic
               |           |           |
               v           v           v
         +----------+ +----------+ +----------+
         | Stream   | | Saga     | | Event-   |
         | Processor| | Handler  | | Sourced  |
         | (windows)| | (choreog)| | Aggregate|
         +----------+ +----------+ +----------+
               |           |           |
               v           v           v
         Window        Saga        Aggregate
         Results       State        State
               |           |           |
               +-----------+-----------+
                           |
                           v
                   +---------------+
                   |  QueryService |  (read model)
                   +---------------+
```

---

## Deliverables

### 1. Written design document (design.md)

A written document (400–600 words) covering:

**a) Kafka topology** (even though the implementation is in-memory, design as if real Kafka):
- Topic names, partition count for each, and partition key selection
- Justification for partition count based on estimated throughput
- Retention strategy for each topic

**b) Capacity estimate** for production deployment:
- Assume 10,000 events/second peak, avg message size 1.5 KB
- Calculate: required partitions, 30-day retention storage (3x replication, 4:1 compression)
- Consumer group sizing: 3 consumer groups, max 8 consumers each

**c) Failure mode analysis**:
- Identify 3 specific failure modes for this system
- For each: component that fails, observable symptom, and mitigation

**d) Consistency model**:
- Describe the consistency guarantees between the event-sourced aggregate and the stream processor's window results
- Is the system eventually consistent or strongly consistent at the read model? Explain why.

### 2. In-memory implementation

A complete, runnable Python package with the following structure:

```
analytics_ingestion/
    __init__.py
    models.py          # All Pydantic models
    event_bus.py       # In-memory pub/sub with dedup (reusable from Day 5 example)
    event_store.py     # EventStore from Day 6
    aggregate.py       # EventSourcedAggregate base + at least one concrete aggregate
    stream_processor.py # TumblingWindowProcessor from Day 6
    saga.py            # Choreography-based saga for order fulfillment
    query_service.py   # Read model: projections over stored events
    generator.py       # EventGenerator: produces synthetic events for testing

tests/
    __init__.py
    test_models.py
    test_event_bus.py
    test_aggregate.py
    test_stream_processor.py
    test_saga.py
    test_query_service.py
    test_integration.py
```

### 3. Reference event-sourced aggregate

Implement an `AnalyticsSession` aggregate (or extend `InventoryItem` from Day 6):

- Commands: `start_session(user_id, referrer)`, `record_pageview(page)`, `record_purchase(amount)`, `end_session()`
- Events: `SessionStarted`, `PageViewed`, `PurchaseRecorded`, `SessionEnded`
- Derived state: `page_count`, `total_spend`, `duration_seconds` (from start to end), `is_active`
- Guard: commands after `end_session` raise `SessionAlreadyEnded`

### 4. Windowed stream processor with tests

Extend `TumblingWindowProcessor` to support:
- Per-topic windows (multiple topics in one processor)
- Configurable window sizes per topic (`topic_windows: dict[str, int]` — key=topic, value=size_seconds)
- `summary() -> dict[str, list[WindowResult]]` — returns results keyed by topic

### 5. Full test suite

Minimum 30 pytest tests total across all modules. Distribution:
- `test_aggregate.py`: >= 10 tests (creation, commands, reconstitution, snapshot)
- `test_stream_processor.py`: >= 8 tests (window accuracy, topic isolation, edge cases)
- `test_saga.py`: >= 6 tests (happy path, each failure step, compensation order)
- `test_integration.py`: >= 6 tests (end-to-end flows: event → store → projection → query)

---

## Acceptance criteria

### Functional

- [ ] `AnalyticsSession` aggregate reconstitutes correctly from stored events
- [ ] Snapshot + delta reconstitution produces identical state to full replay
- [ ] `TumblingWindowProcessor` produces correct `record_count`, `total_value`, `avg_value` for each window
- [ ] Per-topic windowing maintains separate windows with no cross-contamination
- [ ] Order fulfillment saga emits correct compensation events when step 2 or 3 fails
- [ ] `EventBus` deduplication prevents reprocessing of events with duplicate `event_id`
- [ ] `QueryService.active_sessions()` returns only sessions where `is_active=True`
- [ ] `QueryService.revenue_by_hour(date)` returns a dict of `{hour: total_spend}`

### Code quality

- [ ] All public functions and class methods have type hints
- [ ] All data crossing module boundaries uses Pydantic models (no raw dicts)
- [ ] `logging` used throughout — no `print` statements in implementation files
- [ ] No hardcoded secrets, credentials, or environment-specific paths
- [ ] All external-ish calls (event publishing, store I/O) are mockable via dependency injection

### Tests

- [ ] All 30+ tests pass with `python -m pytest tests/ -v`
- [ ] No test modifies global state without cleanup (no cross-test pollution)
- [ ] Each test has a clear name following `test_<what>_<when>_<expected>` convention
- [ ] Tests do not assert on log output (use return values and state assertions)

---

## Getting started hints

### Step 1: Copy and extend Day 6 modules

Start with `event_store.py`, `aggregate.py`, and `stream_processor.py` from Day 6. They are complete and tested — extend, do not rewrite.

```python
# Minimal AnalyticsSession skeleton to get started:
from event_sourcing.aggregate import EventSourcedAggregate

class AnalyticsSession(EventSourcedAggregate):
    aggregate_type = "AnalyticsSession"

    def __init__(self, aggregate_id: str) -> None:
        super().__init__(aggregate_id)
        self.user_id: str = ""
        self.page_count: int = 0
        self.total_spend: float = 0.0
        self.is_active: bool = False

    @classmethod
    def start(cls, session_id: str, user_id: str, referrer: str) -> "AnalyticsSession":
        session = cls(session_id)
        session._raise_event("SessionStarted", {"user_id": user_id, "referrer": referrer})
        return session

    def _apply_SessionStarted(self, payload: dict) -> None:
        self.user_id = payload["user_id"]
        self.is_active = True

    def to_snapshot_state(self) -> dict:
        return {
            "user_id": self.user_id,
            "page_count": self.page_count,
            "total_spend": self.total_spend,
            "is_active": self.is_active,
        }

    def from_snapshot_state(self, state: dict) -> None:
        self.user_id = state["user_id"]
        self.page_count = state["page_count"]
        self.total_spend = state["total_spend"]
        self.is_active = state["is_active"]
```

### Step 2: Build the EventGenerator

The generator produces synthetic events for integration tests. Keep it simple:

```python
import uuid
from datetime import datetime, timedelta, timezone
from event_sourcing.models import StreamRecord

def generate_clickstream(
    n: int,
    start: datetime,
    interval_seconds: float = 0.5,
) -> list[StreamRecord]:
    t = start
    records = []
    for _ in range(n):
        records.append(StreamRecord(
            timestamp=t,
            topic="clickstream",
            value=1.0,
            attributes={"page": "/home"},
        ))
        t += timedelta(seconds=interval_seconds)
    return records
```

### Step 3: Build QueryService last

`QueryService` is a read model — it depends on the EventStore being populated. Build it after `AnalyticsSession` and the EventStore are solid. It only needs to call `store.load_events(aggregate_id)` and reduce over them.

### Step 4: Integration test structure

```python
# tests/test_integration.py  (skeleton)
def test_session_ingest_and_query() -> None:
    store = EventStore()
    query = QueryService(store)

    session = AnalyticsSession.start("s-1", "user-42", "google.com")
    session.record_pageview("/products")
    session.record_purchase(49.99)
    session.end_session()
    store.save(session)

    active = query.active_sessions()
    assert "s-1" not in active  # session ended

    revenue = query.revenue_by_hour(datetime(2024, 1, 1).date())
    # depends on when the session was created — mock datetime for determinism
```

---

## Stretch goals

Complete these only after all acceptance criteria pass. They are not required for a pass.

1. **Sliding window**: Implement `SlidingWindowProcessor(window_size_seconds, step_seconds)`. A record at t=5s in a window [0,10) with step 5 also appears in [5,15). Write 4 tests.

2. **Late-arrival handling**: Add `allowed_lateness_seconds: int` to `TumblingWindowProcessor`. Records arriving up to N seconds after a window closes are added to the closed window (update the result). Records beyond N seconds are dropped and logged as `LATE_DROPPED`. Write 3 tests.

3. **Projection replay**: Add a `rebuild(aggregate_class, aggregate_id)` method to `QueryService` that replays all events from scratch (ignoring snapshots) and returns the reconstituted aggregate. Useful for auditing.

4. **Concurrent write simulation**: Write a test that simulates two concurrent writers to the same aggregate. Both load at `version=5`. One saves first (succeeds), the other catches `AggregateVersionConflict`, reloads, and retries. Assert the retry succeeds and the final state is consistent.

5. **Schema migration**: Add `event_version: int = 1` to `StoredEvent`. Implement an `EventMigrator` that upcasts `PageViewed` v1 `{"page": str}` to v2 `{"page": str, "duration_ms": int}` (default 0). Apply migration on `load_events`. Write 3 tests including idempotency.

---

## Supplementary links

- Confluent — Kafka architecture deep dive: https://developer.confluent.io/courses/architecture/get-started/
- Eventuate — Saga framework for microservices: https://eventuate.io/docs/manual/eventuate-tram/latest/getting-started-eventuate-tram-sagas.html
- Flink — Windowing in stream processing: https://nightlies.apache.org/flink/flink-docs-stable/docs/dev/datastream/operators/windows/
- Pat Helland — Immutability Changes Everything: https://queue.acm.org/detail.cfm?id=2884038
- Gunnar Morling — Change Data Capture + Outbox: https://www.infoq.com/presentations/microservices-data-cdc/
