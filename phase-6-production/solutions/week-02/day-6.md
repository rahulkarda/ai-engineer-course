# Phase 6 · Week 02 · Day 6 — Solutions

> Solutions for: DLQ, Ordering, Backpressure, Broker Choice

---

## Exercise 1 — DLQ replay with filtering: replay only messages matching a predicate

**Problem**: Add DLQ replay with filtering — replay only messages matching a predicate (e.g., `error_type == "timeout"`).

**Explanation**: Replaying the entire DLQ is risky when it contains a mix of recoverable and unrecoverable errors. A predicate-filtered replay lets you selectively replay only the messages that are likely to succeed after a fix (e.g., timeout errors after the downstream service recovered) while leaving schema errors in the DLQ for manual inspection.

```python
"""
ex1_dlq_replay_filter.py

Extends InMemoryBrokerWithDLQ (from day-6 worked example) to support
filtered DLQ replay: only replay messages where predicate(record) is True.

Run tests: pytest ex1_dlq_replay_filter.py -v
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import uuid4

import pytest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
)
log = logging.getLogger(__name__)


@dataclass
class Message:
    id: str
    topic: str
    payload: Any
    attempt: int = 0
    error_type: str = ""            # set on failure — used for filtering
    published_at: float = field(default_factory=time.monotonic)


@dataclass
class DeadLetterRecord:
    original_message: Message
    original_topic: str
    failure_reason: str
    error_type: str                 # "timeout" | "schema" | "unknown" etc.
    failed_at: float = field(default_factory=time.monotonic)


DLQPredicate = Callable[[DeadLetterRecord], bool]


class InMemoryBrokerWithDLQ:
    def __init__(self, max_retries: int = 3) -> None:
        self._max_retries = max_retries
        self._queues: dict[str, list[Message]] = defaultdict(list)
        self._handlers: dict[str, list[Callable[[Message], None]]] = defaultdict(list)
        self._dlq: dict[str, list[DeadLetterRecord]] = defaultdict(list)

    def publish(self, topic: str, payload: Any, error_type: str = "") -> str:
        msg = Message(id=str(uuid4()), topic=topic, payload=payload, error_type=error_type)
        self._queues[topic].append(msg)
        return msg.id

    def subscribe(self, topic: str, handler: Callable[[Message], None]) -> None:
        self._handlers[topic].append(handler)

    def consume_one(self, topic: str) -> bool:
        queue = self._queues[topic]
        if not queue:
            return False
        msg = queue.pop(0)
        msg.attempt += 1
        for handler in self._handlers.get(topic, []):
            try:
                handler(msg)
            except Exception as exc:
                self._handle_failure(msg, str(exc))
        return True

    def consume_all(self, topic: str) -> int:
        count = 0
        while self.consume_one(topic):
            count += 1
        return count

    def _handle_failure(self, msg: Message, reason: str) -> None:
        if msg.attempt < self._max_retries:
            self._queues[msg.topic].append(msg)
        else:
            record = DeadLetterRecord(
                original_message=msg,
                original_topic=msg.topic,
                failure_reason=reason,
                error_type=msg.error_type,
            )
            self._dlq[msg.topic].append(record)
            log.error("DLQ msg_id=%s topic=%s error_type=%s", msg.id, msg.topic, msg.error_type)

    def dlq_depth(self, topic: str) -> int:
        return len(self._dlq[topic])

    def dlq_inspect(self, topic: str) -> list[DeadLetterRecord]:
        return list(self._dlq[topic])

    def dlq_replay(
        self,
        topic: str,
        predicate: DLQPredicate | None = None,
    ) -> tuple[int, int]:
        """
        Replay DLQ messages for topic.

        If predicate is None, replay all (original behaviour).
        If predicate is given, replay only records where predicate(record) is True.
        Records that do not match the predicate remain in the DLQ.

        Returns: (replayed_count, remaining_in_dlq)
        """
        records = self._dlq.pop(topic, [])
        replayed = 0
        kept: list[DeadLetterRecord] = []

        for record in records:
            if predicate is None or predicate(record):
                fresh = Message(
                    id=record.original_message.id,
                    topic=record.original_topic,
                    payload=record.original_message.payload,
                    error_type=record.error_type,
                    attempt=0,
                )
                self._queues[record.original_topic].append(fresh)
                log.info("REPLAY msg_id=%s error_type=%s", fresh.id, fresh.error_type)
                replayed += 1
            else:
                kept.append(record)
                log.debug("KEPT in DLQ msg_id=%s (predicate=False)", record.original_message.id)

        if kept:
            self._dlq[topic] = kept

        return replayed, len(kept)

    def dlq_purge(self, topic: str) -> int:
        records = self._dlq.pop(topic, [])
        return len(records)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def always_fail(error_type: str) -> Callable[[Message], None]:
    def handler(msg: Message) -> None:
        msg.error_type = error_type
        raise RuntimeError(f"simulated {error_type} error")
    return handler


@pytest.fixture()
def broker() -> InMemoryBrokerWithDLQ:
    return InMemoryBrokerWithDLQ(max_retries=1)


class TestDLQReplayFilter:
    def _populate_dlq(self, broker: InMemoryBrokerWithDLQ) -> None:
        broker.subscribe("orders", always_fail("timeout"))
        broker.subscribe("orders", always_fail("schema"))   # second handler also fails

        broker.publish("orders", {"order_id": "timeout-1"}, error_type="timeout")
        broker.publish("orders", {"order_id": "schema-1"}, error_type="schema")
        broker.publish("orders", {"order_id": "timeout-2"}, error_type="timeout")
        broker.consume_all("orders")

    def test_replay_all_without_predicate(self, broker: InMemoryBrokerWithDLQ) -> None:
        broker.subscribe("orders", always_fail("timeout"))
        broker.publish("orders", {"order_id": "o1"}, error_type="timeout")
        broker.consume_all("orders")

        assert broker.dlq_depth("orders") == 1
        replayed, remaining = broker.dlq_replay("orders")
        assert replayed == 1
        assert remaining == 0

    def test_replay_only_timeout_errors(self, broker: InMemoryBrokerWithDLQ) -> None:
        broker_2 = InMemoryBrokerWithDLQ(max_retries=1)

        def handler(msg: Message) -> None:
            msg.error_type = msg.payload.get("error_type_hint", "unknown")
            raise RuntimeError("always fail")

        broker_2.subscribe("t", handler)
        broker_2.publish("t", {"id": "timeout-1", "error_type_hint": "timeout"})
        broker_2.publish("t", {"id": "schema-1", "error_type_hint": "schema"})
        broker_2.publish("t", {"id": "timeout-2", "error_type_hint": "timeout"})
        broker_2.consume_all("t")

        assert broker_2.dlq_depth("t") == 3

        replayed, remaining = broker_2.dlq_replay(
            "t",
            predicate=lambda r: r.error_type == "timeout",
        )

        assert replayed == 2
        assert remaining == 1

        # Verify remaining record is the schema error
        kept = broker_2.dlq_inspect("t")
        assert len(kept) == 1
        assert kept[0].error_type == "schema"

    def test_replay_with_false_predicate_keeps_all(self, broker: InMemoryBrokerWithDLQ) -> None:
        broker.subscribe("orders", always_fail("timeout"))
        broker.publish("orders", {"id": "o1"}, error_type="timeout")
        broker.consume_all("orders")

        replayed, remaining = broker.dlq_replay("orders", predicate=lambda r: False)
        assert replayed == 0
        assert remaining == 1
        assert broker.dlq_depth("orders") == 1

    def test_replay_empty_dlq_returns_zero(self, broker: InMemoryBrokerWithDLQ) -> None:
        replayed, remaining = broker.dlq_replay("nonexistent-topic")
        assert replayed == 0
        assert remaining == 0
```

---

## Exercise 2 — Per-partition ordering enforcement using key hash → partition assignment

**Problem**: Implement per-partition ordering enforcement using a per-key hash → partition assignment.

**Explanation**: Kafka's ordering guarantee is per-partition. The hash-based partitioner ensures messages with the same key always land in the same partition, which has a monotonically increasing offset. Two events for the same entity (e.g., `order_id`) processed from the same partition are always in write order. This breaks if the key is `None` (use a stable fallback) or if the partition count changes (messages re-route to a different partition).

```python
"""
ex2_per_key_ordering.py

Demonstrates per-partition ordering via deterministic key → partition routing.
Verifies that same-key messages are always consecutive within a partition.

Run tests: pytest ex2_per_key_ordering.py -v
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import BaseModel, field_validator

log = logging.getLogger(__name__)


class KafkaRecord(BaseModel):
    key: str | None
    value: dict[str, Any]
    partition: int
    offset: int

    @field_validator("offset")
    @classmethod
    def offset_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("offset must be >= 0")
        return v


@dataclass
class Partition:
    partition_id: int
    log: list[KafkaRecord] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def append(self, key: str | None, value: dict[str, Any]) -> KafkaRecord:
        with self._lock:
            offset = len(self.log)
            record = KafkaRecord(key=key, value=value, partition=self.partition_id, offset=offset)
            self.log.append(record)
            return record

    def records_for_key(self, key: str | None) -> list[KafkaRecord]:
        with self._lock:
            return [r for r in self.log if r.key == key]


class OrderEnforcedBroker:
    """
    In-memory broker with per-key partition enforcement.

    Partitioner: murmur2-style via Python's built-in hash.
    Same key → same hash → same partition modulo num_partitions.

    Ordering guarantee: within a partition, records are appended in order.
    Since same-key records always go to the same partition, they are always
    retrievable in insertion order from that partition's log.
    """

    def __init__(self) -> None:
        self._topics: dict[str, list[Partition]] = {}
        self._num_partitions: dict[str, int] = {}
        self._offsets: dict[str, dict[str, dict[int, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )

    def create_topic(self, topic: str, num_partitions: int) -> None:
        if topic not in self._topics:
            self._topics[topic] = [Partition(partition_id=i) for i in range(num_partitions)]
            self._num_partitions[topic] = num_partitions

    def partition_for_key(self, key: str | None, num_partitions: int) -> int:
        """
        Deterministic key → partition mapping.
        None key → partition 0 (stable fallback, not round-robin).
        Using Python's built-in hash is deterministic within a process
        (hash randomization is per-interpreter invocation but stable within a run).
        """
        if key is None:
            return 0
        return hash(key) % num_partitions

    def produce(self, topic: str, key: str | None, value: dict[str, Any]) -> KafkaRecord:
        num_parts = self._num_partitions[topic]
        pid = self.partition_for_key(key, num_parts)
        record = self._topics[topic][pid].append(key, value)
        log.debug("produce: %s[%d]@%d key=%s", topic, pid, record.offset, key)
        return record

    def poll(self, group: str, topic: str, partition: int, max_records: int = 10) -> list[KafkaRecord]:
        offset = self._offsets[group][topic][partition]
        with self._topics[topic][partition]._lock:
            return self._topics[topic][partition].log[offset : offset + max_records]

    def commit(self, group: str, topic: str, partition: int, offset: int) -> None:
        self._offsets[group][topic][partition] = offset

    def records_for_key(self, topic: str, key: str | None) -> list[KafkaRecord]:
        """Return all records for a key in insertion order (reads from the assigned partition)."""
        num_parts = self._num_partitions[topic]
        pid = self.partition_for_key(key, num_parts)
        return self._topics[topic][pid].records_for_key(key)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def broker() -> OrderEnforcedBroker:
    b = OrderEnforcedBroker()
    b.create_topic("events", num_partitions=4)
    return b


class TestPerKeyOrdering:
    def test_same_key_always_same_partition(self, broker: OrderEnforcedBroker) -> None:
        records = [broker.produce("events", key="order-42", value={"seq": i}) for i in range(5)]
        partitions = {r.partition for r in records}
        assert len(partitions) == 1, f"same key must land in one partition, got {partitions}"

    def test_same_key_consecutive_offsets(self, broker: OrderEnforcedBroker) -> None:
        records = [broker.produce("events", key="order-42", value={"seq": i}) for i in range(5)]
        offsets = [r.offset for r in records]
        # Offsets within a partition are monotonically increasing
        assert offsets == sorted(offsets)
        assert all(offsets[i + 1] == offsets[i] + 1 for i in range(len(offsets) - 1))

    def test_different_keys_may_use_different_partitions(self, broker: OrderEnforcedBroker) -> None:
        partitions_seen: set[int] = set()
        for i in range(20):
            r = broker.produce("events", key=f"order-{i}", value={"seq": i})
            partitions_seen.add(r.partition)
        # With 20 distinct keys and 4 partitions, multiple partitions should be used
        assert len(partitions_seen) > 1

    def test_insertion_order_preserved_per_key(self, broker: OrderEnforcedBroker) -> None:
        for i in range(10):
            broker.produce("events", key="entity-1", value={"step": i})

        records = broker.records_for_key("events", "entity-1")
        steps = [r.value["step"] for r in records]
        assert steps == list(range(10)), f"insertion order violated: {steps}"

    def test_none_key_goes_to_partition_zero(self, broker: OrderEnforcedBroker) -> None:
        r = broker.produce("events", key=None, value={"type": "unkeyed"})
        assert r.partition == 0

    def test_interleaved_keys_independent_ordering(self, broker: OrderEnforcedBroker) -> None:
        """
        Interleave messages from two keys.
        Each key's records must still be in insertion order.
        """
        for i in range(5):
            broker.produce("events", key="A", value={"seq": i})
            broker.produce("events", key="B", value={"seq": i})

        a_records = broker.records_for_key("events", "A")
        b_records = broker.records_for_key("events", "B")

        assert [r.value["seq"] for r in a_records] == list(range(5))
        assert [r.value["seq"] for r in b_records] == list(range(5))
```

---

## Exercise 3 — Backpressure-aware producer: linear rate limiting and pause above 90%

**Problem**: Build a backpressure-aware producer: measure queue depth, apply linear rate limiting when depth > 50%, pause when > 90%.

**Explanation**: Backpressure protects the broker from overflow. The producer checks queue depth before each publish:
- depth 0–50%: no throttle (full speed)
- depth 50–90%: linear throttle — `sleep = max_sleep * ((depth_pct - 50) / 40)`
- depth > 90%: hard pause — sleep until depth drops below threshold

This is the same strategy Kafka's producer uses via `buffer.memory` and `max.block.ms`.

```python
"""
ex3_backpressure_producer.py

BackpressureAwareProducer: measures queue depth and applies rate limiting.

- 0-50% full: produce at full speed.
- 50-90% full: linear sleep proportional to depth above 50%.
- >90% full: pause until depth drops below 90%.

Run tests: pytest ex3_backpressure_producer.py -v
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import pytest

log = logging.getLogger(__name__)

BACKPRESSURE_LOW_PCT = 0.50    # start throttling above 50%
BACKPRESSURE_HIGH_PCT = 0.90   # hard pause above 90%
MAX_SLEEP_S = 0.1              # max linear sleep per message
PAUSE_POLL_INTERVAL_S = 0.01   # how often to re-check depth when paused


class BackpressureError(Exception):
    """Raised when the queue is at max_depth and pause_on_full=False."""


@dataclass
class InMemoryQueue:
    """Simple bounded FIFO queue simulating a broker topic."""
    max_depth: int = 1000
    _items: list[Any] = field(default_factory=list)

    def enqueue(self, item: Any) -> None:
        if len(self._items) >= self.max_depth:
            raise BackpressureError(f"Queue full at {self.max_depth}")
        self._items.append(item)

    def dequeue(self, n: int = 1) -> list[Any]:
        items, self._items = self._items[:n], self._items[n:]
        return items

    @property
    def depth(self) -> int:
        return len(self._items)

    @property
    def depth_pct(self) -> float:
        return self.depth / self.max_depth if self.max_depth > 0 else 0.0


class BackpressureAwareProducer:
    """
    Produces messages to an InMemoryQueue with depth-aware rate limiting.

    Behaviour by queue depth percentage:
        [0, 50%)  → produce immediately, no sleep
        [50%, 90%) → linear sleep: sleep = max_sleep * (depth_pct - 0.5) / 0.4
        [90%, 100%] → hard pause: spin-wait until depth drops below 90%

    All sleeps are observable via the throttle_events list (for testing).
    """

    def __init__(self, queue: InMemoryQueue, max_sleep_s: float = MAX_SLEEP_S) -> None:
        self._queue = queue
        self._max_sleep = max_sleep_s
        self.throttle_events: list[dict[str, Any]] = []   # for test inspection

    def produce(self, payload: Any) -> None:
        """Produce one message, applying backpressure as needed."""
        depth_pct = self._queue.depth_pct

        if depth_pct >= BACKPRESSURE_HIGH_PCT:
            self._pause_until_below_high_water()
        elif depth_pct >= BACKPRESSURE_LOW_PCT:
            self._linear_throttle(depth_pct)

        self._queue.enqueue(payload)

    def _linear_throttle(self, depth_pct: float) -> None:
        """Sleep proportionally to how far above 50% we are."""
        ratio = (depth_pct - BACKPRESSURE_LOW_PCT) / (BACKPRESSURE_HIGH_PCT - BACKPRESSURE_LOW_PCT)
        sleep_s = self._max_sleep * ratio
        event = {"type": "throttle", "depth_pct": depth_pct, "sleep_s": sleep_s}
        self.throttle_events.append(event)
        log.debug("Throttle: depth=%.1f%% sleep=%.4fs", depth_pct * 100, sleep_s)
        time.sleep(sleep_s)

    def _pause_until_below_high_water(self) -> None:
        """Spin-wait until queue depth drops below the high-water mark."""
        log.warning(
            "Backpressure PAUSE: queue depth %.1f%% >= %.0f%%. Waiting for consumer.",
            self._queue.depth_pct * 100,
            BACKPRESSURE_HIGH_PCT * 100,
        )
        paused_at = self._queue.depth_pct
        while self._queue.depth_pct >= BACKPRESSURE_HIGH_PCT:
            time.sleep(PAUSE_POLL_INTERVAL_S)
        self.throttle_events.append({
            "type": "pause",
            "depth_pct_at_pause": paused_at,
            "depth_pct_resumed": self._queue.depth_pct,
        })
        log.info("Backpressure RESUME: queue depth %.1f%%", self._queue.depth_pct * 100)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBackpressureProducer:
    def test_no_throttle_below_50_pct(self) -> None:
        queue = InMemoryQueue(max_depth=100)
        producer = BackpressureAwareProducer(queue, max_sleep_s=0.001)

        # Fill to 49 items (49%)
        for i in range(49):
            producer.produce({"seq": i})

        assert len(producer.throttle_events) == 0
        assert queue.depth == 49

    def test_linear_throttle_between_50_and_90_pct(self) -> None:
        queue = InMemoryQueue(max_depth=100)
        producer = BackpressureAwareProducer(queue, max_sleep_s=0.0)  # zero sleep for speed

        # Fill to 50 (50% = just at threshold, no throttle yet)
        for i in range(50):
            producer.produce({"seq": i})
        assert len(producer.throttle_events) == 0

        # 51st item triggers throttle
        producer.produce({"seq": 50})
        assert len(producer.throttle_events) == 1
        assert producer.throttle_events[0]["type"] == "throttle"
        assert producer.throttle_events[0]["depth_pct"] >= 0.50

    def test_sleep_increases_with_depth(self) -> None:
        """Sleep at 70% should be greater than sleep at 55%."""
        queue = InMemoryQueue(max_depth=100)
        producer = BackpressureAwareProducer(queue, max_sleep_s=1.0)

        # Get a throttle event at ~55%
        for i in range(55):
            producer.produce({"seq": i})
        sleep_at_55 = producer.throttle_events[-1]["sleep_s"]

        # Get a throttle event at ~70%
        for i in range(15):
            producer.produce({"seq": 100 + i})
        sleep_at_70 = producer.throttle_events[-1]["sleep_s"]

        assert sleep_at_70 > sleep_at_55, (
            f"sleep at 70% ({sleep_at_70:.4f}) should exceed sleep at 55% ({sleep_at_55:.4f})"
        )

    def test_pause_triggered_above_90_pct(self) -> None:
        """Producer pauses when depth >= 90%; resumes after consumer drains."""
        import threading

        queue = InMemoryQueue(max_depth=10)
        producer = BackpressureAwareProducer(queue, max_sleep_s=0.0)

        # Fill to 9 (90%)
        for i in range(9):
            queue.enqueue({"seq": i})

        assert queue.depth_pct >= 0.90

        # Producer should pause; start in background thread
        results: list[str] = []

        def producer_thread() -> None:
            producer.produce({"seq": 99})  # this should block
            results.append("produced")

        t = threading.Thread(target=producer_thread)
        t.start()

        # Let producer enter pause loop
        time.sleep(0.05)
        assert results == [], "Producer should still be paused"

        # Consumer drains enough to drop below 90%
        queue.dequeue(n=5)  # depth now 4/10 = 40%
        t.join(timeout=2.0)
        assert t.is_alive() is False, "Producer thread should have unblocked"
        assert results == ["produced"]

        pause_events = [e for e in producer.throttle_events if e["type"] == "pause"]
        assert len(pause_events) == 1

    def test_max_depth_raises_backpressure_error_when_full(self) -> None:
        queue = InMemoryQueue(max_depth=5)
        for i in range(5):
            queue.enqueue(i)
        with pytest.raises(BackpressureError):
            queue.enqueue("overflow")
```

---

## Exercise 4 — DLQ alerting: warn when DLQ depth exceeds threshold, include message sample

**Problem**: Implement DLQ alerting: emit a warning log when DLQ depth exceeds threshold, include message sample.

**Explanation**: DLQ depth > 0 means real messages have failed. In production, this connects to a PagerDuty / Slack alert. Here we use the standard `logging` module at WARNING level, which production alerting systems scrape. The message sample gives on-call engineers immediate context without needing to query the DLQ separately.

```python
"""
ex4_dlq_alerting.py

DLQ alerting: emit WARNING log when DLQ depth exceeds threshold.
Includes a message sample for immediate context.

Run tests: pytest ex4_dlq_alerting.py -v
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import uuid4

import pytest

log = logging.getLogger(__name__)

DLQ_ALERT_THRESHOLD = int(10)   # warn when DLQ depth >= this value
DLQ_SAMPLE_SIZE = int(3)        # include this many sample messages in the alert


@dataclass
class Message:
    id: str
    topic: str
    payload: Any
    attempt: int = 0
    published_at: float = field(default_factory=time.monotonic)


@dataclass
class DeadLetterRecord:
    original_message: Message
    original_topic: str
    failure_reason: str
    failed_at: float = field(default_factory=time.monotonic)


class InMemoryBrokerWithDLQAlerts:
    """
    InMemoryBroker with DLQ depth alerting.

    After each message is routed to the DLQ, checks if depth >= threshold.
    If so, emits one WARNING log with:
      - topic name
      - current DLQ depth
      - sample of up to DLQ_SAMPLE_SIZE recent failure reasons + message IDs
    """

    def __init__(
        self,
        max_retries: int = 3,
        alert_threshold: int = DLQ_ALERT_THRESHOLD,
        sample_size: int = DLQ_SAMPLE_SIZE,
    ) -> None:
        self._max_retries = max_retries
        self._alert_threshold = alert_threshold
        self._sample_size = sample_size
        self._queues: dict[str, list[Message]] = defaultdict(list)
        self._handlers: dict[str, list[Callable[[Message], None]]] = defaultdict(list)
        self._dlq: dict[str, list[DeadLetterRecord]] = defaultdict(list)
        self._alerts_emitted: dict[str, int] = defaultdict(int)  # topic → alert count (for tests)

    def publish(self, topic: str, payload: Any) -> str:
        msg = Message(id=str(uuid4()), topic=topic, payload=payload)
        self._queues[topic].append(msg)
        return msg.id

    def subscribe(self, topic: str, handler: Callable[[Message], None]) -> None:
        self._handlers[topic].append(handler)

    def consume_one(self, topic: str) -> bool:
        queue = self._queues[topic]
        if not queue:
            return False
        msg = queue.pop(0)
        msg.attempt += 1
        for handler in self._handlers.get(topic, []):
            try:
                handler(msg)
            except Exception as exc:
                self._handle_failure(msg, str(exc))
        return True

    def consume_all(self, topic: str) -> int:
        count = 0
        while self.consume_one(topic):
            count += 1
        return count

    def _handle_failure(self, msg: Message, reason: str) -> None:
        if msg.attempt < self._max_retries:
            self._queues[msg.topic].append(msg)
        else:
            record = DeadLetterRecord(
                original_message=msg,
                original_topic=msg.topic,
                failure_reason=reason,
            )
            self._dlq[msg.topic].append(record)
            self._check_and_alert(msg.topic)

    def _check_and_alert(self, topic: str) -> None:
        """Check DLQ depth; emit WARNING alert if threshold exceeded."""
        depth = len(self._dlq[topic])
        if depth < self._alert_threshold:
            return

        # Build sample from most-recent DLQ records
        sample = self._dlq[topic][-self._sample_size:]
        sample_lines = [
            f"  msg_id={r.original_message.id} reason={r.failure_reason!r}"
            for r in sample
        ]
        sample_text = "\n".join(sample_lines)

        log.warning(
            "DLQ ALERT: topic=%s depth=%d >= threshold=%d\nSample (last %d):\n%s",
            topic,
            depth,
            self._alert_threshold,
            len(sample),
            sample_text,
        )
        self._alerts_emitted[topic] += 1

    def dlq_depth(self, topic: str) -> int:
        return len(self._dlq[topic])

    def dlq_inspect(self, topic: str) -> list[DeadLetterRecord]:
        return list(self._dlq[topic])

    def alerts_emitted(self, topic: str) -> int:
        return self._alerts_emitted[topic]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def always_fail(msg: Message) -> None:
    raise RuntimeError(f"simulated failure for {msg.id}")


@pytest.fixture()
def broker() -> InMemoryBrokerWithDLQAlerts:
    return InMemoryBrokerWithDLQAlerts(max_retries=1, alert_threshold=3, sample_size=2)


class TestDLQAlerting:
    def test_no_alert_below_threshold(
        self, broker: InMemoryBrokerWithDLQAlerts, caplog: pytest.LogCaptureFixture
    ) -> None:
        broker.subscribe("t", always_fail)
        for i in range(2):
            broker.publish("t", {"seq": i})
        broker.consume_all("t")

        assert broker.dlq_depth("t") == 2
        assert broker.alerts_emitted("t") == 0

    def test_alert_emitted_at_threshold(
        self, broker: InMemoryBrokerWithDLQAlerts, caplog: pytest.LogCaptureFixture
    ) -> None:
        broker.subscribe("t", always_fail)
        with caplog.at_level(logging.WARNING, logger=__name__):
            for i in range(3):  # threshold=3
                broker.publish("t", {"seq": i})
            broker.consume_all("t")

        assert broker.dlq_depth("t") == 3
        assert broker.alerts_emitted("t") >= 1

        warning_records = [r for r in caplog.records if "DLQ ALERT" in r.message]
        assert len(warning_records) >= 1
        alert_msg = warning_records[0].message
        assert "topic=t" in alert_msg
        assert "depth=3" in alert_msg
        assert "Sample" in alert_msg

    def test_alert_contains_message_sample(
        self, broker: InMemoryBrokerWithDLQAlerts, caplog: pytest.LogCaptureFixture
    ) -> None:
        broker.subscribe("t", always_fail)
        with caplog.at_level(logging.WARNING, logger=__name__):
            for i in range(5):
                broker.publish("t", {"seq": i})
            broker.consume_all("t")

        warning_records = [r for r in caplog.records if "DLQ ALERT" in r.message]
        assert warning_records, "Expected at least one DLQ ALERT warning"
        # Sample should include msg_id and reason
        assert "msg_id=" in warning_records[-1].message
        assert "reason=" in warning_records[-1].message

    def test_alert_independent_per_topic(
        self, broker: InMemoryBrokerWithDLQAlerts
    ) -> None:
        broker.subscribe("topic-a", always_fail)
        broker.subscribe("topic-b", always_fail)

        for i in range(3):
            broker.publish("topic-a", {"i": i})
        broker.consume_all("topic-a")

        assert broker.alerts_emitted("topic-a") >= 1
        assert broker.alerts_emitted("topic-b") == 0
```

---

## Exercise 5 — Broker factory: `BrokerFactory.create(broker_type, **config) -> AbstractBroker`

**Problem**: Build a `BrokerFactory.create(broker_type: str, **config) -> AbstractBroker` supporting "memory", "kafka" (stub), "redis" (stub).

**Explanation**: The factory pattern decouples broker selection from broker usage. Application code calls `BrokerFactory.create("memory")` in tests and `BrokerFactory.create("kafka", bootstrap_servers="...")` in production — the same `AbstractBroker` interface throughout. Adding a new broker type (e.g., "redis") requires only a new `elif` branch in the factory and a new implementation class.

```python
"""
ex5_broker_factory.py

BrokerFactory: create an AbstractBroker by name.
Supported types: "memory", "kafka" (stub), "redis" (stub).

Run tests: pytest ex5_broker_factory.py -v
"""

from __future__ import annotations

import logging
import os
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

import pytest

log = logging.getLogger(__name__)

DeliveryCallback = Callable[[Exception | None, "RecordMetadata | None"], None]


@dataclass
class RecordMetadata:
    topic: str
    partition: int
    offset: int


class AbstractBroker(ABC):
    """Minimal broker interface — same as day-5."""

    @abstractmethod
    def produce(self, topic: str, key: str | None, value: bytes, on_delivery: DeliveryCallback | None = None) -> None: ...

    @abstractmethod
    def flush(self, timeout_s: float = 5.0) -> None: ...

    @abstractmethod
    def poll(self, group_id: str, topic: str, partition: int, max_records: int = 10) -> list[tuple[int, bytes]]: ...

    @abstractmethod
    def commit(self, group_id: str, topic: str, partition: int, offset: int) -> None: ...

    @abstractmethod
    def create_topic(self, topic: str, num_partitions: int = 1) -> None: ...

    def partition_for_key(self, key: str | None, num_partitions: int) -> int:
        if key is None:
            return 0
        return hash(key.encode()) % num_partitions


# ---------------------------------------------------------------------------
# InMemoryBroker (production-grade in-memory implementation)
# ---------------------------------------------------------------------------


@dataclass
class _PartitionLog:
    log: list[bytes] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def append(self, value: bytes) -> int:
        with self._lock:
            offset = len(self.log)
            self.log.append(value)
            return offset

    def read_from(self, offset: int, max_records: int) -> list[tuple[int, bytes]]:
        with self._lock:
            return [(offset + i, v) for i, v in enumerate(self.log[offset : offset + max_records])]


class InMemoryBroker(AbstractBroker):
    def __init__(self) -> None:
        self._topics: dict[str, list[_PartitionLog]] = {}
        self._num_partitions: dict[str, int] = {}
        self._offsets: dict[str, dict[str, dict[int, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        self._lock = threading.Lock()

    def create_topic(self, topic: str, num_partitions: int = 1) -> None:
        with self._lock:
            if topic not in self._topics:
                self._topics[topic] = [_PartitionLog() for _ in range(num_partitions)]
                self._num_partitions[topic] = num_partitions

    def produce(self, topic: str, key: str | None, value: bytes, on_delivery: DeliveryCallback | None = None) -> None:
        if topic not in self._topics:
            raise KeyError(f"Topic '{topic}' does not exist.")
        pid = self.partition_for_key(key, self._num_partitions[topic])
        offset = self._topics[topic][pid].append(value)
        if on_delivery:
            on_delivery(None, RecordMetadata(topic=topic, partition=pid, offset=offset))

    def flush(self, timeout_s: float = 5.0) -> None:
        pass

    def poll(self, group_id: str, topic: str, partition: int, max_records: int = 10) -> list[tuple[int, bytes]]:
        offset = self._offsets[group_id][topic][partition]
        return self._topics[topic][partition].read_from(offset, max_records)

    def commit(self, group_id: str, topic: str, partition: int, offset: int) -> None:
        self._offsets[group_id][topic][partition] = offset

    def record_count(self, topic: str, partition: int = 0) -> int:
        return len(self._topics[topic][partition].log)


# ---------------------------------------------------------------------------
# KafkaBrokerStub — configuration-only; raises on actual I/O
# ---------------------------------------------------------------------------


class KafkaBrokerStub(AbstractBroker):
    """
    Stub Kafka broker for configuration validation.
    Holds the connection config but raises NotImplementedError on all I/O.
    Replace with the real KafkaBroker (confluent-kafka) for production.
    """

    def __init__(self, bootstrap_servers: str, **kwargs: Any) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.config = kwargs
        log.info(
            "KafkaBrokerStub: configured bootstrap_servers=%s extra=%s",
            bootstrap_servers,
            kwargs,
        )

    def create_topic(self, topic: str, num_partitions: int = 1) -> None:
        raise NotImplementedError("KafkaBrokerStub: connect to a real Kafka cluster.")

    def produce(self, topic: str, key: str | None, value: bytes, on_delivery: DeliveryCallback | None = None) -> None:
        raise NotImplementedError("KafkaBrokerStub: connect to a real Kafka cluster.")

    def flush(self, timeout_s: float = 5.0) -> None:
        raise NotImplementedError("KafkaBrokerStub: connect to a real Kafka cluster.")

    def poll(self, group_id: str, topic: str, partition: int, max_records: int = 10) -> list[tuple[int, bytes]]:
        raise NotImplementedError("KafkaBrokerStub: connect to a real Kafka cluster.")

    def commit(self, group_id: str, topic: str, partition: int, offset: int) -> None:
        raise NotImplementedError("KafkaBrokerStub: connect to a real Kafka cluster.")


# ---------------------------------------------------------------------------
# RedisBrokerStub — configuration-only; raises on actual I/O
# ---------------------------------------------------------------------------


class RedisBrokerStub(AbstractBroker):
    """
    Stub Redis Streams broker.
    Holds connection config; raises NotImplementedError on all I/O.
    Replace with a real Redis Streams implementation using `redis-py` for production.

    Key Redis Streams commands mapped to AbstractBroker:
        produce  → XADD {topic} * key value
        poll     → XREADGROUP GROUP {group_id} {consumer_id} COUNT {n} STREAMS {topic} >
        commit   → XACK {topic} {group_id} {message_id}
        create_topic → XGROUP CREATE {topic} {group_id} $ MKSTREAM
    """

    def __init__(self, host: str = "localhost", port: int = 6379, **kwargs: Any) -> None:
        self.host = host
        self.port = port
        self.config = kwargs
        log.info("RedisBrokerStub: configured host=%s port=%d", host, port)

    def create_topic(self, topic: str, num_partitions: int = 1) -> None:
        # Redis Streams are single-stream (no partitions); num_partitions ignored.
        raise NotImplementedError("RedisBrokerStub: connect to a real Redis instance.")

    def produce(self, topic: str, key: str | None, value: bytes, on_delivery: DeliveryCallback | None = None) -> None:
        raise NotImplementedError("RedisBrokerStub: connect to a real Redis instance.")

    def flush(self, timeout_s: float = 5.0) -> None:
        pass  # Redis XADD is synchronous; nothing to flush.

    def poll(self, group_id: str, topic: str, partition: int, max_records: int = 10) -> list[tuple[int, bytes]]:
        raise NotImplementedError("RedisBrokerStub: connect to a real Redis instance.")

    def commit(self, group_id: str, topic: str, partition: int, offset: int) -> None:
        raise NotImplementedError("RedisBrokerStub: connect to a real Redis instance.")


# ---------------------------------------------------------------------------
# BrokerFactory
# ---------------------------------------------------------------------------


class BrokerFactory:
    """
    Factory for creating AbstractBroker instances by name.

    Supported broker_type values:
        "memory" — InMemoryBroker (no external dependencies)
        "kafka"  — KafkaBrokerStub (requires bootstrap_servers kwarg)
        "redis"  — RedisBrokerStub (optional host/port kwargs)

    Usage:
        broker = BrokerFactory.create("memory")
        broker = BrokerFactory.create("kafka", bootstrap_servers="localhost:9092")
        broker = BrokerFactory.create("redis", host="redis.internal", port=6379)
    """

    _SUPPORTED = {"memory", "kafka", "redis"}

    @classmethod
    def create(cls, broker_type: str, **config: Any) -> AbstractBroker:
        """
        Instantiate and return the requested broker.

        Raises:
            ValueError: if broker_type is not one of the supported values.
            TypeError:  if required config kwargs are missing (e.g. kafka without bootstrap_servers).
        """
        bt = broker_type.lower().strip()

        if bt == "memory":
            if config:
                log.warning("BrokerFactory: 'memory' broker ignores config kwargs: %s", config)
            return InMemoryBroker()

        elif bt == "kafka":
            bootstrap_servers = config.get("bootstrap_servers") or os.getenv(
                "KAFKA_BOOTSTRAP_SERVERS"
            )
            if not bootstrap_servers:
                raise TypeError(
                    "BrokerFactory.create('kafka') requires bootstrap_servers kwarg "
                    "or KAFKA_BOOTSTRAP_SERVERS env var."
                )
            remaining = {k: v for k, v in config.items() if k != "bootstrap_servers"}
            return KafkaBrokerStub(bootstrap_servers=bootstrap_servers, **remaining)

        elif bt == "redis":
            host = config.get("host", "localhost")
            port = int(config.get("port", 6379))
            remaining = {k: v for k, v in config.items() if k not in {"host", "port"}}
            return RedisBrokerStub(host=host, port=port, **remaining)

        else:
            raise ValueError(
                f"Unknown broker_type: {broker_type!r}. "
                f"Supported: {sorted(cls._SUPPORTED)}"
            )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBrokerFactory:
    def test_create_memory_returns_in_memory_broker(self) -> None:
        broker = BrokerFactory.create("memory")
        assert isinstance(broker, InMemoryBroker)

    def test_create_memory_is_functional(self) -> None:
        broker = BrokerFactory.create("memory")
        broker.create_topic("t", num_partitions=1)
        broker.produce("t", key=None, value=b"hello")
        records = broker.poll("g", "t", partition=0)
        assert len(records) == 1
        assert records[0][1] == b"hello"

    def test_create_kafka_stub(self) -> None:
        broker = BrokerFactory.create("kafka", bootstrap_servers="localhost:9092")
        assert isinstance(broker, KafkaBrokerStub)
        assert broker.bootstrap_servers == "localhost:9092"

    def test_create_kafka_missing_bootstrap_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
        with pytest.raises(TypeError, match="bootstrap_servers"):
            BrokerFactory.create("kafka")

    def test_create_redis_stub(self) -> None:
        broker = BrokerFactory.create("redis", host="redis.internal", port=6380)
        assert isinstance(broker, RedisBrokerStub)
        assert broker.host == "redis.internal"
        assert broker.port == 6380

    def test_create_redis_defaults(self) -> None:
        broker = BrokerFactory.create("redis")
        assert isinstance(broker, RedisBrokerStub)
        assert broker.host == "localhost"
        assert broker.port == 6379

    def test_unknown_broker_type_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown broker_type"):
            BrokerFactory.create("rabbitmq")

    def test_case_insensitive(self) -> None:
        broker = BrokerFactory.create("MEMORY")
        assert isinstance(broker, InMemoryBroker)

    def test_all_implementations_satisfy_abstract_interface(self) -> None:
        """All factory outputs must be AbstractBroker instances."""
        brokers = [
            BrokerFactory.create("memory"),
            BrokerFactory.create("kafka", bootstrap_servers="localhost:9092"),
            BrokerFactory.create("redis"),
        ]
        for broker in brokers:
            assert isinstance(broker, AbstractBroker), f"{type(broker)} does not satisfy AbstractBroker"

    def test_kafka_stub_raises_on_io(self) -> None:
        broker = BrokerFactory.create("kafka", bootstrap_servers="localhost:9092")
        with pytest.raises(NotImplementedError):
            broker.create_topic("t")

    def test_redis_stub_raises_on_io(self) -> None:
        broker = BrokerFactory.create("redis")
        with pytest.raises(NotImplementedError):
            broker.produce("t", key=None, value=b"x")
```

---

## Summary

| Exercise | Core concept | Key insight |
|---|---|---|
| 1 | DLQ filtered replay | Predicate keeps non-matching records in DLQ; only replay what's fixable |
| 2 | Per-key ordering | `hash(key) % n` → deterministic partition; offsets within partition are write-ordered |
| 3 | Backpressure producer | Linear throttle at 50-90%; hard pause above 90%; resume when consumer drains |
| 4 | DLQ alerting | Emit WARNING with depth + sample after each DLQ routing; threshold configurable |
| 5 | BrokerFactory | Registry pattern with `create(type, **config)`; stubs for non-memory backends |
