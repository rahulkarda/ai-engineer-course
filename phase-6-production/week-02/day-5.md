# Phase 6 · Week 02 · Day 5 — Producing and Consuming in Python: confluent-kafka, aiokafka, and a Mockable Abstraction

> Objective: Build a typed, testable broker abstraction that runs in-memory for tests and against real Kafka in production.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

---

## Concept

Two dominant Python Kafka clients exist:

| | confluent-kafka | aiokafka |
|---|---|---|
| Transport | librdkafka (C) via cffi | pure Python asyncio |
| API style | sync (blocking poll loop) | async/await |
| Throughput | very high — librdkafka is battle-tested | good — overhead from Python event loop |
| Best for | CPU-bound pipelines, workers, stream processors | I/O-bound async services (FastAPI, aiohttp) |
| Install | `uv add confluent-kafka` | `uv add aiokafka` |

The critical insight: **your application should not import either directly.** Both clients are infrastructure. Wrap them behind an abstract interface so:
- Unit tests run without a broker
- Integration tests can use an in-memory broker or a test container
- Swapping libraries (or adding a schema registry) touches one class, not every caller

This is the **Repository pattern** applied to messaging.

---

## How it works internally

### Producer path
1. `produce(topic, key, value)` serializes key and value (bytes)
2. A **partitioner** maps key → partition (murmur2 hash by default in librdkafka)
3. The record is batched into a per-partition buffer (`queue.buffering.max.ms`, `batch.size`)
4. A background thread in librdkafka drains the buffer to the broker leader
5. The broker appends and, depending on `acks`, responds with the assigned offset
6. A **delivery callback** fires with `(error, record_metadata)` — the only way to confirm delivery

`acks` values:
- `0`: producer does not wait — maximum throughput, zero durability
- `1`: leader wrote to its local log — fast, loses data if leader crashes before replication
- `all` (or `-1`): all ISR replicas acknowledged — safest, adds latency proportional to ISR sync

### Consumer path
1. `subscribe([topic])` registers the consumer with the group coordinator
2. Coordinator assigns partitions (rebalance protocol: eager or cooperative sticky)
3. `poll(timeout_ms)` fetches a batch from the broker for each assigned partition
4. Application processes records — **nothing is auto-committed in the examples below**
5. `commit(offsets)` tells the broker the new position for this group
6. On rebalance (member joins/leaves), Kafka calls the `on_revoke` / `on_assign` callbacks — use these to commit in-flight offsets before they are reassigned

### Why manual commit matters
Auto-commit fires on a timer. If the consumer crashes between the timer firing and processing completing, messages are silently skipped. Manual commit — committing only after processing — gives at-least-once delivery. Idempotent consumers + manual commit = effectively-once.

### Pydantic as lightweight schema
Avro and JSON Schema (via schema registry) provide schema evolution and cross-language guarantees. For Python-internal services, Pydantic v2 provides:
- Runtime validation on deserialization
- Forward compatibility via `model_config = ConfigDict(extra="ignore")`
- Documentation via field descriptions

Use Pydantic at the application boundary (deserialize once, pass typed models). Keep serialization in the broker layer (JSON bytes or msgpack).

---

## Worked example

```python
"""
broker_abstraction.py

Mockable broker abstraction for Kafka.

Architecture:
  AbstractBroker (ABC)
    ├── InMemoryBroker   -- used in tests
    └── KafkaBroker      -- wraps confluent-kafka (requires `uv add confluent-kafka`)

Usage:
  python broker_abstraction.py          # runs demo with InMemoryBroker
  pytest test_broker_abstraction.py     # runs tests with InMemoryBroker

No external dependencies required for InMemoryBroker.
For KafkaBroker: uv add confluent-kafka
For AsyncKafkaBroker: uv add aiokafka
"""

from __future__ import annotations

import json
import logging
import os
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, field_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message schema — Pydantic v2
# ---------------------------------------------------------------------------


class OrderEvent(BaseModel):
    """
    Typed message for the 'orders' topic.

    model_config extra='ignore' makes this forward-compatible:
    new fields from a newer producer are silently dropped, not rejected.
    """

    model_config = ConfigDict(extra="ignore")

    order_id: str
    user_id: str
    amount_cents: int
    event_type: str  # "placed" | "cancelled" | "fulfilled"

    @field_validator("amount_cents")
    @classmethod
    def non_negative_amount(cls, v: int) -> int:
        if v < 0:
            raise ValueError("amount_cents must be >= 0")
        return v

    @field_validator("event_type")
    @classmethod
    def valid_event_type(cls, v: str) -> str:
        allowed = {"placed", "cancelled", "fulfilled"}
        if v not in allowed:
            raise ValueError(f"event_type must be one of {allowed}")
        return v


# ---------------------------------------------------------------------------
# Abstract broker interface
# ---------------------------------------------------------------------------


@dataclass
class RecordMetadata:
    """Result of a successful produce call."""

    topic: str
    partition: int
    offset: int


DeliveryCallback = Callable[[Exception | None, RecordMetadata | None], None]


class AbstractBroker(ABC):
    """
    Minimal broker interface.

    Implementations:
    - InMemoryBroker: in-process, no I/O — for unit tests
    - KafkaBroker: wraps confluent-kafka — for production

    All methods are synchronous. For async usage, wrap in asyncio.to_thread().
    """

    @abstractmethod
    def produce(
        self,
        topic: str,
        key: str | None,
        value: bytes,
        on_delivery: DeliveryCallback | None = None,
    ) -> None:
        """
        Enqueue a record for delivery.
        on_delivery(error, metadata) fires when the broker acknowledges.
        """

    @abstractmethod
    def flush(self, timeout_s: float = 5.0) -> None:
        """Wait until all enqueued records are delivered or timeout expires."""

    @abstractmethod
    def poll(
        self,
        group_id: str,
        topic: str,
        partition: int,
        max_records: int = 10,
    ) -> list[tuple[int, bytes]]:
        """
        Return up to max_records from (group_id, topic, partition).
        Returns list of (offset, value_bytes).
        Caller must call commit() after processing.
        """

    @abstractmethod
    def commit(
        self,
        group_id: str,
        topic: str,
        partition: int,
        offset: int,
    ) -> None:
        """Advance the committed offset for this consumer group + partition."""

    @abstractmethod
    def create_topic(self, topic: str, num_partitions: int = 1) -> None:
        """Create a topic if it does not already exist."""

    def partition_for_key(self, key: str | None, num_partitions: int) -> int:
        """Default partitioner: murmur2-like hash of key bytes."""
        if key is None:
            return 0
        return hash(key.encode()) % num_partitions


# ---------------------------------------------------------------------------
# InMemoryBroker — test implementation
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
            slice_ = self.log[offset : offset + max_records]
            return [(offset + i, v) for i, v in enumerate(slice_)]


class InMemoryBroker(AbstractBroker):
    """
    In-process broker. Thread-safe, zero I/O.

    Suitable for:
    - pytest unit tests
    - local development without a Kafka cluster
    - CI pipelines where spinning up Kafka is impractical
    """

    def __init__(self) -> None:
        # topics[topic][partition] = _PartitionLog
        self._topics: dict[str, list[_PartitionLog]] = {}
        self._num_partitions: dict[str, int] = {}
        # offsets[group_id][topic][partition] = int
        self._offsets: dict[str, dict[str, dict[int, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        self._pending_callbacks: list[tuple[DeliveryCallback, RecordMetadata]] = []
        self._lock = threading.Lock()

    def create_topic(self, topic: str, num_partitions: int = 1) -> None:
        with self._lock:
            if topic in self._topics:
                return
            self._topics[topic] = [_PartitionLog() for _ in range(num_partitions)]
            self._num_partitions[topic] = num_partitions
            logger.info("InMemoryBroker: created topic '%s' (%d partitions)", topic, num_partitions)

    def produce(
        self,
        topic: str,
        key: str | None,
        value: bytes,
        on_delivery: DeliveryCallback | None = None,
    ) -> None:
        if topic not in self._topics:
            raise KeyError(f"Topic '{topic}' does not exist. Call create_topic() first.")
        num_parts = self._num_partitions[topic]
        partition = self.partition_for_key(key, num_parts)
        offset = self._topics[topic][partition].append(value)
        logger.debug("InMemoryBroker produce: %s[%d]@%d", topic, partition, offset)
        if on_delivery:
            meta = RecordMetadata(topic=topic, partition=partition, offset=offset)
            on_delivery(None, meta)

    def flush(self, timeout_s: float = 5.0) -> None:
        # In-memory: produce is synchronous, nothing to flush
        pass

    def poll(
        self,
        group_id: str,
        topic: str,
        partition: int,
        max_records: int = 10,
    ) -> list[tuple[int, bytes]]:
        offset = self._offsets[group_id][topic][partition]
        return self._topics[topic][partition].read_from(offset, max_records)

    def commit(
        self,
        group_id: str,
        topic: str,
        partition: int,
        offset: int,
    ) -> None:
        self._offsets[group_id][topic][partition] = offset
        logger.debug("InMemoryBroker commit: group=%s %s[%d]@%d", group_id, topic, partition, offset)

    def record_count(self, topic: str, partition: int) -> int:
        """Test helper: total records in a partition."""
        return len(self._topics[topic][partition].log)


# ---------------------------------------------------------------------------
# KafkaBroker — production implementation (requires confluent-kafka)
# ---------------------------------------------------------------------------


class KafkaBroker(AbstractBroker):
    """
    Thin wrapper around confluent-kafka.

    Install: uv add confluent-kafka

    Config values are passed through to librdkafka.
    See: https://github.com/confluentinc/librdkafka/blob/master/CONFIGURATION.md
    """

    def __init__(self, bootstrap_servers: str | None = None) -> None:
        # Import here so the rest of the module works without confluent-kafka installed
        try:
            from confluent_kafka import Consumer, Producer
            from confluent_kafka.admin import AdminClient, NewTopic
        except ImportError as e:
            raise ImportError(
                "confluent-kafka not installed. Run: uv add confluent-kafka"
            ) from e

        servers = bootstrap_servers or os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
        )
        self._producer = Producer(
            {
                "bootstrap.servers": servers,
                "acks": "all",           # wait for all ISR replicas
                "enable.idempotence": True,
                "linger.ms": 5,          # small batch window
            }
        )
        self._consumer_cache: dict[str, Any] = {}
        self._admin = AdminClient({"bootstrap.servers": servers})
        self._Consumer = Consumer
        self._NewTopic = NewTopic
        self._num_partitions: dict[str, int] = {}

    def create_topic(self, topic: str, num_partitions: int = 1) -> None:
        new_topic = self._NewTopic(topic, num_partitions=num_partitions, replication_factor=1)
        fs = self._admin.create_topics([new_topic])
        for t, f in fs.items():
            try:
                f.result()
                logger.info("KafkaBroker: created topic '%s'", t)
            except Exception as e:
                logger.warning("KafkaBroker create_topic '%s': %s", t, e)
        self._num_partitions[topic] = num_partitions

    def produce(
        self,
        topic: str,
        key: str | None,
        value: bytes,
        on_delivery: DeliveryCallback | None = None,
    ) -> None:
        def _cb(err: Any, msg: Any) -> None:
            if on_delivery:
                if err:
                    on_delivery(Exception(str(err)), None)
                else:
                    meta = RecordMetadata(
                        topic=msg.topic(),
                        partition=msg.partition(),
                        offset=msg.offset(),
                    )
                    on_delivery(None, meta)

        self._producer.produce(
            topic=topic,
            key=key.encode() if key else None,
            value=value,
            on_delivery=_cb,
        )
        # Serve delivery callbacks without blocking
        self._producer.poll(0)

    def flush(self, timeout_s: float = 5.0) -> None:
        remaining = self._producer.flush(timeout=timeout_s)
        if remaining > 0:
            logger.warning("KafkaBroker.flush: %d messages not delivered within %.1fs", remaining, timeout_s)

    def poll(
        self,
        group_id: str,
        topic: str,
        partition: int,
        max_records: int = 10,
    ) -> list[tuple[int, bytes]]:
        consumer = self._get_consumer(group_id, topic)
        records: list[tuple[int, bytes]] = []
        for _ in range(max_records):
            msg = consumer.poll(timeout=0.5)
            if msg is None:
                break
            if msg.error():
                logger.error("KafkaBroker consumer error: %s", msg.error())
                break
            records.append((msg.offset(), msg.value()))
        return records

    def commit(
        self,
        group_id: str,
        topic: str,
        partition: int,
        offset: int,
    ) -> None:
        consumer = self._get_consumer(group_id, topic)
        from confluent_kafka import TopicPartition

        consumer.commit(
            offsets=[TopicPartition(topic, partition, offset)],
            asynchronous=False,
        )

    def _get_consumer(self, group_id: str, topic: str) -> Any:
        key = f"{group_id}::{topic}"
        if key not in self._consumer_cache:
            consumer = self._Consumer(
                {
                    "bootstrap.servers": self._producer._producer_config.get(  # type: ignore[attr-defined]
                        "bootstrap.servers", "localhost:9092"
                    ),
                    "group.id": group_id,
                    "auto.offset.reset": "earliest",
                    "enable.auto.commit": False,  # manual commit only
                }
            )
            consumer.subscribe([topic])
            self._consumer_cache[key] = consumer
        return self._consumer_cache[key]


# ---------------------------------------------------------------------------
# Typed producer / consumer — thin wrappers adding serialization
# ---------------------------------------------------------------------------


class OrderProducer:
    """
    Produces OrderEvent messages to the 'orders' topic.
    Serializes to JSON bytes. Uses the broker abstraction — no Kafka import here.
    """

    TOPIC = "orders"

    def __init__(self, broker: AbstractBroker, num_partitions: int = 3) -> None:
        self._broker = broker
        self._num_partitions = num_partitions
        self._broker.create_topic(self.TOPIC, num_partitions)

    def send(self, event: OrderEvent, on_delivery: DeliveryCallback | None = None) -> None:
        payload = event.model_dump_json().encode()
        self._broker.produce(
            topic=self.TOPIC,
            key=event.user_id,  # same user always lands in same partition
            value=payload,
            on_delivery=on_delivery,
        )

    def flush(self) -> None:
        self._broker.flush()


class OrderConsumer:
    """
    Consumes OrderEvent messages from the 'orders' topic.
    Deserializes JSON bytes → OrderEvent via Pydantic validation.
    Manual commit after processing.
    """

    TOPIC = "orders"

    def __init__(
        self,
        broker: AbstractBroker,
        group_id: str,
        partition: int,
    ) -> None:
        self._broker = broker
        self._group_id = group_id
        self._partition = partition

    def poll_and_process(
        self,
        handler: Callable[[OrderEvent], None],
        max_records: int = 10,
    ) -> int:
        """
        Poll up to max_records, validate each with Pydantic, call handler, commit.
        Returns number of records processed.
        """
        records = self._broker.poll(
            self._group_id, self.TOPIC, self._partition, max_records
        )
        if not records:
            return 0
        processed = 0
        last_offset = -1
        for offset, raw in records:
            try:
                event = OrderEvent.model_validate_json(raw)
                handler(event)
                last_offset = offset
                processed += 1
            except Exception as exc:
                # Dead letter: log and skip — do not lose the offset
                logger.error("Failed to process offset %d: %s", offset, exc)
                last_offset = offset  # still advance past the bad record
        if last_offset >= 0:
            self._broker.commit(self._group_id, self.TOPIC, self._partition, last_offset + 1)
        return processed


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def run_demo(broker: AbstractBroker) -> None:
    logger.info("=== Demo start ===")

    producer = OrderProducer(broker, num_partitions=3)

    events = [
        OrderEvent(order_id="o1", user_id="u1", amount_cents=1000, event_type="placed"),
        OrderEvent(order_id="o2", user_id="u2", amount_cents=500, event_type="placed"),
        OrderEvent(order_id="o3", user_id="u1", amount_cents=0, event_type="cancelled"),
        OrderEvent(order_id="o4", user_id="u3", amount_cents=2500, event_type="fulfilled"),
        OrderEvent(order_id="o5", user_id="u2", amount_cents=750, event_type="placed"),
        OrderEvent(order_id="o6", user_id="u1", amount_cents=100, event_type="placed"),
    ]

    delivered: list[RecordMetadata] = []

    def on_delivery(err: Exception | None, meta: RecordMetadata | None) -> None:
        if err:
            logger.error("Delivery failed: %s", err)
        elif meta:
            delivered.append(meta)
            logger.info("Delivered: %s[%d]@%d", meta.topic, meta.partition, meta.offset)

    for event in events:
        producer.send(event, on_delivery=on_delivery)
    producer.flush()

    logger.info("Delivered %d messages", len(delivered))

    # Consume from each partition
    def handle(event: OrderEvent) -> None:
        logger.info("Processed: order_id=%s user_id=%s type=%s", event.order_id, event.user_id, event.event_type)

    total = 0
    for partition in range(3):
        consumer = OrderConsumer(broker, group_id="app-group", partition=partition)
        total += consumer.poll_and_process(handle)

    logger.info("=== Demo end: processed %d records ===", total)


if __name__ == "__main__":
    broker = InMemoryBroker()
    run_demo(broker)
```

---

### Test suite

```python
"""
test_broker_abstraction.py

Unit tests for the broker abstraction.
All tests use InMemoryBroker — no Kafka required.

Run: pytest test_broker_abstraction.py -v
"""

from __future__ import annotations

import json
import pytest

from broker_abstraction import (
    AbstractBroker,
    InMemoryBroker,
    OrderConsumer,
    OrderEvent,
    OrderProducer,
    RecordMetadata,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def broker() -> InMemoryBroker:
    return InMemoryBroker()


@pytest.fixture()
def producer(broker: InMemoryBroker) -> OrderProducer:
    return OrderProducer(broker, num_partitions=3)


# ---------------------------------------------------------------------------
# InMemoryBroker unit tests
# ---------------------------------------------------------------------------


class TestInMemoryBroker:
    def test_create_topic(self, broker: InMemoryBroker) -> None:
        broker.create_topic("test-topic", num_partitions=2)
        assert "test-topic" in broker._topics
        assert len(broker._topics["test-topic"]) == 2

    def test_create_topic_idempotent(self, broker: InMemoryBroker) -> None:
        broker.create_topic("test-topic", num_partitions=2)
        broker.create_topic("test-topic", num_partitions=5)  # should not replace
        assert len(broker._topics["test-topic"]) == 2

    def test_produce_and_poll(self, broker: InMemoryBroker) -> None:
        broker.create_topic("t", num_partitions=1)
        broker.produce("t", key=None, value=b"hello")
        broker.produce("t", key=None, value=b"world")

        records = broker.poll("grp", "t", partition=0)
        assert len(records) == 2
        assert records[0] == (0, b"hello")
        assert records[1] == (1, b"world")

    def test_commit_advances_offset(self, broker: InMemoryBroker) -> None:
        broker.create_topic("t", num_partitions=1)
        broker.produce("t", key=None, value=b"a")
        broker.produce("t", key=None, value=b"b")
        broker.produce("t", key=None, value=b"c")

        # First poll: reads from 0
        records = broker.poll("grp", "t", partition=0)
        assert len(records) == 3
        broker.commit("grp", "t", partition=0, offset=3)

        # Second poll: nothing new
        records = broker.poll("grp", "t", partition=0)
        assert records == []

        # Produce more, then poll
        broker.produce("t", key=None, value=b"d")
        records = broker.poll("grp", "t", partition=0)
        assert len(records) == 1
        assert records[0] == (3, b"d")

    def test_two_groups_independent(self, broker: InMemoryBroker) -> None:
        broker.create_topic("t", num_partitions=1)
        broker.produce("t", key=None, value=b"msg")

        records_a = broker.poll("group-a", "t", partition=0)
        records_b = broker.poll("group-b", "t", partition=0)

        assert len(records_a) == 1
        assert len(records_b) == 1  # both groups see the same message

    def test_delivery_callback_fires(self, broker: InMemoryBroker) -> None:
        broker.create_topic("t", num_partitions=1)
        received: list[RecordMetadata] = []

        def cb(err: Exception | None, meta: RecordMetadata | None) -> None:
            if meta:
                received.append(meta)

        broker.produce("t", key=None, value=b"x", on_delivery=cb)
        assert len(received) == 1
        assert received[0].offset == 0

    def test_same_key_same_partition(self, broker: InMemoryBroker) -> None:
        broker.create_topic("t", num_partitions=4)
        partitions_seen: set[int] = set()
        for _ in range(5):
            meta_list: list[RecordMetadata] = []
            broker.produce(
                "t",
                key="stable-key",
                value=b"v",
                on_delivery=lambda e, m: meta_list.append(m) if m else None,
            )
            partitions_seen.add(meta_list[0].partition)
        assert len(partitions_seen) == 1, "same key must always land in same partition"

    def test_produce_to_nonexistent_topic_raises(self, broker: InMemoryBroker) -> None:
        with pytest.raises(KeyError, match="does not exist"):
            broker.produce("no-such-topic", key=None, value=b"x")


# ---------------------------------------------------------------------------
# OrderProducer / OrderConsumer integration tests
# ---------------------------------------------------------------------------


class TestOrderProducerConsumer:
    def test_round_trip(self, broker: InMemoryBroker) -> None:
        producer = OrderProducer(broker, num_partitions=1)
        event = OrderEvent(order_id="o1", user_id="u1", amount_cents=100, event_type="placed")
        producer.send(event)
        producer.flush()

        received: list[OrderEvent] = []
        consumer = OrderConsumer(broker, group_id="test-group", partition=0)
        consumer.poll_and_process(received.append)

        assert len(received) == 1
        assert received[0].order_id == "o1"
        assert received[0].amount_cents == 100

    def test_manual_commit_prevents_reread(self, broker: InMemoryBroker) -> None:
        producer = OrderProducer(broker, num_partitions=1)
        for i in range(3):
            producer.send(
                OrderEvent(order_id=f"o{i}", user_id="u1", amount_cents=i * 10, event_type="placed")
            )

        consumer = OrderConsumer(broker, group_id="test-group", partition=0)
        first_batch: list[OrderEvent] = []
        consumer.poll_and_process(first_batch.append)
        assert len(first_batch) == 3

        second_batch: list[OrderEvent] = []
        consumer.poll_and_process(second_batch.append)
        assert len(second_batch) == 0  # already committed

    def test_bad_record_skipped_good_records_committed(self, broker: InMemoryBroker) -> None:
        """Corrupt message in the middle should not block subsequent messages."""
        broker.create_topic("orders", num_partitions=1)
        # Produce valid, corrupt, valid
        broker.produce("orders", key="u1", value=b'{"order_id":"o1","user_id":"u1","amount_cents":10,"event_type":"placed"}')
        broker.produce("orders", key="u1", value=b"not-json{{{")
        broker.produce("orders", key="u1", value=b'{"order_id":"o3","user_id":"u1","amount_cents":30,"event_type":"placed"}')

        received: list[OrderEvent] = []
        consumer = OrderConsumer(broker, group_id="g", partition=0)
        processed = consumer.poll_and_process(received.append)

        assert processed == 2  # corrupt record counted but not appended to received
        assert len(received) == 2
        assert received[0].order_id == "o1"
        assert received[1].order_id == "o3"

    def test_pydantic_validation_rejects_invalid_event_type(self) -> None:
        with pytest.raises(Exception, match="event_type"):
            OrderEvent(order_id="x", user_id="u", amount_cents=0, event_type="refunded")

    def test_pydantic_forward_compat_extra_fields_ignored(self) -> None:
        raw = b'{"order_id":"o1","user_id":"u1","amount_cents":0,"event_type":"placed","new_field":"future"}'
        event = OrderEvent.model_validate_json(raw)
        assert event.order_id == "o1"

    def test_broker_is_abstract(self) -> None:
        """AbstractBroker cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AbstractBroker()  # type: ignore[abstract]
```

---

## Exercises

1. **Async wrapper**: Wrap `InMemoryBroker.produce` and `InMemoryBroker.poll` in an `AsyncInMemoryBroker` class that exposes `async def produce(...)` and `async def poll(...)` using `asyncio.to_thread`. Write a pytest-asyncio test that produces 10 events and consumes them concurrently with two coroutines.

2. **Dead letter topic**: When `poll_and_process` encounters a record that fails Pydantic validation, instead of logging and skipping, produce the raw bytes to a `orders.dlq` topic with the error reason as the key. Add a test verifying the DLQ receives exactly one record for a corrupt message.

3. **Rebalance callback**: Add `on_partitions_assigned(partitions)` and `on_partitions_revoked(partitions)` hooks to `AbstractBroker`. In `InMemoryBroker`, fire `on_partitions_revoked` before reassigning any partition to a new consumer. Write a test showing committed offsets survive a simulated rebalance.

4. **Batch producer**: Modify `OrderProducer.send` to buffer events and only call `broker.produce` when either the buffer reaches 10 events or `flush()` is called explicitly. Write tests for both flush paths (buffer-full and explicit flush).

5. **Throughput benchmark**: Using `InMemoryBroker`, write a script that produces 100,000 events and measures produce throughput (events/sec) and consume throughput (events/sec) separately. Profile where time is spent (serialization vs. broker append vs. Pydantic validation) and suggest one optimization for each.

6. **aiokafka variant**: Study the aiokafka docs (link below) and write an `AsyncKafkaBroker` that implements `AbstractBroker` using `aiokafka.AIOKafkaProducer` and `aiokafka.AIOKafkaConsumer`. The class should require `uv add aiokafka`. The existing test suite should pass unchanged when `InMemoryBroker` is swapped with `AsyncKafkaBroker` pointed at a local cluster.

(Solutions: ../solutions/week-02/day-5.md)

---

## Self-check

- [ ] I can explain the key difference between confluent-kafka and aiokafka and name a scenario where each is preferred.
- [ ] I can describe what `acks=all` means and why it costs latency.
- [ ] I can implement `AbstractBroker` from memory, including the method signatures and their contracts.
- [ ] I can explain why manual commit gives at-least-once delivery and what extra guarantee is needed for effectively-once.
- [ ] I can write an `InMemoryBroker` unit test that verifies two consumer groups read the same data independently.
- [ ] I can explain why Pydantic's `extra="ignore"` matters for forward compatibility in message consumers.
- [ ] I can describe the delivery callback pattern and what happens if `flush()` is never called.

---

## Supplementary links

- [confluent-kafka-python docs](https://docs.confluent.io/platform/current/clients/confluent-kafka-python/html/index.html)
- [aiokafka docs](https://aiokafka.readthedocs.io/en/stable/)
- [librdkafka configuration reference](https://github.com/confluentinc/librdkafka/blob/master/CONFIGURATION.md)
- [Kafka consumer group protocol internals](https://developer.confluent.io/courses/architecture/consumer-group-protocol/)
- [Pydantic v2 model config](https://docs.pydantic.dev/latest/concepts/config/)
- [At-least-once vs. exactly-once semantics](https://kafka.apache.org/documentation/#semantics)
