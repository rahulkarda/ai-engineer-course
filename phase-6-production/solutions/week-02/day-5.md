# Phase 6 · Week 02 · Day 5 — Solutions

> Solutions for: Producing and Consuming in Python — Mockable Abstraction

---

## Exercise 1 — Async wrapper: `AsyncInMemoryBroker` using `asyncio.to_thread`

**Problem**: Wrap `InMemoryBroker.produce` and `InMemoryBroker.poll` in an `AsyncInMemoryBroker` class exposing `async def produce(...)` and `async def poll(...)` using `asyncio.to_thread`. Write a pytest-asyncio test that produces 10 events and consumes them concurrently with two coroutines.

**Explanation**: `asyncio.to_thread` dispatches a synchronous callable to the default `ThreadPoolExecutor` and returns a coroutine. This lets you use a thread-safe synchronous broker inside an async service (FastAPI, aiohttp) without blocking the event loop. The broker itself does not need to change — only the async wrapper is new.

```python
"""
ex1_async_broker.py

AsyncInMemoryBroker: async wrapper around InMemoryBroker using asyncio.to_thread.

Run tests:
    pytest ex1_async_broker.py -v       # requires pytest-asyncio
    pip install pytest pytest-asyncio pydantic
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

import pytest
from pydantic import BaseModel, ConfigDict, field_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared models (copied from day-5 worked example)
# ---------------------------------------------------------------------------


class OrderEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    order_id: str
    user_id: str
    amount_cents: int
    event_type: str

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


@dataclass
class RecordMetadata:
    topic: str
    partition: int
    offset: int


DeliveryCallback = Callable[[Exception | None, RecordMetadata | None], None]


# ---------------------------------------------------------------------------
# Synchronous InMemoryBroker (unchanged from day-5)
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


class InMemoryBroker:
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

    def partition_for_key(self, key: str | None, num_partitions: int) -> int:
        if key is None:
            return 0
        return hash(key.encode()) % num_partitions

    def produce(
        self,
        topic: str,
        key: str | None,
        value: bytes,
        on_delivery: DeliveryCallback | None = None,
    ) -> None:
        if topic not in self._topics:
            raise KeyError(f"Topic '{topic}' does not exist.")
        num_parts = self._num_partitions[topic]
        partition = self.partition_for_key(key, num_parts)
        offset = self._topics[topic][partition].append(value)
        if on_delivery:
            on_delivery(None, RecordMetadata(topic=topic, partition=partition, offset=offset))

    def flush(self, timeout_s: float = 5.0) -> None:
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

    def commit(self, group_id: str, topic: str, partition: int, offset: int) -> None:
        self._offsets[group_id][topic][partition] = offset

    def record_count(self, topic: str, partition: int) -> int:
        return len(self._topics[topic][partition].log)

    def num_partitions(self, topic: str) -> int:
        return self._num_partitions[topic]


# ---------------------------------------------------------------------------
# AsyncInMemoryBroker
# ---------------------------------------------------------------------------


class AsyncInMemoryBroker:
    """
    Async wrapper around InMemoryBroker.

    Every method dispatches to the synchronous broker via asyncio.to_thread,
    which runs it in a thread pool without blocking the event loop.

    Usage in an async service:
        broker = AsyncInMemoryBroker()
        await broker.create_topic("orders", num_partitions=3)
        await broker.produce("orders", key="u1", value=b"...")
        records = await broker.poll("group", "orders", partition=0)
    """

    def __init__(self, sync_broker: InMemoryBroker | None = None) -> None:
        self._broker = sync_broker or InMemoryBroker()

    async def create_topic(self, topic: str, num_partitions: int = 1) -> None:
        await asyncio.to_thread(self._broker.create_topic, topic, num_partitions)

    async def produce(
        self,
        topic: str,
        key: str | None,
        value: bytes,
        on_delivery: DeliveryCallback | None = None,
    ) -> None:
        await asyncio.to_thread(self._broker.produce, topic, key, value, on_delivery)

    async def flush(self, timeout_s: float = 5.0) -> None:
        await asyncio.to_thread(self._broker.flush, timeout_s)

    async def poll(
        self,
        group_id: str,
        topic: str,
        partition: int,
        max_records: int = 10,
    ) -> list[tuple[int, bytes]]:
        return await asyncio.to_thread(self._broker.poll, group_id, topic, partition, max_records)

    async def commit(self, group_id: str, topic: str, partition: int, offset: int) -> None:
        await asyncio.to_thread(self._broker.commit, group_id, topic, partition, offset)

    def record_count(self, topic: str, partition: int) -> int:
        return self._broker.record_count(topic, partition)

    def num_partitions(self, topic: str) -> int:
        return self._broker.num_partitions(topic)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def async_broker() -> AsyncInMemoryBroker:
    return AsyncInMemoryBroker()


@pytest.mark.asyncio
async def test_produce_and_poll(async_broker: AsyncInMemoryBroker) -> None:
    await async_broker.create_topic("t", num_partitions=1)
    await async_broker.produce("t", key=None, value=b"hello")
    records = await async_broker.poll("g", "t", partition=0)
    assert len(records) == 1
    assert records[0] == (0, b"hello")


@pytest.mark.asyncio
async def test_concurrent_produce_consume_10_events(async_broker: AsyncInMemoryBroker) -> None:
    """
    Produce 10 OrderEvent messages from one coroutine.
    Consume them concurrently with two coroutines (each reads from partition 0).
    All 10 events must be processed exactly once.
    """
    TOPIC = "orders-async"
    GROUP = "test-group"
    await async_broker.create_topic(TOPIC, num_partitions=1)

    events = [
        OrderEvent(
            order_id=f"o{i}",
            user_id=f"u{i % 3}",
            amount_cents=i * 100,
            event_type="placed",
        )
        for i in range(10)
    ]

    # Producer coroutine
    async def producer_coro() -> None:
        for event in events:
            payload = event.model_dump_json().encode()
            await async_broker.produce(TOPIC, key=event.user_id, value=payload)
        await async_broker.flush()
        logger.info("Producer: sent %d events", len(events))

    consumed: list[OrderEvent] = []
    lock = asyncio.Lock()

    # Consumer coroutine — both read from the same partition; first one drains it
    async def consumer_coro(consumer_id: str) -> None:
        while True:
            records = await async_broker.poll(GROUP, TOPIC, partition=0, max_records=3)
            if not records:
                break
            for offset, raw in records:
                event = OrderEvent.model_validate_json(raw)
                async with lock:
                    consumed.append(event)
                logger.info("[%s] consumed order_id=%s offset=%d", consumer_id, event.order_id, offset)
            await async_broker.commit(GROUP, TOPIC, partition=0, offset=records[-1][0] + 1)

    # Run producer first, then two consumers concurrently
    await producer_coro()
    await asyncio.gather(consumer_coro("consumer-1"), consumer_coro("consumer-2"))

    assert len(consumed) == 10, f"Expected 10 events, got {len(consumed)}"
    order_ids = {e.order_id for e in consumed}
    assert order_ids == {f"o{i}" for i in range(10)}, "All 10 order IDs must be present"
    logger.info("PASS: all 10 events consumed exactly once via async broker.")
```

---

## Exercise 2 — DLQ in AbstractBroker: failed messages go to `{topic}.dlq`

**Problem**: When `poll_and_process` encounters a record that fails Pydantic validation, produce the raw bytes to a `orders.dlq` topic with the error reason as the key. Add a test verifying the DLQ receives exactly one record for a corrupt message.

**Explanation**: The DLQ is a regular topic. The consumer catches the validation error, serialises the original raw bytes along with metadata, and produces to `{original_topic}.dlq`. The original offset is committed so the main consumer advances past the bad record. A separate consumer group can read from the DLQ for inspection and replay.

```python
"""
ex2_dlq.py

Extends OrderConsumer from day-5 to route Pydantic validation failures
to an orders.dlq topic instead of silently skipping them.

Run tests: pytest ex2_dlq.py -v
"""

from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

import pytest
from pydantic import BaseModel, ConfigDict, field_validator

logger = logging.getLogger(__name__)


class OrderEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    order_id: str
    user_id: str
    amount_cents: int
    event_type: str

    @field_validator("amount_cents")
    @classmethod
    def non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("amount_cents must be >= 0")
        return v

    @field_validator("event_type")
    @classmethod
    def valid_type(cls, v: str) -> str:
        if v not in {"placed", "cancelled", "fulfilled"}:
            raise ValueError(f"invalid event_type: {v!r}")
        return v


@dataclass
class RecordMetadata:
    topic: str
    partition: int
    offset: int


DeliveryCallback = Callable[[Exception | None, RecordMetadata | None], None]


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


class InMemoryBroker:
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

    def partition_for_key(self, key: str | None, num_partitions: int) -> int:
        if key is None:
            return 0
        return hash(key.encode()) % num_partitions

    def produce(
        self,
        topic: str,
        key: str | None,
        value: bytes,
        on_delivery: DeliveryCallback | None = None,
    ) -> None:
        if topic not in self._topics:
            raise KeyError(f"Topic '{topic}' does not exist.")
        num_parts = self._num_partitions[topic]
        partition = self.partition_for_key(key, num_parts)
        offset = self._topics[topic][partition].append(value)
        if on_delivery:
            on_delivery(None, RecordMetadata(topic=topic, partition=partition, offset=offset))

    def flush(self, timeout_s: float = 5.0) -> None:
        pass

    def poll(self, group_id: str, topic: str, partition: int, max_records: int = 10) -> list[tuple[int, bytes]]:
        offset = self._offsets[group_id][topic][partition]
        return self._topics[topic][partition].read_from(offset, max_records)

    def commit(self, group_id: str, topic: str, partition: int, offset: int) -> None:
        self._offsets[group_id][topic][partition] = offset

    def record_count(self, topic: str, partition: int) -> int:
        return len(self._topics[topic][partition].log)

    def raw_records(self, topic: str, partition: int) -> list[bytes]:
        return list(self._topics[topic][partition].log)


class OrderConsumerWithDLQ:
    """
    Consumes OrderEvent messages from 'orders'.
    On Pydantic validation failure: routes to 'orders.dlq' (creates if needed).
    Manual commit after each batch.
    """

    TOPIC = "orders"
    DLQ_TOPIC = "orders.dlq"

    def __init__(self, broker: InMemoryBroker, group_id: str, partition: int) -> None:
        self._broker = broker
        self._group_id = group_id
        self._partition = partition
        # Ensure DLQ topic exists
        self._broker.create_topic(self.DLQ_TOPIC, num_partitions=1)

    def poll_and_process(
        self,
        handler: Callable[[OrderEvent], None],
        max_records: int = 10,
    ) -> tuple[int, int]:
        """
        Poll, validate, handle. Route failures to DLQ.

        Returns: (processed_ok, routed_to_dlq)
        """
        records = self._broker.poll(self._group_id, self.TOPIC, self._partition, max_records)
        if not records:
            return 0, 0

        processed_ok = 0
        routed_to_dlq = 0
        last_offset = -1

        for offset, raw in records:
            try:
                event = OrderEvent.model_validate_json(raw)
                handler(event)
                processed_ok += 1
            except Exception as exc:
                # Build DLQ record: original bytes + error metadata
                dlq_payload = json.dumps({
                    "original_topic": self.TOPIC,
                    "original_partition": self._partition,
                    "original_offset": offset,
                    "error": str(exc),
                    "raw": raw.decode(errors="replace"),
                }).encode()
                # Key = error reason (truncated) for partitioning & filtering
                error_key = type(exc).__name__ + ":" + str(exc)[:64]
                self._broker.produce(self.DLQ_TOPIC, key=error_key, value=dlq_payload)
                logger.warning(
                    "DLQ: routed offset=%d from %s → %s reason=%s",
                    offset, self.TOPIC, self.DLQ_TOPIC, exc,
                )
                routed_to_dlq += 1
            finally:
                last_offset = offset

        if last_offset >= 0:
            self._broker.commit(self._group_id, self.TOPIC, self._partition, last_offset + 1)

        return processed_ok, routed_to_dlq


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def broker() -> InMemoryBroker:
    b = InMemoryBroker()
    b.create_topic("orders", num_partitions=1)
    return b


class TestDLQRouting:
    def test_valid_message_not_dlqd(self, broker: InMemoryBroker) -> None:
        broker.produce(
            "orders",
            key="u1",
            value=b'{"order_id":"o1","user_id":"u1","amount_cents":100,"event_type":"placed"}',
        )
        consumer = OrderConsumerWithDLQ(broker, group_id="g", partition=0)
        received: list[OrderEvent] = []
        ok, dlq_count = consumer.poll_and_process(received.append)
        assert ok == 1
        assert dlq_count == 0
        assert broker.record_count("orders.dlq", partition=0) == 0

    def test_corrupt_message_routed_to_dlq(self, broker: InMemoryBroker) -> None:
        broker.produce("orders", key="u1", value=b"not-json{{{")
        consumer = OrderConsumerWithDLQ(broker, group_id="g", partition=0)
        ok, dlq_count = consumer.poll_and_process(lambda e: None)
        assert ok == 0
        assert dlq_count == 1
        assert broker.record_count("orders.dlq", partition=0) == 1

    def test_dlq_record_contains_original_payload(self, broker: InMemoryBroker) -> None:
        raw = b"not-json{{{"
        broker.produce("orders", key="u1", value=raw)
        consumer = OrderConsumerWithDLQ(broker, group_id="g", partition=0)
        consumer.poll_and_process(lambda e: None)

        dlq_records = broker.raw_records("orders.dlq", partition=0)
        assert len(dlq_records) == 1
        dlq_data = json.loads(dlq_records[0])
        assert dlq_data["original_topic"] == "orders"
        assert dlq_data["original_offset"] == 0
        assert "raw" in dlq_data
        assert "error" in dlq_data

    def test_mixed_batch_routes_only_corrupt_to_dlq(self, broker: InMemoryBroker) -> None:
        broker.produce("orders", key="u1", value=b'{"order_id":"o1","user_id":"u1","amount_cents":10,"event_type":"placed"}')
        broker.produce("orders", key="u1", value=b"bad-payload")
        broker.produce("orders", key="u1", value=b'{"order_id":"o3","user_id":"u1","amount_cents":30,"event_type":"placed"}')

        consumer = OrderConsumerWithDLQ(broker, group_id="g", partition=0)
        received: list[OrderEvent] = []
        ok, dlq_count = consumer.poll_and_process(received.append)

        assert ok == 2
        assert dlq_count == 1
        assert len(received) == 2
        assert broker.record_count("orders.dlq", partition=0) == 1

    def test_invalid_event_type_routed_to_dlq(self, broker: InMemoryBroker) -> None:
        broker.produce(
            "orders",
            key="u1",
            value=b'{"order_id":"o1","user_id":"u1","amount_cents":0,"event_type":"refunded"}',
        )
        consumer = OrderConsumerWithDLQ(broker, group_id="g", partition=0)
        ok, dlq_count = consumer.poll_and_process(lambda e: None)
        assert ok == 0
        assert dlq_count == 1
```

---

## Exercise 3 — Rebalance callback: `on_partitions_assigned` / `on_partitions_revoked`

**Problem**: Add `on_partitions_assigned(partitions)` and `on_partitions_revoked(partitions)` hooks to `AbstractBroker`. In `InMemoryBroker`, fire `on_partitions_revoked` before reassigning any partition to a new consumer. Write a test showing committed offsets survive a simulated rebalance.

**Explanation**: Rebalance callbacks let consumers commit in-flight offsets before their partitions are reassigned (`on_partitions_revoked`) and set up any local state after assignment (`on_partitions_assigned`). This is the same contract as confluent-kafka's `on_revoke` / `on_assign`. The in-memory implementation fires them synchronously; real Kafka fires them from the poll thread.

```python
"""
ex3_rebalance_callback.py

Adds on_partitions_assigned / on_partitions_revoked hooks.
Demonstrates that committed offsets survive a simulated rebalance.

Run tests: pytest ex3_rebalance_callback.py -v
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import pytest

logger = logging.getLogger(__name__)

PartitionCallback = Callable[[list[int]], None]


class AbstractBrokerWithCallbacks(ABC):
    """
    Broker interface extended with rebalance callbacks.

    on_partitions_assigned(partitions: list[int]) — called after new partitions assigned.
    on_partitions_revoked(partitions: list[int])  — called before partitions are taken away.
    """

    @abstractmethod
    def produce(self, topic: str, key: str | None, value: bytes) -> None: ...

    @abstractmethod
    def poll(self, group_id: str, topic: str, partition: int, max_records: int = 10) -> list[tuple[int, bytes]]: ...

    @abstractmethod
    def commit(self, group_id: str, topic: str, partition: int, offset: int) -> None: ...

    @abstractmethod
    def create_topic(self, topic: str, num_partitions: int = 1) -> None: ...

    @abstractmethod
    def trigger_rebalance(
        self,
        group_id: str,
        topic: str,
        old_assignment: dict[str, list[int]],
        new_assignment: dict[str, list[int]],
        on_revoked: dict[str, PartitionCallback] | None = None,
        on_assigned: dict[str, PartitionCallback] | None = None,
    ) -> None:
        """
        Simulate a rebalance:
        1. Fire on_revoked for each member losing partitions.
        2. Reassign offsets.
        3. Fire on_assigned for each member gaining partitions.
        """


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


class InMemoryBrokerWithCallbacks(AbstractBrokerWithCallbacks):
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

    def produce(self, topic: str, key: str | None, value: bytes) -> None:
        if topic not in self._topics:
            raise KeyError(f"Topic '{topic}' not found.")
        num_parts = self._num_partitions[topic]
        pid = (hash(key.encode()) if key else 0) % num_parts
        self._topics[topic][pid].append(value)

    def poll(self, group_id: str, topic: str, partition: int, max_records: int = 10) -> list[tuple[int, bytes]]:
        offset = self._offsets[group_id][topic][partition]
        return self._topics[topic][partition].read_from(offset, max_records)

    def commit(self, group_id: str, topic: str, partition: int, offset: int) -> None:
        self._offsets[group_id][topic][partition] = offset
        logger.debug("Committed group=%s %s[%d]@%d", group_id, topic, partition, offset)

    def committed_offset(self, group_id: str, topic: str, partition: int) -> int:
        return self._offsets[group_id][topic][partition]

    def trigger_rebalance(
        self,
        group_id: str,
        topic: str,
        old_assignment: dict[str, list[int]],
        new_assignment: dict[str, list[int]],
        on_revoked: dict[str, PartitionCallback] | None = None,
        on_assigned: dict[str, PartitionCallback] | None = None,
    ) -> None:
        """
        Fire callbacks and update logical assignment.

        Step 1: For each member in old_assignment, fire on_partitions_revoked
                with the partitions they are losing (old - new).
        Step 2: For each member in new_assignment, fire on_partitions_assigned
                with the partitions they are gaining (new - old_of_this_member).
        """
        on_revoked = on_revoked or {}
        on_assigned = on_assigned or {}

        # Fire revoke callbacks
        for member, old_parts in old_assignment.items():
            new_parts = set(new_assignment.get(member, []))
            losing = [p for p in old_parts if p not in new_parts]
            if losing and member in on_revoked:
                logger.info("Rebalance: revoking %s from member '%s'", losing, member)
                on_revoked[member](losing)

        # Fire assigned callbacks
        for member, new_parts in new_assignment.items():
            old_parts = set(old_assignment.get(member, []))
            gaining = [p for p in new_parts if p not in old_parts]
            if gaining and member in on_assigned:
                logger.info("Rebalance: assigning %s to member '%s'", gaining, member)
                on_assigned[member](gaining)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def broker() -> InMemoryBrokerWithCallbacks:
    b = InMemoryBrokerWithCallbacks()
    b.create_topic("t", num_partitions=3)
    return b


class TestRebalanceCallbacks:
    def test_on_revoked_fires_before_reassignment(self, broker: InMemoryBrokerWithCallbacks) -> None:
        revoked_partitions: list[int] = []

        def on_revoke(partitions: list[int]) -> None:
            revoked_partitions.extend(partitions)

        old_assignment = {"consumer-A": [0, 1, 2]}
        new_assignment = {"consumer-A": [0], "consumer-B": [1, 2]}

        broker.trigger_rebalance(
            group_id="g",
            topic="t",
            old_assignment=old_assignment,
            new_assignment=new_assignment,
            on_revoked={"consumer-A": on_revoke},
        )

        assert set(revoked_partitions) == {1, 2}, f"Expected {{1,2}}, got {revoked_partitions}"

    def test_on_assigned_fires_for_new_member(self, broker: InMemoryBrokerWithCallbacks) -> None:
        assigned_partitions: list[int] = []

        def on_assign(partitions: list[int]) -> None:
            assigned_partitions.extend(partitions)

        old_assignment = {"consumer-A": [0, 1, 2]}
        new_assignment = {"consumer-A": [0], "consumer-B": [1, 2]}

        broker.trigger_rebalance(
            group_id="g",
            topic="t",
            old_assignment=old_assignment,
            new_assignment=new_assignment,
            on_assigned={"consumer-B": on_assign},
        )

        assert set(assigned_partitions) == {1, 2}

    def test_committed_offsets_survive_rebalance(self, broker: InMemoryBrokerWithCallbacks) -> None:
        """
        consumer-A commits offsets for partitions 0 and 1.
        After rebalance, consumer-B takes partition 1.
        consumer-B must resume from consumer-A's committed offset for partition 1.
        """
        GROUP = "g"
        TOPIC = "t"

        # Produce 5 records to partition 1
        for i in range(5):
            broker._topics[TOPIC][1].append(f"msg-{i}".encode())

        # consumer-A polls and commits up to offset 3 on partition 1
        records = broker.poll(GROUP, TOPIC, partition=1, max_records=3)
        assert len(records) == 3
        broker.commit(GROUP, TOPIC, partition=1, offset=3)

        committed_before = broker.committed_offset(GROUP, TOPIC, partition=1)
        assert committed_before == 3

        # Rebalance: consumer-A loses partition 1 to consumer-B
        revoked_log: list[int] = []

        def on_revoke_a(parts: list[int]) -> None:
            # In a real consumer: commit any in-flight offsets here
            revoked_log.extend(parts)
            logger.info("consumer-A revoke callback: %s", parts)

        old_assignment = {"consumer-A": [0, 1, 2]}
        new_assignment = {"consumer-A": [0, 2], "consumer-B": [1]}

        broker.trigger_rebalance(
            GROUP, TOPIC,
            old_assignment=old_assignment,
            new_assignment=new_assignment,
            on_revoked={"consumer-A": on_revoke_a},
        )

        assert 1 in revoked_log

        # consumer-B polls partition 1 — must start from committed offset 3
        records_b = broker.poll(GROUP, TOPIC, partition=1, max_records=10)
        assert len(records_b) == 2, f"Expected 2 remaining records, got {len(records_b)}"
        assert records_b[0][0] == 3  # first offset returned to consumer-B
        assert records_b[1][0] == 4

        logger.info("PASS: consumer-B resumes from committed offset 3 after rebalance.")
```

---

## Exercise 4 — Batch producer: buffer messages, flush on batch size or timeout

**Problem**: Modify `OrderProducer.send` to buffer events and only call `broker.produce` when the buffer reaches 10 events or `flush()` is called explicitly. Write tests for both flush paths.

**Explanation**: Batching reduces the number of round trips to the broker and improves throughput. librdkafka does this natively with `linger.ms` and `batch.size`. In an in-memory broker we implement it manually. A timer-based flush requires a background thread; here we expose explicit flush and a `batch_size` limit to keep the implementation synchronous and testable.

```python
"""
ex4_batch_producer.py

BatchOrderProducer: buffers messages and flushes when batch_size is reached
or when flush() is called explicitly.

Run tests: pytest ex4_batch_producer.py -v
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import pytest
from pydantic import BaseModel, ConfigDict, field_validator

logger = logging.getLogger(__name__)


class OrderEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")
    order_id: str
    user_id: str
    amount_cents: int
    event_type: str

    @field_validator("amount_cents")
    @classmethod
    def non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("amount_cents must be >= 0")
        return v

    @field_validator("event_type")
    @classmethod
    def valid_type(cls, v: str) -> str:
        if v not in {"placed", "cancelled", "fulfilled"}:
            raise ValueError(f"invalid: {v!r}")
        return v


@dataclass
class RecordMetadata:
    topic: str
    partition: int
    offset: int


DeliveryCallback = Callable[[Exception | None, RecordMetadata | None], None]


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


class InMemoryBroker:
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

    def partition_for_key(self, key: str | None, num_partitions: int) -> int:
        if key is None:
            return 0
        return hash(key.encode()) % num_partitions

    def produce(
        self,
        topic: str,
        key: str | None,
        value: bytes,
        on_delivery: DeliveryCallback | None = None,
    ) -> None:
        if topic not in self._topics:
            raise KeyError(f"Topic '{topic}' does not exist.")
        num_parts = self._num_partitions[topic]
        partition = self.partition_for_key(key, num_parts)
        offset = self._topics[topic][partition].append(value)
        if on_delivery:
            on_delivery(None, RecordMetadata(topic=topic, partition=partition, offset=offset))

    def record_count(self, topic: str, partition: int = 0) -> int:
        return len(self._topics[topic][partition].log)

    def total_record_count(self, topic: str) -> int:
        return sum(len(p.log) for p in self._topics[topic])


class BatchOrderProducer:
    """
    Buffers OrderEvent messages and sends them to the broker in batches.

    Flush triggers:
    1. Buffer reaches batch_size — automatic flush.
    2. Caller calls flush() explicitly.
    3. (Optional, not implemented here) linger_ms timeout via background thread.

    Thread safety: uses a lock to protect the buffer.
    """

    TOPIC = "orders"

    def __init__(
        self,
        broker: InMemoryBroker,
        num_partitions: int = 3,
        batch_size: int = 10,
    ) -> None:
        self._broker = broker
        self._batch_size = batch_size
        self._buffer: list[tuple[str | None, bytes]] = []  # (key, value_bytes)
        self._lock = threading.Lock()
        self._flushed_count = 0
        self._broker.create_topic(self.TOPIC, num_partitions)

    def send(self, event: OrderEvent) -> None:
        """Buffer the event. Flushes automatically if batch_size is reached."""
        payload = event.model_dump_json().encode()
        with self._lock:
            self._buffer.append((event.user_id, payload))
            if len(self._buffer) >= self._batch_size:
                self._flush_locked()

    def flush(self) -> int:
        """Explicitly flush all buffered messages. Returns number of messages flushed."""
        with self._lock:
            return self._flush_locked()

    def _flush_locked(self) -> int:
        """Must be called with self._lock held."""
        count = len(self._buffer)
        if count == 0:
            return 0
        for key, value in self._buffer:
            self._broker.produce(self.TOPIC, key=key, value=value)
        self._buffer.clear()
        self._flushed_count += count
        logger.info("BatchProducer: flushed %d messages (total flushed: %d)", count, self._flushed_count)
        return count

    @property
    def buffer_size(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def total_flushed(self) -> int:
        return self._flushed_count


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def make_event(order_id: str, user_id: str = "u1") -> OrderEvent:
    return OrderEvent(order_id=order_id, user_id=user_id, amount_cents=100, event_type="placed")


@pytest.fixture()
def broker() -> InMemoryBroker:
    return InMemoryBroker()


class TestBatchProducer:
    def test_messages_buffered_not_sent_until_flush(self, broker: InMemoryBroker) -> None:
        producer = BatchOrderProducer(broker, num_partitions=1, batch_size=10)
        for i in range(5):
            producer.send(make_event(f"o{i}"))

        assert producer.buffer_size == 5
        assert broker.total_record_count("orders") == 0  # nothing sent yet

    def test_explicit_flush_sends_all_buffered(self, broker: InMemoryBroker) -> None:
        producer = BatchOrderProducer(broker, num_partitions=1, batch_size=10)
        for i in range(7):
            producer.send(make_event(f"o{i}"))

        flushed = producer.flush()
        assert flushed == 7
        assert producer.buffer_size == 0
        assert broker.total_record_count("orders") == 7

    def test_batch_size_triggers_auto_flush(self, broker: InMemoryBroker) -> None:
        producer = BatchOrderProducer(broker, num_partitions=1, batch_size=10)

        # Send exactly batch_size events
        for i in range(10):
            producer.send(make_event(f"o{i}"))

        # Auto-flush must have fired
        assert producer.buffer_size == 0
        assert broker.total_record_count("orders") == 10

    def test_partial_batch_not_auto_flushed(self, broker: InMemoryBroker) -> None:
        producer = BatchOrderProducer(broker, num_partitions=1, batch_size=10)
        for i in range(9):
            producer.send(make_event(f"o{i}"))

        # 9 < batch_size=10, not flushed yet
        assert producer.buffer_size == 9
        assert broker.total_record_count("orders") == 0

        # 10th message triggers flush
        producer.send(make_event("o9"))
        assert producer.buffer_size == 0
        assert broker.total_record_count("orders") == 10

    def test_flush_returns_zero_when_buffer_empty(self, broker: InMemoryBroker) -> None:
        producer = BatchOrderProducer(broker, num_partitions=1, batch_size=10)
        assert producer.flush() == 0

    def test_two_flushes_cumulative_count(self, broker: InMemoryBroker) -> None:
        producer = BatchOrderProducer(broker, num_partitions=1, batch_size=10)
        for i in range(3):
            producer.send(make_event(f"o{i}"))
        producer.flush()

        for i in range(4):
            producer.send(make_event(f"o{i+3}"))
        producer.flush()

        assert producer.total_flushed == 7
        assert broker.total_record_count("orders") == 7
```

---

## Exercise 5 — Throughput metrics: messages/sec produced, consumed, consumer lag

**Problem**: Using `InMemoryBroker`, write a script that produces 100,000 events and measures produce throughput (events/sec) and consume throughput (events/sec) separately. Profile where time is spent and suggest one optimization for each stage.

**Explanation**: Measuring throughput separately isolates bottlenecks. In an in-memory broker the dominant cost is Pydantic serialization (`model_dump_json`), not the append. For the consumer side, Pydantic validation (`model_validate_json`) dominates. The optimizations are: use `model_dump(mode='json')` + `orjson.dumps` for produce, and `model.model_validate` with pre-parsed dict for consume (avoid double parsing).

```python
"""
ex5_throughput.py

Measures produce and consume throughput against InMemoryBroker.
Profiles serialization vs. broker append vs. Pydantic validation.

Run: python ex5_throughput.py
Requires: Python 3.12+, pydantic>=2
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, field_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

N = 100_000


class OrderEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")
    order_id: str
    user_id: str
    amount_cents: int
    event_type: str

    @field_validator("amount_cents")
    @classmethod
    def non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("amount_cents must be >= 0")
        return v

    @field_validator("event_type")
    @classmethod
    def valid_type(cls, v: str) -> str:
        if v not in {"placed", "cancelled", "fulfilled"}:
            raise ValueError(f"invalid: {v!r}")
        return v


@dataclass
class _PartitionLog:
    log: list[bytes] = field(default_factory=list)

    def append(self, value: bytes) -> int:
        offset = len(self.log)
        self.log.append(value)
        return offset

    def read_from(self, offset: int, max_records: int) -> list[tuple[int, bytes]]:
        return [(offset + i, v) for i, v in enumerate(self.log[offset : offset + max_records])]


class InMemoryBroker:
    def __init__(self) -> None:
        self._topics: dict[str, list[_PartitionLog]] = {}
        self._num_partitions: dict[str, int] = {}
        self._offsets: dict[str, dict[str, dict[int, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )

    def create_topic(self, topic: str, num_partitions: int = 1) -> None:
        if topic not in self._topics:
            self._topics[topic] = [_PartitionLog() for _ in range(num_partitions)]
            self._num_partitions[topic] = num_partitions

    def partition_for_key(self, key: str | None, num_partitions: int) -> int:
        if key is None:
            return 0
        return hash(key.encode()) % num_partitions

    def produce(self, topic: str, key: str | None, value: bytes) -> int:
        num_parts = self._num_partitions[topic]
        pid = self.partition_for_key(key, num_parts)
        return self._topics[topic][pid].append(value)

    def poll(self, group_id: str, topic: str, partition: int, max_records: int = 100) -> list[tuple[int, bytes]]:
        offset = self._offsets[group_id][topic][partition]
        return self._topics[topic][partition].read_from(offset, max_records)

    def commit(self, group_id: str, topic: str, partition: int, offset: int) -> None:
        self._offsets[group_id][topic][partition] = offset

    def total_records(self, topic: str) -> int:
        return sum(len(p.log) for p in self._topics[topic])


def benchmark_produce(broker: InMemoryBroker, n: int) -> dict[str, float]:
    """
    Benchmark: produce N OrderEvent messages to single partition.
    Measures time split across: serialization, broker append.
    """
    TOPIC = "bench"
    broker.create_topic(TOPIC, num_partitions=1)

    events = [
        OrderEvent(
            order_id=f"o{i}",
            user_id=f"u{i % 1000}",
            amount_cents=i,
            event_type="placed",
        )
        for i in range(n)
    ]

    # Measure serialization separately
    t0 = time.perf_counter()
    payloads = [e.model_dump_json().encode() for e in events]
    t_serialize = time.perf_counter() - t0

    # Measure broker append separately
    t0 = time.perf_counter()
    for i, payload in enumerate(payloads):
        broker.produce(TOPIC, key=f"u{i % 1000}", value=payload)
    t_append = time.perf_counter() - t0

    total = t_serialize + t_append
    return {
        "n": n,
        "total_s": total,
        "msgs_per_sec": n / total,
        "serialize_s": t_serialize,
        "append_s": t_append,
        "serialize_pct": 100 * t_serialize / total,
        "append_pct": 100 * t_append / total,
    }


def benchmark_consume(broker: InMemoryBroker, n: int) -> dict[str, float]:
    """
    Benchmark: consume all N messages from single partition.
    Measures time split across: poll (disk/buffer read), Pydantic validation.
    """
    TOPIC = "bench"
    GROUP = "bench-group"
    PARTITION = 0
    BATCH = 200

    # Separate poll time from validation time
    all_raw: list[bytes] = []

    t0 = time.perf_counter()
    while True:
        records = broker.poll(GROUP, TOPIC, PARTITION, max_records=BATCH)
        if not records:
            break
        all_raw.extend(raw for _, raw in records)
        broker.commit(GROUP, TOPIC, PARTITION, records[-1][0] + 1)
    t_poll = time.perf_counter() - t0

    # Reset and measure validation only
    t0 = time.perf_counter()
    for raw in all_raw:
        OrderEvent.model_validate_json(raw)
    t_validate = time.perf_counter() - t0

    total = t_poll + t_validate
    return {
        "n": len(all_raw),
        "total_s": total,
        "msgs_per_sec": len(all_raw) / total,
        "poll_s": t_poll,
        "validate_s": t_validate,
        "poll_pct": 100 * t_poll / total,
        "validate_pct": 100 * t_validate / total,
    }


def main() -> None:
    broker = InMemoryBroker()

    logger.info("=== Produce benchmark (%d events) ===", N)
    produce_stats = benchmark_produce(broker, N)
    logger.info("  Total:       %.2f s", produce_stats["total_s"])
    logger.info("  Throughput:  %.0f msg/s", produce_stats["msgs_per_sec"])
    logger.info("  Serialization: %.2f s (%.1f%%)", produce_stats["serialize_s"], produce_stats["serialize_pct"])
    logger.info("  Broker append: %.2f s (%.1f%%)", produce_stats["append_s"], produce_stats["append_pct"])

    logger.info("=== Consume benchmark (%d events) ===", N)
    consume_stats = benchmark_consume(broker, N)
    logger.info("  Total:       %.2f s", consume_stats["total_s"])
    logger.info("  Throughput:  %.0f msg/s", consume_stats["msgs_per_sec"])
    logger.info("  Poll time:       %.2f s (%.1f%%)", consume_stats["poll_s"], consume_stats["poll_pct"])
    logger.info("  Pydantic validate: %.2f s (%.1f%%)", consume_stats["validate_s"], consume_stats["validate_pct"])

    logger.info("=== Optimization recommendations ===")
    logger.info(
        "  Serialization: replace model_dump_json() with orjson.dumps(model.model_dump()) — "
        "orjson is ~3-5x faster than stdlib json for Python objects."
    )
    logger.info(
        "  Broker append: list.append is O(1) amortised; no bottleneck at this scale. "
        "For multi-partition: use per-partition locks instead of a global broker lock."
    )
    logger.info(
        "  Pydantic validation: pre-parse JSON with orjson.loads(), then call "
        "OrderEvent.model_validate(dict) instead of model_validate_json(bytes) — "
        "avoids double parsing when you need the raw dict for logging anyway."
    )


if __name__ == "__main__":
    main()
```

---

## Exercise 6 — `AsyncKafkaBroker` using aiokafka

**Problem**: Write an `AsyncKafkaBroker` implementing `AbstractBroker` using `aiokafka.AIOKafkaProducer` and `aiokafka.AIOKafkaConsumer`. The class should require `uv add aiokafka`. The existing test suite should pass when `InMemoryBroker` is swapped for `AsyncKafkaBroker` pointed at a local cluster.

**Explanation**: `aiokafka` is fully async — its producer and consumer use `async/await`. The adaptor wraps the async calls and matches the `AbstractBroker` synchronous interface by running them in an event loop (or requiring callers to use the async variants). In practice, async services use the async methods directly; the sync shim exists for test compatibility.

```python
"""
ex6_aiokafka_broker.py

AsyncKafkaBroker: AbstractBroker implementation using aiokafka.

Requires: uv add aiokafka pydantic
Real usage: must point at a running Kafka cluster.
This file shows the full implementation structure; tests run against InMemoryBroker.

Key design decisions:
  - AsyncKafkaBroker exposes async_ prefixed methods for native async use.
  - The sync AbstractBroker methods use asyncio.get_event_loop().run_until_complete()
    as a compatibility shim — only safe outside an existing event loop.
  - In a real FastAPI service, inject AsyncKafkaBroker and call async_ methods directly.
"""

from __future__ import annotations

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from typing import Callable

logger = logging.getLogger(__name__)

DeliveryCallback = Callable[[Exception | None, "RecordMetadata | None"], None]


class RecordMetadata:
    def __init__(self, topic: str, partition: int, offset: int) -> None:
        self.topic = topic
        self.partition = partition
        self.offset = offset


class AbstractBroker(ABC):
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


class AsyncKafkaBroker(AbstractBroker):
    """
    AbstractBroker implementation backed by aiokafka.

    Install: uv add aiokafka

    Lifecycle:
        broker = AsyncKafkaBroker()
        await broker.start()
        ... use async_ methods ...
        await broker.stop()

    The sync AbstractBroker shim methods (produce, poll, commit) use
    run_until_complete and are only valid outside a running event loop.
    For async services, use async_produce / async_poll / async_commit directly.
    """

    def __init__(
        self,
        bootstrap_servers: str | None = None,
        group_id: str = "default-group",
    ) -> None:
        try:
            import aiokafka  # noqa: F401
        except ImportError as exc:
            raise ImportError("aiokafka not installed. Run: uv add aiokafka") from exc

        self._bootstrap_servers = bootstrap_servers or os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
        )
        self._group_id = group_id
        self._producer: "aiokafka.AIOKafkaProducer | None" = None  # type: ignore[name-defined]
        self._consumer: "aiokafka.AIOKafkaConsumer | None" = None  # type: ignore[name-defined]
        self._subscribed_topics: set[str] = set()

    async def start(self) -> None:
        """Start producer and consumer. Must be called before any produce/poll."""
        import aiokafka

        self._producer = aiokafka.AIOKafkaProducer(
            bootstrap_servers=self._bootstrap_servers,
            acks="all",
            enable_idempotence=True,
        )
        await self._producer.start()
        logger.info("AsyncKafkaBroker: producer started (%s)", self._bootstrap_servers)

    async def stop(self) -> None:
        """Flush and stop producer and consumer."""
        if self._producer:
            await self._producer.stop()
        if self._consumer:
            await self._consumer.stop()
        logger.info("AsyncKafkaBroker: stopped.")

    # ------------------------------------------------------------------
    # Async native API (preferred for async services)
    # ------------------------------------------------------------------

    async def async_produce(
        self,
        topic: str,
        key: str | None,
        value: bytes,
    ) -> RecordMetadata:
        """Produce a single record and await delivery confirmation."""
        if self._producer is None:
            raise RuntimeError("Call await broker.start() before producing.")
        record_metadata = await self._producer.send_and_wait(
            topic=topic,
            key=key.encode() if key else None,
            value=value,
        )
        return RecordMetadata(
            topic=record_metadata.topic,
            partition=record_metadata.partition,
            offset=record_metadata.offset,
        )

    async def async_subscribe(self, topics: list[str]) -> None:
        """Subscribe consumer to topics. Creates consumer on first call."""
        import aiokafka

        if self._consumer is None:
            self._consumer = aiokafka.AIOKafkaConsumer(
                *topics,
                bootstrap_servers=self._bootstrap_servers,
                group_id=self._group_id,
                auto_offset_reset="earliest",
                enable_auto_commit=False,
            )
            await self._consumer.start()
        else:
            self._consumer.subscribe(topics)
        self._subscribed_topics.update(topics)
        logger.info("AsyncKafkaBroker: subscribed to %s", topics)

    async def async_poll(self, timeout_ms: int = 500) -> list[tuple[str, int, int, bytes]]:
        """
        Poll for records. Returns list of (topic, partition, offset, value).
        Does NOT commit — caller must call async_commit().
        """
        if self._consumer is None:
            return []
        records = await self._consumer.getmany(timeout_ms=timeout_ms, max_records=100)
        result: list[tuple[str, int, int, bytes]] = []
        for tp, msgs in records.items():
            for msg in msgs:
                result.append((msg.topic, msg.partition, msg.offset, msg.value))
        return result

    async def async_commit(self) -> None:
        """Commit all fetched offsets."""
        if self._consumer:
            await self._consumer.commit()

    # ------------------------------------------------------------------
    # AbstractBroker sync shims (for test-compat outside event loop)
    # ------------------------------------------------------------------

    def _run(self, coro: "Any") -> "Any":  # noqa: ANN001
        """Run a coroutine synchronously. Not safe inside a running event loop."""
        return asyncio.get_event_loop().run_until_complete(coro)

    def create_topic(self, topic: str, num_partitions: int = 1) -> None:
        # aiokafka relies on auto-create or AdminClient; here we log only.
        logger.info("AsyncKafkaBroker.create_topic: '%s' (ensure broker auto-creates or pre-create via AdminClient)", topic)

    def produce(
        self,
        topic: str,
        key: str | None,
        value: bytes,
        on_delivery: DeliveryCallback | None = None,
    ) -> None:
        try:
            meta = self._run(self.async_produce(topic, key, value))
            if on_delivery:
                on_delivery(None, meta)
        except Exception as exc:
            if on_delivery:
                on_delivery(exc, None)
            else:
                raise

    def flush(self, timeout_s: float = 5.0) -> None:
        if self._producer:
            self._run(self._producer.flush())

    def poll(
        self,
        group_id: str,
        topic: str,
        partition: int,
        max_records: int = 10,
    ) -> list[tuple[int, bytes]]:
        records = self._run(self.async_poll(timeout_ms=500))
        return [
            (offset, value)
            for (t, p, offset, value) in records
            if t == topic and p == partition
        ][:max_records]

    def commit(self, group_id: str, topic: str, partition: int, offset: int) -> None:
        self._run(self.async_commit())


# ---------------------------------------------------------------------------
# Usage example and integration test notes
# ---------------------------------------------------------------------------

USAGE_EXAMPLE = '''
# Integration test against a real Kafka cluster:

import asyncio
from ex6_aiokafka_broker import AsyncKafkaBroker

async def integration_test():
    broker = AsyncKafkaBroker(bootstrap_servers="localhost:9092", group_id="test-group")
    await broker.start()

    # Produce 5 messages
    for i in range(5):
        meta = await broker.async_produce("test-topic", key=f"key-{i}", value=f"value-{i}".encode())
        print(f"Produced: partition={meta.partition} offset={meta.offset}")

    # Subscribe and consume
    await broker.async_subscribe(["test-topic"])
    import asyncio; await asyncio.sleep(1)  # let partition assignment complete

    records = await broker.async_poll(timeout_ms=2000)
    for topic, partition, offset, value in records:
        print(f"Consumed: {topic}[{partition}]@{offset} = {value}")
    await broker.async_commit()

    await broker.stop()

asyncio.run(integration_test())
'''

if __name__ == "__main__":
    print("AsyncKafkaBroker is defined. See USAGE_EXAMPLE for integration test instructions.")
    print("For unit tests, use InMemoryBroker (from day-5 worked example).")
    print(USAGE_EXAMPLE)
```

---

## Summary

| Exercise | Core concept | Key insight |
|---|---|---|
| 1 | `asyncio.to_thread` | Sync broker usable in async services without blocking event loop |
| 2 | DLQ routing | Commit the offset even for failed records; route bad bytes to `{topic}.dlq` |
| 3 | Rebalance callbacks | `on_partitions_revoked` fires before reassignment; commit in-flight offsets there |
| 4 | Batch producer | Buffer until `batch_size` or explicit `flush()`; separate locking from I/O |
| 5 | Throughput profiling | Serialization and Pydantic validation dominate; orjson cuts both significantly |
| 6 | aiokafka wrapper | Implement `AbstractBroker` with async native methods; add sync shims for test compat |
