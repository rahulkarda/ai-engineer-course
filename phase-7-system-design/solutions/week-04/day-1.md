# Solutions — Week 04 · Day 1
# Event-driven architecture + pub/sub at scale

---

## Exercise 1

**Implement an in-memory pub/sub broker with topic partitioning. Support multiple consumer groups. Show that two consumer groups on the same topic each receive all messages (fan-out). Show that consumers in the same group share the load.**

```python
"""
in_memory_pubsub.py

In-memory pub/sub broker with topic partitioning and consumer groups.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)


@dataclass
class Message:
    topic: str
    key: str | None
    value: bytes
    partition: int = 0


MessageHandler = Callable[[Message], Awaitable[None]]


@dataclass
class ConsumerGroupMember:
    group_id: str
    member_id: str
    handler: MessageHandler
    assigned_partitions: set[int] = field(default_factory=set)


class Partition:
    """Single ordered queue of messages for one partition."""

    def __init__(self, partition_id: int) -> None:
        self.partition_id = partition_id
        self._queue: asyncio.Queue[Message] = asyncio.Queue()

    async def put(self, message: Message) -> None:
        await self._queue.put(message)

    async def get(self) -> Message:
        return await self._queue.get()

    def qsize(self) -> int:
        return self._queue.qsize()


class Topic:
    """Topic with N partitions. Each partition is a separate queue."""

    def __init__(self, name: str, num_partitions: int = 4) -> None:
        self.name = name
        self.num_partitions = num_partitions
        self.partitions: list[Partition] = [
            Partition(i) for i in range(num_partitions)
        ]
        # group_id -> list[ConsumerGroupMember]
        self._groups: dict[str, list[ConsumerGroupMember]] = defaultdict(list)

    def _partition_for_key(self, key: str | None) -> int:
        if key is None:
            return 0
        digest = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return digest % self.num_partitions

    async def publish(self, key: str | None, value: bytes) -> None:
        partition_id = self._partition_for_key(key)
        message = Message(
            topic=self.name,
            key=key,
            value=value,
            partition=partition_id,
        )
        logger.debug(
            "Published to %s partition=%d key=%s", self.name, partition_id, key
        )
        await self.partitions[partition_id].put(message)

    def register_consumer(
        self,
        group_id: str,
        member_id: str,
        handler: MessageHandler,
    ) -> ConsumerGroupMember:
        member = ConsumerGroupMember(
            group_id=group_id,
            member_id=member_id,
            handler=handler,
        )
        self._groups[group_id].append(member)
        self._rebalance(group_id)
        logger.info(
            "Registered consumer %s in group %s on topic %s",
            member_id,
            group_id,
            self.name,
        )
        return member

    def _rebalance(self, group_id: str) -> None:
        """Assign partitions round-robin across members of the group."""
        members = self._groups[group_id]
        for m in members:
            m.assigned_partitions.clear()
        for p_id in range(self.num_partitions):
            members[p_id % len(members)].assigned_partitions.add(p_id)
        for m in members:
            logger.debug(
                "Group %s member %s assigned partitions %s",
                group_id,
                m.member_id,
                m.assigned_partitions,
            )

    async def _consume_partition(
        self, partition: Partition, member: ConsumerGroupMember
    ) -> None:
        """Drain a single partition for one group member."""
        while True:
            message = await partition.get()
            logger.debug(
                "Group %s member %s consuming partition %d",
                member.group_id,
                member.member_id,
                message.partition,
            )
            await member.handler(message)

    async def start_consuming(self) -> list[asyncio.Task[None]]:
        """
        For every group, for every member, launch consume tasks for
        that member's assigned partitions. Fan-out across groups is
        achieved because each group gets its own copy of every message.

        NOTE: Fan-out requires publishing to per-group queues. Here we
        implement it by duplicating messages into per-group partition queues
        at publish time via a Broker wrapper (see Broker.publish below).
        """
        tasks: list[asyncio.Task[None]] = []
        for group_id, members in self._groups.items():
            for member in members:
                for p_id in member.assigned_partitions:
                    t = asyncio.create_task(
                        self._consume_partition(
                            self.partitions[p_id], member
                        ),
                        name=f"{self.name}/{group_id}/{member.member_id}/p{p_id}",
                    )
                    tasks.append(t)
        return tasks


class Broker:
    """
    Top-level broker. Manages topics and routes published messages to
    all consumer groups subscribed to that topic (fan-out).

    Fan-out implementation: per-group per-partition queues. Each group
    has its own set of Partition objects so messages are independent.
    """

    def __init__(self) -> None:
        # topic -> group_id -> list[Partition]
        self._topic_group_partitions: dict[
            str, dict[str, list[Partition]]
        ] = defaultdict(lambda: defaultdict(list))
        # topic -> num_partitions
        self._topic_num_partitions: dict[str, int] = {}
        # all tasks
        self._tasks: list[asyncio.Task[None]] = []

    def create_topic(self, topic: str, num_partitions: int = 4) -> None:
        self._topic_num_partitions[topic] = num_partitions
        logger.info("Created topic %s with %d partitions", topic, num_partitions)

    def _ensure_group_partitions(
        self, topic: str, group_id: str
    ) -> list[Partition]:
        if group_id not in self._topic_group_partitions[topic]:
            n = self._topic_num_partitions.get(topic, 4)
            self._topic_group_partitions[topic][group_id] = [
                Partition(i) for i in range(n)
            ]
        return self._topic_group_partitions[topic][group_id]

    def subscribe(
        self,
        topic: str,
        group_id: str,
        member_id: str,
        handler: MessageHandler,
    ) -> ConsumerGroupMember:
        partitions = self._ensure_group_partitions(topic, group_id)
        n = self._topic_num_partitions.get(topic, 4)

        # build a lightweight member for rebalancing
        existing = self._get_group_members(topic, group_id)
        member = ConsumerGroupMember(
            group_id=group_id, member_id=member_id, handler=handler
        )
        existing.append(member)
        self._rebalance_group(topic, group_id, partitions, existing)
        return member

    def _get_group_members(
        self, topic: str, group_id: str
    ) -> list[ConsumerGroupMember]:
        # stored separately so we can rebalance
        key = (topic, group_id)
        if not hasattr(self, "_members"):
            self._members: dict[tuple[str, str], list[ConsumerGroupMember]] = {}
        if key not in self._members:
            self._members[key] = []
        return self._members[key]

    def _rebalance_group(
        self,
        topic: str,
        group_id: str,
        partitions: list[Partition],
        members: list[ConsumerGroupMember],
    ) -> None:
        for m in members:
            m.assigned_partitions.clear()
        for p_id in range(len(partitions)):
            members[p_id % len(members)].assigned_partitions.add(p_id)

    async def publish(self, topic: str, key: str | None, value: bytes) -> None:
        if topic not in self._topic_group_partitions:
            logger.warning("No subscribers for topic %s", topic)
            return
        p_id = self._partition_for_key(
            key, self._topic_num_partitions.get(topic, 4)
        )
        msg = Message(topic=topic, key=key, value=value, partition=p_id)
        for group_id, group_partitions in self._topic_group_partitions[
            topic
        ].items():
            await group_partitions[p_id].put(msg)
            logger.debug(
                "Routed to group %s partition %d on topic %s",
                group_id,
                p_id,
                topic,
            )

    @staticmethod
    def _partition_for_key(key: str | None, num_partitions: int) -> int:
        if key is None:
            return 0
        digest = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return digest % num_partitions

    async def start(self) -> None:
        """Launch all consumer tasks."""
        for (topic, group_id), members in self._members.items():
            partitions = self._topic_group_partitions[topic][group_id]
            for member in members:
                for p_id in member.assigned_partitions:
                    t = asyncio.create_task(
                        self._consume_loop(partitions[p_id], member),
                        name=f"{topic}/{group_id}/{member.member_id}/p{p_id}",
                    )
                    self._tasks.append(t)

    async def _consume_loop(
        self, partition: Partition, member: ConsumerGroupMember
    ) -> None:
        while True:
            msg = await partition.get()
            await member.handler(msg)

    def cancel(self) -> None:
        for t in self._tasks:
            t.cancel()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

import pytest


@pytest.mark.asyncio
async def test_fan_out_two_groups_each_receive_all_messages() -> None:
    """
    Two consumer groups on the same topic must each receive every message.
    """
    broker = Broker()
    broker.create_topic("orders", num_partitions=2)

    received_a: list[bytes] = []
    received_b: list[bytes] = []

    async def handler_a(msg: Message) -> None:
        received_a.append(msg.value)

    async def handler_b(msg: Message) -> None:
        received_b.append(msg.value)

    broker.subscribe("orders", "group-a", "a1", handler_a)
    broker.subscribe("orders", "group-b", "b1", handler_b)

    await broker.start()

    messages = [b"order-1", b"order-2", b"order-3", b"order-4"]
    for i, m in enumerate(messages):
        await broker.publish("orders", str(i), m)

    await asyncio.sleep(0.05)
    broker.cancel()

    assert sorted(received_a) == sorted(messages)
    assert sorted(received_b) == sorted(messages)


@pytest.mark.asyncio
async def test_consumers_in_same_group_share_load() -> None:
    """
    Two consumers in the same group must together process each message
    exactly once (no duplication within the group).
    """
    broker = Broker()
    broker.create_topic("events", num_partitions=4)

    received: list[tuple[str, bytes]] = []

    async def make_handler(member_id: str) -> MessageHandler:
        async def handler(msg: Message) -> None:
            received.append((member_id, msg.value))

        return handler

    broker.subscribe("events", "workers", "w1", await make_handler("w1"))
    broker.subscribe("events", "workers", "w2", await make_handler("w2"))

    await broker.start()

    messages = [f"evt-{i}".encode() for i in range(8)]
    for i, m in enumerate(messages):
        await broker.publish("events", str(i), m)

    await asyncio.sleep(0.05)
    broker.cancel()

    received_values = [v for _, v in received]
    # Each message received exactly once across the group
    assert sorted(received_values) == sorted(messages)
    # Both workers received at least one message (load shared)
    workers_used = {wid for wid, _ in received}
    assert len(workers_used) == 2
```

---

## Exercise 2

**Add backpressure: if a consumer's buffer exceeds a threshold, slow the producer via an async semaphore. Write a test demonstrating the pressure.**

```python
"""
backpressure.py

Backpressure-aware broker. Producer acquires a semaphore slot before
publishing. Slot is released only after the consumer processes the message.
This bounds the number of in-flight messages per consumer to `max_inflight`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class BPMessage:
    key: str | None
    value: bytes
    _done_callback: asyncio.Event = field(
        default_factory=asyncio.Event, repr=False
    )


MessageHandler = "Callable[[BPMessage], Awaitable[None]]"


class BackpressureChannel:
    """
    Single-consumer channel with backpressure.

    The producer calls `publish()` which blocks when `max_inflight` messages
    are already queued but not yet acknowledged by the consumer.
    The consumer calls `consume()` which yields messages; after processing
    the consumer calls `ack()` to release the semaphore slot.
    """

    def __init__(self, max_inflight: int = 10) -> None:
        self._semaphore = asyncio.Semaphore(max_inflight)
        self._queue: asyncio.Queue[BPMessage] = asyncio.Queue()
        self.max_inflight = max_inflight

    async def publish(self, key: str | None, value: bytes) -> None:
        """
        Acquire a slot before enqueuing. If all slots are taken the
        coroutine suspends here — this is the backpressure point.
        """
        await self._semaphore.acquire()
        msg = BPMessage(key=key, value=value)
        await self._queue.put(msg)
        logger.debug(
            "Published key=%s qsize=%d/%d",
            key,
            self._queue.qsize(),
            self.max_inflight,
        )

    async def consume(self) -> BPMessage:
        """Return next message. Caller must call ack() when done."""
        return await self._queue.get()

    def ack(self, msg: BPMessage) -> None:
        """Release the semaphore slot for this message."""
        self._semaphore.release()

    def pending(self) -> int:
        return self._queue.qsize()

    def available_slots(self) -> int:
        return self._semaphore._value  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

import pytest


@pytest.mark.asyncio
async def test_backpressure_slows_producer() -> None:
    """
    With max_inflight=3 the producer must not be able to publish a 4th
    message until the consumer processes at least one.
    Measured by timing: a slow consumer should cause the producer to wait.
    """
    channel = BackpressureChannel(max_inflight=3)
    publish_times: list[float] = []
    consume_count = 0

    async def slow_consumer() -> None:
        nonlocal consume_count
        while consume_count < 5:
            msg = await channel.consume()
            await asyncio.sleep(0.05)  # slow processing
            channel.ack(msg)
            consume_count += 1

    async def producer() -> None:
        for i in range(5):
            t0 = time.monotonic()
            await channel.publish(str(i), f"msg-{i}".encode())
            publish_times.append(time.monotonic() - t0)

    consumer_task = asyncio.create_task(slow_consumer())
    await asyncio.sleep(0)  # yield so consumer starts

    await asyncio.gather(producer(), consumer_task)

    # First 3 publishes should be near-instant (slots available).
    # 4th and 5th publish must wait for the consumer to ack.
    assert publish_times[0] < 0.01, "First publish should be fast"
    assert publish_times[1] < 0.01, "Second publish should be fast"
    assert publish_times[2] < 0.01, "Third publish should be fast"
    # 4th publish must have waited
    assert publish_times[3] > 0.03, (
        f"4th publish should be delayed by backpressure, got {publish_times[3]:.3f}s"
    )


@pytest.mark.asyncio
async def test_backpressure_buffer_never_exceeds_threshold() -> None:
    """Buffer (pending messages) must never exceed max_inflight."""
    max_inflight = 5
    channel = BackpressureChannel(max_inflight=max_inflight)
    max_seen: list[int] = [0]

    async def slow_consumer() -> None:
        for _ in range(20):
            msg = await channel.consume()
            await asyncio.sleep(0.01)
            max_seen[0] = max(max_seen[0], channel.pending())
            channel.ack(msg)

    async def fast_producer() -> None:
        for i in range(20):
            await channel.publish(str(i), b"x")

    consumer_task = asyncio.create_task(slow_consumer())
    await asyncio.gather(fast_producer(), consumer_task)

    assert max_seen[0] <= max_inflight
```

---

## Exercise 3

**Explain when you'd choose Kafka over Redis Streams over an in-process event bus. Give a decision matrix.**

### Answer

#### Core capabilities at a glance

| Capability | In-process bus | Redis Streams | Apache Kafka |
|---|---|---|---|
| Persistence | None (in-memory) | Yes (AOF / RDB) | Yes (durable log) |
| Durability guarantee | None | Configurable | Configurable (replication) |
| Throughput ceiling | CPU-bound (millions msg/s in single process) | ~1 M msg/s (single shard) | 10s of millions msg/s (partitioned) |
| Consumer groups (competing consumers) | Manual (as above) | Native | Native |
| Fan-out (multiple groups) | Manual | Native | Native |
| Message retention | None | TTL or maxlen | Configurable (days/forever) |
| Replay from offset | No | Yes (ID-based) | Yes |
| Ordering guarantee | Per-queue | Per-stream | Per-partition |
| Multi-region / cross-DC | No | Redis Cluster (limited) | MirrorMaker 2 |
| Latency | sub-millisecond | 1-5 ms | 5-20 ms (batch) |
| Operational complexity | None | Low (managed: Elasticache) | Medium-High (ZK/KRaft, brokers) |
| Schema registry | No | No | Confluent / Apicurio |
| Exactly-once semantics | Manual (see day-4) | At-least-once | Yes (transactions) |
| Cost | Free | Low | Medium-High |

#### Decision rules

**Choose in-process event bus when:**
- Single process / single host (monolith, test harness, lambda function).
- No persistence needed; data is ephemeral.
- Ultra-low latency is required (game loop, real-time inference pipeline inside a container).
- Operational simplicity trumps all.

**Choose Redis Streams when:**
- Cross-process messaging within one data center.
- Already using Redis; adding another infra component is undesirable.
- Moderate throughput (< 500 K msg/s).
- Need persistence with simple replay but full Kafka operability is overkill.
- Consumer groups needed with automatic failover.
- Message sizes are small (< 1 MB).

**Choose Kafka when:**
- High throughput (> 500 K msg/s) or large messages.
- Long-term event retention for audit, reprocessing, or ML feature pipelines.
- Exactly-once end-to-end semantics required (financial transactions, billing).
- Multiple independent teams consume the same event stream.
- Event sourcing or CQRS at scale (event log is the system of record).
- Multi-region replication needed.
- Schema evolution with backward/forward compatibility required.

#### Quick decision flowchart (text)

```
Need persistence?
  NO  --> In-process bus
  YES --> Need > 500K msg/s OR long retention OR exactly-once?
            YES --> Kafka
            NO  --> Already running Redis?
                      YES --> Redis Streams
                      NO  --> Kafka (if ops cost acceptable) else Redis Streams
```

### Discussion notes

- **Partitioning key matters**: Kafka and Redis both route by key hash to a partition. Wrong key choice (e.g., all messages with `key=null`) causes hot partitions and defeats horizontal scaling.
- **In-process fan-out vs. broker fan-out**: In-process buses duplicate in memory (O(groups * messages)). Kafka stores once and consumers pull independently — much more memory-efficient for many groups.
- **Backpressure is implicit in Kafka**: Consumers control their own offset; a slow consumer does not block the broker or the producer. In-process and Redis Streams require explicit flow control (as shown in Exercise 2).
- **Exactly-once in Kafka** requires `enable.idempotence=true` + transactional producer + consumer `isolation.level=read_committed`. This is non-trivial to configure and adds latency (~2x). Only use when the cost of a duplicate is higher than the cost of latency.
- **Redis Streams `XACK`** acknowledges per-consumer-group, not globally. Messages stay in the stream until trimmed by `MAXLEN` or explicit `XDEL`. This is different from Kafka, where the "ack" is just advancing the offset.

---
