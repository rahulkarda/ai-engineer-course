# Phase 7 · Week 04 · Day 1 — Event-Driven Architecture and Pub/Sub at Scale

> Objective: Understand event-driven architecture principles, pub/sub delivery semantics, and build an in-memory broker that demonstrates fan-out, consumer groups, backpressure, and at-least-once delivery.  ·  Time: ~1 hr  ·  Prereqs: [[week-03/day-4]]

---

## Concept

Event-driven architecture (EDA) decouples producers from consumers by replacing direct calls with **events** — immutable facts that something happened. Producers emit events without knowing who will consume them. Consumers subscribe and react independently.

### Why EDA over Request/Response

| Concern | Request/Response | Event-Driven |
|---|---|---|
| Coupling | Tight — caller knows callee | Loose — via topic name only |
| Availability | Both parties must be up | Producer continues if consumer is down |
| Scale | N calls from N producers | One event, N consumers (fan-out) |
| Auditability | Hard — ephemeral | Natural — events are facts |
| Backpressure | Caller blocks or times out | Consumer controls its own rate |

### Core Vocabulary

**Event** — an immutable, timestamped record of a state change. Contains what happened, when, and the relevant payload.

**Topic** — a named channel. Producers write to topics; consumers read from topics.

**Partition** — a topic is split into ordered, append-only logs. Each partition is consumed by at most one consumer within a consumer group at a time. Partitioning enables horizontal scaling.

**Consumer group** — a set of consumers that jointly consume a topic. Each message goes to exactly one consumer in the group (competing consumers pattern). Different groups each get all messages (fan-out).

**Offset** — each message in a partition has a monotonically increasing integer position. Consumers commit offsets to track progress.

### Delivery Semantics

**At-most-once** — commit offset before processing. Message may be lost if consumer crashes mid-process.

**At-least-once** — commit offset after processing. Message may be re-delivered if crash happens between process and commit. Consumer must be idempotent.

**Exactly-once** — requires coordination between broker and consumer (transactional commits). Covered in depth on Day 4.

### Fan-out vs Fan-in

```
Fan-out (broadcast):
  Producer --> Topic --> Consumer Group A (all members share load)
                    --> Consumer Group B (all members share load)
                    --> Consumer Group C (all members share load)

Fan-in (aggregation):
  Producer A -->
  Producer B --> Topic --> Single Consumer
  Producer C -->
```

### Backpressure

When consumers process slower than producers emit, messages queue up. Backpressure mechanisms:

1. **Blocking queue** — producer blocks when buffer is full (pull model)
2. **Rate limiting** — producer throttles based on consumer lag signal
3. **Drop** — oldest or newest messages discarded (lossy, only for metrics/telemetry)
4. **Dead-letter queue** — unprocessable messages routed aside

---

## How It Works Internally

### Kafka-Style Partitioned Log (Conceptual)

```
Topic: "orders"
  Partition 0: [msg_0][msg_1][msg_4][msg_7]...  <- offset 0,1,4,7
  Partition 1: [msg_2][msg_3][msg_5][msg_8]...
  Partition 2: [msg_6][msg_9]...

Consumer Group "billing":
  Consumer 0 -> Partition 0  (offset committed: 7)
  Consumer 1 -> Partition 1  (offset committed: 8)
  Consumer 2 -> Partition 2  (offset committed: 9)

Consumer Group "analytics":
  Consumer 0 -> Partition 0, 1, 2  (reads independently; own offsets)
```

**Why partitions?**

- Ordering guaranteed within partition, not across partitions
- Partition key (e.g., `order_id % num_partitions`) routes related events to same partition, preserving per-entity order
- Replication factor: each partition has a leader + N replicas on different brokers for fault tolerance

### In-Memory Broker Architecture

```
+------------------+       publish(topic, event)       +-------------------+
|    Producer A    | ---------------------------------> |                   |
+------------------+                                   |    In-Memory      |
+------------------+                                   |    Broker         |
|    Producer B    | ---------------------------------> |                   |
+------------------+                                   |  topics: dict     |
                                                        |    topic ->       |
+------------------+  subscribe(topic, group)           |    partitions[]   |
|   Consumer Grp A | <--------------------------------- |                   |
|  (load balanced) |                                    |  groups: dict     |
+------------------+                                    |    group ->       |
+------------------+  subscribe(topic, group)           |    offsets{}      |
|   Consumer Grp B | <--------------------------------- |                   |
|  (independent)   |                                    +-------------------+
+------------------+
```

### Offset Commit and At-Least-Once

The broker stores `(group, partition) -> committed_offset`. After crash recovery, consumer resumes from last committed offset. If the consumer processed message at offset 5 but crashed before committing, it will re-read offset 5 on restart — at-least-once delivery.

Idempotency requirement: processing the same event twice must produce the same result. Techniques: deduplication keys, upsert semantics, conditional writes.

---

## Worked Example

```python
"""
eda_broker.py — In-memory event-driven broker demonstrating pub/sub at scale.

Covers:
- Topic partitioning by key
- Consumer groups with offset tracking
- Fan-out (multiple groups, each gets all messages)
- At-least-once delivery simulation
- Backpressure via bounded queues
- Async producer and consumer loops

Run:
    python eda_broker.py

No external dependencies beyond Python 3.12 stdlib.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("eda_broker")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    """Immutable event record."""
    event_id: str
    topic: str
    key: str          # used for partition routing
    payload: dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def create(cls, topic: str, key: str, payload: dict[str, Any]) -> "Event":
        return cls(
            event_id=str(uuid.uuid4()),
            topic=topic,
            key=key,
            payload=payload,
        )


@dataclass
class PartitionedMessage:
    """A message stored within a partition."""
    offset: int
    event: Event


# ---------------------------------------------------------------------------
# Partition — append-only log for one partition of a topic
# ---------------------------------------------------------------------------

class Partition:
    def __init__(self, partition_id: int, max_size: int = 10_000) -> None:
        self.partition_id = partition_id
        self._log: list[PartitionedMessage] = []
        self._max_size = max_size
        self._lock = asyncio.Lock()

    async def append(self, event: Event) -> int:
        """Append event and return its offset."""
        async with self._lock:
            if len(self._log) >= self._max_size:
                raise OverflowError(
                    f"Partition {self.partition_id} full ({self._max_size} msgs). "
                    "Backpressure: producer must slow down."
                )
            offset = len(self._log)
            self._log.append(PartitionedMessage(offset=offset, event=event))
            return offset

    def read_from(self, offset: int, max_messages: int = 10) -> list[PartitionedMessage]:
        """Read up to max_messages starting at offset (non-blocking)."""
        return self._log[offset: offset + max_messages]

    @property
    def size(self) -> int:
        return len(self._log)


# ---------------------------------------------------------------------------
# Broker — routes events to partitions, tracks consumer group offsets
# ---------------------------------------------------------------------------

class InMemoryBroker:
    """
    Simplified Kafka-like broker.

    - Topics are created on first publish.
    - Partition count is fixed at topic creation.
    - Consumer groups track per-partition offsets.
    - Fan-out: each group gets independent offset tracking.
    """

    def __init__(self, default_partitions: int = 3) -> None:
        self._default_partitions = default_partitions
        # topic -> list of Partition
        self._topics: dict[str, list[Partition]] = {}
        # (group_id, topic, partition_id) -> committed_offset
        self._offsets: dict[tuple[str, str, int], int] = defaultdict(int)
        self._subscribers: dict[str, set[str]] = defaultdict(set)
        self._log = logging.getLogger("broker")

    def create_topic(self, topic: str, num_partitions: int | None = None) -> None:
        if topic in self._topics:
            return
        n = num_partitions or self._default_partitions
        self._topics[topic] = [Partition(i) for i in range(n)]
        self._log.info("Created topic '%s' with %d partitions", topic, n)

    def _route_partition(self, topic: str, key: str) -> int:
        """Consistent hash routing: same key always -> same partition."""
        num_partitions = len(self._topics[topic])
        digest = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return digest % num_partitions

    async def publish(self, event: Event) -> tuple[int, int]:
        """
        Publish event to its topic.
        Returns (partition_id, offset).
        Raises OverflowError on backpressure (partition full).
        """
        if event.topic not in self._topics:
            self.create_topic(event.topic)

        partition_id = self._route_partition(event.topic, event.key)
        partition = self._topics[event.topic][partition_id]
        offset = await partition.append(event)

        self._log.debug(
            "Published event_id=%s to %s[%d] offset=%d",
            event.event_id[:8], event.topic, partition_id, offset,
        )
        return partition_id, offset

    def subscribe(self, group_id: str, topic: str) -> None:
        """Register consumer group interest in a topic."""
        if topic not in self._topics:
            self.create_topic(topic)
        self._subscribers[topic].add(group_id)
        self._log.info("Group '%s' subscribed to topic '%s'", group_id, topic)

    def poll(
        self,
        group_id: str,
        topic: str,
        partition_id: int,
        max_messages: int = 5,
    ) -> list[PartitionedMessage]:
        """
        Fetch messages for a consumer group from one partition.
        Returns messages starting at the group's committed offset.
        """
        committed = self._offsets[(group_id, topic, partition_id)]
        return self._topics[topic][partition_id].read_from(committed, max_messages)

    def commit(self, group_id: str, topic: str, partition_id: int, offset: int) -> None:
        """
        Commit offset for a group+partition.
        Next poll will start at offset+1.
        """
        self._offsets[(group_id, topic, partition_id)] = offset + 1
        self._log.debug(
            "Committed: group=%s topic=%s partition=%d offset=%d",
            group_id, topic, partition_id, offset,
        )

    def lag(self, group_id: str, topic: str) -> dict[int, int]:
        """Return per-partition lag (messages behind head) for a group."""
        result: dict[int, int] = {}
        for pid, partition in enumerate(self._topics.get(topic, [])):
            committed = self._offsets[(group_id, topic, pid)]
            result[pid] = partition.size - committed
        return result

    def topic_stats(self, topic: str) -> dict[str, Any]:
        if topic not in self._topics:
            return {}
        return {
            "topic": topic,
            "partitions": len(self._topics[topic]),
            "total_messages": sum(p.size for p in self._topics[topic]),
            "subscribers": list(self._subscribers[topic]),
        }


# ---------------------------------------------------------------------------
# Consumer — processes messages from assigned partitions
# ---------------------------------------------------------------------------

class Consumer:
    """
    Single consumer instance within a group.
    Assigned a subset of partitions (simulated static assignment).
    Uses at-least-once: commit only after successful process.
    """

    def __init__(
        self,
        consumer_id: str,
        group_id: str,
        topic: str,
        partitions: list[int],
        broker: InMemoryBroker,
        process_delay: float = 0.01,
    ) -> None:
        self.consumer_id = consumer_id
        self.group_id = group_id
        self.topic = topic
        self.partitions = partitions
        self._broker = broker
        self._process_delay = process_delay
        self._processed: list[dict[str, Any]] = []
        self._log = logging.getLogger(f"consumer.{group_id}.{consumer_id}")

    async def process_message(self, msg: PartitionedMessage) -> None:
        """Simulate message processing with some work."""
        await asyncio.sleep(self._process_delay)
        self._processed.append({
            "event_id": msg.event.event_id,
            "partition": msg.offset,
            "key": msg.event.key,
            "payload": msg.event.payload,
        })
        self._log.debug(
            "Processed event_id=%s key=%s",
            msg.event.event_id[:8], msg.event.key,
        )

    async def poll_once(self) -> int:
        """Poll all assigned partitions, process and commit. Returns total processed."""
        total = 0
        for pid in self.partitions:
            messages = self._broker.poll(self.group_id, self.topic, pid)
            for msg in messages:
                await self.process_message(msg)
                # at-least-once: commit after processing
                self._broker.commit(self.group_id, self.topic, pid, msg.offset)
                total += 1
        return total

    async def run(self, max_polls: int = 20) -> None:
        """Run poll loop until no more messages or max_polls reached."""
        polls_with_no_data = 0
        for _ in range(max_polls):
            count = await self.poll_once()
            if count == 0:
                polls_with_no_data += 1
                if polls_with_no_data >= 3:
                    break
                await asyncio.sleep(0.05)
            else:
                polls_with_no_data = 0
        self._log.info(
            "Consumer %s done. Processed %d messages total.",
            self.consumer_id, len(self._processed),
        )

    @property
    def processed_count(self) -> int:
        return len(self._processed)


# ---------------------------------------------------------------------------
# Producer — publishes events with optional rate limiting
# ---------------------------------------------------------------------------

class Producer:
    def __init__(
        self,
        producer_id: str,
        broker: InMemoryBroker,
        rate_limit: float = 0.005,  # seconds between publishes
    ) -> None:
        self.producer_id = producer_id
        self._broker = broker
        self._rate_limit = rate_limit
        self._log = logging.getLogger(f"producer.{producer_id}")

    async def publish_batch(
        self,
        topic: str,
        events: list[tuple[str, dict[str, Any]]],
    ) -> int:
        """
        Publish a batch of (key, payload) tuples.
        Respects rate_limit between publishes.
        Returns count of successfully published events.
        """
        published = 0
        for key, payload in events:
            event = Event.create(topic=topic, key=key, payload=payload)
            try:
                partition_id, offset = await self._broker.publish(event)
                published += 1
            except OverflowError as exc:
                self._log.warning("Backpressure! Dropping event: %s", exc)
            if self._rate_limit > 0:
                await asyncio.sleep(self._rate_limit)
        return published


# ---------------------------------------------------------------------------
# Demo: fan-out with two independent consumer groups
# ---------------------------------------------------------------------------

async def demo_fanout(broker: InMemoryBroker) -> None:
    """
    Two consumer groups (billing, analytics) both read from 'orders'.
    Each gets all messages independently (fan-out).
    """
    logger.info("=== Fan-out Demo ===")

    topic = "orders"
    broker.create_topic(topic, num_partitions=3)

    # Two groups subscribe independently
    broker.subscribe("billing", topic)
    broker.subscribe("analytics", topic)

    # Producer publishes 30 order events
    producer = Producer("order-service", broker)
    order_events = [
        (f"order-{i}", {"order_id": i, "amount": i * 10.5, "user_id": f"u{i % 5}"})
        for i in range(30)
    ]
    published = await producer.publish_batch(topic, order_events)
    logger.info("Published %d events", published)

    # Billing group: 2 consumers, each handles partitions 0,1 and 2 respectively
    billing_c0 = Consumer("c0", "billing", topic, [0, 1], broker)
    billing_c1 = Consumer("c1", "billing", topic, [2], broker)

    # Analytics group: single consumer handles all partitions
    analytics_c0 = Consumer("c0", "analytics", topic, [0, 1, 2], broker)

    # Run all consumers concurrently
    await asyncio.gather(
        billing_c0.run(),
        billing_c1.run(),
        analytics_c0.run(),
    )

    billing_total = billing_c0.processed_count + billing_c1.processed_count
    logger.info("Billing group processed: %d", billing_total)
    logger.info("Analytics group processed: %d", analytics_c0.processed_count)
    logger.info("Fan-out: both groups should see all %d messages", published)

    stats = broker.topic_stats(topic)
    logger.info("Topic stats: %s", stats)


# ---------------------------------------------------------------------------
# Demo: backpressure with small partitions
# ---------------------------------------------------------------------------

async def demo_backpressure(broker: InMemoryBroker) -> None:
    """
    Publish more events than the partition buffer can hold.
    Broker raises OverflowError; producer logs drops.
    """
    logger.info("=== Backpressure Demo ===")

    topic = "telemetry"
    # Tiny partition buffer to trigger backpressure quickly
    broker._topics[topic] = [Partition(0, max_size=5)]
    broker.subscribe("monitor", topic)

    producer = Producer("sensor", broker, rate_limit=0.0)
    events = [("sensor-1", {"temp": 22.0 + i}) for i in range(15)]
    published = await producer.publish_batch(topic, events)
    logger.info("Sent 15 events, published %d (rest dropped due to backpressure)", published)


# ---------------------------------------------------------------------------
# Demo: at-least-once — simulate a crash mid-process
# ---------------------------------------------------------------------------

async def demo_at_least_once(broker: InMemoryBroker) -> None:
    """
    Crash consumer after processing but before committing.
    On restart the same messages are redelivered (at-least-once).
    """
    logger.info("=== At-Least-Once Demo ===")

    topic = "payments"
    broker.create_topic(topic, num_partitions=1)
    broker.subscribe("ledger", topic)

    producer = Producer("payment-svc", broker, rate_limit=0.0)
    await producer.publish_batch(
        topic,
        [("payment", {"amount": i * 5}) for i in range(5)],
    )

    # First run: process messages 0,1,2 then "crash" without committing 2
    messages = broker.poll("ledger", topic, partition_id=0, max_messages=3)
    for msg in messages[:2]:  # commit only first two
        broker.commit("ledger", topic, 0, msg.offset)
    logger.info("Crashed after processing msg offsets 0,1 (offset 2 not committed)")

    # Restart: at-least-once means we re-read from offset 2
    redelivered = broker.poll("ledger", topic, partition_id=0, max_messages=5)
    logger.info(
        "After restart, redelivered offsets: %s",
        [m.offset for m in redelivered],
    )
    assert redelivered[0].offset == 2, "Expected re-delivery starting at offset 2"
    logger.info("At-least-once confirmed: offset 2 redelivered after simulated crash")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    broker = InMemoryBroker(default_partitions=3)

    await demo_fanout(broker)
    print()

    await demo_backpressure(broker)
    print()

    await demo_at_least_once(broker)
    print()

    logger.info("All demos complete.")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Exercises

1. **Partition rebalancing.** The current broker assigns partitions statically. Add a `rebalance(group_id, topic, num_consumers)` method that redistributes partitions evenly across a new consumer count. Verify no messages are lost or double-counted across the rebalance.

2. **Dead-letter queue.** Add a `max_retries` parameter to `Consumer`. If processing raises an exception more than `max_retries` times, route the message to a `<topic>-dlq` topic instead of retrying. Write a test that injects a failing message and confirms it lands in the DLQ.

3. **Consumer lag monitor.** Write an async `lag_monitor(broker, group_id, topic, interval=1.0)` coroutine that logs per-partition lag every `interval` seconds and raises an alert (log at WARNING) when any partition lag exceeds a threshold. Test it by pausing a consumer for a few seconds.

4. **Exactly-once simulation.** Extend the broker with a transactional API: `begin_transaction()`, `publish_transactional()`, and `commit_transaction()`. Within a transaction, events are held in a staging area and only become visible to consumers on `commit_transaction`. Simulate an abort where staged messages never appear.

5. **Benchmark partitioning.** Publish 10,000 events with 100 unique keys across 3 and 10 partitions. Plot (or log) the distribution of messages per partition. Verify that consistent hashing produces a reasonably even spread.

*(Solutions: ../solutions/week-04/day-1.md)*

---

## Self-check

- [ ] I can explain why partitioning enables horizontal scale without sacrificing per-key ordering
- [ ] I can describe the difference between at-least-once and exactly-once delivery and the trade-off for each
- [ ] I can implement a consumer group that distributes partition ownership across multiple consumers
- [ ] I can explain what backpressure is and name two mechanisms to handle it
- [ ] I can trace what happens to consumer group offsets when a consumer crashes and restarts

---

## Supplementary Reading

- [The Log: What every software engineer should know about real-time data](https://engineering.linkedin.com/distributed-systems/log-what-every-software-engineer-should-know-about-real-time-datas-unifying) — Jay Kreps, LinkedIn Engineering
- [Kafka: The Definitive Guide (ch. 1–3)](https://www.confluent.io/resources/kafka-the-definitive-guide/) — architectural overview, partitions, consumer groups
- [Designing Event-Driven Systems](https://www.confluent.io/designing-event-driven-systems/) — Ben Stopford (free PDF from Confluent)
- [Apache Kafka documentation: Introduction](https://kafka.apache.org/documentation/#introduction)
- [Backpressure explained](https://medium.com/@jayphelps/backpressure-explained-the-flow-of-data-through-software-2350b3e77ce7) — Jay Phelps
