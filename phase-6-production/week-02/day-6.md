# Phase 6 · Week 02 · Day 6 — Dead-Letter Queues, Ordering & Backpressure, Broker Choice

> Objective: Master failure isolation with DLQs, understand ordering guarantees, control backpressure, and select the right broker for a given workload.  ·  Time: ~1 hr  ·  Prereqs: [[day-5]]

## Concept

### Dead-letter queues

A **dead-letter queue (DLQ)** is a holding area for messages that cannot be processed successfully. Every production messaging system needs one; without it, a single bad message ("poison pill") can block an entire consumer forever.

A message ends up in the DLQ when any of the following occur:

| Trigger | Example |
|---|---|
| Max retries exceeded | Consumer throws an exception on every attempt |
| Parse / schema failure | JSON field missing, wrong type, encoding error |
| TTL expired | Message sat in queue longer than its time-to-live |
| Explicit rejection | Consumer calls `nack(requeue=False)` after detecting a poison pill |
| DLQ-forwarding rule | Broker-side routing rule (RabbitMQ dead-letter exchange) |

**What goes in the DLQ record:** the original message payload, the original topic/queue name, the timestamp, the failure reason, and the retry count. You need all of this to debug and replay.

**DLQ operations:**
- **Inspect**: read messages without consuming (non-destructive peek).
- **Replay**: re-publish to the original topic after fixing the root cause.
- **Purge**: discard permanently after confirming messages are unrecoverable.

DLQ monitoring is a production SLA requirement. Alert on DLQ depth > 0; a non-zero DLQ means real messages from real users have failed.

### Ordering guarantees

Ordering is one of the most misunderstood properties of message queues.

| Guarantee | How achieved | Trade-offs |
|---|---|---|
| **Global order** | Single partition / single queue | No parallelism; bottleneck at scale |
| **Per-key order** | Partition by key (Kafka), consistent hashing | Parallel partitions; order within each key |
| **No order** | Round-robin across partitions | Maximum throughput; consumers must be order-independent |

Kafka's model: messages with the same partition key always go to the same partition. All messages in a partition are ordered by offset. Different partitions are consumed in parallel — no cross-partition ordering.

Practical rule: if two events for the same entity (e.g. same `order_id`) must be processed in sequence, use that entity's ID as the partition key.

**Gotcha:** Even per-key order breaks if a consumer crashes mid-batch and retries from an earlier offset. Idempotent consumers (Day 5) are required alongside ordering guarantees.

### Backpressure

**Backpressure** is the mechanism by which a slow consumer signals to a fast producer to slow down. Without backpressure, queues grow unboundedly until disk or memory exhausts.

The failure cascade:
1. Producer publishes faster than consumer processes.
2. Queue depth grows.
3. Broker memory fills; broker may write to disk (Kafka: always disk; RabbitMQ: memory alarm triggers disk spill).
4. Broker disk fills; new publishes start failing or blocking.
5. Producers back up; their memory fills; upstream services start failing.

**Backpressure strategies:**

| Strategy | Where applied | Mechanism |
|---|---|---|
| Consumer pause/resume | Consumer side | Pause polling when internal buffer full; resume when drained |
| Rate limiting producers | Producer side | Token bucket / leaky bucket; reject publishes above rate |
| Queue depth limits | Broker side | Reject publish when queue exceeds max depth (RabbitMQ `x-max-length`) |
| Kafka consumer lag alerts | Ops | Alert + autoscale consumers when lag exceeds threshold |

In Python async consumers: `await asyncio.sleep()` in the poll loop effectively rate-limits consumption. In Kafka (confluent-kafka): call `consumer.pause([TopicPartition(...)])` to stop fetching from specific partitions without closing the connection.

### Broker choice guide

No single broker is best for every workload. The decision matrix:

| Criterion | Kafka | RabbitMQ | Redis Streams | AWS SQS/SNS |
|---|---|---|---|---|
| **Throughput** | Very high (millions/s) | Medium (tens of thousands/s) | High (hundreds of thousands/s) | High (managed scaling) |
| **Latency** | Low–medium (ms) | Very low (sub-ms) | Very low (sub-ms, in-memory) | Low–medium (ms) |
| **Ordering** | Per-partition (per-key) | Per-queue (single consumer) | Per-stream | FIFO queues only |
| **Replay** | Yes (configurable retention) | No (consumed = gone) | Yes (consumer group offsets) | No |
| **Routing complexity** | Simple (topic/partition) | Rich (exchanges, bindings, fanout) | Minimal | Topic/fanout via SNS |
| **Ops overhead** | High (ZooKeeper/KRaft, storage) | Medium (cluster management) | Low (reuse existing Redis) | Zero (fully managed) |
| **Event sourcing** | Yes | No | Limited | No |
| **Consumer groups** | Yes (native, powerful) | Yes (competing consumers) | Yes (XREADGROUP) | Yes (multiple subscribers via SNS) |
| **Cost** | Infrastructure + ops | Infrastructure + ops | Incremental to Redis cost | Pay-per-message, scales to zero |

**Decision heuristic:**
- Need replay / event sourcing / >100k msg/s? → **Kafka**
- Need complex routing / per-message TTL / sub-ms latency for tasks? → **RabbitMQ**
- Already running Redis, moderate throughput, want consumer groups? → **Redis Streams**
- AWS-native, no ops budget, FIFO ordering or fan-out? → **SQS/SNS**

## How it works internally

### DLQ routing mechanics

In **Kafka**, there is no native DLQ concept — you build it: the consumer catches the exception, publishes to a `<topic>.dlq` topic (a regular Kafka topic), and commits the original offset. The DLQ topic can have a separate consumer group for inspection.

In **RabbitMQ**, DLQs are a first-class feature: a queue declares a `x-dead-letter-exchange` argument; when a message is nacked or expires, the broker routes it to that exchange automatically.

In an in-memory broker (what we use in this course), you implement DLQ as a secondary `dict[str, list[DeadLetterRecord]]` keyed by original topic.

### Backpressure at the kernel level

The TCP stack already implements backpressure: the receiver advertises a window size; when its buffer fills, it sets the window to zero; the sender blocks. Most message broker clients sit on top of TCP, so this low-level backpressure propagates up — but only if the broker client does not buffer indefinitely in user space.

Kafka's `max.in.flight.requests.per.connection` and `buffer.memory` settings control exactly where the producer blocks. When `buffer.memory` fills, `producer.send()` blocks (or raises after `max.block.ms`). This is the backpressure signal reaching the producer.

### Partition assignment and rebalancing

When a Kafka consumer group member joins or leaves, the group coordinator triggers a **rebalance**: partitions are reassigned across live members. During a rebalance, all consumption pauses. This is why sticky partition assignment (`partition.assignment.strategy=cooperative-sticky`) matters for large consumer groups — it minimizes partition movement and reduces pause time.

## Worked example

The following extends the `InMemoryBroker` from Day 5 with:
- Configurable `max_retries` per topic
- Automatic DLQ routing on retry exhaustion
- A `DLQInspector` for non-destructive reads
- A `DLQReplayer` that re-publishes failed messages to the original topic

```python
"""
day6_dlq.py — InMemoryBroker with dead-letter queue, retry counting,
               DLQ inspector, and replayer.

Run: python day6_dlq.py
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import uuid4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Message:
    """A single message on the broker."""

    id: str
    topic: str
    payload: Any
    attempt: int = 0
    published_at: float = field(default_factory=time.monotonic)


@dataclass
class DeadLetterRecord:
    """A message that exhausted all retries or was explicitly rejected."""

    original_message: Message
    original_topic: str
    failure_reason: str
    failed_at: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# Broker
# ---------------------------------------------------------------------------


class InMemoryBrokerWithDLQ:
    """
    Thread-naive in-memory broker with DLQ support.

    - Each topic has a FIFO queue of Message objects.
    - max_retries: how many times to attempt delivery before DLQ routing.
    - Handlers registered via subscribe(); called synchronously on consume().
    - On handler exception: increment attempt, re-enqueue or route to DLQ.
    """

    def __init__(self, max_retries: int = 3) -> None:
        self._max_retries = max_retries
        self._queues: dict[str, list[Message]] = defaultdict(list)
        self._handlers: dict[str, list[Callable[[Message], None]]] = defaultdict(list)
        self._dlq: dict[str, list[DeadLetterRecord]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Producer interface
    # ------------------------------------------------------------------

    def publish(self, topic: str, payload: Any) -> str:
        """Enqueue a new message. Returns the message id."""
        msg = Message(id=str(uuid4()), topic=topic, payload=payload)
        self._queues[topic].append(msg)
        log.debug("Published msg_id=%s to topic=%s", msg.id, topic)
        return msg.id

    # ------------------------------------------------------------------
    # Consumer interface
    # ------------------------------------------------------------------

    def subscribe(
        self,
        topic: str,
        handler: Callable[[Message], None],
    ) -> None:
        """Register a handler for a topic. Multiple handlers = fanout."""
        self._handlers[topic].append(handler)
        log.debug("Subscribed handler %s to topic=%s", handler.__name__, topic)

    def consume_one(self, topic: str) -> bool:
        """
        Dequeue and process one message from topic.

        Returns True if a message was processed (successfully or routed to DLQ),
        False if the queue is empty.
        """
        queue = self._queues[topic]
        if not queue:
            return False

        msg = queue.pop(0)
        msg.attempt += 1
        handlers = self._handlers.get(topic, [])

        for handler in handlers:
            try:
                handler(msg)
                log.info(
                    "OK    msg_id=%s topic=%s handler=%s attempt=%d",
                    msg.id,
                    topic,
                    handler.__name__,
                    msg.attempt,
                )
            except Exception as exc:
                self._handle_failure(msg, handler.__name__, str(exc))

        return True

    def consume_all(self, topic: str) -> int:
        """Drain the topic queue. Returns number of messages processed."""
        count = 0
        while self.consume_one(topic):
            count += 1
        return count

    # ------------------------------------------------------------------
    # Internal failure routing
    # ------------------------------------------------------------------

    def _handle_failure(
        self, msg: Message, handler_name: str, reason: str
    ) -> None:
        if msg.attempt < self._max_retries:
            log.warning(
                "RETRY msg_id=%s topic=%s handler=%s attempt=%d/%d reason=%s",
                msg.id,
                msg.topic,
                handler_name,
                msg.attempt,
                self._max_retries,
                reason,
            )
            # Re-enqueue at the back — simple retry strategy
            self._queues[msg.topic].append(msg)
        else:
            record = DeadLetterRecord(
                original_message=msg,
                original_topic=msg.topic,
                failure_reason=reason,
            )
            self._dlq[msg.topic].append(record)
            log.error(
                "DLQ   msg_id=%s topic=%s after %d attempts reason=%s",
                msg.id,
                msg.topic,
                msg.attempt,
                reason,
            )

    # ------------------------------------------------------------------
    # DLQ interface
    # ------------------------------------------------------------------

    def dlq_depth(self, topic: str) -> int:
        return len(self._dlq[topic])

    def dlq_inspect(self, topic: str) -> list[DeadLetterRecord]:
        """Non-destructive read of all DLQ records for a topic."""
        return list(self._dlq[topic])

    def dlq_replay(self, topic: str) -> int:
        """
        Re-publish all DLQ messages for topic back to their original topic.

        Resets attempt counter so they get a fresh set of retries.
        Returns number of messages replayed.
        """
        records = self._dlq.pop(topic, [])
        for record in records:
            # Clone with reset attempt counter
            fresh = Message(
                id=record.original_message.id,
                topic=record.original_topic,
                payload=record.original_message.payload,
                attempt=0,
            )
            self._queues[record.original_topic].append(fresh)
            log.info(
                "REPLAY msg_id=%s back to topic=%s",
                fresh.id,
                record.original_topic,
            )
        return len(records)

    def dlq_purge(self, topic: str) -> int:
        """Permanently discard all DLQ records for topic. Returns count removed."""
        records = self._dlq.pop(topic, [])
        log.warning("PURGE  %d DLQ records for topic=%s", len(records), topic)
        return len(records)


# ---------------------------------------------------------------------------
# Demo: order processing with a flaky handler
# ---------------------------------------------------------------------------

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# Track which message IDs have been "seen" to simulate a transient bug
_fail_first_n: dict[str, int] = {}


def flaky_order_handler(msg: Message) -> None:
    """
    Simulates a handler that fails on the first two attempts for any message,
    succeeds on the third.  After replay it should succeed immediately.
    """
    order_id = msg.payload.get("order_id", "unknown")
    seen = _fail_first_n.get(order_id, 0)
    if seen < 2:
        _fail_first_n[order_id] = seen + 1
        raise ValueError(f"Transient error processing order {order_id} (attempt {msg.attempt})")
    log.info("Processed order_id=%s successfully", order_id)


def poison_pill_handler(msg: Message) -> None:
    """Always fails — simulates a parse error."""
    raise ValueError(f"Unrecognised payload schema: {msg.payload!r}")


def run_demo() -> None:
    broker = InMemoryBrokerWithDLQ(max_retries=MAX_RETRIES)

    # Register handlers
    broker.subscribe("orders", flaky_order_handler)
    broker.subscribe("orders.poison", poison_pill_handler)

    # Publish messages
    good_id = broker.publish("orders", {"order_id": "ORD-001", "total": 99.0})
    bad_id = broker.publish("orders.poison", {"bad_key": "????"})

    log.info("--- Processing 'orders' (expect retries then success) ---")
    broker.consume_all("orders")

    log.info("--- Processing 'orders.poison' (expect DLQ routing) ---")
    broker.consume_all("orders.poison")

    # Inspect DLQ
    dlq_records = broker.dlq_inspect("orders.poison")
    log.info("DLQ depth for 'orders.poison': %d", len(dlq_records))
    for rec in dlq_records:
        log.info(
            "  DLQ record: msg_id=%s reason=%s",
            rec.original_message.id,
            rec.failure_reason,
        )

    # Replay the DLQ (would normally happen after fixing the handler)
    replayed = broker.dlq_replay("orders.poison")
    log.info("Replayed %d messages from DLQ", replayed)

    # After replay the messages are back in the queue — but the handler
    # is still broken, so they will DLQ again. In a real system you would
    # fix the handler first, then replay.
    broker.consume_all("orders.poison")
    log.info(
        "DLQ depth after second round: %d",
        broker.dlq_depth("orders.poison"),
    )


if __name__ == "__main__":
    run_demo()
```

### Trace walkthrough

1. `ORD-001` is published to `orders`. `flaky_order_handler` fails on attempts 1 and 2 (transient error). On attempt 3 it succeeds — message never reaches DLQ.
2. The poison-pill message is published to `orders.poison`. `poison_pill_handler` fails every time. After 3 attempts the broker routes it to the DLQ for `orders.poison`.
3. `dlq_inspect` returns the record non-destructively.
4. `dlq_replay` re-publishes the message to `orders.poison` with `attempt=0`. The handler is still broken, so it DLQs again — illustrating why you fix the root cause before replaying.

## Exercises

1. Add a `max_queue_depth` parameter to `InMemoryBrokerWithDLQ`. When `publish()` is called and the topic queue length equals `max_queue_depth`, raise a `BackpressureError` instead of enqueuing. Write a test that verifies this behavior at exactly the limit.

2. Implement a `consume_with_rate_limit(topic: str, messages_per_second: float)` method that processes messages from a topic but inserts `time.sleep()` between calls to stay at or below the given rate. What are the limits of `time.sleep`-based rate limiting in production?

3. Extend `dlq_replay` to accept an optional `transform: Callable[[Message], Message]` argument. This allows fixing message payloads before requeuing (e.g. adding a missing field). Write a test that replays a DLQ message with a corrected payload.

4. The current retry strategy re-enqueues at the back of the queue (FIFO retry). Implement an **exponential-backoff retry** variant: instead of immediately re-enqueuing, store the message with a `retry_after: float` timestamp. Add a `consume_due(topic: str)` method that only processes messages whose `retry_after <= time.monotonic()`. How does this interact with ordering guarantees?

5. Build a `BrokerChoiceAdvisor` that accepts a `WorkloadProfile` Pydantic model (`throughput_msgs_per_sec: int`, `requires_replay: bool`, `requires_ordering: bool`, `managed_only: bool`, `existing_redis: bool`) and returns a recommended broker name with a one-line justification. No external calls — pure logic.

(Solutions: ../solutions/week-02/day-6.md)

## Self-check

- [ ] I can explain what a DLQ is and name three conditions under which a message lands there
- [ ] I can distinguish global order, per-key order, and no-order guarantees and state when each is appropriate
- [ ] I can describe the backpressure failure cascade and name two strategies to prevent it
- [ ] I can read the broker decision matrix and select a broker for a given workload profile
- [ ] I can implement DLQ routing, inspection, and replay in the in-memory broker

## Supplementary reading

- [Kafka Documentation: Dead Letter Queue pattern](https://kafka.apache.org/documentation/#design_deadletter)
- [RabbitMQ Dead Letter Exchanges](https://www.rabbitmq.com/docs/dlx)
- [Redis Streams consumer groups and delivery guarantees](https://redis.io/docs/latest/develop/data-types/streams/)
- [AWS SQS Dead-Letter Queues](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-dead-letter-queues.html)
- [Confluent: Kafka Producer Backpressure](https://docs.confluent.io/platform/current/installation/configuration/producer-configs.html)
- Designing Data-Intensive Applications (Kleppmann) — Chapter 11: Stream Processing
