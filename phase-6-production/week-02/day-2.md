# Phase 6 · Week 02 · Day 2 — Queues vs Pub/Sub: Messaging Patterns

> Objective: Distinguish point-to-point queues from publish/subscribe and know which pattern fits which problem.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

---

## Concept

On Day 1 we saw that async messaging decouples producer from consumer in time. Today we look at *topology* — how messages fan out from a single producer to one or many consumers.

There are two fundamental patterns, and real brokers (RabbitMQ, Kafka, Google Pub/Sub) are combinations of them.

### Point-to-point queue (work queue)

One producer. Many consumers. Each message is delivered to **exactly one** consumer. Once a consumer acknowledges it, the message is gone from the queue.

```
Producer ──► [Queue] ──► Consumer A  (gets message 1)
                    └──► Consumer B  (gets message 2)
```

Use case: distributing work. An image resize job should be done once, not N times. Adding consumers scales throughput.

Key property: **competing consumers** — they race to claim messages. Load is shared automatically.

### Publish/subscribe (fanout)

One producer. Many consumers. Each message is delivered to **every subscriber**. Consumers each maintain their own cursor (offset, pointer, or private queue copy).

```
Producer ──► [Topic/Exchange]
               ├──► Consumer A  (gets a copy)
               ├──► Consumer B  (gets a copy)
               └──► Consumer C  (gets a copy)
```

Use case: event notification. An `OrderPlaced` event should trigger both the inventory service and the email service independently.

Key property: **independent consumption** — adding a new subscriber does not affect existing ones.

### Exchange types (conceptual)

RabbitMQ makes the routing logic explicit through an **exchange** object between the producer and the queues. Three common types:

| Exchange type | Routing rule | Use case |
|---|---|---|
| **fanout** | Send to all bound queues, no inspection | Broadcast notifications |
| **direct** | Route by exact routing key | Task dispatch by type |
| **topic** | Route by wildcard pattern on key | Multi-dimensional filtering (e.g. `orders.eu.*`) |

Kafka uses a different model: a **topic** is a partitioned, ordered log. Consumers in different **consumer groups** each get a full copy (pub/sub). Consumers in the same group share partitions (work queue). Kafka unifies both patterns through the consumer group concept.

### Filtering and routing keys

A routing key is a string attached to each message. Consumers (or their queue bindings) declare which keys they care about.

- `direct` exchange: key `"payment"` → only payment queue
- `topic` exchange: key `"orders.eu.express"` matches `"orders.eu.*"` and `"orders.#"` but not `"orders.us.*"`

Filtering lets a single topic carry heterogeneous events without forcing every consumer to deserialize and discard unwanted types.

### Choosing the pattern

| Situation | Pattern | Reason |
|---|---|---|
| Background job workers (resize, transcode) | Work queue | Each job done once, parallelism via multiple consumers |
| Audit log + email + metrics on same event | Pub/Sub fanout | Each service needs its own copy |
| Route `payment` jobs to payment workers only | Direct exchange | Routing key filters destination |
| Per-region routing (`orders.eu.*`) | Topic exchange | Wildcard pattern on key |
| Replay past events (new consumer joins later) | Kafka topic | Persistent log, consumer starts from offset 0 |

---

## How it works internally

### Work queue mechanics

The broker holds messages in a FIFO structure. When a consumer connects, the broker assigns the next unacked message to it and marks it **in-flight**. The message stays in the broker's in-flight set until the consumer sends an explicit ack. If the consumer crashes, the broker requeues the in-flight message.

Multiple consumers connect to the same queue. The broker distributes via round-robin (RabbitMQ default) or by availability (first consumer to signal readiness gets the next message). This is why competing consumers scale throughput linearly — each consumer processes its own slice.

### Pub/Sub mechanics

In a true pub/sub model, the broker maintains **one queue per subscriber**. When a message is published to a topic, the broker copies it into each subscriber's queue. Each subscriber drains its own queue independently.

This is memory-intensive for many subscribers and ephemeral for new ones (they miss past messages). Kafka solves this differently: the topic is a durable log. Each consumer group tracks its own **offset** (a pointer into the log). New consumers can rewind to offset 0 and replay all history. The log is never deleted on consumption — it ages out by time or size.

### In-memory broker design

The in-memory broker in the worked example below models both patterns:

- **Queue mode**: a single `asyncio.Queue` shared by all consumers. First consumer to `get()` wins.
- **Pub/Sub mode**: a `dict[subscriber_id, asyncio.Queue]`. The broker fans out by iterating the dict and calling `put()` on each private queue.

This is structurally identical to how RabbitMQ handles fanout exchanges.

---

## Worked example

```python
"""
day_2_broker.py

In-memory broker supporting both work-queue (competing consumers)
and pub/sub (fanout) modes.

Usage:
    python day_2_broker.py
"""

import asyncio
import logging
import time
import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
)
log = logging.getLogger("day2")


# ---------------------------------------------------------------------------
# Message model
# ---------------------------------------------------------------------------

class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    topic: str
    routing_key: str = ""
    payload: dict[str, Any]
    created_at: float = Field(default_factory=time.time)

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Broker
# ---------------------------------------------------------------------------

class BrokerMode(StrEnum):
    WORK_QUEUE = "work_queue"
    PUBSUB = "pubsub"


class InMemoryBroker:
    """
    Minimal in-memory message broker.

    WORK_QUEUE mode: one shared queue, first consumer to get() wins.
    PUBSUB mode:     each subscriber gets its own queue; broker fans out.
    """

    def __init__(self, mode: BrokerMode = BrokerMode.WORK_QUEUE) -> None:
        self.mode = mode
        # work_queue mode: one queue per topic
        self._work_queues: dict[str, asyncio.Queue[Message]] = {}
        # pubsub mode: topic -> {subscriber_id -> queue}
        self._subscriptions: dict[str, dict[str, asyncio.Queue[Message]]] = {}
        self._published: int = 0

    # ------------------------------------------------------------------
    # Work queue API
    # ------------------------------------------------------------------

    def _get_or_create_work_queue(self, topic: str) -> asyncio.Queue[Message]:
        if topic not in self._work_queues:
            self._work_queues[topic] = asyncio.Queue()
        return self._work_queues[topic]

    async def publish(self, message: Message) -> None:
        """Publish to whichever mode is active."""
        self._published += 1
        if self.mode == BrokerMode.WORK_QUEUE:
            q = self._get_or_create_work_queue(message.topic)
            await q.put(message)
            log.debug(
                "[BROKER] work_queue topic=%s depth=%d msg=%s",
                message.topic,
                q.qsize(),
                message.id,
            )
        else:
            # Fanout to every subscriber on this topic
            subs = self._subscriptions.get(message.topic, {})
            if not subs:
                log.warning("[BROKER] No subscribers for topic=%s", message.topic)
                return
            for sub_id, q in subs.items():
                await q.put(message)
            log.debug(
                "[BROKER] pubsub topic=%s fanout_to=%d msg=%s",
                message.topic,
                len(subs),
                message.id,
            )

    # ------------------------------------------------------------------
    # Consumer API — work queue
    # ------------------------------------------------------------------

    async def consume_one(self, topic: str, timeout_s: float = 2.0) -> Message | None:
        """
        Claim one message from the work queue (competing consumers).
        Returns None on timeout.
        """
        q = self._get_or_create_work_queue(topic)
        try:
            return await asyncio.wait_for(q.get(), timeout=timeout_s)
        except TimeoutError:
            return None

    def ack(self, topic: str) -> None:
        """Signal that the last consumed message was processed successfully."""
        q = self._work_queues.get(topic)
        if q:
            q.task_done()

    # ------------------------------------------------------------------
    # Consumer API — pub/sub
    # ------------------------------------------------------------------

    def subscribe(self, topic: str, subscriber_id: str) -> asyncio.Queue[Message]:
        """Register a subscriber and return its private queue."""
        if topic not in self._subscriptions:
            self._subscriptions[topic] = {}
        if subscriber_id not in self._subscriptions[topic]:
            self._subscriptions[topic][subscriber_id] = asyncio.Queue()
            log.info("[BROKER] subscriber %s joined topic=%s", subscriber_id, topic)
        return self._subscriptions[topic][subscriber_id]

    def unsubscribe(self, topic: str, subscriber_id: str) -> None:
        self._subscriptions.get(topic, {}).pop(subscriber_id, None)
        log.info("[BROKER] subscriber %s left topic=%s", subscriber_id, topic)

    async def receive(
        self,
        queue: asyncio.Queue[Message],
        subscriber_id: str,
        timeout_s: float = 2.0,
    ) -> Message | None:
        try:
            return await asyncio.wait_for(queue.get(), timeout=timeout_s)
        except TimeoutError:
            log.info("[BROKER] subscriber %s timed out waiting", subscriber_id)
            return None


# ---------------------------------------------------------------------------
# Demo 1: Work queue — competing consumers
# ---------------------------------------------------------------------------

async def work_queue_demo() -> None:
    log.info("=== WORK QUEUE DEMO (competing consumers) ===")
    broker = InMemoryBroker(mode=BrokerMode.WORK_QUEUE)
    topic = "resize-jobs"

    # Publish 6 jobs
    for i in range(1, 7):
        msg = Message(
            topic=topic,
            payload={"image_id": f"img-{i:03d}", "target_width": 800},
        )
        await broker.publish(msg)

    results: list[tuple[int, str]] = []

    async def worker(worker_id: int) -> None:
        while True:
            msg = await broker.consume_one(topic, timeout_s=1.0)
            if msg is None:
                break
            log.info(
                "[WRK-%d] Processing %s → image_id=%s",
                worker_id,
                msg.id,
                msg.payload["image_id"],
            )
            await asyncio.sleep(0.1)  # simulate work
            broker.ack(topic)
            results.append((worker_id, msg.payload["image_id"]))

    # Two workers competing for the same queue
    await asyncio.gather(worker(1), worker(2))

    log.info("Work distribution:")
    from collections import Counter
    dist = Counter(wid for wid, _ in results)
    for wid, count in sorted(dist.items()):
        items = [img for w, img in results if w == wid]
        log.info("  Worker %d processed %d jobs: %s", wid, count, items)
    log.info("Total processed: %d / 6\n", len(results))


# ---------------------------------------------------------------------------
# Demo 2: Pub/Sub — fanout to all subscribers
# ---------------------------------------------------------------------------

async def pubsub_demo() -> None:
    log.info("=== PUB/SUB DEMO (fanout) ===")
    broker = InMemoryBroker(mode=BrokerMode.PUBSUB)
    topic = "order-placed"

    # Three independent services subscribe
    inventory_q = broker.subscribe(topic, "inventory-service")
    email_q = broker.subscribe(topic, "email-service")
    metrics_q = broker.subscribe(topic, "metrics-service")

    # Publish 3 order events
    for i in range(1, 4):
        msg = Message(
            topic=topic,
            routing_key="orders.eu.standard",
            payload={"order_id": f"ORD-{i:03d}", "amount": i * 50.0},
        )
        await broker.publish(msg)
        log.info("[PUB  ] Published order %s", msg.payload["order_id"])

    # Each service drains its own private queue
    async def service_consumer(name: str, queue: asyncio.Queue[Message]) -> None:
        while True:
            msg = await broker.receive(queue, name, timeout_s=1.0)
            if msg is None:
                break
            log.info(
                "[%-20s] Received %s → order_id=%s",
                name,
                msg.id,
                msg.payload["order_id"],
            )
            await asyncio.sleep(0.05)

    await asyncio.gather(
        service_consumer("inventory-service", inventory_q),
        service_consumer("email-service", email_q),
        service_consumer("metrics-service", metrics_q),
    )

    log.info("Each service received all 3 messages independently.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    await work_queue_demo()
    await pubsub_demo()


if __name__ == "__main__":
    asyncio.run(main())
```

**What to observe:**

1. **Work queue**: 6 jobs are split between Worker 1 and Worker 2. Neither worker gets the same job twice. Total processed = 6.
2. **Pub/Sub**: 3 messages are published. Each of the 3 services receives all 3 messages. Total receives = 9. The fanout is silent to the publisher.
3. The broker code for the two modes is structurally different: one shared queue vs. per-subscriber queues. This mirrors the RabbitMQ direct/fanout exchange distinction.

---

## Exercises

1. Extend `InMemoryBroker` to support **topic-based filtering**: add a `subscribe(topic, subscriber_id, routing_key_pattern)` parameter. Use `fnmatch.fnmatch` to filter messages. Only deliver a message to a subscriber if its routing key matches the subscriber's pattern. Test with `orders.eu.*` and `orders.us.*` patterns.

2. Add a **dead letter queue** to the work queue mode: if `consume_one` returns a message but `ack()` is never called within 1 second (simulate with `asyncio.wait_for`), the broker moves the message to a `_dlq` queue. Implement a `drain_dlq(topic)` method and demonstrate recovery.

3. Add a new subscriber `analytics-service` to the pub/sub demo **after** the first message has already been published. What does it receive? Implement a **replay** mechanism: store all published messages in `self._history[topic]` and deliver history to new subscribers on join.

4. Implement a `broadcast_to_topic` method that works in `WORK_QUEUE` mode by checking all known work queues whose names start with the given prefix. Demonstrate routing `orders.eu.express` to both `orders.eu.express` and `orders.eu` queues.

5. Measure the throughput difference between `WORK_QUEUE` with N=1,2,4 consumers processing 20 jobs each taking `0.1 s`. Plot (or log) jobs/second. At what consumer count does adding more consumers stop helping? Why?

(Solutions: ../solutions/week-02/day-2.md)

---

## Self-check

- [ ] I can explain the difference between point-to-point and pub/sub without looking at notes.
- [ ] I can describe why competing consumers scale throughput but fanout does not reduce work.
- [ ] I can name three exchange types and give a use case for each.
- [ ] I can implement an in-memory broker supporting both modes from scratch.
- [ ] I can explain why adding a new subscriber to a queue-based system is a bigger change than adding one to a pub/sub topic.

---

## Supplementary links

- RabbitMQ — Exchanges, routing, bindings: https://www.rabbitmq.com/tutorials/amqp-concepts
- Kafka — Consumer groups and offsets: https://kafka.apache.org/documentation/#intro_consumers
- Enterprise Integration Patterns — Message Channel: https://www.enterpriseintegrationpatterns.com/patterns/messaging/MessageChannel.html
- Enterprise Integration Patterns — Publish-Subscribe Channel: https://www.enterpriseintegrationpatterns.com/patterns/messaging/PublishSubscribeChannel.html
- Google Cloud — Pub/Sub architecture overview: https://cloud.google.com/pubsub/docs/overview
- Python `asyncio.Queue` docs: https://docs.python.org/3/library/asyncio-queue.html
