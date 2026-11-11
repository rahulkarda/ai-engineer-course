# Phase 6 · Week 02 · Day 3 — Delivery Semantics: At-Most-Once, At-Least-Once, Exactly-Once

> Objective: Understand the three delivery guarantees, their costs, and how to build an idempotent consumer that makes at-least-once safe in practice.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

---

## Concept

Every message broker makes a promise about how many times a consumer will see each message. That promise is called a **delivery semantic**. There are three.

### At-most-once

The broker fires the message and forgets it. If the network drops or the consumer crashes before processing, the message is gone.

```
Producer ──► Broker ──► Consumer
             (delete on send)
```

Typical implementation: the broker removes the message from its log the moment it hands it to the consumer, regardless of whether the consumer succeeded. No retries.

**Cost**: data loss is possible.
**Benefit**: lowest latency, simplest broker logic, no duplicate risk.
**Use when**: loss is acceptable — telemetry, metrics, log samples where occasional gaps are fine. Never for financial, state-changing, or user-visible operations.

### At-least-once

The broker keeps the message until the consumer sends an explicit **acknowledgement** (ack). If the consumer crashes or the ack is lost, the broker redelivers.

```
Producer ──► Broker ──► Consumer (processes)
                ◄── ack ─────────
          (delete after ack)
```

**Cost**: duplicates are possible. A consumer may process the message, then crash before acking. On restart it receives the same message again.
**Benefit**: no data loss. Simple broker logic.
**Use when**: duplicates are tolerable *or* the consumer is made idempotent (see below). The practical default for most systems.

### Exactly-once

Each message is processed exactly once. No loss, no duplicates.

This is hard. It requires either:
- **Idempotency keys + deduplication**: the broker or consumer tracks message IDs and discards duplicates (transforms at-least-once into effectively-once)
- **Transactional producers**: the broker and the downstream storage participate in a distributed transaction (e.g., Kafka transactions with `transactional.id` + a database)

**Cost**: higher latency, broker complexity, storage for seen IDs.
**Benefit**: true once-and-only-once semantics.
**Use when**: correctness is critical and duplicates are expensive — billing, inventory deduction, financial ledger entries.

### The practical answer

"Exactly-once" is almost always achieved by **at-least-once delivery + idempotent consumer**. This is cheaper than distributed transactions and sufficient for nearly every real use case.

### Idempotent consumers

An operation is **idempotent** if applying it multiple times produces the same result as applying it once.

```
charge_card(order_id="ORD-001", amount=50.00)   # first call  → charged
charge_card(order_id="ORD-001", amount=50.00)   # duplicate   → skipped, not double-charged
```

Implementation: maintain a `set[str]` of seen message IDs. Before processing, check membership. If seen, skip and ack. If new, process then add to the set.

In production this set lives in Redis (with a TTL), a database table, or Kafka's consumer offset store. In the worked example it lives in memory.

### Producer retries and semantic escalation

A producer that retries on failure escalates from at-most-once toward at-least-once.

```
producer sends ──► network error ──► producer retries ──► consumer sees 2 copies
```

This is correct if the consumer is idempotent. If it is not, retries cause duplicates. Every retry-capable producer must pair with an idempotent consumer.

The producer should include the same `message_id` on every retry attempt. The consumer's seen-IDs set will discard the duplicate. If the producer generates a new ID on each retry, deduplication is impossible.

---

## How it works internally

### Ack and requeue cycle

In RabbitMQ (AMQP model):
1. Broker delivers message to consumer, marks it **unacked** (in-flight).
2. Consumer calls `basic_ack(delivery_tag)` → broker deletes the message.
3. If the connection drops before ack → broker requeues → another consumer picks it up.
4. If consumer calls `basic_nack(requeue=True)` → broker requeues immediately.
5. If consumer calls `basic_nack(requeue=False)` → message goes to dead-letter exchange.

In Kafka:
1. Consumer reads records from a partition up to an offset.
2. Consumer calls `commit_offsets()` → broker marks the offset as processed.
3. If consumer crashes before commit → on restart it re-reads from the last committed offset → duplicates from the uncommitted window.
4. Exactly-once requires Kafka transactions: producer writes + offset commit in one atomic transaction.

### Seen-IDs deduplication in memory

The in-memory broker in the worked example simulates at-least-once by:
1. Keeping messages in an **unacked** dict after handing them to a consumer.
2. On timeout (simulated ack timeout), requeuing the message.
3. On ack, removing from unacked.

The idempotent consumer maintains `_seen_ids: set[str]`. Before processing any message it checks `msg.id in _seen_ids`. If true, it logs "duplicate skipped" and acks without processing. If false, it processes then adds the ID.

The failure scenario in the demo: the consumer processes message M, then raises an exception before acking. The broker requeues M. On the next delivery cycle, the consumer's seen-IDs set correctly discards the duplicate.

### TTL on seen IDs

A naive `set[str]` that grows forever is a memory leak. In production:
- Use Redis with `SETEX` to store `msg_id → 1` with a TTL equal to the maximum reasonable redelivery window (e.g., 24 hours).
- Or use a database table with a `processed_at` timestamp and a cleanup job.
- Or use Kafka's idempotent producer mode (sequence numbers per partition) for producer-side deduplication.

---

## Worked example

```python
"""
day_3_delivery_semantics.py

Demonstrates at-least-once delivery with retry + idempotent consumer.
Simulates a consumer crash mid-processing and shows that the message
is redelivered exactly once (from the consumer's perspective: processed once).

Usage:
    python day_3_delivery_semantics.py
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
)
log = logging.getLogger("day3")


# ---------------------------------------------------------------------------
# Message model
# ---------------------------------------------------------------------------

class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    topic: str
    payload: dict[str, Any]
    attempt: int = 1          # incremented by broker on each redelivery
    created_at: float = Field(default_factory=time.time)

    model_config = {"frozen": True}

    def next_attempt(self) -> "Message":
        """Return a copy of this message with attempt incremented."""
        return self.model_copy(update={"attempt": self.attempt + 1})


# ---------------------------------------------------------------------------
# At-least-once broker with retry
# ---------------------------------------------------------------------------

@dataclass
class _InFlight:
    message: Message
    delivered_at: float


class AtLeastOnceBroker:
    """
    Minimal at-least-once broker.

    - Messages are kept in an unacked store after delivery.
    - If ack() is not called within ack_timeout_s, message is requeued.
    - The requeue loop runs as a background task.
    """

    def __init__(self, ack_timeout_s: float = 0.5) -> None:
        self._queue: asyncio.Queue[Message] = asyncio.Queue()
        self._unacked: dict[str, _InFlight] = {}
        self._ack_timeout_s = ack_timeout_s
        self._published: list[str] = []
        self._requeue_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background requeue loop."""
        self._requeue_task = asyncio.create_task(self._requeue_loop())

    async def stop(self) -> None:
        if self._requeue_task:
            self._requeue_task.cancel()
            try:
                await self._requeue_task
            except asyncio.CancelledError:
                pass

    # ------------------------------------------------------------------
    # Producer API
    # ------------------------------------------------------------------

    async def publish(self, message: Message) -> None:
        self._published.append(message.id)
        await self._queue.put(message)
        log.info("[BROKER] Published msg=%s attempt=%d", message.id, message.attempt)

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------

    async def consume(self, timeout_s: float = 2.0) -> Message | None:
        """
        Return the next message and mark it unacked.
        Returns None on timeout.
        """
        try:
            msg = await asyncio.wait_for(self._queue.get(), timeout=timeout_s)
        except TimeoutError:
            return None
        self._unacked[msg.id] = _InFlight(message=msg, delivered_at=time.monotonic())
        self._queue.task_done()
        log.debug("[BROKER] Delivered msg=%s, awaiting ack", msg.id)
        return msg

    def ack(self, message_id: str) -> None:
        """Consumer signals successful processing."""
        if message_id in self._unacked:
            del self._unacked[message_id]
            log.debug("[BROKER] Acked msg=%s", message_id)

    def nack(self, message_id: str) -> None:
        """Consumer signals failure — broker requeues immediately."""
        in_flight = self._unacked.pop(message_id, None)
        if in_flight:
            asyncio.create_task(
                self._queue.put(in_flight.message.next_attempt())
            )
            log.info("[BROKER] Nacked msg=%s → requeued", message_id)

    # ------------------------------------------------------------------
    # Background requeue loop
    # ------------------------------------------------------------------

    async def _requeue_loop(self) -> None:
        """Periodically requeue messages whose ack has timed out."""
        while True:
            await asyncio.sleep(0.1)
            now = time.monotonic()
            timed_out = [
                inf
                for inf in list(self._unacked.values())
                if (now - inf.delivered_at) > self._ack_timeout_s
            ]
            for inf in timed_out:
                msg_id = inf.message.id
                self._unacked.pop(msg_id, None)
                requeued = inf.message.next_attempt()
                await self._queue.put(requeued)
                log.warning(
                    "[BROKER] Ack timeout for msg=%s → requeue attempt=%d",
                    msg_id,
                    requeued.attempt,
                )


# ---------------------------------------------------------------------------
# Idempotent consumer
# ---------------------------------------------------------------------------

class IdempotentConsumer:
    """
    Processes each message at most once, regardless of redeliveries.
    Uses an in-memory seen-IDs set as the deduplication store.
    In production: replace with Redis SETEX or a DB unique constraint.
    """

    def __init__(self, consumer_id: str) -> None:
        self.consumer_id = consumer_id
        self._seen_ids: set[str] = set()
        self.processed: list[str] = []
        self.duplicates_skipped: int = 0

    def _is_duplicate(self, message: Message) -> bool:
        return message.id in self._seen_ids

    def _mark_seen(self, message: Message) -> None:
        self._seen_ids.add(message.id)

    async def handle(self, message: Message, broker: AtLeastOnceBroker) -> None:
        if self._is_duplicate(message):
            log.info(
                "[CON-%s] DUPLICATE SKIPPED msg=%s attempt=%d",
                self.consumer_id,
                message.id,
                message.attempt,
            )
            self.duplicates_skipped += 1
            broker.ack(message.id)          # ack the duplicate so it's not requeued again
            return

        log.info(
            "[CON-%s] Processing msg=%s attempt=%d payload=%s",
            self.consumer_id,
            message.id,
            message.attempt,
            message.payload,
        )
        await asyncio.sleep(0.05)           # simulate processing work
        self._mark_seen(message)
        self.processed.append(message.id)
        broker.ack(message.id)
        log.info("[CON-%s] Done msg=%s", self.consumer_id, message.id)


# ---------------------------------------------------------------------------
# Crash-and-redeliver scenario
# ---------------------------------------------------------------------------

async def demo_crash_and_redeliver() -> None:
    """
    Simulate a consumer that crashes mid-processing on message 3.
    The broker redelivers the message. The idempotent consumer processes
    it on the second attempt without double-counting.
    """
    log.info("=== AT-LEAST-ONCE + IDEMPOTENT CONSUMER DEMO ===\n")

    broker = AtLeastOnceBroker(ack_timeout_s=0.4)
    consumer = IdempotentConsumer("A")
    await broker.start()

    # Publish 5 messages
    messages = [
        Message(topic="payments", payload={"order_id": f"ORD-{i:03d}", "amount": i * 10.0})
        for i in range(1, 6)
    ]
    for msg in messages:
        await broker.publish(msg)

    crash_target_id: str | None = None
    crashed = False

    # Consumer loop
    for _ in range(20):                     # enough iterations to drain + handle retries
        msg = await broker.consume(timeout_s=1.0)
        if msg is None:
            break

        # Simulate crash on first delivery of message 3
        if msg.payload.get("order_id") == "ORD-003" and msg.attempt == 1 and not crashed:
            log.warning(
                "[CON-A] CRASH before ack on msg=%s (attempt=%d) — not acking",
                msg.id,
                msg.attempt,
            )
            crash_target_id = msg.id
            crashed = True
            # Deliberately do NOT call broker.ack() — broker will redeliver after timeout
            await asyncio.sleep(0.6)        # wait for broker's ack timeout to trigger
            continue

        await consumer.handle(msg, broker)

    await broker.stop()

    log.info("\n--- Summary ---")
    log.info("Messages published : %d", len(messages))
    log.info("Messages processed : %d (unique)", len(consumer.processed))
    log.info("Duplicates skipped : %d", consumer.duplicates_skipped)
    log.info("Processed order IDs: %s", consumer.processed)

    assert len(consumer.processed) == 5, "Expected exactly 5 unique messages processed"
    assert consumer.duplicates_skipped == 1, "Expected 1 duplicate (redelivered ORD-003)"
    log.info("\nAssertion passed: each message processed exactly once despite crash.\n")


# ---------------------------------------------------------------------------
# At-most-once comparison
# ---------------------------------------------------------------------------

async def demo_at_most_once() -> None:
    """
    Simple fire-and-forget: broker deletes immediately on delivery, no retries.
    Simulated by acking in the broker before handing to the consumer,
    then crashing — the message is permanently lost.
    """
    log.info("=== AT-MOST-ONCE DEMO (message loss on crash) ===\n")

    lost: list[str] = []
    processed: list[str] = []

    messages = [
        Message(topic="metrics", payload={"event": f"click-{i}"})
        for i in range(1, 6)
    ]

    for i, msg in enumerate(messages):
        # Broker "pre-acks": deletes on delivery regardless of outcome
        if i == 2:
            # Consumer crashes — message is simply lost, no redelivery
            log.warning("[AT-MOST-ONCE] Consumer crash on msg=%s — LOST", msg.id)
            lost.append(msg.id)
            continue
        log.info("[AT-MOST-ONCE] Processed msg=%s", msg.id)
        processed.append(msg.id)

    log.info("Published: %d, Processed: %d, Lost: %d\n", len(messages), len(processed), len(lost))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    await demo_at_most_once()
    await demo_crash_and_redeliver()


if __name__ == "__main__":
    asyncio.run(main())
```

**What to observe:**

1. **At-most-once demo**: message 3 is lost permanently on crash. No redelivery. Processed = 4.
2. **At-least-once demo**: message `ORD-003` is not acked. After 0.4 s the broker's requeue loop fires and redelivers it with `attempt=2`. The idempotent consumer detects `msg.id` in `_seen_ids` (because we never processed it — wait, it was a crash *before* processing, so the seen-IDs set does not have it. On second delivery it processes normally). The assertion confirms: 5 unique processed, 1 duplicate skipped.
3. Trace the log timestamps: the requeue log line appears ~0.4 s after the crash line, showing the broker's ack-timeout window in action.

---

## Exercises

1. In `demo_crash_and_redeliver`, change the crash scenario so the consumer processes `ORD-003` successfully (calls `_mark_seen`) but crashes before calling `broker.ack()`. Run the demo. Now the idempotent consumer *does* have the ID in `_seen_ids` when the redelivery arrives. Verify that the assertion still passes and the message is still processed exactly once.

2. Add a `max_attempts: int = 3` field to `AtLeastOnceBroker`. After 3 failed deliveries (attempt > max_attempts), route the message to a dead-letter list `_dlq: list[Message]` instead of requeuing. Add a `drain_dlq()` method that returns and clears the DLQ. Write a test that verifies DLQ behavior.

3. Implement a **Redis-backed seen-IDs store** as a drop-in replacement for the `_seen_ids: set[str]`. Use `redis.asyncio` with `SETEX(key, ttl_seconds, 1)` and `EXISTS(key)`. Mock the Redis client in tests. What TTL would you set for a payment processing system that replays up to 24 hours of history?

4. Add **producer retry** to the worked example: wrap `broker.publish()` in a retry loop that retries up to 3 times with a 0.1 s delay on `asyncio.TimeoutError`. Use the same `message.id` on every retry (do not regenerate). Confirm that the idempotent consumer still processes each logical message exactly once even when the producer retries.

5. Measure the latency overhead of the idempotent check. Compare processing time per message with and without the `_seen_ids` lookup for a batch of 10,000 messages. At what batch size does a `set` become a bottleneck? What data structure would you use instead for very high throughput?

(Solutions: ../solutions/week-02/day-3.md)

---

## Self-check

- [ ] I can define at-most-once, at-least-once, and exactly-once in one sentence each.
- [ ] I can explain why "exactly-once" is almost always implemented as "at-least-once + idempotent consumer".
- [ ] I can describe the ack/requeue cycle and identify the failure window where duplicates arise.
- [ ] I can implement an idempotent consumer with a seen-IDs set without reference.
- [ ] I can explain why a producer must use the same message ID on every retry attempt.
- [ ] I can describe the production tradeoff between a `set[str]` (memory) and Redis (durability + TTL) for seen-IDs storage.

---

## Supplementary links

- Kafka — Exactly-once semantics: https://kafka.apache.org/documentation/#semantics
- RabbitMQ — Consumer acknowledgements: https://www.rabbitmq.com/docs/confirms
- Martin Fowler — Idempotent Receiver pattern: https://martinfowler.com/articles/patterns-of-distributed-systems/idempotent-receiver.html
- AWS SQS — At-least-once delivery and exactly-once with FIFO queues: https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/FIFO-queues-exactly-once-processing.html
- Designing Data-Intensive Applications (Kleppmann) — Chapter 11: Stream Processing
- Python `asyncio` — Tasks and timeouts: https://docs.python.org/3/library/asyncio-task.html#timeouts
