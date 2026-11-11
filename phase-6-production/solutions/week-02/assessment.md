# Phase 6 · Week 02 · Assessment — Answer Key

> Full answer key for graders and self-assessment. All code is runnable; no external services required.

---

## Part 1: Conceptual questions — detailed answers

### Q1. When would you choose async over sync communication between services?

Choose **async** when:
- The caller does not need an immediate response (fire and forget — order placed, email sent later).
- You want to decouple availability: the producer continues publishing even if the consumer is down; messages accumulate in the queue until the consumer recovers.
- The workload is bursty and you need to absorb spikes without overloading downstream services (queue acts as a buffer).
- Multiple independent downstream services need to react to the same event (fan-out): e.g. an `order_placed` event triggers inventory, notifications, and analytics independently.
- You need durable event replay: events can be re-read from a Kafka topic for backfills, auditing, or new consumers catching up.

Choose **sync** (HTTP/gRPC) when:
- The caller needs an immediate result before proceeding — e.g. a payment authorisation that must succeed before the user sees a confirmation page.
- You need strong consistency across two systems in one request.
- The operation is user-interactive with a latency SLA under a few hundred milliseconds.

Rule of thumb: if the sentence "let me know when it's done" applies, use async. If "tell me now or the request fails" applies, use sync.

---

### Q2. What is the difference between a queue and pub/sub? Give a concrete example.

**Queue (point-to-point / work queue):** One message, one consumer. The broker assigns each message to exactly one consumer from a pool. Used for load-balanced work distribution.

Example: `payment_jobs` queue consumed by a pool of 5 payment-processor workers. Each payment is processed exactly once; adding workers increases throughput linearly.

**Pub/sub (publish-subscribe / fan-out):** One message, many independent subscribers — each subscriber receives its own copy.

Example: `order_placed` topic subscribed to by an `inventory` service, a `notifications` service, and an `analytics` service. All three receive every event; they operate independently and know nothing about each other.

**Kafka supports both patterns simultaneously:**
- Consumer group = queue semantics: messages split across group members, each partition owned by exactly one member.
- Independent groups on same topic = pub/sub: each group gets all messages. Kafka is therefore more accurately described as a hybrid log-based broker rather than purely one or the other.

---

### Q3. Explain at-least-once delivery semantics and how you make a consumer idempotent.

**At-least-once:** The broker guarantees the message is delivered at least once but may redeliver on consumer failure or network interruption before the consumer has committed its offset (or acknowledged the message). The consumer may therefore process the same message more than once.

**Making a consumer idempotent — three strategies:**

1. **Natural deduplication key + fast lookup.** Use a field that uniquely identifies the event's effect (e.g. `order_id`, `event_id`). Store processed keys in a Redis SET (`SETNX order_id 1 EX 86400`) or DB unique constraint. Before processing, check: if present → skip; if absent → process then mark.

2. **Idempotent operations.** Design the operation itself to be safe to repeat: `INSERT … ON CONFLICT DO NOTHING`, state-machine guard (`if order.status != "placed": return`), `PUT` semantics instead of `POST`.

3. **Transactional outbox.** Combine the business DB write and the offset commit in a single transaction (Kafka Streams, or an outbox table polled by a relay). Ensures exactly-once effect even across crashes.

The `_seen_order_ids: set[str]` used in `InventoryConsumer` is a simple in-memory implementation of strategy 1. In production, replace with Redis for durability and cross-replica consistency.

---

### Q4. What is a Kafka partition, and why does partition count matter?

A **partition** is an ordered, immutable, append-only log segment. Each topic is divided into N partitions stored across broker nodes. Within a partition, messages have a monotonically increasing offset and are consumed in order.

**Why partition count matters:**

1. **Parallelism cap.** A consumer group can have at most one active consumer per partition. 10 partitions → max 10 parallel consumers. Add an 11th consumer and it sits idle until a rebalance.

2. **Throughput ceiling.** Single-partition throughput is bounded by one broker's disk I/O. More partitions → higher aggregate throughput.

3. **Ordering scope.** Order is guaranteed only within a partition. Messages with the same partition key (e.g. `customer_id`) go to the same partition, preserving per-customer ordering. Global ordering requires a single partition, sacrificing parallelism.

4. **Operational cost.** Each partition is a file handle, a replication stream, and a factor in leader election. Too many partitions increases broker memory, rebalance latency, and ZooKeeper/KRaft metadata overhead.

Sizing heuristic: `partition_count = target_throughput_MB_s ÷ single_partition_throughput_MB_s`, then round up to the next convenient number (powers of 2 or multiples of replica count).

---

### Q5. What is a consumer group, and how does it enable horizontal scaling?

A **consumer group** is a named set of consumer instances that cooperatively consume a topic. Kafka assigns each partition to exactly one member of the group at a time (partition ownership). When a member joins or leaves, Kafka triggers a **rebalance** to redistribute partitions.

**Horizontal scaling:** Adding consumers (up to partition count) linearly increases throughput — each new consumer takes ownership of one or more partitions. There is no central coordinator bottleneck; Kafka handles assignment automatically.

**Multiple groups:** Independent groups each receive all messages (pub/sub fan-out). This means the same topic can simultaneously serve both a work-queue model (one group with multiple workers) and fan-out (multiple independent groups each seeing everything).

Example: topic `orders` with 12 partitions.
- Group `inventory-service`: 4 consumers → each owns 3 partitions.
- Group `notification-service`: 2 consumers → each owns 6 partitions.
- Both groups operate independently; adding consumers to one group does not affect the other.

---

### Q6. How does Kafka maintain ordering within a topic?

Kafka guarantees ordering **only within a single partition**. All messages written to the same partition are delivered to consumers in the order they were produced.

**Partition routing:** The producer chooses a partition via:
1. Explicit `key`: messages with the same key always go to the same partition (hash(key) % N). Example: `order_id` as key ensures all events for one order are ordered and processed sequentially.
2. Round-robin (no key): messages distributed across partitions for load balancing; no ordering guarantee.
3. Custom partitioner: application-defined routing logic.

**Producer ordering under retries:** Enable `enable.idempotence=true` on the producer. Combined with `max.in.flight.requests.per.connection ≤ 5`, this ensures retries do not reorder messages within a partition.

**Cross-partition ordering:** There is no global ordering across partitions. If you need strict total ordering for a topic, use a single partition — but this is a single-threaded bottleneck and must be a deliberate architectural decision.

---

### Q7. When would you choose RabbitMQ over Kafka?

| Dimension | RabbitMQ | Kafka |
|---|---|---|
| Routing | Rich: topic/direct/fanout/headers exchanges, per-message routing keys | Simple: topic only; routing via partitions |
| Message lifetime | Deleted after ACK (push model, broker owns lifecycle) | Retained for configured period; consumers manage offsets |
| Replay | Not built-in; messages gone after consume | Native replay: rewind offset to any point |
| Ordering | Per-queue FIFO; priority queues built-in | Per-partition ordered |
| RPC pattern | Native (reply-to header, correlation-id) | Awkward |
| Scale | Excellent for ≤ tens of thousands msg/s | Millions msg/s at scale |
| Ops complexity | Simpler for small teams; management UI built-in | Higher (ZooKeeper/KRaft, partition management) |

**Choose RabbitMQ when:** complex routing logic, per-message TTL and priority, request/reply RPC, or when simplicity matters more than throughput.

**Choose Kafka when:** high-throughput event streaming, long retention and event replay, event sourcing, ordered per-key processing, or integration into a data-lake pipeline.

Rule of thumb: RabbitMQ for task queues with routing logic; Kafka for event logs and data pipelines.

---

### Q8. What triggers a message to be routed to the DLQ?

A message is routed to the dead-letter queue (DLQ) when it cannot be processed successfully after a configured maximum number of retries. Triggers include:

1. **Consumer exception on every attempt.** Any unhandled exception causes retry. After `max_retries` failures the message is dead-lettered.
2. **Schema / validation error (poison pill).** A message that fails Pydantic validation every time. Will never succeed; must be DLQ'd, inspected, and the root-cause corrected (bad producer, schema drift).
3. **Deserialization error.** Corrupt bytes or wrong encoding.
4. **Downstream dependency permanently unavailable.** Database down, external API returning 500 on every attempt.
5. **Message TTL expired** (RabbitMQ feature). Message lived longer than its time-to-live before being consumed.

**What the DLQ preserves:** original payload + metadata (retry count, last error message, original topic name, timestamp). This enables:
- Manual inspection to identify the root cause.
- Replay after the fix is deployed (`dlq_replay`).
- Alerting: non-empty DLQ is a production signal requiring investigation.

Without a DLQ, a single poison pill can block an entire queue indefinitely (head-of-line blocking), preventing all subsequent messages from being processed.

---

## Part 2: Coding challenge — complete solution

### Full implementation

```python
"""
assessment_solution.py
Event-driven notification service demonstrating:
- AbstractBroker / InMemoryBroker
- UserSignupEvent Pydantic v2 model
- SignupProducer
- WelcomeEmailConsumer (idempotent by user_id)
- AuditConsumer
- DLQ with max_retries=3
- No external dependencies beyond pydantic

Run directly: python assessment_solution.py
Test: pytest test_assessment_solution.py -v
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class UserSignupEvent(BaseModel):
    event_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    user_id: str
    email: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# Broker data structures
# ---------------------------------------------------------------------------

@dataclass
class Message:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    payload: Any = None
    headers: dict[str, str] = field(default_factory=dict)
    attempt: int = 0


@dataclass
class DeadLetterRecord:
    message: Message
    topic: str
    last_error: str
    total_attempts: int

    @property
    def payload(self) -> Any:
        return self.message.payload


# ---------------------------------------------------------------------------
# AbstractBroker
# ---------------------------------------------------------------------------

class AbstractBroker(ABC):
    @abstractmethod
    def publish(self, topic: str, payload: Any) -> None:
        """Publish payload to topic."""

    @abstractmethod
    def subscribe(self, topic: str, handler: Callable[..., None]) -> None:
        """Register a handler. Multiple handlers = fanout."""

    def dlq_depth(self, topic: str) -> int:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# InMemoryBroker
# ---------------------------------------------------------------------------

class InMemoryBroker(AbstractBroker):
    """
    Synchronous, thread-safe, in-process broker.
    Retries failing handlers up to max_retries, then routes to DLQ.
    """

    def __init__(self, max_retries: int = 3) -> None:
        self.max_retries = max_retries
        self._lock = threading.Lock()
        self._queues: dict[str, list[Message]] = {}
        self._handlers: dict[str, list[Callable[..., None]]] = {}
        self._dlq: dict[str, list[DeadLetterRecord]] = {}

    def publish(self, topic: str, payload: Any) -> None:
        msg = Message(payload=payload)
        with self._lock:
            self._queues.setdefault(topic, []).append(msg)
        log.debug("publish topic=%s msg_id=%s", topic, msg.id)

    def subscribe(self, topic: str, handler: Callable[..., None]) -> None:
        with self._lock:
            self._handlers.setdefault(topic, []).append(handler)

    def deliver_all(self, topic: str) -> int:
        """
        Process all messages queued on `topic`.
        Returns count of successfully delivered messages.
        This is the consume_all equivalent — named differently per assessment spec.
        """
        processed = 0
        while True:
            with self._lock:
                queue = self._queues.get(topic, [])
                if not queue:
                    break
                msg = queue.pop(0)
                handlers = list(self._handlers.get(topic, []))

            all_ok = True
            last_error = ""
            for handler in handlers:
                try:
                    handler(msg)
                except Exception as exc:  # noqa: BLE001
                    all_ok = False
                    last_error = str(exc)
                    log.warning(
                        "handler %s failed msg_id=%s attempt=%d: %s",
                        handler.__qualname__,
                        msg.id,
                        msg.attempt + 1,
                        exc,
                    )
                    break

            if all_ok:
                processed += 1
            else:
                msg.attempt += 1
                if msg.attempt >= self.max_retries:
                    self._dlq_route(msg, topic, last_error)
                else:
                    with self._lock:
                        self._queues.setdefault(topic, []).append(msg)

        return processed

    def dlq_depth(self, topic: str) -> int:
        with self._lock:
            return len(self._dlq.get(topic, []))

    def dlq_inspect(self, topic: str) -> list[DeadLetterRecord]:
        with self._lock:
            return list(self._dlq.get(topic, []))

    def _dlq_route(self, msg: Message, topic: str, last_error: str) -> None:
        record = DeadLetterRecord(
            message=msg,
            topic=topic,
            last_error=last_error,
            total_attempts=msg.attempt,
        )
        dlq_topic = f"dlq.{topic}"
        with self._lock:
            self._dlq.setdefault(dlq_topic, []).append(record)
        log.error(
            "DLQ topic=%s → dlq.%s msg_id=%s attempts=%d",
            topic,
            topic,
            msg.id,
            msg.attempt,
        )

    def dlq_depth(self, topic: str) -> int:  # type: ignore[override]
        """topic should be the DLQ topic name, e.g. 'dlq.user.signup'."""
        with self._lock:
            return len(self._dlq.get(topic, []))


# ---------------------------------------------------------------------------
# SignupProducer
# ---------------------------------------------------------------------------

class SignupProducer:
    def __init__(self, broker: AbstractBroker) -> None:
        self._broker = broker

    def publish(self, event: UserSignupEvent) -> None:
        self._broker.publish("user.signup", event.model_dump(mode="json"))
        log.info("SignupProducer published user_id=%s email=%s", event.user_id, event.email)


# ---------------------------------------------------------------------------
# WelcomeEmailConsumer (idempotent by user_id)
# ---------------------------------------------------------------------------

class WelcomeEmailConsumer:
    """
    Subscribes to 'user.signup'. Sends a welcome email (mocked via log).
    Idempotent: processes each user_id exactly once.

    Production note: replace _seen_user_ids with Redis SETNX for
    distributed deduplication across replicas.
    """

    def __init__(self, broker: AbstractBroker) -> None:
        self._broker = broker
        self._seen_user_ids: set[str] = set()
        broker.subscribe("user.signup", self._handle)

    def _is_duplicate(self, user_id: str) -> bool:
        if user_id in self._seen_user_ids:
            log.warning("WelcomeEmail: duplicate user_id=%s — skipping", user_id)
            return True
        self._seen_user_ids.add(user_id)
        return False

    def _handle(self, msg: object) -> None:
        assert isinstance(msg, Message)
        event = UserSignupEvent.model_validate(msg.payload)
        if self._is_duplicate(event.user_id):
            return
        self._send_welcome_email(event)

    def _send_welcome_email(self, event: UserSignupEvent) -> None:
        """Mock email dispatch — logs only."""
        log.info("WELCOME_EMAIL to=%s user_id=%s", event.email, event.user_id)


# ---------------------------------------------------------------------------
# AuditConsumer
# ---------------------------------------------------------------------------

class AuditConsumer:
    """
    Subscribes to 'user.signup'. Records every event to an audit log.
    Not idempotent — audits every delivery (including duplicates, by design).
    """

    def __init__(self, broker: AbstractBroker) -> None:
        self._broker = broker
        self.audit_log: list[UserSignupEvent] = []
        broker.subscribe("user.signup", self._handle)

    def _handle(self, msg: object) -> None:
        assert isinstance(msg, Message)
        event = UserSignupEvent.model_validate(msg.payload)
        self.audit_log.append(event)
        log.info("AUDIT recorded event_id=%s user_id=%s", event.event_id, event.user_id)


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------

def run_demo() -> None:
    log.info("=== Notification Service Demo ===")

    broker = InMemoryBroker(max_retries=3)
    producer = SignupProducer(broker)
    welcome = WelcomeEmailConsumer(broker)
    audit = AuditConsumer(broker)

    users = [
        UserSignupEvent(user_id="alice", email="alice@example.com"),
        UserSignupEvent(user_id="bob", email="bob@example.com"),
        UserSignupEvent(user_id="alice", email="alice@example.com"),  # duplicate
    ]

    for u in users:
        producer.publish(u)

    broker.deliver_all("user.signup")

    log.info("Welcome emails sent: %d (expected 2 unique)", len(welcome._seen_user_ids))
    log.info("Audit log entries: %d (expected 2 — idempotent handler saw duplicate)", len(audit.audit_log))
    log.info("DLQ depth: %d", broker.dlq_depth("dlq.user.signup"))


if __name__ == "__main__":
    run_demo()
```

---

### Test suite

```python
"""
test_assessment_solution.py
pytest suite for the assessment coding challenge.

Run: pytest test_assessment_solution.py -v
"""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest

from assessment_solution import (
    AuditConsumer,
    InMemoryBroker,
    Message,
    SignupProducer,
    UserSignupEvent,
    WelcomeEmailConsumer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _event(user_id: str = "user-1", email: str = "u@example.com") -> UserSignupEvent:
    return UserSignupEvent(user_id=user_id, email=email)


def _broker(max_retries: int = 3) -> InMemoryBroker:
    return InMemoryBroker(max_retries=max_retries)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

def test_user_signup_event_has_required_fields() -> None:
    e = _event()
    assert isinstance(e.event_id, uuid.UUID)
    assert e.user_id == "user-1"
    assert e.email == "u@example.com"
    assert e.timestamp is not None


# ---------------------------------------------------------------------------
# Fan-out: both consumers receive the same event
# ---------------------------------------------------------------------------

def test_fanout_both_consumers_receive_event() -> None:
    """Publishing one event triggers both WelcomeEmailConsumer and AuditConsumer."""
    b = _broker()
    producer = SignupProducer(b)
    welcome = WelcomeEmailConsumer(b)
    audit = AuditConsumer(b)

    welcome_received: list[UserSignupEvent] = []
    original_send = welcome._send_welcome_email
    welcome._send_welcome_email = lambda e: (welcome_received.append(e), original_send(e))[1]  # type: ignore[method-assign]

    producer.publish(_event("alice", "alice@example.com"))
    b.deliver_all("user.signup")

    assert len(welcome_received) == 1
    assert welcome_received[0].user_id == "alice"
    assert len(audit.audit_log) == 1
    assert audit.audit_log[0].user_id == "alice"


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def test_welcome_email_idempotent_same_user_id() -> None:
    """Publishing same user_id twice results in exactly one welcome email."""
    b = _broker()
    producer = SignupProducer(b)
    welcome = WelcomeEmailConsumer(b)

    emails_sent: list[UserSignupEvent] = []
    with patch.object(welcome, "_send_welcome_email", side_effect=emails_sent.append):
        producer.publish(_event("alice"))
        producer.publish(_event("alice"))  # duplicate user_id
        b.deliver_all("user.signup")

    assert len(emails_sent) == 1


def test_welcome_email_different_user_ids_both_sent() -> None:
    """Two different user_ids → two welcome emails."""
    b = _broker()
    producer = SignupProducer(b)
    welcome = WelcomeEmailConsumer(b)

    emails_sent: list[UserSignupEvent] = []
    with patch.object(welcome, "_send_welcome_email", side_effect=emails_sent.append):
        producer.publish(_event("alice", "alice@example.com"))
        producer.publish(_event("bob", "bob@example.com"))
        b.deliver_all("user.signup")

    assert len(emails_sent) == 2
    sent_ids = {e.user_id for e in emails_sent}
    assert sent_ids == {"alice", "bob"}


# ---------------------------------------------------------------------------
# DLQ routing
# ---------------------------------------------------------------------------

def test_dlq_routing_after_max_retries() -> None:
    """A consumer that always raises is retried 3× then event appears on DLQ."""
    b = InMemoryBroker(max_retries=3)
    call_count = 0

    def always_fails(msg: Message) -> None:
        nonlocal call_count
        call_count += 1
        raise RuntimeError("Simulated failure")

    b.subscribe("user.signup", always_fails)
    b.publish("user.signup", _event().model_dump(mode="json"))

    for _ in range(10):
        b.deliver_all("user.signup")

    assert call_count == 3, f"Expected exactly 3 attempts, got {call_count}"
    assert b.dlq_depth("dlq.user.signup") == 1


def test_dlq_preserves_original_payload() -> None:
    """DLQ record contains the original payload."""
    b = InMemoryBroker(max_retries=2)
    b.subscribe("user.signup", lambda m: (_ for _ in ()).throw(RuntimeError("fail")))

    event = _event("preserved-user")
    b.publish("user.signup", event.model_dump(mode="json"))

    for _ in range(5):
        b.deliver_all("user.signup")

    records = b.dlq_inspect("dlq.user.signup")
    assert len(records) == 1
    assert records[0].payload["user_id"] == "preserved-user"


def test_dlq_empty_on_successful_delivery() -> None:
    """No DLQ entries when handler succeeds."""
    b = _broker()
    b.subscribe("user.signup", lambda m: None)
    b.publish("user.signup", _event().model_dump(mode="json"))
    b.deliver_all("user.signup")
    assert b.dlq_depth("dlq.user.signup") == 0


def test_retry_then_success_no_dlq() -> None:
    """Handler fails twice, succeeds on third → DLQ stays empty."""
    b = InMemoryBroker(max_retries=3)
    attempts = 0

    def sometimes_fails(msg: Message) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("transient")

    b.subscribe("user.signup", sometimes_fails)
    b.publish("user.signup", _event().model_dump(mode="json"))

    for _ in range(5):
        b.deliver_all("user.signup")

    assert attempts == 3
    assert b.dlq_depth("dlq.user.signup") == 0


# ---------------------------------------------------------------------------
# Integration: full flow
# ---------------------------------------------------------------------------

def test_full_notification_service_flow() -> None:
    """End-to-end: publish 3 events (1 duplicate), verify audit and welcome counts."""
    b = _broker()
    producer = SignupProducer(b)
    welcome = WelcomeEmailConsumer(b)
    audit = AuditConsumer(b)

    emails_sent: list[UserSignupEvent] = []
    with patch.object(welcome, "_send_welcome_email", side_effect=emails_sent.append):
        producer.publish(_event("alice", "alice@example.com"))
        producer.publish(_event("bob", "bob@example.com"))
        producer.publish(_event("alice", "alice@example.com"))  # duplicate
        b.deliver_all("user.signup")

    # 3 events published, 2 unique users → 2 welcome emails
    assert len(emails_sent) == 2
    # AuditConsumer records unique deliveries (broker deduplicates at handler level
    # only for WelcomeEmail; duplicate message was still delivered to AuditConsumer)
    assert len(audit.audit_log) == 3
    # No DLQ entries
    assert b.dlq_depth("dlq.user.signup") == 0
```

---

## Part 3: Grading guide

### Total: 100 points — Pass threshold: 75 points

---

### Dimension 1: Independent implementation (40 points)

| Criterion | Points | Notes |
|---|---|---|
| `UserSignupEvent` Pydantic v2 model with all 4 fields (`event_id: UUID`, `user_id: str`, `email: str`, `timestamp: datetime`) | 5 | Deduct 1 per missing field |
| `AbstractBroker` ABC with `publish` and `subscribe` abstract methods | 5 | Must be ABC with `@abstractmethod` |
| `InMemoryBroker` concrete implementation with fanout (multiple handlers per topic) | 8 | Multiple handlers must ALL receive message |
| `InMemoryBroker` retry logic: max_retries with attempt counter | 7 | Off-by-one counts as partial credit (3 pts) |
| `InMemoryBroker` DLQ routing on max_retries exceeded | 5 | DLQ must preserve original payload |
| `SignupProducer` publishes serialised event | 4 | Must use `model_dump(mode="json")` |
| `WelcomeEmailConsumer` subscribes and sends notification | 3 | |
| `AuditConsumer` subscribes and records to audit log | 3 | |

**Penalties:** −5 for `print()` instead of `logging`. −3 for hardcoded secrets (none expected in this challenge, but flag `os.getenv` usage). −2 for missing type hints on function signatures.

---

### Dimension 2: Conceptual understanding (25 points)

Score each Q1–Q8 answer independently:

| Per-question rubric | Points |
|---|---|
| Fully correct with concrete example or trade-off | 3 |
| Mostly correct, minor gap (no example, or imprecise) | 2 |
| Partially correct (right direction, missing key distinction) | 1 |
| Incorrect or missing | 0 |

Maximum 24 points from 8 × 3. +1 bonus if all 8 are fully correct.

**Common gaps to watch for:**
- Q3: must name Redis or DB unique constraint as production alternative, not just describe in-memory set.
- Q4: must mention partition count = max consumer parallelism, not just "more partitions = more throughput."
- Q6: must state ordering guaranteed only within partition, and explain partition key routing.
- Q8: must mention "poison pill" as a concrete trigger and explain that DLQ prevents head-of-line blocking.

---

### Dimension 3: Debugging & problem-solving (20 points)

Assessed by reading the submitted code for evidence of anticipating and handling failure modes.

| Criterion | Points |
|---|---|
| Re-raises `ValidationError` in consumer (or equivalent) so broker can DLQ bad payloads | 5 |
| Uses `model_dump(mode="json")` when publishing (not raw Pydantic object) | 5 |
| `_is_duplicate` check-and-mark is atomic (not check then mark separately after processing) | 5 |
| Retry counter starts at 0 and condition is `>= max_retries` (not `> max_retries`) | 5 |

---

### Dimension 4: Code quality & testing (15 points)

| Criterion | Points |
|---|---|
| pytest suite present with at least 3 passing tests | 3 |
| Fan-out test: both consumers receive the same event | 3 |
| Idempotency test: same user_id twice → exactly one welcome email | 3 |
| DLQ routing test: always-failing handler retried N times then DLQ'd | 3 |
| `logging` used throughout, no `print()` | 2 |
| Type hints on all function signatures | 1 |

**Bonus (+5, not capped):** DLQ test also verifies `total_attempts == max_retries` and original payload preserved.

---

### Grade boundaries

| Score | Grade |
|---|---|
| 90–100 | Distinction — production-ready understanding |
| 75–89 | Pass — solid grasp, minor gaps |
| 60–74 | Marginal fail — revisit retry/DLQ mechanics and idempotency |
| < 60 | Fail — revisit Week 2 from Day 1 |
