# Phase 6 · Week 02 · Day 7 — Solutions & Reference Implementations

> Complete reference solutions for the Saturday build session. Read the exercise hints in `week-02/day-7.md` first; come here when stuck or to verify your implementation.

---

## Part 1: Design review answers

1. **Why does `InMemoryBroker` need `max_retries` at broker level?** Retry policy is an infrastructure concern, not a business concern. If each consumer set its own, retry counts would diverge and DLQ routing would be inconsistent. Centralising it means every topic gets the same guarantee, and the policy can be changed in one place.

2. **Two handlers on same topic — both receive?** Yes. This is the **fan-out** (pub/sub) delivery model. The broker iterates all registered handlers and calls each one per message.

3. **Where should deduplication state live?** In the consumer. The broker must stay stateless about business semantics. The consumer holds `_seen_order_ids: set[str]`; in production this moves to Redis or a DB unique constraint.

4. **Idempotency for notifications:** If the broker retries the notification event, `_seen_order_ids` in `NotificationConsumer` detects the duplicate `order_id` and returns early without calling `_send_notification` again. The user receives exactly one email.

---

## Complete Implementation

### `src/order_pipeline/broker.py`

```python
"""
src/order_pipeline/broker.py
AbstractBroker ABC + InMemoryBroker (thread-safe, DLQ, pub/sub + work-queue modes).
KafkaBroker stub for conceptual scaffolding only.
"""

from __future__ import annotations

import logging
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
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


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class AbstractBroker(ABC):
    """Interface all broker implementations must satisfy."""

    @abstractmethod
    def publish(
        self,
        topic: str,
        payload: Any,
        headers: dict[str, str] | None = None,
    ) -> str:
        """Publish a message. Returns message id."""

    @abstractmethod
    def subscribe(self, topic: str, handler: Callable[..., None]) -> None:
        """Register a handler for topic. Multiple handlers = fanout."""

    @abstractmethod
    def consume_all(self, topic: str) -> int:
        """Process all queued messages for topic. Returns count processed."""

    def dlq_depth(self, topic: str) -> int:  # noqa: ARG002
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement dlq_depth. "
            "Override this method in a concrete subclass."
        )


# ---------------------------------------------------------------------------
# In-memory implementation
# ---------------------------------------------------------------------------

class InMemoryBroker(AbstractBroker):
    """
    Thread-safe, synchronous in-memory broker.

    Supports:
    - pub/sub fan-out (multiple handlers per topic)
    - At-least-once delivery with configurable max_retries
    - Per-topic DLQ with inspect and replay
    - Optional headers on every message (for tracing correlation IDs)
    """

    def __init__(self, max_retries: int = 3) -> None:
        self.max_retries = max_retries
        self._lock = threading.Lock()
        # Per-topic FIFO queues
        self._queues: dict[str, list[Message]] = {}
        # Per-topic handler lists (fanout)
        self._handlers: dict[str, list[Callable[..., None]]] = {}
        # Per-topic dead-letter store
        self._dlq: dict[str, list[DeadLetterRecord]] = {}

    # ------------------------------------------------------------------
    # AbstractBroker implementation
    # ------------------------------------------------------------------

    def publish(
        self,
        topic: str,
        payload: Any,
        headers: dict[str, str] | None = None,
    ) -> str:
        msg = Message(payload=payload, headers=headers or {})
        with self._lock:
            self._queues.setdefault(topic, []).append(msg)
        log.debug("Broker.publish topic=%s msg_id=%s", topic, msg.id)
        return msg.id

    def subscribe(self, topic: str, handler: Callable[..., None]) -> None:
        with self._lock:
            self._handlers.setdefault(topic, []).append(handler)
        log.debug("Broker.subscribe topic=%s handler=%s", topic, handler.__qualname__)

    def consume_all(self, topic: str) -> int:
        """
        Drain the queue for `topic`, calling all registered handlers per message.
        On handler exception: increments attempt count and re-queues until max_retries,
        then routes to DLQ.
        Returns the total number of messages that were fully processed (not DLQ'd).
        """
        processed = 0
        while True:
            with self._lock:
                queue = self._queues.get(topic, [])
                if not queue:
                    break
                msg = queue.pop(0)
                handlers = list(self._handlers.get(topic, []))

            # Call every handler (fanout). One failure re-queues/DLQs.
            all_ok = True
            last_error = ""
            for handler in handlers:
                try:
                    handler(msg)
                except Exception as exc:  # noqa: BLE001
                    all_ok = False
                    last_error = str(exc)
                    log.warning(
                        "Handler %s failed on msg_id=%s attempt=%d: %s",
                        handler.__qualname__,
                        msg.id,
                        msg.attempt + 1,
                        exc,
                    )
                    break  # stop calling remaining handlers; re-queue message

            if all_ok:
                processed += 1
            else:
                msg.attempt += 1
                if msg.attempt >= self.max_retries:
                    self._route_to_dlq(msg, topic, last_error)
                else:
                    with self._lock:
                        # Re-enqueue at the back
                        self._queues.setdefault(topic, []).append(msg)
                    log.info(
                        "Re-queued msg_id=%s attempt=%d/%d topic=%s",
                        msg.id,
                        msg.attempt,
                        self.max_retries,
                        topic,
                    )

        return processed

    # ------------------------------------------------------------------
    # DLQ helpers
    # ------------------------------------------------------------------

    def dlq_depth(self, topic: str) -> int:
        with self._lock:
            return len(self._dlq.get(topic, []))

    def dlq_inspect(self, topic: str) -> list[DeadLetterRecord]:
        """Return a copy of DLQ records for `topic`."""
        with self._lock:
            return list(self._dlq.get(topic, []))

    def dlq_replay(self, topic: str) -> int:
        """
        Re-enqueue all DLQ messages for `topic` back onto the main queue.
        Clears DLQ. Returns count replayed.
        """
        with self._lock:
            records = self._dlq.pop(topic, [])
            for rec in records:
                rec.message.attempt = 0  # reset retry counter
                self._queues.setdefault(topic, []).append(rec.message)
        log.info("DLQ replay topic=%s count=%d", topic, len(records))
        return len(records)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _route_to_dlq(self, msg: Message, topic: str, last_error: str) -> None:
        record = DeadLetterRecord(
            message=msg,
            topic=topic,
            last_error=last_error,
            total_attempts=msg.attempt,
        )
        with self._lock:
            self._dlq.setdefault(topic, []).append(record)
        log.error(
            "DLQ msg_id=%s topic=%s attempts=%d error=%s",
            msg.id,
            topic,
            msg.attempt,
            last_error,
        )


# ---------------------------------------------------------------------------
# Kafka stub — conceptual only, never instantiated in tests
# ---------------------------------------------------------------------------

class KafkaBroker(AbstractBroker):
    """
    Conceptual stub for a real Kafka broker integration.

    In production this would wrap confluent-kafka Producer/Consumer.
    Not instantiated in this course — in-memory only.
    """

    def publish(
        self,
        topic: str,
        payload: Any,
        headers: dict[str, str] | None = None,
    ) -> str:
        # Would: serialize payload to bytes (JSON/Avro/Protobuf),
        #        call producer.produce(topic, value=bytes, key=partition_key,
        #                              headers=list(headers.items())),
        #        call producer.flush() or use delivery-report callbacks.
        raise NotImplementedError("Use InMemoryBroker in this course")

    def subscribe(self, topic: str, handler: Callable[..., None]) -> None:
        # Would: store handler, configure consumer group id (GROUP_ID env var),
        #        call consumer.subscribe([topic]).
        raise NotImplementedError("Use InMemoryBroker in this course")

    def consume_all(self, topic: str) -> int:
        # Would: poll in a loop (consumer.poll(timeout=1.0)),
        #        call handler(msg), commit offset on success,
        #        route to DLQ topic on max retries via a separate producer.
        raise NotImplementedError("Use InMemoryBroker in this course")
```

---

### `src/order_pipeline/models.py`

```python
"""
src/order_pipeline/models.py
Pydantic v2 event models for the order processing pipeline.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class OrderItem(BaseModel):
    sku: str
    quantity: int
    unit_price: float


class OrderEvent(BaseModel):
    order_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    items: list[OrderItem]
    total: float
    status: Literal["pending", "confirmed", "cancelled"] = "pending"
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @model_validator(mode="after")
    def validate_total(self) -> "OrderEvent":
        expected = sum(i.quantity * i.unit_price for i in self.items)
        if abs(self.total - expected) > 0.01:
            raise ValueError(
                f"total={self.total} does not match sum of items={expected:.2f}"
            )
        return self


class InventoryReservedEvent(BaseModel):
    order_id: str
    reserved_items: list[OrderItem]
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class InventoryFailedEvent(BaseModel):
    order_id: str
    reason: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class NotificationEvent(BaseModel):
    order_id: str
    status: Literal["reserved", "failed"]
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class DLQMessage(BaseModel):
    original_topic: str
    payload: dict
    error: str
    attempts: int
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
```

---

### `src/order_pipeline/producer.py`

```python
"""
src/order_pipeline/producer.py
OrderProducer: serialises OrderEvent and publishes to 'orders' topic.
"""

from __future__ import annotations

import logging
import uuid

from order_pipeline.broker import AbstractBroker
from order_pipeline.models import OrderEvent

log = logging.getLogger(__name__)

ORDERS_TOPIC = "orders"


class OrderProducer:
    def __init__(self, broker: AbstractBroker) -> None:
        self._broker = broker

    def publish_order(self, event: OrderEvent) -> str:
        """
        Serialise event and publish to 'orders' topic.
        Attaches a correlation_id header for distributed tracing.
        Returns message id.
        """
        correlation_id = str(uuid.uuid4())
        msg_id = self._broker.publish(
            ORDERS_TOPIC,
            event.model_dump(mode="json"),
            headers={
                "correlation_id": correlation_id,
                "source": "order_producer",
            },
        )
        log.info(
            "Published order_id=%s msg_id=%s correlation_id=%s",
            event.order_id,
            msg_id,
            correlation_id,
        )
        return msg_id
```

---

### `src/order_pipeline/consumers.py`

```python
"""
src/order_pipeline/consumers.py
InventoryConsumer and NotificationConsumer.

Both are idempotent: each maintains _seen_order_ids: set[str].

Production note: replace _seen_order_ids with a Redis SET (SETNX order_id 1 EX 86400)
for distributed idempotency across multiple consumer replicas. The idempotency window
(TTL) is a trade-off between memory/storage and safety against re-processing stale
duplicates.
"""

from __future__ import annotations

import logging
import random

from pydantic import ValidationError

from order_pipeline.broker import AbstractBroker, Message
from order_pipeline.config import settings
from order_pipeline.models import (
    InventoryFailedEvent,
    InventoryReservedEvent,
    OrderEvent,
)

log = logging.getLogger(__name__)


class InventoryConsumer:
    """
    Subscribes to 'orders'. Checks mock inventory.
    Publishes InventoryReservedEvent or InventoryFailedEvent.
    Idempotent: skips duplicate order_ids.
    """

    def __init__(self, broker: AbstractBroker) -> None:
        self._broker = broker
        self._seen_order_ids: set[str] = set()
        broker.subscribe("orders", self._handle)

    # ------------------------------------------------------------------
    # Idempotency helper
    # ------------------------------------------------------------------

    def _is_duplicate(self, order_id: str) -> bool:
        """
        Return True if order_id was already processed.

        Production note: replace self._seen_order_ids with a Redis SET
        (SETNX order_id 1 EX 86400) for distributed idempotency.
        """
        if order_id in self._seen_order_ids:
            log.warning("Duplicate order_id=%s — skipping", order_id)
            return True
        self._seen_order_ids.add(order_id)
        return False

    # ------------------------------------------------------------------
    # Handler
    # ------------------------------------------------------------------

    def _handle(self, msg: object) -> None:
        assert isinstance(msg, Message)
        try:
            event = OrderEvent.model_validate(msg.payload)
        except ValidationError:
            log.exception("Invalid OrderEvent payload msg_id=%s — routing to DLQ", msg.id)  # type: ignore[union-attr]
            raise  # broker will retry then DLQ

        if self._is_duplicate(event.order_id):
            return

        correlation_id = msg.headers.get("correlation_id", "n/a")
        log.info(
            "InventoryConsumer processing order_id=%s correlation_id=%s",
            event.order_id,
            correlation_id,
        )

        if self._check_inventory(event):
            reserved = InventoryReservedEvent(
                order_id=event.order_id,
                reserved_items=event.items,
            )
            self._broker.publish(
                "inventory.reserved",
                reserved.model_dump(mode="json"),
                headers={"correlation_id": correlation_id},
            )
            log.info("Inventory RESERVED order_id=%s", event.order_id)
        else:
            failed = InventoryFailedEvent(
                order_id=event.order_id,
                reason="Insufficient stock",
            )
            self._broker.publish(
                "inventory.failed",
                failed.model_dump(mode="json"),
                headers={"correlation_id": correlation_id},
            )
            log.info("Inventory FAILED order_id=%s", event.order_id)

    def _check_inventory(self, event: OrderEvent) -> bool:  # noqa: ARG002
        """Mock inventory check. Returns True if stock available."""
        return random.random() < settings.inventory_pass_rate


# ---------------------------------------------------------------------------

class NotificationConsumer:
    """
    Subscribes to 'inventory.reserved' AND 'inventory.failed' (fanout).
    Sends a mock notification for each order.
    Idempotent: skips duplicate order_ids.
    """

    def __init__(self, broker: AbstractBroker) -> None:
        self._broker = broker
        self._seen_order_ids: set[str] = set()
        broker.subscribe("inventory.reserved", self._handle_reserved)
        broker.subscribe("inventory.failed", self._handle_failed)

    def _is_duplicate(self, order_id: str) -> bool:
        """
        Return True if order_id was already processed.

        Production note: replace self._seen_order_ids with a Redis SET
        (SETNX order_id 1 EX 86400) for distributed idempotency.
        """
        if order_id in self._seen_order_ids:
            log.warning("Duplicate order_id=%s in NotificationConsumer — skipping", order_id)
            return True
        self._seen_order_ids.add(order_id)
        return False

    def _handle_reserved(self, msg: object) -> None:
        assert isinstance(msg, Message)
        event = InventoryReservedEvent.model_validate(msg.payload)
        if self._is_duplicate(event.order_id):
            return
        self._send_notification(event.order_id, "reserved")

    def _handle_failed(self, msg: object) -> None:
        assert isinstance(msg, Message)
        event = InventoryFailedEvent.model_validate(msg.payload)
        if self._is_duplicate(event.order_id):
            return
        self._send_notification(event.order_id, "failed")

    def _send_notification(self, order_id: str, status: str) -> None:
        """Mock notification — logs only. Replace with real email/SMS in production."""
        log.info("NOTIFY order_id=%s status=%s", order_id, status)
```

---

### `src/order_pipeline/dlq.py`

```python
"""
src/order_pipeline/dlq.py
DLQMonitor: reports DLQ depth per topic (Prometheus/Datadog in production).
DLQReplayer: replays DLQ messages back onto main queues with optional filtering.
"""

from __future__ import annotations

import logging

from order_pipeline.broker import InMemoryBroker

log = logging.getLogger(__name__)


class DLQMonitor:
    """
    Reports DLQ depth per topic.

    In production: push these metrics to Datadog / Prometheus.
    Set an alert if any topic's DLQ depth exceeds a threshold.
    """

    def __init__(self, broker: InMemoryBroker, topics: list[str]) -> None:
        self._broker = broker
        self._topics = topics

    def report(self) -> dict[str, int]:
        return {t: self._broker.dlq_depth(t) for t in self._topics}

    def log_report(self) -> None:
        for topic, depth in self.report().items():
            level = logging.WARNING if depth > 0 else logging.DEBUG
            log.log(level, "DLQ depth topic=%s depth=%d", topic, depth)


class DLQReplayer:
    """
    Replays dead-lettered messages back onto their original topic queues.

    Use after fixing the root cause of repeated failures.
    In production: replay selectively, not blindly — fix the poison pill first.
    """

    def __init__(self, broker: InMemoryBroker) -> None:
        self._broker = broker

    def replay(self, topic: str) -> int:
        """Replay all DLQ messages for `topic`. Returns count replayed."""
        count = self._broker.dlq_replay(topic)
        log.info("DLQReplayer replayed %d messages for topic=%s", count, topic)
        return count

    def replay_all(self, topics: list[str]) -> dict[str, int]:
        return {t: self.replay(t) for t in topics}
```

---

### `src/order_pipeline/config.py`

```python
"""
src/order_pipeline/config.py
Application settings loaded from environment variables.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Broker behaviour
    broker_max_retries: int = 3

    # Inventory mock: fraction of orders that pass the inventory check
    inventory_pass_rate: float = 0.8

    # Log level
    log_level: str = "INFO"


settings = Settings()
```

---

## Complete Test Suite

### `tests/test_broker.py`

```python
"""
tests/test_broker.py
Unit tests for InMemoryBroker: DLQ routing, retry behaviour, fanout.
"""

from __future__ import annotations

import pytest

from order_pipeline.broker import DeadLetterRecord, InMemoryBroker, Message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_broker(max_retries: int = 3) -> InMemoryBroker:
    return InMemoryBroker(max_retries=max_retries)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_successful_delivery_no_dlq() -> None:
    """Handler succeeds on first attempt; DLQ stays empty."""
    broker = _make_broker()
    received: list[Message] = []

    broker.subscribe("test", received.append)
    broker.publish("test", {"value": 1})
    broker.consume_all("test")

    assert len(received) == 1
    assert received[0].payload == {"value": 1}
    assert broker.dlq_depth("test") == 0


def test_retry_then_success() -> None:
    """Handler fails twice, succeeds on third attempt; DLQ stays empty."""
    broker = _make_broker(max_retries=3)
    call_count = 0

    def flaky_handler(msg: Message) -> None:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("Transient error")

    broker.subscribe("test", flaky_handler)
    broker.publish("test", "payload")

    # Need to call consume_all multiple times since re-queued messages
    # get processed in subsequent consume_all sweeps
    for _ in range(5):
        broker.consume_all("test")

    assert call_count == 3
    assert broker.dlq_depth("test") == 0


def test_max_retries_routes_to_dlq() -> None:
    """Handler always fails; after max_retries attempts, message lands in DLQ."""
    broker = _make_broker(max_retries=3)

    def always_fails(msg: Message) -> None:
        raise ValueError("Permanent error")

    broker.subscribe("test", always_fails)
    broker.publish("test", {"key": "value"})

    for _ in range(10):
        broker.consume_all("test")

    assert broker.dlq_depth("test") == 1
    records = broker.dlq_inspect("test")
    assert len(records) == 1
    assert isinstance(records[0], DeadLetterRecord)
    assert records[0].payload == {"key": "value"}
    assert records[0].total_attempts == 3
    assert "Permanent error" in records[0].last_error


def test_dlq_replay_re_enqueues() -> None:
    """After DLQ routing, dlq_replay re-enqueues the message. Swap in a succeeding handler."""
    broker = _make_broker(max_retries=2)
    call_count = 0

    def always_fails(msg: Message) -> None:
        nonlocal call_count
        call_count += 1
        raise RuntimeError("fail")

    broker.subscribe("test", always_fails)
    broker.publish("test", "data")

    for _ in range(5):
        broker.consume_all("test")

    assert broker.dlq_depth("test") == 1

    # Swap handler and replay
    broker._handlers["test"] = []
    received: list[Message] = []
    broker.subscribe("test", received.append)

    replayed = broker.dlq_replay("test")
    assert replayed == 1
    assert broker.dlq_depth("test") == 0

    broker.consume_all("test")
    assert len(received) == 1
    assert received[0].payload == "data"


def test_fanout_multiple_handlers() -> None:
    """Two handlers subscribed to same topic both receive the message."""
    broker = _make_broker()
    results_a: list[Message] = []
    results_b: list[Message] = []

    broker.subscribe("fanout", results_a.append)
    broker.subscribe("fanout", results_b.append)
    broker.publish("fanout", "shared_payload")
    broker.consume_all("fanout")

    assert len(results_a) == 1
    assert len(results_b) == 1
    assert results_a[0].payload == "shared_payload"
    assert results_b[0].payload == "shared_payload"


def test_publish_returns_message_id() -> None:
    """publish() returns a non-empty string message id."""
    broker = _make_broker()
    msg_id = broker.publish("test", "x")
    assert isinstance(msg_id, str) and len(msg_id) > 0


def test_message_headers_stored() -> None:
    """Headers passed to publish are preserved on the Message."""
    broker = _make_broker()
    received: list[Message] = []
    broker.subscribe("hdr", received.append)
    broker.publish("hdr", "data", headers={"x-source": "test"})
    broker.consume_all("hdr")

    assert received[0].headers == {"x-source": "test"}


def test_consume_empty_topic_returns_zero() -> None:
    broker = _make_broker()
    broker.subscribe("empty", lambda m: None)
    count = broker.consume_all("empty")
    assert count == 0


def test_dlq_depth_zero_on_clean_run() -> None:
    broker = _make_broker()
    broker.subscribe("ok", lambda m: None)
    broker.publish("ok", "x")
    broker.consume_all("ok")
    assert broker.dlq_depth("ok") == 0
```

---

### `tests/test_consumers.py`

```python
"""
tests/test_consumers.py
Unit tests for InventoryConsumer and NotificationConsumer.
"""

from __future__ import annotations

from unittest.mock import patch

from order_pipeline.broker import InMemoryBroker, Message
from order_pipeline.consumers import InventoryConsumer, NotificationConsumer
from order_pipeline.models import OrderEvent, OrderItem
from order_pipeline.producer import OrderProducer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_order(order_id: str = "ORD-001") -> OrderEvent:
    items = [OrderItem(sku="SKU-A", quantity=2, unit_price=10.0)]
    return OrderEvent(order_id=order_id, user_id="USR-1", items=items, total=20.0)


def _make_broker() -> InMemoryBroker:
    return InMemoryBroker(max_retries=3)


# ---------------------------------------------------------------------------
# InventoryConsumer tests
# ---------------------------------------------------------------------------

def test_inventory_consumer_idempotent() -> None:
    """
    Publishing same order_id twice does not result in two inventory events.
    'inventory.reserved' should have exactly 1 message.
    """
    broker = _make_broker()
    producer = OrderProducer(broker)
    consumer = InventoryConsumer(broker)

    with patch.object(consumer, "_check_inventory", return_value=True):
        producer.publish_order(_make_order("ORD-DUPE"))
        producer.publish_order(_make_order("ORD-DUPE"))  # duplicate
        broker.consume_all("orders")

    reserved = broker._queues.get("inventory.reserved", [])
    failed = broker._queues.get("inventory.failed", [])
    assert len(reserved) == 1, "Expected exactly 1 reserved event"
    assert len(failed) == 0, "Expected no failed events"


def test_inventory_consumer_routes_reserved() -> None:
    """When _check_inventory returns True, InventoryReservedEvent published."""
    broker = _make_broker()
    producer = OrderProducer(broker)
    consumer = InventoryConsumer(broker)

    with patch.object(consumer, "_check_inventory", return_value=True):
        producer.publish_order(_make_order())
        broker.consume_all("orders")

    assert broker._queues.get("inventory.reserved"), "Expected reserved event"
    assert not broker._queues.get("inventory.failed"), "Should not have failed event"


def test_inventory_consumer_routes_failed() -> None:
    """When _check_inventory returns False, InventoryFailedEvent published."""
    broker = _make_broker()
    producer = OrderProducer(broker)
    consumer = InventoryConsumer(broker)

    with patch.object(consumer, "_check_inventory", return_value=False):
        producer.publish_order(_make_order())
        broker.consume_all("orders")

    assert broker._queues.get("inventory.failed"), "Expected failed event"
    assert not broker._queues.get("inventory.reserved"), "Should not have reserved event"


def test_inventory_consumer_invalid_payload_routes_to_dlq() -> None:
    """Invalid payload re-raises ValidationError and eventually lands in DLQ."""
    broker = InMemoryBroker(max_retries=2)
    InventoryConsumer(broker)
    # Publish garbage that cannot be parsed as OrderEvent
    broker.publish("orders", {"bad": "data"})

    for _ in range(5):
        broker.consume_all("orders")

    assert broker.dlq_depth("orders") == 1


def test_inventory_consumer_correlation_id_propagated() -> None:
    """Correlation ID from order message is forwarded to inventory event."""
    broker = _make_broker()
    producer = OrderProducer(broker)
    consumer = InventoryConsumer(broker)

    with patch.object(consumer, "_check_inventory", return_value=True):
        producer.publish_order(_make_order())
        broker.consume_all("orders")

    reserved_msgs = broker._queues.get("inventory.reserved", [])
    assert len(reserved_msgs) == 1
    assert "correlation_id" in reserved_msgs[0].headers


# ---------------------------------------------------------------------------
# NotificationConsumer tests
# ---------------------------------------------------------------------------

def test_notification_consumer_receives_both_topics() -> None:
    """NotificationConsumer handles messages from both inventory topics."""
    broker = _make_broker()
    notif = NotificationConsumer(broker)
    notifications: list[tuple[str, str]] = []

    with patch.object(notif, "_send_notification", side_effect=lambda oid, s: notifications.append((oid, s))):
        broker.publish("inventory.reserved", {"order_id": "ORD-R", "reserved_items": [], "timestamp": "2024-01-01T00:00:00Z"})
        broker.publish("inventory.failed", {"order_id": "ORD-F", "reason": "out of stock", "timestamp": "2024-01-01T00:00:00Z"})
        broker.consume_all("inventory.reserved")
        broker.consume_all("inventory.failed")

    assert ("ORD-R", "reserved") in notifications
    assert ("ORD-F", "failed") in notifications
    assert len(notifications) == 2


def test_notification_consumer_idempotent() -> None:
    """Duplicate order_id on same topic triggers notification only once."""
    broker = _make_broker()
    notif = NotificationConsumer(broker)
    notifications: list[tuple[str, str]] = []

    with patch.object(notif, "_send_notification", side_effect=lambda oid, s: notifications.append((oid, s))):
        broker.publish("inventory.reserved", {"order_id": "ORD-X", "reserved_items": [], "timestamp": "2024-01-01T00:00:00Z"})
        broker.publish("inventory.reserved", {"order_id": "ORD-X", "reserved_items": [], "timestamp": "2024-01-01T00:00:00Z"})
        broker.consume_all("inventory.reserved")

    assert len(notifications) == 1
```

---

### `tests/test_integration.py`

```python
"""
tests/test_integration.py
End-to-end pipeline tests.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from order_pipeline.broker import InMemoryBroker
from order_pipeline.consumers import InventoryConsumer, NotificationConsumer
from order_pipeline.dlq import DLQMonitor
from order_pipeline.models import OrderEvent, OrderItem
from order_pipeline.producer import OrderProducer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_orders(n: int) -> list[OrderEvent]:
    return [
        OrderEvent(
            order_id=f"ORD-{i:03d}",
            user_id=f"USR-{i}",
            items=[OrderItem(sku="SKU-A", quantity=1, unit_price=50.0)],
            total=50.0,
        )
        for i in range(n)
    ]


def _build_pipeline(max_retries: int = 3) -> tuple[InMemoryBroker, OrderProducer, InventoryConsumer, NotificationConsumer]:
    broker = InMemoryBroker(max_retries=max_retries)
    producer = OrderProducer(broker)
    inv_consumer = InventoryConsumer(broker)
    notif_consumer = NotificationConsumer(broker)
    return broker, producer, inv_consumer, notif_consumer


PIPELINE_TOPICS = ["orders", "inventory.reserved", "inventory.failed"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_full_pipeline_all_orders_notified() -> None:
    """
    Publish 5 orders. All pass inventory (mocked).
    Verify 5 notifications sent and DLQ empty on all topics.
    """
    broker, producer, inv_consumer, notif_consumer = _build_pipeline()
    notifications_sent: list[tuple[str, str]] = []

    def mock_notify(order_id: str, status: str) -> None:
        notifications_sent.append((order_id, status))

    with patch.object(notif_consumer, "_send_notification", side_effect=mock_notify):
        with patch.object(inv_consumer, "_check_inventory", return_value=True):
            for order in _make_orders(5):
                producer.publish_order(order)

            broker.consume_all("orders")
            broker.consume_all("inventory.reserved")
            broker.consume_all("inventory.failed")

    assert len(notifications_sent) == 5
    assert all(status == "reserved" for _, status in notifications_sent)
    assert broker.dlq_depth("orders") == 0
    assert broker.dlq_depth("inventory.reserved") == 0


def test_full_pipeline_mixed_inventory() -> None:
    """
    Publish 5 orders. 2 fail inventory, 3 pass.
    Verify exactly 5 notifications (3 reserved, 2 failed).
    DLQ empty on all topics.
    """
    broker, producer, inv_consumer, notif_consumer = _build_pipeline()
    notifications_sent: list[tuple[str, str]] = []

    def mock_notify(order_id: str, status: str) -> None:
        notifications_sent.append((order_id, status))

    # First 3 pass, last 2 fail
    inventory_results = [True, True, True, False, False]

    def controlled_inventory(event: OrderEvent) -> bool:
        idx = int(event.order_id.split("-")[1])
        return inventory_results[idx]

    with patch.object(notif_consumer, "_send_notification", side_effect=mock_notify):
        with patch.object(inv_consumer, "_check_inventory", side_effect=controlled_inventory):
            for order in _make_orders(5):
                producer.publish_order(order)

            broker.consume_all("orders")
            broker.consume_all("inventory.reserved")
            broker.consume_all("inventory.failed")

    assert len(notifications_sent) == 5
    reserved = [s for _, s in notifications_sent if s == "reserved"]
    failed = [s for _, s in notifications_sent if s == "failed"]
    assert len(reserved) == 3
    assert len(failed) == 2

    monitor = DLQMonitor(broker, PIPELINE_TOPICS)
    assert all(depth == 0 for depth in monitor.report().values())


def test_dlq_monitor_reports_correctly() -> None:
    """After clean run, DLQMonitor.report() shows 0 depth for all topics."""
    broker, producer, inv_consumer, notif_consumer = _build_pipeline()

    with patch.object(notif_consumer, "_send_notification"):
        with patch.object(inv_consumer, "_check_inventory", return_value=True):
            for order in _make_orders(3):
                producer.publish_order(order)
            broker.consume_all("orders")
            broker.consume_all("inventory.reserved")
            broker.consume_all("inventory.failed")

    monitor = DLQMonitor(broker, PIPELINE_TOPICS)
    report = monitor.report()
    assert report == {"orders": 0, "inventory.reserved": 0, "inventory.failed": 0}


def test_duplicate_orders_not_double_processed() -> None:
    """
    Sending the same order_id twice results in exactly one inventory event
    and one notification — idempotency end-to-end.
    """
    broker, producer, inv_consumer, notif_consumer = _build_pipeline()
    notifications_sent: list[tuple[str, str]] = []

    def mock_notify(order_id: str, status: str) -> None:
        notifications_sent.append((order_id, status))

    with patch.object(notif_consumer, "_send_notification", side_effect=mock_notify):
        with patch.object(inv_consumer, "_check_inventory", return_value=True):
            order = _make_orders(1)[0]
            producer.publish_order(order)
            producer.publish_order(order)  # duplicate

            broker.consume_all("orders")
            broker.consume_all("inventory.reserved")

    assert len(notifications_sent) == 1


def test_dlq_replay_recovers_failed_message() -> None:
    """
    Permanently-failing handler routes to DLQ.
    After fixing the handler, dlq_replay re-delivers the message.
    """
    broker = InMemoryBroker(max_retries=2)
    received: list[object] = []

    def bad_handler(msg: object) -> None:
        raise RuntimeError("broken")

    broker.subscribe("retry-test", bad_handler)
    broker.publish("retry-test", "important_payload")

    for _ in range(5):
        broker.consume_all("retry-test")

    assert broker.dlq_depth("retry-test") == 1

    # Fix: replace handler and replay
    broker._handlers["retry-test"] = [received.append]
    broker.dlq_replay("retry-test")
    broker.consume_all("retry-test")

    assert len(received) == 1
    assert received[0].payload == "important_payload"  # type: ignore[union-attr]
    assert broker.dlq_depth("retry-test") == 0
```

---

## Common Mistakes & Debugging Tips

### 1. `model_validate` vs raw dict

**Symptom:** `ValidationError: 1 validation error for OrderEvent — items: list[OrderItem] Input should be a valid list`

**Cause:** Publishing a Pydantic model object directly instead of serialising it:
```python
# Wrong — broker stores the model object, consumer gets it back as an object
broker.publish("orders", event)

# Correct — serialise to dict first
broker.publish("orders", event.model_dump(mode="json"))
```
Use `mode="json"` not `mode="python"` — `datetime` objects won't serialise to JSON-safe strings with `mode="python"`.

### 2. Off-by-one in max_retries

**Symptom:** `assert broker.dlq_depth("test") == 1` fails; depth is 0 and message was delivered on attempt 4.

**Cause:** Retry counter starts at 0. If your condition is `if msg.attempt > max_retries` instead of `>= max_retries`, you get one extra attempt.

```python
# broker.py inside consume_all
msg.attempt += 1
if msg.attempt >= self.max_retries:  # >= not >
    self._route_to_dlq(...)
```

### 3. Handler not registered before consume_all

**Symptom:** Messages disappear, handlers never called.

**Cause:** Consumer `__init__` must call `broker.subscribe(...)`. If you call `consume_all` before constructing the consumer, the queue drains with no handlers registered. Always construct consumers before calling `consume_all`.

### 4. Thread-safety in tests

**Symptom:** Intermittent test failures when running `pytest -n4`.

**Cause:** `InMemoryBroker` uses a `threading.Lock` but tests that directly access `broker._queues` bypass it. In tests, always use the public API (`publish`, `consume_all`, `dlq_depth`) rather than accessing private attributes directly — except in the simple assertion `broker._queues.get(...)` which is read-only and safe in single-threaded test runs.

### 5. `model_dump` called on a dict

**Symptom:** `AttributeError: 'dict' object has no attribute 'model_dump'`

**Cause:** Calling `.model_dump()` on something already serialised. Check whether the variable is the Pydantic model or the already-serialised dict.

### 6. DLQ replay resets attempt counter

**Symptom:** After `dlq_replay`, messages immediately DLQ again even though the handler is now fixed.

**Cause:** If you forgot to reset `msg.attempt = 0` in `dlq_replay`, the message re-enters with `attempt == max_retries` and is immediately re-routed to DLQ on the first failure. The reference implementation sets `rec.message.attempt = 0` before re-queuing.

### 7. `_seen_order_ids` shared between test runs

**Symptom:** Second test that uses the same consumer instance shows idempotency triggering unexpectedly.

**Cause:** Tests must construct a fresh consumer per test (not a module-level singleton). Each test function should call `InventoryConsumer(broker)` inside its body.
