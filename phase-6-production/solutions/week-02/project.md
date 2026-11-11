# Phase 6 · Week 02 · Project — Event-Driven Order Processing Pipeline (Full Solution)

> Production-quality reference implementation. Covers the full project spec from `week-02/day-7.md`.
> Run with: `python main.py` · Test with: `pytest tests/ -v`

---

## Project layout

```
src/order_pipeline/
    __init__.py
    config.py       # pydantic-settings BrokerSettings
    models.py       # all Pydantic event models
    broker.py       # AbstractBroker ABC + InMemoryBroker + KafkaBroker stub
    producers.py    # OrderProducer
    consumers.py    # InventoryConsumer + NotificationConsumer
    dlq.py          # DLQMonitor + DLQReplayer
    main.py         # full pipeline orchestration
tests/
    test_broker.py
    test_consumers.py
    test_integration.py
pyproject.toml
.env.example
```

---

## `src/order_pipeline/config.py`

```python
"""
Application settings loaded from environment variables.

Set BROKER_TYPE, BROKER_MAX_RETRIES, BROKER_QUEUE_DEPTH_LIMIT
in .env or the process environment to override defaults.
"""

from __future__ import annotations

import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class BrokerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Which broker implementation to use at runtime
    broker_type: Literal["memory", "kafka"] = "memory"

    # Max delivery attempts before DLQ routing
    broker_max_retries: int = int(os.getenv("BROKER_MAX_RETRIES", "3"))

    # Optional soft cap on queue depth (informational; not enforced by broker)
    broker_queue_depth_limit: int = int(os.getenv("BROKER_QUEUE_DEPTH_LIMIT", "10000"))

    # Inventory mock: fraction of orders that pass inventory check
    inventory_pass_rate: float = 0.8

    # Logging
    log_level: str = "INFO"


settings = BrokerSettings()
```

---

## `src/order_pipeline/models.py`

```python
"""
Pydantic v2 event models for the order processing pipeline.

All models use timezone-aware datetimes and Literal status fields
so consumers can pattern-match on status without string comparisons.
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
    """Envelope stored when a message exceeds max_retries."""
    original_topic: str
    payload: dict
    error: str
    attempts: int
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
```

---

## `src/order_pipeline/broker.py`

```python
"""
Broker abstraction layer.

AbstractBroker  — interface all concrete implementations must satisfy.
InMemoryBroker  — thread-safe, synchronous, in-process broker.
                  Supports pub/sub fanout, at-least-once retry, DLQ.
KafkaBroker     — conceptual stub only; raises NotImplementedError.
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
    """
    An envelope around a broker payload.

    Attributes:
        id:       Unique message identifier (UUID4 string).
        payload:  The raw message body (any JSON-serialisable value).
        headers:  Key-value metadata (e.g. correlation_id, source).
        attempt:  How many delivery attempts have been made so far.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    payload: Any = None
    headers: dict[str, str] = field(default_factory=dict)
    attempt: int = 0


@dataclass
class DeadLetterRecord:
    """A message that has been moved to the DLQ after max_retries failures."""
    message: Message
    topic: str
    last_error: str
    total_attempts: int

    @property
    def payload(self) -> Any:
        return self.message.payload


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class AbstractBroker(ABC):
    """
    Minimal contract every broker implementation must fulfil.

    Concrete subclasses: InMemoryBroker (tests/dev), KafkaBroker (production).
    """

    @abstractmethod
    def publish(
        self,
        topic: str,
        payload: Any,
        headers: dict[str, str] | None = None,
    ) -> str:
        """Publish a message. Returns the message id."""

    @abstractmethod
    def subscribe(self, topic: str, handler: Callable[..., None]) -> None:
        """
        Register a handler for topic.
        Multiple handlers on the same topic = fan-out (all receive every message).
        """

    @abstractmethod
    def consume_all(self, topic: str) -> int:
        """
        Process all queued messages for topic synchronously.
        Returns count of successfully delivered messages.
        """

    def dlq_depth(self, topic: str) -> int:  # noqa: ARG002
        """Return count of dead-lettered messages for topic."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement dlq_depth."
        )


# ---------------------------------------------------------------------------
# In-memory concrete implementation
# ---------------------------------------------------------------------------

class InMemoryBroker(AbstractBroker):
    """
    Synchronous, thread-safe, in-process message broker.

    Design choices:
    - A single threading.Lock protects all mutable state (_queues, _handlers, _dlq).
    - Fan-out: every handler registered for a topic receives every message.
    - At-least-once: on handler exception the message is re-queued (attempt += 1).
      After max_retries attempts the message is routed to the DLQ.
    - DLQ is per-topic; use dlq_inspect() + dlq_replay() for manual recovery.
    - publish() accepts optional headers for correlation/tracing metadata.
    """

    def __init__(self, max_retries: int = 3) -> None:
        self.max_retries = max_retries
        self._lock = threading.Lock()
        # Per-topic FIFO message queues
        self._queues: dict[str, list[Message]] = {}
        # Per-topic registered handlers (fan-out list)
        self._handlers: dict[str, list[Callable[..., None]]] = {}
        # Per-topic dead-letter records
        self._dlq: dict[str, list[DeadLetterRecord]] = {}

    # ------------------------------------------------------------------
    # AbstractBroker
    # ------------------------------------------------------------------

    def publish(
        self,
        topic: str,
        payload: Any,
        headers: dict[str, str] | None = None,
    ) -> str:
        """Enqueue a message on `topic`. Thread-safe."""
        msg = Message(payload=payload, headers=headers or {})
        with self._lock:
            self._queues.setdefault(topic, []).append(msg)
        log.debug("publish topic=%s msg_id=%s", topic, msg.id)
        return msg.id

    def subscribe(self, topic: str, handler: Callable[..., None]) -> None:
        """Register `handler` to receive all future messages on `topic`."""
        with self._lock:
            self._handlers.setdefault(topic, []).append(handler)
        log.debug("subscribe topic=%s handler=%s", topic, handler.__qualname__)

    def consume_all(self, topic: str) -> int:
        """
        Drain the queue for `topic`.

        For each message:
        1. Call all registered handlers (fan-out).
        2. On success: increment processed count.
        3. On any handler exception:
           a. If attempt < max_retries: re-enqueue at tail.
           b. If attempt >= max_retries: route to DLQ.

        Returns the count of messages fully delivered (not DLQ'd).
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
                        "handler %s failed msg_id=%s attempt=%d error=%s",
                        handler.__qualname__,
                        msg.id,
                        msg.attempt + 1,
                        exc,
                    )
                    break  # stop calling remaining handlers; message will be re-queued

            if all_ok:
                processed += 1
            else:
                msg.attempt += 1
                if msg.attempt >= self.max_retries:
                    self._route_to_dlq(msg, topic, last_error)
                else:
                    with self._lock:
                        self._queues.setdefault(topic, []).append(msg)
                    log.info(
                        "re-queued msg_id=%s attempt=%d/%d topic=%s",
                        msg.id,
                        msg.attempt,
                        self.max_retries,
                        topic,
                    )

        return processed

    # ------------------------------------------------------------------
    # DLQ API
    # ------------------------------------------------------------------

    def dlq_depth(self, topic: str) -> int:
        with self._lock:
            return len(self._dlq.get(topic, []))

    def dlq_inspect(self, topic: str) -> list[DeadLetterRecord]:
        """Return a snapshot copy of DLQ records for `topic`."""
        with self._lock:
            return list(self._dlq.get(topic, []))

    def dlq_replay(self, topic: str) -> int:
        """
        Move all DLQ messages for `topic` back onto the main queue.
        Resets attempt counter so they get a full retry budget.
        Returns count of messages re-enqueued.

        Use only after the root cause of failures has been fixed.
        """
        with self._lock:
            records = self._dlq.pop(topic, [])
            for rec in records:
                rec.message.attempt = 0  # full retry budget
                self._queues.setdefault(topic, []).append(rec.message)
        log.info("dlq_replay topic=%s count=%d", topic, len(records))
        return len(records)

    # ------------------------------------------------------------------
    # Internal
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
            "DLQ topic=%s msg_id=%s attempts=%d error=%s",
            topic,
            msg.id,
            msg.attempt,
            last_error,
        )


# ---------------------------------------------------------------------------
# Kafka stub — conceptual scaffolding, never instantiated in this course
# ---------------------------------------------------------------------------

class KafkaBroker(AbstractBroker):
    """
    Conceptual stub for a real Kafka broker integration.

    In production this would wrap confluent-kafka's Producer and Consumer.
    It is included here to document what a real implementation would require,
    and to satisfy the AbstractBroker interface for type-checking purposes.

    NOT instantiated or called in this course — InMemoryBroker only.
    """

    def publish(
        self,
        topic: str,
        payload: Any,
        headers: dict[str, str] | None = None,
    ) -> str:
        # Would:
        # 1. Serialize payload to bytes: json.dumps(payload).encode() or Avro/Protobuf.
        # 2. Determine partition key (e.g. order_id) for ordering guarantees.
        # 3. Call producer.produce(topic, value=serialized, key=partition_key,
        #                          headers=list((headers or {}).items()),
        #                          on_delivery=delivery_report_callback).
        # 4. Call producer.poll(0) to trigger callbacks; or producer.flush() for sync.
        # 5. Return the message offset as a string (no UUID in real Kafka).
        raise NotImplementedError("Use InMemoryBroker in this course")

    def subscribe(self, topic: str, handler: Callable[..., None]) -> None:
        # Would:
        # 1. Store handler in a registry keyed by topic.
        # 2. Set consumer group.id (from config; typically service-name + env).
        # 3. Call consumer.subscribe([topic]) — Kafka handles partition assignment.
        # 4. Note: multiple independent groups on the same topic = fan-out semantics.
        raise NotImplementedError("Use InMemoryBroker in this course")

    def consume_all(self, topic: str) -> int:
        # Would:
        # 1. Loop: msg = consumer.poll(timeout=1.0); if msg is None: break.
        # 2. Check msg.error(); on KafkaException route to DLQ topic via a Producer.
        # 3. Deserialize msg.value(); call stored handler.
        # 4. On success: consumer.commit(asynchronous=False) — manual offset commit.
        # 5. On handler exception: increment retry counter (stored in msg headers or
        #    a Redis key); after max_retries produce to "dlq.<topic>".
        raise NotImplementedError("Use InMemoryBroker in this course")
```

---

## `src/order_pipeline/producers.py`

```python
"""
OrderProducer: serialises OrderEvent and publishes to the 'orders' topic.

Attaches a correlation_id header on every message for distributed tracing
(log correlation across InventoryConsumer and NotificationConsumer).
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

## `src/order_pipeline/consumers.py`

```python
"""
InventoryConsumer  — subscribes to 'orders', checks mock inventory,
                     publishes InventoryReservedEvent or InventoryFailedEvent.
NotificationConsumer — subscribes to BOTH inventory topics (fan-out),
                       sends a mock notification per order.

Both consumers are idempotent: each maintains _seen_order_ids: set[str].

Production idempotency note:
    Replace _seen_order_ids with a Redis SET using SETNX + EX TTL:
        SETNX order_pipeline:seen:{order_id} 1 EX 86400
    This makes deduplication durable across restarts and distributed across
    multiple consumer replicas. The TTL (idempotency window) is a trade-off:
    too short → risk of reprocessing old duplicates;
    too long  → memory growth for old IDs never seen again.
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
    Consumes orders and performs a mock inventory check.

    On success  → publishes InventoryReservedEvent to 'inventory.reserved'
    On failure  → publishes InventoryFailedEvent  to 'inventory.failed'
    Duplicate   → skips without publishing (idempotent)
    Bad payload → re-raises ValidationError → broker retries → DLQ
    """

    def __init__(self, broker: AbstractBroker) -> None:
        self._broker = broker
        self._seen_order_ids: set[str] = set()
        # Register immediately so messages published before consume_all are handled
        broker.subscribe("orders", self._handle)

    # ------------------------------------------------------------------
    # Idempotency
    # ------------------------------------------------------------------

    def _is_duplicate(self, order_id: str) -> bool:
        """
        Return True if order_id was already processed.

        Production note: replace self._seen_order_ids with a Redis SET
        (SETNX order_id 1 EX 86400) for distributed idempotency.
        """
        if order_id in self._seen_order_ids:
            log.warning("Duplicate order_id=%s in InventoryConsumer — skipping", order_id)
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
            log.exception(
                "Invalid OrderEvent payload msg_id=%s — will DLQ after retries", msg.id
            )
            raise  # broker handles retry → DLQ

        if self._is_duplicate(event.order_id):
            return

        correlation_id = msg.headers.get("correlation_id", "n/a")
        log.info(
            "InventoryConsumer order_id=%s correlation_id=%s",
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
        """Mock inventory check based on settings.inventory_pass_rate."""
        return random.random() < settings.inventory_pass_rate


# ---------------------------------------------------------------------------

class NotificationConsumer:
    """
    Subscribes to BOTH 'inventory.reserved' and 'inventory.failed' (fan-out).
    Sends one notification per order regardless of source topic.
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

## `src/order_pipeline/dlq.py`

```python
"""
DLQMonitor  — reports per-topic DLQ depths (push to Prometheus/Datadog in prod).
DLQReplayer — replays dead-lettered messages after root-cause fix.
"""

from __future__ import annotations

import logging

from order_pipeline.broker import InMemoryBroker

log = logging.getLogger(__name__)


class DLQMonitor:
    """
    Observes DLQ depth across a set of topics.

    In production: replace log.log() calls with metric pushes to
    Datadog (statsd) or a Prometheus Gauge.
    Set an alert if any topic depth exceeds a threshold.
    """

    def __init__(self, broker: InMemoryBroker, topics: list[str]) -> None:
        self._broker = broker
        self._topics = topics

    def report(self) -> dict[str, int]:
        """Return {topic: dlq_depth} for each watched topic."""
        return {t: self._broker.dlq_depth(t) for t in self._topics}

    def log_report(self) -> None:
        for topic, depth in self.report().items():
            level = logging.WARNING if depth > 0 else logging.DEBUG
            log.log(level, "DLQ topic=%s depth=%d", topic, depth)

    def any_non_empty(self) -> bool:
        return any(d > 0 for d in self.report().values())


class DLQReplayer:
    """
    Re-enqueues dead-lettered messages onto their original topic queues.

    Usage pattern:
    1. Identify root cause (via DLQMonitor + dlq_inspect).
    2. Fix the handler or the upstream data.
    3. Call replay(topic) — messages get a fresh retry budget.
    4. Call broker.consume_all(topic) to process them.

    WARNING: never replay without fixing the root cause — you will
    immediately re-DLQ the same messages and loop forever.
    """

    def __init__(self, broker: InMemoryBroker) -> None:
        self._broker = broker

    def replay(self, topic: str) -> int:
        """Replay all DLQ messages for `topic`. Returns count replayed."""
        count = self._broker.dlq_replay(topic)
        log.info("DLQReplayer replayed %d messages topic=%s", count, topic)
        return count

    def replay_all(self, topics: list[str]) -> dict[str, int]:
        return {t: self.replay(t) for t in topics}
```

---

## `src/order_pipeline/main.py`

```python
"""
main.py — Full pipeline orchestration.

Publishes 10 orders (mix of pass/fail inventory), runs the full
consume chain, prints a DLQ report, and exits.

Run: python main.py
     BROKER_MAX_RETRIES=5 python main.py
"""

from __future__ import annotations

import logging
import os
from unittest.mock import patch

from order_pipeline.broker import InMemoryBroker
from order_pipeline.config import settings
from order_pipeline.consumers import InventoryConsumer, NotificationConsumer
from order_pipeline.dlq import DLQMonitor
from order_pipeline.models import OrderEvent, OrderItem
from order_pipeline.producers import OrderProducer

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
)
log = logging.getLogger(__name__)

PIPELINE_TOPICS = ["orders", "inventory.reserved", "inventory.failed"]


def build_orders(n: int = 10) -> list[OrderEvent]:
    return [
        OrderEvent(
            order_id=f"ORD-{i:04d}",
            user_id=f"USR-{i % 3:02d}",
            items=[OrderItem(sku=f"SKU-{i % 5}", quantity=1, unit_price=float(10 + i))],
            total=float(10 + i),
        )
        for i in range(n)
    ]


def run_pipeline() -> None:
    log.info("=== Order Processing Pipeline START ===")
    log.info(
        "config: broker_type=%s max_retries=%d inventory_pass_rate=%.0f%%",
        settings.broker_type,
        settings.broker_max_retries,
        settings.inventory_pass_rate * 100,
    )

    broker = InMemoryBroker(max_retries=settings.broker_max_retries)
    producer = OrderProducer(broker)
    inv_consumer = InventoryConsumer(broker)
    notif_consumer = NotificationConsumer(broker)
    monitor = DLQMonitor(broker, PIPELINE_TOPICS)

    orders = build_orders(10)
    log.info("Publishing %d orders...", len(orders))
    for order in orders:
        producer.publish_order(order)

    # Consume in pipeline order
    log.info("--- Consuming orders ---")
    delivered = broker.consume_all("orders")
    log.info("orders: %d delivered", delivered)

    log.info("--- Consuming inventory.reserved ---")
    delivered = broker.consume_all("inventory.reserved")
    log.info("inventory.reserved: %d delivered", delivered)

    log.info("--- Consuming inventory.failed ---")
    delivered = broker.consume_all("inventory.failed")
    log.info("inventory.failed: %d delivered", delivered)

    log.info("=== DLQ Report ===")
    monitor.log_report()
    report = monitor.report()
    total_dlq = sum(report.values())
    if total_dlq == 0:
        log.info("All messages delivered successfully — DLQ is empty.")
    else:
        log.warning("DLQ non-empty: %s", report)

    log.info("=== Pipeline DONE ===")


if __name__ == "__main__":
    run_pipeline()
```

---

## `src/order_pipeline/__init__.py`

```python
"""Order processing pipeline — event-driven, in-memory broker."""
```

---

## `.env.example`

```
# Broker configuration
BROKER_TYPE=memory
BROKER_MAX_RETRIES=3
BROKER_QUEUE_DEPTH_LIMIT=10000

# Mock inventory pass rate (0.0 – 1.0)
INVENTORY_PASS_RATE=0.8

# Logging
LOG_LEVEL=INFO
```

---

## `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "order-pipeline"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

---

## `tests/test_broker.py`

```python
"""
tests/test_broker.py
Unit tests for InMemoryBroker.
"""

from __future__ import annotations

import pytest

from order_pipeline.broker import DeadLetterRecord, InMemoryBroker, Message


def _broker(max_retries: int = 3) -> InMemoryBroker:
    return InMemoryBroker(max_retries=max_retries)


def test_successful_delivery_no_dlq() -> None:
    """Handler succeeds on first attempt; DLQ stays empty."""
    b = _broker()
    received: list[Message] = []

    b.subscribe("t", received.append)
    b.publish("t", {"value": 42})
    count = b.consume_all("t")

    assert count == 1
    assert len(received) == 1
    assert received[0].payload == {"value": 42}
    assert b.dlq_depth("t") == 0


def test_retry_then_success() -> None:
    """Handler fails twice, succeeds on third attempt; DLQ stays empty."""
    b = _broker(max_retries=3)
    attempts = 0

    def flaky(msg: Message) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("transient")

    b.subscribe("t", flaky)
    b.publish("t", "data")

    for _ in range(5):
        b.consume_all("t")

    assert attempts == 3
    assert b.dlq_depth("t") == 0


def test_max_retries_routes_to_dlq() -> None:
    """Always-failing handler routes to DLQ after max_retries."""
    b = _broker(max_retries=3)

    def always_fails(msg: Message) -> None:
        raise ValueError("Permanent failure")

    b.subscribe("t", always_fails)
    b.publish("t", {"key": "val"})

    for _ in range(10):
        b.consume_all("t")

    assert b.dlq_depth("t") == 1
    records = b.dlq_inspect("t")
    assert len(records) == 1
    rec = records[0]
    assert isinstance(rec, DeadLetterRecord)
    assert rec.payload == {"key": "val"}
    assert rec.total_attempts == 3
    assert "Permanent failure" in rec.last_error


def test_dlq_replay_re_enqueues() -> None:
    """dlq_replay re-enqueues message; a fixed handler processes it successfully."""
    b = _broker(max_retries=2)
    received: list[Message] = []

    b.subscribe("t", lambda m: (_ for _ in ()).throw(RuntimeError("fail")))
    b.publish("t", "payload")

    for _ in range(5):
        b.consume_all("t")

    assert b.dlq_depth("t") == 1

    # Fix: replace handler
    b._handlers["t"] = [received.append]
    replayed = b.dlq_replay("t")
    assert replayed == 1
    assert b.dlq_depth("t") == 0

    b.consume_all("t")
    assert len(received) == 1
    assert received[0].payload == "payload"


def test_fanout_multiple_handlers() -> None:
    """Two handlers on same topic both receive every message."""
    b = _broker()
    a_received: list[Message] = []
    c_received: list[Message] = []

    b.subscribe("fan", a_received.append)
    b.subscribe("fan", c_received.append)
    b.publish("fan", "shared")
    b.consume_all("fan")

    assert len(a_received) == 1
    assert len(c_received) == 1
    assert a_received[0].payload == "shared"
    assert c_received[0].payload == "shared"


def test_publish_returns_unique_ids() -> None:
    b = _broker()
    ids = {b.publish("t", i) for i in range(5)}
    assert len(ids) == 5


def test_message_headers_preserved() -> None:
    b = _broker()
    received: list[Message] = []
    b.subscribe("h", received.append)
    b.publish("h", "body", headers={"x-cid": "abc123"})
    b.consume_all("h")
    assert received[0].headers == {"x-cid": "abc123"}


def test_consume_empty_topic() -> None:
    b = _broker()
    b.subscribe("empty", lambda m: None)
    assert b.consume_all("empty") == 0


def test_multiple_messages_all_delivered() -> None:
    b = _broker()
    received: list[object] = []
    b.subscribe("m", lambda msg: received.append(msg.payload))
    for i in range(5):
        b.publish("m", i)
    b.consume_all("m")
    assert received == [0, 1, 2, 3, 4]
```

---

## `tests/test_consumers.py`

```python
"""
tests/test_consumers.py
Unit tests for InventoryConsumer and NotificationConsumer.
"""

from __future__ import annotations

from unittest.mock import patch

from order_pipeline.broker import InMemoryBroker
from order_pipeline.consumers import InventoryConsumer, NotificationConsumer
from order_pipeline.models import OrderEvent, OrderItem
from order_pipeline.producers import OrderProducer


def _order(order_id: str = "ORD-001", total: float = 20.0) -> OrderEvent:
    return OrderEvent(
        order_id=order_id,
        user_id="USR-1",
        items=[OrderItem(sku="SKU-A", quantity=2, unit_price=10.0)],
        total=total,
    )


def _setup() -> tuple[InMemoryBroker, OrderProducer, InventoryConsumer]:
    b = InMemoryBroker(max_retries=3)
    p = OrderProducer(b)
    c = InventoryConsumer(b)
    return b, p, c


# ---------------------------------------------------------------------------
# InventoryConsumer
# ---------------------------------------------------------------------------

def test_inventory_consumer_routes_reserved() -> None:
    """inventory pass → InventoryReservedEvent on inventory.reserved."""
    b, p, c = _setup()
    with patch.object(c, "_check_inventory", return_value=True):
        p.publish_order(_order())
        b.consume_all("orders")

    assert b._queues.get("inventory.reserved"), "Expected reserved event"
    assert not b._queues.get("inventory.failed"), "Should have no failed event"


def test_inventory_consumer_routes_failed() -> None:
    """inventory fail → InventoryFailedEvent on inventory.failed."""
    b, p, c = _setup()
    with patch.object(c, "_check_inventory", return_value=False):
        p.publish_order(_order())
        b.consume_all("orders")

    assert b._queues.get("inventory.failed"), "Expected failed event"
    assert not b._queues.get("inventory.reserved"), "Should have no reserved event"


def test_inventory_consumer_idempotent() -> None:
    """Same order_id published twice → exactly one inventory event."""
    b, p, c = _setup()
    with patch.object(c, "_check_inventory", return_value=True):
        p.publish_order(_order("ORD-DUPE"))
        p.publish_order(_order("ORD-DUPE"))
        b.consume_all("orders")

    assert len(b._queues.get("inventory.reserved", [])) == 1
    assert len(b._queues.get("inventory.failed", [])) == 0


def test_inventory_consumer_invalid_payload_routes_to_dlq() -> None:
    """Poison-pill payload retried then DLQ'd."""
    b = InMemoryBroker(max_retries=2)
    InventoryConsumer(b)
    b.publish("orders", {"garbage": True})

    for _ in range(5):
        b.consume_all("orders")

    assert b.dlq_depth("orders") == 1


def test_inventory_consumer_correlation_id_forwarded() -> None:
    """Correlation ID from the order message is present on inventory event."""
    b, p, c = _setup()
    with patch.object(c, "_check_inventory", return_value=True):
        p.publish_order(_order())
        b.consume_all("orders")

    msgs = b._queues.get("inventory.reserved", [])
    assert msgs and "correlation_id" in msgs[0].headers


# ---------------------------------------------------------------------------
# NotificationConsumer
# ---------------------------------------------------------------------------

def test_notification_consumer_receives_reserved() -> None:
    b = InMemoryBroker(max_retries=3)
    n = NotificationConsumer(b)
    sent: list[tuple[str, str]] = []

    with patch.object(n, "_send_notification", side_effect=lambda oid, s: sent.append((oid, s))):
        b.publish("inventory.reserved", {"order_id": "ORD-R", "reserved_items": [], "timestamp": "2024-01-01T00:00:00Z"})
        b.consume_all("inventory.reserved")

    assert sent == [("ORD-R", "reserved")]


def test_notification_consumer_receives_failed() -> None:
    b = InMemoryBroker(max_retries=3)
    n = NotificationConsumer(b)
    sent: list[tuple[str, str]] = []

    with patch.object(n, "_send_notification", side_effect=lambda oid, s: sent.append((oid, s))):
        b.publish("inventory.failed", {"order_id": "ORD-F", "reason": "no stock", "timestamp": "2024-01-01T00:00:00Z"})
        b.consume_all("inventory.failed")

    assert sent == [("ORD-F", "failed")]


def test_notification_consumer_idempotent() -> None:
    """Same order_id on same topic triggers notification exactly once."""
    b = InMemoryBroker(max_retries=3)
    n = NotificationConsumer(b)
    sent: list[tuple[str, str]] = []

    with patch.object(n, "_send_notification", side_effect=lambda oid, s: sent.append((oid, s))):
        payload = {"order_id": "ORD-DUP", "reserved_items": [], "timestamp": "2024-01-01T00:00:00Z"}
        b.publish("inventory.reserved", payload)
        b.publish("inventory.reserved", payload)
        b.consume_all("inventory.reserved")

    assert len(sent) == 1
```

---

## `tests/test_integration.py`

```python
"""
tests/test_integration.py
End-to-end pipeline tests: producer → inventory → notification.
"""

from __future__ import annotations

from unittest.mock import patch

from order_pipeline.broker import InMemoryBroker
from order_pipeline.consumers import InventoryConsumer, NotificationConsumer
from order_pipeline.dlq import DLQMonitor
from order_pipeline.models import OrderEvent, OrderItem
from order_pipeline.producers import OrderProducer


TOPICS = ["orders", "inventory.reserved", "inventory.failed"]


def _orders(n: int) -> list[OrderEvent]:
    return [
        OrderEvent(
            order_id=f"ORD-{i:03d}",
            user_id=f"USR-{i}",
            items=[OrderItem(sku="SKU-A", quantity=1, unit_price=50.0)],
            total=50.0,
        )
        for i in range(n)
    ]


def _pipeline(max_retries: int = 3) -> tuple[InMemoryBroker, OrderProducer, InventoryConsumer, NotificationConsumer]:
    b = InMemoryBroker(max_retries=max_retries)
    return b, OrderProducer(b), InventoryConsumer(b), NotificationConsumer(b)


def _drain(broker: InMemoryBroker) -> None:
    for t in TOPICS:
        broker.consume_all(t)


# ---------------------------------------------------------------------------

def test_all_pass_inventory_all_notified() -> None:
    """5 orders all pass → 5 'reserved' notifications, DLQ empty."""
    b, p, ic, nc = _pipeline()
    sent: list[tuple[str, str]] = []

    with patch.object(nc, "_send_notification", side_effect=lambda oid, s: sent.append((oid, s))):
        with patch.object(ic, "_check_inventory", return_value=True):
            for o in _orders(5):
                p.publish_order(o)
            _drain(b)

    assert len(sent) == 5
    assert all(s == "reserved" for _, s in sent)
    assert all(b.dlq_depth(t) == 0 for t in TOPICS)


def test_mixed_inventory_all_notified() -> None:
    """5 orders: 3 pass, 2 fail → 5 notifications (3 reserved, 2 failed)."""
    b, p, ic, nc = _pipeline()
    sent: list[tuple[str, str]] = []

    results = [True, True, True, False, False]

    def controlled(event: OrderEvent) -> bool:
        return results[int(event.order_id.split("-")[1])]

    with patch.object(nc, "_send_notification", side_effect=lambda oid, s: sent.append((oid, s))):
        with patch.object(ic, "_check_inventory", side_effect=controlled):
            for o in _orders(5):
                p.publish_order(o)
            _drain(b)

    assert len(sent) == 5
    assert len([s for _, s in sent if s == "reserved"]) == 3
    assert len([s for _, s in sent if s == "failed"]) == 2

    monitor = DLQMonitor(b, TOPICS)
    assert all(d == 0 for d in monitor.report().values())


def test_duplicate_orders_idempotent_end_to_end() -> None:
    """Same order published twice → exactly one notification."""
    b, p, ic, nc = _pipeline()
    sent: list[tuple[str, str]] = []

    with patch.object(nc, "_send_notification", side_effect=lambda oid, s: sent.append((oid, s))):
        with patch.object(ic, "_check_inventory", return_value=True):
            order = _orders(1)[0]
            p.publish_order(order)
            p.publish_order(order)  # duplicate
            _drain(b)

    assert len(sent) == 1


def test_dlq_monitor_all_zeros_after_clean_run() -> None:
    """DLQMonitor.report() returns 0 for all topics after clean pipeline run."""
    b, p, ic, nc = _pipeline()

    with patch.object(nc, "_send_notification"):
        with patch.object(ic, "_check_inventory", return_value=True):
            for o in _orders(3):
                p.publish_order(o)
            _drain(b)

    monitor = DLQMonitor(b, TOPICS)
    assert monitor.report() == {t: 0 for t in TOPICS}
    assert not monitor.any_non_empty()


def test_poison_pill_routes_to_dlq_not_blocking() -> None:
    """
    One malformed message in 'orders' routes to DLQ.
    Remaining valid orders still processed.
    """
    b = InMemoryBroker(max_retries=2)
    ic = InventoryConsumer(b)
    nc = NotificationConsumer(b)
    p = OrderProducer(b)
    sent: list[tuple[str, str]] = []

    with patch.object(nc, "_send_notification", side_effect=lambda oid, s: sent.append((oid, s))):
        with patch.object(ic, "_check_inventory", return_value=True):
            # One poison pill, then two valid orders
            b.publish("orders", {"malformed": True})
            for o in _orders(2):
                p.publish_order(o)

            for _ in range(5):
                b.consume_all("orders")
            b.consume_all("inventory.reserved")

    assert b.dlq_depth("orders") == 1
    assert len(sent) == 2  # valid orders delivered despite poison pill
```

---

## Running the project

```bash
# Install in editable mode
pip install -e ".[dev]"

# Run the pipeline
python src/order_pipeline/main.py

# Run tests
pytest tests/ -v --tb=short

# With coverage
pytest tests/ --cov=order_pipeline --cov-report=term-missing
```

### Expected `pytest` output

```
tests/test_broker.py::test_successful_delivery_no_dlq PASSED
tests/test_broker.py::test_retry_then_success PASSED
tests/test_broker.py::test_max_retries_routes_to_dlq PASSED
tests/test_broker.py::test_dlq_replay_re_enqueues PASSED
tests/test_broker.py::test_fanout_multiple_handlers PASSED
tests/test_broker.py::test_publish_returns_unique_ids PASSED
tests/test_broker.py::test_message_headers_preserved PASSED
tests/test_broker.py::test_consume_empty_topic PASSED
tests/test_broker.py::test_multiple_messages_all_delivered PASSED
tests/test_consumers.py::test_inventory_consumer_routes_reserved PASSED
tests/test_consumers.py::test_inventory_consumer_routes_failed PASSED
tests/test_consumers.py::test_inventory_consumer_idempotent PASSED
tests/test_consumers.py::test_inventory_consumer_invalid_payload_routes_to_dlq PASSED
tests/test_consumers.py::test_inventory_consumer_correlation_id_forwarded PASSED
tests/test_consumers.py::test_notification_consumer_receives_reserved PASSED
tests/test_consumers.py::test_notification_consumer_receives_failed PASSED
tests/test_consumers.py::test_notification_consumer_idempotent PASSED
tests/test_integration.py::test_all_pass_inventory_all_notified PASSED
tests/test_integration.py::test_mixed_inventory_all_notified PASSED
tests/test_integration.py::test_duplicate_orders_idempotent_end_to_end PASSED
tests/test_integration.py::test_dlq_monitor_all_zeros_after_clean_run PASSED
tests/test_integration.py::test_poison_pill_routes_to_dlq_not_blocking PASSED
```
