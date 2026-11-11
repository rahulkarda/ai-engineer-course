# Phase 6 · Week 02 · Day 7 — Project Day: Event-Driven Order Processing Pipeline

> Objective: Build a complete event-driven order processing pipeline from scratch, integrating everything from Week 2: broker abstraction, Pydantic models, producers, consumers, DLQ, idempotency, and tests.  ·  Time: ~4 hr  ·  Prereqs: [[day-1]] through [[day-6]]

## Overview

This is the Saturday build session. You will assemble a production-pattern order processing pipeline using only the in-memory broker (no external services required), then verify it with a pytest suite.

**What you will build:**

```
OrderProducer
     |
     | publishes OrderEvent → "orders" topic
     v
InventoryConsumer
     |
     |-- success → publishes InventoryReservedEvent → "inventory.reserved"
     |-- failure → publishes InventoryFailedEvent  → "inventory.failed"
     |-- 3 retries exceeded → DLQ
     v
NotificationConsumer (subscribed to BOTH "inventory.reserved" AND "inventory.failed")
     |
     | sends notification (mocked)
```

**By the end you will have:**
- `AbstractBroker` + `InMemoryBroker` (with DLQ)
- `OrderEvent`, `InventoryReservedEvent`, `InventoryFailedEvent` Pydantic models
- `OrderProducer`, `InventoryConsumer`, `NotificationConsumer`
- Idempotent consumers (dedup by `order_id`)
- pytest suite: retry, DLQ routing, idempotency, integration

**Target structure:**
```
src/order_pipeline/
    __init__.py
    broker.py          # AbstractBroker + InMemoryBroker
    models.py          # Pydantic event models
    producer.py        # OrderProducer
    consumers.py       # InventoryConsumer + NotificationConsumer
    config.py          # Settings via pydantic-settings
tests/
    test_broker.py
    test_consumers.py
    test_integration.py
```

---

## Part 1 (~30 min): Design review + scaffold

### Design review questions

Before writing any code, answer these in your head (or a scratch comment):

1. Why does `InMemoryBroker` need a `max_retries` parameter at the broker level rather than inside each consumer?
2. If `InventoryConsumer` and `NotificationConsumer` both subscribe to `"inventory.reserved"`, will both receive every message? What delivery model is this called?
3. An `OrderEvent` with `order_id="ORD-123"` arrives twice (at-least-once delivery). The `InventoryConsumer` must not reserve inventory twice. Where should deduplication state live — in the consumer, the broker, or the model?
4. The `NotificationConsumer` sends an external email. If the broker retries the notification event, the user gets two emails. How does idempotency help here?

### Scaffold

Create the directory structure and empty files first. The `config.py` and `__init__.py` content is provided; you write the rest.

```python
# src/order_pipeline/config.py
"""Application settings loaded from environment variables."""

from __future__ import annotations

import os
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

```python
# src/order_pipeline/__init__.py
"""Order processing pipeline — event-driven, in-memory broker."""
```

### Hint

Use `pyproject.toml` to declare `src/` as the package root so imports work cleanly:

```toml
[tool.setuptools.packages.find]
where = ["src"]
```

Run `pip install -e .` once so `import order_pipeline` resolves from `tests/`.

---

## Part 2 (~60 min): Broker abstraction + models

### 2a. Broker abstraction (`broker.py`)

Implement `AbstractBroker` as an ABC with three abstract methods:

```python
from abc import ABC, abstractmethod
from typing import Any, Callable

class AbstractBroker(ABC):
    @abstractmethod
    def publish(self, topic: str, payload: Any) -> str:
        """Publish a message. Returns message id."""
        ...

    @abstractmethod
    def subscribe(self, topic: str, handler: Callable[..., None]) -> None:
        """Register a handler for topic. Multiple handlers = fanout."""
        ...

    @abstractmethod
    def consume_all(self, topic: str) -> int:
        """Process all queued messages for topic. Returns count."""
        ...
```

Then implement `InMemoryBroker(AbstractBroker)` with:
- `_queues: dict[str, list[Message]]` — per-topic FIFO
- `_handlers: dict[str, list[Callable]]` — per-topic handler list
- `_dlq: dict[str, list[DeadLetterRecord]]` — per-topic dead-letter store
- `_retry_counts: dict[str, int]` — `{msg_id: attempt_count}`
- `max_retries: int` parameter on `__init__`

The `Message` and `DeadLetterRecord` dataclasses from Day 6 are your starting point. Add one field to `Message`: `headers: dict[str, str] = field(default_factory=dict)` — you will use this for tracing in Part 4.

Also add a stub `KafkaBroker(AbstractBroker)` class that raises `NotImplementedError` on all methods with a docstring explaining what each would require (topic creation, producer config, consumer group config). This is conceptual scaffolding only — it is never instantiated in tests.

```python
class KafkaBroker(AbstractBroker):
    """
    Conceptual stub for a real Kafka broker integration.

    In production this would wrap confluent-kafka Producer/Consumer.
    Not instantiated in this course — in-memory only.
    """

    def publish(self, topic: str, payload: Any) -> str:
        # Would: serialize payload to bytes (JSON/Avro/Protobuf),
        #        call producer.produce(topic, value=bytes, key=partition_key),
        #        call producer.flush() or use async callbacks.
        raise NotImplementedError("Use InMemoryBroker in this course")

    def subscribe(self, topic: str, handler: Callable[..., None]) -> None:
        # Would: store handler, configure consumer group id,
        #        call consumer.subscribe([topic]).
        raise NotImplementedError("Use InMemoryBroker in this course")

    def consume_all(self, topic: str) -> int:
        # Would: poll in a loop (consumer.poll(timeout_ms)),
        #        call handler, commit offset on success,
        #        route to DLQ topic on max retries.
        raise NotImplementedError("Use InMemoryBroker in this course")
```

### 2b. Event models (`models.py`)

Implement these Pydantic v2 models:

```python
# Hint: use datetime.datetime with timezone awareness
# Hint: use Literal for status fields so consumers can pattern-match

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, Field


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
```

**Validation rule to add:** `total` must equal `sum(item.quantity * item.unit_price for item in items)` within a floating-point tolerance of 0.01. Implement this as a `model_validator(mode="after")`.

### Hint: serialising Pydantic models into broker payloads

The broker payload is `Any`. Use `model.model_dump(mode="json")` when publishing and `OrderEvent.model_validate(payload)` when consuming. This round-trips cleanly and keeps the broker type-agnostic.

---

## Part 3 (~60 min): Producers and consumers

### 3a. OrderProducer (`producer.py`)

```python
import logging
from order_pipeline.broker import AbstractBroker
from order_pipeline.models import OrderEvent

log = logging.getLogger(__name__)

ORDERS_TOPIC = "orders"


class OrderProducer:
    def __init__(self, broker: AbstractBroker) -> None:
        self._broker = broker

    def publish_order(self, event: OrderEvent) -> str:
        """Serialise event and publish to 'orders' topic. Returns message id."""
        msg_id = self._broker.publish(ORDERS_TOPIC, event.model_dump(mode="json"))
        log.info("Published order_id=%s msg_id=%s", event.order_id, msg_id)
        return msg_id
```

### 3b. InventoryConsumer (`consumers.py`)

The `InventoryConsumer`:
- Subscribes to `"orders"`
- Deserialises payload into `OrderEvent`
- Calls `_check_inventory(event)` — a mock that randomly passes or fails based on `settings.inventory_pass_rate`
- On success: publishes `InventoryReservedEvent` to `"inventory.reserved"`
- On failure: publishes `InventoryFailedEvent` to `"inventory.failed"`
- Is **idempotent**: maintains `_seen_order_ids: set[str]`; if `order_id` already processed, logs a warning and returns without re-processing

```python
import logging
import random
from order_pipeline.broker import AbstractBroker
from order_pipeline.models import (
    InventoryFailedEvent,
    InventoryReservedEvent,
    OrderEvent,
)
from order_pipeline.config import settings

log = logging.getLogger(__name__)


class InventoryConsumer:
    def __init__(self, broker: AbstractBroker) -> None:
        self._broker = broker
        self._seen_order_ids: set[str] = set()
        # Register handler immediately on construction
        broker.subscribe("orders", self._handle)

    def _handle(self, msg: object) -> None:
        # msg is a Message dataclass; payload is a dict
        ...  # your implementation here

    def _check_inventory(self, event: OrderEvent) -> bool:
        """Mock inventory check. Returns True if stock available."""
        return random.random() < settings.inventory_pass_rate
```

### 3c. NotificationConsumer (`consumers.py`, continued)

The `NotificationConsumer`:
- Subscribes to **both** `"inventory.reserved"` and `"inventory.failed"`
- Is idempotent: dedup by `order_id`
- Calls `_send_notification(order_id, status)` — a mock that logs the notification
- `status` is `"reserved"` or `"failed"` depending on the source topic

```python
class NotificationConsumer:
    def __init__(self, broker: AbstractBroker) -> None:
        self._broker = broker
        self._seen_order_ids: set[str] = set()
        broker.subscribe("inventory.reserved", self._handle_reserved)
        broker.subscribe("inventory.failed", self._handle_failed)

    def _handle_reserved(self, msg: object) -> None:
        ...

    def _handle_failed(self, msg: object) -> None:
        ...

    def _send_notification(self, order_id: str, status: str) -> None:
        """Mock notification — logs only. Replace with real email/SMS in production."""
        log.info("NOTIFY order_id=%s status=%s", order_id, status)
```

### Hint: accessing `msg.payload`

Your `Message` dataclass carries `.payload: Any`. Inside a handler, cast it:

```python
def _handle(self, msg: object) -> None:  # msg is actually Message
    from order_pipeline.broker import Message
    assert isinstance(msg, Message)
    event = OrderEvent.model_validate(msg.payload)
    ...
```

---

## Part 4 (~60 min): DLQ + idempotency

### 4a. Wire up DLQ

The `InMemoryBroker` already routes to DLQ after `max_retries`. In this part you will:

1. Add a `dlq_depth(topic: str) -> int` method to `AbstractBroker` (as a non-abstract method that raises `NotImplementedError` with a helpful message — subclasses override it).

2. Add `dlq_inspect(topic: str) -> list[DeadLetterRecord]` and `dlq_replay(topic: str) -> int` to `InMemoryBroker`.

3. In `InventoryConsumer._handle`, if the payload cannot be parsed as `OrderEvent` (catches `ValidationError`), **re-raise** — this turns the message into a candidate for DLQ routing after retries.

4. Add a `DLQMonitor` class that accepts a broker and a list of topics to watch. It exposes `report() -> dict[str, int]` returning DLQ depth per topic. In production this would push metrics to Datadog/Prometheus.

```python
# Hint for DLQMonitor
class DLQMonitor:
    def __init__(self, broker: InMemoryBroker, topics: list[str]) -> None:
        self._broker = broker
        self._topics = topics

    def report(self) -> dict[str, int]:
        return {t: self._broker.dlq_depth(t) for t in self._topics}
```

### 4b. Strengthen idempotency

The current `_seen_order_ids: set[str]` is in-memory only. Add a note (comment in the code) explaining:
- In production, this would be a Redis SET with TTL, or a DB unique-constraint check.
- The idempotency window (how long you remember seen IDs) is a trade-off between memory and safety.

Add a helper method `_is_duplicate(order_id: str) -> bool` to both consumer classes that encapsulates the check-and-mark logic:

```python
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
```

### 4c. Message headers for tracing

Add a `correlation_id` to message headers in `OrderProducer.publish_order`:

```python
import uuid

def publish_order(self, event: OrderEvent) -> str:
    correlation_id = str(uuid.uuid4())
    msg_id = self._broker.publish(
        ORDERS_TOPIC,
        event.model_dump(mode="json"),
        headers={"correlation_id": correlation_id, "source": "order_producer"},
    )
    log.info(
        "Published order_id=%s msg_id=%s correlation_id=%s",
        event.order_id,
        msg_id,
        correlation_id,
    )
    return msg_id
```

Update `InMemoryBroker.publish` to accept an optional `headers: dict[str, str] | None = None` parameter and store it on the `Message`.

---

## Part 5 (~30 min): Tests + integration run

### 5a. Unit tests (`tests/test_broker.py`)

Write the following test functions (stubs provided, bodies are yours):

```python
"""
tests/test_broker.py — Unit tests for InMemoryBroker DLQ and retry behaviour.
"""

from __future__ import annotations

import pytest
from order_pipeline.broker import InMemoryBroker, Message


def test_successful_delivery_no_dlq() -> None:
    """Handler succeeds on first attempt; DLQ stays empty."""
    ...


def test_retry_then_success() -> None:
    """Handler fails twice, succeeds on third attempt; DLQ stays empty."""
    ...


def test_max_retries_routes_to_dlq() -> None:
    """
    Handler always fails; after max_retries attempts, message lands in DLQ.
    Verify dlq_depth == 1 and dlq_inspect returns the correct payload.
    """
    ...


def test_dlq_replay_re_enqueues() -> None:
    """
    After DLQ routing, dlq_replay re-enqueues the message.
    Swap in a succeeding handler and verify delivery.
    """
    ...


def test_fanout_multiple_handlers() -> None:
    """Two handlers subscribed to same topic both receive the message."""
    ...
```

### 5b. Consumer tests (`tests/test_consumers.py`)

```python
"""
tests/test_consumers.py — Unit tests for InventoryConsumer and NotificationConsumer.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch
from order_pipeline.broker import InMemoryBroker
from order_pipeline.consumers import InventoryConsumer, NotificationConsumer
from order_pipeline.models import OrderEvent, OrderItem
from order_pipeline.producer import OrderProducer


def _make_order(order_id: str = "ORD-001") -> OrderEvent:
    items = [OrderItem(sku="SKU-A", quantity=2, unit_price=10.0)]
    return OrderEvent(order_id=order_id, user_id="USR-1", items=items, total=20.0)


def test_inventory_consumer_idempotent() -> None:
    """
    Publishing same order_id twice does not result in two inventory events.
    'inventory.reserved' or 'inventory.failed' should have exactly 1 message.
    """
    ...


def test_inventory_consumer_routes_reserved() -> None:
    """
    When _check_inventory returns True, an InventoryReservedEvent is published
    to 'inventory.reserved'.
    """
    broker = InMemoryBroker(max_retries=3)
    producer = OrderProducer(broker)
    consumer = InventoryConsumer(broker)

    with patch.object(consumer, "_check_inventory", return_value=True):
        producer.publish_order(_make_order())
        broker.consume_all("orders")

    assert broker._queues["inventory.reserved"], "Expected reserved event"
    assert not broker._queues.get("inventory.failed"), "Should not have failed event"


def test_inventory_consumer_routes_failed() -> None:
    """When _check_inventory returns False, InventoryFailedEvent is published."""
    ...


def test_notification_consumer_receives_both_topics() -> None:
    """
    NotificationConsumer should process messages from both
    'inventory.reserved' and 'inventory.failed'.
    """
    ...
```

### 5c. Integration test (`tests/test_integration.py`)

The full pipeline test: publish 5 orders, run the full consume chain, assert all notifications were sent.

```python
"""
tests/test_integration.py — End-to-end pipeline test.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch
from order_pipeline.broker import InMemoryBroker
from order_pipeline.consumers import InventoryConsumer, NotificationConsumer
from order_pipeline.models import OrderEvent, OrderItem
from order_pipeline.producer import OrderProducer


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


def test_full_pipeline_all_orders_notified() -> None:
    """
    Publish 5 orders.
    All pass inventory (mocked).
    Verify 5 notifications are dispatched (via NotificationConsumer._send_notification).
    Verify DLQ is empty on all topics.
    """
    broker = InMemoryBroker(max_retries=3)
    producer = OrderProducer(broker)
    inv_consumer = InventoryConsumer(broker)
    notif_consumer = NotificationConsumer(broker)

    notifications_sent: list[tuple[str, str]] = []

    def mock_notify(order_id: str, status: str) -> None:
        notifications_sent.append((order_id, status))

    with patch.object(notif_consumer, "_send_notification", side_effect=mock_notify):
        with patch.object(inv_consumer, "_check_inventory", return_value=True):
            orders = _make_orders(5)
            for order in orders:
                producer.publish_order(order)

            broker.consume_all("orders")
            broker.consume_all("inventory.reserved")
            broker.consume_all("inventory.failed")

    assert len(notifications_sent) == 5
    assert all(status == "reserved" for _, status in notifications_sent)
    assert broker.dlq_depth("orders") == 0
    assert broker.dlq_depth("inventory.reserved") == 0
```

### 5d. Run everything

```bash
# From the project root
pytest tests/ -v --tb=short
```

Expected output: all tests pass. If a test fails, read the traceback — the most common issues are:
- Handler not registered before consuming (check consumer `__init__` calls `broker.subscribe`).
- `model_validate` failing because `payload` is already a Pydantic model object, not a dict (use `model_dump(mode="json")` when publishing).
- DLQ depth assertion off-by-one because `max_retries` counts from 1 vs 0 (check your retry counter initialisation).

---

## Exercises

1. The integration test pins `_check_inventory` to always return `True`. Add a second integration test where 2 of 5 orders fail inventory. Verify exactly 5 notifications are sent (3 `"reserved"`, 2 `"failed"`).

2. Add a `DLQMonitor` report to the integration test: after all processing, assert `monitor.report()` returns `{topic: 0}` for every topic.

3. Implement a `BatchOrderProducer` that accepts a `list[OrderEvent]` and publishes all in a single method call. Add a test verifying all messages appear in the broker queue before any are consumed.

4. Add a `circuit_breaker` to `InventoryConsumer`: if more than 3 consecutive inventory checks fail, the consumer stops processing and raises `CircuitBreakerOpenError`. Implement a `reset()` method to re-enable. Write tests for open and reset states.

5. Extend `OrderEvent` with a `priority: Literal["low", "normal", "high"] = "normal"` field. Modify `InMemoryBroker` to maintain separate queues per priority level within a topic and drain `high` before `normal` before `low` in `consume_all`. Write tests verifying the ordering.

(Solutions: ../solutions/week-02/project.md)

## Self-check

- [ ] I can explain why AbstractBroker enables test doubles without changing consumer code
- [ ] I can implement an idempotent consumer and explain the production trade-offs of in-memory vs. Redis deduplication
- [ ] I can trace a message from `OrderProducer.publish_order` through `InventoryConsumer` to `NotificationConsumer`
- [ ] I can write a pytest test that mocks the broker's internal inventory check and asserts on DLQ depth
- [ ] I understand why `model_dump(mode="json")` is required when passing Pydantic models across the broker boundary
- [ ] I can explain what happens to ordering guarantees when `InventoryConsumer` retries a message

## Supplementary reading

- [Pydantic v2 model validators](https://docs.pydantic.dev/latest/concepts/validators/)
- [pydantic-settings documentation](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [pytest monkeypatch and unittest.mock.patch patterns](https://docs.pytest.org/en/stable/how-to/monkeypatch.html)
- [Event-Driven Architecture patterns (Martin Fowler)](https://martinfowler.com/articles/201701-event-driven.html)
- [Idempotent consumers — pattern library](https://microservices.io/patterns/communication-style/idempotent-consumer.html)
- Designing Data-Intensive Applications (Kleppmann) — Chapter 11: Stream Processing, section on exactly-once semantics
