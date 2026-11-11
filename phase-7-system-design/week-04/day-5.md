# Phase 7 · Week 04 · Day 5 — Choreography, Orchestration & Idempotency at Scale

> Objective: Understand when to use choreography vs orchestration, implement saga compensation, and build reliable deduplication strategies.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

---

## Concept

### Two coordination models

When multiple services must cooperate to complete a business transaction, you have two fundamental coordination patterns:

**Orchestration** — one central coordinator (orchestrator) tells each participant what to do and when. The orchestrator holds the full workflow state.

**Choreography** — each service reacts to events from other services. No central brain; the workflow emerges from the chain of reactions.

```
ORCHESTRATION

  ┌─────────────────────────────────────┐
  │           Order Orchestrator        │
  └──┬──────────────┬────────────────┬──┘
     │              │                │
     v              v                v
  Payment       Inventory         Shipping
  Service       Service           Service
  (awaits cmd)  (awaits cmd)      (awaits cmd)


CHOREOGRAPHY

  OrderPlaced ──> PaymentService ──publishes──> PaymentProcessed
                                               │
                                               v
                                       InventoryService ──> InventoryReserved
                                                            │
                                                            v
                                                    ShippingService ──> OrderShipped
```

### Tradeoffs at a glance

| Dimension            | Orchestration                          | Choreography                            |
|----------------------|----------------------------------------|-----------------------------------------|
| Coupling             | Services coupled to orchestrator       | Services coupled only to event schema   |
| Observability        | Easy — one place to trace state        | Hard — trace must correlate event IDs   |
| Adding a step        | Change orchestrator only               | New service subscribes, no changes      |
| Failure handling     | Orchestrator issues compensations      | Compensation events must be pre-agreed  |
| Testing              | Unit-test orchestrator workflow        | Integration-test each service in chain  |
| Blast radius         | Orchestrator outage stalls all flows   | Any service can fail independently      |

### The Saga Pattern

A **saga** is a sequence of local transactions, each publishing an event that triggers the next. If any step fails, compensating transactions undo prior steps.

Sagas exist in both flavors:
- **Orchestration-based saga**: a saga orchestrator issues commands and waits for replies.
- **Choreography-based saga**: each service reacts to domain events and emits its own result.

```
CHOREOGRAPHY SAGA — Order fulfillment

  Step 1: ORDER_PLACED
          |
          v
  PaymentService
    success --> PAYMENT_AUTHORIZED
    failure --> ORDER_CANCELLED (saga ends)
          |
          v (on PAYMENT_AUTHORIZED)
  InventoryService
    success --> INVENTORY_RESERVED
    failure --> PAYMENT_REVERSED (compensate step 1), ORDER_CANCELLED
          |
          v (on INVENTORY_RESERVED)
  ShippingService
    success --> SHIPMENT_SCHEDULED
    failure --> INVENTORY_RELEASED (compensate step 2)
              PAYMENT_REVERSED   (compensate step 1)
              ORDER_CANCELLED
```

Each compensation must be **idempotent** — running it twice must not double-reverse a charge.

---

## How it works internally

### Message ordering

Distributed message brokers like Kafka provide ordering **within a partition**, not globally. Two guarantees you can rely on:

1. **Total order within a partition**: messages in partition 3 arrive in the order they were produced to partition 3.
2. **No ordering guarantees across partitions**: partition 3's messages may interleave arbitrarily with partition 7's.

Common strategy: **partition by aggregate ID** (order_id, user_id). All events for one order land in one partition → local ordering preserved.

```
Partition key = order_id

order_id=A  ──> partition 0: [A:created, A:paid, A:shipped]   (ordered)
order_id=B  ──> partition 1: [B:created, B:paid]              (ordered)
order_id=C  ──> partition 0: [C:created, ...]                  (interleaved with A, but A's own order intact)
```

**Pitfall**: if you change partition count after data exists, the same key will hash to a different partition. Kafka does not reshuffle existing data. Plan partition count upfront.

### At-least-once vs exactly-once

| Delivery guarantee | Meaning                                      | Common source of duplicates |
|--------------------|----------------------------------------------|-----------------------------|
| At-most-once       | May lose messages; never duplicates          | Fire-and-forget producers   |
| At-least-once      | No loss; may deliver same message 2+ times   | Retry on timeout/error      |
| Exactly-once       | No loss, no duplicates — complex to achieve  | Kafka transactions + idempotent producers |

Most production systems choose **at-least-once + consumer-side idempotency** because exactly-once is expensive and has edge cases.

### Idempotency keys

An **idempotency key** (also called deduplication ID) is a stable, unique identifier attached to an operation. The receiver stores processed keys and rejects re-deliveries of the same key.

```
Producer attaches:  event_id = "evt_<uuid4>"

Consumer logic:
  1. Check if event_id is in processed_ids store
  2. If yes  → ack immediately, skip processing
  3. If no   → process, then store event_id, then ack
```

The store must be **durable and consistent with the business operation** — ideally updated in the same DB transaction. Otherwise you get a race condition: crash between processing and storing the key → duplicate processed on restart.

### Deduplication window

Storing every event_id forever is impractical. Common approaches:

1. **TTL-based**: store key for N days (e.g. 7 days). Reject duplicates within that window. Accept that a retry after 8 days may be processed twice — acceptable if reprocessing is rare.
2. **Sliding bloom filter**: probabilistic, very low memory. False positives possible (reject a non-duplicate). False negatives impossible (never miss a real duplicate). Good for high-throughput pre-filter.
3. **Idempotent DB upsert**: `INSERT ... ON CONFLICT DO NOTHING` — DB uniqueness constraint acts as dedup store. Works beautifully when writing to a relational DB.

### Compensation design rules

1. **Compensation must be idempotent**: `PAYMENT_REVERSED` twice must not double-debit.
2. **Compensation must be total**: every forward step needs a defined compensation.
3. **Compensation order matters**: reverse the forward order (LIFO).
4. **Pivot transaction**: the step that cannot be compensated (e.g., SMS sent). Identify it early. Everything after the pivot must be idempotent and retryable.

```
Forward:      A → B → C (pivot) → D → E
Compensate:   E' → D'            (C can't be undone, B and A before pivot also need compensation)
              B' → A'
```

---

## Worked example

Below is a self-contained Python choreography saga with in-memory deduplication. No broker needed — uses a simple in-memory event bus.

```python
# choreography_saga.py
from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DomainEvent:
    event_id: str
    event_type: str
    aggregate_id: str
    payload: dict[str, Any]


# ---------------------------------------------------------------------------
# In-memory event bus with deduplication
# ---------------------------------------------------------------------------


class EventBus:
    """Simple pub/sub bus for choreography. Not thread-safe (demo only)."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[[DomainEvent], None]]] = {}
        self._processed: set[str] = set()  # deduplication store

    def subscribe(self, event_type: str, handler: Callable[[DomainEvent], None]) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    def publish(self, event: DomainEvent) -> None:
        if event.event_id in self._processed:
            logger.info("DUPLICATE skipped: %s %s", event.event_type, event.event_id)
            return
        self._processed.add(event.event_id)
        logger.info("EVENT: %s [%s]", event.event_type, event.aggregate_id)
        for handler in self._handlers.get(event.event_type, []):
            handler(event)


# ---------------------------------------------------------------------------
# Services (each service handles one step and publishes its result)
# ---------------------------------------------------------------------------


def make_payment_service(bus: EventBus) -> None:
    def handle(event: DomainEvent) -> None:
        order_id = event.aggregate_id
        # Simulate payment success
        result = DomainEvent(
            event_id=str(uuid.uuid4()),
            event_type="PAYMENT_AUTHORIZED",
            aggregate_id=order_id,
            payload={"amount": event.payload["amount"]},
        )
        bus.publish(result)

    bus.subscribe("ORDER_PLACED", handle)


def make_inventory_service(bus: EventBus) -> None:
    def handle_reserve(event: DomainEvent) -> None:
        order_id = event.aggregate_id
        result = DomainEvent(
            event_id=str(uuid.uuid4()),
            event_type="INVENTORY_RESERVED",
            aggregate_id=order_id,
            payload={"sku": "SKU-001", "qty": 1},
        )
        bus.publish(result)

    def handle_release(event: DomainEvent) -> None:
        order_id = event.aggregate_id
        logger.info("COMPENSATION: releasing inventory for order %s", order_id)

    bus.subscribe("PAYMENT_AUTHORIZED", handle_reserve)
    bus.subscribe("SHIPPING_FAILED", handle_release)


def make_shipping_service(bus: EventBus) -> None:
    def handle(event: DomainEvent) -> None:
        order_id = event.aggregate_id
        result = DomainEvent(
            event_id=str(uuid.uuid4()),
            event_type="SHIPMENT_SCHEDULED",
            aggregate_id=order_id,
            payload={"carrier": "DHL"},
        )
        bus.publish(result)

    bus.subscribe("INVENTORY_RESERVED", handle)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def run_saga() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    bus = EventBus()

    make_payment_service(bus)
    make_inventory_service(bus)
    make_shipping_service(bus)

    # Trigger the saga
    order_placed = DomainEvent(
        event_id=str(uuid.uuid4()),
        event_type="ORDER_PLACED",
        aggregate_id="order-999",
        payload={"amount": 49.99},
    )
    bus.publish(order_placed)

    # Simulate duplicate delivery (same event_id)
    bus.publish(order_placed)


if __name__ == "__main__":
    run_saga()
```

Expected output:
```
INFO  EVENT: ORDER_PLACED [order-999]
INFO  EVENT: PAYMENT_AUTHORIZED [order-999]
INFO  EVENT: INVENTORY_RESERVED [order-999]
INFO  EVENT: SHIPMENT_SCHEDULED [order-999]
INFO  DUPLICATE skipped: ORDER_PLACED <same-event-id>
```

---

## Exercises

1. **Saga failure path**: modify `make_payment_service` to fail 50% of the time (use `random.random() < 0.5`). Publish a `PAYMENT_FAILED` event on failure. Add a handler that publishes `ORDER_CANCELLED` when `PAYMENT_FAILED` is received. Trace which events fire on each run.

2. **Compensation chain**: add a `ShippingService` that fails and must compensate both `INVENTORY_RESERVED` and `PAYMENT_AUTHORIZED`. Implement the two compensating handlers. Verify via logging that compensation fires in reverse order.

3. **Idempotency key collision**: change `handle_reserve` in `make_inventory_service` to use a deterministic `event_id = f"inv-{event.aggregate_id}"` instead of a new UUID. Publish `PAYMENT_AUTHORIZED` twice with different `event_id`s but same `aggregate_id`. Confirm only one `INVENTORY_RESERVED` is emitted. Explain why this works — and what breaks if you tie the key to the aggregate instead of the causing event.

4. **Ordering invariant**: write a test that publishes 100 `ORDER_PLACED` events with sequential order IDs and asserts that for each order the downstream events arrive in the correct order (authorized before reserved, reserved before scheduled). Use a collector handler that appends `(aggregate_id, event_type)` tuples to a list.

5. **Orchestration rewrite**: rewrite the same 3-step flow as an orchestration saga: one `OrderOrchestrator` class that calls payment, inventory, and shipping directly (method calls, no bus) and handles failures with explicit compensation calls. Compare the code complexity and identify which error scenarios are easier to reason about in each model.

(Solutions: ../solutions/week-04/day-5.md)

---

## Self-check

- [ ] Can explain in one sentence why choreography-based sagas are harder to trace than orchestration-based sagas.
- [ ] Can draw the compensation order for a 4-step saga where step 3 is the pivot transaction.
- [ ] Can explain why idempotency keys must be stored atomically with the business operation.
- [ ] Can describe two deduplication strategies and their tradeoffs (TTL vs DB constraint).
- [ ] Knows the difference between "exactly-once delivery" and "at-least-once + idempotent consumer".
- [ ] Can identify which partition strategy preserves ordering for a given aggregate.
- [ ] Can explain what a "pivot transaction" is and why it changes compensation design.

---

## Supplementary links

- Chris Richardson — Saga Pattern: https://microservices.io/patterns/data/saga.html
- Gregor Hohpe — Enterprise Integration Patterns (messaging catalog): https://www.enterpriseintegrationpatterns.com/
- Kafka documentation — Exactly-once semantics: https://kafka.apache.org/documentation/#semantics
- Martin Fowler — Event Choreography vs Orchestration: https://martinfowler.com/articles/201701-event-driven.html
- Stripe idempotency keys design: https://stripe.com/docs/idempotency
