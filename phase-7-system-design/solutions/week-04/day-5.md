# Solutions — Week 04 · Day 5

## Exercise 1: Choreography-based Order Saga

```python
"""
choreography_saga.py

Choreography-based Order saga using an in-memory EventBus.
Events flow: OrderPlaced -> PaymentProcessed|PaymentFailed -> InventoryReserved|OrderCancelled
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrderPlaced:
    order_id: str
    customer_id: str
    amount: float
    items: list[dict[str, Any]]


@dataclass(frozen=True)
class PaymentProcessed:
    order_id: str
    payment_id: str
    amount: float


@dataclass(frozen=True)
class PaymentFailed:
    order_id: str
    reason: str


@dataclass(frozen=True)
class InventoryReserved:
    order_id: str
    reservation_id: str


@dataclass(frozen=True)
class OrderCancelled:
    order_id: str
    reason: str


# ---------------------------------------------------------------------------
# In-memory EventBus
# ---------------------------------------------------------------------------

type EventHandler = Callable[[Any], None]


class EventBus:
    """Synchronous in-memory event bus."""

    def __init__(self) -> None:
        self._handlers: dict[type, list[EventHandler]] = {}

    def subscribe(self, event_type: type, handler: EventHandler) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    def publish(self, event: Any) -> None:
        event_type = type(event)
        logger.info("EventBus.publish: %s", event_type.__name__)
        for handler in self._handlers.get(event_type, []):
            handler(event)


# ---------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------

class PaymentService:
    """Reacts to OrderPlaced. Emits PaymentProcessed or PaymentFailed."""

    def __init__(self, bus: EventBus, *, fail_order_ids: set[str] | None = None) -> None:
        self._bus = bus
        self._fail_order_ids: set[str] = fail_order_ids or set()
        bus.subscribe(OrderPlaced, self.handle_order_placed)

    def handle_order_placed(self, event: OrderPlaced) -> None:
        logger.info("PaymentService: processing payment for order %s", event.order_id)
        if event.order_id in self._fail_order_ids:
            self._bus.publish(PaymentFailed(
                order_id=event.order_id,
                reason="Insufficient funds",
            ))
        else:
            self._bus.publish(PaymentProcessed(
                order_id=event.order_id,
                payment_id=str(uuid.uuid4()),
                amount=event.amount,
            ))


class InventoryService:
    """Reacts to PaymentProcessed. Emits InventoryReserved."""

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        bus.subscribe(PaymentProcessed, self.handle_payment_processed)

    def handle_payment_processed(self, event: PaymentProcessed) -> None:
        logger.info("InventoryService: reserving stock for order %s", event.order_id)
        self._bus.publish(InventoryReserved(
            order_id=event.order_id,
            reservation_id=str(uuid.uuid4()),
        ))


class OrderService:
    """Reacts to PaymentFailed. Emits OrderCancelled (compensation)."""

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        bus.subscribe(PaymentFailed, self.handle_payment_failed)

    def handle_payment_failed(self, event: PaymentFailed) -> None:
        logger.warning("OrderService: cancelling order %s — %s", event.order_id, event.reason)
        self._bus.publish(OrderCancelled(
            order_id=event.order_id,
            reason=event.reason,
        ))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def run_demo() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # --- Happy path ---
    print("\n=== Happy path ===")
    bus = EventBus()
    results: list[Any] = []

    bus.subscribe(InventoryReserved, results.append)
    bus.subscribe(OrderCancelled, results.append)

    PaymentService(bus)
    InventoryService(bus)
    OrderService(bus)

    bus.publish(OrderPlaced(
        order_id="ord-001",
        customer_id="cust-42",
        amount=99.99,
        items=[{"sku": "WIDGET-A", "qty": 2}],
    ))

    assert len(results) == 1
    assert isinstance(results[0], InventoryReserved)
    assert results[0].order_id == "ord-001"
    print(f"Result: {results[0]}")

    # --- Compensation path ---
    print("\n=== Compensation path ===")
    bus2 = EventBus()
    results2: list[Any] = []

    bus2.subscribe(InventoryReserved, results2.append)
    bus2.subscribe(OrderCancelled, results2.append)

    PaymentService(bus2, fail_order_ids={"ord-002"})
    InventoryService(bus2)
    OrderService(bus2)

    bus2.publish(OrderPlaced(
        order_id="ord-002",
        customer_id="cust-99",
        amount=250.00,
        items=[{"sku": "GADGET-X", "qty": 1}],
    ))

    assert len(results2) == 1
    assert isinstance(results2[0], OrderCancelled)
    assert results2[0].order_id == "ord-002"
    print(f"Result: {results2[0]}")

    print("\nAll assertions passed.")


if __name__ == "__main__":
    run_demo()
```

**Key design points:**
- Services are decoupled — each only knows about the EventBus interface.
- Compensation (OrderCancelled) is triggered by the same choreography pattern, no central coordinator.
- `fail_order_ids` set lets tests inject failures without mocking.

---

## Exercise 2: Idempotency Key Handler

```python
"""
idempotency_handler.py

Idempotency key handler that ensures a command is processed at most once.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Command:
    idempotency_key: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class Result:
    success: bool
    value: Any


class CommandHandler:
    """Wraps a processor function with idempotency semantics."""

    def __init__(self) -> None:
        self._seen: dict[str, Result] = {}
        self.call_count: int = 0

    def handle(self, command: Command) -> Result:
        if command.idempotency_key in self._seen:
            logger.info(
                "CommandHandler: duplicate key=%s — returning cached result",
                command.idempotency_key,
            )
            return self._seen[command.idempotency_key]

        logger.info("CommandHandler: processing key=%s", command.idempotency_key)
        self.call_count += 1
        result = self._process(command)
        self._seen[command.idempotency_key] = result
        return result

    def _process(self, command: Command) -> Result:
        # Simulates real work
        total = sum(command.payload.values()) if command.payload else 0
        return Result(success=True, value=total)


# ---------------------------------------------------------------------------
# Tests (runnable with pytest)
# ---------------------------------------------------------------------------

def test_handler_called_exactly_once_on_duplicate() -> None:
    handler = CommandHandler()
    cmd = Command(idempotency_key="key-abc", payload={"amount": 100})

    result1 = handler.handle(cmd)
    result2 = handler.handle(cmd)

    assert result1 == result2
    assert handler.call_count == 1, f"Expected 1 call, got {handler.call_count}"


def test_different_keys_both_processed() -> None:
    handler = CommandHandler()
    cmd_a = Command(idempotency_key="key-1", payload={"amount": 10})
    cmd_b = Command(idempotency_key="key-2", payload={"amount": 20})

    handler.handle(cmd_a)
    handler.handle(cmd_b)

    assert handler.call_count == 2


def test_cached_result_is_identical() -> None:
    handler = CommandHandler()
    cmd = Command(idempotency_key="key-xyz", payload={"x": 5, "y": 3})

    r1 = handler.handle(cmd)
    r2 = handler.handle(cmd)

    assert r1.value == 8
    assert r1 is r2  # same object — no re-computation


def test_many_duplicates_single_execution() -> None:
    handler = CommandHandler()
    cmd = Command(idempotency_key="bulk-key", payload={"n": 42})

    results = [handler.handle(cmd) for _ in range(100)]

    assert handler.call_count == 1
    assert all(r == results[0] for r in results)


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-v"]))
```

**Key design points:**
- O(1) lookup per command via dict.
- `call_count` is an observable side-effect used in tests without mocking.
- Returning the exact same `Result` object on cache hit avoids re-serialization costs in real systems.

---

## Exercise 3: Partition-Key-Based Ordering

```python
"""
partition_assignment.py

Consistent, deterministic partition assignment by key using hash % N.
Demonstrates uniform distribution across keys.
"""
from __future__ import annotations

import random
import string
from collections import Counter


def assign_partition(key: str, num_partitions: int) -> int:
    """
    Assign a string key to a partition deterministically.

    Uses Python's built-in hash with a fixed seed via hashlib for reproducibility
    across processes (Python's hash is randomized by default via PYTHONHASHSEED).
    """
    import hashlib
    digest = int(hashlib.sha256(key.encode()).hexdigest(), 16)
    return digest % num_partitions


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_same_key_always_same_partition() -> None:
    key = "user-12345"
    partitions = assign_partition(key, 16)
    for _ in range(1000):
        assert assign_partition(key, 16) == partitions, "Non-deterministic assignment"


def test_different_num_partitions_gives_valid_range() -> None:
    for n in [1, 4, 8, 16, 32, 64, 128]:
        for key in ["alpha", "beta", "gamma", "delta"]:
            p = assign_partition(key, n)
            assert 0 <= p < n, f"Partition {p} out of range [0, {n})"


def test_distribution_is_roughly_uniform() -> None:
    """
    Generate 1 000 random keys and verify no partition gets more than 3x
    the expected share (expected = 1000 / num_partitions per bucket).
    """
    num_partitions = 10
    num_keys = 1000
    rng = random.Random(42)
    keys = [
        "".join(rng.choices(string.ascii_lowercase + string.digits, k=12))
        for _ in range(num_keys)
    ]
    counts = Counter(assign_partition(k, num_partitions) for k in keys)

    expected = num_keys / num_partitions  # 100 per bucket
    for partition, count in counts.items():
        assert count < expected * 3, (
            f"Partition {partition} has {count} keys — skew too high "
            f"(expected ~{expected:.0f})"
        )
    # Verify all partitions received at least one key
    assert len(counts) == num_partitions


def test_ordering_guarantee_same_partition() -> None:
    """
    Verify that messages for the same entity key always land in the same
    partition, which guarantees ordering within that partition.
    """
    entity_id = "order-9999"
    assigned = assign_partition(entity_id, 32)
    # Simulate 50 messages for the same entity
    for _ in range(50):
        assert assign_partition(entity_id, 32) == assigned


def show_distribution(num_partitions: int = 10, num_keys: int = 1000) -> None:
    rng = random.Random(0)
    keys = [
        "".join(rng.choices(string.ascii_lowercase + string.digits, k=10))
        for _ in range(num_keys)
    ]
    counts = Counter(assign_partition(k, num_partitions) for k in keys)
    print(f"\nDistribution across {num_partitions} partitions ({num_keys} keys):")
    for p in range(num_partitions):
        bar = "#" * (counts[p] // 5)
        print(f"  Partition {p:2d}: {counts[p]:4d}  {bar}")


if __name__ == "__main__":
    show_distribution()
    import pytest, sys
    sys.exit(pytest.main([__file__, "-v"]))
```

**Key design points:**
- `hashlib.sha256` instead of Python's built-in `hash()` — `hash()` is randomized per process via `PYTHONHASHSEED`, making it unsuitable for consistent routing across restarts or processes.
- The modulo approach mirrors how Kafka computes `DefaultPartitioner`.
- Ordering guarantee: all messages for the same key land in the same partition, so partition-level FIFO is preserved.

---

## Discussion Notes

**Choreography vs orchestration (when to choose each):**
- Choreography excels when services are truly independent and the event flow is simple. Adding a new service means subscribing to an event — no central change required.
- Choreography becomes hard to reason about as the number of event types grows (distributed saga anti-pattern: "event spaghetti"). Use an orchestrator (e.g., a Saga orchestrator or a LangGraph subgraph) when you need explicit visibility into saga state.

**Idempotency key placement:**
- In HTTP APIs: `Idempotency-Key` header (Stripe pattern).
- In Kafka consumers: store the key in a compacted offsets topic; deduplication window must exceed max redelivery interval.
- UUID v7 (time-ordered) is preferable to UUID v4 for keys that need chronological sorting in audit logs.

**Partition count planning:**
- Increasing partition count is a one-way operation in Kafka — keys remap, breaking ordering guarantees for in-flight messages.
- Rule of thumb: partition count = max(throughput / per_partition_throughput, desired_parallelism). Start conservative; over-partitioning increases broker and consumer overhead.
- Consumer lag target: partitions >= consumer count for the hottest consumer group.
