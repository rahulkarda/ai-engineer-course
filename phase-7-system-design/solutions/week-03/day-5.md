# Solutions — Phase 7 · Week 3 · Day 5 — CQRS, Denormalization & OLTP vs OLAP

---

## Exercise 1 — Workload Classification

**(a) Fraud detection pipeline**

**OLTP** for the real-time decision path. Fraud checks happen per transaction, at low latency (< 50 ms SLA). The system looks up narrow account or session state by primary key — a row-oriented store with indexed lookups is ideal. A secondary **OLAP-adjacent** component (a feature store or batch model retraining pipeline) runs separately on historical data.

**(b) Product catalog search**

Neither pure OLTP nor OLAP — this is a **search workload**, best served by an inverted-index engine (Elasticsearch, OpenSearch) or a search-optimized column store. If forced to choose between the two: closer to **OLAP** for full-text analytics, but catalogs also need low-latency point reads, so the typical answer is a dedicated search layer backed by a primary OLTP store.

**(c) Daily sales report**

**OLAP**. The query scans months of transaction history, groups by product/region/date, and aggregates totals. A column-store warehouse (BigQuery, Snowflake, Redshift) with a star schema is purpose-built for this. Running it on the live OLTP database would degrade transactional throughput.

**(d) User session state**

**OLTP** — specifically a key-value store (Redis, DynamoDB). Session reads and writes are narrow (one user's state), latency-sensitive (page load budget), and transient. Long-term analytical queries on session data (funnel analysis) would use a separate OLAP pipeline fed from a session event stream.

**(e) Social feed ranking**

A hybrid workload. The **write path** (ingest post events, fan-out to follower feeds) is OLTP-like: high-frequency, narrow writes. The **ranking/scoring** (ML model inference over feature vectors) can be precomputed in batch (OLAP) and cached. At Netflix/Twitter scale, a dedicated **feed store** (separate from both OLTP and OLAP) is used — a combination of a sorted-set cache (Redis ZSETs) and an eventual-consistency update pipeline.

**Summary table:**

| Workload | Best fit | Reasoning |
|---|---|---|
| Fraud detection (real-time) | OLTP | Narrow, low-latency, key-based |
| Product catalog search | Search engine | Inverted index; neither pure OLTP nor OLAP |
| Daily sales report | OLAP | Full-scan aggregation, historical data |
| User session state | OLTP / KV store | Narrow, transient, latency-critical |
| Social feed ranking | Hybrid (precomputed OLAP → OLTP cache) | Mixed read/write patterns |

---

## Exercise 2 — Dictionary Encoding Compression Math

**Setup:**
- Column: `status` with 3 distinct string values: `'pending'` (7 bytes), `'processing'` (10 bytes), `'complete'` (8 bytes)
- 1 000 000 rows
- Assume roughly uniform distribution (~333 333 rows per value)

### Raw VARCHAR(20) storage

UTF-8 storage: each value padded or stored with length prefix. Using 20 bytes per row (VARCHAR(20) max):

```
1_000_000 rows × 20 bytes = 20_000_000 bytes ≈ 19.1 MB
```

More precisely with actual lengths + 1-byte length prefix:

```
'pending'    → 8 bytes × 333_333 = 2_666_664 bytes
'processing' → 11 bytes × 333_333 = 3_666_663 bytes
'complete'   → 9 bytes × 333_333 = 2_999_997 bytes
Total ≈ 9_333_324 bytes ≈ 8.9 MB
```

### Dictionary-encoded storage

Build a dictionary:
```
dict = {0: 'pending', 1: 'processing', 2: 'complete'}
```

Dictionary size: 3 entries × avg ~9.3 bytes = ~28 bytes (negligible).

Each row stores only the dictionary key. With 3 values, 2 bits suffice; real implementations use 1 byte for simplicity:

```
1 byte per row × 1_000_000 = 1_000_000 bytes ≈ 0.95 MB
```

With bit-packing (2 bits per row):
```
2 bits × 1_000_000 / 8 = 250_000 bytes ≈ 244 KB
```

### Compression ratio

| Encoding | Size | Ratio vs VARCHAR(20) |
|---|---|---|
| VARCHAR(20) | 19.1 MB | 1× |
| Actual string lengths | 8.9 MB | 2.1× |
| Dictionary (1 byte/row) | ~1 MB | ~19× |
| Dictionary (bit-packed) | ~244 KB | ~78× |

**Answer:** Dictionary encoding achieves roughly **19–78× compression** depending on whether bit-packing is used. Real column stores (Parquet, ClickHouse, BigQuery) typically use bit-packing plus run-length encoding on top, reaching the high end.

**Why this matters:** OLAP queries that filter on `status = 'pending'` operate directly on the compact integer dictionary codes — the comparison is an integer comparison, not a string comparison, enabling SIMD vectorization and L1 cache efficiency.

---

## Exercise 3 — `OrderSummaryByCustomer` Read Model

Full implementation extending the lesson's CQRS order service.

```python
# cqrs_orders_extended.py
"""
Extension of cqrs_orders.py with an OrderSummaryByCustomer read model.

Subscribes to OrderPlaced events and maintains per-customer
order count and total spend.

Run:  pytest test_cqrs_orders_extended.py
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain types (same as lesson)
# ---------------------------------------------------------------------------

class OrderStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    CANCELLED = "cancelled"


@dataclass
class Customer:
    customer_id: str
    name: str
    email: str


@dataclass
class OrderLine:
    product_id: str
    product_name: str
    quantity: int
    unit_price: float

    @property
    def line_total(self) -> float:
        return self.quantity * self.unit_price


# ---------------------------------------------------------------------------
# Domain Events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrderPlaced:
    event_id: str
    order_id: str
    customer_id: str
    lines: tuple[OrderLine, ...]
    occurred_at: datetime


@dataclass(frozen=True)
class OrderStatusChanged:
    event_id: str
    order_id: str
    new_status: OrderStatus
    occurred_at: datetime


DomainEvent = OrderPlaced | OrderStatusChanged


# ---------------------------------------------------------------------------
# Write Model
# ---------------------------------------------------------------------------

@dataclass
class Order:
    order_id: str
    customer_id: str
    lines: list[OrderLine]
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def confirm(self) -> OrderStatusChanged:
        if self.status != OrderStatus.PENDING:
            raise ValueError(f"Cannot confirm order in status {self.status}")
        self.status = OrderStatus.CONFIRMED
        return OrderStatusChanged(
            event_id=str(uuid.uuid4()),
            order_id=self.order_id,
            new_status=OrderStatus.CONFIRMED,
            occurred_at=datetime.now(timezone.utc),
        )

    @property
    def total(self) -> float:
        return sum(line.line_total for line in self.lines)


class WriteStore:
    def __init__(self) -> None:
        self._orders: dict[str, Order] = {}
        self._customers: dict[str, Customer] = {}

    def save_customer(self, customer: Customer) -> None:
        self._customers[customer.customer_id] = customer

    def save_order(self, order: Order) -> None:
        self._orders[order.order_id] = order

    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def get_customer(self, customer_id: str) -> Optional[Customer]:
        return self._customers.get(customer_id)


# ---------------------------------------------------------------------------
# Event Bus
# ---------------------------------------------------------------------------

EventHandler = Callable[[DomainEvent], None]


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[type, list[EventHandler]] = {}

    def subscribe(self, event_type: type, handler: EventHandler) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    def publish(self, event: DomainEvent) -> None:
        for handler in self._handlers.get(type(event), []):
            try:
                handler(event)
            except Exception:
                log.exception("Event handler failed for %s", type(event).__name__)


# ---------------------------------------------------------------------------
# NEW: OrderSummaryByCustomer Read Model
# ---------------------------------------------------------------------------

@dataclass
class CustomerOrderSummary:
    """Denormalized per-customer aggregate maintained by the read model."""

    customer_id: str
    order_count: int = 0
    total_spend: float = 0.0


class CustomerOrderReadStore:
    """In-memory store for per-customer order aggregates."""

    def __init__(self) -> None:
        self._by_customer: dict[str, CustomerOrderSummary] = {}

    def get(self, customer_id: str) -> Optional[CustomerOrderSummary]:
        return self._by_customer.get(customer_id)

    def upsert(self, summary: CustomerOrderSummary) -> None:
        self._by_customer[summary.customer_id] = summary

    def all(self) -> list[CustomerOrderSummary]:
        return list(self._by_customer.values())

    def clear(self) -> None:
        """Wipe the store — used before a full event replay."""
        self._by_customer.clear()


class OrderSummaryByCustomerBuilder:
    """
    Read model builder that maintains per-customer order count and total spend.

    Subscribes to OrderPlaced events only. OrderStatusChanged events do not
    affect aggregate totals (cancellations are a possible extension).

    This projection is idempotent-by-design: replaying the same OrderPlaced
    event twice would double-count. In production, track processed event_ids
    to de-duplicate. For this exercise, we assume each event appears once.
    """

    def __init__(
        self,
        write_store: WriteStore,
        customer_read_store: CustomerOrderReadStore,
        bus: EventBus,
    ) -> None:
        self._write = write_store
        self._store = customer_read_store
        bus.subscribe(OrderPlaced, self._on_order_placed)

    def _on_order_placed(self, event: DomainEvent) -> None:
        assert isinstance(event, OrderPlaced)
        order = self._write.get_order(event.order_id)
        if order is None:
            log.error(
                "CustomerReadModel: order %s not found in write store",
                event.order_id,
            )
            return

        existing = self._store.get(event.customer_id) or CustomerOrderSummary(
            customer_id=event.customer_id
        )
        existing.order_count += 1
        existing.total_spend += order.total
        self._store.upsert(existing)
        log.info(
            "CustomerReadModel: customer %s — %d orders, total=%.2f",
            event.customer_id,
            existing.order_count,
            existing.total_spend,
        )

    def rebuild_from_events(self, events: list[DomainEvent]) -> None:
        """
        Replay a historical event log to reconstruct the read store.

        Clears the store first, then replays events in order.
        Only OrderPlaced events affect this read model.
        """
        self._store.clear()
        log.info(
            "CustomerReadModel: rebuilding from %d events", len(events)
        )
        for event in events:
            if isinstance(event, OrderPlaced):
                self._on_order_placed(event)
        log.info("CustomerReadModel: rebuild complete")


# ---------------------------------------------------------------------------
# Command Handler (abbreviated — only place_order needed for tests)
# ---------------------------------------------------------------------------

class OrderCommandHandler:
    def __init__(self, write_store: WriteStore, bus: EventBus) -> None:
        self._store = write_store
        self._bus = bus

    def place_order(
        self,
        customer_id: str,
        lines: list[OrderLine],
    ) -> tuple[str, DomainEvent]:
        """Returns (order_id, OrderPlaced event) for replay testing."""
        order_id = str(uuid.uuid4())
        order = Order(order_id=order_id, customer_id=customer_id, lines=lines)
        self._store.save_order(order)
        event = OrderPlaced(
            event_id=str(uuid.uuid4()),
            order_id=order_id,
            customer_id=customer_id,
            lines=tuple(lines),
            occurred_at=order.created_at,
        )
        self._bus.publish(event)
        log.info("Command: placed order %s", order_id)
        return order_id, event
```

**Test file: `test_cqrs_orders_extended.py`**

```python
"""
pytest tests for OrderSummaryByCustomer read model.
"""
import pytest
from cqrs_orders_extended import (
    Customer,
    CustomerOrderReadStore,
    EventBus,
    OrderCommandHandler,
    OrderLine,
    OrderPlaced,
    OrderSummaryByCustomerBuilder,
    WriteStore,
)


def _build_system():
    write_store = WriteStore()
    customer_store = CustomerOrderReadStore()
    bus = EventBus()

    alice = Customer("cust-001", "Alice", "alice@example.com")
    bob = Customer("cust-002", "Bob", "bob@example.com")
    write_store.save_customer(alice)
    write_store.save_customer(bob)

    builder = OrderSummaryByCustomerBuilder(write_store, customer_store, bus)
    commands = OrderCommandHandler(write_store, bus)
    return commands, customer_store, builder


def test_single_order_increments_count_and_spend():
    commands, store, _ = _build_system()
    commands.place_order(
        "cust-001",
        [OrderLine("p1", "Widget", 2, 10.00)],
    )
    summary = store.get("cust-001")
    assert summary is not None
    assert summary.order_count == 1
    assert summary.total_spend == pytest.approx(20.00)


def test_multiple_orders_accumulate():
    commands, store, _ = _build_system()
    commands.place_order("cust-001", [OrderLine("p1", "A", 1, 50.00)])
    commands.place_order("cust-001", [OrderLine("p2", "B", 1, 30.00)])
    commands.place_order("cust-001", [OrderLine("p3", "C", 1, 20.00)])
    summary = store.get("cust-001")
    assert summary.order_count == 3
    assert summary.total_spend == pytest.approx(100.00)


def test_different_customers_are_independent():
    commands, store, _ = _build_system()
    commands.place_order("cust-001", [OrderLine("p1", "A", 1, 100.00)])
    commands.place_order("cust-002", [OrderLine("p2", "B", 1, 50.00)])
    assert store.get("cust-001").order_count == 1
    assert store.get("cust-002").order_count == 1
    assert store.get("cust-001").total_spend == pytest.approx(100.00)
    assert store.get("cust-002").total_spend == pytest.approx(50.00)


def test_rebuild_from_events_restores_state():
    commands, store, builder = _build_system()

    # Place orders and capture events
    _, evt1 = commands.place_order(
        "cust-001", [OrderLine("p1", "A", 1, 40.00)]
    )
    _, evt2 = commands.place_order(
        "cust-001", [OrderLine("p2", "B", 2, 15.00)]
    )
    _, evt3 = commands.place_order(
        "cust-002", [OrderLine("p3", "C", 1, 99.00)]
    )

    # Snapshot state before clearing
    pre_clear_alice = store.get("cust-001")
    pre_clear_bob = store.get("cust-002")
    assert pre_clear_alice.order_count == 2
    assert pre_clear_bob.order_count == 1

    # Clear and rebuild
    store.clear()
    assert store.get("cust-001") is None  # wiped

    builder.rebuild_from_events([evt1, evt2, evt3])

    # Post-rebuild state must match pre-clear state
    post_alice = store.get("cust-001")
    post_bob = store.get("cust-002")
    assert post_alice.order_count == pre_clear_alice.order_count
    assert post_alice.total_spend == pytest.approx(pre_clear_alice.total_spend)
    assert post_bob.order_count == pre_clear_bob.order_count
    assert post_bob.total_spend == pytest.approx(pre_clear_bob.total_spend)


def test_rebuild_from_empty_events_clears_store():
    commands, store, builder = _build_system()
    commands.place_order("cust-001", [OrderLine("p1", "A", 1, 10.00)])
    builder.rebuild_from_events([])  # replay nothing
    assert store.get("cust-001") is None
```

---

## Exercise 4 — Denormalization Consistency: Category Name Update

**Problem:** Product documents embed `category_name` inline (e.g., `{product_id: "p1", name: "Keyboard", category_name: "Electronics"}`). The category "Electronics" is renamed to "Consumer Electronics". How do you propagate this change?

### Approach 1 — Synchronous Bulk Update (Write-Through Fan-Out)

When the category record is updated, immediately issue an update to all product documents in that category:

```sql
UPDATE products SET category_name = 'Consumer Electronics'
WHERE category_id = 42;
```

**In a document store (MongoDB):**
```python
db.products.update_many(
    {"category_id": 42},
    {"$set": {"category_name": "Consumer Electronics"}}
)
```

**Trade-offs:**
- Consistency: strong — all documents updated atomically (if your DB supports it) or near-simultaneously.
- Write cost: O(number of products in category) — potentially millions of writes for large categories.
- Latency: the category rename blocks until all product documents are updated. Unacceptable for large datasets without a background job.
- Risk: if the update process crashes mid-way, the data is transiently inconsistent (some products show old name, some show new name).

Mitigation: run the fan-out asynchronously via a background job with idempotent writes, and keep `category_id` as the source of truth so readers can join if needed.

### Approach 2 — Lazy Update via Reference + Cache (Hybrid Normalization)

Store both the `category_id` (foreign key) and the `category_name` (denormalized) in product documents. Serve reads using the cached `category_name`. When the category name changes:

1. Update the `categories` table/collection immediately.
2. Mark the category as "dirty" (e.g., set `last_updated_at`).
3. A background reconciler sweeps product documents in that category and updates `category_name` over time.
4. Until the background job completes, reads still return the stale name from the embedded field — tolerable if slight staleness is acceptable.

**Trade-offs:**
- Eventual consistency: accepted explicitly.
- No blocking write fan-out at rename time.
- Requires background reconciliation infrastructure.
- Read can optionally bypass the embedded name and look up the category directly if freshness is critical.

**Event sourcing note:** With event sourcing, events are immutable. The `OrderPlaced` event that recorded `category_name = "Electronics"` cannot change. This is intentional — it's an accurate historical record of what the category was called when the order was placed. Current reads that need the *current* category name should look it up from the categories service, not rely on the historical event. The embedded name in events serves as an audit record, not a live cache.

---

## Exercise 5 — ETL vs ELT: Adding a New Transformation to Historical Data

**Setup:** 10 GB/day of raw JSON clickstream → clean Parquet files in BigQuery.

### ETL approach (Python + Pandas)

```
Raw JSON (S3/GCS)
       ↓  Python pipeline reads, transforms, validates
Clean Parquet
       ↓  Load into BigQuery
BigQuery table (only clean data)
```

To add a new transformation to historical data, you must:
1. Re-run the entire Python pipeline over all historical raw JSON files.
2. Re-generate all Parquet files.
3. Re-load everything into BigQuery.

**Problem:** The raw JSON may no longer be available if it was deleted after loading (common cost-saving practice). Even if it is available, re-processing months of data through a Python job is slow and operationally expensive.

### ELT approach (raw JSON → BigQuery → dbt)

```
Raw JSON (GCS)
       ↓  Load as-is (no transformation)
BigQuery raw table (JSON or nested columns)
       ↓  dbt models run SQL transformations
BigQuery cleaned/modeled tables
```

To add a new transformation:
1. Write a new dbt model (a SQL `SELECT` statement).
2. Run `dbt run` — BigQuery executes the SQL over the already-loaded raw data.
3. Historical raw data is always available in BigQuery; no re-ingestion needed.

### Decision

**ELT makes adding new transformations to historical data dramatically easier.**

The raw data stays in the warehouse. Adding a transformation is just adding a SQL file and running `dbt run`. With ETL, historical re-processing requires re-running code over external files that may be large, slow, or no longer present.

| Criterion | ETL | ELT |
|---|---|---|
| Adding new transformation to history | Hard (re-process raw files) | Easy (new dbt model, run on existing raw data) |
| Analyst access to raw data | Delayed (after transformation) | Immediate (raw table always present) |
| Operational complexity | Higher (maintain Python pipeline + Pandas env) | Lower (dbt + BigQuery; SQL-native) |
| Compute cost | Python/Pandas cluster | BigQuery slot-hours (elastic) |
| Schema evolution | Rigid (pre-defined output schema) | Flexible (raw schema preserved, transforms layered) |

**Recommendation for this use case:** ELT. Modern cloud warehouses with elastic compute make in-warehouse transformation cheap and fast. The ability to answer ad-hoc analytical questions directly on raw data — without waiting for a pipeline re-run — is a decisive advantage.
