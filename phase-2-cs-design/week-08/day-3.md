# Phase 2 · Week 08 · Day 3 — Microservices vs Monolith

> Objective: Apply bounded-context thinking to decide when decomposition is warranted, draw defensible service boundaries, and execute the strangler fig migration pattern.  ·  Time: ~1 hr  ·  Prereqs: [[week-08/day-1]], [[week-08/day-2]]

---

## Concept

### The Monolith Is Not Evil

A monolith is a single deployable unit where all modules share one process and one database.  
That is not an anti-pattern — it is the correct starting point for nearly every system.

Microservices are a solution to **organisational and operational scale problems**.  
If you don't have those problems, you pay the distributed-systems tax for nothing.

The default should be: **build a monolith until you have a concrete reason to split**.

---

### Decomposition Criteria

Split a monolith when you have **at least one** of these:

#### 1. Independent deployment velocity

Teams block each other merging to the same codebase and deploying the same artefact.  
Rule of thumb: when the cost of coordination between two groups exceeds the cost of an async API, split.

#### 2. Independent scaling requirements

The search function needs 200 replicas during peak. The admin panel needs 1.  
Running them together wastes resources and creates blast-radius coupling.

#### 3. Technology isolation

A real-time inference service needs Python + GPU.  
The billing service needs strict SQL transactions.  
Forcing them into one process means one dependency tree, one runtime, one build.

#### 4. Fault isolation requirements

A crash in the recommendations engine must not take down the checkout flow.  
Process isolation (separate containers) is the only reliable fault boundary.

#### 5. Regulatory isolation

PCI-DSS scope for card processing must be minimised.  
Running payment code alongside unrelated services expands audit scope unnecessarily.

Do NOT split because:
- The team read a blog post about Netflix.
- "Microservices are the future."
- The codebase is large (modules and packages solve this).
- Different people own different features (ownership works inside a monolith too).

---

### Bounded Contexts and DDD Lite

**Domain-Driven Design (DDD)** provides the language to identify service boundaries.  
You don't need the full DDD toolkit — these three concepts are enough.

#### Bounded Context

A **bounded context** is a region of the domain where a model is internally consistent  
and terms mean the same thing.

The word "Order" means different things in:
- **Sales**: a customer request with a price and a discount code.
- **Warehouse**: a pick-list with SKUs, bin locations, and weight.
- **Finance**: a revenue recognition event with a journal entry.

These are three different bounded contexts.  
Each should own its data model, its validation rules, and its invariants.  
They communicate across boundaries via events or APIs — never by sharing a table.

#### Ubiquitous Language

Every bounded context has a vocabulary shared by engineers and domain experts.  
"Invoice", "Payment", "Reconciliation" mean specific things in Finance.  
If you have to say "the order, I mean the financial order, not the warehouse order", your context is bleeding.

#### Aggregate

An **aggregate** is a cluster of domain objects treated as a single unit for consistency.  
An `Order` aggregate owns its `OrderLine` items — you never modify `OrderLine` directly, only through `Order`.  
The aggregate root (`Order`) is the only entry point for mutations.

**Rule**: one aggregate root owns the transaction boundary.  
If two aggregates need to change in one transaction, they probably belong in the same bounded context.

#### Practical service boundary heuristics

| Signal | Split | Merge |
|--------|-------|-------|
| Separate teams with independent roadmaps | yes | |
| Different data models for the same noun | yes | |
| Single transaction spanning both | | yes |
| Most API calls are cross-service | | yes |
| One service reads the other's tables directly | | yes |
| One team maintains both | | yes |

---

### When NOT to Split

#### The distributed monolith anti-pattern

Services that are technically separate processes but require synchronous calls across 5 services  
to serve a single user request are a **distributed monolith** — all the operational complexity of  
microservices with none of the isolation benefits.

Warning signs:
- Service A cannot deploy without also deploying service B (shared DB schema).
- Tests for service A require service B, C, and D to be running.
- A single business operation touches more than 2 services in a synchronous chain.
- Services share a database (even different schemas in the same cluster).

#### The two-pizza rule is not a technical rule

Amazon's "two-pizza team" is about communication overhead, not service granularity.  
A team of 6 can own a well-structured monolith with 10 modules.  
The team size does not determine the service count.

#### Latency math

A monolith function call is ~10 ns.  
An in-datacenter RPC (gRPC, same AZ) is ~500 µs–2 ms.  
A cross-AZ HTTP call is ~5–20 ms.

If your hot path calls 10 services, you've added up to 200 ms of pure network overhead.  
Batch operations and fan-outs are far worse.

**Rule**: synchronous call chains deeper than 2–3 hops are a decomposition smell.

---

### Strangler Fig Pattern

The **strangler fig** is the standard pattern for migrating from a monolith to services  
without a big-bang rewrite.

Named after the strangler fig tree, which grows around an existing tree until the  
original is completely replaced.

#### Steps

```
1. IDENTIFY — pick one bounded context with clear, stable API surface
2. BUILD alongside — implement the new service; don't touch the monolith yet
3. INTERCEPT — route new traffic to the new service (feature flag, API gateway rule)
4. VALIDATE — shadow-mode: run both, compare results, fix divergence
5. MIGRATE — flip 100% traffic to new service
6. REMOVE — delete the corresponding code from the monolith
```

#### Key constraints for success

- The monolith must become the **anti-corruption layer** during transition:  
  it translates between the old internal model and the new service's API.
- Never share the database between old and new service.  
  If both need the same data, the new service owns it; the monolith calls the service.
- The migration is done when the monolith no longer contains the code AND no longer owns the data.

#### Feature flags and routing

```python
# API gateway / proxy rule (pseudocode)
if feature_flag("use_new_inventory_service") and request.path.startswith("/inventory"):
    proxy_to("http://inventory-service:8001")
else:
    proxy_to("http://monolith:8000")
```

Start at 1% → 10% → 50% → 100%.  
Roll back is instant: flip the flag.  
Shadow mode: send to both, discard new service response, compare in a background job.

---

## How it works internally

### Database ownership and the shared-DB trap

The #1 reason strangler figs fail: the new service reads or writes the monolith's database.

Why it fails:
- The monolith's schema is the monolith's internal API. It changes without notice.
- The new service becomes coupled to the monolith's migration schedule.
- You cannot scale the DB independently.

Fix: **data migration must accompany service extraction**.

```
Phase 1: new service has its own DB; monolith is the system of record
         monolith writes both → monolith DB + publishes event → new service consumes event → new service DB

Phase 2: new service is the system of record; monolith is read-only
         new service writes → publishes event → monolith consumes event → monolith DB (read replica logic)

Phase 3: monolith no longer owns this data; drop old tables
```

### Event-driven vs synchronous decomposition

Two options for cross-service communication:

**Synchronous (HTTP/gRPC)**: A calls B and waits.  
Good for: reads, real-time queries, when you need B's response to continue.  
Bad for: writes (tight coupling, cascading failures, temporal coupling).

**Asynchronous (message queue, Kafka, SNS/SQS)**: A publishes event, B consumes later.  
Good for: writes, fan-out, decoupling deployment schedules.  
Bad for: when A needs an immediate answer from B.

**Rule for writes**: prefer events. "OrderPlaced" is better than `POST /warehouse/pick-orders`.

### Module boundaries inside a monolith

Before you split, make the monolith's internals look like services:

```
src/
  orders/
    api.py          # HTTP layer — knows nothing about other domains
    service.py      # Business logic — no DB imports
    repository.py   # DB access — no HTTP imports
    models.py       # Pydantic models scoped to this domain
  inventory/
    api.py
    service.py
    repository.py
    models.py
```

Each domain communicates with others only through their `service.py` interface.  
No cross-domain direct DB access.  
When you split, the `service.py` boundary becomes the API contract with minimal changes.

---

## Worked example

```python
"""
week_08_day3_strangler_fig.py

Demonstrates:
- Modular monolith structure: orders and inventory as internal bounded contexts
- Anti-corruption layer: orders domain translates to/from inventory domain
- Strangler fig routing: feature-flag-controlled proxy to new inventory service
- Event publishing: orders emits domain events consumed by inventory

This is a simulation — no real HTTP calls or message queues.
In production, replace the stubs with actual FastAPI/aiohttp/Kafka clients.

Run:
    pip install fastapi uvicorn pydantic
    uvicorn week_08_day3_strangler_fig:app --reload
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, field_validator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature flags — controls strangler fig routing
# ---------------------------------------------------------------------------
# In production: read from LaunchDarkly, AWS AppConfig, or a config table.
# Here: read from env so tests can flip the flag without code changes.

def feature_flag(name: str) -> bool:
    return os.getenv(f"FEATURE_{name.upper()}", "false").lower() == "true"


# ---------------------------------------------------------------------------
# Domain events — the contract between bounded contexts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DomainEvent:
    event_id: str
    occurred_at: datetime
    aggregate_id: str


@dataclass(frozen=True)
class OrderPlaced(DomainEvent):
    """Published when an order is confirmed. Inventory consumes this."""

    user_id: int
    items: list[dict[str, int]]  # [{"sku": "A1", "quantity": 2}]


# Simple in-memory event bus (replace with Kafka/SNS in production)
_event_handlers: dict[type, list] = {}


def subscribe(event_type: type):
    def decorator(fn):
        _event_handlers.setdefault(event_type, []).append(fn)
        return fn
    return decorator


def publish(event: DomainEvent) -> None:
    log.info("publishing event type=%s id=%s", type(event).__name__, event.event_id)
    for handler in _event_handlers.get(type(event), []):
        handler(event)


# ---------------------------------------------------------------------------
# INVENTORY BOUNDED CONTEXT — legacy (inside monolith)
# ---------------------------------------------------------------------------


class InventoryItem(BaseModel):
    sku: str
    available: int


# Simulated inventory store
_inventory: dict[str, int] = {
    "A1": 100,
    "B2": 50,
    "C3": 0,  # out of stock
}


class LegacyInventoryService:
    """The old inventory service inside the monolith.

    This is what the strangler fig will eventually replace.
    """

    def check_availability(self, sku: str, quantity: int) -> bool:
        available = _inventory.get(sku, 0)
        return available >= quantity

    def reserve(self, sku: str, quantity: int) -> bool:
        if not self.check_availability(sku, quantity):
            return False
        _inventory[sku] -= quantity
        log.info("legacy inventory reserved sku=%s qty=%d", sku, quantity)
        return True

    def get_stock(self, sku: str) -> int:
        return _inventory.get(sku, 0)


# ---------------------------------------------------------------------------
# NEW INVENTORY SERVICE — the strangler fig target
# ---------------------------------------------------------------------------


class NewInventoryService:
    """Simulates the new standalone inventory microservice.

    In production: this is a separate deployed service.
    The monolith calls it via HTTP (or consumes its events).
    """

    def __init__(self) -> None:
        # New service has its own data store
        self._stock: dict[str, int] = dict(_inventory)

    def check_availability(self, sku: str, quantity: int) -> bool:
        return self._stock.get(sku, 0) >= quantity

    def reserve(self, sku: str, quantity: int) -> bool:
        if not self.check_availability(sku, quantity):
            return False
        self._stock[sku] -= quantity
        log.info("new inventory service reserved sku=%s qty=%d", sku, quantity)
        return True

    def get_stock(self, sku: str) -> int:
        return self._stock.get(sku, 0)


_legacy_inventory = LegacyInventoryService()
_new_inventory = NewInventoryService()


# ---------------------------------------------------------------------------
# Anti-corruption layer — inventory gateway used by orders domain
# ---------------------------------------------------------------------------


class InventoryGateway:
    """
    The anti-corruption layer between the orders bounded context and the
    inventory bounded context.

    Orders never import LegacyInventoryService or NewInventoryService directly.
    This class controls the routing and translation.

    When use_new_inventory_service flag is True:
        - Route to NewInventoryService (or a real HTTP call in production)
    Otherwise:
        - Route to LegacyInventoryService
    """

    def check_and_reserve(self, items: list[dict[str, int]]) -> tuple[bool, list[str]]:
        """
        Check availability and reserve all items atomically.
        Returns (success, list of unavailable SKUs).
        """
        use_new = feature_flag("use_new_inventory_service")
        service = _new_inventory if use_new else _legacy_inventory
        log.info("inventory gateway routing to=%s", "new" if use_new else "legacy")

        unavailable = [
            item["sku"]
            for item in items
            if not service.check_availability(item["sku"], item["quantity"])
        ]
        if unavailable:
            return False, unavailable

        for item in items:
            service.reserve(item["sku"], item["quantity"])

        return True, []


# ---------------------------------------------------------------------------
# ORDERS BOUNDED CONTEXT — API layer
# ---------------------------------------------------------------------------


class OrderItem(BaseModel):
    sku: str
    quantity: int

    @field_validator("quantity")
    @classmethod
    def positive_quantity(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("quantity must be positive")
        return v


class OrderCreateRequest(BaseModel):
    user_id: int
    items: list[OrderItem]

    @field_validator("items")
    @classmethod
    def items_not_empty(cls, v: list[OrderItem]) -> list[OrderItem]:
        if not v:
            raise ValueError("order must have at least one item")
        return v


class OrderResponse(BaseModel):
    order_id: str
    status: str
    user_id: int
    items: list[dict[str, Any]]
    created_at: datetime


# In-memory order store
_orders: dict[str, dict] = {}

# Gateway — orders domain's only entry point to inventory
_inventory_gateway = InventoryGateway()


# ---------------------------------------------------------------------------
# Inventory event consumer — synchronises inventory context on OrderPlaced
# ---------------------------------------------------------------------------


@subscribe(OrderPlaced)
def handle_order_placed(event: OrderPlaced) -> None:
    """
    Inventory context reacts to OrderPlaced events.
    This is how it would work in an async/event-driven split:
    inventory doesn't need to be called synchronously.

    In the legacy synchronous approach, the gateway calls reserve() directly.
    In the async approach, the order is placed first, then inventory is updated.
    Both patterns are valid; sync is simpler; async is more resilient.
    """
    log.info(
        "inventory handler received OrderPlaced order_id=%s items=%s",
        event.aggregate_id,
        event.items,
    )


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="Week 08 Day 3 — Microservices, DDD, Strangler Fig")


@app.post("/v1/orders", response_model=OrderResponse, status_code=status.HTTP_201_CREATED)
async def create_order(payload: OrderCreateRequest) -> OrderResponse:
    """
    Creates an order.

    Internally demonstrates:
    1. Orders bounded context validates its own rules.
    2. InventoryGateway (anti-corruption layer) routes to legacy or new service.
    3. DomainEvent published after successful order creation.
    """
    items = [item.model_dump() for item in payload.items]

    # Step 1: Check and reserve via anti-corruption layer
    success, unavailable = _inventory_gateway.check_and_reserve(items)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Items out of stock: {unavailable}",
        )

    # Step 2: Persist the order (orders bounded context owns this data)
    order_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    _orders[order_id] = {
        "order_id": order_id,
        "status": "confirmed",
        "user_id": payload.user_id,
        "items": items,
        "created_at": now,
    }

    # Step 3: Publish domain event (decoupled notification to other contexts)
    event = OrderPlaced(
        event_id=str(uuid.uuid4()),
        occurred_at=now,
        aggregate_id=order_id,
        user_id=payload.user_id,
        items=items,
    )
    publish(event)

    log.info("order created id=%s user_id=%d", order_id, payload.user_id)
    return OrderResponse(**_orders[order_id])


@app.get("/v1/orders/{order_id}", response_model=OrderResponse)
async def get_order(order_id: str) -> OrderResponse:
    order = _orders.get(order_id)
    if not order:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="order not found")
    return OrderResponse(**order)


@app.get("/v1/inventory/{sku}", response_model=InventoryItem)
async def get_inventory(sku: str) -> InventoryItem:
    """
    Returns inventory for a SKU.

    Routes to legacy or new service based on feature flag.
    The response model is identical — the anti-corruption layer normalises differences.
    """
    use_new = feature_flag("use_new_inventory_service")
    service = _new_inventory if use_new else _legacy_inventory
    stock = service.get_stock(sku)
    return InventoryItem(sku=sku, available=stock)


@app.get("/v1/flags", tags=["ops"])
async def list_flags() -> dict[str, bool]:
    """Returns current feature flag state. For debugging only — remove in production."""
    return {
        "use_new_inventory_service": feature_flag("use_new_inventory_service"),
    }
```

---

### Reading the strangler fig progression

The code above shows **Phase 1** of a strangler fig:

```
Phase 1 (current):
  Orders monolith ─── InventoryGateway (ACL) ─── LegacyInventoryService
                                              └── NewInventoryService  [flag off]

Phase 2 (flag flipped to True):
  Orders monolith ─── InventoryGateway (ACL) ─── NewInventoryService  [flag on]
                                              └── LegacyInventoryService  [unused]

Phase 3 (cleanup):
  Orders monolith ─── InventoryGateway (ACL) ─── HTTP call → Inventory microservice
                   (LegacyInventoryService deleted from codebase)

Phase 4 (done):
  Orders service ─── async event → Inventory microservice
  (InventoryGateway removed — orders no longer calls inventory synchronously)
```

---

## Exercises

1. Add a shadow-mode to `InventoryGateway.check_and_reserve`:  
   When `FEATURE_SHADOW_INVENTORY=true`, call both legacy and new service,  
   compare results, log any divergence, but always return the legacy result.  
   Write a test that verifies: (a) legacy result is always returned; (b) divergence is logged.

2. Implement a second bounded context: `Notifications`.  
   Subscribe `Notifications` to `OrderPlaced` events and simulate sending an email.  
   Show that `Orders` has zero imports from `Notifications` (true decoupling).  
   Write a test using a fake event bus that asserts the handler was called.

3. Identify the two-phase commit problem: what happens if `reserve()` succeeds but writing
   to `_orders` fails? Design a saga pattern to handle this: compensating transaction on failure.
   Write pseudocode for the saga steps and the compensation steps.

4. Draw (in text/ASCII) the bounded context map for a simplified e-commerce system with:  
   Sales, Inventory, Shipping, Finance, and Customer Support contexts.  
   Mark which relationships are upstream/downstream and which use shared kernels.

(Solutions: ../solutions/week-08/day-3.md)

---

## Self-check

- [ ] I can name at least 5 valid technical reasons to split a monolith and 3 common bad reasons.
- [ ] I can define bounded context, ubiquitous language, and aggregate in one sentence each.
- [ ] I can identify the distributed monolith anti-pattern by its symptoms.
- [ ] I can walk through all 6 steps of the strangler fig pattern and explain why each step is necessary.
- [ ] I can explain why two services sharing a database is never a valid service boundary.
- [ ] I can implement an anti-corruption layer that routes between a legacy and a new service without the consumer knowing which is active.

---

## Further Reading

- [Martin Fowler — Strangler Fig Application](https://martinfowler.com/bliki/StranglerFigApplication.html)
- [Sam Newman — Monolith to Microservices (book outline + key heuristics)](https://samnewman.io/books/monolith-to-microservices/)
- [Eric Evans — Domain-Driven Design reference (free summary)](https://www.domainlanguage.com/ddd/reference/)
