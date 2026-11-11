# Phase 2 · Week 08 · Day 6 — 4-Hour Deep Practice: 2-Service System Design

> Objective: Design and implement a production-quality 2-service architecture end-to-end — from ADR through API contract to tested FastAPI skeleton.  ·  Time: ~4 hr  ·  Prereqs: [[day-4]], [[day-5]]

---

## Overview

Today you build a complete system from scratch, applying every concept from Week 08:

- **Order Service** — accepts customer orders, persists them, publishes events.
- **Inventory Service** — consumes order events, checks and reserves stock, exposes a REST API.

You will produce: a system design document, versioned FastAPI skeletons for both services, idempotent POST endpoints, inter-service communication via async messaging, and full pytest coverage of the API contract.

Work through the four parts in order. Time boxes are suggestions — adjust to your pace.

---

## Part 1 — Design Phase (45 min)

### 1.1 System Design Template

Fill this out before writing a single line of code. Completed version becomes your design artifact.

```
# System Design: Order + Inventory Services

## Problem statement
Customers place orders. Each order must check and reserve inventory.
Orders must be idempotent (network retries must not create duplicates).
Inventory must be eventually consistent after order placement.

## Service boundaries
| Service    | Responsibility                          | Data owned               |
|------------|----------------------------------------|--------------------------|
| Order      | Accept, validate, persist orders        | orders table             |
| Inventory  | Manage stock levels, reserve on demand  | inventory table          |

## Communication choice
- Client → Order Service: synchronous REST (client needs order_id immediately)
- Order Service → Inventory Service: async messaging (event-driven, decoupled)
  Rationale: inventory reservation can lag by seconds; order creation should not
  block on inventory availability. Inventory failure does not fail the order write.

## API contracts (see Part 2 for full FastAPI schemas)

### Order Service
POST /v1/orders               Create order (idempotent via Idempotency-Key header)
GET  /v1/orders/{order_id}    Read order by ID
GET  /v1/orders               List orders (pagination: cursor-based)

### Inventory Service
GET  /v1/inventory/{item_id}  Read stock level
POST /v1/inventory/reserve    Reserve stock for an order (idempotent)
POST /v1/inventory/release    Release reserved stock

## Versioning strategy
URI path versioning (/v1/, /v2/). Breaking changes require new version.
Non-breaking additions (new optional fields) allowed within same version.

## Idempotency design
- POST /v1/orders: client sends Idempotency-Key header (UUID or hash).
  Server stores {idempotency_key: order_id} in a KV store with TTL 24h.
  Duplicate key returns 200 with the original response body.
- POST /v1/inventory/reserve: idempotent by order_id. Same order_id
  never double-reserves.

## Data model sketch
Order: {order_id, customer_id, items: [{item_id, qty}], status, created_at, idempotency_key}
InventoryItem: {item_id, total_stock, reserved_stock, available_stock (computed)}
Reservation: {reservation_id, order_id, item_id, qty, status, created_at}

## Failure modes and mitigations
| Failure                        | Mitigation                                    |
|-------------------------------|-----------------------------------------------|
| Order Service crash after POST | Idempotency key prevents duplicate on retry  |
| Message lost before Inventory  | At-least-once delivery; idempotent consumer  |
| Inventory Service down         | Order accepted, event redelivered on recovery|
| Inventory insufficient         | Compensating event: order status → REJECTED  |

## Open questions (resolve before implementation)
1. Should order status update synchronously or via compensation event?
2. What is the SLA for inventory reservation lag? (drives retry/timeout config)
3. Should the API gateway handle auth, or do services validate JWTs directly?
```

### 1.2 Draw the flow

Sketch (on paper or in a comment block) the happy path:

```
Client
  │── POST /v1/orders (Idempotency-Key: <uuid>) ──► Order Service
                                                        │ persist order
                                                        │ publish "order_placed" event
                                                        │◄── 201 Created {order_id}
Client ◄── 201 ──────────────────────────────────────────┘

                                          Inventory Service ◄── consumes "order_placed"
                                                               │ check stock
                                                               │ reserve stock
                                                               │ publish "reservation_confirmed"

Order Service ◄── consumes "reservation_confirmed"
               │ update order.status = CONFIRMED
```

---

## Part 2 — Implementation (2 hr)

### 2.1 Project structure

```
services/
  order_service/
    src/
      order_service/
        main.py          # FastAPI app factory
        routers/
          orders.py      # route handlers
        models/
          order.py       # Pydantic schemas
        services/
          order_svc.py   # business logic
        store/
          memory.py      # in-memory store (replace with DB in prod)
        messaging/
          publisher.py   # publish events to broker
    tests/
      test_orders_api.py
      test_idempotency.py
  inventory_service/
    src/
      inventory_service/
        main.py
        routers/
          inventory.py
        models/
          inventory.py
        services/
          inventory_svc.py
        store/
          memory.py
        messaging/
          consumer.py
    tests/
      test_inventory_api.py
      test_reservation.py
```

### 2.2 Order Service

```python
# services/order_service/src/order_service/models/order.py
"""
Pydantic v2 models for the Order Service API contract.
These schemas define what the API accepts and returns — never leak internal
DB models directly through the API layer.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator


class OrderStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class OrderItem(BaseModel):
    item_id: str = Field(..., min_length=1, max_length=64)
    quantity: int = Field(..., ge=1, le=10_000)


class CreateOrderRequest(BaseModel):
    customer_id: str = Field(..., min_length=1, max_length=64)
    items: list[OrderItem] = Field(..., min_length=1, max_length=100)

    @field_validator("items")
    @classmethod
    def items_unique_item_ids(cls, v: list[OrderItem]) -> list[OrderItem]:
        ids = [item.item_id for item in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate item_ids in order")
        return v


class OrderResponse(BaseModel):
    order_id: str
    customer_id: str
    items: list[OrderItem]
    status: OrderStatus
    created_at: float

    model_config = {"frozen": True}


class PaginatedOrdersResponse(BaseModel):
    items: list[OrderResponse]
    next_cursor: str | None = None
    total: int


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    request_id: str | None = None
```

```python
# services/order_service/src/order_service/store/memory.py
"""
In-memory store for orders and idempotency keys.
In production: replace with PostgreSQL (orders) + Redis (idempotency).
Thread-safe via asyncio — single-threaded event loop.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from order_service.models.order import OrderResponse, OrderStatus

logger = logging.getLogger(__name__)

IDEMPOTENCY_KEY_TTL_SECONDS = 86_400  # 24 hours


@dataclass
class IdempotencyRecord:
    order_id: str
    response: OrderResponse
    created_at: float = field(default_factory=time.time)


class OrderStore:
    def __init__(self) -> None:
        self._orders: dict[str, OrderResponse] = {}
        self._idempotency: dict[str, IdempotencyRecord] = {}

    def get_by_idempotency_key(self, key: str) -> OrderResponse | None:
        record = self._idempotency.get(key)
        if record is None:
            return None
        # Evict expired records lazily
        if time.time() - record.created_at > IDEMPOTENCY_KEY_TTL_SECONDS:
            del self._idempotency[key]
            logger.debug("Evicted expired idempotency key=%s", key)
            return None
        return record.response

    def save(self, order: OrderResponse, idempotency_key: str) -> None:
        self._orders[order.order_id] = order
        self._idempotency[idempotency_key] = IdempotencyRecord(
            order_id=order.order_id, response=order
        )
        logger.info("Saved order_id=%s idempotency_key=%s", order.order_id, idempotency_key)

    def get_by_id(self, order_id: str) -> OrderResponse | None:
        return self._orders.get(order_id)

    def list_orders(
        self, cursor: str | None, limit: int
    ) -> tuple[list[OrderResponse], str | None]:
        all_orders = sorted(self._orders.values(), key=lambda o: o.created_at, reverse=True)
        if cursor:
            # Cursor is the order_id of the last seen item
            try:
                idx = next(i for i, o in enumerate(all_orders) if o.order_id == cursor)
                all_orders = all_orders[idx + 1 :]
            except StopIteration:
                all_orders = []

        page = all_orders[:limit]
        next_cursor = page[-1].order_id if len(all_orders) > limit else None
        return page, next_cursor

    def update_status(self, order_id: str, status: OrderStatus) -> OrderResponse | None:
        order = self._orders.get(order_id)
        if order is None:
            return None
        # Pydantic frozen model — create new instance with updated status
        updated = order.model_copy(update={"status": status})
        self._orders[order_id] = updated
        return updated
```

```python
# services/order_service/src/order_service/services/order_svc.py
"""
Order business logic — pure functions that take the store and return results.
No FastAPI dependencies here; keeps business logic testable without HTTP context.
"""

from __future__ import annotations

import logging
import uuid

from order_service.models.order import (
    CreateOrderRequest,
    OrderResponse,
    OrderStatus,
)
from order_service.store.memory import OrderStore

logger = logging.getLogger(__name__)


async def create_order(
    request: CreateOrderRequest,
    idempotency_key: str,
    store: OrderStore,
) -> tuple[OrderResponse, bool]:
    """
    Returns (order, created: bool).
    created=False means the idempotency key matched an existing order.
    """
    existing = store.get_by_idempotency_key(idempotency_key)
    if existing is not None:
        logger.info(
            "Idempotency key hit: key=%s order_id=%s", idempotency_key, existing.order_id
        )
        return existing, False

    import time

    order = OrderResponse(
        order_id=str(uuid.uuid4()),
        customer_id=request.customer_id,
        items=request.items,
        status=OrderStatus.PENDING,
        created_at=time.time(),
    )
    store.save(order, idempotency_key)
    logger.info("Created order_id=%s customer_id=%s", order.order_id, request.customer_id)
    return order, True
```

```python
# services/order_service/src/order_service/routers/orders.py
"""
FastAPI router for /v1/orders.
All endpoints are versioned via the router prefix.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse

from order_service.models.order import (
    CreateOrderRequest,
    ErrorResponse,
    OrderResponse,
    OrderStatus,
    PaginatedOrdersResponse,
)
from order_service.services.order_svc import create_order
from order_service.store.memory import OrderStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/orders", tags=["orders"])


def get_store(request: Request) -> OrderStore:
    """Dependency: extracts the shared store from app state."""
    return request.app.state.order_store


@router.post(
    "",
    response_model=OrderResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        200: {"model": OrderResponse, "description": "Duplicate request — replayed response"},
        422: {"model": ErrorResponse},
    },
    summary="Create a new order (idempotent)",
    description=(
        "Creates an order. If the same `Idempotency-Key` was used within the last 24 hours, "
        "returns the original response with HTTP 200 instead of 201."
    ),
)
async def create_order_endpoint(
    request_body: CreateOrderRequest,
    idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=1, max_length=128),
    store: OrderStore = Depends(get_store),
) -> JSONResponse:
    order, created = await create_order(request_body, idempotency_key, store)
    http_status = status.HTTP_201_CREATED if created else status.HTTP_200_OK
    return JSONResponse(content=order.model_dump(), status_code=http_status)


@router.get(
    "/{order_id}",
    response_model=OrderResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get order by ID",
)
async def get_order(
    order_id: str,
    store: OrderStore = Depends(get_store),
) -> OrderResponse:
    order = store.get_by_id(order_id)
    if order is None:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
    return order


@router.get(
    "",
    response_model=PaginatedOrdersResponse,
    summary="List orders (cursor-based pagination)",
)
async def list_orders(
    cursor: str | None = Query(None, description="Pagination cursor (order_id of last item)"),
    limit: int = Query(20, ge=1, le=100),
    store: OrderStore = Depends(get_store),
) -> PaginatedOrdersResponse:
    orders, next_cursor = store.list_orders(cursor, limit)
    return PaginatedOrdersResponse(
        items=orders, next_cursor=next_cursor, total=len(orders)
    )
```

```python
# services/order_service/src/order_service/main.py
"""
Order Service — FastAPI application factory.
Keeps app creation in a function so tests can create isolated instances.
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from order_service.routers.orders import router as orders_router
from order_service.store.memory import OrderStore

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Order Service",
        version="1.0.0",
        description="Manages customer orders. Part of the order-inventory system.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("CORS_ORIGINS", "").split(","),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Shared in-memory store — in production inject a real DB client here
    app.state.order_store = OrderStore()

    app.include_router(orders_router)

    @app.get("/health", tags=["ops"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "order-service"}

    logger.info("Order Service started")
    return app


app = create_app()
```

### 2.3 Inventory Service

```python
# services/inventory_service/src/inventory_service/models/inventory.py
"""Pydantic v2 models for the Inventory Service API contract."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ReservationStatus(str, Enum):
    CONFIRMED = "confirmed"
    FAILED = "failed"
    RELEASED = "released"


class InventoryItemResponse(BaseModel):
    item_id: str
    total_stock: int
    reserved_stock: int
    available_stock: int

    model_config = {"frozen": True}


class ReserveStockRequest(BaseModel):
    order_id: str = Field(..., min_length=1, max_length=64)
    item_id: str = Field(..., min_length=1, max_length=64)
    quantity: int = Field(..., ge=1, le=10_000)


class ReserveStockResponse(BaseModel):
    reservation_id: str
    order_id: str
    item_id: str
    quantity: int
    status: ReservationStatus


class ReleaseStockRequest(BaseModel):
    order_id: str
    item_id: str


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
```

```python
# services/inventory_service/src/inventory_service/store/memory.py
"""In-memory inventory store. Replace with PostgreSQL in production."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

from inventory_service.models.inventory import (
    InventoryItemResponse,
    ReservationStatus,
    ReserveStockResponse,
)

logger = logging.getLogger(__name__)


@dataclass
class InventoryItem:
    item_id: str
    total_stock: int
    reserved_stock: int = 0

    @property
    def available_stock(self) -> int:
        return self.total_stock - self.reserved_stock

    def to_response(self) -> InventoryItemResponse:
        return InventoryItemResponse(
            item_id=self.item_id,
            total_stock=self.total_stock,
            reserved_stock=self.reserved_stock,
            available_stock=self.available_stock,
        )


@dataclass
class Reservation:
    reservation_id: str
    order_id: str
    item_id: str
    quantity: int
    status: ReservationStatus


class InventoryStore:
    def __init__(self) -> None:
        self._items: dict[str, InventoryItem] = {}
        self._reservations: dict[str, Reservation] = {}
        # Dedup: order_id+item_id → reservation_id
        self._order_reservations: dict[tuple[str, str], str] = {}

    def seed(self, item_id: str, total_stock: int) -> None:
        """Pre-populate stock — call during app startup or tests."""
        self._items[item_id] = InventoryItem(item_id=item_id, total_stock=total_stock)

    def get_item(self, item_id: str) -> InventoryItemResponse | None:
        item = self._items.get(item_id)
        return item.to_response() if item else None

    def reserve(
        self, order_id: str, item_id: str, quantity: int
    ) -> ReserveStockResponse:
        """
        Idempotent: same (order_id, item_id) returns the existing reservation.
        Returns FAILED status if insufficient stock.
        """
        dedup_key = (order_id, item_id)
        existing_id = self._order_reservations.get(dedup_key)
        if existing_id:
            existing = self._reservations[existing_id]
            logger.info("Idempotent reserve hit order_id=%s item_id=%s", order_id, item_id)
            return ReserveStockResponse(
                reservation_id=existing.reservation_id,
                order_id=existing.order_id,
                item_id=existing.item_id,
                quantity=existing.quantity,
                status=existing.status,
            )

        item = self._items.get(item_id)
        if item is None or item.available_stock < quantity:
            reservation_id = str(uuid.uuid4())
            reservation = Reservation(
                reservation_id=reservation_id,
                order_id=order_id,
                item_id=item_id,
                quantity=quantity,
                status=ReservationStatus.FAILED,
            )
            self._reservations[reservation_id] = reservation
            self._order_reservations[dedup_key] = reservation_id
            logger.warning(
                "Stock insufficient item_id=%s requested=%d available=%d",
                item_id,
                quantity,
                item.available_stock if item else 0,
            )
            return ReserveStockResponse(
                reservation_id=reservation_id,
                order_id=order_id,
                item_id=item_id,
                quantity=quantity,
                status=ReservationStatus.FAILED,
            )

        item.reserved_stock += quantity
        reservation_id = str(uuid.uuid4())
        reservation = Reservation(
            reservation_id=reservation_id,
            order_id=order_id,
            item_id=item_id,
            quantity=quantity,
            status=ReservationStatus.CONFIRMED,
        )
        self._reservations[reservation_id] = reservation
        self._order_reservations[dedup_key] = reservation_id
        logger.info(
            "Reserved item_id=%s qty=%d order_id=%s reservation_id=%s",
            item_id,
            quantity,
            order_id,
            reservation_id,
        )
        return ReserveStockResponse(
            reservation_id=reservation_id,
            order_id=order_id,
            item_id=item_id,
            quantity=quantity,
            status=ReservationStatus.CONFIRMED,
        )

    def release(self, order_id: str, item_id: str) -> bool:
        """Releases a reservation. Returns False if reservation not found."""
        dedup_key = (order_id, item_id)
        reservation_id = self._order_reservations.get(dedup_key)
        if not reservation_id:
            return False
        reservation = self._reservations.get(reservation_id)
        if not reservation or reservation.status != ReservationStatus.CONFIRMED:
            return False
        item = self._items.get(item_id)
        if item:
            item.reserved_stock -= reservation.quantity
        reservation.status = ReservationStatus.RELEASED
        logger.info("Released reservation_id=%s", reservation_id)
        return True
```

```python
# services/inventory_service/src/inventory_service/routers/inventory.py
"""FastAPI router for /v1/inventory."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from inventory_service.models.inventory import (
    ErrorResponse,
    InventoryItemResponse,
    ReleaseStockRequest,
    ReserveStockRequest,
    ReserveStockResponse,
)
from inventory_service.store.memory import InventoryStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/inventory", tags=["inventory"])


def get_store(request: Request) -> InventoryStore:
    return request.app.state.inventory_store


@router.get(
    "/{item_id}",
    response_model=InventoryItemResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get stock level for an item",
)
async def get_item(
    item_id: str,
    store: InventoryStore = Depends(get_store),
) -> InventoryItemResponse:
    item = store.get_item(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
    return item


@router.post(
    "/reserve",
    response_model=ReserveStockResponse,
    status_code=status.HTTP_200_OK,
    summary="Reserve stock for an order (idempotent by order_id + item_id)",
)
async def reserve_stock(
    request_body: ReserveStockRequest,
    store: InventoryStore = Depends(get_store),
) -> ReserveStockResponse:
    return store.reserve(
        order_id=request_body.order_id,
        item_id=request_body.item_id,
        quantity=request_body.quantity,
    )


@router.post(
    "/release",
    response_model=dict[str, str],
    responses={404: {"model": ErrorResponse}},
    summary="Release previously reserved stock",
)
async def release_stock(
    request_body: ReleaseStockRequest,
    store: InventoryStore = Depends(get_store),
) -> dict[str, str]:
    released = store.release(
        order_id=request_body.order_id,
        item_id=request_body.item_id,
    )
    if not released:
        raise HTTPException(
            status_code=404,
            detail=f"No confirmed reservation for order_id={request_body.order_id} item_id={request_body.item_id}",
        )
    return {"status": "released"}
```

```python
# services/inventory_service/src/inventory_service/main.py
"""Inventory Service — FastAPI application factory."""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI

from inventory_service.routers.inventory import router as inventory_router
from inventory_service.store.memory import InventoryStore

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def create_app(seed_data: dict[str, int] | None = None) -> FastAPI:
    """
    seed_data: {item_id: stock_count} for test fixtures and local dev.
    In production, inventory is loaded from the database on startup.
    """
    app = FastAPI(
        title="Inventory Service",
        version="1.0.0",
        description="Manages stock levels and reservations.",
    )

    store = InventoryStore()
    if seed_data:
        for item_id, qty in seed_data.items():
            store.seed(item_id, qty)
    app.state.inventory_store = store

    app.include_router(inventory_router)

    @app.get("/health", tags=["ops"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "inventory-service"}

    logger.info("Inventory Service started with %d seeded items", len(seed_data or {}))
    return app


app = create_app(seed_data={"widget-001": 500, "gadget-002": 100})
```

---

## Part 3 — Testing (45 min)

```python
# services/order_service/tests/test_orders_api.py
"""
Tests for the Order Service API contract.
Uses FastAPI's TestClient (httpx under the hood) — no real server needed.
All external calls (messaging) are mocked.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from order_service.main import create_app


@pytest.fixture()
def client() -> TestClient:
    """Fresh app instance per test — isolated in-memory store."""
    app = create_app()
    return TestClient(app)


@pytest.fixture()
def idempotency_key() -> str:
    return str(uuid.uuid4())


@pytest.fixture()
def valid_order_payload() -> dict:
    return {
        "customer_id": "cust-001",
        "items": [{"item_id": "widget-001", "quantity": 2}],
    }


class TestCreateOrder:
    def test_create_returns_201(
        self, client: TestClient, valid_order_payload: dict, idempotency_key: str
    ) -> None:
        response = client.post(
            "/v1/orders",
            json=valid_order_payload,
            headers={"Idempotency-Key": idempotency_key},
        )
        assert response.status_code == 201
        body = response.json()
        assert "order_id" in body
        assert body["status"] == "pending"
        assert body["customer_id"] == "cust-001"

    def test_idempotent_create_returns_200_on_duplicate(
        self, client: TestClient, valid_order_payload: dict, idempotency_key: str
    ) -> None:
        # First call
        r1 = client.post(
            "/v1/orders",
            json=valid_order_payload,
            headers={"Idempotency-Key": idempotency_key},
        )
        assert r1.status_code == 201

        # Second call — same key
        r2 = client.post(
            "/v1/orders",
            json=valid_order_payload,
            headers={"Idempotency-Key": idempotency_key},
        )
        assert r2.status_code == 200  # replayed
        assert r1.json()["order_id"] == r2.json()["order_id"]

    def test_idempotency_key_required(
        self, client: TestClient, valid_order_payload: dict
    ) -> None:
        response = client.post("/v1/orders", json=valid_order_payload)
        # Missing required header → 422 Unprocessable Entity
        assert response.status_code == 422

    def test_duplicate_item_ids_rejected(
        self, client: TestClient, idempotency_key: str
    ) -> None:
        payload = {
            "customer_id": "cust-001",
            "items": [
                {"item_id": "widget-001", "quantity": 1},
                {"item_id": "widget-001", "quantity": 2},  # duplicate
            ],
        }
        response = client.post(
            "/v1/orders",
            json=payload,
            headers={"Idempotency-Key": idempotency_key},
        )
        assert response.status_code == 422

    def test_different_idempotency_keys_create_different_orders(
        self, client: TestClient, valid_order_payload: dict
    ) -> None:
        r1 = client.post(
            "/v1/orders",
            json=valid_order_payload,
            headers={"Idempotency-Key": str(uuid.uuid4())},
        )
        r2 = client.post(
            "/v1/orders",
            json=valid_order_payload,
            headers={"Idempotency-Key": str(uuid.uuid4())},
        )
        assert r1.status_code == 201
        assert r2.status_code == 201
        assert r1.json()["order_id"] != r2.json()["order_id"]


class TestGetOrder:
    def test_get_existing_order(
        self, client: TestClient, valid_order_payload: dict, idempotency_key: str
    ) -> None:
        create_response = client.post(
            "/v1/orders",
            json=valid_order_payload,
            headers={"Idempotency-Key": idempotency_key},
        )
        order_id = create_response.json()["order_id"]

        get_response = client.get(f"/v1/orders/{order_id}")
        assert get_response.status_code == 200
        assert get_response.json()["order_id"] == order_id

    def test_get_nonexistent_order_returns_404(self, client: TestClient) -> None:
        response = client.get("/v1/orders/nonexistent-id")
        assert response.status_code == 404


class TestListOrders:
    def test_list_empty(self, client: TestClient) -> None:
        response = client.get("/v1/orders")
        assert response.status_code == 200
        body = response.json()
        assert body["items"] == []
        assert body["next_cursor"] is None

    def test_list_returns_created_orders(
        self, client: TestClient, valid_order_payload: dict
    ) -> None:
        for _ in range(3):
            client.post(
                "/v1/orders",
                json=valid_order_payload,
                headers={"Idempotency-Key": str(uuid.uuid4())},
            )

        response = client.get("/v1/orders")
        assert response.status_code == 200
        assert len(response.json()["items"]) == 3

    def test_pagination_limit(
        self, client: TestClient, valid_order_payload: dict
    ) -> None:
        for _ in range(5):
            client.post(
                "/v1/orders",
                json=valid_order_payload,
                headers={"Idempotency-Key": str(uuid.uuid4())},
            )
        response = client.get("/v1/orders?limit=2")
        assert response.status_code == 200
        body = response.json()
        assert len(body["items"]) == 2
        assert body["next_cursor"] is not None
```

```python
# services/inventory_service/tests/test_reservation.py
"""
Tests for the Inventory Service — stock reservation and idempotency.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from inventory_service.main import create_app


@pytest.fixture()
def client() -> TestClient:
    app = create_app(seed_data={"widget-001": 100, "scarce-item": 1})
    return TestClient(app)


class TestGetInventory:
    def test_get_existing_item(self, client: TestClient) -> None:
        response = client.get("/v1/inventory/widget-001")
        assert response.status_code == 200
        body = response.json()
        assert body["total_stock"] == 100
        assert body["available_stock"] == 100
        assert body["reserved_stock"] == 0

    def test_get_missing_item_returns_404(self, client: TestClient) -> None:
        response = client.get("/v1/inventory/does-not-exist")
        assert response.status_code == 404


class TestReserveStock:
    def test_successful_reservation(self, client: TestClient) -> None:
        payload = {"order_id": str(uuid.uuid4()), "item_id": "widget-001", "quantity": 10}
        response = client.post("/v1/inventory/reserve", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "confirmed"
        assert body["quantity"] == 10

    def test_reservation_reduces_available_stock(self, client: TestClient) -> None:
        order_id = str(uuid.uuid4())
        client.post(
            "/v1/inventory/reserve",
            json={"order_id": order_id, "item_id": "widget-001", "quantity": 30},
        )
        inventory = client.get("/v1/inventory/widget-001").json()
        assert inventory["available_stock"] == 70
        assert inventory["reserved_stock"] == 30

    def test_reservation_idempotent(self, client: TestClient) -> None:
        order_id = str(uuid.uuid4())
        payload = {"order_id": order_id, "item_id": "widget-001", "quantity": 5}
        r1 = client.post("/v1/inventory/reserve", json=payload)
        r2 = client.post("/v1/inventory/reserve", json=payload)
        assert r1.json()["reservation_id"] == r2.json()["reservation_id"]
        # Stock only deducted once
        inventory = client.get("/v1/inventory/widget-001").json()
        assert inventory["reserved_stock"] == 5

    def test_insufficient_stock_returns_failed_status(self, client: TestClient) -> None:
        payload = {"order_id": str(uuid.uuid4()), "item_id": "scarce-item", "quantity": 10}
        response = client.post("/v1/inventory/reserve", json=payload)
        assert response.status_code == 200
        assert response.json()["status"] == "failed"

    def test_release_restores_stock(self, client: TestClient) -> None:
        order_id = str(uuid.uuid4())
        client.post(
            "/v1/inventory/reserve",
            json={"order_id": order_id, "item_id": "widget-001", "quantity": 20},
        )
        client.post(
            "/v1/inventory/release",
            json={"order_id": order_id, "item_id": "widget-001"},
        )
        inventory = client.get("/v1/inventory/widget-001").json()
        assert inventory["available_stock"] == 100
        assert inventory["reserved_stock"] == 0

    def test_release_nonexistent_reservation_returns_404(self, client: TestClient) -> None:
        response = client.post(
            "/v1/inventory/release",
            json={"order_id": "nonexistent", "item_id": "widget-001"},
        )
        assert response.status_code == 404
```

---

## Part 4 — Architecture Decision Record (30 min)

Write this ADR and save it alongside your code as `docs/adr/001-inter-service-communication.md`.

```markdown
# ADR-001: Inter-Service Communication Pattern

**Status**: Accepted
**Date**: YYYY-MM-DD
**Deciders**: [your name]

## Context

The Order Service must trigger inventory reservation when an order is placed.
Two options exist: synchronous HTTP call from Order → Inventory, or asynchronous
messaging via a broker.

## Decision

Use asynchronous messaging (event-driven) for Order → Inventory communication.

## Rationale

### Option A: Synchronous REST (rejected)

- Order creation latency increases by inventory service RTT.
- If Inventory Service is down, Order creation fails — availability coupling.
- Retry logic on Order Service must handle partial failures (order written, inventory call failed).
- Temporal coupling: both services must be running simultaneously.

### Option B: Async messaging via broker (chosen)

- Order creation latency is independent of Inventory Service health.
- Inventory Service can be redeployed or scaled independently.
- At-least-once delivery + idempotent consumer handles duplicate events safely.
- Clear separation of concerns: Order Service owns order lifecycle, Inventory owns stock.

### Tradeoffs accepted

- Eventual consistency: inventory reservation lags behind order creation.
- Added infrastructure: requires a message broker (Kafka / SQS / RabbitMQ).
- Harder to trace: distributed trace must correlate across services via trace_id.
- Out-of-order events possible: mitigated by event sequencing on consumer side.

## Consequences

- Order Service must publish `order_placed` events reliably (transactional outbox pattern recommended for production).
- Inventory Service must implement idempotent reservation keyed on `order_id`.
- Monitoring must track broker lag to detect Inventory Service falling behind.
- Order status will be `PENDING` until `reservation_confirmed` event is consumed.
- API contract must document eventual consistency to client teams.

## Alternatives considered

### Synchronous with circuit breaker (not chosen)

Could use a circuit breaker (e.g., Hystrix pattern) to degrade gracefully when
Inventory is down. Still couples availability. Chosen async approach is strictly
better for resilience at the cost of consistency.

### Saga pattern (future consideration)

For multi-step workflows (order → inventory → payment → fulfillment), a saga
coordinator or choreography pattern is recommended. Out of scope for this iteration.
```

---

## Exercises

1. **Add health checks with dependency status**: Extend both `/health` endpoints to also check if the store is accessible (for a real DB: run a simple `SELECT 1`). Return `{"status": "degraded", "reason": "..."}` with HTTP 503 if the dependency is unhealthy.

2. **Add request ID propagation**: Implement middleware on both services that reads an `X-Request-ID` header (or generates a UUID if absent) and injects it into the logging context for all log lines during the request. This enables correlation across services.

3. **API versioning migration**: Add a `/v2/orders` endpoint that returns the same data but with an additional `total_value: float` field (sum of `quantity * unit_price` with a mock price lookup). Write a backward-compatibility test that asserts `/v1/orders` responses still work after adding `/v2`.

4. **Transactional outbox**: The current implementation publishes events in-memory. Implement a transactional outbox pattern: instead of publishing directly, write the event to an `outbox` table in the same transaction as the order write. A background worker reads the outbox and publishes to the broker. Sketch the implementation and explain why this solves the dual-write problem.

(Solutions: ../solutions/week-08/day-6.md)

---

## Self-check

- [ ] I can draw the service boundary diagram and explain why Order and Inventory are separate services.
- [ ] I can explain the dual-write problem and why the outbox pattern solves it.
- [ ] I can implement an idempotent POST endpoint with Idempotency-Key header handling.
- [ ] I can write a complete API contract (request model, response model, error model, versioned prefix) in FastAPI with Pydantic v2.
- [ ] I can write an ADR that captures context, decision, rationale, and consequences concisely.
- [ ] I can write pytest tests that cover both the happy path and idempotency invariants.

---

## Supplementary reading

- [Idempotency Keys — Stripe engineering blog](https://stripe.com/blog/idempotency)
- [Architecture Decision Records (Michael Nygard original)](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
- [The Outbox Pattern (Debezium docs)](https://debezium.io/documentation/reference/stable/transformations/outbox-event-router.html)
