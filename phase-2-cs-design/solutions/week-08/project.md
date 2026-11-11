# Solutions — Week 08 · Project (2-Service System Reference Implementation)

## Architecture

```
                        ┌────────────────────────────────────────────┐
                        │              API Gateway / Client           │
                        └──────────────┬──────────────┬──────────────┘
                                       │              │
                        ┌──────────────▼──┐     ┌─────▼───────────────┐
                        │  Order Service  │     │  Inventory Service  │
                        │  :8000          │     │  :8001              │
                        │                 │     │                     │
                        │  POST /orders   │────▶│  GET /products/{id} │
                        │  GET  /orders   │     │  GET /products      │
                        │  GET  /orders/{id}    │  PATCH /products/{id}│
                        │                 │     │    /reserve         │
                        └────────┬────────┘     └─────────┬───────────┘
                                 │                        │
                        ┌────────▼────────┐     ┌─────────▼───────────┐
                        │  orders DB      │     │  inventory DB       │
                        │  (in-memory)    │     │  (in-memory)        │
                        └─────────────────┘     └─────────────────────┘

Cross-service call: Order Service calls Inventory Service synchronously
during order creation to reserve stock (demo scope).
In production: replace with async event (see ADR-003 in day-7.md).
```

---

## Project Structure

```
week-08-project/
  src/
    shared/
      __init__.py
      errors.py
      pagination.py
    order_service/
      __init__.py
      main.py
      models.py
      store.py
    inventory_service/
      __init__.py
      main.py
      models.py
      store.py
  tests/
    __init__.py
    test_orders.py
    test_inventory.py
    test_integration.py
  adr/
    ADR-001-service-boundaries.md
    ADR-002-pagination-strategy.md
    ADR-003-consistency-model.md
  pyproject.toml
```

---

## Shared Layer

```python
# src/shared/errors.py
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

PROBLEM_JSON = "application/problem+json"


@dataclass
class ProblemDetail:
    type: str
    title: str
    status: int
    detail: str
    instance: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "type": self.type,
            "title": self.title,
            "status": self.status,
            "detail": self.detail,
            "instance": self.instance,
        }
        body.update(self.extra)
        return body


class AppError(Exception):
    def __init__(
        self,
        *,
        problem_type: str,
        title: str,
        status: int,
        detail: str,
        instance: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.problem = ProblemDetail(
            type=problem_type,
            title=title,
            status=status,
            detail=detail,
            instance=instance or f"/errors/{uuid.uuid4()}",
            extra=extra or {},
        )
        super().__init__(detail)


class NotFoundError(AppError):
    def __init__(self, resource: str, resource_id: str | int) -> None:
        super().__init__(
            problem_type="https://api.example.com/problems/not-found",
            title="Resource Not Found",
            status=404,
            detail=f"{resource} with id '{resource_id}' does not exist.",
            instance=f"/{resource.lower()}s/{resource_id}",
        )


class ConflictError(AppError):
    def __init__(self, detail: str, instance: str = "/errors/conflict") -> None:
        super().__init__(
            problem_type="https://api.example.com/problems/conflict",
            title="Conflict",
            status=409,
            detail=detail,
            instance=instance,
        )


class UnprocessableError(AppError):
    def __init__(self, detail: str, instance: str = "/errors/unprocessable") -> None:
        super().__init__(
            problem_type="https://api.example.com/problems/unprocessable",
            title="Unprocessable Request",
            status=422,
            detail=detail,
            instance=instance,
        )


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        p = exc.problem
        logger.warning("app error type=%s status=%d path=%s", p.type, p.status, request.url.path)
        return JSONResponse(
            status_code=p.status,
            content=p.to_dict(),
            headers={"Content-Type": PROBLEM_JSON},
        )

    @app.exception_handler(Exception)
    async def fallback_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error at %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "type": "https://api.example.com/problems/internal-error",
                "title": "Internal Server Error",
                "status": 500,
                "detail": "An unexpected error occurred.",
                "instance": str(request.url.path),
            },
            headers={"Content-Type": PROBLEM_JSON},
        )
```

```python
# src/shared/pagination.py
from __future__ import annotations

import base64
import json
from typing import Generic, TypeVar

from pydantic import BaseModel, field_validator

T = TypeVar("T")


def encode_cursor(last_id: int) -> str:
    payload = json.dumps({"last_id": last_id}, separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode()).decode()


def decode_cursor(cursor: str) -> int:
    try:
        payload = base64.urlsafe_b64decode(cursor.encode()).decode()
        return int(json.loads(payload)["last_id"])
    except Exception as exc:
        raise ValueError(f"Invalid cursor: {cursor!r}") from exc


class PageResponse(BaseModel, Generic[T]):
    data: list[T]
    next_cursor: str | None
    has_next: bool
    limit: int


def paginate(items: list[T], cursor: str | None, limit: int) -> PageResponse[T]:  # type: ignore[type-var]
    start_id = 0
    if cursor is not None:
        start_id = decode_cursor(cursor)  # raises ValueError on bad cursor

    # items must be sorted by id ascending; filter by id > start_id
    filtered = [item for item in items if getattr(item, "id") > start_id]
    page = filtered[: limit + 1]
    has_next = len(page) > limit
    result = page[:limit]
    next_cursor = encode_cursor(getattr(result[-1], "id")) if has_next and result else None
    return PageResponse(data=result, next_cursor=next_cursor, has_next=has_next, limit=limit)
```

---

## Inventory Service

```python
# src/inventory_service/models.py
from __future__ import annotations
from pydantic import BaseModel, Field


class Product(BaseModel):
    id: int
    sku: str
    name: str
    quantity: int
    version: int = 0


class ReservationRequest(BaseModel):
    quantity: int = Field(..., gt=0)


class ReservationResponse(BaseModel):
    product_id: int
    reserved: int
    remaining: int
```

```python
# src/inventory_service/store.py
from __future__ import annotations

import threading
from copy import deepcopy

from .models import Product

_lock = threading.Lock()
_products: dict[int, Product] = {
    i: Product(id=i, sku=f"SKU-{i:04d}", name=f"Product {i}", quantity=100)
    for i in range(1, 21)
}


def all_products() -> list[Product]:
    with _lock:
        return sorted(_products.values(), key=lambda p: p.id)


def get_product(product_id: int) -> Product | None:
    with _lock:
        p = _products.get(product_id)
        return deepcopy(p) if p else None


def reserve(product_id: int, qty: int) -> Product | None:
    """Decrement quantity atomically. Returns updated product or None if not found."""
    with _lock:
        p = _products.get(product_id)
        if p is None:
            return None
        if p.quantity < qty:
            raise ValueError(
                f"Insufficient stock: requested {qty}, available {p.quantity}"
            )
        _products[product_id] = p.model_copy(
            update={"quantity": p.quantity - qty, "version": p.version + 1}
        )
        return deepcopy(_products[product_id])
```

```python
# src/inventory_service/main.py
from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, FastAPI, Query

from shared.errors import ConflictError, NotFoundError, UnprocessableError, register_error_handlers
from shared.pagination import PageResponse, paginate

from .models import Product, ReservationRequest, ReservationResponse
from .store import all_products, get_product, reserve

logger = logging.getLogger(__name__)

app = FastAPI(title="Inventory Service", version="1.0.0")
register_error_handlers(app)

router = APIRouter(prefix="/api/v1/products", tags=["products"])


@router.get("", response_model=PageResponse[Product])
async def list_products(
    cursor: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> PageResponse[Product]:
    try:
        return paginate(all_products(), cursor, limit)
    except ValueError as exc:
        raise UnprocessableError(str(exc), instance="/api/v1/products") from exc


@router.get("/{product_id}", response_model=Product)
async def get_product_endpoint(product_id: int) -> Product:
    product = get_product(product_id)
    if product is None:
        raise NotFoundError("Product", product_id)
    return product


@router.post("/{product_id}/reserve", response_model=ReservationResponse)
async def reserve_stock(product_id: int, body: ReservationRequest) -> ReservationResponse:
    product = get_product(product_id)
    if product is None:
        raise NotFoundError("Product", product_id)
    try:
        updated = reserve(product_id, body.quantity)
    except ValueError as exc:
        raise ConflictError(str(exc), instance=f"/api/v1/products/{product_id}/reserve") from exc
    assert updated is not None
    return ReservationResponse(
        product_id=product_id,
        reserved=body.quantity,
        remaining=updated.quantity,
    )


app.include_router(router)
```

---

## Order Service

```python
# src/order_service/models.py
from __future__ import annotations

import os
from enum import StrEnum

from pydantic import BaseModel, Field


class OrderStatus(StrEnum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"


class OrderItem(BaseModel):
    product_id: int
    quantity: int = Field(..., gt=0)


class CreateOrderRequest(BaseModel):
    customer_id: str = Field(..., min_length=1)
    items: list[OrderItem] = Field(..., min_length=1)
    idempotency_key: str = Field(..., min_length=1)


class Order(BaseModel):
    id: int
    customer_id: str
    items: list[OrderItem]
    status: OrderStatus
    version: int = 0


INVENTORY_BASE_URL: str = os.getenv("INVENTORY_SERVICE_URL", "http://localhost:8001")
```

```python
# src/order_service/store.py
from __future__ import annotations

import threading
from copy import deepcopy

from .models import Order, OrderItem, OrderStatus

_lock = threading.Lock()
_orders: dict[int, Order] = {}
_next_id = 1
_idempotency: dict[str, Order] = {}


def create_order(
    customer_id: str,
    items: list[OrderItem],
    idempotency_key: str,
) -> tuple[Order, bool]:
    """
    Returns (order, created). created=False means idempotent replay.
    """
    with _lock:
        global _next_id
        if idempotency_key in _idempotency:
            return deepcopy(_idempotency[idempotency_key]), False

        order = Order(
            id=_next_id,
            customer_id=customer_id,
            items=items,
            status=OrderStatus.CONFIRMED,
        )
        _orders[_next_id] = order
        _idempotency[idempotency_key] = order
        _next_id += 1
        return deepcopy(order), True


def all_orders() -> list[Order]:
    with _lock:
        return sorted(_orders.values(), key=lambda o: o.id)


def get_order(order_id: int) -> Order | None:
    with _lock:
        o = _orders.get(order_id)
        return deepcopy(o) if o else None
```

```python
# src/order_service/main.py
from __future__ import annotations

import logging
from typing import Annotated

import httpx
from fastapi import APIRouter, FastAPI, Query, status

from shared.errors import ConflictError, NotFoundError, UnprocessableError, register_error_handlers
from shared.pagination import PageResponse, paginate

from .models import INVENTORY_BASE_URL, CreateOrderRequest, Order
from .store import all_orders, create_order, get_order

logger = logging.getLogger(__name__)

app = FastAPI(title="Order Service", version="1.0.0")
register_error_handlers(app)

router = APIRouter(prefix="/api/v1/orders", tags=["orders"])


async def _reserve_inventory(product_id: int, quantity: int) -> None:
    """Call Inventory Service to reserve stock. Raises on failure."""
    url = f"{INVENTORY_BASE_URL}/api/v1/products/{product_id}/reserve"
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.post(url, json={"quantity": quantity})
    if resp.status_code == 404:
        raise NotFoundError("Product", product_id)
    if resp.status_code == 409:
        body = resp.json()
        raise ConflictError(
            body.get("detail", "Insufficient stock"),
            instance=f"/api/v1/orders/items/{product_id}",
        )
    if not resp.is_success:
        raise UnprocessableError(
            f"Inventory service error: {resp.status_code}",
            instance="/api/v1/orders",
        )


@router.post("", response_model=Order, status_code=status.HTTP_201_CREATED)
async def create_order_endpoint(body: CreateOrderRequest) -> Order:
    # Reserve stock for all items (not atomic across items — demo scope)
    for item in body.items:
        await _reserve_inventory(item.product_id, item.quantity)

    order, created = create_order(
        customer_id=body.customer_id,
        items=body.items,
        idempotency_key=body.idempotency_key,
    )
    if not created:
        logger.info("Idempotent order replay key=%s", body.idempotency_key)
    return order


@router.get("", response_model=PageResponse[Order])
async def list_orders(
    cursor: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> PageResponse[Order]:
    try:
        return paginate(all_orders(), cursor, limit)
    except ValueError as exc:
        raise UnprocessableError(str(exc), instance="/api/v1/orders") from exc


@router.get("/{order_id}", response_model=Order)
async def get_order_endpoint(order_id: int) -> Order:
    order = get_order(order_id)
    if order is None:
        raise NotFoundError("Order", order_id)
    return order


app.include_router(router)
```

---

## Test Suite

```python
# tests/test_inventory.py
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.inventory_service.main import app
from src.inventory_service import store as inv_store

client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def reset_inventory():
    from src.inventory_service.models import Product
    inv_store._products = {
        i: Product(id=i, sku=f"SKU-{i:04d}", name=f"Product {i}", quantity=100)
        for i in range(1, 21)
    }
    yield


def test_list_products_default_page():
    resp = client.get("/api/v1/products")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["data"]) == 20
    assert body["has_next"] is False
    assert body["next_cursor"] is None


def test_list_products_cursor_pagination():
    resp1 = client.get("/api/v1/products?limit=5")
    assert resp1.status_code == 200
    body1 = resp1.json()
    assert len(body1["data"]) == 5
    assert body1["has_next"] is True
    cursor = body1["next_cursor"]

    resp2 = client.get(f"/api/v1/products?cursor={cursor}&limit=5")
    assert resp2.status_code == 200
    body2 = resp2.json()
    # IDs must be strictly greater than those on page 1
    ids1 = {p["id"] for p in body1["data"]}
    ids2 = {p["id"] for p in body2["data"]}
    assert ids1.isdisjoint(ids2)


def test_get_product_found():
    resp = client.get("/api/v1/products/1")
    assert resp.status_code == 200
    assert resp.json()["id"] == 1


def test_get_product_not_found():
    resp = client.get("/api/v1/products/9999")
    assert resp.status_code == 404
    assert resp.headers["content-type"] == "application/problem+json"
    body = resp.json()
    assert body["type"] == "https://api.example.com/problems/not-found"


def test_reserve_stock_success():
    resp = client.post("/api/v1/products/1/reserve", json={"quantity": 10})
    assert resp.status_code == 200
    body = resp.json()
    assert body["reserved"] == 10
    assert body["remaining"] == 90


def test_reserve_stock_insufficient():
    resp = client.post("/api/v1/products/1/reserve", json={"quantity": 999})
    assert resp.status_code == 409
    assert resp.headers["content-type"] == "application/problem+json"
    body = resp.json()
    assert body["status"] == 409


def test_reserve_invalid_cursor():
    resp = client.get("/api/v1/products?cursor=not-valid-base64!!!")
    assert resp.status_code == 422
```

```python
# tests/test_orders.py
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from src.order_service.main import app
from src.order_service import store as order_store

client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def reset_orders():
    order_store._orders.clear()
    order_store._idempotency.clear()
    order_store._next_id = 1
    yield


@pytest.fixture
def mock_inventory():
    """Patch the inventory reserve call to succeed."""
    with patch(
        "src.order_service.main._reserve_inventory",
        new_callable=AsyncMock,
        return_value=None,
    ) as mock:
        yield mock


def test_create_order_success(mock_inventory):
    resp = client.post(
        "/api/v1/orders",
        json={
            "customer_id": "cust-1",
            "items": [{"product_id": 1, "quantity": 2}],
            "idempotency_key": "idem-001",
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "confirmed"
    assert body["customer_id"] == "cust-1"


def test_create_order_idempotent_replay(mock_inventory):
    payload = {
        "customer_id": "cust-1",
        "items": [{"product_id": 1, "quantity": 2}],
        "idempotency_key": "idem-002",
    }
    resp1 = client.post("/api/v1/orders", json=payload)
    resp2 = client.post("/api/v1/orders", json=payload)
    assert resp1.status_code == 201
    assert resp2.status_code == 201
    assert resp1.json()["id"] == resp2.json()["id"]
    # inventory called only once
    assert mock_inventory.call_count == 1


def test_list_orders_empty():
    resp = client.get("/api/v1/orders")
    assert resp.status_code == 200
    assert resp.json()["data"] == []


def test_list_orders_cursor_pagination(mock_inventory):
    for i in range(10):
        client.post(
            "/api/v1/orders",
            json={
                "customer_id": f"cust-{i}",
                "items": [{"product_id": 1, "quantity": 1}],
                "idempotency_key": f"idem-{i}",
            },
        )
    resp = client.get("/api/v1/orders?limit=4")
    body = resp.json()
    assert len(body["data"]) == 4
    assert body["has_next"] is True


def test_get_order_not_found():
    resp = client.get("/api/v1/orders/9999")
    assert resp.status_code == 404
    assert resp.headers["content-type"] == "application/problem+json"


def test_insufficient_stock_returns_409(mock_inventory):
    mock_inventory.side_effect = ConflictError(
        "Insufficient stock", instance="/api/v1/orders/items/1"
    )
    resp = client.post(
        "/api/v1/orders",
        json={
            "customer_id": "cust-x",
            "items": [{"product_id": 1, "quantity": 9999}],
            "idempotency_key": "idem-conflict",
        },
    )
    assert resp.status_code == 409


# Fix import needed for last test
from src.shared.errors import ConflictError
```

```python
# tests/test_integration.py
"""
Integration tests: Order Service calls a real Inventory Service (both in-process).
"""
from __future__ import annotations

import threading

import pytest
import uvicorn
from fastapi.testclient import TestClient

from src.inventory_service.main import app as inv_app
from src.inventory_service import store as inv_store
from src.inventory_service.models import Product
from src.order_service.main import app as ord_app
from src.order_service import store as ord_store


@pytest.fixture(autouse=True)
def reset_all():
    inv_store._products = {
        i: Product(id=i, sku=f"SKU-{i:04d}", name=f"Product {i}", quantity=50)
        for i in range(1, 11)
    }
    ord_store._orders.clear()
    ord_store._idempotency.clear()
    ord_store._next_id = 1
    yield


inv_client = TestClient(inv_app, raise_server_exceptions=False)
ord_client = TestClient(ord_app, raise_server_exceptions=False)


def test_inventory_service_health():
    resp = inv_client.get("/api/v1/products/1")
    assert resp.status_code == 200
    assert resp.json()["quantity"] == 50


def test_reserve_reduces_inventory():
    inv_client.post("/api/v1/products/1/reserve", json={"quantity": 10})
    resp = inv_client.get("/api/v1/products/1")
    assert resp.json()["quantity"] == 40


def test_pagination_covers_all_products():
    all_ids: list[int] = []
    cursor = None
    while True:
        url = f"/api/v1/products?limit=3{('&cursor=' + cursor) if cursor else ''}"
        resp = inv_client.get(url)
        body = resp.json()
        all_ids.extend(p["id"] for p in body["data"])
        if not body["has_next"]:
            break
        cursor = body["next_cursor"]
    assert sorted(all_ids) == list(range(1, 11))
```

---

## ADR Documents

### ADR-001: Service Boundary — Orders and Inventory

```markdown
# ADR-001: Decompose by Business Domain, Not Technical Layer

**Status:** Accepted

## Decision
Split into Order Service and Inventory Service, each owning its data store
end-to-end, rather than splitting by presentation/business/data layers.

## Rationale
Domain decomposition aligns deployment boundaries with change boundaries.
The team owning order lifecycle (Order Service) does not need to coordinate
with the team owning stock levels (Inventory Service) for unrelated features.
Technical-layer decomposition would require all three layers to be deployed
together for every feature, negating the value of separate services.

## Consequences
Cross-service queries require API calls. Each service must version its own
API independently. No shared database tables.
```

### ADR-002: Cursor-Based Pagination

```markdown
# ADR-002: Cursor Pagination Over Offset Pagination

**Status:** Accepted

## Decision
All list endpoints use opaque base64-encoded cursor pagination.

## Rationale
Offset pagination scans O(offset) rows before returning results; cursor
pagination seeks directly to the last seen row via an indexed ID filter.
Offset pagination produces duplicate or missing rows when rows are inserted
between pages; cursor pagination is stable for append-heavy workloads.

## Consequences
Clients cannot jump to an arbitrary page. Random access requires cursor
traversal. Page numbers are not exposed in the API.
```

---

## Grading Guide

| Section | Points | Criteria |
|---|---|---|
| RFC 7807 errors | 20 | `Content-Type: application/problem+json`, all 5 required fields present, correct HTTP status codes, registered exception handler |
| Cursor pagination | 20 | Opaque cursor, base64 encoding/decoding, correct `has_next` detection (fetch limit+1), `next_cursor` is null on last page, no duplicates across pages |
| Idempotency | 15 | Client-supplied key, SHA-256 hashing, full response stored and replayed, inventory not called twice on replay |
| Service decomposition | 10 | Orders and inventory in separate FastAPI apps, no shared in-memory state, cross-service call via HTTP |
| API versioning | 5 | All routes under `/api/v1/` prefix |
| Pydantic v2 models | 10 | All request/response bodies typed, no untyped dicts crossing route boundaries |
| Test suite | 15 | Unit tests for pagination, error handlers, idempotency; at least one integration test; all mocked external HTTP calls |
| ADR | 5 | States decision, rationale, accepted trade-offs |
| **Total** | **100** | |

**Deductions:**
- `-10` per hardcoded secret or credential
- `-5` per `print()` instead of `logging`
- `-5` per missing type hint on function signature
- `-10` for shared DB state between services
- `-5` for missing `Content-Type: application/problem+json` header on error responses
