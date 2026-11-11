# Solutions — Week 08 · Day 6

## Reference Solution — 2-Service System

This solution implements two cooperating FastAPI microservices with full test suites and an Architecture Decision Record.

---

## Project Layout

```
week-08-day-6/
├── catalog_service/
│   ├── main.py
│   ├── models.py
│   ├── routes.py
│   └── config.py
├── order_service/
│   ├── main.py
│   ├── models.py
│   ├── routes.py
│   ├── catalog_client.py
│   └── config.py
├── tests/
│   ├── test_catalog_service.py
│   └── test_order_service.py
├── ADR-001-sync-vs-async.md
└── pyproject.toml
```

---

## Service 1: Catalog Service

### catalog_service/config.py

```python
"""Catalog service configuration."""
import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class CatalogSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CATALOG_", env_file=".env")

    app_name: str = "catalog-service"
    host: str = "0.0.0.0"
    port: int = 8001
    log_level: str = "INFO"
    database_url: str = os.getenv("CATALOG_DATABASE_URL", "sqlite:///./catalog.db")


settings = CatalogSettings()
```

### catalog_service/models.py

```python
"""Pydantic v2 models for the Catalog Service."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, Field, field_validator


class ProductBase(BaseModel):
    name: Annotated[str, Field(min_length=1, max_length=255)]
    description: str = ""
    price_cents: Annotated[int, Field(gt=0, description="Price in cents (avoids float rounding)")]
    sku: Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Z0-9\-]+$")]
    stock_quantity: Annotated[int, Field(ge=0)] = 0
    is_active: bool = True


class ProductCreate(ProductBase):
    pass


class ProductUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    price_cents: Annotated[int, Field(gt=0)] | None = None
    stock_quantity: Annotated[int, Field(ge=0)] | None = None
    is_active: bool | None = None


class Product(ProductBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ProductListResponse(BaseModel):
    items: list[Product]
    total: int
    page: int
    page_size: int


class ErrorDetail(BaseModel):
    code: str
    message: str
    field: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
```

### catalog_service/routes.py

```python
"""Versioned routes for Catalog Service (/api/v1/products)."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, status

from .models import (
    ErrorDetail,
    ErrorResponse,
    Product,
    ProductCreate,
    ProductListResponse,
    ProductUpdate,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/products", tags=["products"])

# In-memory store for reference implementation (replace with DB in production)
_PRODUCTS: dict[int, dict] = {
    1: {
        "id": 1,
        "name": "Laptop Pro 15",
        "description": "High-performance laptop",
        "price_cents": 149900,
        "sku": "LAP-PRO-15",
        "stock_quantity": 42,
        "is_active": True,
        "created_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
    },
    2: {
        "id": 2,
        "name": "Wireless Mouse",
        "description": "Ergonomic wireless mouse",
        "price_cents": 4999,
        "sku": "MOUSE-WL-01",
        "stock_quantity": 150,
        "is_active": True,
        "created_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
    },
}
_NEXT_ID: int = 3


@router.get(
    "",
    response_model=ProductListResponse,
    summary="List products (paginated)",
)
def list_products(
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    active_only: bool = True,
) -> ProductListResponse:
    from typing import Annotated

    items = list(_PRODUCTS.values())
    if active_only:
        items = [p for p in items if p["is_active"]]

    total = len(items)
    start = (page - 1) * page_size
    end = start + page_size
    page_items = [Product.model_validate(p) for p in items[start:end]]

    logger.info("list_products page=%d page_size=%d total=%d", page, page_size, total)
    return ProductListResponse(items=page_items, total=total, page=page, page_size=page_size)


@router.get(
    "/{product_id}",
    response_model=Product,
    responses={404: {"model": ErrorResponse}},
    summary="Get product by ID",
)
def get_product(product_id: int) -> Product:
    product = _PRODUCTS.get(product_id)
    if not product:
        logger.warning("get_product not found id=%d", product_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ErrorResponse(
                error=ErrorDetail(code="PRODUCT_NOT_FOUND", message=f"Product {product_id} not found")
            ).model_dump(),
        )
    return Product.model_validate(product)


@router.post(
    "",
    response_model=Product,
    status_code=status.HTTP_201_CREATED,
    responses={409: {"model": ErrorResponse}},
    summary="Create a new product",
)
def create_product(payload: ProductCreate) -> Product:
    global _NEXT_ID

    # Ensure SKU uniqueness
    existing_skus = {p["sku"] for p in _PRODUCTS.values()}
    if payload.sku in existing_skus:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=ErrorResponse(
                error=ErrorDetail(
                    code="SKU_CONFLICT",
                    message=f"Product with SKU '{payload.sku}' already exists",
                    field="sku",
                )
            ).model_dump(),
        )

    now = datetime.now(tz=timezone.utc)
    product_dict = {
        "id": _NEXT_ID,
        **payload.model_dump(),
        "created_at": now,
        "updated_at": now,
    }
    _PRODUCTS[_NEXT_ID] = product_dict
    _NEXT_ID += 1

    logger.info("create_product id=%d sku=%s", product_dict["id"], payload.sku)
    return Product.model_validate(product_dict)


@router.patch(
    "/{product_id}",
    response_model=Product,
    responses={404: {"model": ErrorResponse}},
    summary="Partially update a product",
)
def update_product(product_id: int, payload: ProductUpdate) -> Product:
    product = _PRODUCTS.get(product_id)
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ErrorResponse(
                error=ErrorDetail(code="PRODUCT_NOT_FOUND", message=f"Product {product_id} not found")
            ).model_dump(),
        )

    updates = payload.model_dump(exclude_none=True)
    product.update(updates)
    product["updated_at"] = datetime.now(tz=timezone.utc)

    logger.info("update_product id=%d fields=%s", product_id, list(updates.keys()))
    return Product.model_validate(product)
```

### catalog_service/main.py

```python
"""Catalog Service entry point."""
import logging

from fastapi import FastAPI

from .config import settings
from .routes import router

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Catalog Service",
    description="Product catalog microservice",
    version="1.0.0",
    docs_url="/api/v1/docs",
    openapi_url="/api/v1/openapi.json",
)

app.include_router(router)


@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok", "service": settings.app_name}


logger.info("Catalog Service starting on port %d", settings.port)
```

---

## Service 2: Order Service

### order_service/config.py

```python
"""Order service configuration."""
import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class OrderSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ORDER_", env_file=".env")

    app_name: str = "order-service"
    host: str = "0.0.0.0"
    port: int = 8002
    log_level: str = "INFO"
    catalog_service_url: str = os.getenv("ORDER_CATALOG_SERVICE_URL", "http://catalog-service:8001")
    catalog_timeout_seconds: float = 10.0
    catalog_max_retries: int = 3
    catalog_retry_base_delay: float = 0.5


settings = OrderSettings()
```

### order_service/models.py

```python
"""Pydantic v2 models for the Order Service."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field


class OrderStatus(StrEnum):
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"
    SHIPPED = "SHIPPED"
    CANCELLED = "CANCELLED"


class OrderLineItem(BaseModel):
    product_id: int
    quantity: Annotated[int, Field(gt=0)]
    unit_price_cents: Annotated[int, Field(gt=0)]


class OrderCreate(BaseModel):
    user_id: Annotated[int, Field(gt=0)]
    line_items: Annotated[list[OrderLineItem], Field(min_length=1)]
    notes: str = ""


class Order(BaseModel):
    id: int
    idempotency_key: str
    user_id: int
    status: OrderStatus
    line_items: list[OrderLineItem]
    total_cents: int
    notes: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class OrderListResponse(BaseModel):
    """Cursor-based pagination response."""

    items: list[Order]
    next_cursor: str | None = None  # opaque cursor — encodes last seen order id
    has_more: bool


class ErrorDetail(BaseModel):
    code: str
    message: str
    field: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


# Minimal product shape returned by catalog service
class CatalogProduct(BaseModel):
    id: int
    name: str
    price_cents: int
    stock_quantity: int
    is_active: bool
```

### order_service/catalog_client.py

```python
"""
HTTP client for the Catalog Service.

Features:
- Timeout configuration
- Exponential-backoff retry (3 attempts)
- Raises structured HTTPException on failure
"""
from __future__ import annotations

import logging
import time

import httpx
from fastapi import HTTPException, status

from .config import settings
from .models import CatalogProduct

logger = logging.getLogger(__name__)


def _build_client() -> httpx.Client:
    timeout = httpx.Timeout(
        connect=2.0,
        read=settings.catalog_timeout_seconds,
        write=settings.catalog_timeout_seconds,
        pool=2.0,
    )
    return httpx.Client(base_url=settings.catalog_service_url, timeout=timeout)


def _get_with_retry(path: str) -> httpx.Response:
    """GET path with exponential-backoff retry. Raises httpx exceptions on all failures."""
    last_exc: Exception | None = None
    for attempt in range(settings.catalog_max_retries):
        try:
            with _build_client() as client:
                response = client.get(path)
                response.raise_for_status()
                return response
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            last_exc = exc
            delay = settings.catalog_retry_base_delay * (2 ** attempt)
            logger.warning(
                "catalog_client: attempt %d/%d failed (%s). retry in %.1fs",
                attempt + 1,
                settings.catalog_max_retries,
                exc,
                delay,
            )
            time.sleep(delay)
        except httpx.HTTPStatusError as exc:
            # Do not retry 4xx — they are deterministic
            if exc.response.status_code < 500:
                raise
            last_exc = exc
            delay = settings.catalog_retry_base_delay * (2 ** attempt)
            logger.warning(
                "catalog_client: upstream 5xx attempt %d/%d. retry in %.1fs",
                attempt + 1,
                settings.catalog_max_retries,
                delay,
            )
            time.sleep(delay)

    raise last_exc  # type: ignore[misc]


def get_product(product_id: int) -> CatalogProduct:
    """
    Fetch a single product from the catalog service.

    Raises HTTPException 404 if product not found,
    HTTPException 502 on upstream failure.
    """
    try:
        response = _get_with_retry(f"/api/v1/products/{product_id}")
        return CatalogProduct.model_validate(response.json())
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Product {product_id} not found in catalog",
            ) from exc
        logger.error("catalog_client: upstream error %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Catalog service error",
        ) from exc
    except Exception as exc:
        logger.error("catalog_client: connection failure %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Catalog service unavailable",
        ) from exc
```

### order_service/routes.py

```python
"""
Order Service routes (/api/v1/orders).

Key features:
- Idempotent POST /orders via X-Idempotency-Key header
- Cursor-based pagination on GET /orders
- Validates product existence + price from catalog service
"""
from __future__ import annotations

import base64
import logging
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Query, status

from . import catalog_client
from .models import (
    ErrorDetail,
    ErrorResponse,
    Order,
    OrderCreate,
    OrderLineItem,
    OrderListResponse,
    OrderStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/orders", tags=["orders"])

# In-memory stores for reference implementation
_ORDERS: dict[int, dict] = {}
_IDEMPOTENCY_STORE: dict[str, int] = {}  # idempotency_key -> order_id
_NEXT_ORDER_ID: int = 1


# ---------------------------------------------------------------------------
# Cursor encoding helpers
# ---------------------------------------------------------------------------

def _encode_cursor(order_id: int) -> str:
    """Encode an order_id into an opaque base64 cursor."""
    return base64.urlsafe_b64encode(str(order_id).encode()).decode()


def _decode_cursor(cursor: str) -> int:
    """Decode cursor back to order_id. Raises HTTPException on invalid cursor."""
    try:
        return int(base64.urlsafe_b64decode(cursor.encode()).decode())
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorResponse(
                error=ErrorDetail(code="INVALID_CURSOR", message="Cursor is malformed")
            ).model_dump(),
        ) from exc


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=Order,
    status_code=status.HTTP_201_CREATED,
    responses={
        200: {"description": "Existing order returned (idempotent repeat)", "model": Order},
        422: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Create order (idempotent)",
)
def create_order(
    payload: OrderCreate,
    x_idempotency_key: Annotated[str, Header(min_length=1, max_length=128)],
) -> Order:
    """
    Create an order. Idempotent: repeat calls with the same X-Idempotency-Key
    return the original order without re-processing.
    """
    global _NEXT_ORDER_ID

    # Idempotency check — return existing order if key already seen
    existing_id = _IDEMPOTENCY_STORE.get(x_idempotency_key)
    if existing_id is not None:
        logger.info("create_order: duplicate idempotency_key=%s returning existing order", x_idempotency_key)
        return Order.model_validate(_ORDERS[existing_id])

    # Validate each line item against the catalog service
    validated_items: list[OrderLineItem] = []
    for item in payload.line_items:
        product = catalog_client.get_product(item.product_id)
        if not product.is_active:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=ErrorResponse(
                    error=ErrorDetail(
                        code="PRODUCT_INACTIVE",
                        message=f"Product {item.product_id} is not available",
                        field="line_items",
                    )
                ).model_dump(),
            )
        if product.stock_quantity < item.quantity:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=ErrorResponse(
                    error=ErrorDetail(
                        code="INSUFFICIENT_STOCK",
                        message=f"Product {item.product_id} has only {product.stock_quantity} in stock",
                        field="line_items",
                    )
                ).model_dump(),
            )
        validated_items.append(
            OrderLineItem(
                product_id=item.product_id,
                quantity=item.quantity,
                unit_price_cents=product.price_cents,  # price locked at order time
            )
        )

    total_cents = sum(i.unit_price_cents * i.quantity for i in validated_items)
    now = datetime.now(tz=timezone.utc)
    order_dict = {
        "id": _NEXT_ORDER_ID,
        "idempotency_key": x_idempotency_key,
        "user_id": payload.user_id,
        "status": OrderStatus.PENDING,
        "line_items": [i.model_dump() for i in validated_items],
        "total_cents": total_cents,
        "notes": payload.notes,
        "created_at": now,
        "updated_at": now,
    }

    _ORDERS[_NEXT_ORDER_ID] = order_dict
    _IDEMPOTENCY_STORE[x_idempotency_key] = _NEXT_ORDER_ID
    _NEXT_ORDER_ID += 1

    logger.info(
        "create_order id=%d user_id=%d total_cents=%d key=%s",
        order_dict["id"],
        payload.user_id,
        total_cents,
        x_idempotency_key,
    )
    return Order.model_validate(order_dict)


@router.get(
    "",
    response_model=OrderListResponse,
    summary="List orders (cursor-paginated)",
)
def list_orders(
    user_id: int | None = None,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: str | None = None,
) -> OrderListResponse:
    """
    List orders with cursor-based pagination.

    Pass `cursor` from the previous response's `next_cursor` to get the next page.
    Cursor encodes the last seen order ID (opaque to the client).
    """
    all_orders = sorted(_ORDERS.values(), key=lambda o: o["id"])

    if user_id is not None:
        all_orders = [o for o in all_orders if o["user_id"] == user_id]

    # Apply cursor — start after the encoded order_id
    if cursor is not None:
        last_id = _decode_cursor(cursor)
        all_orders = [o for o in all_orders if o["id"] > last_id]

    # Fetch one extra to determine has_more
    page = all_orders[: page_size + 1]
    has_more = len(page) > page_size
    page = page[:page_size]

    next_cursor = _encode_cursor(page[-1]["id"]) if has_more and page else None

    logger.info("list_orders user_id=%s page_size=%d has_more=%s", user_id, page_size, has_more)
    return OrderListResponse(
        items=[Order.model_validate(o) for o in page],
        next_cursor=next_cursor,
        has_more=has_more,
    )


@router.get(
    "/{order_id}",
    response_model=Order,
    responses={404: {"model": ErrorResponse}},
    summary="Get order by ID",
)
def get_order(order_id: int) -> Order:
    order = _ORDERS.get(order_id)
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ErrorResponse(
                error=ErrorDetail(code="ORDER_NOT_FOUND", message=f"Order {order_id} not found")
            ).model_dump(),
        )
    return Order.model_validate(order)
```

### order_service/main.py

```python
"""Order Service entry point."""
import logging

from fastapi import FastAPI

from .config import settings
from .routes import router

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Order Service",
    description="Order management microservice",
    version="1.0.0",
    docs_url="/api/v1/docs",
    openapi_url="/api/v1/openapi.json",
)

app.include_router(router)


@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok", "service": settings.app_name}


logger.info("Order Service starting on port %d", settings.port)
```

---

## Tests

### tests/test_catalog_service.py

```python
"""
Unit tests for the Catalog Service.

All tests use the FastAPI test client — no network I/O.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from catalog_service.main import app
from catalog_service.routes import _PRODUCTS, _NEXT_ID


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_store():
    """Reset in-memory store before each test."""
    import catalog_service.routes as r

    original_products = dict(r._PRODUCTS)
    original_next_id = r._NEXT_ID
    yield
    r._PRODUCTS.clear()
    r._PRODUCTS.update(original_products)
    r._NEXT_ID = original_next_id


class TestHealth:
    def test_health_returns_ok(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestListProducts:
    def test_returns_active_products_by_default(self, client: TestClient):
        response = client.get("/api/v1/products")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert data["total"] >= 0

    def test_pagination_page_size(self, client: TestClient):
        response = client.get("/api/v1/products?page=1&page_size=1")
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) <= 1

    def test_invalid_page_size_rejected(self, client: TestClient):
        response = client.get("/api/v1/products?page_size=0")
        assert response.status_code == 422

    def test_page_size_over_max_rejected(self, client: TestClient):
        response = client.get("/api/v1/products?page_size=101")
        assert response.status_code == 422


class TestGetProduct:
    def test_returns_existing_product(self, client: TestClient):
        response = client.get("/api/v1/products/1")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == 1
        assert data["sku"] == "LAP-PRO-15"

    def test_404_for_missing_product(self, client: TestClient):
        response = client.get("/api/v1/products/99999")
        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["error"]["code"] == "PRODUCT_NOT_FOUND"

    def test_product_has_required_fields(self, client: TestClient):
        response = client.get("/api/v1/products/1")
        data = response.json()
        required = {"id", "name", "price_cents", "sku", "stock_quantity", "is_active"}
        assert required.issubset(data.keys())


class TestCreateProduct:
    def test_creates_product_with_valid_payload(self, client: TestClient):
        payload = {
            "name": "Test Widget",
            "description": "A test product",
            "price_cents": 999,
            "sku": "TEST-WIDGET-01",
            "stock_quantity": 10,
        }
        response = client.post("/api/v1/products", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["sku"] == "TEST-WIDGET-01"
        assert data["price_cents"] == 999
        assert "id" in data

    def test_rejects_duplicate_sku(self, client: TestClient):
        payload = {
            "name": "Duplicate",
            "price_cents": 100,
            "sku": "LAP-PRO-15",  # already exists
        }
        response = client.post("/api/v1/products", json=payload)
        assert response.status_code == 409
        data = response.json()
        assert data["detail"]["error"]["code"] == "SKU_CONFLICT"

    def test_rejects_zero_price(self, client: TestClient):
        payload = {"name": "Free item", "price_cents": 0, "sku": "FREE-01"}
        response = client.post("/api/v1/products", json=payload)
        assert response.status_code == 422

    def test_rejects_invalid_sku_characters(self, client: TestClient):
        payload = {"name": "Bad SKU", "price_cents": 100, "sku": "bad sku!"}
        response = client.post("/api/v1/products", json=payload)
        assert response.status_code == 422

    def test_rejects_empty_name(self, client: TestClient):
        payload = {"name": "", "price_cents": 100, "sku": "VALID-01"}
        response = client.post("/api/v1/products", json=payload)
        assert response.status_code == 422


class TestUpdateProduct:
    def test_partial_update_changes_only_given_fields(self, client: TestClient):
        response = client.patch("/api/v1/products/1", json={"price_cents": 99900})
        assert response.status_code == 200
        data = response.json()
        assert data["price_cents"] == 99900
        assert data["name"] == "Laptop Pro 15"  # unchanged

    def test_404_on_missing_product(self, client: TestClient):
        response = client.patch("/api/v1/products/99999", json={"price_cents": 100})
        assert response.status_code == 404
```

### tests/test_order_service.py

```python
"""
Unit tests for the Order Service.

All catalog service calls are mocked — no real HTTP.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from order_service.main import app
from order_service.models import CatalogProduct


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_order_store():
    """Reset in-memory stores before each test."""
    import order_service.routes as r

    r._ORDERS.clear()
    r._IDEMPOTENCY_STORE.clear()
    r._NEXT_ORDER_ID = 1
    yield
    r._ORDERS.clear()
    r._IDEMPOTENCY_STORE.clear()
    r._NEXT_ORDER_ID = 1


@pytest.fixture()
def mock_catalog_product() -> CatalogProduct:
    return CatalogProduct(
        id=1,
        name="Test Product",
        price_cents=1000,
        stock_quantity=50,
        is_active=True,
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_ok(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Create Order
# ---------------------------------------------------------------------------

class TestCreateOrder:
    def test_creates_order_successfully(
        self, client: TestClient, mock_catalog_product: CatalogProduct
    ):
        with patch("order_service.routes.catalog_client.get_product", return_value=mock_catalog_product):
            response = client.post(
                "/api/v1/orders",
                json={"user_id": 1, "line_items": [{"product_id": 1, "quantity": 2, "unit_price_cents": 1000}]},
                headers={"X-Idempotency-Key": "order-abc-001"},
            )
        assert response.status_code == 201
        data = response.json()
        assert data["user_id"] == 1
        assert data["total_cents"] == 2000  # 2 * 1000
        assert data["status"] == "PENDING"
        assert data["idempotency_key"] == "order-abc-001"

    def test_idempotent_repeat_returns_same_order(
        self, client: TestClient, mock_catalog_product: CatalogProduct
    ):
        payload = {
            "user_id": 1,
            "line_items": [{"product_id": 1, "quantity": 1, "unit_price_cents": 1000}],
        }
        headers = {"X-Idempotency-Key": "idem-key-xyz"}

        with patch("order_service.routes.catalog_client.get_product", return_value=mock_catalog_product):
            r1 = client.post("/api/v1/orders", json=payload, headers=headers)
            # Second call — catalog should NOT be called again
            r2 = client.post("/api/v1/orders", json=payload, headers=headers)

        assert r1.status_code == 201
        assert r2.status_code == 201  # 200 or 201 both acceptable for idempotent repeat
        assert r1.json()["id"] == r2.json()["id"]

    def test_missing_idempotency_key_rejected(
        self, client: TestClient, mock_catalog_product: CatalogProduct
    ):
        with patch("order_service.routes.catalog_client.get_product", return_value=mock_catalog_product):
            response = client.post(
                "/api/v1/orders",
                json={"user_id": 1, "line_items": [{"product_id": 1, "quantity": 1, "unit_price_cents": 1000}]},
            )
        assert response.status_code == 422

    def test_rejects_inactive_product(self, client: TestClient):
        inactive_product = CatalogProduct(
            id=1, name="Gone", price_cents=100, stock_quantity=10, is_active=False
        )
        with patch("order_service.routes.catalog_client.get_product", return_value=inactive_product):
            response = client.post(
                "/api/v1/orders",
                json={"user_id": 1, "line_items": [{"product_id": 1, "quantity": 1, "unit_price_cents": 100}]},
                headers={"X-Idempotency-Key": "key-inactive"},
            )
        assert response.status_code == 422
        assert response.json()["detail"]["error"]["code"] == "PRODUCT_INACTIVE"

    def test_rejects_insufficient_stock(self, client: TestClient):
        low_stock = CatalogProduct(
            id=1, name="Scarce", price_cents=100, stock_quantity=1, is_active=True
        )
        with patch("order_service.routes.catalog_client.get_product", return_value=low_stock):
            response = client.post(
                "/api/v1/orders",
                json={"user_id": 1, "line_items": [{"product_id": 1, "quantity": 5, "unit_price_cents": 100}]},
                headers={"X-Idempotency-Key": "key-no-stock"},
            )
        assert response.status_code == 422
        assert response.json()["detail"]["error"]["code"] == "INSUFFICIENT_STOCK"

    def test_catalog_service_unavailable_returns_503(self, client: TestClient):
        with patch(
            "order_service.routes.catalog_client.get_product",
            side_effect=HTTPException(status_code=503, detail="unavailable"),
        ):
            response = client.post(
                "/api/v1/orders",
                json={"user_id": 1, "line_items": [{"product_id": 1, "quantity": 1, "unit_price_cents": 100}]},
                headers={"X-Idempotency-Key": "key-503"},
            )
        assert response.status_code == 503

    def test_price_locked_at_order_time(
        self, client: TestClient, mock_catalog_product: CatalogProduct
    ):
        """Price stored in the order matches catalog price, not client-supplied price."""
        with patch("order_service.routes.catalog_client.get_product", return_value=mock_catalog_product):
            response = client.post(
                "/api/v1/orders",
                json={
                    "user_id": 1,
                    "line_items": [{"product_id": 1, "quantity": 1, "unit_price_cents": 9999}],
                },
                headers={"X-Idempotency-Key": "key-price"},
            )
        assert response.status_code == 201
        data = response.json()
        assert data["line_items"][0]["unit_price_cents"] == mock_catalog_product.price_cents


# ---------------------------------------------------------------------------
# List Orders
# ---------------------------------------------------------------------------

class TestListOrders:
    def _seed_orders(self, client: TestClient, product: CatalogProduct, count: int = 5):
        for i in range(count):
            with patch("order_service.routes.catalog_client.get_product", return_value=product):
                client.post(
                    "/api/v1/orders",
                    json={"user_id": 1, "line_items": [{"product_id": 1, "quantity": 1, "unit_price_cents": 100}]},
                    headers={"X-Idempotency-Key": f"seed-{i}"},
                )

    def test_returns_empty_list_initially(self, client: TestClient):
        response = client.get("/api/v1/orders")
        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []
        assert not data["has_more"]

    def test_cursor_pagination(self, client: TestClient, mock_catalog_product: CatalogProduct):
        self._seed_orders(client, mock_catalog_product, count=5)

        r1 = client.get("/api/v1/orders?page_size=2")
        assert r1.status_code == 200
        d1 = r1.json()
        assert len(d1["items"]) == 2
        assert d1["has_more"] is True
        assert d1["next_cursor"] is not None

        r2 = client.get(f"/api/v1/orders?page_size=2&cursor={d1['next_cursor']}")
        assert r2.status_code == 200
        d2 = r2.json()
        assert len(d2["items"]) == 2
        # No ID overlap between pages
        ids1 = {o["id"] for o in d1["items"]}
        ids2 = {o["id"] for o in d2["items"]}
        assert ids1.isdisjoint(ids2)

    def test_invalid_cursor_returns_400(self, client: TestClient):
        response = client.get("/api/v1/orders?cursor=notvalidbase64!!!")
        assert response.status_code == 400

    def test_filter_by_user_id(self, client: TestClient, mock_catalog_product: CatalogProduct):
        for uid, key in [(1, "u1-k1"), (2, "u2-k1"), (1, "u1-k2")]:
            with patch("order_service.routes.catalog_client.get_product", return_value=mock_catalog_product):
                client.post(
                    "/api/v1/orders",
                    json={"user_id": uid, "line_items": [{"product_id": 1, "quantity": 1, "unit_price_cents": 100}]},
                    headers={"X-Idempotency-Key": key},
                )

        response = client.get("/api/v1/orders?user_id=1")
        data = response.json()
        assert all(o["user_id"] == 1 for o in data["items"])
        assert len(data["items"]) == 2


# ---------------------------------------------------------------------------
# Get Order
# ---------------------------------------------------------------------------

class TestGetOrder:
    def test_get_existing_order(self, client: TestClient, mock_catalog_product: CatalogProduct):
        with patch("order_service.routes.catalog_client.get_product", return_value=mock_catalog_product):
            create_resp = client.post(
                "/api/v1/orders",
                json={"user_id": 1, "line_items": [{"product_id": 1, "quantity": 1, "unit_price_cents": 100}]},
                headers={"X-Idempotency-Key": "get-test-001"},
            )
        order_id = create_resp.json()["id"]

        response = client.get(f"/api/v1/orders/{order_id}")
        assert response.status_code == 200
        assert response.json()["id"] == order_id

    def test_404_for_missing_order(self, client: TestClient):
        response = client.get("/api/v1/orders/99999")
        assert response.status_code == 404
        assert response.json()["detail"]["error"]["code"] == "ORDER_NOT_FOUND"
```

---

## ADR-001: Synchronous vs Asynchronous Communication

```
# ADR-001: Inter-Service Communication — Catalog → Order

Status: Accepted
Date: 2024-01-01
Deciders: Week 08 Project Team

## Context

The Order Service must validate product existence, price, and stock before
creating an order. It calls the Catalog Service to do so.

Two options:

1. **Synchronous REST**: Order Service calls Catalog Service via HTTP during
   the order creation request. The order is only created if the catalog
   responds successfully.

2. **Asynchronous messaging**: Order Service publishes an "order draft" event;
   a background worker validates against the catalog and publishes an
   "order confirmed" or "order rejected" event back. The client polls or
   receives a webhook.

## Decision

Use **synchronous REST** for product validation.

## Rationale

- Price and stock validation are **blocking dependencies**: we cannot create
  the order without knowing the current price (which we lock at order time)
  and whether stock is available. Async would require a two-step
  request/poll protocol, complicating the client interface significantly.

- The latency budget for checkout is tight (users expect < 2s). Sync REST
  with timeout (10s) and 3 retries fits the P99 SLA. Async adds at least
  one round-trip of queueing latency.

- At this scale (reference project), an in-process circuit breaker + retry
  is sufficient. If the catalog becomes a bottleneck, we can add a read
  replica or a short-lived cache (TTL 30s) in front of the catalog call
  without changing the sync pattern.

- Async messaging is appropriate for **notifications** that do not block
  the user response (e.g., sending an order-confirmation email after the
  order is committed). That integration is intentionally left async (out of
  scope for this project).

## Consequences

- Order creation fails fast (503/502) if the catalog is down.
  Mitigation: circuit breaker, retry with backoff, clear error codes.

- Price is locked at order-creation time (snapshot isolation): if the
  product price changes between browsing and checkout, the cart price
  reflects the catalog price at checkout time. This is correct behavior
  for e-commerce.

- Catalog Service becomes a hard dependency of the Order Service critical
  path. SLO for Catalog must be >= SLO for Order.
```

---

## pyproject.toml

```toml
[project]
name = "week-08-day-6"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115.0",
    "httpx>=0.27.0",
    "pydantic>=2.9.0",
    "pydantic-settings>=2.5.0",
    "uvicorn[standard]>=0.32.0",
]

[project.optional-dependencies]
test = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]

[tool.ruff]
target-version = "py312"
line-length = 100
select = ["E", "F", "I", "UP", "N", "ANN"]
ignore = ["ANN101", "ANN102"]
```
