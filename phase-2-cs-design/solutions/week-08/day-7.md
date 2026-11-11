# Solutions — Week 08 · Day 7 (Integration Day)

## Exercise 1 — RFC 7807 Problem Details

Add RFC 7807 `application/problem+json` responses to the 2-service system.

### FastAPI exception handler

```python
# src/shared/errors.py
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


@dataclass
class ProblemDetail:
    """RFC 7807 Problem Details object."""

    type: str                      # URI reference identifying the problem type
    title: str                     # Short human-readable summary
    status: int                    # HTTP status code
    detail: str                    # Human-readable explanation for this occurrence
    instance: str                  # URI reference identifying this occurrence
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
    """Base application exception that maps to RFC 7807."""

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


class ValidationError(AppError):
    def __init__(self, detail: str, fields: list[str] | None = None) -> None:
        super().__init__(
            problem_type="https://api.example.com/problems/validation-error",
            title="Validation Error",
            status=422,
            detail=detail,
            extra={"invalid_fields": fields or []},
        )


class ConflictError(AppError):
    def __init__(self, detail: str) -> None:
        super().__init__(
            problem_type="https://api.example.com/problems/conflict",
            title="Conflict",
            status=409,
            detail=detail,
        )
```

```python
# src/shared/middleware.py
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .errors import AppError

logger = logging.getLogger(__name__)

PROBLEM_JSON_CONTENT_TYPE = "application/problem+json"


def register_error_handlers(app: FastAPI) -> None:
    """Register RFC 7807-compliant error handlers on a FastAPI app."""

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        problem = exc.problem
        logger.warning(
            "Application error",
            extra={
                "type": problem.type,
                "status": problem.status,
                "instance": problem.instance,
                "path": request.url.path,
            },
        )
        return JSONResponse(
            status_code=problem.status,
            content=problem.to_dict(),
            headers={"Content-Type": PROBLEM_JSON_CONTENT_TYPE},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception at %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "type": "https://api.example.com/problems/internal-error",
                "title": "Internal Server Error",
                "status": 500,
                "detail": "An unexpected error occurred. Please try again later.",
                "instance": str(request.url.path),
            },
            headers={"Content-Type": PROBLEM_JSON_CONTENT_TYPE},
        )
```

```python
# src/order_service/main.py  (wiring example)
from fastapi import FastAPI
from shared.middleware import register_error_handlers
from shared.errors import NotFoundError, ValidationError

app = FastAPI(title="Order Service", version="1.0.0")
register_error_handlers(app)


@app.get("/api/v1/orders/{order_id}")
async def get_order(order_id: int) -> dict:
    order = _db.get(order_id)
    if order is None:
        raise NotFoundError("Order", order_id)
    return order
```

**Key points:**
- `Content-Type: application/problem+json` distinguishes error bodies from success bodies.
- `type` is a stable URI — consumers can map it to i18n strings without parsing `detail`.
- `instance` lets clients correlate a specific occurrence in logs.
- `extra` carries domain-specific extension members (e.g., `invalid_fields`).

---

## Exercise 2 — Cursor-Based Pagination

Replace offset pagination with cursor pagination on both services.

### Cursor encoding/decoding

```python
# src/shared/pagination.py
from __future__ import annotations

import base64
import json
import logging
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

T = TypeVar("T")


def encode_cursor(last_id: int | str) -> str:
    """Encode last seen ID into an opaque cursor string."""
    payload = json.dumps({"last_id": last_id}, separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode()).decode()


def decode_cursor(cursor: str) -> int | str:
    """Decode cursor back to last seen ID. Raises ValueError on invalid cursor."""
    try:
        payload = base64.urlsafe_b64decode(cursor.encode()).decode()
        data = json.loads(payload)
        return data["last_id"]
    except Exception as exc:
        raise ValueError(f"Invalid cursor: {cursor!r}") from exc


class PageRequest(BaseModel):
    cursor: str | None = None
    limit: int = 20

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, v: int) -> int:
        if not (1 <= v <= 100):
            raise ValueError("limit must be between 1 and 100")
        return v


class PageResponse(BaseModel, Generic[T]):
    data: list[T]
    next_cursor: str | None
    has_next: bool
    limit: int
```

### Order Service paginated endpoint

```python
# src/order_service/routes/orders.py
import logging
from typing import Annotated

from fastapi import APIRouter, Query
from pydantic import BaseModel

from shared.errors import ValidationError
from shared.pagination import PageRequest, PageResponse, decode_cursor, encode_cursor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/orders", tags=["orders"])


class Order(BaseModel):
    id: int
    customer_id: str
    amount: float
    status: str


# In-memory store for demonstration; replace with actual DB query
_orders: list[Order] = [
    Order(id=i, customer_id=f"cust-{i % 10}", amount=float(i * 10), status="confirmed")
    for i in range(1, 201)
]


@router.get("", response_model=PageResponse[Order])
async def list_orders(
    cursor: Annotated[str | None, Query(description="Opaque pagination cursor")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> PageResponse[Order]:
    req = PageRequest(cursor=cursor, limit=limit)

    # Determine starting point
    start_id: int = 0
    if req.cursor is not None:
        try:
            last_id = decode_cursor(req.cursor)
            start_id = int(last_id)
        except ValueError as exc:
            raise ValidationError(str(exc), fields=["cursor"]) from exc

    # Filter rows with id > start_id (cursor points to last seen, not next)
    filtered = [o for o in _orders if o.id > start_id]
    page = filtered[: req.limit + 1]          # fetch one extra to detect has_next
    has_next = len(page) > req.limit
    items = page[: req.limit]

    next_cursor: str | None = None
    if has_next and items:
        next_cursor = encode_cursor(items[-1].id)

    logger.debug(
        "list_orders cursor=%s limit=%d returned=%d has_next=%s",
        cursor, req.limit, len(items), has_next,
    )
    return PageResponse(
        data=items,
        next_cursor=next_cursor,
        has_next=has_next,
        limit=req.limit,
    )
```

### Inventory Service paginated endpoint (same pattern)

```python
# src/inventory_service/routes/products.py
import logging
from typing import Annotated

from fastapi import APIRouter, Query
from pydantic import BaseModel

from shared.errors import ValidationError
from shared.pagination import PageRequest, PageResponse, decode_cursor, encode_cursor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/products", tags=["products"])


class Product(BaseModel):
    id: int
    sku: str
    name: str
    quantity: int


_products: list[Product] = [
    Product(id=i, sku=f"SKU-{i:04d}", name=f"Product {i}", quantity=i * 5)
    for i in range(1, 101)
]


@router.get("", response_model=PageResponse[Product])
async def list_products(
    cursor: Annotated[str | None, Query(description="Opaque pagination cursor")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> PageResponse[Product]:
    req = PageRequest(cursor=cursor, limit=limit)

    start_id: int = 0
    if req.cursor is not None:
        try:
            start_id = int(decode_cursor(req.cursor))
        except ValueError as exc:
            raise ValidationError(str(exc), fields=["cursor"]) from exc

    filtered = [p for p in _products if p.id > start_id]
    page = filtered[: req.limit + 1]
    has_next = len(page) > req.limit
    items = page[: req.limit]

    next_cursor: str | None = encode_cursor(items[-1].id) if has_next and items else None

    return PageResponse(
        data=items,
        next_cursor=next_cursor,
        has_next=has_next,
        limit=req.limit,
    )
```

**Why cursor over offset:**
- Offset skips rows by scanning and discarding — O(offset) work. Cursor filters by indexed ID — O(1) seek.
- Offset pagination produces duplicates or skips rows when rows are inserted between pages. Cursor is stable as long as the sort key is immutable (primary key or created_at).
- Cursor is opaque: callers cannot construct cursors manually, so the internal representation can change.

---

## Exercise 3 — Architecture Decision Record: Eventual Consistency

```markdown
# ADR-003: Eventual Consistency for Cross-Service Reads

**Status:** Accepted
**Date:** 2026-11-11
**Deciders:** Backend Platform Team

---

## Context

The Order Service and Inventory Service are independently deployable. Order creation
requires reading current inventory levels to check stock. Two options exist:
1. Synchronous REST call to Inventory Service during order creation (strong consistency).
2. Order Service holds a replicated read cache of inventory, updated asynchronously
   (eventual consistency).

We have observed that the Inventory Service is the system bottleneck during peak sales
events: synchronous coupling causes cascading latency spikes and upstream timeouts in
the Order Service whenever Inventory is under load.

---

## Decision

We chose **eventual consistency for cross-service inventory reads**.

The Order Service maintains a local inventory snapshot updated via domain events published
by the Inventory Service over a message broker (Apache Kafka). Order creation reads from
this local snapshot rather than calling Inventory Service synchronously.

---

## Rationale

Synchronous coupling violates the independent deployability requirement: an Inventory
Service outage would cascade to order acceptance, taking down revenue-critical flows.
Eventual consistency decouples availability domains — the Order Service continues
accepting orders even when Inventory Service is degraded.

Business analysis shows that inventory data is read far more frequently than it is
updated (approximately 50:1 read-to-write ratio). A local replicated cache therefore
eliminates 98% of cross-service calls while accepting bounded staleness.

---

## Consistency Lag Accepted

Under normal operating conditions the replication lag is under 500 ms. In the worst case
(consumer group lag during a broker restart) the snapshot may be up to 30 seconds stale.

This means a small percentage of orders may be accepted against a stale inventory count,
potentially allowing momentary overselling by at most the volume of orders processed
within the lag window.

---

## Mitigations

| Risk | Mitigation |
|---|---|
| Overselling during replication lag | Inventory Service enforces a hard reservation at fulfillment time; orders that cannot be fulfilled are refunded automatically via compensating transaction |
| Snapshot divergence after broker partition | Periodic full-state sync job (every 5 minutes) resets snapshot from Inventory Service REST API |
| Consumer falling arbitrarily behind | Consumer lag alert fires at > 10 s; on-call SRE investigates within 5 minutes |
| Silent data corruption in snapshot | Checksum comparison between snapshot and authoritative state in nightly reconciliation job |

---

## Consequences

**Positive**
- Order Service p99 latency reduced from 340 ms to 45 ms (no synchronous inventory call).
- Order Service availability decoupled from Inventory Service availability.
- Inventory Service peak load reduced by approximately 80%.

**Negative**
- Replication pipeline is a new operational component requiring monitoring.
- Developers must reason about snapshot staleness when writing order acceptance logic.
- End-to-end test suite must simulate lag to verify compensating transaction path.

---

## Alternatives Considered

**Synchronous REST call with circuit breaker:** Reduces blast radius but does not eliminate
the coupling. When the circuit is open, we must still decide whether to reject orders or
use stale data — the same trade-off, but without explicit SLA guarantees on staleness.

**Two-phase commit across services:** Provides strong consistency but introduces a
distributed coordinator that is a single point of failure and dramatically increases
p99 latency. Rejected: complexity is not justified for the inventory-read use case.
```
