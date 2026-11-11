# Solutions — Week 08 · Day 1

## Exercise 1

**Classify each endpoint by REST Maturity Level (Richardson Maturity Model)**

Given endpoints:
```
(a) POST /api?action=getBook&id=42
(b) GET  /books/42
(c) POST /books  (creates book, returns 200 with body)
(d) GET  /books/42  (returns 200 with _links.self, _links.reserve)
(e) POST /rpc/cancelReservation
```

**Answers:**

| Endpoint | Level | Reason |
|----------|-------|--------|
| (a) | **Level 0** — The Swamp of POX | Single URI as tunnel. Action encoded in query param. HTTP is just a transport protocol. |
| (b) | **Level 1** — Resources | Distinct URI per resource (`/books/42`). But if every operation uses GET regardless of intent, HTTP verbs are still ignored. |
| (c) | **Level 2** — HTTP Verbs | Uses POST for creation, GET for retrieval. However returning `200` instead of `201 Created` with a `Location` header violates the full Level 2 contract. |
| (d) | **Level 3** — Hypermedia Controls (HATEOAS) | Returns resource + links describing next valid actions. Client can navigate the API without out-of-band docs. |
| (e) | **Level 0 / Level 1** | RPC-style verb in the path (`cancelReservation`). Correct move: `DELETE /reservations/{id}` or `POST /reservations/{id}/cancellation`. |

**Why it matters:** Each level reduces client-server coupling. Level 0 requires clients to know every operation name. Level 3 clients only need to know entry-point URLs; everything else is discovered at runtime.

**How to upgrade (a) to Level 2:**
```
Before: POST /api?action=getBook&id=42
After:  GET  /books/42
```

**How to upgrade (c) to full Level 2:**
```python
# Return 201 with Location header
from fastapi import Response

@router.post("/books", status_code=201)
def create_book(payload: BookCreate, response: Response) -> BookOut:
    book = service.create(payload)
    response.headers["Location"] = f"/books/{book.id}"
    return book
```

---

## Exercise 2

**URL design for a Book Lending Library API**

```
# List all books (with optional filters)
GET  /books
GET  /books?genre=fiction&available=true

# Get a single book
GET  /books/{book_id}

# Create a reservation
POST /reservations

# Cancel a reservation
DELETE /reservations/{reservation_id}
# or, if soft-cancel semantics are needed:
POST /reservations/{reservation_id}/cancellation

# List a user's active reservations
GET /users/{user_id}/reservations?status=active
# or with JWT auth (preferred — user from token):
GET /me/reservations?status=active
```

**Design rationale:**

1. **Plural nouns** — `/books`, `/reservations`, `/users`. Never verbs (`/getBook`, `/createReservation`).
2. **Hierarchy reflects ownership** — `/users/{id}/reservations` expresses "reservations belonging to user". Use only one level of nesting; deeper nesting creates brittle URLs.
3. **Cancellation as sub-resource** — `DELETE /reservations/{id}` is cleanest when cancel = destroy. Use `POST /reservations/{id}/cancellation` when cancel is a distinct workflow event that must be logged (audit trail).
4. **Filters as query params** — `?status=active`, `?available=true`. Never encode filter state into the path.
5. **`/me` shortcut** — avoids requiring clients to know their own user ID when authenticated via JWT. Resolves server-side from the token.

**Full FastAPI skeleton:**

```python
from fastapi import APIRouter, Depends
from app.models import BookOut, ReservationCreate, ReservationOut
from app.auth import current_user

router = APIRouter()


@router.get("/books", response_model=list[BookOut])
def list_books(genre: str | None = None, available: bool | None = None):
    ...


@router.get("/books/{book_id}", response_model=BookOut)
def get_book(book_id: int):
    ...


@router.post("/reservations", response_model=ReservationOut, status_code=201)
def create_reservation(payload: ReservationCreate, user=Depends(current_user)):
    ...


@router.delete("/reservations/{reservation_id}", status_code=204)
def cancel_reservation(reservation_id: int, user=Depends(current_user)):
    ...


@router.get("/me/reservations", response_model=list[ReservationOut])
def list_my_reservations(status: str | None = None, user=Depends(current_user)):
    ...
```

---

## Exercise 3

**Idempotent POST endpoint using `X-Idempotency-Key`**

```python
import hashlib
import json
import logging
import os
import time
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field
import redis

logger = logging.getLogger(__name__)

router = APIRouter()

# Redis client — connection string from env, never hardcoded
_redis = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))

TTL_SECONDS = 86_400  # 24 hours


class OrderCreate(BaseModel):
    product_id: int
    quantity: int = Field(gt=0)
    shipping_address: str


class OrderOut(BaseModel):
    order_id: str
    product_id: int
    quantity: int
    status: str


def _idempotency_cache_key(idempotency_key: str) -> str:
    return f"idempotency:{idempotency_key}"


@router.post("/orders", response_model=OrderOut, status_code=201)
def create_order(
    payload: OrderCreate,
    x_idempotency_key: str = Header(..., alias="X-Idempotency-Key"),
) -> OrderOut:
    """
    Creates an order. Idempotent: same key within 24 h returns the cached response.
    """
    cache_key = _idempotency_cache_key(x_idempotency_key)

    # 1. Check cache
    cached = _redis.get(cache_key)
    if cached:
        logger.info("Idempotency hit for key=%s", x_idempotency_key)
        data = json.loads(cached)
        return OrderOut(**data)

    # 2. Create order (real business logic here)
    order_id = hashlib.sha256(
        f"{x_idempotency_key}{time.time()}".encode()
    ).hexdigest()[:12]

    order = OrderOut(
        order_id=order_id,
        product_id=payload.product_id,
        quantity=payload.quantity,
        status="confirmed",
    )

    # 3. Persist to cache before returning
    _redis.setex(cache_key, TTL_SECONDS, order.model_dump_json())
    logger.info("Order created order_id=%s idempotency_key=%s", order_id, x_idempotency_key)

    return order
```

**Why this works:**

- `X-Idempotency-Key` is a client-supplied UUID. The server never generates it; the client retries with the same key.
- Redis stores the serialised response with a 24-hour TTL. Any duplicate within that window gets the cached response — identical status code, identical body.
- The key is namespaced (`idempotency:`) to avoid collisions with other Redis keys.
- The order creation logic runs exactly once per key; subsequent calls never touch the database.
- Pattern follows Stripe's idempotency key spec, which is the industry standard.

---

## Exercise 4

**Pick the correct HTTP status code for each scenario**

| Scenario | Code | Reason |
|----------|------|--------|
| (a) Resource not found | **404 Not Found** | The server understood the request but the resource does not exist at that URI. Use `404` not `200` with `null` body — conflating success and absence couples clients to payload inspection. |
| (b) Validation error (bad input from client) | **422 Unprocessable Entity** | The syntax was valid (not `400`), but semantic validation failed (e.g., `quantity: -5`). FastAPI uses `422` by default for Pydantic validation failures. Alternatively `400 Bad Request` is acceptable when the error is purely structural. |
| (c) Rate limit exceeded | **429 Too Many Requests** | RFC 6585. Include `Retry-After` header (seconds or HTTP-date). Do not use `503` — that implies server overload, not client quota breach. |
| (d) Server cannot determine API version | **400 Bad Request** or **406 Not Acceptable** | `400` if the version header/param is malformed or missing and required. `406` if the requested `Accept` version is syntactically valid but no representation can be produced. Avoid `500` — the client caused this, not the server. |
| (e) Resource created successfully with a Location header | **201 Created** | Signals creation (not just success). Must include `Location: /resource/{id}` header pointing to the new resource. Do not return `200 OK` — it does not communicate that a new resource was created. |

**General rule:** 2xx = success, 4xx = client error (fix your request), 5xx = server error (retry may help). Never return `200` for errors — this forces clients to parse the body to detect failure, which breaks observability tooling.
