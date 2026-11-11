# Phase 2 · Week 08 · Day 7 — Week Review + Integration

> Objective: Extend your 2-service system with RFC 7807 error contracts, cursor-based pagination, and documented CAP trade-off decisions.  ·  Time: ~3 hr project + 1 hr assessment  ·  Prereqs: [[day-6]]

---

## Concept

### What we're integrating

By end of week you built two microservices, wired them over HTTP, designed schemas, and chose a consistency model. Today you harden the system in three specific ways:

1. **RFC 7807 Problem Details** — machine-readable error responses instead of ad-hoc JSON blobs.
2. **Cursor-based pagination** — stable, efficient page traversal for large collections.
3. **Architecture Decision Records (ADR)** — short, structured documents capturing each CAP/consistency trade-off you made so future you (and reviewers) understand *why*, not just *what*.

These three capabilities mark the boundary between a toy API and a production-grade one. They are also the first three things a senior reviewer will check.

---

### RFC 7807 Problem Details

RFC 7807 defines a standard JSON (and `application/problem+json`) structure for HTTP error responses:

```
{
  "type":     "https://example.com/errors/validation-error",
  "title":    "Validation Error",
  "status":   422,
  "detail":   "The field 'email' must be a valid address.",
  "instance": "/requests/abc-123"
}
```

Key fields:

| Field | Required | Purpose |
|---|---|---|
| `type` | yes | URI identifying the error class; should be dereferenceable |
| `title` | yes | Human-readable summary; stable across occurrences |
| `status` | yes | HTTP status code (mirrors the HTTP header) |
| `detail` | no | Human-readable explanation of this specific occurrence |
| `instance` | no | URI identifying this specific occurrence |
| extensions | no | Any extra fields you need (e.g. `errors` array for validation) |

Why this matters: clients can switch on `type` instead of parsing free-text `message` fields. Monitoring systems can group by `type`. Documentation tooling can link to your error catalogue.

---

### Cursor-Based Pagination

**Offset pagination** (`?page=3&size=20`) is simple but breaks under concurrent writes:

- A row inserted before your current offset shifts everything — you skip or re-read rows.
- `OFFSET N` forces the DB to scan and discard N rows, so page 500 is slow.

**Cursor pagination** uses an opaque token encoding the position of the last seen row:

```
GET /orders?limit=20
→ { "items": [...], "next_cursor": "eyJpZCI6IDQyfQ==" }

GET /orders?limit=20&cursor=eyJpZCI6IDQyfQ==
→ { "items": [...], "next_cursor": "eyJpZCI6IDYyfQ==" }
```

The cursor encodes the sort key(s) of the last returned item (typically a timestamp + ID). The query becomes:

```sql
SELECT * FROM orders
WHERE (created_at, id) > (:cursor_ts, :cursor_id)
ORDER BY created_at, id
LIMIT :limit
```

This is a **keyset seek** — the DB uses the index directly, cost is O(1) regardless of page depth.

Rules for stable cursors:

- Sort column must be unique or combined with a tie-breaker (usually primary key).
- Cursors are opaque to clients — base64-encode the JSON payload.
- Always validate and reject tampered cursors (wrap in HMAC or just catch parse errors).
- Cursors are not stable across schema changes to sort columns.

---

### Architecture Decision Records (ADR)

An ADR answers: "Why did we make this choice, given these constraints?"

Standard Nygard format (abbreviated):

```
# ADR-001: Use eventual consistency for order–inventory sync

## Status
Accepted

## Context
Order service and Inventory service are separate deployable units.
Synchronous calls introduce coupling; the inventory DB is on a different host.

## Decision
Publish OrderCreated events to a message broker; Inventory service consumes
asynchronously. Accept that inventory reads may lag by up to ~500 ms.

## Consequences
- Positive: order write path is decoupled from inventory latency.
- Negative: double-booking possible in < 500 ms windows; mitigated by
  reservation pattern in Inventory service.
- CAP trade-off: AP (Availability + Partition tolerance) over CP for reads.
  CP only for the reservation step (distributed lock, 200 ms TTL).
```

One ADR per significant decision. Store them in `docs/adr/` numbered sequentially.

---

## How it works internally

### FastAPI error handler registration

FastAPI lets you register exception handlers globally. Return `JSONResponse` with `media_type="application/problem+json"`:

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        media_type="application/problem+json",
        content={
            "type": "https://api.example.com/errors/validation-error",
            "title": "Validation Error",
            "status": 422,
            "detail": str(exc),
            "instance": str(request.url),
        },
    )
```

### Cursor encoding/decoding

A cursor is a base64url-encoded JSON object. Use `urllib.parse` for URL-safe encoding:

```python
import base64
import json
from datetime import datetime

def encode_cursor(created_at: datetime, row_id: int) -> str:
    payload = {"ts": created_at.isoformat(), "id": row_id}
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()

def decode_cursor(cursor: str) -> tuple[datetime, int]:
    raw = base64.urlsafe_b64decode(cursor.encode()).decode()
    data = json.loads(raw)
    return datetime.fromisoformat(data["ts"]), data["id"]
```

The DB layer translates this into a keyset predicate — never `OFFSET`.

---

## Worked example

Below is a self-contained FastAPI service that demonstrates all three concepts. It uses an in-memory list as the "database" so no external dependency is needed to run it.

```python
"""
day7_integration.py
Demonstrates: RFC 7807 errors, cursor pagination, and ADR logging.

Run:
    pip install fastapi uvicorn pydantic
    uvicorn day7_integration:app --reload
"""
from __future__ import annotations

import base64
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, field_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

app = FastAPI(title="Week-08 Integration Demo")

# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------

class OrderCreate(BaseModel):
    customer_email: EmailStr
    amount_cents: int

    @field_validator("amount_cents")
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("amount_cents must be positive")
        return v


class Order(BaseModel):
    order_id: str
    customer_email: EmailStr
    amount_cents: int
    created_at: datetime


class ProblemDetail(BaseModel):
    type: str
    title: str
    status: int
    detail: str
    instance: str


class CursorPage(BaseModel):
    items: list[Order]
    next_cursor: str | None


# ---------------------------------------------------------------------------
# In-memory store (replace with real DB in production)
# ---------------------------------------------------------------------------

_ORDERS: list[Order] = []


# ---------------------------------------------------------------------------
# RFC 7807 error helpers
# ---------------------------------------------------------------------------

PROBLEM_BASE = os.getenv("PROBLEM_BASE_URI", "https://api.example.com/errors")


def problem_response(
    request: Request,
    status: int,
    type_slug: str,
    title: str,
    detail: str,
) -> JSONResponse:
    body = {
        "type": f"{PROBLEM_BASE}/{type_slug}",
        "title": title,
        "status": status,
        "detail": detail,
        "instance": str(request.url),
    }
    return JSONResponse(
        status_code=status,
        media_type="application/problem+json",
        content=body,
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return problem_response(
        request,
        status=exc.status_code,
        type_slug="http-error",
        title=f"HTTP {exc.status_code}",
        detail=exc.detail,
    )


# ---------------------------------------------------------------------------
# Cursor helpers
# ---------------------------------------------------------------------------

def encode_cursor(created_at: datetime, order_id: str) -> str:
    payload = {"ts": created_at.isoformat(), "id": order_id}
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()


def decode_cursor(raw: str) -> tuple[datetime, str]:
    try:
        data = json.loads(base64.urlsafe_b64decode(raw.encode()).decode())
        return datetime.fromisoformat(data["ts"]), data["id"]
    except Exception as exc:
        raise ValueError("Invalid cursor") from exc


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/orders", response_model=Order, status_code=201)
async def create_order(payload: OrderCreate, request: Request) -> Order:
    order = Order(
        order_id=str(uuid.uuid4()),
        customer_email=payload.customer_email,
        amount_cents=payload.amount_cents,
        created_at=datetime.now(tz=timezone.utc),
    )
    _ORDERS.append(order)
    log.info("Created order %s for %s", order.order_id, order.customer_email)
    return order


@app.get("/orders", response_model=CursorPage)
async def list_orders(
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: Annotated[str | None, Query()] = None,
) -> CursorPage:
    # Sort stable: (created_at, order_id)
    sorted_orders = sorted(_ORDERS, key=lambda o: (o.created_at, o.order_id))

    if cursor:
        try:
            cursor_ts, cursor_id = decode_cursor(cursor)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        # Keyset: keep rows strictly after the cursor position
        sorted_orders = [
            o for o in sorted_orders
            if (o.created_at, o.order_id) > (cursor_ts, cursor_id)
        ]

    page = sorted_orders[:limit]
    next_cursor: str | None = None
    if len(sorted_orders) > limit:
        last = page[-1]
        next_cursor = encode_cursor(last.created_at, last.order_id)

    return CursorPage(items=page, next_cursor=next_cursor)


@app.get("/orders/{order_id}", response_model=Order)
async def get_order(order_id: str, request: Request) -> Order:
    for o in _ORDERS:
        if o.order_id == order_id:
            return o
    return problem_response(  # type: ignore[return-value]
        request,
        status=404,
        type_slug="order-not-found",
        title="Order Not Found",
        detail=f"No order with id '{order_id}'.",
    )
```

Test it manually:

```bash
# Create two orders
curl -s -X POST http://localhost:8000/orders \
  -H "Content-Type: application/json" \
  -d '{"customer_email":"alice@example.com","amount_cents":4999}' | python -m json.tool

# List first page
curl -s "http://localhost:8000/orders?limit=1" | python -m json.tool

# List with cursor (paste next_cursor value from above)
curl -s "http://localhost:8000/orders?limit=1&cursor=<PASTE_HERE>" | python -m json.tool

# Trigger a 404 with Problem Details
curl -s "http://localhost:8000/orders/does-not-exist" | python -m json.tool
```

---

## Exercises

1. **RFC 7807 extension field.** The spec allows custom extension fields on the problem object. Add an `errors` array to the validation error response that lists each failing field by name and its constraint. For example, a bad `amount_cents` should produce `"errors": [{"field": "amount_cents", "message": "must be positive"}]`.

2. **Cursor tamper detection.** Modify `decode_cursor` to sign the payload with an HMAC-SHA256 key sourced from `os.getenv("CURSOR_HMAC_KEY")`. `encode_cursor` should append the signature. `decode_cursor` should reject cursors whose signature does not match.

3. **Keyset SQL.** Translate the in-memory keyset filter from the worked example into a SQLAlchemy Core `SELECT` statement (not ORM). Use a composite `WHERE (created_at, id) > (:cursor_ts, :cursor_id)` and ensure you use a bound parameter for both columns.

4. **Write an ADR.** Your 2-service system makes at least one choice between consistency and availability. Identify it and write a full ADR in `docs/adr/ADR-001.md` using the Nygard format shown above. Your ADR must name the specific CAP position (AP, CP, or CA) and the mitigation for the downside.

5. **Idempotency key.** Add an `Idempotency-Key` header to `POST /orders`. If the same key is submitted twice within 60 seconds return the original response (HTTP 200 with the existing order) instead of creating a duplicate. Store the mapping in a dict keyed by idempotency token. Write a pytest test that POSTs twice with the same key and asserts the same `order_id` is returned.

(Solutions: ../solutions/week-08/day-7.md)

---

## Self-check

- [ ] I can describe all five required fields of an RFC 7807 Problem Details response.
- [ ] I can explain why cursor pagination outperforms offset pagination at large page depths.
- [ ] I can implement a keyset query that does not use `OFFSET`.
- [ ] I can encode and decode a cursor without introducing a timing oracle.
- [ ] I have written at least one ADR that names a specific CAP trade-off and its mitigation.
- [ ] My 2-service system returns `application/problem+json` for all 4xx and 5xx responses.

---

## Supplementary links

- [RFC 7807 — Problem Details for HTTP APIs](https://www.rfc-editor.org/rfc/rfc7807)
- [Use The Index, Luke — Paging through results](https://use-the-index-luke.com/no-offset)
- [Michael Nygard — Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
- [FastAPI — Exception Handlers](https://fastapi.tiangolo.com/tutorial/handling-errors/)
- [Pydantic v2 — Validators](https://docs.pydantic.dev/latest/concepts/validators/)
- *Designing Data-Intensive Applications* — Kleppmann, Chapter 5 (replication + consistency)
