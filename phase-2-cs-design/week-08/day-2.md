# Phase 2 · Week 08 · Day 2 — Error Contracts, Pagination, and HATEOAS

> Objective: Design consistent error payloads, implement cursor-based and keyset pagination, and understand when HATEOAS adds genuine value.  ·  Time: ~1 hr  ·  Prereqs: [[week-08/day-1]]

---

## Concept

### Error Contracts

An error is a first-class part of your API's contract.  
An inconsistent error shape forces clients to write divergent error-handling code — one branch for auth errors, another for validation errors, a third for rate limits — because the fields differ between endpoints.

A contract means: **every error from every endpoint shares the same top-level shape**.

#### The RFC 7807 Problem Details format

RFC 7807 (updated by RFC 9457) defines a standard JSON error envelope:

```json
{
  "type": "https://api.example.com/errors/validation",
  "title": "Request validation failed",
  "status": 422,
  "detail": "The 'email' field is not a valid email address.",
  "instance": "/v1/users",
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "errors": [
    {"field": "email", "message": "not a valid email address", "code": "invalid_format"}
  ]
}
```

| Field | Required | Purpose |
|-------|----------|---------|
| `type` | yes | URI identifying the error class; dereferenceable for docs |
| `title` | yes | Human-readable short description (does not change per instance) |
| `status` | yes | Mirror of the HTTP status code |
| `detail` | no | Human-readable explanation specific to this occurrence |
| `instance` | no | URI of the specific request that caused the error |
| `trace_id` | extension | Correlation ID for distributed tracing |
| `errors` | extension | Per-field validation errors |

Content-Type for RFC 7807 responses: `application/problem+json`.

#### Error taxonomy

```
4xx Client Errors
├── 400 Bad Request        — malformed JSON, missing required field
├── 401 Unauthorized       — no token, expired token
├── 403 Forbidden          — valid token, insufficient scope
├── 404 Not Found          — resource does not exist
├── 409 Conflict           — duplicate, concurrent modification
├── 422 Unprocessable      — valid JSON, invalid business logic
└── 429 Too Many Requests  — rate limit; always include Retry-After

5xx Server Errors
├── 500 Internal Server Error — unexpected fault; log with trace_id
├── 502 Bad Gateway           — upstream returned garbage
├── 503 Service Unavailable   — circuit open, maintenance mode
└── 504 Gateway Timeout       — upstream too slow
```

Rules for production error handling:
1. Never expose stack traces or internal paths in 5xx responses to the client.
2. Log the full internal error with `trace_id`; return only `trace_id` to the client.
3. Validation errors (422) list every failing field, not just the first.
4. `detail` messages are for humans; `code` strings are for machines.

---

### Pagination

Never return unbounded collections.  
Even a table with 1 000 rows today may have 10 000 000 rows next year.

Three strategies — each with a specific use case.

#### Offset / Limit

```
GET /users?offset=40&limit=20
```

How it works: `SELECT * FROM users ORDER BY id LIMIT 20 OFFSET 40`

Response:

```json
{
  "data": [...],
  "pagination": {
    "offset": 40,
    "limit": 20,
    "total": 1842
  }
}
```

Pros: random access, easy to implement, UI can jump to any page.  
Cons:
- **Phantom rows**: if a row is deleted between page 1 and page 2, items shift and page 2 misses one.
- **Slow at high offsets**: `OFFSET 1000000` forces the DB to scan and discard 1 000 000 rows even though you want only 20.
- Does not scale to high-volume streams.

Use when: UI needs page numbers; dataset is small and stable.

#### Cursor Pagination

```
GET /users?cursor=eyJpZCI6NDB9&limit=20
```

`cursor` is a base64-encoded opaque token that encodes the position of the last seen item.

Response:

```json
{
  "data": [...],
  "pagination": {
    "next_cursor": "eyJpZCI6NjB9",
    "prev_cursor": "eyJpZCI6MjB9",
    "has_more": true
  }
}
```

How it works internally:

```sql
-- Decode cursor to get last_seen_id=40
SELECT * FROM users WHERE id > 40 ORDER BY id LIMIT 20
```

Pros: stable across inserts/deletes; O(log n) with an index on the cursor column.  
Cons: no random page access; cursor must be invalidated when sort order changes.

Use when: infinite scroll, live feeds, large datasets.

#### Keyset Pagination

Keyset is a generalisation of cursor pagination for multi-column sort keys.

```
GET /messages?after_created_at=2026-01-01T10:00:00Z&after_id=8921&limit=20
```

SQL:

```sql
SELECT * FROM messages
WHERE (created_at, id) > ('2026-01-01T10:00:00Z', 8921)
ORDER BY created_at, id
LIMIT 20
```

Pros: stable, fast with composite index on `(created_at, id)`, handles ties via secondary key.  
Cons: more complex to implement and explain; requires composite index.

Use when: sorted by a non-unique column (timestamp) where duplicates exist.

#### Pagination envelope comparison

| Feature | Offset/Limit | Cursor | Keyset |
|---------|-------------|--------|--------|
| Random access | yes | no | no |
| Stable under mutations | no | yes | yes |
| DB performance at scale | poor | good | good |
| Multi-column sort | easy | no | yes |
| UI "page 7 of 12" | easy | hard | hard |

---

### HATEOAS

**H**ypermedia **A**s **T**he **E**ngine **O**f **A**pplication **S**tate.

The core idea: responses embed links to valid next actions, so the client never hard-codes URLs.  
The client discovers the API by following links, the way a browser follows HTML `<a>` tags.

```json
{
  "id": 42,
  "name": "Alice",
  "email": "alice@example.com",
  "_links": {
    "self":    {"href": "/v1/users/42", "method": "GET"},
    "update":  {"href": "/v1/users/42", "method": "PUT"},
    "delete":  {"href": "/v1/users/42", "method": "DELETE"},
    "orders":  {"href": "/v1/users/42/orders", "method": "GET"}
  }
}
```

#### Why bother?

1. **Evolvability**: the server can change URLs; clients that follow links don't break.
2. **Discoverability**: new clients explore the API without reading docs.
3. **Workflow encoding**: a payment API can include `_links.capture` only when capture is valid.

#### When NOT to use HATEOAS

HATEOAS adds serialization cost and documentation complexity.  
Skip it when:
- Clients are internal and fully versioned with the server (microservice calling microservice).
- The API has a small, stable surface (< 10 endpoints).
- No public third-party developers.

Use it when:
- Building a public API consumed by many unknown clients.
- URL structure may change over time.
- Workflow state needs to be surfaced (e.g., "you can only confirm an order in PENDING state").

---

## How it works internally

### Why offset pagination breaks at scale

Relational databases implement `OFFSET N` by reading and discarding N rows.  
With an index, the engine still traverses N leaf pages before reaching the target.  
At `OFFSET 1 000 000` with 20-byte rows:
- Index traversal: ~1 000 000 comparisons.
- Estimated latency: 200–500 ms even on a fast NVMe disk.

Keyset avoids this because it uses `WHERE (col) > (value)` — the index seeks directly to the right position in O(log n).

### Cursor encoding security

A cursor encodes position, not identity.  
If the cursor encodes `{"user_id": 40}` and is signed (HMAC), the server can reject tampered cursors.  
If unsigned, a client can forge `{"user_id": 0}` to scan from the beginning — a minor information leak, not usually a security issue but worth noting.

For sensitive datasets (financial records), sign cursors with HMAC-SHA256 and a server secret.

```python
import hashlib, hmac, json, base64, os

CURSOR_SECRET = os.getenv("CURSOR_SECRET")

def encode_cursor(payload: dict) -> str:
    data = json.dumps(payload, separators=(",", ":")).encode()
    sig = hmac.new(CURSOR_SECRET.encode(), data, hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(json.dumps({"d": payload, "s": sig}).encode()).decode()

def decode_cursor(token: str) -> dict:
    raw = json.loads(base64.urlsafe_b64decode(token))
    expected = hmac.new(CURSOR_SECRET.encode(),
                        json.dumps(raw["d"], separators=(",", ":")).encode(),
                        hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, raw["s"]):
        raise ValueError("invalid cursor signature")
    return raw["d"]
```

### How HATEOAS links interact with HTTP caching

A HATEOAS response that embeds user-specific links (`_links.delete` only if you own the resource)
is user-specific and must not be cached at the CDN level.  
Set `Cache-Control: private, max-age=0` or `Vary: Authorization` to prevent shared caching.

---

## Worked example

```python
"""
week_08_day2_errors_pagination_hateoas.py

Demonstrates:
- RFC 7807-compliant error responses via custom exception handler
- Offset pagination with total count
- Cursor pagination with opaque tokens
- Minimal HATEOAS links on resource responses

Run:
    pip install fastapi uvicorn pydantic
    uvicorn week_08_day2_errors_pagination_hateoas:app --reload
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import uuid
from typing import Any, Generic, TypeVar

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, field_validator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Error models (RFC 7807 / RFC 9457)
# ---------------------------------------------------------------------------


class FieldError(BaseModel):
    field: str
    message: str
    code: str


class ProblemDetail(BaseModel):
    """RFC 7807 Problem Details for HTTP APIs."""

    type: str = "about:blank"
    title: str
    status: int
    detail: str | None = None
    instance: str | None = None
    trace_id: str | None = None
    errors: list[FieldError] | None = None


class APIError(Exception):
    """Raise this anywhere in the app to produce a consistent error response."""

    def __init__(
        self,
        status_code: int,
        title: str,
        detail: str | None = None,
        error_type: str = "about:blank",
        field_errors: list[FieldError] | None = None,
    ) -> None:
        self.status_code = status_code
        self.title = title
        self.detail = detail
        self.error_type = error_type
        self.field_errors = field_errors
        super().__init__(title)


app = FastAPI(title="Week 08 Day 2 — Errors, Pagination, HATEOAS")


# ---------------------------------------------------------------------------
# Global exception handlers — all errors become RFC 7807
# ---------------------------------------------------------------------------


@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    trace_id = str(uuid.uuid4())
    log.error(
        "api error title=%s status=%d trace_id=%s path=%s",
        exc.title,
        exc.status_code,
        trace_id,
        request.url.path,
    )
    problem = ProblemDetail(
        type=exc.error_type,
        title=exc.title,
        status=exc.status_code,
        detail=exc.detail,
        instance=str(request.url),
        trace_id=trace_id,
        errors=exc.field_errors,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=problem.model_dump(exclude_none=True),
        media_type="application/problem+json",
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Convert FastAPI's default HTTPException to RFC 7807 format."""
    trace_id = str(uuid.uuid4())
    log.error("http exception status=%d trace_id=%s", exc.status_code, trace_id)
    problem = ProblemDetail(
        title=exc.detail if isinstance(exc.detail, str) else "HTTP error",
        status=exc.status_code,
        instance=str(request.url),
        trace_id=trace_id,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=problem.model_dump(exclude_none=True),
        media_type="application/problem+json",
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all: never leak internals to the client."""
    trace_id = str(uuid.uuid4())
    log.exception("unhandled exception trace_id=%s", trace_id)
    problem = ProblemDetail(
        type="https://api.example.com/errors/internal",
        title="An unexpected error occurred",
        status=500,
        detail=f"Please contact support with trace_id: {trace_id}",
        instance=str(request.url),
        trace_id=trace_id,
    )
    return JSONResponse(
        status_code=500,
        content=problem.model_dump(exclude_none=True),
        media_type="application/problem+json",
    )


# ---------------------------------------------------------------------------
# Data store and models
# ---------------------------------------------------------------------------


class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    _links: dict[str, dict[str, str]] | None = None  # HATEOAS links

    model_config = {"populate_by_name": True}


# Seed 50 fake users for pagination demos
_users: list[dict[str, Any]] = [
    {"id": i, "name": f"User {i}", "email": f"user{i}@example.com"}
    for i in range(1, 51)
]


def _user_links(user_id: int, base_url: str = "") -> dict[str, dict[str, str]]:
    """Generate HATEOAS _links for a user resource."""
    return {
        "self": {"href": f"{base_url}/v1/users/{user_id}", "method": "GET"},
        "update": {"href": f"{base_url}/v1/users/{user_id}", "method": "PUT"},
        "delete": {"href": f"{base_url}/v1/users/{user_id}", "method": "DELETE"},
        "orders": {"href": f"{base_url}/v1/users/{user_id}/orders", "method": "GET"},
    }


# ---------------------------------------------------------------------------
# Pagination models
# ---------------------------------------------------------------------------

T = TypeVar("T")


class OffsetPage(BaseModel, Generic[T]):
    data: list[T]
    pagination: dict[str, int]


class CursorPage(BaseModel, Generic[T]):
    data: list[T]
    pagination: dict[str, Any]


# ---------------------------------------------------------------------------
# Offset pagination endpoint
# ---------------------------------------------------------------------------


@app.get("/v1/users/offset", response_model=OffsetPage[UserResponse])
async def list_users_offset(
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(10, ge=1, le=100, description="Max items to return"),
) -> OffsetPage[UserResponse]:
    """
    Offset-based pagination. Useful for page-number UIs.

    Warning: unstable under concurrent inserts/deletes.
    Use cursor pagination for live-updating feeds.
    """
    total = len(_users)
    page_data = _users[offset: offset + limit]
    users = [UserResponse(**u) for u in page_data]
    log.info("offset pagination offset=%d limit=%d total=%d", offset, limit, total)
    return OffsetPage(
        data=users,
        pagination={
            "offset": offset,
            "limit": limit,
            "total": total,
            "pages": math.ceil(total / limit),
        },
    )


# ---------------------------------------------------------------------------
# Cursor pagination
# ---------------------------------------------------------------------------


def _encode_cursor(payload: dict[str, Any]) -> str:
    """Opaque cursor: base64-encode JSON payload.

    In production, HMAC-sign this to prevent forgery.
    """
    return base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode()
    ).decode()


def _decode_cursor(token: str) -> dict[str, Any]:
    try:
        return json.loads(base64.urlsafe_b64decode(token.encode()))
    except Exception as exc:
        raise APIError(
            status_code=status.HTTP_400_BAD_REQUEST,
            title="Invalid cursor",
            detail="The cursor token could not be decoded. Use the cursor from a previous response.",
            error_type="https://api.example.com/errors/invalid-cursor",
        ) from exc


@app.get("/v1/users/cursor", response_model=CursorPage[UserResponse])
async def list_users_cursor(
    cursor: str | None = Query(None, description="Opaque cursor from previous response"),
    limit: int = Query(10, ge=1, le=100),
) -> CursorPage[UserResponse]:
    """
    Cursor-based pagination. Stable across inserts and deletes.

    The cursor is opaque — clients MUST NOT construct or modify it.
    Pass next_cursor from the previous response to get the next page.
    """
    # Decode cursor to get last_seen_id
    if cursor is not None:
        cursor_data = _decode_cursor(cursor)
        last_id: int = cursor_data["last_id"]
    else:
        last_id = 0  # Start from beginning

    # Efficient query equivalent: WHERE id > last_id ORDER BY id LIMIT limit+1
    filtered = [u for u in _users if u["id"] > last_id]
    # Fetch one extra to determine has_more
    page_data = filtered[: limit + 1]
    has_more = len(page_data) > limit
    page_data = page_data[:limit]

    # Build next/prev cursors
    next_cursor: str | None = None
    if has_more and page_data:
        next_cursor = _encode_cursor({"last_id": page_data[-1]["id"]})

    prev_cursor: str | None = None
    if last_id > 0 and page_data:
        # Find the item just before our first item
        first_id = page_data[0]["id"]
        prev_items = [u for u in _users if u["id"] < first_id]
        if prev_items:
            prev_start_id = max(0, prev_items[-limit]["id"] - 1) if len(prev_items) >= limit else 0
            prev_cursor = _encode_cursor({"last_id": prev_start_id})

    users = [UserResponse(**u) for u in page_data]
    log.info("cursor pagination last_id=%d limit=%d has_more=%s", last_id, limit, has_more)

    return CursorPage(
        data=users,
        pagination={
            "next_cursor": next_cursor,
            "prev_cursor": prev_cursor,
            "has_more": has_more,
            "limit": limit,
        },
    )


# ---------------------------------------------------------------------------
# HATEOAS — single resource with embedded links
# ---------------------------------------------------------------------------


class UserWithLinks(BaseModel):
    id: int
    name: str
    email: str
    links: dict[str, dict[str, str]]

    model_config = {"populate_by_name": True}


@app.get("/v1/users/{user_id}", response_model=UserWithLinks)
async def get_user_hateoas(user_id: int, request: Request) -> UserWithLinks:
    """
    Single resource with HATEOAS _links.

    The client can follow 'orders' without knowing the URL structure.
    A future rename of /v1/users/{id}/orders breaks nothing — clients follow links.
    """
    users_by_id = {u["id"]: u for u in _users}
    user = users_by_id.get(user_id)
    if not user:
        raise APIError(
            status_code=status.HTTP_404_NOT_FOUND,
            title="User not found",
            detail=f"No user with id {user_id} exists.",
            error_type="https://api.example.com/errors/not-found",
        )

    base = str(request.base_url).rstrip("/")
    links = _user_links(user_id, base)

    log.info("returning user id=%d with HATEOAS links", user_id)
    return UserWithLinks(**user, links=links)


# ---------------------------------------------------------------------------
# Validation error example — shows per-field errors
# ---------------------------------------------------------------------------


class OrderCreate(BaseModel):
    user_id: int
    items: list[int]
    currency: str

    @field_validator("items")
    @classmethod
    def items_not_empty(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("order must contain at least one item")
        return v

    @field_validator("currency")
    @classmethod
    def valid_currency(cls, v: str) -> str:
        allowed = {"USD", "EUR", "GBP"}
        if v.upper() not in allowed:
            raise ValueError(f"currency must be one of {sorted(allowed)}")
        return v.upper()


@app.post("/v1/orders", status_code=status.HTTP_201_CREATED)
async def create_order(payload: OrderCreate) -> dict[str, Any]:
    """
    POST with business-rule validation.

    Demonstrates how to raise structured per-field errors when Pydantic
    validation passes but business logic fails.
    """
    users_by_id = {u["id"]: u for u in _users}
    if payload.user_id not in users_by_id:
        raise APIError(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            title="Business rule violation",
            detail=f"User {payload.user_id} does not exist.",
            error_type="https://api.example.com/errors/validation",
            field_errors=[
                FieldError(field="user_id", message="user does not exist", code="reference_not_found")
            ],
        )

    order_id = str(uuid.uuid4())
    log.info("created order id=%s user_id=%d", order_id, payload.user_id)
    return {"order_id": order_id, "status": "pending"}
```

---

## Exercises

1. Extend the cursor pagination endpoint to support a `sort` query parameter (`name_asc` / `name_desc`).
   The cursor must encode both the sort key value and the `id` to handle ties correctly.
   Write a test that verifies cursor stability when a user is inserted between two pages.

2. Add HMAC signing to the cursor using a key loaded from `os.getenv("CURSOR_SECRET")`.
   The endpoint must return 400 with a proper RFC 7807 error when a tampered cursor is submitted.
   Test both the valid and tampered cases.

3. Implement a global Pydantic `RequestValidationError` handler that converts FastAPI's
   default 422 response into a RFC 7807 `ProblemDetail` with per-field `errors`.
   Verify that submitting an invalid `OrderCreate` body returns `application/problem+json`.

4. Add a HATEOAS `collection` link on `GET /v1/users/{id}` that points back to the
   cursor-paginated list endpoint, with the correct cursor for the page that contains
   this user. Explain what constraint this places on the cursor design.

(Solutions: ../solutions/week-08/day-2.md)

---

## Self-check

- [ ] I can write a Pydantic model that matches the RFC 7807 Problem Details shape including optional extension fields.
- [ ] I can explain exactly why offset pagination is unreliable under concurrent writes at the DB level.
- [ ] I can implement cursor pagination from scratch: encode cursor, decode cursor, SQL equivalent, has_more detection.
- [ ] I can identify when HATEOAS adds real value (public API with unstable URLs) vs when it is overhead (internal microservices).
- [ ] I can register a FastAPI exception handler that catches all unhandled exceptions and returns `application/problem+json` without leaking stack traces.

---

## Further Reading

- [RFC 9457 — Problem Details for HTTP APIs (supersedes RFC 7807)](https://datatracker.ietf.org/doc/html/rfc9457)
- [Stripe API — Cursor Pagination reference implementation](https://stripe.com/docs/api/pagination)
- [Roy Fielding's REST dissertation — Chapter 5 (HATEOAS origins)](https://ics.uci.edu/~fielding/pubs/dissertation/rest_arch_style.htm)
