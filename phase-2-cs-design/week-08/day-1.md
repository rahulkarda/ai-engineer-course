# Phase 2 · Week 08 · Day 1 — API Design Principles

> Objective: Design production-quality REST APIs using the Richardson Maturity Model, correct HTTP semantics, idempotency guarantees, and a coherent versioning strategy.  ·  Time: ~1 hr  ·  Prereqs: [[phase-1-python-backend/week-04/day-1]]

---

## Concept

An API is a contract between two programs.  
Good API design minimises surprises, enables safe retries, and lets both sides evolve independently.  
This lesson builds intuition from first principles: HTTP verbs as verbs, URLs as nouns, status codes as machine-readable intent.

### Richardson Maturity Model

The RMM scores a REST API on four levels.  
Higher levels exploit more of HTTP's built-in capabilities.

| Level | Name | What it does |
|-------|------|--------------|
| 0 | The Swamp of POX | Single endpoint, verbs in the payload (`{"action": "getUser"}`) |
| 1 | Resources | Separate URLs per resource (`/users`, `/orders`) |
| 2 | HTTP Verbs | GET/POST/PUT/PATCH/DELETE carry semantic meaning |
| 3 | Hypermedia (HATEOAS) | Responses embed links to valid next actions |

Most production APIs target **Level 2**.  
Level 3 (HATEOAS) is theoretically ideal but rarely implemented in full — covered on Day 2.

#### Level 0 — what it looks like (and why it hurts)

```
POST /api
{"action": "create_user", "name": "alice"}

POST /api
{"action": "delete_user", "id": 42}
```

Problems: caching is impossible (everything is POST), errors are buried in 200 responses,
no HTTP-level tooling (load balancers, CDNs) can help.

#### Level 1 — resources appear

```
POST /users          # create
POST /users/42       # update (still wrong verb)
POST /users/42/del   # delete (verb in URL — wrong)
```

Better: each noun has a home. Still wrong: verbs in URLs.

#### Level 2 — HTTP verbs do the heavy lifting

```
GET    /users/42      # read — safe, idempotent
POST   /users         # create — neither safe nor idempotent
PUT    /users/42      # full replace — idempotent
PATCH  /users/42      # partial update — idempotent when correct
DELETE /users/42      # delete — idempotent
```

This is the target. Caching, proxies, CDNs, and API gateways all speak this language natively.

---

### URL Design

**Nouns, not verbs. Hierarchies mirror ownership.**

Rules:
1. Use plural nouns for collections: `/users`, `/orders`, `/products`
2. Nested paths express ownership: `/users/{user_id}/orders`
3. Nesting beyond 2 levels is a smell — flatten or use query params
4. Kebab-case for multi-word segments: `/payment-methods`, not `/paymentMethods`
5. Query params for filtering, sorting, pagination: `/users?status=active&sort=created_at`
6. Actions that don't map to CRUD use sub-resources or POST to a verb noun: `/orders/{id}/cancel` (POST)

Bad examples and fixes:

| Bad | Good | Reason |
|-----|------|--------|
| `GET /getUser?id=1` | `GET /users/1` | verb in path |
| `POST /users/delete/1` | `DELETE /users/1` | verb in path |
| `GET /users/1/orders/5/items/3/sku` | `GET /order-items/3` | too deep |
| `GET /Users/1` | `GET /users/1` | case inconsistent |

---

### HTTP Verbs — Exact Semantics

| Verb | Safe | Idempotent | Body | Notes |
|------|------|-----------|------|-------|
| GET | yes | yes | no | Never mutate on GET |
| HEAD | yes | yes | no | Same as GET, no body |
| OPTIONS | yes | yes | no | CORS preflight |
| POST | no | no | yes | Creates; outcome differs per call |
| PUT | no | yes | yes | Full replacement |
| PATCH | no | yes* | yes | Partial update; idempotent only if deterministic |
| DELETE | no | yes | optional | Second DELETE returns 404, not error |

**Safe** = no side effects from the server's perspective.  
**Idempotent** = calling N times = calling once (result is the same).

PATCH idempotency nuance: `{"increment_views": 1}` is NOT idempotent. `{"views": 42}` IS.
Always prefer absolute-value PATCH payloads.

---

### Versioning Strategies

APIs must evolve without breaking existing clients.  
Three mainstream strategies, each with genuine trade-offs.

#### Strategy 1 — URL Path Versioning

```
GET /v1/users/42
GET /v2/users/42
```

Pros: visible in logs, easy to route at the gateway level, dead simple.  
Cons: version in URL is technically not RESTful (the resource didn't change, the representation did).  
**Recommended for most teams** — operationally simple.

#### Strategy 2 — Accept Header (Content Negotiation)

```
GET /users/42
Accept: application/vnd.myapi.v2+json
```

Pros: clean URLs, RESTfully correct.  
Cons: invisible in browser address bar, harder to test, proxy caching tricky.  
**Use when** clients are all programmatic and your team has strong HTTP discipline.

#### Strategy 3 — Query Parameter

```
GET /users/42?api_version=2
```

Pros: easy to demo and test in a browser.  
Cons: pollutes query string, can conflict with other params, often cached by accident.  
**Avoid** for core versioning — acceptable as a deprecation escape hatch.

#### Practical rules for versioning

- Version at the API level, not the endpoint level.
- Default to the latest version; never silently break old clients.
- Announce deprecations 6+ months before removal.
- Support N and N-1 simultaneously during the transition window.
- Use `Sunset` and `Deprecation` headers (RFC 8594) to signal removal dates.

```
HTTP/1.1 200 OK
Deprecation: Tue, 01 Jul 2027 00:00:00 GMT
Sunset: Mon, 01 Jan 2028 00:00:00 GMT
Link: <https://api.example.com/v2/users/42>; rel="successor-version"
```

---

### Idempotency

#### What and why

**Idempotency**: applying an operation multiple times produces the same result as applying it once.

Why it matters: networks fail. Clients retry. Without idempotency guarantees:
- A payment POST retried after a timeout creates duplicate charges.
- A user creation POST retried creates duplicate accounts.

#### PUT vs PATCH vs POST idempotency

```
PUT /users/42
{"name": "alice", "email": "alice@example.com"}
# Safe to retry: always results in the same state.

PATCH /users/42
{"email": "alice@example.com"}
# Safe to retry IF the payload specifies absolute values.

PATCH /users/42
{"credits": "+10"}   # BAD — relative, not idempotent
{"credits": 110}     # GOOD — absolute value
```

#### Idempotency Keys

For non-idempotent operations (POST) that must be retry-safe, use **idempotency keys**:  
the client generates a unique key per logical operation and sends it as a header.  
The server stores the result keyed on it; on retry it returns the stored result.

```
POST /payments
Idempotency-Key: a3f9e2c1-7b4d-4e8a-9f0b-1c2d3e4f5a6b
Content-Type: application/json

{"amount": 9900, "currency": "EUR", "recipient_id": 7}
```

Server logic (pseudocode):
1. Check cache for `Idempotency-Key`.
2. If found and completed: return stored response (same status + body).
3. If found and in-flight: return `409 Conflict` or wait.
4. If not found: process, store result, return.
5. Expire keys after 24–48 hours.

Key properties:
- UUIDs (v4) from the client side prevent collisions.
- Keys are scoped per endpoint and per authenticated user.
- Stored responses must include original status code and body.

---

### Status Codes

Status codes are machine-readable intent. Choose the most specific applicable code.

#### 2xx — Success

| Code | Name | When |
|------|------|------|
| 200 | OK | Read, update success with body |
| 201 | Created | POST that created a resource; include `Location` header |
| 202 | Accepted | Async task enqueued; body contains task ID |
| 204 | No Content | Delete or update with no response body |

#### 3xx — Redirection

| Code | Name | When |
|------|------|------|
| 301 | Moved Permanently | Resource URL changed forever; update bookmarks |
| 302 | Found | Temporary redirect |
| 304 | Not Modified | ETag/Last-Modified conditional GET — use cached copy |

#### 4xx — Client Error

| Code | Name | When |
|------|------|------|
| 400 | Bad Request | Malformed input, validation failure |
| 401 | Unauthorized | Missing or invalid authentication |
| 403 | Forbidden | Authenticated but lacks permission |
| 404 | Not Found | Resource does not exist |
| 405 | Method Not Allowed | Verb not supported on this endpoint |
| 409 | Conflict | State conflict (duplicate key, concurrent edit) |
| 410 | Gone | Resource existed and was permanently deleted |
| 422 | Unprocessable Entity | Semantically invalid (field rules failed) |
| 429 | Too Many Requests | Rate limit hit; include `Retry-After` |

#### 5xx — Server Error

| Code | Name | When |
|------|------|------|
| 500 | Internal Server Error | Unexpected server fault |
| 502 | Bad Gateway | Upstream service returned invalid response |
| 503 | Service Unavailable | Overloaded or in maintenance; include `Retry-After` |
| 504 | Gateway Timeout | Upstream service timed out |

Rule of thumb: **never return 200 with an error in the body**. That breaks every monitoring tool.

---

## How it works internally

### Why idempotency keys need server-side storage

HTTP is stateless. The client sends a key; the server must remember the result.  
Common implementations use Redis with TTL:

```
key = f"idempotency:{user_id}:{idempotency_key}"
result = await redis.get(key)
if result:
    return deserialize(result)
# ... process ...
await redis.setex(key, ttl_seconds=86400, value=serialize(response))
```

Why Redis, not the application DB?
- Lookups must be sub-millisecond (every request checks).
- TTL-based expiry is built-in.
- Shared across multiple API server instances.

### How HTTP caching uses verbs and status codes

GET + 200 + `Cache-Control` header = cacheable by CDN, browser, and reverse proxy.  
POST + 200 = never cached (browser treats POST as non-cacheable by default).  
DELETE + 204 = forces cache invalidation if the CDN is configured for it.

This is why verbs matter architecturally, not just semantically.  
A single mistaken POST where a GET belongs can destroy cache hit rates.

### URL routing internals in FastAPI

FastAPI builds a routing tree from path decorators.  
Routes are matched in registration order; more specific paths must come before wildcards.

```python
# Correct order — specific before wildcard
@app.get("/users/me")      # matches /users/me exactly
@app.get("/users/{id}")    # matches /users/anything-else
```

If registered in reverse, `/users/me` matches the `{id}` route with `id="me"`.  
FastAPI does NOT auto-sort by specificity — registration order is final.

### Version routing with a prefix router

```python
# Shared logic, two mounted prefixes
v1_router = APIRouter(prefix="/v1")
v2_router = APIRouter(prefix="/v2")

app.include_router(v1_router)
app.include_router(v2_router)
```

Each router is an independent routing tree.  
Middleware (auth, logging) attached to `app` runs for all versions.

---

## Worked example

```python
"""
week_08_day1_api_design.py

Demonstrates:
- REST Level 2 endpoints with correct verbs
- URL path versioning via APIRouter prefixes
- Idempotency key middleware for POST /payments
- Correct status codes throughout

Run:
    pip install fastapi uvicorn pydantic redis
    uvicorn week_08_day1_api_design:app --reload
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Annotated

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request, Response, status
from pydantic import BaseModel, EmailStr, field_validator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — all values from environment, never hardcoded
# ---------------------------------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
IDEMPOTENCY_TTL_SECONDS = int(os.getenv("IDEMPOTENCY_TTL", "86400"))

# ---------------------------------------------------------------------------
# Pydantic models (v2)
# ---------------------------------------------------------------------------


class UserCreate(BaseModel):
    name: str
    email: EmailStr

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("name must not be blank")
        return v.strip()


class UserUpdate(BaseModel):
    """PATCH payload — all fields optional, values are absolute (not increments)."""

    name: str | None = None
    email: EmailStr | None = None


class UserResponse(BaseModel):
    id: int
    name: str
    email: str


class PaymentCreate(BaseModel):
    amount_cents: int
    currency: str
    recipient_id: int

    @field_validator("amount_cents")
    @classmethod
    def positive_amount(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("amount_cents must be positive")
        return v


class PaymentResponse(BaseModel):
    payment_id: str
    status: str
    amount_cents: int
    currency: str


# ---------------------------------------------------------------------------
# Fake in-memory store (replace with DB in production)
# ---------------------------------------------------------------------------
_users: dict[int, dict] = {
    1: {"id": 1, "name": "Alice", "email": "alice@example.com"},
    2: {"id": 2, "name": "Bob", "email": "bob@example.com"},
}
_next_user_id = 3


# ---------------------------------------------------------------------------
# Redis dependency
# ---------------------------------------------------------------------------
async def get_redis() -> aioredis.Redis:
    client = aioredis.from_url(REDIS_URL, decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# v1 router — users (full CRUD, correct verbs)
# ---------------------------------------------------------------------------
v1 = APIRouter(prefix="/v1", tags=["v1"])


@v1.get("/users", response_model=list[UserResponse], status_code=status.HTTP_200_OK)
async def list_users() -> list[UserResponse]:
    """GET /v1/users — safe and idempotent; CDN-cacheable."""
    log.info("listing all users")
    return [UserResponse(**u) for u in _users.values()]


@v1.get("/users/{user_id}", response_model=UserResponse, status_code=status.HTTP_200_OK)
async def get_user(user_id: int) -> UserResponse:
    """GET /v1/users/{id} — safe, idempotent, cacheable."""
    user = _users.get(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="user not found")
    return UserResponse(**user)


@v1.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(payload: UserCreate, response: Response) -> UserResponse:
    """POST /v1/users — creates resource, returns 201 + Location header."""
    global _next_user_id
    new_id = _next_user_id
    _next_user_id += 1
    _users[new_id] = {"id": new_id, "name": payload.name, "email": payload.email}
    # Location header tells the client where to find the new resource
    response.headers["Location"] = f"/v1/users/{new_id}"
    log.info("created user id=%d", new_id)
    return UserResponse(**_users[new_id])


@v1.put("/users/{user_id}", response_model=UserResponse, status_code=status.HTTP_200_OK)
async def replace_user(user_id: int, payload: UserCreate) -> UserResponse:
    """PUT /v1/users/{id} — full replacement, idempotent.

    Calling this N times with the same body always produces the same result.
    """
    if user_id not in _users:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="user not found")
    _users[user_id] = {"id": user_id, "name": payload.name, "email": payload.email}
    log.info("replaced user id=%d", user_id)
    return UserResponse(**_users[user_id])


@v1.patch("/users/{user_id}", response_model=UserResponse, status_code=status.HTTP_200_OK)
async def patch_user(user_id: int, payload: UserUpdate) -> UserResponse:
    """PATCH /v1/users/{id} — partial update, idempotent because values are absolute.

    Payload uses absolute values, never increments, so retrying is safe.
    """
    if user_id not in _users:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="user not found")
    current = _users[user_id]
    if payload.name is not None:
        current["name"] = payload.name
    if payload.email is not None:
        current["email"] = payload.email
    log.info("patched user id=%d fields=%s", user_id, payload.model_fields_set)
    return UserResponse(**current)


@v1.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: int) -> None:
    """DELETE /v1/users/{id} — idempotent; second call returns 404 (not 500)."""
    if user_id not in _users:
        # Idempotent: already gone is still 'gone', but 404 is correct after first delete
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="user not found")
    del _users[user_id]
    log.info("deleted user id=%d", user_id)
    # 204 No Content — no body


# ---------------------------------------------------------------------------
# Idempotency key middleware for payments
# ---------------------------------------------------------------------------

async def check_idempotency(
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    redis: aioredis.Redis = Depends(get_redis),
) -> str | None:
    """
    Dependency that checks for a stored response for this idempotency key.
    Raises 409 if the key is currently in-flight.
    Returns the key so the handler can store its result.
    """
    if idempotency_key is None:
        return None
    stored = await redis.get(f"idem:{idempotency_key}")
    if stored == "__in_flight__":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="request with this idempotency key is already in flight",
        )
    if stored:
        # Already completed — signal handler to return stored result
        # In a real implementation you'd deserialize the full response here
        log.info("idempotency cache hit key=%s", idempotency_key)
    return idempotency_key


@v1.post("/payments", response_model=PaymentResponse, status_code=status.HTTP_201_CREATED)
async def create_payment(
    payload: PaymentCreate,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    redis: aioredis.Redis = Depends(get_redis),
) -> PaymentResponse:
    """POST /v1/payments — non-idempotent by nature, made safe via idempotency keys.

    Client MUST supply Idempotency-Key for retry safety.
    Server stores result for IDEMPOTENCY_TTL_SECONDS seconds.
    """
    if idempotency_key is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Idempotency-Key header is required for payment requests",
        )

    cache_key = f"idem:payment:{idempotency_key}"

    # Check for existing result
    cached = await redis.get(cache_key)
    if cached:
        log.info("returning cached payment response for key=%s", idempotency_key)
        import json
        return PaymentResponse(**json.loads(cached))

    # Mark in-flight
    await redis.setex(cache_key, 30, "__in_flight__")

    try:
        # Simulate payment processing
        payment_id = str(uuid.uuid4())
        result = PaymentResponse(
            payment_id=payment_id,
            status="authorized",
            amount_cents=payload.amount_cents,
            currency=payload.currency,
        )

        # Store result with full TTL
        import json
        await redis.setex(cache_key, IDEMPOTENCY_TTL_SECONDS, json.dumps(result.model_dump()))
        log.info("created payment id=%s key=%s", payment_id, idempotency_key)
        return result

    except Exception:
        # Remove in-flight marker so client can retry
        await redis.delete(cache_key)
        raise


# ---------------------------------------------------------------------------
# v2 router — breaking change: email is now required on create
# Demonstrates how versioning isolates breaking changes
# ---------------------------------------------------------------------------
v2 = APIRouter(prefix="/v2", tags=["v2"])


class UserCreateV2(BaseModel):
    """v2 adds phone_number (optional) and tightens name validation."""

    name: str
    email: EmailStr
    phone_number: str | None = None

    @field_validator("name")
    @classmethod
    def name_min_length(cls, v: str) -> str:
        if len(v.strip()) < 2:
            raise ValueError("name must be at least 2 characters")
        return v.strip()


@v2.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user_v2(payload: UserCreateV2, response: Response) -> UserResponse:
    """POST /v2/users — same resource, stricter contract, isolated from v1."""
    global _next_user_id
    new_id = _next_user_id
    _next_user_id += 1
    _users[new_id] = {"id": new_id, "name": payload.name, "email": payload.email}
    response.headers["Location"] = f"/v2/users/{new_id}"
    log.info("v2 created user id=%d", new_id)
    return UserResponse(**_users[new_id])


# Mount v2 reads from v1 (DRY — non-breaking read endpoints shared)
v2.include_router(v1, prefix="")  # inherits GET /v1/users under /v2/users via re-use pattern


# ---------------------------------------------------------------------------
# App assembly
# ---------------------------------------------------------------------------
app = FastAPI(title="Week 08 Day 1 — API Design", version="2.0.0")
app.include_router(v1)
app.include_router(v2)


@app.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
```

---

## Exercises

1. Add a `GET /v1/users?email=alice@example.com` filter endpoint.
   The filter must be case-insensitive and must not break the existing list endpoint.
   Write a pytest test that verifies filtering returns only matching users.

2. Implement a `POST /v1/users/bulk` endpoint that creates multiple users in a single
   request and returns 207 Multi-Status with per-item success/failure.
   What HTTP status code does the outer response carry when some items fail?

3. Extend the idempotency key store to handle concurrent in-flight requests using Redis
   `SET NX EX` (set-if-not-exists with expiry).  
   Write a test that sends two simultaneous identical requests and verifies that exactly
   one 201 and one 409 are returned.

4. Add `Sunset` and `Deprecation` headers (RFC 8594) to all v1 endpoints.
   Use FastAPI middleware rather than adding headers to every route individually.

(Solutions: ../solutions/week-08/day-1.md)

---

## Self-check

- [ ] I can place any API endpoint on the correct RMM level (0–3) and explain the gap to the next level.
- [ ] I can explain why DELETE is idempotent but POST is not, in terms of server state.
- [ ] I can choose between URL path, header, and query-param versioning and defend the choice.
- [ ] I can implement idempotency keys end-to-end: client key generation, server lookup, in-flight marker, result storage, TTL.
- [ ] I can select the correct 2xx, 4xx, or 5xx status code for at least 10 scenarios without looking them up.
- [ ] I can implement PATCH correctly using absolute values so the endpoint is idempotent.

---

## Further Reading

- [Richardson Maturity Model — Martin Fowler](https://martinfowler.com/articles/richardsonMaturityModel.html)
- [RFC 7231 — HTTP/1.1 Semantics and Content (verbs, status codes)](https://datatracker.ietf.org/doc/html/rfc7231)
- [RFC 8594 — The Sunset HTTP Header Field](https://datatracker.ietf.org/doc/html/rfc8594)
