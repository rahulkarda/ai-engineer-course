# Phase 1 · Week 07 · Day 1 — HTTP Fundamentals, REST Architecture & FastAPI Basics

> Objective: Understand HTTP request/response cycle, REST constraints, and spin up a FastAPI app  ·  Time: ~1 hr  ·  Prereqs: [[week-06/day-1]]

## Concept

### HTTP request/response cycle

HTTP is a stateless, text-based application protocol. Every interaction is one request → one response, with no server-side memory of previous requests.

**Request anatomy:**
```
METHOD /path?query HTTP/1.1
Host: api.example.com
Content-Type: application/json
Authorization: Bearer <token>

{"field": "value"}
```

**Response anatomy:**
```
HTTP/1.1 200 OK
Content-Type: application/json
X-Request-Id: abc123

{"id": 1, "name": "Alice"}
```

**Methods and their semantics:**

| Method | Safe | Idempotent | Body | Typical use |
|--------|------|------------|------|-------------|
| GET    | yes  | yes        | no   | Read resource |
| POST   | no   | no         | yes  | Create resource |
| PUT    | no   | yes        | yes  | Replace resource entirely |
| PATCH  | no   | no         | yes  | Partial update |
| DELETE | no   | yes        | no   | Remove resource |

- **Safe**: no side effects on the server
- **Idempotent**: calling N times produces the same result as calling once

**Status code families:**

| Range | Meaning | Examples |
|-------|---------|---------|
| 2xx | Success | 200 OK, 201 Created, 204 No Content |
| 3xx | Redirect | 301 Moved, 304 Not Modified |
| 4xx | Client error | 400 Bad Request, 401 Unauthorized, 403 Forbidden, 404 Not Found, 409 Conflict, 422 Unprocessable Entity |
| 5xx | Server error | 500 Internal Server Error, 503 Service Unavailable |

**Headers** carry metadata: `Content-Type` describes body format, `Authorization` carries credentials, `Cache-Control` instructs caches, `Accept` tells the server what format the client wants.

### REST constraints

REST (Representational State Transfer) is an architectural style, not a protocol. Six constraints define it:

1. **Client-server**: UI and storage are decoupled — each can evolve independently.
2. **Stateless**: every request must contain all information needed to process it. No session state on the server between requests.
3. **Cacheable**: responses must declare whether they are cacheable. Caching eliminates redundant requests.
4. **Uniform interface**: resources are identified by URIs; manipulation happens through representations; messages are self-descriptive; hypermedia drives application state (HATEOAS).
5. **Layered system**: client cannot tell whether it is connected directly to the server or through an intermediary (load balancer, cache, gateway).
6. **Code on demand** (optional): servers can extend client functionality by transferring executable code (e.g., JavaScript).

### Resource modeling

Resources are **nouns**, not verbs. The HTTP method expresses the action.

```
# Wrong — verb in path
POST /createUser
GET  /getUserById?id=1
POST /deleteUser

# Correct — noun resources
POST   /users          -> create
GET    /users          -> list
GET    /users/{id}     -> read one
PUT    /users/{id}     -> replace
PATCH  /users/{id}     -> partial update
DELETE /users/{id}     -> delete
```

Hierarchy expresses relationships:
```
GET /users/{user_id}/orders          # orders belonging to a user
GET /users/{user_id}/orders/{id}     # one specific order
POST /users/{user_id}/orders         # create order for that user
```

Keep URIs lowercase, use hyphens not underscores, no trailing slash.

### FastAPI architecture

FastAPI is built on two layers:

- **Starlette**: the ASGI (Asynchronous Server Gateway Interface) framework underneath — handles routing, middleware, WebSockets, request/response objects.
- **Pydantic v2**: all request body parsing, response serialization, and automatic validation are driven by Pydantic models.

FastAPI adds on top:
- Dependency injection system
- Automatic OpenAPI schema generation from type hints
- Interactive docs at `/docs` (Swagger UI) and `/redoc`

ASGI replaces WSGI for async Python. An ASGI server (uvicorn, hypercorn) calls the app with an async interface, enabling true concurrency with `async`/`await` without threads.

## How it works internally

When a request arrives at a FastAPI app:

1. **uvicorn** (ASGI server) receives raw bytes, parses HTTP, and calls the ASGI `__call__` interface.
2. **Starlette router** matches the path+method to a route handler. Routes are stored in a trie for O(log n) matching.
3. **FastAPI** inspects the handler's signature via `inspect.signature`. It classifies each parameter:
   - `Path(...)`: extracted from the URL path segment
   - `Query(...)`: parsed from `?key=value`
   - `Body(...)`: deserialized from the request body using Pydantic
   - `Depends(...)`: resolved through the DI graph (covered Day 3)
4. **Pydantic v2** validates and coerces all inputs. If validation fails → FastAPI automatically returns HTTP 422 with structured error detail.
5. The route handler runs (sync handlers are moved to a threadpool executor automatically).
6. The return value is serialized: if `response_model` is set, Pydantic filters and validates the output; otherwise FastAPI uses `jsonable_encoder`.
7. The ASGI response streams back through uvicorn.

Automatic OpenAPI schema: FastAPI builds the OpenAPI 3.1 spec at startup by reflecting on all route signatures, Pydantic models, and docstrings. No separate spec file needed.

## Worked example

```python
# main.py
# Minimal FastAPI application demonstrating GET, POST, DELETE
# Run: uvicorn main:app --reload --port 8000

from __future__ import annotations

import logging
from typing import Annotated

import uvicorn
from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Item API",
    description="Demo REST API — Week 07 Day 1",
    version="0.1.0",
)

# --- Models ---

class ItemCreate(BaseModel):
    """Payload for creating an item."""
    name: str = Field(..., min_length=1, max_length=100)
    price: float = Field(..., gt=0)


class Item(ItemCreate):
    """Full item including server-assigned id."""
    id: int


# --- In-memory store (replace with DB in real apps) ---
_db: dict[int, Item] = {}
_next_id: int = 1


# --- Endpoints ---

@app.get("/items", response_model=list[Item], status_code=200)
async def list_items() -> list[Item]:
    """Return all items."""
    return list(_db.values())


@app.post("/items", response_model=Item, status_code=201)
async def create_item(payload: ItemCreate) -> Item:
    """Create a new item. Returns the created item with its assigned id."""
    global _next_id
    item = Item(id=_next_id, **payload.model_dump())
    _db[_next_id] = item
    _next_id += 1
    logger.info("Created item id=%d name=%s", item.id, item.name)
    return item


@app.delete("/items/{item_id}", status_code=204)
async def delete_item(
    item_id: Annotated[int, Path(description="The item to delete", ge=1)],
) -> None:
    """Delete an item by id. Returns 404 if not found."""
    if item_id not in _db:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
    del _db[item_id]
    logger.info("Deleted item id=%d", item_id)


if __name__ == "__main__":
    # Development only — use a process manager in production
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

Run and explore:
```bash
pip install fastapi uvicorn
uvicorn main:app --reload

# Create an item
curl -X POST http://localhost:8000/items \
     -H "Content-Type: application/json" \
     -d '{"name": "Widget", "price": 9.99}'

# List items
curl http://localhost:8000/items

# Delete item 1
curl -X DELETE http://localhost:8000/items/1

# Interactive docs
open http://localhost:8000/docs
```

## Exercises

1. Add a `GET /items/{item_id}` endpoint that returns a single item or 404 if not found. Use `Annotated[int, Path(...)]` for the parameter.

2. Add a `PUT /items/{item_id}` endpoint that replaces an item completely. It should return 404 for missing items and 200 with the updated item on success.

3. Add a `GET /items` query parameter `min_price: float | None = None` that filters items to those with `price >= min_price` when provided. Use `Annotated[float | None, Query(ge=0)]`.

4. Write a `pytest` test file `tests/test_main.py` using `TestClient` from `starlette.testclient`. Cover: create returns 201 with correct body, list returns created items, delete returns 204, delete non-existent returns 404.

(Solutions: ../solutions/week-07/day-1.md)

## Self-check

- [ ] I can explain the difference between safe and idempotent HTTP methods
- [ ] I can explain the six REST constraints and why statelessness matters
- [ ] I can explain how ASGI differs from WSGI and why FastAPI uses it
- [ ] I can model a resource hierarchy as URIs using nouns
- [ ] I can implement a minimal FastAPI app with GET, POST, DELETE and correct status codes without reference
