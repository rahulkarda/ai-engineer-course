# Week 07 · Day 1 — Solutions
## HTTP, REST, and FastAPI Basics

---

### Exercise 1
**Create a FastAPI app with GET /health returning `{"status":"ok","version":"1.0"}` and a custom `X-Request-ID` response header using middleware.**

```python
import uuid
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

app = FastAPI(title="Banking API", version="1.0")


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique X-Request-ID header to every response.

    If the caller already sends X-Request-ID we echo it back;
    otherwise we generate a fresh UUID4.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


app.add_middleware(RequestIDMiddleware)


@app.get("/health")
async def health_check() -> dict:
    """Liveness probe — returns service status and current API version."""
    return {"status": "ok", "version": "1.0"}
```

**Key points:**
- `BaseHTTPMiddleware.dispatch` wraps every request/response cycle.
- We echo an existing `X-Request-ID` header from the caller (useful for distributed tracing) or generate one if absent.
- The `/health` route is intentionally unauthenticated so load-balancer probes can reach it.

---

### Exercise 2
**Model a REST resource hierarchy for a banking API: define URI patterns for customers, their accounts, and transactions; explain which HTTP methods apply to each.**

```
REST Resource Hierarchy
=======================

Resource: Customers
-------------------
GET    /customers               List all customers (with pagination)
POST   /customers               Create a new customer
GET    /customers/{customer_id} Retrieve a single customer
PATCH  /customers/{customer_id} Partially update customer details
DELETE /customers/{customer_id} Close / soft-delete a customer record

Resource: Accounts (nested under Customer)
------------------------------------------
GET    /customers/{customer_id}/accounts                List accounts for customer
POST   /customers/{customer_id}/accounts                Open a new account
GET    /customers/{customer_id}/accounts/{account_id}   Get account details
PATCH  /customers/{customer_id}/accounts/{account_id}   Update account (e.g. overdraft limit)
DELETE /customers/{customer_id}/accounts/{account_id}   Close account

Resource: Transactions (nested under Account)
----------------------------------------------
GET    /customers/{customer_id}/accounts/{account_id}/transactions
           List transactions (supports ?from=&to= date filters, ?limit=, ?offset=)
POST   /customers/{customer_id}/accounts/{account_id}/transactions
           Post a new debit/credit entry
GET    /customers/{customer_id}/accounts/{account_id}/transactions/{tx_id}
           Retrieve a specific transaction

Notes
-----
- PUT is avoided here; PATCH is preferred for partial updates so callers
  do not need to re-send immutable fields (account number, currency, etc.).
- DELETE on an account does NOT delete its transaction history — financial
  records must be retained for compliance. The account moves to CLOSED state.
- Transactions are immutable once posted. No PUT/PATCH/DELETE on /transactions/{id}.
- A top-level /transactions endpoint (without customer/account nesting) can
  be added later for admin/ops searches, but the canonical URLs stay nested.
```

**HTTP method semantics quick reference:**

| Method | Safe | Idempotent | Body | Typical use |
|--------|------|------------|------|-------------|
| GET    | yes  | yes        | no   | Read        |
| POST   | no   | no         | yes  | Create      |
| PUT    | no   | yes        | yes  | Full replace|
| PATCH  | no   | no*        | yes  | Partial update |
| DELETE | no   | yes        | no   | Remove      |

_*PATCH idempotency depends on the patch semantics; JSON Merge Patch is idempotent, JSON Patch with `add` is not._

---

### Exercise 3
**Write a FastAPI endpoint `GET /items/{item_id}` that returns 404 with `{"error":"Item not found","item_id":<id>}` when `item_id > 1000`, and 200 with the item otherwise.**

```python
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    item_id: int
    name: str
    price: float


# Simulated data store — in production this would be a DB query.
def _fetch_item(item_id: int) -> Item | None:
    """Return a mock Item if item_id <= 1000, else None."""
    if item_id > 1000:
        return None
    return Item(item_id=item_id, name=f"Item #{item_id}", price=round(item_id * 0.99, 2))


@app.get(
    "/items/{item_id}",
    response_model=Item,
    responses={
        404: {
            "description": "Item not found",
            "content": {
                "application/json": {
                    "example": {"error": "Item not found", "item_id": 9999}
                }
            },
        }
    },
)
async def get_item(item_id: int) -> Item | JSONResponse:
    """Retrieve a single item by its ID.

    Returns 404 when item_id exceeds the known range (> 1000).
    """
    item = _fetch_item(item_id)
    if item is None:
        return JSONResponse(
            status_code=404,
            content={"error": "Item not found", "item_id": item_id},
        )
    return item
```

**Why `JSONResponse` instead of raising `HTTPException`?**
`HTTPException` embeds the detail inside `{"detail": ...}` which doesn't match the required schema `{"error": ..., "item_id": ...}`. Using `JSONResponse` gives full control over the response body.

---

### Exercise 4
**Add query parameters to `GET /items`: `limit` (int, 1-100, default 20), `offset` (int, default 0), `sort_by` (Literal "name"|"created_at", default "name") — validated with Pydantic.**

```python
from typing import Literal
from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field, field_validator

app = FastAPI()


# ---------------------------------------------------------------------------
# Query parameter model
# ---------------------------------------------------------------------------

class ItemListParams(BaseModel):
    """Validated query parameters for the item list endpoint.

    Using a Pydantic model (rather than individual Query() annotations)
    keeps validation logic in one place and makes the params reusable.
    """

    limit: int = Field(default=20, ge=1, le=100, description="Page size, 1-100")
    offset: int = Field(default=0, ge=0, description="Number of records to skip")
    sort_by: Literal["name", "created_at"] = Field(
        default="name",
        description="Field to sort results by",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"limit": 10, "offset": 0, "sort_by": "name"},
                {"limit": 50, "offset": 100, "sort_by": "created_at"},
            ]
        }
    }


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------

class Item(BaseModel):
    item_id: int
    name: str
    price: float


class PaginatedItems(BaseModel):
    items: list[Item]
    total: int
    limit: int
    offset: int


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

# Fake in-memory dataset
_ITEMS: list[Item] = [
    Item(item_id=i, name=f"Item #{i:04d}", price=round(i * 0.99, 2))
    for i in range(1, 201)
]


@app.get("/items", response_model=PaginatedItems)
async def list_items(params: ItemListParams = Depends()) -> PaginatedItems:
    """List items with pagination and sorting.

    FastAPI treats a Pydantic model injected via Depends() as individual
    query parameters — each field maps to a ?field=value query param.
    """
    # Sort
    reverse = False
    if params.sort_by == "name":
        sorted_items = sorted(_ITEMS, key=lambda x: x.name, reverse=reverse)
    else:  # created_at — in a real app this would be a DB column
        sorted_items = sorted(_ITEMS, key=lambda x: x.item_id, reverse=reverse)

    # Paginate
    page = sorted_items[params.offset : params.offset + params.limit]

    return PaginatedItems(
        items=page,
        total=len(_ITEMS),
        limit=params.limit,
        offset=params.offset,
    )
```

**Key design decisions:**
- `ge=1, le=100` constraints on `limit` produce clean 422 errors automatically — no manual validation needed.
- `Literal["name", "created_at"]` restricts the enum at the Pydantic level; FastAPI surfaces this as an enum in the OpenAPI schema.
- `Depends()` with a Pydantic model is the recommended FastAPI pattern for complex query-param groups — it keeps routes clean and the model unit-testable independently.
