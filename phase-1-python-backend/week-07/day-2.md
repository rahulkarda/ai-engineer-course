# Phase 1 · Week 07 · Day 2 — Routing, Request/Response Models & Pydantic Validation

> Objective: Define typed routes, validate request bodies, and shape API responses with Pydantic v2  ·  Time: ~1 hr  ·  Prereqs: [[week-07/day-1]]

## Concept

### Path parameters, query parameters, request body

FastAPI infers where to find a value from the route signature:

```python
from fastapi import FastAPI, Query
from typing import Annotated

app = FastAPI()

@app.get("/users/{user_id}/orders")
async def list_orders(
    user_id: int,                                   # path param — in the URL template
    page: Annotated[int, Query(ge=1)] = 1,          # query param — ?page=2
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[dict]:
    ...
```

Rules:
- Parameter name matches a `{name}` segment → **path parameter**
- Primitive type with no body annotation, not in path → **query parameter**
- Pydantic `BaseModel` subclass → **request body** (deserialized from JSON)
- `Header(...)`, `Cookie(...)`, `Form(...)` override this logic explicitly

Multiple body parameters combine into one JSON object with each param name as a key.

### Pydantic v2 BaseModel

Pydantic v2 (released 2023) rewrote the core in Rust. Key changes from v1:
- `model_dump()` replaces `.dict()`
- `model_validate()` replaces `.parse_obj()`
- `@field_validator` replaces `@validator`
- `@model_validator` replaces `@root_validator`
- `computed_field` is new

**Field validators** run per-field:

```python
from pydantic import BaseModel, Field, field_validator

class UserCreate(BaseModel):
    email: str
    name: str = Field(..., min_length=1, max_length=100)
    age: int = Field(..., ge=0, le=150)

    @field_validator("email")
    @classmethod
    def email_must_contain_at(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("invalid email format")
        return v.lower()  # transform during validation
```

**Model validators** run after all fields are validated — use for cross-field checks:

```python
from pydantic import model_validator
from typing import Self

class DateRange(BaseModel):
    start: date
    end: date

    @model_validator(mode="after")
    def check_range(self) -> Self:
        if self.end <= self.start:
            raise ValueError("end must be after start")
        return self
```

**Computed fields** are read-only derived properties included in serialization:

```python
from pydantic import computed_field

class Product(BaseModel):
    price_cents: int
    tax_rate: float = 0.2

    @computed_field
    @property
    def price_with_tax(self) -> float:
        return round(self.price_cents / 100 * (1 + self.tax_rate), 2)
```

### Response models: filtering sensitive fields

`response_model` tells FastAPI which Pydantic model to serialize the return value through. This filters fields and validates output shape — the handler can return a richer internal object; FastAPI projects it down.

```python
class UserInternal(BaseModel):
    id: int
    email: str
    password_hash: str   # never expose this
    is_admin: bool

class UserResponse(BaseModel):
    id: int
    email: str
    # password_hash excluded — not in this model
    # is_admin excluded

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int) -> UserInternal:
    # Handler returns UserInternal but FastAPI strips it to UserResponse
    return db.get_user(user_id)
```

`response_model_exclude_none=True` drops fields whose value is `None`, reducing response size.

### HTTP 422 Unprocessable Entity

FastAPI returns 422 when Pydantic validation fails on incoming data. The response body is a structured JSON array of validation errors:

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "email"],
      "msg": "Field required",
      "input": {"name": "Alice"},
      "url": "https://errors.pydantic.dev/2.x/v/missing"
    }
  ]
}
```

Each error has:
- `loc`: list of keys showing where in the request the error occurred (path/query/body, then field names)
- `type`: Pydantic error type (e.g. `string_too_short`, `int_parsing`, `value_error`)
- `msg`: human-readable message
- `input`: the value that failed

422 is the correct status for well-formed JSON that fails semantic validation. Use 400 Bad Request for malformed syntax (unparseable JSON).

## How it works internally

**Pydantic v2 validation pipeline** (per field, in order):
1. Type coercion: `"42"` → `int(42)` if the field type is `int` (strict mode disables this)
2. `Field(...)` constraints: `ge`, `le`, `min_length`, etc. checked as Rust validators
3. `@field_validator` functions run in definition order
4. `@model_validator(mode="before")` runs before field parsing; `mode="after"` runs after all fields are set

FastAPI's `response_model` flow:
1. Handler returns any Python object
2. FastAPI calls `jsonable_encoder(return_value)` to convert to JSON-compatible types
3. If `response_model` is set, it calls `response_model.model_validate(encoded_value)` — this applies the output model's validators and strips unknown fields
4. The result is serialized to JSON bytes and sent in the response

This means `response_model` filters by inclusion: only fields declared in the response model survive. Fields with `exclude=True` in the model's `Field(...)` are also dropped.

**Schema generation**: at startup FastAPI walks every route's type annotations and calls `model.model_json_schema()` for each Pydantic model, aggregating into the OpenAPI `components.schemas` section. Changes to models are reflected in `/docs` immediately on restart.

## Worked example

```python
# customers.py
# Customer CRUD demonstrating Pydantic v2 models and response filtering
# Run: uvicorn customers:app --reload

from __future__ import annotations

import logging
from datetime import date
from typing import Annotated

from fastapi import FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, EmailStr, Field, computed_field, field_validator, model_validator
from typing import Self

logger = logging.getLogger(__name__)
app = FastAPI(title="Customer API", version="0.1.0")


# --- Request/response models ---

class CustomerCreate(BaseModel):
    """Payload for creating a new customer."""
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., description="Customer email address")
    birth_year: int = Field(..., ge=1900, le=2010)
    credit_limit: float = Field(default=1000.0, gt=0)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("must contain @")
        return v.strip().lower()


class CustomerUpdate(BaseModel):
    """Partial update — all fields optional."""
    name: str | None = Field(default=None, min_length=1, max_length=100)
    credit_limit: float | None = Field(default=None, gt=0)


class CustomerResponse(BaseModel):
    """Public view — no internal fields like credit_limit."""
    id: int
    name: str
    email: str
    birth_year: int

    @computed_field
    @property
    def age(self) -> int:
        return date.today().year - self.birth_year


class CustomerInternal(CustomerResponse):
    """Internal model — includes sensitive fields, not exposed directly."""
    credit_limit: float


# --- In-memory store ---
_store: dict[int, CustomerInternal] = {}
_next_id: int = 1


# --- Endpoints ---

@app.get("/customers", response_model=list[CustomerResponse])
async def list_customers(
    name_contains: Annotated[str | None, Query(max_length=100)] = None,
) -> list[CustomerInternal]:
    customers = list(_store.values())
    if name_contains:
        customers = [c for c in customers if name_contains.lower() in c.name.lower()]
    return customers  # FastAPI projects each through CustomerResponse


@app.get("/customers/{customer_id}", response_model=CustomerResponse)
async def get_customer(
    customer_id: Annotated[int, Path(ge=1)],
) -> CustomerInternal:
    customer = _store.get(customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    return customer


@app.post("/customers", response_model=CustomerResponse, status_code=201)
async def create_customer(payload: CustomerCreate) -> CustomerInternal:
    global _next_id
    customer = CustomerInternal(
        id=_next_id,
        **payload.model_dump(),
    )
    _store[_next_id] = customer
    _next_id += 1
    logger.info("Created customer id=%d email=%s", customer.id, customer.email)
    return customer


@app.patch("/customers/{customer_id}", response_model=CustomerResponse)
async def update_customer(
    customer_id: Annotated[int, Path(ge=1)],
    payload: CustomerUpdate,
) -> CustomerInternal:
    customer = _store.get(customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")

    # model_dump(exclude_unset=True) — only fields the caller sent
    updates = payload.model_dump(exclude_unset=True)
    updated = customer.model_copy(update=updates)
    _store[customer_id] = updated
    logger.info("Updated customer id=%d fields=%s", customer_id, list(updates.keys()))
    return updated


@app.delete("/customers/{customer_id}", status_code=204)
async def delete_customer(
    customer_id: Annotated[int, Path(ge=1)],
) -> None:
    if customer_id not in _store:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    del _store[customer_id]
```

Test the 422 behavior:
```bash
# Missing required field — returns 422
curl -X POST http://localhost:8000/customers \
     -H "Content-Type: application/json" \
     -d '{"name": "Bob"}'

# Invalid email — returns 422 with field-level error
curl -X POST http://localhost:8000/customers \
     -H "Content-Type: application/json" \
     -d '{"name": "Bob", "email": "notanemail", "birth_year": 1990}'

# Confirm credit_limit absent from response despite being stored
curl -X POST http://localhost:8000/customers \
     -H "Content-Type: application/json" \
     -d '{"name": "Alice", "email": "alice@example.com", "birth_year": 1985}'
```

## Exercises

1. Add a `@model_validator(mode="after")` to `CustomerCreate` that raises `ValueError` if `birth_year > date.today().year - 18` (customers must be 18+).

2. Add a `CustomerListResponse` model with fields `items: list[CustomerResponse]` and `total: int`. Change `list_customers` to return this wrapper. Add a `computed_field` `has_more: bool` that is `True` when `total > len(items)` (hint: you will need to accept a `limit` query parameter).

3. Create a `CustomerSensitiveResponse` model that includes `credit_limit` alongside the public fields. Add a `GET /customers/{customer_id}/full` endpoint that requires an `x-admin-key` header (use `Header(...)`) and returns this richer model. Return 403 if the header is missing.

4. Write `tests/test_customers.py` using `TestClient`. Cover: 422 on missing email, 422 on invalid email format, 201 on valid creation with `credit_limit` absent from response body, PATCH with `exclude_unset` semantics (send only `name`, confirm `credit_limit` unchanged).

(Solutions: ../solutions/week-07/day-2.md)

## Self-check

- [ ] I can explain how FastAPI decides if a parameter is path, query, or body
- [ ] I can explain the difference between `@field_validator` and `@model_validator`
- [ ] I can implement a response model that strips internal fields from a richer return type
- [ ] I can explain what HTTP 422 means and what the error body looks like
- [ ] I can use `model_dump(exclude_unset=True)` correctly for partial updates without reference
