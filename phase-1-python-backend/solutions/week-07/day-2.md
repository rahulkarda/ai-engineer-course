# Week 07 · Day 2 — Solutions
## Routing, Request/Response Models, and Pydantic v2

---

### Exercise 1
**Create a `CustomerCreate` Pydantic v2 model with: `name` (str, 2-100 chars), `email` (EmailStr), `phone` (str, E.164 format), `date_of_birth` (date, must be 18+ years old).**

```python
import re
from datetime import date, timedelta

from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_E164_PATTERN = re.compile(r"^\+[1-9]\d{6,14}$")
"""E.164 format: + followed by country code and subscriber number, 7-15 digits total."""


def _is_18_or_older(dob: date) -> bool:
    """Return True if the person born on `dob` is at least 18 years old today."""
    today = date.today()
    # Subtract 18 years, accounting for leap years.
    try:
        cutoff = date(today.year - 18, today.month, today.day)
    except ValueError:
        # Edge case: today is Feb 29 and the target year is not a leap year.
        cutoff = date(today.year - 18, today.month, today.day - 1)
    return dob <= cutoff


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CustomerCreate(BaseModel):
    """Input model for creating a new customer.

    Validates all fields before they reach application logic.
    """

    name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Full name of the customer",
        examples=["Jane Doe"],
    )
    email: EmailStr = Field(..., description="Primary contact e-mail address")
    phone: str = Field(
        ...,
        description="Phone number in E.164 format (e.g. +14155552671)",
        examples=["+14155552671"],
    )
    date_of_birth: date = Field(..., description="ISO 8601 date; must be 18+ years ago")

    @field_validator("phone")
    @classmethod
    def validate_e164(cls, value: str) -> str:
        """Enforce E.164 phone number format."""
        if not _E164_PATTERN.match(value):
            raise ValueError(
                "phone must be in E.164 format: '+' followed by 7-15 digits "
                "(e.g. +14155552671)"
            )
        return value

    @model_validator(mode="after")
    def validate_minimum_age(self) -> "CustomerCreate":
        """Ensure the customer is at least 18 years old."""
        if not _is_18_or_older(self.date_of_birth):
            raise ValueError(
                "Customer must be at least 18 years old to open an account"
            )
        return self
```

**Why `model_validator` instead of `field_validator` for age?**
The age check only needs `date_of_birth`, so technically a `field_validator` would work.
We use `model_validator(mode="after")` to show the pattern when a check conceptually belongs to the whole record and could reference multiple fields (e.g., `date_of_birth < date_of_enrollment`).

---

### Exercise 2
**Create a `CustomerResponse` model that excludes `password_hash`, includes a `computed_field age: int`, and adds a Config example.**

```python
from datetime import date, datetime
from pydantic import BaseModel, EmailStr, Field, computed_field


class CustomerResponse(BaseModel):
    """Public-facing customer representation returned by the API.

    Deliberately omits `password_hash` — sensitive fields must never
    appear in API responses. The `age` field is derived at serialization
    time so it is always accurate without storing a denormalized value.
    """

    id: int
    name: str
    email: EmailStr
    phone: str
    date_of_birth: date
    created_at: datetime

    # `password_hash` is intentionally absent — it lives only in the DB model.

    @computed_field  # type: ignore[misc]
    @property
    def age(self) -> int:
        """Current age in full years, calculated from date_of_birth."""
        today = date.today()
        years = today.year - self.date_of_birth.year
        # Subtract 1 if the birthday has not occurred yet this calendar year.
        had_birthday = (today.month, today.day) >= (
            self.date_of_birth.month,
            self.date_of_birth.day,
        )
        return years if had_birthday else years - 1

    model_config = {
        "from_attributes": True,  # Allows construction from ORM objects
        "json_schema_extra": {
            "examples": [
                {
                    "id": 42,
                    "name": "Jane Doe",
                    "email": "jane.doe@example.com",
                    "phone": "+14155552671",
                    "date_of_birth": "1990-06-15",
                    "created_at": "2024-01-10T09:30:00Z",
                    "age": 34,
                }
            ]
        },
    }
```

**Key points:**
- `@computed_field` (Pydantic v2) adds a read-only property to the serialized JSON output without storing the value.
- `from_attributes=True` enables `CustomerResponse.model_validate(orm_object)` so SQLAlchemy row objects can be converted directly.
- `password_hash` is excluded by simply not declaring it — no `exclude` config needed.

---

### Exercise 3
**Write `POST /customers`: accepts `CustomerCreate`, mock-stores it, returns `CustomerResponse` with 201, raises 409 if email already exists.**

```python
from datetime import datetime, timezone
from typing import Annotated

from fastapi import FastAPI, HTTPException, status

# Re-use the models from Exercise 1 and 2 (shown inline for completeness).

app = FastAPI(title="Banking API — Customers", version="1.0")


# ---------------------------------------------------------------------------
# In-memory store (replace with a real DB session in production)
# ---------------------------------------------------------------------------

# Maps email -> stored customer dict.  Not thread-safe; use a DB in production.
_customer_store: dict[str, dict] = {}
_next_id: int = 1


def _email_exists(email: str) -> bool:
    return email.lower() in _customer_store


def _save_customer(data: CustomerCreate) -> dict:
    """Persist a new customer to the in-memory store and return the full record."""
    global _next_id
    record = {
        "id": _next_id,
        "name": data.name,
        "email": data.email,
        "phone": data.phone,
        "date_of_birth": data.date_of_birth,
        "created_at": datetime.now(tz=timezone.utc),
        # password_hash would be set here after hashing; omitted for this exercise
    }
    _customer_store[data.email.lower()] = record
    _next_id += 1
    return record


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/customers",
    response_model=CustomerResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        409: {
            "description": "A customer with this email already exists",
            "content": {
                "application/json": {
                    "example": {"detail": "Email already registered"}
                }
            },
        }
    },
)
async def create_customer(payload: CustomerCreate) -> CustomerResponse:
    """Register a new customer.

    Returns the created resource with HTTP 201.
    Raises HTTP 409 Conflict if the email address is already registered.
    """
    if _email_exists(payload.email):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    record = _save_customer(payload)
    return CustomerResponse.model_validate(record)
```

**Why 409 Conflict instead of 400 Bad Request?**
The input is syntactically valid — the conflict is a business-logic constraint (uniqueness), not a malformed request. RFC 7231 describes 409 as "the request could not be completed due to a conflict with the current state of the resource."

---

### Exercise 4
**Demonstrate FastAPI 422 validation errors: write a test using `TestClient` that sends invalid data and asserts the error structure.**

```python
import pytest
from fastapi.testclient import TestClient

# `app` is the FastAPI instance from Exercise 3 (or imported from the module).


client = TestClient(app)


class TestCustomerCreate422:
    """Verify that FastAPI returns well-structured 422 Unprocessable Entity
    responses when the request body fails Pydantic validation."""

    def test_missing_required_fields_returns_422(self):
        response = client.post("/customers", json={})
        assert response.status_code == 422
        body = response.json()
        assert "detail" in body
        # Each element in detail describes one validation error.
        assert isinstance(body["detail"], list)
        assert len(body["detail"]) > 0

    def test_error_item_structure(self):
        """Each detail entry must have 'loc', 'msg', and 'type' keys."""
        response = client.post("/customers", json={})
        errors = response.json()["detail"]
        for error in errors:
            assert "loc" in error, "Missing 'loc' in error item"
            assert "msg" in error, "Missing 'msg' in error item"
            assert "type" in error, "Missing 'type' in error item"

    def test_invalid_email_returns_422(self):
        payload = {
            "name": "Jane Doe",
            "email": "not-an-email",
            "phone": "+14155552671",
            "date_of_birth": "1990-06-15",
        }
        response = client.post("/customers", json=payload)
        assert response.status_code == 422
        locs = [e["loc"] for e in response.json()["detail"]]
        # The error location must mention the 'email' field.
        assert any("email" in loc for loc in locs)

    def test_invalid_phone_returns_422(self):
        payload = {
            "name": "Jane Doe",
            "email": "jane@example.com",
            "phone": "555-1234",  # Not E.164
            "date_of_birth": "1990-06-15",
        }
        response = client.post("/customers", json=payload)
        assert response.status_code == 422
        locs = [e["loc"] for e in response.json()["detail"]]
        assert any("phone" in loc for loc in locs)

    def test_underage_customer_returns_422(self):
        """Date of birth that makes the customer < 18 should fail at model level."""
        from datetime import date, timedelta

        minor_dob = (date.today() - timedelta(days=365 * 17)).isoformat()
        payload = {
            "name": "Minor Child",
            "email": "minor@example.com",
            "phone": "+14155552671",
            "date_of_birth": minor_dob,
        }
        response = client.post("/customers", json=payload)
        assert response.status_code == 422
        msgs = [e["msg"] for e in response.json()["detail"]]
        assert any("18" in msg for msg in msgs)

    def test_name_too_short_returns_422(self):
        payload = {
            "name": "X",  # min_length=2
            "email": "x@example.com",
            "phone": "+14155552671",
            "date_of_birth": "1990-01-01",
        }
        response = client.post("/customers", json=payload)
        assert response.status_code == 422

    def test_valid_customer_returns_201(self):
        """Sanity-check: a valid payload must NOT trigger a 422."""
        payload = {
            "name": "Jane Doe",
            "email": "jane.doe@example.com",
            "phone": "+14155552671",
            "date_of_birth": "1990-06-15",
        }
        response = client.post("/customers", json=payload)
        assert response.status_code == 201
```

**How FastAPI 422 errors work (explanation):**

1. When a request body arrives, FastAPI passes it to Pydantic's `model_validate()`.
2. Pydantic collects **all** field errors (not just the first) before raising `ValidationError`.
3. FastAPI's built-in exception handler converts `RequestValidationError` to a 422 response.
4. The body is `{"detail": [...]}` where each item has:
   - `loc`: list of strings/ints indicating the path to the invalid field, e.g. `["body", "email"]`
   - `msg`: human-readable error message
   - `type`: machine-readable error code, e.g. `"value_error"`, `"missing"`, `"string_too_short"`
5. This structure is defined by the OpenAPI spec and consistent across all FastAPI apps.
