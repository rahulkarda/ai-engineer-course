# Week 7 Project — Financial Operations REST API

## Goal

Build a production-ready REST API for a fintech company. The API handles customer onboarding,
transaction processing, and account balance queries. It must enforce authentication, role-based
authorization, input validation, and structured error handling at every layer.

By the end of this project you will have applied FastAPI, async SQLAlchemy, Pydantic v2, and
JWT-based auth in a realistic, deployable service.

---

## Tech Stack

| Component | Technology |
|---|---|
| Web framework | FastAPI (latest stable) |
| Database | PostgreSQL 15+ |
| ORM / sessions | SQLAlchemy 2.0 async (`asyncpg` driver) |
| Schema validation | Pydantic v2 |
| Authentication | JWT (HS256), `python-jose` or `PyJWT` |
| Testing | pytest, pytest-asyncio, httpx `AsyncClient` |
| Config | `pydantic-settings`, all secrets via `os.getenv()` |

**Required environment variables (no defaults, no hardcoded fallbacks):**

```python
DATABASE_URL = os.getenv("DATABASE_URL")   # e.g. postgresql+asyncpg://...
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
```

Provide a `.env.example` with all variable names and no values.

---

## Required Endpoints

### Authentication context

Every protected endpoint must validate the `Authorization: Bearer <token>` header. The token
payload must include at minimum: `sub` (customer ID as UUID string), `role` (`admin` or
`customer`), and `exp`.

---

### POST /customers

Create a new customer account.

**Auth:** None (public registration endpoint)

**Request body (`CustomerCreate`):**

```python
class CustomerCreate(BaseModel):
    email: EmailStr
    full_name: str = Field(min_length=2, max_length=120)
    password: str = Field(min_length=10)
```

**Response (`CustomerResponse`, 201 Created):**

```python
class CustomerResponse(BaseModel):
    id: uuid.UUID
    email: EmailStr
    full_name: str
    created_at: datetime
```

**Validation rules:**

- Email must be unique; return `409 Conflict` with `{"detail": "Email already registered"}` on
  duplicate.
- Hash the password with `bcrypt` before persisting; never store or return plaintext passwords.
- `id` and `created_at` are server-generated; never accept them from the client.

---

### GET /customers/{id}

Retrieve a customer by UUID.

**Auth:** Required. A `customer` role may only fetch their own record (`id` must equal `sub`
from the token). An `admin` role may fetch any record.

**Response (`CustomerResponse`, 200 OK)**

**Error cases:**

- `404 Not Found` — customer does not exist.
- `403 Forbidden` — customer role attempting to access another customer's record.

---

### POST /transactions

Create a new financial transaction.

**Auth:** Required (`customer` or `admin` role).

**Request body (`TransactionCreate`):**

```python
class TransactionCreate(BaseModel):
    account_id: uuid.UUID
    amount: Decimal = Field(gt=Decimal("0"), decimal_places=2)
    currency: str = Field(min_length=3, max_length=3)
    transaction_type: Literal["credit", "debit"]
    description: str | None = Field(default=None, max_length=255)
```

**Validation rules:**

- `amount` must be strictly greater than zero. Use `Decimal` (not `float`) for monetary values.
- `currency` must be a valid ISO 4217 three-letter code (uppercase). Validate with a
  `field_validator` that checks against a predefined set of accepted currencies.
- `account_id` must belong to the authenticated customer (or any account for `admin`). Return
  `403 Forbidden` if not.
- The transaction is created in `pending` status.

**Response (`TransactionResponse`, 201 Created):**

```python
class TransactionResponse(BaseModel):
    id: uuid.UUID
    account_id: uuid.UUID
    customer_id: uuid.UUID
    amount: Decimal
    currency: str
    transaction_type: str
    status: Literal["pending", "completed", "failed", "reversed"]
    description: str | None
    created_at: datetime
    updated_at: datetime
```

---

### GET /transactions

List transactions with filtering and cursor-based pagination.

**Auth:** Required. `customer` role sees only their own transactions. `admin` role sees all.

**Query parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `cursor` | `str \| None` | `None` | Opaque pagination cursor (base64-encoded ID) |
| `limit` | `int` | `20` | Items per page. Range: 1–100 |
| `status` | `str \| None` | `None` | Filter by transaction status |
| `from_date` | `date \| None` | `None` | Inclusive start date (ISO 8601) |
| `to_date` | `date \| None` | `None` | Inclusive end date (ISO 8601) |

**Response (`PaginatedTransactions`, 200 OK):**

```python
class PaginatedTransactions(BaseModel):
    items: list[TransactionResponse]
    next_cursor: str | None
    total_count: int
```

**Requirements:**

- Sort order: `created_at DESC, id DESC` — stable under concurrent inserts.
- Return `400 Bad Request` with `{"detail": "Invalid cursor"}` for a malformed or unknown cursor.
- Do not use offset-based pagination.

---

### GET /transactions/{id}

Retrieve a single transaction.

**Auth:** Required. `customer` role may only fetch transactions where
`transaction.customer_id == token.sub`. `admin` role may fetch any.

**Response (`TransactionResponse`, 200 OK)**

**Error cases:**

- `404 Not Found` — transaction does not exist.
- `403 Forbidden` — ownership check fails.

---

### PATCH /transactions/{id}

Update the status of an existing transaction.

**Auth:** Required. **Admin role only.** A `customer` token must receive `403 Forbidden`
regardless of ownership.

**Request body (`TransactionStatusUpdate`):**

```python
class TransactionStatusUpdate(BaseModel):
    status: Literal["completed", "failed", "reversed"]
```

**Validation rules:**

- Cannot transition a `reversed` or `failed` transaction (return `409 Conflict` with
  `{"detail": "Terminal status cannot be changed"}`).
- `updated_at` must be refreshed on every successful update.

**Response (`TransactionResponse`, 200 OK)**

---

### GET /accounts/{id}/balance

Return the current balance for an account.

**Auth:** Required. `customer` role may only access accounts they own. `admin` role may access
any account.

**Response (`AccountBalance`, 200 OK):**

```python
class AccountBalance(BaseModel):
    account_id: uuid.UUID
    customer_id: uuid.UUID
    currency: str
    balance: Decimal
    as_of: datetime
```

**Implementation note:** Compute `balance` as the sum of completed `credit` transactions minus
the sum of completed `debit` transactions for the account. Use a single async SQLAlchemy query
with aggregation rather than fetching all rows in Python.

---

## Cross-Cutting Requirements

### Role-Based Authorization

- Define an `admin` role and a `customer` role.
- Implement a reusable `require_role(*roles)` dependency factory that raises
  `403 Forbidden` when the token role is not in the allowed set.
- The dependency must be composable: `Depends(require_role("admin"))` and
  `Depends(require_role("admin", "customer"))` must both work.

### Input Validation

- All request bodies validated by Pydantic v2 models.
- Use `model_validator` for cross-field validation (e.g., `to_date >= from_date`).
- Return RFC 7807-style error bodies for validation failures (FastAPI default `422` is
  acceptable; document the format in your README).

### Error Handling

- Register a global exception handler for unhandled `Exception` that returns
  `500 Internal Server Error` with `{"detail": "Internal server error"}` — never expose
  stack traces in the response body.
- Log the full exception (including traceback) at `ERROR` level using Python's `logging` module
  before returning the 500 response.
- Use structured logging (e.g., JSON lines) for all log output; include `request_id`,
  `method`, `path`, and `status_code` in every log entry.

### Middleware

- Add a request-ID middleware that generates a `UUID` per request, attaches it to the request
  state, and adds it to every response as `X-Request-ID`.
- Add a logging middleware that records method, path, status code, and latency (ms) for every
  request.

### Database

- Use `AsyncSession` and `async_sessionmaker` throughout — no synchronous session calls in
  async path operations.
- Provide Alembic migration files for the initial schema. The `alembic upgrade head` command
  must create all required tables from scratch.
- `DATABASE_URL` must be read exclusively from `os.getenv("DATABASE_URL")`.

### OpenAPI Documentation

- All endpoints must have `summary`, `description`, and `responses` documented.
- Pydantic models must include `model_config = ConfigDict(json_schema_extra={"example": {...}})`
  with realistic example values for every schema shown in the OpenAPI docs.
- The API must be browsable at `/docs` (Swagger UI) and `/redoc`.

---

## Testing Requirements

- Target: greater than 80% line coverage measured by `pytest-cov`.
- Use `httpx.AsyncClient` with `ASGITransport` for integration tests — do not spin up a live
  server.
- Override the `AsyncSession` dependency in tests using FastAPI's `app.dependency_overrides`.
- Required test categories (use `pytest.mark`):
  - `unit` — pure function and validator tests (no DB, no HTTP).
  - `integration` — endpoint tests with a mocked or in-memory DB session.
  - `auth` — token validation, role enforcement, ownership checks.
- Every endpoint must have at least one happy-path test and one error-path test.
- Tests must not require a live database or a live JWT signing service; mock or stub both.

---

## Project Structure

```
src/
  financial_api/
    agents/           # (empty for now, reserved)
    config.py         # pydantic-settings Config class
    database.py       # async engine, sessionmaker, Base
    main.py           # FastAPI app factory, middleware registration
    models/
      customer.py     # SQLAlchemy ORM model
      transaction.py
      account.py
    routers/
      customers.py
      transactions.py
      accounts.py
    schemas/
      customer.py     # Pydantic request/response models
      transaction.py
      account.py
    dependencies/
      auth.py         # JWT decode, get_current_user, require_role
      pagination.py   # cursor encode/decode helpers
      rate_limit.py   # RateLimiter dependency
    middleware/
      logging.py
      request_id.py
tests/
  unit/
  integration/
  auth/
alembic/
  versions/
  env.py
.env.example
pyproject.toml
```

No file in `src/` should exceed 300 lines. Split if it does.

---

## Deliverables Checklist

- [ ] All seven endpoints implemented and returning correct status codes.
- [ ] JWT authentication working; `require_role` dependency enforced on all protected routes.
- [ ] Pydantic v2 models for all request and response schemas, with field-level and
      cross-field validation.
- [ ] Async SQLAlchemy setup with `AsyncSession` dependency and Alembic migrations.
- [ ] Cursor-based pagination on `GET /transactions`.
- [ ] Request-ID and logging middleware registered.
- [ ] Global exception handler returning 500 without stack traces.
- [ ] OpenAPI docs with examples at `/docs`.
- [ ] `.env.example` listing all required variables (no values).
- [ ] `pytest` suite passing with >80% coverage (`pytest --cov=src --cov-report=term-missing`).
- [ ] `ruff check src/ tests/` passes with zero violations.
- [ ] `pyproject.toml` with all dependencies pinned.
- [ ] README with setup instructions, environment variable table, and how to run tests.

---

## Evaluation Criteria

| Criterion | Description |
|---|---|
| Correctness | All endpoints return the specified status codes and response bodies |
| Auth enforcement | JWT validation, role checks, and ownership checks are airtight |
| Validation | Pydantic models reject invalid input with clear error messages |
| Async correctness | No blocking calls inside async path operations |
| Pagination | Cursor-based pagination is stable and does not use OFFSET |
| Test coverage | >80% line coverage; happy-path and error-path tests for every endpoint |
| Code quality | Separation of concerns, typed state, no god files, ruff-clean |
| Security | No hardcoded secrets; passwords hashed; no PII in logs |
