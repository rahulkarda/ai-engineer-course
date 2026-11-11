# Week 07 Assessment

## Conceptual (interview-style)

1. Explain the difference between PUT and PATCH. When would you choose PATCH over PUT in a
   financial transactions API, and what idempotency guarantees does each method carry?

2. A client sends a POST request to create a transaction and the server returns 201. The client
   retries due to a network timeout and receives 409. Describe the correct HTTP status codes for
   each scenario and how you would design the endpoint to be idempotent using a
   client-generated idempotency key.

3. REST defines six architectural constraints. Name them and explain which two most directly
   affect how you structure a FastAPI application with JWT authentication and a PostgreSQL
   backend.

4. Pydantic v2 introduces `model_validator` and `field_validator`. What is the execution order
   between `field_validator` (mode='before') and `model_validator` (mode='before') when both
   are defined on the same model? Write a concrete example showing how you would validate that
   a transaction's `amount` is positive AND that `currency` is uppercase, using the correct
   validator types.

5. FastAPI's dependency injection system uses `Depends()`. Explain the difference between a
   dependency declared at the path-operation level versus one declared in the `APIRouter`
   constructor. Describe a real scenario where router-level injection is preferable to
   path-operation-level injection for an auth requirement.

6. Describe the three Base64URL-encoded segments of a JWT. What information belongs in the
   payload versus the header, and why should sensitive data such as account numbers never be
   stored in the payload without additional encryption? What does FastAPI do — and not do —
   when it receives a bearer token?

7. SQLAlchemy 2.0 async requires `AsyncSession` and `asyncpg`. Explain why you cannot call
   `session.execute()` synchronously inside an async path operation, and describe the correct
   pattern for obtaining an `AsyncSession` per request using FastAPI's dependency injection
   (show the generator function signature and the `async_sessionmaker` setup).

8. A FastAPI middleware receives every request before routing. Compare using middleware versus a
   dependency for implementing rate limiting. What are the trade-offs regarding access to path
   parameters, response mutation, and per-endpoint configurability? In which cases is middleware
   the correct choice over a dependency, and vice versa?

---

## Coding challenge

### Challenge: Rate-Limiting Dependency + Cursor-Based Paginated Endpoint

#### Background

You are extending a fintech REST API. Two features are required:

1. A reusable FastAPI dependency that enforces per-client rate limits.
2. A paginated `GET /transactions` endpoint that uses cursor-based pagination.

#### Part A — Rate-Limiting Dependency

Implement a FastAPI dependency class `RateLimiter` that:

- Accepts constructor arguments `max_requests: int` and `window_seconds: int`.
- Identifies the client by the `X-Forwarded-For` header, falling back to `request.client.host`.
- Tracks request counts in an in-process store (a plain dict is acceptable for this challenge;
  note its limitations in a multi-process deployment).
- Raises `HTTPException(status_code=429)` with a JSON body
  `{"detail": "Rate limit exceeded"}` when the client exceeds `max_requests` within
  `window_seconds`.
- Sets the following headers on every response that is **not** rejected:
  - `X-RateLimit-Limit`: the configured `max_requests` value.
  - `X-RateLimit-Remaining`: how many requests the client has left in the current window.
  - `X-RateLimit-Reset`: Unix timestamp (integer) when the current window expires.

Acceptance criteria:

- `RateLimiter(max_requests=5, window_seconds=60)` used as a dependency on an endpoint.
- The first 5 requests within 60 seconds return 200 with correct headers.
- The 6th request within the same window returns 429 with `{"detail": "Rate limit exceeded"}`.
- After the window expires, the counter resets and the client can make requests again.
- The dependency is injected without modifying the endpoint function's return type or signature
  beyond adding it to `Depends()`.
- At least two pytest tests covering the allow and reject paths.

#### Part B — Cursor-Based Paginated Endpoint

Implement `GET /transactions` with cursor-based pagination. Requirements:

- Query parameters: `cursor: str | None = None`, `limit: int = Query(default=20, ge=1, le=100)`.
- The `cursor` is an opaque base64-encoded string that encodes the `id` (UUID) of the last
  item seen by the client. When absent, return the first page.
- The response model must be:

```python
class PaginatedTransactions(BaseModel):
    items: list[TransactionResponse]
    next_cursor: str | None
    total_count: int
```

- `next_cursor` is `None` when there are no further pages.
- The endpoint must apply an `ORDER BY created_at DESC, id DESC` sort to ensure stable
  pagination across concurrent inserts.
- The endpoint requires authentication (JWT bearer token) and returns only transactions
  belonging to the authenticated customer, unless the caller has the `admin` role.
- Return 400 with `{"detail": "Invalid cursor"}` if the cursor cannot be decoded or does not
  correspond to an existing transaction.
- At least three pytest tests: first page (no cursor), subsequent page (valid cursor), invalid
  cursor.

#### Constraints

- Use FastAPI, Pydantic v2, SQLAlchemy 2.0 async, and Python 3.12+.
- All secrets and connection strings via `os.getenv()` — no hardcoded values.
- Tests must use `pytest-asyncio` and mock or override the database session dependency.
- Do not use offset-based pagination anywhere in Part B.

---

## Rubric (pass >= 75%)

| Dimension | Weight |
|---|---|
| Independent implementation | 40% |
| Conceptual understanding | 25% |
| Debugging & problem-solving | 20% |
| Code quality & testing | 15% |

Answer key: ../solutions/week-07/assessment.md
