# Week 08 Assessment — API & Microservice Architecture + Database Internals

## Conceptual (interview-style)

1. **CAP theorem.** State the CAP theorem precisely. Explain why "CA" is not a practically achievable combination in a distributed system that can experience network partitions.

2. **Consistency models.** Rank these consistency models from strongest to weakest and give a one-sentence definition of each: linearizability, sequential consistency, causal consistency, eventual consistency.

3. **RFC 7807.** What problem does RFC 7807 solve? List the five standard fields and explain why `type` is a URI rather than a plain string.

4. **Cursor vs. offset pagination.** A product manager asks: "Why can't we just use `?page=5&size=20` — it works fine in our prototype." Write the two-paragraph answer you would give, covering correctness under concurrent writes and query performance at scale.

5. **Idempotency.** Define idempotency in the context of HTTP. Which HTTP methods are idempotent by specification? How do you make `POST` idempotent in practice?

6. **Database index internals.** Explain how a B-tree index processes a range query (`WHERE created_at > '2024-01-01'`). Why does a composite index on `(status, created_at)` not help a query that filters only on `created_at`?

7. **ACID vs. BASE.** Describe the four ACID properties. What does BASE stand for and which distributed trade-off does it reflect?

8. **Service-to-service communication.** Compare synchronous REST calls between microservices with asynchronous event-driven communication. Give one concrete scenario where each is the correct choice and justify your answer.

9. **Two-phase commit.** Explain the two phases of 2PC. What is the coordinator failure problem and how does the saga pattern address it?

10. **Read replicas.** A service reads from a read replica. A user writes a record and immediately fetches it — the fetch returns "not found." Name this phenomenon, explain its cause, and describe two mitigation strategies at the application layer.

---

## Coding challenge

### Extend the 2-service system with reliability and observability

You have two FastAPI services from the week project: an **Order Service** and an **Inventory Service**. Both are running locally. Extend them according to the specification below.

**Context:** Orders are created in Order Service. Order Service calls Inventory Service to reserve stock. The reservation call is synchronous today.

### Task 1 — Standardise error contracts (RFC 7807)

Add a global exception handler to **both** services that:

- Returns `Content-Type: application/problem+json` for all `HTTPException` and unhandled `Exception` responses.
- Includes `type`, `title`, `status`, `detail`, and `instance` on every error response.
- For validation errors (Pydantic `ValidationError`), includes an additional `errors` array where each element has `field` (dot-path) and `message` (constraint violated).

Write at least 3 pytest tests:

- A test that triggers a 404 and asserts `Content-Type` is `application/problem+json`.
- A test that submits invalid input and asserts the `errors` array is present and non-empty.
- A test that calls a non-existent route and asserts a 404 Problem Detail is returned.

### Task 2 — Cursor-based pagination on Order Service

Replace any offset-based `GET /orders` endpoint with cursor pagination:

- Accept `limit` (int, 1–100, default 20) and `cursor` (optional opaque string) query parameters.
- Sort by `(created_at, order_id)`.
- Return `{ "items": [...], "next_cursor": "<string or null>" }`.
- Cursor must be base64url-encoded JSON. Invalid or tampered cursors must return a 400 Problem Detail (not a 500).
- The underlying query must not use SQL `OFFSET` (in-memory list filter is acceptable if you implement the keyset predicate correctly).

Write at least 2 pytest tests:

- A test that creates 5 orders, pages through with `limit=2`, and asserts every order is returned exactly once.
- A test that passes a malformed cursor and asserts a 400 `application/problem+json` response.

### Task 3 — Architecture Decision Record

Write `docs/adr/ADR-001.md` in the Nygard format for the consistency choice your system makes when an order is created and inventory is updated. The ADR must:

- Name the specific CAP position (AP, CP, or CA).
- Describe the failure scenario the chosen approach introduces.
- Describe the mitigation at the application layer.

### Acceptance criteria

- All tests pass with `pytest -q`.
- Neither service imports the other's models (services must stay independently deployable).
- No hardcoded secrets or connection strings — all config via `os.getenv()`.
- `ruff check .` reports zero violations.
- The `ADR-001.md` file exists and follows the Nygard template.

---

## Rubric (pass >= 75%)

| Dimension | Weight |
|---|---|
| Independent implementation | 40% |
| Conceptual understanding | 25% |
| Debugging & problem-solving | 20% |
| Code quality & testing | 15% |

### Scoring guide

**Independent implementation (40%)**
- Full marks: all three tasks work end-to-end, tests pass, no scaffolding copied verbatim.
- Partial: one task incomplete or tests missing.
- Failing: two or more tasks not attempted.

**Conceptual understanding (25%)**
- Full marks: answers to conceptual questions are precise, use correct terminology, and include concrete examples.
- Partial: answers are mostly correct but lack precision or miss a key distinction.
- Failing: answers confuse terms or demonstrate a fundamental misunderstanding.

**Debugging & problem-solving (20%)**
- Full marks: student can explain why each design decision was made, and can trace a bug in the cursor or error handler code to root cause.
- Partial: can identify a bug but not explain root cause.
- Failing: cannot diagnose a simple error in their own code.

**Code quality & testing (15%)**
- Full marks: full type hints, Pydantic v2 models, `logging` not `print`, tests cover happy path and error paths, `ruff` clean.
- Partial: some type hints missing, minimal tests, minor linting issues.
- Failing: no tests, `print` statements, untyped functions.

Answer key: ../solutions/week-08/assessment.md
