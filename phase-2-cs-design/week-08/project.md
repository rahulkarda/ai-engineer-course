# Week 08 Project — 2-Service Order and Inventory System

## Overview

Design and partially implement a two-service backend system. The system processes customer orders and manages product inventory. Each service is an independently deployable FastAPI application. By the end of the week you will have working HTTP APIs, a shared event contract, a chosen consistency model with written justification, and a test suite covering both happy paths and failure modes.

This project is the integration target for Days 1–7. Each day's exercises contribute a component. Day 7 ties them together with production-quality error handling and pagination.

---

## Requirements

### Functional requirements

**Order Service**

- `POST /orders` — create an order with `customer_email`, `product_id`, and `quantity`. Validate all fields. Return the created order with a generated `order_id` (UUID) and `created_at` timestamp.
- `GET /orders` — list orders with cursor-based pagination (`limit`, `cursor`). Sort by `(created_at, order_id)`. Return `{ "items": [...], "next_cursor": string | null }`.
- `GET /orders/{order_id}` — retrieve a single order. Return 404 Problem Detail if not found.
- `PATCH /orders/{order_id}/cancel` — cancel an order if its status is `pending`. Return 409 Problem Detail if the order is already `cancelled` or `fulfilled`.
- On order creation, call Inventory Service to reserve stock. If reservation fails, the order must not be created (return 422 Problem Detail).

**Inventory Service**

- `POST /products` — create a product with `sku`, `name`, and `stock_quantity`.
- `GET /products/{sku}` — retrieve a product by SKU. Return 404 Problem Detail if not found.
- `POST /reservations` — reserve `quantity` units for a given `sku` and `order_id`. Return 409 Problem Detail if stock is insufficient.
- `DELETE /reservations/{order_id}` — release a reservation (called on order cancellation).

### Non-functional requirements

- All error responses must conform to RFC 7807 (`application/problem+json`).
- All config (ports, inter-service URLs, DB connection strings if any) must come from environment variables via `os.getenv()`. No literal values in source.
- All public functions and Pydantic models must have complete type hints.
- `logging` module only — no `print` statements.
- `ruff check .` must report zero violations before submission.

---

## Architecture constraints

1. **Independent deployability.** Order Service must not import any module from Inventory Service and vice versa. Shared types (if any) live in a `shared/` package that both depend on, not in either service.

2. **No shared database.** Each service owns its own data store. You may use SQLite (via `aiosqlite` or `sqlite3`) or an in-memory dict/list. Do not let one service query the other's database directly.

3. **Synchronous inter-service call for reservation.** Use `httpx.AsyncClient` in Order Service to call Inventory Service's `/reservations` endpoint. The call must have a timeout (configurable via `os.getenv("INVENTORY_TIMEOUT_SECONDS", "2")`). Handle `httpx.TimeoutException` and return a 503 Problem Detail.

4. **Consistency model.** Choose one of the following and document it in `docs/adr/ADR-001.md`:
   - **Synchronous, strongly consistent:** Order creation is atomic with stock reservation (via 2-phase: call Inventory, if it fails roll back the Order insert). This is CP for the write path.
   - **Asynchronous, eventually consistent:** Order is created immediately; a background task publishes an event for Inventory to consume. Inventory may lag. This is AP.
   - You may not mix approaches without documenting both trade-offs.

5. **Cursor encoding.** Cursors must be base64url-encoded JSON. Malformed or un-decodable cursors must return 400, never 500.

---

## Deliverables

Submit a single directory (or repo) with the following layout:

```
project/
  order_service/
    main.py           # FastAPI app
    models.py         # Pydantic models
    routes/
      orders.py
    services/
      inventory_client.py   # httpx wrapper
    config.py         # pydantic-settings BaseSettings
  inventory_service/
    main.py
    models.py
    routes/
      products.py
      reservations.py
    config.py
  shared/             # optional: shared enums / error helpers
  docs/
    adr/
      ADR-001.md
  tests/
    unit/
      test_order_routes.py
      test_inventory_routes.py
      test_cursor.py
    integration/
      test_order_reservation_flow.py
  .env.example
  pyproject.toml
```

### Minimum viable deliverables (required for pass)

- Both services start with `uvicorn` on different ports.
- `POST /orders` → `POST /reservations` flow works end-to-end.
- Cursor pagination on `GET /orders` works with at least 5 seeded orders.
- All 4xx and 5xx responses are `application/problem+json`.
- `docs/adr/ADR-001.md` exists and follows the Nygard template.
- At least 8 passing pytest tests covering both services.

### Stretch deliverables (for distinction)

- HMAC-signed cursors (Exercise 2 from Day 7).
- Idempotency key on `POST /orders` (Exercise 5 from Day 7).
- A `docker-compose.yml` that starts both services.
- OpenAPI spec exported to `docs/openapi.json` (`GET /openapi.json` from FastAPI).

---

## Grading

| Criterion | Points |
|---|---|
| Both services start and serve requests | 10 |
| `POST /orders` → reservation flow (happy path) | 15 |
| `POST /orders` → reservation failure returns 422 Problem Detail | 10 |
| Cursor pagination correct (no skips, no duplicates across pages) | 15 |
| Invalid cursor returns 400 Problem Detail | 5 |
| RFC 7807 on all error paths (checked with `Content-Type` header) | 10 |
| `ADR-001.md` names CAP position and mitigation | 10 |
| Tests: >= 8 passing, cover both happy and error paths | 15 |
| Code quality: type hints, `logging`, `ruff` clean, no secrets | 10 |
| **Total** | **100** |
| **Pass threshold** | **75** |

Stretch deliverables can compensate for up to 10 points lost elsewhere (cap: 100).

### Submission checklist

- [ ] `.env.example` lists every required env var with a description comment.
- [ ] `.env` is in `.gitignore` and not committed.
- [ ] `ruff check .` exits 0.
- [ ] `pytest -q` exits 0.
- [ ] `ADR-001.md` is present and names a specific CAP trade-off.
- [ ] No service imports the other's source modules.
- [ ] All routes return `application/problem+json` on errors (verify with `curl -I`).
