# Week 06 Project — Financial Transaction Management System

> Time: ~6 hr total (Sat 4 hr + Sun 2 hr)  ·  Prereqs: All Week 06 days

## Goal

Build a production-quality Financial Transaction Management System backed by PostgreSQL. The system handles customer accounts, fund transfers, transaction history, category management, and a complete audit trail — all wired through a FastAPI HTTP layer with SQLAlchemy 2.0.

---

## Requirements

### Functional

| ID | Requirement |
|----|-------------|
| F1 | Create / read / soft-delete customers |
| F2 | Create / read accounts per customer; enforce non-negative balance constraint |
| F3 | Record individual transactions (credit / debit) with idempotency key |
| F4 | Execute atomic fund transfers between two accounts in a single DB transaction |
| F5 | Filter transactions by date range, category, status, and direction |
| F6 | Paginated transaction history (offset pagination) |
| F7 | Auto-generate audit log entries on INSERT/UPDATE to accounts and transactions |
| F8 | Revert a transaction (set status = 'reversed', reverse the balance change) |

### Technical

| ID | Requirement |
|----|-------------|
| T1 | SQLAlchemy 2.0 ORM models with full type annotations |
| T2 | Alembic migrations — every schema change is a migration, not raw DDL |
| T3 | Repository pattern: one class per entity, thin FastAPI layer |
| T4 | `SERIALIZABLE` isolation for transfers, `READ COMMITTED` for reads |
| T5 | All secrets via `os.getenv` / pydantic-settings; no hardcoded values |
| T6 | Indexes: at minimum those from Day 3; verify with EXPLAIN ANALYZE |
| T7 | Test coverage ≥ 70% for repository and service layers (pytest) |
| T8 | `ruff check` passes with zero errors |

---

## Schema

Use the schema defined in Day 3 and Day 5 exactly. All five tables must be present:

```
customers            → customer_id (PK), full_name, email (UNIQUE), phone, is_active, created_at, updated_at
transaction_categories → category_id (PK), name (UNIQUE), description
accounts             → account_id (PK), customer_id (FK), account_number (UNIQUE), account_type (CHECK), balance (≥0), currency, is_active, created_at, updated_at
transactions         → transaction_id (PK), account_id (FK), category_id (FK nullable), amount (>0), direction (CHECK), description, reference_id (UNIQUE), status (CHECK), created_at
audit_logs           → log_id (PK), table_name, record_id, action (CHECK), old_data (JSONB), new_data (JSONB), changed_by, changed_at
```

---

## API Endpoints

Implement at minimum:

```
POST   /customers                         → create customer
GET    /customers/{customer_id}           → get customer
GET    /customers                         → list active customers (paginated)
DELETE /customers/{customer_id}           → soft-delete (set is_active=false)

POST   /accounts                          → create account
GET    /accounts/{account_id}             → get account with balance
GET    /customers/{customer_id}/accounts  → list accounts for customer

POST   /transactions                      → record a single transaction
GET    /accounts/{account_id}/transactions → transaction history (filter + paginate)
POST   /transactions/{tx_id}/reverse      → reverse a transaction

POST   /transfers                         → atomic transfer between accounts

GET    /reports/monthly                   → monthly spend summary (query params: customer_id, months=6)
```

---

## Project structure

```
financial_app/
├── src/
│   └── financial_app/
│       ├── __init__.py
│       ├── config.py          # pydantic-settings
│       ├── database.py        # engine, SessionLocal, get_session
│       ├── models.py          # SQLAlchemy ORM models
│       ├── schemas.py         # Pydantic request/response models
│       ├── repository.py      # CustomerRepo, AccountRepo, TransactionRepo
│       ├── service.py         # TransactionService (business logic)
│       ├── audit.py           # SQLAlchemy event listeners for audit_logs
│       └── main.py            # FastAPI app, routers
├── alembic/
│   ├── env.py
│   └── versions/
│       ├── 0001_initial_schema.py
│       └── 0002_add_parent_account.py   (optional extension)
├── scripts/
│   └── seed.py
├── tests/
│   ├── conftest.py
│   ├── test_customer_repo.py
│   ├── test_transaction_repo.py
│   ├── test_service.py
│   └── test_api.py
├── .env.example
├── alembic.ini
└── pyproject.toml
```

---

## Milestones

### Saturday (Day 6 — 4 hr)

- [ ] Project scaffold created, dependencies installed
- [ ] `alembic init` complete, `env.py` wired to models
- [ ] Initial migration written and applied (`alembic upgrade head`)
- [ ] All five tables exist with constraints and indexes verified via `\d tablename`
- [ ] Seed script inserts 20 customers, ~40 accounts, 500+ transactions
- [ ] Five verification queries from Day 6 run and return expected data

### Sunday (Day 7 — 2 hr)

- [ ] Repository classes for Customer, Account, Transaction complete
- [ ] `TransactionService.transfer()` uses `SERIALIZABLE` isolation
- [ ] `audit.py` event listener auto-inserts audit rows on account/transaction changes
- [ ] FastAPI app runs (`uvicorn financial_app.main:app --reload`)
- [ ] All 9+ API endpoints return correct HTTP status codes
- [ ] `/reports/monthly` returns correct aggregations
- [ ] `pytest` passes with ≥ 70% coverage
- [ ] `ruff check src/` passes with zero errors

---

## Stretch goals (optional)

1. **Cursor-based pagination** — replace offset with a `cursor` (encoded `created_at + id`) for stable pagination under inserts.
2. **Async SQLAlchemy** — switch to `AsyncSession` + `asyncpg`; refactor all repositories to `async`/`await`.
3. **Full-text search** — add a `GIN` index on `transactions.description` using `to_tsvector`; add a `GET /transactions/search?q=` endpoint.
4. **Materialized view** — create a `monthly_account_summary` materialized view and a `REFRESH MATERIALIZED VIEW CONCURRENTLY` endpoint.
5. **Docker Compose** — add `docker-compose.yml` with `postgres:16` and the `financial_app` service; `make dev-up` starts the stack.

---

## Evaluation criteria

| Criterion | Weight | Evidence |
|-----------|--------|---------|
| Schema correctness (constraints, FKs, indexes) | 25% | `\d tablename` output; EXPLAIN confirms index use |
| Transaction integrity (ACID, rollback) | 25% | Transfer test: concurrent insufficient-funds → rollback |
| API completeness and correctness | 20% | All endpoints return correct status codes and payloads |
| Test quality (coverage, isolation) | 20% | `pytest --cov` report; no test leaks between test cases |
| Code quality (types, structure, ruff) | 10% | `ruff check` clean; no god files; no secrets |

**Minimum to pass: all Saturday milestones + 5 of 8 Sunday milestones + tests passing.**
