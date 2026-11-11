# Week 05 Project — Transaction Library: Tests, Logging & CI

> Effort: ~4–5 hr total (split across Day 6 lab + this document)  ·  Prereqs: Week 4 project (transaction library)

## Goal

Transform the Week 4 transaction library into a production-quality package with a full test suite, structured logging, custom exceptions, environment-based configuration, and a CI pipeline. This is the pattern you will repeat for every service you build going forward.

## Context

The Week 4 transaction library is a working Python module that processes financial transactions. It may already handle deposits, withdrawals, transfers, and balance queries. It likely lacks tests, uses `print` for output, hardcodes config, and has no exception hierarchy.

By the end of this project the library will be CI-green, linted, and safe to merge to `main` via a pull request.

## Required deliverables

### 1. Exception hierarchy (`src/exceptions.py`)

```
AppError
├── InvalidTransactionError    — business rule violations
├── AccountNotFoundError       — missing account
├── InsufficientFundsError     — balance too low
├── DuplicateTransactionError  — replay attack / idempotency
└── ConfigurationError         — misconfigured settings at startup
```

Every `raise` in the codebase must use one of these types (or a new subclass you add with justification). No bare `raise Exception(...)`.

### 2. Configuration (`src/config.py`)

A `Settings` class using pydantic-settings that exposes:

| Setting | Env var | Default | Type |
|---------|---------|---------|------|
| Max single transaction | `MAX_TRANSACTION_AMOUNT` | 50000 | `int` |
| Default currency | `DEFAULT_CURRENCY` | `GBP` | `str` |
| Transaction fee (%) | `TRANSACTION_FEE_PCT` | 0.0 | `float` |
| Log level | `LOG_LEVEL` | `INFO` | `str` |
| External API key | `APP_API_KEY` | `` | `str` |

`.env.example` committed; `.env` git-ignored.

### 3. Structured logging

Every public method logs:
- Entry: DEBUG level, all parameters.
- Successful completion: INFO level, key result.
- Validation rejection: WARNING level, rejected value.
- Unexpected error: ERROR level, with `exc_info=True`.

No `print()` calls anywhere in `src/`.

### 4. Test suite

Structure:
```
tests/
├── conftest.py                       — shared fixtures
├── unit/
│   ├── test_transactions.py          — happy-path unit tests
│   ├── test_invalid_inputs.py        — all error-path tests
│   └── test_mocked.py                — tests with mocked external calls
└── integration/                      — (optional bonus)
    └── test_persistence.py
```

Minimum requirements:
- ≥ 15 unit tests across the three files.
- Every public function tested.
- All invalid input paths tested with correct exception type and message.
- All external I/O calls mocked (file, HTTP, random, datetime).
- At least one parametrized test with ≥ 4 parameter sets.
- Coverage ≥ 80% (enforced by CI).

### 5. GitHub Actions CI (`/.github/workflows/ci.yml`)

Pipeline runs on every push to `main` and every PR:

1. `ruff check src/ tests/` — lint
2. `ruff format --check src/ tests/` — format check
3. `pytest tests/unit/ --cov=src --cov-fail-under=80` — tests + coverage gate

Pipeline must be green before any PR can merge.

## Project layout after completion

```
transaction-library/
├── .env.example
├── .gitignore               ← .env listed here
├── .github/
│   └── workflows/
│       └── ci.yml
├── pyproject.toml           ← dev deps: pytest, pytest-cov, ruff, pydantic-settings
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── exceptions.py
│   ├── transactions.py      ← (or your Week 4 module names)
│   └── notifications.py     ← (stub for external calls)
└── tests/
    ├── conftest.py
    └── unit/
        ├── test_transactions.py
        ├── test_invalid_inputs.py
        └── test_mocked.py
```

## Step-by-step guidance

### Step 1 — Git branching strategy

Before touching any code:
```bash
git checkout -b feat/week-05-quality
```

All work goes on this branch. You will open a PR to merge into `main` at the end.

### Step 2 — Custom exceptions first

Create `src/exceptions.py`. Update imports across the codebase: search for `raise ValueError` / `raise Exception` and replace with the appropriate custom type.

Commit: `refactor(exceptions): introduce custom exception hierarchy`

### Step 3 — Config

Create `src/config.py`, `.env.example`. Add `.env` to `.gitignore`. Replace any hardcoded values in the codebase with `settings.<field>`.

Commit: `feat(config): centralise config with pydantic-settings`

### Step 4 — Logging

Replace `print` with `logging`. Add `configure_logging()` call in `main.py` or wherever the app starts. Use `LOG_LEVEL` from settings.

Commit: `feat(logging): replace print with structured logging`

### Step 5 — Tests

Write all three test files. Run `pytest -v` after each file to stay green.

Commit: `test: add unit test suite with mocks and coverage`

### Step 6 — CI

Create `.github/workflows/ci.yml`. Push the branch. Verify the Actions tab shows a green pipeline.

Commit: `ci: add GitHub Actions pipeline with lint and coverage gate`

### Step 7 — PR

Open a PR: base `main` ← compare `feat/week-05-quality`.

PR description must include:
- Summary of what changed and why.
- How to run the tests locally.
- Screenshot or paste of the green CI run.

## Evaluation criteria

| Criteria | Pass |
|----------|------|
| All public functions have unit tests | ≥ 1 test per function |
| Invalid inputs tested | Every error branch has a test |
| External calls mocked | No real I/O in unit tests |
| Custom exception hierarchy | All raises use custom types |
| Logging | No `print` in `src/`; all state changes logged |
| Config | pydantic-settings, `.env` ignored, `.env.example` committed |
| CI green | Actions pipeline passes on the PR branch |
| Coverage | ≥ 80% reported by pytest-cov |
| ruff clean | Zero lint errors, formatted |
| PR opened | Branch with passing CI; not merged |

## Bonus stretch goals

- Add a `pytest-mock` integration and rewrite one mock test using `mocker.patch` (the pytest-mock style) to compare the two approaches.
- Implement `pytest-xdist` parallel execution (`pytest -n auto`) and verify tests pass concurrently.
- Add an integration test that persists and reloads transactions from a temp SQLite DB (use `pytest`'s `tmp_path` fixture for the DB path).
- Add a `CHANGELOG.md` entry describing the changes made this week, following the [Keep a Changelog](https://keepachangelog.com/) format.
