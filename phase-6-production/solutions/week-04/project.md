# Reference Solution — Week 04 Project: Production-Ready Multi-Service Containerization

## 1. Project Directory Structure

```
week-04-project/
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.worker
├── Makefile
├── pyproject.toml
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py
│   └── worker/
│       ├── __init__.py
│       └── main.py
└── tests/
    ├── __init__.py
    └── test_api.py
```

---

## 2. src/api/main.py

```python
from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import redis.asyncio as aioredis
import asyncpg
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    redis_url: str = Field(..., alias="REDIS_URL")
    database_url: str = Field(..., alias="DATABASE_URL")
    app_version: str = Field("0.0.0", alias="APP_VERSION")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    port: int = Field(8000, alias="PORT")


settings = Settings()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TaskRequest(BaseModel):
    task_type: str = Field(..., min_length=1, max_length=128)
    payload: dict[str, Any] = Field(default_factory=dict)


class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    task_type: str | None = None
    result: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

app = FastAPI(title="Task API", version=settings.app_version)

_redis: aioredis.Redis | None = None
_pg_pool: asyncpg.Pool | None = None


@app.on_event("startup")
async def startup() -> None:
    global _redis, _pg_pool

    logger.info("Connecting to Redis at %s", settings.redis_url)
    _redis = aioredis.from_url(settings.redis_url, decode_responses=True)

    logger.info("Connecting to Postgres")
    _pg_pool = await asyncpg.create_pool(settings.database_url, min_size=2, max_size=10)

    async with _pg_pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id    TEXT PRIMARY KEY,
                task_type  TEXT NOT NULL,
                payload    JSONB NOT NULL DEFAULT '{}',
                status     TEXT NOT NULL DEFAULT 'queued',
                result     TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
    logger.info("Startup complete")


@app.on_event("shutdown")
async def shutdown() -> None:
    if _redis:
        await _redis.aclose()
    if _pg_pool:
        await _pg_pool.close()
    logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=settings.app_version)


@app.post("/tasks", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_task(request: TaskRequest) -> TaskResponse:
    if _redis is None or _pg_pool is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    task_id = str(uuid.uuid4())
    job = {"task_id": task_id, "task_type": request.task_type, "payload": request.payload}

    async with _pg_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO tasks (task_id, task_type, payload, status) VALUES ($1, $2, $3, $4)",
            task_id,
            request.task_type,
            json.dumps(request.payload),
            "queued",
        )

    await _redis.lpush("task_queue", json.dumps(job))
    logger.info("Queued task %s type=%s", task_id, request.task_type)

    return TaskResponse(task_id=task_id, status="queued", message="Task accepted")


@app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    if _pg_pool is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    async with _pg_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT task_id, task_type, status, result, created_at, updated_at FROM tasks WHERE task_id = $1",
            task_id,
        )

    if row is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return TaskStatusResponse(
        task_id=row["task_id"],
        status=row["status"],
        task_type=row["task_type"],
        result=row["result"],
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )
```

---

## 3. src/worker/main.py

```python
from __future__ import annotations

import json
import logging
import os
import signal
import time
from pathlib import Path
from typing import Any

import asyncpg
import redis
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    redis_url: str = Field(..., alias="REDIS_URL")
    database_url: str = Field(..., alias="DATABASE_URL")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    heartbeat_path: Path = Field(Path("/tmp/worker.heartbeat"), alias="HEARTBEAT_PATH")
    queue_name: str = Field("task_queue", alias="QUEUE_NAME")
    brpop_timeout: int = Field(5, alias="BRPOP_TIMEOUT")


settings = Settings()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_running = True


def _handle_sigterm(signum: int, frame: Any) -> None:
    global _running
    logger.info("SIGTERM received, draining current job then exiting")
    _running = False


signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigterm)

# ---------------------------------------------------------------------------
# Task processing
# ---------------------------------------------------------------------------

def _process_job(job: dict[str, Any]) -> str:
    """
    Placeholder processing logic. Replace with real business logic.
    Returns a result string.
    """
    task_type = job.get("task_type", "unknown")
    payload = job.get("payload", {})
    logger.info("Processing task_type=%s payload_keys=%s", task_type, list(payload.keys()))
    # Simulate work
    time.sleep(0.1)
    return f"Completed task_type={task_type}"


def _update_heartbeat() -> None:
    settings.heartbeat_path.write_text(str(time.time()))


def run() -> None:
    logger.info("Worker starting, connecting to Redis and Postgres")

    r = redis.from_url(settings.redis_url, decode_responses=True)

    import asyncio

    async def _get_pool() -> asyncpg.Pool:
        return await asyncpg.create_pool(settings.database_url, min_size=1, max_size=5)

    pool: asyncpg.Pool = asyncio.get_event_loop().run_until_complete(_get_pool())

    async def _set_status(
        task_id: str,
        status: str,
        result: str | None = None,
    ) -> None:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE tasks
                SET status = $1, result = $2, updated_at = NOW()
                WHERE task_id = $3
                """,
                status,
                result,
                task_id,
            )

    logger.info("Worker ready, listening on queue=%s", settings.queue_name)
    _update_heartbeat()

    while _running:
        _update_heartbeat()
        raw = r.brpop(settings.queue_name, timeout=settings.brpop_timeout)
        if raw is None:
            # timeout — loop back and check _running
            continue

        _, message = raw
        try:
            job: dict[str, Any] = json.loads(message)
        except json.JSONDecodeError:
            logger.error("Invalid JSON on queue, skipping: %s", message[:200])
            continue

        task_id: str = job.get("task_id", "")
        if not task_id:
            logger.error("Job missing task_id, skipping")
            continue

        logger.info("Starting task %s", task_id)
        asyncio.get_event_loop().run_until_complete(
            _set_status(task_id, "processing")
        )

        try:
            result = _process_job(job)
            asyncio.get_event_loop().run_until_complete(
                _set_status(task_id, "completed", result)
            )
            logger.info("Completed task %s", task_id)
        except Exception as exc:
            logger.exception("Task %s failed: %s", task_id, exc)
            asyncio.get_event_loop().run_until_complete(
                _set_status(task_id, "failed", str(exc))
            )

    logger.info("Worker exiting cleanly")
    asyncio.get_event_loop().run_until_complete(pool.close())
    r.close()


if __name__ == "__main__":
    run()
```

---

## 4. Dockerfile.api

```dockerfile
# ============================================================
# Stage 1 — builder: install Python dependencies
# ============================================================
FROM python:3.12 AS builder

WORKDIR /build

COPY pyproject.toml ./
# Install dependencies into an isolated prefix so the slim stage can copy them
RUN pip install --no-cache-dir --prefix=/install .

# ============================================================
# Stage 2 — runtime: minimal image
# ============================================================
FROM python:3.12-slim AS runtime

ARG APP_VERSION=0.0.0
ENV APP_VERSION=${APP_VERSION} \
    PORT=8000 \
    LOG_LEVEL=INFO \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Non-root user
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app
COPY src/ ./src/

RUN chown -R appuser:appgroup /app
USER appuser

EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')"

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT} --log-level ${LOG_LEVEL}"]
```

---

## 5. Dockerfile.worker

```dockerfile
# ============================================================
# Stage 1 — builder
# ============================================================
FROM python:3.12 AS builder

WORKDIR /build

COPY pyproject.toml ./
RUN pip install --no-cache-dir --prefix=/install .

# ============================================================
# Stage 2 — runtime
# ============================================================
FROM python:3.12-slim AS runtime

ENV LOG_LEVEL=INFO \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=builder /install /usr/local

RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app
COPY src/ ./src/

RUN chown -R appuser:appgroup /app
USER appuser

# Worker writes a heartbeat file; health check verifies it is recent (< 60 s old)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD test -f /tmp/worker.heartbeat

CMD ["python", "-m", "src.worker.main"]
```

---

## 6. docker-compose.yml

```yaml
version: "3.9"

services:

  # ------------------------------------------------------------------
  # PostgreSQL
  # ------------------------------------------------------------------
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: ${DB_NAME}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - app_network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER} -d ${DB_NAME}"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 10s
    restart: unless-stopped

  # ------------------------------------------------------------------
  # Redis
  # ------------------------------------------------------------------
  redis:
    image: redis:7-alpine
    command: >
      redis-server
      --requirepass ${REDIS_PASSWORD}
      --appendonly yes
    volumes:
      - redis_data:/data
    networks:
      - app_network
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "${REDIS_PASSWORD}", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 5s
    restart: unless-stopped

  # ------------------------------------------------------------------
  # API
  # ------------------------------------------------------------------
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
      args:
        APP_VERSION: ${APP_VERSION}
    env_file:
      - .env
    ports:
      - "${PORT:-8000}:${PORT:-8000}"
    networks:
      - app_network
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-8000}/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 15s
    restart: unless-stopped

  # ------------------------------------------------------------------
  # Worker
  # ------------------------------------------------------------------
  worker:
    build:
      context: .
      dockerfile: Dockerfile.worker
    env_file:
      - .env
    networks:
      - app_network
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "test", "-f", "/tmp/worker.heartbeat"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 20s
    restart: unless-stopped

# ------------------------------------------------------------------
# Volumes and network
# ------------------------------------------------------------------
volumes:
  postgres_data:
  redis_data:

networks:
  app_network:
    driver: bridge
```

---

## 7. .env.example

```dotenv
# ---------------------------------------------------------------
# Database (Postgres)
# ---------------------------------------------------------------
# DB_USER: Postgres username
DB_USER=app_user

# DB_PASSWORD: Postgres password — NEVER commit a real value
DB_PASSWORD=

# DB_NAME: Database name
DB_NAME=tasks_db

# ---------------------------------------------------------------
# Redis
# ---------------------------------------------------------------
# REDIS_PASSWORD: Redis AUTH password — NEVER commit a real value
REDIS_PASSWORD=

# ---------------------------------------------------------------
# Connection URLs (constructed from above values by convention)
# ---------------------------------------------------------------
# DATABASE_URL format: postgresql://USER:PASSWORD@HOST:PORT/DBNAME
DATABASE_URL=postgresql://${DB_USER}:${DB_PASSWORD}@postgres:5432/${DB_NAME}

# REDIS_URL format: redis://:PASSWORD@HOST:PORT/0
REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0

# ---------------------------------------------------------------
# Application
# ---------------------------------------------------------------
APP_VERSION=1.0.0
LOG_LEVEL=INFO
PORT=8000
```

---

## 8. Makefile

```makefile
.DEFAULT_GOAL := help

.PHONY: help build up down logs shell-api shell-worker test clean ps

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

build: ## Build all Docker images
	docker compose build

up: ## Start all services in the background
	docker compose up -d

down: ## Stop all services (keep volumes)
	docker compose down

logs: ## Tail logs for all services
	docker compose logs -f

shell-api: ## Open a bash shell in the API container
	docker compose exec api bash

shell-worker: ## Open a bash shell in the worker container
	docker compose exec worker bash

test: ## Run pytest inside the API container
	docker compose run --rm api python -m pytest /app/tests/ -v

clean: ## Stop services, remove volumes and local images
	docker compose down -v --rmi local

ps: ## Show running containers and their health status
	docker compose ps
```

---

## 9. tests/test_api.py

```python
from __future__ import annotations

import json
import logging
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# App factory — import after patching external clients
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_redis() -> AsyncMock:
    r = AsyncMock()
    r.lpush = AsyncMock(return_value=1)
    return r


@pytest.fixture()
def mock_pg_pool() -> AsyncMock:
    pool = AsyncMock()
    conn = AsyncMock()
    conn.execute = AsyncMock()
    pool.acquire = MagicMock(return_value=_AsyncContextManager(conn))
    return pool, conn


class _AsyncContextManager:
    """Minimal async context manager wrapping a mock connection."""

    def __init__(self, obj: Any) -> None:
        self._obj = obj

    async def __aenter__(self) -> Any:
        return self._obj

    async def __aexit__(self, *args: Any) -> None:
        pass


@pytest.fixture()
def client(mock_redis: AsyncMock, mock_pg_pool: tuple[AsyncMock, AsyncMock]) -> TestClient:
    pool, _conn = mock_pg_pool

    with (
        patch("src.api.main.aioredis.from_url", return_value=mock_redis),
        patch("src.api.main.asyncpg.create_pool", new_callable=AsyncMock, return_value=pool),
    ):
        # Import here so patches are applied before module-level code runs
        from src.api.main import app  # noqa: PLC0415

        # Inject mocks into module globals
        import src.api.main as api_module  # noqa: PLC0415

        api_module._redis = mock_redis
        api_module._pg_pool = pool

        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_health_endpoint(client: TestClient) -> None:
    """GET /health returns status ok and a version string."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body
    logger.info("health response: %s", body)


def test_create_task_returns_task_id(
    client: TestClient,
    mock_redis: AsyncMock,
) -> None:
    """POST /tasks returns 202 with a task_id and queues to Redis."""
    payload = {"task_type": "summarise", "payload": {"text": "hello world"}}
    response = client.post("/tasks", json=payload)

    assert response.status_code == 202
    body = response.json()
    assert "task_id" in body
    assert body["status"] == "queued"
    # Verify Redis lpush was called once
    mock_redis.lpush.assert_awaited_once()
    call_args = mock_redis.lpush.call_args
    queue_name: str = call_args[0][0]
    assert queue_name == "task_queue"
    queued_job: dict[str, Any] = json.loads(call_args[0][1])
    assert queued_job["task_type"] == "summarise"
    logger.info("create_task response: %s", body)


def test_get_task_status_found(
    client: TestClient,
    mock_pg_pool: tuple[AsyncMock, AsyncMock],
) -> None:
    """GET /tasks/{task_id} returns task details when task exists in Postgres."""
    _pool, conn = mock_pg_pool
    task_id = str(uuid.uuid4())
    fake_row = {
        "task_id": task_id,
        "task_type": "summarise",
        "status": "completed",
        "result": "done",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:01:00+00:00",
    }
    conn.fetchrow = AsyncMock(return_value=fake_row)

    response = client.get(f"/tasks/{task_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["task_id"] == task_id
    assert body["status"] == "completed"
    assert body["task_type"] == "summarise"
    logger.info("get_task_status response: %s", body)


def test_get_task_status_not_found(
    client: TestClient,
    mock_pg_pool: tuple[AsyncMock, AsyncMock],
) -> None:
    """GET /tasks/{task_id} returns 404 when task does not exist."""
    _pool, conn = mock_pg_pool
    conn.fetchrow = AsyncMock(return_value=None)

    response = client.get(f"/tasks/{uuid.uuid4()}")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
```

---

## 10. pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "week04-project"
version = "1.0.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.111",
    "uvicorn[standard]>=0.29",
    "pydantic>=2.7",
    "pydantic-settings>=2.2",
    "redis>=5.0",
    "asyncpg>=0.29",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.2",
    "pytest-asyncio>=0.23",
    "httpx>=0.27",
    "anyio>=4.4",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

---

## 11. .gitignore

```
.env
__pycache__/
*.pyc
*.pyo
.pytest_cache/
dist/
*.egg-info/
.mypy_cache/
.ruff_cache/
```

---

## 12. Grading Notes

### What each deliverable demonstrates

**src/api/main.py**
- Pydantic models for all request/response boundaries — no untyped dicts across service edges.
- `pydantic-settings` reads env vars at startup; no `os.getenv` literals scattered in handlers.
- Startup/shutdown lifecycle events wire connections once and share them via module-level singletons (`_redis`, `_pg_pool`). Avoids opening a new connection per request.
- `uuid.uuid4()` for task IDs — collision-resistant, no DB sequence dependency.
- `logging` throughout; `print` never appears.
- HTTP 202 for task creation (accepted, not yet processed) — semantically correct.

**src/worker/main.py**
- `SIGTERM` handler sets `_running = False`; the BRPOP loop exits cleanly after finishing the current job. Kubernetes `preStop` hooks rely on this.
- `BRPOP` with a timeout (not `LPOP` in a busy-wait loop) — avoids wasting CPU when the queue is empty.
- Heartbeat file written every iteration — the Docker HEALTHCHECK can use `test -f` to detect a hung worker.
- Status transitions: `queued -> processing -> completed|failed`. Any crash leaves the task in `processing`, making stalled jobs detectable.

**Dockerfile.api / Dockerfile.worker (multi-stage)**
- Builder stage installs into `/install` with `--prefix`; the slim runtime stage only copies the built artifacts. Dramatically reduces final image size.
- Non-root user (`appuser`) prevents privilege escalation if the process is compromised.
- `HEALTHCHECK` uses Python's stdlib `urllib.request` — no extra tooling (`curl`, `wget`) needed in the slim image.
- `ARG APP_VERSION` at build time, `ENV APP_VERSION` at runtime — the version is baked in but overridable.

**docker-compose.yml**
- All secrets via `${VAR}` from `.env`. No literal passwords anywhere in the file.
- `depends_on` with `condition: service_healthy` — API and worker do not start until Postgres and Redis pass their health checks. Eliminates startup race conditions.
- Named volumes (`postgres_data`, `redis_data`) survive `docker compose down`. Data is not lost on container restart.
- Custom bridge network (`app_network`) isolates the stack from other containers on the host.

**tests/test_api.py**
- `TestClient` (sync) with `unittest.mock.AsyncMock` patches on `aioredis` and `asyncpg` — no live services needed in CI.
- Four tests cover the happy path for each endpoint plus the 404 case.
- `mock_redis.lpush.assert_awaited_once()` verifies the side-effect, not just the response body.

### Common mistakes to avoid

| Mistake | Correct approach |
|---|---|
| Hardcoding `password=changeme` in compose | Use `${DB_PASSWORD}` from `.env` |
| `FROM python:3.12` for the final image | Use `python:3.12-slim` — saves ~600 MB |
| `print()` for observability | `logging.getLogger(__name__)` |
| `depends_on: [postgres]` without health condition | `condition: service_healthy` or add a retry loop in the app |
| `HEALTHCHECK` with `curl` in slim image | Python one-liner via `urllib.request` |
| `git add .` staging `.env` | `.gitignore` + explicit staging |
| Missing `SIGTERM` handler in worker | Pod eviction kills jobs silently without graceful drain |
| Untyped `dict` for task payloads | `dict[str, Any]` minimum; Pydantic model preferred |
| Single-stage Dockerfile | Multi-stage keeps secrets and build tools out of runtime layer |
| Running as root in container | `adduser --system` + `USER appuser` |
