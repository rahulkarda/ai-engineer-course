# Phase 6 · Week 04 · Day 6 — Saturday Deep Practice: Containerizing a Python FastAPI + Worker App

> Objective: Containerize a multi-service application (FastAPI + Redis worker + Postgres + Redis) from scratch with optimized Dockerfiles, Compose, and a Makefile  ·  Time: ~4 hours  ·  Prereqs: [[day-5]]

## Concept

This session is a structured build. You will write every file from scratch, in order, making deliberate decisions at each step. The goal is muscle memory: by the end you should be able to containerize a Python multi-service app without referencing documentation for the common patterns.

The application is a job queue system:
- `api` service: FastAPI that accepts task submissions and queues them to Redis.
- `worker` service: reads from the Redis queue, processes jobs, logs results.
- `db` service: Postgres for persistent job state.
- `redis` service: Redis as the message queue.

## How it works internally

The architecture follows a standard producer/consumer pattern. The API is stateless and scales horizontally. The worker is stateful only in the sense that it holds a job while processing it. Postgres stores the authoritative job record. Redis is ephemeral; if it is lost, in-flight jobs are lost (acceptable for this exercise; production systems use acknowledgement patterns).

```
client --> [FastAPI :8080] --> [Redis queue]
                    |                |
                    v                v
              [Postgres]       [Worker] --> [Postgres update]
```

---

## Step 1 (45 min): Application structure

Create the following directory layout:

```
app/
  api/
    __init__.py
    main.py
    models.py
    config.py
  worker/
    __init__.py
    main.py
    config.py
tests/
  test_api.py
requirements-api.txt
requirements-worker.txt
.env.example
.gitignore
.dockerignore
```

**`app/api/config.py`**

```python
import logging
import os

from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)


class APISettings(BaseSettings):
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_queue: str = "jobs"
    db_dsn: str  # postgres+asyncpg://user:pass@host/db — injected from env
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


def get_settings() -> APISettings:
    return APISettings()
```

**`app/api/models.py`**

```python
import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    done = "done"
    failed = "failed"


class TaskRequest(BaseModel):
    payload: dict = Field(..., description="Arbitrary JSON payload for the worker")


class TaskResponse(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: JobStatus = JobStatus.queued
    created_at: datetime = Field(default_factory=datetime.utcnow)
```

**`app/api/main.py`**

```python
import json
import logging
import os

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from app.api.config import get_settings
from app.api.models import TaskRequest, TaskResponse

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

settings = get_settings()
app = FastAPI(title="Task API", version="1.0.0")

_redis: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            decode_responses=True,
        )
    return _redis


@app.on_event("shutdown")
async def shutdown() -> None:
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None
    log.info("Redis connection closed")


@app.get("/health", status_code=status.HTTP_200_OK)
async def health() -> dict[str, str]:
    try:
        r = await get_redis()
        await r.ping()
    except Exception as exc:
        log.warning("Health check failed: Redis unreachable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis unreachable",
        ) from exc
    return {"status": "ok"}


@app.post("/tasks", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_task(body: TaskRequest) -> TaskResponse:
    task = TaskResponse()
    job = {
        "job_id": task.job_id,
        "payload": body.payload,
        "created_at": task.created_at.isoformat(),
    }
    try:
        r = await get_redis()
        await r.lpush(settings.redis_queue, json.dumps(job))
    except Exception as exc:
        log.error("Failed to enqueue job %s: %s", task.job_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Queue unavailable",
        ) from exc
    log.info("Enqueued job %s", task.job_id)
    return task
```

**`app/worker/config.py`**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class WorkerSettings(BaseSettings):
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_queue: str = "jobs"
    poll_timeout: int = 5  # seconds
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


def get_settings() -> WorkerSettings:
    return WorkerSettings()
```

**`app/worker/main.py`**

```python
import asyncio
import json
import logging
import os
import time

import redis.asyncio as aioredis

from app.worker.config import get_settings

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

settings = get_settings()


async def process_job(job: dict) -> None:
    job_id = job.get("job_id", "unknown")
    payload = job.get("payload", {})
    log.info("Processing job %s payload=%s", job_id, payload)
    # Simulate work proportional to a 'duration' key in the payload
    duration = float(payload.get("duration", 1.0))
    await asyncio.sleep(min(duration, 10.0))  # cap at 10s
    log.info("Completed job %s", job_id)


async def run_worker() -> None:
    log.info(
        "Worker starting: redis=%s:%d queue=%s",
        settings.redis_host,
        settings.redis_port,
        settings.redis_queue,
    )
    r = aioredis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        decode_responses=True,
    )
    try:
        while True:
            try:
                result = await r.brpop(settings.redis_queue, timeout=settings.poll_timeout)
                if result is None:
                    continue
                _, raw = result
                job = json.loads(raw)
                await process_job(job)
            except aioredis.RedisError as exc:
                log.error("Redis error: %s — retrying in 5s", exc)
                await asyncio.sleep(5)
            except Exception as exc:
                log.exception("Unexpected error processing job: %s", exc)
    finally:
        await r.aclose()
        log.info("Worker shut down")


if __name__ == "__main__":
    asyncio.run(run_worker())
```

**`requirements-api.txt`**

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
redis[asyncio]==5.0.8
pydantic-settings==2.4.0
```

**`requirements-worker.txt`**

```
redis[asyncio]==5.0.8
pydantic-settings==2.4.0
```

**`.env.example`**

```
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_QUEUE=jobs
DB_DSN=postgresql+asyncpg://appuser@db/appdb
LOG_LEVEL=INFO
POSTGRES_USER=appuser
POSTGRES_DB=appdb
# POSTGRES_PASSWORD must be set — never commit actual value
```

**`.gitignore`** (key entries)

```
.env
secrets/
__pycache__/
*.pyc
.pytest_cache/
dist/
*.egg-info/
```

---

## Step 2 (45 min): Multi-stage Dockerfiles

**`Dockerfile.api`**

```dockerfile
# syntax=docker/dockerfile:1

# ---- build stage ----
FROM python:3.12-slim AS build

WORKDIR /build

RUN pip install --upgrade pip

COPY requirements-api.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-api.txt

# ---- runtime stage ----
FROM python:3.12-slim AS runtime

WORKDIR /app

# Non-root user
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --no-create-home --shell /sbin/nologin appuser

# Copy installed packages from build stage
COPY --from=build /install /usr/local

# Copy application source
COPY --chown=appuser:appgroup app/api ./app/api
COPY --chown=appuser:appgroup app/__init__.py ./app/__init__.py 2>/dev/null || true

USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

CMD ["python", "-m", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

**`Dockerfile.worker`**

```dockerfile
# syntax=docker/dockerfile:1

# ---- build stage ----
FROM python:3.12-slim AS build

WORKDIR /build

RUN pip install --upgrade pip

COPY requirements-worker.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-worker.txt

# ---- runtime stage ----
FROM python:3.12-slim AS runtime

WORKDIR /app

RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --no-create-home --shell /sbin/nologin appuser

COPY --from=build /install /usr/local
COPY --chown=appuser:appgroup app/worker ./app/worker
COPY --chown=appuser:appgroup app/__init__.py ./app/__init__.py 2>/dev/null || true

USER appuser

CMD ["python", "-m", "app.worker.main"]
```

Key decisions:
- Two stages: `build` installs deps into a prefix directory; `runtime` is a fresh `python:3.12-slim` with no build tools.
- The final image has no `pip`, no compilers, no test tools.
- `--prefix=/install` installs to a separate directory so it can be copied cleanly without other site-packages.

---

## Step 3 (45 min): docker-compose.yml

```yaml
# docker-compose.yml
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    env_file: .env
    ports:
      - "${API_PORT:-8080}:8080"
    depends_on:
      redis:
        condition: service_healthy
    networks:
      - backend
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    healthcheck:
      test: ["CMD", "python", "-c",
             "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 15s
    restart: unless-stopped

  worker:
    build:
      context: .
      dockerfile: Dockerfile.worker
    env_file: .env
    depends_on:
      redis:
        condition: service_healthy
    networks:
      - backend
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    restart: unless-stopped

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - db_data:/var/lib/postgresql/data
    networks:
      - backend
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 10s

  redis:
    image: redis:7-alpine
    command: redis-server --save "" --appendonly no
    networks:
      - backend
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5

volumes:
  db_data:

networks:
  backend:
    driver: bridge
```

Notes:
- `POSTGRES_PASSWORD` comes from `.env` at runtime. It is never in the image.
- `redis-server --save "" --appendonly no` disables persistence (queue is ephemeral by design).
- Named volume `db_data` survives `docker compose down` (data not lost unless `down -v`).
- All services share the `backend` bridge network; no service is on the default network.

---

## Step 4 (30 min): Makefile

```makefile
.PHONY: build up down logs shell test clean

IMAGE_API   := taskapi
IMAGE_WORKER := taskworker

build:
	docker compose build --no-cache

up:
	docker compose up -d
	@echo "Waiting for api to be healthy..."
	@until docker compose ps api | grep -q "healthy"; do sleep 2; done
	@echo "Stack is up"

down:
	docker compose down

down-v:
	docker compose down -v

logs:
	docker compose logs -f

logs-api:
	docker compose logs -f api

logs-worker:
	docker compose logs -f worker

shell-api:
	docker compose exec api /bin/sh

shell-worker:
	docker compose exec worker /bin/sh

test:
	docker compose run --rm api python -m pytest tests/ -v

clean:
	docker compose down -v --rmi local
	docker image prune -f
```

---

## Step 5 (30 min): Test the full stack

Start the stack:

```bash
make build
make up
```

Verify services are healthy:

```bash
docker compose ps
```

All services should show `healthy` or `running`.

Submit a task:

```bash
curl -s -X POST http://localhost:8080/tasks \
  -H "Content-Type: application/json" \
  -d '{"payload": {"duration": 2, "task_name": "test-job"}}' | python -m json.tool
```

Expected response:

```json
{
  "job_id": "<uuid>",
  "status": "queued",
  "created_at": "<timestamp>"
}
```

Watch the worker process it:

```bash
make logs-worker
```

You should see: `Processing job <uuid>` followed by `Completed job <uuid>`.

Check the health endpoint directly:

```bash
curl -s http://localhost:8080/health
```

Inspect container health state:

```bash
docker inspect --format='{{json .State.Health.Status}}' $(docker compose ps -q api)
```

---

## Step 6 (45 min): Harden

**`.dockerignore`**

```
.git
.env
secrets/
*.pyc
__pycache__/
.pytest_cache/
tests/
*.md
.dockerignore
Makefile
```

`tests/` is excluded from the runtime image. `.env` is excluded so it can never accidentally be `COPY .`-ed in.

**Add `__init__.py` files** if missing so Python resolves the package correctly inside the container:

```bash
touch app/__init__.py app/api/__init__.py app/worker/__init__.py
```

**Verify non-root user is active:**

```bash
docker compose exec api whoami
# should print: appuser
```

**Verify no secrets in image history:**

```bash
docker image history taskapi --no-trunc | grep -i password
# should return nothing
```

**Add read-only filesystem to the API (it has no local writes):**

```yaml
# in docker-compose.yml under api:
read_only: true
tmpfs:
  - /tmp
```

Restart and confirm health still passes.

---

## Exercises

1. The current worker has no retry logic for failed jobs. Design a dead-letter queue pattern: after N failures, push the job to a `jobs:dead` Redis key instead of discarding it. Implement it in `app/worker/main.py` without changing the API or Compose file. Write a test that verifies the dead-letter behavior by mocking `process_job` to raise an exception.

2. Add a `GET /tasks/{job_id}` endpoint to the API that retrieves job status from Postgres. You will need to: add `asyncpg` and `sqlalchemy[asyncio]` to `requirements-api.txt`, define a `jobs` table, write the job record on enqueue (status=queued), and update it in the worker on completion. Use `pydantic-settings` for the DB connection string. No hardcoded credentials anywhere.

3. The Compose file does not limit container resources. Add `deploy.resources.limits` (memory and CPU) for the `api` and `worker` services. Then run `docker stats` while submitting 20 concurrent tasks (use `xargs` or a small Python script) and observe whether the limits are hit. Tune the limits so the API does not OOM under the load.

(Solutions: ../solutions/week-04/day-6.md)

## Self-check

- Why does a multi-stage build reduce the attack surface of the final image?
- What is the purpose of `condition: service_healthy` in `depends_on`, and what happens if it is omitted?
- Why is `POSTGRES_PASSWORD` in the Compose `environment` block (from `.env`) different from hardcoding it in the Dockerfile?
- What does `cap_drop: ALL` do, and under what circumstances would you need to add back a capability?
- What would happen to in-flight jobs if the Redis container is restarted with `docker compose restart redis`?
