# Phase 1 · Week 08 · Day 6 — Docker & Docker Compose for FastAPI

> Objective: Package a FastAPI app as a production-ready Docker image and orchestrate it with PostgreSQL and Redis via Docker Compose.  ·  Time: ~4 hr  ·  Prereqs: [[day-5]]

---

## Concept

### Containers vs Virtual Machines

A virtual machine emulates entire hardware, boots a full OS kernel, and carries gigabytes of overhead. A container shares the host kernel, isolates only processes and filesystems, and starts in milliseconds. The tradeoff: containers give you speed and density; VMs give you stronger isolation (separate kernels).

**Key insight:** a Docker image is not a running thing — it is a read-only blueprint. A container is a running instance of that blueprint.

### Image Layers

Docker images are built from layers, one per `RUN`/`COPY`/`ADD` instruction. Each layer is a diff on top of the previous. Layers are content-addressed and cached: if a layer's inputs haven't changed, Docker reuses the cached version. This means instruction order matters — put rarely-changing steps first (installing OS packages, copying `requirements.txt`) and frequently-changing steps last (copying your source code).

### Core Dockerfile Instructions

| Instruction | Purpose |
|-------------|---------|
| `FROM` | Base image to build on |
| `COPY` | Copy files from build context into image |
| `RUN` | Execute a shell command during build (creates a layer) |
| `CMD` | Default command when container starts (overridable) |
| `EXPOSE` | Document which port the app listens on |
| `ENV` | Set environment variables baked into the image |
| `ARG` | Build-time variable (not in final image) |
| `WORKDIR` | Set working directory for subsequent instructions |

### Multi-Stage Builds

A multi-stage build uses multiple `FROM` statements in one Dockerfile. Each stage is independent. You copy artifacts between stages, leaving build tools (compilers, test runners) behind. The final image ships only the runtime stage — dramatically smaller and with a smaller attack surface.

```
Stage 1 (builder):  full Python image + pip + all build deps
Stage 2 (runtime):  slim Python image + only the installed packages
```

### .dockerignore

Works like `.gitignore` — prevents files from entering the build context sent to the daemon. Excluding `.git`, `__pycache__`, `*.pyc`, local `.env`, `tests/`, and IDE configs keeps the context small and avoids cache-busting layers unnecessarily.

---

## How It Works Internally

### Linux Namespaces

Docker uses kernel namespaces to provide isolation:

- **`pid`** namespace: container processes get their own PID tree (PID 1 inside = first process)
- **`net`** namespace: each container gets its own network stack, interfaces, and routing table
- **`mnt`** namespace: separate filesystem mount tree per container
- **`uts`** namespace: separate hostname and domain name
- **`user`** namespace: map container UIDs to unprivileged host UIDs (rootless mode)

### cgroups (Control Groups)

Namespaces isolate *what a process can see*; cgroups limit *what it can use* — CPU, memory, I/O, network bandwidth. `--memory 512m --cpus 1.5` in a `docker run` command translates directly to cgroup limits.

### Overlay Filesystem

Docker's default storage driver (`overlay2`) stacks image layers using Linux's OverlayFS. Each layer is a directory of files added or modified. At runtime, Docker adds a thin read-write layer on top. Writes go to that top layer; reads fall through to whichever layer owns the file. Deleting a file in a higher layer creates a "whiteout" entry rather than removing the lower layer.

### Layer Caching in Practice

```
COPY requirements.txt .      # changes rarely → cached
RUN pip install -r requirements.txt   # only reruns when requirements.txt changes
COPY . .                     # changes often → invalidates only from here
```

Reversing these two `COPY` lines would invalidate the pip install layer on every code change.

---

## Worked Example

### Project Structure

```
my-api/
├── app/
│   ├── main.py
│   ├── config.py
│   └── routers/
│       └── items.py
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
└── .env.example
```

### `app/config.py`

```python
import os
import logging
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    database_url: str = os.getenv("DATABASE_URL", "")
    redis_url: str = os.getenv("REDIS_URL", "")
    app_env: str = os.getenv("APP_ENV", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    class Config:
        env_file = ".env"


settings = Settings()
```

### `app/main.py`

```python
import logging
import os

from fastapi import FastAPI

from app.config import settings

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(title="My API", version="1.0.0")


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok", "env": settings.app_env}
```

### `Dockerfile` — Multi-Stage Build

```dockerfile
# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies (creates a cached layer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source
COPY app/ ./app/

# Non-root user for security
RUN useradd --no-create-home --shell /bin/false appuser
USER appuser

EXPOSE 8000

# ENV vars provide defaults; real values injected at runtime via compose/.env
ENV APP_ENV=production \
    LOG_LEVEL=INFO \
    PORT=8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `docker-compose.yml`

```yaml
version: "3.9"

services:
  api:
    build:
      context: .
      target: runtime
    ports:
      - "8000:8000"
    env_file:
      - .env                          # DATABASE_URL, REDIS_URL, etc. from .env
    environment:
      - APP_ENV=production
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - app-net
    restart: unless-stopped

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    volumes:
      - pg_data:/var/lib/postgresql/data   # persistent volume
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s
    networks:
      - app-net
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5
    networks:
      - app-net
    restart: unless-stopped

volumes:
  pg_data:
  redis_data:

networks:
  app-net:
    driver: bridge
```

### `.env.example`

```
# Copy to .env and fill in real values — never commit .env
DATABASE_URL=postgresql+asyncpg://user:password@postgres:5432/mydb
REDIS_URL=redis://redis:6379/0
POSTGRES_USER=
POSTGRES_PASSWORD=
POSTGRES_DB=mydb
LOG_LEVEL=INFO
APP_ENV=production
```

### `.dockerignore`

```
.git
.gitignore
__pycache__
*.pyc
*.pyo
*.pyd
.env
.env.*
!.env.example
*.egg-info
.pytest_cache
.mypy_cache
.ruff_cache
tests/
docs/
*.md
Dockerfile*
docker-compose*
.vscode
.idea
```

### Key Networking Detail

Services on the same Docker Compose network resolve by **service name**. The FastAPI app connects to `postgres:5432` and `redis:6379` — not `localhost`. This is why `DATABASE_URL` in `.env` uses `@postgres:5432`, not `@localhost:5432`.

### Running It

```bash
# Build and start all services
docker compose up --build

# Rebuild after code change
docker compose up --build api

# Check running containers
docker compose ps

# Tail logs
docker compose logs -f api

# Stop and remove containers (volumes preserved)
docker compose down

# Also remove volumes (destructive — destroys DB data)
docker compose down -v
```

### Inspecting Image Size

```bash
# List image layers and sizes
docker image history my-api-api:latest

# Total image size
docker images my-api-api

# Compare before/after .dockerignore optimization
docker build -t my-api:no-ignore .
docker images my-api
```

---

## Exercises

> Solutions: ../solutions/week-08/day-6.md

**Exercise 1 — Multi-Stage Dockerfile**

Write a multi-stage `Dockerfile` for a FastAPI app with these requirements:
- Builder stage: `python:3.12` (full), installs `fastapi`, `uvicorn[standard]`, `asyncpg`, `pydantic-settings` into `/install`
- Runtime stage: `python:3.12-slim`, copies packages from builder, copies `app/` directory
- Creates a non-root user `appuser` and runs as that user
- Exposes port 8000
- Default `CMD` starts uvicorn on `0.0.0.0:8000`

Test: run `docker build -t ex1 . && docker run --rm -p 8000:8000 ex1` and confirm the app responds.

---

**Exercise 2 — PostgreSQL with Persistence and Health Check**

Add a `postgres` service to a `docker-compose.yml`:
- Uses `postgres:16-alpine`
- All credentials (`POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`) come from environment variables — no hardcoded values
- Mounts a named volume `pg_data` at `/var/lib/postgresql/data`
- Includes a `healthcheck` using `pg_isready` with: interval 10s, timeout 5s, retries 5, start_period 30s
- The `api` service declares `depends_on` with `condition: service_healthy`

Verify: `docker compose up`, write data to the DB, run `docker compose down`, run `docker compose up` again — data should persist.

---

**Exercise 3 — Secure Environment Variable Configuration**

Given an `app/config.py` that uses `pydantic-settings`, complete the following:

1. Create `.env.example` listing all required vars (no values, just keys with inline comments)
2. Create `.env` (gitignored) with actual values for local development
3. In `docker-compose.yml`, reference `.env` via `env_file:` — do NOT inline secret values in the compose file
4. In `app/config.py`, add a validator that raises `ValueError` if `DATABASE_URL` is empty at startup
5. Confirm the app fails fast with a clear error when `DATABASE_URL` is missing

---

**Exercise 4 — Add Redis Service and Connect FastAPI**

Extend your compose file with a `redis` service:
- Uses `redis:7-alpine`
- Enables AOF persistence (`--appendonly yes`)
- Named volume `redis_data` at `/data`
- Health check: `redis-cli ping`
- `REDIS_URL` injected into the `api` service via `.env`

In `app/main.py`, add a startup check that logs whether the Redis connection succeeded (use `redis.asyncio` — `pip install redis[asyncio]`). Use `logging`, not `print`. The app should start even if Redis is temporarily unreachable, but log a warning.

```python
import logging
import os

import redis.asyncio as aioredis
from contextlib import asynccontextmanager
from fastapi import FastAPI

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_url = os.getenv("REDIS_URL", "")
    try:
        client = aioredis.from_url(redis_url)
        await client.ping()
        logger.info("Redis connection verified")
        app.state.redis = client
    except Exception as exc:
        logger.warning("Redis not reachable at startup: %s", exc)
        app.state.redis = None
    yield
    if app.state.redis:
        await app.state.redis.aclose()
```

---

**Exercise 5 — Image Size Profiling and Optimization**

Profile and reduce your image size:

1. Build your current image, record size: `docker images <name> --format "{{.Size}}"`
2. Add a `.dockerignore` that excludes: `.git`, `__pycache__`, `*.pyc`, `.env`, `tests/`, `docs/`, IDE configs
3. Rebuild and record new size
4. In the Dockerfile, combine related `RUN` commands into a single layer using `&&` and remove apt cache in the same step:
   ```dockerfile
   RUN apt-get update && apt-get install -y --no-install-recommends libpq5 \
       && rm -rf /var/lib/apt/lists/*
   ```
5. Rebuild and compare. Document size at each step in a comment at the top of the Dockerfile.

Target: final image under 200 MB for a typical FastAPI + asyncpg app.

---

## Self-Check

1. **Why does instruction order in a Dockerfile matter for build performance?** What happens to cached layers when `COPY . .` appears before `COPY requirements.txt .`?

2. **What is the difference between `CMD` and `ENTRYPOINT`?** When would you use each?

3. **A service in Compose fails with "could not connect to postgres."** The `postgres` service is running. What are three things to check first?

4. **Why should secrets never appear in `ENV` or `ARG` instructions in a Dockerfile?** Where should they come from at runtime instead?

---

## Supplementary Resources

- Docker Get Started: https://docs.docker.com/get-started/
- Docker Compose: https://docs.docker.com/compose/
- Docker multi-stage builds: https://docs.docker.com/build/building/multi-stage/
- `pydantic-settings` docs: https://docs.pydantic.dev/latest/concepts/pydantic_settings/
