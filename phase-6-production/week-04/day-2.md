# Phase 6 · Week 04 · Day 2 — Dockerfile Best Practices

> Objective: Write production-quality Dockerfiles that are cache-efficient, secret-free, and minimal.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

## Concept

A Dockerfile is a deterministic recipe for building an image layer by layer. Every instruction (`RUN`, `COPY`, `ADD`) creates a new layer. The choices you make — instruction order, exec vs shell form, ARG vs ENV — determine cache hit rates, image size, security posture, and startup correctness.

**Key instructions**

| Instruction | Purpose |
|---|---|
| `FROM` | Set the base image. Always pin a digest or tag. |
| `RUN` | Execute a shell command and commit the result as a layer. |
| `COPY` | Copy files from build context into the image (preferred over ADD). |
| `ADD` | Like COPY but also handles URLs and auto-extracts archives. Use only when you need those features. |
| `ENV` | Set an environment variable available at build time and in the running container. |
| `ARG` | Define a build-time variable (`--build-arg`). Not available at runtime. |
| `WORKDIR` | Set the working directory; creates it if absent. Prefer over `RUN mkdir && cd`. |
| `EXPOSE` | Document which port the container listens on. Does not publish the port. |
| `CMD` | Default command for `docker run`. Overridable by the caller. |
| `ENTRYPOINT` | Fixed executable; `CMD` becomes its default arguments. |

**Shell vs exec form**

```dockerfile
# Shell form — spawns /bin/sh -c "...", environment variable expansion works
CMD python app.py

# Exec form — no shell, direct exec(), receives signals correctly
CMD ["python", "app.py"]
```

Prefer exec form for `CMD` and `ENTRYPOINT`. Shell form wraps in `/bin/sh -c`, so PID 1 is the shell, not your process. Signals (`SIGTERM` on `docker stop`) reach the shell, not your app — graceful shutdown breaks.

## How it works internally

**Cache invalidation**

Docker caches each layer keyed by the instruction text and the content of any files it touches. Rules:

1. If instruction N changes → all layers N and later are rebuilt.
2. `COPY`/`ADD` invalidate if the file content (checksum) changes.
3. `RUN` invalidates only when the instruction text changes — not when a remote package has a new version (use `--no-cache` or pin versions to force a rebuild).

This means: **put instructions that change least often at the top**. The canonical order for Python apps:

```
FROM                        (rarely changes)
system deps via apt         (changes infrequently)
COPY requirements.txt       (changes per dependency bump)
RUN pip install             (invalidated only by req change)
COPY source code            (changes every commit)
```

If you `COPY . .` before `pip install`, every code change invalidates the pip layer and reinstalls all packages.

**ARG vs ENV**

- `ARG` is build-time only. Values are NOT visible in the running container or in `docker inspect` environment lists (but they do appear in `docker history` — never put secrets in ARG).
- `ENV` is baked into the image and visible to every process at runtime. Use for runtime config defaults.
- To avoid secrets in history: use `--secret` mount (`RUN --mount=type=secret,id=pip_token ...`). Covered in Day 3.

**.dockerignore**

Without `.dockerignore`, `COPY . .` sends the entire build context (including `.git`, `__pycache__`, `.env`, `venv/`) to the Docker daemon. This is slow and risks leaking secrets. `.dockerignore` uses the same glob syntax as `.gitignore`.

## Worked example

Project structure:

```
fastapi-app/
  app/
    main.py
    models.py
    config.py
  requirements.txt
  Dockerfile
  .dockerignore
```

`.dockerignore`:

```
# Version control
.git
.gitignore

# Python artifacts
__pycache__
*.pyc
*.pyo
*.pyd
.Python
*.egg-info
dist/
build/

# Virtual environments
venv/
.venv/
env/

# Local secrets — never ship these
.env
*.env
secrets/

# Editor / OS noise
.DS_Store
.idea/
.vscode/
*.swp
```

`Dockerfile`:

```dockerfile
# syntax=docker/dockerfile:1

# --- base: pin the exact tag, never use :latest ---
FROM python:3.12-slim AS base

# Build-time variable for the port (no secret, safe to use ARG here)
ARG APP_PORT=8000

# Runtime environment variables — no defaults that look like real values
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_PORT=${APP_PORT}

WORKDIR /app

# --- deps: install system dependencies in a single RUN to minimise layers ---
# Clean up apt cache in the same layer to avoid carrying it into the image
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# --- Python deps: copy only requirements first for cache efficiency ---
# This layer is rebuilt only when requirements.txt changes
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# --- source: copy application code last (changes most frequently) ---
COPY app/ ./app/

# Document the port (informational; actual publishing is in docker run / compose)
EXPOSE ${APP_PORT}

# Non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
USER appuser

# Exec form: process receives signals directly, no shell wrapper
ENTRYPOINT ["python", "-m", "uvicorn", "app.main:app"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
```

`app/config.py` (runtime config via environment, never hardcoded):

```python
import logging
import os
from pydantic_settings import BaseSettings

log = logging.getLogger(__name__)


class Settings(BaseSettings):
    app_env: str = "production"
    database_url: str  # required — no default, must be set via env
    secret_key: str    # required

    class Config:
        env_file = ".env"


settings = Settings()
log.info("App starting in environment: %s", settings.app_env)
```

Build and run:

```bash
# Build — pass the port as a build arg
docker build --build-arg APP_PORT=8080 -t fastapi-app:latest .

# Run — pass secrets at runtime via environment, never in Dockerfile
docker run \
  -p 8080:8000 \
  -e DATABASE_URL="${DATABASE_URL}" \
  -e SECRET_KEY="${SECRET_KEY}" \
  fastapi-app:latest

# Verify the cache: change only app/main.py, rebuild, observe pip layer is cached
touch app/main.py
docker build -t fastapi-app:latest .
# Expected: CACHED for the pip install step
```

## Exercises

1. Take the Dockerfile above and intentionally move `COPY app/ ./app/` to before `COPY requirements.txt ./`. Rebuild after making a trivial change to `app/main.py`. Measure how many layers are rebuilt vs the original ordering. Write a one-paragraph explanation of why the ordering matters for a team committing code 20 times per day.

2. The Dockerfile above uses `CMD ["--host", "0.0.0.0", "--port", "8000"]`. Explain what happens when a user runs `docker run fastapi-app:latest --workers 4`. Then rewrite the `ENTRYPOINT`/`CMD` pair so the user can override only the uvicorn arguments without replacing the entire command, and demonstrate the correct `docker run` invocation.

3. Create a `.dockerignore` file for a project that has: a `data/` directory with large CSV files, a `notebooks/` directory, a `.env` file, a `tests/` directory that should NOT be excluded (it contains fixtures used at runtime), and standard Python build artifacts. Verify your file is correct by running `docker build` and confirming the excluded directories are absent inside the image using `docker run --rm <image> find / -name "*.csv" 2>/dev/null`.

(Solutions: ../solutions/week-04/day-2.md)

## Self-check

- [ ] I can explain why exec form is preferred over shell form for ENTRYPOINT and CMD.
- [ ] I can explain the Dockerfile instruction ordering rule for maximising cache hits and apply it to a new project.
- [ ] I can write a .dockerignore that excludes secrets, build artifacts, and large data files without excluding test fixtures.
- [ ] I can implement runtime configuration using pydantic-settings and environment variables with no secrets in the Dockerfile.
