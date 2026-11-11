# Phase 6 · Week 04 · Day 5 — Secrets, Config, Health Checks, and Container Security

> Objective: Harden a container by externalizing secrets, adding health checks, and running as a non-root user  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

## Concept

Production containers must satisfy three security invariants:

1. No secrets baked into the image layer history.
2. The process inside the container runs with the least privilege necessary.
3. The orchestrator can determine liveness and readiness without manual inspection.

These are not optional hardening steps; they are the baseline for any container that will be deployed outside a developer laptop.

## How it works internally

### Container secrets — three patterns

**Environment variables (simple, 12-factor)**

Environment variables are injected at runtime by the container runtime or orchestrator. They never appear in the image layer. The application reads them with `os.getenv()`. This is the correct default for most workloads.

Limitation: the value is visible to any process in the container and to anyone who can run `docker inspect`. Use this for non-critical config (ports, feature flags, log levels) and for secrets in environments where the secret injection layer itself (Kubernetes Secrets, AWS Secrets Manager, Vault Agent) is already secured.

**Docker Swarm secrets (files mounted at `/run/secrets/<name>`)**

Docker Swarm mounts secrets as in-memory tmpfs files. The application reads the file:

```python
import pathlib, logging

log = logging.getLogger(__name__)

def read_secret(name: str) -> str:
    path = pathlib.Path(f"/run/secrets/{name}")
    if not path.exists():
        raise RuntimeError(f"Secret {name} not found at {path}")
    return path.read_text().strip()
```

**Bind-mount secrets (development / CI)**

A secrets file on the host is mounted into the container at run time with `-v /host/secrets/db.env:/run/config/db.env:ro`. The `:ro` flag makes it read-only inside the container. Never commit the secrets file.

**Secret scanning**

Before every commit, scan staged files for patterns that look like credentials. Tools: `truffleHog`, `gitleaks`, `detect-secrets`. A pre-commit hook is the correct integration point. The scan should run in CI as well.

### Config patterns: 12-factor app

The [12-factor app](https://12factor.net/config) rule: config is everything that varies between deploys (dev/staging/prod). It must be stored in environment variables, not in the codebase.

- Local development: `.env` file + `python-dotenv` or `pydantic-settings` with `env_file=".env"`.
- Production: real environment variables injected by the orchestrator. The `.env` file is never deployed.
- `.env` must be in `.gitignore`. Always maintain a `.env.example` with placeholder keys and no values.

```python
# config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    db_host: str
    db_port: int = 5432
    db_name: str
    db_user: str
    db_password: str  # injected from env, never default here
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()  # raises ValidationError if required vars are missing
```

### Health checks: Dockerfile HEALTHCHECK instruction

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=15s \
    CMD curl -f http://localhost:8080/health || exit 1
```

Parameters:
- `--interval`: time between checks (default 30s).
- `--timeout`: how long to wait for the command to finish (default 30s).
- `--retries`: consecutive failures before status becomes `unhealthy` (default 3).
- `--start-period`: grace period during container startup before failures count (default 0s). Set this to the expected startup time.

The check command must exit 0 for healthy, 1 for unhealthy. Use `curl -f` (fail on HTTP error codes) or a small Python script if `curl` is not in the image.

Inspect health status:

```bash
docker inspect --format='{{json .State.Health}}' <container>
```

### Health checks in Compose

```yaml
services:
  api:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 15s

  worker:
    depends_on:
      api:
        condition: service_healthy
```

`condition: service_healthy` blocks the dependent service from starting until the dependency passes its health check.

### Non-root users

By default, Docker runs the container process as root (UID 0). Root inside a container can:
- Write to any file in the container filesystem.
- Exploit kernel vulnerabilities with higher privilege.
- If the container runtime is misconfigured, escalate to the host.

Create a dedicated application user in the Dockerfile:

```dockerfile
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --no-create-home --shell /sbin/nologin appuser

USER appuser
```

After the `USER` instruction, all subsequent `RUN`, `CMD`, and `ENTRYPOINT` instructions execute as that user. Ensure the application files are owned by this user:

```dockerfile
COPY --chown=appuser:appgroup . /app
```

### Read-only filesystem

```bash
docker run --read-only --tmpfs /tmp --tmpfs /app/cache myimage
```

`--read-only` mounts the root filesystem as read-only. `--tmpfs` provides writable in-memory directories for paths the application must write to (temp files, cache, PID files). In Compose:

```yaml
read_only: true
tmpfs:
  - /tmp
  - /app/cache
```

### Minimizing attack surface

**`no-new-privileges`** prevents the process from gaining additional privileges via `setuid`/`setgid` binaries:

```bash
docker run --security-opt no-new-privileges:true myimage
```

In Compose:

```yaml
security_opt:
  - no-new-privileges:true
```

**Dropping Linux capabilities**: containers run with a default set of capabilities. Drop all and add back only what is needed:

```yaml
cap_drop:
  - ALL
cap_add:
  - NET_BIND_SERVICE  # only if binding to ports < 1024
```

## Worked example

**Secure Dockerfile:**

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

WORKDIR /app

# Create non-root user before installing deps
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --no-create-home --shell /sbin/nologin appuser

# Install dependencies as root (pip needs write access to site-packages)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and transfer ownership
COPY --chown=appuser:appgroup . .

# Health check — /health must return 2xx
HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

# Switch to non-root user
USER appuser

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

No `ENV` instructions with secrets. No `ARG` instructions carrying secrets (they appear in `docker history`). The image carries no credentials.

**docker-compose.yml with health check and non-root:**

```yaml
services:
  api:
    build: .
    env_file: .env          # injected at runtime, never baked in
    ports:
      - "${API_PORT:-8080}:8080"
    read_only: true
    tmpfs:
      - /tmp
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
      start_period: 10s

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: ${DB_NAME}
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
    secrets:
      - db_password
    volumes:
      - db_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER} -d ${DB_NAME}"]
      interval: 10s
      timeout: 5s
      retries: 5

secrets:
  db_password:
    file: ./secrets/db_password.txt   # never commit this file

volumes:
  db_data:
```

`.env.example` (committed):
```
API_PORT=8080
DB_NAME=myapp
DB_USER=myapp
# DB_PASSWORD set via secrets or env injection
```

`.env` is in `.gitignore`. `./secrets/db_password.txt` is in `.gitignore`.

## Exercises

1. Take an existing Dockerfile that runs as root and has a hardcoded `ENV DB_PASSWORD=...` line. Remove the hardcoded secret, add a non-root user, and add a health check. Verify with `docker inspect` that the container transitions to `healthy`.

2. Write a `config.py` using `pydantic-settings` that loads `DB_HOST`, `DB_PORT`, `DB_USER`, and `DB_PASSWORD` from environment variables. The module must raise a `ValidationError` with a clear message if any required variable is missing. Write a `pytest` test that confirms the error is raised when `DB_PASSWORD` is unset.

3. Start a container with `--read-only` and `--security-opt no-new-privileges:true`. Confirm the application still starts correctly by checking the health endpoint. Identify which paths require `--tmpfs` mounts by observing the `PermissionError` logs, then add the minimal set of `tmpfs` mounts.

(Solutions: ../solutions/week-04/day-5.md)

## Self-check

- Can you explain why secrets in `ENV` Dockerfile instructions are dangerous even after a later `RUN unset` command?
- What is the difference between `--timeout` and `--start-period` in a `HEALTHCHECK`?
- Why does `cap_drop: ALL` improve security even for a non-root user?
- What does `condition: service_healthy` in a Compose `depends_on` block actually wait for?
- What is the 12-factor rule for config, and why does it matter for containerized workloads?
