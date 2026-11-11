# Solutions — Day 5: Secrets/Config + Health Checks + Non-root Security

---

## Exercise 1: Add a HEALTHCHECK to a FastAPI Dockerfile

### Solution

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=root:root . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Parameter breakdown

| Parameter | Value | Meaning |
|---|---|---|
| `--interval` | `30s` | Docker runs the health check every 30 seconds |
| `--timeout` | `5s` | If the check doesn't complete within 5 seconds, it counts as a failure |
| `--start-period` | `10s` | For the first 10 seconds after the container starts, failed checks don't count toward `--retries`. Gives the app time to boot. |
| `--retries` | `3` | After 3 consecutive failures, the container is marked `unhealthy` |

### Why `|| exit 1`?

`curl -f` returns a non-zero exit code on HTTP errors (4xx, 5xx). The `|| exit 1` makes the intent explicit: any failure → non-zero exit → Docker registers a failed check. Without it, a network error might still return 0 depending on shell context.

### Alternative without curl

`curl` is not in `python:3.12-slim` by default. Either install it:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
```

Or use the Python standard library instead (no extra package needed):

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1
```

The Python version raises an exception (and exits non-zero) on any HTTP error or connection failure, so it behaves identically.

### The `/health` endpoint (FastAPI side)

The health check is only meaningful if `/health` actually reflects application state:

```python
import logging
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
app = FastAPI()


@app.get("/health", status_code=status.HTTP_200_OK)
async def health() -> dict[str, str]:
    """Liveness probe — returns 200 when the app is up."""
    return {"status": "ok"}
```

For a readiness probe (app is up **and** dependencies are reachable) also check the DB connection, cache, etc., and return `503` if they fail.

---

## Exercise 2: Non-root User + Read-only Filesystem

### Solution — Dockerfile

```dockerfile
FROM python:3.12-slim

# 1. Create a dedicated system group and user.
#    -r creates a system account (no home dir, no login shell, no password aging).
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# 2. Install dependencies as root (needs write access to site-packages).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy application source and immediately assign ownership to the app user.
#    --chown ensures the process can read its own code even with a read-only FS.
COPY --chown=appuser:appuser . .

# 4. Switch to non-root for all subsequent instructions and at runtime.
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Solution — docker-compose.yml service

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    read_only: true          # mount the container's root FS read-only
    tmpfs:
      - /tmp                 # uvicorn / stdlib need a writable /tmp
      - /var/run             # PID files, sockets
    security_opt:
      - no-new-privileges:true  # prevent setuid binaries from escalating
```

### Why not run as root in a container?

1. **Container escape → host root.** If an attacker finds a container breakout vulnerability (kernel exploit, misconfigured `--privileged`, etc.) and the container process runs as root (UID 0), they land as root on the host. Running as an unprivileged UID limits the blast radius to whatever that UID can do on the host.

2. **File modification / supply chain attacks.** A root process can overwrite `/usr/local/lib/python3.12/`, swap out installed packages, or modify the application binary. A non-root user with `--chown` ownership and `read_only: true` cannot change any file in the image.

3. **Principle of least privilege.** The app only needs to listen on a port (>1024) and read its own files. Root grants hundreds of Linux capabilities the app will never use.

4. **Defense in depth.** Even if your code has a remote code execution bug, the attacker's shell is sandboxed to UID `appuser` with no write access to the filesystem.

Additional hardening options (mention in solutions for advanced learners):

```yaml
cap_drop:
  - ALL                # drop every Linux capability
cap_add:
  - NET_BIND_SERVICE   # add back only what you need (binding ports < 1024)
```

---

## Exercise 3: Secrets Strategy for a Multi-service Compose App

### Solution — `.env.example`

Document every variable so the next developer knows exactly what to provide. **Never put real values here.**

```dotenv
# Database — PostgreSQL
DB_HOST=postgres
DB_PORT=5432
DB_NAME=myapp
DB_USER=myapp_user
DB_PASSWORD=              # set a strong random password (e.g. openssl rand -hex 32)

# Cache — Redis
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=           # set a strong random password

# Application
APP_SECRET_KEY=           # Django/FastAPI signing key (openssl rand -hex 32)
APP_ENV=development       # development | staging | production
APP_LOG_LEVEL=INFO

# External APIs
OPENAI_API_KEY=           # sk-...
SMTP_PASSWORD=            # email relay password
```

### Solution — `.gitignore` (relevant entries)

```gitignore
# Secrets — never commit
.env
.env.local
.env.*.local

# Keep the example
!.env.example
```

### Solution — `docker-compose.yml`

```yaml
services:
  postgres:
    image: postgres:16-alpine
    env_file:
      - .env                    # DB_USER, DB_PASSWORD, DB_NAME come from here
    environment:
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: ${DB_NAME}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER} -d ${DB_NAME}"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    env_file:
      - .env
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "${REDIS_PASSWORD}", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  api:
    build: .
    env_file:
      - .env                    # all vars injected at runtime, not baked into image
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    ports:
      - "8000:8000"
    read_only: true
    tmpfs:
      - /tmp
      - /var/run

volumes:
  postgres_data:
```

### Why never use `ENV` for secrets in a Dockerfile?

```dockerfile
# BAD — never do this
ENV DB_PASSWORD=supersecret
```

Two attack vectors:

1. **`docker inspect <container>`** — shows all `ENV` values in plain text in the JSON output, accessible to anyone who can run Docker commands on the host.

2. **`docker history <image>`** — shows each layer command. `ARG` values used inside `RUN` instructions also appear in history unless squashed with `--squash` (experimental) or multi-stage builds.

```bash
# An attacker or curious developer runs:
docker inspect my-api --format='{{json .Config.Env}}'
# Output: ["DB_PASSWORD=supersecret","PATH=/usr/local/bin:..."]
```

**Correct runtime injection pattern:** use `env_file` in compose, environment variable injection at the orchestrator level (Kubernetes Secrets, AWS Secrets Manager, HashiCorp Vault), or Docker Swarm secrets (`/run/secrets/<name>`).

### Docker Swarm secrets (bonus — production pattern)

```yaml
# docker-compose.yml for Swarm mode
secrets:
  db_password:
    external: true           # created with: docker secret create db_password -

services:
  api:
    secrets:
      - db_password          # mounted at /run/secrets/db_password (file, not env var)
```

```python
# Read in application code
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def get_db_password() -> str:
    secret_path = Path("/run/secrets/db_password")
    if secret_path.exists():
        return secret_path.read_text().strip()
    # Fallback for local dev (env var)
    import os
    password = os.getenv("DB_PASSWORD")
    if not password:
        raise RuntimeError("DB_PASSWORD not set and /run/secrets/db_password not found")
    return password
```

This approach means the secret is never in environment variables, never in `docker inspect`, and is mounted as a tmpfs file that only the container process can read.
