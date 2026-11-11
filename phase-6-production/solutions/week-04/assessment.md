# Answer Key — Week 04 Assessment

---

## Conceptual Questions

### Q1. Containers vs VMs: how do they differ in isolation, startup time, and overhead?

**Containers** share the host OS kernel and use Linux kernel features to create isolation:

- **Namespaces** partition global resources: `pid` (process IDs), `net` (network stack), `mnt` (filesystem mount points), `uts` (hostname/domain), `ipc` (inter-process communication), `user` (UID/GID mapping).
- **cgroups (control groups)** enforce resource limits: CPU shares, memory limits, I/O bandwidth.
- **Union filesystems** (OverlayFS) layer images on top of each other efficiently.

**Virtual machines** virtualize hardware via a hypervisor (Type 1: bare metal like KVM/Hyper-V; Type 2: hosted like VirtualBox). Each VM runs a full OS kernel.

| Dimension | Containers | VMs |
|---|---|---|
| Kernel | Shared (host kernel) | Separate per VM |
| Startup | Milliseconds | Seconds to minutes |
| Overhead | ~MB per container | ~GB per VM (OS + kernel) |
| Isolation strength | Process-level (kernel namespace bugs can escape) | Hardware-level (stronger boundary) |
| Portability | Image = consistent across any Linux host | Image tied to hypervisor/hardware |
| Use case | Microservices, CI, ephemeral workloads | Full OS isolation, Windows on Linux, strong multi-tenant security |

**Security note:** containers are not as strongly isolated as VMs. A kernel vulnerability can allow a container to escape to the host. For multi-tenant environments with untrusted code, use gVisor, Kata Containers, or VMs.

---

### Q2. How does Docker image layer caching work, and how should you order Dockerfile instructions to maximize cache hits?

Docker builds images layer by layer. Each `RUN`, `COPY`, and `ADD` instruction creates a new layer. Layers are content-addressed: Docker computes a hash of the instruction and its inputs.

**Invalidation is top-down and cascading:** if layer N is invalidated (instruction changed, or an earlier layer changed), layers N through the last layer are all rebuilt — even if those later instructions didn't change.

**Practical rule:** put instructions that change rarely near the top; instructions that change often near the bottom.

```dockerfile
# WRONG — changing any source file rebuilds pip install
COPY . .
RUN pip install -r requirements.txt

# CORRECT — pip install is cached as long as requirements.txt doesn't change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .        # source code changes here — only this layer and below are rebuilt
```

**Why this matters:** a developer workflow where `pip install` takes 2 minutes can be reduced to near-zero on subsequent builds if requirements haven't changed.

**Additional cache optimization:**
- Use `.dockerignore` to exclude files that change frequently (logs, `.git`, test output) from invalidating the `COPY . .` layer.
- Multi-stage builds allow the builder stage to have its own cache chain independent of the runtime stage.

---

### Q3. Why does using `ENV` or `ARG` for secrets in a Dockerfile create a security risk?

**`ENV` (runtime environment variable baked into image):**

```dockerfile
ENV DB_PASSWORD=supersecret   # BAD
```

Anyone who can inspect the running container sees it:

```bash
docker inspect <container> --format='{{json .Config.Env}}'
# ["DB_PASSWORD=supersecret", "PATH=/usr/local/bin:..."]
```

The value is also stored permanently in the image layer — every container started from this image inherits it.

**`ARG` (build-time variable used in a `RUN` instruction):**

```dockerfile
ARG DB_PASSWORD
RUN echo "password=$DB_PASSWORD" > /app/.config   # BAD
```

`docker history <image>` shows the full `RUN` command text including the interpolated `ARG` value. Even if you delete the resulting file in a later layer, the value is still in history.

**Correct patterns:**

| Pattern | When to use |
|---|---|
| `env_file` in compose | Local development and single-host deployments |
| Kubernetes Secrets | K8s workloads (note: stored base64, not encrypted by default — use Sealed Secrets or Vault) |
| HashiCorp Vault | Production multi-service, dynamic secrets, audit logging needed |
| Docker Swarm secrets | Swarm mode — mounted as `/run/secrets/<name>`, never in env vars |
| AWS Secrets Manager / Azure Key Vault | Cloud-native deployments |

**Key principle:** secrets should be injected at **runtime**, never baked into the **image**.

---

### Q4. Explain multi-stage builds. What problem do they solve and how does `COPY --from=builder` work?

**Problem:** build tools (gcc, pip, npm, go build) are large and not needed at runtime. Without multi-stage, a Python image that compiled C extensions would include all of gcc, headers, and build cache in the final image.

**Multi-stage solution:** use multiple `FROM` instructions in one Dockerfile. Each `FROM` starts a new stage. Only the final stage becomes the image that gets pushed/run.

```dockerfile
# Stage 1: builder — has all build tools
FROM python:3.12 AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY . .
RUN python -m compileall .


# Stage 2: runtime — starts fresh, copies only what's needed
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /root/.local /root/.local
# Copy compiled source
COPY --from=builder /build/app ./app

ENV PATH=/root/.local/bin:$PATH
USER nobody

CMD ["python", "-m", "app.main"]
```

**`COPY --from=builder`:** copies files from a named stage (`builder`) rather than the local filesystem. Only the specified paths are copied — all other builder artifacts (pip cache, gcc, headers, test files) are discarded.

**Typical size reduction:** 500 MB+ builder image → 50–80 MB runtime image (5–10×).

**Multi-stage also works with external images:**
```dockerfile
COPY --from=alpine:latest /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
```

---

### Q5. What are the limitations of `depends_on: condition: service_healthy` in docker-compose?

`service_healthy` waits for the **HEALTHCHECK** defined on the dependency to return healthy before starting the dependent service. This is much better than the default `service_started`, but has important limitations:

1. **Only checks what the HEALTHCHECK checks.** A Postgres HEALTHCHECK might only verify `pg_isready` (TCP connection accepted) — the database could still be running migrations, loading data, or initializing before accepting queries.

2. **Healthcheck can pass; service still warming up.** Some applications pass the health endpoint immediately on boot but take additional time to establish connection pools, warm caches, or load models. The first real request may still fail.

3. **`start_period` must be calibrated per service.** If `start_period` is too short, failed checks during startup count toward `retries` and the container goes `unhealthy` before it's had a chance to start properly.

4. **Only works within a single `docker compose` project.** Cannot wait for health of services in another compose file or on a remote host.

5. **Not a replacement for retry logic in the application.** Even with `service_healthy`, network blips, restarts, and failovers happen at runtime. Applications should implement connection retry with exponential backoff regardless of compose health checks.

**Best practice:** use `service_healthy` in compose AND implement application-level retry/backoff for database and external service connections.

---

### Q6. Explain the four HEALTHCHECK parameters: `--interval`, `--timeout`, `--retries`, `--start-period`.

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

| Parameter | Default | What it controls |
|---|---|---|
| `--interval` | 30s | How often Docker runs the health check command. Lower = more responsive detection of failures, higher CPU cost. |
| `--timeout` | 30s | If the health check command does not complete within this duration, it counts as a failure. Should be shorter than `--interval`. |
| `--start-period` | 0s | Grace period after container start. Failed checks during `start-period` do NOT count toward `--retries`. Gives the application time to initialize. |
| `--retries` | 3 | Number of **consecutive** failures required to mark the container `unhealthy`. A single failure doesn't immediately trigger unhealthy — transient network issues are tolerated. |

**Container health states:**
- `starting` — within `start-period` OR not enough checks have run yet
- `healthy` — last check succeeded
- `unhealthy` — `retries` consecutive failures

**Choosing values:**
- `--interval`: 10–30s for production (30s is safe default).
- `--timeout`: 3–10s (must be < `--interval`). Match your SLA for the health endpoint.
- `--start-period`: set to your application's P95 cold-start time. For a Python app loading ML models, this might be 30–60s.
- `--retries`: 3 is a reasonable default. Increase to 5 for services that have occasional slow responses.

---

### Q7. Why is running containers as root a security risk? What additional hardening measures complement non-root users?

**Root in container = UID 0 on host (unless user namespaces are configured).** Most Docker deployments use the default single-user-namespace mapping, meaning root in the container IS root on the host for filesystem purposes.

**Attack scenarios:**

1. **Container escape:** a CVE in the container runtime (runc, containerd) combined with a root process gives the attacker full host access. A non-root process limits the escape to an unprivileged UID.

2. **Volume mount exploitation:** if `/etc/passwd` or host paths are mounted into a root container, the process can modify them.

3. **Capability abuse:** root has all Linux capabilities by default. Even without a full escape, capabilities like `CAP_NET_ADMIN`, `CAP_SYS_ADMIN`, `CAP_CHOWN` can be exploited.

4. **Supply chain / RCE:** if a dependency has a remote code execution bug, the attacker's shell runs as root — potentially rewriting binaries, installing backdoors, or accessing other containers' filesystems via shared volumes.

**Complementary hardening measures:**

```yaml
# docker-compose.yml
services:
  api:
    user: "1000:1000"                    # explicit UID:GID (matches Dockerfile USER)
    read_only: true                      # container FS immutable
    tmpfs:
      - /tmp
    security_opt:
      - no-new-privileges:true          # blocks setuid/setgid escalation
    cap_drop:
      - ALL                             # drop every Linux capability
    cap_add:
      - NET_BIND_SERVICE                # add back only what's needed
```

**Defense in depth principle:** no single measure is sufficient. Combine non-root + read-only FS + no-new-privileges + cap_drop + network segmentation + image scanning (Trivy, Grype) + runtime monitoring (Falco).

---

### Q8. How do you debug an `unhealthy` container? Walk through the diagnostic commands.

```bash
# Step 1: see the health check history (last 5 results)
docker inspect --format='{{json .State.Health}}' <container_name_or_id> | python -m json.tool

# Output shows:
# {
#   "Status": "unhealthy",
#   "FailingStreak": 3,
#   "Log": [
#     {
#       "Start": "2026-11-11T10:00:00Z",
#       "End": "2026-11-11T10:00:05Z",
#       "ExitCode": 1,
#       "Output": "curl: (7) Failed to connect to localhost port 8000"
#     }
#   ]
# }
```

```bash
# Step 2: check application logs (is the app crashing? binding to the wrong port?)
docker logs <container_name> --tail 50

# Step 3: get a shell inside the container and run the health check manually
docker exec -it <container_name> /bin/sh
# Inside container:
curl -v http://localhost:8000/health
# or
python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
```

```bash
# Step 4: verify the health check binary actually exists in the image
docker exec <container_name> which curl
# If not found: curl is not installed — use the python alternative or RUN apt-get install curl

# Step 5: check if the process is actually listening on the expected port
docker exec <container_name> ss -tlnp
# or
docker exec <container_name> netstat -tlnp
```

```bash
# Step 6: for a stopped/crashed container
docker inspect <container_name> --format='{{.State.ExitCode}} {{.State.Error}}'

# Step 7: watch health check events in real time
docker events --filter container=<container_name> --filter event=health_status
```

**Common root causes:**

| Symptom | Likely cause |
|---|---|
| `curl: (7) Failed to connect` | App not started, wrong port, binding to `127.0.0.1` only in IPv6 container |
| `curl: (22) HTTP error 503` | App started but not ready (still loading), dependency not available |
| `curl: not found` | curl not installed in the image — switch to python alternative |
| `ExitCode: 137` | Health check process OOM-killed — container memory limit too tight |
| Health check passes but compose `depends_on` still fails | `start_period` too short — increase it |

---

## Coding Challenge Solution

### Task

Build a production-ready containerized FastAPI application with:
- Multi-stage Dockerfile (non-root, health check)
- docker-compose.yml with Postgres, named volumes, env_file, health checks, no hardcoded credentials

### Solution — `Dockerfile`

```dockerfile
# ============================================================
# Stage 1: builder — installs dependencies with full toolchain
# ============================================================
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies (needed for some C-extension packages)
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# --user installs into /root/.local so we can COPY the whole dir to final stage
RUN pip install --no-cache-dir --user -r requirements.txt


# ============================================================
# Stage 2: runtime — minimal image, no build tools
# ============================================================
FROM python:3.12-slim AS runtime

# Install only runtime system libraries (libpq for psycopg2)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application source with correct ownership
COPY --chown=appuser:appuser app/ ./app/

# Create log directory with correct ownership
RUN mkdir -p /app/logs && chown appuser:appuser /app/logs

# Switch to non-root
USER appuser

# Add user's local bin to PATH (where --user installs scripts)
ENV PATH=/home/appuser/.local/bin:$PATH

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Solution — `app/main.py`

```python
# app/main.py
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import asyncpg
from fastapi import FastAPI, status

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("APP_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

_pool: asyncpg.Pool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _pool
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        logger.info("Connecting to database")
        _pool = await asyncpg.create_pool(database_url, min_size=2, max_size=10)
        logger.info("Database pool created")
    yield
    if _pool:
        await _pool.close()
        logger.info("Database pool closed")


app = FastAPI(lifespan=lifespan)


@app.get("/health", status_code=status.HTTP_200_OK)
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready", status_code=status.HTTP_200_OK)
async def ready() -> dict[str, str]:
    """Readiness probe — checks DB connectivity."""
    if _pool is None:
        return {"status": "no_db_configured"}
    async with _pool.acquire() as conn:
        await conn.fetchval("SELECT 1")
    return {"status": "ok"}
```

### Solution — `requirements.txt`

```
fastapi>=0.115.0
uvicorn[standard]>=0.31.0
asyncpg>=0.29.0
```

### Solution — `.env.example`

```dotenv
# PostgreSQL
DB_HOST=postgres
DB_PORT=5432
DB_NAME=myapp
DB_USER=myapp_user
DB_PASSWORD=          # openssl rand -hex 32

# Constructed from above — used by asyncpg in the app
DATABASE_URL=postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}

# Application
APP_LOG_LEVEL=INFO
APP_ENV=development
```

### Solution — `docker-compose.yml`

```yaml
services:
  postgres:
    image: postgres:16-alpine
    env_file:
      - .env
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
      start_period: 5s
      retries: 5
    networks:
      - app_net
    restart: unless-stopped

  api:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    depends_on:
      postgres:
        condition: service_healthy
    volumes:
      - app_logs:/app/logs
    read_only: true
    tmpfs:
      - /tmp
      - /var/run
    security_opt:
      - no-new-privileges:true
    networks:
      - app_net
    restart: unless-stopped

volumes:
  postgres_data:
    driver: local
  app_logs:
    driver: local

networks:
  app_net:
    driver: bridge
```

### Solution — `.gitignore` (relevant section)

```gitignore
# Secrets
.env
.env.local
.env.*.local
!.env.example

# Python
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
dist/
*.egg-info/

# Docker
*.log
```

### Solution — `.dockerignore`

```dockerignore
# Version control
.git
.gitignore

# Secrets — never in build context
.env
.env.*
!.env.example

# Python cache
__pycache__
*.py[cod]
.pytest_cache
.ruff_cache

# Tests (not needed in runtime image)
tests/

# Docs
*.md
docs/

# IDE
.vscode/
.idea/
```

### Verification commands

```bash
# Build and start
docker compose up --build -d

# Wait for healthy, then check
docker compose ps

# Verify health check
docker inspect --format='{{json .State.Health.Status}}' <project>-api-1

# Test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/ready

# Confirm non-root
docker exec <project>-api-1 whoami
# Expected: appuser

# Confirm read-only filesystem
docker exec <project>-api-1 touch /test-file
# Expected: touch: cannot touch '/test-file': Read-only file system

# Inspect image size (multi-stage benefit)
docker images | grep <project>

# View secrets are NOT in env
docker inspect <project>-api-1 --format='{{json .Config.Env}}'
# Should show variable names only if DATABASE_URL is set via env_file,
# not hardcoded values in the Dockerfile
```
