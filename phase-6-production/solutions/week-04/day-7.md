# Solutions — Day 7 (Sunday): Project Completion + Assessment Prep

---

## Exercise 1: Debug a Broken docker-compose.yml

### The broken file (with issues annotated)

```yaml
# docker-compose.yml  — BROKEN VERSION (do not use)
version: "3.8"

services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: myuser
      POSTGRES_PASSWORD: mypassword        # BUG 1: hardcoded secret
      POSTGRES_DB: mydb
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD", "wget", "--spider", "http://localhost:5432"]  # BUG 2: wget on TCP port
      interval: 10s
      timeout: 5s
      retries: 5

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://myuser:mypassword@postgres:5432/mydb  # BUG 3: hardcoded creds in env
    depends_on:
      postgres:
        condition: service_healthy
    volumes:
      - app_logs:/app/logs                  # BUG 4: volume exists but isn't declared

volumes:
  postgres_data:
  # app_logs missing here  ←  BUG 4
  # no networks declared   ←  BUG 5 (services can still communicate via default network,
  #                                   but explicit networks are required by the exercises)
```

### Bug analysis

| # | Location | Issue | Why it breaks |
|---|---|---|---|
| 1 | `postgres.environment.POSTGRES_PASSWORD` | Hardcoded password | Committed to git history; visible in `docker inspect` |
| 2 | `postgres.healthcheck.test` | `wget --spider http://localhost:5432` | PostgreSQL speaks the Postgres wire protocol on 5432, not HTTP. `wget` will always fail. |
| 3 | `api.environment.DATABASE_URL` | Hardcoded credentials in `environment` | Same as bug 1 — visible in `docker inspect` |
| 4 | `api.volumes` references `app_logs` | `app_logs` not declared in top-level `volumes:` | Compose refuses to start: `service "api" refers to undefined volume app_logs` |
| 5 | No `networks:` block | Implicit default network | Not a hard error, but explicit networks are best practice for isolation and clarity |

### The corrected file

```yaml
# docker-compose.yml  — FIXED VERSION
services:
  postgres:
    image: postgres:16-alpine
    env_file:
      - .env                             # FIX 1: load creds from .env, not hardcoded
    environment:
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: ${DB_NAME}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      # FIX 2: use pg_isready — the correct tool for checking Postgres readiness
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER} -d ${DB_NAME}"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 5s
    networks:
      - app_net

  api:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env                             # FIX 3: inject DATABASE_URL at runtime from .env
    depends_on:
      postgres:
        condition: service_healthy
    volumes:
      - app_logs:/app/logs               # now references a declared volume
    networks:
      - app_net
    healthcheck:
      test: ["CMD", "python", "-c",
             "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 5s
      start_period: 10s
      retries: 3

volumes:
  postgres_data:
  app_logs:                              # FIX 4: declare the volume

networks:
  app_net:                               # FIX 5: explicit named network
    driver: bridge
```

### `.env` (not committed) and `.env.example` (committed)

```dotenv
# .env.example — commit this
DB_USER=myapp_user
DB_PASSWORD=           # openssl rand -hex 32
DB_NAME=mydb
DATABASE_URL=postgresql://${DB_USER}:${DB_PASSWORD}@postgres:5432/${DB_NAME}
```

### Fix 2 explained in detail: why `wget --spider http://localhost:5432` fails

PostgreSQL listens on TCP port 5432 but speaks the [Postgres Frontend/Backend Protocol](https://www.postgresql.org/docs/current/protocol.html), not HTTP. When `wget` connects it sends an HTTP request; Postgres responds with an error or closes the connection. `wget` exits non-zero — the health check always fails — and the container is permanently `unhealthy`.

**Correct tools for Postgres health checks:**

```bash
# Option A: pg_isready (ships with the official postgres image)
pg_isready -U ${DB_USER} -d ${DB_NAME}

# Option B: psql (more thorough — actually runs a query)
psql -U ${DB_USER} -d ${DB_NAME} -c "SELECT 1"
```

### Fix 4 explained: volume permission issue

If the application writes to `/app/logs` but the directory is owned by root and the process runs as a non-root user, writes will fail with `Permission denied`. Fix options:

**Option A — correct ownership in Dockerfile (preferred):**
```dockerfile
RUN mkdir -p /app/logs && chown appuser:appuser /app/logs
USER appuser
```

**Option B — entrypoint that fixes permissions at runtime:**
```dockerfile
# entrypoint.sh
#!/bin/sh
chown -R appuser:appuser /app/logs
exec gosu appuser "$@"
```

**Option C — init container in compose (Swarm/K8s pattern):**
```yaml
  init-permissions:
    image: busybox
    volumes:
      - app_logs:/app/logs
    command: ["chown", "-R", "1000:1000", "/app/logs"]
```

Option A is cleanest for single-service setups. Option B works when the volume is mounted from the host and ownership changes at runtime. Option C is useful in Kubernetes.

---

## Exercise 2: Docker Best Practices Checklist (Self-assessment)

Rate yourself: **Yes / Partial / No** for each item.

### Dockerfile

- [ ] **Layer order: dependencies before source code.** `COPY requirements.txt .` and `RUN pip install` appear before `COPY . .` so code changes don't invalidate the dependency cache.
- [ ] **Multi-stage build used.** Builder stage compiles/installs; final stage copies only artifacts. Final image contains no build tools (gcc, pip cache, test deps).
- [ ] **No secrets in the image.** No `ENV SECRET=...`, no `ARG` with a real value used in `RUN`, no credentials files `COPY`ed in.
- [ ] **Non-root user.** `RUN groupadd -r appuser && useradd -r -g appuser appuser` + `USER appuser` before `CMD`.
- [ ] **HEALTHCHECK defined.** `HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD ...` checks a real liveness endpoint.
- [ ] **.dockerignore present.** Excludes `.git`, `.env`, `__pycache__`, `*.pyc`, `tests/`, `*.md` to keep build context small and prevent secret leaks.
- [ ] **Pinned base image tag.** `python:3.12-slim` not `python:latest` — ensures reproducible builds.
- [ ] **`--no-cache-dir` on pip install.** Avoids storing the pip cache in the image layer.
- [ ] **`apt-get` cleanup in same layer.** `apt-get install` followed by `rm -rf /var/lib/apt/lists/*` in the same `RUN` to avoid leaving package index in the layer.

### docker-compose.yml

- [ ] **`env_file` used, not hardcoded `environment`.** All secrets loaded from `.env` at runtime.
- [ ] **`.env` in `.gitignore`.** Only `.env.example` committed.
- [ ] **Named networks declared explicitly.** Services communicate over a named bridge network, not the default network.
- [ ] **Named volumes declared.** All volumes listed in the top-level `volumes:` block with meaningful names.
- [ ] **`depends_on` uses `condition: service_healthy`.** Services with dependencies wait for health checks, not just container start.
- [ ] **`stop_grace_period` set for long-running services.** Workers and consumers have sufficient time to drain work before SIGKILL.
- [ ] **`restart: unless-stopped`** (or `always`) on services that should recover from crashes in production.

### Security

- [ ] **`read_only: true`** on containers that don't need to write to the filesystem.
- [ ] **`tmpfs`** mounts provided for `/tmp` and other runtime write locations when `read_only: true`.
- [ ] **`security_opt: [no-new-privileges:true]`** on all services.
- [ ] **`cap_drop: [ALL]`** with only required capabilities added back via `cap_add`.

### Operations

- [ ] **Meaningful image tags.** Images tagged with git SHA or semantic version, not just `latest` in production.
- [ ] **`dive` inspection done.** `dive <image>` reviewed to catch large layers, unnecessary files, wasted space.
- [ ] **Image size monitored.** Final image size noted and compared to previous builds (multi-stage typically achieves 5–10× reduction).
- [ ] **Health check tested manually.** `docker inspect --format='{{json .State.Health}}' <container>` verified to show `healthy`.
- [ ] **Compose file validated.** `docker compose config` run to catch YAML errors before deploying.

### Scoring guide

| Score | Interpretation |
|---|---|
| 20–24 Yes | Production-ready mindset. Ship it. |
| 14–19 Yes | Solid foundation. Address the gaps before going to production. |
| 8–13 Yes | Good start. Prioritize secrets, non-root, and health checks next. |
| < 8 Yes | Review Days 3–7 before building production workloads. |
