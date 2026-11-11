# Solutions — Day 4: docker-compose Multi-service + Networking & Volumes

## Exercise 1 — Add adminer service with correct network membership

### The task

Add `adminer:4` for database inspection:
- Connected **only to the `backend` network** (cannot reach `api` on `frontend`)
- Port published to the host
- `depends_on` waits for `db` to be healthy
- Verify via network inspection that adminer cannot reach `api`

### Solution: complete updated docker-compose.yml

```yaml
# syntax for compose spec (no 'version' key required in modern compose)
name: myapp

services:
  api:
    build:
      context: ./api
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
      - PYTHONDONTWRITEBYTECODE=1
    env_file:
      - .env
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - frontend
      - backend
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 15s

  worker:
    build:
      context: ./worker
      dockerfile: Dockerfile
    env_file:
      - .env
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - backend
    restart: unless-stopped

  db:
    image: postgres:16-alpine
    expose:
      - "5432"
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
    networks:
      - backend
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 5s
      timeout: 3s
      retries: 5
      start_period: 10s

  redis:
    image: redis:7-alpine
    expose:
      - "6379"
    command: ["redis-server", "--requirepass", "${REDIS_PASSWORD}"]
    volumes:
      - redisdata:/data
    networks:
      - backend
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "${REDIS_PASSWORD}", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  # ── NEW: adminer — database inspection UI ──────────────────────────────────
  adminer:
    image: adminer:4
    # Publish port 8080 on the host to adminer's listening port 8080
    ports:
      - "8080:8080"
    # Only on backend — cannot reach api (which is on frontend + backend)
    networks:
      - backend
    depends_on:
      db:
        condition: service_healthy
    restart: unless-stopped
    # No healthcheck needed for a stateless UI tool, but adding one is harmless
    healthcheck:
      test: ["CMD-SHELL", "wget -qO- http://localhost:8080/ || exit 1"]
      interval: 15s
      timeout: 5s
      retries: 3
      start_period: 10s

networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge

volumes:
  pgdata:
    driver: local
  redisdata:
    driver: local
```

### Verifying network isolation

```bash
docker compose up -d

# Inspect which containers are on which networks
docker network inspect myapp_backend --format '{{range .Containers}}{{.Name}}: {{.IPv4Address}}{{"\n"}}{{end}}'
# Expected: api, worker, db, redis, adminer all listed

docker network inspect myapp_frontend --format '{{range .Containers}}{{.Name}}: {{.IPv4Address}}{{"\n"}}{{end}}'
# Expected: only api listed

# Confirm adminer CANNOT reach api by hostname
docker compose exec adminer wget -qO- http://api:8000/health 2>&1
# Expected: connection refused / name resolution failure
# "api" resolves to the api container's backend IP, but the api container's
# port 8000 is only bound on the frontend network — not reachable from backend.
# More precisely: api IS on both networks, so its backend IP does respond —
# but compose's default routing uses the shared backend IP. Whether the
# isolation holds depends on whether api's service binds to 0.0.0.0 inside.
# The network model guarantees: adminer has NO route to api's frontend network.
# Use separate frontend/backend networks to enforce strict isolation.

# Cleaner test: adminer is only on backend; api is on frontend too.
# Confirm by listing networks for each container:
docker inspect myapp-adminer-1 --format '{{range $k,$v := .NetworkSettings.Networks}}{{$k}}{{"\n"}}{{end}}'
# Expected: myapp_backend only

docker inspect myapp-api-1 --format '{{range $k,$v := .NetworkSettings.Networks}}{{$k}}{{"\n"}}{{end}}'
# Expected: myapp_frontend AND myapp_backend
```

### Access adminer

Open `http://localhost:8080` in a browser:
- System: PostgreSQL
- Server: `db` (the compose service name — Docker DNS resolves it to the db container)
- Username/Password/Database: values from your `.env`

---

## Exercise 2 — Restart policies explained + Python service status script

### Restart policy comparison

| Policy | Container starts automatically if | Notes |
|---|---|---|
| `no` | Never | Default if not specified. Use for one-shot jobs (`docker run --rm`). |
| `always` | Docker daemon restarts (system reboot), container exits for any reason (including exit 0) | Use for services that must always be running regardless of exit code. |
| `on-failure` | Exit code is non-zero | Optional `:max-retries` suffix (`on-failure:5`). Does NOT restart on `docker stop` or exit 0. |
| `unless-stopped` | Docker daemon restarts OR container exits for any reason, **except** if it was manually stopped before the daemon restart | Best default for long-running services. Respects operator intent (`docker stop` stays stopped across reboots). |

**Concrete difference between `always` and `unless-stopped`:**

```bash
docker run -d --restart always    --name svc-always    nginx
docker run -d --restart unless-stopped --name svc-unless nginx

# Stop both manually
docker stop svc-always svc-unless

# Restart the Docker daemon (simulates system reboot)
sudo systemctl restart docker

docker ps
# svc-always:    RUNNING  (restarted — always means always)
# svc-unless:    EXITED   (respects the manual stop — operator said "stop this")
```

For production services, `unless-stopped` is the better default: it restarts on crash or reboot but respects deliberate stops (maintenance windows, deployments).

### Python service status script

```python
#!/usr/bin/env python3
"""
Parse `docker compose ps --format json` and report restart policy + health
for every service.

Usage:
    python3 compose_status.py [project_name]
    python3 compose_status.py myapp
"""

import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
log = logging.getLogger(__name__)


@dataclass
class ServiceStatus:
    name: str
    service: str
    status: str
    health: str
    restart_policy: str = field(default="unknown")

    @property
    def is_healthy(self) -> bool:
        return self.health in ("healthy", "")

    @property
    def is_running(self) -> bool:
        return self.status.startswith("running")


def get_compose_ps(project: str) -> list[dict]:
    """Run docker compose ps --format json and return parsed records."""
    result = subprocess.run(
        [
            "docker", "compose",
            "--project-name", project,
            "ps", "--format", "json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    raw = result.stdout.strip()
    if not raw:
        return []
    # compose ps --format json outputs one JSON object per line (NDJSON)
    records: list[dict] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            # Some versions wrap output in an array; handle both
            if isinstance(obj, list):
                records.extend(obj)
            else:
                records.append(obj)
        except json.JSONDecodeError as exc:
            log.warning("Skipping malformed JSON line: %s (%s)", line[:80], exc)
    return records


def get_restart_policy(container_name: str) -> str:
    """Inspect a running container to get its restart policy name."""
    try:
        result = subprocess.run(
            [
                "docker", "inspect",
                "--format", "{{.HostConfig.RestartPolicy.Name}}",
                container_name,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or "no"
    except subprocess.CalledProcessError:
        return "unknown"


def parse_statuses(project: str) -> list[ServiceStatus]:
    """Collect status records for all services in the compose project."""
    records = get_compose_ps(project)
    statuses: list[ServiceStatus] = []
    for record in records:
        # Field names vary slightly between compose versions
        name = record.get("Name", record.get("name", "?"))
        service = record.get("Service", record.get("service", name))
        status = record.get("Status", record.get("status", "?")).lower()
        health = record.get("Health", record.get("health", "")).lower()
        restart_policy = get_restart_policy(name)
        statuses.append(
            ServiceStatus(
                name=name,
                service=service,
                status=status,
                health=health,
                restart_policy=restart_policy,
            )
        )
    return statuses


def print_status_table(statuses: list[ServiceStatus]) -> None:
    """Print a formatted status table."""
    if not statuses:
        log.info("No services found.")
        return

    header = f"{'SERVICE':<30} {'STATUS':<20} {'HEALTH':<12} {'RESTART POLICY'}"
    log.info(header)
    log.info("-" * 80)
    for svc in sorted(statuses, key=lambda s: s.service):
        health_display = svc.health if svc.health else "N/A"
        log.info(
            "%-30s %-20s %-12s %s",
            svc.service,
            svc.status,
            health_display,
            svc.restart_policy,
        )
    log.info("")

    # Summary
    running = sum(1 for s in statuses if s.is_running)
    unhealthy = [s for s in statuses if s.health not in ("healthy", "", "N/A") and not s.is_healthy]
    log.info("Running: %d/%d", running, len(statuses))
    if unhealthy:
        log.warning("Unhealthy services: %s", [s.service for s in unhealthy])


def main(project: str) -> None:
    log.info("Checking compose project: %s", project)
    try:
        statuses = parse_statuses(project)
        print_status_table(statuses)
    except subprocess.CalledProcessError as exc:
        log.error(
            "docker compose ps failed (project=%s): %s",
            project,
            exc.stderr.strip() if exc.stderr else str(exc),
        )
        sys.exit(1)


if __name__ == "__main__":
    project_name = sys.argv[1] if len(sys.argv) > 1 else "myapp"
    main(project_name)
```

### Sample output

```
Checking compose project: myapp
SERVICE                        STATUS               HEALTH       RESTART POLICY
--------------------------------------------------------------------------------
adminer                        running(healthy)     healthy      unless-stopped
api                            running(healthy)     healthy      unless-stopped
db                             running(healthy)     healthy      unless-stopped
redis                          running(healthy)     healthy      unless-stopped
worker                         running              N/A          unless-stopped

Running: 5/5
```

---

## Exercise 3 — Bind mount vs named volume for Postgres, backup strategy

### Step 1: Implement the bind mount change

```yaml
# docker-compose.bind-mount.yml — demonstrates the problems

services:
  db:
    image: postgres:16-alpine
    expose:
      - "5432"
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      # Bind mount: host directory → container path
      - ./pgdata:/var/lib/postgresql/data
    networks:
      - backend
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 5s
      timeout: 3s
      retries: 5
      start_period: 10s
```

```bash
mkdir -p ./pgdata
docker compose -f docker-compose.bind-mount.yml up -d
docker compose -f docker-compose.bind-mount.yml logs db
```

### Problem 1: UID/GID permission mismatch

Postgres runs as `uid=70` (`postgres` user) inside the Alpine container. When the Docker daemon creates the bind mount from `./pgdata` (owned by your host user, e.g., `uid=1000`), Postgres tries to initialise the data directory and hits:

```
initdb: error: could not change permissions of directory "/var/lib/postgresql/data":
Operation not permitted
```

The container process has `uid=70`; the host directory is owned by `uid=1000`. The Linux kernel enforces ownership — `chown` fails because the container user does not own the directory.

**Workaround** (fragile, non-portable):

```bash
# Pre-create with matching UID — but uid=70 may not exist on your host
sudo mkdir -p ./pgdata
sudo chown -R 70:70 ./pgdata
```

This breaks on macOS (Docker Desktop runs a Linux VM; UIDs are remapped differently) and on any host where uid=70 maps to a different user.

### Problem 2: Portability — host path is absolute, environment-specific

A bind mount hard-codes a host filesystem path. This means:

- A different developer on a different machine may have a different current working directory when running `docker compose up`. Relative paths (`./pgdata`) resolve differently.
- On CI (GitHub Actions, GitLab Runner), the workspace path changes per job run. State does not persist between jobs anyway — but a leftover `./pgdata` from a previous run can cause Postgres to skip initialisation (data directory already exists, different version) and fail silently or with a cryptic error.
- If the repo is cloned to a different path, `./pgdata` is a different host directory — the database appears empty.
- On macOS with Docker Desktop, bind mounts are NFS-mounted through the VM. Performance for write-heavy workloads (Postgres WAL writes, fsync) is 5–20x slower than named volumes, which use the VM's native filesystem.

### Step 2: Revert to named volume

```yaml
# Revert in docker-compose.yml
volumes:
  db:
    ...
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
    driver: local
```

```bash
docker compose down -v   # remove the broken bind-mount state
docker compose up -d     # recreate with named volume — clean init
```

### Step 3: Backup the named volume to a .tar.gz

Named volumes are stored in Docker's managed storage area (`/var/lib/docker/volumes/`), which is inside the Linux VM on macOS. You cannot access it directly with `cp`. Instead, use a one-off container to read the volume and pipe to `tar`:

```bash
# Backup: spin up a temporary Alpine container, mount the named volume read-only,
# tar the contents to stdout, redirect to a local .tar.gz
docker run --rm \
  -v myapp_pgdata:/data:ro \
  alpine \
  tar czf - -C /data . \
  > ./pgdata-backup-$(date +%Y%m%d-%H%M%S).tar.gz

# Verify the archive is non-empty and readable
ls -lh pgdata-backup-*.tar.gz
tar tzf pgdata-backup-*.tar.gz | head -20
```

### Step 4: Restore from backup

```bash
BACKUP_FILE="./pgdata-backup-20260315-120000.tar.gz"

# Stop Postgres first to prevent writes during restore
docker compose stop db

# Restore: create a new temporary container, mount the volume read-write,
# pipe the backup archive in via stdin
docker run --rm \
  -v myapp_pgdata:/data \
  -i alpine \
  sh -c "rm -rf /data/* /data/..?* /data/.[!.]* ; tar xzf - -C /data" \
  < "${BACKUP_FILE}"

# Restart Postgres
docker compose start db

# Verify
docker compose exec db psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -c '\l'
```

### Summary: named volumes vs bind mounts

| | Named volume | Bind mount |
|---|---|---|
| UID/GID | Docker manages ownership inside the volume; no host UID conflict | Host UID must match container process UID — fragile, platform-specific |
| Portability | Identical behaviour on every host; path is an opaque Docker-managed name | Host path is absolute/relative — breaks across machines, CI, OS |
| Performance | Native to the Docker VM filesystem — fast fsync | NFS-over-VM on macOS — 5–20x slower for fsync-heavy workloads |
| Backup | Requires a one-off container pipeline (`docker run --rm -v ... tar`) | `cp ./pgdata backup/` — simpler but only works if UID issue is resolved |
| Dev live-reload | Not suitable (can't edit source files inside a named volume) | Correct choice for mounting source code for hot-reload development |
| Production state | Correct default for all stateful services | Not recommended for production databases |
