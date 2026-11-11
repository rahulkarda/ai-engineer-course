# Phase 6 · Week 04 · Day 4 — docker-compose for Multi-service Apps + Networking & Volumes

> Objective: Declaratively orchestrate multi-container applications with compose, and understand Docker's networking and storage primitives.  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

## Concept

**docker-compose** (now `docker compose` as a CLI plugin) lets you describe a complete application stack — multiple services, their networks, volumes, environment, and dependencies — in a single YAML file. One command (`docker compose up`) starts every service in the right order.

This replaces sequences of `docker run` commands that are hard to reproduce, impossible to diff, and error-prone to maintain. The compose file is the authoritative record of how the application is assembled.

**Compose file anatomy**

```yaml
services:          # one key per container; the key is the service name
  api:
    build: ./api
    ...
  db:
    image: postgres:16
    ...

networks:          # named networks; services attach to one or more
  backend:
    driver: bridge

volumes:           # named volumes; survive container removal
  pgdata:
```

**Service DNS resolution**

Every service on the same Docker network is reachable by its service name as a hostname. If `api` and `db` are on the same network, the API container can connect to `postgres://db:5432/...` — Docker's embedded DNS resolves `db` to the container's IP automatically. No hard-coded IPs needed.

**Port semantics**

- `expose: ["5432"]` — makes the port reachable by other containers on the same network. Not visible from the host.
- `ports: ["5432:5432"]` — publishes the port to the host (`HOST:CONTAINER`). Reachable from localhost and potentially from the network.

Rule: only publish ports that the developer or operator needs to reach directly. Internal service-to-service traffic uses `expose`, not `ports`.

## How it works internally

**Bridge network**

By default, compose creates a project-scoped bridge network named `<project>_default`. All services join it unless overridden. A bridge network is a virtual Ethernet switch in software: the Docker daemon maintains a `veth` pair for each container and routes traffic through the bridge.

Custom named networks isolate groups of services. A service can join multiple networks.

**Named volumes vs bind mounts vs tmpfs**

| Type | Definition | Lifecycle | Use case |
|---|---|---|---|
| Named volume | `volumes: pgdata:` | Managed by Docker; persists until `docker volume rm` | Database data, any state that must survive container restarts |
| Bind mount | `./local/path:/container/path` | Host filesystem; survives `docker compose down` | Dev: mount source code for live reload |
| tmpfs | `tmpfs: /tmp` | In-memory; gone on container stop | Secrets, ephemeral scratch space, test isolation |

Named volumes are the correct default for stateful services in production. Bind mounts have permission issues in Linux (UID mismatch between host and container user) and expose host paths to containers.

**`depends_on` and health checks**

`depends_on` with no condition only waits for the container to start (the process to launch), not for the service inside it to be ready. A Postgres container starts in ~200 ms but takes 1–3 seconds to accept connections. Use `condition: service_healthy` with a `healthcheck` to wait for readiness:

```yaml
healthcheck:
  test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
  interval: 5s
  timeout: 3s
  retries: 5
  start_period: 10s
```

**Environment variables**

Three patterns, in order of preference:

1. `env_file: .env` — load a file of `KEY=VALUE` pairs. Keep `.env` out of version control; commit `.env.example`.
2. `environment:` with `${VAR}` substitution — compose reads from the shell environment or a `.env` file in the project root.
3. Inline values — only for non-sensitive defaults (`PYTHONUNBUFFERED=1`).

Never hard-code credentials in the compose file.

## Worked example

Project structure:

```
myapp/
  api/
    Dockerfile
    app/
  worker/
    Dockerfile
    worker/
  .env.example
  .env              # gitignored
  docker-compose.yml
```

`.env.example` (committed):

```dotenv
POSTGRES_USER=app
POSTGRES_DB=appdb
# Generate with: openssl rand -hex 32
POSTGRES_PASSWORD=
SECRET_KEY=
REDIS_PASSWORD=
```

`docker-compose.yml`:

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
    # Only exposed internally; not published to the host
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
    # Pass password via command arg — env var REDIS_PASSWORD is read from .env
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

networks:
  # api is on both; worker, db, redis are on backend only
  # The frontend network is where external-facing services live
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

Running the stack:

```bash
# Copy and fill in secrets
cp .env.example .env
# edit .env: set POSTGRES_PASSWORD, SECRET_KEY, REDIS_PASSWORD

# Start all services; rebuild images if Dockerfiles changed
docker compose up --build -d

# Follow logs for a specific service
docker compose logs -f api

# Scale the worker to 3 replicas
docker compose up --scale worker=3 -d

# Exec into the running api container
docker compose exec api bash

# Run a one-off command against the db service
docker compose exec db psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}"

# Stop and remove containers; volumes are preserved
docker compose down

# Stop and remove containers AND volumes (destructive — deletes all data)
docker compose down -v

# Inspect which containers are on which networks
docker network inspect myapp_backend
```

Python health-check utility (use during integration tests):

```python
#!/usr/bin/env python3
"""Wait for all compose services to report healthy before running tests."""

import logging
import subprocess
import sys
import time
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


@dataclass
class ServiceStatus:
    name: str
    status: str

    @property
    def is_healthy(self) -> bool:
        return self.status == "healthy"


def get_service_statuses(project: str) -> list[ServiceStatus]:
    result = subprocess.run(
        [
            "docker", "compose",
            "--project-name", project,
            "ps", "--format", "{{.Name}}\t{{.Health}}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    statuses: list[ServiceStatus] = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2:
            statuses.append(ServiceStatus(name=parts[0], status=parts[1].strip()))
    return statuses


def wait_for_healthy(project: str, timeout_s: int = 60) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        statuses = get_service_statuses(project)
        unhealthy = [s for s in statuses if not s.is_healthy and s.status != ""]
        if not unhealthy:
            log.info("All services healthy.")
            return
        log.info("Waiting on: %s", [s.name for s in unhealthy])
        time.sleep(3)
    log.error("Timed out waiting for services to become healthy.")
    sys.exit(1)


if __name__ == "__main__":
    project = sys.argv[1] if len(sys.argv) > 1 else "myapp"
    wait_for_healthy(project)
```

## Exercises

1. Add a fifth service to the compose file: `adminer` (image `adminer:4`) for database inspection, connected only to the `backend` network and with its port published to the host. Add a `depends_on` condition so adminer waits for `db` to be healthy. Verify that adminer cannot reach the `api` service by examining its network membership.

2. The compose file uses `restart: unless-stopped`. Research and explain the difference between `no`, `always`, `on-failure`, and `unless-stopped`. Then write a Python script (type-annotated, uses `subprocess`, logs via `logging`) that parses `docker compose ps --format json` output and prints the restart policy and health status for each service.

3. A team member suggests using a bind mount instead of a named volume for Postgres data (`./pgdata:/var/lib/postgresql/data`) to make backups easier. Implement this change, bring up the stack, and document two concrete problems you encounter (permissions, portability, or otherwise). Then revert to a named volume and write a `docker compose` command pipeline that backs up the named volume to a local `.tar.gz` file without requiring direct filesystem access.

(Solutions: ../solutions/week-04/day-4.md)

## Self-check

- [ ] I can explain why services on the same compose network resolve each other by service name without hard-coded IPs.
- [ ] I can write a compose file with named volumes, custom bridge networks, health checks, and `depends_on: condition: service_healthy` without reference.
- [ ] I can articulate the difference between named volumes and bind mounts and choose the right type for a given use case.
- [ ] I can implement a Python utility that parses `docker compose ps` output with proper type hints and structured logging.
