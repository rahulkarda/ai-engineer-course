# Phase 6 · Week 04 Project — Production-Ready Multi-Service Containerization

## Overview

Fully containerize a four-service application (FastAPI API + background worker + Postgres + Redis) to production-ready standards. The deliverable is a repository that any engineer can clone, run `make up`, and have a working, hardened stack within 90 seconds.

This project integrates every concept from the week: multi-stage builds, image size optimization, health checks, non-root users, read-only filesystems, secret management, and Compose orchestration.

## Architecture

```
                    +-----------------+
  HTTP client ----> |  API  :8080     |
                    | (FastAPI)       |
                    +--------+--------+
                             |  LPUSH jobs
                             v
                    +-----------------+     BRPOP jobs
                    |  Redis  :6379   | <-----------+
                    +-----------------+             |
                                               +----+----+
                                               | Worker  |
                                               +---------+
                                                    |
                             +----------------------+
                             |  INSERT/UPDATE
                             v
                    +-----------------+
                    | Postgres :5432  |
                    | (job state)     |
                    +-----------------+

  Networks: all services on bridge network "backend"
  Volumes:  db_data (named, Postgres data)
            redis_data (named, optional persistence)
```

All inter-service communication uses Docker's embedded DNS (service name as hostname). No service is exposed to the host network except the API on `${API_PORT:-8080}`.

## Requirements

### Functional
- `POST /tasks` — accepts `{"payload": dict}`, enqueues to Redis, writes initial record to Postgres (status=queued), returns `{"job_id": str, "status": "queued", "created_at": str}`.
- `GET /tasks/{job_id}` — reads job record from Postgres, returns current status.
- `GET /health` — returns 200 if Redis is reachable, 503 otherwise.
- Worker reads from Redis queue, logs processing, updates Postgres record to `done` or `failed`.

### Container / Compose
- Multi-stage `Dockerfile.api` and `Dockerfile.worker`: final images based on `python:3.12-slim`, no build tools, no test deps.
- Both images run as a non-root user (UID >= 1000).
- No secrets in any `ENV` or `ARG` Dockerfile instruction.
- `HEALTHCHECK` defined in both Dockerfiles.
- `docker-compose.yml` with four services, named bridge network, named volumes, `env_file`, `healthcheck` blocks, `depends_on` with `condition: service_healthy`.
- All credentials and config injected via `.env` at runtime.

### Security
- `security_opt: no-new-privileges:true` on API and worker.
- `cap_drop: ALL` on API and worker.
- `read_only: true` on API (tmpfs for `/tmp`).
- `.dockerignore` excludes `.env`, `tests/`, `.git`, `__pycache__`, `*.md`.
- `.env` in `.gitignore`. `.env.example` committed with all keys documented.

### Tooling
- `Makefile` with targets: `build`, `up`, `down`, `down-v`, `logs`, `logs-api`, `logs-worker`, `shell-api`, `shell-worker`, `test`, `clean`.

### Testing
- `pytest` test suite in `tests/`:
  - Unit tests for API endpoints (mock Redis and Postgres).
  - Integration test that spins up the real stack (`docker compose up`) and hits the endpoints.
  - All tests pass via `make test`.

## Deliverables

- [ ] `app/api/main.py` — FastAPI app, type-annotated, logging, no print.
- [ ] `app/api/config.py` — `pydantic-settings`, all config from env.
- [ ] `app/api/models.py` — Pydantic request/response models.
- [ ] `app/worker/main.py` — asyncio worker, reconnect on error, logging.
- [ ] `app/worker/config.py` — `pydantic-settings`.
- [ ] `Dockerfile.api` — multi-stage, non-root, healthcheck, no secrets.
- [ ] `Dockerfile.worker` — multi-stage, non-root.
- [ ] `docker-compose.yml` — 4 services, network, volumes, health checks, env_file.
- [ ] `Makefile` — all targets listed above.
- [ ] `.env.example` — all variables documented, no values.
- [ ] `.dockerignore` — excludes sensitive and unnecessary files.
- [ ] `.gitignore` — includes `.env`, `secrets/`.
- [ ] `tests/test_api.py` — unit tests (mocked dependencies).
- [ ] `tests/test_integration.py` — integration test (real stack, skippable in CI without Docker).
- [ ] `README.md` — setup, env vars, how to run, how to test, architecture diagram.

## Grading

| Criterion | Points |
|---|---|
| Stack starts cleanly with `make up`; all services healthy within 90s | 20 |
| `POST /tasks` enqueues correctly; worker processes and logs | 15 |
| `GET /tasks/{job_id}` returns correct status from Postgres | 10 |
| Multi-stage builds: final images contain no pip/build tools (verified with `docker run --rm <image> pip --version`) | 10 |
| Non-root user in both images (verified with `docker compose exec api whoami`) | 10 |
| No secrets in image history (verified with `docker image history --no-trunc`) | 10 |
| Health checks pass; `depends_on` uses `condition: service_healthy` | 10 |
| `make test` passes with all tests green | 10 |
| Code quality: type hints, logging not print, `ruff check` clean | 5 |

Pass threshold: 70 / 100.

## Hints (not solutions)

- Start by writing the Python application code and testing it locally without Docker. Get the logic right before adding container complexity.
- The `--prefix=/install` flag on `pip install` lets you copy the installed packages cleanly to the runtime stage without other files in `site-packages`.
- `python:3.12-slim` does not include `curl`. For health checks in the Dockerfile `HEALTHCHECK` instruction, use `python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"`.
- For the read-only API container, run it once without `--read-only` first, submit a few requests, then run `docker diff <container>` to discover which paths the application writes to. Those are the paths that need `tmpfs` mounts.
- The `pg_isready` command is available in the `postgres:16-alpine` image and is the correct tool for the Postgres health check.
- If `docker compose up` exits immediately with a non-zero code, run `docker compose logs <service>` before reaching for anything else. 90% of startup failures are visible in the first 10 log lines.
- Use `docker inspect --format='{{json .State.Health}}' <container>` rather than `docker compose ps` when a health check is stuck — it shows the actual output and exit code of the last 5 check invocations.
- The Makefile `shell-api` target is useful for live debugging inside the container. Run `env | sort` inside to verify that all expected environment variables are present.
