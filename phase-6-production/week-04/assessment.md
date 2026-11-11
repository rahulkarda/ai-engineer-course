# Week 04 Assessment

## Conceptual (interview-style)

1. A colleague says "containers are just lightweight VMs." What is technically incorrect about this statement? Explain the actual isolation mechanism (namespaces and cgroups) in two to three sentences.

2. You have a Dockerfile with these instructions in order: `COPY requirements.txt .`, `RUN pip install -r requirements.txt`, `COPY . .`. Your colleague swaps the last two lines to: `COPY . .`, `RUN pip install -r requirements.txt`. What is the build-cache consequence of this change? On which instruction will the cache miss occur most frequently, and why?

3. A developer adds `ENV API_KEY=sk-abc123` to a Dockerfile for testing, then removes it in a later PR with `RUN unset API_KEY`. Is the secret safe in the final image? Explain why or why not, and name two commands that would reveal it.

4. Explain multi-stage builds: what problem they solve, how `COPY --from=<stage>` works, and name two concrete things that are present in the build stage but absent from the runtime image.

5. You have a `docker-compose.yml` with `services.worker.depends_on.db.condition: service_healthy`. The Postgres service has a health check. A colleague argues this means the worker will never fail to connect to Postgres on startup. Identify the flaw in this argument and describe a scenario where the worker could still fail to connect.

6. Describe the four parameters of a `HEALTHCHECK` instruction (`--interval`, `--timeout`, `--retries`, `--start-period`) and what each controls. What container health status is set while the start period is active?

7. A containerized Python application runs as root. List three concrete attack vectors that running as root enables compared to running as UID 1001 with no elevated capabilities. What Dockerfile instructions create a non-root user and switch to it?

8. You run `docker compose up -d` and `docker compose ps` shows all services as `running` but your API returns 502 Bad Gateway when proxied. Name three debugging commands you would run, in order, and what each tells you.

---

## Coding challenge

### Service specification

You are given a Python message-processing service with the following behavior:

- `POST /messages` — accepts `{"content": str, "priority": int}`, enqueues the message to a Redis list named `messages`, returns `{"message_id": "<uuid>", "status": "queued"}`.
- `GET /health` — returns `{"status": "ok"}` if Redis is reachable, `503` otherwise.
- A background worker reads from the `messages` Redis list using `BRPOP`, logs each message (`logging`, not `print`), and increments a `processed_count` counter in Redis.
- All configuration (Redis host/port, queue name) comes from environment variables via `pydantic-settings`. No default values for credentials.

### Deliverables

Produce a directory with the following files:

1. `app/api/main.py` — FastAPI application (type-annotated, logging, no hardcoded values).
2. `app/worker/main.py` — asyncio worker (type-annotated, logging, reconnect on Redis error).
3. `Dockerfile.api` — multi-stage build; final stage is `python:3.12-slim`; non-root user; `HEALTHCHECK` using Python (no curl).
4. `Dockerfile.worker` — multi-stage build; non-root user.
5. `docker-compose.yml` — four services (api, worker, redis); named bridge network; health checks on all services; `depends_on` with `condition: service_healthy`; `env_file: .env`; no hardcoded secrets.
6. `.env.example` — documents every variable; no actual values.
7. `tests/test_api.py` — at least two `pytest` tests: one for `POST /messages` (mock Redis), one for `GET /health` with Redis unreachable returning 503.

### Acceptance criteria

- `docker compose up -d` starts all services; `docker compose ps` shows all as healthy within 60 seconds.
- `curl -s -X POST http://localhost:8080/messages -H "Content-Type: application/json" -d '{"content":"hello","priority":1}'` returns HTTP 202 with a `message_id`.
- Worker logs show the message being processed within 5 seconds.
- `docker image history --no-trunc <api-image>` contains no credentials.
- `docker compose exec api whoami` prints a non-root username.
- `pytest tests/test_api.py` passes with all tests green.
- No `print()` calls in application code. No untyped function signatures.

---

## Rubric (pass >= 75%)

| Dimension | Weight |
|---|---|
| Independent implementation (all files produced without looking up solutions) | 40% |
| Conceptual understanding (conceptual questions answered correctly) | 25% |
| Debugging and problem-solving (correct use of docker logs/inspect/exec/stats) | 20% |
| Code quality and testing (type hints, logging, tests pass, no secrets) | 15% |

Answer key: ../solutions/week-04/assessment.md
