# Phase 6 · Week 04 · Day 7 — Sunday: Project Completion, Debugging, and Assessment Prep

> Objective: Finalize the week project, review Docker best practices and common pitfalls, and consolidate knowledge for the assessment  ·  Time: ~3 hr  ·  Prereqs: [[day-6]]

## Concept

The final day of the week serves two purposes: completing and polishing the week project to a shippable state, and consolidating the week's conceptual knowledge so you can recall it under interview conditions. Debugging skill is as important as build skill — this session covers the systematic approach to diagnosing container failures.

## How it works internally

### Finalizing the project

Before declaring a project done, run through this checklist:

**Code quality**
- [ ] All Python files pass `ruff check` with no errors.
- [ ] All functions and classes have type annotations.
- [ ] No `print()` calls — all output goes through `logging`.
- [ ] No hardcoded secrets, ports, or environment-specific values in source files.
- [ ] `.env.example` documents every variable the application reads.
- [ ] `.env` is in `.gitignore`.

**Docker/Compose**
- [ ] Every service has a health check.
- [ ] No service starts before its dependencies are healthy (`condition: service_healthy`).
- [ ] Application processes run as non-root.
- [ ] No secrets in `ENV` Dockerfile instructions or `docker history`.
- [ ] `.dockerignore` excludes `.env`, `tests/`, `*.md`, `.git`, `__pycache__`.
- [ ] Named volumes for persistent data; tmpfs for ephemeral writable dirs in read-only containers.
- [ ] Services communicate on a named bridge network, not the default network.

**Testing**
- [ ] At least one integration test that runs the full API + Redis stack.
- [ ] All external calls (DB, Redis, LLM) are mocked in unit tests.
- [ ] Tests pass inside the container: `make test` succeeds.

**Documentation**
- [ ] `README.md` contains: setup steps, environment variables, how to run, how to test.
- [ ] Architecture diagram (even ASCII) is present.

---

### Common Docker pitfalls and debugging

**Pitfall 1: Container exits immediately after starting**

Symptom: `docker compose ps` shows `Exited (1)` seconds after `up`.

Diagnosis:
```bash
docker compose logs <service>
```
The exit code is in `docker compose ps`. Exit code 1 usually means the application crashed. Exit code 127 means the `CMD` binary was not found. Exit code 137 means OOM kill.

Common causes:
- Missing environment variable — the app raised `ValidationError` on startup.
- Wrong `CMD` path (typo in module name, missing `__init__.py`).
- Port already in use on the host (exit code 1, logs show `Address already in use`).

**Pitfall 2: Health check stuck in `starting` or transitions to `unhealthy`**

Diagnosis:
```bash
docker inspect --format='{{json .State.Health}}' <container_id>
```
This shows the last 5 check results, each with exit code, stdout, and stderr.

Common causes:
- `--start-period` too short: the application is still initializing when the first check runs. Increase `--start-period`.
- Health endpoint is slow: increase `--timeout`.
- Health check command not found in the image: if the image has no `curl`, use `python` or `wget`.
- Application bound to `127.0.0.1` instead of `0.0.0.0`: the health check from outside the container cannot reach it.

**Pitfall 3: Port conflict**

Symptom: `bind: address already in use`.

Diagnosis:
```bash
lsof -i :<port>
# or on Linux:
ss -tulpn | grep <port>
```

Fix: change the host-side port in Compose (`"8081:8080"`) or stop the conflicting process.

**Pitfall 4: Volume permission errors**

Symptom: application starts but cannot write to a mounted directory; logs show `PermissionError: [Errno 13]`.

Cause: the host directory is owned by root or another UID. The container process runs as UID 1001.

Fix options:
- `chown 1001:1001 ./data` on the host before mounting.
- Use named volumes (Docker manages ownership).
- Add `user: "1001:1001"` to the Compose service to align UIDs.

**Pitfall 5: Network isolation — service cannot reach another service**

Symptom: `Connection refused` or `Name or service not known` inside a container.

Diagnosis:
```bash
docker compose exec api ping redis
docker compose exec api nslookup redis
```

Common causes:
- Services are on different networks (one uses the default `bridge`, another uses a named network).
- Service name in `.env` does not match the Compose service name (e.g., `REDIS_HOST=localhost` instead of `REDIS_HOST=redis`).
- Container is healthy but the application inside is not yet listening.

Fix: ensure all services that need to communicate share a named network. Use the Compose service name as the hostname.

---

### Debugging commands reference

```bash
# Tail logs for a service (follow mode)
docker compose logs -f <service>

# Tail last 50 lines
docker compose logs --tail=50 <service>

# Open a shell in a running container (if sh is available)
docker compose exec <service> /bin/sh

# Run a one-off command in a new container (does not affect the running one)
docker compose run --rm <service> python -c "from app.api.config import get_settings; print(get_settings())"

# Inspect full container metadata (environment, health, mounts, network)
docker inspect <container_id>

# Inspect only health state
docker inspect --format='{{json .State.Health}}' <container_id>

# Resource usage (CPU, memory, network I/O)
docker stats

# Event stream (start, stop, die, health_status events)
docker events --filter type=container

# List all containers including stopped ones
docker ps -a

# View image layer history (check for accidental secrets)
docker image history --no-trunc <image>

# Diff filesystem changes vs the original image
docker diff <container_id>
```

---

### Week review: key concepts

**Containers vs VMs**

VMs virtualize hardware; each VM runs a full OS kernel. Containers share the host kernel; isolation is via Linux namespaces (pid, net, mnt, uts, ipc, user) and cgroups (resource limits). Containers start in milliseconds, VMs in seconds to minutes. Container isolation is weaker than VM isolation.

**Image layers**

Each `RUN`, `COPY`, and `ADD` instruction creates a layer. Layers are content-addressed (SHA256) and cached. A cache miss on layer N invalidates all layers after N. Order instructions from least-frequently-changed (OS deps) to most-frequently-changed (application code). Layer contents are never deleted by later instructions — a secret written in one layer and deleted in the next is still in the image history.

**Multi-stage builds**

`FROM base AS build` followed by `FROM base AS runtime` and `COPY --from=build` produces a final image that contains only what was explicitly copied from the build stage. Build tools, compilers, test dependencies, and intermediate artifacts are not in the final image.

**Compose networking**

Each named network is a Docker bridge network. Services on the same named network can reach each other by service name (Docker's embedded DNS). Services on different networks cannot communicate unless connected to both.

**Health checks**

Status transitions: `starting` (within `--start-period`) -> `healthy` (check passed) or `unhealthy` (failed `--retries` consecutive checks). An `unhealthy` container is still running; the orchestrator decides what to do (restart policy, Swarm/Kubernetes action).

---

### Assessment preparation guide

Know these cold (no notes):

1. The difference between a container and a VM at the kernel level.
2. Why layer ordering matters for build cache efficiency.
3. Why secrets must never appear in `ENV` or `ARG` Dockerfile instructions.
4. What `COPY --from=build` does and why it reduces image size.
5. The four `HEALTHCHECK` parameters and what each controls.
6. How `depends_on: condition: service_healthy` works and what it does not guarantee (it guarantees the health check passed, not that the application is ready to serve traffic — those are the same only if the health check is well-designed).
7. Why running as root in a container is a security risk.
8. What `cap_drop: ALL` does and when you would add back a capability.
9. How Docker's embedded DNS enables service-to-service communication in Compose.
10. The 12-factor rule for config and how `pydantic-settings` implements it.

---

## Worked example

**Systematic debugging walkthrough**

Scenario: `make up` completes but `docker compose ps` shows the `api` service as `unhealthy`.

Step 1 — read the logs:
```bash
docker compose logs api
```
Output shows: `pydantic_settings.env_settings: field required [type=missing, input_value=..., input_url=...]`

Diagnosis: a required environment variable (`DB_DSN`) is missing from `.env`.

Fix: add `DB_DSN=postgresql+asyncpg://appuser:${POSTGRES_PASSWORD}@db/appdb` to `.env`.

Step 2 — inspect health check details:
```bash
docker inspect --format='{{json .State.Health}}' $(docker compose ps -q api)
```
Output shows exit code 1, stderr: `urllib.error.HTTPError: HTTP Error 503: Service Unavailable`.

Diagnosis: the app started but `/health` returns 503 because Redis is not yet ready.

Fix: `--start-period` is too short. Increase to `30s` and add `depends_on: redis: condition: service_healthy`.

Step 3 — verify after fix:
```bash
docker compose down && docker compose up -d
docker compose ps
# api: healthy
```

---

## Exercises

1. Take a Compose stack where the `api` service has no `depends_on` and the `worker` has `depends_on: api` with no `condition`. Start the stack and use `docker events` to observe the startup order. Measure (with `docker logs --timestamps`) whether the worker ever attempts to connect before the API is ready. Then add `condition: service_healthy` to all `depends_on` entries and compare the startup sequence.

2. Introduce a deliberate secret leak: add `ENV DB_PASSWORD=test123` to `Dockerfile.api`, build the image, then run `docker image history --no-trunc taskapi` and `docker inspect` to find where the secret appears. Document every location where it is visible. Then remove the leak, rebuild, and verify it is gone from history. Write a one-paragraph explanation of why `RUN unset DB_PASSWORD` in a later layer does not fix the leak.

(Solutions: ../solutions/week-04/day-7.md)

## Self-check

- What is the difference between `docker compose down` and `docker compose down -v`?
- A container's health check uses `curl`. The final image is `python:3.12-slim`. Will `curl` be available? How do you verify?
- Why does a container running as root represent a higher risk if the container runtime has a kernel vulnerability?
- What does `docker diff <container>` show, and when is it useful for debugging?
- If `service_healthy` in `depends_on` guarantees the health check passed, why might a dependent service still fail to connect immediately after starting?
