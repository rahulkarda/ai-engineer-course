# Solutions — Day 3: Multi-stage Builds + Image Size Optimization

## Exercise 1 — Add a test stage to the multi-stage Dockerfile

### The requirement

The `runtime` stage must:
- Still copy only from `builder` (not from `test`)
- Contain no pytest installation

The `test` stage must:
- Be buildable as a target (`--target test`)
- Run `pytest tests/` using the builder venv

### Solution: three-stage Dockerfile

```dockerfile
# syntax=docker/dockerfile:1
# Dockerfile.multistage-with-tests

# ── builder: installs all dependencies (including dev deps for tests) ─────────
FROM python:3.12-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt ./

# Production venv — contains only runtime deps, no pytest
RUN python -m venv /build/venv \
    && /build/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /build/venv/bin/pip install --no-cache-dir -r requirements.txt

# ── test: extends builder with dev deps and runs the test suite ───────────────
# This stage is NOT referenced by runtime — it is a dead-end branch.
FROM builder AS test

COPY requirements-dev.txt ./

# Install dev dependencies (pytest, coverage, etc.) into a separate test venv
# so they never contaminate the production venv that runtime copies.
RUN python -m venv /build/venv-test \
    && /build/venv-test/bin/pip install --no-cache-dir --upgrade pip \
    && /build/venv-test/bin/pip install --no-cache-dir -r requirements.txt \
    && /build/venv-test/bin/pip install --no-cache-dir -r requirements-dev.txt

COPY app/ ./app/
COPY tests/ ./tests/

# Run tests as the default command for this stage
CMD ["/build/venv-test/bin/pytest", "tests/", "-v"]

# ── runtime: copies only the production venv from builder ────────────────────
# Does NOT depend on test — runtime is always buildable without running tests.
FROM gcr.io/distroless/python3-debian12:nonroot AS runtime

WORKDIR /app

COPY --from=builder /build/venv /app/venv
COPY app/ ./app/

ENV PATH="/app/venv/bin:$PATH"
EXPOSE 8000

CMD ["/app/venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### requirements-dev.txt (example)

```
pytest>=8.0
pytest-cov>=5.0
httpx>=0.27        # for FastAPI TestClient
```

### Build and verify

```bash
# Build only the runtime target (default — what CI ships)
docker build --target runtime -t app:runtime .

# Build and run tests in the test stage (CI test step)
docker build --target test -t app:test .
docker run --rm app:test
# Expected: pytest output, all tests pass

# Verify pytest is NOT in the runtime image
docker run --rm app:runtime /app/venv/bin/pip show pytest 2>&1
# Expected: WARNING: Package(s) not found: pytest
# (or command not found for distroless — confirming absence)

# Verify distroless has no shell (expected error)
docker run --rm app:runtime bash 2>&1
# Expected: exec: "bash": executable file not found
```

### Why the stages are independent

The DAG of stages is:

```
builder ──┬──► test     (dead-end; never referenced by runtime)
          └──► runtime  (production image; copies only /build/venv)
```

`runtime` copies `--from=builder`, not `--from=test`. BuildKit evaluates only the stages needed for the requested `--target`. When building `--target runtime`, the `test` stage is not evaluated at all — its `pytest` run is skipped entirely. This is intentional: CI controls when tests run; the production image build should not be gated on tests passing (that is the CI pipeline's job, not the Dockerfile's).

---

## Exercise 2 — Refactor with --mount=type=cache and proper COPY ordering

### The colleague's broken Dockerfile

```dockerfile
# Dockerfile.colleague — deliberately bad
FROM python:3.12-slim

WORKDIR /app

# ❌ COPY all source first — any code change invalidates pip
COPY . .
RUN pip install -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Measuring the before state

```bash
docker build -f Dockerfile.colleague -t app:colleague .

# Simulate a developer change
echo "# comment" >> app/main.py

# Time the rebuild
time docker build -f Dockerfile.colleague -t app:colleague .
# COPY . .       <-- MISS (source changed)
# RUN pip install <-- MISS (must reinstall everything)
# real  0m 47.3s   (network + pip install for 20 packages)
```

### Refactored Dockerfile with cache mount

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Step 1: copy only the dependency manifest (changes infrequently)
COPY requirements.txt ./

# Step 2: install with a persistent cache mount
# --mount=type=cache: the /root/.cache/pip directory is bind-mounted from a
# host-side BuildKit cache. Downloaded wheels are stored there across builds.
# The mount is NOT committed to the image layer — zero size impact.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install -r requirements.txt

# Step 3: copy source last — code changes do NOT invalidate pip layer
COPY app/ ./app/

EXPOSE 8000

RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Measuring the after state

```bash
# First build: pip downloads packages and populates the cache mount
DOCKER_BUILDKIT=1 docker build -f Dockerfile -t app:refactored .
# real  0m 48.1s  (first build: same, cache is empty)

# Simulate a code change
echo "# comment" >> app/main.py

# Second build: only the COPY app/ layer is rebuilt
DOCKER_BUILDKIT=1 time docker build -f Dockerfile -t app:refactored .
# COPY requirements.txt  <-- CACHED
# RUN pip install        <-- CACHED (requirements unchanged)
# COPY app/              <-- MISS  (source changed)
# real  0m  1.8s   (only the COPY — no pip run)

# Now simulate a dependency change
echo "httpx>=0.27" >> requirements.txt

# Third build: pip install runs but uses cached wheels
DOCKER_BUILDKIT=1 time docker build -f Dockerfile -t app:refactored .
# RUN pip install  <-- MISS (requirements.txt changed)
#   but httpx wheel is in /root/.cache/pip — downloaded in ~0.1s instead of ~3s
# real  0m  8.2s   (faster than cold install — wheel cache hit)
```

### Why --mount=type=cache does not increase image size

A `--mount=type=cache` bind-mount is attached to the container's filesystem **only during the RUN step** that declares it. When the `RUN` command exits, the mount is detached. BuildKit snapshots the container's writable layer **without the mount point's contents** — the cache directory is never part of the committed layer. This is fundamentally different from `pip install` without `--no-cache-dir`, where pip's wheel cache would be written into the layer and committed.

### BuildKit requirement

`--mount=type=cache` requires BuildKit. Enable it with:

```bash
# Option 1: environment variable (older Docker)
DOCKER_BUILDKIT=1 docker build .

# Option 2: docker buildx (always uses BuildKit)
docker buildx build .

# Option 3: set as default in daemon config
# Add to /etc/docker/daemon.json:
# { "features": { "buildkit": true } }
```

BuildKit is the default builder since Docker 23.0. If you're on Docker 20.x or older, use `DOCKER_BUILDKIT=1`.

---

## Exercise 3 — Debugging a distroless container without a shell

### Reproducing the error

```bash
# Build the distroless image from the worked example
docker build -f Dockerfile.multistage -t app:distroless .

# Start the container
docker run -d --name distro-debug -p 8000:8000 app:distroless

# Try to exec into it
docker exec -it distro-debug bash
```

**Error output:**

```
OCI runtime exec failed: exec failed: unable to start container process:
exec: "bash": executable file not found in $PATH: unknown
```

Or more precisely with distroless nonroot:

```
Error response from daemon: No such exec instance
```

### Why this error occurs

Distroless images are built from scratch with only the minimal runtime dependencies: the Python interpreter, its stdlib, and `glibc`. There is no shell (`bash`, `sh`, `dash`), no package manager (`apt`, `apk`), no coreutils (`ls`, `cat`, `find`), and no user-space tools of any kind. This is intentional: eliminating the shell eliminates an entire class of container escape techniques (arbitrary command execution via shell injection). The attack surface is limited to the Python runtime and your application code.

`docker exec` works by launching a new process inside the container's namespaces. When you ask for `bash`, Docker calls `execve("/usr/bin/bash", ...)` — and `bash` does not exist in the distroless filesystem, so the kernel returns `ENOENT`.

### Debugging techniques without a shell

**Technique 1: docker cp — copy files out for inspection**

```bash
# Copy the application source out to the host for inspection
docker cp distro-debug:/app/app ./debug-app-copy

# Inspect config files
ls ./debug-app-copy/

# Copy logs if the app writes to a file
docker cp distro-debug:/app/app.log ./debug-app.log 2>/dev/null || echo "No log file"
```

**Technique 2: docker inspect — examine container state**

```bash
# Full container metadata (mounts, env, network config, exit code)
docker inspect distro-debug | python3 -m json.tool

# Just the environment variables
docker inspect distro-debug --format '{{range .Config.Env}}{{println .}}{{end}}'

# Exit code and error details
docker inspect distro-debug --format 'State: {{.State.Status}}, Exit: {{.State.ExitCode}}'

# Check if the container restarted (crash loop indicator)
docker inspect distro-debug --format 'RestartCount: {{.RestartCount}}'
```

**Technique 3: docker logs — read stdout/stderr**

```bash
# Follow logs (most useful for running containers)
docker logs -f distro-debug

# Show last 50 lines with timestamps
docker logs --tail 50 --timestamps distro-debug
```

**Technique 4: override with a debug variant that has a shell**

For deep debugging, build a parallel debug image with a shell:

```dockerfile
# Dockerfile.debug — NOT for production
FROM gcr.io/distroless/python3-debian12:debug AS debug

COPY --from=builder /build/venv /app/venv
COPY app/ ./app/

ENV PATH="/app/venv/bin:$PATH"
WORKDIR /app

# debug tag includes busybox — can exec with /busybox/sh
CMD ["/app/venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build --target debug -f Dockerfile.debug -t app:debug .
docker run -d --name distro-debug-shell -p 8001:8000 app:debug
docker exec -it distro-debug-shell /busybox/sh
# Now you have a busybox shell for debugging
```

**Technique 5: ephemeral sidecar debug container (Docker 26+)**

Docker 26 introduced `--pid` sharing and `nsenter`. You can attach a debugging container to the same namespaces:

```bash
# Get the PID of the distroless container's root process
TARGET_PID=$(docker inspect distro-debug --format '{{.State.Pid}}')

# Run a debug container that shares the target's namespaces
docker run --rm -it \
  --pid=container:distro-debug \
  --net=container:distro-debug \
  --cap-add SYS_PTRACE \
  busybox \
  sh -c "nsenter -t ${TARGET_PID} -m -u -i -n sh"
```

### Summary of debugging options

| Technique | When to use |
|---|---|
| `docker logs` | Application crashed or is misbehaving — first thing to check |
| `docker inspect` | Verify config (env vars, mounts, network, restart count) without code |
| `docker cp` | Extract files (logs, configs, generated artefacts) for local inspection |
| Debug image variant (`:debug` tag) | When you need an interactive shell for live debugging |
| Sidecar nsenter | Live debugging of a running container without rebuilding |
| `docker stats` | CPU/memory/IO profiling of a running container |
