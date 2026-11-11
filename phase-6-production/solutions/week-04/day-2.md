# Solutions — Day 2: Dockerfile Best Practices

## Exercise 1 — Cache invalidation from instruction ordering

### Setup: deliberately broken ordering

```dockerfile
# Dockerfile.bad-order — DO NOT USE in production; illustrates the cache problem
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ❌ Wrong: source copied before dependencies
# Any code change invalidates the pip install layer
COPY app/ ./app/
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Measuring the difference

```bash
# Build both variants — first build always misses cache
docker build -f Dockerfile.bad-order -t app:bad-order .
docker build -f Dockerfile -t app:good-order .   # the lesson's Dockerfile

# Simulate a code change (what a developer does 20 times/day)
echo "# trivial change" >> app/main.py

# Rebuild bad ordering — watch the output
time docker build --no-cache=false -f Dockerfile.bad-order -t app:bad-order .
# Step 4/7 : COPY app/ ./app/      <-- MISS (content changed)
# Step 5/7 : COPY requirements.txt ./ <-- MISS (preceding layer changed)
# Step 6/7 : RUN pip install ...   <-- MISS (must reinstall all packages)
# Time: ~45 seconds (network + pip)

# Rebuild good ordering — watch the cache hits
time docker build --no-cache=false -f Dockerfile -t app:good-order .
# Step 5/8 : COPY requirements.txt ./  <-- CACHED (req.txt unchanged)
# Step 6/8 : RUN pip install ...       <-- CACHED (req.txt unchanged)
# Step 7/8 : COPY app/ ./app/          <-- MISS (code changed — correct)
# Time: ~2 seconds (only source copy)
```

### Written explanation

Docker invalidates the build cache the moment any layer's input changes, and **every subsequent layer is also invalidated** regardless of whether its own input changed. When `COPY app/` precedes `COPY requirements.txt`, a one-line edit to `main.py` changes the checksum of the first `COPY` instruction. This forces a cache miss on `requirements.txt` copy and on the `pip install` step even though neither `requirements.txt` nor its contents changed at all.

For a team committing 20 times a day, that means 20 full `pip install` runs per developer per day — each taking 30–90 seconds depending on the dependency tree. On a team of 5 engineers that is 100 wasted pip installs and potentially 90+ minutes of cumulative build time, every single day. With the correct ordering (`COPY requirements.txt` → `RUN pip install` → `COPY app/`), the pip layer is rebuilt only when a dependency actually changes — perhaps once per sprint. The CI pipeline benefits equally: faster builds mean shorter feedback loops and lower compute costs.

---

## Exercise 2 — ENTRYPOINT/CMD split and argument override

### What happens with the current Dockerfile

```dockerfile
ENTRYPOINT ["python", "-m", "uvicorn", "app.main:app"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
```

When the user runs:

```bash
docker run fastapi-app:latest --workers 4
```

Docker treats `--workers 4` as a **replacement for `CMD`**, not an addition. The final command executed is:

```
python -m uvicorn app.main:app --workers 4
```

The `--host 0.0.0.0` and `--port 8000` defaults are **gone**. The server binds to `127.0.0.1:8000` (uvicorn's default), which is unreachable from outside the container. This is a silent misconfiguration — the container starts but no traffic reaches the app.

### Correct behaviour explained

With `ENTRYPOINT` in exec form + `CMD` as default arguments, the user can:

- Run with defaults: `docker run fastapi-app:latest` → runs `python -m uvicorn app.main:app --host 0.0.0.0 --port 8000`
- Override all args: `docker run fastapi-app:latest --workers 4` → runs `python -m uvicorn app.main:app --workers 4` (host/port defaults lost)
- Override with all needed args: `docker run fastapi-app:latest --host 0.0.0.0 --port 8000 --workers 4` → correct

The user must supply all required flags when overriding CMD. This is intentional: CMD is a complete set of default arguments, not a base that gets extended.

### Rewritten Dockerfile with safe defaults baked into ENTRYPOINT

If the host and port should always be `0.0.0.0:8000` and the user should only be able to add uvicorn tuning flags, move those into `ENTRYPOINT`:

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ARG APP_PORT=8000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_PORT=${APP_PORT}

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE ${APP_PORT}

RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
USER appuser

# Host and port are fixed in ENTRYPOINT — they cannot be overridden by CMD
# CMD provides only additional/optional flags; empty CMD is fine too
ENTRYPOINT ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD []
```

### Correct docker run invocations

```bash
# Default: host=0.0.0.0, port=8000, workers=1
docker run -p 8000:8000 fastapi-app:latest

# Add workers — host and port are still enforced by ENTRYPOINT
docker run -p 8000:8000 fastapi-app:latest --workers 4

# Add logging level
docker run -p 8000:8000 fastapi-app:latest --workers 4 --log-level warning

# The user cannot accidentally drop --host/--port because they are in ENTRYPOINT
```

### When to keep CMD as default args (the original pattern)

Use `ENTRYPOINT` + `CMD` as default args when you want the user to have full control over all flags. Document in the README that host/port must be passed if overriding CMD. The pattern from the lesson's Dockerfile is still correct — the exercise just surfaces the implication so operators understand the behaviour.

---

## Exercise 3 — .dockerignore for a mixed-content project

### Project layout

```
myproject/
  app/
    main.py
    models.py
  data/           ← large CSV files — EXCLUDE
  notebooks/      ← Jupyter notebooks — EXCLUDE
  tests/          ← pytest fixtures used at runtime — KEEP
  .env            ← secrets — EXCLUDE
  requirements.txt
  Dockerfile
  .dockerignore
```

### .dockerignore

```
# ── Secrets — never ship these ───────────────────────────────────────────────
.env
.env.*
secrets/
*.pem
*.key

# ── Large data files ─────────────────────────────────────────────────────────
data/
*.csv
*.parquet
*.feather

# ── Notebooks — not part of the runtime app ─────────────────────────────────
notebooks/
*.ipynb

# ── Python build artefacts ───────────────────────────────────────────────────
__pycache__
*.pyc
*.pyo
*.pyd
*.egg-info/
dist/
build/
.eggs/

# ── Virtual environments ─────────────────────────────────────────────────────
venv/
.venv/
env/
.env.d/

# ── Test cache / coverage artefacts (NOT the tests/ directory itself) ────────
.pytest_cache/
.coverage
coverage.xml
htmlcov/
.mypy_cache/
.ruff_cache/

# ── Version control ──────────────────────────────────────────────────────────
.git/
.gitignore
.gitattributes

# ── Editor / OS noise ────────────────────────────────────────────────────────
.DS_Store
.idea/
.vscode/
*.swp
*.swo
Thumbs.db

# ── Documentation ────────────────────────────────────────────────────────────
docs/
*.md
!README.md

# ── CI / deployment configs not needed in image ──────────────────────────────
.github/
Makefile
docker-compose*.yml
```

### Why each group is excluded

| Group | Reason |
|---|---|
| `.env`, `secrets/` | Contains credentials. If included in the build context, any `COPY . .` instruction would bake them into an image layer — potentially visible via `docker history`. Even if never `COPY`'d explicitly, sending them to the daemon is unnecessary risk. |
| `data/`, `*.csv` | Large files (can be GB) slow down `docker build` even if not `COPY`'d, because the entire build context is serialised and sent to the Docker daemon over a socket. Exclude everything that isn't needed in the runtime image. |
| `notebooks/` | Development artefacts. Not part of the production application. May contain data, API keys in outputs, or large cell outputs. |
| `__pycache__`, `*.pyc` | Compiled bytecode specific to the build machine's Python version. If included and `COPY`'d, they could override the container's bytecode or cause subtle import errors. Let Python regenerate them inside the container. |
| `venv/`, `.venv/` | The local virtual environment contains the host's compiled C extensions, which may be ABI-incompatible with the container's glibc/musl. Always install dependencies inside the container via `pip install`. |
| `.pytest_cache/`, `.mypy_cache/` | Tool caches. No runtime value. Add noise and size to the context. |
| `.git/` | Can be hundreds of MB. Contains full commit history — no runtime value and a potential information-disclosure risk. |
| `*.md` (except `README.md`) | Documentation. Excluded to keep the image focused. `README.md` is kept as a convention (many images include it for `docker inspect` users). |

Note: `tests/` is **not** excluded because the exercise specifies that it contains fixtures used at runtime. If tests were pure CI artefacts, they would be excluded too.

### Verification

```bash
# Build the image
docker build -t myproject:test .

# Confirm data/ CSV files are absent
docker run --rm myproject:test find / -name "*.csv" 2>/dev/null
# Expected: no output

# Confirm notebooks are absent
docker run --rm myproject:test find / -name "*.ipynb" 2>/dev/null
# Expected: no output

# Confirm tests/ is present (fixture used at runtime)
docker run --rm myproject:test ls /app/tests/
# Expected: fixture files listed

# Confirm .env is absent
docker run --rm myproject:test ls -la /app/.env 2>&1 | head
# Expected: "No such file or directory"
```
