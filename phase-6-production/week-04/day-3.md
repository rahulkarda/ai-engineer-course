# Phase 6 · Week 04 · Day 3 — Multi-stage Builds + Image Size Optimization

> Objective: Shrink production images by separating build tools from runtime and applying systematic size-reduction techniques.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

## Concept

A naive Python image built with `FROM python:3.12` ends up shipping compilers, header files, test suites, and documentation that are needed only at build time. The resulting image is 1–2 GB. Production containers need none of that.

**Multi-stage builds** solve this with a two-phase approach:

1. **Builder stage** — starts from a full image, installs build tools, compiles extensions, runs `pip install`.
2. **Runtime stage** — starts from a minimal base image, copies only the final artefacts from the builder. Build tools never appear in the runtime image.

```dockerfile
FROM python:3.12 AS builder
# ... install everything, compile ...

FROM python:3.12-slim AS runtime
COPY --from=builder /app/venv /app/venv
# Build tools absent from this stage
```

`COPY --from=<stage>` copies files between stages. Only the `runtime` stage's filesystem is shipped in the final image.

**Base image choices** (roughly smallest to largest):

| Base | Size | Contents | Use when |
|---|---|---|---|
| `scratch` | 0 B | Nothing | Statically compiled binaries only |
| `distroless/python3` (Google) | ~50 MB | Python runtime, no shell | Production Python, max security |
| `python:3.12-alpine` | ~60 MB | Python + musl libc + BusyBox | Scripts, avoid C extensions (musl ABI issues) |
| `python:3.12-slim` | ~130 MB | Python + minimal Debian | Most Python apps, best balance |
| `python:3.12` | ~1 GB | Python + full Debian + build tools | Building, dev |

Alpine is tempting but causes subtle issues: musl libc is binary-incompatible with glibc-compiled wheels. Many Python packages (numpy, cryptography, psycopg2) ship glibc wheels on PyPI; Alpine forces recompilation from source, which requires installing build tools anyway — negating the size advantage and adding build time.

## How it works internally

**Build cache mechanics**

The build cache key for each instruction is a hash of:
- The instruction text.
- For `COPY`/`ADD`: the content checksum of the copied files.
- The cache of the preceding layer.

A cache miss at layer N invalidates all subsequent layers. Two strategies to preserve cache:

1. **Instruction ordering** — least-changing instructions first (covered in Day 2).
2. **`--mount=type=cache`** — bind-mount a persistent cache directory into the `RUN` step without committing it to a layer. The cache survives across builds without appearing in the image.

```dockerfile
# pip cache persists across builds, never enters the image layer
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt
```

```dockerfile
# apt cache similarly
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends gcc
```

**Size-reduction techniques**

- `pip install --no-cache-dir` — prevents pip from storing downloaded wheels in the layer.
- `apt-get ... && rm -rf /var/lib/apt/lists/*` — removes the package index in the same layer.
- `pip install --no-deps` — when you control the dependency tree tightly.
- Remove `__pycache__` and `.pyc` files if using distroless (no interpreter at build time writes them anyway).
- Use `--mount=type=secret` for build secrets (tokens, private registry creds) — they never appear in any layer or `docker history`.

**Measuring image size**

```bash
docker image ls --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}"
```

The `dive` CLI tool shows per-layer size, efficiency score (ratio of unique bytes to total bytes), and which files each layer adds or removes. Install: `brew install dive` or `go install github.com/wagoodman/dive@latest`.

```bash
dive python:3.12-slim
```

## Worked example

Three Dockerfiles showing size progression for the same FastAPI app.

**Stage 1 — Baseline (full image, no optimisation): ~1.1 GB**

```dockerfile
# Dockerfile.full
FROM python:3.12

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY app/ ./app/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Stage 2 — Slim base with cleanup: ~250 MB**

```dockerfile
# Dockerfile.slim
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Stage 3 — Multi-stage with distroless: ~95 MB**

```dockerfile
# Dockerfile.multistage
# syntax=docker/dockerfile:1

# ── builder: full image with gcc for any C extensions ──────────────────────
FROM python:3.12-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt ./

# Install into an isolated venv so we copy a clean, self-contained directory
RUN python -m venv /build/venv \
    && /build/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /build/venv/bin/pip install --no-cache-dir -r requirements.txt

# ── runtime: distroless — no shell, no package manager, minimal attack surface ─
FROM gcr.io/distroless/python3-debian12:nonroot AS runtime

WORKDIR /app

# Copy only the venv and application source from the builder stage
COPY --from=builder /build/venv /app/venv
COPY app/ ./app/

# Distroless has no shell; exec form is required
ENV PATH="/app/venv/bin:$PATH"
EXPOSE 8000

# nonroot base runs as uid 65532 by default
CMD ["/app/venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and compare:

```bash
docker build -f Dockerfile.full    -t app:full    .
docker build -f Dockerfile.slim    -t app:slim    .
docker build -f Dockerfile.multistage -t app:distroless .

docker image ls app --format "table {{.Tag}}\t{{.Size}}"
# TAG          SIZE
# full         1.12GB
# slim         248MB
# distroless   97MB
```

Inspect size with dive:

```bash
# Install once
brew install dive   # or: go install github.com/wagoodman/dive@latest

dive app:distroless
# Shows: efficiency score, wasted space, per-layer file diff
```

Python helper — automate size comparison across builds:

```python
#!/usr/bin/env python3
"""Compare sizes of multiple Docker images."""

import json
import logging
import subprocess
import sys
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


@dataclass
class ImageInfo:
    tag: str
    size_bytes: int

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


def get_image_size(tag: str) -> ImageInfo:
    result = subprocess.run(
        ["docker", "image", "inspect", tag, "--format", "{{.Size}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    size_bytes = int(result.stdout.strip())
    return ImageInfo(tag=tag, size_bytes=size_bytes)


def compare_images(tags: list[str]) -> None:
    infos = [get_image_size(t) for t in tags]
    baseline = infos[0].size_bytes
    log.info("%-40s %10s %12s", "IMAGE", "SIZE (MB)", "VS BASELINE")
    log.info("-" * 65)
    for info in infos:
        ratio = info.size_bytes / baseline * 100
        log.info("%-40s %10.1f %11.0f%%", info.tag, info.size_mb, ratio)


if __name__ == "__main__":
    compare_images(sys.argv[1:] or ["app:full", "app:slim", "app:distroless"])
```

## Exercises

1. Take `Dockerfile.multistage` above and add a third intermediate stage named `test` that runs `pytest tests/` using the builder venv. The `runtime` stage must not depend on `test` — it should still copy only from `builder`. Build the image normally (runtime target) and separately as the test stage. Verify tests run in the test stage and the runtime image contains no pytest installation.

2. A colleague's Dockerfile has `RUN pip install -r requirements.txt` as the second instruction (right after `FROM`), with `COPY . .` as the first `RUN` instruction. Profile the rebuild time after changing one line of application code. Then refactor the Dockerfile to use `--mount=type=cache,target=/root/.cache/pip` and proper COPY ordering. Measure the rebuild time improvement.

3. Build `app:distroless` from the worked example, then try `docker exec -it <container> bash`. Document the error and explain why it occurs. Then demonstrate how you would debug a running distroless container without a shell, using `docker cp` and `docker inspect` instead.

(Solutions: ../solutions/week-04/day-3.md)

## Self-check

- [ ] I can explain the multi-stage build pattern and articulate why the runtime stage is smaller than the builder stage.
- [ ] I can choose between alpine, slim, and distroless base images given a set of application requirements.
- [ ] I can write a Dockerfile that uses `--mount=type=cache` for pip and apt without committing cache to the image.
- [ ] I can implement a Python script that compares image sizes with proper dataclasses, type hints, and logging.
