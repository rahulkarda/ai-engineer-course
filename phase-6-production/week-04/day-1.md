# Phase 6 · Week 04 · Day 1 — Containers vs VMs + Images & Layers

> Objective: Understand the isolation model behind Docker containers and how image layers work internally.  ·  Time: ~1 hr  ·  Prereqs: [[phase-6-week-03]]

## Concept

**Virtual Machines** run a full guest OS on top of a hypervisor (Type-1: bare-metal like KVM/VMware ESXi; Type-2: hosted like VirtualBox). Each VM carries its own kernel, drivers, and OS userland — isolation is strong but overhead is high (GBs of RAM, tens of seconds to boot).

**Containers** share the host kernel. Isolation comes from two Linux primitives:

- **Namespaces** — partition kernel resources per process group: `pid`, `net`, `mnt`, `uts`, `ipc`, `user`. A container process sees only its own process tree, network stack, filesystem mount points, and hostname.
- **cgroups (control groups)** — enforce resource limits (CPU shares, memory ceiling, block I/O weight). The kernel scheduler enforces these; no hypervisor needed.

The practical differences:

| Property | VM | Container |
|---|---|---|
| Isolation unit | Full OS | Process namespace |
| Startup time | 10–60 s | < 1 s |
| Memory overhead | 256 MB–4 GB per instance | 5–50 MB per instance |
| Density on one host | Tens | Hundreds–thousands |
| Kernel | Guest kernel (separate) | Shared host kernel |
| Security boundary | Strong (hypervisor) | Moderate (kernel attack surface shared) |

Containers are not "lightweight VMs" — they are isolated process trees. A container process is visible in the host's process table (`ps aux`) with a different PID namespace.

## How it works internally

**Union filesystem (OverlayFS)**

Docker images are stacks of read-only layers stored on disk. OverlayFS merges them into a single coherent filesystem view:

```
Container writable layer  (upperdir)   <-- your writes go here
────────────────────────────────────
Layer 3: COPY ./app /app               (read-only)
Layer 2: RUN pip install -r req.txt    (read-only)
Layer 1: RUN apt-get install ...       (read-only)
Layer 0: python:3.12-slim base image   (read-only)
```

When a container reads a file, the kernel looks from the top down and returns the first match. When a container **writes** a file that exists in a lower layer, OverlayFS copies the file up to the writable layer first (**copy-on-write**), then modifies it. The original lower layer is untouched.

This means:
- Multiple containers sharing the same image share the read-only layers in memory and on disk — zero duplication.
- A container's writable layer disappears when the container is removed (unless a named volume is used).
- Image layers are content-addressed by SHA256 digest. `docker pull` skips layers already present locally.

**Image anatomy**

- **Base image** (`FROM python:3.12-slim`) — the bottom layer, usually a minimal OS + runtime.
- **Intermediate layers** — each `RUN`, `COPY`, `ADD` instruction in a Dockerfile creates one layer.
- **Writable container layer** — allocated at `docker run` time, ephemeral, not part of the image.

## Worked example

```bash
# Pull a specific image (explicit digest pins the exact content)
docker pull python:3.12-slim

# List local images — note SIZE column
docker image ls python

# Inspect full metadata as JSON
docker image inspect python:3.12-slim

# Show every layer and the command that created it
# --no-trunc shows full commands
docker history --no-trunc python:3.12-slim

# Run an interactive container — removed automatically on exit
docker run --rm -it python:3.12-slim bash

# Inside the container: verify kernel sharing
uname -r          # same kernel version as the host
ps aux            # only processes in this namespace visible
cat /etc/os-release   # Debian slim — different from host OS

# From another terminal: inspect the running container
docker ps
docker inspect <container_id> | python3 -m json.tool | head -60

# Run a named, detached container with a published port and env var
docker run \
  -d \
  -p 8080:8000 \
  -e APP_ENV=development \
  --name my-app \
  python:3.12-slim \
  python3 -m http.server 8000

# Exec into a running container
docker exec -it my-app bash

# Stop and remove
docker stop my-app && docker rm my-app
```

Layer inspection script — shows cumulative size growth per layer:

```python
#!/usr/bin/env python3
"""Inspect Docker image layers and print size breakdown."""

import json
import logging
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def inspect_layers(image: str) -> None:
    result = subprocess.run(
        ["docker", "history", "--format", "{{json .}}", image],
        capture_output=True,
        text=True,
        check=True,
    )
    layers = [json.loads(line) for line in result.stdout.strip().splitlines()]
    log.info("%-12s  %-60s  %s", "SIZE", "CREATED BY", "COMMENT")
    log.info("-" * 90)
    for layer in layers:
        created_by = (layer.get("CreatedBy") or "")[:60]
        size = layer.get("Size", "0B")
        comment = layer.get("Comment", "")
        log.info("%-12s  %-60s  %s", size, created_by, comment)


if __name__ == "__main__":
    image = sys.argv[1] if len(sys.argv) > 1 else "python:3.12-slim"
    inspect_layers(image)
```

Run with:

```bash
python3 inspect_layers.py python:3.12-slim
```

## Exercises

1. Pull `ubuntu:24.04` and `python:3.12-slim`. Run `docker history` on both. Identify which layers are shared (same SHA256). Explain in one paragraph why layer sharing matters for disk usage on a host running 50 containers.

2. Start a container from `python:3.12-slim`, create a file inside it (`touch /tmp/proof.txt`), then `docker commit` it to a new image tag. Run `docker history` on the new image and locate the layer your commit created. Verify the writable layer is now a read-only layer in the committed image.

3. Write a Python script (type-annotated, uses `subprocess`, logs via `logging`) that accepts an image name as a CLI argument and prints the total uncompressed size of all its layers by parsing `docker image inspect` JSON output. Run it against three different images and compare.

(Solutions: ../solutions/week-04/day-1.md)

## Self-check

- [ ] I can explain the difference between namespace isolation and hypervisor isolation without looking at notes.
- [ ] I can describe the OverlayFS copy-on-write mechanism and what happens to a container's writable layer on `docker rm`.
- [ ] I can use `docker history`, `docker image inspect`, and `docker exec` confidently.
- [ ] I can implement a script that parses `docker inspect` JSON output with proper type hints and logging.
