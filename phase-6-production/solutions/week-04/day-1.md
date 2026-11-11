# Solutions — Day 1: Containers vs VMs + Images & Layers

## Exercise 1 — Pull ubuntu:24.04 and python:3.12-slim, compare shared layers

### Commands

```bash
# Pull both images
docker pull ubuntu:24.04
docker pull python:3.12-slim

# Inspect layer history for each
docker history --no-trunc ubuntu:24.04
docker history --no-trunc python:3.12-slim

# Get the layer digests (content-addressed SHA256)
docker image inspect ubuntu:24.04 --format '{{range .RootFS.Layers}}{{println .}}{{end}}'
docker image inspect python:3.12-slim --format '{{range .RootFS.Layers}}{{println .}}{{end}}'

# Cross-reference — shared digests appear in both lists
comm -12 \
  <(docker image inspect ubuntu:24.04 --format '{{range .RootFS.Layers}}{{println .}}{{end}}' | sort) \
  <(docker image inspect python:3.12-slim --format '{{range .RootFS.Layers}}{{println .}}{{end}}' | sort)
```

### What to look for in docker history output

`python:3.12-slim` is built on top of Debian slim. The base Debian filesystem layers appear at the bottom of the history. If you also have any other Debian-based image locally (e.g., `nginx:bookworm`), those base layers will appear in your local layer cache — Docker does not download them again.

`ubuntu:24.04` uses Ubuntu's own base layers (different distribution, different SHA256s). Typically **zero layers are shared** between `ubuntu:24.04` and `python:3.12-slim` because they derive from different parent trees.

However, **two images that both derive from the same base** (e.g., `python:3.12-slim` and `fastapi:latest` if the latter starts `FROM python:3.12-slim`) share every common base layer.

### Why layer sharing matters at scale

When 50 containers on one host all run images based on `python:3.12-slim`, the ~130 MB base image exists **once on disk** in `/var/lib/docker/overlay2`. Each container's writable layer is a few KB for typical applications. Without layer sharing, 50 containers × 130 MB = 6.5 GB just for base images. With layer sharing, the total is ~130 MB base + negligible per-container overhead — roughly a **50x reduction in disk use**. The same applies to memory: the read-only layers are mapped into memory once via the page cache and shared across all container processes reading those files. This is the core economic argument for container density over VMs.

---

## Exercise 2 — Commit a container to a new image, inspect the committed layer

### Commands

```bash
# Start the container interactively
docker run -it --name layer-demo python:3.12-slim bash

# Inside the container: create the proof file and exit
touch /tmp/proof.txt
exit

# Commit the stopped container to a new image tag
docker commit layer-demo python:3.12-slim-modified

# Compare history: the new layer appears at the top
docker history python:3.12-slim-modified

# Verify the proof file exists in the committed image
docker run --rm python:3.12-slim-modified ls -la /tmp/proof.txt

# Verify the original image is unchanged (file absent)
docker run --rm python:3.12-slim ls /tmp/proof.txt 2>&1 || echo "Not in original image — correct"

# Clean up
docker rm layer-demo
docker rmi python:3.12-slim-modified
```

### Explanation

When you ran `touch /tmp/proof.txt` inside the container, OverlayFS wrote the new file into the container's **writable layer** (the `upperdir`). That writable layer is ephemeral — it lives only as long as the container.

`docker commit` snapshots the writable layer and appends it as a new **read-only layer** to the image stack. The output of `docker history python:3.12-slim-modified` will show a new entry at the top with a small size (a few bytes for the inode metadata of `proof.txt`) and a comment like `buildkit.dockerfile.v0` or a blank `CREATED BY` field — reflecting that it was committed manually rather than built from a Dockerfile instruction.

Key insight: `docker commit` is a debugging and migration tool. **Do not use it for production images.** Committed images are not reproducible (no Dockerfile), not diffable in version control, and accumulate stale state. Always build from a Dockerfile.

---

## Exercise 3 — Python script to report total uncompressed image size

```python
#!/usr/bin/env python3
"""
Report the total uncompressed size of all layers in one or more Docker images.

Usage:
    python3 image_size_report.py python:3.12-slim ubuntu:24.04 alpine:3.20
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
class LayerInfo:
    digest: str
    size_bytes: int

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


@dataclass
class ImageSizeReport:
    image: str
    layers: list[LayerInfo] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(layer.size_bytes for layer in self.layers)

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 * 1024)

    @property
    def layer_count(self) -> int:
        return len(self.layers)


def inspect_image(image: str) -> ImageSizeReport:
    """Run docker image inspect and parse layer size information."""
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
        check=True,
    )
    data: list[dict] = json.loads(result.stdout)
    if not data:
        raise ValueError(f"No inspect data returned for image: {image}")

    inspect = data[0]

    # RootFS.Layers gives digests; GraphDriver.Data gives overlay dirs.
    # The top-level Size field is the total uncompressed size in bytes.
    total_size: int = inspect.get("Size", 0)

    # Per-layer sizes come from docker history JSON format
    history_result = subprocess.run(
        ["docker", "history", "--format", "{{json .}}", "--no-trunc", image],
        capture_output=True,
        text=True,
        check=True,
    )

    layers: list[LayerInfo] = []
    for line in history_result.stdout.strip().splitlines():
        entry: dict = json.loads(line)
        raw_size: str = entry.get("Size", "0B")
        # docker history Size is a human-readable string like "47.4MB" — parse it
        size_bytes = _parse_human_size(raw_size)
        digest = entry.get("ID", "")
        layers.append(LayerInfo(digest=digest, size_bytes=size_bytes))

    # If history parsing yields zero (common with older format), fall back to
    # the top-level Size from inspect (most reliable for the total).
    if not layers:
        layers = [LayerInfo(digest="(aggregate)", size_bytes=total_size)]

    return ImageSizeReport(image=image, layers=layers)


def _parse_human_size(size_str: str) -> int:
    """Convert docker human-readable size string to bytes."""
    size_str = size_str.strip()
    if not size_str or size_str in ("0B", "0"):
        return 0

    units: dict[str, int] = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
        "kB": 1024,
        "KiB": 1024,
        "MiB": 1024**2,
        "GiB": 1024**3,
    }
    for suffix, multiplier in sorted(units.items(), key=lambda x: -len(x[0])):
        if size_str.endswith(suffix):
            try:
                value = float(size_str[: -len(suffix)])
                return int(value * multiplier)
            except ValueError:
                return 0
    # No unit suffix — assume bytes
    try:
        return int(size_str)
    except ValueError:
        return 0


def print_report(reports: list[ImageSizeReport]) -> None:
    """Print a formatted comparison table."""
    log.info("")
    log.info("%-45s %8s %12s", "IMAGE", "LAYERS", "TOTAL SIZE")
    log.info("-" * 70)
    baseline: int | None = None
    for report in reports:
        if baseline is None:
            baseline = report.total_bytes or 1
        reduction = (1 - report.total_bytes / baseline) * 100
        comparison = f"baseline" if report.total_bytes == baseline else f"{reduction:+.0f}%"
        log.info(
            "%-45s %8d %9.1f MB  (%s)",
            report.image,
            report.layer_count,
            report.total_mb,
            comparison,
        )
    log.info("")


def main(images: list[str]) -> None:
    reports: list[ImageSizeReport] = []
    for image in images:
        log.info("Inspecting %s ...", image)
        try:
            report = inspect_image(image)
            reports.append(report)
            log.info(
                "  %d layers, %.1f MB total",
                report.layer_count,
                report.total_mb,
            )
        except subprocess.CalledProcessError as exc:
            log.error("Failed to inspect %s: %s", image, exc.stderr.strip())

    if reports:
        print_report(reports)


if __name__ == "__main__":
    target_images: list[str] = sys.argv[1:] if len(sys.argv) > 1 else [
        "python:3.12-slim",
        "ubuntu:24.04",
        "alpine:3.20",
    ]
    main(target_images)
```

### Sample output

```
Inspecting python:3.12-slim ...
  8 layers, 131.4 MB total
Inspecting ubuntu:24.04 ...
  3 layers, 77.9 MB total
Inspecting alpine:3.20 ...
  1 layers,  7.8 MB total

IMAGE                                         LAYERS   TOTAL SIZE
----------------------------------------------------------------------
python:3.12-slim                                   8   131.4 MB  (baseline)
ubuntu:24.04                                       3    77.9 MB  (-41%)
alpine:3.20                                        1     7.8 MB  (-94%)
```

### Key observations

- `alpine:3.20` achieves its tiny size by using musl libc and BusyBox instead of glibc + GNU coreutils. Only 1 layer.
- `ubuntu:24.04` has fewer layers than `python:3.12-slim` because Python's image adds apt installs and pip bootstrap layers on top of Debian slim.
- `python:3.12-slim` is larger than `ubuntu:24.04` because it includes the Python runtime on top of a Debian base — but it is the correct trade-off for Python applications since the runtime is already included and does not need to be installed in your Dockerfile.

### Run it

```bash
# Pull all three first if not local
docker pull python:3.12-slim ubuntu:24.04 alpine:3.20

# Run the script (use -I to prevent loading code from the current directory)
python3 -I image_size_report.py python:3.12-slim ubuntu:24.04 alpine:3.20
```
