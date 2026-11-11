# Phase 6 · Week 07 · Day 1 — CI/CD Concepts & Pipeline Stages

> Objective: Understand CI/CD fundamentals and the stages of a modern delivery pipeline.  ·  Time: ~1 hr  ·  Prereqs: [[week-06/day-7]]

## Concept

**Continuous Integration (CI)** means every code change is automatically validated: built, tested, and scanned — within minutes of the commit. **Continuous Delivery (CD)** extends that: every validated artifact is automatically promoted through environments and is *always* releasable. **Continuous Deployment** (a stricter variant) automatically ships to production without a human gate.

Why it matters for AI services: model configs, prompt templates, and inference code change constantly. Without automation, a team of three can't safely release ten times a day.

### Core principles

| Principle | What it means in practice |
|---|---|
| Single source of truth | One repo, one pipeline definition |
| Fast feedback | Lint + unit tests finish in < 5 min |
| Fail fast | First cheap checks (lint) before expensive ones (integration tests) |
| Reproducible builds | Container image = same binary in every environment |
| Immutable artifacts | Never rebuild; promote the same image from dev → staging → prod |

## How it works internally

A pipeline is a **directed acyclic graph** (DAG) of jobs. Each job runs on an isolated ephemeral runner (a container or VM). Jobs communicate only via **artifacts** (uploaded files) or **environment promotion** (a named deployment slot).

```
commit push
    │
    ▼
[lint] ──→ fail fast (seconds)
    │
    ▼
[unit-tests] ──→ coverage gate
    │
    ▼
[build image] ──→ push to registry
    │
    ▼
[security scan] ──→ CVE gate
    │
    ▼
[deploy staging] ──→ smoke tests
    │
    ▼
[deploy prod] ──→ (manual gate or auto)
```

Each stage is **idempotent**: running it twice with the same inputs produces the same output. This is what enables reruns and rollbacks.

### Key concepts

**Artifact**: immutable output of a job (binary, container image, test report). Tagged with the commit SHA.  
**Environment**: a named deployment target (dev/staging/prod) with its own secrets and approval policies.  
**Promotion**: moving the same artifact (not rebuilding it) from a lower to a higher environment.  
**Gate**: a quality check that blocks promotion — coverage %, security score, manual approval.

## Worked example

A minimal pipeline definition — conceptual pseudocode first, then real YAML on Day 2.

```
Pipeline: ai-service
trigger: push to main, push to feature/*

Stage 1 — lint (30 s)
  - ruff check src/
  - mypy src/

Stage 2 — test (2 min)
  - pytest tests/unit --cov=src --cov-fail-under=80
  needs: [lint]

Stage 3 — build (3 min)
  - docker build -t ai-service:$COMMIT_SHA .
  - docker push registry/ai-service:$COMMIT_SHA
  needs: [test]

Stage 4 — scan (2 min)
  - trivy image registry/ai-service:$COMMIT_SHA
  needs: [build]

Stage 5 — deploy-staging (1 min)
  - deploy registry/ai-service:$COMMIT_SHA to staging
  needs: [scan]
  environment: staging

Stage 6 — deploy-prod (manual gate)
  - deploy registry/ai-service:$COMMIT_SHA to prod
  needs: [deploy-staging]
  environment: prod
  requires: manual approval
```

Total wall time for automatic stages: ~8 minutes. A developer gets a red/green signal before making coffee.

### Python quality gate script

```python
#!/usr/bin/env python3
"""CI quality gate: run lint + tests, exit non-zero on failure.

Usage: python scripts/quality_gate.py [--coverage-min 80]
"""
import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run(cmd: list[str], label: str) -> int:
    """Run a subprocess command and return its exit code."""
    logger.info("Running %s: %s", label, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False)  # noqa: S603
    if result.returncode != 0:
        logger.error("%s failed (exit %d)", label, result.returncode)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage-min", type=int, default=80)
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    failures: list[str] = []

    checks: list[tuple[str, list[str]]] = [
        ("ruff", ["ruff", "check", str(repo_root / "src")]),
        ("mypy", ["mypy", str(repo_root / "src")]),
        (
            "pytest",
            [
                "pytest",
                "tests/unit",
                f"--cov=src",
                f"--cov-fail-under={args.coverage_min}",
                "-q",
            ],
        ),
    ]

    for label, cmd in checks:
        code = run(cmd, label)
        if code != 0:
            failures.append(label)

    if failures:
        logger.error("Gate FAILED — %s did not pass: %s", len(failures), failures)
        sys.exit(1)

    logger.info("Gate PASSED")


if __name__ == "__main__":
    main()
```

## Exercises

1. Draw the DAG for a pipeline that runs `lint` and `type-check` in parallel, then `unit-test` (depends on both), then `build`, then `integration-test` (depends on build), with a manual gate before `deploy-prod`.

2. A team ships twice a week manually. Estimate the risk surface (days between deployments, blast radius of each change). How does daily CI/CD change that estimate?

3. Add a `notify-on-failure` stage to the pseudocode pipeline above. Where in the DAG should it run? What triggers it?

4. The `quality_gate.py` script runs all checks sequentially. Refactor it to run `ruff` and `mypy` in parallel using `subprocess.Popen`, collecting both results before proceeding to `pytest`.

5. What makes a build "reproducible"? List three things that can silently break reproducibility in a Python project.

(Solutions: ../solutions/week-07/day-1.md)

## Self-check

- [ ] I can explain the difference between CI, CD (delivery), and CD (deployment)
- [ ] I can sketch a pipeline DAG for a real project
- [ ] I understand what an artifact is and why images should be promoted not rebuilt
- [ ] I can define a quality gate and name three possible gate criteria

## Supplementary links

- [GitHub Actions documentation](https://docs.github.com/en/actions)
- [DORA metrics — key CI/CD KPIs](https://dora.dev/guides/dora-metrics-four-keys/)
- [Trunk-based development](https://trunkbaseddevelopment.com/)
