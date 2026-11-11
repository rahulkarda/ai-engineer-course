# Solutions — Week 07 · Day 1

## Exercise 1 — DAG with parallel lint/type-check

```
[lint] ──┐
         ├──→ [unit-test] ──→ [build] ──→ [integration-test] ──→ [manual-gate] ──→ [deploy-prod]
[type-check] ──┘
```

In GitHub Actions YAML:
```yaml
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - run: ruff check src/

  type-check:
    runs-on: ubuntu-latest
    steps:
      - run: mypy src/

  unit-test:
    needs: [lint, type-check]   # both must pass
    runs-on: ubuntu-latest
    steps:
      - run: pytest tests/unit

  build:
    needs: [unit-test]
    runs-on: ubuntu-latest
    steps:
      - run: docker build .

  integration-test:
    needs: [build]
    runs-on: ubuntu-latest
    steps:
      - run: pytest tests/integration

  deploy-prod:
    needs: [integration-test]
    environment: production     # manual gate
    runs-on: ubuntu-latest
    steps:
      - run: echo "Deploy"
```

---

## Exercise 2 — Risk surface analysis

**Manual twice-weekly deploy:**
- Change batch size: ~3–4 days of commits per deploy.
- Blast radius: all changes from those days ship together. A breaking change is buried among many others — hard to identify and isolate.
- Mean time to detection: a bug introduced on day 1 may not be observed until deploy day 4, plus any investigation time.
- Mean time to recovery: requires identifying the offending commit in a large batch, often a partial rollback is impossible.

**Daily CI/CD:**
- Change batch: 1–10 commits per deploy, often a single logical change.
- Blast radius: one change at a time, trivially bisectable.
- Detection: automated tests catch regressions within minutes of commit. Production issues surface same day.
- Recovery: rollback targets one commit; previous image SHA is always known.

The **deployment frequency** (DORA key metric) directly correlates with lower change failure rate — smaller batches are inherently safer.

---

## Exercise 3 — notify-on-failure placement

```
[lint] → [test] → [build] → [scan] → [deploy-staging] → [deploy-prod]
                                                                │
                                                       [notify-on-failure]
                                                       if: failure()
```

The `notify-on-failure` job should:
- Declare `needs: [lint, test, build, scan, deploy-staging, deploy-prod]`.
- Use `if: failure()` — GitHub evaluates this even when upstream jobs fail.
- Use `always()` combined with `failure()` if it must also run when previous jobs are skipped.

```yaml
notify-on-failure:
  runs-on: ubuntu-latest
  needs: [lint, test, build, scan, deploy-staging, deploy-prod]
  if: failure()
  steps:
    - name: Notify Slack
      env:
        SLACK_WEBHOOK: ${{ secrets.SLACK_WEBHOOK }}
      run: |
        curl -X POST "$SLACK_WEBHOOK" \
          -H 'Content-type: application/json' \
          --data "{\"text\": \"Pipeline failed: ${{ github.workflow }} run ${{ github.run_id }}\"}"
```

---

## Exercise 4 — Parallel ruff + mypy

```python
#!/usr/bin/env python3
"""CI quality gate with parallel lint/type-check."""
import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage-min", type=int, default=80)
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent

    # Launch ruff and mypy in parallel
    ruff_proc = subprocess.Popen(  # noqa: S603
        ["ruff", "check", str(repo_root / "src")]
    )
    mypy_proc = subprocess.Popen(  # noqa: S603
        ["mypy", str(repo_root / "src")]
    )

    ruff_rc = ruff_proc.wait()
    mypy_rc = mypy_proc.wait()

    failures: list[str] = []
    if ruff_rc != 0:
        failures.append("ruff")
        logger.error("ruff failed")
    if mypy_rc != 0:
        failures.append("mypy")
        logger.error("mypy failed")

    if failures:
        logger.error("Static analysis failed: %s", failures)
        sys.exit(1)

    # Only run pytest if static analysis passed
    pytest_rc = subprocess.run(  # noqa: S603
        [
            "pytest",
            "tests/unit",
            "--cov=src",
            f"--cov-fail-under={args.coverage_min}",
            "-q",
        ]
    ).returncode

    if pytest_rc != 0:
        logger.error("pytest failed")
        sys.exit(1)

    logger.info("Gate PASSED")


if __name__ == "__main__":
    main()
```

---

## Exercise 5 — What makes a build reproducible?

A build is reproducible if the same source inputs always produce the same binary output.

**Three common silent reproducibility breakers in Python:**

1. **Unpinned dependencies**: `requirements.txt` contains `fastapi>=0.100` — a new release appears between the dev build and the prod build, installing a different version with different behavior. Fix: pin with `pip-compile` or lock files (`uv lock`).

2. **`latest` base image**: `FROM python:3.12` resolves to different patch versions (3.12.3 vs 3.12.4) on different days. A library update in the base image can introduce a CVE or a behavior change. Fix: use digest-pinned base images: `FROM python:3.12@sha256:abc123...`.

3. **Non-deterministic filesystem ordering**: Python's `dict` is insertion-ordered since 3.7, but older pattern-match code or `os.listdir()` returns files in OS-dependent order. If your build step generates code from file scans, the output order can differ. Fix: always sort inputs before processing (`sorted(Path(".").glob("*.py"))`).

Bonus: **timezone/timestamp in build output** — some tools embed build timestamps in binaries. Fix: set `SOURCE_DATE_EPOCH` environment variable for reproducible builds.
