# Phase 6 · Week 07 · Day 3 — Build/Test/Lint/Scan/Deploy Pipeline + Image Publishing

> Objective: Build a complete, production-quality GitHub Actions pipeline with security scanning and container publishing.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

## Concept

A production pipeline is more than "run tests and push." It enforces:

- **SAST** (Static Application Security Testing) — code-level vulnerability scanning (Bandit, Semgrep).
- **Container image scanning** — CVE scanning of the built image (Trivy, Grype).
- **Dependency auditing** — known-vulnerable packages (pip-audit, Safety).
- **SBOM generation** — Software Bill of Materials for supply chain compliance.
- **Signed images** — cryptographic provenance (Sigstore/Cosign).
- **Promotion not rebuild** — the *same* image SHA promoted through environments.

For an AI service specifically, you also want:
- Secret detection to prevent committing API keys (Gitleaks, detect-secrets).
- Prompt injection / config drift scanning.
- Model artifact integrity checks (hash the model weights file).

## How it works internally

### Container image lifecycle

```
Dockerfile + source code
        │
        ▼ docker build
  Local image (untagged)
        │
        ▼ docker tag / docker push
  Registry (GHCR / ECR / GCR)
    ├── ai-service:abc123f   ← commit SHA (immutable)
    └── ai-service:latest    ← mutable pointer (use sparingly)
        │
        ▼ deploy command
  Runtime (Cloud Run / ECS / k8s)
    references immutable SHA tag
```

**Why SHA tags not `latest`?** `latest` is mutable — a rollback becomes `which latest?`. SHA tags are immutable by construction.

### Trivy scan exit codes

| Code | Meaning |
|---|---|
| 0 | No vulnerabilities (or all below threshold) |
| 1 | Vulnerabilities found above threshold |
| 2 | Scanner error |

Use `--exit-code 1 --severity HIGH,CRITICAL` to fail the pipeline only on actionable CVEs.

## Worked example

### Full pipeline: lint → test → audit → build → scan → sign → deploy

```yaml
# .github/workflows/pipeline.yml
name: Full Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

permissions:
  contents: read
  packages: write
  id-token: write   # for OIDC and Cosign signing

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}/ai-service

jobs:
  # ── Stage 1: Static analysis ──────────────────────────────────────────────
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements*.txt') }}
      - run: pip install ruff mypy bandit pip-audit -r requirements.txt
      - run: ruff check src/
      - run: mypy src/
      # SAST: Bandit security linter
      - run: bandit -r src/ -ll   # -ll = warn only on medium+ severity

  # ── Stage 2: Tests with coverage ─────────────────────────────────────────
  test:
    runs-on: ubuntu-latest
    needs: [lint]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements*.txt') }}
      - run: pip install -r requirements-dev.txt
      - name: Run tests
        run: |
          pytest tests/ \
            --cov=src \
            --cov-fail-under=80 \
            --cov-report=xml \
            --junitxml=test-results.xml \
            -q
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: test-results
          path: |
            coverage.xml
            test-results.xml

  # ── Stage 3: Dependency audit ─────────────────────────────────────────────
  audit:
    runs-on: ubuntu-latest
    needs: [lint]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install pip-audit
      # Fail on known-vulnerable packages
      - run: pip-audit -r requirements.txt --strict

  # ── Stage 4: Build & push image ──────────────────────────────────────────
  build:
    runs-on: ubuntu-latest
    needs: [test, audit]
    # Only push on main, not on PRs (PRs just build to validate Dockerfile)
    outputs:
      image-digest: ${{ steps.push.outputs.digest }}
      image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}@${{ steps.push.outputs.digest }}
    steps:
      - uses: actions/checkout@v4

      - uses: docker/setup-buildx-action@v3

      - name: Log in to GHCR
        if: github.event_name == 'push'
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract Docker metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=sha,format=long
            type=ref,event=branch

      - name: Build and push
        id: push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: ${{ github.event_name == 'push' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          # Generate SBOM and provenance attestation
          sbom: true
          provenance: true

  # ── Stage 5: Container image scan ────────────────────────────────────────
  scan:
    runs-on: ubuntu-latest
    needs: [build]
    if: github.event_name == 'push'
    steps:
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:sha-${{ github.sha }}
          format: table
          exit-code: "1"
          ignore-unfixed: true
          severity: HIGH,CRITICAL
          # Save report as SARIF for GitHub Security tab
      - name: Run Trivy (SARIF output)
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:sha-${{ github.sha }}
          format: sarif
          output: trivy-results.sarif
          severity: HIGH,CRITICAL
      - uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: trivy-results.sarif

  # ── Stage 6: Sign image with Cosign ──────────────────────────────────────
  sign:
    runs-on: ubuntu-latest
    needs: [scan]
    if: github.event_name == 'push'
    steps:
      - name: Install Cosign
        uses: sigstore/cosign-installer@v3

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Sign image with keyless Cosign (OIDC)
        run: |
          cosign sign --yes \
            ${{ needs.build.outputs.image-ref }}

  # ── Stage 7: Deploy to staging ───────────────────────────────────────────
  deploy-staging:
    runs-on: ubuntu-latest
    needs: [sign]
    environment: staging
    steps:
      - uses: actions/checkout@v4
      - name: Deploy to staging
        env:
          IMAGE_DIGEST: ${{ needs.build.outputs.image-digest }}
        run: |
          # Use digest (not tag) for immutable reference
          echo "Deploying ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}@${IMAGE_DIGEST}"
          # kubectl set image deployment/ai-service \
          #   ai-service=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}@${IMAGE_DIGEST}
```

### Dockerfile for an AI service

```dockerfile
# syntax=docker/dockerfile:1.6
FROM python:3.12-slim AS builder

WORKDIR /app

# Copy dependency spec first (cache layer)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install -r requirements.txt

FROM python:3.12-slim AS runtime

# Non-root user
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY --chown=appuser:appgroup src/ ./src/

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Secret scanning step (Gitleaks)

```yaml
- name: Gitleaks secret scan
  uses: gitleaks/gitleaks-action@v2
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

## Exercises

1. The `audit` job runs `pip-audit -r requirements.txt`. Add a second step that also audits `requirements-dev.txt` but only warns (non-blocking) — how do you handle the exit code?

2. Modify the Dockerfile to use a multi-stage build where the `builder` stage installs build tools (`gcc`, `build-essential`) but the `runtime` stage is pure `python:3.12-slim`. Why does this reduce image size and attack surface?

3. Write a step that publishes the Trivy SARIF report as a PR comment using the GitHub API (`${{ secrets.GITHUB_TOKEN }}` + `gh` CLI). What permissions does the job need?

4. Add a `deploy-prod` job with `environment: production` that requires a manual approval and only runs when a Git tag matching `v*.*.*` is pushed.

5. The build job uses `type=sha,format=long` for tags. Why is a long SHA safer than `type=sha` (short SHA) for production image references?

(Solutions: ../solutions/week-07/day-3.md)

## Self-check

- [ ] I can write a pipeline that builds, scans, and signs a container image
- [ ] I understand the difference between SAST (Bandit) and image scanning (Trivy)
- [ ] I can explain why images should be referenced by digest in production
- [ ] I know what SBOM and Cosign keyless signing provide
- [ ] I can write a multi-stage Dockerfile for a Python service

## Supplementary links

- [Trivy documentation](https://aquasecurity.github.io/trivy/)
- [Cosign keyless signing](https://docs.sigstore.dev/cosign/keyless/)
- [docker/build-push-action SBOM/provenance](https://docs.docker.com/build/ci/github-actions/attestations/)
- [Gitleaks](https://github.com/gitleaks/gitleaks-action)
- [pip-audit](https://pypi.org/project/pip-audit/)
