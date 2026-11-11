# Solutions — Week 07 · Day 3

## Exercise 1 — Non-blocking dev dependency audit

```yaml
- name: Audit runtime dependencies (blocking)
  run: pip-audit -r requirements.txt --strict

- name: Audit dev dependencies (warning only)
  run: pip-audit -r requirements-dev.txt --strict || true
  # `|| true` forces exit code 0 regardless of pip-audit result
  # Alternatively, capture exit code and emit a warning annotation:

- name: Audit dev dependencies (annotated warning)
  run: |
    pip-audit -r requirements-dev.txt --strict
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
      echo "::warning::Dev dependency vulnerabilities found — review pip-audit output above"
    fi
    exit 0
```

The `::warning::` syntax emits a GitHub Actions annotation visible in the PR without failing the job.

---

## Exercise 2 — Multi-stage Dockerfile analysis

```dockerfile
# syntax=docker/dockerfile:1.6
FROM python:3.12-slim AS builder

# Install build tools only in the builder stage
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    libpq-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install -r requirements.txt

# Runtime stage — starts from a clean base
FROM python:3.12-slim AS runtime

# Only copy compiled packages from builder
COPY --from=builder /install /usr/local

RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup --no-create-home appuser

WORKDIR /app
COPY --chown=appuser:appgroup src/ ./src/

USER appuser
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
EXPOSE 8080
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

**Why this reduces image size:**
- `gcc`, `build-essential`, `libpq-dev` are ~200 MB. They exist only in the builder layer, not in the final image.
- `apt-get` cache and build artifacts are discarded between stages.
- Result: runtime image is typically 150–200 MB instead of 500+ MB.

**Why it reduces attack surface:**
- No compiler in the runtime image — an attacker who gains code execution cannot compile new binaries locally.
- Fewer packages = fewer CVEs surfaced by Trivy scans.
- `pip` is still present (from the base image) but build tools are absent.

---

## Exercise 3 — Post SARIF as PR comment

```yaml
jobs:
  scan:
    runs-on: ubuntu-latest
    permissions:
      security-events: write    # for SARIF upload
      pull-requests: write      # for PR comment
      contents: read

    steps:
      - uses: actions/checkout@v4

      - name: Run Trivy (JSON for comment)
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ghcr.io/org/ai-service:sha-${{ github.sha }}
          format: json
          output: trivy-results.json
          severity: HIGH,CRITICAL

      - name: Run Trivy (SARIF for Security tab)
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ghcr.io/org/ai-service:sha-${{ github.sha }}
          format: sarif
          output: trivy-results.sarif
          severity: HIGH,CRITICAL
          exit-code: "1"

      - uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: trivy-results.sarif

      - name: Comment Trivy summary on PR
        if: github.event_name == 'pull_request' && always()
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          VULN_COUNT=$(jq '[.Results[].Vulnerabilities // [] | .[] | select(.Severity == "HIGH" or .Severity == "CRITICAL")] | length' trivy-results.json)
          gh pr comment ${{ github.event.pull_request.number }} \
            --body "**Trivy scan**: ${VULN_COUNT} HIGH/CRITICAL vulnerabilities found. See Security tab for details."
```

**Required permissions**: `pull-requests: write` to post comments; `security-events: write` to upload SARIF.

---

## Exercise 4 — deploy-prod triggered by version tag

```yaml
deploy-prod:
  runs-on: ubuntu-latest
  needs: [scan]
  # Only run when a version tag is pushed
  if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')
  environment: production   # manual approval required

  steps:
    - uses: actions/checkout@v4

    - name: Extract version from tag
      id: version
      run: echo "version=${GITHUB_REF#refs/tags/}" >> $GITHUB_OUTPUT

    - name: Deploy to production
      env:
        VERSION: ${{ steps.version.outputs.version }}
      run: |
        echo "Deploying version ${VERSION} to production"
        # kubectl set image ... or terraform apply ...
```

Add to the `on:` block:
```yaml
on:
  push:
    branches: [main]
    tags:
      - "v*.*.*"
  pull_request:
    branches: [main]
```

---

## Exercise 5 — Long SHA vs. short SHA

`type=sha,format=long` produces a 40-character full SHA, e.g. `sha-a1b2c3d4e5f6...full40chars`.  
`type=sha` (default) produces a 7-character prefix, e.g. `sha-a1b2c3d`.

**Safety of long SHA:**

Short SHAs can collide. In a very large repository with thousands of commits, two different commits can share the same 7-character prefix — this is known as a "birthday collision" and is non-trivial to produce intentionally but happens accidentally in long-lived large repos.

More importantly: an adversary with write access to the registry could push a malicious image with a SHA prefix that matches a legitimate short tag. The full 40-character SHA is collision-resistant by the security properties of SHA-1 (and effectively of SHA-256 in modern Git).

For production image references, use the **digest** (`sha256:...`) rather than any tag at all — digests are immutable by construction and cannot be retagged.
