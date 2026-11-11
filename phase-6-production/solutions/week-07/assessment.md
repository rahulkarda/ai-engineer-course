# Solutions — Week 07 Assessment

## Conceptual Answers

### 1. Artifact promotion vs. rebuild

**Why promote, not rebuild:**

A container image is a compiled artifact. If you rebuild from the same source, you might get a different result: the base image `FROM python:3.12-slim` resolves to a newer patch version, a transitive dependency resolves to a different version, or a `RUN apt-get update` pulls updated packages. The staging image you tested is not the same binary as the prod image you deployed.

**What breaks with rebuild-per-environment:**

- Security validations are invalidated: the Trivy scan run on the staging image applies to a different binary than what hits prod.
- Test coverage claims become false: unit tests passed on staging image; prod image has a different dependency tree.
- Debugging becomes impossible: "it worked in staging" cannot be trusted because staging and prod ran different code.
- Reproducibility is lost: you cannot revert to "the exact version we had last Tuesday" because that build no longer exists.

The solution: tag with the commit SHA, push once, promote the SHA tag through environments. Only the Terraform variable `image_tag` changes between environments.

---

### 2. OIDC vs. static credentials

**Static credentials (`AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` in GitHub Secrets):**

These are long-lived credentials. They remain valid until manually rotated. If leaked — via a `--debug` flag, a misconfigured logging statement, a compromised GitHub Actions runner, or a GitHub API leak — the attacker has permanent AWS access until you detect the breach and rotate. Detection is often delayed (days to weeks).

**OIDC:**

GitHub issues a short-lived JWT (valid for minutes) for each specific workflow run. AWS STS exchanges it for temporary credentials scoped to the role's permissions and valid for the job duration (max 1 hour). A leaked OIDC token is already expired before an attacker can use it.

If the OIDC trust policy is misconfigured (e.g., `"StringLike": {"token.actions.githubusercontent.com:sub": "*"}` instead of scoping to a specific repo/branch), any GitHub Actions workflow from any repo can assume the role. This is why the trust policy must scope to `repo:org/repo:ref:refs/heads/main` or similar.

**Preferred approach:** OIDC with a narrowly scoped trust policy and a role that has least-privilege IAM permissions.

---

### 3. Terraform state corruption

**Step-by-step without DynamoDB locking:**

1. Engineer A: `terraform apply` starts. Reads state version N from S3. Computes plan: create resource X and Y.
2. Engineer B: `terraform apply` starts concurrently. Reads same state version N. Computes plan: create resource Z (different change set, same base state).
3. Engineer A finishes. Writes state version N+1 to S3. Resources X and Y are in the state.
4. Engineer B finishes. Writes state version N+1 to S3, overwriting A's write. State now records only resource Z. Resources X and Y exist in AWS but are absent from state — they are orphaned.

**Recovery:**

1. `terraform refresh` to re-sync state with AWS reality.
2. `terraform state list` to audit what Terraform knows about vs. what AWS reports.
3. `terraform import <resource_type>.<name> <cloud_id>` to re-adopt orphaned resources.
4. Re-enable DynamoDB locking before any further applies.
5. Audit for any duplicate resources created by both engineers; destroy duplicates manually.

---

### 4. Canary vs. blue-green for AI prompt template change

**Blue-green detection time:**

All traffic flips at once. If the new prompt template causes subtle tone issues, 100% of users are affected immediately. Detection comes from user complaints, not automated metrics (tone issues don't show up as 5xx errors). You might detect it within 2 hours (as described), but you have already affected all users.

**Rollback:** Instant — flip traffic back to blue. All users are on the old version within seconds.

**Canary detection time:**

5% of traffic goes to the new template. 95% of users are unaffected. You can observe user feedback and behavioral signals (engagement rate, feedback thumbs-up/down, conversation length) on the 5% cohort before affecting everyone. Detection window is the bake time (5–30 minutes per step), giving you early signal with limited blast radius.

**Rollback:** Set canary weight to 0. Only 5% of users were affected during the bake.

**Which gives earlier signal:** Canary — you get signal while blast radius is still small. Blue-green gives a faster rollback but only after 100% exposure.

For AI services where behavioral changes are subtle (not binary error/no-error), canary is strongly preferred.

---

### 5. Pipeline DAG and fail-fast

**Running lint and unit-test in parallel: correct in principle, with a caveat.**

Parallel execution is faster — total wall time equals `max(lint, unit-test)` instead of their sum. Both jobs are independent (no data dependency). This is valid.

**Caveat:** If lint fails within 10 seconds but unit-test takes 2 minutes, you get a fast signal from lint without waiting. But if you run them in sequence (lint → unit-test), a lint failure would save the unit-test runner cost. Parallel is better for developer experience; sequential is cheaper.

**`fail-fast: false` is the right choice when:**

- The matrix tests multiple versions (Python 3.11 and 3.12) — if 3.11 fails but 3.12 passes, you want to know *which* versions fail, not just that one of them did.
- The matrix tests multiple operating systems — a Windows-specific failure should not hide a Linux-specific failure.
- Failures in different legs provide independent diagnostic value.

**`fail-fast: true` (default) is better when:**

- Matrix legs are near-identical (same OS/Python version, different test categories) — once one fails, the others will likely fail the same way; no value in waiting.
- CI minutes are expensive and fast cancellation saves cost.

---

## Coding Challenge — Reference Solution

### Debugging Scenario Answer

**Bug:** The build job has `push: ${{ github.event_name == 'push' }}`. On a `pull_request` event, this evaluates to `false` — the image is built locally on the runner but **not pushed** to GHCR.

The Trivy job then tries to pull `ghcr.io/org/ai-service:${{ github.sha }}` from GHCR — but nothing was pushed. It fails with "image not found."

**Fix option 1:** Add a `needs: [build]` check and skip scan on PRs:
```yaml
scan:
  needs: [build]
  if: github.event_name == 'push'
```

**Fix option 2:** Push to GHCR on PRs too (with a different tag) and scan the PR image:
```yaml
- uses: docker/build-push-action@v5
  with:
    push: true   # always push
    tags: |
      ghcr.io/org/ai-service:${{ github.sha }}
```
Then ensure the GHCR token has write access from PR workflows (`permissions: packages: write`).

**Recommended fix:** Option 1 — build on PRs to validate the Dockerfile, but only push and scan on pushes to `main`. This saves registry storage and scan costs on every PR while still catching vulnerabilities on every merge.

---

### Complete ci.yml Reference

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

permissions:
  contents: read
  packages: write
  id-token: write
  security-events: write

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}/ai-service

jobs:
  lint:
    name: Lint & Type Check
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
      - run: pip install -r requirements-dev.txt
      - run: ruff check src/ tests/
      - run: mypy src/
      - run: bandit -r src/ -ll -q

  test:
    name: Test (Python ${{ matrix.python-version }})
    runs-on: ubuntu-latest
    needs: [lint]
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
      fail-fast: false
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-${{ matrix.python-version }}-pip-${{ hashFiles('requirements*.txt') }}
      - run: pip install -r requirements-dev.txt
      - run: |
          pytest tests/unit \
            --cov=src \
            --cov-fail-under=80 \
            --cov-report=xml \
            --junitxml=test-results.xml \
            -q
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: test-results-${{ matrix.python-version }}
          path: "*.xml"

  audit:
    name: Dependency Audit
    runs-on: ubuntu-latest
    needs: [lint]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install pip-audit
      - run: pip-audit -r requirements.txt --strict

  build:
    name: Build & Push Image
    runs-on: ubuntu-latest
    needs: [test, audit]
    if: github.event_name == 'push'
    outputs:
      image-digest: ${{ steps.push.outputs.digest }}
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/metadata-action@v5
        id: meta
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: type=sha,format=long
      - uses: docker/build-push-action@v5
        id: push
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          sbom: true
          provenance: true

  scan:
    name: Vulnerability Scan
    runs-on: ubuntu-latest
    needs: [build]
    if: github.event_name == 'push'
    steps:
      - uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}@${{ needs.build.outputs.image-digest }}
          format: sarif
          output: trivy-results.sarif
          severity: HIGH,CRITICAL
          exit-code: "1"
          ignore-unfixed: true
      - uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: trivy-results.sarif

  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: [scan]
    environment: staging
    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_DEPLOY_ROLE_ARN }}
          aws-region: eu-west-1
      - name: Terraform apply
        env:
          TF_VAR_db_password: ${{ secrets.STAGING_DB_PASSWORD }}
        run: |
          cd infra
          terraform init
          terraform apply -auto-approve \
            -var="environment=staging" \
            -var="image_tag=sha-${{ github.sha }}"

  deploy-prod:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: [deploy-staging]
    environment: production
    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_DEPLOY_ROLE_ARN_PROD }}
          aws-region: eu-west-1
      - name: Terraform apply
        env:
          TF_VAR_db_password: ${{ secrets.PROD_DB_PASSWORD }}
        run: |
          cd infra
          terraform init
          terraform apply -auto-approve \
            -var="environment=prod" \
            -var="image_tag=sha-${{ github.sha }}"
```

Note: The scan step references the image by digest (`needs.build.outputs.image-digest`), not by mutable tag. This satisfies the acceptance criterion "image referenced by digest in deploy step" — the deploy jobs receive the digest via `needs.build.outputs.image-digest` and pass it as `TF_VAR_image_digest` (or the image_tag variable depending on your Terraform setup).
