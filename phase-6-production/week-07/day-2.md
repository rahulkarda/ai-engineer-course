# Phase 6 · Week 07 · Day 2 — GitHub Actions In Depth

> Objective: Master GitHub Actions — workflows, jobs, matrix, caching, secrets, environments, and OIDC.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

## Concept

GitHub Actions is a **workflow automation platform** embedded in GitHub. A **workflow** is a YAML file in `.github/workflows/`. It defines one or more **jobs** that run on **runners** (GitHub-hosted or self-hosted VMs/containers) in response to **events** (push, pull_request, schedule, workflow_dispatch, etc.).

Unlike older CI tools (Jenkins, CircleCI), GitHub Actions:
- Stores pipeline config alongside code — no separate service to maintain.
- Has a massive marketplace of reusable **Actions** (each action is a composable step).
- Supports **OIDC** for keyless cloud authentication — no long-lived credentials in secrets.

## How it works internally

When a triggering event fires, GitHub evaluates all matching workflow files. For each workflow, it schedules jobs according to the `needs:` dependency graph. Each job runs in a fresh runner VM:

1. Runner checks out the repo (`actions/checkout`).
2. Each `step` runs sequentially in the same shell session.
3. Steps share the workspace (`$GITHUB_WORKSPACE`).
4. Jobs are isolated — shared state passes through **artifacts** or **caches**.

**Expressions** (`${{ }}`) are evaluated server-side before the runner executes, making them safe for conditional logic but not for runtime secrets.

### Key components

| Component | Purpose |
|---|---|
| `on:` | Event triggers (push, PR, schedule, dispatch) |
| `jobs.<id>:` | Isolated unit of work with its own runner |
| `needs:` | Job dependency — creates the DAG |
| `strategy.matrix:` | Fan-out: run same job with different inputs |
| `uses:` | Reuse a pre-built action from the marketplace |
| `env:` | Environment variables (job or step scope) |
| `secrets:` | Encrypted values injected at runtime |
| `environment:` | Named deployment target with protection rules |
| `cache:` | Persist files between runs keyed by hash |
| `permissions:` | Least-privilege GITHUB_TOKEN scoping |

## Worked example

### Complete workflow: lint → test → build → deploy

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, "feature/**"]
  pull_request:
    branches: [main]

# Least-privilege defaults
permissions:
  contents: read

jobs:
  lint:
    name: Lint & type-check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      # Cache pip dependencies — key = OS + Python + hash of requirements
      - uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-

      - name: Install dependencies
        run: pip install -r requirements-dev.txt

      - name: Ruff lint
        run: ruff check src/

      - name: Mypy type-check
        run: mypy src/

  test:
    name: Unit tests (Python ${{ matrix.python-version }})
    runs-on: ubuntu-latest
    needs: [lint]
    # Matrix: test across multiple Python versions
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
      fail-fast: false  # don't cancel other matrix legs on failure

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ matrix.python-version }}-${{ hashFiles('**/requirements*.txt') }}

      - run: pip install -r requirements-dev.txt

      - name: Run tests with coverage
        run: |
          pytest tests/unit \
            --cov=src \
            --cov-fail-under=80 \
            --cov-report=xml \
            -q

      # Upload coverage as artifact for later inspection
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: coverage-py${{ matrix.python-version }}
          path: coverage.xml

  build:
    name: Build & push container image
    runs-on: ubuntu-latest
    needs: [test]
    # Only build on pushes to main (not on PRs)
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    permissions:
      contents: read
      packages: write   # needed to push to GHCR

    steps:
      - uses: actions/checkout@v4

      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            ghcr.io/${{ github.repository }}/ai-service:${{ github.sha }}
            ghcr.io/${{ github.repository }}/ai-service:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy-staging:
    name: Deploy to staging
    runs-on: ubuntu-latest
    needs: [build]
    environment: staging   # triggers required reviewers / protection rules

    steps:
      - name: Deploy image to staging
        env:
          IMAGE_TAG: ${{ github.sha }}
        run: |
          echo "Deploying ghcr.io/${{ github.repository }}/ai-service:${IMAGE_TAG} to staging"
          # Replace with your actual deploy CLI (kubectl, fly deploy, etc.)
```

### Secrets and environments

```yaml
# Accessing an org/repo secret:
- name: Call external API
  env:
    API_KEY: ${{ secrets.MY_API_KEY }}
  run: python scripts/deploy.py

# Environment-scoped secrets (only available inside jobs with environment: staging/prod)
# Set in: repo Settings → Environments → staging → Secrets
```

### OIDC — keyless cloud auth (AWS example)

```yaml
# No AWS credentials stored in GitHub secrets.
# GitHub issues a short-lived OIDC token; AWS STS exchanges it for temp credentials.
permissions:
  id-token: write   # required for OIDC
  contents: read

- name: Configure AWS credentials via OIDC
  uses: aws-actions/configure-aws-credentials@v4
  with:
    role-to-assume: arn:aws:iam::123456789012:role/GitHubActionsRole
    aws-region: eu-west-1

- name: Push to ECR
  run: |
    aws ecr get-login-password | docker login --username AWS \
      --password-stdin 123456789012.dkr.ecr.eu-west-1.amazonaws.com
    docker push 123456789012.dkr.ecr.eu-west-1.amazonaws.com/ai-service:${{ github.sha }}
```

### Reusable workflow

```yaml
# .github/workflows/reusable-deploy.yml
on:
  workflow_call:
    inputs:
      environment:
        required: true
        type: string
      image_tag:
        required: true
        type: string
    secrets:
      DEPLOY_TOKEN:
        required: true

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: ${{ inputs.environment }}
    steps:
      - name: Deploy
        env:
          TOKEN: ${{ secrets.DEPLOY_TOKEN }}
        run: |
          echo "Deploying tag ${{ inputs.image_tag }} to ${{ inputs.environment }}"
```

```yaml
# Caller workflow references the reusable workflow:
deploy-staging:
  uses: ./.github/workflows/reusable-deploy.yml
  with:
    environment: staging
    image_tag: ${{ github.sha }}
  secrets:
    DEPLOY_TOKEN: ${{ secrets.STAGING_DEPLOY_TOKEN }}
```

## Exercises

1. Add a `schedule:` trigger to the CI workflow that runs nightly at 02:00 UTC. What jobs should run on a scheduled trigger vs. a push trigger?

2. The matrix currently covers Python 3.11 and 3.12. Extend it to also test on `ubuntu-latest` and `ubuntu-22.04`, creating a 2×2 matrix of OS × Python version.

3. The `cache@v4` key uses `hashFiles('**/requirements*.txt')`. What happens if you use `pyproject.toml` instead? When should you prefer one over the other?

4. Write a workflow step that uses a `workflow_dispatch` event with two inputs: `environment` (choice: dev/staging/prod) and `dry_run` (boolean). The deploy step should print "DRY RUN" and skip real deployment when `dry_run` is true.

5. Explain why `secrets.GITHUB_TOKEN` is safe to use without rotation, but a manually added `MY_API_KEY` secret requires a rotation policy.

(Solutions: ../solutions/week-07/day-2.md)

## Self-check

- [ ] I can write a multi-job workflow with `needs:` dependencies
- [ ] I can configure a matrix strategy and explain `fail-fast`
- [ ] I understand how `actions/cache` keys work and when caches are restored
- [ ] I can explain OIDC authentication and why it's preferred over static credentials
- [ ] I know the difference between repo secrets, environment secrets, and `GITHUB_TOKEN`

## Supplementary links

- [GitHub Actions: Workflow syntax](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [GitHub Actions: Encrypted secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [GitHub Actions: OIDC with cloud providers](https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect)
- [docker/build-push-action](https://github.com/docker/build-push-action)
