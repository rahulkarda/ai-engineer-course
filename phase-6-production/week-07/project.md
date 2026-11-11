# Week 07 Project — Complete CI/CD Pipeline + Terraform Module

> Time: ~4 hr (Day 6 build) + ~3 hr (Day 7 polish and runbook)

## Goal

Ship a production-grade CI/CD pipeline and IaC module for an AI service. When complete, every push to `main` automatically lints, tests, builds, scans, and deploys the application — and the underlying infrastructure is fully described in Terraform.

## Background

You are the platform engineer for an AI inference service built on FastAPI + Anthropic Claude. The service needs:
- Automated quality checks on every PR.
- Immutable container images pushed to GHCR on merge to main.
- Vulnerability scanning before any deployment.
- Cloud infrastructure (ECS Fargate + RDS + ALB) provisioned by Terraform.
- A canary-capable deployment strategy.
- A runbook so any team member can deploy or roll back.

## Deliverables

### 1. Application (`src/`)

- `src/main.py` — FastAPI with `/health` (GET) and `/infer` (POST) endpoints.
- `src/config.py` — `pydantic-settings` config; reads `ENVIRONMENT`, `DATABASE_URL`.
- `src/models.py` — Pydantic models for all request/response shapes.
- `Dockerfile` — multi-stage build; non-root user; `python:3.12-slim`.
- `requirements.txt`, `requirements-dev.txt`, `pyproject.toml`.

### 2. Tests (`tests/`)

- `tests/unit/test_main.py` — at minimum three tests covering health, infer-no-key, infer-mocked.
- `tests/conftest.py` — shared fixtures if needed.
- All external calls (LLM, DB) must be mocked. 80%+ coverage required.

### 3. GitHub Actions (`.github/workflows/ci.yml`)

Full pipeline:
```
lint (ruff + mypy + bandit)
    └─ test (matrix: py3.11 + py3.12, coverage ≥ 80%)
           └─ audit (pip-audit)
                  └─ build (buildx → GHCR, SHA tag, SBOM)
                         └─ scan (Trivy HIGH/CRITICAL gate, SARIF upload)
                                └─ deploy-staging (OIDC + terraform apply)
                                       └─ deploy-prod (manual gate + OIDC + terraform apply)
```

Additional workflow: `.github/workflows/rollback.yml`
- Trigger: `workflow_dispatch` with inputs `environment` and `image_tag`.
- Runs `terraform apply` with the specified image tag.

### 4. Terraform (`infra/`)

**Module** (`infra/modules/container-service/`):
- Variables: `name`, `environment`, `image_url`, `cpu`, `memory`, `desired_count`, `subnet_ids`, `vpc_id`, `app_port`, `tags`.
- Resources: ECS cluster, task definition (Fargate), ECS service, ALB target group, security group, CloudWatch log group.
- Outputs: `service_name`, `cluster_name`, `target_group_arn`.

**Root** (`infra/`):
- `main.tf`: VPC + subnets, RDS Postgres, S3 (artifacts, versioned, encrypted), calls the module for staging and prod.
- `variables.tf`: `environment`, `aws_region`, `image_tag`, `db_password` (sensitive), `canary_weight`.
- `outputs.tf`: ALB DNS name, ECR URL, RDS endpoint.
- `terraform.tf`: S3 backend, required providers (hashicorp/aws ~> 5.0), required_version >= 1.6.0.

**Environment files** (gitignored):
- `infra/staging.tfvars` — `environment = "staging"`, `db_instance_class = "db.t3.micro"`.
- `infra/prod.tfvars` — `environment = "prod"`, `db_instance_class = "db.t3.small"`.

### 5. Deploy Runbook (`runbook.md`)

Sections:
1. **Prerequisites** — `terraform >= 1.6`, `aws-cli >= 2`, `gh cli`, `docker`, required IAM role.
2. **Normal deploy** — push to main → watch pipeline → approve prod gate → verify health endpoint.
3. **Canary deploy** — run `scripts/canary_promote.py --steps 5,20,50,100 --bake-time 300`.
4. **Rollback** — trigger `rollback.yml` workflow with previous `image_tag`; or manual ECS task definition rollback.
5. **Post-deploy checks** — `/health` endpoint, CloudWatch `HTTPCode_Target_5XX_Count < 5/min`, p99 latency < 500 ms.

## Acceptance Criteria

### Pipeline
- [ ] `ci.yml` is valid YAML and passes `yamllint`.
- [ ] Lint job runs ruff, mypy, bandit — any failure blocks subsequent jobs.
- [ ] Test matrix covers Python 3.11 and 3.12; both must pass.
- [ ] Build job only runs on `push` to `main`, not on PRs.
- [ ] Trivy scan fails the pipeline on HIGH or CRITICAL CVEs.
- [ ] SARIF results uploaded to the Security tab.
- [ ] `deploy-prod` requires manual approval (`environment: production`).
- [ ] No secrets hardcoded anywhere in YAML — all via `${{ secrets.* }}`.
- [ ] OIDC used for AWS auth in deploy jobs.

### Terraform
- [ ] `terraform validate` passes on `infra/`.
- [ ] `terraform plan` produces no errors on a fresh workspace with valid vars.
- [ ] All resource names use `var.environment` as prefix for isolation.
- [ ] `db_password` variable is marked `sensitive = true`.
- [ ] RDS has `deletion_protection = true` in prod.
- [ ] S3 bucket has versioning and SSE-AES256 enabled.
- [ ] Module is defined once and called with environment-specific variables.

### Application & Tests
- [ ] `pytest tests/unit -q` exits 0.
- [ ] Coverage ≥ 80%.
- [ ] No `print()` in source — `logging` only.
- [ ] `ruff check src/` exits 0.
- [ ] `mypy src/` exits 0.
- [ ] Dockerfile builds without errors (single `docker build .` command).
- [ ] `/health` returns `{"status": "ok", ...}` when running locally.

### Runbook
- [ ] Runbook documents exact commands for normal deploy, canary, and rollback.
- [ ] Rollback section specifies how to identify the previous stable image tag.
- [ ] Post-deploy checks include at least two observable metrics.

## Stretch Goals

1. Add a `smoke-test` job between `deploy-staging` and `deploy-prod` that curls the staging `/health` endpoint and fails if the response is not 200.
2. Implement `aws_appautoscaling_target` and `aws_appautoscaling_policy` in the Terraform module for CPU-based ECS autoscaling.
3. Add a `notify-slack` job that runs `if: failure()` and posts a message to `secrets.SLACK_WEBHOOK` with the run URL and failed job name.
4. Write a `scripts/canary_promote.py` that queries a CloudWatch metric for the canary target group's 5xx count before each promotion step.

## Submission

Push all files to a branch `week-07-project` and open a PR against `main`. The PR description should include:
- Screenshot or log of a passing pipeline run.
- `terraform plan` output for staging environment (redact any sensitive values).
- Link to the Security tab showing Trivy SARIF results.
