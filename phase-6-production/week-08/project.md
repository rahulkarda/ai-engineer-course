# Week 08 Project — Deploy a Containerized AI API to ECS Fargate

> Duration: Day 7 + weekend if needed  ·  Type: Integration project  ·  Solo or pair

---

## Overview

Bring together every concept from Week 8 into a single, deployable, production-ready AI API. You will containerize a FastAPI app that calls Claude, write the full Terraform IaC, integrate secrets management and autoscaling, document the architecture, and produce a cost estimate.

This project is the practical capstone of Phase 6's cloud deployment track. The output — a working Dockerfile, Terraform codebase, architecture diagram, cost spreadsheet, and ADR — is a portfolio artifact you can reference in engineering discussions.

---

## Goals

1. **Containerize** a FastAPI AI app using a multi-stage Docker build with a non-root user and no secrets baked in.
2. **Write Terraform IaC** for the full stack: ECS Fargate + RDS (PostgreSQL) + ElastiCache (Redis) + Secrets Manager + ALB + Application Auto Scaling.
3. **Document** the architecture cloud-agnostically (design doc) with a concrete AWS walkthrough.
4. **Produce a cost estimate** for three traffic scenarios using the cost estimator from Day 5.
5. **Write one ADR** capturing the most significant architectural decision you made during the project.

---

## Deliverables

All deliverables live under `week-08-project/` in your course workspace.

### 1. `Dockerfile`
- Multi-stage build (builder + runtime).
- Non-root user (`appuser:appuser`).
- `HEALTHCHECK` instruction present.
- No `RUN` instructions that touch secrets.
- Final image size below 200 MB (use `docker image inspect` to verify).

### 2. Terraform codebase (`terraform/`)

Modules:
- `modules/ecs/` — cluster, task definition, service, task/execution roles.
- `modules/networking/` — VPC, subnets, security groups, ALB, target group.
- `modules/data/` — RDS PostgreSQL instance, ElastiCache Redis cluster, subnet groups.
- `modules/secrets/` — Secrets Manager secrets for DB credentials + API key.
- `modules/autoscaling/` — Application Auto Scaling target + policies.

Root module (`main.tf`) wires the modules together. All sensitive values via variables; no literals.

Minimum resources:
- `aws_ecr_repository`
- `aws_ecs_cluster` with Container Insights enabled
- `aws_ecs_task_definition` (Fargate, awsvpc)
- `aws_ecs_service` with `ignore_changes = [desired_count]`
- `aws_lb` + `aws_lb_target_group` + `aws_lb_listener`
- `aws_db_instance` (RDS PostgreSQL, `db.t3.micro` for dev)
- `aws_elasticache_cluster` (Redis, `cache.t3.micro` for dev)
- `aws_secretsmanager_secret` × 2 (API key, DB password)
- `aws_appautoscaling_target` + at least one `aws_appautoscaling_policy`
- `aws_cloudwatch_log_group`
- Appropriate `aws_iam_role` × 2 (execution + task)

### 3. FastAPI application (`app/`)

Functional endpoints:
- `GET /health` — returns `{"status": "ok"}` (ALB health check).
- `POST /infer` — accepts `{prompt, max_tokens}`, calls Claude, returns structured response.
- `POST /store` — stores a prompt/response pair to RDS via SQLAlchemy (async).
- `GET /cached/{key}` — returns a cached result from ElastiCache Redis.

Requirements:
- Pydantic models for all request/response types.
- Secrets fetched from Secrets Manager at startup (boto3).
- DB connection string assembled from Secrets Manager values — never hardcoded.
- Structured JSON logging (`python-json-logger` or equivalent).
- `os.getenv()` for all environment-level config.
- No `print()` statements anywhere.

### 4. Architecture document (`docs/architecture.md`)

Sections:
1. **Cloud-agnostic design**: The problem, the pattern (managed container service + managed DB + secrets manager + autoscaling), equivalents table across AWS / GCP / Azure.
2. **AWS concrete walkthrough**: How each AWS service maps to the pattern; data flow diagram (ASCII acceptable).
3. **Security model**: IAM role separation, secrets at rest vs. in transit, network isolation (private subnets, security group rules).
4. **Scaling model**: What triggers scale-out, what the limits are, what fails first under extreme load.

### 5. Cost estimate (`docs/cost-estimate.md` or `.xlsx`)

Three scenarios:
- **Dev/staging**: 1 ECS task, minimal RDS, 0.5 RPS average.
- **Production (low traffic)**: 2–4 tasks, `db.t3.small`, 10 RPS average.
- **Production (medium traffic)**: 5–15 tasks autoscaling, `db.t3.medium`, 50 RPS average.

For each scenario, break down: compute, LLM API calls, RDS, ElastiCache, ALB, data transfer. Show monthly total.

Use the `cost_estimator.py` from Day 5 as a starting point; extend it to cover RDS and ElastiCache.

### 6. ADR (`docs/adr/ADR-001-*.md`)

One ADR on the most significant decision you made. Candidates:
- Why ECS Fargate over EC2 launch type (or Lambda, or App Runner).
- Why Secrets Manager over SSM Parameter Store.
- Why target tracking over step scaling for your workload.
- Why multi-stage Docker build.

Follow the ADR format from Day 7.

---

## Acceptance Criteria

| Criterion | Verified by |
|---|---|
| App deploys with `terraform apply` + ECR push; `/health` returns 200 | `curl http://<alb-dns>/health` |
| No secret values in Dockerfile, Terraform files, or Python source | `grep -r "sk-ant\|password\s*=\s*[\"'][^$]" .` returns nothing |
| Autoscaling configured; `aws application-autoscaling describe-scalable-targets` shows target | AWS CLI |
| Cost estimate covers all three scenarios with line-item breakdown | Manual review |
| `POST /infer` returns a non-empty `result` field using Claude | `curl -X POST .../infer -d '{"prompt":"say hello","max_tokens":32}'` |
| Terraform has no warnings on `terraform validate` | `terraform validate` exit code 0 |
| All Python files pass `ruff check` with no errors | `ruff check app/` |
| ADR is ≥ 300 words and includes alternatives table | Manual review |

---

## Tech Stack

| Component | Technology | Notes |
|---|---|---|
| Language | Python 3.12+ | Type hints everywhere |
| Web framework | FastAPI | Async, Pydantic v2 |
| LLM | Anthropic Claude (`claude-sonnet-5-5`) | Via `anthropic` SDK |
| ORM | SQLAlchemy 2.x (async) | For RDS interaction |
| Cache client | `redis-py` (async) | For ElastiCache |
| Secrets | boto3 + AWS Secrets Manager | Fetched at startup |
| Container | Docker (multi-stage) | Non-root, minimal |
| Registry | Amazon ECR | |
| Compute | ECS Fargate | Serverless containers |
| Database | Amazon RDS PostgreSQL | `db.t3.micro` in dev |
| Cache | Amazon ElastiCache Redis | `cache.t3.micro` in dev |
| Load balancer | Application Load Balancer | HTTP:80 for dev |
| Autoscaling | AWS Application Auto Scaling | Target tracking |
| IaC | Terraform >= 1.6 | AWS provider ~> 5.0 |
| Linting | ruff | All Python files |
| Logging | Python `logging` + JSON formatter | No print() |

---

## Suggested Time Split (~4 hr total)

| Task | Time |
|---|---|
| App skeleton + Dockerfile | 60 min |
| Terraform modules (networking + ECS + secrets) | 75 min |
| Terraform modules (RDS + ElastiCache + autoscaling) | 45 min |
| Deploy + smoke test | 30 min |
| Architecture doc + ADR | 30 min |
| Cost estimate | 20 min |

---

## Hints

- Start with `terraform apply` on the networking module alone, verify VPC exists, then add ECS.
- Use `terraform plan -target=module.networking` to iterate quickly on a single module.
- The RDS and ElastiCache instances take 5–10 minutes to provision — start them early.
- Test the Dockerfile locally with `docker run --env-file .env.local -p 8080:8080 ai-api:latest` before pushing to ECR.
- For local dev without a real Secrets Manager, mock `get_secret` by reading from a local `.env` file — but remove this code path before building the final image.
- `ruff check --fix app/` handles most auto-fixable style issues.

---

## Extension Challenges (optional)

- Add HTTPS to the ALB with an ACM certificate (requires a domain you control or Route 53 hosted zone).
- Implement a CI/CD pipeline (GitHub Actions or AWS CodePipeline) that builds, pushes, and deploys on every push to `main`.
- Add distributed tracing with AWS X-Ray.
- Implement secret rotation for the Anthropic API key using a Lambda rotation function.
- Replace the ALB with an API Gateway HTTP API and compare cost/performance.
