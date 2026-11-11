# Week 07 Assessment — CI/CD & Infrastructure as Code

> Time: ~60–90 min  ·  Pass gate: 75%

## Conceptual (interview-style)

Answer each question in 3–6 sentences. Precision matters more than length.

1. **Artifact promotion vs. rebuild.** Explain why the same container image SHA should be deployed to staging and then promoted to production rather than rebuilt for each environment. What breaks when teams rebuild per environment?

2. **OIDC vs. static credentials.** A colleague proposes storing `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in GitHub Secrets for the deploy job. You recommend OIDC instead. Explain the security difference and what happens if the static key is leaked vs. if the OIDC trust policy is misconfigured.

3. **Terraform state corruption.** Two engineers run `terraform apply` simultaneously. The state backend has no locking (DynamoDB table was accidentally deleted). Describe what can go wrong, step by step. How do you recover?

4. **Canary vs. blue-green.** An AI service rolls out a new prompt template that subtly changes response tone. Users complain after 2 hours. Compare how quickly you can detect and roll back with canary vs. blue-green. Which strategy gives you earlier signal?

5. **Pipeline DAG and fail-fast.** Your pipeline runs `lint`, `unit-test`, `integration-test`, `build`, `scan` in sequence. A colleague says you should run `lint` and `unit-test` in parallel. Is this correct? When is `fail-fast: false` on a matrix job the right choice?

---

## Coding Challenge

Build a **complete, runnable CI/CD pipeline** for the `ai-service` application (from Day 6).

### Spec

Your submission must include:

**1. GitHub Actions workflow** (`.github/workflows/ci.yml`)
- Triggers on push to `main` and on pull requests to `main`.
- Jobs: `lint` → `test` (matrix: Python 3.11 + 3.12) → `build` → `scan` → `deploy-staging` → `deploy-prod`.
- `lint`: ruff + mypy + bandit.
- `test`: pytest with 80% coverage gate, upload XML artifacts.
- `build`: docker buildx push to GHCR with SHA tag; outputs `image-digest`.
- `scan`: Trivy on the pushed image; fail on HIGH/CRITICAL; upload SARIF.
- `deploy-staging`: OIDC AWS auth + `terraform apply` with `environment=staging`.
- `deploy-prod`: requires `environment: production` (manual approval) + OIDC.

**2. Terraform module** (`infra/modules/container-service/`)
- Variables: `name`, `environment`, `image_url`, `cpu`, `memory`, `desired_count`, `subnet_ids`, `vpc_id`, `app_port`, `tags`.
- Resources: `aws_ecs_cluster`, `aws_ecs_task_definition`, `aws_ecs_service` (Fargate), `aws_lb_target_group`, `aws_security_group`, `aws_cloudwatch_log_group`.
- Outputs: `service_name`, `cluster_name`, `target_group_arn`.

**3. Root Terraform** (`infra/main.tf`)
- Calls the module twice: once for `staging`, once for `prod`.
- Includes `aws_db_instance` (RDS Postgres) and `aws_s3_bucket` (artifacts).
- Remote S3 backend declared in `terraform.tf`.
- All secrets via `var.*` (sensitive) or `aws_secretsmanager_secret`.

**4. Tests** (`tests/unit/`)
- `test_health` — asserts `/health` returns 200 and `status == "ok"`.
- `test_infer_no_key` — asserts `/infer` returns 500 when `ANTHROPIC_API_KEY` is unset.
- `test_infer_mocked` — mocks `anthropic.Anthropic`, asserts response model and content.

### Acceptance criteria

- [ ] Workflow file is valid YAML (parseable, correct indentation).
- [ ] All jobs have `needs:` that enforce the lint → test → build → scan → deploy order.
- [ ] Build job only runs on `push` events (not PRs).
- [ ] Image is referenced by digest in the deploy step, not by mutable tag.
- [ ] Terraform module is called with `for_each` or two explicit module blocks.
- [ ] All sensitive values passed via `var.*` or environment-level secrets — nothing hardcoded.
- [ ] Three unit tests pass against the FastAPI app.
- [ ] `terraform validate` passes on the infra directory.

---

## Rubric (pass ≥ 75%)

| Dimension | Weight | What earns full marks |
|---|---|---|
| Independent implementation | 40% | Working workflow + module without consulting solution; all acceptance criteria met |
| Conceptual understanding | 25% | Conceptual answers are precise, not vague; demonstrate tradeoffs |
| Debugging & problem-solving | 20% | Correctly identifies the issue in the debugging scenario below |
| Code quality & testing | 15% | YAML/HCL consistent, secrets never hardcoded, tests mock external calls |

### Debugging scenario (counts toward 20%)

The following GitHub Actions step runs but the subsequent Trivy scan fails with "image not found":

```yaml
- name: Build and push
  uses: docker/build-push-action@v5
  with:
    context: .
    push: ${{ github.event_name == 'push' }}
    tags: ghcr.io/org/ai-service:${{ github.sha }}
```

The Trivy job that follows:
```yaml
- uses: aquasecurity/trivy-action@master
  with:
    image-ref: ghcr.io/org/ai-service:${{ github.sha }}
```

The pipeline runs on a pull_request event. Identify the bug and provide the fix.

---

Answer key: ../solutions/week-07/assessment.md
