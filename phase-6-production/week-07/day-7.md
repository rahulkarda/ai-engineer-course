# Phase 6 · Week 07 · Day 7 — Project Day

> Objective: Complete the week project and submit for assessment.  ·  Time: ~3 hr project + 1 hr assessment  ·  Prereqs: [[day-6]]

## Review & Synthesis

Before starting the project, spend 20 minutes reviewing the week:

### Key takeaways

**CI/CD concepts (Day 1)**
- Pipeline = DAG of jobs. Stages: lint → test → build → scan → deploy.
- Immutable artifacts: push once, promote. Never rebuild for staging vs. prod.
- Quality gates block promotion automatically — no manual gatekeeping.

**GitHub Actions (Day 2)**
- Workflows are YAML in `.github/workflows/`. Events trigger jobs.
- `needs:` creates the DAG. `strategy.matrix:` fans out.
- `actions/cache` is keyed by content hash — cache invalidates when deps change.
- OIDC = keyless cloud auth. No static credentials in secrets.

**Full pipeline (Day 3)**
- Bandit = SAST (code-level). Trivy = image CVE scan. pip-audit = dependency audit.
- Cosign signs images with OIDC — zero secrets, verifiable provenance.
- Multi-stage Dockerfile separates build tools from runtime.

**Terraform (Day 4)**
- State file maps HCL to real resource IDs. Remote + locked state is required for teams.
- `+`, `~`, `-`, `-/+` symbols in plan. `-/+` on a database = data loss risk.
- Modules package reusable infra. `sensitive = true` masks secrets in output.

**IaC for AI + Deployment Strategies (Day 5)**
- AI service infra: ECS + RDS + ALB + S3 + Secrets Manager.
- Blue-green: two environments, atomic traffic flip, instant rollback.
- Canary: weighted routing, bake time + metrics gate, gradual promotion.

---

## Week Project

See [[project.md]] for the full spec and acceptance criteria.

---

## Pre-Assessment Review

Work through these questions before opening `assessment.md`:

1. What is the minimum set of pipeline stages for a production AI service?
2. How does `actions/cache@v4` decide when to restore vs. create a new cache entry?
3. Draw the sequence of events when Cosign signs an image with OIDC keyless signing.
4. What is the Terraform state file and why must it be stored remotely?
5. Explain the canary rollback decision: when do you stop promotion and how is it automated?
6. Why does `terraform plan` sometimes show `-/+` (replace) instead of `~` (update)?
7. What GitHub permission is required for a job to push to GHCR?
8. How does a `environment:` block in a GitHub Actions job enforce manual approval?

---

## Final Exercises

1. A `terraform apply` on prod fails mid-way. Some resources are created, others are not. The state is now partially updated. What do you do? In what order?

2. The canary is at 20% when you observe a 2x increase in p99 latency. Walk through the exact commands to roll back.

3. Your Trivy scan fails with `UNKNOWN` CVEs (scanner database outdated). The team is blocked. What are your options? Which is acceptable for production gating?

4. Write a GitHub Actions step that posts a Slack notification on pipeline failure using a webhook URL stored as `secrets.SLACK_WEBHOOK`. The message should include the workflow name, run URL, and failing job.

5. You need to add a new environment variable (`REDIS_URL`) to the ECS task definition. Describe every file you need to change (Terraform HCL, GitHub Actions YAML, application config) and in what order the changes should be deployed.

(Solutions: ../solutions/week-07/day-7.md)

## Self-check

- [ ] I can explain every stage in a production AI service CI/CD pipeline
- [ ] I can write a complete GitHub Actions workflow from memory
- [ ] I understand Terraform state, providers, modules, and the plan/apply cycle
- [ ] I can implement blue-green and canary deployments with Terraform + GitHub Actions
- [ ] I have a completed week project with a working pipeline and Terraform module

---

Take the [[assessment]] when ready.
