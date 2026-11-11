# Phase 6 · Week 08 · Day 7 — Project Day & Architecture Decision Record

> Objective: Integrate this week's deployed system into a coherent production architecture, write an ADR, and complete the week assessment.  ·  Time: ~3 hr project + 1 hr assessment review  ·  Prereqs: [[day-6]], working deploy from Day 6

## Concept

**Architecture Decision Records (ADRs)** are short documents that capture a significant architectural choice: the context, the decision, the alternatives considered, and the consequences. They are the institutional memory of a system — more valuable than code comments because they capture *why*, not *what*.

ADR format (Nygard / RFC-style):
1. **Status** — proposed / accepted / deprecated / superseded
2. **Context** — the forces at play (technical, team, business constraints)
3. **Decision** — the choice made
4. **Alternatives considered** — what was rejected and why
5. **Consequences** — trade-offs, follow-up work, monitoring needed

ADRs live in `docs/adr/` alongside the code. They are immutable once accepted — if a decision changes, a new ADR supersedes the old one.

**Week 8 integration checklist:**

By end of Day 6 you should have:
- A containerized FastAPI AI app
- An ECR repository with a pushed image
- ECS Fargate service with task execution role + task role
- Secrets Manager secret (populated)
- ALB with health-check-passing target group
- Application Auto Scaling target tracking policy
- CloudWatch log group

Today: verify the system end-to-end, document it, write the ADR, and run the assessment.

## How it works internally

**Why ADRs matter for AI systems specifically:**

AI system architecture decisions are higher-stakes than typical web apps:
- Model choice affects cost by orders of magnitude.
- Prompt location (inline vs. external store vs. system prompt) affects debuggability.
- Secrets handling for LLM API keys is a frequent security failure mode.
- Autoscaling thresholds interact with per-token cost in non-obvious ways.

Undocumented decisions create "haunted" systems — engineers are afraid to change things because they don't know why they were built that way.

**What makes a good ADR:**
- Written *at decision time*, not after the fact.
- Concise: 1–2 pages max.
- Honest about the trade-offs.
- Linked to the code/IaC that implements it.
- Includes a "review trigger" — a condition that would make you revisit the decision.

## Worked example

### End-to-end verification script

```python
"""
week_08/verify_deploy.py

Smoke-test a deployed AI API.
Usage: python -I verify_deploy.py

Environment variables:
  ALB_DNS   - ALB DNS name from terraform output (no protocol prefix)
  TEST_PROMPT - (optional) prompt to send to /infer
"""
from __future__ import annotations

import logging
import os
import sys
import urllib.request
import urllib.error
import json

logging.basicConfig(level="INFO", format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def check_health(base_url: str) -> bool:
    """Check /health returns 200."""
    url = f"{base_url}/health"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            body = json.loads(resp.read())
            ok = resp.status == 200 and body.get("status") == "ok"
            log.info("Health check: %s (status=%d)", "PASS" if ok else "FAIL", resp.status)
            return ok
    except urllib.error.URLError as exc:
        log.error("Health check FAIL: %s", exc)
        return False


def check_infer(base_url: str, prompt: str) -> bool:
    """POST to /infer and verify a non-empty result."""
    url = f"{base_url}/infer"
    payload = json.dumps({"prompt": prompt, "max_tokens": 64}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
            has_result = bool(body.get("result"))
            log.info(
                "Infer check: %s (model=%s, in=%d, out=%d)",
                "PASS" if has_result else "FAIL",
                body.get("model", "?"),
                body.get("input_tokens", 0),
                body.get("output_tokens", 0),
            )
            return has_result
    except urllib.error.URLError as exc:
        log.error("Infer check FAIL: %s", exc)
        return False


def main() -> int:
    alb_dns = os.getenv("ALB_DNS", "")
    if not alb_dns:
        log.error("ALB_DNS environment variable not set")
        return 1

    base_url = f"http://{alb_dns}"
    prompt = os.getenv("TEST_PROMPT", "Reply with exactly: deployment verified")

    results = [
        check_health(base_url),
        check_infer(base_url, prompt),
    ]

    passed = sum(results)
    log.info("Smoke tests: %d/%d passed", passed, len(results))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
```

---

### Sample ADR: ADR-001 — Use AWS Secrets Manager for API Key Storage

```markdown
# ADR-001: Use AWS Secrets Manager for LLM API Key Storage

**Status:** Accepted  
**Date:** 2026-11-11  
**Author:** Week 8 project  
**Supersedes:** —  
**Superseded by:** —

## Context

The AI API requires an Anthropic API key at runtime. This key must not appear in:
- Container images (ECR is accessible to any AWS principal with ECR:GetAuthorizationToken)
- Terraform state files (stored in S3, potentially accessible to IaC pipeline roles)
- Environment variables set at task-definition time (visible in `aws ecs describe-task-definition`)
- Application logs (logged accidentally as part of config dumps)

The key must be available to running tasks with low latency and must support rotation
without redeploying the container.

Team constraint: no dedicated secrets management infrastructure (no HashiCorp Vault).

## Decision

Fetch the Anthropic API key from AWS Secrets Manager at application startup using boto3,
via the ECS task role (not the task execution role). Cache the result in-process for the
task lifetime. The `SECRET_ARN` (not the value) is injected as an environment variable
via the ECS task definition.

## Alternatives Considered

| Option | Rejected because |
|---|---|
| Hardcode in Dockerfile | Obvious security failure; image reuse impossible |
| ECS secrets injection (native) | Value appears in `docker inspect` output |
| SSM Parameter Store | Secrets Manager has better rotation + KMS integration |
| Build-time env var | Leaks into image layer history |
| HashiCorp Vault | Not available; would add operational complexity |

## Consequences

**Positive:**
- Secret value never appears in IaC state, image, or AWS API responses.
- Rotation possible without task restart (requires cache invalidation — see follow-up).
- Audit trail via CloudTrail on every `GetSecretValue` call.

**Negative:**
- ~50ms cold start penalty for first secret fetch (mitigated by startup caching).
- Task role must grant `secretsmanager:GetSecretValue` — adds IAM complexity.
- Secret rotation requires application-level cache invalidation (not yet implemented).

## Follow-up Work

- [ ] Implement cache invalidation on secret rotation via EventBridge → ECS task drain.
- [ ] Add CloudWatch alarm on failed `GetSecretValue` calls.
- [ ] Document rotation runbook.

## Review Trigger

Revisit if: (a) Vault becomes available in the organization, (b) the number of secrets
grows beyond 10 (centralized management becomes valuable), or (c) a secret is
accidentally exposed via any vector listed in Context.
```

---

### Week integration: ASCII architecture diagram

```
                         ┌─────────────────────────────────────────────┐
                         │              AWS Account                     │
                         │                                              │
  Internet               │  ┌──────────────────────────────────────┐   │
  ─────────              │  │  VPC                                  │   │
  User / Client          │  │                                       │   │
      │                  │  │  Public Subnets      Private Subnets  │   │
      │  HTTP:80         │  │  ┌──────────┐        ┌────────────┐  │   │
      └──────────────────┼──┼─►│   ALB    │───────►│ ECS Fargate│  │   │
                         │  │  └──────────┘        │  Tasks     │  │   │
                         │  │                      │  (ai-api)  │  │   │
                         │  │                      └──────┬─────┘  │   │
                         │  │                             │         │   │
                         │  └─────────────────────────────┼────────┘   │
                         │                                │             │
                         │  ┌─────────────────┐          │             │
                         │  │ Secrets Manager │◄─boto3───┘             │
                         │  └─────────────────┘                        │
                         │                                              │
                         │  ┌─────────────────┐                        │
                         │  │  ECR (images)   │                        │
                         │  └─────────────────┘                        │
                         │                                              │
                         │  ┌─────────────────┐                        │
                         │  │  CloudWatch Logs│                        │
                         │  └─────────────────┘                        │
                         │                                              │
                         │  ┌─────────────────────────────────┐        │
                         │  │  Application Auto Scaling        │        │
                         │  │  (target: ALB rps/task ≤ 80)    │        │
                         │  └─────────────────────────────────┘        │
                         └─────────────────────────────────────────────┘
                                              │ HTTPS
                                              ▼
                                    ┌─────────────────┐
                                    │  Anthropic API  │
                                    │  (Claude)       │
                                    └─────────────────┘
```

## Exercises

1. **ADR practice**: Write ADR-002 for the autoscaling metric choice: why `ALBRequestCountPerTarget` over `ECSServiceAverageCPUUtilization`. Follow the same format as ADR-001. Include at least two alternatives and a review trigger.

2. **Cost audit**: Run the `cost_estimator.py` from Day 5 against your actual deployment profile (check CloudWatch for real RPS, check your Anthropic usage dashboard for actual token counts). How close was your Day 5 estimate?

3. **Log analysis**: Write a Python script that uses boto3 to fetch the last 50 log events from your CloudWatch log group and prints a summary: request count, error count, and the three most-frequent error messages. Use `logs.filter_log_events`.

4. **Runbook stub**: Write a 10-line runbook (plain text) for: "ECS tasks are failing health checks after a deploy." Include: first check, escalation path, rollback command.

(Solutions: ../solutions/week-08/day-7.md)

## Self-check

- [ ] I can describe the purpose of an ADR and explain what distinguishes a good one from a bad one.
- [ ] I can draw the Week 8 architecture from memory (ALB, ECS, Secrets Manager, ECR, CloudWatch, autoscaling).
- [ ] I can run a smoke test against my deployed API and interpret the results.
- [ ] I can write an ADR for an architectural decision I've made this week.
- [ ] I completed the Week 8 assessment with a passing score.

---

## Running the Week 8 Assessment

```bash
# From the week-08 directory:
# 1. Read assessment.md
# 2. Complete the coding challenge in a new file: week_08_challenge.tf + week_08_secrets.py
# 3. Self-grade against the rubric
# 4. Check solutions only after completing your attempt
cat assessment.md
```

**Time budget suggestion:**
- 20 min: conceptual questions (write answers, don't look up)
- 40 min: coding challenge (Terraform + Python from scratch)
- 15 min: self-review against rubric
- 15 min: review solution, note gaps

---

## Supplementary Links

- [Documenting Architecture Decisions — Michael Nygard (original ADR post)](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
- [ADR GitHub org — templates and tooling](https://adr.github.io/)
- [CloudWatch Logs Insights query syntax](https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/CWL_QuerySyntax.html)
- [ECS deployment rollback](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/deployment-type-ecs.html)
- [boto3 Secrets Manager docs](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/secretsmanager.html)
