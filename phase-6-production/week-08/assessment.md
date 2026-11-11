# Week 08 Assessment — Cloud Deployment for AI APIs

> Complete all sections independently before checking the answer key.  
> Target time: ~75 min  
> Answer key: ../solutions/week-08/assessment.md

---

## Conceptual (interview-style)

Answer each question in 3–6 sentences. No code required.

**Q1.** An ECS Fargate task for an AI API has CPU utilization sitting at 15%, yet users are reporting high latency. Your autoscaling policy is CPU-based (target 60%) and is not triggering. Explain why this happens and what metric you would use instead.

**Q2.** Describe the difference between the ECS *task execution role* and the ECS *task role*. Give a concrete example of an IAM permission that belongs on each. Why is it a security risk to combine them into a single role?

**Q3.** You have a Secrets Manager secret containing an Anthropic API key. List every place in the AWS stack where this key could accidentally appear if you used ECS native secrets injection (the `secrets` field in the task definition container definition), and explain why fetching it via boto3 in application code is safer.

**Q4.** Explain target tracking autoscaling vs. step scaling. For an AI API that experiences very sudden, sharp traffic spikes (0 to 200 RPS in under 30 seconds), which would you choose and why? What is the fundamental limitation of both approaches?

**Q5.** A manager asks why the Terraform code sets `ignore_changes = [desired_count]` on the ECS service. Explain the root cause of the conflict that would occur without this setting, using the terms "Terraform state", "Application Auto Scaling", and "desired count drift."

---

## Coding Challenge

Implement both parts from scratch. No copy-paste from the lesson files — write it from memory, then verify.

### Part A — Terraform: ECS Service with Autoscaling

Write valid Terraform (`week_08_challenge.tf`) that provisions:

1. An `aws_appautoscaling_target` for an ECS service named `"my-ai-api"` in cluster `"prod-cluster"`, with min capacity 2 and max capacity 30.

2. A **target tracking** `aws_appautoscaling_policy` named `"my-ai-api-rpt"` that:
   - Targets `ALBRequestCountPerTarget` at a value of 100 requests per task.
   - Uses a 60-second scale-out cooldown and 300-second scale-in cooldown.
   - References an `alb_resource_label` variable for the metric's `resource_label`.

3. A **scheduled** `aws_appautoscaling_scheduled_action` that:
   - Fires at 07:50 UTC every weekday (Mon–Fri).
   - Sets minimum capacity to 8 and maximum capacity to 30.
   - Name it `"morning-prewarm"`.

**Acceptance criteria:**
- All three resources present and correctly typed.
- No hardcoded account IDs, ARNs, or secret values.
- Cooldown values correct.
- Cron expression syntactically valid for AWS (use `cron(...)` format).
- Variables used instead of literals for cluster name, service name, min/max capacity.

### Part B — Python: Secrets Manager fetcher

Write a Python 3.12+ module (`week_08_secrets.py`) that:

1. Defines a function `fetch_secret(secret_arn: str, region: str) -> dict[str, Any]` that:
   - Uses `boto3` to call `secretsmanager.get_secret_value`.
   - Parses the JSON string from `SecretString`.
   - Raises a `RuntimeError` with a meaningful message on `ClientError`.
   - Uses `logging` (not `print`) for all output.

2. Defines a function `get_api_key(secret_arn: str, region: str, key_name: str) -> str` that:
   - Calls `fetch_secret`.
   - Returns the string value for `key_name`.
   - Raises `KeyError` if the key is missing from the secret payload.

3. Has a `__main__` block that:
   - Reads `SECRET_ARN` from `os.getenv` (raises `SystemExit` with a message if not set).
   - Reads `AWS_REGION` from `os.getenv` with a default of `"us-east-1"`.
   - Reads `KEY_NAME` from `os.getenv` with a default of `"anthropic_api_key"`.
   - Calls `get_api_key` and logs the key length (NOT the key value).

**Acceptance criteria:**
- Full type hints on both function signatures.
- No hardcoded secret values anywhere.
- `ClientError` caught and re-raised as `RuntimeError`.
- `logging` used, not `print`.
- `__main__` uses `os.getenv` for all config — no argparse required but allowed.
- Key value never logged; only its length.

---

## Rubric (pass ≥ 75%)

| Dimension | Weight | Indicators |
|---|---|---|
| Independent implementation | 40% | Terraform and Python written without referencing lesson code; resources correct; variables used throughout |
| Conceptual understanding | 25% | Q1–Q5 answers demonstrate first-principles understanding, not surface-level parroting |
| Debugging & problem-solving | 20% | Q1 and Q4 show systematic diagnosis; Trade-offs articulated with specifics, not generalities |
| Code quality & testing | 15% | Type hints present; no secrets in code; logging not print; error handling present; code is readable and would pass ruff |

**Scoring guide:**
- 90–100%: Ready to own cloud AI deployments in production.
- 75–89%: Solid — review any missed concepts before moving to Week 9.
- 60–74%: Re-read Day 5 + Day 6; re-attempt the coding challenge.
- <60%: Revisit the full week before proceeding.

Answer key: ../solutions/week-08/assessment.md
