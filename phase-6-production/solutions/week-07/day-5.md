# Solutions — Week 07 · Day 5

## Exercise 1 — create_before_destroy on target groups

```hcl
resource "aws_lb_target_group" "blue" {
  name        = "${local.name_prefix}-blue"
  port        = var.app_port
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    path              = "/health"
    healthy_threshold = 2
    interval          = 30
  }

  lifecycle {
    create_before_destroy = true
  }
}
```

**Why necessary for zero-downtime replacement:**

ALB target groups cannot have the same name within the same VPC. If Terraform needs to replace a target group (e.g., because `target_type` changed from `instance` to `ip` — a destroy-and-create operation), the default behavior is:
1. Destroy old target group (listener now points at nothing — downtime).
2. Create new target group.
3. Update listener to point at new group.

With `create_before_destroy = true`:
1. Create new target group (both exist simultaneously).
2. Update listener to point at new group.
3. Destroy old target group.

This requires unique name generation (use a random suffix or include the target type in the name) to avoid name conflicts during the overlap period.

---

## Exercise 2 — Workspace strategy with .tfvars files

```bash
# Directory layout:
infra/
├── main.tf
├── variables.tf
├── terraform.tf
├── dev.tfvars        # gitignored for any sensitive values
├── staging.tfvars    # gitignored if it contains db_password
└── prod.tfvars       # gitignored (contains sensitive values)
```

```hcl
# infra/staging.tfvars
environment       = "staging"
aws_region        = "eu-west-1"
db_instance_class = "db.t3.micro"
canary_weight     = 0
# db_password passed via TF_VAR_db_password env var — never in file
```

```hcl
# infra/prod.tfvars
environment       = "prod"
aws_region        = "eu-west-1"
db_instance_class = "db.t3.small"
canary_weight     = 0
```

```bash
# Usage:
terraform -chdir=infra apply -var-file=staging.tfvars
terraform -chdir=infra apply -var-file=prod.tfvars

# With sensitive vars via env:
export TF_VAR_db_password="$(aws secretsmanager get-secret-value \
  --secret-id staging-ai-service/db-password --query SecretString --output text)"
terraform -chdir=infra apply -var-file=staging.tfvars
```

**What .tfvars files contain vs. what stays in env vars:**
- `.tfvars`: non-sensitive configuration (environment name, instance sizes, region, feature flags).
- Environment variables (`TF_VAR_*`): sensitive values (passwords, API keys, tokens). Never in `.tfvars` files that could be accidentally committed.

---

## Exercise 3 — skip_final_snapshot conditional safety

```hcl
resource "aws_db_instance" "main" {
  skip_final_snapshot = var.environment != "prod"
  # Evaluates to: true for dev/staging, false for prod
}
```

**Why this is safe:**

- For dev/staging: `skip_final_snapshot = true` means `terraform destroy` deletes the RDS instance without taking a final snapshot. This is intentional — dev/staging data is ephemeral; the cost of storing an unnecessary snapshot is wasteful.
- For prod: `skip_final_snapshot = false` means `terraform destroy` requires a `final_snapshot_identifier` to be set, or it will fail. This forces an explicit decision point before prod data is destroyed.

**What happens on accidental `terraform destroy` on prod without this:**

If `skip_final_snapshot = true` is set on prod, `terraform destroy` immediately deletes the database and all its data with no recovery option. The accidental destroy completes in < 2 minutes and is irreversible.

Defense-in-depth: also set `deletion_protection = true` on prod RDS (a separate safeguard at the AWS API level — Terraform destroy fails even before reaching snapshot logic).

---

## Exercise 4 — Real CloudWatch metrics check with boto3

```python
import logging
import time
from datetime import UTC, datetime, timedelta

import boto3

logger = logging.getLogger(__name__)


def check_metrics(
    target_group_arn: str,
    alb_arn: str,
    bake_seconds: int,
    region: str = "eu-west-1",
) -> bool:
    """Check CloudWatch metrics for canary target group health.

    Returns True if metrics are healthy, False if rollback should be triggered.
    """
    logger.info("Baking for %d seconds before checking metrics...", bake_seconds)
    time.sleep(bake_seconds)

    cw = boto3.client("cloudwatch", region_name=region)

    now = datetime.now(UTC)
    start = now - timedelta(minutes=5)

    # Extract short form of ARNs for metric dimensions
    # ALB dimension: "app/my-alb/1234abc"
    alb_suffix = alb_arn.split("loadbalancer/")[-1]
    # Target group dimension: "targetgroup/my-tg/5678def"
    tg_suffix = target_group_arn.split(":")[-1]

    response = cw.get_metric_statistics(
        Namespace="AWS/ApplicationELB",
        MetricName="HTTPCode_Target_5XX_Count",
        Dimensions=[
            {"Name": "LoadBalancer", "Value": alb_suffix},
            {"Name": "TargetGroup", "Value": tg_suffix},
        ],
        StartTime=start,
        EndTime=now,
        Period=300,
        Statistics=["Sum"],
    )

    datapoints = response.get("Datapoints", [])
    total_5xx = sum(dp["Sum"] for dp in datapoints)

    # Also check request count for context
    req_response = cw.get_metric_statistics(
        Namespace="AWS/ApplicationELB",
        MetricName="RequestCount",
        Dimensions=[
            {"Name": "LoadBalancer", "Value": alb_suffix},
            {"Name": "TargetGroup", "Value": tg_suffix},
        ],
        StartTime=start,
        EndTime=now,
        Period=300,
        Statistics=["Sum"],
    )
    total_requests = sum(dp["Sum"] for dp in req_response.get("Datapoints", []))

    if total_requests == 0:
        logger.warning("No requests to canary in last 5 min — cannot assess health")
        return True  # No traffic yet — conservative: allow promotion

    error_rate = total_5xx / total_requests
    logger.info(
        "Canary error rate: %.2f%% (%d errors / %d requests)",
        error_rate * 100,
        int(total_5xx),
        int(total_requests),
    )

    threshold = 0.01  # 1% error rate
    healthy = error_rate < threshold
    if not healthy:
        logger.warning("Error rate %.2f%% exceeds threshold %.2f%% — triggering rollback", error_rate * 100, threshold * 100)
    return healthy
```

---

## Exercise 5 — Blue-green swap script

```bash
#!/usr/bin/env bash
# scripts/blue_green_swap.sh
# Promote green to 100%, decommission blue.
set -euo pipefail

WORKSPACE="${1:-staging}"
IMAGE_TAG="${2:-}"

if [[ -z "$IMAGE_TAG" ]]; then
  echo "Usage: $0 <workspace> <image_tag>" >&2
  exit 1
fi

echo "Swapping to green — workspace=${WORKSPACE} image=${IMAGE_TAG}"

# Step 1: Set canary_weight to 100 (all traffic to green)
terraform -chdir=infra apply -auto-approve \
  -var="environment=${WORKSPACE}" \
  -var="image_tag=${IMAGE_TAG}" \
  -var="canary_weight=100"

echo "Traffic fully on green. Waiting 60s before decommissioning blue..."
sleep 60

# Step 2: Scale blue ECS service to 0 desired count
# (In Terraform, controlled by a separate variable; here we call AWS CLI directly)
CLUSTER_NAME="${WORKSPACE}-ai-service"
BLUE_SERVICE="${WORKSPACE}-ai-service-blue"

aws ecs update-service \
  --cluster "${CLUSTER_NAME}" \
  --service "${BLUE_SERVICE}" \
  --desired-count 0 \
  --region eu-west-1 \
  --query 'service.{name:serviceName,desired:desiredCount}' \
  --output table

echo "Blue decommissioned. Swap complete."
```

**Terraform changes required for a full blue-green implementation:**

1. Two ECS services: one named `-blue`, one named `-green`, each with its own task definition and target group.
2. Two ALB target groups with weighted listener rules (`blue_weight` + `green_weight` variables that sum to 100).
3. A `active_color` variable (or output) indicating which side is "stable" — used by the rollback script to know which side to promote back.
4. The swap is a `terraform apply` with `canary_weight=100` followed by decommissioning the old side.
