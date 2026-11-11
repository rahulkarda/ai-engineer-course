# Phase 6 · Week 07 · Day 5 — IaC for an AI Service + Deployment Strategies

> Objective: Provision complete AI service infrastructure with Terraform; implement blue-green and canary deployments.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

## Concept

An AI service typically needs more infrastructure than a standard web service:

- **Container service** — to run the FastAPI/LangGraph app.
- **Managed database** — persistent state (conversation history, evaluations).
- **Object storage** — model artifacts, prompt templates, eval datasets.
- **Secrets management** — API keys for LLM providers.
- **Load balancer** — traffic routing for deployment strategies.

All of this can be expressed as Terraform code, making every environment identical and every change auditable.

### Deployment strategies

| Strategy | How it works | Rollback | Risk |
|---|---|---|---|
| **Recreate** | Stop old, start new | Redeploy old version | Downtime |
| **Rolling** | Replace instances one by one | Wait for rollback | Partial exposure |
| **Blue-green** | Two identical envs, flip traffic | Flip back instantly | Double infra cost |
| **Canary** | Route X% to new, increase over time | Reduce % to 0 | Slowest, safest |
| **A/B test** | Route by user segment | Per-segment control | Complex routing |

For AI services, **canary** is preferred because model behavior changes are subtle and need time to surface in metrics (latency, error rate, response quality scores).

## How it works internally

### Blue-green

```
DNS / Load Balancer
        │
   ┌────┴────┐
   │         │
[Blue]    [Green]
 v1.2       v1.3  ← deploy here first
   │
 100% traffic initially

After health checks pass:
   DNS / Load Balancer
        │
   ┌────┴────┐
   │         │
[Blue]    [Green]
 v1.2       v1.3
              │
           100% traffic (swap)

Blue kept alive for instant rollback window (e.g. 30 min)
```

### Canary

```
Load Balancer (weighted routing)
   ├── 95% → Stable (v1.2)
   └──  5% → Canary  (v1.3)

Metrics gate (5 min):
  error rate < 1%? latency p99 < 500ms?
    YES → promote to 20%, then 50%, then 100%
    NO  → rollback: set canary weight to 0%
```

In AWS: use **ALB weighted target groups** or **ECS service with CodeDeploy**. In Kubernetes: use `Argo Rollouts` or `Flagger`.

## Worked example

### Complete Terraform: ECS + RDS + ALB (production-ready)

```hcl
# infra/main.tf

locals {
  name_prefix = "${var.environment}-ai-service"
  common_tags = {
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# ── Networking ────────────────────────────────────────────────────────────────
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  tags                 = merge(local.common_tags, { Name = "${local.name_prefix}-vpc" })
}

resource "aws_subnet" "public" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet("10.0.0.0/16", 8, count.index)
  availability_zone = data.aws_availability_zones.available.names[count.index]
  tags              = merge(local.common_tags, { Name = "${local.name_prefix}-public-${count.index}" })
}

resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet("10.0.0.0/16", 8, count.index + 10)
  availability_zone = data.aws_availability_zones.available.names[count.index]
  tags              = merge(local.common_tags, { Name = "${local.name_prefix}-private-${count.index}" })
}

data "aws_availability_zones" "available" {
  state = "available"
}

# ── ALB for blue-green / canary traffic routing ───────────────────────────────
resource "aws_lb" "main" {
  name               = "${local.name_prefix}-alb"
  internal           = false
  load_balancer_type = "application"
  subnets            = aws_subnet.public[*].id
  tags               = local.common_tags
}

resource "aws_lb_target_group" "blue" {
  name        = "${local.name_prefix}-blue"
  port        = var.app_port
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    path                = "/health"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    interval            = 30
  }
}

resource "aws_lb_target_group" "green" {
  name        = "${local.name_prefix}-green"
  port        = var.app_port
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    path                = "/health"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    interval            = 30
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = "forward"
    forward {
      # Weighted routing: canary gets var.canary_weight percent
      target_group {
        arn    = aws_lb_target_group.blue.arn
        weight = 100 - var.canary_weight
      }
      target_group {
        arn    = aws_lb_target_group.green.arn
        weight = var.canary_weight
      }
    }
  }
}

# ── RDS (managed PostgreSQL) ──────────────────────────────────────────────────
resource "aws_db_subnet_group" "main" {
  name       = local.name_prefix
  subnet_ids = aws_subnet.private[*].id
  tags       = local.common_tags
}

resource "aws_db_instance" "main" {
  identifier              = local.name_prefix
  engine                  = "postgres"
  engine_version          = "16.1"
  instance_class          = var.db_instance_class
  allocated_storage       = 20
  storage_encrypted       = true
  db_name                 = "aiservice"
  username                = "aiservice"
  password                = var.db_password   # sensitive var, from Secrets Manager in prod
  db_subnet_group_name    = aws_db_subnet_group.main.name
  skip_final_snapshot     = var.environment != "prod"
  deletion_protection     = var.environment == "prod"
  backup_retention_period = var.environment == "prod" ? 7 : 1
  tags                    = local.common_tags

  lifecycle {
    prevent_destroy = false   # set to true in prod module call
  }
}

# ── S3 for model artifacts / eval datasets ────────────────────────────────────
resource "aws_s3_bucket" "artifacts" {
  bucket = "${local.name_prefix}-artifacts"
  tags   = local.common_tags
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# ── ECS Fargate service ───────────────────────────────────────────────────────
resource "aws_ecs_service" "ai_service" {
  name            = local.name_prefix
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.ai_service.arn
  desired_count   = var.environment == "prod" ? 3 : 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.private[*].id
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.blue.arn
    container_name   = "ai-service"
    container_port   = var.app_port
  }

  # Enable blue-green via CodeDeploy (production)
  deployment_controller {
    type = var.environment == "prod" ? "CODE_DEPLOY" : "ECS"
  }

  tags = local.common_tags
}
```

### variables.tf

```hcl
variable "environment"       { type = string }
variable "aws_region"        { type = string; default = "eu-west-1" }
variable "app_port"          { type = number; default = 8080 }
variable "image_tag"         { type = string; description = "Container image tag to deploy" }
variable "db_password"       { type = string; sensitive = true }
variable "db_instance_class" { type = string; default = "db.t3.micro" }
variable "canary_weight" {
  type        = number
  default     = 0
  description = "Percentage of traffic routed to canary (green) target group"
  validation {
    condition     = var.canary_weight >= 0 && var.canary_weight <= 100
    error_message = "canary_weight must be 0-100"
  }
}
```

### Canary promotion script

```python
#!/usr/bin/env python3
"""Canary promotion: gradually shift traffic from blue to green.

Usage:
    python scripts/canary_promote.py \
        --workspace staging \
        --steps 5,20,50,100 \
        --bake-time 300
"""
import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def terraform_apply(workspace: str, canary_weight: int, image_tag: str) -> int:
    cmd = [
        "terraform",
        "-chdir=infra",
        "apply",
        "-auto-approve",
        f"-var=canary_weight={canary_weight}",
        f"-var=image_tag={image_tag}",
        f"-var=environment={workspace}",
    ]
    logger.info("Setting canary weight to %d%%", canary_weight)
    result = subprocess.run(cmd, capture_output=False)  # noqa: S603
    return result.returncode


def check_metrics(bake_seconds: int) -> bool:
    """Placeholder: query your observability platform for error rate / latency."""
    logger.info("Baking for %d seconds — observing metrics...", bake_seconds)
    time.sleep(bake_seconds)
    # Replace with real metrics check: Datadog, CloudWatch, Prometheus
    # Return False to trigger rollback
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--image-tag", required=True)
    parser.add_argument("--steps", default="5,20,50,100")
    parser.add_argument("--bake-time", type=int, default=300)
    args = parser.parse_args()

    weights = [int(w) for w in args.steps.split(",")]

    for weight in weights:
        rc = terraform_apply(args.workspace, weight, args.image_tag)
        if rc != 0:
            logger.error("Terraform apply failed at weight %d%%", weight)
            sys.exit(1)

        if weight < 100:
            healthy = check_metrics(args.bake_time)
            if not healthy:
                logger.warning("Metrics unhealthy — rolling back to 0%%")
                terraform_apply(args.workspace, 0, args.image_tag)
                sys.exit(1)

    logger.info("Canary promotion complete — 100%% traffic on new version")


if __name__ == "__main__":
    main()
```

## Exercises

1. Add a `lifecycle { create_before_destroy = true }` block to the `aws_lb_target_group` resources. Why is this necessary for zero-downtime target group replacement?

2. The `canary_weight` variable defaults to 0. Write a `terraform workspace` strategy that stores different variable files per environment (`dev.tfvars`, `staging.tfvars`, `prod.tfvars`). What does each file contain?

3. Explain why `skip_final_snapshot = var.environment != "prod"` is a safe conditional. What happens if you accidentally run `terraform destroy` on prod without this?

4. The canary script sleeps for `bake_time` seconds. Replace the `check_metrics` stub with a real implementation that queries a CloudWatch metric using `boto3`. The metric: `HTTPCode_Target_5XX_Count` on the ALB target group.

5. Write a `blue_green_swap` shell script (can call Terraform) that promotes the green target group to receive 100% traffic and terminates the blue ECS service. What Terraform changes does this require?

(Solutions: ../solutions/week-07/day-5.md)

## Self-check

- [ ] I can provision ECS + RDS + ALB with Terraform
- [ ] I can explain the difference between blue-green and canary deployment
- [ ] I know how to implement weighted routing via ALB target groups
- [ ] I can write a canary promotion script with a rollback path
- [ ] I understand when to use `prevent_destroy` and `skip_final_snapshot`

## Supplementary links

- [AWS ECS blue/green deployments](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/deployment-type-bluegreen.html)
- [ALB weighted target groups](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/load-balancer-listeners.html#weighted-target-groups)
- [Terraform aws_lb_listener forward action](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/lb_listener#forward-action)
- [Argo Rollouts (Kubernetes canary)](https://argoproj.github.io/rollouts/)
