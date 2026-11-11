# Week 8 Assessment — Answer Key

## Conceptual Questions (5 Questions)

---

### Q1: Compare containers-on-ECS, serverless Lambda, and managed SageMaker for deploying an AI inference API. When would you choose each?

**Model Answer:**

| Dimension | ECS Fargate | Lambda | SageMaker Endpoints |
|-----------|-------------|--------|---------------------|
| Cold start | None (pre-warmed min tasks) | 2–30s with large container images | Minutes (model loading) |
| Max runtime | Unlimited | 15 minutes | Unlimited |
| GPU support | Via EC2 capacity provider | No | Yes (native) |
| Pricing model | Per task-second (min 1 min) | Per 100ms invocation | Per hour of endpoint |
| Scaling speed | 30–60s (new task) | Near-instant (Provisioned Concurrency) | 2–5 min |
| Operational burden | Medium | Low | Low |
| Idle cost | Min ~$60/month (2 tasks) | $0 (pure on-demand) | Min ~$60/month |

**Choose ECS Fargate when:**
- The application has significant startup overhead (model loading, connection pooling)
- You need persistent TCP connections to RDS/Redis
- p95 latency SLA is tight and cold starts are unacceptable
- Traffic is steady enough that idle costs are justified
- The team already has Docker expertise

**Choose Lambda when:**
- Workload is event-driven or bursty (batch triggered, webhook handler)
- The function is stateless and short-running (< 2 min)
- Traffic is sparse — paying $60/month idle for a dev endpoint is wasteful
- You need true scale-to-zero cost efficiency
- Cold start is acceptable (background jobs, async processing)
- **Caveat:** Use Provisioned Concurrency if Lambda is user-facing and cold starts are a problem

**Choose SageMaker Endpoints when:**
- You are deploying your own trained model (not calling a third-party API)
- You need GPU inference (SageMaker manages GPU instance types, multi-model endpoints)
- You want A/B traffic splitting between model versions natively
- You need model monitoring (data drift, prediction quality) built in
- You are already deep in the SageMaker MLOps ecosystem (Pipelines, Model Registry)

**Key insight:** For an API that *calls* Claude (not hosts a model), ECS is the right default. SageMaker is only relevant when *you* host the model weights. Lambda is a strong choice for async/batch AI workloads.

---

### Q2: Explain how ECS Fargate autoscaling with target tracking works. What happens during a traffic spike?

**Model Answer:**

Target tracking autoscaling works like a thermostat: you set a desired metric value (e.g., CPU = 60%), and AWS continuously adjusts the task count to maintain that value.

**Mechanism:**

1. **Continuous evaluation:** CloudWatch evaluates the metric (e.g., `ECSServiceAverageCPUUtilization`) every 60 seconds.
2. **Scale-out trigger:** When the metric exceeds the target for two consecutive periods, Application Auto Scaling calculates how many additional tasks are needed to bring the metric back to target. Formula approximation: `desired_tasks = ceil(current_tasks × current_metric / target_metric)`.
3. **New task startup:** ECS pulls the container image (from ECR, cached on the Fargate host), starts the container, and waits for the health check to pass. Total time: ~30–90 seconds.
4. **ALB registration:** Once healthy, the new task joins the ALB target group and receives traffic.
5. **Scale-in protection:** The `scale_in_cooldown` (typically 300s) prevents scale-in until the metric has been below target for the full cooldown, preventing oscillation.

**During a traffic spike:**
```
t=0s:   Spike hits; 2 running tasks at 40% CPU → surge to 85% CPU
t=60s:  CloudWatch alarm triggers (1st evaluation period above threshold)
t=120s: 2nd period above threshold — scale-out fires
        desired = ceil(2 × 85 / 60) = ceil(2.83) = 3 tasks
t=150s: ECS starts 3rd task; health check starts
t=210s: 3rd task healthy, joins ALB; CPU drops to ~57%
        If traffic continues growing, another scale-out fires
t=510s: Traffic subsides; scale-in cooldown begins (300s)
t=810s: Metric stable below 60%; scale in to 2 tasks
```

**Why two cooldown values differ:**
- `scale_out_cooldown = 60s`: respond fast to protect availability
- `scale_in_cooldown = 300s`: wait longer to avoid "scale in, spike, scale out, spike" oscillation (thrashing)

---

### Q3: What is the cost difference between storing secrets in ECS environment variables vs. Secrets Manager? What are the security implications of each?

**Model Answer:**

**Cost comparison:**

| Approach | Monthly cost (3 secrets) | Notes |
|----------|--------------------------|-------|
| ECS environment variables | $0 | No external service call |
| SSM Parameter Store (Standard) | $0 | Free tier; no rotation |
| Secrets Manager | $1.20 | $0.40/secret/month + $0.05/10K API calls |

With caching (fetch once at startup, re-use for 5 minutes), 2 ECS tasks fetching 3 secrets restart once per day: 2 × 3 × 30 = 180 API calls/month — well within the 10K free tier. Effective extra cost: **$1.20/month**.

**Security implications:**

ECS environment variables (plaintext):
- Value stored in the task definition JSON — visible in ECS console to anyone with `ecs:DescribeTaskDefinitions`
- Stored in Terraform state file (S3 bucket) — readable by anyone with S3 access to state bucket
- Visible in `aws ecs describe-tasks` API output
- No rotation mechanism — key rotation requires a new task definition and deployment
- No audit trail of who accessed the value

Secrets Manager:
- Encrypted at rest with KMS (AES-256)
- Access governed by IAM — `secretsmanager:GetSecretValue` is explicit and auditable
- Every access logged in CloudTrail (who accessed, when, from which IP/role)
- Native auto-rotation for RDS passwords; custom rotation Lambda for other secret types
- Value never appears in task definition, Terraform state, or ECS API outputs
- Enables zero-downtime secret rotation (new version created, tasks pick it up on next cache TTL)

**Conclusion:** The $1.20/month cost of Secrets Manager is trivially offset by the security and operational benefits. For any production AI API with external API keys, Secrets Manager is the correct choice.

---

### Q4: Describe the role of the NAT Gateway in a VPC with private subnets. What alternatives exist for AWS service access, and why does it matter for cost?

**Model Answer:**

**NAT Gateway role:**

ECS tasks run in private subnets (no direct internet route) for security. When they need to reach the public internet — to call `api.anthropic.com`, pull ECR images, or fetch Secrets Manager secrets — traffic must be routed through a NAT Gateway (Network Address Translation).

Architecture: `ECS task (private subnet) → NAT Gateway (public subnet) → Internet Gateway → Internet`

**Cost structure:**
- NAT Gateway hourly: $0.045/hour × 24 × 30 = **$32.40/month**
- Data processing: **$0.045/GB** for every byte that passes through NAT

For an AI API pulling 100 GB/month from S3 (model files, training data): $32.40 + $4.50 = **$36.90/month just for S3 traffic**.

**Alternatives for AWS service access:**

1. **VPC Gateway Endpoints** (S3, DynamoDB) — **FREE**
   - Traffic routes directly to AWS backbone, bypassing NAT
   - Added as route table entries in private subnet routing tables
   - Zero data processing charge
   - Eliminates NAT for the most data-intensive AWS services

2. **VPC Interface Endpoints** (Secrets Manager, ECR, CloudWatch, STS, etc.)
   - Creates an ENI in the private subnet; traffic never hits NAT
   - Cost: ~$0.01/hour per AZ + $0.01/GB → ~$7.20/month per endpoint per AZ
   - Still cheaper than NAT for services with high call volume

3. **NAT Gateway per AZ** (for resilience)
   - For production, deploy NAT Gateway in each AZ ($32.40 × 3 = $97.20/month)
   - For dev/staging, single NAT is fine (saves $65/month, accept AZ dependency)

**Why it matters:**

For an AI workload fetching 1 TB/month of data from S3 (embeddings, datasets, model weights):
- Through NAT: $32.40 + (1024 × $0.045) = **$78.48/month**
- Through S3 Gateway Endpoint: **$0/month**

Gateway endpoints for S3 are a mandatory optimization for any AI workload. Adding one takes 5 minutes and 5 lines of Terraform.

---

### Q5: What is a deployment circuit breaker in ECS, and why is it important for a production AI service?

**Model Answer:**

**What it is:**

The ECS deployment circuit breaker is a feature that automatically detects when a rolling deploy is failing and rolls back to the last healthy task definition.

**How it works:**
1. ECS starts new tasks with the updated image
2. If the new tasks fail to reach `RUNNING` state and pass health checks within a threshold (default: 10 consecutive failures, or failure rate > 50% of the new desired count), the circuit breaker trips
3. With `rollback = true`, ECS immediately reverts the service to the previous task definition version
4. The deployment state transitions to `FAILED` rather than getting stuck

```hcl
deployment_circuit_breaker {
  enable   = true
  rollback = true  # Automatically restore previous version
}
```

**Without circuit breaker — failure scenario:**

1. Deploy new image with a bug (e.g., wrong `ANTHROPIC_API_KEY` secret name → startup crash)
2. New tasks start, crash, ECS tries to replace them, they crash again
3. Service stuck in crash loop — rolling deploy holds `minimum_healthy_percent = 100` so old tasks stay running, but you get paged and must manually intervene
4. Mean time to recovery (MTTR) = time-to-alert + time-to-diagnose + time-to-rollback manually

**With circuit breaker:**

1. Same deployment with bug
2. 10 new tasks fail health checks
3. Circuit breaker trips — ECS rolls back to previous task definition automatically
4. Old tasks continue serving traffic uninterrupted
5. MTTR = minutes (automatic) vs. 30–60 minutes (manual)

**Why critical for AI services specifically:**

AI services often fail at startup in non-obvious ways:
- Model file download from S3 fails (wrong IAM role, missing VPC endpoint)
- Secret injection fails (wrong secret name or execution role missing permission)
- Out-of-memory during model loading (task memory too low)
- External API key invalid (Anthropic key rotated, new secret not yet propagated)

All of these cause the container to crash immediately after start. Without circuit breaker, a bad deploy to production takes down the service until manually remediated. With it, the blast radius is zero — old tasks keep serving.

---

## Coding Challenge

### Challenge 1: Terraform — ECS Service with Autoscaling

```hcl
# ── variables.tf ──────────────────────────────────────────────────────────────
variable "app_name" {
  type    = string
  default = "ai-api"
}

variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "environment" {
  type    = string
  default = "production"
}

variable "image_uri" {
  description = "Full ECR image URI including tag"
  type        = string
}

variable "task_cpu" {
  type    = number
  default = 1024
}

variable "task_memory" {
  type    = number
  default = 2048
}

variable "min_tasks" {
  type    = number
  default = 2
}

variable "max_tasks" {
  type    = number
  default = 20
}

variable "cpu_target" {
  description = "Target CPU utilization % for autoscaling"
  type        = number
  default     = 60
}

variable "vpc_id" {
  type = string
}

variable "private_subnet_ids" {
  type = list(string)
}

variable "alb_target_group_arn" {
  type = string
}

variable "alb_security_group_id" {
  type = string
}

# ── main.tf ───────────────────────────────────────────────────────────────────
terraform {
  required_version = ">= 1.7"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_caller_identity" "current" {}

# ── CloudWatch Logs ───────────────────────────────────────────────────────────
resource "aws_cloudwatch_log_group" "app" {
  name              = "/ecs/${var.app_name}"
  retention_in_days = 30
}

# ── IAM: Task Execution Role ──────────────────────────────────────────────────
resource "aws_iam_role" "execution" {
  name = "${var.app_name}-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "execution_managed" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "execution_secrets" {
  name = "${var.app_name}-secrets"
  role = aws_iam_role.execution.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue", "kms:Decrypt"]
      Resource = "arn:aws:secretsmanager:${var.aws_region}:${data.aws_caller_identity.current.account_id}:secret:${var.app_name}/*"
    }]
  })
}

# ── IAM: Task Role ────────────────────────────────────────────────────────────
resource "aws_iam_role" "task" {
  name = "${var.app_name}-task-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

# ── Secrets Manager ───────────────────────────────────────────────────────────
resource "aws_secretsmanager_secret" "api_keys" {
  name        = "${var.app_name}/keys"
  description = "External API keys"
}

# ── Security Group ────────────────────────────────────────────────────────────
resource "aws_security_group" "ecs" {
  name        = "${var.app_name}-ecs"
  description = "ECS tasks — accept only from ALB"
  vpc_id      = var.vpc_id

  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [var.alb_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ── ECS Cluster ───────────────────────────────────────────────────────────────
resource "aws_ecs_cluster" "main" {
  name = var.app_name
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# ── Task Definition ───────────────────────────────────────────────────────────
resource "aws_ecs_task_definition" "app" {
  family                   = var.app_name
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([{
    name      = var.app_name
    image     = var.image_uri
    essential = true

    portMappings = [{ containerPort = 8000, protocol = "tcp" }]

    secrets = [{
      name      = "ANTHROPIC_API_KEY"
      valueFrom = "${aws_secretsmanager_secret.api_keys.arn}:ANTHROPIC_API_KEY::"
    }]

    environment = [
      { name = "APP_ENV",    value = var.environment },
      { name = "AWS_REGION", value = var.aws_region }
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.app.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ecs"
      }
    }

    healthCheck = {
      command     = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 60
    }

    stopTimeout = 30
  }])
}

# ── ECS Service ───────────────────────────────────────────────────────────────
resource "aws_ecs_service" "app" {
  name            = var.app_name
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = var.min_tasks

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = var.alb_target_group_arn
    container_name   = var.app_name
    container_port   = 8000
  }

  deployment_configuration {
    maximum_percent         = 200
    minimum_healthy_percent = 100
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  lifecycle {
    ignore_changes = [desired_count]
  }

  depends_on = [aws_iam_role_policy_attachment.execution_managed]
}

# ── Autoscaling ───────────────────────────────────────────────────────────────
resource "aws_appautoscaling_target" "ecs" {
  max_capacity       = var.max_tasks
  min_capacity       = var.min_tasks
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.app.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
  depends_on         = [aws_ecs_service.app]
}

resource "aws_appautoscaling_policy" "cpu" {
  name               = "${var.app_name}-cpu-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value       = var.cpu_target
    scale_in_cooldown  = 300
    scale_out_cooldown = 60
  }
}

# ── Outputs ───────────────────────────────────────────────────────────────────
output "cluster_name" {
  value = aws_ecs_cluster.main.name
}

output "service_name" {
  value = aws_ecs_service.app.name
}

output "secret_arn" {
  value = aws_secretsmanager_secret.api_keys.arn
}
```

---

### Challenge 2: Python boto3 Secret Fetcher

```python
# secret_fetcher.py
"""
Week 8 Assessment — Python boto3 Secrets Manager fetcher.

Features:
- Fetches JSON secrets from AWS Secrets Manager
- In-memory TTL cache (configurable, default 5 min)
- Thread-safe via lock
- Structured logging
- Type-annotated (Python 3.12+)
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class SecretCache:
    """Thread-safe TTL cache for Secrets Manager values."""

    def __init__(self, ttl_seconds: int = 300) -> None:
        self._ttl = ttl_seconds
        self._store: dict[str, tuple[dict[str, Any], float]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            if key not in self._store:
                return None
            value, fetched_at = self._store[key]
            if time.monotonic() - fetched_at > self._ttl:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._store[key] = (value, time.monotonic())

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)


class SecretFetcher:
    """
    Fetches and caches secrets from AWS Secrets Manager.

    Args:
        region: AWS region. Defaults to AWS_REGION env var or us-east-1.
        ttl_seconds: Cache TTL. Set to 0 to disable caching.

    Example:
        fetcher = SecretFetcher()
        api_key = fetcher.get_value("myapp/keys", "ANTHROPIC_API_KEY")
    """

    def __init__(
        self,
        region: str | None = None,
        ttl_seconds: int = 300,
    ) -> None:
        self._region = region or os.getenv("AWS_REGION", "us-east-1")
        self._cache = SecretCache(ttl_seconds=ttl_seconds)
        self._client = boto3.client("secretsmanager", region_name=self._region)
        logger.info("SecretFetcher ready", extra={"region": self._region, "ttl": ttl_seconds})

    def get(self, secret_name: str) -> dict[str, Any]:
        """
        Return parsed JSON secret dict, using cache when available.

        Raises:
            ClientError: On AWS API failure (permission denied, not found, etc.)
            ValueError: If the secret is not valid JSON.
        """
        cached = self._cache.get(secret_name)
        if cached is not None:
            logger.debug("Cache hit", extra={"secret": secret_name})
            return cached

        logger.info("Fetching secret", extra={"secret": secret_name})
        try:
            response = self._client.get_secret_value(SecretId=secret_name)
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            logger.error("Secrets Manager error", extra={"secret": secret_name, "code": code})
            raise

        raw = response.get("SecretString", "{}")
        try:
            parsed: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Secret '{secret_name}' is not valid JSON: {exc}") from exc

        self._cache.set(secret_name, parsed)
        return parsed

    def get_value(self, secret_name: str, key: str) -> str:
        """
        Convenience: extract one key from a JSON secret.

        Raises:
            KeyError: If the key does not exist in the secret.
        """
        secret = self.get(secret_name)
        if key not in secret:
            raise KeyError(f"Key '{key}' not found in secret '{secret_name}'")
        return str(secret[key])

    def invalidate(self, secret_name: str) -> None:
        """Force re-fetch on next access (use after rotation)."""
        self._cache.invalidate(secret_name)
        logger.info("Cache invalidated", extra={"secret": secret_name})


# ── Singleton for module-level use ────────────────────────────────────────────
_fetcher: SecretFetcher | None = None
_fetcher_lock = threading.Lock()


def get_fetcher() -> SecretFetcher:
    """Return module-level singleton SecretFetcher."""
    global _fetcher
    if _fetcher is None:
        with _fetcher_lock:
            if _fetcher is None:
                _fetcher = SecretFetcher()
    return _fetcher
```

**Usage in a FastAPI lifespan:**

```python
# main.py
import logging
import os
from contextlib import asynccontextmanager

import anthropic
from fastapi import FastAPI

from secret_fetcher import get_fetcher

logger = logging.getLogger(__name__)

_anthropic_client: anthropic.AsyncAnthropic | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _anthropic_client

    # Load API key at startup — not on every request
    if os.getenv("APP_ENV", "development") == "development":
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
    else:
        api_key = get_fetcher().get_value("ai-api/keys", "ANTHROPIC_API_KEY")

    _anthropic_client = anthropic.AsyncAnthropic(api_key=api_key)
    logger.info("Application startup complete")

    yield

    logger.info("Application shutdown")


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
```

---

### Scoring Rubric

**Conceptual (50 points — 10 each):**
- Q1: Award full marks for correct identification of cold start as Lambda blocker, GPU as SageMaker differentiator, and ECS as steady-traffic default
- Q2: Must mention CloudWatch metric evaluation, scale-out formula, task startup sequence, and why two different cooldowns
- Q3: Must mention Terraform state exposure, CloudTrail auditing, and rotation as key differentiators
- Q4: Must correctly state Gateway endpoints are free, identify S3 as highest-value endpoint, and calculate comparative cost
- Q5: Must explain failure threshold mechanism, automatic rollback, and give an AI-specific failure scenario

**Coding challenge (50 points):**
- Terraform (30 points): ECS task definition with `secrets` (not `environment`) field, autoscaling target + policy, circuit breaker, correct IAM roles (execution vs task), health check
- Python (20 points): boto3 client, JSON parsing, TTL cache, thread safety, proper error handling, no hardcoded credentials
