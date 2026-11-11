# Phase 6 · Week 08 · Day 6 — Full Deploy Walkthrough: AI FastAPI on ECS Fargate

> Objective: Deploy a production-ready containerized AI FastAPI app end-to-end on ECS Fargate with Secrets Manager, ALB, and autoscaling.  ·  Time: ~4 hr  ·  Prereqs: [[day-5]], Docker installed, AWS CLI configured

## Concept

A complete production deploy for an AI API involves six layers, each with a different failure mode:

1. **Container image** — reproducible build, minimal surface area, no secrets baked in.
2. **Container registry** — ECR; image scanning, lifecycle policies to control cost.
3. **Compute** — ECS Fargate: serverless containers, no EC2 management, per-second billing.
4. **Secrets** — AWS Secrets Manager fetched at runtime; never in env vars at build time, never in IaC state.
5. **Networking** — VPC, ALB, security groups; least-privilege ingress/egress.
6. **Autoscaling** — Application Auto Scaling on top of ECS service (covered Day 5, wired here).

Cloud-agnostic equivalents:

| Concern | AWS | GCP | Azure |
|---|---|---|---|
| Container registry | ECR | Artifact Registry | ACR |
| Serverless containers | ECS Fargate | Cloud Run | Container Apps |
| Secrets | Secrets Manager | Secret Manager | Key Vault |
| Load balancer | ALB | Cloud Load Balancing | App Gateway |
| Managed DB | RDS | Cloud SQL | Azure DB |

The design principle stays constant: **secrets at runtime, image is public-safe, IaC is the source of truth.**

## How it works internally

**ECS task execution flow:**

1. ECS scheduler places the task on Fargate infrastructure.
2. The **task execution role** (IAM) authorizes ECS to pull the image from ECR and fetch secrets from Secrets Manager — this role is used by the ECS control plane, not your app code.
3. The **task role** (IAM) is assumed by your container at runtime — only this role should have permissions your app needs (Bedrock, S3, etc.).
4. ECS injects secrets as environment variables OR your app calls `boto3` directly. Direct call is preferred — avoids secrets in `docker inspect` output.
5. The ALB listener forwards `HTTPS:443 → HTTP:8080` inside the VPC. The target group health-checks `/health` every 30s; unhealthy tasks are deregistered and replaced.

**Why Fargate over EC2 launch type:**

Fargate abstracts the underlying EC2 fleet. No AMI patching, no cluster capacity management, no SSH access surface. You pay per task (vCPU + memory × seconds). Trade-off: cold start is ~10–30s (vs near-instant for warm EC2). Mitigate with `minimumHealthyPercent: 100` + pre-warm scheduled scaling.

**Secrets Manager fetch flow:**

```
App startup
  └── boto3 secretsmanager.get_secret_value(SecretId=ARN)
        └── Decrypts with KMS CMK (or AWS-managed key)
              └── Returns JSON string → parse → use
```

The task role must have `secretsmanager:GetSecretValue` + `kms:Decrypt` on the specific secret ARN. Principle of least privilege: scope the IAM policy to the exact ARN, not `*`.

## Worked example

### Project structure

```
ai-api/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI entry point
│   ├── config.py        # pydantic-settings config
│   ├── secrets.py       # Secrets Manager client
│   ├── models.py        # Pydantic request/response models
│   └── routers/
│       ├── __init__.py
│       └── inference.py # /infer endpoint
├── Dockerfile
├── requirements.txt
└── terraform/
    ├── main.tf
    ├── variables.tf
    └── outputs.tf
```

---

### Dockerfile

```dockerfile
# Dockerfile
# Multi-stage build: builder installs deps; runtime is minimal.
# No secrets, no AWS credentials, no API keys.

# ── Stage 1: dependency builder ──────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build tools for any compiled deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime image ───────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Non-root user — principle of least privilege
RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid 1001 --no-create-home appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY app/ ./app/

# Health check baked into image (ALB also checks /health)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

USER appuser

EXPOSE 8080

# Uvicorn: 1 worker per task (Fargate tasks are single-purpose; scale via ECS)
CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1", \
     "--log-level", "info", \
     "--access-log"]
```

---

### App skeleton

```python
# app/config.py
"""Application configuration via pydantic-settings.

All sensitive values are loaded from Secrets Manager at runtime (see secrets.py).
Only non-sensitive config is read from environment variables here.
"""
from __future__ import annotations

import logging
import os

from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Non-sensitive application configuration."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "ai-api"
    log_level: str = "INFO"
    port: int = 8080

    # ARN of the Secrets Manager secret — not the secret value itself
    secret_arn: str = os.getenv("SECRET_ARN", "")

    # AWS region — injected by ECS task environment
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")

    # Claude model to use — no secrets here
    claude_model_id: str = "claude-sonnet-5-5"


settings = Settings()
```

```python
# app/secrets.py
"""Fetch secrets from AWS Secrets Manager at runtime.

The task execution role must grant secretsmanager:GetSecretValue.
Secret value is a JSON string: {"anthropic_api_key": "..."}
"""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Any

import boto3
from botocore.exceptions import ClientError

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_secret(secret_arn: str, region: str) -> dict[str, Any]:
    """Fetch and parse the secret JSON from Secrets Manager.

    Cached for process lifetime — secrets rotate requires task restart or
    manual cache invalidation.

    Args:
        secret_arn: Full ARN of the Secrets Manager secret.
        region: AWS region.

    Returns:
        Parsed secret dict.

    Raises:
        RuntimeError: If secret cannot be fetched.
    """
    if not secret_arn:
        raise RuntimeError(
            "SECRET_ARN environment variable not set. "
            "Set it to the Secrets Manager ARN for this environment."
        )

    client = boto3.client("secretsmanager", region_name=region)

    try:
        response = client.get_secret_value(SecretId=secret_arn)
    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        log.error("Failed to fetch secret %s: %s", secret_arn, error_code)
        raise RuntimeError(f"Cannot fetch secret: {error_code}") from exc

    raw = response.get("SecretString", "{}")
    parsed: dict[str, Any] = json.loads(raw)
    log.info("Secret fetched successfully from %s", secret_arn)
    return parsed


def get_anthropic_api_key() -> str:
    """Return the Anthropic API key from Secrets Manager."""
    from app.config import settings

    secret = get_secret(settings.secret_arn, settings.aws_region)
    key: str = secret.get("anthropic_api_key", "")
    if not key:
        raise RuntimeError("anthropic_api_key missing from secret payload")
    return key
```

```python
# app/models.py
"""Pydantic models for request/response boundaries."""
from __future__ import annotations

from pydantic import BaseModel, Field


class InferRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)
    max_tokens: int = Field(default=512, ge=1, le=4096)


class InferResponse(BaseModel):
    result: str
    model: str
    input_tokens: int
    output_tokens: int
```

```python
# app/routers/inference.py
"""Inference router — calls Claude via Anthropic SDK."""
from __future__ import annotations

import logging

import anthropic
from fastapi import APIRouter, HTTPException

from app.models import InferRequest, InferResponse
from app.secrets import get_anthropic_api_key
from app.config import settings

log = logging.getLogger(__name__)
router = APIRouter(prefix="/infer", tags=["inference"])


def _get_client() -> anthropic.Anthropic:
    """Construct Anthropic client with key from Secrets Manager."""
    return anthropic.Anthropic(api_key=get_anthropic_api_key())


@router.post("", response_model=InferResponse)
async def infer(body: InferRequest) -> InferResponse:
    """Run a single-turn inference against Claude."""
    client = _get_client()

    try:
        message = client.messages.create(
            model=settings.claude_model_id,
            max_tokens=body.max_tokens,
            messages=[{"role": "user", "content": body.prompt}],
        )
    except anthropic.APIError as exc:
        log.error("Anthropic API error: %s", exc)
        raise HTTPException(status_code=502, detail="Upstream API error") from exc

    content_block = message.content[0]
    result_text = content_block.text if hasattr(content_block, "text") else ""

    return InferResponse(
        result=result_text,
        model=message.model,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
    )
```

```python
# app/main.py
"""FastAPI application entry point."""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.config import settings
from app.routers import inference

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name, version="0.1.0")
app.include_router(inference.router)


@app.get("/health")
async def health() -> JSONResponse:
    """ALB health check endpoint — must return 200."""
    return JSONResponse({"status": "ok"})


@app.on_event("startup")
async def startup_event() -> None:
    log.info("Starting %s on port %d", settings.app_name, settings.port)
    # Eagerly validate secret access at startup — fail fast before traffic arrives
    try:
        from app.secrets import get_secret
        get_secret(settings.secret_arn, settings.aws_region)
        log.info("Secrets validated successfully")
    except RuntimeError as exc:
        log.error("Startup secret validation failed: %s", exc)
        # Do not raise — allow health check to pass, alert via logs
```

---

### Terraform IaC

```hcl
# terraform/main.tf
# ECS Fargate + ALB + Secrets Manager integration
# All sensitive values come from variables (tfvars, not hardcoded here).

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  # Remote state — replace with your backend config
  # backend "s3" {
  #   bucket = var.tf_state_bucket
  #   key    = "week08/terraform.tfstate"
  #   region = var.aws_region
  # }
}

provider "aws" {
  region = var.aws_region
}

# ── Data sources ─────────────────────────────────────────────────────────────
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# ── ECR repository ────────────────────────────────────────────────────────────
resource "aws_ecr_repository" "ai_api" {
  name                 = "${var.project_name}-ai-api"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  lifecycle {
    prevent_destroy = false
  }
}

resource "aws_ecr_lifecycle_policy" "ai_api" {
  repository = aws_ecr_repository.ai_api.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

# ── Secrets Manager ───────────────────────────────────────────────────────────
# The secret itself is created here but populated out-of-band (CI/CD or manual).
# Never store secret values in Terraform state.
resource "aws_secretsmanager_secret" "ai_api" {
  name                    = "${var.project_name}/ai-api/credentials"
  description             = "AI API runtime credentials"
  recovery_window_in_days = 7
}

# ── IAM: Task execution role (used by ECS control plane) ─────────────────────
resource "aws_iam_role" "task_execution" {
  name = "${var.project_name}-task-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "task_execution_managed" {
  role       = aws_iam_role.task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "task_execution_secrets" {
  name = "secrets-access"
  role = aws_iam_role.task_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = aws_secretsmanager_secret.ai_api.arn
    }]
  })
}

# ── IAM: Task role (used by the running container) ────────────────────────────
resource "aws_iam_role" "task_role" {
  name = "${var.project_name}-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "task_role_secrets" {
  name = "runtime-secrets-access"
  role = aws_iam_role.task_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = aws_secretsmanager_secret.ai_api.arn
    }]
  })
}

# ── ECS Cluster ───────────────────────────────────────────────────────────────
resource "aws_ecs_cluster" "main" {
  name = "${var.project_name}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_ecs_cluster_capacity_providers" "main" {
  cluster_name       = aws_ecs_cluster.main.name
  capacity_providers = ["FARGATE", "FARGATE_SPOT"]

  default_capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight            = 1
    base              = var.min_tasks  # guarantee this many on-demand
  }
}

# ── CloudWatch Log Group ──────────────────────────────────────────────────────
resource "aws_cloudwatch_log_group" "ai_api" {
  name              = "/ecs/${var.project_name}/ai-api"
  retention_in_days = 14
}

# ── ECS Task Definition ───────────────────────────────────────────────────────
resource "aws_ecs_task_definition" "ai_api" {
  family                   = "${var.project_name}-ai-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.task_execution.arn
  task_role_arn            = aws_iam_role.task_role.arn

  container_definitions = jsonencode([{
    name      = "ai-api"
    image     = "${aws_ecr_repository.ai_api.repository_url}:${var.image_tag}"
    essential = true

    portMappings = [{
      containerPort = 8080
      protocol      = "tcp"
    }]

    environment = [
      { name = "AWS_REGION", value = var.aws_region },
      { name = "SECRET_ARN",  value = aws_secretsmanager_secret.ai_api.arn },
      { name = "LOG_LEVEL",   value = "INFO" },
    ]

    # NOTE: secrets are fetched by the app at runtime via boto3, NOT injected
    # here as ECS secrets — this avoids secrets appearing in docker inspect output.

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.ai_api.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ai-api"
      }
    }

    healthCheck = {
      command     = ["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 15
    }
  }])
}

# ── ALB ───────────────────────────────────────────────────────────────────────
resource "aws_lb" "main" {
  name               = "${var.project_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = var.public_subnet_ids

  enable_deletion_protection = false
}

resource "aws_lb_target_group" "ai_api" {
  name        = "${var.project_name}-ai-api-tg"
  port        = 8080
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"  # required for Fargate awsvpc

  health_check {
    enabled             = true
    path                = "/health"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 30
    matcher             = "200"
  }

  deregistration_delay = 60  # reduced for faster scale-in / rolling deploy
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  # Redirect HTTP to HTTPS in production; simplified here for the exercise
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.ai_api.arn
  }
}

# ── Security Groups ───────────────────────────────────────────────────────────
resource "aws_security_group" "alb" {
  name        = "${var.project_name}-alb-sg"
  description = "ALB inbound HTTP/HTTPS"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "ecs_tasks" {
  name        = "${var.project_name}-ecs-tasks-sg"
  description = "ECS tasks — allow from ALB only"
  vpc_id      = var.vpc_id

  ingress {
    from_port       = 8080
    to_port         = 8080
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ── ECS Service ───────────────────────────────────────────────────────────────
resource "aws_ecs_service" "ai_api" {
  name            = "${var.project_name}-ai-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.ai_api.arn
  desired_count   = var.min_tasks

  capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight            = 1
    base              = var.min_tasks
  }

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = false  # tasks in private subnets; NAT gateway for egress
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.ai_api.arn
    container_name   = "ai-api"
    container_port   = 8080
  }

  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200

  lifecycle {
    ignore_changes = [desired_count]  # autoscaling manages this
  }

  depends_on = [aws_lb_listener.http]
}

# ── Autoscaling (from Day 5 module) ──────────────────────────────────────────
resource "aws_appautoscaling_target" "ecs_service" {
  max_capacity       = var.max_tasks
  min_capacity       = var.min_tasks
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.ai_api.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "requests_per_task" {
  name               = "${var.project_name}-rpt"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs_service.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_service.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs_service.service_namespace

  target_tracking_scaling_policy_configuration {
    target_value       = 80
    scale_in_cooldown  = 300
    scale_out_cooldown = 60

    predefined_metric_specification {
      predefined_metric_type = "ALBRequestCountPerTarget"
      resource_label = "${aws_lb.main.arn_suffix}/${aws_lb_target_group.ai_api.arn_suffix}"
    }
  }
}
```

```hcl
# terraform/variables.tf

variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "project_name" {
  type        = string
  description = "Short prefix for all resource names"
}

variable "image_tag" {
  type        = string
  default     = "latest"
  description = "ECR image tag to deploy"
}

variable "task_cpu" {
  type    = number
  default = 512  # 0.5 vCPU
}

variable "task_memory" {
  type    = number
  default = 1024  # 1 GB
}

variable "min_tasks" {
  type    = number
  default = 2
}

variable "max_tasks" {
  type    = number
  default = 20
}

variable "vpc_id" {
  type = string
}

variable "public_subnet_ids" {
  type = list(string)
}

variable "private_subnet_ids" {
  type = list(string)
}
```

```hcl
# terraform/outputs.tf

output "alb_dns_name" {
  value       = aws_lb.main.dns_name
  description = "ALB DNS — use this as your API base URL"
}

output "ecr_repository_url" {
  value       = aws_ecr_repository.ai_api.repository_url
  description = "ECR URL for docker push"
}

output "secret_arn" {
  value       = aws_secretsmanager_secret.ai_api.arn
  description = "Secrets Manager ARN — populate via CLI after apply"
  sensitive   = true
}

output "ecs_cluster_name" {
  value = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  value = aws_ecs_service.ai_api.name
}
```

---

### Deploy sequence (no secrets in any command)

```bash
# 1. Build and push image
ECR_URL=$(terraform -chdir=terraform output -raw ecr_repository_url)
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin "$ECR_URL"

docker build -t ai-api:latest .
docker tag ai-api:latest "$ECR_URL:latest"
docker push "$ECR_URL:latest"

# 2. Populate secret (value comes from stdin / password manager — NOT from env)
SECRET_ARN=$(terraform -chdir=terraform output -raw secret_arn)
# Paste the JSON interactively or pipe from a secrets vault CLI:
aws secretsmanager put-secret-value \
  --secret-id "$SECRET_ARN" \
  --secret-string '{"anthropic_api_key": "<paste from vault>"}'

# 3. Apply IaC
terraform -chdir=terraform apply -var="project_name=week08demo"

# 4. Verify health
ALB_DNS=$(terraform -chdir=terraform output -raw alb_dns_name)
curl -f "http://$ALB_DNS/health"
```

## Exercises

1. **Add HTTPS**: The current ALB listener forwards plain HTTP. Add an `aws_acm_certificate` resource and an HTTPS listener (port 443) that uses it. Update the HTTP listener to redirect `301` to HTTPS instead of forwarding.

2. **Secrets rotation**: Secrets Manager supports automatic rotation via Lambda. Sketch the IAM policy changes needed on the task role and the Lambda execution role. What application-level change would you make to handle mid-rotation key transitions without a task restart?

3. **Multi-region failover**: Identify which Terraform resources would need to be duplicated for a second region, and which can be shared. What DNS strategy (Route 53 health-check-based failover) would you use?

4. **VPC hardening**: The current ECS task security group allows all egress (`0.0.0.0/0`). Restrict egress to only: ECR VPC endpoint (port 443), Secrets Manager VPC endpoint (port 443), and the Anthropic API IP range (or use a NAT gateway with an egress-only SG). Write the updated `aws_security_group` resource.

5. **Blue/green deploy**: Replace the rolling `deployment_minimum_healthy_percent` approach with CodeDeploy blue/green for zero-downtime deploys. Add the `aws_codedeploy_app` and `aws_codedeploy_deployment_group` resources.

(Solutions: ../solutions/week-08/day-6.md)

## Self-check

- [ ] I can explain the difference between the ECS task execution role and the task role, and why each exists.
- [ ] I can describe the Fargate task startup sequence from scheduler placement to health check passing.
- [ ] I can build a multi-stage Dockerfile that produces a non-root, minimal runtime image.
- [ ] I can wire Secrets Manager into a FastAPI app using boto3 without storing secrets in environment variables at build time.
- [ ] I can write Terraform for ECS Fargate + ALB + autoscaling from scratch in under 60 minutes.
- [ ] I can explain why `ignore_changes = [desired_count]` is required on the ECS service when autoscaling is active.

---

## Supplementary Links

- [ECS Fargate launch type documentation](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/AWS_Fargate.html)
- [ECS task execution IAM role](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task_execution_IAM_role.html)
- [Secrets Manager integration with ECS](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/secrets-app-secrets-manager.html)
- [Terraform aws_ecs_service](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/ecs_service)
- [Terraform aws_lb + aws_lb_target_group](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/lb)
- [Multi-stage Docker builds](https://docs.docker.com/build/building/multi-stage/)
- [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
