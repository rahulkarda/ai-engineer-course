# Day 6 Solution — Full Deploy Walkthrough

## Exercise 1: Complete Dockerfile for Python FastAPI AI App

```dockerfile
# syntax=docker/dockerfile:1.7
# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps for common AI/ML packages (psycopg2-binary, httpx, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install into /build/venv — keeps final image lean
RUN python -m venv /build/venv \
    && /build/venv/bin/pip install --upgrade pip \
    && /build/venv/bin/pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Security: non-root user
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

# Minimal runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual env from builder
COPY --from=builder /build/venv /app/venv

# Copy application source
COPY --chown=appuser:appgroup src/ ./src/

# Activate venv for all subsequent commands
ENV PATH="/app/venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Prevents model downloads to /root (not writable as appuser)
    TRANSFORMERS_CACHE=/tmp/hf_cache \
    HF_HOME=/tmp/hf_cache

USER appuser

EXPOSE 8000

# Health check — ALB uses /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use exec form — forwards signals correctly (SIGTERM for graceful shutdown)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--timeout-keep-alive", "30"]
```

**Key decisions explained:**

| Decision | Why |
|----------|-----|
| Multi-stage build | Final image excludes build tools — reduces size from ~1.5 GB to ~400 MB and attack surface |
| Non-root user (uid 1001) | ECS blocks containers that run as root by default when `readonlyRootFilesystem = true` |
| `PYTHONDONTWRITEBYTECODE=1` | No `.pyc` files in image layers |
| `PYTHONUNBUFFERED=1` | CloudWatch Logs receives log lines in real time, not buffered |
| `exec` CMD form | PID 1 is uvicorn, so SIGTERM from ECS reaches the process directly for graceful drain |
| `/health` endpoint | ALB target group uses HTTP health checks; separate from `/` to avoid auth middleware |
| `start-period=60s` | AI apps often need time to load model weights before they can serve traffic |
| `workers=1` | In ECS, scale horizontally by adding tasks, not workers per container — simpler OOM management |

---

## Exercise 2: Terraform — ECS Fargate Deployment with ALB

### `main.tf`

```hcl
terraform {
  required_version = ">= 1.7"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    # Configure with -backend-config or environment variables
    # bucket = var.tf_state_bucket  # not allowed in backend block — use -backend-config
    key    = "week08/ai-api/terraform.tfstate"
    region = "us-east-1"
    encrypt = true
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.app_name
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# ── Data sources ──────────────────────────────────────────────────────────────
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# ── VPC ───────────────────────────────────────────────────────────────────────
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${var.app_name}-vpc"
  cidr = var.vpc_cidr

  azs             = slice(data.aws_availability_zones.available.names, 0, 3)
  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs

  enable_nat_gateway   = true
  single_nat_gateway   = var.environment != "production"
  enable_dns_hostnames = true
  enable_dns_support   = true
}

# Free S3 Gateway endpoint — eliminates NAT charges for S3 traffic
resource "aws_vpc_endpoint" "s3" {
  vpc_id            = module.vpc.vpc_id
  service_name      = "com.amazonaws.${var.aws_region}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = module.vpc.private_route_table_ids
}

# ── ECR Repository ────────────────────────────────────────────────────────────
resource "aws_ecr_repository" "app" {
  name                 = var.app_name
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "KMS"
  }
}

resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name

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

# ── IAM ───────────────────────────────────────────────────────────────────────
resource "aws_iam_role" "ecs_task_execution" {
  name = "${var.app_name}-task-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution_managed" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Allow task execution role to read secrets (for injection at task start)
resource "aws_iam_role_policy" "ecs_task_execution_secrets" {
  name = "${var.app_name}-secrets-read"
  role = aws_iam_role.ecs_task_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "secretsmanager:GetSecretValue",
        "kms:Decrypt"
      ]
      Resource = [
        "arn:aws:secretsmanager:${var.aws_region}:${data.aws_caller_identity.current.account_id}:secret:${var.app_name}/*"
      ]
    }]
  })
}

# Task role — what the application code can do at runtime
resource "aws_iam_role" "ecs_task" {
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

resource "aws_iam_role_policy" "ecs_task_s3" {
  name = "${var.app_name}-s3-access"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
      Resource = [
        "arn:aws:s3:::${var.app_name}-*",
        "arn:aws:s3:::${var.app_name}-*/*"
      ]
    }]
  })
}

# ── CloudWatch Logs ───────────────────────────────────────────────────────────
resource "aws_cloudwatch_log_group" "app" {
  name              = "/ecs/${var.app_name}"
  retention_in_days = 30
}

# ── ECS Cluster ───────────────────────────────────────────────────────────────
resource "aws_ecs_cluster" "main" {
  name = var.app_name

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_ecs_cluster_capacity_providers" "main" {
  cluster_name       = aws_ecs_cluster.main.name
  capacity_providers = ["FARGATE", "FARGATE_SPOT"]

  default_capacity_provider_strategy {
    base              = 1
    weight            = 1
    capacity_provider = "FARGATE"
  }
}

# ── ECS Task Definition ───────────────────────────────────────────────────────
resource "aws_ecs_task_definition" "app" {
  family                   = var.app_name
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name      = var.app_name
    image     = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
    essential = true

    portMappings = [{
      containerPort = 8000
      protocol      = "tcp"
    }]

    # Secrets injected as environment variables from Secrets Manager
    # The execution role reads them at task start — never in code or logs
    secrets = [
      {
        name      = "ANTHROPIC_API_KEY"
        valueFrom = "${aws_secretsmanager_secret.anthropic_key.arn}:ANTHROPIC_API_KEY::"
      },
      {
        name      = "DATABASE_URL"
        valueFrom = "${aws_secretsmanager_secret.db_credentials.arn}:DATABASE_URL::"
      }
    ]

    environment = [
      { name = "APP_ENV",       value = var.environment },
      { name = "LOG_LEVEL",     value = var.log_level },
      { name = "AWS_REGION",    value = var.aws_region }
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

    readonlyRootFilesystem = false  # Set true if app writes no local files
    stopTimeout            = 30     # Seconds ECS waits for graceful shutdown
  }])
}

# ── Security Groups ───────────────────────────────────────────────────────────
resource "aws_security_group" "alb" {
  name        = "${var.app_name}-alb-sg"
  description = "ALB inbound from internet"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

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

resource "aws_security_group" "ecs" {
  name        = "${var.app_name}-ecs-sg"
  description = "ECS tasks — only accept traffic from ALB"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 8000
    to_port         = 8000
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

# ── ALB ───────────────────────────────────────────────────────────────────────
resource "aws_lb" "main" {
  name               = "${var.app_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = module.vpc.public_subnets

  enable_deletion_protection = var.environment == "production"
}

resource "aws_lb_target_group" "app" {
  name        = "${var.app_name}-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = module.vpc.vpc_id
  target_type = "ip"  # required for Fargate awsvpc network mode

  health_check {
    enabled             = true
    path                = "/health"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 30
    matcher             = "200"
  }

  deregistration_delay = 30  # Match stopTimeout in task def
}

resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.main.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.acm_certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}

resource "aws_lb_listener" "http_redirect" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = "redirect"
    redirect {
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }
}

# ── ECS Service ───────────────────────────────────────────────────────────────
resource "aws_ecs_service" "app" {
  name            = var.app_name
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = var.min_tasks

  network_configuration {
    subnets          = module.vpc.private_subnets
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = var.app_name
    container_port   = 8000
  }

  deployment_configuration {
    maximum_percent         = 200
    minimum_healthy_percent = 100  # Zero-downtime rolling deploy
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  depends_on = [
    aws_lb_listener.https,
    aws_iam_role_policy_attachment.ecs_task_execution_managed
  ]

  lifecycle {
    ignore_changes = [desired_count]  # Managed by autoscaling
  }
}

# ── Autoscaling ───────────────────────────────────────────────────────────────
resource "aws_appautoscaling_target" "ecs" {
  max_capacity       = var.max_tasks
  min_capacity       = var.min_tasks
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.app.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
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
    target_value       = 60.0
    scale_in_cooldown  = 300
    scale_out_cooldown = 60
  }
}
```

### `variables.tf`

```hcl
variable "app_name" {
  description = "Application name — used as prefix for all resources"
  type        = string
  default     = "ai-api"
}

variable "environment" {
  description = "Deployment environment (development, staging, production)"
  type        = string
  validation {
    condition     = contains(["development", "staging", "production"], var.environment)
    error_message = "Must be development, staging, or production."
  }
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "private_subnet_cidrs" {
  description = "Private subnet CIDRs (ECS tasks)"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnet_cidrs" {
  description = "Public subnet CIDRs (ALB)"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

variable "task_cpu" {
  description = "Fargate task CPU units (1024 = 1 vCPU)"
  type        = number
  default     = 1024
}

variable "task_memory" {
  description = "Fargate task memory in MB"
  type        = number
  default     = 2048
}

variable "image_tag" {
  description = "Docker image tag to deploy"
  type        = string
  default     = "latest"
}

variable "min_tasks" {
  description = "Minimum ECS task count"
  type        = number
  default     = 2
}

variable "max_tasks" {
  description = "Maximum ECS task count"
  type        = number
  default     = 20
}

variable "log_level" {
  description = "Application log level"
  type        = string
  default     = "INFO"
}

variable "acm_certificate_arn" {
  description = "ARN of ACM TLS certificate for ALB HTTPS listener"
  type        = string
}
```

### `outputs.tf`

```hcl
output "alb_dns_name" {
  description = "ALB DNS name — point your CNAME here"
  value       = aws_lb.main.dns_name
}

output "alb_zone_id" {
  description = "ALB hosted zone ID for Route 53 alias record"
  value       = aws_lb.main.zone_id
}

output "ecr_repository_url" {
  description = "ECR repository URL for docker push"
  value       = aws_ecr_repository.app.repository_url
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "ECS service name"
  value       = aws_ecs_service.app.name
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group for application logs"
  value       = aws_cloudwatch_log_group.app.name
}
```

---

## Exercise 3: Python Secret Fetching Module with Caching

```python
# src/secrets_manager.py
"""
AWS Secrets Manager client with in-process caching.

Fetches secrets once per cache_ttl_seconds, not on every request.
This avoids Secrets Manager API costs and latency on hot paths.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

_DEFAULT_TTL = int(os.getenv("SECRETS_CACHE_TTL_SECONDS", "300"))  # 5 minutes


@dataclass
class _CacheEntry:
    value: dict[str, Any]
    fetched_at: float = field(default_factory=time.monotonic)


class SecretsManagerClient:
    """
    Thin wrapper around boto3 Secrets Manager with TTL-based in-memory cache.

    Usage:
        client = SecretsManagerClient()
        value = client.get_secret_value("my-app/anthropic-key", key="ANTHROPIC_API_KEY")
    """

    def __init__(
        self,
        region: str | None = None,
        cache_ttl_seconds: int = _DEFAULT_TTL,
    ) -> None:
        self._region = region or os.getenv("AWS_REGION", "us-east-1")
        self._ttl = cache_ttl_seconds
        self._cache: dict[str, _CacheEntry] = {}
        self._client = boto3.client("secretsmanager", region_name=self._region)
        logger.info(
            "SecretsManagerClient initialized",
            extra={"region": self._region, "cache_ttl": self._ttl},
        )

    def _is_expired(self, entry: _CacheEntry) -> bool:
        return (time.monotonic() - entry.fetched_at) > self._ttl

    def _fetch_raw(self, secret_name: str) -> dict[str, Any]:
        """Fetch from AWS — raises on error, never returns partial data."""
        logger.debug("Fetching secret from AWS", extra={"secret_name": secret_name})
        try:
            response = self._client.get_secret_value(SecretId=secret_name)
        except ClientError as exc:
            error_code = exc.response["Error"]["Code"]
            logger.error(
                "Failed to fetch secret",
                extra={"secret_name": secret_name, "error_code": error_code},
            )
            raise

        secret_string = response.get("SecretString", "{}")
        try:
            return json.loads(secret_string)
        except json.JSONDecodeError:
            # Plain-string secret (not JSON) — wrap in dict
            return {"value": secret_string}

    def get_secret(self, secret_name: str) -> dict[str, Any]:
        """
        Return the full secret dict, using cache when fresh.

        Args:
            secret_name: The Secrets Manager secret name or ARN.

        Returns:
            Parsed JSON dict of the secret.

        Raises:
            botocore.exceptions.ClientError: On AWS API failure.
        """
        entry = self._cache.get(secret_name)
        if entry is not None and not self._is_expired(entry):
            logger.debug("Cache hit", extra={"secret_name": secret_name})
            return entry.value

        logger.debug("Cache miss — fetching", extra={"secret_name": secret_name})
        value = self._fetch_raw(secret_name)
        self._cache[secret_name] = _CacheEntry(value=value)
        return value

    def get_secret_value(self, secret_name: str, key: str) -> str:
        """
        Convenience: fetch one key from a JSON secret.

        Args:
            secret_name: Secrets Manager secret name or ARN.
            key: JSON key within the secret.

        Returns:
            String value for the key.

        Raises:
            KeyError: If key is not in the secret JSON.
        """
        secret = self.get_secret(secret_name)
        if key not in secret:
            raise KeyError(f"Key '{key}' not found in secret '{secret_name}'")
        return str(secret[key])

    def invalidate(self, secret_name: str) -> None:
        """Force next access to re-fetch from AWS (e.g. after rotation)."""
        self._cache.pop(secret_name, None)
        logger.info("Cache invalidated", extra={"secret_name": secret_name})


@lru_cache(maxsize=1)
def get_secrets_client() -> SecretsManagerClient:
    """Module-level singleton — one client per process lifetime."""
    return SecretsManagerClient()
```

**Usage in FastAPI app:**

```python
# src/config.py
import os
import logging
from functools import lru_cache

from pydantic_settings import BaseSettings
from src.secrets_manager import get_secrets_client

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    app_env: str = "development"
    aws_region: str = "us-east-1"
    secret_name_prefix: str = "ai-api"
    log_level: str = "INFO"

    # Populated at startup from Secrets Manager (or env in local dev)
    anthropic_api_key: str = ""
    database_url: str = ""

    class Config:
        env_file = ".env"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()

    # In non-local environments, override with Secrets Manager values
    if settings.app_env != "development":
        client = get_secrets_client()
        try:
            settings.anthropic_api_key = client.get_secret_value(
                f"{settings.secret_name_prefix}/keys",
                "ANTHROPIC_API_KEY",
            )
            settings.database_url = client.get_secret_value(
                f"{settings.secret_name_prefix}/db",
                "DATABASE_URL",
            )
            logger.info("Loaded secrets from Secrets Manager")
        except Exception:
            logger.exception("Failed to load secrets — falling back to env vars")
            # Falls through to env-var values set in task definition
    else:
        # Local dev: read from environment / .env file
        settings.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        settings.database_url = os.getenv("DATABASE_URL", "")

    return settings
```

---

## Exercise 4: ECS Task Definition with Secrets Manager References

This is the Terraform resource for the task definition. The critical point is that secrets are referenced by ARN — the ECS agent fetches them at task startup, so the values are never in Terraform state or application logs.

```hcl
# secrets.tf — create the secrets first (values populated separately)
resource "aws_secretsmanager_secret" "anthropic_key" {
  name        = "${var.app_name}/keys"
  description = "API keys for external AI services"
  # KMS key for encryption at rest (use aws_kms_key resource for custom key)
}

resource "aws_secretsmanager_secret" "db_credentials" {
  name        = "${var.app_name}/db"
  description = "Database connection credentials"
}

# Populate via CLI (not Terraform — keeps secrets out of state):
# aws secretsmanager put-secret-value \
#   --secret-id ai-api/keys \
#   --secret-string '{"ANTHROPIC_API_KEY": "..."}'

# ── Task definition with secret references ─────────────────────────────────
resource "aws_ecs_task_definition" "app_with_secrets" {
  family                   = "${var.app_name}-v2"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name      = var.app_name
    image     = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
    essential = true

    portMappings = [{ containerPort = 8000, protocol = "tcp" }]

    # ── SECRET INJECTION (correct pattern) ──────────────────────────────────
    # valueFrom accepts:
    #   Full ARN                           → whole secret string as env var value
    #   ARN:json-key::                     → just that JSON key's value
    #   ARN:json-key:version-stage:        → specific rotation version
    secrets = [
      {
        # JSON secret: {"ANTHROPIC_API_KEY": "..."}
        # The :ANTHROPIC_API_KEY:: suffix extracts just that key
        name      = "ANTHROPIC_API_KEY"
        valueFrom = "${aws_secretsmanager_secret.anthropic_key.arn}:ANTHROPIC_API_KEY::"
      },
      {
        name      = "DATABASE_URL"
        valueFrom = "${aws_secretsmanager_secret.db_credentials.arn}:DATABASE_URL::"
      },
      {
        # Redis URL — can also be a plain string secret
        name      = "REDIS_URL"
        valueFrom = aws_secretsmanager_secret.redis_url.arn
      }
    ]

    # ── NON-SECRET CONFIG ────────────────────────────────────────────────────
    # Only non-sensitive values go here — visible in ECS console
    environment = [
      { name = "APP_ENV",    value = var.environment },
      { name = "LOG_LEVEL",  value = var.log_level },
      { name = "AWS_REGION", value = var.aws_region },
      { name = "PORT",       value = "8000" }
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = "/ecs/${var.app_name}"
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

    ulimits = [{
      name      = "nofile"
      softLimit = 65536
      hardLimit = 65536
    }]
  }])

  tags = {
    Name = "${var.app_name}-task-def"
  }
}
```

**Why secrets go in `secrets` not `environment`:**

| `environment` | `secrets` |
|--------------|-----------|
| Plaintext in task definition | Encrypted in Secrets Manager |
| Visible in ECS console & CloudTrail | Fetched by ECS agent at runtime only |
| Stays in Terraform state (S3) | Never in Terraform state |
| Never rotatable without redeploy | Rotatable; new tasks pick up new value |
| Visible in `describe-tasks` API | Not in describe-tasks output |

The execution role's `secretsmanager:GetSecretValue` permission is what makes this work — without it, the task fails to start with `CannotPullContainerError`.
