# Week 8 Project Solution — Production AI API on AWS

## Architecture Overview

```
                         ┌─────────────────────────────────────────────────────┐
                         │                    AWS Account                       │
                         │                                                       │
  Internet               │  ┌─────────────────────────────────────────────────┐ │
  Clients                │  │                  VPC 10.0.0.0/16                 │ │
     │                   │  │                                                   │ │
     │  HTTPS 443        │  │  Public Subnets  ┌──────────────────────────┐    │ │
     └──────────────────────▶  10.0.101-103.0  │   Application Load        │    │ │
                         │  │                  │   Balancer (ALB)          │    │ │
                         │  │                  │   HTTPS → Port 8000       │    │ │
                         │  │                  └────────────┬─────────────┘    │ │
                         │  │                               │                   │ │
                         │  │  Private Subnets ┌────────────▼─────────────┐    │ │
                         │  │  10.0.1-3.0      │   ECS Fargate Tasks       │    │ │
                         │  │                  │   (2–20 tasks)            │    │ │
                         │  │                  │   FastAPI + Anthropic     │    │ │
                         │  │                  │   claude-sonnet-5-5       │    │ │
                         │  │                  └──┬──────────┬────────┬───┘    │ │
                         │  │                     │          │        │         │ │
                         │  │              ┌──────▼──┐  ┌────▼────┐   │         │ │
                         │  │              │   RDS   │  │  Redis  │   │         │ │
                         │  │              │Postgres │  │Elasticache   │         │ │
                         │  │              │ Multi-AZ│  │         │   │         │ │
                         │  │              └─────────┘  └─────────┘   │         │ │
                         │  │                                          │ VPC     │ │
                         │  │  ┌───────────────────────────────────────▼──────┐  │ │
                         │  │  │  AWS Services (via VPC Endpoints)            │  │ │
                         │  │  │  S3 (Gateway) · ECR · Secrets Manager       │  │ │
                         │  │  │  CloudWatch · X-Ray · STS                   │  │ │
                         │  │  └──────────────────────────────────────────────┘  │ │
                         │  └─────────────────────────────────────────────────┘ │
                         │                                                       │
                         │  ┌──────────────────┐  ┌──────────────────────────┐  │
                         │  │ Secrets Manager  │  │  ECR (Container Images) │  │
                         │  │ - ANTHROPIC_KEY  │  │  Immutable tags          │  │
                         │  │ - DB_PASS        │  │  Scan on push            │  │
                         │  │ - REDIS_TOKEN    │  └──────────────────────────┘  │
                         │  └──────────────────┘                                │
                         └─────────────────────────────────────────────────────┘

  External:  api.anthropic.com  (HTTPS via NAT Gateway)
```

---

## Part 1: Dockerfile

```dockerfile
# syntax=docker/dockerfile:1.7
# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m venv /build/venv \
    && /build/venv/bin/pip install --upgrade pip \
    && /build/venv/bin/pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/venv /app/venv
COPY --chown=appuser:appgroup src/ ./src/

ENV PATH="/app/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--timeout-keep-alive", "30"]
```

**`requirements.txt`:**
```
fastapi==0.115.0
uvicorn[standard]==0.30.6
anthropic==0.40.0
asyncpg==0.30.0
redis[asyncio]==5.2.0
pydantic==2.10.0
pydantic-settings==2.7.0
boto3==1.35.0
```

---

## Part 2: Application Code

```python
# src/main.py
"""FastAPI AI API — Week 8 project reference implementation."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import anthropic
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from src.config import get_settings, Settings
from src.models import ChatRequest, ChatResponse, HealthResponse
from src.database import init_db, close_db
from src.cache import init_cache, close_cache

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)

_client: anthropic.AsyncAnthropic | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _client
    settings = get_settings()

    logger.info("Initializing application", extra={"env": settings.app_env})
    await init_db(settings.database_url)
    await init_cache(settings.redis_url)

    _client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    logger.info("Application ready")

    yield

    logger.info("Shutting down")
    await close_db()
    await close_cache()


app = FastAPI(
    title="AI API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None if os.getenv("APP_ENV") == "production" else "/docs",
)


def get_client() -> anthropic.AsyncAnthropic:
    if _client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI client not initialized",
        )
    return _client


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """ALB health check endpoint — no auth required."""
    return HealthResponse(status="ok")


@app.post("/v1/chat", response_model=ChatResponse, tags=["inference"])
async def chat(
    request: ChatRequest,
    client: anthropic.AsyncAnthropic = Depends(get_client),
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    """Single-turn chat completion via Claude."""
    logger.info("Chat request", extra={"model": settings.model_id, "turns": 1})

    try:
        message = await client.messages.create(
            model=settings.model_id,
            max_tokens=request.max_tokens,
            messages=[{"role": "user", "content": request.message}],
        )
    except anthropic.APIError as exc:
        logger.error("Anthropic API error", extra={"status": exc.status_code})
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Upstream AI service error",
        ) from exc

    return ChatResponse(
        content=message.content[0].text,
        model=message.model,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
    )
```

```python
# src/config.py
"""Application configuration — loaded from Secrets Manager in production."""
from __future__ import annotations

import logging
import os
from functools import lru_cache

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    app_env: str = "development"
    aws_region: str = "us-east-1"
    log_level: str = "INFO"
    model_id: str = "claude-sonnet-5-5"

    # Populated at startup — from env in dev, from Secrets Manager in prod
    anthropic_api_key: str = ""
    database_url: str = ""
    redis_url: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()

    if settings.app_env != "development":
        _load_from_secrets_manager(settings)

    return settings


def _load_from_secrets_manager(settings: Settings) -> None:
    """Populate sensitive fields from Secrets Manager at startup."""
    try:
        import boto3
        import json

        client = boto3.client("secretsmanager", region_name=settings.aws_region)

        def fetch(secret_name: str) -> dict:
            resp = client.get_secret_value(SecretId=secret_name)
            return json.loads(resp["SecretString"])

        app_name = os.getenv("APP_NAME", "ai-api")
        keys = fetch(f"{app_name}/keys")
        db = fetch(f"{app_name}/db")
        cache = fetch(f"{app_name}/cache")

        settings.anthropic_api_key = keys["ANTHROPIC_API_KEY"]
        settings.database_url = db["DATABASE_URL"]
        settings.redis_url = cache["REDIS_URL"]

        logger.info("Loaded configuration from Secrets Manager")
    except Exception:
        logger.exception("Failed to load from Secrets Manager — falling back to env vars")
```

```python
# src/models.py
"""Pydantic request/response models."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8192)
    max_tokens: int = Field(default=1024, ge=1, le=4096)


class ChatResponse(BaseModel):
    content: str
    model: str
    input_tokens: int
    output_tokens: int


class HealthResponse(BaseModel):
    status: str
```

---

## Part 3: Terraform IaC

### `main.tf`

```hcl
terraform {
  required_version = ">= 1.7"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }

  backend "s3" {
    key     = "week08/project/terraform.tfstate"
    region  = "us-east-1"
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
      Week        = "08"
    }
  }
}

data "aws_availability_zones" "available" { state = "available" }
data "aws_caller_identity" "current" {}

# ── VPC ───────────────────────────────────────────────────────────────────────
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${var.app_name}-vpc"
  cidr = "10.0.0.0/16"

  azs             = slice(data.aws_availability_zones.available.names, 0, 3)
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

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

# Interface endpoint for Secrets Manager (removes NAT dependency for secrets)
resource "aws_vpc_endpoint" "secretsmanager" {
  vpc_id              = module.vpc.vpc_id
  service_name        = "com.amazonaws.${var.aws_region}.secretsmanager"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = module.vpc.private_subnets
  security_group_ids  = [aws_security_group.vpc_endpoints.id]
  private_dns_enabled = true
}

resource "aws_security_group" "vpc_endpoints" {
  name        = "${var.app_name}-vpce-sg"
  description = "VPC Interface Endpoints"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 443
    to_port         = 443
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs.id]
  }
}

# ── ECR ───────────────────────────────────────────────────────────────────────
resource "aws_ecr_repository" "app" {
  name                 = var.app_name
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration { scan_on_push = true }
  encryption_configuration { encryption_type = "KMS" }
}

resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 10 images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 10 }
      action       = { type = "expire" }
    }]
  })
}

# ── RDS PostgreSQL ────────────────────────────────────────────────────────────
resource "aws_db_subnet_group" "main" {
  name       = "${var.app_name}-db-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_security_group" "rds" {
  name        = "${var.app_name}-rds-sg"
  description = "RDS — accept only from ECS"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs.id]
  }
}

resource "random_password" "db" {
  length  = 32
  special = false
}

resource "aws_secretsmanager_secret" "db" {
  name        = "${var.app_name}/db"
  description = "RDS PostgreSQL credentials"
}

resource "aws_secretsmanager_secret_version" "db" {
  secret_id = aws_secretsmanager_secret.db.id
  secret_string = jsonencode({
    DATABASE_URL = "postgresql://${var.db_username}:${random_password.db.result}@${aws_db_instance.main.endpoint}/${var.db_name}"
    DB_PASSWORD  = random_password.db.result
  })
}

resource "aws_db_instance" "main" {
  identifier        = var.app_name
  engine            = "postgres"
  engine_version    = "16.4"
  instance_class    = var.db_instance_class
  allocated_storage = 20
  storage_encrypted = true

  db_name  = var.db_name
  username = var.db_username
  password = random_password.db.result

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]

  multi_az               = var.environment == "production"
  skip_final_snapshot    = var.environment != "production"
  deletion_protection    = var.environment == "production"
  backup_retention_period = 7

  performance_insights_enabled = true
}

# ── ElastiCache Redis ─────────────────────────────────────────────────────────
resource "aws_elasticache_subnet_group" "main" {
  name       = "${var.app_name}-cache-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_security_group" "redis" {
  name        = "${var.app_name}-redis-sg"
  description = "Redis — accept only from ECS"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs.id]
  }
}

resource "random_password" "redis" {
  length  = 32
  special = false
}

resource "aws_secretsmanager_secret" "redis" {
  name        = "${var.app_name}/cache"
  description = "Redis auth token and connection URL"
}

resource "aws_elasticache_replication_group" "main" {
  replication_group_id = var.app_name
  description          = "${var.app_name} Redis cache"

  engine               = "redis"
  engine_version       = "7.1"
  node_type            = var.redis_node_type
  num_cache_clusters   = var.environment == "production" ? 2 : 1

  subnet_group_name  = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.redis.id]

  auth_token                 = random_password.redis.result
  transit_encryption_enabled = true
  at_rest_encryption_enabled = true
}

resource "aws_secretsmanager_secret_version" "redis" {
  secret_id = aws_secretsmanager_secret.redis.id
  secret_string = jsonencode({
    REDIS_URL = "rediss://:${random_password.redis.result}@${aws_elasticache_replication_group.main.primary_endpoint_address}:6379"
  })
}

# ── Secrets Manager: API Keys ─────────────────────────────────────────────────
resource "aws_secretsmanager_secret" "api_keys" {
  name        = "${var.app_name}/keys"
  description = "External AI service API keys — populate via CLI, not Terraform"
}

# Populate separately:
# aws secretsmanager put-secret-value \
#   --secret-id ai-api/keys \
#   --secret-string '{"ANTHROPIC_API_KEY":"..."}'

# ── IAM ───────────────────────────────────────────────────────────────────────
resource "aws_iam_role" "execution" {
  name = "${var.app_name}-execution"
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
  name = "secrets-read"
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

resource "aws_iam_role" "task" {
  name = "${var.app_name}-task"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "task_s3" {
  name = "s3-access"
  role = aws_iam_role.task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
      Resource = ["arn:aws:s3:::${var.app_name}-*", "arn:aws:s3:::${var.app_name}-*/*"]
    }]
  })
}

# ── CloudWatch Logs ───────────────────────────────────────────────────────────
resource "aws_cloudwatch_log_group" "app" {
  name              = "/ecs/${var.app_name}"
  retention_in_days = 30
}

# ── ECS ───────────────────────────────────────────────────────────────────────
resource "aws_ecs_cluster" "main" {
  name = var.app_name
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

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
    image     = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
    essential = true

    portMappings = [{ containerPort = 8000, protocol = "tcp" }]

    secrets = [
      {
        name      = "ANTHROPIC_API_KEY"
        valueFrom = "${aws_secretsmanager_secret.api_keys.arn}:ANTHROPIC_API_KEY::"
      },
      {
        name      = "DATABASE_URL"
        valueFrom = "${aws_secretsmanager_secret.db.arn}:DATABASE_URL::"
      },
      {
        name      = "REDIS_URL"
        valueFrom = "${aws_secretsmanager_secret.redis.arn}:REDIS_URL::"
      }
    ]

    environment = [
      { name = "APP_ENV",    value = var.environment },
      { name = "LOG_LEVEL",  value = var.log_level },
      { name = "AWS_REGION", value = var.aws_region },
      { name = "APP_NAME",   value = var.app_name }
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

# ── Security Groups ───────────────────────────────────────────────────────────
resource "aws_security_group" "alb" {
  name        = "${var.app_name}-alb"
  description = "ALB public ingress"
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
  name        = "${var.app_name}-ecs"
  description = "ECS tasks — ALB only"
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
  target_type = "ip"

  health_check {
    enabled             = true
    path                = "/health"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 30
    matcher             = "200"
  }

  deregistration_delay = 30
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

resource "aws_lb_listener" "http" {
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
    minimum_healthy_percent = 100
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  depends_on = [
    aws_lb_listener.https,
    aws_iam_role_policy_attachment.execution_managed,
  ]

  lifecycle { ignore_changes = [desired_count] }
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
  name               = "${var.app_name}-cpu"
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
variable "app_name"           { type = string; default = "ai-api" }
variable "environment"        { type = string; default = "production" }
variable "aws_region"         { type = string; default = "us-east-1" }
variable "image_tag"          { type = string; default = "latest" }
variable "task_cpu"           { type = number; default = 1024 }
variable "task_memory"        { type = number; default = 2048 }
variable "min_tasks"          { type = number; default = 2 }
variable "max_tasks"          { type = number; default = 20 }
variable "log_level"          { type = string; default = "INFO" }
variable "db_instance_class"  { type = string; default = "db.t4g.small" }
variable "db_name"            { type = string; default = "aiapi" }
variable "db_username"        { type = string; default = "appuser" }
variable "redis_node_type"    { type = string; default = "cache.t4g.small" }
variable "acm_certificate_arn" { type = string }
```

### `outputs.tf`

```hcl
output "alb_dns_name"      { value = aws_lb.main.dns_name }
output "alb_zone_id"       { value = aws_lb.main.zone_id }
output "ecr_repo_url"      { value = aws_ecr_repository.app.repository_url }
output "cluster_name"      { value = aws_ecs_cluster.main.name }
output "service_name"      { value = aws_ecs_service.app.name }
output "db_endpoint"       { value = aws_db_instance.main.endpoint; sensitive = true }
output "redis_endpoint"    { value = aws_elasticache_replication_group.main.primary_endpoint_address; sensitive = true }
output "log_group"         { value = aws_cloudwatch_log_group.app.name }
output "api_keys_secret"   { value = aws_secretsmanager_secret.api_keys.arn }
```

---

## Part 4: Cost Estimate Table

Pricing based on us-east-1, November 2025. Production configuration (3 AZs).

| Component | Specification | Monthly Cost |
|-----------|--------------|-------------|
| **ECS Fargate (compute)** | 2 tasks avg × 1 vCPU × 720h | $58.29 |
| **ECS Fargate (memory)** | 2 tasks avg × 2 GB × 720h | $12.80 |
| **ALB** | 1 ALB + 1M req/day LCUs | $18.00 |
| **NAT Gateway (fixed)** | 1 NAT × 720h | $32.40 |
| **NAT Gateway (data)** | 50 GB/month outbound | $2.25 |
| **RDS PostgreSQL** | db.t4g.small, Multi-AZ, 20 GB | $58.40 |
| **ElastiCache Redis** | cache.t4g.small, 2 nodes | $37.14 |
| **Secrets Manager** | 3 secrets × $0.40 + API calls | $1.20 |
| **ECR** | 1 GB storage | $0.10 |
| **CloudWatch Logs** | 10 GB ingestion + 10 GB storage | $7.50 |
| **Data transfer out** | 60 GB/month | $5.40 |
| **KMS** | CMK for ECR + Secrets | $2.00 |
| **VPC Interface Endpoints** | Secrets Manager, 2 AZ | $14.40 |
| **Total** | | **~$249.88/month** |

**Notes:**
- S3 Gateway endpoint is free — no S3 traffic costs
- Secrets Manager Interface endpoint ($14.40) eliminates ~$30/month NAT charges for secrets
- At 10× traffic (autoscaling to 15 tasks avg): +$437 compute = ~$687/month total
- Dev/staging: remove Multi-AZ RDS, single Redis node, single NAT → ~$140/month

**Cost optimization opportunities:**
1. Add Fargate Spot for non-minimum tasks → saves ~40% on compute above min baseline
2. Reserved RDS instance (1-year, no upfront) → saves ~30% on RDS
3. ElastiCache Reserved Nodes (1-year) → saves ~30% on Redis
4. At full optimization: production cost drops to ~$180/month

---

## Part 5: Architecture Decision Record

### ADR-003: PostgreSQL + Redis for Persistence and Caching

**Status:** Accepted  
**Date:** 2025-01-12  
**Context:** The AI API needs to store conversation history, rate-limit counters, and cached Anthropic responses.

**Decision:** Use RDS PostgreSQL for durable storage and ElastiCache Redis for caching/rate-limiting.

**Alternatives considered:**

| Option | Reason rejected |
|--------|----------------|
| DynamoDB | Strong consistency guarantees for relational conversation data require careful key design; team has SQL expertise, not DynamoDB |
| Aurora Serverless | Scale-to-zero causes cold starts; cluster min cost > RDS t4g.small for steady traffic |
| Memcached | No persistence, no pub/sub, no Lua scripting; Redis is strictly superior for our use case |

**Consequences:**
- (+) Standard SQL queries; easy to migrate data
- (+) Redis handles rate limiting with atomic INCR + EXPIRE patterns
- (-) RDS Multi-AZ doubles the DB cost; acceptable for production
- (-) Two additional services to monitor and manage

**Revisit:** If we move to multi-region, evaluate Aurora Global Database to replace RDS.

---

### ADR-004: Single-Region Deployment for Initial Launch

**Status:** Accepted  
**Date:** 2025-01-12  
**Context:** Should we deploy to multiple AWS regions from day one?

**Decision:** Single region (us-east-1) for initial launch. Revisit at 10K daily active users.

**Rationale:**
- Multi-region adds significant complexity: Route 53 latency routing, Aurora Global, ElastiCache Global Datastore, secrets replication
- Anthropic's API is globally available; our latency is dominated by LLM inference time (500ms+), not our infrastructure
- 99.9% SLA is achievable in a single region with Multi-AZ RDS and 2+ ECS tasks
- Cost of multi-region setup would double infrastructure costs without user-visible benefit at current scale

**Consequences:**
- (+) Faster time to production; lower monthly cost
- (-) Single-region failure affects all users; recovery time is ~15 minutes for RDS failover
- Revisit when: latency-sensitive customers in EU/APAC, or availability SLA > 99.9%

---

## Deployment Guide

### Initial setup

```bash
# 1. Authenticate to AWS
export AWS_PROFILE=production

# 2. Create ECR repository and push image
cd /path/to/project
IMAGE_TAG=$(git rev-parse --short HEAD)
AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${AWS_ACCOUNT}.dkr.ecr.us-east-1.amazonaws.com/ai-api"

aws ecr get-login-password | docker login --username AWS --password-stdin "${AWS_ACCOUNT}.dkr.ecr.us-east-1.amazonaws.com"
docker build -t ai-api:${IMAGE_TAG} .
docker tag ai-api:${IMAGE_TAG} ${ECR_URI}:${IMAGE_TAG}
docker push ${ECR_URI}:${IMAGE_TAG}

# 3. Populate secrets (before Terraform apply)
aws secretsmanager create-secret \
  --name ai-api/keys \
  --description "External API keys"

# Store actual key value — never hardcode in scripts
aws secretsmanager put-secret-value \
  --secret-id ai-api/keys \
  --secret-string "{\"ANTHROPIC_API_KEY\": \"$(read -sp 'API Key: ' k; echo $k)\"}"

# 4. Terraform init and apply
cd terraform/
terraform init -backend-config=backend.hcl
terraform plan -var="image_tag=${IMAGE_TAG}" -out=tfplan
terraform apply tfplan
```

### Rolling deploy

```bash
# Build and push new image
IMAGE_TAG=$(git rev-parse --short HEAD)
docker build -t ${ECR_URI}:${IMAGE_TAG} .
docker push ${ECR_URI}:${IMAGE_TAG}

# Update task definition
terraform apply -var="image_tag=${IMAGE_TAG}" -auto-approve

# ECS automatically performs rolling deploy with circuit breaker protection
# Monitor:
aws ecs describe-services \
  --cluster ai-api \
  --services ai-api \
  --query 'services[0].deployments'
```
