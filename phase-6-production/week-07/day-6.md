# Phase 6 · Week 07 · Day 6 — Saturday Deep Build: Full Pipeline + Terraform Module

> Objective: Build a complete GitHub Actions CI/CD pipeline and a Terraform module for an AI service from scratch.  ·  Time: ~4 hr  ·  Prereqs: [[day-1]]–[[day-5]]

## Overview

Today you assemble everything from the week into a working system:

1. A FastAPI AI service with health endpoint.
2. GitHub Actions workflow: lint → test → build → scan → deploy (staging and prod with manual gate).
3. A Terraform module: ECS Fargate + RDS + ALB, environment-parameterised.
4. A deploy runbook.

Work in a real Git repository. All YAML and HCL runs against real tooling — no stubs.

---

## Part 1 — AI Service Application (45 min)

### Repository structure

```
ai-service/
├── src/
│   ├── __init__.py
│   ├── main.py          # FastAPI app
│   ├── config.py        # pydantic-settings config
│   └── models.py        # Pydantic request/response models
├── tests/
│   ├── unit/
│   │   └── test_main.py
│   └── conftest.py
├── infra/
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   ├── terraform.tf
│   └── modules/
│       └── container-service/
│           ├── main.tf
│           ├── variables.tf
│           └── outputs.tf
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── deploy.yml
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
└── runbook.md
```

### FastAPI service

```python
# src/config.py
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    environment: str = os.getenv("ENVIRONMENT", "dev")
    log_level: str = "INFO"
    db_url: str = os.getenv("DATABASE_URL", "")
    model_id: str = "claude-opus-5-5"

    class Config:
        env_file = ".env"


settings = Settings()
```

```python
# src/models.py
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    environment: str
    version: str


class InferRequest(BaseModel):
    prompt: str
    max_tokens: int = 1024


class InferResponse(BaseModel):
    content: str
    model: str
    input_tokens: int
    output_tokens: int
```

```python
# src/main.py
import logging
import os
from importlib.metadata import version as pkg_version

from fastapi import FastAPI, HTTPException

from .config import settings
from .models import HealthResponse, InferRequest, InferResponse

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Service", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        environment=settings.environment,
        version="0.1.0",
    )


@app.post("/infer", response_model=InferResponse)
async def infer(request: InferRequest) -> InferResponse:
    """Call the LLM and return the response."""
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")

    client = anthropic.Anthropic(api_key=api_key)
    logger.info("Inference request — model=%s tokens=%d", settings.model_id, request.max_tokens)

    message = client.messages.create(
        model=settings.model_id,
        max_tokens=request.max_tokens,
        messages=[{"role": "user", "content": request.prompt}],
    )

    return InferResponse(
        content=message.content[0].text,
        model=message.model,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
    )
```

```python
# tests/unit/test_main.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from src.main import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "environment" in data
    assert "version" in data


def test_infer_without_api_key_returns_500() -> None:
    with patch.dict("os.environ", {}, clear=False):
        import os
        os.environ.pop("ANTHROPIC_API_KEY", None)
        response = client.post("/infer", json={"prompt": "Hello"})
    assert response.status_code == 500


@patch("anthropic.Anthropic")
def test_infer_returns_response(mock_anthropic_cls: MagicMock) -> None:
    import os
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="Hello world")]
    mock_message.model = "claude-opus-5-5"
    mock_message.usage.input_tokens = 10
    mock_message.usage.output_tokens = 5
    mock_client.messages.create.return_value = mock_message

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        response = client.post("/infer", json={"prompt": "Hello"})

    assert response.status_code == 200
    data = response.json()
    assert data["content"] == "Hello world"
    assert data["model"] == "claude-opus-5-5"
```

---

## Part 2 — GitHub Actions Pipeline (60 min)

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, "feature/**"]
  pull_request:
    branches: [main]

permissions:
  contents: read
  packages: write
  id-token: write
  security-events: write

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}/ai-service

jobs:
  lint:
    name: Lint & Type Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements*.txt', 'pyproject.toml') }}
          restore-keys: ${{ runner.os }}-pip-
      - run: pip install -r requirements-dev.txt
      - run: ruff check src/ tests/
      - run: mypy src/
      - run: bandit -r src/ -ll -q

  test:
    name: Test (Python ${{ matrix.python-version }})
    runs-on: ubuntu-latest
    needs: [lint]
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
      fail-fast: false
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-${{ matrix.python-version }}-pip-${{ hashFiles('requirements*.txt') }}
      - run: pip install -r requirements-dev.txt
      - name: Run tests
        run: |
          pytest tests/unit \
            --cov=src \
            --cov-fail-under=80 \
            --cov-report=xml \
            --junitxml=test-results.xml \
            -q
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: test-results-py${{ matrix.python-version }}
          path: "*.xml"

  audit:
    name: Dependency Audit
    runs-on: ubuntu-latest
    needs: [lint]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install pip-audit
      - run: pip-audit -r requirements.txt --strict

  build:
    name: Build & Push Image
    runs-on: ubuntu-latest
    needs: [test, audit]
    if: github.event_name == 'push'
    outputs:
      image-digest: ${{ steps.push.outputs.digest }}
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/metadata-action@v5
        id: meta
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=sha,format=long
            type=ref,event=branch
      - uses: docker/build-push-action@v5
        id: push
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          sbom: true
          provenance: true

  scan:
    name: Vulnerability Scan
    runs-on: ubuntu-latest
    needs: [build]
    steps:
      - uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:sha-${{ github.sha }}
          format: sarif
          output: trivy-results.sarif
          severity: HIGH,CRITICAL
          exit-code: "1"
          ignore-unfixed: true
      - uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: trivy-results.sarif

  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: [scan]
    environment: staging
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - name: Configure AWS credentials (OIDC)
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_DEPLOY_ROLE_ARN }}
          aws-region: eu-west-1
      - name: Terraform apply (staging)
        env:
          TF_VAR_db_password: ${{ secrets.STAGING_DB_PASSWORD }}
        run: |
          cd infra
          terraform init
          terraform apply -auto-approve \
            -var="environment=staging" \
            -var="image_tag=sha-${{ github.sha }}"

  deploy-prod:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: [deploy-staging]
    environment: production   # requires manual approval in GitHub settings
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - name: Configure AWS credentials (OIDC)
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_DEPLOY_ROLE_ARN_PROD }}
          aws-region: eu-west-1
      - name: Terraform apply (prod)
        env:
          TF_VAR_db_password: ${{ secrets.PROD_DB_PASSWORD }}
        run: |
          cd infra
          terraform init
          terraform apply -auto-approve \
            -var="environment=prod" \
            -var="image_tag=sha-${{ github.sha }}"
```

---

## Part 3 — Terraform Module (60 min)

Build a reusable `container-service` module that is called from the root for both staging and prod.

```hcl
# infra/modules/container-service/variables.tf
variable "name"              { type = string }
variable "environment"       { type = string }
variable "image_url"         { type = string }
variable "app_port"          { type = number; default = 8080 }
variable "cpu"               { type = number; default = 512 }
variable "memory"            { type = number; default = 1024 }
variable "desired_count"     { type = number; default = 1 }
variable "subnet_ids"        { type = list(string) }
variable "alb_listener_arn"  { type = string }
variable "vpc_id"            { type = string }
variable "canary_weight"     { type = number; default = 0 }
variable "tags"              { type = map(string); default = {} }
```

```hcl
# infra/modules/container-service/main.tf
resource "aws_ecs_task_definition" "this" {
  family                   = var.name
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = tostring(var.cpu)
  memory                   = tostring(var.memory)

  container_definitions = jsonencode([{
    name         = var.name
    image        = var.image_url
    essential    = true
    portMappings = [{ containerPort = var.app_port, protocol = "tcp" }]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = "/ecs/${var.name}"
        "awslogs-region"        = data.aws_region.current.name
        "awslogs-stream-prefix" = "ecs"
      }
    }
  }])

  tags = var.tags
}

data "aws_region" "current" {}

resource "aws_cloudwatch_log_group" "this" {
  name              = "/ecs/${var.name}"
  retention_in_days = 30
  tags              = var.tags
}

resource "aws_security_group" "service" {
  name   = "${var.name}-sg"
  vpc_id = var.vpc_id

  ingress {
    from_port   = var.app_port
    to_port     = var.app_port
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/8"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = var.tags
}

resource "aws_lb_target_group" "this" {
  name        = "${var.name}-tg"
  port        = var.app_port
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"

  health_check {
    path                = "/health"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    interval            = 30
  }

  tags = var.tags
}

resource "aws_ecs_service" "this" {
  name            = var.name
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.this.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets         = var.subnet_ids
    security_groups = [aws_security_group.service.id]
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.this.arn
    container_name   = var.name
    container_port   = var.app_port
  }

  tags = var.tags
}

resource "aws_ecs_cluster" "this" {
  name = var.name
  tags = var.tags
}
```

```hcl
# infra/modules/container-service/outputs.tf
output "service_name"     { value = aws_ecs_service.this.name }
output "target_group_arn" { value = aws_lb_target_group.this.arn }
output "cluster_name"     { value = aws_ecs_cluster.this.name }
```

---

## Part 4 — Deploy Runbook (30 min)

Write `runbook.md` (this file belongs in your repo — it is project documentation, not generated for the learner):

The runbook should cover these sections:
1. **Prerequisites** — tools required (terraform, aws cli, gh cli, docker)
2. **Normal deploy** — step-by-step for a standard release (push to main → observe pipeline → approve prod gate)
3. **Canary deploy** — how to run `scripts/canary_promote.py` and what to watch
4. **Rollback procedure** — set image tag back, `terraform apply`
5. **Emergency rollback** — manual ECS task definition rollback via AWS console
6. **Post-deploy checks** — health endpoint, CloudWatch metrics, error rate thresholds

---

## Exercises

1. The `deploy-prod` job uses `environment: production` to gate on manual approval. How do you configure the required reviewers and approval timeout in GitHub? Where in the GitHub UI?

2. The Terraform module creates an ECS cluster per environment. Refactor so staging and prod share a cluster (pass `cluster_arn` as a variable) but keep separate services and task definitions.

3. Add a `smoke-test` job between `deploy-staging` and `deploy-prod` that calls `curl -f https://staging.example.com/health` and fails the pipeline if the health check does not return 200.

4. The canary promotion script uses `time.sleep`. Production-grade: what observability data should replace the sleep, and how would you query it with CloudWatch SDK (`boto3.client("cloudwatch")`)?

5. Write a GitHub Actions workflow (`rollback.yml`) that is triggered only via `workflow_dispatch` with inputs `environment` (staging/prod) and `image_tag`, and runs `terraform apply` with those values.

(Solutions: ../solutions/week-07/day-6.md)

## Self-check

- [ ] I have a working FastAPI service with health endpoint and unit tests
- [ ] I have a GitHub Actions workflow that covers lint → test → build → scan → deploy
- [ ] I have a Terraform module that accepts environment and image_tag variables
- [ ] I can explain the difference between the `ci.yml` and a `rollback.yml` workflow
- [ ] I have a written runbook covering normal deploy, canary, and rollback
