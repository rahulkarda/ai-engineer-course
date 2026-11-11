# Solutions — Week 07 Project

## Reference Implementation

This solution provides working code for all project deliverables. Study it after completing your own implementation.

---

## Application

### requirements.txt

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
pydantic==2.9.2
pydantic-settings==2.5.2
anthropic==0.34.2
```

### requirements-dev.txt

```
-r requirements.txt
pytest==8.3.3
pytest-cov==5.0.0
pytest-asyncio==0.24.0
httpx==0.27.2
ruff==0.6.9
mypy==1.11.2
bandit==1.7.10
pip-audit==2.7.3
```

### pyproject.toml

```toml
[tool.ruff]
line-length = 100
target-version = "py312"
select = ["E", "F", "I", "N", "UP", "S", "B"]
ignore = ["S101"]   # allow assert in tests

[tool.mypy]
python_version = "3.12"
strict = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

### Dockerfile

```dockerfile
# syntax=docker/dockerfile:1.6
FROM python:3.12-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install -r requirements.txt

FROM python:3.12-slim AS runtime

RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup --no-create-home appuser

WORKDIR /app
COPY --from=builder /install /usr/local
COPY --chown=appuser:appgroup src/ ./src/

USER appuser
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

---

## GitHub Actions Workflows

### .github/workflows/rollback.yml

```yaml
name: Rollback

on:
  workflow_dispatch:
    inputs:
      environment:
        description: "Environment to roll back"
        required: true
        type: choice
        options: [staging, production]
      image_tag:
        description: "Image tag to deploy (e.g. sha-abc1234)"
        required: true
        type: string

permissions:
  id-token: write
  contents: read

jobs:
  rollback:
    name: Rollback ${{ inputs.environment }}
    runs-on: ubuntu-latest
    environment: ${{ inputs.environment }}
    steps:
      - uses: actions/checkout@v4

      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: >-
            ${{ inputs.environment == 'production'
              && secrets.AWS_DEPLOY_ROLE_ARN_PROD
              || secrets.AWS_DEPLOY_ROLE_ARN }}
          aws-region: eu-west-1

      - name: Terraform rollback
        env:
          TF_VAR_db_password: >-
            ${{ inputs.environment == 'production'
              && secrets.PROD_DB_PASSWORD
              || secrets.STAGING_DB_PASSWORD }}
        run: |
          cd infra
          terraform init
          terraform apply -auto-approve \
            -var="environment=${{ inputs.environment }}" \
            -var="image_tag=${{ inputs.image_tag }}" \
            -var="canary_weight=0"

      - name: Verify health
        run: |
          BASE_URL="${{ inputs.environment == 'production' && 'https://api.example.com' || 'https://staging.example.com' }}"
          sleep 30
          curl -f "${BASE_URL}/health"
```

---

## Terraform

### infra/terraform.tf

```hcl
terraform {
  required_version = ">= 1.6.0"

  backend "s3" {
    bucket         = "my-terraform-state-bucket"
    key            = "ai-service/terraform.tfstate"
    region         = "eu-west-1"
    encrypt        = true
    dynamodb_table = "terraform-state-locks"
  }

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
```

### infra/outputs.tf

```hcl
output "alb_dns_name" {
  description = "ALB DNS name for the service"
  value       = aws_lb.main.dns_name
}

output "ecr_repository_url" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.ai_service.repository_url
}

output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.main.endpoint
  sensitive   = true
}

output "s3_artifacts_bucket" {
  description = "S3 bucket for model artifacts"
  value       = aws_s3_bucket.artifacts.bucket
}
```

### infra/staging.tfvars

```hcl
environment       = "staging"
aws_region        = "eu-west-1"
db_instance_class = "db.t3.micro"
canary_weight     = 0
```

### infra/prod.tfvars

```hcl
environment       = "prod"
aws_region        = "eu-west-1"
db_instance_class = "db.t3.small"
canary_weight     = 0
```

---

## Deploy Runbook (reference content)

```markdown
# AI Service Deploy Runbook

## Prerequisites
- terraform >= 1.6 (`terraform version`)
- aws-cli >= 2 configured with profile that can assume the deploy role
- gh cli authenticated (`gh auth status`)
- docker running locally

## Normal Deploy

1. Merge your PR to `main`.
2. Watch the pipeline: `gh run list --branch main --limit 5`.
3. When `deploy-staging` passes, observe staging for 5 minutes:
   - `curl https://staging.example.com/health`
   - Check CloudWatch dashboard for error rates.
4. Approve the `deploy-prod` job in the GitHub Actions UI (Settings → Environments → production → pending deployments).
5. After prod deploy completes, verify:
   - `curl https://api.example.com/health`
   - CloudWatch: `HTTPCode_Target_5XX_Count` < 5 per minute for 10 minutes.

## Canary Deploy

```bash
export TF_VAR_db_password="$(aws secretsmanager get-secret-value \
  --secret-id staging-ai-service/db-password --query SecretString --output text)"

python scripts/canary_promote.py \
  --workspace staging \
  --image-tag sha-$(git rev-parse HEAD) \
  --steps 5,20,50,100 \
  --bake-time 300
```

Monitor: CloudWatch → ai-service dashboard → error rate, p99 latency.

## Rollback

### Via GitHub Actions (preferred):
1. Go to Actions → Rollback → Run workflow.
2. Select environment and the last known-good image tag (from the previous pipeline run).
3. Approve if environment requires it.

### Via CLI (emergency):
```bash
cd infra
terraform apply -auto-approve \
  -var="environment=prod" \
  -var="image_tag=sha-<LAST_GOOD_SHA>" \
  -var="canary_weight=0"
```

### Via AWS console (last resort):
1. ECS → Clusters → prod-ai-service → Services → prod-ai-service.
2. Update → Force new deployment.
3. Change task definition revision to previous version.

## Post-Deploy Checks
- `/health` returns `{"status": "ok"}`
- CloudWatch: `HTTPCode_Target_5XX_Count` < 5/min
- CloudWatch: `TargetResponseTime` p99 < 500 ms
- CloudWatch: `UnHealthyHostCount` = 0
```

---

## Smoke Test Addition (stretch goal)

```yaml
# Add between deploy-staging and deploy-prod in ci.yml:
smoke-test:
  name: Staging Smoke Test
  runs-on: ubuntu-latest
  needs: [deploy-staging]
  environment: staging
  steps:
    - name: Health check (with retry)
      run: |
        for i in $(seq 1 10); do
          STATUS=$(curl -s -o /dev/null -w "%{http_code}" https://staging.example.com/health)
          [ "$STATUS" = "200" ] && echo "OK" && exit 0
          echo "Attempt $i: HTTP $STATUS — waiting 15s"
          sleep 15
        done
        echo "Smoke test FAILED" && exit 1

deploy-prod:
  needs: [smoke-test]   # changed from [deploy-staging]
  # ...
```

## Slack Notification (stretch goal)

```yaml
notify-failure:
  runs-on: ubuntu-latest
  needs: [lint, test, audit, build, scan, deploy-staging, deploy-prod]
  if: failure()
  steps:
    - name: Slack alert
      env:
        SLACK_WEBHOOK: ${{ secrets.SLACK_WEBHOOK }}
        RUN_URL: ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}
      run: |
        curl -X POST "$SLACK_WEBHOOK" \
          -H 'Content-type: application/json' \
          --data "{\"text\":\":red_circle: *${{ github.workflow }}* failed on \`${{ github.ref_name }}\` — <${RUN_URL}|View run>\"}"
```
