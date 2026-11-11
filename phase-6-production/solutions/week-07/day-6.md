# Solutions — Week 07 · Day 6

## Exercise 1 — Configuring environment protection rules

In GitHub:
1. Navigate to repo **Settings → Environments**.
2. Click **New environment** → name it `production`.
3. Under **Deployment protection rules**:
   - Enable **Required reviewers** → add team members or a team (e.g., `@org/platform-team`).
   - Set **Wait timer** (optional) — e.g., 30 minutes after staging passes before prod approval can be given.
   - Enable **Prevent self-review** if the deploying engineer should not be able to approve their own deployment.
4. Under **Environment secrets** — add `PROD_DB_PASSWORD`, `AWS_DEPLOY_ROLE_ARN_PROD` here (they are only available to jobs with `environment: production`).

When a job with `environment: production` runs, the workflow pauses at that job, sends a review notification to required reviewers, and resumes only when approved (or times out and fails).

---

## Exercise 2 — Shared cluster, separate services

```hcl
# infra/main.tf

# Single shared cluster
resource "aws_ecs_cluster" "shared" {
  name = "ai-service-shared"
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
  tags = { ManagedBy = "terraform" }
}

# Call module for staging — pass cluster_arn
module "ai_service_staging" {
  source      = "./modules/container-service"
  name        = "staging-ai-service"
  environment = "staging"
  image_url   = "${aws_ecr_repository.ai_service.repository_url}:${var.image_tag}"
  cluster_arn = aws_ecs_cluster.shared.arn   # new variable
  subnet_ids  = aws_subnet.private[*].id
  vpc_id      = aws_vpc.main.id
  desired_count = 1
}

# Call module for prod — same cluster, separate service
module "ai_service_prod" {
  source      = "./modules/container-service"
  name        = "prod-ai-service"
  environment = "prod"
  image_url   = "${aws_ecr_repository.ai_service.repository_url}:${var.image_tag}"
  cluster_arn = aws_ecs_cluster.shared.arn
  subnet_ids  = aws_subnet.private[*].id
  vpc_id      = aws_vpc.main.id
  desired_count = 3
}
```

Module change: remove `aws_ecs_cluster.this` resource from the module, replace with a variable:

```hcl
# modules/container-service/variables.tf — add:
variable "cluster_arn" {
  type        = string
  description = "ARN of the ECS cluster to deploy into"
}

# modules/container-service/main.tf — replace cluster reference:
resource "aws_ecs_service" "this" {
  cluster         = var.cluster_arn   # use passed-in ARN
  # ... rest unchanged
}
```

**Tradeoff:** A shared cluster means a misbehaving prod workload can starve staging CPU/memory. For strict isolation (security or cost allocation), keep separate clusters. Shared clusters reduce idle capacity cost.

---

## Exercise 3 — Smoke test job

```yaml
smoke-test:
  name: Staging Smoke Test
  runs-on: ubuntu-latest
  needs: [deploy-staging]
  environment: staging

  steps:
    - name: Wait for service to be healthy
      run: |
        STAGING_URL="https://staging.example.com"
        MAX_RETRIES=10
        RETRY_INTERVAL=15

        for i in $(seq 1 $MAX_RETRIES); do
          echo "Attempt $i/$MAX_RETRIES — checking ${STAGING_URL}/health"
          STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${STAGING_URL}/health" || echo "000")

          if [ "$STATUS" = "200" ]; then
            echo "Health check passed (HTTP ${STATUS})"
            exit 0
          fi

          echo "Health check returned HTTP ${STATUS} — retrying in ${RETRY_INTERVAL}s"
          sleep $RETRY_INTERVAL
        done

        echo "Smoke test FAILED — /health did not return 200 after ${MAX_RETRIES} attempts"
        exit 1
```

Add `needs: [smoke-test]` to the `deploy-prod` job:
```yaml
deploy-prod:
  needs: [smoke-test]
  environment: production
  # ...
```

---

## Exercise 4 — CloudWatch-based canary bake

See Day 5 Solution Exercise 4 for the full `check_metrics` implementation with `boto3`.

Key points for the canary promote script:

```python
def check_metrics_with_cloudwatch(
    target_group_arn: str,
    alb_arn: str,
    bake_seconds: int,
) -> bool:
    """Real implementation replacing time.sleep."""
    # Initial bake period — wait for metrics to populate
    time.sleep(min(bake_seconds, 60))  # at least 60s, or bake_time if shorter

    # Then poll every 30s for remaining bake period
    deadline = time.time() + bake_seconds
    check_interval = 30

    while time.time() < deadline:
        if not _cloudwatch_check(target_group_arn, alb_arn):
            return False
        remaining = deadline - time.time()
        logger.info("Metrics healthy — %.0f seconds remaining in bake window", remaining)
        time.sleep(min(check_interval, max(0, remaining)))

    return True
```

The key replacement: instead of sleeping the full bake time and checking once, poll at 30-second intervals so you catch a degradation early and roll back faster.

---

## Exercise 5 — rollback.yml workflow

```yaml
# .github/workflows/rollback.yml
name: Rollback Deploy

on:
  workflow_dispatch:
    inputs:
      environment:
        description: "Environment to roll back"
        required: true
        type: choice
        options: [staging, production]
      image_tag:
        description: "Image tag to roll back to (e.g. sha-abc123)"
        required: true
        type: string

permissions:
  id-token: write
  contents: read

jobs:
  rollback:
    name: Rollback ${{ inputs.environment }} to ${{ inputs.image_tag }}
    runs-on: ubuntu-latest
    environment: ${{ inputs.environment }}

    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials (OIDC)
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ inputs.environment == 'production' && secrets.AWS_DEPLOY_ROLE_ARN_PROD || secrets.AWS_DEPLOY_ROLE_ARN }}
          aws-region: eu-west-1

      - name: Rollback via Terraform
        env:
          TF_VAR_db_password: ${{ inputs.environment == 'production' && secrets.PROD_DB_PASSWORD || secrets.STAGING_DB_PASSWORD }}
        run: |
          echo "Rolling back ${{ inputs.environment }} to image tag ${{ inputs.image_tag }}"
          cd infra
          terraform init
          terraform apply -auto-approve \
            -var="environment=${{ inputs.environment }}" \
            -var="image_tag=${{ inputs.image_tag }}" \
            -var="canary_weight=0"

      - name: Verify rollback health
        run: |
          ENV="${{ inputs.environment }}"
          if [ "$ENV" = "production" ]; then
            URL="https://api.example.com"
          else
            URL="https://staging.example.com"
          fi
          sleep 30
          curl -f "${URL}/health" || (echo "Health check failed after rollback" && exit 1)
```
