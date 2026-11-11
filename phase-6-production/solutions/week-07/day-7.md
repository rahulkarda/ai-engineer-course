# Solutions — Week 07 · Day 7

## Exercise 1 — Partial terraform apply failure recovery

**Scenario:** `terraform apply` on prod fails mid-way. Some resources created, others not.

**Step-by-step recovery:**

1. **Do not panic and do not run `terraform apply` again immediately.** The state may be partially written; a second apply might duplicate resources.

2. **Check current state:**
   ```bash
   terraform state list
   ```
   Compare against what the plan intended to create.

3. **Run `terraform refresh`** to sync the state file with actual cloud reality:
   ```bash
   terraform refresh -var-file=prod.tfvars
   ```

4. **Re-run `terraform plan`** to see what still needs to happen:
   ```bash
   terraform plan -var-file=prod.tfvars
   ```
   The plan should now show only the resources that were NOT created.

5. **Run `terraform apply` again.** Terraform is idempotent — resources that already exist will be `~` (update/no-op); missing resources will be `+` (create).

6. **For resources that were half-created** (e.g., an ECS service with tasks in an error state): use `-target` to address only those resources:
   ```bash
   terraform apply -target=aws_ecs_service.ai_service -var-file=prod.tfvars
   ```

7. **Post-recovery:** Check CloudWatch alarms, run the smoke test, and verify the health endpoint.

**Prevention:** Enable the S3 backend with DynamoDB locking and versioning. The state file versions let you roll back the state itself if it becomes corrupted:
```bash
# List state file versions in S3
aws s3api list-object-versions --bucket my-terraform-state --prefix ai-service/terraform.tfstate
# Restore a previous version
aws s3api copy-object --copy-source my-terraform-state/ai-service/terraform.tfstate?versionId=XXXXXXX \
  --bucket my-terraform-state --key ai-service/terraform.tfstate
```

---

## Exercise 2 — Canary rollback at 20% with latency spike

**Exact commands:**

```bash
# 1. Immediately set canary weight to 0 (stop routing to new version)
terraform -chdir=infra apply -auto-approve \
  -var="environment=staging" \
  -var="image_tag=<NEW_TAG>" \
  -var="canary_weight=0"

# 2. Verify traffic is back to 100% on stable version
# Check ALB CloudWatch metrics for the stable (blue) target group
aws cloudwatch get-metric-statistics \
  --namespace AWS/ApplicationELB \
  --metric-name RequestCount \
  --dimensions Name=TargetGroup,Value=targetgroup/staging-ai-service-blue/xxxx \
  --start-time "$(date -u -v-5M +%Y-%m-%dT%H:%M:%SZ)" \
  --end-time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --period 60 \
  --statistics Sum

# 3. Confirm health endpoint returns 200
curl -f https://staging.example.com/health

# 4. File incident report: log the image_tag that failed, the latency metric,
#    and the timestamp. Investigate before re-promoting.
```

**What to watch during rollback:**
- ALB `TargetResponseTime` p99 — should drop back within 1–2 minutes.
- `HTTPCode_Target_5XX_Count` — should return to baseline.
- If latency does not improve after routing 0% to new version, the issue may be in shared infrastructure (RDS, external API) not the new image.

---

## Exercise 3 — Trivy UNKNOWN CVEs: options for production gating

**Root cause:** Trivy's vulnerability database (`trivy-db`) is stale (> 24 hours old). `UNKNOWN` severity means the scanner found a CVE entry but cannot determine severity because the DB is incomplete.

**Options:**

1. **Update the Trivy database before scanning (preferred):**
   ```yaml
   - name: Update Trivy DB
     run: trivy image --download-db-only
   - uses: aquasecurity/trivy-action@master
     with:
       skip-db-update: false   # default
   ```
   The `trivy-action` always downloads the latest DB if not cached. Check if the `cache` step is restoring a stale DB version.

2. **Ignore UNKNOWN severity** (acceptable for non-blocking scans, not for hard gates):
   ```yaml
   severity: CRITICAL,HIGH   # omit UNKNOWN
   ```
   This is acceptable if your policy is "only block on confirmed HIGH/CRITICAL."

3. **Set `ignore-unfixed: true`** — skip CVEs with no available fix regardless of severity. This is the pragmatic choice for production gates; you cannot remediate what has no fix.

4. **Use a fixed DB commit** for reproducible scans — advanced, rarely needed.

**Acceptable for production gating:** Option 2 (scope to `HIGH,CRITICAL`) combined with `ignore-unfixed: true`. This focuses the gate on actionable, confirmed high-risk CVEs. `UNKNOWN` severity items should be tracked in a separate non-blocking scan for periodic review.

---

## Exercise 4 — Slack notification on failure

```yaml
notify-slack:
  name: Notify Slack on failure
  runs-on: ubuntu-latest
  needs: [lint, test, build, scan, deploy-staging, deploy-prod]
  if: failure()   # runs when any upstream job fails

  steps:
    - name: Send Slack notification
      env:
        SLACK_WEBHOOK: ${{ secrets.SLACK_WEBHOOK }}
      run: |
        curl -X POST "$SLACK_WEBHOOK" \
          -H 'Content-type: application/json' \
          --data "$(cat <<EOF
        {
          "text": ":red_circle: Pipeline failed",
          "blocks": [
            {
              "type": "section",
              "text": {
                "type": "mrkdwn",
                "text": ":red_circle: *${{ github.workflow }}* failed on \`${{ github.ref_name }}\`\n<${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}|View run>"
              }
            }
          ]
        }
        EOF
        )"
```

**Note:** The job needs `if: failure()` not `if: always()`. With `always()`, it runs on success too. With `failure()`, it runs only when any `needs:` job has failed or been cancelled.

---

## Exercise 5 — Adding REDIS_URL end-to-end

**Files to change, in deployment order:**

**1. `infra/main.tf`** — Add ElastiCache Redis cluster resource:
```hcl
resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "${local.name_prefix}-redis"
  engine               = "redis"
  node_type            = "cache.t3.micro"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  port                 = 6379
  subnet_group_name    = aws_elasticache_subnet_group.main.name
  tags                 = local.common_tags
}

output "redis_endpoint" {
  value = "${aws_elasticache_cluster.redis.cache_nodes[0].address}:${aws_elasticache_cluster.redis.port}"
}
```

**2. `infra/modules/container-service/variables.tf`** — Add `redis_url` variable.

**3. `infra/modules/container-service/main.tf`** — Add to `container_definitions` environment:
```hcl
{ name = "REDIS_URL", value = var.redis_url }
```

**4. `src/config.py`** — Add field:
```python
redis_url: str = os.getenv("REDIS_URL", "")
```

**5. `.github/workflows/ci.yml`** — Add `REDIS_URL` to the deploy job's `TF_VAR_*` or pass it as a Terraform variable (value sourced from the Terraform output of the Redis cluster).

**Deployment order:**
1. Apply Terraform to create the Redis cluster first (a separate targeted apply or initial environment setup).
2. Note the output `redis_endpoint`.
3. Apply the ECS task definition change with the new env var pointing to the Redis endpoint.
4. ECS performs a rolling update of tasks — new tasks have `REDIS_URL`, old tasks don't.
5. Run smoke test to verify the service uses Redis correctly.

**Why order matters:** If you deploy the application code expecting `REDIS_URL` before the Redis cluster exists, the app will fail health checks and ECS will roll back automatically (if health check is configured). Always apply infrastructure changes before application changes.
