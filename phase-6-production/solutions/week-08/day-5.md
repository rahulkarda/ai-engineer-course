# Day 5 Solution — Autoscaling & Cost Awareness

## Exercise 1: Terraform ECS Autoscaling Policy (Target Tracking on CPU)

```hcl
# autoscaling.tf

resource "aws_appautoscaling_target" "ecs_target" {
  max_capacity       = 20
  min_capacity       = 2
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.ai_api.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"

  depends_on = [aws_ecs_service.ai_api]
}

# --- Target Tracking: CPU Utilization ---
resource "aws_appautoscaling_policy" "cpu_target_tracking" {
  name               = "${var.app_name}-cpu-target-tracking"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs_target.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs_target.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }

    # Scale out when CPU > 60%; scale in when CPU < 60% for the cooldown period
    target_value       = 60.0
    scale_in_cooldown  = 300  # seconds — wait 5 min before scaling in to avoid flapping
    scale_out_cooldown = 60   # seconds — scale out fast to absorb traffic spikes
  }
}

# --- Target Tracking: Memory Utilization (secondary guard) ---
resource "aws_appautoscaling_policy" "memory_target_tracking" {
  name               = "${var.app_name}-memory-target-tracking"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs_target.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs_target.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageMemoryUtilization"
    }

    target_value       = 75.0
    scale_in_cooldown  = 300
    scale_out_cooldown = 60
  }
}
```

**Why this is correct:**

- `TargetTrackingScaling` is the simplest and most responsive policy type — AWS automatically calculates scale-out and scale-in actions to keep the metric at the target value. You do not write separate scale-out / scale-in alarms.
- `target_value = 60.0` for CPU is a standard production value. Going higher (e.g. 80%) risks latency spikes while new tasks start up; going lower wastes money.
- `scale_out_cooldown = 60` is aggressive on purpose for an AI API where inference can spike suddenly. `scale_in_cooldown = 300` prevents oscillation.
- Two policies (CPU + memory) are independent; AWS honors whichever triggers first — correct for AI workloads where memory pressure from model weights can occur before CPU saturates.
- `min_capacity = 2` ensures high availability across AZs even at zero load.

---

## Exercise 2: Cost Calculation — 1M Requests/Day to ECS Fargate (1 vCPU / 2 GB)

### Given
| Parameter | Value |
|-----------|-------|
| Requests/day | 1,000,000 |
| Avg request duration | 500 ms |
| Task spec | 1 vCPU, 2 GB RAM |
| Region | us-east-1 |
| Fargate vCPU-hour price | $0.04048 |
| Fargate GB-hour price | $0.004445 |
| ALB LCU price | $0.008/LCU-hour |
| Data transfer out | $0.09/GB |

### Step 1 — Determine task count needed

```
Requests per second = 1,000,000 / 86,400 ≈ 11.6 RPS (average)
Assume peak = 3× average = ~35 RPS
Assume 1 task handles ~20 RPS at <60% CPU target
Tasks needed at peak = ceil(35 / 20) = 2 tasks
With min=2, average running tasks ≈ 2–3; use 2.5 for estimate
```

### Step 2 — Fargate compute cost

```
Task-hours/day = 2.5 tasks × 24 hours = 60 task-hours/day
Monthly task-hours = 60 × 30 = 1,800

vCPU cost = 1,800 task-hours × 1 vCPU × $0.04048 = $72.86/month
Memory cost = 1,800 task-hours × 2 GB × $0.004445 = $16.00/month
Fargate total = $88.86/month
```

### Step 3 — ALB cost

```
1M requests/day × 30 days = 30M requests/month
Rule evaluations ≈ 1 per request = 30M
LCU-hours: 30M / (1,000 × 720 hours) ≈ 41.7 LCU-hours/month
ALB fixed = $0.008 × 720 = $5.76/month
ALB LCU = 41.7 × $0.008 × 720 = ~$0.23 (negligible)
ALB total ≈ $18/month (fixed charge + LCU)
```

### Step 4 — Data transfer

```
Assume avg response = 2 KB
1M req/day × 30 × 2 KB = 60 GB/month outbound
Transfer cost = 60 × $0.09 = $5.40/month
```

### Step 5 — Secrets Manager (if using per-request fetch — wrong pattern)

```
Correct pattern = fetch once at startup + cache in memory
Cost = ~$0.40/month per secret (10,000 API calls threshold negligible with caching)
```

### Total Estimate

| Component | Monthly Cost |
|-----------|-------------|
| Fargate compute | $88.86 |
| ALB | ~$18.00 |
| Data transfer out | $5.40 |
| Secrets Manager | $0.40 |
| ECR storage (1 GB image) | $0.10 |
| CloudWatch Logs (10 GB) | $5.00 |
| **Total** | **~$117.76/month** |

**Why this calculation matters:**
- Fargate pricing is per-second, billed in 1-minute minimum. Short-lived tasks (batch jobs) waste money; long-running services are efficient.
- NAT Gateway adds ~$0.045/GB data processed — if the service calls S3 or DynamoDB heavily through NAT, that can dwarf compute costs (see Exercise 4).
- At 10× traffic (10M req/day) with autoscaling to 25 tasks, cost scales ~linearly to ~$1,100/month — useful for capacity planning.

---

## Exercise 3: Spot Instance Strategy for AI Batch Processing

### Strategy Design

**Use case:** Nightly batch embedding generation, weekly fine-tuning jobs, async inference queues.

```
Architecture:
  SQS Queue → ECS Service (Spot) → S3 (results)
                    ↓ (if spot interrupted)
              ECS Service (On-Demand fallback, 1 task min)
```

**Key decisions:**

1. **Capacity provider mix**

```hcl
resource "aws_ecs_cluster_capacity_providers" "batch" {
  cluster_name = aws_ecs_cluster.batch.name

  capacity_providers = ["FARGATE_SPOT", "FARGATE"]

  default_capacity_provider_strategy {
    base              = 1          # 1 on-demand task always running
    weight            = 1
    capacity_provider = "FARGATE"
  }

  default_capacity_provider_strategy {
    base              = 0
    weight            = 4          # 80% spot, 20% on-demand
    capacity_provider = "FARGATE_SPOT"
  }
}
```

2. **Interruption handling**

```python
# In the worker process — handle SIGTERM from 2-minute spot warning
import signal
import logging

logger = logging.getLogger(__name__)

def _handle_sigterm(signum: int, frame) -> None:
    """Gracefully checkpoint on Spot interruption notice."""
    logger.warning("SIGTERM received — checkpointing current batch item to SQS")
    # Re-enqueue the in-flight item with visibility timeout = 0
    # so another worker picks it up immediately
    _checkpoint_current_item()
    raise SystemExit(0)

signal.signal(signal.SIGTERM, _handle_sigterm)
```

3. **Instance type diversification** (EC2-backed ECS, not Fargate):

```hcl
# For GPU batch jobs, use a launch template with multiple instance types
resource "aws_launch_template" "gpu_spot" {
  name = "gpu-spot-template"

  instance_market_options {
    market_type = "spot"
    spot_options {
      interruption_behavior = "terminate"
      max_price             = "0.50"  # cap at 50% of on-demand
    }
  }
}

# Diversify across instance families to maximize spot availability
variable "gpu_instance_types" {
  default = ["g4dn.xlarge", "g4dn.2xlarge", "g5.xlarge", "p3.2xlarge"]
}
```

4. **Idempotent jobs** — critical for spot:
   - Every batch item has a unique ID stored in SQS message
   - Results written to S3 with key = `{job_id}/{item_id}.json`
   - S3 `PutObject` is idempotent — safe to retry
   - Use SQS visibility timeout = 2× expected item processing time

**Savings:** Fargate Spot is ~70% cheaper than Fargate on-demand. For a 4-hour nightly batch job with 10 tasks: saves ~$8.40/run vs ~$28/run on-demand.

---

## Exercise 4: NAT Gateway vs VPC Endpoints for S3 — Cost Difference

### NAT Gateway (current default for private subnets)

```
NAT Gateway hourly: $0.045/hour × 720 = $32.40/month
Data processed: $0.045/GB

If ECS tasks pull 100 GB/month from S3 (model weights, datasets):
  NAT data cost = 100 GB × $0.045 = $4.50/month
  Total = $32.40 + $4.50 = $36.90/month just for S3 access
```

### VPC Gateway Endpoint for S3

```
Cost: $0.00 (Gateway endpoints for S3 and DynamoDB are FREE)
Data processed: $0.00 (traffic stays on AWS backbone, no data processing charge)

For S3-heavy AI workloads (100 GB/month):
  Total = $0.00
```

### Difference

| Traffic | NAT Gateway | VPC Endpoint | Savings |
|---------|-------------|--------------|---------|
| 100 GB/month to S3 | $36.90/month | $0/month | $36.90 |
| 1 TB/month to S3 | $77.40/month | $0/month | $77.40 |
| NAT fixed cost alone | $32.40/month | $0/month | $32.40 |

**Why VPC endpoints win:**

1. **Gateway endpoints** (S3, DynamoDB) — free, no route through NAT, add a route table entry that directs `.s3.amazonaws.com` traffic straight to AWS's backbone.
2. **Interface endpoints** (most other AWS services) — cost ~$7.30/month per AZ per service but still far cheaper than NAT for high-volume services like Secrets Manager or ECR.
3. **Security posture** improves — traffic never leaves the AWS network, reducing attack surface.
4. **Latency** drops slightly — no NAT translation hop.

**Terraform to add S3 Gateway endpoint:**

```hcl
resource "aws_vpc_endpoint" "s3" {
  vpc_id            = aws_vpc.main.id
  service_name      = "com.amazonaws.${var.aws_region}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = aws_route_table.private[*].id

  tags = {
    Name = "${var.app_name}-s3-endpoint"
  }
}
```

**Rule of thumb:** For any AI workload that reads model weights, training data, or embeddings from S3, always add the Gateway endpoint first — it's free and saves significant money at scale.
