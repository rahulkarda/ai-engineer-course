# Phase 6 · Week 08 · Day 5 — Autoscaling & Cost Awareness in Cloud AI

> Objective: Design autoscaling policies for containerized AI workloads and build a cost model for a production AI API.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

## Concept

Autoscaling keeps compute matched to demand without manual intervention. For AI APIs, demand is spiky — model inference is expensive, so over-provisioning burns money and under-provisioning causes latency spikes or timeouts.

Three scaling axes:
- **Horizontal scaling** — add/remove container instances (ECS tasks, K8s pods).
- **Vertical scaling** — resize instance type (disruptive; usually done at deploy time).
- **Scheduled scaling** — pre-warm capacity before known traffic events.

Three policy types (AWS ECS / Application Auto Scaling):
- **Target tracking** — maintain a metric at a target value (CPU at 60%, requests/task at 100). Simplest; AWS manages scale-in/out math.
- **Step scaling** — explicit thresholds map to discrete capacity adjustments. More control; more configuration.
- **Scheduled scaling** — set min/max/desired on a cron. Useful for batch AI jobs that run on a fixed schedule.

Cost anatomy for a containerized AI API:
| Layer | AWS example | Notes |
|---|---|---|
| Compute | ECS Fargate vCPU-hours + GB-hours | Dominates at scale |
| LLM API calls | Claude API tokens | Per-token pricing; biggest variable cost |
| Data transfer | $0.09/GB egress | Often overlooked |
| Load balancer | ALB: $0.008/LCU-hr | Constant baseline |
| Storage | S3, EFS, ECR | Usually small |
| Secrets Manager | $0.40/secret/month + API calls | Negligible |

**Spot / preemptible instances** — AWS Spot Instances (GCP preemptible, Azure Spot) offer 60–90% discount at cost of possible interruption. Suitable for:
- Batch AI processing (embeddings, re-ranking pipelines, offline eval)
- Non-latency-sensitive inference queues
- NOT suitable for synchronous user-facing inference (without fallback)

Pattern: run baseline capacity on On-Demand, burst to Spot. ECS Fargate Spot handles this natively via capacity provider strategies.

## How it works internally

**Target tracking under the hood:**

AWS Application Auto Scaling publishes a CloudWatch alarm on your chosen metric. A PID-like controller runs on AWS side: when the metric breaches the threshold by more than a cooldown window, it calls `UpdateService` to change `desiredCount`. It tracks two alarms — one for scale-out (metric > target), one for scale-in (metric < target × scale-in fraction, default 0.9).

Scale-in is deliberately slow (default 300s cooldown) to avoid flapping. Scale-out is fast (default 0s warmup for CPU; configurable for custom metrics like SQS depth).

**Why CPU target tracking underperforms for LLM APIs:**

LLM inference is I/O bound (waiting for token generation) not CPU bound. A task can be handling 5 concurrent streaming requests, all waiting on the Claude API, while CPU sits at 10%. Better metric: **requests per target** (ALB `RequestCountPerTarget`) or a custom CloudWatch metric for active inference requests.

**Spot interruption handling:**

AWS sends a 2-minute warning via EC2 instance metadata and EventBridge before reclaiming a Spot instance. ECS drains the task — stops new requests routing to it, lets in-flight requests complete within the draining timeout. If draining timeout exceeds 2 minutes, in-flight requests may fail. Set `deregistration_delay` on the target group ≤ 90s for Spot tasks.

## Worked example

```hcl
# terraform/modules/autoscaling/main.tf
# ECS Service autoscaling — target tracking on requests per task
# No secrets, no hardcoded account IDs

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# ── Scalable target ──────────────────────────────────────────────────────────
resource "aws_appautoscaling_target" "ecs_service" {
  max_capacity       = var.max_capacity       # e.g. 20
  min_capacity       = var.min_capacity       # e.g. 2
  resource_id        = "service/${var.ecs_cluster_name}/${var.ecs_service_name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

# ── Target tracking: keep requests/task near 80 ──────────────────────────────
resource "aws_appautoscaling_policy" "requests_per_task" {
  name               = "${var.ecs_service_name}-rpt-tracking"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs_service.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_service.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs_service.service_namespace

  target_tracking_scaling_policy_configuration {
    target_value       = 80  # requests per task target
    scale_in_cooldown  = 300 # seconds; wait before scale-in to avoid flap
    scale_out_cooldown = 60  # seconds; fast scale-out for latency

    predefined_metric_specification {
      predefined_metric_type = "ALBRequestCountPerTarget"
      # resource_label links ALB + target group; injected from outputs
      resource_label = var.alb_target_group_resource_label
    }
  }
}

# ── Step scaling: emergency scale-out on high CPU ────────────────────────────
resource "aws_appautoscaling_policy" "cpu_step_out" {
  name               = "${var.ecs_service_name}-cpu-step-out"
  policy_type        = "StepScaling"
  resource_id        = aws_appautoscaling_target.ecs_service.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_service.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs_service.service_namespace

  step_scaling_policy_configuration {
    adjustment_type          = "ChangeInCapacity"
    cooldown                 = 60
    metric_aggregation_type  = "Average"

    step_adjustment {
      metric_interval_lower_bound = 0   # CPU >= 80%
      metric_interval_upper_bound = 20  # CPU < 100%
      scaling_adjustment          = 2   # add 2 tasks
    }
    step_adjustment {
      metric_interval_lower_bound = 20  # CPU >= 100%
      scaling_adjustment          = 5   # add 5 tasks (emergency)
    }
  }
}

# ── Scheduled scaling: pre-warm before 9am UTC business hours ───────────────
resource "aws_appautoscaling_scheduled_action" "morning_prewarm" {
  name               = "${var.ecs_service_name}-morning-prewarm"
  service_namespace  = aws_appautoscaling_target.ecs_service.service_namespace
  resource_id        = aws_appautoscaling_target.ecs_service.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_service.scalable_dimension
  schedule           = "cron(45 8 ? * MON-FRI *)"  # 08:45 UTC Mon-Fri

  scalable_target_action {
    min_capacity = 5
    max_capacity = var.max_capacity
  }
}

resource "aws_appautoscaling_scheduled_action" "night_scaledown" {
  name               = "${var.ecs_service_name}-night-scaledown"
  service_namespace  = aws_appautoscaling_target.ecs_service.service_namespace
  resource_id        = aws_appautoscaling_target.ecs_service.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_service.scalable_dimension
  schedule           = "cron(0 22 ? * MON-FRI *)"  # 22:00 UTC Mon-Fri

  scalable_target_action {
    min_capacity = var.min_capacity
    max_capacity = var.max_capacity
  }
}
```

```hcl
# terraform/modules/autoscaling/variables.tf

variable "ecs_cluster_name" {
  type        = string
  description = "ECS cluster name"
}

variable "ecs_service_name" {
  type        = string
  description = "ECS service name"
}

variable "min_capacity" {
  type        = number
  default     = 2
  description = "Minimum number of ECS tasks"
}

variable "max_capacity" {
  type        = number
  default     = 20
  description = "Maximum number of ECS tasks"
}

variable "alb_target_group_resource_label" {
  type        = string
  description = "ALB/targetgroup resource label for RequestCountPerTarget metric"
  # Format: app/<alb-name>/<alb-id>/targetgroup/<tg-name>/<tg-id>
}
```

**Cost estimation worksheet (Python):**

```python
"""
week_08/cost_estimator.py

Estimate monthly cost for a containerized AI API on ECS Fargate.
All prices are US East (us-east-1) illustrative values — verify current pricing.
Env vars: none required (pure calculation).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


@dataclass
class TrafficProfile:
    """Monthly traffic assumptions."""
    avg_rps: float            # average requests per second
    peak_rps: float           # peak RPS (autoscaling target)
    avg_input_tokens: int     # tokens per request (input)
    avg_output_tokens: int    # tokens per request (output)
    avg_response_kb: float    # response size in KB (for egress calc)
    hours_per_day: float = 24.0
    days_per_month: float = 30.0


@dataclass
class FargatePricing:
    """Fargate pricing — verify at https://aws.amazon.com/fargate/pricing/"""
    vcpu_per_hour: float = 0.04048
    gb_per_hour: float = 0.004445


@dataclass
class TaskSpec:
    """Per-task resource allocation."""
    vcpu: float = 0.5
    memory_gb: float = 1.0
    requests_per_task_target: int = 80  # matches autoscaling target


@dataclass
class LLMPricing:
    """Claude API illustrative pricing — verify at https://www.anthropic.com/pricing"""
    input_per_million_tokens: float = 3.0    # $ per 1M input tokens
    output_per_million_tokens: float = 15.0  # $ per 1M output tokens


@dataclass
class CostBreakdown:
    compute_usd: float = 0.0
    llm_api_usd: float = 0.0
    data_transfer_usd: float = 0.0
    alb_usd: float = 0.0
    total_usd: float = field(init=False)

    def __post_init__(self) -> None:
        self.total_usd = (
            self.compute_usd
            + self.llm_api_usd
            + self.data_transfer_usd
            + self.alb_usd
        )


def estimate_monthly_cost(
    traffic: TrafficProfile,
    fargate: FargatePricing,
    task: TaskSpec,
    llm: LLMPricing,
) -> CostBreakdown:
    """Calculate estimated monthly cost."""
    total_hours = traffic.hours_per_day * traffic.days_per_month
    total_requests = traffic.avg_rps * total_hours * 3600

    # Tasks needed at average load (with 20% headroom)
    avg_tasks = max(2.0, (traffic.avg_rps / task.requests_per_task_target) * 1.2)
    log.info("Estimated average task count: %.1f", avg_tasks)

    # Compute cost
    compute = avg_tasks * total_hours * (
        task.vcpu * fargate.vcpu_per_hour
        + task.memory_gb * fargate.gb_per_hour
    )

    # LLM API cost
    total_input_tokens = total_requests * traffic.avg_input_tokens
    total_output_tokens = total_requests * traffic.avg_output_tokens
    llm_cost = (
        (total_input_tokens / 1_000_000) * llm.input_per_million_tokens
        + (total_output_tokens / 1_000_000) * llm.output_per_million_tokens
    )

    # Data transfer egress (requests out)
    total_egress_gb = (total_requests * traffic.avg_response_kb) / (1024 * 1024)
    egress_cost = total_egress_gb * 0.09  # $0.09/GB after 1GB free tier

    # ALB cost (simplified: $16/month baseline + LCU cost)
    alb_cost = 16.0 + (traffic.avg_rps * 0.008 * total_hours)

    breakdown = CostBreakdown(
        compute_usd=round(compute, 2),
        llm_api_usd=round(llm_cost, 2),
        data_transfer_usd=round(egress_cost, 2),
        alb_usd=round(alb_cost, 2),
    )
    return breakdown


if __name__ == "__main__":
    profile = TrafficProfile(
        avg_rps=10.0,
        peak_rps=50.0,
        avg_input_tokens=500,
        avg_output_tokens=200,
        avg_response_kb=2.0,
    )
    result = estimate_monthly_cost(
        profile,
        FargatePricing(),
        TaskSpec(),
        LLMPricing(),
    )
    log.info("=== Monthly Cost Estimate ===")
    log.info("Compute:       $%.2f", result.compute_usd)
    log.info("LLM API:       $%.2f", result.llm_api_usd)
    log.info("Data transfer: $%.2f", result.data_transfer_usd)
    log.info("ALB:           $%.2f", result.alb_usd)
    log.info("TOTAL:         $%.2f", result.total_usd)
```

## Exercises

1. Modify `TrafficProfile` in `cost_estimator.py` to model a bursty workload: 2 RPS off-peak (18 hrs/day) and 40 RPS peak (6 hrs/day). Compare total cost against the flat 10 RPS scenario. Which cost component grows most?

2. The current target tracking policy uses `ALBRequestCountPerTarget`. Research the `ECSServiceAverageCPUUtilization` predefined metric. Write a second `aws_appautoscaling_policy` resource using CPU target tracking at 60%. When would you use each? Add comments explaining the trade-off.

3. Design an autoscaling strategy for an offline embedding pipeline: 10,000 documents arrive every night at 02:00 UTC, processing takes ~45 minutes, then the cluster should idle. Sketch the scheduled scaling actions (times, min/max values) and explain why target tracking alone is insufficient here.

4. Calculate the Spot savings for the batch pipeline in Exercise 3 assuming a 70% Spot discount. Add a `spot_savings_usd` field to `CostBreakdown` and a parameter to `estimate_monthly_cost` to toggle Spot mode.

(Solutions: ../solutions/week-08/day-5.md)

## Self-check

- [ ] I can explain the difference between target tracking, step scaling, and scheduled scaling, and name a use case for each.
- [ ] I can list the five main cost components of a containerized AI API and identify which dominates at high token volume.
- [ ] I can explain why CPU utilization is a poor autoscaling metric for LLM-backed APIs and name a better alternative.
- [ ] I can describe Spot instance interruption behavior and the draining timeout constraint.
- [ ] I can implement a target tracking policy in Terraform without referring to documentation.

---

## Supplementary Links

- [AWS Application Auto Scaling — Target Tracking](https://docs.aws.amazon.com/autoscaling/application/userguide/application-auto-scaling-target-tracking.html)
- [ECS Fargate Spot — Capacity Provider Strategy](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/fargate-capacity-providers.html)
- [AWS Pricing Calculator](https://calculator.aws/pricing/2/home)
- [Terraform aws_appautoscaling_policy](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/appautoscaling_policy)
- [CloudWatch Metrics for ECS](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/cloudwatch-metrics.html)
