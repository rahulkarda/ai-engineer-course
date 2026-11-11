# Phase 6 · Week 08 · Day 2 — Managed Container Services

> Objective: Understand how managed container platforms (ECS, Cloud Run, Container Apps) abstract Kubernetes, and how to deploy AI workloads on them securely.  ·  Time: ~1 hr  ·  Prereqs: [[week-08/day-1]]

---

## Concept

**Managed container services** sit between raw Kubernetes and FaaS. You ship a container image; the platform handles scheduling, health restarts, networking, and auto-scaling. You never touch a node or control plane.

The three major equivalents:

| Platform | Service | Pricing unit |
|---|---|---|
| AWS | ECS (Fargate launch type) | vCPU-second + GB-second |
| GCP | Cloud Run | CPU-second + request count |
| Azure | Container Apps | vCPU-second + replica count |

**Key differences from raw Kubernetes:**
- No nodes to patch, no `kubectl` required for routine ops.
- Declarative task/service definitions (JSON/YAML or IaC) rather than Deployment + Service + Ingress objects.
- Scaling policies are simpler: target-tracking on CPU/memory or request-count, not full HPA/KEDA configuration.
- Networking is VPC-native but simplified: no CNI plugin selection, no kube-proxy.

**When raw k8s still wins:** multi-tenant workloads, custom schedulers, GPU fractioning, stateful sets with complex storage, existing Helm ecosystem.

---

## How it works internally

### ECS conceptual model

```
Cluster
  └── Service  (desired count, load balancer, auto-scaling policy)
        └── Task  (running instance of a Task Definition)
              └── Container(s)  (image, CPU, memory, env, ports, log driver)
```

- **Task Definition**: immutable versioned blueprint (image URI, resource limits, IAM task role, log config).
- **Service**: long-running process; ECS scheduler reconciles desired vs running count.
- **Task**: ephemeral execution unit. Fargate provisions an isolated microVM per task; no shared kernel with other customers.

### Networking

Each Fargate task gets an **Elastic Network Interface (ENI)** injected into your VPC subnet. The task has a private IP. Traffic flows:

```
Internet → ALB (public subnet) → Target Group → Task ENI (private subnet)
```

Tasks in private subnets need a **NAT Gateway** (or VPC endpoints) to reach ECR, Secrets Manager, and external APIs. This is the most common misconfiguration: tasks in private subnets with no egress route silently fail to pull images.

### IAM for containers

Two IAM roles matter:

1. **Task Execution Role** — used by the ECS agent *before* the container starts. Needs: `ecr:GetAuthorizationToken`, `logs:CreateLogStream`, `secretsmanager:GetSecretValue` (if using secrets injection).
2. **Task Role** — injected into the running container via the instance metadata endpoint (IMDS). Used by your application code. Scope to only what the app needs.

Never put credentials in environment variables or bake them into images. Use the Task Role + IAM policies.

### Service discovery

ECS integrates with **AWS Cloud Map**: each task registers a DNS record. Services discover each other via `http://my-service.local:8080` — no service mesh required for basic use cases.

---

## Worked example

Two artifacts: an ECS Task Definition (JSON) and a minimal Terraform snippet that creates the service.

### Task Definition (JSON)

```json
{
  "family": "rag-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::ACCOUNT_ID:role/ecs-execution-role",
  "taskRoleArn": "arn:aws:iam::ACCOUNT_ID:role/rag-api-task-role",
  "containerDefinitions": [
    {
      "name": "rag-api",
      "image": "ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/rag-api:latest",
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "LOG_LEVEL",
          "value": "INFO"
        },
        {
          "name": "VECTOR_DB_HOST",
          "value": "vectordb.internal"
        }
      ],
      "secrets": [
        {
          "name": "ANTHROPIC_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:REGION:ACCOUNT_ID:secret:prod/rag-api/anthropic-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/rag-api",
          "awslogs-region": "REGION",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

**Notes:**
- `secrets` block fetches from Secrets Manager at task start — the value is injected as an env var inside the container but never stored in the task definition plaintext.
- `executionRoleArn` must have `secretsmanager:GetSecretValue` for the ARN above.
- `startPeriod: 60` gives the container grace time before health checks begin — important for AI apps that load models on startup.

### Terraform snippet (ECS Service + auto-scaling)

```hcl
# ecs_service.tf
# Assumes: VPC, subnets, security groups, ALB target group, and task definition already created.

resource "aws_ecs_service" "rag_api" {
  name            = "rag-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.rag_api.arn
  desired_count   = 2
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.rag_api.id]
    assign_public_ip = false  # tasks in private subnet; egress via NAT GW
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.rag_api.arn
    container_name   = "rag-api"
    container_port   = 8080
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true  # auto-rollback on consecutive health check failures
  }

  lifecycle {
    ignore_changes = [desired_count]  # managed by autoscaling
  }
}

# Auto-scaling: scale out when CPU > 70%
resource "aws_appautoscaling_target" "rag_api" {
  max_capacity       = 10
  min_capacity       = 2
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.rag_api.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "rag_api_cpu" {
  name               = "rag-api-cpu-tracking"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.rag_api.resource_id
  scalable_dimension = aws_appautoscaling_target.rag_api.scalable_dimension
  service_namespace  = aws_appautoscaling_target.rag_api.service_namespace

  target_tracking_scaling_policy_configuration {
    target_value = 70.0
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    scale_in_cooldown  = 300
    scale_out_cooldown = 60  # scale out fast, scale in slow
  }
}
```

### Python health check endpoint (FastAPI)

```python
# app/health.py
# Minimal health check router — attach to your FastAPI app.

from __future__ import annotations

import logging
import os

from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    """
    ECS health check target.
    Returns 200 with service metadata; no secrets exposed.
    """
    return {
        "status": "ok",
        "service": os.getenv("SERVICE_NAME", "rag-api"),
        "version": os.getenv("IMAGE_TAG", "unknown"),
    }
```

---

## Exercises

1. **Networking gap**: A Fargate task in a private subnet fails to start with `CannotPullContainerError`. List the three most likely root causes and how you would diagnose each using AWS console or CLI (no code required — describe the steps).

2. **IAM least-privilege**: Your RAG API task needs to: (a) read from an S3 bucket `rag-docs`, (b) call Bedrock runtime, (c) read one secret from Secrets Manager. Write the IAM policy JSON for the **Task Role** (not execution role). Include only the minimum actions needed.

3. **Multi-container task**: You want to add an OpenTelemetry sidecar container to the task definition above. The sidecar runs on port 4317 (gRPC) and the main container sends traces to `localhost:4317`. Modify the task definition JSON to add the sidecar. What log configuration should the sidecar use? What resource limits are appropriate for a sidecar?

4. **Cloud Run equivalent**: Translate the key concepts of the ECS task definition above to a Cloud Run service YAML (`apiVersion: serving.knative.dev/v1`). Identify which fields map directly, which have no equivalent, and which Cloud Run handles automatically.

(Solutions: ../solutions/week-08/day-2.md)

---

## Self-check

- [ ] I can explain the difference between a Task Definition and a Service in ECS, without notes.
- [ ] I can describe why Fargate tasks in private subnets need a NAT Gateway or VPC endpoints.
- [ ] I can explain why there are two separate IAM roles (execution role vs task role) and what each grants.
- [ ] I can read an ECS task definition JSON and identify where secrets are injected and how they flow.
- [ ] I can implement the Terraform auto-scaling policy without reference.

---

## Supplementary links

- [AWS ECS — Task definitions](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task_definitions.html)
- [AWS ECS — Fargate task networking](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task-networking-awsvpc.html)
- [GCP Cloud Run — Service YAML reference](https://cloud.google.com/run/docs/reference/rest/v1/namespaces.services)
- [Azure Container Apps — Concepts](https://learn.microsoft.com/en-us/azure/container-apps/overview)
- [AWS — IAM roles for ECS tasks](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task-iam-roles.html)
- [Terraform AWS provider — ECS service](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/ecs_service)
