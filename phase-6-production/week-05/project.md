# Week 05 Project — Kubernetes Manifests for the FastAPI App

## Overview

Deploy the week-4 containerized FastAPI app to a local Kubernetes cluster using production-quality manifests. Write a complete operations runbook. Perform a zero-downtime rolling update and a rollback.

## Learning outcomes

- Author all six core manifest types from scratch
- Configure probes and resource limits correctly
- Manage configuration and secrets following k8s best practices
- Operate a deployment through its full lifecycle (deploy → update → rollback)
- Document operational procedures in a runbook

## Prerequisites

- Week 4 FastAPI app containerized and locally tagged as `ghcr.io/myorg/fastapi-app:1.0.0`
- `kind`, `kubectl`, `docker` installed
- Days 1–6 of this week complete

## Deliverables

```
k8s/
├── kind-config.yaml       # cluster configuration
├── namespace.yaml
├── configmap.yaml
├── deployment.yaml
├── service.yaml
├── hpa.yaml
└── RUNBOOK.md
```

A `secret-generated.yaml` is produced at deploy time from environment variables — it is NOT committed to the repo. Add `*secret-generated*` to `.gitignore`.

## Requirements

### Namespace

- Name: `ai-platform`
- Labels: `env: production`, `team: ml-infra`

### ConfigMap

- Name: `fastapi-config`, namespace `ai-platform`
- Must include: `APP_ENV`, `LOG_LEVEL`, `API_HOST`, `API_PORT`
- Must include a multi-line config file key (`settings.toml`) mountable at `/app/config/settings.toml`

### Secret

- Name: `fastapi-secrets`, namespace `ai-platform`
- Keys: `DATABASE_URL`, `JWT_SECRET_KEY`
- Created imperatively from env vars — no plaintext in any committed file
- YAML file contains only placeholder base64 values with inline documentation

### Deployment

| Requirement | Value |
|-------------|-------|
| Replicas | 3 |
| Image | `ghcr.io/myorg/fastapi-app:1.0.0` |
| Strategy | RollingUpdate: maxSurge=1, maxUnavailable=0 |
| startupProbe | `/health`, 30 retries × 2s = 60s window |
| livenessProbe | `/health`, period=10s, fail=3 |
| readinessProbe | `/ready`, period=5s, fail=3 |
| CPU request | 100m |
| CPU limit | 500m |
| Memory request | 128Mi |
| Memory limit | 256Mi |
| terminationGracePeriodSeconds | 30 |
| ConfigMap injection | envFrom + volume mount |
| Secret injection | specific keys via secretKeyRef |

### Service

- Type: `ClusterIP`
- Port 80 → targetPort `http` (named port)
- Selector: `app: fastapi`

### HPA

- Min replicas: 2, max: 10
- CPU target: 70% utilization
- Scale-down stabilization: 300s

### Runbook

Cover all of: first-time setup, standard deploy, rolling update, rollback, health check commands, port-forward for local testing, troubleshooting table (5 failure modes).

## Verification steps

Run these in order and confirm each passes:

```bash
# 1. Cluster is up
kubectl get nodes

# 2. All manifests validate
kubectl apply -f k8s/ --dry-run=server -n ai-platform

# 3. Deployment rolls out
kubectl rollout status deployment/fastapi-app -n ai-platform

# 4. All 3 pods running
kubectl get pods -n ai-platform
# Expected: 3 pods in Running state

# 5. Endpoints populated
kubectl get endpoints fastapi-service -n ai-platform
# Expected: 3 pod IPs listed

# 6. Health check passes
kubectl port-forward service/fastapi-service 8080:80 -n ai-platform &
curl -s http://localhost:8080/health | grep '"status":"ok"'
curl -s http://localhost:8080/ready | grep '"status":"ready"'

# 7. Rolling update
kubectl set image deployment/fastapi-app \
  api=ghcr.io/myorg/fastapi-app:2.0.0 -n ai-platform
kubectl rollout status deployment/fastapi-app -n ai-platform

# 8. Rollback
kubectl rollout undo deployment/fastapi-app -n ai-platform
kubectl rollout status deployment/fastapi-app -n ai-platform
kubectl get pods -n ai-platform -o jsonpath='{.items[0].spec.containers[0].image}'
# Expected: image ends in :1.0.0
```

## Extension tasks (optional)

1. Add a `PodDisruptionBudget` ensuring at least 2 pods are always available.
2. Add a `NetworkPolicy` allowing ingress only from within the namespace.
3. Add a `LimitRange` to the namespace setting defaults for any pod that doesn't specify resources.
4. Write a `CronJob` that hits `/health` every minute and logs the response.

## Grading

| Criterion | Points |
|-----------|--------|
| All 6 manifest files present and valid YAML | 20 |
| Deployment: correct selectors, probes, resources | 25 |
| ConfigMap + Secret used correctly (no secrets in YAML) | 20 |
| Rolling update executed and verified | 15 |
| Rollback executed and verified | 10 |
| Runbook: complete, accurate, runnable commands | 10 |
| **Total** | **100** |
| **Pass** | **75** |

Solution: ../solutions/week-05/project.md
