# Phase 6 · Week 05 · Day 5 — Probes, Resource Limits, and Rolling Updates

> Objective: Configure liveness/readiness probes, set resource requests/limits, and control rolling update behavior.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

## Concept

### Liveness probe — is the container healthy?

A **liveness probe** tells kubelet whether the container is alive. If it fails `failureThreshold` times, kubelet **restarts** the container. Use it to recover from deadlocks or stuck processes that are running but not functional.

### Readiness probe — is the container ready for traffic?

A **readiness probe** tells kubelet whether the container is ready to receive traffic. If it fails, the pod is **removed from the Service Endpoints** (no traffic) but NOT restarted. Use it to:
- Hold traffic during startup (wait for DB migrations, model loading)
- Remove traffic during temporary overload

### Startup probe — is the container done initializing?

A **startup probe** disables liveness/readiness probes until it succeeds. For slow-starting apps (ML model loading, JVM warmup). Once it succeeds, normal probes take over.

### Resource requests and limits

| Field | Meaning | Effect |
|-------|---------|--------|
| `requests.cpu` | Minimum CPU reserved | Scheduler uses this for placement |
| `requests.memory` | Minimum memory reserved | Scheduler uses this for placement |
| `limits.cpu` | Maximum CPU allowed | Container is throttled (not killed) |
| `limits.memory` | Maximum memory allowed | Container is OOM-killed if exceeded |

CPU is measured in **millicores** (m): `100m` = 0.1 vCPU, `1000m` = 1 vCPU.
Memory in bytes with suffixes: `128Mi`, `1Gi`.

**Rule of thumb:** always set both requests and limits. Requests without limits create noisy-neighbor risk. Limits without requests means the pod can land on an overloaded node.

### Quality of Service classes

k8s assigns a QoS class to each pod based on its resource config:

| Class | Condition | Eviction priority |
|-------|-----------|------------------|
| `Guaranteed` | All containers: `requests == limits` | Last evicted |
| `Burstable` | At least one container has requests set | Middle |
| `BestEffort` | No requests or limits at all | First evicted |

For production: target `Guaranteed` (requests = limits) or `Burstable` with explicit requests.

### Rolling update controls

```yaml
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxSurge: 1        # extra pods above desired during rollout (int or %)
    maxUnavailable: 0  # pods below desired that may be unavailable (int or %)
```

`maxUnavailable: 0` + `maxSurge: 1` = **one pod at a time, no downtime**. Needs capacity for N+1 pods.
`maxUnavailable: 25%` + `maxSurge: 25%` = faster rollout, some unavailability OK.

## How it works internally

**Probe execution** — kubelet runs probes directly, not through the API server. Three mechanisms:
- `httpGet` — kubelet sends HTTP GET; 2xx-3xx = success
- `exec` — kubelet runs a command in the container; exit 0 = success
- `tcpSocket` — kubelet opens a TCP connection; success = port is open
- `grpc` — calls gRPC Health Checking Protocol

**Probe timing parameters:**
- `initialDelaySeconds` — wait before first probe (legacy; startup probe is better)
- `periodSeconds` — how often to probe
- `timeoutSeconds` — probe timeout
- `failureThreshold` — consecutive failures before action
- `successThreshold` — consecutive successes to be considered healthy (readiness)

**CPU throttling** — CPU limits are enforced by Linux cgroups. A container using more than its limit has CPU cycles throttled (CFS bandwidth control). It stays running but slows down. This is different from memory: exceeding memory limit → OOM kill (immediate termination).

**Horizontal Pod Autoscaler (HPA)** — reads CPU/memory utilization from the metrics-server and adjusts `spec.replicas` on the Deployment. Only works when requests are set (it compares usage / request to a target percentage).

## Worked example

### Complete container spec with probes and resources

```yaml
# deployment-production.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fastapi-app
  namespace: default
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fastapi
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: fastapi
        version: "2.0"
    spec:
      containers:
        - name: api
          image: ghcr.io/myorg/fastapi-app:2.0.0
          ports:
            - containerPort: 8000

          # Resource budget
          resources:
            requests:
              cpu: "100m"
              memory: "128Mi"
            limits:
              cpu: "500m"
              memory: "256Mi"

          # Startup probe: wait up to 60s for the app to initialize
          # (failureThreshold × periodSeconds = 30 × 2 = 60s)
          startupProbe:
            httpGet:
              path: /health
              port: 8000
            failureThreshold: 30
            periodSeconds: 2

          # Liveness probe: restart if app becomes unresponsive
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 0   # startup probe handles the wait
            periodSeconds: 10
            timeoutSeconds: 3
            failureThreshold: 3
            successThreshold: 1

          # Readiness probe: hold traffic until app is warm
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 0
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 3
            successThreshold: 1
```

### FastAPI health and readiness endpoints

```python
# app/main.py
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response

logger = logging.getLogger(__name__)

# Startup state
_ready: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ready
    logger.info("Loading model...")
    # Simulate model load — in production: load actual model here
    import asyncio
    await asyncio.sleep(2)
    _ready = True
    logger.info("App ready")
    yield
    _ready = False


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    """Liveness check — always returns 200 if process is running."""
    return {"status": "ok"}


@app.get("/ready")
async def ready(response: Response) -> dict:
    """Readiness check — 200 when ready to serve, 503 otherwise."""
    if not _ready:
        response.status_code = 503
        return {"status": "initializing"}
    return {"status": "ready"}
```

### Exec and TCP probe alternatives

```yaml
# Exec probe — run a script inside the container
livenessProbe:
  exec:
    command:
      - python
      - -c
      - "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
  periodSeconds: 15

# TCP probe — useful for databases, message brokers
livenessProbe:
  tcpSocket:
    port: 5432
  periodSeconds: 10
```

### Observing probe failures

```bash
# Watch events for probe failures
kubectl describe pod <pod-name>
# Look for: Liveness probe failed / Readiness probe failed events

# Watch restart count
kubectl get pods -w
# RESTARTS column increments on liveness failure

# If pod is crashing, get logs from the previous container
kubectl logs <pod-name> --previous
```

### Horizontal Pod Autoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fastapi-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fastapi-app
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70   # scale up if avg CPU > 70% of request
```

```bash
kubectl apply -f hpa.yaml
kubectl get hpa fastapi-hpa
kubectl describe hpa fastapi-hpa
```

## Exercises

1. Write a liveness probe for a FastAPI app at `/health` on port 8000 that: waits 0 seconds initially, checks every 10 seconds, times out after 3 seconds, and restarts after 3 consecutive failures.

2. What is the difference between a liveness probe failing and a readiness probe failing? Describe the exact action kubelet takes for each.

3. A pod has `requests.memory: 128Mi` and `limits.memory: 256Mi`. The app's memory usage grows to 300Mi. What happens? Is this a restart or a termination?

4. Your Deployment has `maxUnavailable: 0, maxSurge: 1` and 4 replicas. Trace the exact pod lifecycle during a rolling update: how many pods run at each step?

5. Write an HPA manifest that scales the `model-server` Deployment between 1 and 8 replicas, targeting 60% CPU utilization. What must be true about the Deployment for HPA to work?

(Solutions: ../solutions/week-05/day-5.md)

## Self-check

- [ ] I can configure all three probe types (liveness, readiness, startup) with correct parameters
- [ ] I can explain the difference between liveness and readiness probe consequences
- [ ] I can set resource requests and limits and explain the scheduling implications
- [ ] I understand QoS classes and which one my pod gets
- [ ] I can write an HPA manifest and explain its prerequisites

## Supplementary links

- [Configure Liveness, Readiness, and Startup Probes](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/)
- [Resource Management for Pods and Containers](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/)
- [Horizontal Pod Autoscaling](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [QoS Classes](https://kubernetes.io/docs/concepts/workloads/pods/pod-qos/)
