# Phase 6 · Week 06 · Day 1 — Autoscaling: HPA, Resource Metrics & Custom Metrics

> Objective: Configure Horizontal Pod Autoscaler with CPU/memory resource metrics and Prometheus custom metrics.  ·  Time: ~1 hr  ·  Prereqs: [[week-05/day-7]]

## Concept

Kubernetes autoscaling adjusts the number of pod replicas automatically in response to observed load. The **Horizontal Pod Autoscaler (HPA)** is the primary mechanism for stateless workloads — it scales out (more pods) or in (fewer pods) based on a target utilization metric.

Three metric sources are supported:

| Source | Where it comes from | Typical use |
|--------|---------------------|-------------|
| `Resource` | kubelet → metrics-server | CPU %, Memory % |
| `Pods` | custom-metrics API | requests/sec per pod |
| `External` | external-metrics API | queue depth, SQS depth |

The HPA control loop:
1. Scrapes the metrics API at a configurable interval (default 15 s).
2. Computes desired replicas: `ceil(currentReplicas × (currentMetric / targetMetric))`.
3. Applies stabilization windows to prevent flapping (default: 5 min scale-down, 0 min scale-up).

## How it works internally

**metrics-server** aggregates resource usage from each node's kubelet summary API. It is a cluster add-on — not installed by default in bare clusters but required for `kubectl top` and HPA resource metrics.

**custom-metrics-apiserver** (e.g., Prometheus Adapter) translates Prometheus query results into Kubernetes API objects at path `apis/custom.metrics.k8s.io/v1beta2`. The HPA controller fetches from that API exactly as it does from metrics-server.

Scale algorithm edge cases:
- Missing metrics: HPA uses the last known value for up to `--horizontal-pod-autoscaler-initial-readiness-delay` (default 30 s), then applies a 1.1× tolerance.
- Readiness: newly started pods are excluded from averages for `--horizontal-pod-autoscaler-cpu-initialization-period` (default 5 min).
- Scale-down stabilization window: HPA tracks the highest desired replica count over the window before committing a scale-down.

## Worked example

### 1. Metrics-server (if not present)

```bash
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
# Verify
kubectl top nodes
kubectl top pods -n default
```

### 2. Deployment with resource requests/limits

```yaml
# deploy-api.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-api
  namespace: production
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ai-api
  template:
    metadata:
      labels:
        app: ai-api
    spec:
      containers:
        - name: api
          image: registry.example.com/ai-api:v1.0.0   # placeholder
          resources:
            requests:
              cpu: "200m"       # 0.2 cores — baseline for HPA math
              memory: "256Mi"
            limits:
              cpu: "800m"
              memory: "512Mi"
          env:
            - name: MODEL_ENDPOINT
              valueFrom:
                secretKeyRef:
                  name: ai-secrets
                  key: model-endpoint
```

> Resource requests MUST be set — HPA cannot compute utilization without them.

### 3. HPA v2 — resource metrics (CPU + memory)

```yaml
# hpa-resource.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-api-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-api
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 60   # scale up if avg CPU > 60% of request
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 70
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30
      policies:
        - type: Percent
          value: 100        # allow doubling replicas in one step
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300   # wait 5 min before scaling down
      policies:
        - type: Pods
          value: 2          # remove at most 2 pods per 60 s
          periodSeconds: 60
```

```bash
kubectl apply -f deploy-api.yaml
kubectl apply -f hpa-resource.yaml
kubectl get hpa -n production -w   # watch events
kubectl describe hpa ai-api-hpa -n production
```

### 4. HPA v2 — custom metric (requests per second via Prometheus Adapter)

```yaml
# hpa-custom.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-api-rps-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-api
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Pods
      pods:
        metric:
          name: http_requests_per_second   # must match Prometheus Adapter config
        target:
          type: AverageValue
          averageValue: "50"   # target 50 req/s per pod
```

Prometheus Adapter rule (excerpt in `custom-metrics-apiserver` ConfigMap):

```yaml
rules:
  - seriesQuery: 'http_requests_total{namespace!="",pod!=""}'
    resources:
      overrides:
        namespace: {resource: "namespace"}
        pod: {resource: "pod"}
    name:
      matches: "^(.*)_total$"
      as: "${1}_per_second"
    metricsQuery: 'rate(<<.Series>>{<<.LabelMatchers>>}[2m])'
```

### 5. Inspect HPA state

```bash
# Current metrics being observed
kubectl get hpa ai-api-hpa -n production -o json | jq '.status'

# Event log (scaling decisions)
kubectl describe hpa ai-api-hpa -n production | grep -A 20 Events

# Simulate load (in a test cluster)
kubectl run load-gen --image=busybox --rm -it --restart=Never -- \
  /bin/sh -c "while true; do wget -q -O- http://ai-api.production.svc; done"
```

## Exercises

1. Write an HPA manifest for a `text-processor` Deployment that scales between 3 and 15 replicas, targeting 70% CPU utilization, with a 2-minute scale-down stabilization window and a maximum scale-down rate of 3 pods per 60 seconds.

2. The HPA shows `<unknown>` in the `TARGETS` column. List three distinct causes and a `kubectl` command to diagnose each.

3. A Deployment has `requests.cpu: "100m"` and the HPA target is 80% utilization. At what observed CPU per pod (in millicores) will the HPA trigger a scale-up?

4. Write a Prometheus Adapter rule that exposes the metric `llm_queue_length` (a gauge, not a counter) per namespace for use in an HPA `External` metric source.

5. Explain the risk of setting `minReplicas: 0` on an HPA that serves synchronous HTTP traffic. What pattern allows zero-replica scale-to-zero safely?

(Solutions: ../solutions/week-06/day-1.md)

## Self-check

- [ ] I can explain how HPA computes desired replicas from current and target metrics.
- [ ] I can write an HPA v2 manifest with both CPU and a custom Pods metric.
- [ ] I can diagnose why an HPA shows `<unknown>` targets.
- [ ] I can configure `behavior` blocks to control scale-up/down velocity.
- [ ] I understand the difference between `Utilization`, `AverageValue`, and `Value` target types.

## Supplementary links

- [HPA documentation](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [HPA walkthrough](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale-walkthrough/)
- [Prometheus Adapter](https://github.com/kubernetes-sigs/prometheus-adapter)
- [metrics-server](https://github.com/kubernetes-sigs/metrics-server)
