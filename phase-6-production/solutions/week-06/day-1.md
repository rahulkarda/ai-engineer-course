# Solutions — Week 06 · Day 1 (HPA & Autoscaling)

## Exercise 1 — HPA for text-processor

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: text-processor-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: text-processor
  minReplicas: 3
  maxReplicas: 15
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 120   # 2 minutes
      policies:
        - type: Pods
          value: 3
          periodSeconds: 60             # max 3 pods removed per 60 s
```

## Exercise 2 — `<unknown>` TARGETS: three causes and diagnosis

| Cause | kubectl command |
|---|---|
| metrics-server not installed or unhealthy | `kubectl get apiservice v1beta1.metrics.k8s.io` — should be `True`; if not: `kubectl get pods -n kube-system -l k8s-app=metrics-server` |
| Resource `requests` not set on pods | `kubectl get pod <pod> -o jsonpath='{.spec.containers[0].resources.requests}'` — must be non-empty |
| HPA controller cannot reach custom metrics API | `kubectl get apiservice v1beta2.custom.metrics.k8s.io` — for custom metric HPAs |

## Exercise 3 — Scale-up trigger threshold

Formula: `desiredReplicas = ceil(current × (observed/target))`

Trigger condition: `observed > target × 1.1` (10% tolerance applied by HPA)

```
request = 100m
target  = 80% of request = 80m
trigger = 80m × 1.1 = 88m
```

At **88 millicores** observed CPU per pod, the HPA will initiate a scale-up.

## Exercise 4 — Prometheus Adapter rule for gauge metric

```yaml
# In Prometheus Adapter ConfigMap
rules:
  - seriesQuery: 'llm_queue_length{namespace!=""}'
    resources:
      overrides:
        namespace: {resource: "namespace"}
    name:
      matches: "^llm_queue_length$"
      as: "llm_queue_length"
    metricsQuery: 'llm_queue_length{<<.LabelMatchers>>}'
```

HPA External metric reference:

```yaml
metrics:
  - type: External
    external:
      metric:
        name: llm_queue_length
        selector:
          matchLabels:
            namespace: production
      target:
        type: AverageValue
        averageValue: "100"
```

Note: gauges do not use `rate()` — query the gauge directly. The `resources.overrides` maps the Prometheus `namespace` label to the Kubernetes namespace resource so the adapter scopes queries correctly.

## Exercise 5 — Risk of minReplicas: 0 and zero-scale alternatives

**Risk:** With `minReplicas: 0`, the HPA can scale down to 0 pods. The first HTTP request after scale-to-zero has no pod to handle it — the Ingress/Service sees 503 for the entire cold-start period (seconds to minutes depending on image pull time + container startup). This is unacceptable for synchronous HTTP traffic where clients do not retry automatically.

**Pattern for safe zero-scale:** Use **KEDA (Kubernetes Event-Driven Autoscaling)**. KEDA extends HPA with a ScaledObject CRD. It:
1. Scales to 0 when a queue/topic is empty (no events to process).
2. Wakes up 1 pod before routing the first event (using its own event-driven trigger, not HTTP).
3. Scales based on event backlog, not CPU.

This works for asynchronous workloads (queue consumers, batch processors) because the event already exists in the queue before the pod needs to exist. Synchronous HTTP callers cannot be buffered this way.
