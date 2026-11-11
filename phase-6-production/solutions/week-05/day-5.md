# Solutions — Day 5: Probes, Resource Limits, and Rolling Updates

## Exercise 1 — Liveness probe

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 0
  periodSeconds: 10
  timeoutSeconds: 3
  failureThreshold: 3
  successThreshold: 1
```

## Exercise 2 — Liveness vs readiness failure consequences

**Liveness probe failure:** kubelet counts consecutive failures. After `failureThreshold` failures, kubelet calls the container runtime to **restart** the container (stops and starts it). The pod remains on the same node; only the container restarts. The restart count increments.

**Readiness probe failure:** kubelet removes the pod's IP from the Service Endpoints object. The pod receives **no new traffic** but keeps running. No restart. Traffic is restored when the probe starts passing again. Use for temporary degradation (high load, warming up) where a restart would not help.

## Exercise 3 — Memory limit exceeded

The container's memory usage grows to 300Mi, exceeding the 256Mi limit. Linux cgroups immediately **OOM-kills** the container process. kubelet detects the exit and restarts the container (because the pod restart policy defaults to `Always`). The restart count increments. Unlike CPU throttling (gradual slowdown), memory limit violation is an immediate hard kill.

## Exercise 4 — Rolling update with 4 replicas, maxSurge=1, maxUnavailable=0

Step-by-step (desired=4, maxSurge=1 means max 5 total, maxUnavailable=0 means min 4 must be ready):

1. Start: 4 old pods running
2. Create 1 new pod → 4 old + 1 new = 5 total (at maxSurge limit)
3. New pod becomes Ready → 4 old + 1 new ready = 5 ready; terminate 1 old → 3 old + 1 new = 4
4. Create 1 new pod → 3 old + 2 new = 5; new pod Ready → terminate 1 old → 2 old + 2 new = 4
5. Create 1 new pod → 2 old + 3 new = 5; new pod Ready → terminate 1 old → 1 old + 3 new = 4
6. Create 1 new pod → 1 old + 4 new = 5; new pod Ready → terminate last old → 0 old + 4 new = 4
7. Done: 4 new pods

Total pods running at peak: 5 (always 1 extra). Available traffic: never below 4.

## Exercise 5 — HPA manifest and prerequisites

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-server-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-server
  minReplicas: 1
  maxReplicas: 8
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 60
```

**Prerequisites for HPA to work:**
1. **metrics-server** must be installed in the cluster (provides CPU/memory metrics)
2. All containers in the Deployment must have **`resources.requests.cpu` set** — HPA computes utilization as `actual_cpu / request_cpu`; without a request, the calculation is impossible and HPA will not scale
