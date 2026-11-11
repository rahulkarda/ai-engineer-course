# Solutions — Day 2: Pods, ReplicaSets, and Deployments

## Exercise 1 — Deployment manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fastapi-app
spec:
  replicas: 2
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
    spec:
      containers:
        - name: api
          image: ghcr.io/myorg/fastapi:2.0.0
          ports:
            - containerPort: 8000
```

## Exercise 2 — Selector mismatch

The `kubectl apply` call will succeed (the manifest is valid YAML), but the ReplicaSet will immediately create 3 pods to reach its desired count. However, it will never recognize those pods as "its own" because their labels (`app: api-service`) don't match the selector (`app: api`). The RS will create more and more pods trying to reach desired count while each created pod is also ignored by the selector. End result: infinite pod creation loop. Fix: change the pod template labels to match the selector exactly.

```yaml
selector:
  matchLabels:
    app: api
template:
  metadata:
    labels:
      app: api      # must match selector
```

## Exercise 3 — Update, watch, rollback

```bash
# (a) Update image
kubectl set image deployment/fastapi-app api=ghcr.io/myorg/fastapi:2.1.0

# (b) Watch rollout
kubectl rollout status deployment/fastapi-app

# (c) Rollback if it fails
kubectl rollout undo deployment/fastapi-app
kubectl rollout status deployment/fastapi-app
```

## Exercise 4 — Bare Pod vs Deployment

A bare Pod has no owner. If it dies or a node fails, it is gone permanently — no controller recreates it. A Deployment's pod has an OwnerReference pointing to the ReplicaSet. When the pod disappears (crash, eviction, node failure), the ReplicaSet controller detects the count is below desired and creates a replacement. Kubernetes does not reschedule bare pods because no controller watches for them.

## Exercise 5 — Node failure

When the node fails, the Node controller marks it `NotReady` after the node heartbeat times out (default ~40s + 5min grace period). The ReplicaSet controller detects that 2 pods are no longer running (they move to `Unknown` state) and creates 2 new pods on remaining healthy nodes. The **ReplicaSet controller** (running in `kube-controller-manager`) is responsible for this.
