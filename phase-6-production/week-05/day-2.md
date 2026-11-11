# Phase 6 · Week 05 · Day 2 — Pods, ReplicaSets, and Deployments

> Objective: Write Pod specs, understand why ReplicaSets exist, and manage Deployments.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

## Concept

### Pod — the atomic unit

A **Pod** is the smallest deployable unit in Kubernetes. It wraps one or more containers that:

- Share a **network namespace** (same IP, same localhost)
- Share **storage volumes**
- Are scheduled together on the same node

Most pods run **one container**. Multi-container pods are for tightly coupled helpers (sidecars, init containers).

```
┌──────────────────────────────────────┐
│  Pod  (IP: 10.244.0.5)               │
│  ┌─────────────┐  ┌──────────────┐   │
│  │  app:api    │  │ sidecar:logs │   │
│  │  port 8000  │  │              │   │
│  └─────────────┘  └──────────────┘   │
│  Volume: /data  (shared)             │
└──────────────────────────────────────┘
```

Pods are **ephemeral**. When a pod dies, it is gone — a new pod gets a new IP. You never manage pods directly in production; higher-level controllers do.

### ReplicaSet — desired count of identical pods

A **ReplicaSet** ensures N copies of a pod are always running. If a pod crashes, the ReplicaSet controller creates a replacement. If a node fails, pods are rescheduled on healthy nodes.

```
ReplicaSet (replicas: 3)
  ├── Pod api-xxx-abc   (running)
  ├── Pod api-xxx-def   (running)
  └── Pod api-xxx-ghi   (running) ← crash → RS creates api-xxx-jkl
```

ReplicaSets use **label selectors** to find their pods. Any pod matching the selector is "owned" by the RS.

### Deployment — ReplicaSet with history

A **Deployment** owns a ReplicaSet and adds:

- **Rolling updates** — swap pods gradually, no downtime
- **Rollback** — revert to any previous ReplicaSet
- **Pause/resume** — batch changes before rolling out

```
Deployment
  └── ReplicaSet (current, 3 replicas)
  └── ReplicaSet (previous, 0 replicas — kept for rollback)
```

In practice: **always use Deployments**, not bare pods or ReplicaSets.

## How it works internally

**Pod lifecycle** — API server assigns `Pending`. Scheduler finds a node, updates the pod's `nodeName`. kubelet on that node detects the assignment via watch, calls the CRI to pull the image and start the container. Pod moves `ContainerCreating` → `Running`. On success: `Succeeded` (batch) or stays `Running` (services).

**ReplicaSet controller loop** — watches pods with its selector. Counts running pods. If count < desired: creates pods. If count > desired: deletes pods. This loop runs every time a relevant event fires (pod deleted, new pod appears).

**Deployment rolling update** — when you change the image, the Deployment controller creates a new ReplicaSet with the new spec. It scales up the new RS one pod at a time (controlled by `maxSurge`) and scales down the old RS one pod at a time (controlled by `maxUnavailable`). This continues until old RS = 0 and new RS = desired.

**Revision history** — each rollout creates a new RS. Old RSes are kept (scaled to 0) up to `revisionHistoryLimit` (default 10). `kubectl rollout undo` just flips the pointer.

## Worked example

### Minimal Pod manifest

```yaml
# pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: fastapi-pod
  labels:
    app: fastapi
spec:
  containers:
    - name: api
      image: python:3.12-slim
      command: ["python", "-m", "http.server", "8000"]
      ports:
        - containerPort: 8000
      resources:
        requests:
          cpu: "100m"
          memory: "128Mi"
        limits:
          cpu: "500m"
          memory: "256Mi"
```

```bash
kubectl apply -f pod.yaml
kubectl get pod fastapi-pod
kubectl describe pod fastapi-pod
kubectl logs fastapi-pod
kubectl exec -it fastapi-pod -- /bin/bash
kubectl delete pod fastapi-pod
```

### Deployment manifest (production pattern)

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fastapi-app
  labels:
    app: fastapi
    version: "1.0"
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fastapi        # must match template labels
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1         # at most 1 extra pod during rollout
      maxUnavailable: 0   # never drop below desired count
  template:               # pod template — everything below is a Pod spec
    metadata:
      labels:
        app: fastapi       # must match selector above
        version: "1.0"
    spec:
      containers:
        - name: api
          image: ghcr.io/myorg/fastapi-app:1.0.0
          ports:
            - containerPort: 8000
          env:
            - name: APP_ENV
              value: production
          resources:
            requests:
              cpu: "100m"
              memory: "128Mi"
            limits:
              cpu: "500m"
              memory: "256Mi"
```

```bash
kubectl apply -f deployment.yaml

# Watch rollout progress
kubectl rollout status deployment/fastapi-app

# Inspect the ReplicaSet the Deployment created
kubectl get replicaset
kubectl describe replicaset <rs-name>

# Scale up/down
kubectl scale deployment/fastapi-app --replicas=5
```

### Triggering and observing a rolling update

```bash
# Update the image — triggers a new rollout
kubectl set image deployment/fastapi-app api=ghcr.io/myorg/fastapi-app:1.1.0

# Or edit the YAML and re-apply
kubectl apply -f deployment-v2.yaml

# Watch pods cycling
kubectl get pods -w

# See rollout history
kubectl rollout history deployment/fastapi-app

# Show detail of a specific revision
kubectl rollout history deployment/fastapi-app --revision=2

# Rollback to previous version
kubectl rollout undo deployment/fastapi-app

# Rollback to a specific revision
kubectl rollout undo deployment/fastapi-app --to-revision=1
```

### Init containers

Init containers run **before** main containers, sequentially. Use them to wait for a dependency or pre-populate a volume.

```yaml
spec:
  initContainers:
    - name: wait-for-db
      image: busybox:1.36
      command:
        - sh
        - -c
        - |
          until nc -z postgres-service 5432; do
            echo "waiting for postgres..."
            sleep 2
          done
  containers:
    - name: api
      image: ghcr.io/myorg/fastapi-app:1.0.0
```

## Exercises

1. Write a Deployment manifest for a FastAPI app (`ghcr.io/myorg/fastapi:2.0.0`, 2 replicas, port 8000). Include `RollingUpdate` strategy with `maxSurge: 1, maxUnavailable: 0`.

2. A ReplicaSet has `selector.matchLabels: {app: api}` but the pod template has `labels: {app: api-service}`. What happens when you apply it, and why?

3. Write the kubectl commands to: (a) update the image to version `2.1.0`, (b) watch the rollout, (c) rollback if it fails.

4. What is the difference between a bare Pod and a Pod managed by a Deployment? Why does k8s reschedule Deployment pods but not bare pods?

5. Your Deployment has 3 replicas. One node fails. What happens to the 2 pods that were on that node? Which component detects this and acts?

(Solutions: ../solutions/week-05/day-2.md)

## Self-check

- [ ] I can write a Pod manifest from memory
- [ ] I can write a Deployment manifest with correct label selectors
- [ ] I can trigger a rolling update and rollback
- [ ] I understand why ReplicaSet selector and pod template labels must match
- [ ] I can explain the Deployment → ReplicaSet → Pod ownership chain

## Supplementary links

- [Pods — concepts](https://kubernetes.io/docs/concepts/workloads/pods/)
- [Deployments — concepts](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
- [kubectl rollout](https://kubernetes.io/docs/reference/generated/kubectl/kubectl-commands#rollout)
