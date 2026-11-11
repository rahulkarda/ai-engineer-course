# Solutions — Day 1: Kubernetes Architecture & kubectl

## Exercise 1 — Control plane diagram and failure impact

```
┌─────────────────────────────────────────┐
│             Control Plane               │
│                                         │
│  kube-apiserver  ←──  all traffic       │
│       │                                 │
│  etcd            ←──  only source of    │
│       │                truth            │
│  kube-scheduler  ←──  pod placement     │
│       │                                 │
│  kube-controller-manager ← loops       │
└─────────────────────────────────────────┘
```

- **kube-apiserver** goes down → cluster writes stop; existing workloads keep running but no changes possible
- **kube-scheduler** goes down → new pods stay Pending; existing pods unaffected

## Exercise 2 — etcd as source of truth

etcd is the only durable store for all cluster state (objects, configs, secrets). Every component reads from and writes to it exclusively via the API server. If etcd data is lost without a backup, the cluster loses all knowledge of what was running — you cannot restore workloads, configs, or history, and the cluster is effectively blank.

## Exercise 3 — List running pods across all namespaces

```bash
kubectl get pods --all-namespaces | grep Running | wc -l

# Or with -A shorthand
kubectl get pods -A --field-selector=status.phase=Running
```

## Exercise 4 — Full YAML of a pod

```bash
kubectl get pod api-server -n production -o yaml
```

## Exercise 5 — Pod stuck Pending

**Component responsible:** `kube-scheduler`

**Possible causes:**
- Insufficient CPU or memory on all nodes (requests cannot be satisfied)
- Node selector / affinity rules that no node satisfies
- Taints on all nodes with no matching toleration in the pod spec
- PersistentVolumeClaim not bound (volume unavailable)

**Diagnose:**
```bash
kubectl describe pod <pod-name>
# Look at the Events section at the bottom — scheduler logs why it couldn't place the pod
```
