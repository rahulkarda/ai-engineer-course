# Solutions — Day 7: Week Project Runbook & Assessment Prep

## Exercise 1 — Why not commit Secret YAML with real values

Committing base64-encoded Secret values to git permanently exposes them in version history — even if deleted later, old commits retain the values and anyone with repo access can decode them with a single `base64 -d` command. Instead, create Secrets imperatively at deploy time using `kubectl create secret generic --from-literal` with values sourced from environment variables (set by your CI/CD system from a secure store). The YAML file in the repo should contain only placeholder comments, never real values.

## Exercise 2 — Stuck rollout diagnosis

```bash
# 1. Check overall rollout status
kubectl rollout status deployment/fastapi-app -n ai-platform

# 2. See how many pods are in each state
kubectl get pods -n ai-platform

# 3. Describe the stuck new pod — look at Events and Conditions
kubectl describe pod <new-pod-name> -n ai-platform

# 4. Check logs of the new pod
kubectl logs <new-pod-name> -n ai-platform

# 5. Check if the readiness probe is failing
# (Events will show: Readiness probe failed)

# 6. Check the old pod — why isn't it terminating?
kubectl describe pod <old-pod-name> -n ai-platform
# If it has status Terminating, check for finalizers or long shutdown hooks
```

If the rollout is stuck because the new pods can't become Ready (bad image, failing probe), rollback:
```bash
kubectl rollout undo deployment/fastapi-app -n ai-platform
```

## Exercise 3 — Dry-run diff of all files

```bash
kubectl diff -f k8s/ -n ai-platform
```

Or to validate all without diffing:
```bash
kubectl apply -f k8s/ --dry-run=server -n ai-platform
```

## Exercise 4 — LimitRange

```yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: ai-platform-defaults
  namespace: ai-platform
spec:
  limits:
    - type: Container
      default:          # default limit if not specified
        cpu: "500m"
        memory: "256Mi"
      defaultRequest:   # default request if not specified
        cpu: "100m"
        memory: "128Mi"
      max:              # hard max per container
        cpu: "2"
        memory: "2Gi"
```

## Exercise 5 — kubectl delete namespace step-by-step

1. API server receives the `DELETE` request for the `ai-platform` namespace
2. Namespace moves to `Terminating` state (not yet deleted)
3. The **namespace controller** (in `kube-controller-manager`) starts garbage collection:
   - Deletes all objects in the namespace in order: Pods → ReplicaSets → Deployments → Services → ConfigMaps → Secrets → etc.
   - For each workload object, the GC controller follows OwnerReferences to cascade deletions
4. Pods receive `SIGTERM` and have `terminationGracePeriodSeconds` to shut down; kubelet sends `SIGKILL` if they don't exit
5. Once all objects are deleted and all finalizers are removed, the namespace object itself is deleted from etcd
6. The namespace disappears from `kubectl get namespaces`

The `kube-controller-manager` (namespace controller) drives this entire cleanup. External resources (cloud load balancers, PVs with `Retain` policy) may NOT be deleted automatically.
