# Solutions — Day 6: Writing k8s Manifests (Deep Practice)

## Exercise 1 — Sidecar container

```yaml
spec:
  containers:
    - name: api
      image: ghcr.io/myorg/fastapi-app:2.0.0
      # ... (existing spec)
      volumeMounts:
        - name: config-volume
          mountPath: /app/config
          readOnly: true

    # Fluent Bit sidecar — shares the same pod network and volume
    - name: log-collector
      image: fluent/fluent-bit:3.1
      resources:
        requests:
          cpu: "50m"
          memory: "32Mi"
        limits:
          cpu: "100m"
          memory: "64Mi"
      volumeMounts:
        - name: config-volume
          mountPath: /fluent-bit/etc
          readOnly: true
        - name: varlog
          mountPath: /var/log

  volumes:
    - name: config-volume
      configMap:
        name: fastapi-config
    - name: varlog
      emptyDir: {}
```

## Exercise 2 — All readiness probes returning 503

When all pods fail readiness simultaneously, the Endpoints controller removes all pod IPs from the Service Endpoints. The Service has `<none>` endpoints. New requests to the Service ClusterIP get connection refused (no backends). Existing in-flight requests complete normally since the pods keep running.

**No pods are restarted** — readiness failure never triggers a restart; only liveness failure does.

**Investigation commands:**
```bash
# 1. Check endpoints are empty
kubectl get endpoints fastapi-service -n ai-platform

# 2. Describe a pod to see readiness probe failures in Events
kubectl describe pod -l app=fastapi -n ai-platform

# 3. Hit the ready endpoint directly
kubectl exec <pod-name> -- wget -qO- http://localhost:8000/ready

# 4. Check logs for the dependency that failed
kubectl logs -l app=fastapi -n ai-platform --tail=50
```

**Remediation:** fix the downstream dependency (e.g., restore database connectivity). Once `/ready` returns 200, Endpoints controller re-adds pod IPs and traffic flows again. No restart needed.

## Exercise 3 — kubectl patch for maxUnavailable

```bash
kubectl patch deployment fastapi-app \
  -n ai-platform \
  --type=json \
  -p='[{"op":"replace","path":"/spec/strategy/rollingUpdate/maxUnavailable","value":1}]'
```

Or using merge patch:
```bash
kubectl patch deployment fastapi-app \
  -n ai-platform \
  --type=merge \
  -p='{"spec":{"strategy":{"rollingUpdate":{"maxUnavailable":1}}}}'
```

## Exercise 4 — PodDisruptionBudget

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: fastapi-pdb
  namespace: ai-platform
spec:
  minAvailable: 2        # at least 2 pods must be available during disruptions
  selector:
    matchLabels:
      app: fastapi
```

## Exercise 5 — NetworkPolicy

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: fastapi-ingress-policy
  namespace: ai-platform
spec:
  podSelector:
    matchLabels:
      app: fastapi
  policyTypes:
    - Ingress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: ai-platform
      ports:
        - protocol: TCP
          port: 8000
```

Note: requires a CNI plugin that enforces NetworkPolicy (e.g., Calico, Cilium). kind with the default CNI does not enforce NetworkPolicy by default — use `kind create cluster --config` with Calico for enforcement.
