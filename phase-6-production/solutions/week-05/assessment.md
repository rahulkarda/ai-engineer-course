# Solutions — Week 05 Assessment

## Conceptual answers

**1. etcd role**

etcd is a distributed key-value store that is the single source of truth for all cluster state — every object (pods, services, configs, secrets, RBAC rules) is stored there. All reads and writes go through the API server, which persists changes to etcd. If etcd data is lost without a backup, the cluster loses all state: it has no record of what workloads were running, how they were configured, or what resources existed, and cannot be recovered without a restore.

**2. Liveness vs readiness probe**

A liveness probe tests whether the container is alive and functional. Failure causes kubelet to **restart the container**. A readiness probe tests whether the container is ready to handle traffic. Failure causes the pod IP to be **removed from Service Endpoints** — traffic stops, but the container keeps running. Liveness repairs broken processes; readiness gates traffic during initialization or temporary overload.

**3. Pending pod — three causes and diagnosis**

```bash
kubectl describe pod <pod-name>
# Read the Events section at the bottom
```

Three root causes:
- **Insufficient resources:** no node has enough free CPU/memory to satisfy the pod's requests → Events: `0/N nodes are available: N Insufficient cpu`
- **No matching nodes:** nodeSelector, nodeAffinity, or taints prevent placement → Events: `0/N nodes are available: N node(s) had untolerated taint`
- **PVC not bound:** a required PersistentVolumeClaim is in Pending state → Events: `persistentvolumeclaim "my-pvc" not found` or `pod has unbound immediate PersistentVolumeClaims`

**4. QoS classes**

| Class | Condition |
|-------|-----------|
| Guaranteed | requests == limits for ALL containers |
| Burstable | at least one container has a request set |
| BestEffort | no requests or limits at all |

With `requests.cpu=100m, limits.cpu=500m` (and no memory spec): the pod is **Burstable**. It is evicted after BestEffort pods but before Guaranteed pods when the node is under memory pressure.

**5. Service with empty endpoints — two causes**

1. **Label selector mismatch:** the Service selector `app: api` does not match any pod's actual labels (e.g., pods are labeled `app: api-v2`)
2. **No pods passing readiness probe:** pods exist and match the selector, but all are failing their readiness probe — the Endpoints controller only adds pods that are Ready

Diagnose: `kubectl get pods --show-labels` and compare with `kubectl describe service my-service | grep Selector`.

**6. k8s Secrets not truly secure**

By default, Secrets are stored **base64-encoded** in etcd — encoding, not encryption. Anyone with etcd access or sufficient RBAC permissions (`get` on Secrets) can decode them instantly. Cluster backups also contain all Secret values.

Production alternatives: **Encryption at Rest** (AES-GCM key in API server config encrypts etcd data); **External Secrets Operator** (syncs from Vault, AWS Secrets Manager, Azure Key Vault); **Secrets Store CSI Driver** (mounts external vault secrets as pod files); strict RBAC restricting Secret read access.

**7. Reconciliation loop in a rolling update**

The reconciliation loop continuously compares desired state (stored in etcd) with actual state and drives toward desired. During a rolling update, the Deployment controller creates a new ReplicaSet with the updated pod spec. On each loop iteration: it checks current pod counts, determines the gap from desired (accounting for maxSurge/maxUnavailable), and creates or deletes pods accordingly. It repeats until new RS = desired count and old RS = 0. The loop is event-driven: each pod state change triggers a new reconcile.

**8. terminationGracePeriodSeconds**

When a pod is deleted, k8s sends `SIGTERM` to the container processes and waits up to `terminationGracePeriodSeconds` (default 30) before sending `SIGKILL`. During this window, a well-coded app can finish in-flight requests, flush buffers, close DB connections, and deregister from service discovery. The pod is also removed from Endpoints immediately on deletion (before SIGTERM), so no new traffic arrives. Without a grace period, long-running requests would be abruptly cut mid-flight during rolling updates, causing client errors.

---

## Coding challenge — Reference solution

```yaml
# 1. namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: inference
  labels:
    env: production
    team: ml-infra

---
# 2. configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-config
  namespace: inference
data:
  MODEL_NAME: claude-sonnet-5-5
  inference.json: |
    {
      "model_name": "claude-sonnet-5-5",
      "max_tokens": 8192,
      "temperature": 0.0,
      "timeout_seconds": 120
    }

---
# 3. secret.yaml
# IMPORTANT: Do NOT commit real values.
# Create with: kubectl create secret generic model-secrets \
#   --from-literal=API_KEY="${API_KEY}" \
#   --namespace=inference
apiVersion: v1
kind: Secret
metadata:
  name: model-secrets
  namespace: inference
type: Opaque
data:
  API_KEY: <base64-of-api-key>

---
# 4. deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server
  namespace: inference
  labels:
    app: model-server
spec:
  replicas: 2
  selector:
    matchLabels:
      app: model-server
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0          # never drop below 2
  template:
    metadata:
      labels:
        app: model-server
    spec:
      terminationGracePeriodSeconds: 30
      containers:
        - name: server
          image: ghcr.io/myorg/model-server:3.0.0
          ports:
            - name: http
              containerPort: 9000
          envFrom:
            - configMapRef:
                name: model-config
          env:
            - name: API_KEY
              valueFrom:
                secretKeyRef:
                  name: model-secrets
                  key: API_KEY
          resources:
            requests:
              cpu: "500m"
              memory: "512Mi"
            limits:
              cpu: "2"
              memory: "2Gi"
          # 30 retries × 2s = 60s window for model loading
          startupProbe:
            httpGet:
              path: /health
              port: 9000
            failureThreshold: 30
            periodSeconds: 2
            timeoutSeconds: 2
          livenessProbe:
            httpGet:
              path: /health
              port: 9000
            periodSeconds: 10
            timeoutSeconds: 3
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /ready
              port: 9000
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 3

---
# 5. service.yaml
apiVersion: v1
kind: Service
metadata:
  name: model-server
  namespace: inference
spec:
  selector:
    app: model-server
  ports:
    - name: http
      port: 80
      targetPort: http
  type: ClusterIP

---
# 6. hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-server-hpa
  namespace: inference
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-server
  minReplicas: 2
  maxReplicas: 6
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 65
```

### Why this solution satisfies the criteria

- **Selectors match:** Deployment `matchLabels: app: model-server`, pod template `labels: app: model-server`, Service `selector: app: model-server` — all three consistent
- **Startup probe gives 60s:** 30 failures × 2s period = 60s before liveness probe can restart
- **Resources match spec:** requests = guaranteed values (500m/512Mi), limits = max (2/2Gi)
- **HPA bounds:** min=2 matches replicas, max=6 as specified
- **No real secrets:** placeholder comment with imperative creation instruction
