# Solutions — Week 05 Project

## Complete manifest set

All manifests below assume the `ai-platform` namespace. Apply in the order shown.

### kind-config.yaml

```yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: ai-platform
nodes:
  - role: control-plane
  - role: worker
  - role: worker
```

### namespace.yaml

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ai-platform
  labels:
    env: production
    team: ml-infra
    managed-by: kubectl
```

### configmap.yaml

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fastapi-config
  namespace: ai-platform
data:
  APP_ENV: production
  LOG_LEVEL: INFO
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  WORKERS: "2"
  settings.toml: |
    [server]
    host = "0.0.0.0"
    port = 8000
    workers = 2

    [logging]
    level = "INFO"
    format = "json"

    [features]
    enable_metrics = true
    enable_tracing = false
```

### secret.yaml (placeholder — do NOT commit real values)

```yaml
# Create imperatively:
#   kubectl create secret generic fastapi-secrets \
#     --from-literal=DATABASE_URL="${DATABASE_URL}" \
#     --from-literal=JWT_SECRET_KEY="${JWT_SECRET_KEY}" \
#     --namespace=ai-platform
apiVersion: v1
kind: Secret
metadata:
  name: fastapi-secrets
  namespace: ai-platform
type: Opaque
data:
  DATABASE_URL: <base64-of-database-url>
  JWT_SECRET_KEY: <base64-of-jwt-secret>
```

### deployment.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fastapi-app
  namespace: ai-platform
  labels:
    app: fastapi
    version: "1.0"
  annotations:
    kubernetes.io/change-cause: "v1.0 initial deployment"
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
        version: "1.0"
        tier: backend
    spec:
      terminationGracePeriodSeconds: 30
      containers:
        - name: api
          image: ghcr.io/myorg/fastapi-app:1.0.0
          imagePullPolicy: IfNotPresent
          ports:
            - name: http
              containerPort: 8000
          envFrom:
            - configMapRef:
                name: fastapi-config
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: fastapi-secrets
                  key: DATABASE_URL
            - name: JWT_SECRET_KEY
              valueFrom:
                secretKeyRef:
                  name: fastapi-secrets
                  key: JWT_SECRET_KEY
          volumeMounts:
            - name: config-volume
              mountPath: /app/config
              readOnly: true
          resources:
            requests:
              cpu: "100m"
              memory: "128Mi"
            limits:
              cpu: "500m"
              memory: "256Mi"
          startupProbe:
            httpGet:
              path: /health
              port: 8000
            failureThreshold: 30
            periodSeconds: 2
            timeoutSeconds: 2
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            periodSeconds: 10
            timeoutSeconds: 3
            failureThreshold: 3
            successThreshold: 1
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 3
            successThreshold: 1
      volumes:
        - name: config-volume
          configMap:
            name: fastapi-config
            items:
              - key: settings.toml
                path: settings.toml
```

### service.yaml

```yaml
apiVersion: v1
kind: Service
metadata:
  name: fastapi-service
  namespace: ai-platform
  labels:
    app: fastapi
spec:
  selector:
    app: fastapi
  ports:
    - name: http
      port: 80
      targetPort: http
      protocol: TCP
  type: ClusterIP
```

### hpa.yaml

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fastapi-hpa
  namespace: ai-platform
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
          averageUtilization: 70
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
```

---

## Operations Runbook

```markdown
# FastAPI App — Kubernetes Operations Runbook

## Cluster info
- Cluster: ai-platform (kind, local)
- Namespace: ai-platform
- Context: kind-ai-platform

## First-time setup
\```bash
kind create cluster --config kind-config.yaml
kubectl config use-context kind-ai-platform
kind load docker-image ghcr.io/myorg/fastapi-app:1.0.0 --name ai-platform
kubectl get nodes   # verify 3 nodes in Ready state
\```

## Deploy

\```bash
kubectl apply -f k8s/namespace.yaml
kubectl create secret generic fastapi-secrets \
  --from-literal=DATABASE_URL="${DATABASE_URL}" \
  --from-literal=JWT_SECRET_KEY="${JWT_SECRET_KEY}" \
  --namespace=ai-platform
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl rollout status deployment/fastapi-app -n ai-platform
\```

## Rolling update
\```bash
kind load docker-image ghcr.io/myorg/fastapi-app:2.0.0 --name ai-platform
kubectl set image deployment/fastapi-app \
  api=ghcr.io/myorg/fastapi-app:2.0.0 -n ai-platform
kubectl rollout status deployment/fastapi-app -n ai-platform
\```

## Rollback
\```bash
kubectl rollout history deployment/fastapi-app -n ai-platform
kubectl rollout undo deployment/fastapi-app -n ai-platform
kubectl rollout status deployment/fastapi-app -n ai-platform
\```

## Health checks
\```bash
# Quick status
kubectl get all -n ai-platform
kubectl get endpoints fastapi-service -n ai-platform

# Logs
kubectl logs -l app=fastapi -n ai-platform --tail=50
kubectl logs <pod-name> -n ai-platform --previous  # after crash

# Local test
kubectl port-forward service/fastapi-service 8080:80 -n ai-platform &
curl http://localhost:8080/health
curl http://localhost:8080/ready
\```

## Troubleshooting

| Symptom | Command | Likely cause |
|---------|---------|--------------|
| Pod Pending | `kubectl describe pod <name> -n ai-platform` | Insufficient resources |
| CrashLoopBackOff | `kubectl logs <name> --previous -n ai-platform` | App crash at startup |
| No traffic reaching pods | `kubectl get endpoints fastapi-service -n ai-platform` | Readiness probe failing |
| Rollout stuck | `kubectl rollout status deployment/fastapi-app -n ai-platform` | New pods not becoming Ready |
| Secret mount error | `kubectl describe pod <name> -n ai-platform` | Secret not created or wrong namespace |

## Cleanup
\```bash
kubectl delete namespace ai-platform
kind delete cluster --name ai-platform
\```
```

---

## Extension task solutions

### PodDisruptionBudget

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: fastapi-pdb
  namespace: ai-platform
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: fastapi
```

### NetworkPolicy (namespace-internal only)

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: fastapi-network-policy
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

### LimitRange

```yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: ai-platform-defaults
  namespace: ai-platform
spec:
  limits:
    - type: Container
      defaultRequest:
        cpu: "100m"
        memory: "128Mi"
      default:
        cpu: "500m"
        memory: "256Mi"
```
