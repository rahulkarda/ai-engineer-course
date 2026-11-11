# Phase 6 · Week 05 · Day 6 — Writing k8s Manifests: Deploy the FastAPI App (4 hr)

> Objective: Write a complete, production-ready set of k8s manifests for the week-4 containerized FastAPI app and deploy it to a local kind cluster.  ·  Time: ~4 hr  ·  Prereqs: [[day-1]], [[day-2]], [[day-3]], [[day-4]], [[day-5]]

## Objective

Take the FastAPI app containerized in week 4, write all required Kubernetes manifests, deploy to a local kind cluster, verify it works, and perform a rolling update + rollback.

**Deliverables by end of session:**
- [ ] Namespace, ConfigMap, Secret, Deployment, Service manifests written
- [ ] All manifests validated with `--dry-run=server`
- [ ] App running in kind cluster with 3 replicas
- [ ] Readiness probe passing before traffic routes
- [ ] Rolling update from v1 to v2 observed
- [ ] Rollback to v1 confirmed

## Part 1 — Set up the local cluster (30 min)

### Install kind (Kubernetes in Docker)

```bash
# macOS
brew install kind kubectl

# Verify
kind version
kubectl version --client
```

### kind cluster config

```yaml
# kind-config.yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: ai-platform
nodes:
  - role: control-plane
  - role: worker
  - role: worker
```

```bash
kind create cluster --config kind-config.yaml
kubectl config use-context kind-ai-platform
kubectl get nodes
# Expect: control-plane + 2 workers in Ready state
```

### Load local images into kind

```bash
# If you built images locally in week 4:
kind load docker-image ghcr.io/myorg/fastapi-app:1.0.0 --name ai-platform
kind load docker-image ghcr.io/myorg/fastapi-app:2.0.0 --name ai-platform

# Or pull a public placeholder
docker pull python:3.12-slim
kind load docker-image python:3.12-slim --name ai-platform
```

## Part 2 — Project layout (15 min)

Organize all manifests under a `k8s/` directory:

```
k8s/
├── namespace.yaml
├── configmap.yaml
├── secret.yaml
├── deployment.yaml
├── service.yaml
└── hpa.yaml
```

## Part 3 — Write the manifests (90 min)

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
  # Mounted as a file at /app/config/settings.toml
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

### secret.yaml

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: fastapi-secrets
  namespace: ai-platform
type: Opaque
data:
  # Placeholder values — replace with real base64-encoded values at deploy time
  # Use: echo -n "$VALUE" | base64
  # Or: kubectl create secret generic fastapi-secrets --from-env-file=.env.production
  DATABASE_URL: <base64-of-your-database-url>
  JWT_SECRET_KEY: <base64-of-your-jwt-secret>
```

> Note: In a real deployment, do NOT commit real Secret values. Use `kubectl create secret` from environment variables, or use External Secrets Operator pointing at Vault/AWS Secrets Manager.

```bash
# Create the secret imperatively (preferred for real values)
kubectl create secret generic fastapi-secrets \
  --from-literal=DATABASE_URL="${DATABASE_URL}" \
  --from-literal=JWT_SECRET_KEY="${JWT_SECRET_KEY}" \
  --namespace=ai-platform \
  --dry-run=client -o yaml > k8s/secret-generated.yaml

# Apply it
kubectl apply -f k8s/secret-generated.yaml
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
    deployment.kubernetes.io/change-cause: "initial deployment v1.0"
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
      # Graceful shutdown: wait for in-flight requests to complete
      terminationGracePeriodSeconds: 30

      containers:
        - name: api
          image: ghcr.io/myorg/fastapi-app:1.0.0
          imagePullPolicy: IfNotPresent
          ports:
            - name: http
              containerPort: 8000

          # Config injection
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

          # Config file mount
          volumeMounts:
            - name: config-volume
              mountPath: /app/config
              readOnly: true

          # Resources — Burstable QoS
          resources:
            requests:
              cpu: "100m"
              memory: "128Mi"
            limits:
              cpu: "500m"
              memory: "256Mi"

          # Startup probe: app has 60s to initialize
          startupProbe:
            httpGet:
              path: /health
              port: 8000
            failureThreshold: 30
            periodSeconds: 2
            timeoutSeconds: 2

          # Liveness probe: restart if stuck
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            periodSeconds: 10
            timeoutSeconds: 3
            failureThreshold: 3
            successThreshold: 1

          # Readiness probe: gate traffic
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
      targetPort: http    # refers to the named port in the pod spec
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
      stabilizationWindowSeconds: 300   # wait 5 min before scaling down
```

## Part 4 — Apply and verify (45 min)

```bash
# Validate all manifests before applying
kubectl apply -f k8s/ --dry-run=server

# Apply in dependency order
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret-generated.yaml  # or secret.yaml with placeholders
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# Watch rollout
kubectl rollout status deployment/fastapi-app -n ai-platform

# Verify all pods are running
kubectl get pods -n ai-platform -w

# Check endpoints (should show pod IPs)
kubectl get endpoints fastapi-service -n ai-platform

# Port-forward to test locally
kubectl port-forward service/fastapi-service 8080:80 -n ai-platform &
curl http://localhost:8080/health
curl http://localhost:8080/ready

# Check logs
kubectl logs -l app=fastapi -n ai-platform --tail=20

# Describe a pod to see probe status
kubectl describe pod -l app=fastapi -n ai-platform
```

## Part 5 — Rolling update (30 min)

```bash
# Tag the current deployment for history
kubectl annotate deployment/fastapi-app \
  kubernetes.io/change-cause="v1.0 initial release" \
  -n ai-platform

# Trigger update to v2
kubectl set image deployment/fastapi-app \
  api=ghcr.io/myorg/fastapi-app:2.0.0 \
  -n ai-platform

kubectl annotate deployment/fastapi-app \
  kubernetes.io/change-cause="v2.0 updated endpoint handling" \
  -n ai-platform

# Watch the rolling update happen
kubectl rollout status deployment/fastapi-app -n ai-platform
kubectl get pods -n ai-platform -w

# Inspect history
kubectl rollout history deployment/fastapi-app -n ai-platform

# Rollback
kubectl rollout undo deployment/fastapi-app -n ai-platform
kubectl rollout status deployment/fastapi-app -n ai-platform

# Verify back on v1
kubectl get pods -n ai-platform -o jsonpath='{.items[0].spec.containers[0].image}'
```

## Part 6 — Cleanup (15 min)

```bash
# Delete all resources in namespace
kubectl delete namespace ai-platform

# Delete the kind cluster
kind delete cluster --name ai-platform
```

## Exercises

1. Add a second container (sidecar) to the Deployment pod template — a `fluent-bit` log collector that mounts the same `config-volume`. Write the sidecar spec with its own resource limits.

2. The `ready` endpoint starts returning 503 for all pods simultaneously (downstream dependency down). Describe exactly what happens to traffic and explain how you would investigate and remediate.

3. Write a `kubectl patch` command to update `maxUnavailable` from 0 to 1 in the Deployment strategy without editing the YAML file.

4. Add a `PodDisruptionBudget` manifest that ensures at least 2 pods are always available during voluntary disruptions (node drains, cluster upgrades).

5. Write a `NetworkPolicy` that restricts ingress to the `fastapi-app` pods to only traffic from pods in the `ai-platform` namespace.

(Solutions: ../solutions/week-05/day-6.md)

## Self-check

- [ ] I can write all six manifest types from memory
- [ ] I can apply manifests in correct dependency order
- [ ] I can verify a deployment is healthy using get/describe/logs/endpoints
- [ ] I performed a rolling update and observed pods cycling
- [ ] I can rollback to a previous revision with kubectl

## Supplementary links

- [kind — quick start](https://kind.sigs.k8s.io/docs/user/quick-start/)
- [kubectl apply](https://kubernetes.io/docs/reference/generated/kubectl/kubectl-commands#apply)
- [PodDisruptionBudget](https://kubernetes.io/docs/tasks/run-application/configure-pdb/)
- [NetworkPolicy](https://kubernetes.io/docs/concepts/services-networking/network-policies/)
