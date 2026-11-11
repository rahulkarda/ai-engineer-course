# Phase 6 · Week 05 · Day 4 — ConfigMaps, Secrets, Labels, and Selectors

> Objective: Externalize config with ConfigMaps and Secrets; master labels and selectors for targeting resources.  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

## Concept

### ConfigMap — external configuration

A **ConfigMap** stores non-sensitive configuration data as key-value pairs. It decouples config from container images so you can update configuration without rebuilding images.

Consumption modes:
- **Environment variables** — each key becomes an env var
- **Volume mount** — keys become files in a directory
- **Single env var** from a specific key

### Secret — sensitive configuration

A **Secret** is structurally identical to a ConfigMap but:
- Values are **base64-encoded** (NOT encrypted by default — just encoded)
- kubectl masks Secret values in `describe` output
- Can be encrypted at rest with the Encryption Configuration feature
- Should be paired with RBAC to restrict who can read them

In production, use an external secret store (Vault, AWS Secrets Manager, Azure Key Vault) with a Secrets Store CSI Driver or External Secrets Operator. Base64 in etcd is not production-grade security.

### Labels and selectors

A **label** is a key-value pair attached to any k8s object:

```yaml
labels:
  app: fastapi
  version: "2.0"
  env: production
  tier: backend
```

A **selector** targets objects by their labels. Two types:

| Type | Syntax | Used by |
|------|--------|---------|
| Equality | `app=fastapi` | Services, kubectl |
| Set-based | `env in (prod, staging)` | ReplicaSets, Deployments, Jobs |

Labels are the glue that connects Deployments → ReplicaSets → Pods → Services. **Get them wrong and nothing connects.**

### Annotations

Like labels but for tool metadata (not for selection). Examples: Prometheus scrape config, deployment timestamps, Helm chart metadata. Not indexable by selectors.

## How it works internally

**ConfigMap projection** — when a ConfigMap is mounted as a volume, kubelet writes each key as a file under the mount path. When the ConfigMap is updated, kubelet detects the change (via watch) and updates the files on the filesystem. For env vars, the values are read at pod start and never updated until the pod restarts.

**Secret storage** — by default, Secrets are stored base64-encoded in etcd, just like ConfigMaps. The difference is API-level: the Secret API object type is treated with more care by `kubectl describe` (values shown as `<redacted>`), and RBAC policies typically restrict `get`/`list` on Secrets more tightly than ConfigMaps.

**Label index** — the API server maintains an in-memory label index for fast selector queries. This is why selectors are efficient even on clusters with thousands of pods.

**Selector ownership and OwnerReferences** — when a ReplicaSet creates a pod, it adds an `ownerReference` to the pod pointing back to the RS. If you delete the pod directly, the RS controller sees it's gone (via watch) and creates a replacement. The OwnerReference is also how cascading deletes work: delete a Deployment and all its RSes and Pods are garbage-collected.

## Worked example

### ConfigMap manifest

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fastapi-config
  namespace: default
data:
  APP_ENV: production
  LOG_LEVEL: INFO
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  # Multi-line config file as a key
  app.toml: |
    [server]
    host = "0.0.0.0"
    port = 8000
    workers = 4
    
    [logging]
    level = "INFO"
    format = "json"
```

### Secret manifest

```yaml
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: fastapi-secrets
  namespace: default
type: Opaque
data:
  # Values MUST be base64-encoded
  # Generate: echo -n "value" | base64
  DATABASE_URL: <base64-encoded-database-url>
  JWT_SECRET_KEY: <base64-encoded-jwt-key>
```

```bash
# Create a secret imperatively (safer — no plaintext in files)
kubectl create secret generic fastapi-secrets \
  --from-literal=DATABASE_URL="${DATABASE_URL}" \
  --from-literal=JWT_SECRET_KEY="${JWT_SECRET_KEY}"

# Or from a .env file
kubectl create secret generic fastapi-secrets \
  --from-env-file=.env.production

# Verify (values are redacted)
kubectl describe secret fastapi-secrets
kubectl get secret fastapi-secrets -o jsonpath='{.data.DATABASE_URL}' | base64 -d
```

### Consuming ConfigMap and Secret in a Deployment

```yaml
# deployment-with-config.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fastapi-app
spec:
  replicas: 2
  selector:
    matchLabels:
      app: fastapi
  template:
    metadata:
      labels:
        app: fastapi
        version: "2.0"
        env: production
    spec:
      containers:
        - name: api
          image: ghcr.io/myorg/fastapi-app:2.0.0
          ports:
            - containerPort: 8000

          # Option 1: load all ConfigMap keys as env vars
          envFrom:
            - configMapRef:
                name: fastapi-config
            - secretRef:
                name: fastapi-secrets

          # Option 2: load specific keys (explicit, preferred for clarity)
          env:
            - name: APP_ENV
              valueFrom:
                configMapKeyRef:
                  name: fastapi-config
                  key: APP_ENV
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: fastapi-secrets
                  key: DATABASE_URL

          # Option 3: mount config file from ConfigMap
          volumeMounts:
            - name: config-volume
              mountPath: /app/config
              readOnly: true

      volumes:
        - name: config-volume
          configMap:
            name: fastapi-config
            items:
              - key: app.toml
                path: app.toml     # file will appear at /app/config/app.toml
```

### Label queries with kubectl

```bash
# Filter by label
kubectl get pods -l app=fastapi
kubectl get pods -l app=fastapi,env=production
kubectl get pods -l 'env in (production, staging)'
kubectl get pods -l 'env notin (development)'
kubectl get pods -l '!env'   # pods WITHOUT the env label

# Label a resource imperatively
kubectl label pod fastapi-pod-abc tier=backend

# Remove a label
kubectl label pod fastapi-pod-abc tier-

# Show labels in output
kubectl get pods --show-labels

# Annotate a resource
kubectl annotate deployment fastapi-app \
  deployment.kubernetes.io/revision-note="added auth middleware"
```

### ConfigMap update strategy

```bash
# Update a ConfigMap
kubectl create configmap fastapi-config \
  --from-literal=LOG_LEVEL=DEBUG \
  --dry-run=client -o yaml | kubectl apply -f -

# Pods using env vars must restart to pick up changes
kubectl rollout restart deployment/fastapi-app

# Pods using volume mounts pick up changes automatically (with slight delay)
```

## Exercises

1. Write a ConfigMap named `model-config` with keys: `MODEL_NAME=claude-sonnet-5-5`, `MAX_TOKENS=4096`, and a multi-line JSON config stored under key `model.json`.

2. Write the kubectl command to create a Secret named `api-credentials` from environment variables `API_KEY` and `API_SECRET` without putting values in a YAML file.

3. A Deployment's pods are not receiving updated values from a ConfigMap that was changed. The ConfigMap is mounted as environment variables. Explain why and describe two ways to fix it.

4. Write a label selector that matches pods with `app=worker` AND `env` not equal to `development`. Show both kubectl syntax and the YAML `matchExpressions` form.

5. Why are Secrets not truly secure by default in Kubernetes, and what should you use instead in production?

(Solutions: ../solutions/week-05/day-4.md)

## Self-check

- [ ] I can write ConfigMap and Secret manifests
- [ ] I can inject config as env vars and as mounted files
- [ ] I can create Secrets imperatively without plaintext in YAML
- [ ] I can write label selectors and filter resources with kubectl
- [ ] I understand the limitations of k8s Secrets and what to use in production

## Supplementary links

- [ConfigMaps — concepts](https://kubernetes.io/docs/concepts/configuration/configmap/)
- [Secrets — concepts](https://kubernetes.io/docs/concepts/configuration/secret/)
- [Labels and Selectors](https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/)
- [External Secrets Operator](https://external-secrets.io/)
