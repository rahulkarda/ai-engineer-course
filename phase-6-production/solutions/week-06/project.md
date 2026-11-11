# Solutions — Week 06 Project

## Complete reference implementation

This file contains annotated solutions for the Week 06 project deliverables and explains key design decisions.

---

## Acceptance criteria verification

### 1. Staging render — CronJob absent, HPA maxReplicas=5

```bash
helm template ai-platform ./ai-platform \
  --set image.repository=registry.example.com/ai-api \
  --set "ingress.hosts[0].host=staging-ai.example.com" \
  -f values-staging.yaml | grep -E "^kind:|maxReplicas"
```

Expected output:
```
kind: ServiceAccount
kind: Role
kind: RoleBinding
kind: ConfigMap
kind: Service
kind: Ingress
kind: Deployment
kind: HorizontalPodAutoscaler
  maxReplicas: 5
```
No `kind: CronJob` line — correct.

### 2. Production render — CronJob present, HPA maxReplicas=40

```bash
helm template ai-platform ./ai-platform \
  --set image.repository=registry.example.com/ai-api \
  --set "ingress.hosts[0].host=ai.prod.example.com" \
  -f values-production.yaml | grep -E "^kind:|maxReplicas"
```

Expected output includes `kind: CronJob` and `maxReplicas: 40`.

### 3. Pod security context passes `restricted`

Required fields for `restricted` profile:
```yaml
# Pod-level (podSecurityContext)
runAsNonRoot: true
runAsUser: 1000          # non-zero
fsGroup: 2000
seccompProfile:
  type: RuntimeDefault   # or Localhost

# Container-level (securityContext)
allowPrivilegeEscalation: false
readOnlyRootFilesystem: true
capabilities:
  drop: ["ALL"]
```

The `/tmp` emptyDir volume mount is required when `readOnlyRootFilesystem: true` — many frameworks write temp files there.

### 4. RBAC restricted to configmaps

The Role in `templates/role.yaml` grants only:
- `get`, `list` on `configmaps`
- `get` on a specific named Secret (via `resourceNames`)

No wildcards. Verification:
```bash
kubectl auth can-i get secrets/ai-prod-secrets \
  --as=system:serviceaccount:production:ai-platform -n production
# yes

kubectl auth can-i delete pods \
  --as=system:serviceaccount:production:ai-platform -n production
# no
```

### 5. No plaintext secrets

Secrets are referenced by name (via `secrets.name` value), never embedded. The chart creates no Secret resources. The actual Secret must be created out-of-band before install.

---

## Design decisions

### checksum/config annotation

```yaml
annotations:
  checksum/config: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
```

This annotation on the Deployment pod template ensures that whenever `values.configmap.data` changes and `helm upgrade` is run, the pod template spec changes (new hash), triggering a rolling restart. Without this, updated ConfigMap values would only be visible to new pods (not running ones).

### required guards

```
image: "{{ required "image.repository must be set" .Values.image.repository }}:..."
host: {{ required "ingress.hosts[].host must be set" .host | quote }}
```

These cause `helm template`/`helm install` to fail with a descriptive message if the operator forgets to provide a value. Better than deploying with an empty string and debugging a failing pod.

### CronJob concurrencyPolicy: Forbid default

AI evaluation jobs are often idempotent but expensive. `Forbid` prevents two runs from consuming double the cluster resources if one run runs long. `Allow` would permit overlap. `Replace` would kill the old run, which could corrupt partial results.

### Atomic upgrade + scale-down stabilization

`helm upgrade --atomic` plus a 300-second scale-down stabilization window together prevent two failure modes:
1. Bad deploy → `--atomic` auto-rollbacks.
2. Traffic spike that subsides during upgrade → stabilization prevents scaling down pods that just finished starting, which would create a flap.

---

## Runbook design notes

Each runbook procedure is self-contained because:
- On-call engineers may have no context from the previous procedure.
- Runbooks are read under stress; the reader should not need to cross-reference.
- Every step that changes state (`helm install`, `kubectl create`) has a verification step immediately after it.

The five procedures cover the four most common production failure modes for a Kubernetes workload:
1. Initial deploy complexity (namespace, secrets bootstrap).
2. Image upgrades (99% of production changes).
3. Rollback (urgent fix for a bad upgrade).
4. CrashLoopBackOff (most common pod failure type).
5. HPA not scaling (silent failure: pod CPU is high but HPA does nothing).

---

## Extension challenge solutions

### PodDisruptionBudget

```yaml
{{/* templates/pdb.yaml */}}
{{- if .Values.pdb.enabled }}
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: {{ include "ai-platform.fullname" . }}
  labels:
    {{- include "ai-platform.labels" . | nindent 4 }}
spec:
  minAvailable: {{ .Values.pdb.minAvailable | default "50%" | quote }}
  selector:
    matchLabels:
      {{- include "ai-platform.selectorLabels" . | nindent 6 }}
{{- end }}
```

Values: `pdb: { enabled: true, minAvailable: "50%" }`

With 4 production replicas and `minAvailable: "50%"`, at least 2 pods must remain available during a node drain. Kubernetes will drain at most 2 pods at a time, respecting this budget.

### NetworkPolicy

```yaml
{{/* templates/networkpolicy.yaml */}}
{{- if .Values.networkPolicy.enabled }}
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: {{ include "ai-platform.fullname" . }}
  labels:
    {{- include "ai-platform.labels" . | nindent 4 }}
spec:
  podSelector:
    matchLabels:
      {{- include "ai-platform.selectorLabels" . | nindent 6 }}
  policyTypes:
    - Ingress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: ingress-nginx
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: {{ .Release.Namespace }}
          podSelector:
            matchLabels:
              role: internal
{{- end }}
```

### Helm test

```yaml
{{/* templates/tests/test-connection.yaml */}}
apiVersion: v1
kind: Pod
metadata:
  name: {{ include "ai-platform.fullname" . }}-test-connection
  labels:
    {{- include "ai-platform.labels" . | nindent 4 }}
  annotations:
    "helm.sh/hook": test
    "helm.sh/hook-delete-policy": before-hook-creation,hook-succeeded
spec:
  restartPolicy: Never
  containers:
    - name: curl
      image: curlimages/curl:8.5.0
      command:
        - curl
        - --silent
        - --fail
        - "http://{{ include "ai-platform.fullname" . }}:{{ .Values.service.port }}/health"
```

Run: `helm test ai-platform -n production --logs`

### OCI push sequence

```bash
# Package
helm package ./ai-platform
# Produces ai-platform-0.1.0.tgz

# Login (credentials from environment)
echo "${REGISTRY_PASSWORD}" | helm registry login registry.example.com \
  --username "${REGISTRY_USERNAME}" \
  --password-stdin

# Push
helm push ai-platform-0.1.0.tgz oci://registry.example.com/helm

# Verify
helm show chart oci://registry.example.com/helm/ai-platform --version 0.1.0

# Argo CD Application using OCI source:
# source:
#   chart: ai-platform
#   repoURL: oci://registry.example.com/helm
#   targetRevision: "0.1.0"
```
