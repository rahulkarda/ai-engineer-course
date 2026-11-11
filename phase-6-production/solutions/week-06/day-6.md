# Solutions — Week 06 · Day 6 (Saturday Build)

## Exercise 1 — PodDisruptionBudget template

```yaml
{{/* ai-platform/templates/pdb.yaml */}}
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

Values addition:
```yaml
pdb:
  enabled: true
  minAvailable: "50%"
```

`minAvailable: "50%"` means at least 50% of matching pods must remain available during voluntary disruptions (node drains, cluster upgrades). With 4 production replicas, at least 2 must be up at all times during a drain.

## Exercise 2 — CronJob with PVC for results

Values addition:
```yaml
cronjob:
  # ... existing fields ...
  persistentVolume:
    enabled: false
    storageClass: standard
    size: 10Gi
    mountPath: /results
```

Add to `templates/cronjob.yaml` inside the Job template pod spec:

```yaml
{{- if .Values.cronjob.persistentVolume.enabled }}
volumes:
  - name: results
    persistentVolumeClaim:
      claimName: {{ include "ai-platform.fullname" . }}-eval-results
{{- end }}
containers:
  - name: {{ .Values.cronjob.name }}
    # ...
    {{- if .Values.cronjob.persistentVolume.enabled }}
    volumeMounts:
      - name: results
        mountPath: {{ .Values.cronjob.persistentVolume.mountPath }}
    {{- end }}
```

Add `templates/cronjob-pvc.yaml`:

```yaml
{{- if and .Values.cronjob.enabled .Values.cronjob.persistentVolume.enabled }}
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: {{ include "ai-platform.fullname" . }}-eval-results
  labels:
    {{- include "ai-platform.labels" . | nindent 4 }}
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: {{ .Values.cronjob.persistentVolume.storageClass }}
  resources:
    requests:
      storage: {{ .Values.cronjob.persistentVolume.size }}
{{- end }}
```

## Exercise 3 — Helm test hook

```yaml
{{/* ai-platform/templates/tests/test-connection.yaml */}}
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

Run after install: `helm test ai-platform -n production --logs`

The `helm.sh/hook-delete-policy: hook-succeeded` annotation deletes the test pod after a successful run, keeping the namespace clean.

## Exercise 4 — NetworkPolicy template

```yaml
{{/* ai-platform/templates/networkpolicy.yaml */}}
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
    # Allow from ingress-nginx namespace
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: ingress-nginx
    # Allow from same namespace with role=internal
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: {{ .Release.Namespace }}
          podSelector:
            matchLabels:
              role: internal
{{- end }}
```

Values addition:
```yaml
networkPolicy:
  enabled: false
```

Note: `namespaceSelector` and `podSelector` in the same `from` entry are AND'd (both must match). Two separate entries in the `from` list are OR'd (either can match). The manifest above correctly uses two separate entries.

## Exercise 5 — Package and push chart to OCI registry

```bash
# 1. Package the chart
helm package ./ai-platform
# Produces: ai-platform-0.1.0.tgz

# 2. Log in to OCI registry (credentials from environment — do not hardcode)
echo "${REGISTRY_PASSWORD}" | helm registry login registry.example.com \
  --username "${REGISTRY_USERNAME}" \
  --password-stdin

# 3. Push to registry
helm push ai-platform-0.1.0.tgz oci://registry.example.com/helm

# 4. Verify
helm show chart oci://registry.example.com/helm/ai-platform --version 0.1.0

# 5. Install from OCI registry
helm install ai-platform oci://registry.example.com/helm/ai-platform \
  --version 0.1.0 \
  --namespace production \
  -f values-production.yaml \
  --atomic
```

The Argo CD Application can reference the OCI chart:
```yaml
source:
  chart: ai-platform
  repoURL: oci://registry.example.com/helm
  targetRevision: "0.1.0"
  helm:
    valueFiles: ["values-production.yaml"]
```

OCI charts skip the `helm repo add` step — the chart is addressed directly by its OCI URL, matching how container images are addressed.
