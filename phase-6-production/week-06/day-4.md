# Phase 6 · Week 06 · Day 4 — Helm: Charts, Values & Templating

> Objective: Build a production-grade Helm chart from scratch, use values for environment parameterization, and apply templating best practices.  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

## Concept

**Helm** is the Kubernetes package manager. A **chart** is a directory of YAML templates plus metadata. Helm renders templates by merging `values.yaml` with overrides, then applies the result to the cluster via the Kubernetes API.

Why Helm over plain YAML:
- **DRY**: values shared across templates; no copy-paste.
- **Release management**: `helm install/upgrade/rollback` with atomic history.
- **Templating**: conditionals, loops, named templates, `required`/`default` guards.
- **Distribution**: charts are OCI artifacts (push to any OCI registry) or tarballs.

Helm 3 key concepts:
| Term | Meaning |
|---|---|
| Chart | Package (directory or `.tgz`) |
| Release | Deployed instance of a chart in a namespace |
| Values | User-supplied configuration |
| Template | Go `text/template` + Sprig functions |
| Hook | Run resources at specific lifecycle points (pre-install, post-upgrade, etc.) |

## How it works internally

`helm install` flow:
1. Load chart + merge values (CLI `--set` > `-f values.yaml` > chart `values.yaml`).
2. Render all templates in `templates/` to Kubernetes manifests.
3. Apply rendered manifests to the cluster (equivalent to `kubectl apply`).
4. Store the release metadata as a Secret in the target namespace (`helm.sh/release.v1`).

Templates use Go `text/template`. Helm adds Sprig functions (`default`, `required`, `toYaml`, `tpl`, `include`, `nindent`). The `_helpers.tpl` file is the canonical place for named templates (partials). Templates in `_*.tpl` files are never rendered directly.

`helm upgrade --atomic` wraps the upgrade: if any resource fails to reach Ready within `--timeout`, the upgrade is automatically rolled back to the previous release.

## Worked example

### 1. Chart scaffold

```bash
helm create ai-platform
```

Generated layout:
```
ai-platform/
├── Chart.yaml
├── values.yaml
├── charts/           # sub-charts (dependencies)
└── templates/
    ├── _helpers.tpl
    ├── deployment.yaml
    ├── service.yaml
    ├── ingress.yaml
    ├── hpa.yaml
    └── NOTES.txt
```

### 2. Chart.yaml

```yaml
# ai-platform/Chart.yaml
apiVersion: v2
name: ai-platform
description: Production Helm chart for the AI Platform API
type: application
version: 0.1.0        # chart version (SemVer)
appVersion: "1.0.0"   # application version (informational)
keywords:
  - ai
  - api
  - production
```

### 3. values.yaml — complete defaults

```yaml
# ai-platform/values.yaml
replicaCount: 2

image:
  repository: registry.example.com/ai-api   # placeholder
  tag: ""          # overridden at deploy time; falls back to .Chart.AppVersion
  pullPolicy: IfNotPresent

imagePullSecrets: []
nameOverride: ""
fullnameOverride: ""

serviceAccount:
  create: true
  annotations: {}
  name: ""

service:
  type: ClusterIP
  port: 80
  targetPort: 8000

ingress:
  enabled: true
  className: nginx
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
  hosts:
    - host: ai.example.com    # placeholder
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: ai-example-com-tls
      hosts:
        - ai.example.com

resources:
  requests:
    cpu: "200m"
    memory: "256Mi"
  limits:
    cpu: "800m"
    memory: "512Mi"

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 20
  targetCPUUtilizationPercentage: 60
  targetMemoryUtilizationPercentage: 70

env: {}
  # MODEL_ENDPOINT: "https://model.example.com"   # placeholder

secrets:
  enabled: false
  name: ""        # name of existing Secret to expose as envFrom

livenessProbe:
  httpGet:
    path: /health
    port: http
  initialDelaySeconds: 10
  periodSeconds: 15

readinessProbe:
  httpGet:
    path: /ready
    port: http
  initialDelaySeconds: 5
  periodSeconds: 10

nodeSelector: {}
tolerations: []
affinity: {}
```

### 4. _helpers.tpl — named templates

```
{{/* ai-platform/templates/_helpers.tpl */}}

{{/*
Expand the name of the chart.
*/}}
{{- define "ai-platform.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "ai-platform.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "ai-platform.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{ include "ai-platform.selectorLabels" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "ai-platform.selectorLabels" -}}
app.kubernetes.io/name: {{ include "ai-platform.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
ServiceAccount name
*/}}
{{- define "ai-platform.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "ai-platform.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}
```

### 5. deployment.yaml template

```yaml
{{/* ai-platform/templates/deployment.yaml */}}
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "ai-platform.fullname" . }}
  labels:
    {{- include "ai-platform.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "ai-platform.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "ai-platform.selectorLabels" . | nindent 8 }}
    spec:
      {{- with .Values.imagePullSecrets }}
      imagePullSecrets:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      serviceAccountName: {{ include "ai-platform.serviceAccountName" . }}
      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - name: http
              containerPort: {{ .Values.service.targetPort }}
              protocol: TCP
          {{- with .Values.livenessProbe }}
          livenessProbe:
            {{- toYaml . | nindent 12 }}
          {{- end }}
          {{- with .Values.readinessProbe }}
          readinessProbe:
            {{- toYaml . | nindent 12 }}
          {{- end }}
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
          {{- if .Values.env }}
          env:
            {{- range $key, $value := .Values.env }}
            - name: {{ $key }}
              value: {{ $value | quote }}
            {{- end }}
          {{- end }}
          {{- if .Values.secrets.enabled }}
          envFrom:
            - secretRef:
                name: {{ required "secrets.name is required when secrets.enabled=true" .Values.secrets.name }}
          {{- end }}
      {{- with .Values.nodeSelector }}
      nodeSelector:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.affinity }}
      affinity:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
```

### 6. hpa.yaml template

```yaml
{{/* ai-platform/templates/hpa.yaml */}}
{{- if .Values.autoscaling.enabled }}
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ include "ai-platform.fullname" . }}
  labels:
    {{- include "ai-platform.labels" . | nindent 4 }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ include "ai-platform.fullname" . }}
  minReplicas: {{ .Values.autoscaling.minReplicas }}
  maxReplicas: {{ .Values.autoscaling.maxReplicas }}
  metrics:
    {{- if .Values.autoscaling.targetCPUUtilizationPercentage }}
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetCPUUtilizationPercentage }}
    {{- end }}
    {{- if .Values.autoscaling.targetMemoryUtilizationPercentage }}
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetMemoryUtilizationPercentage }}
    {{- end }}
{{- end }}
```

### 7. Rendering and deployment

```bash
# Render without applying (dry-run)
helm template ai-platform ./ai-platform \
  --namespace production \
  --set image.tag=v1.2.3 \
  --set ingress.hosts[0].host=staging.example.com

# Install
helm install ai-platform ./ai-platform \
  --namespace production \
  --create-namespace \
  --set image.tag=v1.2.3 \
  -f values-production.yaml

# Upgrade
helm upgrade ai-platform ./ai-platform \
  --namespace production \
  --set image.tag=v1.3.0 \
  --atomic \
  --timeout 5m

# Rollback to previous release
helm rollback ai-platform 0 --namespace production

# History
helm history ai-platform --namespace production
```

### 8. Environment-specific values file

```yaml
# values-production.yaml
replicaCount: 4
ingress:
  hosts:
    - host: ai.prod.example.com   # placeholder
      paths:
        - path: /
          pathType: Prefix
resources:
  requests:
    cpu: "500m"
    memory: "512Mi"
  limits:
    cpu: "2"
    memory: "2Gi"
autoscaling:
  maxReplicas: 40
secrets:
  enabled: true
  name: ai-prod-secrets
```

## Exercises

1. Add a `cronjob.yaml` template to the `ai-platform` chart that creates a CronJob when `.Values.cronjob.enabled` is `true`, with schedule, image, and resource values from `.Values.cronjob.*`.

2. The `required` function raises an error if a value is missing. Add `required` guards to the chart for `image.repository` and `ingress.hosts[0].host`. Show the template syntax.

3. Write a `values-staging.yaml` that overrides: 2 replicas, HPA max 5, ingress host `staging.example.com`, and disables liveness probe.

4. Explain what `helm upgrade --atomic` does differently from a plain `helm upgrade`. What happens if the upgrade times out?

5. A chart's `templates/secret.yaml` would embed a real password if rendered. Explain two correct patterns (ExternalSecrets Operator and Sealed Secrets) and how neither stores the plaintext value in the chart repository.

(Solutions: ../solutions/week-06/day-4.md)

## Self-check

- [ ] I can scaffold a Helm chart and understand all top-level files.
- [ ] I can write `values.yaml` with sensible defaults and use `required`/`default` guards.
- [ ] I can write named templates in `_helpers.tpl` and call them with `include`.
- [ ] I can template a Deployment, HPA, and Ingress with conditionals and loops.
- [ ] I can use `helm template`, `helm install`, `helm upgrade --atomic`, and `helm rollback`.

## Supplementary links

- [Helm documentation](https://helm.sh/docs/)
- [Chart best practices](https://helm.sh/docs/chart_best_practices/)
- [Sprig function reference](https://masterminds.github.io/sprig/)
- [Helm template functions](https://helm.sh/docs/chart_template_guide/function_list/)
