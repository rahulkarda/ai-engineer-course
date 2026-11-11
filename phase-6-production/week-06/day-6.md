# Phase 6 · Week 06 · Day 6 (Saturday) — Build: Helm Chart + Operations Runbook

> Objective: Build a complete, production-grade Helm chart for the AI Platform with HPA, Ingress, CronJob, and write the operations runbook.  ·  Time: ~4 hr  ·  Prereqs: [[day-1]] – [[day-5]]

## Overview

Today you assemble everything from the week into a single shippable artefact:

1. A complete Helm chart (`ai-platform`) with: Deployment, Service, Ingress, HPA, CronJob, ConfigMap, ServiceAccount, RBAC.
2. Environment-specific values files for staging and production.
3. An operations runbook covering: deploy, scale, rollback, and debug procedures.

Work through each section sequentially. The final chart structure and runbook are fully specified here.

---

## Part 1 — Complete Helm chart

### Target structure

```
ai-platform/
├── Chart.yaml
├── values.yaml
├── values-staging.yaml
├── values-production.yaml
└── templates/
    ├── _helpers.tpl
    ├── serviceaccount.yaml
    ├── role.yaml
    ├── rolebinding.yaml
    ├── configmap.yaml
    ├── deployment.yaml
    ├── service.yaml
    ├── ingress.yaml
    ├── hpa.yaml
    ├── cronjob.yaml
    └── NOTES.txt
```

### Chart.yaml

```yaml
apiVersion: v2
name: ai-platform
description: Production-grade Helm chart for the AI Platform
type: application
version: 0.1.0
appVersion: "1.0.0"
maintainers:
  - name: Platform Team
    email: platform@example.com   # placeholder
keywords: [ai, api, production]
```

### values.yaml (complete)

```yaml
replicaCount: 2

image:
  repository: registry.example.com/ai-api   # placeholder — override at deploy time
  tag: ""
  pullPolicy: IfNotPresent

imagePullSecrets: []
nameOverride: ""
fullnameOverride: ""

serviceAccount:
  create: true
  annotations: {}
  name: ""
  # Roles granted to the ServiceAccount (list of {apiGroups, resources, verbs})
  rules:
    - apiGroups: [""]
      resources: ["configmaps"]
      verbs: ["get", "list"]

service:
  type: ClusterIP
  port: 80
  targetPort: 8000

ingress:
  enabled: true
  className: nginx
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "120"
  hosts:
    - host: ai.example.com   # placeholder
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: ai-example-com-tls
      hosts:
        - ai.example.com   # placeholder

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
  scaleDownStabilizationSeconds: 300

configmap:
  data:
    LOG_LEVEL: "INFO"
    MODEL_TIMEOUT_MS: "30000"

secrets:
  enabled: false
  name: ""   # name of existing Secret mounted via envFrom

cronjob:
  enabled: true
  name: nightly-eval
  schedule: "0 2 * * *"
  timeZone: "UTC"
  concurrencyPolicy: Forbid
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 1
  activeDeadlineSeconds: 7200
  backoffLimit: 1
  image:
    repository: registry.example.com/ai-tools   # placeholder
    tag: ""
  command: ["python", "-m", "eval.nightly"]
  resources:
    requests:
      cpu: "500m"
      memory: "1Gi"
    limits:
      cpu: "2"
      memory: "4Gi"

podSecurityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 2000
  seccompProfile:
    type: RuntimeDefault

securityContext:
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: true
  capabilities:
    drop: ["ALL"]

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

### templates/_helpers.tpl

```
{{- define "ai-platform.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "ai-platform.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name (default .Chart.Name .Values.nameOverride) | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{- define "ai-platform.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | trunc 63 }}
{{ include "ai-platform.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "ai-platform.selectorLabels" -}}
app.kubernetes.io/name: {{ include "ai-platform.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "ai-platform.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "ai-platform.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}
```

### templates/configmap.yaml

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ include "ai-platform.fullname" . }}-config
  labels:
    {{- include "ai-platform.labels" . | nindent 4 }}
data:
  {{- toYaml .Values.configmap.data | nindent 2 }}
```

### templates/deployment.yaml

```yaml
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
      annotations:
        checksum/config: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
    spec:
      {{- with .Values.imagePullSecrets }}
      imagePullSecrets:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      serviceAccountName: {{ include "ai-platform.serviceAccountName" . }}
      securityContext:
        {{- toYaml .Values.podSecurityContext | nindent 8 }}
      volumes:
        - name: tmp
          emptyDir: {}
      containers:
        - name: api
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          securityContext:
            {{- toYaml .Values.securityContext | nindent 12 }}
          ports:
            - name: http
              containerPort: {{ .Values.service.targetPort }}
              protocol: TCP
          volumeMounts:
            - name: tmp
              mountPath: /tmp
          envFrom:
            - configMapRef:
                name: {{ include "ai-platform.fullname" . }}-config
            {{- if .Values.secrets.enabled }}
            - secretRef:
                name: {{ required "secrets.name required when secrets.enabled=true" .Values.secrets.name }}
            {{- end }}
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

### templates/hpa.yaml

```yaml
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
  behavior:
    scaleDown:
      stabilizationWindowSeconds: {{ .Values.autoscaling.scaleDownStabilizationSeconds }}
      policies:
        - type: Pods
          value: 2
          periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 30
      policies:
        - type: Percent
          value: 100
          periodSeconds: 60
{{- end }}
```

### templates/cronjob.yaml

```yaml
{{- if .Values.cronjob.enabled }}
apiVersion: batch/v1
kind: CronJob
metadata:
  name: {{ include "ai-platform.fullname" . }}-{{ .Values.cronjob.name }}
  labels:
    {{- include "ai-platform.labels" . | nindent 4 }}
spec:
  schedule: {{ .Values.cronjob.schedule | quote }}
  timeZone: {{ .Values.cronjob.timeZone | quote }}
  concurrencyPolicy: {{ .Values.cronjob.concurrencyPolicy }}
  successfulJobsHistoryLimit: {{ .Values.cronjob.successfulJobsHistoryLimit }}
  failedJobsHistoryLimit: {{ .Values.cronjob.failedJobsHistoryLimit }}
  jobTemplate:
    spec:
      backoffLimit: {{ .Values.cronjob.backoffLimit }}
      activeDeadlineSeconds: {{ .Values.cronjob.activeDeadlineSeconds }}
      template:
        spec:
          restartPolicy: OnFailure
          serviceAccountName: {{ include "ai-platform.serviceAccountName" . }}
          securityContext:
            {{- toYaml .Values.podSecurityContext | nindent 12 }}
          containers:
            - name: {{ .Values.cronjob.name }}
              image: "{{ .Values.cronjob.image.repository }}:{{ .Values.cronjob.image.tag | default .Chart.AppVersion }}"
              command: {{ toYaml .Values.cronjob.command | nindent 16 }}
              securityContext:
                {{- toYaml .Values.securityContext | nindent 16 }}
              resources:
                {{- toYaml .Values.cronjob.resources | nindent 16 }}
              envFrom:
                - configMapRef:
                    name: {{ include "ai-platform.fullname" . }}-config
                {{- if .Values.secrets.enabled }}
                - secretRef:
                    name: {{ .Values.secrets.name }}
                {{- end }}
{{- end }}
```

### values-staging.yaml

```yaml
replicaCount: 1
autoscaling:
  enabled: true
  minReplicas: 1
  maxReplicas: 5
ingress:
  hosts:
    - host: staging-ai.example.com   # placeholder
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: staging-ai-tls
      hosts: [staging-ai.example.com]
resources:
  requests:
    cpu: "100m"
    memory: "128Mi"
  limits:
    cpu: "500m"
    memory: "512Mi"
configmap:
  data:
    LOG_LEVEL: "DEBUG"
    MODEL_TIMEOUT_MS: "60000"
cronjob:
  enabled: false   # disable nightly eval in staging
```

### values-production.yaml

```yaml
replicaCount: 4
autoscaling:
  enabled: true
  minReplicas: 4
  maxReplicas: 40
  targetCPUUtilizationPercentage: 60
  scaleDownStabilizationSeconds: 300
ingress:
  hosts:
    - host: ai.prod.example.com   # placeholder
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: ai-prod-tls
      hosts: [ai.prod.example.com]
resources:
  requests:
    cpu: "500m"
    memory: "512Mi"
  limits:
    cpu: "2"
    memory: "2Gi"
secrets:
  enabled: true
  name: ai-prod-secrets   # must exist before deploy
configmap:
  data:
    LOG_LEVEL: "INFO"
    MODEL_TIMEOUT_MS: "30000"
```

---

## Part 2 — Operations Runbook

### Deploy (first install)

```bash
# 1. Verify context
kubectl config current-context
kubectl config get-contexts

# 2. Create namespace and secrets (one-time setup)
kubectl create namespace production
kubectl create secret generic ai-prod-secrets \
  --from-literal=OPENAI_API_KEY="$(cat /run/secrets/openai_api_key)" \
  -n production
# Store secret in your org's secrets manager; do NOT commit to Git.

# 3. Install
helm install ai-platform ./ai-platform \
  --namespace production \
  --create-namespace \
  -f ./ai-platform/values-production.yaml \
  --set image.tag=1.0.0 \
  --atomic \
  --timeout 10m

# 4. Verify
kubectl get pods -n production -l app.kubernetes.io/name=ai-platform
kubectl get hpa -n production
kubectl get ingress -n production
```

### Deploy (upgrade)

```bash
# 1. Render diff (optional, requires helm-diff plugin)
helm diff upgrade ai-platform ./ai-platform \
  -n production \
  -f ./ai-platform/values-production.yaml \
  --set image.tag=1.1.0

# 2. Upgrade atomically
helm upgrade ai-platform ./ai-platform \
  --namespace production \
  -f ./ai-platform/values-production.yaml \
  --set image.tag=1.1.0 \
  --atomic \
  --timeout 10m

# 3. Verify rollout
kubectl rollout status deployment/ai-platform -n production
helm history ai-platform -n production
```

### Scale manually (bypass HPA temporarily)

```bash
# Temporarily override HPA (HPA will re-assert after next reconcile)
kubectl scale deployment ai-platform --replicas=10 -n production
# To keep the override, patch HPA minReplicas:
kubectl patch hpa ai-platform -n production \
  -p '{"spec":{"minReplicas":10}}'
# Restore:
kubectl patch hpa ai-platform -n production \
  -p '{"spec":{"minReplicas":4}}'
```

### Rollback

```bash
# See history
helm history ai-platform -n production

# Rollback to previous release (revision 0 = previous)
helm rollback ai-platform 0 -n production --wait --timeout 5m

# Rollback to a specific revision
helm rollback ai-platform 3 -n production --wait --timeout 5m

# Verify
kubectl rollout status deployment/ai-platform -n production
kubectl get pods -n production
```

### Debug: Pod not starting

```bash
# Step 1 — Event summary
kubectl describe pod <pod-name> -n production | tail -30

# Step 2 — Common causes and checks
# ImagePullBackOff: wrong tag or missing pull secret
kubectl get events -n production --field-selector reason=Failed
kubectl get pod <pod-name> -n production -o jsonpath='{.status.containerStatuses[0].state}'

# CrashLoopBackOff: application error
kubectl logs <pod-name> -n production --previous

# Pending (no node): resource pressure or scheduling constraint
kubectl describe pod <pod-name> -n production | grep -A5 Events
kubectl get nodes -o wide
kubectl describe node <node-name> | grep -A10 Conditions

# OOMKilled (exit 137)
kubectl get pod <pod-name> -n production \
  -o jsonpath='{.status.containerStatuses[0].lastState.terminated.exitCode}'
# Increase memory limit or profile memory usage
```

### Debug: HPA not scaling

```bash
kubectl describe hpa ai-platform -n production
kubectl get hpa ai-platform -n production
# If TARGETS shows <unknown>:
kubectl get apiservices | grep metrics    # check metrics-server
kubectl top pods -n production           # verify metrics available
# Check resource requests are set on pods (required for HPA)
kubectl get pod <pod-name> -n production -o jsonpath='{.spec.containers[0].resources}'
```

### Debug: Ingress 502/404

```bash
# Check ingress rule
kubectl describe ingress ai-platform -n production

# Check backend service and endpoints
kubectl get svc ai-platform -n production
kubectl get endpoints ai-platform -n production   # must not be empty

# Controller logs
kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx --tail=50

# Test from inside cluster
kubectl run curl-debug --image=curlimages/curl --rm -it --restart=Never -- \
  curl -v http://ai-platform.production.svc.cluster.local/health
```

### Debug: CronJob not running

```bash
kubectl describe cronjob ai-platform-nightly-eval -n production
kubectl get jobs -n production --sort-by=.metadata.creationTimestamp
# If LAST SCHEDULE is <none> or stale:
# Check for timezone misconfiguration, forbidden concurrency, or suspended flag
kubectl get cronjob ai-platform-nightly-eval -n production \
  -o jsonpath='{.spec.suspend}'
# Trigger manually to verify image/command
kubectl create job manual-eval \
  --from=cronjob/ai-platform-nightly-eval -n production
kubectl logs job/manual-eval -n production
```

---

## Part 3 — Exercises

1. Add a `PodDisruptionBudget` to the chart that ensures at least 50% of `ai-platform` pods are available during node drains. Create the template in `templates/pdb.yaml` and gate it with a `.Values.pdb.enabled` flag.

2. The nightly CronJob needs to write results to a shared PVC. Add a `cronjob.persistentVolume.enabled` value. When true, create a PVC template and mount it at `/results` in the CronJob pod.

3. Write a `helm test` hook that runs a `curl` container, calls `/health`, and exits 0 on success. Place it in `templates/tests/test-connection.yaml`.

4. Add a `NetworkPolicy` template that restricts ingress to the `ai-platform` pods: only allow traffic from the ingress-nginx namespace and from pods in the same namespace with label `role=internal`. Gate with `.Values.networkPolicy.enabled`.

5. Package the chart and push it to an OCI registry: `helm package ./ai-platform && helm push ai-platform-0.1.0.tgz oci://registry.example.com/helm`. Write the complete sequence including login.

(Solutions: ../solutions/week-06/day-6.md)

## Self-check

- [ ] Chart renders cleanly with `helm template` for both staging and production values.
- [ ] HPA, Ingress, and CronJob are properly gated with enabled flags.
- [ ] `required` guards catch missing values before deploy.
- [ ] Pod security context passes the `restricted` Pod Security Standard.
- [ ] Runbook covers deploy, upgrade, rollback, and four debug scenarios.
