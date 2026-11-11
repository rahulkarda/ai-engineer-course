# Solutions — Week 06 · Day 7 / Project

## Reference implementation

The complete project solution is provided below. It meets all acceptance criteria in `project.md`.

---

### Chart.yaml

```yaml
apiVersion: v2
name: ai-platform
description: Production-grade Helm chart for the AI Platform API
type: application
version: 0.1.0
appVersion: "1.0.0"
maintainers:
  - name: Platform Team
    email: platform@example.com   # placeholder
keywords: [ai, api, production]
```

---

### values.yaml

```yaml
replicaCount: 2

image:
  repository: ""    # required: set at install time (e.g. registry.example.com/ai-api)
  tag: ""           # falls back to Chart.AppVersion
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
    nginx.ingress.kubernetes.io/proxy-read-timeout: "120"
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: ""    # required: set at install time (e.g. ai.example.com)
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: ai-platform-tls
      hosts: []   # populated by values override

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
  name: ""

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
    repository: ""   # placeholder: set to registry.example.com/ai-tools
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

---

### values-staging.yaml

```yaml
replicaCount: 1
autoscaling:
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
configmap:
  data:
    LOG_LEVEL: "DEBUG"
    MODEL_TIMEOUT_MS: "60000"
cronjob:
  enabled: false
```

---

### values-production.yaml

```yaml
replicaCount: 4
autoscaling:
  minReplicas: 4
  maxReplicas: 40
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
  name: ai-prod-secrets
configmap:
  data:
    LOG_LEVEL: "INFO"
    MODEL_TIMEOUT_MS: "30000"
```

---

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

---

### templates/serviceaccount.yaml

```yaml
{{- if .Values.serviceAccount.create }}
apiVersion: v1
kind: ServiceAccount
metadata:
  name: {{ include "ai-platform.serviceAccountName" . }}
  labels:
    {{- include "ai-platform.labels" . | nindent 4 }}
  {{- with .Values.serviceAccount.annotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
{{- end }}
```

---

### templates/role.yaml

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: {{ include "ai-platform.fullname" . }}
  labels:
    {{- include "ai-platform.labels" . | nindent 4 }}
rules:
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["get", "list"]
  - apiGroups: [""]
    resources: ["secrets"]
    resourceNames: [{{ .Values.secrets.name | default "ai-secrets" | quote }}]
    verbs: ["get"]
```

---

### templates/rolebinding.yaml

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: {{ include "ai-platform.fullname" . }}
  labels:
    {{- include "ai-platform.labels" . | nindent 4 }}
subjects:
  - kind: ServiceAccount
    name: {{ include "ai-platform.serviceAccountName" . }}
    namespace: {{ .Release.Namespace }}
roleRef:
  kind: Role
  name: {{ include "ai-platform.fullname" . }}
  apiGroup: rbac.authorization.k8s.io
```

---

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

---

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
          image: "{{ required "image.repository must be set" .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
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
                name: {{ required "secrets.name is required when secrets.enabled=true" .Values.secrets.name }}
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

---

### templates/service.yaml

```yaml
apiVersion: v1
kind: Service
metadata:
  name: {{ include "ai-platform.fullname" . }}
  labels:
    {{- include "ai-platform.labels" . | nindent 4 }}
spec:
  type: {{ .Values.service.type }}
  selector:
    {{- include "ai-platform.selectorLabels" . | nindent 4 }}
  ports:
    - port: {{ .Values.service.port }}
      targetPort: http
      protocol: TCP
      name: http
```

---

### templates/ingress.yaml

```yaml
{{- if .Values.ingress.enabled }}
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {{ include "ai-platform.fullname" . }}
  labels:
    {{- include "ai-platform.labels" . | nindent 4 }}
  {{- with .Values.ingress.annotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
spec:
  ingressClassName: {{ .Values.ingress.className }}
  {{- if .Values.ingress.tls }}
  tls:
    {{- toYaml .Values.ingress.tls | nindent 4 }}
  {{- end }}
  rules:
    {{- range .Values.ingress.hosts }}
    - host: {{ required "ingress.hosts[].host must be set" .host | quote }}
      http:
        paths:
          {{- range .paths }}
          - path: {{ .path }}
            pathType: {{ .pathType }}
            backend:
              service:
                name: {{ include "ai-platform.fullname" $ }}
                port:
                  number: {{ $.Values.service.port }}
          {{- end }}
    {{- end }}
{{- end }}
```

---

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

---

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
  schedule: {{ required "cronjob.schedule required" .Values.cronjob.schedule | quote }}
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
          volumes:
            - name: tmp
              emptyDir: {}
          containers:
            - name: {{ .Values.cronjob.name }}
              image: "{{ .Values.cronjob.image.repository }}:{{ .Values.cronjob.image.tag | default .Chart.AppVersion }}"
              command: {{ toYaml .Values.cronjob.command | nindent 16 }}
              securityContext:
                {{- toYaml .Values.securityContext | nindent 16 }}
              volumeMounts:
                - name: tmp
                  mountPath: /tmp
              envFrom:
                - configMapRef:
                    name: {{ include "ai-platform.fullname" . }}-config
                {{- if .Values.secrets.enabled }}
                - secretRef:
                    name: {{ .Values.secrets.name }}
                {{- end }}
              resources:
                {{- toYaml .Values.cronjob.resources | nindent 16 }}
{{- end }}
```

---

### argocd-app.yaml

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: ai-platform-production
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/example/ai-platform-infra.git   # placeholder
    targetRevision: main
    path: charts/ai-platform
    helm:
      releaseName: ai-platform
      valueFiles:
        - values-production.yaml
  destination:
    server: https://kubernetes.default.svc
    namespace: production
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
      - PrunePropagationPolicy=foreground
    retry:
      limit: 3
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
```

---

### runbook.md

```markdown
# AI Platform Operations Runbook

## Procedure 1 — First install

```bash
# 1. Verify cluster context
kubectl config current-context
kubectl config get-contexts

# 2. Create namespace
kubectl create namespace production

# 3. Create secrets (values from environment — never hardcode)
kubectl create secret generic ai-prod-secrets \
  --from-literal=OPENAI_API_KEY="${OPENAI_API_KEY}" \
  -n production

# 4. Install chart
helm install ai-platform ./ai-platform \
  --namespace production \
  --create-namespace \
  --set image.repository=registry.example.com/ai-api \
  --set image.tag=1.0.0 \
  --set ingress.hosts[0].host=ai.prod.example.com \
  --set ingress.tls[0].secretName=ai-prod-tls \
  --set "ingress.tls[0].hosts[0]=ai.prod.example.com" \
  -f ./ai-platform/values-production.yaml \
  --atomic \
  --timeout 10m

# 5. Verify
kubectl get pods -n production -l app.kubernetes.io/name=ai-platform
kubectl get hpa -n production
kubectl get ingress -n production
```

## Procedure 2 — Image upgrade

```bash
# 1. Upgrade atomically
helm upgrade ai-platform ./ai-platform \
  --namespace production \
  -f ./ai-platform/values-production.yaml \
  --set image.tag=1.1.0 \
  --atomic \
  --timeout 10m

# 2. Verify rollout
kubectl rollout status deployment/ai-platform -n production

# 3. Confirm HPA still active
kubectl get hpa ai-platform -n production

# 4. Check release history
helm history ai-platform -n production
```

## Procedure 3 — Rollback

```bash
# 1. List release history
helm history ai-platform -n production

# 2. Roll back to previous revision (0 = previous)
helm rollback ai-platform 0 -n production --wait --timeout 5m

# 3. Verify rollout completed
kubectl rollout status deployment/ai-platform -n production

# 4. Check pod versions
kubectl get pods -n production -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.containers[0].image}{"\n"}{end}'
```

## Procedure 4 — On-call: CrashLoopBackOff

```bash
# 1. Identify crashing pods
kubectl get pods -n production | grep CrashLoop

# 2. Get recent events
kubectl get events -n production --sort-by='.lastTimestamp' | tail -20

# 3. Read logs from the previous (crashed) container
kubectl logs <pod-name> -n production -c api --previous

# 4. Decode exit code
kubectl get pod <pod-name> -n production \
  -o jsonpath='{.status.containerStatuses[0].lastState.terminated}'
# exitCode 1 = application error; 137 = OOMKilled; 143 = SIGTERM

# 5. Fix (example: wrong env var)
# Edit values-production.yaml or the Secret, then:
helm upgrade ai-platform ./ai-platform \
  --namespace production \
  -f ./ai-platform/values-production.yaml \
  --set image.tag=<current-tag> \
  --atomic
```

## Procedure 5 — On-call: HPA stuck at minReplicas despite high CPU

```bash
# 1. Check HPA status and current metrics
kubectl describe hpa ai-platform -n production
kubectl get hpa ai-platform -n production
# If TARGETS shows <unknown>: metrics pipeline broken

# 2. Verify metrics-server is healthy
kubectl get apiservice v1beta1.metrics.k8s.io
kubectl get pods -n kube-system -l k8s-app=metrics-server

# 3. Check current metrics
kubectl top pods -n production -l app.kubernetes.io/name=ai-platform

# 4. Verify resource requests are set (HPA requires requests to compute %)
kubectl get pod <pod-name> -n production \
  -o jsonpath='{.spec.containers[0].resources.requests}'
# Must be non-empty; if empty: fix resources in values.yaml and upgrade

# 5. Check HPA events
kubectl describe hpa ai-platform -n production | grep -A10 Events
# Look for: "FailedGetScale", "DesiredReplicas", "unable to fetch metrics"
```
```

---

## Verification commands

```bash
# Render staging — CronJob must be absent, HPA maxReplicas=5
helm template ai-platform ./ai-platform -f values-staging.yaml | grep -E "kind:|maxReplicas"

# Render production — CronJob present, HPA maxReplicas=40
helm template ai-platform ./ai-platform \
  --set image.repository=registry.example.com/ai-api \
  --set "ingress.hosts[0].host=ai.prod.example.com" \
  -f values-production.yaml | grep -E "kind:|maxReplicas|CronJob"

# Verify no secrets in rendered output
helm template ai-platform ./ai-platform \
  --set image.repository=registry.example.com/ai-api \
  --set "ingress.hosts[0].host=ai.example.com" | grep -i password
# Must return nothing
```
