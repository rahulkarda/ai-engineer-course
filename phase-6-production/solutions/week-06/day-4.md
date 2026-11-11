# Solutions — Week 06 · Day 4 (Helm Charts, Values & Templating)

## Exercise 1 — CronJob template in ai-platform chart

```yaml
{{/* ai-platform/templates/cronjob.yaml */}}
{{- if .Values.cronjob.enabled }}
apiVersion: batch/v1
kind: CronJob
metadata:
  name: {{ include "ai-platform.fullname" . }}-{{ .Values.cronjob.name }}
  labels:
    {{- include "ai-platform.labels" . | nindent 4 }}
spec:
  schedule: {{ required "cronjob.schedule is required when cronjob.enabled=true" .Values.cronjob.schedule | quote }}
  timeZone: {{ .Values.cronjob.timeZone | default "UTC" | quote }}
  concurrencyPolicy: {{ .Values.cronjob.concurrencyPolicy | default "Forbid" }}
  successfulJobsHistoryLimit: {{ .Values.cronjob.successfulJobsHistoryLimit | default 3 }}
  failedJobsHistoryLimit: {{ .Values.cronjob.failedJobsHistoryLimit | default 1 }}
  jobTemplate:
    spec:
      backoffLimit: {{ .Values.cronjob.backoffLimit | default 1 }}
      activeDeadlineSeconds: {{ .Values.cronjob.activeDeadlineSeconds | default 7200 }}
      template:
        spec:
          restartPolicy: OnFailure
          serviceAccountName: {{ include "ai-platform.serviceAccountName" . }}
          containers:
            - name: {{ .Values.cronjob.name }}
              image: "{{ .Values.cronjob.image.repository }}:{{ .Values.cronjob.image.tag | default .Chart.AppVersion }}"
              command: {{ toYaml .Values.cronjob.command | nindent 16 }}
              resources:
                {{- toYaml .Values.cronjob.resources | nindent 16 }}
{{- end }}
```

Values additions in `values.yaml`:
```yaml
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
```

## Exercise 2 — `required` guards for image.repository and ingress host

In `templates/deployment.yaml`:
```
image: "{{ required "image.repository must be set" .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
```

In `templates/ingress.yaml` (within the hosts loop):
```
{{- range .Values.ingress.hosts }}
- host: {{ required "ingress.hosts[].host must be set" .host | quote }}
{{- end }}
```

The `required` function syntax: `required "error message" .Values.path`. It renders as the value if set and non-empty; otherwise it raises a template error that stops `helm template`/`helm install` and prints the message. This is superior to `default` when the value must be explicitly provided by the operator.

## Exercise 3 — values-staging.yaml

```yaml
replicaCount: 2

autoscaling:
  minReplicas: 2
  maxReplicas: 5

ingress:
  hosts:
    - host: staging.example.com   # placeholder
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: staging-tls
      hosts:
        - staging.example.com

configmap:
  data:
    LOG_LEVEL: "DEBUG"
    MODEL_TIMEOUT_MS: "60000"

livenessProbe: {}   # empty map disables the liveness probe in the Deployment template
```

Note: setting `livenessProbe: {}` with `{{- with .Values.livenessProbe }}` in the template means the block is skipped when the map is empty. Alternatively, add a top-level `livenessProbe.enabled` boolean and gate with `{{- if .Values.livenessProbe.enabled }}`.

## Exercise 4 — `helm upgrade --atomic` vs plain `helm upgrade`

**Plain `helm upgrade`:** submits the rendered manifests to Kubernetes API and returns. If a pod fails to start, Helm does not notice — the release is marked `deployed` but the cluster may be in a broken state.

**`helm upgrade --atomic`:** sets `--wait` internally (waits until all resources reach Ready within `--timeout`). If any resource fails to reach Ready within the timeout, Helm **automatically rolls back** to the previous release revision and marks the current release as `failed`. This makes the upgrade transactional.

If the upgrade times out:
1. Helm initiates a `helm rollback` to the previous revision automatically.
2. The release history shows the failed upgrade revision with status `failed`.
3. The cluster returns to the prior state.

`--atomic` is the correct default for production CI/CD pipelines. The main downside is that a long pod startup time (e.g., pulling a large AI model on startup) requires a correspondingly long `--timeout`.

## Exercise 5 — Secrets without plaintext in the chart repo

**Pattern 1: External Secrets Operator (ESO)**

ESO runs as a controller. You commit an `ExternalSecret` CRD resource to Git (safe — no actual secret data). ESO reads it, fetches the real secret from an external store (AWS Secrets Manager, Vault, GCP Secret Manager), and creates a Kubernetes `Secret` in the namespace at runtime.

```yaml
# ExternalSecret committed to Git — no secret value here
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: ai-secrets
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: ClusterSecretStore
  target:
    name: ai-prod-secrets   # creates this Secret in Kubernetes
  data:
    - secretKey: OPENAI_API_KEY
      remoteRef:
        key: production/ai-platform/openai-api-key   # path in Secrets Manager
```

**Pattern 2: Sealed Secrets (Bitnami)**

`kubeseal` CLI encrypts a Kubernetes Secret with the cluster's public key. The resulting `SealedSecret` is safe to commit to Git — only the cluster can decrypt it (holds the private key). The Sealed Secrets controller decrypts it and creates the actual `Secret`.

```bash
# Encrypt a secret
kubectl create secret generic ai-secrets \
  --from-literal=OPENAI_API_KEY=sk-... \
  --dry-run=client -o yaml | \
  kubeseal --format yaml > sealed-ai-secrets.yaml
# commit sealed-ai-secrets.yaml to Git; delete the plaintext
```

Both patterns store zero plaintext secret data in the chart or Git repository.
