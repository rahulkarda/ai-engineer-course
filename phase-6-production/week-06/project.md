# Week 06 Project — AI Platform Helm Chart & Operations Runbook

> Full specification for the week-06 project. Build deliverables, then self-grade against acceptance criteria.

## What you are building

A production-ready, parameterized Helm chart for an AI Platform API service, plus a GitOps Application manifest and a five-procedure operations runbook. This is the capstone for Phase 6, Week 6 — it combines every topic from the week: HPA, Ingress, StatefulSets/PVCs (optional extension), Jobs/CronJobs, Helm templating, Argo CD, RBAC, pod security, and debugging.

## Context

The AI Platform serves HTTP inference requests at `POST /v1/infer`. It is stateless (no local state). A nightly CronJob runs model evaluation using a separate image/command. The cluster already has:
- `ingress-nginx` installed in namespace `ingress-nginx`
- `cert-manager` with ClusterIssuer `letsencrypt-prod`
- A StorageClass `standard` for RWO volumes
- metrics-server installed

## Deliverables

```
week-06-project/
├── ai-platform/                        # Helm chart
│   ├── Chart.yaml
│   ├── values.yaml                     # defaults (base)
│   ├── values-staging.yaml             # staging overrides
│   ├── values-production.yaml          # production overrides
│   └── templates/
│       ├── _helpers.tpl
│       ├── serviceaccount.yaml
│       ├── role.yaml
│       ├── rolebinding.yaml
│       ├── configmap.yaml
│       ├── deployment.yaml
│       ├── service.yaml
│       ├── ingress.yaml
│       ├── hpa.yaml
│       └── cronjob.yaml
├── argocd-app.yaml                     # Argo CD Application
└── runbook.md                          # Operations runbook
```

## Specification per file

### Chart.yaml

- `name: ai-platform`
- `type: application`
- `version: 0.1.0`
- `appVersion: "1.0.0"`
- `description`: one clear sentence

### values.yaml defaults

| Value | Default |
|---|---|
| `image.repository` | `registry.example.com/ai-api` (placeholder) |
| `image.tag` | `""` (falls back to appVersion) |
| `replicaCount` | `2` |
| `service.port` | `80` |
| `service.targetPort` | `8000` |
| `autoscaling.enabled` | `true` |
| `autoscaling.minReplicas` | `2` |
| `autoscaling.maxReplicas` | `20` |
| `autoscaling.targetCPUUtilizationPercentage` | `60` |
| `autoscaling.scaleDownStabilizationSeconds` | `300` |
| `ingress.enabled` | `true` |
| `ingress.className` | `nginx` |
| `ingress.hosts[0].host` | `ai.example.com` (placeholder) |
| `cronjob.enabled` | `true` |
| `cronjob.schedule` | `"0 2 * * *"` |
| `cronjob.concurrencyPolicy` | `Forbid` |
| `configmap.data.LOG_LEVEL` | `INFO` |
| `configmap.data.MODEL_TIMEOUT_MS` | `"30000"` |
| `secrets.enabled` | `false` |
| `secrets.name` | `""` |

### values-staging.yaml (overrides)

- `replicaCount: 1`
- `autoscaling.minReplicas: 1`, `maxReplicas: 5`
- `ingress.hosts[0].host: staging-ai.example.com`
- `configmap.data.LOG_LEVEL: DEBUG`
- `cronjob.enabled: false`

### values-production.yaml (overrides)

- `replicaCount: 4`
- `autoscaling.minReplicas: 4`, `maxReplicas: 40`
- `ingress.hosts[0].host: ai.prod.example.com`
- `secrets.enabled: true`
- `secrets.name: ai-prod-secrets`
- resources: `requests.cpu: "500m"`, `requests.memory: "512Mi"`, `limits.cpu: "2"`, `limits.memory: "2Gi"`

### Deployment

- Pod security: `runAsNonRoot: true`, `runAsUser: 1000`, `fsGroup: 2000`, `seccompProfile.type: RuntimeDefault`
- Container security: `allowPrivilegeEscalation: false`, `readOnlyRootFilesystem: true`, `capabilities.drop: [ALL]`
- `tmp` emptyDir volume mounted at `/tmp` (needed for readOnlyRootFilesystem)
- `checksum/config` annotation triggering restart on ConfigMap change
- envFrom: ConfigMap + optional Secret
- Liveness: `GET /health` port `http`; Readiness: `GET /ready` port `http`

### HPA

- Target the Deployment
- Use autoscaling/v2 API
- Include `behavior.scaleDown.stabilizationWindowSeconds` from values
- Both CPU and memory metrics if both target values are set (use `if` guard)

### Ingress

- `ingressClassName` from values
- cert-manager annotation `cert-manager.io/cluster-issuer: letsencrypt-prod` when `ingress.tls` is non-empty
- TLS from values
- Host/path rules from values

### CronJob

- All fields from values
- `restartPolicy: OnFailure`
- Same pod security context as Deployment
- ConfigMap envFrom; optional Secret envFrom

### RBAC

- ServiceAccount: created when `serviceAccount.create=true`
- Role: `get`, `list` on `configmaps` in the release namespace
- RoleBinding: binds Role to ServiceAccount

### Argo CD Application

- `metadata.namespace: argocd`
- `spec.source.repoURL`: placeholder `https://github.com/example/ai-platform-infra.git`
- `spec.source.path: charts/ai-platform`
- `spec.source.targetRevision: main`
- `spec.source.helm.valueFiles: ["values-production.yaml"]`
- `spec.destination.namespace: production`
- `spec.syncPolicy.automated.prune: true`
- `spec.syncPolicy.automated.selfHeal: true`
- `spec.syncPolicy.syncOptions: [CreateNamespace=true]`
- Retry: `limit: 3`, exponential backoff

### runbook.md

Five numbered procedures, each self-contained:

1. **First install**: context check → namespace creation → secret creation (from env, not hardcoded) → `helm install --atomic` → verify pods/hpa/ingress
2. **Image upgrade**: `helm upgrade --atomic`, `kubectl rollout status`, `kubectl get hpa` to confirm still active
3. **Rollback**: `helm history` → `helm rollback 0 --wait` → `kubectl rollout status` → `kubectl get pods`
4. **CrashLoopBackOff**: `kubectl get events`, `kubectl logs --previous`, decode exit code, fix and upgrade
5. **HPA stuck at minReplicas**: `kubectl describe hpa`, `kubectl top pods`, check metrics-server, check resource requests set on pods

## Acceptance criteria

- [ ] `helm template ai-platform ./ai-platform` renders without errors.
- [ ] `helm template ai-platform ./ai-platform -f values-staging.yaml` renders without errors.
- [ ] `helm template ai-platform ./ai-platform -f values-production.yaml` renders without errors.
- [ ] Staging render: CronJob absent, HPA maxReplicas=5.
- [ ] Production render: CronJob present, HPA maxReplicas=40, envFrom includes Secret.
- [ ] All pods pass `restricted` Pod Security Standard (verify fields in deployment.yaml).
- [ ] RBAC Role is scoped to `configmaps` only (not wildcards).
- [ ] No real credentials or secrets in any file.
- [ ] Argo CD Application has `automated` with `selfHeal: true` and `prune: true`.
- [ ] All five runbook procedures include exact kubectl commands (no "run the upgrade command" placeholders).

## Extension challenges (optional)

1. Add a `PodDisruptionBudget` template with `minAvailable: 50%`, gated by `pdb.enabled`.
2. Add a `NetworkPolicy` allowing ingress only from `ingress-nginx` namespace pods and same-namespace pods with `role=internal`, gated by `networkPolicy.enabled`.
3. Add a Helm test (`templates/tests/test-connection.yaml`) that `curl`s `/health` and exits 0 on 200.
4. Publish the chart to an OCI registry and update the Argo CD Application to pull from the registry instead of Git.

(Project solution: ../solutions/week-06/project.md)
