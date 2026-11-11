# Solutions — Week 06 Assessment

## Conceptual answers

**1. HPA control loop and formula**

The HPA controller polls the metrics API every 15 seconds (configurable via `--horizontal-pod-autoscaler-sync-period`). It computes:

```
desiredReplicas = ceil(currentReplicas × (currentMetricValue / desiredMetricValue))
```

A 10% tolerance is applied: scale-up only triggers if `currentMetric / desiredMetric > 1.1`; scale-down only if `< 0.9`. This prevents flapping around the threshold.

The scale-down stabilization window (default 5 min) records the recommended replica count at each tick. The actual scale-down only fires when the recommendation has been consistently lower for the full window duration. This prevents aggressive scale-down during temporary traffic lulls.

**2. Ingress pathType**

- `Exact`: path must match character-for-character. `/api` matches `/api` only.
- `Prefix`: path is a URL prefix. `/api` matches `/api`, `/api/`, `/api/v2`, `/api/v1/users`.
- `ImplementationSpecific`: controller decides (NGINX supports regex here).

`Prefix` with rule path `/api` **would match `/api/v2`**.

**3. StatefulSet vs Deployment**

| Property | Deployment | StatefulSet |
|---|---|---|
| Pod identity | Random suffix (`-7d9f8c`) | Ordinal (`-0`, `-1`, `-2`) |
| DNS | Shared Service DNS | Per-pod DNS via Headless Service |
| Storage | Shared PVC or ephemeral | Per-pod PVC via `volumeClaimTemplates` |
| Scale-down order | Random | Reverse ordinal (N-1 first) |

Use StatefulSet for: databases, message brokers, distributed caches, vector DBs — any workload where each instance has distinct identity or data.

**4. CronJob concurrencyPolicy: Forbid at 03:00**

At 03:00, the CronJob controller checks if the 02:00 Job is still running. Since `concurrencyPolicy: Forbid`, the 03:00 tick is **skipped entirely** — no new Job is created. The 02:00 job continues. The missed schedule is recorded in the CronJob status.

If `successfulJobsHistoryLimit` jobs are missed consecutively (default threshold: 100 missed schedules), the CronJob controller stops scheduling entirely until manually reset.

**5. Argo CD reconciliation and selfHeal**

The reconciliation loop: repo-server clones Git → renders manifests → application controller diffs rendered manifests against live cluster state → computes sync status (`Synced` or `OutOfSync`).

`selfHeal: true` means: if someone manually edits a resource in the cluster (e.g., `kubectl edit deployment`), Argo CD detects the drift and automatically reverts it to match Git. This enforces Git as the sole source of truth.

`selfHeal: false` is used for staging/development environments where teams need to temporarily patch live resources for testing without Argo immediately reverting them.

**6. Role vs ClusterRole**

`Role` is namespace-scoped — it grants permissions on resources within one namespace.  
`ClusterRole` is cluster-scoped — it grants permissions on cluster-wide resources (Nodes, PersistentVolumes) or on namespaced resources across all namespaces (when bound with a ClusterRoleBinding).

Example of each:
- `Role`: grant `ai-api-sa` permission to read Secrets in the `production` namespace.
- `ClusterRole`: grant an ops engineer permission to `kubectl get/list/watch` all resources across all namespaces (cluster-viewer pattern).

**7. OOMKilled exit code**

Exit code **137** (128 + 9, where 9 = SIGKILL). Verify:

```bash
kubectl get pod <pod> -n production \
  -o jsonpath='{.status.containerStatuses[0].lastState.terminated.reason}'
# Returns: OOMKilled

kubectl get pod <pod> -n production \
  -o jsonpath='{.status.containerStatuses[0].lastState.terminated.exitCode}'
# Returns: 137
```

**8. Importance of allowPrivilegeEscalation: false for a root-default container**

A container running as root with `allowPrivilegeEscalation: true` can call `setuid` binaries or acquire additional Linux capabilities. If the container is compromised (e.g., via prompt injection triggering code execution in an AI API), the attacker could escalate to full root on the node via a kernel exploit or a container escape.

`allowPrivilegeEscalation: false` sets the `no_new_privs` flag in the Linux process, making it impossible for any child process to gain more privileges than the parent. Combined with `runAsNonRoot: true` and `capabilities.drop: [ALL]`, the blast radius of a container compromise is constrained to the container's own process space.

**9. helm upgrade --atomic with failing readiness probe**

Sequence:
1. Helm submits the new Deployment spec to Kubernetes.
2. Kubernetes starts a rolling update — new pod(s) created alongside old ones.
3. New pod fails its readiness probe → pod is never added to Endpoints → Service traffic continues to old pods.
4. Helm watches with `--wait` — it is waiting for all Deployment pods to reach Ready.
5. After `--timeout` (e.g., 5 min), the wait deadline expires.
6. `--atomic` triggers: Helm runs `helm rollback` to the previous revision automatically.
7. The failed upgrade revision is recorded in history with status `failed`.
8. Old (working) pods remain live throughout — zero downtime.

**10. checksum/config annotation**

```yaml
annotations:
  checksum/config: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
```

**Problem it solves:** By default, updating a ConfigMap does not trigger a Deployment rolling restart — the pods keep their old environment variables. The `checksum` annotation stores a hash of the ConfigMap's rendered content as a pod template annotation. When the ConfigMap changes, the hash changes, which changes the pod template spec, which forces Kubernetes to perform a rolling restart. This ensures ConfigMap changes are applied to running pods automatically on `helm upgrade`.

---

## Coding challenge — reference solution

### Chart.yaml

```yaml
apiVersion: v2
name: inference-engine
description: Helm chart for the AI Inference Engine
type: application
version: 0.1.0
appVersion: "1.0.0"
```

### values.yaml

```yaml
replicaCount: 2

image:
  repository: ""   # required
  tag: ""
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80
  targetPort: 8000

serviceAccount:
  create: true
  name: ""

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  scaleDownStabilizationSeconds: 180

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: inference.example.com   # placeholder
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: inference-tls
      hosts: [inference.example.com]

configmap:
  data:
    BATCH_SIZE: "32"
    LOG_LEVEL: "INFO"

secrets:
  enabled: false
  name: ""

cronjob:
  enabled: false
  schedule: "0 3 * * 0"
  timeZone: "UTC"
  concurrencyPolicy: Forbid
  activeDeadlineSeconds: 7200
  command: ["python", "-m", "eval.weekly"]

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

resources:
  requests:
    cpu: "200m"
    memory: "256Mi"
  limits:
    cpu: "800m"
    memory: "512Mi"
```

### values-production.yaml

```yaml
autoscaling:
  minReplicas: 5
  maxReplicas: 30
configmap:
  data:
    BATCH_SIZE: "32"
    LOG_LEVEL: "WARNING"
secrets:
  enabled: true
  name: inference-prod-secrets
cronjob:
  enabled: true
```

### Key template patterns

**deployment.yaml** — required guard on image.repository:
```yaml
image: "{{ required "image.repository must be set" .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
```

**hpa.yaml** — gated by enabled flag, scaleDown stabilization from values:
```yaml
{{- if .Values.autoscaling.enabled }}
# ... HPA spec with .Values.autoscaling.scaleDownStabilizationSeconds ...
{{- end }}
```

**cronjob.yaml** — gated by cronjob.enabled:
```yaml
{{- if .Values.cronjob.enabled }}
# ... CronJob with schedule, concurrencyPolicy, activeDeadlineSeconds from values ...
{{- end }}
```

**Verification commands:**
```bash
# Default render — CronJob absent
helm template inference-engine ./inference-engine \
  --set image.repository=registry.example.com/inference-engine | \
  grep kind
# Should NOT list CronJob

# Production render — CronJob present, HPA maxReplicas=30
helm template inference-engine ./inference-engine \
  --set image.repository=registry.example.com/inference-engine \
  -f values-production.yaml | \
  grep -E "kind:|maxReplicas"
```

---

## Rubric scoring guidance

**40% — independent implementation:**
- All 12 templates present and render without error: full marks.
- Missing HPA, CronJob, or RBAC templates: -5 points each.
- Broken template syntax (unclosed `{{-`, wrong indent): -3 points each.
- Value not wired from values.yaml (hardcoded in template): -2 points each occurrence.
- Missing `required` guard on `image.repository`: -4 points.

**25% — conceptual:**
- 10 questions × 2.5 points each.
- Partial credit (1.25 pts): answer is directionally correct but missing key detail.
- Zero credit: factually wrong or describes a different concept.

**20% — debugging:**
- Q3 (OOMKilled): 10 pts. Must name exit code 137, provide both kubectl commands.
- Day-5 Q3 (CrashLoopBackOff walkthrough): 10 pts. Must cover `logs --previous`, exit code decode, and escalation path.

**15% — code quality:**
- Pod security context has all 5 required fields for `restricted` (runAsNonRoot, runAsUser, fsGroup, seccompProfile, allowPrivilegeEscalation=false, capabilities.drop=ALL, readOnlyRootFilesystem=true): 5 pts.
- RBAC Role grants only configmaps (not wildcard `*`): 5 pts.
- No hardcoded values or secrets in any template: 5 pts.
