# Phase 6 · Week 06 · Day 7 (Sunday) — Week 06 Project

> Objective: Deliver the complete AI Platform Helm chart and operations runbook as a shippable artefact.  ·  Time: ~3 hr build + 1 hr assessment  ·  Prereqs: [[day-6]]

## Project brief

You are the platform engineer responsible for a production AI Platform. The team has been deploying with raw `kubectl apply` and manual YAML files. Your job is to:

1. Package the existing workload as a Helm chart with configurable values.
2. Add autoscaling, Ingress, and a nightly CronJob.
3. Harden the chart with proper RBAC, Pod Security, and secret handling.
4. Write an operations runbook covering the four most common on-call scenarios.
5. Wire up a basic Argo CD `Application` for GitOps delivery.

This is a design and delivery day — you do not run a cluster, but every YAML must be valid and every command must be executable in a real cluster.

---

## Requirements

### Helm chart

| Feature | Requirement |
|---|---|
| Deployment | Image parameterized; configmap envFrom; `restricted` pod security |
| Service | ClusterIP; port from values |
| Ingress | Host/path from values; TLS secretName from values; cert-manager annotation |
| HPA | CPU + memory targets; configurable min/max; scale-down stabilization |
| CronJob | Schedule, command, image, resources all in values; `Forbid` concurrency default |
| ServiceAccount + Role | Least-privilege: only `get/list configmaps` and `get` of a named secret |
| ConfigMap | LOG_LEVEL, MODEL_TIMEOUT_MS from values |
| values.yaml | Full defaults; `required` guards on `image.repository` and ingress host |
| values-staging.yaml | 1 replica, HPA max 5, debug log level, CronJob disabled |
| values-production.yaml | 4 replicas, HPA max 40, secrets.enabled, INFO log level |

### Argo CD Application

- Targets production namespace.
- `automated` sync with `selfHeal: true` and `prune: true`.
- Points at `charts/ai-platform` in a Git repo.
- Uses `values-production.yaml`.

### Operations runbook

Must cover (each as a numbered procedure with exact kubectl commands):

1. First install (namespace, secrets, helm install, verify).
2. Image upgrade (helm upgrade --atomic, verify rollout, confirm HPA still active).
3. Rollback (list history, rollback, verify).
4. On-call: CrashLoopBackOff (logs, exit code, fix and redeploy).
5. On-call: HPA stuck at minReplicas with high CPU (diagnose metrics pipeline).

---

## Deliverables

```
week-06-project/
├── ai-platform/
│   ├── Chart.yaml
│   ├── values.yaml
│   ├── values-staging.yaml
│   ├── values-production.yaml
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
├── argocd-app.yaml
└── runbook.md
```

---

## Acceptance criteria

- [ ] `helm template ai-platform ./ai-platform -f values-staging.yaml` renders without errors.
- [ ] `helm template ai-platform ./ai-platform -f values-production.yaml` renders without errors.
- [ ] HPA `minReplicas` and `maxReplicas` differ between staging and production values.
- [ ] CronJob is absent in the staging render (disabled via values).
- [ ] No real secrets or credentials appear anywhere in the chart files.
- [ ] Pod spec passes `restricted` Pod Security Standard (runAsNonRoot, no privilege escalation, capabilities dropped, readOnlyRootFilesystem).
- [ ] Argo CD Application `automated` block has both `selfHeal` and `prune` set to `true`.
- [ ] Runbook includes exact kubectl commands for each procedure (no placeholders like "run the command").
- [ ] All five runbook procedures are present and complete.

---

## Grading (self-assessment)

| Dimension | Weight | Your score |
|---|---|---|
| Chart renders correctly for both environments | 25% | /25 |
| Security: RBAC, pod security, no leaked secrets | 25% | /25 |
| HPA and CronJob correctly templated and gated | 20% | /20 |
| Argo CD Application complete and correct | 15% | /15 |
| Runbook: all 5 procedures with exact commands | 15% | /15 |

Pass threshold: 75%.

---

## Notes

- Keep `values.yaml` as the single source of truth for defaults. Never hardcode values in templates.
- The `checksum/config` annotation on the Deployment causes a rolling restart when the ConfigMap changes — include it.
- For the runbook, every step must be independently executable by an on-call engineer with no context from the previous step.
- Do not store the operations runbook in a location that requires cluster access to read — it should live in the Git repo alongside the chart.

(Project solution: ../solutions/week-06/project.md)
