# Week 06 Assessment — Kubernetes Advanced & Operations

> Time: ~1 hr  ·  Pass threshold: 75%

## Conceptual (interview-style)

1. Explain the HPA control loop. What formula does it use to compute desired replicas? What is the effect of the scale-down stabilization window?

2. What is the difference between `pathType: Prefix` and `pathType: Exact` in a Kubernetes Ingress rule? Which pathType would match `/api/v2` when the rule path is `/api`?

3. Explain what makes a StatefulSet different from a Deployment in terms of: pod identity, storage, and scale-down order. In what scenarios should you use a StatefulSet?

4. A CronJob has `concurrencyPolicy: Forbid` and runs a 3-hour job. The job is scheduled at 02:00 and 03:00. What happens at the 03:00 tick?

5. What is the reconciliation loop in Argo CD? What does `selfHeal: true` do? When would you set it to `false`?

6. Explain the difference between a `Role` and a `ClusterRole`. Give a concrete example of when you would use each.

7. What exit code does a container receive when it is OOMKilled? How do you verify this with kubectl?

8. The Pod Security Standard `restricted` profile requires `allowPrivilegeEscalation: false`. Why is this important for a containerized AI API that runs as root by default?

9. Describe the sequence of events when `helm upgrade --atomic` is used and one of the Deployment pods fails its readiness probe.

10. What is the purpose of the `checksum/config` annotation pattern in a Deployment template? What problem does it solve?

---

## Coding challenge

### Scenario

Your team runs an AI inference API on Kubernetes. You need to deliver the following in a single Helm chart named `inference-engine`:

**Requirements:**

1. **Deployment**: 2 default replicas. Container image `registry.example.com/inference-engine` (placeholder), tag from values (falls back to `Chart.AppVersion`). ConfigMap-backed env vars: `BATCH_SIZE` (default `"32"`) and `LOG_LEVEL` (default `"INFO"`). Pod must pass `restricted` Pod Security Standard.

2. **HPA**: Enabled by default. Min 2, max 10 replicas. Target 70% CPU utilization. Scale-down stabilization: 180 seconds.

3. **Ingress**: Enabled by default. Host `inference.example.com` (placeholder). Path `/` Prefix. TLS using `cert-manager` annotation `cert-manager.io/cluster-issuer: letsencrypt-prod`.

4. **CronJob**: Runs every Sunday at 03:00 UTC. Command `["python", "-m", "eval.weekly"]`. Image same repository as main image. Disabled via `cronjob.enabled=false`. `concurrencyPolicy: Forbid`. Hard cap 2 hours.

5. **RBAC**: ServiceAccount created by the chart. Role grants `get` and `list` on `configmaps` only.

6. **Values overrides**: Provide a `values-production.yaml` with 5 min replicas, 30 max replicas, `LOG_LEVEL=WARNING`, CronJob enabled, and `secrets.enabled=true` with `secrets.name=inference-prod-secrets`.

**Deliverables:**
- `inference-engine/Chart.yaml`
- `inference-engine/values.yaml`
- `inference-engine/values-production.yaml`
- `inference-engine/templates/_helpers.tpl` (fullname, labels, selectorLabels, serviceAccountName)
- `inference-engine/templates/deployment.yaml`
- `inference-engine/templates/hpa.yaml`
- `inference-engine/templates/ingress.yaml`
- `inference-engine/templates/cronjob.yaml`
- `inference-engine/templates/serviceaccount.yaml`
- `inference-engine/templates/role.yaml`
- `inference-engine/templates/rolebinding.yaml`
- `inference-engine/templates/configmap.yaml`

**Acceptance criteria:**
- `helm template inference-engine ./inference-engine` renders without error.
- `helm template inference-engine ./inference-engine -f values-production.yaml` renders without error.
- CronJob is absent in the default render; present in the production render.
- No real credentials or secrets in any file.
- Pod passes `restricted` profile (demonstrate with security context fields).

---

## Rubric (pass ≥ 75%)

| Dimension | Weight |
|---|---|
| Independent implementation (chart renders, all resources present, values wired correctly) | 40% |
| Conceptual understanding (all 10 conceptual answers substantially correct) | 25% |
| Debugging & problem-solving (troubleshooting questions and on-call scenarios) | 20% |
| Code quality & testing (RBAC least-privilege, pod security, no secrets, clean templates) | 15% |

**Scoring guide:**
- 40% implementation: award 4 points per template (12 templates × ~3.3 = 40); deduct for: missing resource, broken template syntax, hardcoded values, no `required` guard on `image.repository`.
- 25% conceptual: ~2.5 points per question; partial credit for partially correct answers.
- 20% debugging: full marks for correct root cause + correct kubectl command in exercises; half marks for correct cause without command.
- 15% code quality: 5 pts for pod security context correct; 5 pts for RBAC restricted to configmaps only; 5 pts for no hardcoded values/secrets.

Answer key: ../solutions/week-06/assessment.md
