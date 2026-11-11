# Solutions — Week 06 · Day 5 (GitOps, RBAC, Pod Security, Debugging)

## Exercise 1 — Argo CD Application for staging (no auto-sync)

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: ai-platform-staging
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
        - values-staging.yaml
  destination:
    server: https://kubernetes.default.svc
    namespace: staging
  syncPolicy:
    automated: null          # no automated sync — manual approval required
    syncOptions:
      - CreateNamespace=true
    retry:
      limit: 2
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 1m
```

With `automated` omitted (or set to `null`), Argo CD will detect `OutOfSync` state but require a human to click "Sync" in the UI or run `argocd app sync ai-platform-staging`. This is the standard pattern for staging: the team reviews the diff before applying.

## Exercise 2 — job-manager Role with RoleBinding

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: job-manager
  namespace: production
rules:
  - apiGroups: ["batch"]
    resources: ["jobs", "cronjobs"]
    verbs: ["create", "delete", "get", "list", "watch", "patch", "update"]
  # Deliberately no Deployments, StatefulSets — omitting them denies access
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: batch-runner
  namespace: production
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: job-manager-binding
  namespace: production
subjects:
  - kind: ServiceAccount
    name: batch-runner
    namespace: production
roleRef:
  kind: Role
  name: job-manager
  apiGroup: rbac.authorization.k8s.io
```

Verification:
```bash
kubectl auth can-i create jobs \
  --as=system:serviceaccount:production:batch-runner -n production
# yes

kubectl auth can-i delete deployments \
  --as=system:serviceaccount:production:batch-runner -n production
# no
```

## Exercise 3 — Exit code 137 diagnosis

Exit code 137 = container received SIGKILL, which in Kubernetes context means **OOMKilled** (Out Of Memory killed by the kernel).

```bash
# Step 1: Confirm exit code
kubectl get pod <pod-name> -n production \
  -o jsonpath='{.status.containerStatuses[0].lastState.terminated.exitCode}'
# Returns: 137

# Step 2: Confirm OOM reason
kubectl get pod <pod-name> -n production \
  -o jsonpath='{.status.containerStatuses[0].lastState.terminated.reason}'
# Returns: OOMKilled

# Evidence from describe:
kubectl describe pod <pod-name> -n production
# Look for: "OOMKilled" in State/Last State, or "Limits" showing memory
```

Fix: increase `resources.limits.memory` or profile memory usage (add `-Xmx` for JVM, adjust Python worker pool size for Python apps).

## Exercise 4 — Namespace PSS labels

```bash
kubectl label namespace development \
  pod-security.kubernetes.io/enforce=restricted \
  pod-security.kubernetes.io/warn=baseline \
  pod-security.kubernetes.io/audit=baseline
```

**What happens when a developer deploys a pod with `privileged: true`:**

The `privileged: true` field violates the `restricted` profile (and even `baseline`). The admission controller:
1. **Rejects** the pod at creation time with a clear error message referencing the violated policy.
2. Also emits an audit log entry and a warning (per the `warn=baseline` label, though `restricted` enforcement would already catch it).

The developer sees: `Error from server (Forbidden): pods "myapp-..." is forbidden: violates PodSecurity "restricted:latest": privileged (container "myapp" must not set securityContext.privileged=true), ...`

The pod is never created. This is the correct behavior — the error surfaces at deploy time, not at runtime.

## Exercise 5 — GitOps promotion pattern

**Pattern: Two Application resources (one per environment)**

```
Git repo layout:
charts/
  ai-platform/
    values.yaml
    values-staging.yaml
    values-production.yaml
```

1. Developer opens a PR → CI runs `helm lint`, `helm template`, unit tests.
2. PR merged to `main` → Argo CD auto-syncs `ai-platform-staging` (detects Git change).
3. QA verifies staging. Team lead runs `argocd app sync ai-platform-production` (or approves via UI) — production has `automated: null`.

**Pattern: ApplicationSet with environment generator**

```yaml
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: ai-platform
  namespace: argocd
spec:
  generators:
    - list:
        elements:
          - env: staging
            namespace: staging
            autoSync: "true"
          - env: production
            namespace: production
            autoSync: "false"
  template:
    metadata:
      name: "ai-platform-{{env}}"
    spec:
      source:
        repoURL: https://github.com/example/ai-platform-infra.git
        path: charts/ai-platform
        helm:
          valueFiles: ["values-{{env}}.yaml"]
      destination:
        namespace: "{{namespace}}"
      syncPolicy:
        automated:
          selfHeal: "{{autoSync}}"
          prune: "{{autoSync}}"
```

This single ApplicationSet manages both environments. Staging gets full auto-sync; production requires manual approval. Promotion means the production `values-production.yaml` is updated (e.g., `image.tag` bumped) in a PR, reviewed, merged, and then manually synced.
