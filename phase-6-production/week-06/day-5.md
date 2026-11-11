# Phase 6 · Week 06 · Day 5 — GitOps (Argo CD), RBAC, Pod Security & Debugging

> Objective: Understand GitOps with Argo CD, configure RBAC, apply Pod Security Standards, and master pod debugging techniques.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

## Concept

### GitOps
GitOps treats Git as the single source of truth for cluster state. An operator continuously reconciles the cluster to match what is declared in Git. The key properties:
- **Declarative**: desired state is files in Git, not imperative `kubectl apply` commands.
- **Versioned**: every change has a commit; rollback = revert the commit.
- **Auditable**: who changed what, when, why — from `git log`.
- **Continuously reconciled**: if someone manually patches a resource, the operator reverts it.

**Argo CD** is the dominant GitOps operator for Kubernetes. It watches a Git repo (or OCI registry) and syncs `Application` resources to one or more clusters.

### RBAC
Kubernetes RBAC (Role-Based Access Control) governs which subjects (users, groups, ServiceAccounts) can perform which verbs (get, list, watch, create, delete, patch) on which resources.

- `Role` / `ClusterRole`: the permission set.
- `RoleBinding` / `ClusterRoleBinding`: binds a subject to a role.

### Pod Security
**Pod Security Standards** (PSS) replaced PodSecurityPolicy in 1.25. Three built-in profiles enforced by admission: `privileged`, `baseline`, `restricted`. Applied via namespace labels.

## How it works internally

**Argo CD** architecture:
- **API server**: exposes REST/gRPC API and the web UI.
- **Application controller**: core reconciliation loop — compares desired state (Git) with live state (cluster) and computes diff.
- **Repo server**: clones and caches Git repos; renders Helm/Kustomize/plain-YAML manifests.
- **Redis**: caches rendered manifests and cluster state.

Sync loop: repo server polls Git (or receives a webhook), renders manifests, API server diffed against live resources, if `OutOfSync` → optionally auto-sync.

**RBAC in Kubernetes**: the API server intercepts every request, runs it through authentication (who is this?) → authorization (are they allowed?) → admission controllers (is it valid?). RBAC is the authorization layer. ServiceAccount tokens are the standard in-cluster identity.

## Worked example

### 1. Argo CD installation

```bash
kubectl create namespace argocd
kubectl apply -n argocd \
  -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Access UI (port-forward for local)
kubectl port-forward svc/argocd-server -n argocd 8080:443

# Get initial admin password
kubectl get secret argocd-initial-admin-secret -n argocd \
  -o jsonpath='{.data.password}' | base64 -d
```

### 2. Argo CD Application pointing at Helm chart in Git

```yaml
# argocd-app.yaml
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
      parameters:
        - name: image.tag
          value: "1.3.0"    # override; usually set by CI pipeline
  destination:
    server: https://kubernetes.default.svc    # in-cluster
    namespace: production
  syncPolicy:
    automated:
      prune: true        # delete resources removed from Git
      selfHeal: true     # revert manual cluster changes
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

```bash
kubectl apply -f argocd-app.yaml
# Watch sync
kubectl get app ai-platform-production -n argocd -w
# Force sync
argocd app sync ai-platform-production
```

### 3. Kubernetes RBAC — least-privilege ServiceAccount for the API

```yaml
# rbac-api.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ai-api-sa
  namespace: production
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: ai-api-role
  namespace: production
rules:
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["get", "list"]
  - apiGroups: [""]
    resources: ["secrets"]
    resourceNames: ["ai-secrets"]   # restrict to one named Secret
    verbs: ["get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: ai-api-rolebinding
  namespace: production
subjects:
  - kind: ServiceAccount
    name: ai-api-sa
    namespace: production
roleRef:
  kind: Role
  name: ai-api-role
  apiGroup: rbac.authorization.k8s.io
```

```yaml
# ClusterRole for a read-only ops engineer
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: cluster-viewer
rules:
  - apiGroups: ["*"]
    resources: ["*"]
    verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: cluster-viewer-binding
subjects:
  - kind: User
    name: alice@example.com   # placeholder
    apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: ClusterRole
  name: cluster-viewer
  apiGroup: rbac.authorization.k8s.io
```

```bash
# Verify what a ServiceAccount can do
kubectl auth can-i get secrets \
  --as=system:serviceaccount:production:ai-api-sa -n production
kubectl auth can-i delete pods \
  --as=system:serviceaccount:production:ai-api-sa -n production
```

### 4. Pod Security Standards

```bash
# Label a namespace with the 'restricted' profile
kubectl label namespace production \
  pod-security.kubernetes.io/enforce=restricted \
  pod-security.kubernetes.io/warn=restricted \
  pod-security.kubernetes.io/audit=restricted
```

Container spec adjustments for `restricted` profile:

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  runAsGroup: 3000
  fsGroup: 2000
  seccompProfile:
    type: RuntimeDefault
containers:
  - name: api
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop: ["ALL"]
    volumeMounts:
      - name: tmp
        mountPath: /tmp    # needed if readOnlyRootFilesystem=true
volumes:
  - name: tmp
    emptyDir: {}
```

### 5. Debugging pods — the full toolkit

```bash
# 1. Pod events and status
kubectl describe pod <pod-name> -n production

# 2. Container logs (current)
kubectl logs <pod-name> -n production -c api
# Previous container (after crash)
kubectl logs <pod-name> -n production -c api --previous

# 3. Shell into running container
kubectl exec -it <pod-name> -n production -- /bin/sh

# 4. Ephemeral debug container (no shell in image)
kubectl debug -it <pod-name> -n production \
  --image=busybox \
  --target=api \
  --copy-to=debug-pod

# 5. Cluster events
kubectl get events -n production --sort-by='.lastTimestamp'
kubectl get events -n production --field-selector reason=BackOff

# 6. Check resource usage
kubectl top pod <pod-name> -n production --containers

# 7. Port-forward to test directly
kubectl port-forward pod/<pod-name> 8080:8000 -n production

# 8. Decode a CrashLoopBackOff
# - Check exit code in `describe pod` → ContainerStatuses.lastState.terminated.exitCode
# - exit 1 → application error; exit 137 → OOMKilled; exit 143 → SIGTERM
```

```bash
# Network debugging
kubectl run netshoot --rm -it --image=nicolaka/netshoot --restart=Never -- \
  curl -v http://ai-api.production.svc.cluster.local/health
```

## Exercises

1. Write an Argo CD `Application` manifest for a Helm chart at `charts/ai-platform` in a Git repo, targeting namespace `staging`, with `selfHeal: false` and `prune: false` (manual approval for staging).

2. Create a `Role` named `job-manager` in namespace `production` that allows creating and deleting `Jobs` and `CronJobs` but cannot touch Deployments or StatefulSets. Bind it to a ServiceAccount `batch-runner`.

3. A pod's container exits with code 137 repeatedly. Identify the cause and the two kubectl commands you run to confirm it and to get the evidence.

4. Label the `development` namespace with `restricted` enforce policy and `baseline` warn policy. Explain what happens when a developer deploys a pod with `privileged: true`.

5. Describe the GitOps promotion pattern: how does a change move from a developer's branch to staging to production using Argo CD `ApplicationSet` or separate `Application` resources?

(Solutions: ../solutions/week-06/day-5.md)

## Self-check

- [ ] I can describe the GitOps reconciliation loop and what `selfHeal` does.
- [ ] I can write a Role, ClusterRole, RoleBinding, and verify permissions with `kubectl auth can-i`.
- [ ] I can apply Pod Security Standards to a namespace and fix a pod to pass `restricted`.
- [ ] I can walk through every step of debugging a CrashLoopBackOff.
- [ ] I can use `kubectl debug` ephemeral containers for distroless images.

## Supplementary links

- [Argo CD documentation](https://argo-cd.readthedocs.io/en/stable/)
- [Kubernetes RBAC](https://kubernetes.io/docs/reference/access-authn-authz/rbac/)
- [Pod Security Standards](https://kubernetes.io/docs/concepts/security/pod-security-standards/)
- [Debugging pods](https://kubernetes.io/docs/tasks/debug/debug-application/debug-pods/)
- [Ephemeral containers](https://kubernetes.io/docs/concepts/workloads/pods/ephemeral-containers/)
