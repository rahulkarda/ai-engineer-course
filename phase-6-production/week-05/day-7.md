# Phase 6 · Week 05 · Day 7 — Week Project: Kubernetes Deploy Runbook

> Objective: Finalize the week project (manifests + runbook), then complete the weekly assessment.  ·  Time: ~3 hr project + 1 hr assessment  ·  Prereqs: [[day-6]]

## Project summary

See [[project.md]] for full specification. Today you write the **operations runbook** and finalize the manifest set.

## Part 1 — Finalize manifests (45 min)

Review all manifests from Day 6 against this checklist:

```
□ Namespace created with labels
□ ConfigMap holds all non-secret config
□ Secret created imperatively (no plaintext in YAML file)
□ Deployment:
    □ replicas: 3
    □ selector matches pod template labels exactly
    □ RollingUpdate: maxSurge=1, maxUnavailable=0
    □ startupProbe, livenessProbe, readinessProbe configured
    □ requests AND limits set on every container
    □ terminationGracePeriodSeconds set
    □ imagePullPolicy: IfNotPresent (for local kind)
□ Service: ClusterIP, named targetPort, correct selector
□ HPA: minReplicas=2, maxReplicas=10, CPU target set
□ All resources in correct namespace
□ All manifests validate with kubectl apply --dry-run=server
```

## Part 2 — Write the operations runbook (60 min)

A runbook is a concise operational guide for your app. Write it as `k8s/RUNBOOK.md`:

### Runbook template

```markdown
# FastAPI App — Kubernetes Operations Runbook

## Cluster info
- Cluster: ai-platform (kind, local)
- Namespace: ai-platform
- Context: kind-ai-platform

## Deploy

### First-time setup
\```bash
kind create cluster --config kind-config.yaml
kubectl config use-context kind-ai-platform
kind load docker-image ghcr.io/myorg/fastapi-app:1.0.0 --name ai-platform
\```

### Apply all manifests
\```bash
kubectl apply -f k8s/namespace.yaml
kubectl create secret generic fastapi-secrets \
  --from-literal=DATABASE_URL="${DATABASE_URL}" \
  --from-literal=JWT_SECRET_KEY="${JWT_SECRET_KEY}" \
  --namespace=ai-platform
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl rollout status deployment/fastapi-app -n ai-platform
\```

## Rolling update

### Update image
\```bash
kubectl set image deployment/fastapi-app \
  api=ghcr.io/myorg/fastapi-app:<NEW_VERSION> \
  -n ai-platform
kubectl rollout status deployment/fastapi-app -n ai-platform
\```

### Rollback
\```bash
kubectl rollout undo deployment/fastapi-app -n ai-platform
kubectl rollout status deployment/fastapi-app -n ai-platform
\```

## Health checks

### Quick status
\```bash
kubectl get all -n ai-platform
kubectl get endpoints fastapi-service -n ai-platform
\```

### Pod logs
\```bash
# All pods
kubectl logs -l app=fastapi -n ai-platform --tail=50

# Specific pod
kubectl logs <pod-name> -n ai-platform

# Previous container (after crash)
kubectl logs <pod-name> -n ai-platform --previous
\```

### Port forward for local testing
\```bash
kubectl port-forward service/fastapi-service 8080:80 -n ai-platform
curl http://localhost:8080/health
curl http://localhost:8080/ready
\```

## Troubleshooting

| Symptom | First command | Likely cause |
|---------|---------------|--------------|
| Pod stuck Pending | `kubectl describe pod <name> -n ai-platform` | Insufficient resources, no matching nodes |
| Pod CrashLoopBackOff | `kubectl logs <name> --previous -n ai-platform` | App crash at startup |
| Pod not receiving traffic | `kubectl get endpoints fastapi-service -n ai-platform` | Readiness failing or selector mismatch |
| Rollout stuck | `kubectl rollout status deployment/fastapi-app -n ai-platform` | New pods not becoming ready |
| Secret missing | `kubectl describe secret fastapi-secrets -n ai-platform` | Secret not created or wrong namespace |

## Cleanup
\```bash
kubectl delete namespace ai-platform
kind delete cluster --name ai-platform
\```
```

## Part 3 — Assessment review (30 min)

Before taking the assessment, review:

- [ ] Control plane components and their roles (Day 1)
- [ ] Pod lifecycle states and the difference between RS and Deployment (Day 2)
- [ ] Service types and DNS resolution patterns (Day 3)
- [ ] ConfigMap vs Secret, label selectors (Day 4)
- [ ] Probe types, resource QoS classes, rolling update params (Day 5)
- [ ] Full manifest set structure (Day 6)

## Part 4 — Weekly Assessment (45 min)

See [[assessment.md]].

## Exercises

1. A new engineer asks: "Should I commit the Secret YAML with real base64 values to git?" Write a 3-sentence explanation of why not and what to do instead.

2. Your rollout is stuck at 2/3 new pods running and 1 old pod not terminating. List the kubectl commands you run to diagnose, in order.

3. Write the kubectl command to perform a dry-run diff of all files in `k8s/` against the current cluster state.

4. Add a `LimitRange` manifest to the `ai-platform` namespace that sets default CPU request `100m` and memory request `128Mi` for any container that doesn't specify them.

5. Explain what happens step-by-step when you run `kubectl delete namespace ai-platform`. What is deleted, in what order, and which Kubernetes component is responsible for cleanup?

(Solutions: ../solutions/week-05/day-7.md)

## Self-check

- [ ] I have a complete manifest set that passes `--dry-run=server`
- [ ] I wrote a runbook covering deploy, update, rollback, and troubleshooting
- [ ] I can diagnose the 5 most common pod failure modes
- [ ] I am ready for the week 5 assessment

## Supplementary links

- [LimitRange](https://kubernetes.io/docs/concepts/policy/limit-range/)
- [Debugging Pods](https://kubernetes.io/docs/tasks/debug/debug-application/debug-pods/)
- [kubectl cheatsheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)
