# Week 05 Assessment — Kubernetes Core

## Conceptual (interview-style)

Answer each in 2–4 sentences. Be precise.

1. Explain the role of **etcd** in a Kubernetes cluster. What happens if etcd data is lost without a backup?

2. What is the difference between a **liveness probe** and a **readiness probe**? What does Kubernetes do when each fails?

3. A pod is in `Pending` state for 10 minutes. List three different root causes and the kubectl command you would run to determine which one applies.

4. Explain **QoS classes** in Kubernetes. Which QoS class is a pod assigned when `requests.cpu = "100m"` and `limits.cpu = "500m"`? When would that pod be evicted relative to other pods?

5. You have a Service with `type: ClusterIP` and selector `app: api`. There are 3 pods with label `app: api`, but `kubectl get endpoints my-service` shows no endpoints. Name two possible causes.

6. Why is storing sensitive data in a Kubernetes Secret not truly secure by default? What production alternatives exist?

7. Describe the **reconciliation loop** pattern. How does a Deployment controller use it during a rolling update?

8. What is the purpose of `terminationGracePeriodSeconds` in a pod spec, and why does it matter for zero-downtime deployments?

## Coding challenge

### Specification

You are deploying a **model inference service** to Kubernetes. The service:
- Container image: `ghcr.io/myorg/model-server:3.0.0`
- Port: 9000
- Takes 45 seconds to load the model at startup
- Exposes `GET /health` (always 200 if process is alive) and `GET /ready` (200 once model is loaded)
- Requires env vars: `MODEL_NAME` (non-secret), `API_KEY` (secret)
- Should run 2 replicas normally, scale up to 6 under load
- Must never have 0 pods available during updates
- Needs `512Mi` memory guaranteed, up to `2Gi` allowed; `500m` CPU guaranteed, up to `2` CPU allowed

### Deliverables

1. A `Namespace` manifest named `inference`.
2. A `ConfigMap` named `model-config` in the `inference` namespace with `MODEL_NAME=claude-sonnet-5-5` and a multi-line `inference.json` config key with at least two fields.
3. A `Secret` manifest (with placeholder base64 values) for `API_KEY`. Include a comment explaining how to create it imperatively.
4. A `Deployment` manifest in the `inference` namespace with:
   - Correct label selectors
   - RollingUpdate strategy that never drops below 2 pods
   - startupProbe that gives the app 60 seconds to initialize
   - livenessProbe and readinessProbe on correct endpoints
   - Resources matching the spec above (requests = guaranteed values, limits = max values)
   - Config injected from the ConfigMap and Secret
5. A `ClusterIP` Service exposing the Deployment on port 80.
6. An `HPA` targeting 65% CPU utilization with the correct replica range.

### Acceptance criteria

- [ ] All six manifests are syntactically valid YAML
- [ ] Label selectors in Deployment and Service both resolve to the same pods
- [ ] Startup probe gives at least 60s before liveness probe can restart the pod
- [ ] Resources match the specification exactly (requests ≤ limits)
- [ ] HPA `scaleTargetRef` matches the Deployment name
- [ ] No real secret values in any file

## Rubric (pass ≥ 75%)

| Dimension | Weight | Criteria |
|-----------|--------|----------|
| Independent implementation | 40% | All 6 manifests written without reference; selectors correct; probes configured |
| Conceptual understanding | 25% | Conceptual questions answered accurately and concisely |
| Debugging & problem-solving | 20% | Correct diagnosis of Pending/endpoint/probe failure scenarios |
| Code quality & testing | 15% | Manifests pass `kubectl apply --dry-run=client`; no secrets in files; resources set |

Answer key: ../solutions/week-05/assessment.md
