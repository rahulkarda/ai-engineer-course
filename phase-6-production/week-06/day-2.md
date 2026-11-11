# Phase 6 · Week 06 · Day 2 — Ingress & Ingress Controllers

> Objective: Understand Ingress resources, deploy an NGINX ingress controller, configure TLS and path-based routing.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

## Concept

A Kubernetes **Service** of type `ClusterIP` or `NodePort` exposes a single backend. **Ingress** is the API object that expresses HTTP(S) routing rules — host-based, path-based, or both — at Layer 7, without needing a separate LoadBalancer per service. This matters for cost (one cloud LB instead of N) and for TLS termination.

Ingress itself is just a rule set. The **Ingress Controller** is the component that reads those rules and enforces them — it watches `Ingress` resources via the Kubernetes API and reconfigures the underlying proxy (NGINX, Envoy, HAProxy, Traefik, etc.).

Common ingress controllers:

| Controller | Underlying proxy | Notes |
|---|---|---|
| ingress-nginx (community) | NGINX | Most common in self-managed clusters |
| AWS Load Balancer Controller | ALB | Integrates with AWS ACM/WAF |
| Traefik | Traefik | Operator-friendly, built-in dashboard |
| Istio Gateway | Envoy | Service mesh; richer traffic policy |
| Kong | NGINX/Kong | Plugin ecosystem |

## How it works internally

1. Controller pod runs in-cluster with a ServiceAccount that has RBAC `get/list/watch` on `Ingress`, `Service`, `Endpoints`, `Secret`.
2. On each Ingress change, the controller reconciles its internal proxy config (e.g., regenerates `nginx.conf`, calls reload or hot-reload).
3. A `LoadBalancer`-type Service fronts the controller pods; the cloud provider allocates an external IP/hostname for it.
4. TLS: the controller reads `kubernetes.io/tls` Secrets (cert + key) referenced in `spec.tls`. cert-manager automates certificate issuance from Let's Encrypt or an internal CA.

**Annotations** are the escape hatch for controller-specific settings (`nginx.ingress.kubernetes.io/proxy-body-size`, `rewrite-target`, etc.). They are per-resource and controller-specific — not portable.

**IngressClass** (GA in 1.18+) replaces the deprecated `kubernetes.io/ingress.class` annotation and enables multiple controller deployments in one cluster.

## Worked example

### 1. Install ingress-nginx (Helm)

```bash
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update

helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace \
  --set controller.replicaCount=2 \
  --set controller.resources.requests.cpu=100m \
  --set controller.resources.requests.memory=128Mi
```

```bash
# Wait for external IP
kubectl get svc -n ingress-nginx ingress-nginx-controller -w
```

### 2. Services to route to

```yaml
# services.yaml
apiVersion: v1
kind: Service
metadata:
  name: ai-api
  namespace: production
spec:
  selector:
    app: ai-api
  ports:
    - port: 80
      targetPort: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: ai-ui
  namespace: production
spec:
  selector:
    app: ai-ui
  ports:
    - port: 80
      targetPort: 3000
```

### 3. Ingress resource — path-based routing

```yaml
# ingress-api.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ai-platform-ingress
  namespace: production
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "120"
    nginx.ingress.kubernetes.io/enable-cors: "true"
spec:
  ingressClassName: nginx       # matches the IngressClass installed by ingress-nginx
  tls:
    - hosts:
        - ai.example.com
      secretName: ai-example-com-tls   # must exist in same namespace
  rules:
    - host: ai.example.com
      http:
        paths:
          - path: /api
            pathType: Prefix
            backend:
              service:
                name: ai-api
                port:
                  number: 80
          - path: /
            pathType: Prefix
            backend:
              service:
                name: ai-ui
                port:
                  number: 80
```

### 4. TLS Secret (manual; use cert-manager in production)

```bash
# Generate self-signed cert for local testing
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout tls.key \
  -out tls.crt \
  -subj "/CN=ai.example.com/O=example"

kubectl create secret tls ai-example-com-tls \
  --key tls.key \
  --cert tls.crt \
  -n production
```

### 5. cert-manager for automated TLS

```bash
helm repo add jetstack https://charts.jetstack.io
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --set crds.enabled=true
```

```yaml
# cluster-issuer-letsencrypt.yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: ops@example.com          # placeholder — replace with real ops email
    privateKeySecretRef:
      name: letsencrypt-prod-key
    solvers:
      - http01:
          ingress:
            ingressClassName: nginx
```

```yaml
# Ingress with cert-manager annotation
metadata:
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
    - hosts:
        - ai.example.com
      secretName: ai-example-com-tls   # cert-manager creates/renews this Secret
```

### 6. Ingress with rewrite rule

```yaml
# Route /inference → ai-api/v1/inference
metadata:
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /v1/inference/$2
spec:
  rules:
    - host: ai.example.com
      http:
        paths:
          - path: /inference(/|$)(.*)
            pathType: ImplementationSpecific
            backend:
              service:
                name: ai-api
                port:
                  number: 80
```

### 7. Debugging Ingress

```bash
# Check Ingress state
kubectl describe ingress ai-platform-ingress -n production

# Controller logs
kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx --tail=50

# Verify backend endpoints are populated
kubectl get endpoints ai-api -n production

# Test routing from inside cluster
kubectl run curl-test --image=curlimages/curl --rm -it --restart=Never -- \
  curl -H "Host: ai.example.com" http://<controller-cluster-ip>/api/health
```

## Exercises

1. Write an Ingress manifest that routes `api.example.com` to `ai-api:80` and `app.example.com` to `ai-ui:80`, with TLS for both hosts using cert-manager and the `letsencrypt-prod` ClusterIssuer.

2. An Ingress rule has `pathType: Exact` for path `/api`. Explain the difference between `Exact`, `Prefix`, and `ImplementationSpecific`. Which pathType would match `/api/v1/users`?

3. The NGINX ingress controller returns a 502 for a valid path. Walk through the full diagnostic sequence from `kubectl describe ingress` to checking Endpoints to controller logs.

4. Write an annotation to set a 30-second proxy timeout and a 10 MB maximum body size on an ingress for an AI API that streams large payloads.

5. Why would you use an `IngressClass` with `controller.ingressClassByName=true` instead of the default class? Give a realistic multi-team scenario.

(Solutions: ../solutions/week-06/day-2.md)

## Self-check

- [ ] I can explain the difference between a Service and an Ingress.
- [ ] I can install ingress-nginx and create a working path-based and host-based Ingress.
- [ ] I can configure TLS manually and via cert-manager.
- [ ] I can diagnose a 502/404 from an Ingress rule.
- [ ] I understand IngressClass and when to use multiple controllers.

## Supplementary links

- [Ingress concepts](https://kubernetes.io/docs/concepts/services-networking/ingress/)
- [ingress-nginx documentation](https://kubernetes.github.io/ingress-nginx/)
- [cert-manager documentation](https://cert-manager.io/docs/)
- [Ingress controllers comparison](https://kubernetes.io/docs/concepts/services-networking/ingress-controllers/)
