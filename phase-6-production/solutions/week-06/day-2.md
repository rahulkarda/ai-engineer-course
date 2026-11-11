# Solutions — Week 06 · Day 2 (Ingress & Controllers)

## Exercise 1 — Multi-host Ingress with TLS and cert-manager

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: multi-host-ingress
  namespace: production
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - api.example.com
      secretName: api-example-com-tls
    - hosts:
        - app.example.com
      secretName: app-example-com-tls
  rules:
    - host: api.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: ai-api
                port:
                  number: 80
    - host: app.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: ai-ui
                port:
                  number: 80
```

## Exercise 2 — pathType comparison

| pathType | `/api` | `/api/v1/users` | `/apiv2` |
|---|---|---|---|
| `Exact` | matches | no match | no match |
| `Prefix` | matches | matches | no match |
| `ImplementationSpecific` | controller-specific | controller-specific | controller-specific |

`Prefix` is evaluated after stripping a trailing `/`. Path `/api` as a Prefix matches `/api`, `/api/`, `/api/v1/users`. **`Prefix` would match `/api/v1/users`.**

`ImplementationSpecific` allows regex in ingress-nginx but is non-portable.

## Exercise 3 — 502 diagnostic sequence

```bash
# Step 1: check Ingress rule and events
kubectl describe ingress <name> -n production
# Look for: "Backends" section shows service name and port
# Look for: Events showing warning/errors

# Step 2: check the Service exists and selector matches pods
kubectl get svc ai-api -n production
kubectl get endpoints ai-api -n production
# Empty endpoints = selector mismatch or no Ready pods

# Step 3: verify pods are ready
kubectl get pods -n production -l app=ai-api
# If no pods: deploy missing. If pods NotReady: probe failing.

# Step 4: check pod readiness probe
kubectl describe pod <pod-name> -n production | grep -A10 Readiness

# Step 5: controller logs
kubectl logs -n ingress-nginx \
  -l app.kubernetes.io/name=ingress-nginx \
  --tail=100 | grep -E "502|upstream"
# Look for: "upstream closed connection" or "no live upstreams"
```

Root cause of 502: almost always the backend pod is unhealthy (not in Endpoints) or the container port does not match `service.targetPort`.

## Exercise 4 — Proxy timeout and body size annotations

```yaml
metadata:
  annotations:
    nginx.ingress.kubernetes.io/proxy-read-timeout: "30"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "30"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
```

Note: `proxy-read-timeout` applies to reading the response from the upstream; `proxy-send-timeout` covers sending the request body. Both are in seconds. `proxy-body-size: "0"` disables the limit entirely (use with caution for streaming).

## Exercise 5 — IngressClass with controller.ingressClassByName=true

**Scenario:** A company has two teams:
- Team A runs a public API requiring NGINX with WAF annotations.
- Team B runs internal tooling behind an auth-proxy Ingress.

Without explicit IngressClass scoping, both teams' Ingress resources would be picked up by the single cluster-default controller. By deploying two ingress-nginx controllers — each with a unique `ingressClassName` (e.g., `public-nginx` and `internal-nginx`) and `controller.ingressClassByName=true` — each controller only processes Ingress resources that reference its specific class. This enables:
- Independent controller configuration (different NGINX configs, replicas, rate limits).
- Network policy segmentation (public controller in a DMZ node pool).
- Team isolation without namespace separation.

`controller.ingressClassByName=true` tells the controller to use its own name as the class name filter rather than the annotation `kubernetes.io/ingress.class`. It must be set when running multiple ingress-nginx instances to prevent both controllers from fighting over the same Ingress resources.
