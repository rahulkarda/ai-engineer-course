# Solutions — Day 3: Services, Networking, and Namespaces

## Exercise 1 — ClusterIP Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: model-server
  namespace: default
spec:
  selector:
    app: model-server
  ports:
    - name: grpc
      port: 9000
      targetPort: 9000
  type: ClusterIP
```

DNS name from `default` namespace: `model-server` (short form) or `model-server.default.svc.cluster.local` (FQDN).

## Exercise 2 — Selector mismatch → empty endpoints

`kubectl get endpoints my-service` will show:
```
NAME         ENDPOINTS   AGE
my-service   <none>      5s
```

No pod IPs are listed because no pods match the selector `app: fastapi`. Pods have `app: fastapi-api`.

**Fix:** change the Service selector to match the actual pod labels:
```yaml
selector:
  app: fastapi-api
```
Or change the pod labels to `app: fastapi` — but this requires changing the Deployment template AND the Deployment selector simultaneously, which requires deleting and recreating the Deployment.

## Exercise 3 — Cross-namespace DNS

Full DNS name: `redis.data.svc.cluster.local`

The short name `redis` fails because CoreDNS resolves short names within the caller's namespace first. A pod in `backend` resolves `redis` as `redis.backend.svc.cluster.local` — which doesn't exist. Cross-namespace access requires at minimum `redis.data` (which works within the same cluster's search domain).

## Exercise 4 — NodePort vs LoadBalancer

Use **NodePort** for local development or bare-metal clusters where no cloud load balancer is available. It is simple but exposes a high port (30000–32767) on every node, which requires knowing node IPs and is not stable for production.

Use **LoadBalancer** in production on cloud providers — it provisions a cloud load balancer with a stable external IP/DNS. Main drawback of NodePort in production: each service needs a unique port, clients must know node IPs, nodes can change, and there is no TLS termination or path-based routing.

## Exercise 5 — Namespace and ResourceQuota

```yaml
# namespace-staging.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: staging
  labels:
    env: staging

---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: staging-quota
  namespace: staging
spec:
  hard:
    pods: "10"
    requests.cpu: "2"
    limits.cpu: "4"
```
