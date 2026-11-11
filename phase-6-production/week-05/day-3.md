# Phase 6 · Week 05 · Day 3 — Services, Networking, and Namespaces

> Objective: Expose pods with Services, understand k8s networking, and organize workloads with Namespaces.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

## Concept

### Why Services exist

Pods are ephemeral — they die and restart with new IPs. A **Service** gives you a stable IP and DNS name that always routes to healthy pods matching its selector.

```
Client → Service (ClusterIP: 10.96.100.1) → {Pod1, Pod2, Pod3}
              (stable)                      (pods come and go)
```

Services also provide **load balancing** across pods.

### Service types

| Type | Accessible from | Use case |
|------|----------------|----------|
| `ClusterIP` (default) | Inside cluster only | Internal microservice communication |
| `NodePort` | Outside cluster via `NodeIP:NodePort` | Dev/testing, bare-metal simple exposure |
| `LoadBalancer` | Outside cluster via cloud LB IP | Production on cloud providers |
| `ExternalName` | Inside cluster → external DNS | Route to external service by name |

### Namespaces — virtual clusters

A **Namespace** partitions cluster resources. Resources in different namespaces are isolated from each other (by default). Used for:

- Environment separation (dev/staging/prod — though separate clusters is better)
- Team isolation
- Resource quota enforcement

Default namespaces every cluster has:

| Namespace | Purpose |
|-----------|---------|
| `default` | Where resources land if you don't specify |
| `kube-system` | Control plane and system components |
| `kube-public` | Publicly readable cluster info |
| `kube-node-lease` | Node heartbeat objects |

### DNS in Kubernetes

k8s runs a DNS server (CoreDNS) in `kube-system`. Every Service gets a DNS entry:

```
<service-name>.<namespace>.svc.cluster.local
```

From within the same namespace: `my-service` resolves.
Cross-namespace: `my-service.other-namespace` resolves.
Full FQDN: `my-service.other-namespace.svc.cluster.local`.

## How it works internally

**kube-proxy and iptables** — when a Service is created, the API server creates an `Endpoints` object listing the IPs of matching pods. kube-proxy watches Endpoints and programs iptables (or IPVS) rules on every node. Packets to the ClusterIP are rewritten (DNAT) to a real pod IP. This is in-kernel packet manipulation — no userspace proxy.

**Endpoints controller** — watches Services and Pods. Maintains the Endpoints object: adds pod IPs when pods become Ready, removes them when pods fail readiness probes or die. This is how the Service load balancer stays current.

**NodePort** — kube-proxy also opens a high port (30000–32767) on every node. Traffic to `<any-node-IP>:30080` is NATted to a pod. The cloud LoadBalancer type creates a cloud LB that points at NodePorts.

**CoreDNS** — runs as a Deployment in `kube-system`. The kubelet configures each pod's `/etc/resolv.conf` to point to the CoreDNS ClusterIP. DNS queries go to CoreDNS, which looks up k8s Services and returns ClusterIPs.

**Network policy** — by default pods can talk to any other pod in the cluster. `NetworkPolicy` resources (enforced by the CNI plugin, e.g. Calico) add firewall rules at the pod level.

## Worked example

### ClusterIP Service (internal)

```yaml
# service-clusterip.yaml
apiVersion: v1
kind: Service
metadata:
  name: fastapi-service
  namespace: default
spec:
  selector:
    app: fastapi       # routes to pods with this label
  ports:
    - name: http
      port: 80         # Service port (what callers use)
      targetPort: 8000 # container port
  type: ClusterIP
```

```bash
kubectl apply -f service-clusterip.yaml
kubectl get service fastapi-service
# NAME               TYPE        CLUSTER-IP      PORT(S)   AGE
# fastapi-service    ClusterIP   10.96.200.5     80/TCP    10s

# Test from another pod in the cluster
kubectl run test --image=curlimages/curl --rm -it -- \
  curl http://fastapi-service/health
```

### NodePort Service (dev access)

```yaml
# service-nodeport.yaml
apiVersion: v1
kind: Service
metadata:
  name: fastapi-nodeport
spec:
  selector:
    app: fastapi
  ports:
    - port: 80
      targetPort: 8000
      nodePort: 30080    # optional: k8s assigns one if omitted
  type: NodePort
```

```bash
# With kind: forward the node port
kubectl port-forward service/fastapi-service 8080:80
# Now accessible at localhost:8080
```

### LoadBalancer Service (cloud)

```yaml
# service-lb.yaml
apiVersion: v1
kind: Service
metadata:
  name: fastapi-lb
spec:
  selector:
    app: fastapi
  ports:
    - port: 80
      targetPort: 8000
  type: LoadBalancer
```

```bash
kubectl get service fastapi-lb
# EXTERNAL-IP column shows the cloud LB IP when provisioned
```

### Namespace operations

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ai-platform
  labels:
    env: production
    team: ml-infra
```

```bash
kubectl apply -f namespace.yaml

# Or imperatively
kubectl create namespace ai-platform

# Deploy into a namespace
kubectl apply -f deployment.yaml -n ai-platform

# Switch default namespace
kubectl config set-context --current --namespace=ai-platform

# Cross-namespace DNS test
kubectl run test -n default --image=curlimages/curl --rm -it -- \
  curl http://fastapi-service.ai-platform.svc.cluster.local/health
```

### Resource Quota on a Namespace

```yaml
# quota.yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: ai-platform-quota
  namespace: ai-platform
spec:
  hard:
    pods: "20"
    requests.cpu: "4"
    requests.memory: "8Gi"
    limits.cpu: "8"
    limits.memory: "16Gi"
```

### Inspecting Endpoints

```bash
# See which pod IPs are behind the service
kubectl get endpoints fastapi-service

# Watch endpoints change as pods restart
kubectl get endpoints fastapi-service -w
```

## Exercises

1. Write a ClusterIP Service for a Deployment with label `app: model-server`, listening on port 9000, targeting container port 9000. What DNS name will other pods in the `default` namespace use to reach it?

2. Your Service selector is `app: fastapi` but all your pods have `app: fastapi-api`. What does `kubectl get endpoints my-service` show, and how do you fix it?

3. A pod in namespace `backend` needs to call a service named `redis` in namespace `data`. Write the full DNS name it must use. Explain why the short name `redis` would fail.

4. Explain when you would use `NodePort` vs `LoadBalancer`. What is the main drawback of `NodePort` in production?

5. Create a Namespace manifest for `staging`. Write a `ResourceQuota` that limits it to 10 pods and 2 CPU requests.

(Solutions: ../solutions/week-05/day-3.md)

## Self-check

- [ ] I can write ClusterIP, NodePort, and LoadBalancer Service manifests
- [ ] I understand how kube-proxy routes traffic without a userspace proxy
- [ ] I can predict the DNS name of any Service in any namespace
- [ ] I can create Namespaces and deploy resources into them
- [ ] I understand what Endpoints are and how they stay current

## Supplementary links

- [Services — concepts](https://kubernetes.io/docs/concepts/services-networking/service/)
- [DNS for Services and Pods](https://kubernetes.io/docs/concepts/services-networking/dns-pod-service/)
- [Namespaces](https://kubernetes.io/docs/concepts/overview/working-with-objects/namespaces/)
- [Network Policies](https://kubernetes.io/docs/concepts/services-networking/network-policies/)
