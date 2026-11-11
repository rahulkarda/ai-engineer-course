# Phase 6 · Week 05 · Day 1 — Kubernetes Architecture & kubectl

> Objective: Understand the k8s control plane, node components, and issue your first kubectl commands.  ·  Time: ~1 hr  ·  Prereqs: [[week-04/day-7]]

## Concept

Kubernetes (k8s) is a **container orchestration platform**: it schedules containers across a fleet of machines, maintains desired state, and self-heals when things fail. You describe *what* you want; Kubernetes figures out *how* to make it happen.

### The cluster split

A Kubernetes cluster has two logical tiers:

| Tier | Who | What it does |
|------|-----|--------------|
| **Control plane** | Runs on master node(s) | Makes decisions (scheduling, scaling, healing) |
| **Data plane** | Runs on worker nodes | Actually runs your workloads |

### Control-plane components

```
┌─────────────────────────────────────────────────────┐
│                   Control Plane                     │
│                                                     │
│  ┌──────────────┐  ┌────────────┐  ┌─────────────┐ │
│  │  kube-api-   │  │  etcd      │  │  kube-       │ │
│  │  server      │  │  (store)   │  │  scheduler   │ │
│  └──────────────┘  └────────────┘  └─────────────┘ │
│  ┌──────────────────────────────────────────────┐   │
│  │       kube-controller-manager                │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

**kube-apiserver** — the single front door. Every kubectl command, every controller, every node talks to the API server. It validates and persists all state changes to etcd.

**etcd** — a distributed key-value store that is the *only* source of truth. All cluster state (pods, services, configs) lives here. Losing etcd without a backup means losing the cluster.

**kube-scheduler** — watches for unscheduled pods and picks the best node based on resource requests, affinity rules, taints/tolerations.

**kube-controller-manager** — runs control loops (controllers). A controller watches the current state in etcd and reconciles it toward the desired state. Examples: the ReplicaSet controller, the Deployment controller, the Node controller.

### Node (worker) components

```
┌──────────────────────────────────────────────┐
│                  Worker Node                 │
│                                              │
│  ┌──────────┐  ┌────────────┐  ┌──────────┐ │
│  │ kubelet  │  │ kube-proxy │  │ container│ │
│  │          │  │            │  │ runtime  │ │
│  └──────────┘  └────────────┘  └──────────┘ │
│                                              │
│  Pod  Pod  Pod  Pod  Pod                     │
└──────────────────────────────────────────────┘
```

**kubelet** — an agent on every node. It receives a PodSpec (what to run), talks to the container runtime (containerd or docker), and reports pod status back to the API server. If a container dies unexpectedly, kubelet restarts it.

**kube-proxy** — maintains network rules (iptables or IPVS) on each node so that Service IPs route to the right pods.

**Container runtime** — actually starts/stops containers. Modern k8s uses containerd or CRI-O via the Container Runtime Interface (CRI).

### The reconciliation loop

Everything in Kubernetes is a control loop:

```
Desired state (in etcd) ──▶ Controller ──▶ Actual state
        ▲                                        │
        └────────────────────────────────────────┘
              (controller drives toward desired)
```

When you `kubectl apply`, you write desired state. Controllers detect the gap and act. This is **declarative** infrastructure.

## How it works internally

**API server admission pipeline** — every write goes through: authentication → authorization (RBAC) → admission controllers (mutating webhooks → validating webhooks) → write to etcd. No component ever writes to etcd directly.

**Watch mechanism** — controllers don't poll. They call `List+Watch` on the API server. Etcd sends change events; the API server fans them out. This makes controllers event-driven with O(1) watch overhead per controller, not O(n) polling.

**Scheduler algorithm** — two phases: *filter* (remove nodes that can't fit the pod) then *score* (rank remaining nodes by resource utilization, affinity, spread). Picks highest scorer. If no node fits, pod stays `Pending`.

**kubelet reconcile** — kubelet calls `CRI.RunPodSandbox` (sets up network namespace), then `CRI.CreateContainer` + `CRI.StartContainer` for each container in the pod. It also runs liveness/readiness probes and calls `CRI.StopContainer` on failures.

## Worked example

### Installing kubectl and connecting to a cluster

```bash
# macOS
brew install kubectl

# Verify — shows client version even without a cluster
kubectl version --client

# Check config — points at your current cluster context
kubectl config view

# With kind (local cluster)
kind create cluster --name dev
kubectl config use-context kind-dev
```

### Exploring cluster components

```bash
# List nodes in the cluster
kubectl get nodes

# Wide output: see IP, OS, container runtime version
kubectl get nodes -o wide

# Inspect a node
kubectl describe node <node-name>

# Show all system pods (control plane lives in kube-system)
kubectl get pods -n kube-system

# Watch pods update in real time
kubectl get pods -n kube-system -w

# Show all namespaces
kubectl get namespaces

# Get everything in a namespace
kubectl get all -n kube-system
```

### kubectl basics — the four verbs

```bash
# CREATE (or update) from a manifest
kubectl apply -f manifest.yaml

# READ — list resources
kubectl get pods
kubectl get pods -o yaml          # full spec
kubectl get pods -o jsonpath='{.items[0].metadata.name}'

# DESCRIBE — human-readable detail + events
kubectl describe pod <name>

# DELETE
kubectl delete pod <name>
kubectl delete -f manifest.yaml   # delete what the file defines
```

### kubectl dry-run and diff

```bash
# Validate without applying
kubectl apply -f manifest.yaml --dry-run=client

# Show what would change server-side
kubectl apply -f manifest.yaml --dry-run=server

# Diff current cluster state vs manifest
kubectl diff -f manifest.yaml
```

### Context and namespace switching

```bash
# List contexts (each context = cluster + user + namespace)
kubectl config get-contexts

# Switch context
kubectl config use-context kind-dev

# Set default namespace for current context
kubectl config set-context --current --namespace=my-app

# Override namespace per command
kubectl get pods -n kube-system
```

## Exercises

1. Draw the control-plane components and label what each one does. Identify which component you would lose cluster writes to if it went down, and which component you would lose scheduling to.

2. Explain in two sentences why etcd is the "source of truth" and what the consequence of etcd data loss is.

3. Write the kubectl command to list all pods across **all namespaces** and pipe the output to `grep Running` to count only running pods.

4. Write the kubectl command to display the full YAML of a pod named `api-server` in the `production` namespace.

5. A pod is stuck in `Pending`. Which control-plane component is responsible for moving it to a node, and what could cause it to remain pending?

(Solutions: ../solutions/week-05/day-1.md)

## Self-check

- [ ] I can name all four control-plane components and their roles
- [ ] I can explain the reconciliation loop in my own words
- [ ] I can use `kubectl get`, `describe`, `apply`, `delete` confidently
- [ ] I can switch contexts and namespaces with kubectl
- [ ] I can explain what kubelet does on a worker node

## Supplementary links

- [Kubernetes Components — official docs](https://kubernetes.io/docs/concepts/overview/components/)
- [kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)
- [etcd — what it is](https://etcd.io/docs/v3.5/learning/why/)
