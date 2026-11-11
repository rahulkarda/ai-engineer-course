# Phase 6 · Week 06 · Day 3 — StatefulSets, Persistent Volumes, Jobs & CronJobs

> Objective: Deploy stateful workloads with stable storage, run batch jobs, and schedule recurring tasks.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

## Concept

**StatefulSet** is the controller for workloads that require:
- Stable, unique pod names (e.g., `redis-0`, `redis-1`, `redis-2`)
- Stable network identity via a Headless Service (DNS: `redis-0.redis.production.svc`)
- Ordered, graceful startup and shutdown
- Persistent storage per pod via `volumeClaimTemplates`

**PersistentVolume (PV)** is cluster storage provisioned by an admin or dynamically by a StorageClass. A **PersistentVolumeClaim (PVC)** is a namespaced request for storage — size, access mode, StorageClass. Kubernetes binds PVC → PV.

Access modes:
| Mode | Abbreviation | Meaning |
|---|---|---|
| ReadWriteOnce | RWO | One node read/write |
| ReadOnlyMany | ROX | Many nodes read |
| ReadWriteMany | RWX | Many nodes read/write |

**Job** runs pods to completion for batch work. **CronJob** wraps a Job with a cron schedule.

## How it works internally

StatefulSet pods are numbered from 0. The controller creates them in order (0, then 1, then 2) and waits for each to become Ready before proceeding (configurable via `podManagementPolicy: Parallel` to skip this). Scale-down proceeds in reverse order (2, then 1, then 0).

`volumeClaimTemplates` in a StatefulSet causes each pod to own a uniquely named PVC (`data-redis-0`, `data-redis-1`). These PVCs are NOT deleted when the StatefulSet is deleted — intentional to protect data. Delete them manually or with `kubectl delete pvc`.

**StorageClass** with `provisioner` drives dynamic provisioning. Cloud providers supply their own StorageClasses (`gp2`, `premium-rwo`, etc.). `WaitForFirstConsumer` binding mode defers PV creation until the pod is scheduled — important for topology-aware storage (PV must be in the same zone as the pod).

**Job** spec fields that matter:
- `completions`: total successful pod executions needed
- `parallelism`: concurrent pods
- `backoffLimit`: retries before the Job is marked failed
- `activeDeadlineSeconds`: hard time cap

**CronJob** creates a new Job at each tick. `concurrencyPolicy: Forbid` skips a tick if the previous Job is still running — critical for AI batch jobs that must not overlap.

## Worked example

### 1. StatefulSet — Redis sentinel with persistent storage

```yaml
# statefulset-redis.yaml
apiVersion: v1
kind: Service
metadata:
  name: redis
  namespace: production
spec:
  clusterIP: None       # headless — required for StatefulSet DNS
  selector:
    app: redis
  ports:
    - port: 6379
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis
  namespace: production
spec:
  serviceName: redis    # must match the headless Service name
  replicas: 3
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
        - name: redis
          image: redis:7.2-alpine
          command: ["redis-server", "--appendonly", "yes"]
          ports:
            - containerPort: 6379
          resources:
            requests:
              cpu: "100m"
              memory: "128Mi"
            limits:
              cpu: "500m"
              memory: "512Mi"
          volumeMounts:
            - name: data
              mountPath: /data
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: standard   # replace with cluster's StorageClass
        resources:
          requests:
            storage: 5Gi
```

```bash
kubectl apply -f statefulset-redis.yaml
kubectl get statefulset -n production -w
kubectl get pvc -n production   # data-redis-0, data-redis-1, data-redis-2
# Stable DNS entry for each pod
kubectl run dns-test --image=busybox --rm -it --restart=Never -- \
  nslookup redis-0.redis.production.svc.cluster.local
```

### 2. Standalone PVC and Deployment using it

```yaml
# pvc-model-cache.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache
  namespace: production
spec:
  accessModes:
    - ReadWriteMany     # shared across multiple AI-api pods (requires RWX StorageClass)
  storageClassName: efs-sc    # example: AWS EFS CSI driver
  resources:
    requests:
      storage: 50Gi
```

```yaml
# mount in a Deployment
volumes:
  - name: model-cache
    persistentVolumeClaim:
      claimName: model-cache
containers:
  - name: api
    volumeMounts:
      - name: model-cache
        mountPath: /models
```

### 3. Job — one-shot embedding generation

```yaml
# job-embed.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: embed-corpus-v2
  namespace: production
spec:
  completions: 1
  parallelism: 1
  backoffLimit: 3
  activeDeadlineSeconds: 3600   # fail after 1 hour
  template:
    spec:
      restartPolicy: Never      # Job pods must use Never or OnFailure
      containers:
        - name: embedder
          image: registry.example.com/ai-tools:v2.0.0   # placeholder
          command: ["python", "-m", "tools.embed_corpus"]
          env:
            - name: CORPUS_PATH
              value: "/data/corpus-v2"
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: ai-secrets
                  key: openai-api-key
          resources:
            requests:
              cpu: "1"
              memory: "2Gi"
            limits:
              cpu: "4"
              memory: "8Gi"
```

```bash
kubectl apply -f job-embed.yaml
kubectl get jobs -n production -w
kubectl logs job/embed-corpus-v2 -n production
```

### 4. Parallel Job — process N documents with worker pool

```yaml
# job-parallel.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: chunk-documents
  namespace: production
spec:
  completions: 100      # 100 work items total
  parallelism: 10       # 10 concurrent pods
  completionMode: Indexed   # each pod gets JOB_COMPLETION_INDEX env var (0..99)
  backoffLimit: 5
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: chunker
          image: registry.example.com/ai-tools:v2.0.0   # placeholder
          command: ["python", "-m", "tools.chunk_shard"]
          env:
            - name: SHARD_INDEX
              valueFrom:
                fieldRef:
                  fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
```

### 5. CronJob — nightly model evaluation

```yaml
# cronjob-eval.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: nightly-model-eval
  namespace: production
spec:
  schedule: "0 2 * * *"          # 02:00 UTC daily
  timeZone: "UTC"
  concurrencyPolicy: Forbid       # skip if previous run still going
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 1
  jobTemplate:
    spec:
      backoffLimit: 1
      activeDeadlineSeconds: 7200  # 2-hour hard cap
      template:
        spec:
          restartPolicy: OnFailure
          containers:
            - name: evaluator
              image: registry.example.com/ai-tools:v2.0.0   # placeholder
              command: ["python", "-m", "eval.nightly"]
              resources:
                requests:
                  cpu: "500m"
                  memory: "1Gi"
```

```bash
kubectl apply -f cronjob-eval.yaml
# Trigger manually
kubectl create job --from=cronjob/nightly-model-eval manual-eval -n production
# Inspect
kubectl describe cronjob nightly-model-eval -n production
kubectl get jobs -n production --sort-by=.metadata.creationTimestamp
```

## Exercises

1. A StatefulSet `vector-db` has 3 replicas and `podManagementPolicy: OrderedReady`. You scale from 3 to 5. Describe the exact sequence of events Kubernetes performs, including what happens if `vector-db-3` fails its readiness probe.

2. Write a PVC manifest for a 100 Gi RWX volume using a `StorageClass` named `nfs-client`. Then write the Deployment volume mount that attaches it to `/embeddings` in all pods.

3. The CronJob `nightly-eval` shows `LAST SCHEDULE: <none>` 10 minutes after apply. List three causes and the command to diagnose each.

4. Write a CronJob that runs every 6 hours, processes a queue (image: `registry.example.com/ai-tools:v2.0.0`), allows at most one concurrent run, keeps 2 successful histories, and hard-caps execution at 1 hour.

5. What happens to the PVCs of a StatefulSet after `kubectl delete statefulset redis --cascade=foreground`? How do you fully clean up including the storage?

(Solutions: ../solutions/week-06/day-3.md)

## Self-check

- [ ] I can deploy a StatefulSet with a Headless Service and persistent storage per pod.
- [ ] I can write a PVC and attach it to a Deployment.
- [ ] I can write a batch Job with correct `restartPolicy`, `backoffLimit`, and resource limits.
- [ ] I can write a CronJob with appropriate concurrency and history settings.
- [ ] I understand what happens to PVCs when a StatefulSet is deleted.

## Supplementary links

- [StatefulSets](https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/)
- [Persistent Volumes](https://kubernetes.io/docs/concepts/storage/persistent-volumes/)
- [Jobs](https://kubernetes.io/docs/concepts/workloads/controllers/job/)
- [CronJobs](https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/)
- [Indexed Jobs](https://kubernetes.io/docs/tasks/job/indexed-parallel-processing-static/)
