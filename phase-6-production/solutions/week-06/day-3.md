# Solutions — Week 06 · Day 3 (StatefulSets, PVs, Jobs & CronJobs)

## Exercise 1 — StatefulSet scale-up from 3 to 5

Sequence with `podManagementPolicy: OrderedReady`:

1. Controller creates `vector-db-3` pod.
2. Controller waits until `vector-db-3` passes its readiness probe and becomes `Ready`.
   - If `vector-db-3` fails its readiness probe: it never becomes `Ready`, so the controller waits indefinitely. The StatefulSet stays at 4 pods (3 original + 1 stuck). No progress toward `vector-db-4`.
3. Once `vector-db-3` is `Ready`, controller creates `vector-db-4`.
4. Once `vector-db-4` is `Ready`, the scale-out is complete (5 replicas running).

**Effect of readiness probe failure on vector-db-3:** Scale-up is blocked until the probe passes or you intervene (fix the pod config, delete and recreate the pod). The StatefulSet will not proceed past a stuck pod.

To observe: `kubectl describe statefulset vector-db -n production | grep -A10 Events`

## Exercise 2 — PVC and Deployment volume mount

```yaml
# pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: embedding-store
  namespace: production
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: nfs-client
  resources:
    requests:
      storage: 100Gi
```

```yaml
# Deployment volume mount (within spec.template.spec)
volumes:
  - name: embeddings
    persistentVolumeClaim:
      claimName: embedding-store
containers:
  - name: api
    volumeMounts:
      - name: embeddings
        mountPath: /embeddings
```

Note: `ReadWriteMany` requires a StorageClass that supports it (NFS, CephFS, AWS EFS). Standard block storage (`gp2`, `pd-standard`) only supports `ReadWriteOnce`.

## Exercise 3 — CronJob LAST SCHEDULE is `<none>`: causes and diagnosis

| Cause | Diagnosis command |
|---|---|
| CronJob is suspended | `kubectl get cronjob nightly-eval -n production -o jsonpath='{.spec.suspend}'` — if `true`, unsuspend with `kubectl patch cronjob nightly-eval -p '{"spec":{"suspend":false}}'` |
| Schedule has not yet fired (first fire is in the future) | `kubectl get cronjob nightly-eval -n production` — check SCHEDULE column; calculate next fire from cron expression |
| Timezone or cron expression error | `kubectl describe cronjob nightly-eval -n production` — look for Events with type=Warning |

After 10 minutes with no fires and no obvious cause, trigger manually:
```bash
kubectl create job --from=cronjob/nightly-eval test-run -n production
```

## Exercise 4 — CronJob every 6 hours

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: queue-processor
  namespace: production
spec:
  schedule: "0 */6 * * *"
  timeZone: "UTC"
  concurrencyPolicy: Forbid
  successfulJobsHistoryLimit: 2
  failedJobsHistoryLimit: 1
  jobTemplate:
    spec:
      backoffLimit: 1
      activeDeadlineSeconds: 3600   # 1 hour hard cap
      template:
        spec:
          restartPolicy: OnFailure
          containers:
            - name: processor
              image: registry.example.com/ai-tools:v2.0.0   # placeholder
              command: ["python", "-m", "tools.process_queue"]
              resources:
                requests:
                  cpu: "200m"
                  memory: "256Mi"
                limits:
                  cpu: "1"
                  memory: "1Gi"
```

## Exercise 5 — PVC lifecycle after StatefulSet deletion

`kubectl delete statefulset redis --cascade=foreground` deletes the StatefulSet controller and its pods in dependency order. **The PVCs (`data-redis-0`, `data-redis-1`, `data-redis-2`) are NOT deleted.** This is intentional — Kubernetes protects persistent data from accidental deletion.

Full cleanup sequence:
```bash
# 1. Delete StatefulSet (and its pods)
kubectl delete statefulset redis -n production

# 2. Verify pods are gone
kubectl get pods -n production -l app=redis

# 3. Delete PVCs manually
kubectl delete pvc data-redis-0 data-redis-1 data-redis-2 -n production

# 4. Verify PVs are released (if reclaimPolicy=Retain, PVs remain; delete manually)
kubectl get pv | grep redis
kubectl delete pv <pv-name>   # if reclaimPolicy=Retain

# 5. Delete Headless Service
kubectl delete svc redis -n production
```

If StorageClass has `reclaimPolicy: Delete` (default for most cloud providers), deleting the PVC automatically triggers deletion of the underlying cloud volume. With `reclaimPolicy: Retain`, the PV and the actual storage block remain until manually deleted — useful for data recovery.
