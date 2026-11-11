# Solutions — Day 4: ConfigMaps, Secrets, Labels, and Selectors

## Exercise 1 — ConfigMap with multi-line key

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-config
  namespace: default
data:
  MODEL_NAME: claude-sonnet-5-5
  MAX_TOKENS: "4096"
  model.json: |
    {
      "model_name": "claude-sonnet-5-5",
      "max_tokens": 4096,
      "temperature": 0.7,
      "timeout_seconds": 30
    }
```

## Exercise 2 — Create Secret imperatively

```bash
kubectl create secret generic api-credentials \
  --from-literal=API_KEY="${API_KEY}" \
  --from-literal=API_SECRET="${API_SECRET}"
```

This creates the Secret without any YAML file containing the values. The values come from shell environment variables and never appear in git.

## Exercise 3 — ConfigMap update not reflected in pods

**Cause:** Environment variables are captured at pod start time from the ConfigMap values. Updating a ConfigMap does not restart pods, so existing pods keep their old env vars until they restart.

**Two fixes:**
1. Restart the Deployment after updating the ConfigMap:
   ```bash
   kubectl rollout restart deployment/fastapi-app
   ```
2. Switch to **volume mounts** instead of `envFrom`. Volume-mounted ConfigMap files are updated automatically (with a short delay — typically ~60 seconds) when the ConfigMap changes. No pod restart needed for file-based config.

## Exercise 4 — Label selector

**kubectl syntax:**
```bash
kubectl get pods -l app=worker,'env notin (development)'
```

**YAML matchExpressions form:**
```yaml
selector:
  matchExpressions:
    - key: app
      operator: In
      values:
        - worker
    - key: env
      operator: NotIn
      values:
        - development
```

## Exercise 5 — Secrets security limitations

Kubernetes Secrets are stored as **base64-encoded** data in etcd by default. Base64 is encoding, not encryption — anyone with etcd access or the right RBAC permissions can read them in plaintext. etcd backups also contain all Secret values in readable form.

**Production alternatives:**
- Enable **Encryption at Rest** in the API server config (encrypts Secret data in etcd with a key)
- Use **External Secrets Operator** to sync secrets from AWS Secrets Manager, Azure Key Vault, or HashiCorp Vault
- Use **Secrets Store CSI Driver** to mount secrets from external vaults as files
- Restrict Secret access with RBAC (least-privilege `get`/`list` on Secrets)
