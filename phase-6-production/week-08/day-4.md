# Phase 6 · Week 08 · Day 4 — Secrets Managers, Load Balancers, and API Gateways

> Objective: Wire together Secrets Manager, ALB, and API Gateway into a production-grade AI API entry point, and implement runtime secret fetching correctly.  ·  Time: ~1 hr  ·  Prereqs: [[week-08/day-3]]

---

## Concept

A production AI API endpoint is never a single compute resource exposed directly to the internet. It is a **chain of layers**, each responsible for a distinct concern:

```
Client
  │
  ▼
API Gateway          ← rate limiting, auth, request validation, WAF
  │
  ▼
Load Balancer        ← health-based routing, TLS termination, target group management
  │
  ▼
Compute (ECS / Lambda / EC2)
  │
  ▼
Secrets Manager      ← runtime credential fetch (LLM API keys, DB passwords)
```

### Secrets Manager

**Never** bake secrets into container images, environment variable plain text in task definitions, or source code. AWS Secrets Manager (and equivalents: GCP Secret Manager, Azure Key Vault) stores secrets encrypted at rest (KMS) and delivers them to authorised callers at runtime.

Two retrieval patterns:
1. **ECS injection**: ECS agent fetches the secret at task start and injects it as an env var. Simple; secret is available as `os.getenv(...)`. Downside: the value is fixed at container start — no live rotation without a restart.
2. **SDK fetch in code**: your app calls `secretsmanager:GetSecretValue` at startup (or on each request with caching). Enables rotation without redeployment. This is the preferred pattern for LLM API keys that may rotate.

**Secret rotation**: Secrets Manager supports automated rotation via a Lambda function. For an LLM API key, you would write a rotation Lambda that calls the provider's key rotation API and updates the secret value. Applications using pattern 2 (SDK fetch) pick up the new value on the next cache refresh.

### Load Balancers: ALB vs NLB

| | ALB (Application LB) | NLB (Network LB) |
|---|---|---|
| Layer | 7 (HTTP/HTTPS) | 4 (TCP/UDP/TLS) |
| Routing | Path, host, header, query string | IP + port only |
| TLS termination | Yes (at ALB) | Yes (pass-through or terminate) |
| WebSockets | Yes | Yes |
| gRPC | Yes | Yes |
| Use for AI APIs | Default choice — HTTP routing rules, path-based canary | Low-latency TCP, non-HTTP protocols |

For AI REST APIs: **always ALB**. ALB adds ~1–2 ms latency but provides path-based routing (e.g., `/v1/*` → service A, `/v2/*` → service B), header-based routing for canary releases, and native integration with WAF and Cognito.

### API Gateway

AWS API Gateway has two relevant product lines:

| | REST API (v1) | HTTP API (v2) |
|---|---|---|
| Latency | ~6 ms overhead | ~1 ms overhead |
| Price | Higher | ~70% cheaper |
| Features | Full: usage plans, API keys, request/response transforms, SDK gen | Streamlined: JWT auth, Lambda proxy, CORS |
| WebSockets | Via separate API GW WS | Not supported |
| For AI | Use when you need usage plans per customer or request transforms | Default for most AI APIs |

**API Gateway vs ALB — when to use which:**
- API Gateway: when you need per-client rate limiting (usage plans), monetisation metering, or serverless-only (no container backend).
- ALB: when backend is containerised (ECS), you need WebSocket upgrades, or you want to avoid API Gateway's per-request cost at scale (>10M req/day ALB is cheaper).
- They can be chained: API Gateway → ALB → ECS (though this adds latency; evaluate the need).

### WAF (Web Application Firewall)

Attach AWS WAF to either an ALB or API Gateway. For AI APIs, the most relevant WAF rules:

- **Rate-based rules**: block IPs exceeding N requests per 5 minutes — protection against scraping and abuse.
- **AWS Managed Rules — Common Rule Set**: OWASP Top 10 protections (SQL injection, XSS, path traversal).
- **Size constraints**: block request bodies over a set size — prevents prompt injection via oversized inputs.
- **Geographic restrictions**: block traffic from regions you do not serve.

---

## How it works internally

### Secret rotation mechanism

```
Rotation Lambda (triggered by Secrets Manager on rotation schedule)
  │
  ├─ Step 1: createSecret  → generate new key value (call provider API)
  ├─ Step 2: setSecret     → store new value in Secrets Manager as AWSPENDING
  ├─ Step 3: testSecret    → verify new key works against the provider
  └─ Step 4: finishSecret  → promote AWSPENDING to AWSCURRENT, demote old to AWSPREVIOUS

Application (fetching at runtime)
  └─ always fetches AWSCURRENT → automatically gets new key after rotation completes
     (grace period: AWSPREVIOUS still valid for ~1 rotation cycle)
```

Your application code should cache the secret for a configurable TTL (e.g., 5 minutes) to avoid a Secrets Manager API call on every request. On a `GetSecretValue` call that returns a different value than the cached one, refresh the downstream client.

### ALB target group health checks

ALB continuously probes each target (ECS task / EC2 instance) at the configured path and interval. A target is marked **healthy** only after `HealthyThresholdCount` consecutive successes. Only healthy targets receive traffic. This is how zero-downtime deployments work: new tasks become healthy before old ones are deregistered.

```
Deployment step:
  1. New task starts → ALB health check begins
  2. New task returns 200 on /health for N consecutive checks → marked Healthy
  3. Old task deregistered from target group → connection draining (in-flight requests complete)
  4. Old task stopped
```

---

## Worked example

Python module to fetch a secret from AWS Secrets Manager at runtime, with caching, structured logging, and type hints.

```python
# secrets.py
# Runtime secret fetcher with in-memory cache.
# No secrets hardcoded anywhere in this file.
# IAM requirement: execution role needs secretsmanager:GetSecretValue on the target ARN.

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

_REGION = os.getenv("AWS_REGION", "us-east-1")
_CACHE_TTL_SECONDS = int(os.getenv("SECRET_CACHE_TTL_SECONDS", "300"))  # 5 min default


@dataclass
class _CachedSecret:
    value: str
    fetched_at: float = field(default_factory=time.monotonic)

    def is_expired(self, ttl_seconds: int) -> bool:
        return (time.monotonic() - self.fetched_at) > ttl_seconds


class SecretsClient:
    """
    Thread-safe Secrets Manager client with in-memory TTL cache.
    Instantiate once at module level; reuse across requests.
    """

    def __init__(
        self,
        region: str = _REGION,
        cache_ttl_seconds: int = _CACHE_TTL_SECONDS,
    ) -> None:
        self._client = boto3.client("secretsmanager", region_name=region)
        self._cache: dict[str, _CachedSecret] = {}
        self._lock = threading.Lock()
        self._ttl = cache_ttl_seconds
        logger.info("SecretsClient initialised", extra={"region": region, "ttl_s": cache_ttl_seconds})

    def get_secret_string(self, secret_name: str) -> str:
        """
        Return the plaintext string value of a secret.
        Returns from cache if still fresh; otherwise fetches from Secrets Manager.
        Raises SecretsError on fetch failure.
        """
        with self._lock:
            cached = self._cache.get(secret_name)
            if cached and not cached.is_expired(self._ttl):
                logger.debug("Secret cache hit", extra={"secret_name": secret_name})
                return cached.value

        logger.info("Fetching secret", extra={"secret_name": secret_name})
        value = self._fetch(secret_name)

        with self._lock:
            self._cache[secret_name] = _CachedSecret(value=value)

        return value

    def get_secret_json(self, secret_name: str) -> dict[str, Any]:
        """
        Parse the secret value as JSON and return the dict.
        Useful for secrets that store multiple values (e.g. DB credentials).
        """
        raw = self.get_secret_string(secret_name)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SecretsError(f"Secret '{secret_name}' is not valid JSON.") from exc

    def _fetch(self, secret_name: str) -> str:
        try:
            response = self._client.get_secret_value(SecretId=secret_name)
        except ClientError as exc:
            error_code = exc.response["Error"]["Code"]
            logger.error(
                "Failed to fetch secret",
                extra={"secret_name": secret_name, "error_code": error_code},
            )
            raise SecretsError(
                f"Could not retrieve secret '{secret_name}': {error_code}"
            ) from exc

        # Secrets Manager returns either SecretString or SecretBinary.
        if "SecretString" in response:
            return response["SecretString"]
        raise SecretsError(
            f"Secret '{secret_name}' is binary — only string secrets supported."
        )

    def invalidate(self, secret_name: str) -> None:
        """Force cache eviction for a specific secret (e.g., after detecting a 401)."""
        with self._lock:
            self._cache.pop(secret_name, None)
        logger.info("Secret cache invalidated", extra={"secret_name": secret_name})


class SecretsError(Exception):
    """Raised when a secret cannot be retrieved or parsed."""


# --- Module-level singleton for use across the application ---
_secrets = SecretsClient()


def get_llm_api_key() -> str:
    """
    Fetch the LLM API key from Secrets Manager.
    Secret name read from env var — no ARN hardcoded here.
    """
    secret_name = os.getenv("LLM_API_KEY_SECRET_NAME")
    if not secret_name:
        raise SecretsError("LLM_API_KEY_SECRET_NAME environment variable not set.")
    return _secrets.get_secret_string(secret_name)


def get_db_credentials() -> dict[str, Any]:
    """
    Fetch DB credentials JSON: {"username": ..., "password": ..., "host": ...}
    """
    secret_name = os.getenv("DB_CREDENTIALS_SECRET_NAME")
    if not secret_name:
        raise SecretsError("DB_CREDENTIALS_SECRET_NAME environment variable not set.")
    return _secrets.get_secret_json(secret_name)
```

### Usage in a FastAPI app

```python
# app/main.py  (fragment)
from __future__ import annotations

import logging
import os

import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.secrets import SecretsError, get_llm_api_key

logger = logging.getLogger(__name__)
app = FastAPI()

# Client initialised lazily on first request; key fetched from Secrets Manager.
_anthropic_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        try:
            api_key = get_llm_api_key()
        except SecretsError as exc:
            logger.error("Cannot initialise Anthropic client: %s", exc)
            raise RuntimeError("LLM client unavailable.") from exc
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


class QueryRequest(BaseModel):
    prompt: str
    max_tokens: int = 1024


@app.post("/v1/query")
async def query(req: QueryRequest) -> dict[str, str]:
    client = _get_client()
    try:
        message = client.messages.create(
            model=os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5-20251001"),
            max_tokens=req.max_tokens,
            messages=[{"role": "user", "content": req.prompt}],
        )
    except anthropic.APIError as exc:
        logger.error("Anthropic API error", extra={"status_code": exc.status_code})
        raise HTTPException(status_code=502, detail="LLM API error.") from exc
    return {"response": message.content[0].text}
```

### Tests for SecretsClient

```python
# tests/unit/test_secrets.py

import json
import time
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from app.secrets import SecretsClient, SecretsError


@pytest.fixture()
def mock_boto_client() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def secrets_client(mock_boto_client: MagicMock) -> SecretsClient:
    with patch("boto3.client", return_value=mock_boto_client):
        return SecretsClient(cache_ttl_seconds=10)


def test_get_secret_string_returns_value(
    secrets_client: SecretsClient, mock_boto_client: MagicMock
) -> None:
    mock_boto_client.get_secret_value.return_value = {"SecretString": "top-secret-value"}
    result = secrets_client.get_secret_string("my/secret")
    assert result == "top-secret-value"
    mock_boto_client.get_secret_value.assert_called_once_with(SecretId="my/secret")


def test_cache_prevents_second_fetch(
    secrets_client: SecretsClient, mock_boto_client: MagicMock
) -> None:
    mock_boto_client.get_secret_value.return_value = {"SecretString": "value"}
    secrets_client.get_secret_string("my/secret")
    secrets_client.get_secret_string("my/secret")  # second call
    assert mock_boto_client.get_secret_value.call_count == 1


def test_expired_cache_refetches(
    secrets_client: SecretsClient, mock_boto_client: MagicMock
) -> None:
    mock_boto_client.get_secret_value.return_value = {"SecretString": "value"}
    secrets_client._ttl = 0  # force immediate expiry
    secrets_client.get_secret_string("my/secret")
    time.sleep(0.01)
    secrets_client.get_secret_string("my/secret")
    assert mock_boto_client.get_secret_value.call_count == 2


def test_client_error_raises_secrets_error(
    secrets_client: SecretsClient, mock_boto_client: MagicMock
) -> None:
    mock_boto_client.get_secret_value.side_effect = ClientError(
        {"Error": {"Code": "ResourceNotFoundException", "Message": "Not found"}},
        "GetSecretValue",
    )
    with pytest.raises(SecretsError, match="ResourceNotFoundException"):
        secrets_client.get_secret_string("nonexistent/secret")


def test_get_secret_json_parses_dict(
    secrets_client: SecretsClient, mock_boto_client: MagicMock
) -> None:
    payload = json.dumps({"username": "admin", "password": "REDACTED"})
    mock_boto_client.get_secret_value.return_value = {"SecretString": payload}
    result = secrets_client.get_secret_json("db/credentials")
    assert result["username"] == "admin"
```

---

## Exercises

1. **Full stack design**: Design the complete request path for a production RAG API endpoint. Include: Route53 → CloudFront (or skip — justify) → WAF → API Gateway (type and auth mechanism) → ALB → ECS Fargate service → Secrets Manager. For each component, state: what it does in this context, how it is configured (one sentence), and what happens if it fails. Draw the architecture as an ASCII diagram.

2. **WAF rule set**: Your AI API is being abused — large prompts (>50 KB request bodies) are causing slow responses and high LLM costs. Write the WAF rule in JSON (AWS Managed Rules format or custom rule) that blocks requests with a body larger than 16 KB. Also add a rate-based rule that blocks any single IP making more than 100 requests per 5 minutes.

3. **Secret rotation handler**: Sketch the four Lambda rotation steps (createSecret, setSecret, testSecret, finishSecret) as Python function signatures with docstrings. For the `testSecret` step, implement the body: fetch the pending secret, make a minimal API call to verify it works, raise an exception if it does not.

4. **ALB vs API Gateway decision**: You are building an AI platform with three endpoints: (a) `/chat` — streaming SSE, moderate traffic, (b) `/embed` — batch, internal only, (c) `/search` — public, needs per-tenant rate limiting and usage metering. For each endpoint, recommend ALB or API Gateway (or both) and justify. Identify the one endpoint where the choice is most consequential and explain why.

(Solutions: ../solutions/week-08/day-4.md)

---

## Self-check

- [ ] I can explain the two patterns for retrieving secrets at runtime and when to use each.
- [ ] I can describe the Secrets Manager rotation lifecycle (four steps) without notes.
- [ ] I can state three meaningful differences between ALB and API Gateway and when each is the right choice for an AI API.
- [ ] I can implement the `SecretsClient` class with caching, thread safety, and proper error handling without reference.
- [ ] I can identify three WAF rule types relevant to an AI API and describe what each protects against.

---

## Supplementary links

- [AWS Secrets Manager — Rotating secrets](https://docs.aws.amazon.com/secretsmanager/latest/userguide/rotating-secrets.html)
- [AWS Secrets Manager — Python SDK (boto3) reference](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/secretsmanager.html)
- [AWS ALB — How it works](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/introduction.html)
- [AWS API Gateway — HTTP vs REST APIs](https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api-vs-rest.html)
- [AWS WAF — Getting started](https://docs.aws.amazon.com/waf/latest/developerguide/getting-started.html)
- [AWS — Well-Architected security pillar: protecting data in transit and at rest](https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/protecting-data-in-transit.html)
- [GCP Secret Manager — Rotating secrets](https://cloud.google.com/secret-manager/docs/rotation-recommendations)
- [Azure Key Vault — Best practices](https://learn.microsoft.com/en-us/azure/key-vault/general/best-practices)
