# Week 8, Day 4 — Secrets Manager, Load Balancers, API Gateway: Solution Key

## Exercise 1: Python Code to Fetch a Secret from AWS Secrets Manager with Caching

The naive pattern (call `GetSecretValue` on every request) creates latency spikes and risks hitting API rate limits (default 10 000 requests/s per region, but per-account limits apply). The correct pattern caches the secret in-process and refreshes only after TTL expiry.

```python
"""
Secrets Manager client with in-process TTL cache.

Why in-process cache vs SSM Parameter Store:
- Secrets Manager supports automatic rotation; cache must handle mid-TTL rotation.
- TTL of 5 minutes ensures stale credentials live at most 5 minutes after rotation.
- No external cache (Redis) needed — avoids network hop + cache invalidation complexity.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    value: dict[str, Any]
    fetched_at: float


class SecretsManagerCache:
    """Thread-safe, TTL-based cache for AWS Secrets Manager secrets."""

    def __init__(
        self,
        region: str = "us-east-1",
        ttl_seconds: int = 300,  # 5 minutes — refresh well before rotation window
    ) -> None:
        self._client = boto3.client("secretsmanager", region_name=region)
        self._ttl = ttl_seconds
        self._cache: dict[str, _CacheEntry] = {}
        self._lock = Lock()

    def get_secret(self, secret_id: str) -> dict[str, Any]:
        """
        Return parsed secret dict. Fetches from AWS if cache is empty or stale.
        Thread-safe via lock; concurrent callers for the same key wait for the fetch.
        """
        with self._lock:
            entry = self._cache.get(secret_id)
            if entry and (time.monotonic() - entry.fetched_at) < self._ttl:
                logger.debug("Secret cache HIT", extra={"secret_id": secret_id})
                return entry.value

            logger.info("Secret cache MISS — fetching from Secrets Manager", extra={"secret_id": secret_id})
            value = self._fetch(secret_id)
            self._cache[secret_id] = _CacheEntry(value=value, fetched_at=time.monotonic())
            return value

    def get_string(self, secret_id: str, key: str) -> str:
        """Convenience: get a single key from a JSON secret."""
        secret = self.get_secret(secret_id)
        if key not in secret:
            raise KeyError(f"Key '{key}' not found in secret '{secret_id}'")
        return str(secret[key])

    def invalidate(self, secret_id: str) -> None:
        """Force re-fetch on next access — call after catching auth errors."""
        with self._lock:
            self._cache.pop(secret_id, None)
            logger.info("Secret cache invalidated", extra={"secret_id": secret_id})

    def _fetch(self, secret_id: str) -> dict[str, Any]:
        try:
            response = self._client.get_secret_value(SecretId=secret_id)
        except ClientError as exc:
            error_code = exc.response["Error"]["Code"]
            if error_code == "ResourceNotFoundException":
                raise ValueError(f"Secret not found: {secret_id}") from exc
            if error_code in ("InvalidRequestException", "InvalidParameterException"):
                raise ValueError(f"Invalid secret request for {secret_id}: {exc}") from exc
            if error_code == "DecryptionFailureException":
                raise RuntimeError(f"KMS decryption failed for {secret_id}") from exc
            raise  # re-raise unexpected errors

        raw = response.get("SecretString")
        if raw is None:
            raise ValueError(f"Secret '{secret_id}' is binary — only string secrets supported")

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Secret is a plain string, not JSON — wrap in dict for uniform interface
            return {"value": raw}


# -------------------------------------------------------------------
# Usage pattern in application startup
# -------------------------------------------------------------------

# Module-level singleton — one cache per Lambda execution environment / ECS task.
_secrets_cache = SecretsManagerCache(region="us-east-1", ttl_seconds=300)


def get_anthropic_api_key() -> str:
    """Fetch Anthropic API key, using cache to avoid per-request Secrets Manager calls."""
    return _secrets_cache.get_string(
        secret_id="prod/ai-api/anthropic",
        key="api_key",
    )


def get_db_credentials() -> dict[str, Any]:
    """Fetch DB credentials dict {username, password, host, port}."""
    return _secrets_cache.get_secret("prod/ai-api/database")
```

**Why TTL-based over event-driven invalidation**: Secrets Manager rotation runs on a schedule (every N days). With 5-minute TTL, stale credentials live at most 5 minutes. True event-driven invalidation would require a Lambda rotation hook to publish to SNS/SQS — unnecessary complexity for most apps.

**Why thread lock**: ECS tasks run multi-threaded Python WSGI/ASGI servers. Without a lock, two threads can both see a cache miss and make concurrent `GetSecretValue` calls — harmless but wasteful. The lock serialises the fetch; second caller gets the freshly cached value.

---

## Exercise 2: ALB + API Gateway + ECS Architecture for a RAG Endpoint

```
                         ┌──────────────────────────────────────────────┐
                         │  Route 53 (DNS)                               │
                         │  api.example.com → API Gateway domain        │
                         └─────────────────────┬────────────────────────┘
                                               │ HTTPS
                         ┌─────────────────────▼────────────────────────┐
                         │  API Gateway (HTTP API)                       │
                         │  ┌─────────────────────────────────────────┐ │
                         │  │ Routes:                                 │ │
                         │  │  POST /v1/query  → VPC Link → ALB       │ │
                         │  │  GET  /v1/health → Lambda (cheap check) │ │
                         │  ├─────────────────────────────────────────┤ │
                         │  │ Auth: JWT Authorizer (Cognito / custom)  │ │
                         │  │ Throttling: 1 000 req/s burst            │ │
                         │  │ Usage plans per API key                  │ │
                         │  └─────────────────────────────────────────┘ │
                         └─────────────────────┬────────────────────────┘
                                               │ VPC Link (private)
                         ┌─────────────────────▼────────────────────────┐
                         │  ALB (internal, multi-AZ)                    │
                         │  Listener: HTTP 80 (VPC only)                │
                         │  Target group: ECS service (IP targets)      │
                         │  Health check: GET /health, 2xx             │
                         └──────────┬───────────────────────────────────┘
                   ┌────────────────┼────────────────┐
                   │                │                │
         ┌─────────▼──────┐  ┌─────▼──────┐  ┌─────▼──────┐
         │  ECS Task 1    │  │ ECS Task 2 │  │ ECS Task N │
         │  RAG API       │  │ RAG API    │  │ RAG API    │
         │  (Fargate)     │  │ (Fargate)  │  │ (Fargate)  │
         └────────┬───────┘  └─────┬──────┘  └─────┬──────┘
                  │                │                │
         ┌────────▼────────────────▼────────────────▼──────┐
         │  Private Subnet (no internet access)             │
         │                                                  │
         │  ┌─────────────┐  ┌──────────────┐  ┌────────┐ │
         │  │  Aurora      │  │  ElastiCache  │  │  S3    │ │
         │  │  PostgreSQL  │  │  Redis        │  │ (docs) │ │
         │  │  + pgvector  │  │  (LLM cache)  │  │        │ │
         │  └─────────────┘  └──────────────┘  └────────┘ │
         └──────────────────────────────────────────────────┘
```

**Data flow for a RAG query**:
1. Client sends `POST /v1/query` with Bearer token to API Gateway.
2. JWT Authorizer validates token; rejects with 401 if invalid (never reaches ECS).
3. API Gateway forwards over VPC Link to internal ALB — no public internet exposure for ECS.
4. ALB round-robins to a healthy ECS task. Connection draining on scale-in.
5. ECS task: checks Redis cache → if miss, queries pgvector (Aurora), reranks, calls Claude API, stores result in Redis.
6. Response streams back through ALB → VPC Link → API Gateway → client.

**Security groups**:
- ALB SG: inbound 80 from API Gateway VPC Link CIDR only.
- ECS SG: inbound 8080 from ALB SG only.
- Aurora SG: inbound 5432 from ECS SG only.
- Redis SG: inbound 6379 from ECS SG only.

---

## Exercise 3: When to Use ALB Directly vs API Gateway in Front

### Use ALB directly (skip API Gateway) when:

1. **Internal service-to-service traffic**: Microservices within the same VPC calling each other. API Gateway's public endpoint is unnecessary overhead; an internal ALB is simpler and cheaper.
2. **WebSocket or long-lived streaming connections**: API Gateway HTTP API supports WebSocket but has a 29-second idle timeout for REST APIs. ALB supports WebSocket natively with no per-message charge.
3. **Very high sustained throughput (> 10 000 req/s)**: API Gateway has account-level throttling limits. ALB scales to millions of req/s with no per-request cost.
4. **gRPC**: ALB supports gRPC natively. API Gateway does not.
5. **Cost sensitivity at scale**: API Gateway REST API costs ~$3.50 per million requests. ALB costs ~$0.008 per LCU-hour (very roughly $0.05–0.20 per million requests at typical usage). At 100M req/month: API Gateway = $350 vs ALB ≈ $10.
6. **ECS service-to-service with service discovery**: Use Cloud Map + ALB or just service DNS. No API Gateway needed.

### Use API Gateway in front of ALB when:

1. **Auth/AuthZ at the edge**: JWT authorizers, Cognito user pools, custom Lambda authorizers. Keeps auth logic out of application code.
2. **Rate limiting and usage plans**: Per-API-key throttling, usage tiers. API Gateway manages this natively; implementing it at ALB requires WAF + custom logic.
3. **Public API with developer portal / SDK generation**: API Gateway integrates with AWS marketplace and generates client SDKs.
4. **Request/response transformation**: Mapping templates to reshape payloads before they reach ECS (e.g., legacy client format → modern backend format).
5. **Caching at edge**: API Gateway cache reduces backend calls for GET endpoints. ALB has no response cache.
6. **Multi-backend routing**: Route `/v1/model-a` to one ECS service and `/v1/model-b` to another, with API Gateway managing routing logic.

**Summary rule**: ALB alone if internal, high-volume, or streaming. API Gateway + ALB if you need auth, per-client throttling, or a public-facing developer API surface.

---

## Exercise 4: WAF Rule to Rate-Limit an AI API Endpoint

### Approach

AWS WAF v2 rate-based rules block IPs that exceed a threshold within a 5-minute window. For AI APIs, two rules are needed: a per-IP rate limit and a per-API-key rate limit (header-based).

```json
{
  "Name": "AIAPIRateLimitWebACL",
  "DefaultAction": { "Allow": {} },
  "Rules": [
    {
      "Name": "PerIPRateLimit",
      "Priority": 10,
      "Statement": {
        "RateBasedStatement": {
          "Limit": 100,
          "AggregateKeyType": "IP",
          "ScopeDownStatement": {
            "ByteMatchStatement": {
              "SearchString": "/v1/query",
              "FieldToMatch": { "UriPath": {} },
              "TextTransformations": [{ "Priority": 0, "Type": "LOWERCASE" }],
              "PositionalConstraint": "STARTS_WITH"
            }
          }
        }
      },
      "Action": {
        "Block": {
          "CustomResponse": {
            "ResponseCode": 429,
            "CustomResponseBodyKey": "rate_limit_body"
          }
        }
      },
      "VisibilityConfig": {
        "SampledRequestsEnabled": true,
        "CloudWatchMetricsEnabled": true,
        "MetricName": "PerIPRateLimit"
      }
    },
    {
      "Name": "PerAPIKeyRateLimit",
      "Priority": 20,
      "Statement": {
        "RateBasedStatement": {
          "Limit": 500,
          "AggregateKeyType": "CUSTOM_KEYS",
          "CustomKeys": [
            {
              "Header": {
                "Name": "x-api-key",
                "TextTransformations": [{ "Priority": 0, "Type": "NONE" }]
              }
            }
          ],
          "ScopeDownStatement": {
            "ByteMatchStatement": {
              "SearchString": "/v1/",
              "FieldToMatch": { "UriPath": {} },
              "TextTransformations": [{ "Priority": 0, "Type": "LOWERCASE" }],
              "PositionalConstraint": "STARTS_WITH"
            }
          }
        }
      },
      "Action": {
        "Block": {
          "CustomResponse": {
            "ResponseCode": 429,
            "CustomResponseBodyKey": "rate_limit_body"
          }
        }
      },
      "VisibilityConfig": {
        "SampledRequestsEnabled": true,
        "CloudWatchMetricsEnabled": true,
        "MetricName": "PerAPIKeyRateLimit"
      }
    },
    {
      "Name": "BlockMalformedJSON",
      "Priority": 5,
      "Statement": {
        "AndStatement": {
          "Statements": [
            {
              "ByteMatchStatement": {
                "SearchString": "/v1/query",
                "FieldToMatch": { "UriPath": {} },
                "TextTransformations": [{ "Priority": 0, "Type": "LOWERCASE" }],
                "PositionalConstraint": "STARTS_WITH"
              }
            },
            {
              "NotStatement": {
                "Statement": {
                  "ByteMatchStatement": {
                    "SearchString": "application/json",
                    "FieldToMatch": {
                      "SingleHeader": { "Name": "content-type" }
                    },
                    "TextTransformations": [{ "Priority": 0, "Type": "LOWERCASE" }],
                    "PositionalConstraint": "CONTAINS"
                  }
                }
              }
            }
          ]
        }
      },
      "Action": { "Block": { "CustomResponse": { "ResponseCode": 415 } } },
      "VisibilityConfig": {
        "SampledRequestsEnabled": true,
        "CloudWatchMetricsEnabled": true,
        "MetricName": "BlockMalformedJSON"
      }
    }
  ],
  "CustomResponseBodies": {
    "rate_limit_body": {
      "ContentType": "APPLICATION_JSON",
      "Content": "{\"error\": \"rate_limit_exceeded\", \"retry_after_seconds\": 60}"
    }
  },
  "VisibilityConfig": {
    "SampledRequestsEnabled": true,
    "CloudWatchMetricsEnabled": true,
    "MetricName": "AIAPIWebACL"
  }
}
```

**Why two separate rate rules**:
- `PerIPRateLimit` (100 req / 5 min per IP): protects against single-IP abuse — scrapers, runaway clients.
- `PerAPIKeyRateLimit` (500 req / 5 min per API key): protects against compromised keys being used at scale from multiple IPs. The IP rule wouldn't catch a distributed attack using a single stolen API key.

**Why `STARTS_WITH "/v1/"` scope-down**: Rate rules apply only to AI API paths. Health check endpoints (`/health`, `/metrics`) are excluded — their traffic pattern is legitimately high-frequency from ALB health checks and monitoring.

**Threshold calibration**:
```
PerIPRateLimit threshold = (expected peak req/s per normal user) * 300 s * safety_factor
= 0.1 req/s * 300 * 3 = 90 → round to 100

PerAPIKeyRateLimit threshold = (max allowed sustained rate per key) * 300 s
= If SLA allows 1 req/s per API key: 1 * 300 * safety = 300-500
```

**Additional rules to add in production**:
- AWS Managed Rule Group: `AWSManagedRulesKnownBadInputsRuleSet` — blocks Log4Shell, XXE, SSRF payloads.
- `SizeConstraintStatement` on request body: reject bodies > 100 KB to prevent prompt injection via massive payloads.
- Geo-restriction if the product is region-specific.

**Attach WAF to ALB** (Terraform snippet):
```hcl
resource "aws_wafv2_web_acl_association" "ai_api" {
  resource_arn = aws_lb.ai_api.arn
  web_acl_arn  = aws_wafv2_web_acl.ai_api.arn
}
```

Set `scope = "REGIONAL"` in the Web ACL (not `CLOUDFRONT`) when attaching to an ALB.
