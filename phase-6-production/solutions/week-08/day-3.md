# Week 8, Day 3 — Serverless for AI, Managed DBs, Caching: Solution Key

## Exercise 1: Lambda Handler Calling an AI API with Structured Output

```python
"""
Lambda handler: AI API call with structured output.

Runtime: Python 3.12
Handler: handler.lambda_handler
Memory: 512 MB (see Exercise 2)
Timeout: 30 s (see Exercise 2)
Required env vars: ANTHROPIC_API_KEY (from Secrets Manager via Lambda env injection)
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import anthropic
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# Module-level client — survives warm invocations, amortises SDK init cost.
_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    """Lazy-initialise Anthropic client; reused across warm Lambda invocations."""
    global _client
    if _client is None:
        api_key = os.environ["ANTHROPIC_API_KEY"]  # injected by Lambda env / Secrets Manager
        _client = anthropic.Anthropic(api_key=api_key)
        logger.info("Anthropic client initialised")
    return _client


class SentimentResult(BaseModel):
    sentiment: str = Field(description="positive | negative | neutral")
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="One-sentence rationale")


def _parse_structured_response(raw: str) -> SentimentResult:
    """Extract JSON block from model response and validate with Pydantic."""
    # Model is instructed to return JSON; strip markdown fences if present.
    clean = raw.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        clean = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    data = json.loads(clean)
    return SentimentResult.model_validate(data)


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """
    Event shape:
        { "text": "The product is amazing!" }

    Response shape (200):
        { "sentiment": "positive", "confidence": 0.95, "reasoning": "..." }
    """
    logger.info("Event received", extra={"request_id": context.aws_request_id})

    text: str | None = event.get("text")
    if not text or not text.strip():
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Field 'text' is required and must be non-empty"}),
        }

    client = _get_client()

    system_prompt = (
        "You are a sentiment analysis assistant. "
        "Always respond with a single JSON object matching this schema:\n"
        '{"sentiment": "positive|negative|neutral", "confidence": 0.0-1.0, "reasoning": "string"}\n'
        "No markdown, no explanation outside the JSON object."
    )

    try:
        message = client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=256,
            system=system_prompt,
            messages=[{"role": "user", "content": f"Analyse this text: {text}"}],
        )
        raw_content = message.content[0].text
        logger.debug("Raw model response", extra={"response": raw_content})

        result = _parse_structured_response(raw_content)
        logger.info(
            "Sentiment classified",
            extra={
                "sentiment": result.sentiment,
                "confidence": result.confidence,
                "request_id": context.aws_request_id,
            },
        )

        return {
            "statusCode": 200,
            "body": json.dumps(result.model_dump()),
            "headers": {"Content-Type": "application/json"},
        }

    except json.JSONDecodeError as exc:
        logger.error("Model returned non-JSON response", exc_info=exc)
        return {
            "statusCode": 502,
            "body": json.dumps({"error": "Model response was not valid JSON"}),
        }
    except anthropic.APIStatusError as exc:
        logger.error("Anthropic API error", extra={"status_code": exc.status_code}, exc_info=exc)
        return {
            "statusCode": 502,
            "body": json.dumps({"error": "Upstream AI API error", "detail": str(exc.message)}),
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error", exc_info=exc)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Internal server error"}),
        }
```

**Why module-level client**: Lambda reuses the execution environment between warm invocations. A module-level client is initialised once on cold start and reused for all warm invocations, saving ~200–400 ms of SDK init per request. This is standard Lambda optimisation practice.

**Why Pydantic validation**: The model response is untrusted external data. Pydantic validates types and constraints (e.g., `confidence` must be 0–1) before the response leaves the function, preventing downstream type errors.

---

## Exercise 2: Memory and Timeout Settings for an LLM Inference Lambda

### Memory sizing

Lambda memory directly controls allocated vCPU share (linear relationship: 1 792 MB = 1 full vCPU).

**For a lightweight inference Lambda (API proxy, no local model weights)**:
- SDK initialisation, JSON parsing, HTTP calls to external AI API — CPU-bound for < 100 ms.
- Recommended: **512 MB – 1 024 MB**.
- 512 MB gives 0.28 vCPU. Sufficient for I/O-bound tasks. 1 024 MB doubles CPU share if SDK parsing is a bottleneck.
- Increasing beyond 1 024 MB rarely helps for pure API proxy workloads — network RTT dominates.

**For a Lambda loading a local embedding model (e.g., sentence-transformers)**:
- Model weights: 90–500 MB in memory depending on model.
- Tokenisation + inference: CPU-intensive.
- Recommended: **3 008 MB** (1.68 vCPU) — gets you nearly 2 full CPUs without moving to the 4 096 MB tier.
- Use Graviton2 (`arm64`) architecture: ~20% better price/performance for Python workloads.

### Timeout sizing

```
timeout = p99_model_latency + network_overhead + buffer
```

| Scenario | Typical AI API latency | Recommended timeout |
|----------|----------------------|---------------------|
| Simple prompt, claude-haiku-class model | 1–3 s | 15 s |
| Long-context prompt, claude-sonnet-class | 5–20 s | 60 s |
| Streaming response (non-streaming Lambda) | Up to 120 s | Not suitable — use ECS |
| Local embedding model (MiniLM) | 50–200 ms | 10 s |

**Rule**: Set timeout to 3× the p99 you measured in testing. Never set it to 15 minutes unless you have a specific batch use case — that ties up concurrency slots and makes cascading failures possible.

**Cost calculation example** (512 MB, 5 s average duration, 100 req/s):
```
Invocations/month = 100 * 60 * 60 * 24 * 30 = 259,200,000
GB-seconds/month  = 259,200,000 * 5 * 0.5 = 648,000,000
Cost (compute)    = 648,000,000 * $0.0000166667 = $10,800
Free tier         = 400,000 GB-s free
Net cost          ≈ $10,793/month

Compare ECS Fargate (4 vCPU, 8 GB, always on):
= 2 tasks * $0.04048/vCPU-hr * 4 * 720 hr + 2 * $0.004445/GB-hr * 8 * 720
= $467 + $51 = $518/month
```
At sustained 100 req/s Lambda is 20× more expensive than Fargate. This illustrates why serverless is wrong for high-sustained-throughput AI workloads.

---

## Exercise 3: Aurora Serverless v2 vs Provisioned RDS for an AI App

### Aurora Serverless v2

**What it is**: Aurora capacity scales in Aurora Capacity Units (ACUs) between a minimum and maximum, billed per ACU-second. Scales up in < 1 second. Scales down to minimum ACUs (but not zero in v2 — unlike v1).

**When to choose it for an AI app**:
- Traffic is **spiky and unpredictable**: chat apps, demo platforms, event-driven pipelines. DB load follows LLM call volume, which can be bursty.
- You want **no capacity planning**: let AWS auto-scale DB compute. Useful for startups and new products where traffic patterns are unknown.
- Dev/test environments: set min_capacity = 0.5 ACU, pay near-zero during off-hours.
- **Multi-tenant SaaS** where each tenant's usage is small but many tenants may spike simultaneously.

**Limitations**:
- Slightly higher latency for the first query after a scale-up (< 1 s in v2, previously minutes in v1).
- ACU pricing is higher per unit than provisioned at sustained load. Break-even is roughly < 70% utilisation.
- I/O-optimised storage (new default) changes cost model — evaluate for write-heavy AI logging workloads.

### Provisioned RDS (Aurora or standard)

**When to choose it for an AI app**:
- Traffic is **predictable and sustained**: production search/RAG APIs with known steady-state QPS.
- You need **maximum performance**: provisioned gives consistent, predictable latency. No scale-up delays.
- Cost certainty: flat monthly bill, easier to forecast.
- Workloads requiring **Read Replicas with specific instance sizes** (e.g., r6g.8xlarge for analytics).

**Decision table**:

| Condition | Choose |
|-----------|--------|
| Traffic CV > 0.5 (high variance) | Aurora Serverless v2 |
| Traffic CV < 0.3 (stable) | Provisioned |
| Cost is primary concern, low avg load | Serverless v2 |
| SLA requires < 5 ms DB latency consistently | Provisioned |
| New product, uncertain traffic | Serverless v2 |
| pgvector for RAG (moderate writes, heavy reads) | Provisioned with read replicas |

**Hybrid pattern**: Use Aurora Serverless v2 for the primary writer. Add a provisioned Aurora read replica for high-frequency read workloads (embedding lookups, session retrieval). This gets you elastic write capacity with predictable read performance.

---

## Exercise 4: Caching Strategy for LLM Responses Using Redis

### Architecture

```
Request
  │
  ▼
[Cache Key Generator]
  │  key = sha256(model + system_prompt + user_message_normalized)
  │
  ▼
[Redis GET]──── HIT ───► Return cached response (add X-Cache: HIT header)
  │
  MISS
  │
  ▼
[LLM API Call]
  │
  ▼
[Redis SET with TTL]
  │
  ▼
Return response (X-Cache: MISS header)
```

### Python implementation

```python
"""
LLM response cache using Redis.
Requires: redis>=5.0, anthropic, pydantic
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Any

import anthropic
import redis.asyncio as aioredis
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class CacheConfig(BaseModel):
    ttl_seconds: int = 3600          # 1 hour default; tune per use case
    max_entry_bytes: int = 50_000    # skip caching oversized responses
    key_prefix: str = "llm:resp:"


class LLMCacheService:
    """Semantic-key LLM response cache backed by Redis."""

    def __init__(self, redis_url: str, config: CacheConfig | None = None) -> None:
        self._redis = aioredis.from_url(redis_url, decode_responses=True)
        self._config = config or CacheConfig()
        self._client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    @staticmethod
    def _normalise(text: str) -> str:
        """Normalise whitespace so trivially different prompts share a cache entry."""
        return re.sub(r"\s+", " ", text.strip().lower())

    def _cache_key(self, model: str, system: str, user_message: str) -> str:
        payload = json.dumps(
            {
                "model": model,
                "system": self._normalise(system),
                "user": self._normalise(user_message),
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode()).hexdigest()
        return f"{self._config.key_prefix}{digest}"

    async def complete(
        self,
        *,
        model: str,
        system: str,
        user_message: str,
        max_tokens: int = 1024,
        extra_metadata: dict[str, Any] | None = None,
    ) -> tuple[str, bool]:
        """
        Returns (response_text, cache_hit).
        cache_hit=True means the response came from Redis.
        """
        key = self._cache_key(model, system, user_message)

        cached = await self._redis.get(key)
        if cached:
            logger.info("Cache HIT", extra={"key_prefix": key[:20]})
            return cached, True

        logger.info("Cache MISS — calling LLM", extra={"key_prefix": key[:20]})
        message = await self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        response_text: str = message.content[0].text

        # Only cache if response is within size limit.
        if len(response_text.encode()) <= self._config.max_entry_bytes:
            await self._redis.setex(key, self._config.ttl_seconds, response_text)
            logger.info(
                "Cached response",
                extra={"ttl": self._config.ttl_seconds, "bytes": len(response_text.encode())},
            )
        else:
            logger.warning(
                "Response too large to cache",
                extra={"bytes": len(response_text.encode()), "limit": self._config.max_entry_bytes},
            )

        return response_text, False

    async def invalidate(self, model: str, system: str, user_message: str) -> None:
        key = self._cache_key(model, system, user_message)
        await self._redis.delete(key)
        logger.info("Cache entry invalidated", extra={"key_prefix": key[:20]})

    async def close(self) -> None:
        await self._redis.aclose()
```

### TTL strategy by use case

| Use case | Recommended TTL | Rationale |
|----------|----------------|-----------|
| Static FAQ / docs Q&A | 24 h | Answers don't change frequently |
| Product recommendations | 1–4 h | Inventory changes, but not per-minute |
| News summarisation | 15–30 min | Content freshness matters |
| Code generation (deterministic prompts) | 7 days | Same input → same good output |
| Personalised responses (user state in prompt) | **Do not cache** | Cache key would be unique per user; no hit rate benefit |
| Real-time analytics queries | **Do not cache** | Data changes; stale answers are harmful |

### Eviction policy

Set Redis `maxmemory-policy` to `allkeys-lru` for LLM caches. When memory is full, LRU evicts the least-recently-used entry — naturally keeps hot prompts in cache. Alternative: `volatile-lru` if you mix cached data with session data (only evicts keys with TTL set).

### Semantic cache (advanced)

For higher hit rates, replace the exact-string hash with an **embedding similarity lookup**:
1. Embed the user message with `text-embedding-3-small` (or equivalent).
2. Store embedding vector in Redis using `redis-py` with RedisSearch / RediSearch module.
3. On incoming request, embed query and search for nearest neighbour with cosine similarity > 0.97.
4. If found, return cached response without calling LLM.

This captures paraphrase equivalences ("What is the capital of France?" / "France's capital city?") that exact-hash misses. Hit rate improvement: 15–40% depending on domain.
