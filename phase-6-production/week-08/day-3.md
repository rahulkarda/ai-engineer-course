# Phase 6 · Week 08 · Day 3 — Serverless for AI + Managed Databases and Caches

> Objective: Design Lambda-based AI inference handlers correctly, understand cold start trade-offs, and choose managed data services for AI app state.  ·  Time: ~1 hr  ·  Prereqs: [[week-08/day-2]]

---

## Concept

### Serverless (FaaS) for AI workloads

**Lambda** (AWS), **Cloud Functions** (GCP), **Azure Functions** execute a handler in response to an event. The platform provisions compute, handles concurrency, and bills per 100 ms of execution and per GB-second of memory.

For AI workloads, three constraints dominate:

| Constraint | Lambda limit | AI impact |
|---|---|---|
| Max memory | 10 GB | Enough for small models; not for 7B+ |
| Max timeout | 15 min | Fine for inference; not for training |
| Deployment package | 250 MB unzipped (or container up to 10 GB) | Use container images for anything with torch/transformers |
| Ephemeral storage (`/tmp`) | 10 GB (configurable) | Usable for model cache per invocation |

**When FaaS is right for AI:**
- Wrapping a managed LLM API (Bedrock, Claude API) — the handler is I/O-bound.
- Async document processing triggered by S3 events.
- Lightweight inference where model fits comfortably in memory.
- Scheduled tasks (re-embedding, index refresh).

**When FaaS is wrong:**
- Real-time inference from a locally loaded large model (cold start + memory).
- Streaming responses longer than Lambda's 15-min timeout (use HTTP streaming via Function URLs or a container service instead).
- Workloads needing GPU (Lambda has no GPU support).

### Cold starts

A cold start occurs when Lambda has no warm execution environment available. The platform:
1. Allocates a microVM.
2. Downloads and initialises the runtime.
3. Runs your initialisation code (module-level imports, model loading).
4. Invokes the handler.

Steps 1–3 add latency ranging from 200 ms (minimal Python handler) to 10+ seconds (large container image with heavy imports). Mitigation strategies:

- **Provisioned Concurrency**: pre-warm N instances; you pay for them even when idle.
- **Keep-warm pings**: scheduled EventBridge event to invoke the function every few minutes.
- **Minimise package size**: lazy imports, slim base images.
- **Move model loading outside the handler**: module-level code runs once per container lifetime.

### Concurrency model

Lambda scales by spawning **one execution environment per concurrent request**. Each environment handles exactly one request at a time (no threading per environment). Account-level default soft limit: 1,000 concurrent executions. For AI APIs that call a downstream LLM with rate limits, this means Lambda's scaling can overrun the upstream rate limit — design for back-pressure (retry with exponential back-off, reserve concurrency).

---

### Managed databases for AI apps

| Service | Type | AI use case |
|---|---|---|
| RDS PostgreSQL | Relational + pgvector extension | Vector similarity search at moderate scale |
| Aurora Serverless v2 | Serverless relational | Conversation history, user data; auto-pause when idle |
| AlloyDB (GCP) | Postgres-compatible, columnar | High-throughput embedding queries |
| DynamoDB | Key-value / document | Session state, per-user conversation context |
| S3 | Object store | Raw documents, model artefacts, embeddings at rest |

**Aurora Serverless v2** is particularly useful for AI apps with bursty traffic: it scales ACUs (Aurora Capacity Units) from minimum (0.5 ACU) up to max in seconds, and can pause completely after a configurable idle period — cost-effective for development and low-traffic production.

### Managed caches

**ElastiCache (Redis)** and **MemoryDB** are the primary options on AWS:

| Use in AI apps | Pattern |
|---|---|
| Prompt/response caching | `key = hash(prompt + model + params)` → cache hit avoids LLM API call |
| Semantic cache | Store embedding + response; on new request, compute cosine similarity to cached embeddings — return if above threshold |
| Rate limiting | Token bucket per user/API key using Redis `INCR` + `EXPIRE` |
| Session state | Conversation history keyed by session ID; TTL-based expiry |

Redis's **single-threaded event loop** means it is fast for small values but not for large binary blobs. Do not store raw model weights in Redis. Keep individual cache entries under 1 MB.

---

## How it works internally

### Lambda execution environment lifecycle

```
Cold path:
  Event → [microVM boot] → [runtime init] → [module-level code] → handler() → response
                                                                      ↑
                                              stays warm for ~15 min of inactivity

Warm path:
  Event → handler() → response   (microVM already warm, module-level already ran)
```

The execution environment is a **Firecracker microVM** — lightweight virtualisation with ~125 ms boot time. The function package (or container image) is layered on top. Lambda's internal load balancer routes requests to warm environments first; only spawns new ones when all existing environments are busy.

**Provisioned Concurrency** pre-initialises N environments and keeps them warm indefinitely. These are billed at a lower per-hour rate than on-demand but still represent a baseline cost. For latency-sensitive AI endpoints, provisioned concurrency is the correct lever — not code optimisation alone.

### Aurora Serverless v2 scaling

Aurora v2 maintains a **warm pool** of capacity at the configured minimum ACU level. Scaling up is done by adjusting the ACU assignment on the same DB instance (no failover, no connection drop). This is different from v1 which had a visible pause. The scale-up trigger is CPU or connections crossing a threshold; scale-out latency is ~1–5 seconds.

---

## Worked example

Lambda function wrapping an AI inference call. Uses `boto3` for Bedrock (Claude model) with structured logging, typed input/output, and correct error handling.

```python
# lambda_ai_handler.py
# Deploy as a Lambda function (container image recommended for production).
# Environment variables: BEDROCK_MODEL_ID, AWS_REGION (auto-set by Lambda runtime)
# IAM: Lambda execution role needs bedrock:InvokeModel on the configured model ARN.

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

# boto3 is pre-installed in the Lambda runtime; no package needed.
import boto3
from botocore.exceptions import ClientError

# Module-level initialisation: runs ONCE per execution environment (not per request).
# This is the correct place for expensive setup (clients, model loading, config).
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

_REGION = os.getenv("AWS_REGION", "us-east-1")
_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-sonnet-5-5-v1:0")
_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

# Initialise the Bedrock client once — reused across warm invocations.
_bedrock_client = boto3.client("bedrock-runtime", region_name=_REGION)

logger.info(
    "Execution environment initialised",
    extra={
        "model_id": _MODEL_ID,
        "region": _REGION,
    },
)


class InferenceError(Exception):
    """Raised when Bedrock invocation fails."""


def _invoke_model(prompt: str) -> str:
    """
    Invoke a Claude model via Bedrock.
    Returns the assistant response text.
    Raises InferenceError on Bedrock client errors.
    """
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": _MAX_TOKENS,
        "temperature": _TEMPERATURE,
        "messages": [
            {"role": "user", "content": prompt},
        ],
    }

    try:
        response = _bedrock_client.invoke_model(
            modelId=_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(request_body),
        )
    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        logger.error(
            "Bedrock ClientError",
            extra={"error_code": error_code, "model_id": _MODEL_ID},
        )
        raise InferenceError(f"Bedrock error: {error_code}") from exc

    response_body: dict[str, Any] = json.loads(response["body"].read())
    return response_body["content"][0]["text"]


def lambda_handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    """
    Lambda entry point.
    Expects event: {"prompt": str}
    Returns: {"statusCode": int, "body": str (JSON)}
    """
    start_ms = time.monotonic_ns() // 1_000_000

    # --- Input validation ---
    prompt: str | None = event.get("prompt")
    if not prompt or not isinstance(prompt, str):
        logger.warning("Missing or invalid prompt in event")
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Field 'prompt' is required and must be a string."}),
        }

    prompt = prompt.strip()
    if len(prompt) > 10_000:
        logger.warning("Prompt exceeds 10k character limit", extra={"length": len(prompt)})
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Prompt exceeds 10,000 character limit."}),
        }

    logger.info(
        "Inference request received",
        extra={
            "prompt_length": len(prompt),
            "aws_request_id": getattr(context, "aws_request_id", "local"),
        },
    )

    # --- Inference ---
    try:
        response_text = _invoke_model(prompt)
    except InferenceError as exc:
        logger.error("Inference failed", extra={"error": str(exc)})
        return {
            "statusCode": 502,
            "body": json.dumps({"error": "Inference service unavailable. Retry with backoff."}),
        }

    elapsed_ms = (time.monotonic_ns() // 1_000_000) - start_ms
    logger.info(
        "Inference complete",
        extra={
            "elapsed_ms": elapsed_ms,
            "response_length": len(response_text),
        },
    )

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "response": response_text,
                "model_id": _MODEL_ID,
                "elapsed_ms": elapsed_ms,
            }
        ),
    }
```

**Key patterns demonstrated:**
- `_bedrock_client` initialised at module level — reused across warm invocations.
- All config via `os.getenv` — no hardcoded values.
- Structured logging with `extra={}` — parseable by CloudWatch Logs Insights.
- Input validation before any API call.
- Specific error handling: `ClientError` caught and wrapped; no bare `except Exception`.
- Timing measured and logged for cold-start analysis.
- No secrets in code or event payload — IAM role grants Bedrock access.

### Testing locally (no Lambda runtime needed)

```python
# test_lambda_handler.py
import json
from unittest.mock import MagicMock, patch

import pytest

# Patch boto3 before importing the handler to avoid credential errors locally.
with patch("boto3.client"):
    from lambda_ai_handler import lambda_handler


@pytest.fixture()
def mock_bedrock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_client = MagicMock()
    mock_response_body = json.dumps(
        {"content": [{"text": "Hello from mock Claude."}]}
    ).encode()
    mock_client.invoke_model.return_value = {
        "body": MagicMock(read=MagicMock(return_value=mock_response_body))
    }
    import lambda_ai_handler
    monkeypatch.setattr(lambda_ai_handler, "_bedrock_client", mock_client)
    return mock_client


def test_valid_prompt_returns_200(mock_bedrock: MagicMock) -> None:
    event = {"prompt": "What is 2 + 2?"}
    result = lambda_handler(event, context=MagicMock())
    assert result["statusCode"] == 200
    body = json.loads(result["body"])
    assert body["response"] == "Hello from mock Claude."


def test_missing_prompt_returns_400() -> None:
    result = lambda_handler({}, context=MagicMock())
    assert result["statusCode"] == 400


def test_oversized_prompt_returns_400() -> None:
    result = lambda_handler({"prompt": "x" * 10_001}, context=MagicMock())
    assert result["statusCode"] == 400
```

---

## Exercises

1. **Cold start mitigation**: You have a Lambda function that imports `anthropic` (the Python SDK) and loads a small config object at startup. Currently p99 cold start latency is 3.2 seconds. List four concrete changes — in order of expected impact — that would reduce this. For each, estimate the reduction and note the cost trade-off.

2. **Concurrency design**: Your Lambda AI handler calls an LLM API with a rate limit of 100 requests per minute. Your Lambda can scale to 1,000 concurrent executions. Design the back-pressure mechanism: what Lambda settings do you configure, and what retry logic does the caller implement? Write the retry logic in Python.

3. **Database selection**: Design the data layer for a multi-user chatbot with these requirements: (a) conversation history per session (up to 50 messages), (b) semantic search over a 500k document corpus, (c) user preference settings (small JSON, per user). For each, choose a specific managed service, justify the choice, and sketch the access pattern (key structure or query).

4. **Prompt caching with Redis**: Implement a Python function `cached_inference(prompt: str, ttl_seconds: int) -> str` that checks a Redis cache before calling the LLM. Use `redis.asyncio` for the Redis client and `os.getenv` for the Redis URL. The cache key should be a SHA-256 hash of the prompt. Do not cache error responses.

(Solutions: ../solutions/week-08/day-3.md)

---

## Self-check

- [ ] I can explain Lambda's execution environment lifecycle and identify which code runs once vs per-request.
- [ ] I can list the three Lambda constraints most relevant to AI workloads and their limits.
- [ ] I can explain the difference between provisioned concurrency and keep-warm pings.
- [ ] I can implement the Lambda handler with proper structured logging and input validation without reference.
- [ ] I can choose between RDS, Aurora Serverless, DynamoDB, and Redis for a given AI app data access pattern.

---

## Supplementary links

- [AWS Lambda — Function configuration](https://docs.aws.amazon.com/lambda/latest/dg/configuration-function-common.html)
- [AWS Lambda — Container images](https://docs.aws.amazon.com/lambda/latest/dg/images-create.html)
- [AWS Lambda — Provisioned concurrency](https://docs.aws.amazon.com/lambda/latest/dg/provisioned-concurrency.html)
- [AWS Bedrock — Invoke model API](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html)
- [Aurora Serverless v2 — How it works](https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/aurora-serverless-v2.how-it-works.html)
- [ElastiCache — Best practices for caching](https://docs.aws.amazon.com/AmazonElastiCache/latest/red-ug/BestPractices.html)
- [AWS Firecracker — MicroVM architecture](https://firecracker-microvm.github.io/)
