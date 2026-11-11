# Phase 6 · Week 09 · Day 5 — Cost Attribution + A/B Testing Models + Request/Response Logging

> Objective: Tag every LLM request with tenant/user/feature dimensions, implement deterministic A/B model routing, and emit structured JSON audit logs without leaking PII.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

---

## Concept

A production gateway must answer three operational questions at any moment:

1. **Who is spending what?** — cost attribution per tenant, user, and feature
2. **Which model performs better for this use case?** — A/B testing with controlled rollout
3. **What happened on this request?** — structured audit log for debugging and compliance

These three concerns are orthogonal but compose naturally: every request generates one `CostRecord`, one routing decision (possibly from an `ABExperiment`), and one `LogEntry`.

---

## How it works internally

### Cost Attribution

Every call to a model has a measurable token cost. To allocate that cost accurately:

- **Tag at call site**: before dispatching, stamp the request with `tenant_id`, `user_id`, and `feature_tag` (e.g., `"summarization"`, `"chat"`, `"code-review"`).
- **Record on response**: when the provider returns usage, create a `CostRecord` and hand it to `CostAttributor`.
- **Aggregate on demand**: `CostAttributor` accumulates cost by `(tenant_id, feature_tag)` for billing exports.

Token-to-USD conversion (Anthropic, 2024 reference — verify against current pricing):

```
input_cost  = (input_tokens  / 1_000_000) * price_per_m_input
output_cost = (output_tokens / 1_000_000) * price_per_m_output
cost_usd    = input_cost + output_cost
```

### A/B Testing Models

Goal: split traffic between two model variants and measure which performs better (quality, latency, cost) without exposing a single user to inconsistent behavior across requests.

**Deterministic assignment** — critical requirement:

```
variant_index = hash(user_id + experiment_id) % total_weight_units
```

Using a stable hash (e.g., `hashlib.md5`) means the same `(user_id, experiment_id)` pair always maps to the same variant. This prevents the same user from seeing different models on repeated requests, which would corrupt any quality measurement.

**Weight normalization**: weights are floats summing to 1.0. Convert to integer buckets (multiply by 100, use modulo) for reliable integer arithmetic.

### Request/Response Logging

Structured logging sends every log record as a parseable JSON object — no free-text `print()` calls. Python's `logging` module supports this via a custom `Formatter`.

Rules for production LLM logging:

| Log | Do NOT log |
|-----|------------|
| `request_id`, `tenant_id`, `model` | Message content (user prompts, completions) |
| `latency_ms`, `input_tokens`, `output_tokens` | PII in any field |
| `cost_usd`, `status_code` | API keys or auth headers |
| `experiment_id`, `variant` | Full response objects from the provider |

A `LogEntry` Pydantic model enforces this contract: only explicitly declared fields are logged.

---

## Worked example

```python
# models.py
import hashlib
import os
import logging
import json
from datetime import datetime, UTC
from typing import Optional
import uuid
from pydantic import BaseModel, Field


# ── Cost Attribution ──────────────────────────────────────────────

class CostRecord(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    user_id: str
    feature_tag: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CostAttributor:
    def __init__(self) -> None:
        self._records: list[CostRecord] = []

    def record(self, rec: CostRecord) -> None:
        self._records.append(rec)

    def aggregate(self, tenant_id: str) -> dict[str, float]:
        """Return {feature_tag: total_cost_usd} for a tenant."""
        totals: dict[str, float] = {}
        for r in self._records:
            if r.tenant_id == tenant_id:
                totals[r.feature_tag] = totals.get(r.feature_tag, 0.0) + r.cost_usd
        return totals

    def total(self, tenant_id: str) -> float:
        return sum(r.cost_usd for r in self._records if r.tenant_id == tenant_id)


# ── A/B Routing ───────────────────────────────────────────────────

class ABVariant(BaseModel):
    model: str          # e.g. "claude-haiku-5-5"
    provider: str       # e.g. "anthropic"
    weight: float       # 0.0 – 1.0; all variants must sum to 1.0


class ABExperiment(BaseModel):
    experiment_id: str
    variants: list[ABVariant]
    active: bool = True

    def model_post_init(self, __context: object) -> None:
        total = sum(v.weight for v in self.variants)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Variant weights must sum to 1.0, got {total}")


class ModelABRouter:
    def __init__(self, experiment: ABExperiment) -> None:
        self._exp = experiment

    def assign(self, user_id: str) -> ABVariant:
        """Deterministic assignment: same user always gets same variant."""
        if not self._exp.active:
            # Fallback: always first variant when experiment is inactive
            return self._exp.variants[0]

        key = f"{user_id}:{self._exp.experiment_id}"
        digest = hashlib.md5(key.encode()).hexdigest()  # noqa: S324 — not crypto use
        bucket = int(digest, 16) % 100  # 0 – 99

        cumulative = 0
        for variant in self._exp.variants:
            cumulative += int(variant.weight * 100)
            if bucket < cumulative:
                return variant

        return self._exp.variants[-1]  # safety fallback


# ── Structured Logging ────────────────────────────────────────────

class LogEntry(BaseModel):
    request_id: str
    tenant_id: str
    user_id: str
    model: str
    provider: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    status_code: int
    experiment_id: Optional[str] = None
    variant_model: Optional[str] = None
    feature_tag: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = record.__dict__.get("extra_data", {})
        payload["level"] = record.levelname
        payload["logger"] = record.name
        payload["message"] = record.getMessage()
        return json.dumps(payload, default=str)


def build_gateway_logger(name: str = "gateway") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


# ── Usage ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Setup
    logger = build_gateway_logger()
    attributor = CostAttributor()

    experiment = ABExperiment(
        experiment_id="exp-model-tier",
        variants=[
            ABVariant(model="claude-haiku-5-5",   provider="anthropic", weight=0.5),
            ABVariant(model="claude-sonnet-5-5",  provider="anthropic", weight=0.5),
        ],
    )
    router = ModelABRouter(experiment)

    # Simulate 3 requests
    for user_id in ["user-001", "user-002", "user-001"]:  # user-001 appears twice
        variant = router.assign(user_id)
        print(f"{user_id} → {variant.model}")
        # user-001 must map to the same variant both times

        rec = CostRecord(
            tenant_id="acme",
            user_id=user_id,
            feature_tag="chat",
            model=variant.model,
            provider=variant.provider,
            input_tokens=500,
            output_tokens=150,
            cost_usd=0.00032,
        )
        attributor.record(rec)

        entry = LogEntry(
            request_id=rec.request_id,
            tenant_id=rec.tenant_id,
            user_id=user_id,
            model=variant.model,
            provider=variant.provider,
            latency_ms=320.0,
            input_tokens=rec.input_tokens,
            output_tokens=rec.output_tokens,
            cost_usd=rec.cost_usd,
            status_code=200,
            experiment_id=experiment.experiment_id,
            variant_model=variant.model,
            feature_tag="chat",
        )
        logger.info("request_complete", extra={"extra_data": entry.model_dump()})

    print("\nCost breakdown for acme:")
    print(attributor.aggregate("acme"))
    print(f"Total: ${attributor.total('acme'):.5f}")
```

Expected output — `user-001` maps to the same variant on both calls:
```
user-001 → claude-haiku-5-5
user-002 → claude-sonnet-5-5
user-001 → claude-haiku-5-5   ← same as first, deterministic
```

---

## Exercises

1. **Extend `CostAttributor`**: add a `top_spenders(tenant_id: str, n: int) -> list[tuple[str, float]]` method that returns the top-N `user_id`s by spend within a tenant, sorted descending. Write a pytest test using only `CostRecord` objects — no provider calls.

2. **Weighted split**: modify `ABExperiment` to support a 70/30 split instead of 50/50. Simulate 1000 users (generate IDs `"user-0000"` through `"user-0999"`), count how many land in each variant, and verify the ratio is within ±5% of the declared weights.

3. **PII guard**: add a `sanitize_user_id(user_id: str) -> str` function that hashes the user ID with SHA-256 before it reaches `LogEntry`. Explain in a comment why deterministic A/B assignment still works after hashing.

4. **Inactive experiment fallback**: write a pytest test that creates an `ABExperiment` with `active=False`, calls `assign()` for 10 different user IDs, and asserts every result is `variants[0]`.

(Solutions: ../solutions/week-09/day-5.md)

---

## Self-check

1. Two calls with `user_id="alice"` and the same `experiment_id` must return the same variant. What property of the hash function guarantees this, and what would break if you used `random.random()` instead?

2. `LogEntry` deliberately omits the message content field. Name two concrete risks that would arise if you logged the raw user prompt in a multi-tenant SaaS product.

3. `CostAttributor.aggregate()` iterates over all records on every call. What data structure change would make aggregation O(1) instead of O(n), and what is the trade-off?
