# Solutions — Day 5: Cost Attribution + A/B Testing + Logging

## Exercise 1: `CostAttributor.top_spenders(n)`

Add a method that returns the top `n` tenants sorted by total spend descending.

```python
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class CostRecord:
    tenant_id: str
    model: str
    cost_usd: float
    input_tokens: int
    output_tokens: int

class CostAttributor:
    def __init__(self):
        self._records: list[CostRecord] = []
        self._tenant_totals: dict[str, float] = defaultdict(float)

    def record(self, cost: CostRecord) -> None:
        self._records.append(cost)
        self._tenant_totals[cost.tenant_id] += cost.cost_usd

    def get_tenant_cost(self, tenant_id: str) -> float:
        return self._tenant_totals.get(tenant_id, 0.0)

    def top_spenders(self, n: int) -> list[tuple[str, float]]:
        sorted_tenants = sorted(
            self._tenant_totals.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_tenants[:n]
```

**Key points:**
- `sorted(..., reverse=True)` sorts descending by cost.
- `[:n]` slice handles `n > len(tenants)` gracefully — returns all.
- `_tenant_totals` is maintained incrementally — O(1) per record, O(k log k) for `top_spenders`.

---

## Exercise 2: `ModelABRouter` with variant distribution tracking

Extend with `_counts` dict and `get_variant_distribution()`.

```python
import hashlib
from dataclasses import dataclass, field

@dataclass
class ABVariant:
    model: str
    provider: str
    weight: float  # 0.0–1.0, weights must sum to 1.0

class ModelABRouter:
    def __init__(self, experiment_id: str, variants: list[ABVariant]):
        self.experiment_id = experiment_id
        self.variants = variants
        self._counts: dict[str, int] = {v.model: 0 for v in variants}

    def assign(self, user_id: str) -> ABVariant:
        key = f"{self.experiment_id}:{user_id}".encode()
        digest = int(hashlib.sha256(key).hexdigest(), 16)
        bucket = (digest % 1000) / 1000.0  # 0.0 to 0.999

        cumulative = 0.0
        for variant in self.variants:
            cumulative += variant.weight
            if bucket < cumulative:
                self._counts[variant.model] += 1
                return variant

        # safety fallback — handles floating point imprecision at boundary
        self._counts[self.variants[-1].model] += 1
        return self.variants[-1]

    def get_variant_distribution(self) -> dict[str, float]:
        total = sum(self._counts.values())
        if total == 0:
            return {m: 0.0 for m in self._counts}
        return {m: count / total for m, count in self._counts.items()}
```

**Key points:**
- SHA-256 hash of `experiment_id:user_id` → deterministic, uniform bucket.
- `% 1000 / 1000.0` gives 1000 buckets — enough granularity for most weight configs.
- Safety fallback after loop handles the edge case where cumulative weights sum to slightly less than 1.0 due to floating-point errors.
- `get_variant_distribution()` returns fractions (0.0–1.0), not percentages — multiply by 100 for display.

---

## Exercise 3: `LogSanitizer` — strip PII keys before logging

```python
class LogSanitizer:
    REDACTED_KEYS = frozenset({"content", "text", "message", "messages"})

    def sanitize(self, data: dict) -> dict:
        return {
            k: "[REDACTED]" if k in self.REDACTED_KEYS else v
            for k, v in data.items()
        }
```

**Usage pattern:**
```python
import logging

logger = logging.getLogger(__name__)
sanitizer = LogSanitizer()

def log_request(request_data: dict) -> None:
    safe = sanitizer.sanitize(request_data)
    logger.info("llm request: %s", safe)
```

**Key points:**
- `frozenset` for O(1) lookup.
- Replaces value with `"[REDACTED]"` — preserves key presence so log structure stays parseable.
- Shallow sanitization — nested dicts are not recursed. For deep sanitization, implement recursively:

```python
def sanitize_deep(self, data: dict) -> dict:
    result = {}
    for k, v in data.items():
        if k in self.REDACTED_KEYS:
            result[k] = "[REDACTED]"
        elif isinstance(v, dict):
            result[k] = self.sanitize_deep(v)
        elif isinstance(v, list):
            result[k] = [
                self.sanitize_deep(item) if isinstance(item, dict) else item
                for item in v
            ]
        else:
            result[k] = v
    return result
```

---

## Exercise 4: pytest tests for `ModelABRouter` determinism

```python
import pytest
from gateway.ab_testing import ModelABRouter, ABVariant


def test_ab_router_deterministic():
    """Same user_id must always receive the same variant."""
    variants = [
        ABVariant(model="claude-haiku-5-5", provider="anthropic", weight=0.5),
        ABVariant(model="claude-sonnet-5-5", provider="anthropic", weight=0.5),
    ]
    router = ModelABRouter(experiment_id="exp-001", variants=variants)

    user_id = "user-abc-123"
    first_assignment = router.assign(user_id)

    for _ in range(99):
        assignment = router.assign(user_id)
        assert assignment.model == first_assignment.model, (
            "Assignment must be deterministic for the same user_id"
        )


def test_ab_router_different_users_can_get_different_variants():
    """With enough users and 50/50 weights, both variants should appear."""
    variants = [
        ABVariant(model="claude-haiku-5-5", provider="anthropic", weight=0.5),
        ABVariant(model="claude-sonnet-5-5", provider="anthropic", weight=0.5),
    ]
    router = ModelABRouter(experiment_id="exp-001", variants=variants)

    assignments = {router.assign(f"user-{i}").model for i in range(100)}
    assert len(assignments) == 2, (
        "With 100 users and equal weights, both variants should appear"
    )


def test_ab_router_distribution_sums_to_one():
    """get_variant_distribution() fractions must sum to 1.0."""
    variants = [
        ABVariant(model="claude-haiku-5-5", provider="anthropic", weight=0.3),
        ABVariant(model="claude-sonnet-5-5", provider="anthropic", weight=0.7),
    ]
    router = ModelABRouter(experiment_id="exp-dist", variants=variants)

    for i in range(200):
        router.assign(f"user-{i}")

    dist = router.get_variant_distribution()
    assert abs(sum(dist.values()) - 1.0) < 1e-9, "Distribution fractions must sum to 1.0"


def test_ab_router_empty_distribution():
    """Before any assigns, distribution should be all zeros."""
    variants = [
        ABVariant(model="claude-haiku-5-5", provider="anthropic", weight=0.5),
        ABVariant(model="claude-sonnet-5-5", provider="anthropic", weight=0.5),
    ]
    router = ModelABRouter(experiment_id="exp-empty", variants=variants)
    dist = router.get_variant_distribution()
    assert all(v == 0.0 for v in dist.values())


def test_ab_router_weighted_distribution_approximate():
    """30/70 split should land within ±10% with 1000 users."""
    variants = [
        ABVariant(model="claude-haiku-5-5", provider="anthropic", weight=0.3),
        ABVariant(model="claude-sonnet-5-5", provider="anthropic", weight=0.7),
    ]
    router = ModelABRouter(experiment_id="exp-weighted", variants=variants)

    for i in range(1000):
        router.assign(f"user-{i}")

    dist = router.get_variant_distribution()
    assert 0.2 <= dist["claude-haiku-5-5"] <= 0.4, "haiku share should be ~30%"
    assert 0.6 <= dist["claude-sonnet-5-5"] <= 0.8, "sonnet share should be ~70%"
```

**Why test determinism across 100 calls rather than 2?**
The hash function is stateless — a second call with the same input always produces the same output. Testing 100 times rules out any accidental mutable state (e.g., a counter-based implementation that only _looks_ deterministic on first call).
