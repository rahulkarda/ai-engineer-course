# Solutions — Week 05 · Day 7: When Reflection Helps vs. Wastes Tokens

## Exercise 1: Budget override

```python
from pydantic import BaseModel


class TaskProfile(BaseModel):
    description: str
    has_verifiable_answer: bool
    failure_cost: str
    latency_budget_ms: int
    current_reliability: ReliabilityTier
    requires_compliance: bool = False
    budget_usd_per_1k_calls: float = 999.0  # default: no limit


BASE_COST_USD_PER_1K = 0.50  # rough baseline for single call


def recommend_pattern(profile: TaskProfile) -> PatternRecommendation:
    # ... (existing logic) ...
    rec = _base_recommend(profile)  # existing logic extracted

    # Budget check
    if rec.primary != "none":
        estimated_cost = BASE_COST_USD_PER_1K * rec.estimated_cost_multiplier
        if estimated_cost > profile.budget_usd_per_1k_calls:
            return PatternRecommendation(
                primary="none",
                secondary=None,
                rationale="Pattern exceeds cost budget",
                estimated_cost_multiplier=1.0,
                skip_reason=(
                    f"Cost {estimated_cost:.2f}/1k > budget "
                    f"{profile.budget_usd_per_1k_calls:.2f}/1k"
                ),
            )
    return rec
```

Test:

```python
def test_tight_budget_overrides_pattern():
    profile = TaskProfile(
        description="Legal doc review",
        has_verifiable_answer=False,
        failure_cost="high",
        latency_budget_ms=60_000,
        current_reliability=ReliabilityTier.LOW,
        budget_usd_per_1k_calls=0.10,  # very tight
    )
    rec = recommend_pattern(profile)
    assert rec.primary == "none"
    assert rec.skip_reason is not None
    assert "budget" in rec.skip_reason.lower()
```

---

## Exercise 2: Temperature collapse warning

```python
class TaskProfile(BaseModel):
    # ... existing fields ...
    low_temperature_collapse_risk: bool = False


def recommend_pattern(profile: TaskProfile) -> PatternRecommendation:
    rec = _base_recommend(profile)
    if rec.primary == "self_consistency" and profile.low_temperature_collapse_risk:
        rec = rec.model_copy(
            update={
                "rationale": rec.rationale + " WARNING: low-temperature collapse risk — ensure temperature >= 0.7."
            }
        )
    return rec
```

---

## Exercise 3: `batch_recommend`

```python
def batch_recommend(profiles: list[TaskProfile]) -> dict[str, PatternRecommendation]:
    return {p.description: recommend_pattern(p) for p in profiles}
```

Test:

```python
def test_no_tight_latency_gets_pattern():
    import random
    profiles = [
        TaskProfile(
            description=f"Task {i}",
            has_verifiable_answer=bool(i % 2),
            failure_cost="low",
            latency_budget_ms=random.choice([500, 1000, 2000]),  # all tight
            current_reliability=ReliabilityTier.MEDIUM,
        )
        for i in range(10)
    ]
    recs = batch_recommend(profiles)
    for profile in profiles:
        assert recs[profile.description].primary == "none", (
            f"Profile with latency {profile.latency_budget_ms}ms got pattern "
            f"{recs[profile.description].primary}"
        )
```

---

## Exercise 4: `adaptive_recommend`

```python
def adaptive_recommend(
    profile: TaskProfile,
    observed_pass_rate: float,
) -> PatternRecommendation:
    """Adjust recommendation based on empirical pass rate."""
    if observed_pass_rate > 0.95:
        return PatternRecommendation(
            primary="none",
            secondary=None,
            rationale=f"Observed pass rate {observed_pass_rate:.0%} > 0.95; no reflection needed",
            estimated_cost_multiplier=1.0,
            skip_reason="High empirical reliability",
        )

    rec = recommend_pattern(profile)

    if observed_pass_rate < 0.5 and rec.primary != "none" and rec.secondary is None:
        # Add a secondary safety net when pass rate is very low
        secondary = "supervisor" if rec.primary != "supervisor" else "critic_agent"
        rec = rec.model_copy(update={
            "secondary": secondary,
            "rationale": rec.rationale + f" (secondary added: pass rate {observed_pass_rate:.0%} < 0.5)",
        })

    return rec
```

Tests:

```python
def test_high_pass_rate_returns_none():
    profile = TaskProfile(
        description="Code completion",
        has_verifiable_answer=True,
        failure_cost="low",
        latency_budget_ms=30_000,
        current_reliability=ReliabilityTier.HIGH,
    )
    rec = adaptive_recommend(profile, observed_pass_rate=0.97)
    assert rec.primary == "none"


def test_low_pass_rate_adds_secondary():
    profile = TaskProfile(
        description="Complex reasoning",
        has_verifiable_answer=False,
        failure_cost="high",
        latency_budget_ms=60_000,
        current_reliability=ReliabilityTier.LOW,
    )
    rec = adaptive_recommend(profile, observed_pass_rate=0.40)
    assert rec.primary != "none"
    assert rec.secondary is not None
```
