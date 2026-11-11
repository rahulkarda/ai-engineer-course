# Phase 5 · Week 05 · Day 7 — When Reflection Helps vs. Wastes Tokens

> Objective: Develop a decision framework for when to apply reflection, self-consistency, and supervision — and when these patterns add cost without value.  ·  Time: ~3 hr project + 1 hr assessment  ·  Prereqs: [[day-6]]

## Concept

The patterns this week add latency and cost. Applied blindly, they degrade a system. Applied strategically, they unlock reliability that a single LLM call cannot achieve.

Today you build the mental model for when each pattern earns its cost.

### Decision framework

#### 1. Does the task have a verifiable answer?

```
Verifiable → self-consistency, test-based Reflexion
Not verifiable → critic agent, rubric-based reflection
```

Math problems, code tasks, factual Q&A: verifiable. Use self-consistency (sample N, take majority) or Reflexion with an automated evaluator.

Creative writing, strategy docs, design advice: not objectively verifiable. Use a critic agent with a rubric.

#### 2. What is the cost of a bad output?

```
High (medical, legal, financial, safety) → supervisor + guardrails mandatory
Medium (internal tools, dev aids)         → critic loop, 2–3 retries
Low (brainstorming, first drafts)         → single pass, no reflection
```

Reflection does not guarantee correctness — it shifts probability. If the cost of failure is catastrophic, guardrails + human escalation are the right layer, not more reflection.

#### 3. What is the latency budget?

| Pattern | Typical overhead vs. single call |
|---|---|
| Self-reflection (1 iter) | 3× |
| Reflexion (3 attempts) | 4–6× |
| Self-consistency (5 samples) | 5× parallel, 5× serial |
| Supervisor + critic | 2–4× |

If response time is user-facing and under 2 seconds, none of these patterns fit. Use them in batch, async, or offline pipelines.

#### 4. Is the model already reliable on this task?

Run a quick eval (50 examples). If accuracy > 90%, adding reflection gains <1 point at 3× cost. The cross-over point is usually around 70%: below that, reflection meaningfully helps.

### Specific guidance per pattern

**Self-reflection**: Use when the task has clear quality dimensions (format, completeness) and the model writes better under structured critique than free generation. Do NOT use for pure factual questions — the model can confidently "fix" a correct answer to be wrong.

**Self-consistency**: Use for chain-of-thought tasks where sampling diversity helps (math reasoning, multi-step deduction). Do NOT use when one authoritative answer exists in training data — sampling just adds noise.

**Critic agent**: Use when you need an auditable quality gate with specific rubric requirements. Do NOT use when the rubric maps 1:1 to automated checks (format, length, JSON validity) — just run the checks directly without an LLM.

**Reflexion**: Use for complex multi-attempt tasks (code generation, planning) where each failure contains learnable information. Do NOT use when failures are random (flaky tests, network errors) — lessons won't generalise.

**Supervisor**: Use when compliance, audit, and policy enforcement are requirements. Do NOT use for internal dev tooling where latency matters more than policy.

## How it works internally

The token cost model:

```
single call:         P tokens
reflection (3 iter): 3P + 2×critique_tokens + 2×revision_tokens
                   ≈ 4–5P
self-consistency:    N × P  (but parallel → wall-clock ≈ P + scheduling)
supervisor:          P + guardrail_tokens + critic_tokens
                   ≈ 2–3P
```

Most teams underestimate the **latency** impact of serial patterns (reflection, Reflexion) and overestimate the cost of **parallel** patterns (self-consistency, concurrent critic + worker).

When designing systems: parallelise what you can, serialise only what must be sequential.

## Worked example: pattern selection decision tree

```python
"""
day7_pattern_selector.py — Decision tree for choosing reflection patterns.

This is a planning tool, not a runtime wrapper.
Run:  python day7_pattern_selector.py
"""
import logging
from enum import Enum

from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


class ReliabilityTier(str, Enum):
    HIGH = "high"     # >90% on task type
    MEDIUM = "medium" # 70-90%
    LOW = "low"       # <70%


class TaskProfile(BaseModel):
    description: str
    has_verifiable_answer: bool
    failure_cost: str  # "low" | "medium" | "high"
    latency_budget_ms: int
    current_reliability: ReliabilityTier
    requires_compliance: bool = False


class PatternRecommendation(BaseModel):
    primary: str
    secondary: str | None
    rationale: str
    estimated_cost_multiplier: float
    skip_reason: str | None = None  # set if no pattern recommended


def recommend_pattern(profile: TaskProfile) -> PatternRecommendation:
    # No pattern if latency is tight
    if profile.latency_budget_ms < 3000:
        return PatternRecommendation(
            primary="none",
            secondary=None,
            rationale="Latency budget too tight for reflection overhead",
            estimated_cost_multiplier=1.0,
            skip_reason="Latency < 3s: single-pass only",
        )

    # Model already reliable
    if profile.current_reliability == ReliabilityTier.HIGH:
        return PatternRecommendation(
            primary="none",
            secondary=None,
            rationale="Model reliability >90%; reflection gain < cost",
            estimated_cost_multiplier=1.0,
            skip_reason="High reliability: not cost-effective",
        )

    # Compliance required → supervisor mandatory
    if profile.requires_compliance:
        primary = "supervisor"
        secondary = "critic_agent"
    elif profile.has_verifiable_answer:
        if profile.failure_cost == "high":
            primary = "self_consistency"
            secondary = "supervisor"
        else:
            primary = "reflexion" if profile.current_reliability == ReliabilityTier.LOW else "self_consistency"
            secondary = None
    else:
        primary = "critic_agent" if profile.failure_cost in ("medium", "high") else "self_reflection"
        secondary = "supervisor" if profile.failure_cost == "high" else None

    cost = {
        "supervisor": 3.0,
        "self_consistency": 5.0,
        "reflexion": 5.0,
        "critic_agent": 2.5,
        "self_reflection": 3.5,
    }

    return PatternRecommendation(
        primary=primary,
        secondary=secondary,
        rationale=f"Failure cost={profile.failure_cost}, verifiable={profile.has_verifiable_answer}, "
                  f"reliability={profile.current_reliability.value}",
        estimated_cost_multiplier=cost.get(primary, 1.0),
    )


if __name__ == "__main__":
    profiles = [
        TaskProfile(
            description="Math tutoring step-by-step solutions",
            has_verifiable_answer=True,
            failure_cost="medium",
            latency_budget_ms=30_000,
            current_reliability=ReliabilityTier.MEDIUM,
        ),
        TaskProfile(
            description="Medical triage recommendation",
            has_verifiable_answer=False,
            failure_cost="high",
            latency_budget_ms=60_000,
            current_reliability=ReliabilityTier.MEDIUM,
            requires_compliance=True,
        ),
        TaskProfile(
            description="Autocomplete for IDE",
            has_verifiable_answer=True,
            failure_cost="low",
            latency_budget_ms=500,
            current_reliability=ReliabilityTier.HIGH,
        ),
        TaskProfile(
            description="Legal contract summarisation",
            has_verifiable_answer=False,
            failure_cost="high",
            latency_budget_ms=120_000,
            current_reliability=ReliabilityTier.LOW,
            requires_compliance=True,
        ),
    ]

    for p in profiles:
        rec = recommend_pattern(p)
        print(f"\nTask: {p.description}")
        print(f"  Primary pattern : {rec.primary}")
        print(f"  Secondary        : {rec.secondary or 'none'}")
        print(f"  Cost multiplier  : {rec.estimated_cost_multiplier}×")
        if rec.skip_reason:
            print(f"  Skip reason      : {rec.skip_reason}")
        else:
            print(f"  Rationale        : {rec.rationale}")
```

## Exercises

1. Add a `budget_usd_per_1k_calls: float` field to `TaskProfile`. If the estimated cost (`base_cost × multiplier × calls_per_day / 1000`) exceeds the budget, override the recommendation to `"none"` and populate `skip_reason`. Write a test with a tight budget that forces the override.

2. The current recommendation ignores the **task diversity** dimension: self-consistency only helps when the model explores genuinely different paths. Add a `low_temperature_collapse_risk: bool` field — if True and the recommendation is `self_consistency`, add a warning note to `rationale`.

3. Write a `batch_recommend(profiles: list[TaskProfile]) -> dict[str, PatternRecommendation]` function and a test that processes 10 profiles, verifying no profile with `latency_budget_ms < 3000` gets a non-none pattern.

4. The decision tree is static. Implement an `adaptive_recommend` function that adjusts recommendation based on empirical data: `observed_pass_rate: float` (0–1) from recent runs. Below 0.5 pass rate, always add a secondary pattern. Above 0.95, always return none.

(Solutions: ../solutions/week-05/day-7.md)

## Self-check

- [ ] I can recite the four decision dimensions (verifiable, cost, latency, reliability)
- [ ] I know which pattern to apply for code generation tasks vs. open-ended writing tasks
- [ ] I can calculate the approximate token cost multiplier for each pattern
- [ ] I understand why reflection can hurt on high-reliability tasks
- [ ] I can justify a "no reflection" recommendation for a given task profile

---

## Supplementary links

- "Large Language Models as Optimizers" (reflect for prompt optimisation) — https://arxiv.org/abs/2309.03409
- "Is Self-Repair Good?" (when reflection helps/hurts) — https://arxiv.org/abs/2401.01123
- OpenAI evals blog (evaluating LLMs systematically) — https://github.com/openai/evals
