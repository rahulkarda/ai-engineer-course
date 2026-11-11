# Solutions — Day 2: Model Routing

---

## Cost Table (shared reference)

```python
# gateway/routing/costs.py
from pydantic import BaseModel


class ModelCost(BaseModel):
    model_id: str
    provider: str
    input_cost_per_1k: float
    output_cost_per_1k: float


COST_TABLE: dict[str, ModelCost] = {
    "claude-haiku-5-5": ModelCost(
        model_id="claude-haiku-5-5",
        provider="anthropic",
        input_cost_per_1k=0.00025,
        output_cost_per_1k=0.00125,
    ),
    "claude-sonnet-5-5": ModelCost(
        model_id="claude-sonnet-5-5",
        provider="anthropic",
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
    ),
    "claude-opus-5-5": ModelCost(
        model_id="claude-opus-5-5",
        provider="anthropic",
        input_cost_per_1k=0.015,
        output_cost_per_1k=0.075,
    ),
}
```

---

## Exercise 1: LATENCY routing policy — always pick `claude-haiku-5-5`

The LATENCY policy ignores prompt content entirely and returns the fastest (cheapest) model.

```python
# gateway/routing/router.py
import logging
from enum import Enum

from gateway.providers.base import CompletionRequest
from gateway.routing.costs import COST_TABLE

logger = logging.getLogger(__name__)


class RoutingPolicy(str, Enum):
    AUTO = "auto"
    COST = "cost"
    LATENCY = "latency"
    QUALITY = "quality"


# Fastest model for LATENCY policy — lowest p50 latency in the Anthropic family
_LATENCY_MODEL = "claude-haiku-5-5"
# Highest-capability model for QUALITY policy
_QUALITY_MODEL = "claude-opus-5-5"


def complexity_score(prompt: str) -> float:
    """
    Return a 0–1 score estimating how complex / demanding the prompt is.
    Higher score → heavier model warranted.
    """
    score = 0.0
    word_count = len(prompt.split())

    # Length signal
    if word_count > 500:
        score += 0.4
    elif word_count > 100:
        score += 0.2

    # Reasoning / analysis keywords
    reasoning_keywords = [
        "analyze", "explain", "compare", "evaluate",
        "synthesize", "critique", "design", "architecture",
    ]
    for kw in reasoning_keywords:
        if kw.lower() in prompt.lower():
            score += 0.1

    # Code-related keywords
    code_keywords = ["def ", "class ", "import ", "```"]
    for kw in code_keywords:
        if kw in prompt:
            score += 0.1

    return min(score, 1.0)


def route(request: CompletionRequest, policy: RoutingPolicy) -> str:
    """Return the model ID to use for this request under the given policy."""
    if policy == RoutingPolicy.LATENCY:
        logger.info("routing policy=LATENCY → %s", _LATENCY_MODEL)
        return _LATENCY_MODEL

    if policy == RoutingPolicy.QUALITY:
        logger.info("routing policy=QUALITY → %s", _QUALITY_MODEL)
        return _QUALITY_MODEL

    if policy == RoutingPolicy.COST:
        # Always pick cheapest available model
        cheapest = min(COST_TABLE.values(), key=lambda m: m.input_cost_per_1k)
        logger.info("routing policy=COST → %s", cheapest.model_id)
        return cheapest.model_id

    # AUTO: use complexity heuristic
    prompt = " ".join(
        m.get("content", "") for m in request.messages if isinstance(m, dict)
    )
    score = complexity_score(prompt)
    if score >= 0.7:
        model = "claude-opus-5-5"
    elif score >= 0.35:
        model = "claude-sonnet-5-5"
    else:
        model = "claude-haiku-5-5"

    logger.info("routing policy=AUTO score=%.2f → %s", score, model)
    return model
```

---

## Exercise 2: Test `route()` with AUTO policy on a long, complex prompt

The test constructs a prompt that is long (>500 words) and contains reasoning + code keywords, so `complexity_score` should exceed 0.7 and the router should return `claude-opus-5-5`.

```python
# tests/unit/test_router.py
import pytest

from gateway.providers.base import CompletionRequest
from gateway.routing.router import RoutingPolicy, complexity_score, route


def _make_request(content: str) -> CompletionRequest:
    return CompletionRequest(
        model="claude-haiku-5-5",  # initial model; router overrides this
        messages=[{"role": "user", "content": content}],
    )


def test_auto_policy_long_complex_prompt_picks_opus() -> None:
    """AUTO policy: long prompt with reasoning + code keywords → opus."""
    # Build a prompt that is clearly long and complex
    base = (
        "Please analyze and evaluate the following architecture. "
        "Compare the design patterns and explain the trade-offs. "
        "def process(): pass\n```python\nimport os\nclass Foo: pass\n```\n"
    )
    # Pad to well over 500 words so the length signal fires
    padding = "word " * 520
    prompt = base + padding

    request = _make_request(prompt)
    model = route(request, RoutingPolicy.AUTO)

    assert model == "claude-opus-5-5", f"Expected opus, got {model}"


def test_auto_policy_short_simple_prompt_picks_haiku() -> None:
    request = _make_request("hi there")
    model = route(request, RoutingPolicy.AUTO)
    assert model == "claude-haiku-5-5"


def test_latency_policy_always_haiku() -> None:
    for content in ["short", "a " * 1000, "analyze compare design"]:
        request = _make_request(content)
        assert route(request, RoutingPolicy.LATENCY) == "claude-haiku-5-5"


def test_quality_policy_always_opus() -> None:
    request = _make_request("short")
    assert route(request, RoutingPolicy.QUALITY) == "claude-opus-5-5"
```

---

## Exercise 3: Extend `complexity_score()` with code-keyword detection

Already integrated above. Isolated implementation for clarity:

```python
# Additions to complexity_score() — drop-in replacement
CODE_KEYWORDS: list[str] = ["def ", "class ", "import ", "```"]

def complexity_score(prompt: str) -> float:
    score = 0.0
    word_count = len(prompt.split())

    # --- length signal ---
    if word_count > 500:
        score += 0.4
    elif word_count > 100:
        score += 0.2

    # --- reasoning / analysis keywords ---
    reasoning_keywords = [
        "analyze", "explain", "compare", "evaluate",
        "synthesize", "critique", "design", "architecture",
    ]
    for kw in reasoning_keywords:
        if kw.lower() in prompt.lower():
            score += 0.1

    # --- code-related keywords (NEW) ---
    for kw in CODE_KEYWORDS:
        if kw in prompt:
            score += 0.1

    return min(score, 1.0)
```

Unit test:

```python
def test_complexity_score_code_keywords_boost() -> None:
    plain = "Tell me about the weather today please."
    with_code = plain + "\ndef helper():\n    pass\nimport os\n```python\n```"

    plain_score = complexity_score(plain)
    code_score = complexity_score(with_code)

    # Code keywords should increase the score
    assert code_score > plain_score
    # Each of the 4 code keywords contributes 0.1 — at least 3 are present
    assert code_score >= plain_score + 0.3
```

---

## Exercise 4: `route_with_budget()` — refuse if cost exceeds budget

Estimates cost using `COST_TABLE` and a rough token estimate (word_count ≈ tokens for budget purposes). Raises `ValueError` if even the cheapest model exceeds the per-request budget.

```python
# gateway/routing/router.py  (add to existing module)
from gateway.routing.costs import COST_TABLE


class BudgetExceededError(Exception):
    """Raised when no model fits within the per-request cost budget."""


def _estimate_cost(model_id: str, prompt: str, max_output_tokens: int = 512) -> float:
    """Rough USD cost estimate: words ≈ tokens for budgeting purposes."""
    cost = COST_TABLE.get(model_id)
    if cost is None:
        return float("inf")
    input_tokens = len(prompt.split())
    input_cost = (input_tokens / 1000) * cost.input_cost_per_1k
    output_cost = (max_output_tokens / 1000) * cost.output_cost_per_1k
    return input_cost + output_cost


def route_with_budget(
    request: CompletionRequest,
    policy: RoutingPolicy,
    max_cost_per_request_usd: float,
) -> str:
    """
    Route like `route()` but raise BudgetExceededError if the chosen model
    (or any model) exceeds max_cost_per_request_usd.

    Falls back to cheaper models when the policy choice is too expensive.
    Raises BudgetExceededError if even haiku exceeds the budget.
    """
    prompt = " ".join(
        m.get("content", "") for m in request.messages if isinstance(m, dict)
    )

    # Preferred model from normal routing
    preferred = route(request, policy)

    # Check preferred model first
    if _estimate_cost(preferred, prompt) <= max_cost_per_request_usd:
        logger.info(
            "route_with_budget: preferred=%s within budget=%.6f",
            preferred,
            max_cost_per_request_usd,
        )
        return preferred

    # Try cheaper models in ascending cost order
    models_by_cost = sorted(
        COST_TABLE.keys(),
        key=lambda m: COST_TABLE[m].input_cost_per_1k,
    )
    for model_id in models_by_cost:
        if _estimate_cost(model_id, prompt) <= max_cost_per_request_usd:
            logger.warning(
                "route_with_budget: preferred=%s too expensive, downgraded to %s",
                preferred,
                model_id,
            )
            return model_id

    raise BudgetExceededError(
        f"All models exceed budget of ${max_cost_per_request_usd:.6f} USD"
    )
```

Tests:

```python
# tests/unit/test_router_budget.py
import pytest

from gateway.providers.base import CompletionRequest
from gateway.routing.router import BudgetExceededError, RoutingPolicy, route_with_budget


def _req(content: str) -> CompletionRequest:
    return CompletionRequest(
        model="claude-haiku-5-5",
        messages=[{"role": "user", "content": content}],
    )


def test_route_with_budget_accepts_haiku_on_tiny_budget() -> None:
    # Short prompt → very low cost; even the tightest realistic budget passes haiku
    request = _req("hi")
    model = route_with_budget(request, RoutingPolicy.AUTO, max_cost_per_request_usd=1.0)
    assert model == "claude-haiku-5-5"


def test_route_with_budget_downgrades_opus_to_haiku() -> None:
    # Use QUALITY policy (wants opus) but give a budget that haiku fits under
    # With a tiny prompt and $0.001 budget, opus (~$0.015/1k input) won't fit
    # haiku (~$0.00025/1k input) will
    request = _req("a short question")
    model = route_with_budget(request, RoutingPolicy.QUALITY, max_cost_per_request_usd=0.001)
    assert model == "claude-haiku-5-5"


def test_route_with_budget_raises_when_impossible() -> None:
    # $0.0000001 is below what any model can achieve even for a single token
    request = _req("hi")
    with pytest.raises(BudgetExceededError):
        route_with_budget(request, RoutingPolicy.AUTO, max_cost_per_request_usd=0.0000001)
```
