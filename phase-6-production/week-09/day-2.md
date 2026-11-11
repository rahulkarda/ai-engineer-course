# Phase 6 · Week 09 · Day 2 — Model Routing (Cost / Latency / Capability)

> Objective: Implement a model router that selects the right model tier for each request based on cost, latency, and task complexity.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

## Concept

Not every request needs `claude-opus-5-5`. Sending a "summarise this sentence" task to the most capable model wastes money and adds latency. **Model routing** selects the cheapest model that can handle the task reliably.

Three routing dimensions:

| Dimension | Haiku | Sonnet | Opus |
|-----------|-------|--------|------|
| Cost | lowest | medium | highest |
| Latency | fastest | medium | slowest |
| Capability | simple tasks | general | complex / long-context |

Routing policies:

- **CHEAPEST** — always pick the lowest-cost model (`claude-haiku-5-5`).
- **FASTEST** — pick the model with lowest time-to-first-token (also `claude-haiku-5-5` in the Anthropic tier).
- **BEST** — always pick the most capable model (`claude-opus-5-5`).
- **AUTO** — score the request's complexity, then pick the tier that matches.

AUTO is the interesting one: it examines prompt length, presence of reasoning keywords ("step by step", "analyse", "compare"), and instruction density to assign a complexity score, then maps that score to a model tier.

## How it works internally

```
CompletionRequest + RoutingPolicy
        │
        ▼
  ModelRouter.route()
        │
        ├── policy == CHEAPEST  →  haiku
        ├── policy == BEST      →  opus
        ├── policy == FASTEST   →  haiku
        └── policy == AUTO
                │
                ▼
          complexity_score(prompt)
                │
                ├── score < 0.4  →  haiku
                ├── score < 0.75 →  sonnet
                └── score >= 0.75 →  opus
                        │
                        ▼
              (provider_name, model_id)
```

The cost table is a static dict keyed on model id. Cost-per-call estimation uses:

```
estimated_cost = (input_tokens * input_cost_per_1k / 1000)
               + (output_tokens * output_cost_per_1k / 1000)
```

The router does **not** call providers — it only returns a `(provider_name, model_id)` tuple. Decoupling routing from execution makes each piece independently testable.

Complexity scoring is a heuristic, not a classifier. Heuristics are intentional here: a real classifier would require a model call to route a model call (circular dependency). Instead, proxy signals like prompt length and keyword density are cheap to compute.

## Worked example

```python
# gateway/routing.py
import logging
import re
from enum import Enum

from pydantic import BaseModel

from gateway.providers import CompletionRequest

logger = logging.getLogger(__name__)


# ── Cost model ────────────────────────────────────────────────────────────────

class ModelCost(BaseModel):
    """Per-1k-token pricing for a model (illustrative rates)."""
    input_cost_per_1k: float   # USD per 1 000 input tokens
    output_cost_per_1k: float  # USD per 1 000 output tokens

    def estimate(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_cost_per_1k / 1000
            + output_tokens * self.output_cost_per_1k / 1000
        )


# Illustrative pricing — not official Anthropic rates.
COST_TABLE: dict[str, ModelCost] = {
    "claude-haiku-5-5":  ModelCost(input_cost_per_1k=0.00025, output_cost_per_1k=0.00125),
    "claude-sonnet-5-5": ModelCost(input_cost_per_1k=0.003,   output_cost_per_1k=0.015),
    "claude-opus-5-5":   ModelCost(input_cost_per_1k=0.015,   output_cost_per_1k=0.075),
}


# ── Routing policy ────────────────────────────────────────────────────────────

class RoutingPolicy(str, Enum):
    CHEAPEST = "cheapest"   # minimise cost
    FASTEST  = "fastest"    # minimise latency
    BEST     = "best"       # maximise capability
    AUTO     = "auto"       # score complexity, pick tier automatically


# ── Complexity scorer ─────────────────────────────────────────────────────────

# Keywords that signal a reasoning-heavy task.
_COMPLEXITY_KEYWORDS = re.compile(
    r"\b(analyse|analyze|compare|contrast|step by step|reason|explain why|"
    r"critique|evaluate|synthesise|synthesize|argue|debate)\b",
    re.IGNORECASE,
)


def complexity_score(text: str) -> float:
    """Return a float in [0, 1] representing estimated task complexity.

    Higher → more complex → prefer a larger model.

    Heuristics (intentional — see concept section):
      - Prompt length (longer prompts often carry more context / subtlety).
      - Reasoning keyword density.
      - Number of distinct instructions (sentences ending in '?').
    """
    if not text:
        return 0.0

    word_count = len(text.split())
    keyword_hits = len(_COMPLEXITY_KEYWORDS.findall(text))
    question_count = text.count("?")

    # Normalise each signal to [0, 1] with soft caps.
    length_signal  = min(word_count / 500, 1.0)          # cap at 500 words
    keyword_signal = min(keyword_hits / 5, 1.0)          # cap at 5 hits
    question_signal = min(question_count / 3, 1.0)       # cap at 3 questions

    score = 0.5 * length_signal + 0.35 * keyword_signal + 0.15 * question_signal
    logger.debug(
        "complexity_score words=%d keywords=%d questions=%d → %.3f",
        word_count, keyword_hits, question_count, score,
    )
    return round(score, 4)


# ── Model router ──────────────────────────────────────────────────────────────

class ModelRouter:
    """Selects (provider_name, model_id) for a request given a routing policy.

    Does NOT call providers — pure decision logic, easy to unit-test.
    """

    def __init__(self, default_provider: str = "anthropic") -> None:
        self._default_provider = default_provider

    def route(
        self,
        request: CompletionRequest,
        policy: RoutingPolicy = RoutingPolicy.AUTO,
    ) -> tuple[str, str]:
        """Return (provider_name, model_id) based on the policy."""
        model = self._select_model(request, policy)
        logger.info("router.route policy=%s → model=%s", policy.value, model)
        return self._default_provider, model

    def _select_model(
        self,
        request: CompletionRequest,
        policy: RoutingPolicy,
    ) -> str:
        if policy is RoutingPolicy.CHEAPEST:
            return "claude-haiku-5-5"
        if policy is RoutingPolicy.FASTEST:
            return "claude-haiku-5-5"
        if policy is RoutingPolicy.BEST:
            return "claude-opus-5-5"

        # AUTO: score the last user message.
        user_text = self._last_user_message(request)
        score = complexity_score(user_text)

        if score < 0.4:
            return "claude-haiku-5-5"
        if score < 0.75:
            return "claude-sonnet-5-5"
        return "claude-opus-5-5"

    @staticmethod
    def _last_user_message(request: CompletionRequest) -> str:
        for msg in reversed(request.messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""


# ── Cost estimator (convenience) ─────────────────────────────────────────────

def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated cost in USD. Returns 0.0 for unknown models."""
    cost = COST_TABLE.get(model)
    if cost is None:
        logger.warning("estimate_cost: unknown model %s", model)
        return 0.0
    return cost.estimate(input_tokens, output_tokens)
```

```python
# tests/unit/test_routing.py
import pytest

from gateway.routing import (
    COST_TABLE,
    ModelRouter,
    RoutingPolicy,
    complexity_score,
    estimate_cost,
)
from gateway.providers import CompletionRequest


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_request(content: str, model: str = "claude-haiku-5-5") -> CompletionRequest:
    return CompletionRequest(
        model=model,
        messages=[{"role": "user", "content": content}],
    )


# ── complexity_score ──────────────────────────────────────────────────────────

def test_empty_text_scores_zero():
    assert complexity_score("") == 0.0


def test_short_simple_text_scores_low():
    score = complexity_score("Hello, how are you?")
    assert score < 0.4


def test_long_reasoning_text_scores_high():
    text = (
        "Please analyse and compare the economic theories of Keynes and Hayek. "
        "Step by step, evaluate their differences and synthesise a modern view. "
        "Why did each approach gain traction? What are the critiques? "
        "Argue which framework better explains 21st-century recessions. " * 3
    )
    score = complexity_score(text)
    assert score >= 0.5


# ── ModelRouter ───────────────────────────────────────────────────────────────

def test_cheapest_always_returns_haiku():
    router = ModelRouter()
    _, model = router.route(make_request("Hi"), RoutingPolicy.CHEAPEST)
    assert model == "claude-haiku-5-5"


def test_best_always_returns_opus():
    router = ModelRouter()
    _, model = router.route(make_request("Hi"), RoutingPolicy.BEST)
    assert model == "claude-opus-5-5"


def test_auto_simple_prompt_routes_to_haiku():
    router = ModelRouter()
    _, model = router.route(make_request("What is 2 + 2?"), RoutingPolicy.AUTO)
    assert model == "claude-haiku-5-5"


def test_auto_complex_prompt_routes_to_sonnet_or_opus():
    text = (
        "Analyse and compare the economic policies of the EU and USA. "
        "Step by step, evaluate the differences. "
        "Why did each policy gain traction? " * 5
    )
    router = ModelRouter()
    _, model = router.route(make_request(text), RoutingPolicy.AUTO)
    assert model in ("claude-sonnet-5-5", "claude-opus-5-5")


def test_router_returns_configured_provider():
    router = ModelRouter(default_provider="anthropic")
    provider, _ = router.route(make_request("Hi"), RoutingPolicy.CHEAPEST)
    assert provider == "anthropic"


# ── Cost table ────────────────────────────────────────────────────────────────

def test_cost_table_contains_all_models():
    for model in ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"):
        assert model in COST_TABLE


def test_estimate_cost_ordering():
    """Haiku should cost less than sonnet, sonnet less than opus."""
    haiku_cost  = estimate_cost("claude-haiku-5-5",  1000, 500)
    sonnet_cost = estimate_cost("claude-sonnet-5-5", 1000, 500)
    opus_cost   = estimate_cost("claude-opus-5-5",   1000, 500)

    assert haiku_cost < sonnet_cost < opus_cost


def test_estimate_cost_unknown_model_returns_zero():
    assert estimate_cost("gpt-999", 100, 100) == 0.0
```

Usage:

```python
from gateway.routing import ModelRouter, RoutingPolicy
from gateway.providers import CompletionRequest, AnthropicProvider, ProviderRegistry

registry = ProviderRegistry()
registry.register(AnthropicProvider())

router = ModelRouter()
request = CompletionRequest(
    model="",  # router will fill this
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)

provider_name, model_id = router.route(request, RoutingPolicy.AUTO)
request = request.model_copy(update={"model": model_id})
response = await registry.get(provider_name).complete(request)
```

## Exercises

1. Add a `BALANCED` routing policy to `RoutingPolicy` that always picks `claude-sonnet-5-5`. Update `ModelRouter._select_model()` to handle it and add a test asserting the correct model is returned.

2. Extend `complexity_score()` with a fourth signal: count markdown code fences (` ``` `) in the text, treating each as +0.1 to the raw score (uncapped). Add at least two tests — one with zero fences and one with three fences — verifying the effect.

3. Write a `CostAwareRouter` class that wraps `ModelRouter` and adds a `max_cost_usd` budget parameter. In `route()`, after selecting a model, estimate cost using a provided `estimated_input_tokens: int` argument. If the estimate exceeds the budget, downgrade one tier (opus → sonnet → haiku). Write a test that forces a downgrade.

4. Write a `route_batch()` method on `ModelRouter` that accepts `list[CompletionRequest]` and a policy, and returns `list[tuple[str, str]]`. Test it with a mixed list where AUTO would pick different tiers for different requests.

(Solutions: ../solutions/week-09/day-2.md)

## Self-check

- [ ] I can explain the four routing policies and when to use each
- [ ] I can implement a `complexity_score()` function using only cheap, local heuristics (no model calls)
- [ ] I can write tests that verify routing decisions without touching any provider
