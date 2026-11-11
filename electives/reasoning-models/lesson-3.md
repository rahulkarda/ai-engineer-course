# Reasoning Models · Lesson 3 — When to Use Reasoning vs Fast Models

> Objective: Build a cost/latency decision framework and implement a routing heuristic.
> Time: ~1 hr · Prereqs: [lesson-2.md](lesson-2.md)

---

## Concept

### The tradeoff in numbers

Extended thinking is not free. A rough comparison (illustrative — check current pricing):

| Model | Input | Output | Latency (typical) |
|-------|-------|--------|-------------------|
| `claude-haiku-5-5` | low | low | < 1 s |
| `claude-sonnet-5-5` | medium | medium | 2–5 s |
| `claude-opus-5-5` | high | high | 5–15 s |
| `claude-sonnet-5-5` + thinking 8k | medium + thinking | medium | 15–40 s |

For a chatbot answering "What's the capital of France?", paying for 8000 thinking tokens
is waste. For a code review that must catch subtle security bugs, skipping thinking may
produce wrong answers at low cost — false economy.

The decision is not "fast vs smart". It is **expected error cost vs. inference cost**:

```
Use reasoning when:
  P(error | fast model) × cost_of_error  >  extra_cost_of_reasoning
```

If a wrong answer means a financial transaction goes through incorrectly, the error cost
is high. If a wrong answer is a slightly off recipe suggestion, the error cost is low.

### Problem taxonomy

Problems where reasoning models help most:

| Category | Examples |
|----------|---------|
| Multi-step logic | Constraint satisfaction, scheduling, deduction puzzles |
| Mathematical reasoning | Proofs, algebra, combinatorics |
| Code correctness | Security audits, bug finding, algorithm verification |
| Complex planning | Multi-step agent plans with dependencies |
| Legal/medical reasoning | Rule application across many interacting conditions |

Problems where fast models are sufficient:

| Category | Examples |
|----------|---------|
| Classification | Sentiment, intent, category labelling |
| Extraction | Named entity recognition, structured data pull |
| Translation | Document translation with no ambiguity |
| Simple Q&A | Factual lookups, definitions |
| Reformatting | Markdown → HTML, JSON reshaping |

### Routing heuristics

A practical router inspects the problem before deciding:

```
complexity score = f(
    problem_length,
    presence of quantitative / logical keywords,
    number of explicit constraints,
    user-flagged difficulty,
    historical accuracy on similar problems
)
```

Route to reasoning when `complexity_score > threshold`. Tune `threshold` against a
labelled eval set.

---

## How it works internally

### Latency budget

Extended thinking adds latency proportional to the thinking budget because the model
generates those tokens sequentially before starting the answer. A 10000-token thinking
budget at 40 tokens/s adds ~250 s before the first answer token appears.

For user-facing applications this is often unacceptable. Mitigations:

1. **Stream** — Surface partial answer tokens as they arrive; hide the thinking phase.
2. **Async prefetch** — Start the reasoning call before the user finishes submitting (if
   you can predict the query).
3. **Cache thinking results** — If the same hard problem recurs, cache the answer+trace
   keyed by (model, problem hash, budget).

### Routing as a preprocessing step

A common architecture:

```
User query
    │
    ▼
[Router — lightweight classifier, no LLM call]
    │                │
    ▼                ▼
[Fast path]   [Reasoning path]
haiku-5-5    sonnet-5-5 + thinking
    │                │
    └────────────────┘
              │
          [Response]
```

The router itself must be fast (rule-based or a tiny classifier). Never use a reasoning
model to decide whether to use a reasoning model — that defeats the purpose.

---

## Worked example

```python
"""lesson_3_router.py — Cost-aware model router."""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from enum import Enum

import anthropic

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class Complexity(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class RouteDecision:
    complexity: Complexity
    model: str
    thinking_budget: int | None
    reason: str


# Keywords that strongly indicate hard reasoning
HARD_KEYWORDS: frozenset[str] = frozenset({
    "prove", "proof", "algorithm", "optimize", "verify",
    "security", "vulnerability", "bug", "audit", "constraint",
    "schedule", "plan", "deduce", "calculate", "derive",
})

EASY_KEYWORDS: frozenset[str] = frozenset({
    "translate", "summarize", "summarise", "list", "define",
    "what is", "who is", "when did", "format", "convert",
})


def classify_complexity(problem: str) -> Complexity:
    """Heuristic classifier — no LLM call."""
    text = problem.lower()
    words = set(re.findall(r"\b\w+\b", text))

    hard_hits = words & HARD_KEYWORDS
    easy_hits = words & EASY_KEYWORDS

    # Count explicit constraints / numbers
    number_count = len(re.findall(r"\b\d+\b", text))
    word_count = len(text.split())

    hard_score = len(hard_hits) * 3 + (number_count > 3) * 2 + (word_count > 100) * 1
    easy_score = len(easy_hits) * 2

    log.debug("hard_score=%d easy_score=%d", hard_score, easy_score)

    if hard_score >= 5:
        return Complexity.HARD
    if easy_score >= 4 or (hard_score == 0 and word_count < 30):
        return Complexity.EASY
    return Complexity.MEDIUM


def route(problem: str) -> RouteDecision:
    """Return the routing decision for a problem."""
    complexity = classify_complexity(problem)

    match complexity:
        case Complexity.EASY:
            return RouteDecision(
                complexity=complexity,
                model="claude-haiku-5-5",
                thinking_budget=None,
                reason="simple query, fast model sufficient",
            )
        case Complexity.MEDIUM:
            return RouteDecision(
                complexity=complexity,
                model="claude-sonnet-5-5",
                thinking_budget=None,
                reason="moderate complexity, Sonnet without thinking",
            )
        case Complexity.HARD:
            return RouteDecision(
                complexity=complexity,
                model="claude-sonnet-5-5",
                thinking_budget=8000,
                reason="high complexity, extended thinking enabled",
            )


def call_model(problem: str, decision: RouteDecision) -> str:
    """Execute the API call per the routing decision."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    kwargs: dict = {
        "model": decision.model,
        "max_tokens": (decision.thinking_budget or 0) + 2048,
        "messages": [{"role": "user", "content": problem}],
    }

    if decision.thinking_budget:
        kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": decision.thinking_budget,
        }

    response = client.messages.create(**kwargs)

    for block in response.content:
        if block.type == "text":
            return block.text

    return ""


def solve(problem: str) -> str:
    """Route and solve a problem."""
    decision = route(problem)
    log.info(
        "route: model=%s thinking=%s reason=%s",
        decision.model,
        decision.thinking_budget,
        decision.reason,
    )
    return call_model(problem, decision)


if __name__ == "__main__":
    easy = "What is the capital of Germany?"
    hard = (
        "Verify whether this scheduling algorithm avoids deadlock: "
        "three processes each holding one resource and requesting the next. "
        "Prove your answer using the resource allocation graph."
    )

    for q in [easy, hard]:
        answer = solve(q)
        log.info("answer: %s", answer[:100])
```

---

## Exercises

1. **Calibrate the classifier** — Write 10 test cases (5 easy, 5 hard) as pytest
   parametrize entries and assert `classify_complexity` returns the correct label.
   No API calls — this is pure logic.

2. **Cost estimator integration** — Extend `RouteDecision` with a `estimated_cost_usd`
   field. Populate it in `route()` using a cost table you define as a module-level dict.
   Rates should be overridable via a parameter, not hardcoded constants.

3. **A/B accuracy comparison** — Write a `compare_routes(problems, answers)` function
   that runs each problem through both the fast and reasoning paths (mocked) and reports
   which path matches the known correct answer. Assert the function returns a
   `dict[str, int]` with keys `"fast_correct"` and `"reasoning_correct"`.

4. **Dynamic threshold** — Modify `route()` to accept a `hard_threshold: int = 5`
   parameter and write three tests showing different threshold values change the routing
   decision for the same borderline problem.

5. **Latency simulation** — Write a pytest test that patches `call_model` to sleep for
   0.1 s (fast) or 0.5 s (thinking) and asserts that the total wall time for 5 easy
   problems is < 1 s when batched via `asyncio.gather`. (Hint: make `call_model` async.)

(Solutions: [solutions/lesson-3.md](solutions/lesson-3.md))

---

## Self-check

- [ ] I can articulate the cost/accuracy tradeoff as an inequality
- [ ] I can build a rule-based complexity classifier with no LLM call
- [ ] I know which problem categories benefit most from extended thinking
- [ ] I can design a routing architecture that adds no LLM overhead on the routing step
- [ ] I understand how thinking budget affects end-to-end latency

## Supplementary links

- [Model overview & pricing](https://docs.anthropic.com/en/docs/about-claude/models)
- [FrugalGPT: routing for cost/quality](https://arxiv.org/abs/2305.05176)
- [LLM routing survey](https://arxiv.org/abs/2406.18665)
