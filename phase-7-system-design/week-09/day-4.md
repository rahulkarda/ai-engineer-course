# Phase 7 · Week 09 · Day 4 — LLM/Agent-Specific Issues: Latency Spikes, Cost Blowups, Hallucination, Tool Failures & Prompt Regressions

> Objective: Diagnose and fix the four failure modes unique to LLM-powered agents: runaway latency, cost explosions, hallucination, and prompt regressions.  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

## Concept

LLM agents fail differently from traditional services — they fail *silently and statistically*. A database either returns a row or it doesn't; an LLM returns text that may be wrong, expensive, slow, or semantically broken in ways no error code captures.

### Four agent-specific failure modes

| Failure | Symptom | Root cause |
|---------|---------|------------|
| Latency spike | p99 > 30 s, high variability | Long context (input tokens dominate TTFT), unbounded tool loops, prompt that triggers chain-of-thought | 
| Cost blowup | Daily spend 10×+ expected | Runaway retry loops, oversized context, wrong model for task, output caching disabled |
| Hallucination | Factually wrong answers, invented citations | Model confidence ≠ accuracy; retrieval miss; prompt that encourages generation over retrieval |
| Prompt regression | After prompt edit, quality drops on previously-passing cases | Missing eval suite; prompt change tested only on "it feels better" |

### Tool failures

Agents invoke external tools (web search, DB, APIs). Tool failures manifest as:
- **Hard failure**: tool raises exception → agent may hallucinate a response or enter retry loop
- **Soft failure**: tool returns empty / malformed result → agent silently proceeds with wrong context
- **Loop failure**: agent re-invokes same tool in a loop because output never satisfies a poorly-specified stopping condition

## How it works internally

### Token budget and TTFT

Time-to-first-token (TTFT) is dominated by the *prefill* phase, which is O(n) in input tokens. A 10K-token context takes ~10× longer to prefill than a 1K context. This is why context window growth causes latency spikes more than output length does.

Cost formula (per request):
```
cost = (input_tokens * input_price_per_1k) + (output_tokens * output_price_per_1k)
```

If an agent is looping over tools, `output_tokens` accumulates per loop iteration. A loop of 10 iterations with 500 output tokens each = 5000 output tokens.

### Hallucination mechanics

A language model maximizes `P(next_token | context)`. It has no internal knowledge-vs-confabulation switch. The model generates fluently regardless of factual grounding. Mitigations:
- Ground with retrieval (provide facts in context)
- Require citations (`"Answer only with quotes from the provided documents"`)
- Use structured output (reduces surface area for hallucination)
- Verify answers with a separate judge call

### Eval-driven prompt development

A prompt regression is undetectable without a *frozen test set* — a collection of (input, expected_output) pairs evaluated before and after every prompt change. The score is the proportion of cases where the output meets acceptance criteria.

## Worked example

```python
"""
day4_llm_agent_issues.py — diagnose LLM/agent-specific failure modes

All LLM calls are MOCKED — no real API key required.
Uses os.getenv("ANTHROPIC_API_KEY") as placeholder.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from unittest.mock import MagicMock, patch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
log = logging.getLogger("llm_debug")

# API key via env — never hardcoded
_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


# ─────────────────────────────────────────────────────────────────────────────
# Mock LLM client (replaces real Anthropic SDK)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MockUsage:
    input_tokens: int
    output_tokens: int

    @property
    def total_cost_usd(self) -> float:
        # claude-claude-sonnet-5-5 pricing approximation
        return self.input_tokens * 3e-6 + self.output_tokens * 15e-6


@dataclass
class MockResponse:
    content: str
    usage: MockUsage
    latency_ms: float


def mock_llm_call(
    prompt: str,
    *,
    inject_latency_spike: bool = False,
    inject_hallucination: bool = False,
    loop_iteration: int = 0,
) -> MockResponse:
    """Simulate LLM call with configurable failure injection."""
    input_tokens = len(prompt.split())  # rough approximation
    output_tokens = random.randint(50, 200)

    base_latency_ms = 300.0
    if inject_latency_spike:
        base_latency_ms = 8000.0  # long context TTFT spike
        input_tokens = 15_000  # simulating bloated context

    # Simulate loop accumulation
    input_tokens += loop_iteration * 500

    time.sleep(base_latency_ms / 1000)

    if inject_hallucination:
        content = "The GDP of France in 2023 was $47.2 trillion."  # wrong
    else:
        content = "The GDP of France in 2023 was approximately $3.0 trillion USD."

    return MockResponse(
        content=content,
        usage=MockUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        latency_ms=base_latency_ms,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. LATENCY SPIKE diagnosis
# ─────────────────────────────────────────────────────────────────────────────

def measure_request_latency(n: int = 5, inject_spike: bool = False) -> None:
    latencies: list[float] = []
    for i in range(n):
        start = time.monotonic()
        resp = mock_llm_call(
            f"Summarize this document: {'word ' * (500 if not inject_spike else 5000)}",
            inject_latency_spike=inject_spike,
        )
        latencies.append((time.monotonic() - start) * 1000)

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p99 = latencies[-1]
    log.info(
        "Latency %s spike: p50=%.0f ms, p99=%.0f ms",
        "WITH" if inject_spike else "without",
        p50, p99,
    )
    if p99 > 5000:
        log.warning("  → p99 > 5 s: check input token count (context bloat)")


# ─────────────────────────────────────────────────────────────────────────────
# 2. COST BLOWUP from runaway tool loop
# ─────────────────────────────────────────────────────────────────────────────

def run_agent_loop_buggy(max_steps: int = 50) -> float:
    """BUG: no step budget — agent loops until some arbitrary condition."""
    total_cost = 0.0
    step = 0
    # Simulate a loop where the stopping condition is never met
    while step < max_steps:
        resp = mock_llm_call(
            f"Continue searching. Previous context: {'data ' * 100}",
            loop_iteration=step,
        )
        total_cost += resp.usage.total_cost_usd
        step += 1
        if step >= 8:  # hit our artificial loop limit
            break
    return total_cost


def run_agent_loop_fixed(max_steps: int = 5) -> float:
    """FIX: hard step budget + cost circuit breaker."""
    total_cost = 0.0
    cost_limit_usd = 0.10  # hard cap

    for step in range(max_steps):
        resp = mock_llm_call(
            "Answer concisely.",
            loop_iteration=step,
        )
        total_cost += resp.usage.total_cost_usd

        if total_cost > cost_limit_usd:
            log.warning("Cost circuit breaker: $%.4f > $%.2f — stopping", total_cost, cost_limit_usd)
            break

        log.info("Step %d: cost so far $%.4f", step + 1, total_cost)

    return total_cost


def demo_cost_blowup() -> None:
    log.info("=== Cost blowup demo ===")
    buggy_cost = run_agent_loop_buggy()
    log.info("Buggy loop total cost: $%.4f", buggy_cost)

    fixed_cost = run_agent_loop_fixed()
    log.info("Fixed loop total cost: $%.4f", fixed_cost)


# ─────────────────────────────────────────────────────────────────────────────
# 3. HALLUCINATION detection
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalCase:
    question: str
    expected_keyword: str  # must appear in answer


def run_eval_suite(
    cases: list[EvalCase],
    inject_hallucination: bool = False,
) -> float:
    """Returns pass rate (0–1)."""
    passed = 0
    for case in cases:
        resp = mock_llm_call(case.question, inject_hallucination=inject_hallucination)
        ok = case.expected_keyword.lower() in resp.content.lower()
        log.debug("Q: %s | expected: %s | pass: %s | answer: %s",
                  case.question[:40], case.expected_keyword, ok, resp.content[:60])
        if ok:
            passed += 1
    return passed / len(cases)


GDP_CASES = [
    EvalCase("What was France's GDP in 2023?", "3.0 trillion"),
    EvalCase("What was France's GDP in 2023?", "trillion"),
]


def demo_hallucination() -> None:
    log.info("=== Hallucination eval ===")
    clean_rate = run_eval_suite(GDP_CASES, inject_hallucination=False)
    hallucinated_rate = run_eval_suite(GDP_CASES, inject_hallucination=True)
    log.info("Pass rate — clean: %.0f%%, hallucinated: %.0f%%",
             clean_rate * 100, hallucinated_rate * 100)
    if hallucinated_rate < 1.0:
        log.warning("Hallucination detected — add retrieval grounding or citation requirement")


# ─────────────────────────────────────────────────────────────────────────────
# 4. PROMPT REGRESSION detection
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PromptVersion:
    name: str
    system_prompt: str


PROMPT_V1 = PromptVersion(
    name="v1",
    system_prompt="You are a helpful assistant. Answer factually.",
)
PROMPT_V2_REGRESSION = PromptVersion(
    name="v2-regression",
    system_prompt="You are a creative storyteller. Imagine vividly.",  # introduces hallucination risk
)


def score_prompt(version: PromptVersion, cases: list[EvalCase]) -> float:
    """Score is fraction of cases where expected_keyword appears."""
    passed = 0
    for case in cases:
        # v2 creative prompt → higher hallucination rate
        inject = version.name == "v2-regression"
        resp = mock_llm_call(case.question, inject_hallucination=inject)
        if case.expected_keyword.lower() in resp.content.lower():
            passed += 1
    return passed / len(cases)


def demo_prompt_regression() -> None:
    log.info("=== Prompt regression detection ===")
    v1_score = score_prompt(PROMPT_V1, GDP_CASES)
    v2_score = score_prompt(PROMPT_V2_REGRESSION, GDP_CASES)
    log.info("v1 score: %.0f%%  |  v2 score: %.0f%%", v1_score * 100, v2_score * 100)
    if v2_score < v1_score:
        log.warning("REGRESSION: v2 scores %.0f%% lower — do not ship", (v1_score - v2_score) * 100)


if __name__ == "__main__":
    measure_request_latency(n=3, inject_spike=False)
    measure_request_latency(n=3, inject_spike=True)
    demo_cost_blowup()
    demo_hallucination()
    demo_prompt_regression()
```

## Exercises

1. **Token budget guard**: Implement a `TokenBudgetGuard` class that wraps a mock LLM call and raises `TokenBudgetExceededError` if `input_tokens + output_tokens > budget`. Add tests for: (a) call within budget succeeds, (b) call over budget raises, (c) budget is cumulative across multiple calls.

2. **Tool failure handling**: Write an `AgentWithToolFallback` that calls a `search_tool` first; if it raises `ToolError` or returns an empty list, it falls back to a `cache_tool`. Test with three scenarios: search succeeds, search fails (use fallback), both fail (raise `AllToolsFailedError`).

3. **Regression CI gate**: Implement `compare_prompt_versions(v1, v2, cases, threshold=0.05) -> bool` that returns `True` if v2 is not more than `threshold` (5%) worse than v1. Write four tests: v2 equal → True, v2 1% worse → True, v2 6% worse → False, v2 better → True.

(Solutions: ../solutions/week-09/day-4.md)

## Self-check

- [ ] I can explain why TTFT scales with input tokens
- [ ] I can implement a cost circuit breaker for an agent loop
- [ ] I can write an eval suite that catches hallucination regressions
- [ ] I understand how a creative system prompt can introduce factual regressions
- [ ] I can detect prompt regressions with a score-comparison gate in CI

## Supplementary links

- [Anthropic API — Token counting](https://docs.anthropic.com/en/docs/build-with-claude/token-counting)
- [RAGAS — RAG evaluation framework](https://docs.ragas.io/)
- [PromptFoo — prompt regression testing](https://www.promptfoo.dev/)
