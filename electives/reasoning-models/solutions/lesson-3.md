# Solutions · Lesson 3 — When to Use Reasoning vs Fast Models

## Exercise 1 — Calibrate the classifier

```python
"""tests/test_lesson_3_ex1.py"""
import pytest
from electives.reasoning_models.lesson_3_router import classify_complexity, Complexity


@pytest.mark.parametrize("problem,expected", [
    # Easy cases
    ("What is the capital of Germany?", Complexity.EASY),
    ("Translate this to French: Hello world.", Complexity.EASY),
    ("Define machine learning.", Complexity.EASY),
    ("Who wrote Romeo and Juliet?", Complexity.EASY),
    ("Summarize the following: The sky is blue.", Complexity.EASY),
    # Hard cases
    (
        "Prove that there are infinitely many prime numbers using Euclid's algorithm. "
        "Verify each step formally.",
        Complexity.HARD,
    ),
    (
        "Audit this Python code for SQL injection vulnerabilities and buffer overflows. "
        "Verify each finding with a proof-of-concept.",
        Complexity.HARD,
    ),
    (
        "Optimize the scheduling of 5 tasks with precedence constraints A→B, B→C, A→D, D→E "
        "to minimize total completion time. Prove optimality.",
        Complexity.HARD,
    ),
    (
        "Derive the backpropagation algorithm from first principles for a 3-layer neural network. "
        "Verify your gradient expressions are correct.",
        Complexity.HARD,
    ),
    (
        "Given 12 coins where one is counterfeit (lighter or heavier), "
        "find the counterfeit coin in exactly 3 weighings. Prove your strategy works.",
        Complexity.HARD,
    ),
])
def test_classify_complexity(problem: str, expected: Complexity):
    result = classify_complexity(problem)
    assert result == expected, f"Problem: {problem!r}\nExpected {expected}, got {result}"
```

---

## Exercise 2 — Cost estimator integration

```python
"""solutions/lesson_3_ex2.py"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import re

# Default cost table (USD per 1M tokens)
DEFAULT_COSTS: dict[str, dict[str, float]] = {
    "claude-haiku-5-5": {"input": 0.08, "output": 0.40},
    "claude-sonnet-5-5": {"input": 3.0, "output": 15.0},
    "claude-opus-5-5": {"input": 15.0, "output": 75.0},
}

# Rough token estimates
TYPICAL_INPUT_TOKENS = 200
TYPICAL_OUTPUT_TOKENS = 500
TYPICAL_THINKING_TOKENS_PER_BUDGET = 0.8  # model typically uses 80% of budget


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
    estimated_cost_usd: float


def _estimate_cost(
    model: str,
    thinking_budget: int | None,
    costs: dict[str, dict[str, float]],
) -> float:
    rates = costs.get(model, {"input": 1.0, "output": 5.0})
    input_cost = (TYPICAL_INPUT_TOKENS / 1e6) * rates["input"]
    output_cost = (TYPICAL_OUTPUT_TOKENS / 1e6) * rates["output"]
    thinking_cost = 0.0
    if thinking_budget:
        effective_tokens = thinking_budget * TYPICAL_THINKING_TOKENS_PER_BUDGET
        thinking_cost = (effective_tokens / 1e6) * rates.get("output", 5.0)
    return round(input_cost + output_cost + thinking_cost, 8)
```

```python
"""tests/test_lesson_3_ex2.py"""
from solutions.lesson_3_ex2 import _estimate_cost, DEFAULT_COSTS


def test_haiku_no_thinking():
    cost = _estimate_cost("claude-haiku-5-5", None, DEFAULT_COSTS)
    assert cost > 0
    assert cost < 0.001  # haiku is cheap


def test_sonnet_with_thinking_costs_more():
    no_think = _estimate_cost("claude-sonnet-5-5", None, DEFAULT_COSTS)
    with_think = _estimate_cost("claude-sonnet-5-5", 8000, DEFAULT_COSTS)
    assert with_think > no_think


def test_custom_rate_override():
    custom_costs = {"my-model": {"input": 100.0, "output": 200.0}}
    cost = _estimate_cost("my-model", None, custom_costs)
    assert cost > 0
```

---

## Exercise 3 — A/B accuracy comparison

```python
"""solutions/lesson_3_ex3.py"""
from __future__ import annotations
from typing import Callable


def compare_routes(
    problems: list[str],
    answers: list[str],
    fast_solver: Callable[[str], str],
    reasoning_solver: Callable[[str], str],
) -> dict[str, int]:
    """
    Run each problem through both solvers and count correct answers.

    Returns:
        {"fast_correct": int, "reasoning_correct": int}
    """
    fast_correct = 0
    reasoning_correct = 0

    for problem, correct_answer in zip(problems, answers):
        fast_answer = fast_solver(problem)
        reasoning_answer = reasoning_solver(problem)

        if fast_answer.strip().lower() == correct_answer.strip().lower():
            fast_correct += 1
        if reasoning_answer.strip().lower() == correct_answer.strip().lower():
            reasoning_correct += 1

    return {"fast_correct": fast_correct, "reasoning_correct": reasoning_correct}
```

```python
"""tests/test_lesson_3_ex3.py"""
from solutions.lesson_3_ex3 import compare_routes


def test_compare_routes_structure():
    problems = ["Q1", "Q2", "Q3"]
    answers = ["A", "B", "C"]
    fast_solver = lambda p: {"Q1": "A", "Q2": "X", "Q3": "C"}[p]
    reasoning_solver = lambda p: {"Q1": "A", "Q2": "B", "Q3": "C"}[p]

    result = compare_routes(problems, answers, fast_solver, reasoning_solver)

    assert isinstance(result, dict)
    assert set(result.keys()) == {"fast_correct", "reasoning_correct"}
    assert result["fast_correct"] == 2   # Q1 and Q3
    assert result["reasoning_correct"] == 3  # all correct
```

---

## Exercise 4 — Dynamic threshold

```python
"""solutions/lesson_3_ex4.py"""
from __future__ import annotations
import re
from enum import Enum

HARD_KEYWORDS: frozenset[str] = frozenset({
    "prove", "proof", "algorithm", "optimize", "verify",
    "security", "vulnerability", "bug", "audit", "constraint",
    "schedule", "plan", "deduce", "calculate", "derive",
})


class Complexity(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


def classify_complexity(problem: str, hard_threshold: int = 5) -> Complexity:
    text = problem.lower()
    words = set(re.findall(r"\b\w+\b", text))
    hard_hits = words & HARD_KEYWORDS
    number_count = len(re.findall(r"\b\d+\b", text))
    word_count = len(text.split())
    hard_score = len(hard_hits) * 3 + (number_count > 3) * 2 + (word_count > 100) * 1

    if hard_score >= hard_threshold:
        return Complexity.HARD
    if hard_score >= 2 or word_count > 40:
        return Complexity.MEDIUM
    return Complexity.EASY
```

```python
"""tests/test_lesson_3_ex4.py"""
from solutions.lesson_3_ex4 import classify_complexity, Complexity

BORDERLINE = (
    "Please verify whether this calculation is correct: 15 × 8 = 120."
)


def test_low_threshold_routes_hard():
    result = classify_complexity(BORDERLINE, hard_threshold=2)
    assert result == Complexity.HARD


def test_high_threshold_routes_medium():
    result = classify_complexity(BORDERLINE, hard_threshold=10)
    assert result == Complexity.MEDIUM


def test_default_threshold():
    easy = "What is the capital of France?"
    assert classify_complexity(easy) == Complexity.EASY
```

---

## Exercise 5 — Latency simulation (async)

```python
"""tests/test_lesson_3_ex5.py"""
import asyncio
import time
from unittest.mock import AsyncMock, patch
import pytest


async def fast_call(problem: str) -> str:
    await asyncio.sleep(0.1)
    return "fast_answer"


async def thinking_call(problem: str) -> str:
    await asyncio.sleep(0.5)
    return "thinking_answer"


async def call_model_async(problem: str, use_thinking: bool) -> str:
    if use_thinking:
        return await thinking_call(problem)
    return await fast_call(problem)


@pytest.mark.asyncio
async def test_five_easy_problems_under_one_second():
    problems = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    start = time.monotonic()
    results = await asyncio.gather(*[call_model_async(p, use_thinking=False) for p in problems])
    elapsed = time.monotonic() - start
    assert len(results) == 5
    assert elapsed < 1.0, f"Took {elapsed:.2f}s — expected < 1.0s"
```
