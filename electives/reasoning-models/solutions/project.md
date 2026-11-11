# Solutions · Capstone Project — Adaptive Problem Router

Full reference implementation. All LLM calls mocked in tests.

---

## pyproject.toml

```toml
[project]
name = "reasoning-router"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "anthropic>=0.40.0",
    "pydantic-settings>=2.0.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23"]

[tool.pytest.ini_options]
asyncio_mode = "auto"

[tool.ruff]
line-length = 100
target-version = "py312"
```

## .env.example

```
ANTHROPIC_API_KEY=your-key-here
FAST_MODEL=claude-haiku-5-5
REASONING_MODEL=claude-sonnet-5-5
THINKING_BUDGET=8000
BEST_OF_N=3
HARD_THRESHOLD=5
```

---

## reasoning_router/config.py

```python
"""reasoning_router/config.py"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str
    fast_model: str = "claude-haiku-5-5"
    reasoning_model: str = "claude-sonnet-5-5"
    thinking_budget: int = 8000
    best_of_n: int = 3
    hard_threshold: int = 5

    model_config = {"env_file": ".env"}
```

---

## reasoning_router/classifier.py

```python
"""reasoning_router/classifier.py"""
from __future__ import annotations

import re
from enum import Enum

HARD_KEYWORDS: frozenset[str] = frozenset({
    "prove", "proof", "algorithm", "optimize", "optimise", "verify",
    "security", "vulnerability", "audit", "constraint",
    "schedule", "plan", "deduce", "calculate", "derive",
    "complexity", "deadlock", "formal", "invariant",
})

EASY_KEYWORDS: frozenset[str] = frozenset({
    "translate", "summarize", "summarise", "list", "define",
    "format", "convert", "what is", "who is", "when did",
})


class Complexity(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


def classify(problem: str, hard_threshold: int = 5) -> Complexity:
    """Rule-based classifier. No LLM call."""
    text = problem.lower()
    words = set(re.findall(r"\b\w+\b", text))

    hard_hits = words & HARD_KEYWORDS
    easy_hits = words & EASY_KEYWORDS
    number_count = len(re.findall(r"\b\d+\b", text))
    word_count = len(text.split())

    hard_score = len(hard_hits) * 3 + (number_count > 4) * 2 + (word_count > 80) * 1
    easy_score = len(easy_hits) * 2

    if hard_score >= hard_threshold:
        return Complexity.HARD
    if easy_score >= 4 or (hard_score == 0 and word_count < 25):
        return Complexity.EASY
    return Complexity.MEDIUM
```

---

## reasoning_router/router.py

```python
"""reasoning_router/router.py"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import anthropic

from .classifier import Complexity, classify
from .config import Settings
from .sampler import sample_n
from .verifier import numeric_match, select_best

log = logging.getLogger(__name__)

# Rough cost table (USD per 1M tokens) — update as pricing changes
_COST_TABLE: dict[str, dict[str, float]] = {
    "claude-haiku-5-5":  {"input": 0.08,  "output": 0.40},
    "claude-sonnet-5-5": {"input": 3.00,  "output": 15.0},
    "claude-opus-5-5":   {"input": 15.00, "output": 75.0},
}

_TYPICAL_INPUT = 300    # tokens
_TYPICAL_OUTPUT = 600   # tokens


@dataclass
class RouteDecision:
    complexity: Complexity
    model: str
    thinking_budget: int | None
    reason: str
    estimated_cost_usd: float


@dataclass
class SolveResult:
    answer: str
    complexity: Complexity
    model: str
    thinking_budget: int | None
    input_tokens: int
    output_tokens: int
    cost_usd: float


def _estimate_cost(
    model: str,
    thinking_budget: int | None,
    cost_table: dict[str, dict[str, float]] | None = None,
) -> float:
    table = cost_table or _COST_TABLE
    rates = table.get(model, {"input": 1.0, "output": 5.0})
    cost = (
        (_TYPICAL_INPUT / 1e6) * rates["input"] +
        (_TYPICAL_OUTPUT / 1e6) * rates["output"]
    )
    if thinking_budget:
        cost += (thinking_budget * 0.8 / 1e6) * rates["output"]
    return round(cost, 8)


def route(problem: str, settings: Settings) -> RouteDecision:
    complexity = classify(problem, hard_threshold=settings.hard_threshold)

    match complexity:
        case Complexity.EASY:
            model = settings.fast_model
            budget = None
            reason = "simple query — fast model, no thinking"
        case Complexity.MEDIUM:
            model = settings.reasoning_model
            budget = None
            reason = "moderate complexity — Sonnet, no thinking"
        case Complexity.HARD:
            model = settings.reasoning_model
            budget = settings.thinking_budget
            reason = "high complexity — Sonnet + extended thinking + best-of-N"

    return RouteDecision(
        complexity=complexity,
        model=model,
        thinking_budget=budget,
        reason=reason,
        estimated_cost_usd=_estimate_cost(model, budget),
    )


def _single_call(
    client: anthropic.Anthropic,
    problem: str,
    decision: RouteDecision,
) -> tuple[str, int, int]:
    """Issue one API call; return (answer, input_tokens, output_tokens)."""
    kwargs: dict = {
        "model": decision.model,
        "max_tokens": (decision.thinking_budget or 0) + 2048,
        "messages": [{"role": "user", "content": problem}],
    }
    if decision.thinking_budget:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": decision.thinking_budget}

    response = client.messages.create(**kwargs)
    answer = ""
    for block in response.content:
        if block.type == "text":
            answer = block.text
            break

    return answer, response.usage.input_tokens, response.usage.output_tokens


async def solve(problem: str, settings: Settings) -> SolveResult:
    """Route and solve a problem."""
    decision = route(problem, settings)
    log.info(
        "route: complexity=%s model=%s thinking=%s",
        decision.complexity, decision.model, decision.thinking_budget,
    )

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    if decision.complexity == Complexity.HARD and settings.best_of_n > 1:
        candidates = await sample_n(problem, settings.best_of_n, settings)
        answer = select_best(candidates, lambda a: 1.0 if a.strip() else 0.0)
        in_tok, out_tok = 0, 0  # summarised across all samples
    else:
        answer, in_tok, out_tok = _single_call(client, problem, decision)

    cost = _estimate_cost(decision.model, decision.thinking_budget)
    return SolveResult(
        answer=answer,
        complexity=decision.complexity,
        model=decision.model,
        thinking_budget=decision.thinking_budget,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=cost,
    )
```

---

## reasoning_router/sampler.py

```python
"""reasoning_router/sampler.py"""
from __future__ import annotations

import asyncio
import logging
import os

import anthropic

from .config import Settings

log = logging.getLogger(__name__)


async def _call_once(
    client: anthropic.AsyncAnthropic,
    problem: str,
    settings: Settings,
) -> str:
    response = await client.messages.create(
        model=settings.reasoning_model,
        max_tokens=settings.thinking_budget + 2048,
        thinking={"type": "enabled", "budget_tokens": settings.thinking_budget},
        messages=[{"role": "user", "content": problem}],
    )
    for block in response.content:
        if block.type == "text":
            return block.text
    return ""


async def sample_n(problem: str, n: int, settings: Settings) -> list[str]:
    """Issue n concurrent calls; return list of answer strings."""
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    tasks = [_call_once(client, problem, settings) for _ in range(n)]
    results = await asyncio.gather(*tasks)
    log.info("sampled %d responses for problem (first 60 chars): %s", n, problem[:60])
    return list(results)
```

---

## reasoning_router/verifier.py

```python
"""reasoning_router/verifier.py"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Callable

import anthropic

log = logging.getLogger(__name__)


@dataclass
class JudgeResult:
    score: int
    reason: str


def numeric_match(answer: str, ground_truth: str, tol: float = 0.01) -> bool:
    """Float comparison with tolerance; falls back to string match."""
    nums_a = re.findall(r"-?\d+(?:\.\d+)?", answer)
    nums_gt = re.findall(r"-?\d+(?:\.\d+)?", ground_truth)
    if nums_a and nums_gt:
        return abs(float(nums_a[-1]) - float(nums_gt[-1])) <= tol
    return answer.strip().lower() == ground_truth.strip().lower()


_JUDGE_PROMPT = """\
You are a reasoning quality judge.
Given a problem, a ground-truth answer, and a candidate answer, return:
{{"score": <int 0-10>, "reason": "<one sentence>"}}

Problem: {problem}
Ground truth: {ground_truth}
Candidate: {candidate}

Return only valid JSON."""


def llm_judge(
    client: anthropic.Anthropic,
    problem: str,
    ground_truth: str,
    candidate: str,
) -> JudgeResult:
    response = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=128,
        messages=[{"role": "user", "content": _JUDGE_PROMPT.format(
            problem=problem, ground_truth=ground_truth, candidate=candidate,
        )}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    data = json.loads(text)
    return JudgeResult(score=int(data["score"]), reason=data["reason"])


def select_best(
    candidates: list[str],
    verifier_fn: Callable[[str], float],
) -> str:
    """Return candidate with highest verifier score; fallback to first on tie."""
    if not candidates:
        return ""
    scored = [(verifier_fn(c), c) for c in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    log.info("best_of_n: best_score=%.2f from %d candidates", scored[0][0], len(candidates))
    return scored[0][1]
```

---

## reasoning_router/eval.py

```python
"""reasoning_router/eval.py"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, asdict

from .config import Settings
from .router import solve, SolveResult
from .verifier import numeric_match

log = logging.getLogger(__name__)


@dataclass
class EvalProblem:
    id: str
    problem: str
    answer: str


@dataclass
class EvalResult:
    id: str
    model_answer: str
    correct: bool
    cost_usd: float
    complexity: str


@dataclass
class EvalReport:
    accuracy: float
    total_cost_usd: float
    avg_latency_s: float
    n: int
    per_problem: list[dict]


@dataclass
class StrategyComparison:
    routing: dict
    always_thinking: dict
    never_thinking: dict


async def run_eval(problems: list[EvalProblem], settings: Settings) -> EvalReport:
    """Solve all problems and report accuracy + cost."""
    results: list[EvalResult] = []

    for prob in problems:
        result: SolveResult = await solve(prob.problem, settings)
        correct = numeric_match(result.answer, prob.answer)
        results.append(EvalResult(
            id=prob.id,
            model_answer=result.answer,
            correct=correct,
            cost_usd=result.cost_usd,
            complexity=result.complexity.value,
        ))
        log.info("id=%s correct=%s cost=%.6f", prob.id, correct, result.cost_usd)

    n = len(results)
    accuracy = sum(r.correct for r in results) / n if n else 0.0
    total_cost = sum(r.cost_usd for r in results)

    return EvalReport(
        accuracy=round(accuracy, 4),
        total_cost_usd=round(total_cost, 6),
        avg_latency_s=0.0,   # would populate with real timing
        n=n,
        per_problem=[asdict(r) for r in results],
    )


async def _run_strategy(
    problems: list[EvalProblem],
    settings: Settings,
    force_thinking: bool | None,
) -> dict:
    """Run all problems with a given strategy; return accuracy + cost summary."""
    overrides = Settings(
        anthropic_api_key=settings.anthropic_api_key,
        fast_model=settings.fast_model,
        reasoning_model=settings.reasoning_model,
        thinking_budget=settings.thinking_budget if force_thinking is not False else 0,
        best_of_n=1,
        hard_threshold=0 if force_thinking else 999,  # force routing
    )
    report = await run_eval(problems, overrides)
    return {"accuracy": report.accuracy, "total_cost_usd": report.total_cost_usd}


async def compare_strategies(
    problems: list[EvalProblem],
    settings: Settings,
) -> StrategyComparison:
    """Compare routing, always-thinking, never-thinking strategies."""
    routing_result, always_result, never_result = await asyncio.gather(
        _run_strategy(problems, settings, force_thinking=None),
        _run_strategy(problems, settings, force_thinking=True),
        _run_strategy(problems, settings, force_thinking=False),
    )
    comparison = StrategyComparison(
        routing=routing_result,
        always_thinking=always_result,
        never_thinking=never_result,
    )
    log.info("strategy comparison: %s", comparison)
    return comparison
```

---

## reasoning_router/main.py

```python
"""reasoning_router/main.py — CLI entry point."""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

from .config import Settings
from .eval import EvalProblem, compare_strategies, run_eval
from .router import solve

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)


def _load_problems(path: str) -> list[EvalProblem]:
    problems = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            data = json.loads(line)
            problems.append(EvalProblem(**data))
    return problems


async def _cmd_solve(problem: str, settings: Settings) -> None:
    result = await solve(problem, settings)
    print(f"Answer: {result.answer}")
    print(f"Model: {result.model}  Complexity: {result.complexity.value}  Cost: ${result.cost_usd:.6f}")


async def _cmd_eval(path: str, settings: Settings) -> None:
    problems = _load_problems(path)
    report = await run_eval(problems, settings)
    print(json.dumps({
        "accuracy": report.accuracy,
        "total_cost_usd": report.total_cost_usd,
        "n": report.n,
    }, indent=2))


async def _cmd_compare(path: str, settings: Settings) -> None:
    problems = _load_problems(path)
    comparison = await compare_strategies(problems, settings)
    print(json.dumps({
        "routing": comparison.routing,
        "always_thinking": comparison.always_thinking,
        "never_thinking": comparison.never_thinking,
    }, indent=2))


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m reasoning_router.main <solve|eval|compare> [args]")
        sys.exit(1)

    settings = Settings()
    command = sys.argv[1]

    match command:
        case "solve":
            problem = " ".join(sys.argv[2:])
            asyncio.run(_cmd_solve(problem, settings))
        case "eval":
            asyncio.run(_cmd_eval(sys.argv[2], settings))
        case "compare":
            asyncio.run(_cmd_compare(sys.argv[2], settings))
        case _:
            print(f"Unknown command: {command}")
            sys.exit(1)


if __name__ == "__main__":
    main()
```

---

## tests/test_classifier.py

```python
"""tests/test_classifier.py"""
import pytest
from reasoning_router.classifier import classify, Complexity


@pytest.mark.parametrize("problem,expected", [
    ("What is the capital of France?", Complexity.EASY),
    ("Define recursion in one sentence.", Complexity.EASY),
    ("Translate 'hello' to Spanish.", Complexity.EASY),
    ("List the planets in the solar system.", Complexity.EASY),
    ("Who wrote Hamlet?", Complexity.EASY),
    (
        "Prove using mathematical induction that sum of first n odd numbers equals n^2.",
        Complexity.HARD,
    ),
    (
        "Audit this API endpoint for SQL injection vulnerabilities. Verify each finding.",
        Complexity.HARD,
    ),
    (
        "Optimize the schedule for 6 tasks with constraints: A before B, C after D. "
        "Minimize total time. Prove your solution is optimal.",
        Complexity.HARD,
    ),
])
def test_classify(problem: str, expected: Complexity):
    assert classify(problem) == expected


def test_hard_threshold_override():
    hard_problem = "Prove that P≠NP."
    # With very high threshold, hard problems route to MEDIUM or EASY
    result = classify(hard_problem, hard_threshold=100)
    assert result != Complexity.HARD


def test_short_query_is_easy():
    assert classify("Hello.") == Complexity.EASY
```

---

## tests/test_router.py

```python
"""tests/test_router.py"""
from unittest.mock import MagicMock, patch
from reasoning_router.classifier import Complexity
from reasoning_router.config import Settings
from reasoning_router.router import route


def _settings(**overrides) -> Settings:
    base = dict(
        anthropic_api_key="test-key",
        fast_model="claude-haiku-5-5",
        reasoning_model="claude-sonnet-5-5",
        thinking_budget=8000,
        best_of_n=3,
        hard_threshold=5,
    )
    base.update(overrides)
    return Settings(**base)


def test_easy_routes_to_haiku():
    decision = route("What is 2+2?", _settings())
    assert decision.model == "claude-haiku-5-5"
    assert decision.thinking_budget is None
    assert decision.complexity == Complexity.EASY


def test_hard_routes_to_sonnet_with_thinking():
    problem = (
        "Prove that the halting problem is undecidable using a diagonalisation argument. "
        "Verify each step formally and derive a contradiction."
    )
    decision = route(problem, _settings())
    assert decision.model == "claude-sonnet-5-5"
    assert decision.thinking_budget == 8000
    assert decision.complexity == Complexity.HARD


def test_estimated_cost_is_positive():
    decision = route("What is 2+2?", _settings())
    assert decision.estimated_cost_usd > 0


def test_hard_cost_greater_than_easy():
    easy_decision = route("What is 2+2?", _settings())
    hard_problem = (
        "Verify this distributed locking algorithm for deadlock-freedom. "
        "Prove correctness using formal methods."
    )
    hard_decision = route(hard_problem, _settings())
    assert hard_decision.estimated_cost_usd > easy_decision.estimated_cost_usd
```

---

## tests/test_sampler.py

```python
"""tests/test_sampler.py"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from reasoning_router.config import Settings
from reasoning_router.sampler import sample_n


def _settings() -> Settings:
    return Settings(
        anthropic_api_key="test-key",
        reasoning_model="claude-sonnet-5-5",
        thinking_budget=4000,
        best_of_n=3,
        hard_threshold=5,
    )


def _make_response(text: str) -> MagicMock:
    block = MagicMock(); block.type = "text"; block.text = text
    r = MagicMock(); r.content = [block]
    return r


@pytest.mark.asyncio
async def test_sample_n_issues_n_calls():
    stub = _make_response("42")
    with patch("reasoning_router.sampler.anthropic.AsyncAnthropic") as MockAsync:
        mock_create = AsyncMock(return_value=stub)
        MockAsync.return_value.messages.create = mock_create

        results = await sample_n("What is 6×7?", n=3, settings=_settings())

    assert len(results) == 3
    assert mock_create.call_count == 3


@pytest.mark.asyncio
async def test_sample_n_returns_text():
    stub = _make_response("The answer is 1.2")
    with patch("reasoning_router.sampler.anthropic.AsyncAnthropic") as MockAsync:
        MockAsync.return_value.messages.create = AsyncMock(return_value=stub)
        results = await sample_n("Q?", n=2, settings=_settings())

    assert all("1.2" in r for r in results)
```

---

## tests/test_verifier.py

```python
"""tests/test_verifier.py"""
import pytest
from reasoning_router.verifier import numeric_match, select_best


@pytest.mark.parametrize("answer,gt,expected", [
    ("The answer is 5", "5", True),
    ("5.0", "5", True),
    ("4.99", "5", True),     # within tol=0.01
    ("4.98", "5", False),    # outside tol
    ("42", "43", False),
    ("Paris", "Paris", True),
    ("Paris", "Berlin", False),
    ("no number here", "5", False),
])
def test_numeric_match(answer: str, gt: str, expected: bool):
    assert numeric_match(answer, gt) == expected


def test_select_best_highest_score():
    candidates = ["bad answer", "good answer", "ok answer"]
    scores = {"bad answer": 0.1, "good answer": 0.9, "ok answer": 0.5}
    best = select_best(candidates, lambda a: scores[a])
    assert best == "good answer"


def test_select_best_empty():
    assert select_best([], lambda a: 1.0) == ""


def test_select_best_tie_returns_first():
    candidates = ["a", "b"]
    best = select_best(candidates, lambda a: 1.0)
    assert best == "a"
```

---

## tests/test_eval.py

```python
"""tests/test_eval.py"""
import asyncio
import pytest
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch
from reasoning_router.eval import EvalProblem, run_eval
from reasoning_router.config import Settings
from reasoning_router.router import SolveResult
from reasoning_router.classifier import Complexity


def _settings() -> Settings:
    return Settings(
        anthropic_api_key="test-key",
        fast_model="claude-haiku-5-5",
        reasoning_model="claude-sonnet-5-5",
        thinking_budget=4000,
        best_of_n=1,
        hard_threshold=5,
    )


def _make_solve_result(answer: str) -> SolveResult:
    return SolveResult(
        answer=answer,
        complexity=Complexity.EASY,
        model="claude-haiku-5-5",
        thinking_budget=None,
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.00005,
    )


@pytest.mark.asyncio
async def test_run_eval_accuracy_and_cost():
    problems = [
        EvalProblem("p1", "What is 2+2?", "4"),
        EvalProblem("p2", "What is 3+3?", "6"),
        EvalProblem("p3", "What is 10-5?", "5"),
    ]
    answers = {"What is 2+2?": "4", "What is 3+3?": "wrong", "What is 10-5?": "5"}

    async def mock_solve(problem: str, settings: Settings) -> SolveResult:
        return _make_solve_result(answers[problem])

    with patch("reasoning_router.eval.solve", side_effect=mock_solve):
        report = await run_eval(problems, _settings())

    assert report.n == 3
    assert abs(report.accuracy - 2 / 3) < 0.01
    assert report.total_cost_usd > 0
    assert len(report.per_problem) == 3
```
