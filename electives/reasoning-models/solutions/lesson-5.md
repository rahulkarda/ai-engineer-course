# Solutions · Lesson 5 — Verifiers, Evals, and Agents

## Exercise 1 — Process reward model

```python
"""solutions/lesson_5_ex1.py"""
from __future__ import annotations
import json
import os

import anthropic

STEP_JUDGE_PROMPT = """\
You are a reasoning step evaluator.
Given a problem and one step in a reasoning chain, score the step 0.0 to 1.0.
Return only valid JSON: {{"score": <float>, "reason": "<one sentence>"}}

Problem: {problem}
Reasoning step: {step}"""


def step_verifier(
    client: anthropic.Anthropic,
    steps: list[str],
    problem: str,
) -> list[float]:
    """Score each reasoning step with an LLM judge."""
    scores: list[float] = []
    for step in steps:
        prompt = STEP_JUDGE_PROMPT.format(problem=problem, step=step)
        response = client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=128,
            messages=[{"role": "user", "content": prompt}],
        )
        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)
        scores.append(float(data["score"]))
    return scores
```

```python
"""tests/test_lesson_5_ex1.py"""
from unittest.mock import MagicMock, patch
from solutions.lesson_5_ex1 import step_verifier


def _make_response(text: str) -> MagicMock:
    block = MagicMock(); block.type = "text"; block.text = text
    r = MagicMock(); r.content = [block]
    return r


def test_step_verifier_returns_same_length():
    steps = ["Let x = 5.", "Then 2x = 10.", "Therefore the answer is 10."]
    problem = "Solve 2x = 10."

    stub = _make_response('{"score": 0.9, "reason": "correct step"}')

    with patch("solutions.lesson_5_ex1.anthropic.Anthropic") as MockClient:
        mock_create = MockClient.return_value.messages.create
        mock_create.return_value = stub
        client = MockClient()
        scores = step_verifier(client, steps, problem)

    assert len(scores) == len(steps)
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert mock_create.call_count == len(steps)
```

---

## Exercise 2 — Calibration curve

```python
"""solutions/lesson_5_ex2.py"""
from __future__ import annotations
import math


def calibration_error(pairs: list[tuple[float, bool]], n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).

    pairs: list of (confidence, correct) tuples
    Returns mean absolute difference between binned confidence and bin accuracy.
    """
    if not pairs:
        return 0.0

    bins: dict[int, list[tuple[float, bool]]] = {i: [] for i in range(n_bins)}
    for conf, correct in pairs:
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx].append((conf, correct))

    total_error = 0.0
    total_samples = len(pairs)

    for bin_data in bins.values():
        if not bin_data:
            continue
        avg_conf = sum(c for c, _ in bin_data) / len(bin_data)
        avg_acc = sum(int(c) for _, c in bin_data) / len(bin_data)
        weight = len(bin_data) / total_samples
        total_error += weight * abs(avg_conf - avg_acc)

    return round(total_error, 6)
```

```python
"""tests/test_lesson_5_ex2.py"""
from solutions.lesson_5_ex2 import calibration_error


def test_perfectly_calibrated():
    # 50% confidence, 50% accuracy → ECE ≈ 0
    pairs = [(0.5, True), (0.5, False)] * 50
    ece = calibration_error(pairs)
    assert ece < 0.05  # near zero


def test_overconfident():
    # 90% confidence but 50% accuracy → high ECE
    pairs = [(0.9, True), (0.9, False)] * 50
    ece = calibration_error(pairs)
    assert ece > 0.3


def test_empty_input():
    assert calibration_error([]) == 0.0
```

---

## Exercise 3 — Eval harness with cost tracking

```python
"""solutions/lesson_5_ex3.py"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable


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


def run_eval(
    problems: list[EvalProblem],
    solver: Callable[[str], tuple[str, int, int]],
    verifier: Callable[[str, str], bool],
    usage_tracker: Callable[[int, int], None],
    cost_table: dict[str, float],
) -> dict:
    """
    Run eval with cost tracking.

    solver: callable(problem) -> (answer, input_tokens, output_tokens)
    usage_tracker: callable(input_tokens, output_tokens) -> None
    cost_table: {"input_per_m": float, "output_per_m": float}
    """
    results = []
    total_cost = 0.0

    for prob in problems:
        answer, in_tok, out_tok = solver(prob.problem)
        usage_tracker(in_tok, out_tok)
        cost = (in_tok / 1e6) * cost_table["input_per_m"] + \
               (out_tok / 1e6) * cost_table["output_per_m"]
        total_cost += cost
        correct = verifier(answer, prob.answer)
        results.append(EvalResult(id=prob.id, model_answer=answer, correct=correct))

    n = len(results)
    accuracy = sum(r.correct for r in results) / n if n else 0.0

    return {
        "n": n,
        "accuracy": round(accuracy, 4),
        "total_cost": round(total_cost, 8),
        "results": [asdict(r) for r in results],
    }
```

```python
"""tests/test_lesson_5_ex3.py"""
from solutions.lesson_5_ex3 import run_eval, EvalProblem


def test_run_eval_with_cost_tracking():
    problems = [
        EvalProblem("p1", "What is 2+2?", "4"),
        EvalProblem("p2", "What is 3+3?", "6"),
        EvalProblem("p3", "What is 10-5?", "5"),
    ]

    answers_map = {"What is 2+2?": "4", "What is 3+3?": "wrong", "What is 10-5?": "5"}
    calls_log: list[tuple[int, int]] = []

    def solver(problem: str) -> tuple[str, int, int]:
        return answers_map[problem], 100, 50

    def verifier(a: str, gt: str) -> bool:
        return a.strip() == gt.strip()

    def usage_tracker(inp: int, out: int) -> None:
        calls_log.append((inp, out))

    cost_table = {"input_per_m": 1.0, "output_per_m": 5.0}
    summary = run_eval(problems, solver, verifier, usage_tracker, cost_table)

    assert summary["n"] == 3
    assert summary["accuracy"] == pytest.approx(2 / 3, rel=1e-3)
    assert summary["total_cost"] > 0
    assert len(calls_log) == 3


import pytest
```

---

## Exercise 4 — Tool use + thinking test

```python
"""tests/test_lesson_5_ex4.py"""
from unittest.mock import MagicMock, patch, call
from lesson_5_verifier_eval import agentic_solver  # adjust import


def _thinking_block() -> MagicMock:
    b = MagicMock(); b.type = "thinking"; b.thinking = "Let me calculate."
    return b


def _tool_use_block(tool_id: str, expression: str) -> MagicMock:
    b = MagicMock()
    b.type = "tool_use"
    b.id = tool_id
    b.name = "calculate"
    b.input = {"expression": expression}
    return b


def _text_block(text: str) -> MagicMock:
    b = MagicMock(); b.type = "text"; b.text = text
    return b


def _make_response(blocks, stop_reason: str) -> MagicMock:
    r = MagicMock()
    r.content = blocks
    r.stop_reason = stop_reason
    return r


def test_agentic_solver_tool_loop():
    tool_response = _make_response(
        [_thinking_block(), _tool_use_block("tool-1", "3 ** 10")],
        stop_reason="tool_use",
    )
    final_response = _make_response(
        [_thinking_block(), _text_block("3^10 = 59049")],
        stop_reason="end_turn",
    )

    with patch("lesson_5_verifier_eval.anthropic.Anthropic") as MockClient:
        mock_create = MockClient.return_value.messages.create
        mock_create.side_effect = [tool_response, final_response]

        client = MockClient()
        result = agentic_solver(client, "What is 3^10?", budget=2000)

    assert result == "3^10 = 59049"
    assert mock_create.call_count == 2

    # Second call's messages should include the tool result
    second_call_messages = mock_create.call_args_list[1][1]["messages"]
    tool_result_turn = next(
        m for m in second_call_messages if m["role"] == "user"
        and isinstance(m["content"], list)
        and any(c.get("type") == "tool_result" for c in m["content"])
    )
    assert tool_result_turn is not None
```

---

## Exercise 5 — Multi-judge aggregation

```python
"""solutions/lesson_5_ex5.py"""
from __future__ import annotations


def aggregate_judges(scores: list[dict]) -> float:
    """
    Average the 'score' field from multiple judge responses.

    Handles missing 'score' keys gracefully (skips them).
    """
    valid_scores = []
    for judge in scores:
        if isinstance(judge, dict) and "score" in judge:
            try:
                valid_scores.append(float(judge["score"]))
            except (TypeError, ValueError):
                pass

    if not valid_scores:
        return 0.0

    return round(sum(valid_scores) / len(valid_scores), 4)
```

```python
"""tests/test_lesson_5_ex5.py"""
from solutions.lesson_5_ex5 import aggregate_judges


def test_three_judges():
    scores = [{"score": 8}, {"score": 7}, {"score": 9}]
    result = aggregate_judges(scores)
    assert abs(result - 8.0) < 1e-4


def test_missing_score_skipped():
    scores = [{"score": 8}, {"reason": "no score here"}, {"score": 6}]
    result = aggregate_judges(scores)
    assert abs(result - 7.0) < 1e-4


def test_all_missing():
    scores = [{"reason": "no"}, {}]
    assert aggregate_judges(scores) == 0.0


def test_empty_list():
    assert aggregate_judges([]) == 0.0
```
