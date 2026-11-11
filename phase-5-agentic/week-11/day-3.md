# Phase 5 · Week 11 · Day 3 — Agent Evaluation: Trajectory-Based Eval & Task-Success Metrics

> Objective: Design and implement trajectory-based agent evaluation; define task-success metrics and build a harness to score agent runs.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

## Concept

An agent evaluation harness answers: **did the agent accomplish the task correctly, and did it take the right path?**

Two complementary lenses:

| Lens | What it measures | Typical metric |
|------|-----------------|----------------|
| **Task success** | Final answer quality | Pass/fail, F1, ROUGE, exact match |
| **Trajectory quality** | Steps taken to reach the answer | Tool call correctness, step efficiency, error rate |

### Why trajectory matters

Two agents can both answer correctly but differ radically in path:
- Agent A: calls `search → summarise → answer` (3 steps, correct)
- Agent B: calls `search → search → search → hallucinate → answer` (5 steps, reached same answer by luck)

Agent B has a **fragile trajectory** — it will degrade on harder tasks. Trajectory eval catches this before production.

### Core vocabulary

- **Golden trajectory** — the expected sequence of tool calls + their arguments for a given input.
- **Step** — a single model or tool action in the message list.
- **Trajectory match** — how closely the actual path matches the golden path.
- **Task success** — whether the final output meets acceptance criteria (exact match, semantic match, or LLM judge).

## How it works internally

An evaluation run:

```
EvalCase(input, golden_trajectory, expected_output)
         │
         ├─ run agent with input → RunResult
         │
         ├─ extract actual_trajectory from result.all_messages()
         │
         ├─ score_trajectory(actual, golden) → TrajectoryScore
         │
         ├─ score_task(result.data, expected_output) → TaskScore
         │
         └─ EvalResult(trajectory_score, task_score, passed)
```

`result.all_messages()` returns the full message list: `SystemPrompt`, `UserPrompt`, `ModelTextResponse`, `ToolCallPart`, `ToolReturnPart`. Parsing these gives the trajectory.

## Worked example

```python
# src/eval/trajectory_eval.py
"""Trajectory-based agent evaluation harness."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel
from pydantic_ai.messages import (
    ModelMessage,
    ToolCallPart,
    ToolReturnPart,
)

logger = logging.getLogger(__name__)


# ── Data models ───────────────────────────────────────────────────────────────
class ToolCall(BaseModel):
    """A single recorded tool invocation."""
    tool_name: str
    args: dict[str, Any]


class Trajectory(BaseModel):
    """Ordered list of tool calls made during an agent run."""
    steps: list[ToolCall]

    @property
    def tool_names(self) -> list[str]:
        return [s.tool_name for s in self.steps]


@dataclass
class EvalCase:
    """One evaluation case: input + golden trajectory + expected output."""
    case_id: str
    user_input: str
    golden_trajectory: Trajectory
    expected_output: str  # simplified: exact-match string for now
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryScore:
    tool_name_match: float   # fraction of tool names that match in order
    step_count_delta: int    # actual_steps - golden_steps (0 = perfect)
    exact_match: bool        # full trajectory matches golden exactly


@dataclass
class TaskScore:
    passed: bool
    detail: str


@dataclass
class EvalResult:
    case_id: str
    trajectory_score: TrajectoryScore
    task_score: TaskScore

    @property
    def overall_pass(self) -> bool:
        return self.task_score.passed


# ── Trajectory extraction ─────────────────────────────────────────────────────
def extract_trajectory(messages: list[ModelMessage]) -> Trajectory:
    """Pull tool calls from a Pydantic AI message list."""
    steps: list[ToolCall] = []
    for msg in messages:
        for part in getattr(msg, "parts", []):
            if isinstance(part, ToolCallPart):
                # args_as_dict() returns dict regardless of how the model encoded them
                steps.append(
                    ToolCall(
                        tool_name=part.tool_name,
                        args=part.args_as_dict(),
                    )
                )
    return Trajectory(steps=steps)


# ── Scoring functions ─────────────────────────────────────────────────────────
def score_trajectory(actual: Trajectory, golden: Trajectory) -> TrajectoryScore:
    """Compare actual trajectory against golden."""
    actual_names = actual.tool_names
    golden_names = golden.tool_names

    # Fraction of positions where tool names match
    min_len = min(len(actual_names), len(golden_names))
    if min_len == 0:
        name_match = 1.0 if not golden_names and not actual_names else 0.0
    else:
        matches = sum(a == g for a, g in zip(actual_names, golden_names))
        name_match = matches / max(len(golden_names), 1)

    exact = actual_names == golden_names
    delta = len(actual.steps) - len(golden.steps)

    score = TrajectoryScore(
        tool_name_match=name_match,
        step_count_delta=delta,
        exact_match=exact,
    )
    logger.debug(
        "Trajectory score: name_match=%.2f delta=%d exact=%s",
        name_match, delta, exact,
    )
    return score


def score_task_exact(actual_output: str, expected: str) -> TaskScore:
    """Simple exact-match task scorer (case-insensitive strip)."""
    passed = actual_output.strip().lower() == expected.strip().lower()
    return TaskScore(
        passed=passed,
        detail=f"exact match: {passed!r}",
    )


# ── Evaluator ────────────────────────────────────────────────────────────────
class AgentEvaluator:
    """Run evaluation cases against an agent and collect results."""

    def __init__(self, agent_runner: Any) -> None:
        # agent_runner: async callable(user_input, deps) -> RunResult
        self._runner = agent_runner

    async def run_case(self, case: EvalCase, deps: Any) -> EvalResult:
        logger.info("Evaluating case: %s", case.case_id)
        run_result = await self._runner(case.user_input, deps)

        actual_traj = extract_trajectory(run_result.all_messages())
        traj_score = score_trajectory(actual_traj, case.golden_trajectory)

        # For now: compare string representation of result.data
        task_score = score_task_exact(str(run_result.data), case.expected_output)

        return EvalResult(
            case_id=case.case_id,
            trajectory_score=traj_score,
            task_score=task_score,
        )

    async def run_suite(self, cases: list[EvalCase], deps: Any) -> list[EvalResult]:
        results = []
        for case in cases:
            result = await self.run_case(case, deps)
            results.append(result)

        passed = sum(r.overall_pass for r in results)
        logger.info("Eval suite: %d/%d passed", passed, len(results))
        return results


# ── Summary helpers ───────────────────────────────────────────────────────────
def summarise_results(results: list[EvalResult]) -> dict[str, float]:
    if not results:
        return {}
    n = len(results)
    return {
        "pass_rate": sum(r.overall_pass for r in results) / n,
        "avg_tool_name_match": sum(
            r.trajectory_score.tool_name_match for r in results
        ) / n,
        "avg_step_delta": sum(
            abs(r.trajectory_score.step_count_delta) for r in results
        ) / n,
        "exact_trajectory_rate": sum(
            r.trajectory_score.exact_match for r in results
        ) / n,
    }
```

### Writing a test with this harness

```python
# tests/unit/test_trajectory_eval.py
import pytest
from src.eval.trajectory_eval import (
    EvalCase,
    Trajectory,
    ToolCall,
    extract_trajectory,
    score_trajectory,
    summarise_results,
)
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart
import json


def _make_tool_call_part(name: str, args: dict) -> ToolCallPart:
    return ToolCallPart(tool_name=name, args=json.dumps(args), tool_call_id="x")


def test_extract_trajectory_empty():
    traj = extract_trajectory([])
    assert traj.steps == []


def test_extract_trajectory_single_tool():
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    msg = ModelResponse(parts=[_make_tool_call_part("search", {"q": "test"})])
    traj = extract_trajectory([msg])
    assert len(traj.steps) == 1
    assert traj.steps[0].tool_name == "search"


def test_score_trajectory_exact_match():
    golden = Trajectory(steps=[ToolCall(tool_name="a", args={}), ToolCall(tool_name="b", args={})])
    actual = Trajectory(steps=[ToolCall(tool_name="a", args={}), ToolCall(tool_name="b", args={})])
    score = score_trajectory(actual, golden)
    assert score.exact_match is True
    assert score.tool_name_match == 1.0
    assert score.step_count_delta == 0


def test_score_trajectory_extra_steps():
    golden = Trajectory(steps=[ToolCall(tool_name="a", args={})])
    actual = Trajectory(steps=[ToolCall(tool_name="a", args={}), ToolCall(tool_name="b", args={})])
    score = score_trajectory(actual, golden)
    assert score.step_count_delta == 1
    assert score.exact_match is False


def test_summarise_results():
    from src.eval.trajectory_eval import EvalResult, TrajectoryScore, TaskScore
    results = [
        EvalResult("c1", TrajectoryScore(1.0, 0, True), TaskScore(True, "")),
        EvalResult("c2", TrajectoryScore(0.5, 1, False), TaskScore(False, "")),
    ]
    summary = summarise_results(results)
    assert summary["pass_rate"] == 0.5
    assert summary["avg_tool_name_match"] == 0.75
```

## Exercises

1. Extend `TrajectoryScore` with an `arg_match_rate: float` field that checks whether tool call arguments (not just names) match the golden. Implement `score_trajectory_args(actual, golden) -> float`.

2. Add a `score_task_contains(actual: str, keywords: list[str]) -> TaskScore` function that passes if all keywords are present in the output. Write two tests.

3. Modify `AgentEvaluator.run_suite` to run cases concurrently with `asyncio.gather`. Add a `concurrency: int = 1` parameter.

4. Define an `EvalReport` Pydantic model that holds `List[EvalResult]` plus the `summarise_results` dict. Add a `to_json_file(path: pathlib.Path)` method.

(Solutions: ../solutions/week-11/day-3.md)

## Self-check

- [ ] I can explain the difference between task-success eval and trajectory eval.
- [ ] I can extract a `Trajectory` from `RunResult.all_messages()`.
- [ ] I can implement `score_trajectory` with tool-name match and step-delta metrics.
- [ ] I can write an `AgentEvaluator` that runs a suite of cases and returns structured results.
- [ ] I can unit-test the eval harness without any real LLM calls.

## Supplementary links

- [Pydantic AI — Evals](https://ai.pydantic.dev/testing-evals/)
- [Pydantic AI — Messages API](https://ai.pydantic.dev/message-history/)
