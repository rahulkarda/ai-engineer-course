# Solutions — Week 11 · Day 3: Trajectory-Based Eval & Task-Success Metrics

## Exercise 1 — arg_match_rate in TrajectoryScore

```python
# Extended TrajectoryScore
from dataclasses import dataclass
from src.eval.trajectory_eval import Trajectory


@dataclass
class TrajectoryScore:
    tool_name_match: float
    step_count_delta: int
    exact_match: bool
    arg_match_rate: float = 0.0  # fraction of steps where args match golden


def score_trajectory_args(actual: Trajectory, golden: Trajectory) -> float:
    """Return fraction of tool calls where args exactly match the golden."""
    if not golden.steps:
        return 1.0 if not actual.steps else 0.0

    min_len = min(len(actual.steps), len(golden.steps))
    if min_len == 0:
        return 0.0

    matches = 0
    for a_step, g_step in zip(actual.steps[:min_len], golden.steps[:min_len]):
        if a_step.tool_name == g_step.tool_name and a_step.args == g_step.args:
            matches += 1

    return matches / len(golden.steps)
```

**Tests:**

```python
def test_arg_match_rate_perfect():
    from src.eval.trajectory_eval import ToolCall, Trajectory
    golden = Trajectory(steps=[ToolCall(tool_name="search", args={"q": "ai"})])
    actual = Trajectory(steps=[ToolCall(tool_name="search", args={"q": "ai"})])
    assert score_trajectory_args(actual, golden) == 1.0


def test_arg_match_rate_wrong_args():
    from src.eval.trajectory_eval import ToolCall, Trajectory
    golden = Trajectory(steps=[ToolCall(tool_name="search", args={"q": "ai"})])
    actual = Trajectory(steps=[ToolCall(tool_name="search", args={"q": "ml"})])  # different args
    assert score_trajectory_args(actual, golden) == 0.0
```

---

## Exercise 2 — score_task_contains

```python
def score_task_contains(actual: str, keywords: list[str]) -> "TaskScore":
    """Pass if all keywords are present in the actual output (case-insensitive)."""
    from src.eval.trajectory_eval import TaskScore
    lower = actual.lower()
    missing = [kw for kw in keywords if kw.lower() not in lower]
    passed = len(missing) == 0
    detail = "all keywords present" if passed else f"missing: {missing}"
    return TaskScore(passed=passed, detail=detail)
```

**Tests:**

```python
def test_score_task_contains_all_present():
    result = score_task_contains("Python agents use LLMs", ["python", "agents"])
    assert result.passed is True


def test_score_task_contains_missing_keyword():
    result = score_task_contains("Python agents use LLMs", ["python", "RAG"])
    assert result.passed is False
    assert "RAG" in result.detail
```

---

## Exercise 3 — Concurrent suite with asyncio.gather

```python
import asyncio
from typing import Any


class AgentEvaluator:
    def __init__(self, agent_runner: Any, concurrency: int = 1) -> None:
        self._runner = agent_runner
        self._concurrency = concurrency

    async def run_case(self, case, deps) -> "EvalResult":
        # ... same as before
        pass

    async def run_suite(self, cases: list, deps: Any) -> list:
        if self._concurrency == 1:
            return [await self.run_case(c, deps) for c in cases]

        # Run up to `concurrency` cases in parallel
        semaphore = asyncio.Semaphore(self._concurrency)

        async def _bounded(case):
            async with semaphore:
                return await self.run_case(case, deps)

        results = await asyncio.gather(*(_bounded(c) for c in cases))
        return list(results)
```

**Design note:** Semaphore limits parallel LLM calls to avoid rate-limit errors. `concurrency=1` is the safe default.

---

## Exercise 4 — EvalReport model

```python
import json
import pathlib
from typing import Any
from pydantic import BaseModel
from src.eval.trajectory_eval import EvalResult, summarise_results


class EvalReport(BaseModel):
    results: list[EvalResult]
    summary: dict[str, float]

    @classmethod
    def build(cls, results: list[EvalResult]) -> "EvalReport":
        return cls(results=results, summary=summarise_results(results))

    def to_json_file(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
```

`EvalResult` must be a Pydantic `BaseModel` (not a dataclass) for this to serialise cleanly with `model_dump_json`. Convert the dataclass version if needed:

```python
class EvalResult(BaseModel):
    case_id: str
    trajectory_score: TrajectoryScore
    task_score: TaskScore

    @property
    def overall_pass(self) -> bool:
        return self.task_score.passed
```
