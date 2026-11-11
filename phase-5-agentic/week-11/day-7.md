# Phase 5 · Week 11 · Day 7 — Week Project: Evaluation & Observability Harness

> Objective: Integrate all week's concepts into a complete eval + observability harness for a typed Pydantic AI agent.  ·  Time: ~3 hr project + 1 hr assessment review  ·  Prereqs: [[day-1]]…[[day-6]]

## Project overview

You will take the **code review agent** from Day 2, wrap it with Pydantic AI typing, build a full evaluation harness (golden trajectories + LLM judge), and add token/cost tracing. All LLM calls are mocked in tests.

See [[project.md]] for full spec and deliverables.

## Integration recap

Before building, map the components:

```
┌─────────────────────────────────────────────────────┐
│                  week-11 stack                      │
│                                                     │
│  Pydantic AI Agent                                  │
│    deps_type: ReviewDeps                            │
│    result_type: ReviewResult                        │
│    tools: count_functions, check_imports,           │
│            estimate_complexity                      │
│    model: claude-sonnet-5-5                         │
│                                                     │
│  Evaluation Harness                                 │
│    GoldenStore  (tests/golden/*.json)               │
│    AgentEvaluator  (trajectory + task success)      │
│    LLM Judge  (factual_accuracy, completeness,      │
│                conciseness)                         │
│    RegressionGate  (CI gate on drift)               │
│                                                     │
│  Observability                                      │
│    CostTracker  (tokens + USD)                      │
│    Tracer  (spans per tool call + model call)       │
│    InstrumentedRunner  (wraps agent.run())          │
└─────────────────────────────────────────────────────┘
```

## Step-by-step guide

### Step 1 — Project scaffold

```
week-11-project/
├── src/
│   ├── agents/
│   │   └── code_review_agent.py   # from day-2 (typed)
│   ├── eval/
│   │   ├── trajectory_eval.py     # from day-3
│   │   ├── llm_judge.py           # from day-4
│   │   ├── golden_store.py        # from day-5
│   │   └── regression_gate.py    # from day-5
│   └── observability/
│       ├── cost_tracker.py        # from day-6
│       └── tracer.py              # from day-6
├── tests/
│   ├── unit/
│   │   ├── test_trajectory.py
│   │   ├── test_judge.py
│   │   └── test_gate.py
│   ├── integration/
│   │   └── test_eval_suite.py
│   └── golden/                    # captured golden files
├── scripts/
│   └── capture_golden.py
└── pyproject.toml
```

### Step 2 — Confirm agent types

Ensure `code_review_agent` from Day 2 is fully typed:
- `Agent[ReviewDeps, ReviewResult]`
- All three tools registered with type-safe signatures
- `ReviewResult` with `issues: list[str]`, `severity: str`, `recommendation: str`

### Step 3 — Capture golden trajectories

Run the agent with known inputs using `TestModel` to get deterministic trajectories, then save them as goldens:

```python
# scripts/capture_golden.py
"""Capture golden trajectories for the code review agent."""
import asyncio
import json
import logging
from pathlib import Path

from pydantic_ai.models.test import TestModel

from src.agents.code_review_agent import ReviewDeps, ReviewResult, review_agent
from src.eval.golden_store import GoldenStore, capture_golden
from src.eval.trajectory_eval import extract_trajectory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOLDEN_DIR = Path("tests/golden")

CAPTURE_CASES = [
    {
        "case_id": "review-001",
        "input": "Review this code:\n\ndef f(x):\n    return eval(x)\n",
    },
    {
        "case_id": "review-002",
        "input": "Review this code:\n\nimport os\ndef g():\n    return os.listdir('.')\n",
    },
]


async def main() -> None:
    store = GoldenStore(GOLDEN_DIR)
    deps = ReviewDeps(allowed_imports=["os", "sys"])

    for case in CAPTURE_CASES:
        with review_agent.override(model=TestModel()):
            result = await review_agent.run(case["input"], deps=deps)

        traj = extract_trajectory(result.all_messages())
        record = capture_golden(
            case_id=case["case_id"],
            user_input=case["input"],
            actual_trajectory=traj,
            actual_output=str(result.data),
            store=store,
        )
        # Auto-approve captured goldens in this script (human review in real CI)
        store.approve(case["case_id"])
        logger.info("Captured and approved: %s  steps=%d", case["case_id"], len(traj.steps))


if __name__ == "__main__":
    asyncio.run(main())
```

### Step 4 — Full evaluation test

```python
# tests/integration/test_eval_suite.py
"""Integration: full eval suite with regression gate."""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic_ai.models.test import TestModel

from src.agents.code_review_agent import ReviewDeps, ReviewResult, review_agent
from src.eval.regression_gate import RegressionGate
from src.eval.golden_store import GoldenStore, GoldenRecord
from src.eval.trajectory_eval import Trajectory, ToolCall, extract_trajectory


@pytest.fixture
def golden_store(tmp_path) -> GoldenStore:
    store = GoldenStore(tmp_path)
    record = GoldenRecord(
        case_id="review-001",
        user_input="Review: def f(x): return eval(x)",
        golden_trajectory=Trajectory(steps=[
            ToolCall(tool_name="count_functions", args={}),
            ToolCall(tool_name="check_imports", args={}),
        ]),
        expected_output="high",  # simplified: severity field
        captured_at="2025-01-01T00:00:00+00:00",
        approved=True,
    )
    store.save(record)
    return store


@pytest.mark.asyncio
async def test_regression_gate_passes_on_correct_run(golden_store):
    gate = RegressionGate(golden_store, max_step_delta=2, min_name_match=0.5)
    actual = Trajectory(steps=[
        ToolCall(tool_name="count_functions", args={}),
        ToolCall(tool_name="check_imports", args={}),
    ])
    result = gate.evaluate("review-001", actual, "high", True)
    assert result is None  # no regression


@pytest.mark.asyncio
async def test_regression_gate_detects_task_failure(golden_store):
    gate = RegressionGate(golden_store)
    actual = Trajectory(steps=[ToolCall(tool_name="count_functions", args={})])
    result = gate.evaluate("review-001", actual, "", False)
    assert result is not None
    assert result.kind == "task_failure"
```

### Step 5 — Cost tracking test

```python
# tests/unit/test_cost_tracker.py
import pytest
from unittest.mock import MagicMock
from src.observability.cost_tracker import CostTracker, cost_usd


def _mock_usage(input_tokens: int, output_tokens: int):
    usage = MagicMock()
    usage.request_tokens = input_tokens
    usage.response_tokens = output_tokens
    return usage


def test_cost_usd_sonnet():
    usage = _mock_usage(1_000_000, 1_000_000)
    cost = cost_usd("claude-sonnet-5-5", usage)
    assert cost == pytest.approx(18.0)


def test_cost_usd_unknown_model():
    usage = _mock_usage(1_000, 500)
    cost = cost_usd("unknown-model", usage)
    assert cost == 0.0


def test_tracker_summary_empty():
    tracker = CostTracker()
    assert tracker.summary() == {}


def test_tracker_records_and_summarises():
    tracker = CostTracker()
    usage = _mock_usage(100, 200)
    tracker.record("run-1", "claude-haiku-5-5", usage, latency_ms=150.0, tool_calls=2, success=True)
    summary = tracker.summary()
    assert summary["runs"] == 1
    assert summary["success_rate"] == 1.0
    assert summary["avg_latency_ms"] == 150.0
```

### Step 6 — LLM judge integration test

```python
# tests/unit/test_judge_integration.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.eval.llm_judge import JudgeRequest, JudgeVerdict, judge, batch_judge


@pytest.fixture
def good_verdict():
    return JudgeVerdict(
        factual_accuracy=5, completeness=4, conciseness=5,
        reasoning="Good response.", overall_score=0.933, passed=True,
    )


@pytest.mark.asyncio
async def test_batch_judge_all_pass(good_verdict):
    mock_result = MagicMock()
    mock_result.data = good_verdict
    reqs = [JudgeRequest("q1", "a1"), JudgeRequest("q2", "a2")]
    with patch("src.eval.llm_judge.judge_agent.run", new_callable=AsyncMock) as m:
        m.return_value = mock_result
        verdicts = await batch_judge(reqs)
    assert len(verdicts) == 2
    assert all(v.passed for v in verdicts)
```

## Learning consolidation

Review the concepts from each day:

| Day | Key pattern | Where it lives in the project |
|-----|-------------|-------------------------------|
| 1 | Agent[Deps, Result], dynamic prompts | `code_review_agent.py` |
| 2 | @tool, ModelRetry, Annotated validation | `code_review_agent.py` tools |
| 3 | Trajectory extraction, task scoring | `trajectory_eval.py` |
| 4 | LLM judge, JudgeVerdict schema | `llm_judge.py` |
| 5 | GoldenStore, RegressionGate | `golden_store.py`, `regression_gate.py` |
| 6 | Spans, CostTracker, OTel | `observability/` |

## Self-check for the week

- [ ] Agent is typed end-to-end: `Agent[ReviewDeps, ReviewResult]`.
- [ ] All tools use type-annotated signatures; `ModelRetry` on recoverable errors.
- [ ] `extract_trajectory` correctly parses all tool calls from `all_messages()`.
- [ ] `AgentEvaluator` runs a suite of cases and returns `EvalResult` per case.
- [ ] LLM judge uses `Annotated[int, Field(ge=1, le=5)]` for scores; validated output.
- [ ] `GoldenStore` can save/load/approve `GoldenRecord` models.
- [ ] `RegressionGate` detects task_failure and trajectory_drift.
- [ ] `CostTracker` computes USD cost from `Usage` for all three Claude model tiers.
- [ ] All LLM calls in tests use `TestModel` or `unittest.mock` — no real API keys.
- [ ] Test suite runs `pytest -x` with no failures.

## Supplementary links

- [Pydantic AI — full API reference](https://ai.pydantic.dev/)
- [Pydantic AI — TestModel](https://ai.pydantic.dev/testing-evals/#unit-testing-with-testmodel)
- [Logfire pricing](https://logfire.pydantic.dev/docs/pricing/)
