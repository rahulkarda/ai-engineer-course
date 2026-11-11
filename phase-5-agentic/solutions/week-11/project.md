# Project Reference Implementation — Week 11

> Full reference for the evaluation & observability harness project.

## Project structure

```
week-11-project/
├── src/
│   ├── agents/
│   │   └── code_review_agent.py
│   ├── eval/
│   │   ├── trajectory_eval.py
│   │   ├── llm_judge.py
│   │   ├── golden_store.py
│   │   └── regression_gate.py
│   └── observability/
│       ├── cost_tracker.py
│       └── tracer.py
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_tools.py
│   │   ├── test_trajectory.py
│   │   ├── test_judge.py
│   │   ├── test_gate.py
│   │   └── test_cost.py
│   ├── integration/
│   │   └── test_eval_suite.py
│   └── golden/
├── scripts/
│   └── capture_golden.py
└── pyproject.toml
```

## src/agents/code_review_agent.py (complete)

```python
"""Typed code review agent — Phase 5 Week 11 project."""
from __future__ import annotations

import ast
import logging
import os
import pathlib
from dataclasses import dataclass, field
from typing import Annotated

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext

logger = logging.getLogger(__name__)


class ReviewResult(BaseModel):
    issues: list[str]
    severity: Annotated[str, Field(pattern=r"^(low|medium|high)$")]
    recommendation: str


@dataclass
class ReviewDeps:
    allowed_imports: list[str] = field(default_factory=list)
    max_complexity: int = 10
    base_dir: pathlib.Path = field(default_factory=pathlib.Path.cwd)


review_agent: Agent[ReviewDeps, ReviewResult] = Agent(
    model="anthropic:claude-sonnet-5-5",
    deps_type=ReviewDeps,
    result_type=ReviewResult,
    system_prompt="You are a senior Python code reviewer. Use all available tools.",
)


@review_agent.system_prompt
def add_complexity_context(ctx: RunContext[ReviewDeps]) -> str:
    return f"Maximum acceptable cyclomatic complexity: {ctx.deps.max_complexity}"


@review_agent.tool_plain
def count_functions(source_code: str) -> dict[str, int]:
    """Count functions, classes, and imports in source code."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        raise ModelRetry(f"SyntaxError: {exc}") from exc
    return {
        "functions": sum(1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)),
        "classes": sum(1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef)),
        "imports": sum(
            1 for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
        ),
    }


@review_agent.tool
def check_imports(ctx: RunContext[ReviewDeps], source_code: str) -> list[str]:
    """Return disallowed imports. Raises ModelRetry on SyntaxError."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        raise ModelRetry(f"SyntaxError: {exc}") from exc

    if not ctx.deps.allowed_imports:
        return []

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ctx.deps.allowed_imports:
                    violations.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root not in ctx.deps.allowed_imports:
                violations.append(node.module)
    return violations


@review_agent.tool_plain
def estimate_complexity(source_code: str) -> int:
    """Estimate cyclomatic complexity as count of branching statements."""
    tree = ast.parse(source_code)
    return sum(
        1
        for n in ast.walk(tree)
        if isinstance(n, (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.With))
    )


@review_agent.tool_plain
def detect_security_issues(source_code: str) -> list[str]:
    """Detect dangerous patterns: eval, exec, subprocess shell=True, pickle.load."""
    patterns = ["eval(", "exec(", "subprocess(shell=True", "pickle.load("]
    return [p for p in patterns if p in source_code]


async def review_code(source: str, deps: ReviewDeps) -> ReviewResult:
    logger.info("Starting code review (max_complexity=%d)", deps.max_complexity)
    result = await review_agent.run(
        f"Review this Python code:\n\n```python\n{source}\n```",
        deps=deps,
    )
    logger.info("Review complete. Tokens: %s", result.usage())
    return result.data
```

## tests/unit/test_tools.py

```python
"""Unit tests for code review agent tools."""
import pytest
from src.agents.code_review_agent import (
    count_functions,
    detect_security_issues,
    estimate_complexity,
)


def test_count_functions_basic():
    code = "def f(): pass\ndef g(): pass\nclass C: pass\nimport os\n"
    result = count_functions(code)
    assert result["functions"] == 2
    assert result["classes"] == 1
    assert result["imports"] == 1


def test_count_functions_empty():
    result = count_functions("")
    assert result == {"functions": 0, "classes": 0, "imports": 0}


def test_detect_security_issues_finds_eval():
    code = "def f(x): return eval(x)\n"
    issues = detect_security_issues(code)
    assert "eval(" in issues


def test_detect_security_issues_finds_exec():
    code = "exec('print(1)')\n"
    issues = detect_security_issues(code)
    assert "exec(" in issues


def test_detect_security_issues_clean_code():
    code = "def f(x): return x * 2\n"
    issues = detect_security_issues(code)
    assert issues == []


def test_estimate_complexity_branches():
    code = """
def f(x):
    if x > 0:
        for i in range(x):
            if i % 2 == 0:
                pass
    return x
"""
    complexity = estimate_complexity(code)
    assert complexity == 3  # if + for + if
```

## tests/unit/test_trajectory.py

```python
"""Unit tests for trajectory evaluation."""
import json
import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from src.eval.trajectory_eval import (
    Trajectory,
    ToolCall,
    extract_trajectory,
    score_trajectory,
    summarise_results,
)


def _make_tool_msg(name: str, args: dict) -> ModelResponse:
    return ModelResponse(
        parts=[ToolCallPart(tool_name=name, args=json.dumps(args), tool_call_id="x")]
    )


def test_extract_trajectory_empty():
    assert extract_trajectory([]).steps == []


def test_extract_trajectory_single():
    msg = _make_tool_msg("search", {"q": "test"})
    traj = extract_trajectory([msg])
    assert len(traj.steps) == 1
    assert traj.steps[0].tool_name == "search"


def test_score_exact_match():
    g = Trajectory(steps=[ToolCall(tool_name="a", args={}), ToolCall(tool_name="b", args={})])
    a = Trajectory(steps=[ToolCall(tool_name="a", args={}), ToolCall(tool_name="b", args={})])
    s = score_trajectory(a, g)
    assert s.exact_match is True
    assert s.tool_name_match == 1.0
    assert s.step_count_delta == 0


def test_score_extra_steps():
    g = Trajectory(steps=[ToolCall(tool_name="a", args={})])
    a = Trajectory(steps=[ToolCall(tool_name="a", args={}), ToolCall(tool_name="b", args={})])
    s = score_trajectory(a, g)
    assert s.step_count_delta == 1
    assert s.exact_match is False
```

## tests/unit/test_cost.py

```python
"""Unit tests for cost tracker."""
import pytest
from unittest.mock import MagicMock
from src.observability.cost_tracker import CostTracker, cost_usd


def _usage(inp, out):
    u = MagicMock()
    u.request_tokens = inp
    u.response_tokens = out
    return u


def test_cost_haiku():
    cost = cost_usd("claude-haiku-5-5", _usage(1_000_000, 1_000_000))
    assert cost == pytest.approx(1.50)  # 0.25 + 1.25


def test_cost_sonnet():
    cost = cost_usd("claude-sonnet-5-5", _usage(1_000_000, 1_000_000))
    assert cost == pytest.approx(18.0)


def test_cost_opus():
    cost = cost_usd("claude-opus-5-5", _usage(1_000_000, 1_000_000))
    assert cost == pytest.approx(90.0)


def test_cost_unknown_model_returns_zero():
    assert cost_usd("unknown", _usage(1000, 1000)) == 0.0


def test_tracker_total_tokens():
    tracker = CostTracker()
    tracker.record("r1", "claude-haiku-5-5", _usage(500, 100), 80.0, 2, True)
    assert tracker.summary()["total_tokens"] == 600


def test_tracker_success_rate():
    tracker = CostTracker()
    tracker.record("r1", "claude-haiku-5-5", _usage(100, 50), 80.0, 1, True)
    tracker.record("r2", "claude-haiku-5-5", _usage(100, 50), 80.0, 1, False)
    assert tracker.summary()["success_rate"] == 0.5
```

## tests/integration/test_eval_suite.py

```python
"""Integration: regression gate end-to-end."""
import pytest
from pathlib import Path
from src.eval.golden_store import GoldenRecord, GoldenStore
from src.eval.regression_gate import RegressionGate
from src.eval.trajectory_eval import Trajectory, ToolCall


@pytest.fixture
def approved_store(tmp_path):
    store = GoldenStore(tmp_path)
    store.save(GoldenRecord(
        case_id="c1",
        user_input="review code",
        golden_trajectory=Trajectory(steps=[
            ToolCall(tool_name="count_functions", args={}),
            ToolCall(tool_name="detect_security_issues", args={}),
        ]),
        expected_output="low",
        captured_at="2025-01-01T00:00:00+00:00",
        approved=True,
    ))
    return store


def test_gate_passes_on_matching_run(approved_store):
    gate = RegressionGate(approved_store, max_step_delta=1)
    traj = Trajectory(steps=[
        ToolCall(tool_name="count_functions", args={}),
        ToolCall(tool_name="detect_security_issues", args={}),
    ])
    assert gate.evaluate("c1", traj, "low", True) is None


def test_gate_fails_on_task_failure(approved_store):
    gate = RegressionGate(approved_store)
    traj = Trajectory(steps=[ToolCall(tool_name="count_functions", args={})])
    reg = gate.evaluate("c1", traj, "", False)
    assert reg is not None
    assert reg.kind == "task_failure"


def test_gate_report_all_pass(approved_store):
    gate = RegressionGate(approved_store)
    traj = Trajectory(steps=[
        ToolCall(tool_name="count_functions", args={}),
        ToolCall(tool_name="detect_security_issues", args={}),
    ])
    report = gate.run_report([{
        "case_id": "c1",
        "trajectory": traj,
        "output": "low",
        "passed": True,
    }])
    assert report.gate_passed is True
    assert report.pass_rate == 1.0
```

## pyproject.toml

```toml
[project]
name = "week-11-project"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "pydantic-ai>=0.0.14",
    "pydantic>=2.10",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.6",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]
```
