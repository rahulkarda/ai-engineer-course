# Solutions — Phase 5 · Week 03 · Day 6 — Full Plan-and-Execute Agent Build

This is the complete reference implementation for the `plan_execute_agent` package.
Every file is production-ready with full type annotations, logging, and a pytest suite
where all LLM calls are mocked via `unittest.mock.patch`.

---

## Directory layout

```
plan_execute_agent/
    __init__.py
    models.py
    planner.py
    executor.py
    tools.py
    agent.py
tests/
    __init__.py
    test_models.py
    test_planner.py
    test_executor.py
    test_agent.py
pyproject.toml
```

---

## `plan_execute_agent/__init__.py`

```python
# Expose the public API at package level.
from plan_execute_agent.agent import PlanExecuteAgent
from plan_execute_agent.models import ExecutionReport, Plan, StepResult

__all__ = ["PlanExecuteAgent", "Plan", "StepResult", "ExecutionReport"]
```

---

## `plan_execute_agent/models.py`

```python
"""
Pydantic v2 data models for the plan-and-execute agent.

Design decision: discriminated unions on `step_type` give us exhaustive
type-safe dispatch in the executor without isinstance chains.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Step types (discriminated union)
# ---------------------------------------------------------------------------

class ToolStep(BaseModel):
    """A step that calls a registered tool with explicit inputs."""
    step_type: Literal["tool"] = "tool"
    step_id: str
    tool: str
    input: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    max_retries: int = 1


class LLMStep(BaseModel):
    """A step that sends a prompt directly to the language model."""
    step_type: Literal["llm"] = "llm"
    step_id: str
    prompt: str
    depends_on: list[str] = Field(default_factory=list)
    max_retries: int = 1


class SubAgentStep(BaseModel):
    """A step that delegates to a child plan-and-execute agent."""
    step_type: Literal["subagent"] = "subagent"
    step_id: str
    sub_goal: str
    depends_on: list[str] = Field(default_factory=list)
    max_retries: int = 1


# Annotated discriminator so Pydantic can deserialise mixed lists correctly.
Step = Annotated[
    Union[ToolStep, LLMStep, SubAgentStep],
    Field(discriminator="step_type"),
]


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

class Plan(BaseModel):
    """An ordered (or dependency-linked) sequence of steps."""
    goal: str
    steps: list[Step]


# ---------------------------------------------------------------------------
# Results & Report
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    step_id: str
    tool: str                # for LLMStep this will be "llm"; SubAgentStep "subagent"
    status: Literal["success", "failed", "skipped"]
    output: str | None = None
    error: str | None = None
    attempt: int = 1


class ExecutionReport(BaseModel):
    goal: str
    steps: list[StepResult]
    replan_count: int
    success: bool

    def summary(self) -> str:
        succeeded = sum(1 for s in self.steps if s.status == "success")
        return (
            f"Goal: {self.goal}\n"
            f"Steps: {len(self.steps)} total, {succeeded} succeeded\n"
            f"Re-plans: {self.replan_count}\n"
            f"Success: {self.success}"
        )
```

---

## `plan_execute_agent/planner.py`

```python
"""
Planner: translates a high-level goal into a validated Plan.

The LLM interface is extracted into a Protocol so tests can inject a mock
without patching the Anthropic SDK globally.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Protocol, runtime_checkable

from plan_execute_agent.models import LLMStep, Plan, Step, ToolStep

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMInterface(Protocol):
    """Minimal interface for generating text — mockable in tests."""

    def generate(self, prompt: str) -> str:
        """Return the model's text response to `prompt`."""
        ...


class AnthropicLLM:
    """
    Production LLM backend using the Anthropic SDK.

    model is set to claude-sonnet-5-5 per course constraints.
    api_key is read from the environment — never hard-coded.
    """

    def __init__(self, model: str = "claude-sonnet-5-5") -> None:
        self._model = model
        self._api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self._api_key:
            logger.warning("ANTHROPIC_API_KEY not set; real LLM calls will fail.")

    def generate(self, prompt: str) -> str:
        import anthropic  # deferred import — not needed in test environments

        client = anthropic.Anthropic(api_key=self._api_key)
        message = client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text


_PLANNING_PROMPT_TEMPLATE = """
You are a planning assistant. Decompose the following goal into a sequence of steps.

GOAL: {goal}

Output a JSON object with the schema:
{{
  "goal": "<goal string>",
  "steps": [
    {{
      "step_type": "tool",
      "step_id": "step_1",
      "tool": "<tool_name>",
      "input": {{}},
      "depends_on": []
    }},
    ...
  ]
}}

Available tools: search, summarize, write_file
Output ONLY valid JSON — no prose, no markdown fences.
""".strip()


class Planner:
    """
    Converts a goal string into a Plan by calling an LLM.

    Design decision: the JSON parsing is done here, not in the LLM wrapper,
    so the Planner owns the contract between "raw text" and "typed Plan."
    """

    def __init__(self, llm: LLMInterface) -> None:
        self._llm = llm

    def decompose(self, goal: str) -> Plan:
        prompt = _PLANNING_PROMPT_TEMPLATE.format(goal=goal)
        logger.info("Planner: generating plan for goal=%r", goal)
        raw = self._llm.generate(prompt)
        return self._parse(raw, goal)

    def replan(
        self,
        goal: str,
        completed_ids: list[str],
        failed_step_id: str,
        error: str,
    ) -> Plan:
        """
        Generate a replacement plan tail starting after `completed_ids`.
        """
        prompt = (
            f"Goal: {goal}\n"
            f"Already completed: {completed_ids}\n"
            f"Failed step: {failed_step_id} — {error}\n"
            "Produce a revised JSON plan tail (same schema as before) "
            "containing only the remaining steps."
        )
        logger.warning(
            "Planner.replan: triggered by failed step %r — %s", failed_step_id, error
        )
        raw = self._llm.generate(prompt)
        return self._parse(raw, goal)

    @staticmethod
    def _parse(raw: str, goal: str) -> Plan:
        try:
            data: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Planner: LLM returned non-JSON: {exc}") from exc
        # Pydantic validates and coerces the discriminated union automatically.
        return Plan.model_validate(data)
```

---

## `plan_execute_agent/executor.py`

```python
"""
Executor: runs a Plan, handles retries, and fires the replan hook on failure.

Topological sort ensures dependency-respecting execution order.
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Callable

from plan_execute_agent.models import (
    ExecutionReport,
    LLMStep,
    Plan,
    StepResult,
    SubAgentStep,
    ToolStep,
)

logger = logging.getLogger(__name__)

# A replan hook receives (goal, completed_results, failed_result, error_str)
# and returns a new Plan tail.
ReplanHook = Callable[[str, list[StepResult], StepResult, str], Plan]


def _topological_sort(plan: Plan) -> list[str]:
    """
    Return step_ids in dependency-respecting execution order.

    Uses Kahn's algorithm (BFS-based). Raises ValueError for cycles.
    Design decision: fail fast at plan-validation time rather than at runtime
    so the agent never partially executes a cyclic plan.
    """
    # Build adjacency: edge A -> B means B depends on A.
    in_degree: dict[str, int] = {s.step_id: 0 for s in plan.steps}
    dependents: dict[str, list[str]] = defaultdict(list)

    for step in plan.steps:
        deps = (
            step.depends_on
            if isinstance(step, (ToolStep, LLMStep, SubAgentStep))
            else []
        )
        for dep in deps:
            dependents[dep].append(step.step_id)
            in_degree[step.step_id] += 1

    queue: deque[str] = deque(
        sid for sid, deg in in_degree.items() if deg == 0
    )
    order: list[str] = []

    while queue:
        sid = queue.popleft()
        order.append(sid)
        for child in dependents[sid]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(order) != len(plan.steps):
        raise ValueError(
            "Plan contains a dependency cycle — cannot execute. "
            "Cycle involves: "
            + str(set(in_degree) - set(order))
        )
    return order


class Executor:
    """
    Execute a Plan step by step.

    Retry logic: each ToolStep / LLMStep has a `max_retries` field. On
    exhausting retries the replan_hook (if provided) is called to obtain a
    replacement plan tail; if no hook is provided the executor aborts.
    """

    def __init__(
        self,
        tool_fns: dict[str, Callable[..., str]],
        llm_fn: Callable[[str], str] | None = None,
        replan_hook: ReplanHook | None = None,
        max_replans: int = 3,
    ) -> None:
        self._tools = tool_fns
        self._llm = llm_fn
        self._replan_hook = replan_hook
        self._max_replans = max_replans

    def run(self, plan: Plan) -> ExecutionReport:
        completed: list[StepResult] = []
        replan_count = 0
        current_plan = plan

        while True:
            try:
                order = _topological_sort(current_plan)
            except ValueError as exc:
                logger.error("Executor: invalid plan — %s", exc)
                return ExecutionReport(
                    goal=plan.goal,
                    steps=completed,
                    replan_count=replan_count,
                    success=False,
                )

            step_map = {s.step_id: s for s in current_plan.steps}
            failed_result: StepResult | None = None

            for sid in order:
                step = step_map[sid]
                result = self._execute_step(step)
                completed.append(result)

                if result.status == "failed":
                    failed_result = result
                    break  # Stop this plan; trigger replan or abort.

            if failed_result is None:
                # All steps completed successfully.
                return ExecutionReport(
                    goal=plan.goal,
                    steps=completed,
                    replan_count=replan_count,
                    success=True,
                )

            # A step failed — attempt replan.
            if self._replan_hook is None or replan_count >= self._max_replans:
                logger.error(
                    "Executor: aborting — no replan hook or max_replans reached."
                )
                return ExecutionReport(
                    goal=plan.goal,
                    steps=completed,
                    replan_count=replan_count,
                    success=False,
                )

            replan_count += 1
            logger.warning(
                "Executor: replan %d/%d triggered by step '%s'",
                replan_count, self._max_replans, failed_result.step_id,
            )
            current_plan = self._replan_hook(
                plan.goal,
                completed,
                failed_result,
                failed_result.error or "unknown error",
            )

    def _execute_step(self, step: ToolStep | LLMStep | SubAgentStep) -> StepResult:
        tool_label = (
            step.tool if isinstance(step, ToolStep) else
            "llm" if isinstance(step, LLMStep) else
            "subagent"
        )
        for attempt in range(1, getattr(step, "max_retries", 1) + 1):
            try:
                output = self._dispatch(step)
                return StepResult(
                    step_id=step.step_id,
                    tool=tool_label,
                    status="success",
                    output=output,
                    attempt=attempt,
                )
            except Exception as exc:
                logger.warning(
                    "Step '%s' attempt %d/%d failed: %s",
                    step.step_id, attempt, getattr(step, "max_retries", 1), exc,
                )
                last_exc = exc

        return StepResult(
            step_id=step.step_id,
            tool=tool_label,
            status="failed",
            error=str(last_exc),
            attempt=getattr(step, "max_retries", 1),
        )

    def _dispatch(self, step: ToolStep | LLMStep | SubAgentStep) -> str:
        if isinstance(step, ToolStep):
            fn = self._tools.get(step.tool)
            if fn is None:
                raise KeyError(f"No tool registered for '{step.tool}'")
            return fn(**step.input)
        if isinstance(step, LLMStep):
            if self._llm is None:
                raise RuntimeError("No LLM function provided to Executor")
            return self._llm(step.prompt)
        if isinstance(step, SubAgentStep):
            # Simplified: treat sub-goal as an LLM call in this reference impl.
            if self._llm is None:
                raise RuntimeError("No LLM function provided for SubAgentStep")
            return self._llm(f"Sub-goal: {step.sub_goal}")
        raise TypeError(f"Unknown step type: {type(step)}")
```

---

## `plan_execute_agent/tools.py`

```python
"""
Tool implementations — all deterministic and mock-able.

ToolRegistry: maps tool names to callables. Keeping registration
centralised here means the executor stays tool-agnostic.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

ToolFn = Callable[..., str]


class ToolRegistry:
    """Simple name -> callable registry with validation on registration."""

    def __init__(self) -> None:
        self._registry: dict[str, ToolFn] = {}

    def register(self, name: str, fn: ToolFn) -> None:
        if name in self._registry:
            logger.warning("ToolRegistry: overwriting existing tool '%s'", name)
        self._registry[name] = fn
        logger.debug("ToolRegistry: registered tool '%s'", name)

    def get(self, name: str) -> ToolFn | None:
        return self._registry.get(name)

    def as_dict(self) -> dict[str, ToolFn]:
        return dict(self._registry)

    def __contains__(self, name: str) -> bool:
        return name in self._registry


# ---------------------------------------------------------------------------
# Built-in tool implementations
# ---------------------------------------------------------------------------

def search(query: str, max_results: int = 5) -> str:
    """
    Deterministic stub search tool.

    In production this would call a real search API.
    Returns a predictable string so tests do not need network access.
    """
    logger.info("tool:search query=%r max_results=%d", query, max_results)
    return f"[SEARCH RESULTS for '{query}': result_1, result_2, result_3]"


def write_file(path: str, content: str) -> str:
    """
    Write `content` to `path` and return a confirmation string.

    Uses pathlib.Path; creates parent directories automatically.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    logger.info("tool:write_file written %d bytes to %s", len(content), dest)
    return f"Wrote {len(content)} bytes to {dest}"


def summarize(text: str, max_words: int = 100) -> str:
    """
    Deterministic stub summarizer.

    In production this would call an LLM summarizer endpoint.
    """
    logger.info("tool:summarize len=%d max_words=%d", len(text), max_words)
    words = text.split()
    truncated = " ".join(words[:max_words])
    suffix = "..." if len(words) > max_words else ""
    return f"[SUMMARY]: {truncated}{suffix}"


def build_default_registry() -> ToolRegistry:
    """Return a ToolRegistry pre-loaded with the built-in tools."""
    registry = ToolRegistry()
    registry.register("search", search)
    registry.register("write_file", write_file)
    registry.register("summarize", summarize)
    return registry
```

---

## `plan_execute_agent/agent.py`

```python
"""
PlanExecuteAgent: top-level orchestrator.

Design decision: the agent owns the retry/replan policy at a high level
while delegating the mechanics to Planner and Executor. This keeps the
agent thin — it wires components together but contains no business logic.
"""
from __future__ import annotations

import logging

from plan_execute_agent.executor import Executor, ReplanHook
from plan_execute_agent.models import ExecutionReport, Plan, StepResult
from plan_execute_agent.planner import LLMInterface, Planner
from plan_execute_agent.tools import ToolRegistry, build_default_registry

logger = logging.getLogger(__name__)


class PlanExecuteAgent:
    """
    Orchestrates Planner + Executor for a given goal.

    Parameters
    ----------
    llm:
        Any object satisfying LLMInterface (AnthropicLLM in production,
        a mock in tests).
    registry:
        ToolRegistry supplying callable tools. Defaults to the built-in set.
    max_replans:
        Hard cap on how many times the executor may trigger a replan before
        aborting. Guards against infinite loops.
    """

    def __init__(
        self,
        llm: LLMInterface,
        registry: ToolRegistry | None = None,
        max_replans: int = 3,
    ) -> None:
        self._planner = Planner(llm)
        self._registry = registry or build_default_registry()
        self._max_replans = max_replans

    def run(self, goal: str) -> ExecutionReport:
        logger.info("PlanExecuteAgent: starting goal=%r", goal)

        plan: Plan = self._planner.decompose(goal)
        logger.info(
            "PlanExecuteAgent: plan has %d steps", len(plan.steps)
        )

        def replan_hook(
            g: str,
            completed: list[StepResult],
            failed: StepResult,
            error: str,
        ) -> Plan:
            completed_ids = [r.step_id for r in completed if r.status == "success"]
            return self._planner.replan(g, completed_ids, failed.step_id, error)

        executor = Executor(
            tool_fns=self._registry.as_dict(),
            replan_hook=replan_hook,
            max_replans=self._max_replans,
        )

        report = executor.run(plan)
        logger.info(
            "PlanExecuteAgent: finished success=%s replan_count=%d",
            report.success, report.replan_count,
        )
        return report
```

---

## `tests/test_models.py`

```python
"""Unit tests for Pydantic models."""
from __future__ import annotations

import json

import pytest

from plan_execute_agent.models import (
    ExecutionReport,
    LLMStep,
    Plan,
    StepResult,
    SubAgentStep,
    ToolStep,
)


class TestToolStep:
    def test_defaults(self) -> None:
        s = ToolStep(step_id="s1", tool="search")
        assert s.step_type == "tool"
        assert s.input == {}
        assert s.depends_on == []
        assert s.max_retries == 1

    def test_with_input(self) -> None:
        s = ToolStep(step_id="s1", tool="search", input={"query": "foo"})
        assert s.input == {"query": "foo"}


class TestLLMStep:
    def test_defaults(self) -> None:
        s = LLMStep(step_id="l1", prompt="Summarise this")
        assert s.step_type == "llm"
        assert s.depends_on == []


class TestSubAgentStep:
    def test_defaults(self) -> None:
        s = SubAgentStep(step_id="a1", sub_goal="research X")
        assert s.step_type == "subagent"


class TestPlanDiscriminatedUnion:
    def test_mixed_step_list(self) -> None:
        data = {
            "goal": "do stuff",
            "steps": [
                {"step_type": "tool", "step_id": "s1", "tool": "search"},
                {"step_type": "llm",  "step_id": "s2", "prompt": "summarise"},
                {"step_type": "subagent", "step_id": "s3", "sub_goal": "sub"},
            ],
        }
        plan = Plan.model_validate(data)
        assert len(plan.steps) == 3
        assert isinstance(plan.steps[0], ToolStep)
        assert isinstance(plan.steps[1], LLMStep)
        assert isinstance(plan.steps[2], SubAgentStep)

    def test_invalid_discriminator(self) -> None:
        with pytest.raises(Exception):
            Plan.model_validate({
                "goal": "g",
                "steps": [{"step_type": "unknown", "step_id": "x"}],
            })


class TestExecutionReport:
    def test_serialises_to_json(self) -> None:
        report = ExecutionReport(
            goal="test",
            steps=[StepResult(step_id="s1", tool="search", status="success")],
            replan_count=0,
            success=True,
        )
        parsed = json.loads(report.model_dump_json())
        assert parsed["success"] is True
        assert parsed["replan_count"] == 0

    def test_summary_method(self) -> None:
        report = ExecutionReport(
            goal="my goal",
            steps=[StepResult(step_id="s1", tool="search", status="success")],
            replan_count=1,
            success=True,
        )
        summary = report.summary()
        assert "my goal" in summary
        assert "Re-plans: 1" in summary
```

---

## `tests/test_planner.py`

```python
"""Unit tests for Planner — all LLM calls mocked."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from plan_execute_agent.models import Plan, ToolStep
from plan_execute_agent.planner import Planner


def _make_plan_json(goal: str = "test goal") -> str:
    return json.dumps({
        "goal": goal,
        "steps": [
            {"step_type": "tool", "step_id": "s1", "tool": "search",
             "input": {"query": "test"}, "depends_on": []},
            {"step_type": "tool", "step_id": "s2", "tool": "summarize",
             "input": {"text": "results"}, "depends_on": ["s1"]},
            {"step_type": "tool", "step_id": "s3", "tool": "write_file",
             "input": {"path": "out.txt", "content": "done"}, "depends_on": ["s2"]},
        ],
    })


@pytest.fixture()
def mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.generate.return_value = _make_plan_json()
    return llm


@pytest.fixture()
def planner(mock_llm: MagicMock) -> Planner:
    return Planner(llm=mock_llm)


class TestPlannerDecompose:
    def test_returns_plan_instance(self, planner: Planner) -> None:
        plan = planner.decompose("Research AI history")
        assert isinstance(plan, Plan)

    def test_step_count(self, planner: Planner) -> None:
        plan = planner.decompose("Research AI history")
        assert len(plan.steps) == 3

    def test_step_types(self, planner: Planner) -> None:
        plan = planner.decompose("goal")
        assert all(isinstance(s, ToolStep) for s in plan.steps)

    def test_llm_called_once(self, planner: Planner, mock_llm: MagicMock) -> None:
        planner.decompose("goal")
        mock_llm.generate.assert_called_once()

    def test_invalid_json_raises(self, mock_llm: MagicMock) -> None:
        mock_llm.generate.return_value = "not json"
        planner = Planner(llm=mock_llm)
        with pytest.raises(ValueError, match="non-JSON"):
            planner.decompose("goal")


class TestPlannerReplan:
    def test_replan_returns_plan(self, mock_llm: MagicMock) -> None:
        mock_llm.generate.return_value = json.dumps({
            "goal": "goal",
            "steps": [
                {"step_type": "tool", "step_id": "s2_retry", "tool": "summarize",
                 "input": {}, "depends_on": []},
            ],
        })
        planner = Planner(llm=mock_llm)
        plan = planner.replan("goal", ["s1"], "s2", "TimeoutError")
        assert isinstance(plan, Plan)
        assert plan.steps[0].step_id == "s2_retry"  # type: ignore[union-attr]
```

---

## `tests/test_executor.py`

```python
"""Unit tests for Executor and topological sort."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from plan_execute_agent.executor import Executor, _topological_sort
from plan_execute_agent.models import (
    ExecutionReport, LLMStep, Plan, StepResult, ToolStep,
)


def _make_plan(*steps: ToolStep | LLMStep) -> Plan:
    return Plan(goal="test goal", steps=list(steps))


def _tool_step(
    sid: str,
    tool: str = "search",
    depends_on: list[str] | None = None,
    max_retries: int = 1,
) -> ToolStep:
    return ToolStep(
        step_id=sid, tool=tool,
        depends_on=depends_on or [],
        max_retries=max_retries,
    )


class TestTopologicalSort:
    def test_linear_chain(self) -> None:
        plan = _make_plan(
            _tool_step("s1"),
            _tool_step("s2", depends_on=["s1"]),
            _tool_step("s3", depends_on=["s2"]),
        )
        order = _topological_sort(plan)
        assert order.index("s1") < order.index("s2") < order.index("s3")

    def test_no_dependencies(self) -> None:
        plan = _make_plan(_tool_step("a"), _tool_step("b"), _tool_step("c"))
        order = _topological_sort(plan)
        assert set(order) == {"a", "b", "c"}

    def test_cycle_raises(self) -> None:
        plan = _make_plan(
            _tool_step("s1", depends_on=["s2"]),
            _tool_step("s2", depends_on=["s1"]),
        )
        with pytest.raises(ValueError, match="cycle"):
            _topological_sort(plan)


class TestExecutorSuccess:
    def test_all_steps_succeed(self) -> None:
        executor = Executor(tool_fns={"search": lambda **_: "results"})
        plan = _make_plan(_tool_step("s1", tool="search"))
        report = executor.run(plan)
        assert report.success is True
        assert len(report.steps) == 1
        assert report.steps[0].status == "success"

    def test_report_type(self) -> None:
        executor = Executor(tool_fns={"search": lambda **_: "r"})
        plan = _make_plan(_tool_step("s1", tool="search"))
        assert isinstance(executor.run(plan), ExecutionReport)


class TestExecutorRetry:
    def test_step_retried_on_failure(self) -> None:
        call_count = {"n": 0}

        def flaky(**kwargs: object) -> str:
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise ConnectionError("transient")
            return "ok"

        executor = Executor(tool_fns={"search": flaky})
        plan = _make_plan(_tool_step("s1", tool="search", max_retries=2))
        report = executor.run(plan)
        assert report.success is True
        assert call_count["n"] == 2


class TestExecutorReplan:
    def test_replan_hook_called_on_failure(self) -> None:
        replan_mock = MagicMock()
        # Return a simple one-step replacement plan.
        replan_mock.return_value = Plan(
            goal="test goal",
            steps=[_tool_step("s1_retry", tool="ok")],
        )
        executor = Executor(
            tool_fns={
                "fail": lambda **_: (_ for _ in ()).throw(ValueError("bad")),
                "ok":   lambda **_: "done",
            },
            replan_hook=replan_mock,
            max_replans=3,
        )
        plan = _make_plan(_tool_step("s1", tool="fail"))
        report = executor.run(plan)
        replan_mock.assert_called_once()
        assert report.replan_count == 1
        assert report.success is True

    def test_abort_when_no_replan_hook(self) -> None:
        executor = Executor(
            tool_fns={"fail": lambda **_: (_ for _ in ()).throw(RuntimeError("x"))},
        )
        plan = _make_plan(_tool_step("s1", tool="fail"))
        report = executor.run(plan)
        assert report.success is False
        assert report.replan_count == 0

    def test_max_replans_guard(self) -> None:
        replan_mock = MagicMock(return_value=_make_plan(_tool_step("retry", tool="fail")))
        executor = Executor(
            tool_fns={"fail": lambda **_: (_ for _ in ()).throw(ValueError("x"))},
            replan_hook=replan_mock,
            max_replans=2,
        )
        plan = _make_plan(_tool_step("s1", tool="fail"))
        report = executor.run(plan)
        assert report.replan_count == 2
        assert report.success is False
```

---

## `tests/test_agent.py`

```python
"""Integration tests for PlanExecuteAgent — LLM fully mocked."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from plan_execute_agent.agent import PlanExecuteAgent
from plan_execute_agent.models import ExecutionReport
from plan_execute_agent.tools import ToolRegistry


def _three_step_plan_json(goal: str = "test") -> str:
    return json.dumps({
        "goal": goal,
        "steps": [
            {"step_type": "tool", "step_id": "s1", "tool": "search",
             "input": {"query": "q"}, "depends_on": []},
            {"step_type": "tool", "step_id": "s2", "tool": "summarize",
             "input": {"text": "t"}, "depends_on": ["s1"]},
            {"step_type": "tool", "step_id": "s3", "tool": "write_file",
             "input": {"path": "/tmp/out.txt", "content": "c"}, "depends_on": ["s2"]},
        ],
    })


@pytest.fixture()
def mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.generate.return_value = _three_step_plan_json()
    return llm


@pytest.fixture()
def agent(mock_llm: MagicMock) -> PlanExecuteAgent:
    return PlanExecuteAgent(llm=mock_llm)


class TestPlanExecuteAgentHappyPath:
    def test_returns_execution_report(self, agent: PlanExecuteAgent) -> None:
        report = agent.run("Research AI history")
        assert isinstance(report, ExecutionReport)

    def test_success_true(self, agent: PlanExecuteAgent) -> None:
        report = agent.run("Research AI history")
        assert report.success is True

    def test_three_steps_completed(self, agent: PlanExecuteAgent) -> None:
        report = agent.run("Research AI history")
        assert len(report.steps) == 3

    def test_no_replans(self, agent: PlanExecuteAgent) -> None:
        report = agent.run("Research AI history")
        assert report.replan_count == 0

    def test_report_serialises(self, agent: PlanExecuteAgent) -> None:
        report = agent.run("goal")
        data = json.loads(report.model_dump_json())
        assert "success" in data


class TestPlanExecuteAgentWithReplan:
    def test_replan_triggered_on_tool_failure(
        self, mock_llm: MagicMock
    ) -> None:
        call_state = {"n": 0}

        def flaky_summarize(**kwargs: object) -> str:
            call_state["n"] += 1
            if call_state["n"] == 1:
                raise ValueError("bad input")
            return "summary"

        # Second LLM call (replan) returns a two-step tail.
        replan_json = json.dumps({
            "goal": "test",
            "steps": [
                {"step_type": "tool", "step_id": "s2_retry", "tool": "summarize",
                 "input": {"text": "fixed"}, "depends_on": []},
                {"step_type": "tool", "step_id": "s3", "tool": "write_file",
                 "input": {"path": "/tmp/out.txt", "content": "done"}, "depends_on": []},
            ],
        })
        mock_llm.generate.side_effect = [
            _three_step_plan_json(),
            replan_json,
        ]

        registry = ToolRegistry()
        registry.register("search", lambda **_: "results")
        registry.register("summarize", flaky_summarize)
        registry.register("write_file", lambda **_: "written")

        agent = PlanExecuteAgent(llm=mock_llm, registry=registry)
        report = agent.run("test goal")
        assert report.replan_count == 1
        assert report.success is True
```

---

## `pyproject.toml` (minimal)

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "plan-execute-agent"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.0",
    "anthropic>=0.25",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.4"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
target-version = "py312"
line-length = 100
```

---

## Running the tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
# Expected: all tests pass; no network calls made.
```
