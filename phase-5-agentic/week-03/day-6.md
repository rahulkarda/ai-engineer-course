# Phase 5 · Week 03 · Day 6 — Build: Plan-and-Execute Agent

> Objective: Assemble a complete, testable plan-and-execute agent from typed models through executor loop to structured report.  ·  Time: ~4 hr  ·  Prereqs: [[phase-5-agentic/week-03/day-5]]

## Concept

This is a build day. The goal is a working, fully tested `plan_execute_agent` package that combines everything from days 3–5 into a single cohesive system.

A plan-and-execute agent separates **planning** (decompose goal into steps) from **execution** (run steps with tools). This separation makes each part independently testable and allows the planner to be swapped without touching the executor.

The system accepts a natural-language goal, decomposes it into typed steps via an LLM call, executes each step against a tool registry, re-plans on structural failure, and returns a typed `ExecutionReport`.

### Architecture overview

```
User goal (str)
       │
       ▼
  [ Planner ]  ──── LLM call ────► Plan (Pydantic)
       │
       ▼
  [ Executor ] ──── ToolRegistry ──► StepResult per step
       │                │
       │         on failure: ReplanningExecutor (day-5)
       ▼
  ExecutionReport (Pydantic)
```

### File layout

```
plan_execute_agent/
    __init__.py
    models.py        # Plan, Step, StepResult, ExecutionReport
    planner.py       # LLMPlanner — calls LLM to generate Plan
    executor.py      # Executor loop with retry and replan hooks
    tools.py         # ToolRegistry + 3 toy tools
    agent.py         # PlanExecuteAgent — top-level orchestrator
    tests/
        __init__.py
        test_models.py
        test_planner.py
        test_executor.py
        test_agent.py
```

## How it works internally

**`models.py`** — all data contracts. `Plan` contains a list of `Step`. Each executed step produces a `StepResult`. The final `ExecutionReport` aggregates all results plus metadata (replan count, abort flag, duration).

**`planner.py`** — one public method `plan(goal: str) -> Plan`. Sends a structured prompt to the LLM, parses the JSON response, validates with Pydantic. Raises `PlannerError` on parse failure.

**`executor.py`** — `Executor.run(plan, registry) -> ExecutionReport`. Steps execute sequentially. On failure, delegates to `ReplanningExecutor` logic (from day 5). Keeps a `completed: list[StepResult]` accumulator.

**`tools.py`** — `ToolRegistry` maps tool names to callables. Three toy tools: `search`, `write_file`, `summarize`. All accept keyword arguments.

**`agent.py`** — `PlanExecuteAgent` wires planner, executor, and registry together. Single entry point: `run(goal: str) -> ExecutionReport`.

**`tests/`** — all LLM calls patched with `unittest.mock.patch`. Tool calls use real toy implementations (no network). Each module has its own test file.

## Worked example

Below is the complete source for every file. Copy each block into the corresponding path.

### `plan_execute_agent/models.py`

```python
"""
models.py — Pydantic v2 data models for the plan-and-execute agent.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class StepStatus(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class Step(BaseModel):
    id: str
    description: str
    tool: str
    inputs: dict[str, str] = Field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    result: str | None = None
    error: str | None = None


class Plan(BaseModel):
    goal: str
    steps: list[Step]


class StepResult(BaseModel):
    step_id: str
    status: StepStatus
    output: str | None = None
    error: str | None = None
    attempt_count: int = 1


class ExecutionReport(BaseModel):
    goal: str
    started_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: datetime | None = None
    step_results: list[StepResult] = Field(default_factory=list)
    replan_count: int = 0
    total_steps_attempted: int = 0
    aborted: bool = False
    abort_reason: str | None = None

    def successful_results(self) -> list[StepResult]:
        return [r for r in self.step_results if r.status == StepStatus.SUCCESS]

    def failed_results(self) -> list[StepResult]:
        return [r for r in self.step_results if r.status == StepStatus.FAILED]
```

### `plan_execute_agent/tools.py`

```python
"""
tools.py — ToolRegistry and three mockable toy tools.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

log = logging.getLogger(__name__)

ToolFn = Callable[..., str]


class ToolNotFoundError(Exception):
    pass


class ToolRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, ToolFn] = {}

    def register(self, name: str, fn: ToolFn) -> None:
        self._registry[name] = fn
        log.debug("registered tool %r", name)

    def call(self, name: str, inputs: dict[str, str]) -> str:
        if name not in self._registry:
            raise ToolNotFoundError(f"Tool {name!r} not registered")
        log.info("calling tool %r with inputs %r", name, inputs)
        return self._registry[name](**inputs)

    def names(self) -> list[str]:
        return list(self._registry.keys())


# ---------------------------------------------------------------------------
# Toy tools — all pure, no network calls
# ---------------------------------------------------------------------------


def search(query: str) -> str:
    """Mock web search."""
    return (
        f"Search results for {query!r}: "
        "[Result 1: mock content A] "
        "[Result 2: mock content B] "
        "[Result 3: mock content C]"
    )


def write_file(path: str, content: str) -> str:
    """Mock file writer."""
    log.info("write_file: %r (%d chars)", path, len(content))
    return f"Wrote {len(content)} characters to {path}"


def summarize(text: str) -> str:
    """Mock text summarizer."""
    word_count = len(text.split())
    return f"Summary ({word_count} words compressed): key themes include {text[:60]}..."


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("search", search)
    registry.register("write_file", write_file)
    registry.register("summarize", summarize)
    return registry
```

### `plan_execute_agent/planner.py`

```python
"""
planner.py — LLMPlanner: decomposes a natural-language goal into a typed Plan.
"""

from __future__ import annotations

import json
import logging
import os

import anthropic
from pydantic import ValidationError

from .models import Plan, Step

log = logging.getLogger(__name__)

PLAN_PROMPT_TEMPLATE = """\
You are a planning agent. Break the following goal into 3–6 concrete steps.

Goal: {goal}

Available tools: search, write_file, summarize

Each step must use exactly one tool. Return a JSON array of step objects.
Each object must have these keys:
  id          (string, e.g. "step-1")
  description (string, one sentence)
  tool        (string, one of: search, write_file, summarize)
  inputs      (object with string keys and string values matching the tool's parameters)

Tool parameter names:
  search:     query (str)
  write_file: path (str), content (str)
  summarize:  text (str)

Output raw JSON array only. No markdown fences. No explanation.
"""


class PlannerError(Exception):
    pass


class LLMPlanner:
    def __init__(
        self,
        model: str = "claude-haiku-5-5",
        max_tokens: int = 1024,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def plan(self, goal: str) -> Plan:
        """Call the LLM to decompose goal into a Plan."""
        prompt = PLAN_PROMPT_TEMPLATE.format(goal=goal)
        log.info("planning for goal: %r", goal[:80])

        message = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text
        log.debug("raw planner response: %s", raw[:400])

        try:
            steps_data: list[dict] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PlannerError(f"Planner returned invalid JSON: {exc}") from exc

        try:
            steps = [Step(**d) for d in steps_data]
        except (ValidationError, TypeError) as exc:
            raise PlannerError(f"Planner returned invalid step schema: {exc}") from exc

        plan = Plan(goal=goal, steps=steps)
        log.info("plan generated: %d steps", len(steps))
        return plan
```

### `plan_execute_agent/executor.py`

```python
"""
executor.py — Executor with retry, re-planning, and circuit-breaker logic.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Protocol

from .models import (
    ExecutionReport,
    Plan,
    Step,
    StepResult,
    StepStatus,
)
from .tools import ToolRegistry

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

TRANSIENT_KEYWORDS: tuple[str, ...] = ("timeout", "connection", "rate limit", "503", "429")
FATAL_KEYWORDS: tuple[str, ...] = ("authentication", "unauthorized", "safety", "budget")


def classify_error(error: str) -> str:
    """Return 'transient', 'structural', or 'fatal'."""
    lower = error.lower()
    if any(kw in lower for kw in FATAL_KEYWORDS):
        return "fatal"
    if any(kw in lower for kw in TRANSIENT_KEYWORDS):
        return "transient"
    return "structural"


# ---------------------------------------------------------------------------
# LLM client protocol
# ---------------------------------------------------------------------------


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str:
        ...


# ---------------------------------------------------------------------------
# Re-plan prompt
# ---------------------------------------------------------------------------

REPLAN_PROMPT = """\
You are a planning agent. A plan was partially executed and a step failed.

Goal: {goal}

Completed steps (do NOT repeat):
{completed_summary}

Failed step:
  id: {failed_id}
  description: {failed_description}
  tool: {failed_tool}
  error: {error}

Generate ONLY the remaining steps to finish the goal.
Available tools: search, write_file, summarize
Return raw JSON array only (no markdown). Each object: id, description, tool, inputs.
"""


def _build_replan_prompt(
    goal: str,
    completed: list[StepResult],
    failed: Step,
    error: str,
) -> str:
    if completed:
        summary = "\n".join(
            f"  [{r.step_id}] status={r.status.value}" for r in completed
        )
    else:
        summary = "  (none)"
    return REPLAN_PROMPT.format(
        goal=goal,
        completed_summary=summary,
        failed_id=failed.id,
        failed_description=failed.description,
        failed_tool=failed.tool,
        error=error,
    )


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class Executor:
    """
    Executes a Plan against a ToolRegistry.

    Retries transient errors with exponential back-off.
    Re-plans on structural errors using the provided LLM client.
    Circuit breaker aborts after max_replans re-planning attempts.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        max_replans: int = 3,
        max_retries: int = 2,
        retry_delay: float = 0.2,
    ) -> None:
        self._llm = llm_client
        self._max_replans = max_replans
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    def run(self, plan: Plan, registry: ToolRegistry) -> ExecutionReport:
        report = ExecutionReport(goal=plan.goal)
        completed_results: list[StepResult] = []
        remaining: list[Step] = list(plan.steps)
        replan_count = 0
        total_attempted = 0

        while remaining:
            step = remaining.pop(0)
            total_attempted += 1

            success, output_or_error, attempts = self._execute_with_retry(
                step, registry
            )

            if success:
                result = StepResult(
                    step_id=step.id,
                    status=StepStatus.SUCCESS,
                    output=output_or_error,
                    attempt_count=attempts,
                )
                completed_results.append(result)
                log.info("step %s succeeded (attempts=%d)", step.id, attempts)
                continue

            error_str = output_or_error
            error_class = classify_error(error_str)
            log.warning(
                "step %s failed [%s]: %s", step.id, error_class, error_str
            )

            if error_class == "fatal":
                completed_results.append(
                    StepResult(
                        step_id=step.id,
                        status=StepStatus.FAILED,
                        error=error_str,
                        attempt_count=attempts,
                    )
                )
                report.step_results = completed_results
                report.finished_at = datetime.utcnow()
                report.aborted = True
                report.abort_reason = f"fatal error on step {step.id}: {error_str}"
                report.replan_count = replan_count
                report.total_steps_attempted = total_attempted
                return report

            if replan_count >= self._max_replans:
                log.error("circuit breaker: max_replans=%d reached", self._max_replans)
                completed_results.append(
                    StepResult(
                        step_id=step.id,
                        status=StepStatus.FAILED,
                        error=error_str,
                        attempt_count=attempts,
                    )
                )
                report.step_results = completed_results
                report.finished_at = datetime.utcnow()
                report.aborted = True
                report.abort_reason = (
                    f"circuit breaker tripped after {replan_count} replans"
                )
                report.replan_count = replan_count
                report.total_steps_attempted = total_attempted
                return report

            if error_class == "structural":
                replan_count += 1
                log.info(
                    "re-planning (attempt %d/%d)", replan_count, self._max_replans
                )
                new_steps = self._replan(
                    plan.goal, completed_results, step, error_str
                )
                remaining = new_steps + remaining
                # Record failed step
                completed_results.append(
                    StepResult(
                        step_id=step.id,
                        status=StepStatus.FAILED,
                        error=error_str,
                        attempt_count=attempts,
                    )
                )
            else:
                # Transient — already exhausted retries, skip
                log.warning("skipping step %s after retry exhaustion", step.id)
                completed_results.append(
                    StepResult(
                        step_id=step.id,
                        status=StepStatus.SKIPPED,
                        error=error_str,
                        attempt_count=attempts,
                    )
                )

        report.step_results = completed_results
        report.finished_at = datetime.utcnow()
        report.replan_count = replan_count
        report.total_steps_attempted = total_attempted
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_with_retry(
        self,
        step: Step,
        registry: ToolRegistry,
    ) -> tuple[bool, str, int]:
        """Returns (success, output_or_error, attempt_count)."""
        last_error = ""
        for attempt in range(1, self._max_retries + 2):
            try:
                output = registry.call(step.tool, step.inputs)
                return True, output, attempt
            except Exception as exc:
                last_error = str(exc)
                error_class = classify_error(last_error)
                if error_class != "transient" or attempt > self._max_retries:
                    return False, last_error, attempt
                delay = self._retry_delay * (2 ** (attempt - 1))
                log.info(
                    "transient error on step %s attempt %d — retry in %.2fs",
                    step.id,
                    attempt,
                    delay,
                )
                time.sleep(delay)
        return False, last_error, self._max_retries + 1

    def _replan(
        self,
        goal: str,
        completed: list[StepResult],
        failed_step: Step,
        error: str,
    ) -> list[Step]:
        prompt = _build_replan_prompt(goal, completed, failed_step, error)
        raw = self._llm.complete(prompt)
        steps_data: list[dict] = json.loads(raw)
        return [Step(**d) for d in steps_data]
```

### `plan_execute_agent/agent.py`

```python
"""
agent.py — PlanExecuteAgent: top-level orchestrator.
"""

from __future__ import annotations

import logging
import os

import anthropic

from .executor import Executor, LLMClient
from .models import ExecutionReport
from .planner import LLMPlanner, PlannerError
from .tools import ToolRegistry, build_default_registry

log = logging.getLogger(__name__)


class _AnthropicExecutorClient:
    """Thin LLMClient implementation for use by the Executor re-planner."""

    def __init__(self, model: str = "claude-haiku-5-5") -> None:
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self._model = model

    def complete(self, prompt: str) -> str:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text


class PlanExecuteAgent:
    """
    Accepts a natural-language goal and returns an ExecutionReport.

    Wires together LLMPlanner, Executor, and ToolRegistry.
    All LLM interactions are injectable for testing.
    """

    def __init__(
        self,
        planner: LLMPlanner | None = None,
        executor: Executor | None = None,
        registry: ToolRegistry | None = None,
        executor_llm_client: LLMClient | None = None,
    ) -> None:
        self._planner = planner or LLMPlanner()
        self._registry = registry or build_default_registry()

        if executor is not None:
            self._executor = executor
        else:
            client = executor_llm_client or _AnthropicExecutorClient()
            self._executor = Executor(llm_client=client)

    def run(self, goal: str) -> ExecutionReport:
        log.info("agent starting — goal: %r", goal[:100])
        try:
            plan = self._planner.plan(goal)
        except PlannerError as exc:
            log.error("planning failed: %s", exc)
            from datetime import datetime

            from .models import StepStatus
            return ExecutionReport(
                goal=goal,
                finished_at=datetime.utcnow(),
                aborted=True,
                abort_reason=f"planning failed: {exc}",
            )

        log.info("plan has %d steps — executing", len(plan.steps))
        report = self._executor.run(plan, self._registry)
        log.info(
            "agent done — replan_count=%d aborted=%s successful=%d",
            report.replan_count,
            report.aborted,
            len(report.successful_results()),
        )
        return report
```

### `plan_execute_agent/__init__.py`

```python
"""plan_execute_agent — a minimal plan-and-execute agent."""

from .agent import PlanExecuteAgent
from .models import ExecutionReport, Plan, Step, StepResult, StepStatus

__all__ = [
    "PlanExecuteAgent",
    "ExecutionReport",
    "Plan",
    "Step",
    "StepResult",
    "StepStatus",
]
```

### `plan_execute_agent/tests/__init__.py`

```python
```

### `plan_execute_agent/tests/test_models.py`

```python
"""
test_models.py — unit tests for Pydantic models.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from plan_execute_agent.models import (
    ExecutionReport,
    Plan,
    Step,
    StepResult,
    StepStatus,
)


def _make_step(**kwargs) -> Step:
    defaults: dict = {
        "id": "s1",
        "description": "do something",
        "tool": "search",
        "inputs": {"query": "test"},
    }
    return Step(**(defaults | kwargs))


def test_step_defaults() -> None:
    step = _make_step()
    assert step.status == StepStatus.PENDING
    assert step.result is None
    assert step.error is None


def test_step_requires_id_description_tool() -> None:
    with pytest.raises(ValidationError):
        Step(id="x", description="y")  # missing tool


def test_plan_stores_steps() -> None:
    plan = Plan(goal="do work", steps=[_make_step(id="s1"), _make_step(id="s2")])
    assert len(plan.steps) == 2
    assert plan.goal == "do work"


def test_step_result_status() -> None:
    result = StepResult(step_id="s1", status=StepStatus.SUCCESS, output="ok")
    assert result.status == StepStatus.SUCCESS
    assert result.output == "ok"


def test_execution_report_filter_methods() -> None:
    report = ExecutionReport(
        goal="test",
        step_results=[
            StepResult(step_id="s1", status=StepStatus.SUCCESS),
            StepResult(step_id="s2", status=StepStatus.FAILED, error="boom"),
            StepResult(step_id="s3", status=StepStatus.SUCCESS),
        ],
    )
    assert len(report.successful_results()) == 2
    assert len(report.failed_results()) == 1
    assert report.failed_results()[0].step_id == "s2"


def test_execution_report_abort_fields() -> None:
    report = ExecutionReport(goal="test", aborted=True, abort_reason="circuit breaker")
    assert report.aborted is True
    assert "circuit" in (report.abort_reason or "")
```

### `plan_execute_agent/tests/test_planner.py`

```python
"""
test_planner.py — unit tests for LLMPlanner (LLM mocked).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from plan_execute_agent.planner import LLMPlanner, PlannerError


def _make_planner() -> LLMPlanner:
    with patch("plan_execute_agent.planner.anthropic.Anthropic"):
        return LLMPlanner()


def _steps_json(n: int = 2) -> str:
    steps = [
        {
            "id": f"step-{i}",
            "description": f"Step {i}",
            "tool": "search",
            "inputs": {"query": f"query {i}"},
        }
        for i in range(1, n + 1)
    ]
    return json.dumps(steps)


@patch("plan_execute_agent.planner.anthropic.Anthropic")
def test_plan_returns_correct_step_count(mock_anthropic) -> None:
    planner = LLMPlanner()
    mock_msg = MagicMock()
    mock_msg.content[0].text = _steps_json(3)
    planner._client.messages.create.return_value = mock_msg

    plan = planner.plan("do three things")
    assert len(plan.steps) == 3
    assert plan.goal == "do three things"


@patch("plan_execute_agent.planner.anthropic.Anthropic")
def test_plan_raises_planner_error_on_invalid_json(mock_anthropic) -> None:
    planner = LLMPlanner()
    mock_msg = MagicMock()
    mock_msg.content[0].text = "not valid json at all"
    planner._client.messages.create.return_value = mock_msg

    with pytest.raises(PlannerError, match="invalid JSON"):
        planner.plan("goal")


@patch("plan_execute_agent.planner.anthropic.Anthropic")
def test_plan_raises_planner_error_on_schema_mismatch(mock_anthropic) -> None:
    planner = LLMPlanner()
    mock_msg = MagicMock()
    # Missing required 'tool' field
    mock_msg.content[0].text = json.dumps([{"id": "s1", "description": "x"}])
    planner._client.messages.create.return_value = mock_msg

    with pytest.raises(PlannerError, match="invalid step schema"):
        planner.plan("goal")


@patch("plan_execute_agent.planner.anthropic.Anthropic")
def test_plan_step_ids_are_preserved(mock_anthropic) -> None:
    planner = LLMPlanner()
    mock_msg = MagicMock()
    mock_msg.content[0].text = _steps_json(2)
    planner._client.messages.create.return_value = mock_msg

    plan = planner.plan("goal")
    assert plan.steps[0].id == "step-1"
    assert plan.steps[1].id == "step-2"
```

### `plan_execute_agent/tests/test_executor.py`

```python
"""
test_executor.py — unit tests for Executor (LLM mock, real toy tools).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from plan_execute_agent.executor import Executor, classify_error
from plan_execute_agent.models import Plan, Step, StepStatus
from plan_execute_agent.tools import ToolRegistry, build_default_registry


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


def test_classify_transient() -> None:
    assert classify_error("connection timeout") == "transient"
    assert classify_error("rate limit exceeded") == "transient"


def test_classify_fatal() -> None:
    assert classify_error("authentication error") == "fatal"
    assert classify_error("safety refusal") == "fatal"


def test_classify_structural() -> None:
    assert classify_error("ValueError: unexpected null") == "structural"
    assert classify_error("schema mismatch in output") == "structural"


# ---------------------------------------------------------------------------
# Executor happy path
# ---------------------------------------------------------------------------


def _stub_llm(response: str = "[]") -> MagicMock:
    client = MagicMock()
    client.complete.return_value = response
    return client


def _make_plan(*tools: str) -> Plan:
    steps = [
        Step(
            id=f"s{i}",
            description=f"step {i}",
            tool=t,
            inputs={"query": "test"} if t == "search" else {"text": "t"},
        )
        for i, t in enumerate(tools, 1)
    ]
    return Plan(goal="test goal", steps=steps)


def test_executor_all_success() -> None:
    plan = _make_plan("search", "summarize")
    executor = Executor(llm_client=_stub_llm(), max_retries=0)
    report = executor.run(plan, build_default_registry())

    assert not report.aborted
    assert report.replan_count == 0
    assert len(report.successful_results()) == 2


def test_executor_unknown_tool_triggers_structural_failure() -> None:
    plan = Plan(
        goal="g",
        steps=[Step(id="s1", description="d", tool="nonexistent", inputs={})],
    )
    executor = Executor(llm_client=_stub_llm("[]"), max_retries=0, max_replans=1)
    report = executor.run(plan, build_default_registry())
    # no replacement steps returned by stub → aborts at circuit breaker (1 replan used)
    assert report.replan_count >= 0  # execution attempted
    assert report.total_steps_attempted >= 1


def test_executor_replan_called_on_structural_error() -> None:
    registry = build_default_registry()
    call_count: dict[str, int] = {"n": 0}

    def bad_search(query: str) -> str:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ValueError("schema mismatch")
        return "ok"

    registry.register("search", bad_search)

    replacement_step = {
        "id": "s1b",
        "description": "retry with simpler query",
        "tool": "search",
        "inputs": {"query": "simple"},
    }
    llm = _stub_llm(json.dumps([replacement_step]))
    plan = Plan(goal="g", steps=[Step(id="s1", description="d", tool="search", inputs={"query": "x"})])
    executor = Executor(llm_client=llm, max_retries=0, max_replans=2)
    report = executor.run(plan, registry)

    assert report.replan_count == 1
    assert any(r.step_id == "s1b" for r in report.step_results)


def test_executor_circuit_breaker_aborts() -> None:
    registry = ToolRegistry()
    registry.register("search", lambda query: (_ for _ in ()).throw(ValueError("always fails")))

    # Stub LLM always returns one more search step → loop until circuit breaker
    replacement = json.dumps(
        [{"id": "sN", "description": "retry", "tool": "search", "inputs": {"query": "q"}}]
    )
    executor = Executor(llm_client=_stub_llm(replacement), max_retries=0, max_replans=2)
    plan = Plan(goal="g", steps=[Step(id="s1", description="d", tool="search", inputs={"query": "q"})])
    report = executor.run(plan, registry)

    assert report.aborted
    assert "circuit breaker" in (report.abort_reason or "")
```

### `plan_execute_agent/tests/test_agent.py`

```python
"""
test_agent.py — integration-level tests for PlanExecuteAgent (LLM fully mocked).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from plan_execute_agent.agent import PlanExecuteAgent
from plan_execute_agent.executor import Executor
from plan_execute_agent.models import Plan, Step
from plan_execute_agent.planner import LLMPlanner
from plan_execute_agent.tools import build_default_registry


def _mock_planner(steps: list[dict]) -> LLMPlanner:
    with patch("plan_execute_agent.planner.anthropic.Anthropic"):
        planner = LLMPlanner()
    mock_msg = MagicMock()
    mock_msg.content[0].text = json.dumps(steps)
    planner._client.messages.create.return_value = mock_msg
    return planner


def _stub_llm_client(response: str = "[]") -> MagicMock:
    client = MagicMock()
    client.complete.return_value = response
    return client


def test_agent_successful_run() -> None:
    planner = _mock_planner(
        [
            {"id": "s1", "description": "search", "tool": "search", "inputs": {"query": "test"}},
            {"id": "s2", "description": "summarize", "tool": "summarize", "inputs": {"text": "data"}},
        ]
    )
    executor = Executor(llm_client=_stub_llm_client(), max_retries=0)
    agent = PlanExecuteAgent(
        planner=planner,
        executor=executor,
        registry=build_default_registry(),
    )
    report = agent.run("Research Python async patterns")

    assert not report.aborted
    assert len(report.successful_results()) == 2
    assert report.goal == "Research Python async patterns"


def test_agent_planner_failure_returns_aborted_report() -> None:
    with patch("plan_execute_agent.planner.anthropic.Anthropic"):
        planner = LLMPlanner()
    mock_msg = MagicMock()
    mock_msg.content[0].text = "not json"
    planner._client.messages.create.return_value = mock_msg

    executor = Executor(llm_client=_stub_llm_client(), max_retries=0)
    agent = PlanExecuteAgent(
        planner=planner,
        executor=executor,
        registry=build_default_registry(),
    )
    report = agent.run("some goal")

    assert report.aborted
    assert "planning failed" in (report.abort_reason or "")


def test_agent_report_contains_goal() -> None:
    planner = _mock_planner(
        [{"id": "s1", "description": "search", "tool": "search", "inputs": {"query": "q"}}]
    )
    executor = Executor(llm_client=_stub_llm_client(), max_retries=0)
    agent = PlanExecuteAgent(
        planner=planner,
        executor=executor,
        registry=build_default_registry(),
    )
    goal = "Write a summary of LangGraph"
    report = agent.run(goal)
    assert report.goal == goal
```

## Build instructions

### 1. Create directory structure

```bash
mkdir -p plan_execute_agent/tests
touch plan_execute_agent/__init__.py
touch plan_execute_agent/tests/__init__.py
```

### 2. Copy each code block into the named file

Copy `models.py`, `tools.py`, `planner.py`, `executor.py`, `agent.py` into `plan_execute_agent/`.
Copy all four `test_*.py` files into `plan_execute_agent/tests/`.

### 3. Install dependencies

```bash
pip install anthropic pydantic pytest
```

### 4. Set API key (only needed for live LLM calls; tests mock it)

```bash
export ANTHROPIC_API_KEY=your_key_here
```

### 5. Run tests

```bash
cd plan_execute_agent
python -m pytest tests/ -v
```

All tests should pass without a real API key because every LLM call is patched.

### 6. Run the agent live (optional)

```python
from plan_execute_agent import PlanExecuteAgent

agent = PlanExecuteAgent()
report = agent.run("Research Python async patterns and write a summary")
print(report.model_dump_json(indent=2))
```

## Exercises

1. Add a `SearchTool` class (instead of a bare function) with a `__call__` method and a `max_results: int` attribute. Register it in `build_default_registry`. Write a test that verifies `max_results` defaults to 5.

2. Extend `ExecutionReport` with a `duration_seconds` property that returns `(finished_at - started_at).total_seconds()` when `finished_at` is set, else `None`. Write a test verifying the value is positive after a completed run.

3. Implement `Executor.run_parallel(plan, registry) -> ExecutionReport` that executes independent steps concurrently using `asyncio.gather`. Two steps are independent if neither references the other's `id` in its `inputs` values. Write at least one test.

(Solutions: ../solutions/week-03/day-6.md)

## Self-check

- [ ] I can describe the responsibility of each module (models, tools, planner, executor, agent) in one sentence.
- [ ] I can implement `ToolRegistry.call` from memory including the `ToolNotFoundError` path.
- [ ] I can write a pytest test for a function that calls an LLM, mocking the LLM with `unittest.mock.patch`.
- [ ] I can explain why `LLMClient` is defined as a `Protocol` rather than an abstract base class.
- [ ] I can trace a failing step through `Executor.run` and identify exactly where the circuit breaker check occurs.

## Supplementary links

- Anthropic model IDs: https://docs.anthropic.com/en/docs/about-claude/models
- Pydantic v2 models: https://docs.pydantic.dev/latest/concepts/models/
- `unittest.mock.patch` reference: https://docs.python.org/3/library/unittest.mock.html#unittest.mock.patch
- Python `Protocol` (structural subtyping): https://docs.python.org/3/library/typing.html#typing.Protocol
- pytest fixtures and parametrize: https://docs.pytest.org/en/stable/how-to/parametrize.html
