# Solutions — Phase 5 · Week 03 · Day 3 — Plan-and-Execute Architecture

## Exercise 1

Integrate `topological_sort` into `PlanExecutor.execute` so steps run in dependency order.

```python
"""
executor/executor.py — PlanExecutor with topological ordering.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any, Callable

from planning.models import Plan, Step

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Topological sort (self-contained copy; or import from planning.sort)
# ---------------------------------------------------------------------------

def _topological_sort(plan: Plan) -> list[Step]:
    """Kahn's algorithm — raises ValueError on cycle."""
    step_map: dict[str, Step] = {s.id: s for s in plan.steps}
    in_degree: dict[str, int] = {s.id: 0 for s in plan.steps}
    dependents: dict[str, list[str]] = {s.id: [] for s in plan.steps}

    for step in plan.steps:
        for dep in step.depends_on:
            in_degree[step.id] += 1
            dependents[dep].append(step.id)

    queue: deque[str] = deque(
        sorted(
            (sid for sid, deg in in_degree.items() if deg == 0),
            key=lambda sid: step_map[sid].priority,
        )
    )
    sorted_steps: list[Step] = []

    while queue:
        node = queue.popleft()
        sorted_steps.append(step_map[node])
        for dep in dependents[node]:
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                queue.append(dep)

    if len(sorted_steps) != len(plan.steps):
        unresolved = [sid for sid, d in in_degree.items() if d > 0]
        raise ValueError(f"Cycle detected; unresolved: {unresolved}")

    return sorted_steps


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

from pydantic import BaseModel


class StepResult(BaseModel):
    step_id: str
    output: Any
    success: bool
    error: str | None = None


# ---------------------------------------------------------------------------
# PlanExecutor
# ---------------------------------------------------------------------------

ToolDispatcher = Callable[[Step, dict[str, Any]], Any]


class PlanExecutor:
    """Execute a Plan in topological dependency order."""

    def __init__(
        self,
        tool_dispatcher: ToolDispatcher,
        on_step_complete: Callable[[StepResult], None] | None = None,
    ) -> None:
        self._dispatch = tool_dispatcher
        self._on_step_complete = on_step_complete

    def execute(self, plan: Plan) -> dict[str, StepResult]:
        """Run all steps in dependency order.

        Returns a mapping of step_id -> StepResult.
        """
        ordered_steps = _topological_sort(plan)
        results: dict[str, StepResult] = {}

        for step in ordered_steps:
            # Inject dependency context (Exercise 2)
            context: dict[str, Any] = self._build_context(step, results)

            logger.info("Executing step '%s'", step.id)
            result = self._run_step(step, context)
            results[step.id] = result

            # Callback hook (Exercise 3)
            if self._on_step_complete is not None:
                self._on_step_complete(result)

        return results

    def _build_context(
        self, step: Step, completed: dict[str, StepResult]
    ) -> dict[str, Any]:
        """Inject the outputs of completed dependency steps into context."""
        context: dict[str, Any] = {}
        for dep_id in step.depends_on:
            if dep_id in completed and completed[dep_id].success:
                context[dep_id] = completed[dep_id].output
                logger.debug("Injecting context['%s'] into step '%s'", dep_id, step.id)
        return context

    def _run_step(self, step: Step, context: dict[str, Any]) -> StepResult:
        try:
            output = self._dispatch(step, context)
            return StepResult(step_id=step.id, output=output, success=True)
        except Exception as exc:
            logger.exception("Step '%s' failed: %s", step.id, exc)
            return StepResult(step_id=step.id, output=None, success=False, error=str(exc))
```

**Explanation:**
- `_topological_sort` is embedded in the executor module (or import from `planning.sort`).
- `execute` calls the sort first, then iterates the ordered list — no step runs before its dependencies.
- `_build_context` and the callback hook are already wired in (Exercises 2 and 3 below).

---

## Exercise 2

`context_injection`: before each step, inject dependency outputs into `context`.

The relevant method from the executor above:

```python
def _build_context(
    self, step: Step, completed: dict[str, StepResult]
) -> dict[str, Any]:
    """Return a context dict keyed by dependency step IDs.

    The tool dispatcher receives this context and can read
    context["dep_step_id"] to access the prior step's output.
    """
    context: dict[str, Any] = {}
    for dep_id in step.depends_on:
        if dep_id in completed and completed[dep_id].success:
            context[dep_id] = completed[dep_id].output
            logger.debug(
                "Step '%s': injecting context from dependency '%s'",
                step.id,
                dep_id,
            )
        elif dep_id in completed and not completed[dep_id].success:
            logger.warning(
                "Step '%s': dependency '%s' failed — context not injected",
                step.id,
                dep_id,
            )
    return context
```

**Usage example:**

```python
def my_tool_dispatcher(step: Step, context: dict[str, Any]) -> str:
    if step.id == "summarize":
        raw_text = context.get("fetch_data", "")  # output from the 'fetch_data' step
        return f"Summary of: {raw_text[:50]}"
    return f"Executed {step.id}"
```

**Explanation:**
- Context is built fresh per step — no shared mutable state.
- Only successful dependency outputs are injected; failed dependencies are logged and skipped.
- The `step.depends_on` list is the single source of truth for which keys appear in context.

---

## Exercise 3

`on_step_complete` callback hook on `PlanExecutor`.

Already shown in Exercise 1. Focused extract:

```python
class PlanExecutor:
    def __init__(
        self,
        tool_dispatcher: ToolDispatcher,
        on_step_complete: Callable[[StepResult], None] | None = None,
    ) -> None:
        self._dispatch = tool_dispatcher
        self._on_step_complete = on_step_complete  # store the optional callback

    def execute(self, plan: Plan) -> dict[str, StepResult]:
        ordered_steps = _topological_sort(plan)
        results: dict[str, StepResult] = {}

        for step in ordered_steps:
            context = self._build_context(step, results)
            result = self._run_step(step, context)
            results[step.id] = result

            # Fire the callback if provided
            if self._on_step_complete is not None:
                self._on_step_complete(result)

        return results
```

**Usage example:**

```python
import logging

logger = logging.getLogger(__name__)

def log_step(result: StepResult) -> None:
    if result.success:
        logger.info("DONE: %s -> %s", result.step_id, result.output)
    else:
        logger.error("FAIL: %s -> %s", result.step_id, result.error)


executor = PlanExecutor(tool_dispatcher=my_dispatcher, on_step_complete=log_step)
executor.execute(plan)
```

**Explanation:**
- `on_step_complete` is `Callable[[StepResult], None] | None` — fully typed, optional.
- The guard `if self._on_step_complete is not None` avoids a call on the default `None`.
- The callback receives the immutable `StepResult` — it cannot mutate execution state.

---

## Exercise 4

Pytest tests for `PlanExecutor.execute` with a 3-step plan.

```python
"""
tests/unit/test_plan_executor.py
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, call

import pytest

from executor.executor import PlanExecutor, StepResult
from planning.models import Plan, Step


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_three_step_plan() -> Plan:
    """Step 3 depends on steps 1 and 2; steps 1 and 2 are independent."""
    return Plan(
        goal="Three-step integration test",
        steps=[
            Step(id="step1", description="First independent step"),
            Step(id="step2", description="Second independent step"),
            Step(id="step3", description="Final step", depends_on=["step1", "step2"]),
        ],
    )


def simple_dispatcher(step: Step, context: dict[str, Any]) -> str:
    return f"output_of_{step.id}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPlanExecutorThreeSteps:

    def test_all_three_steps_complete(self) -> None:
        """All steps produce a StepResult with success=True."""
        plan = make_three_step_plan()
        executor = PlanExecutor(tool_dispatcher=simple_dispatcher)

        results = executor.execute(plan)

        assert set(results.keys()) == {"step1", "step2", "step3"}
        for step_id, result in results.items():
            assert result.success, f"Step '{step_id}' should have succeeded"
            assert result.error is None

    def test_results_stored_with_correct_output(self) -> None:
        """Each step's output matches what the dispatcher returned."""
        plan = make_three_step_plan()
        executor = PlanExecutor(tool_dispatcher=simple_dispatcher)

        results = executor.execute(plan)

        assert results["step1"].output == "output_of_step1"
        assert results["step2"].output == "output_of_step2"
        assert results["step3"].output == "output_of_step3"

    def test_dependency_order_respected(self) -> None:
        """step3 is executed after step1 and step2."""
        execution_order: list[str] = []

        def order_tracking_dispatcher(step: Step, context: dict[str, Any]) -> str:
            execution_order.append(step.id)
            return f"output_of_{step.id}"

        plan = make_three_step_plan()
        executor = PlanExecutor(tool_dispatcher=order_tracking_dispatcher)
        executor.execute(plan)

        assert "step3" not in execution_order[:2], "step3 ran before its deps"
        assert execution_order.index("step3") > execution_order.index("step1")
        assert execution_order.index("step3") > execution_order.index("step2")

    def test_context_injection_into_step3(self) -> None:
        """step3 receives context keys for step1 and step2."""
        received_contexts: dict[str, dict[str, Any]] = {}

        def capturing_dispatcher(step: Step, context: dict[str, Any]) -> str:
            received_contexts[step.id] = dict(context)
            return f"output_of_{step.id}"

        plan = make_three_step_plan()
        executor = PlanExecutor(tool_dispatcher=capturing_dispatcher)
        executor.execute(plan)

        ctx3 = received_contexts["step3"]
        assert "step1" in ctx3, "step3 should receive step1 output in context"
        assert "step2" in ctx3, "step3 should receive step2 output in context"
        assert ctx3["step1"] == "output_of_step1"
        assert ctx3["step2"] == "output_of_step2"

    def test_on_step_complete_called_for_each_step(self) -> None:
        """on_step_complete callback is invoked exactly 3 times, once per step."""
        callback = MagicMock()
        plan = make_three_step_plan()
        executor = PlanExecutor(
            tool_dispatcher=simple_dispatcher,
            on_step_complete=callback,
        )

        executor.execute(plan)

        assert callback.call_count == 3
        called_ids = {c.args[0].step_id for c in callback.call_args_list}
        assert called_ids == {"step1", "step2", "step3"}

    def test_failed_step_stored_as_failure(self) -> None:
        """A dispatcher that raises stores a failure result without crashing."""
        def failing_dispatcher(step: Step, context: dict[str, Any]) -> str:
            if step.id == "step2":
                raise RuntimeError("step2 intentionally failed")
            return f"output_of_{step.id}"

        plan = make_three_step_plan()
        executor = PlanExecutor(tool_dispatcher=failing_dispatcher)

        results = executor.execute(plan)

        assert results["step1"].success is True
        assert results["step2"].success is False
        assert results["step2"].error is not None
        assert "intentionally failed" in results["step2"].error

    def test_mocked_dispatcher_called_for_each_step(self) -> None:
        """Mock dispatcher: verify it is called with the correct Step objects."""
        mock_dispatcher = MagicMock(side_effect=lambda step, ctx: f"out_{step.id}")
        plan = make_three_step_plan()
        executor = PlanExecutor(tool_dispatcher=mock_dispatcher)

        executor.execute(plan)

        assert mock_dispatcher.call_count == 3
        called_step_ids = {c.args[0].id for c in mock_dispatcher.call_args_list}
        assert called_step_ids == {"step1", "step2", "step3"}
```

**Explanation:**
- Seven tests cover correctness, ordering, context injection, callback firing, failure handling, and mock dispatch count.
- `order_tracking_dispatcher` appends to a list — no side-channel communication needed.
- `capturing_dispatcher` records the context dict per step, enabling direct assertion on what each step received.
- `MagicMock` with `side_effect=lambda ...` lets the mock behave like the real dispatcher while still being inspectable.
