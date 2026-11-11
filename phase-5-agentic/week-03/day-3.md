# Phase 5 · Week 03 · Day 3 — Plan-and-Execute Architecture

> Objective: Separate planning from execution, implement a dependency-ordered executor, and accumulate step results into a structured report.  ·  Time: ~1 hr  ·  Prereqs: [[week-03/day-2]]

## Concept

**Plan-and-Execute** (Wang et al., 2023) splits the agent loop into two independent phases:

```
Phase 1 — Plan:   LLM reasons about the goal → structured Plan (no tools called yet)
Phase 2 — Execute: deterministic code walks the DAG, calls tools, stores results
```

### Why this beats pure ReAct for long tasks

| Dimension | ReAct | Plan-and-Execute |
|---|---|---|
| LLM calls | 1 per step | 1 for plan + 1 per uncertain step |
| Parallelism | Strictly sequential | Independent branches can run in parallel |
| Recoverability | Must re-prompt from scratch on failure | Re-run failed step, plan unchanged |
| Inspectability | Hard to audit mid-run | Plan is a serialisable artefact |
| Cost at depth 20 | High (full trace in context) | Low (executor is code, not LLM) |

The key insight: once a plan is validated, most execution steps are deterministic tool calls. There is no need to involve the LLM unless the tool output is ambiguous.

### State management between steps

The executor maintains a **result context** — a dict mapping `step_id -> StepResult`. When a step executes, it receives the outputs of its dependencies as context. This replaces the ever-growing ReAct trace.

### Early termination conditions

- A step fails and `fail_fast=True` — abort immediately.
- All required steps succeed — return early even if optional steps are pending.
- An executor-level timeout fires — return partial results with `success=False`.

## How it works internally

### Topological execution

The executor builds an in-degree counter from the plan's `depends_on` edges. It maintains a ready queue of steps whose dependencies are all resolved. After each step completes, it decrements in-degree for downstream steps and enqueues any that reach zero.

This is Kahn's algorithm again — the same logic used to *detect* cycles during validation is used to *drive* execution order.

### Result accumulation

Each `StepResult` stores:
- `step_id` — which step produced it
- `output` — the raw string or structured value returned by the tool
- `success` — whether the tool call returned without error
- `error` — the exception message if `success=False`

The execution context passed to each step is `dict[str, str]` — `{step_id: output}` for all completed dependencies.

### Structured logging

Every executor event (step started, step completed, step failed, plan done) is logged with `extra={"step_id": ..., "tool": ...}` so structured log aggregators can filter and alert on specific step failures without parsing free text.

## Worked example

```python
"""
day_3_executor.py — Plan-and-Execute architecture.

Depends on the Plan / Step models from day_2_planner.py.
Copy that file to the same directory, or install the package.

Run:
    python day_3_executor.py
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

# Re-use day-2 models — adjust the import path if needed.
from day_2_planner import Plan, Planner, Step, topological_order

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    step_id: str
    output: str
    success: bool
    error: str | None = None
    duration_seconds: float = 0.0


@dataclass
class ExecutionReport:
    plan: Plan
    results: dict[str, StepResult] = field(default_factory=dict)
    success: bool = False
    failed_steps: list[str] = field(default_factory=list)
    total_duration_seconds: float = 0.0

    @property
    def succeeded_steps(self) -> list[str]:
        return [r.step_id for r in self.results.values() if r.success]

    def summary(self) -> str:
        n_total = len(self.plan.steps)
        n_ok = len(self.succeeded_steps)
        n_fail = len(self.failed_steps)
        return (
            f"Plan '{self.plan.goal}': "
            f"{n_ok}/{n_total} steps succeeded, "
            f"{n_fail} failed, "
            f"{self.total_duration_seconds:.2f}s total."
        )


# ---------------------------------------------------------------------------
# Toy tool registry (replicates day-1 tools)
# ---------------------------------------------------------------------------

def _search(query: str, _ctx: dict[str, str]) -> str:
    facts = {
        "eiffel tower": "The Eiffel Tower is 330 metres tall.",
        "python creator": "Python was created by Guido van Rossum.",
    }
    for key, val in facts.items():
        if key in query.lower():
            return val
    return f"No fact found for '{query}'."


def _calculate(expr: str, _ctx: dict[str, str]) -> str:
    import re  # noqa: PLC0415
    if not re.fullmatch(r"[\d\s\+\-\*\/\(\)\.]+", expr.strip()):
        return f"Error: disallowed characters in '{expr}'."
    try:
        result = eval(expr.strip(), {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


def _summarise(text: str, ctx: dict[str, str]) -> str:
    # In a real system this would call the LLM.
    # Here we concatenate dependency outputs for demo purposes.
    context_str = "; ".join(f"{k}: {v}" for k, v in ctx.items())
    return f"Summary based on [{context_str}]: {text}"


# Context-aware tool signature: (argument, dependency_outputs) -> str
ToolFn = Callable[[str, dict[str, str]], str]

DEFAULT_TOOLS: dict[str, ToolFn] = {
    "search":    _search,
    "calculate": _calculate,
    "summarise": _summarise,
}


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class PlanExecutor:
    def __init__(
        self,
        tools: dict[str, ToolFn] | None = None,
        fail_fast: bool = True,
    ) -> None:
        self.tools = tools or DEFAULT_TOOLS
        self.fail_fast = fail_fast

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def execute(self, plan: Plan) -> ExecutionReport:
        """Execute all steps in dependency order. Return a full report."""
        report = ExecutionReport(plan=plan)
        start = time.monotonic()

        log.info(
            "Execution started",
            extra={"goal": plan.goal, "n_steps": len(plan.steps)},
        )

        results: dict[str, StepResult] = {}

        # Build adjacency for in-degree tracking
        in_degree: dict[str, int] = {s.id: 0 for s in plan.steps}
        adjacency: dict[str, list[str]] = {s.id: [] for s in plan.steps}
        step_map: dict[str, Step] = {s.id: s for s in plan.steps}

        for step in plan.steps:
            for dep in step.depends_on:
                adjacency[dep].append(step.id)
                in_degree[step.id] += 1

        ready: deque[str] = deque(
            sid for sid, deg in in_degree.items() if deg == 0
        )

        while ready:
            step_id = ready.popleft()
            step = step_map[step_id]

            # Build context from completed dependencies
            context: dict[str, str] = {
                dep: results[dep].output
                for dep in step.depends_on
                if dep in results and results[dep].success
            }

            result = self.execute_step(step, context)
            results[step_id] = result

            if not result.success:
                log.error(
                    "Step failed",
                    extra={"step_id": step_id, "error": result.error},
                )
                report.failed_steps.append(step_id)
                if self.fail_fast:
                    log.warning("fail_fast=True — aborting execution.")
                    break
            else:
                log.info(
                    "Step succeeded",
                    extra={"step_id": step_id, "output_preview": result.output[:60]},
                )
                # Unlock downstream steps
                for downstream in adjacency[step_id]:
                    in_degree[downstream] -= 1
                    if in_degree[downstream] == 0:
                        ready.append(downstream)

        report.results = results
        report.success = len(report.failed_steps) == 0
        report.total_duration_seconds = time.monotonic() - start

        log.info("Execution complete: %s", report.summary())
        return report

    def execute_step(self, step: Step, context: dict[str, str]) -> StepResult:
        """Execute a single step. Returns StepResult regardless of success/failure."""
        log.info(
            "Executing step",
            extra={"step_id": step.id, "tool": step.tool, "description": step.description},
        )
        start = time.monotonic()

        if step.tool is None:
            # LLM-only step — substitute with a simple echo for demo
            output = f"[LLM step] {step.description} — context keys: {list(context)}"
            return StepResult(
                step_id=step.id,
                output=output,
                success=True,
                duration_seconds=time.monotonic() - start,
            )

        tool_fn = self.tools.get(step.tool)
        if tool_fn is None:
            return StepResult(
                step_id=step.id,
                output="",
                success=False,
                error=f"Unknown tool '{step.tool}'. Available: {list(self.tools)}",
                duration_seconds=time.monotonic() - start,
            )

        try:
            # Use the step description as the tool argument in this demo.
            # A real executor would extract `tool_args` from the Step model.
            output = tool_fn(step.description, context)
            return StepResult(
                step_id=step.id,
                output=output,
                success=True,
                duration_seconds=time.monotonic() - start,
            )
        except Exception as exc:  # noqa: BLE001
            return StepResult(
                step_id=step.id,
                output="",
                success=False,
                error=str(exc),
                duration_seconds=time.monotonic() - start,
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Build a plan via the day-2 Planner (uses mock LLM if no API key)
    planner = Planner()
    plan = planner.plan("Research the Eiffel Tower and summarise findings.")

    print("\n=== Execution order ===")
    for sid in topological_order(plan):
        print(f"  {sid}")

    executor = PlanExecutor()
    report = executor.execute(plan)

    print("\n=== Execution Report ===")
    print(report.summary())
    for sid, result in report.results.items():
        status = "OK" if result.success else "FAIL"
        print(f"  [{status}] {sid}: {result.output[:80]}")

    # Demonstrate fail_fast with a plan that has a bad tool
    from day_2_planner import Step as PlanStep

    bad_plan_data: dict[str, Any] = {
        "goal": "test fail_fast",
        "steps": [
            {
                "id": "step_good",
                "description": "eiffel tower",
                "depends_on": [],
                "tool": "search",
                "expected_output": "height",
            },
            {
                "id": "step_bad",
                "description": "arg",
                "depends_on": ["step_good"],
                "tool": "nonexistent_tool",
                "expected_output": "x",
            },
            {
                "id": "step_unreachable",
                "description": "should not run",
                "depends_on": ["step_bad"],
                "tool": "search",
                "expected_output": "y",
            },
        ],
    }
    bad_plan = Plan.model_validate(bad_plan_data)
    fail_report = executor.execute(bad_plan)
    print(f"\nFail-fast report: {fail_report.summary()}")
    print(f"Failed steps: {fail_report.failed_steps}")
    step_ids_run = list(fail_report.results)
    print(f"Steps that ran: {step_ids_run}")
```

## Exercises

1. The `execute_step` method uses `step.description` as the tool argument. Extend `Step` (from day-2) with an optional `tool_args: dict[str, str]` field. Update `execute_step` to extract the argument from `tool_args["query"]` (for `search`) or `tool_args["expr"]` (for `calculate`), falling back to `step.description` if the field is absent.

2. Implement `fail_fast=False` behaviour: when a step fails, skip it but continue executing steps that do not depend on it (only skip direct and transitive dependents). Add a pytest test confirming that an independent branch still runs after one branch fails.

3. Add a `timeout_seconds: float | None` parameter to `PlanExecutor`. If set, abort execution when `time.monotonic() - start` exceeds it. Return an `ExecutionReport` with `success=False` and a descriptive error in `failed_steps`.

4. Add a `dry_run: bool = False` parameter to `execute()`. In dry-run mode, log what each step would do but do not call any tools. Return an `ExecutionReport` where all steps have `success=True` and `output="[dry-run]"`.

5. Write a pytest fixture that constructs a 4-step diamond DAG (step A → B, A → C, B → D, C → D) and assert: (a) topological_order includes all 4 steps, (b) D appears after both B and C, (c) the executor runs all 4 steps when all tools succeed.

(Solutions: ../solutions/week-03/day-3.md)

## Self-check

- [ ] I can explain why Plan-and-Execute is cheaper than pure ReAct for long tasks.
- [ ] I can implement Kahn's algorithm-based DAG execution from memory.
- [ ] I can describe how dependency context is passed to each step during execution.
- [ ] I can implement fail_fast and non-fail_fast modes correctly.
- [ ] I can structure logs with `extra={}` kwargs for structured log aggregation.

---

## Supplementary links

- "Plan-and-Solve Prompting" (Wang et al., 2023) — https://arxiv.org/abs/2305.04091
- LangGraph Plan-and-Execute example — https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/plan-and-execute/
- Wikipedia — Topological sorting — https://en.wikipedia.org/wiki/Topological_sorting
- Python `dataclasses` documentation — https://docs.python.org/3.12/library/dataclasses.html
- Python `logging` — LogRecord attributes and extra — https://docs.python.org/3.12/library/logging.html#logrecord-attributes
