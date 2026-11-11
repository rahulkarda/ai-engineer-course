# Phase 5 · Week 03 · Day 4 — Plan Representation with Pydantic + Executor Loop

> Objective: Model heterogeneous step types with discriminated unions, persist plans as JSON, and implement a robust executor loop with per-step retries and exponential backoff.  ·  Time: ~1 hr  ·  Prereqs: [[week-03/day-3]]

## Concept

Day 3's `Step` model treats all steps uniformly — a single `tool` field. This breaks down when steps have fundamentally different shapes:

- A **tool step** needs `tool_name` and `tool_args`.
- An **LLM step** needs a `prompt_template` and an `output_key` to store the result under.
- A **subagent step** needs an `agent_name` and a `goal` string.

Forcing all three into one model means every field is optional and the type system cannot enforce which combinations are valid.

**Discriminated unions** solve this by tagging each variant with a `Literal` type field (`type: Literal["tool"]`, etc.) and letting Pydantic route validation to the correct model automatically.

### Why discriminated unions over inheritance

Pydantic v2 discriminated unions:

- Validate in O(1) — Pydantic checks the tag field first, then validates only the matching model.
- Serialise cleanly — `model_dump(mode="json")` includes the `type` tag, enabling lossless round-trips.
- Are type-safe — `mypy` knows exactly which fields exist on each branch.

With plain inheritance and no discriminator, Pydantic tries each subclass sequentially and picks the first match — O(n) and fragile.

### JSON round-trip as a correctness guarantee

A plan that survives `json.loads(plan.model_dump_json())` without data loss is a plan that can be:

- Saved to disk and resumed after a crash.
- Sent over a network to a remote executor.
- Stored in a database and replayed for debugging.

Round-trip testing (`plan == Plan.model_validate_json(plan.model_dump_json())`) should be part of every CI suite that touches plan models.

### Retry with exponential backoff

Tool calls fail. Network timeouts, rate limits, and transient errors are facts of life. A robust executor retries each step up to `max_retries` times with an exponential delay:

```
attempt 0: call → fail
attempt 1: sleep(base * 2^0) → call → fail
attempt 2: sleep(base * 2^1) → call → fail
attempt 3: give up → record as failed
```

The backoff is a stub in this example (actual `time.sleep` is replaced by a no-op) so tests stay fast. In production, swap the stub for real sleep.

## How it works internally

### Pydantic discriminated union dispatch

```python
StepDefinition = Annotated[ToolStep | LLMStep | SubAgentStep, Field(discriminator="type")]
```

When Pydantic validates `{"type": "llm", "prompt_template": "...", "output_key": "..."}` against `StepDefinition`, it:

1. Reads the `type` field.
2. Looks up `"llm"` in the discriminator map → `LLMStep`.
3. Validates the remaining fields against `LLMStep` only.

If `type` is missing or unrecognised, Pydantic raises `ValidationError` immediately.

### Executor loop with retries

The executor loop extends day-3's DAG execution with a `_execute_with_retry` wrapper:

```
for attempt in range(max_retries + 1):
    result = try_execute(step)
    if result.success:
        return result
    if attempt < max_retries:
        sleep(backoff(attempt))
return result  # final failure
```

The `StepResult` now records `attempts: int` so callers can see whether a step needed retries.

## Worked example

```python
"""
day_4_typed_executor.py — Discriminated unions, JSON round-trip, retry loop.

Run:
    python day_4_typed_executor.py

Does not require ANTHROPIC_API_KEY — all LLM steps use a mock.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, Literal

from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Discriminated step definitions
# ---------------------------------------------------------------------------

class ToolStep(BaseModel):
    type: Literal["tool"] = "tool"
    id: str
    description: str
    depends_on: list[str] = Field(default_factory=list)
    expected_output: str = ""
    tool_name: str
    tool_args: dict[str, Any] = Field(default_factory=dict)


class LLMStep(BaseModel):
    type: Literal["llm"] = "llm"
    id: str
    description: str
    depends_on: list[str] = Field(default_factory=list)
    expected_output: str = ""
    prompt_template: str
    output_key: str


class SubAgentStep(BaseModel):
    type: Literal["subagent"] = "subagent"
    id: str
    description: str
    depends_on: list[str] = Field(default_factory=list)
    expected_output: str = ""
    agent_name: str
    goal: str


# The discriminated union — Pydantic routes by the `type` tag.
StepDefinition = Annotated[
    ToolStep | LLMStep | SubAgentStep,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Typed plan model
# ---------------------------------------------------------------------------

class TypedPlan(BaseModel):
    goal: str
    steps: list[StepDefinition]  # type: ignore[valid-type]

    @property
    def step_ids(self) -> set[str]:
        return {s.id for s in self.steps}

    def get_step(self, step_id: str) -> StepDefinition | None:  # type: ignore[return]
        return next((s for s in self.steps if s.id == step_id), None)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    step_id: str
    output: str
    success: bool
    error: str | None = None
    attempts: int = 1
    duration_seconds: float = 0.0


@dataclass
class ExecutionReport:
    plan: TypedPlan
    results: dict[str, StepResult] = field(default_factory=dict)
    success: bool = False
    failed_steps: list[str] = field(default_factory=list)
    total_duration_seconds: float = 0.0

    def summary(self) -> str:
        n_ok = sum(1 for r in self.results.values() if r.success)
        return (
            f"Plan '{self.plan.goal}': "
            f"{n_ok}/{len(self.plan.steps)} steps succeeded, "
            f"{len(self.failed_steps)} failed."
        )


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

ToolFn = Callable[[dict[str, Any], dict[str, str]], str]

_call_count: dict[str, int] = {}  # used to simulate transient failures


def _search(args: dict[str, Any], _ctx: dict[str, str]) -> str:
    query: str = args.get("query", "")
    facts = {
        "eiffel tower": "The Eiffel Tower is 330 metres tall.",
        "python": "Python was created by Guido van Rossum in 1991.",
    }
    for key, val in facts.items():
        if key in query.lower():
            return val
    return f"No result for '{query}'."


def _flaky_calculate(args: dict[str, Any], _ctx: dict[str, str]) -> str:
    """Fails on the first call to demonstrate retry logic."""
    key = "calculate"
    _call_count[key] = _call_count.get(key, 0) + 1
    if _call_count[key] == 1:
        raise RuntimeError("Transient network error (simulated).")
    import re  # noqa: PLC0415
    expr: str = args.get("expr", "0")
    if not re.fullmatch(r"[\d\s\+\-\*\/\(\)\.]+", expr.strip()):
        return f"Error: disallowed characters in '{expr}'."
    result = eval(expr.strip(), {"__builtins__": {}})  # noqa: S307
    return str(result)


TOOL_REGISTRY: dict[str, ToolFn] = {
    "search":    _search,
    "calculate": _flaky_calculate,
}


# ---------------------------------------------------------------------------
# Typed executor with retry
# ---------------------------------------------------------------------------

class TypedPlanExecutor:
    def __init__(
        self,
        tools: dict[str, ToolFn] | None = None,
        max_retries: int = 3,
        backoff_base_seconds: float = 1.0,
        fail_fast: bool = True,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self.tools = tools or TOOL_REGISTRY
        self.max_retries = max_retries
        self.backoff_base = backoff_base_seconds
        self.fail_fast = fail_fast
        # Stub sleep so tests stay fast; swap for `time.sleep` in production.
        self._sleep = sleep_fn or (lambda _: None)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def execute(self, plan: TypedPlan) -> ExecutionReport:
        report = ExecutionReport(plan=plan)
        start = time.monotonic()
        results: dict[str, StepResult] = {}

        in_degree: dict[str, int] = {s.id: 0 for s in plan.steps}
        adjacency: dict[str, list[str]] = {s.id: [] for s in plan.steps}
        step_map: dict[str, StepDefinition] = {s.id: s for s in plan.steps}  # type: ignore[misc]

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

            context: dict[str, str] = {
                dep: results[dep].output
                for dep in step.depends_on
                if dep in results and results[dep].success
            }

            result = self._execute_with_retry(step, context)
            results[step_id] = result

            if not result.success:
                log.error("Step failed after %d attempt(s): %s", result.attempts, step_id)
                report.failed_steps.append(step_id)
                if self.fail_fast:
                    break
            else:
                log.info(
                    "Step OK (attempt %d/%d): %s",
                    result.attempts,
                    self.max_retries + 1,
                    step_id,
                )
                for downstream in adjacency[step_id]:
                    in_degree[downstream] -= 1
                    if in_degree[downstream] == 0:
                        ready.append(downstream)

        report.results = results
        report.success = len(report.failed_steps) == 0
        report.total_duration_seconds = time.monotonic() - start
        log.info(report.summary())
        return report

    # ------------------------------------------------------------------
    # Step dispatch by type
    # ------------------------------------------------------------------

    def _execute_with_retry(
        self,
        step: StepDefinition,  # type: ignore[valid-type]
        context: dict[str, str],
    ) -> StepResult:
        last_result = StepResult(step_id=step.id, output="", success=False)

        for attempt in range(self.max_retries + 1):
            start = time.monotonic()
            try:
                output = self._dispatch(step, context)
                return StepResult(
                    step_id=step.id,
                    output=output,
                    success=True,
                    attempts=attempt + 1,
                    duration_seconds=time.monotonic() - start,
                )
            except Exception as exc:  # noqa: BLE001
                delay = self.backoff_base * math.pow(2, attempt)
                log.warning(
                    "Step '%s' attempt %d failed: %s. Retry in %.1fs.",
                    step.id, attempt + 1, exc, delay,
                )
                last_result = StepResult(
                    step_id=step.id,
                    output="",
                    success=False,
                    error=str(exc),
                    attempts=attempt + 1,
                    duration_seconds=time.monotonic() - start,
                )
                if attempt < self.max_retries:
                    self._sleep(delay)

        return last_result

    def _dispatch(self, step: StepDefinition, context: dict[str, str]) -> str:  # type: ignore[valid-type]
        if isinstance(step, ToolStep):
            return self._run_tool_step(step, context)
        if isinstance(step, LLMStep):
            return self._run_llm_step(step, context)
        if isinstance(step, SubAgentStep):
            return self._run_subagent_step(step, context)
        raise TypeError(f"Unknown step type: {type(step)}")

    def _run_tool_step(self, step: ToolStep, context: dict[str, str]) -> str:
        tool_fn = self.tools.get(step.tool_name)
        if tool_fn is None:
            raise ValueError(f"Tool '{step.tool_name}' not found. Available: {list(self.tools)}")
        return tool_fn(step.tool_args, context)

    @staticmethod
    def _run_llm_step(step: LLMStep, context: dict[str, str]) -> str:
        """Execute an LLM step. Uses the Anthropic API if key is set, else mock."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            return TypedPlanExecutor._call_anthropic_llm_step(step, context, api_key)
        # Mock: render the template with context values.
        rendered = step.prompt_template
        for k, v in context.items():
            rendered = rendered.replace(f"{{{k}}}", v)
        return f"[mock LLM output for '{step.output_key}']: {rendered[:120]}"

    @staticmethod
    def _call_anthropic_llm_step(
        step: LLMStep,
        context: dict[str, str],
        api_key: str,
    ) -> str:
        try:
            import anthropic  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("pip install anthropic") from exc

        rendered = step.prompt_template
        for k, v in context.items():
            rendered = rendered.replace(f"{{{k}}}", v)

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=256,
            messages=[{"role": "user", "content": rendered}],
        )
        return message.content[0].text

    @staticmethod
    def _run_subagent_step(step: SubAgentStep, _context: dict[str, str]) -> str:
        # In production this would instantiate and run a sub-agent.
        # For this demo, return a stub.
        return f"[SubAgent '{step.agent_name}'] completed goal: {step.goal[:80]}"


# ---------------------------------------------------------------------------
# JSON round-trip helpers
# ---------------------------------------------------------------------------

def plan_to_json(plan: TypedPlan) -> str:
    return plan.model_dump_json(indent=2)


def plan_from_json(raw: str) -> TypedPlan:
    return TypedPlan.model_validate_json(raw)


def plan_to_file(plan: TypedPlan, path: str) -> None:
    import pathlib  # noqa: PLC0415
    pathlib.Path(path).write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    log.info("Plan saved to %s", path)


def plan_from_file(path: str) -> TypedPlan:
    import pathlib  # noqa: PLC0415
    raw = pathlib.Path(path).read_text(encoding="utf-8")
    plan = TypedPlan.model_validate_json(raw)
    log.info("Plan loaded from %s (%d steps)", path, len(plan.steps))
    return plan


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Build a heterogeneous plan
    plan = TypedPlan(
        goal="Research the Eiffel Tower, compute its height in feet, and summarise.",
        steps=[
            ToolStep(
                id="step_search",
                description="Search for Eiffel Tower facts.",
                tool_name="search",
                tool_args={"query": "eiffel tower"},
                expected_output="Height in metres.",
            ),
            ToolStep(
                id="step_calc",
                description="Convert 330 metres to feet.",
                depends_on=["step_search"],
                tool_name="calculate",
                tool_args={"expr": "330 * 3.28084"},
                expected_output="Height in feet.",
            ),
            LLMStep(
                id="step_summarise",
                description="Write a one-sentence summary.",
                depends_on=["step_search", "step_calc"],
                prompt_template=(
                    "Facts: {step_search}. Height in feet: {step_calc}. "
                    "Write a one-sentence summary."
                ),
                output_key="summary",
                expected_output="One-sentence summary.",
            ),
            SubAgentStep(
                id="step_publish",
                description="Publish summary to a report.",
                depends_on=["step_summarise"],
                agent_name="report-writer",
                goal="Write the Eiffel Tower summary to report.txt.",
                expected_output="Confirmation message.",
            ),
        ],
    )

    # JSON round-trip
    serialised = plan_to_json(plan)
    restored = plan_from_json(serialised)
    assert restored == plan, "Round-trip failed!"
    print("JSON round-trip: OK\n")

    # Save and reload
    plan_to_file(plan, "/tmp/eiffel_plan.json")
    loaded = plan_from_file("/tmp/eiffel_plan.json")
    assert loaded == plan
    print("File round-trip: OK\n")

    # Execute — note _flaky_calculate will fail once then succeed on retry
    executor = TypedPlanExecutor(max_retries=3)
    report = executor.execute(plan)

    print("\n=== Execution Report ===")
    print(report.summary())
    for sid, result in report.results.items():
        status = "OK" if result.success else "FAIL"
        attempts = result.attempts
        print(f"  [{status}] {sid} (attempts={attempts}): {result.output[:70]}")
```

## Exercises

1. Add a `HumanStep` variant:

   ```python
   class HumanStep(BaseModel):
       type: Literal["human"] = "human"
       id: str
       description: str
       depends_on: list[str] = ...
       prompt: str          # shown to the human operator
       expected_output: str
   ```

   Add it to `StepDefinition`. In `_dispatch`, raise `NotImplementedError` for now. Write a pytest test that loads a JSON plan containing a `HumanStep` and asserts it deserialises correctly.

2. The backoff stub (`_sleep`) is injected at construction time. Write a pytest test that injects a recording sleep function and asserts: (a) for 2 retries, sleep is called twice, and (b) the second sleep delay is twice the first.

3. Implement `plan_diff(plan_a: TypedPlan, plan_b: TypedPlan) -> dict[str, Any]` that returns a dict with keys `added`, `removed`, `modified` — each a list of step IDs. Use it to show the delta when a planner revises a plan.

4. Add a `max_step_timeout_seconds: float | None` parameter to `TypedPlanExecutor`. Use `concurrent.futures.ThreadPoolExecutor` with a timeout to enforce it per step. Return a failure `StepResult` with `error="timeout"` if the step exceeds the limit.

5. Write a pytest parametrised test over all three step types (`ToolStep`, `LLMStep`, `SubAgentStep`) that: (a) constructs a single-step plan, (b) serialises to JSON, (c) deserialises, and (d) asserts the `type` discriminator field round-trips correctly.

(Solutions: ../solutions/week-03/day-4.md)

## Self-check

- [ ] I can explain why discriminated unions are preferable to a single Step model with optional fields.
- [ ] I can write a Pydantic v2 discriminated union with three variants and verify it in a REPL.
- [ ] I can implement exponential backoff with an injectable sleep function for testability.
- [ ] I can perform a JSON round-trip on a `TypedPlan` and assert equality.
- [ ] I can describe the full week-3 architecture: ReAct (day 1) → Decomposition (day 2) → Executor (day 3) → Typed models + retry (day 4).

---

## Supplementary links

- Pydantic v2 discriminated unions — https://docs.pydantic.dev/latest/concepts/unions/#discriminated-unions
- "Exponential backoff and jitter" (AWS Architecture Blog) — https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
- Pydantic v2 `model_dump_json` / `model_validate_json` — https://docs.pydantic.dev/latest/concepts/serialization/
- LangGraph typed state documentation — https://langchain-ai.github.io/langgraph/concepts/low_level/#state
- Python `concurrent.futures` — https://docs.python.org/3.12/library/concurrent.futures.html
