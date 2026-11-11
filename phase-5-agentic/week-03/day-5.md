# Phase 5 · Week 03 · Day 5 — Re-planning & Failure Recovery

> Objective: Detect step failures in a plan-and-execute agent and dynamically re-plan instead of crashing.  ·  Time: ~1 hr  ·  Prereqs: [[phase-5-agentic/week-03/day-4]]

## Concept

Even a perfectly generated plan will fail at runtime. A file tool hits a permission error. A search API times out. The LLM refuses to generate the next sub-step because of a policy boundary. A step returns a partial result that downstream steps cannot consume.

**Failure recovery** is what separates a toy agent from a production one. There are three responses:

| Response | When to use |
|----------|-------------|
| Retry | Transient error (network timeout, rate-limit 429) — same step, same inputs |
| Re-plan | Structural error (bad assumptions, wrong tool, missing prerequisite) |
| Abort | Unrecoverable (auth failure, budget exceeded, safety refusal) |

Choosing the wrong response wastes tokens at best and causes incorrect results at worst.

### Failure modes taxonomy

**Tool errors** — the tool itself raises an exception. May be transient (timeout) or permanent (bad arguments, missing file).

**LLM refusals** — the model returns a safe-guarded response or an empty tool-call block. These are structural: retrying the same prompt will usually produce the same refusal.

**Timeout** — the tool or LLM call exceeds a deadline. Usually retriable, but with exponential back-off.

**Partial output** — the tool returns something, but the schema does not match what the next step expects (e.g., a search tool returning `None` instead of a list). Often a planning error: the planner assumed wrong output shape.

**Context exhaustion** — the context window fills mid-plan. Structural; retrying will fail again.

### When to re-plan vs. retry

```
error.is_transient()  → retry (max 3 times, exponential back-off)
error.is_structural() → re-plan
error.is_fatal()      → abort, surface to user
```

Classifying an error as transient or structural is itself a heuristic. A simple strategy:

- HTTP 429, 503, connection errors, `TimeoutError` → transient
- `ValueError`, `KeyError`, `TypeError`, LLM refusal, schema mismatch → structural
- `AuthenticationError`, safety refusal, budget exceeded → fatal

### Circuit breaker

Without a guard, an agent can re-plan indefinitely. A circuit breaker limits total re-planning attempts (`max_replans`). When the breaker trips, the agent aborts the run and returns everything completed so far as a partial `ExecutionReport`.

```
max_replans = 3
replan_count = 0

on_step_failure:
    if replan_count >= max_replans → abort
    if should_replan(error):
        replan_count += 1
        new_plan = replan(...)
    else:
        retry or abort
```

### Re-planning prompt design

The re-planner LLM call needs four pieces of context to generate a sensible new plan tail:

1. **Original goal** — what the user asked for
2. **Completed steps** — what has already been accomplished (do not redo these)
3. **Failed step** — the step that broke, with its description and tool
4. **Error** — the exact error message or refusal text

Without completed steps, the LLM will regenerate the full plan and repeat work. Without the error, it has no signal about what went wrong.

```
You are a planning agent. The following goal was partially executed.

Goal: {goal}

Completed steps:
{completed_steps}

Failed step: {failed_step}
Error: {error}

Generate only the remaining steps needed to complete the goal.
Do not repeat completed steps.
Output JSON only.
```

### Updating plan in-place vs. generating new tail

Two strategies:

- **New tail only** (preferred) — keep completed steps, replace only the remaining steps from the failure point onward. Cheaper, preserves context.
- **Full new plan** — regenerate all remaining steps from scratch. Use when the error reveals that the original goal decomposition was fundamentally wrong.

## How it works internally

The `ReplanningExecutor` wraps the base `PlanExecutor` loop with three hooks:

1. `should_replan(step, error) -> bool` — classifies the error
2. `replan(original_goal, completed, failed, error, client) -> Plan` — calls LLM for new tail
3. Circuit-breaker counter checked before each re-plan call

On each step:
```
result = execute_step(step)
if result.success:
    completed.append(step)
    continue
else:
    if circuit_breaker_tripped:
        return partial_report(completed, remaining)
    if should_replan(step, result.error):
        replan_count += 1
        plan = replan(goal, completed, step, result.error, client)
    else:
        retry_or_abort(step, result.error)
```

Structured logging records every re-plan event with: goal (first 80 chars), failed step ID, error type, replan count, new plan length.

## Worked example

```python
"""
day5_replanning.py — ReplanningExecutor with circuit-breaker and LLM-based re-planner.

Requires:
    pip install anthropic pydantic
    export ANTHROPIC_API_KEY=...
"""

from __future__ import annotations

import logging
import os
import time
from enum import Enum
from typing import Protocol

import anthropic
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


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


class ExecutionReport(BaseModel):
    goal: str
    completed_steps: list[Step]
    failed_step: Step | None
    aborted: bool
    replan_count: int
    total_steps_attempted: int


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    TimeoutError,
    ConnectionError,
)

FATAL_KEYWORDS: tuple[str, ...] = (
    "authentication",
    "unauthorized",
    "safety",
    "budget exceeded",
)


def classify_error(error: Exception) -> str:
    """Return 'transient', 'structural', or 'fatal'."""
    msg = str(error).lower()
    if any(kw in msg for kw in FATAL_KEYWORDS):
        return "fatal"
    if isinstance(error, TRANSIENT_ERRORS):
        return "transient"
    return "structural"


# ---------------------------------------------------------------------------
# LLM client protocol — allows easy mocking in tests
# ---------------------------------------------------------------------------


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str:
        """Send prompt, return raw text response."""
        ...


class AnthropicLLMClient:
    """Thin wrapper around the Anthropic Messages API."""

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


# ---------------------------------------------------------------------------
# Tool registry (toy implementations for demonstration)
# ---------------------------------------------------------------------------

ToolFn = type(lambda *_: None)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolFn] = {}

    def register(self, name: str, fn: ToolFn) -> None:
        self._tools[name] = fn

    def call(self, name: str, inputs: dict[str, str]) -> str:
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name!r}")
        return self._tools[name](**inputs)


def _toy_search(query: str) -> str:
    log.info("search(%r)", query)
    return f"Results for {query!r}: [mock result A, mock result B]"


def _toy_write(path: str, content: str) -> str:
    log.info("write_file(%r)", path)
    return f"Written {len(content)} chars to {path}"


def _toy_summarize(text: str) -> str:
    log.info("summarize(len=%d)", len(text))
    return f"Summary of {len(text)} chars: [mock summary]"


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("search", _toy_search)
    registry.register("write_file", _toy_write)
    registry.register("summarize", _toy_summarize)
    return registry


# ---------------------------------------------------------------------------
# Replanning prompt builder
# ---------------------------------------------------------------------------

REPLAN_PROMPT_TEMPLATE = """\
You are a planning agent. A plan was partially executed and a step failed.

Goal: {goal}

Completed steps (do NOT repeat these):
{completed_summary}

Failed step:
  id: {failed_id}
  description: {failed_description}
  tool: {failed_tool}
  error: {error}

Generate ONLY the remaining steps needed to finish the goal.
Return a JSON array of step objects with keys: id, description, tool, inputs.
Use tool names from this list only: search, write_file, summarize.
Output raw JSON array, no markdown fences.
"""


def build_replan_prompt(
    goal: str,
    completed: list[Step],
    failed: Step,
    error: str,
) -> str:
    if completed:
        completed_summary = "\n".join(
            f"  [{s.id}] {s.description} → {s.result or 'ok'}"
            for s in completed
        )
    else:
        completed_summary = "  (none)"

    return REPLAN_PROMPT_TEMPLATE.format(
        goal=goal,
        completed_summary=completed_summary,
        failed_id=failed.id,
        failed_description=failed.description,
        failed_tool=failed.tool,
        error=error,
    )


# ---------------------------------------------------------------------------
# Replanning executor
# ---------------------------------------------------------------------------

import json


class ReplanningExecutor:
    """
    Executes a Plan step by step.
    On failure, classifies the error and optionally re-plans using an LLM.
    A circuit breaker limits total re-planning attempts.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        llm_client: LLMClient,
        max_replans: int = 3,
        max_retries: int = 2,
        retry_delay: float = 0.5,
    ) -> None:
        self._registry = registry
        self._llm = llm_client
        self._max_replans = max_replans
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, plan: Plan) -> ExecutionReport:
        completed: list[Step] = []
        remaining: list[Step] = list(plan.steps)
        replan_count = 0
        total_attempted = 0

        while remaining:
            step = remaining.pop(0)
            total_attempted += 1

            success, result_or_error = self._execute_with_retry(step)

            if success:
                step.status = StepStatus.SUCCESS
                step.result = result_or_error
                completed.append(step)
                log.info("step %s succeeded", step.id)
                continue

            # Step failed after retries
            error_str = result_or_error
            step.status = StepStatus.FAILED
            step.error = error_str
            log.warning("step %s failed: %s", step.id, error_str)

            # Circuit breaker
            if replan_count >= self._max_replans:
                log.error(
                    "circuit breaker tripped after %d replans — aborting",
                    replan_count,
                )
                return ExecutionReport(
                    goal=plan.goal,
                    completed_steps=completed,
                    failed_step=step,
                    aborted=True,
                    replan_count=replan_count,
                    total_steps_attempted=total_attempted,
                )

            error_class = classify_error(Exception(error_str))
            if error_class == "fatal":
                log.error("fatal error on step %s — aborting", step.id)
                return ExecutionReport(
                    goal=plan.goal,
                    completed_steps=completed,
                    failed_step=step,
                    aborted=True,
                    replan_count=replan_count,
                    total_steps_attempted=total_attempted,
                )

            if self.should_replan(step, error_str):
                replan_count += 1
                log.info(
                    "replanning (attempt %d/%d) for step %s",
                    replan_count,
                    self._max_replans,
                    step.id,
                )
                new_tail = self.replan(
                    plan.goal, completed, step, error_str, self._llm
                )
                remaining = new_tail + remaining
            else:
                # Transient but exhausted retries — skip and continue
                log.warning("skipping step %s after retry exhaustion", step.id)
                step.status = StepStatus.SKIPPED
                completed.append(step)

        return ExecutionReport(
            goal=plan.goal,
            completed_steps=completed,
            failed_step=None,
            aborted=False,
            replan_count=replan_count,
            total_steps_attempted=total_attempted,
        )

    # ------------------------------------------------------------------
    # Heuristics
    # ------------------------------------------------------------------

    def should_replan(self, step: Step, error: str) -> bool:
        """Return True when the failure calls for re-planning rather than retry."""
        error_class = classify_error(Exception(error))
        # Structural errors need re-planning; transient ones were already retried
        return error_class == "structural"

    def replan(
        self,
        original_goal: str,
        completed_steps: list[Step],
        failed_step: Step,
        error: str,
        llm_client: LLMClient,
    ) -> list[Step]:
        """Ask the LLM to generate a new plan tail given the failure context."""
        prompt = build_replan_prompt(original_goal, completed_steps, failed_step, error)
        raw = llm_client.complete(prompt)
        log.info(
            "replan raw response (first 200 chars): %s",
            raw[:200].replace("\n", " "),
        )
        steps_data: list[dict] = json.loads(raw)
        return [Step(**d) for d in steps_data]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _execute_with_retry(self, step: Step) -> tuple[bool, str]:
        """Execute a single step with exponential back-off retry for transient errors."""
        last_error: str = ""
        for attempt in range(1, self._max_retries + 2):
            try:
                result = self._registry.call(step.tool, step.inputs)
                return True, result
            except Exception as exc:
                last_error = str(exc)
                error_class = classify_error(exc)
                if error_class != "transient" or attempt > self._max_retries:
                    return False, last_error
                delay = self._retry_delay * (2 ** (attempt - 1))
                log.info(
                    "transient error on step %s (attempt %d) — retrying in %.1fs",
                    step.id,
                    attempt,
                    delay,
                )
                time.sleep(delay)
        return False, last_error


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def _demo() -> None:
    registry = build_default_registry()

    # Inject a failing tool to trigger re-planning
    call_count: dict[str, int] = {"n": 0}

    def flaky_search(query: str) -> str:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ValueError("search service returned schema mismatch: expected list")
        return f"Results for {query!r}: [mock A, mock B]"

    registry.register("search", flaky_search)

    # For demo without a real API key, use a stub client
    class StubLLM:
        def complete(self, prompt: str) -> str:
            # Return a minimal replacement step
            return json.dumps(
                [
                    {
                        "id": "step-1b",
                        "description": "retry search with simplified query",
                        "tool": "search",
                        "inputs": {"query": "async patterns"},
                    }
                ]
            )

    plan = Plan(
        goal="Research async patterns and write summary",
        steps=[
            Step(
                id="step-1",
                description="search for Python async patterns",
                tool="search",
                inputs={"query": "Python async await patterns 2024"},
            ),
            Step(
                id="step-2",
                description="summarize search results",
                tool="summarize",
                inputs={"text": "placeholder"},
            ),
            Step(
                id="step-3",
                description="write summary to file",
                tool="write_file",
                inputs={"path": "summary.md", "content": "placeholder"},
            ),
        ],
    )

    executor = ReplanningExecutor(
        registry=registry,
        llm_client=StubLLM(),
        max_replans=3,
    )
    report = executor.run(plan)

    print("\n--- Execution Report ---")
    print(f"Goal:            {report.goal}")
    print(f"Aborted:         {report.aborted}")
    print(f"Replan count:    {report.replan_count}")
    print(f"Steps attempted: {report.total_steps_attempted}")
    print(f"Completed steps: {[s.id for s in report.completed_steps]}")
    if report.failed_step:
        print(f"Failed step:     {report.failed_step.id}")


if __name__ == "__main__":
    _demo()
```

Run it:
```bash
python day5_replanning.py
```

Expected output (approximate):
```
10:00:01 INFO     __main__ — step step-1 failed: search service returned schema mismatch: expected list
10:00:01 WARNING  __main__ — step step-1 failed: ...
10:00:01 INFO     __main__ — replanning (attempt 1/3) for step step-1
10:00:01 INFO     __main__ — step step-1b succeeded
10:00:01 INFO     __main__ — step step-2 succeeded
10:00:01 INFO     __main__ — step step-3 succeeded

--- Execution Report ---
Goal:            Research async patterns and write summary
Aborted:         False
Replan count:    1
Steps attempted: 4
Completed steps: ['step-1b', 'step-2', 'step-3']
```

## Exercises

1. Add a `max_retries_per_step: int` field to `ReplanningExecutor` and write a pytest test that verifies a step with a tool that always raises `TimeoutError` is retried exactly `max_retries_per_step` times before being classified for re-planning.

2. Extend `classify_error` to handle HTTP status codes: accept an optional `status_code: int | None` parameter. Status codes 429, 503 should return `"transient"`. Status code 401 should return `"fatal"`. Write parametrized pytest tests covering all three cases.

3. Add a `ReplanEvent` Pydantic model (fields: `timestamp`, `failed_step_id`, `error_class`, `replan_number`, `new_tail_length`) and emit one structured log record per re-plan event inside `ReplanningExecutor.run`. Write a test that captures log output with `caplog` and verifies the event is present.

(Solutions: ../solutions/week-03/day-5.md)

## Self-check

- [ ] I can name four distinct failure modes and give a concrete example of each.
- [ ] I can explain the difference between retry, re-plan, and abort, and state the classification heuristic used here.
- [ ] I can implement `should_replan` and `classify_error` without reference.
- [ ] I can describe what information must be passed to the LLM in a re-planning prompt and why each piece is necessary.
- [ ] I can explain the circuit-breaker pattern and implement it as a counter check.

## Supplementary links

- Anthropic model IDs: https://docs.anthropic.com/en/docs/about-claude/models
- Anthropic error handling guide: https://docs.anthropic.com/en/api/errors
- Pydantic v2 field validators: https://docs.pydantic.dev/latest/concepts/validators/
- Python `logging` cookbook: https://docs.python.org/3/howto/logging-cookbook.html
- Circuit breaker pattern (Martin Fowler): https://martinfowler.com/bliki/CircuitBreaker.html
