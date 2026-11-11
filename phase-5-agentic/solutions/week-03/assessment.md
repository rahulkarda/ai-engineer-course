# Solutions — Phase 5 · Week 03 · Assessment

---

## Part 1 — Conceptual Questions

### Q1: ReAct vs. chain-of-thought vs. pure tool-use

Chain-of-thought (CoT) generates a sequence of reasoning tokens before a final answer
but takes no external actions — it operates entirely within the model's context window.
Pure tool-use loops call tools on each turn without an explicit reasoning trace visible
to the model; the model emits a tool-call, receives the result, and emits the next
tool-call.

ReAct interleaves explicit *Thought* tokens (the reasoning trace) with *Act* tokens (tool
calls) and *Observe* tokens (tool results) in a single sequence. The key architectural
difference: the reasoning trace is part of the prompt context the model reads back on the
next step, giving it a persistent scratchpad.

Where each breaks down:
- CoT: fails when the task genuinely requires external information or side effects.
- Pure tool-use: fails when the next tool to call depends on intermediate reasoning that
  has not been made explicit — the model must "re-derive" context on each step.
- ReAct: the reasoning trace grows with each step, so very long tasks push against context
  limits. It also fails when the model produces confident but wrong Thought tokens that
  chain into cascading errors. This is the failure mode ReAct addresses that the others
  do not: it makes errors *visible and auditable* at each step, enabling the executor to
  inject correction signals before the next action.

### Q2: Plan-and-execute architecture tradeoffs

Two concrete advantages over a single ReAct loop for long-horizon tasks:

1. **Separation of planning cost from execution cost.** The planner runs once (or a small
   number of times on replan) against a potentially expensive frontier model. Individual
   execution steps can use cheaper/faster models or deterministic tools.
2. **Parallelism via dependency graph.** Once the plan is a DAG, steps with no mutual
   dependencies can run concurrently. A ReAct loop is strictly sequential because each
   observation must be read before the next action.

Primary disadvantage: the plan is created before execution starts, so it cannot adapt to
discoveries made mid-execution without a replan cycle. This hurts most on tasks that are
*exploratory* — where the optimal next step depends on the content of previous results
(e.g., open-ended research where finding one source suggests entirely different follow-up
queries). In those cases a reactive loop is more efficient.

### Q3: Task decomposition strategies

**Sequential decomposition** — steps are ordered linearly; each step depends on the
previous. Best for: pipeline tasks where each transformation's output is the next one's
input (e.g., fetch raw data -> clean -> analyse -> visualize). Simpler to implement and
reason about; no scheduling complexity.

**Dependency-graph decomposition** — steps form a DAG; independent sub-tasks are
enumerated explicitly. Best for: tasks with parallelisable sub-goals (e.g., researching
five separate topics simultaneously before writing a combined report). Enables concurrent
execution but requires a topological sort and cycle detection at plan-validation time.

### Q4: Re-plan vs. retry

Retry when: the failure is *transient* and the step specification itself is correct —
network timeout, rate limit, momentary unavailability. The distinguishing signal: the
error type is a connectivity/infrastructure exception (`ConnectionError`, `TimeoutError`,
HTTP 429/503) and no input data was rejected by the tool.

Re-plan when: the failure indicates the *step specification is wrong* — the tool rejected
the inputs, a required key was missing, or the tool returned data that makes the
downstream steps impossible. Distinguishing signal: a semantic or data error
(`ValueError`, `KeyError`, HTTP 400, tool returns empty result set) where retrying the
same inputs will produce the same failure.

Concrete signal: check whether `error.is_retryable` on a typed exception hierarchy, or
use the `classify_error` function that maps exception class to `"retriable"` vs
`"replan"`.

### Q5: Pydantic discriminated union for step types

```python
from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field

class SearchStep(BaseModel):
    step_type: Literal["search"] = "search"
    step_id: str
    query: str

class SummarizeStep(BaseModel):
    step_type: Literal["summarize"] = "summarize"
    step_id: str
    source_step_id: str

class WriteReportStep(BaseModel):
    step_type: Literal["write_report"] = "write_report"
    step_id: str
    title: str
    source_step_ids: list[str]

Step = Annotated[
    Union[SearchStep, SummarizeStep, WriteReportStep],
    Field(discriminator="step_type"),
]

class Plan(BaseModel):
    goal: str
    steps: list[Step]
```

Pydantic reads the `step_type` literal on each list element and routes deserialization
to the correct concrete class. No `isinstance` chains needed in application code.

### Q6: Topological sort for dependency execution

A DAG of steps encodes a partial order: step B must not start until step A finishes if
B depends on A's output. Topological sort (e.g., Kahn's BFS algorithm) produces a linear
extension of this partial order — a sequence in which every step appears after all its
dependencies. Any other ordering algorithm (alphabetical, insertion order) could place a
step before its dependency, causing a key-error at runtime when the step tries to read a
not-yet-produced output.

If the dependency graph contains a cycle, Kahn's algorithm stalls: after processing all
nodes with in-degree zero, remaining nodes still have in-degree > 0 but no node can be
dequeued. The detection is straightforward: after the BFS loop completes, if
`len(order) < len(all_steps)` there is a cycle. Handle this at plan-validation time (in
a `Plan.model_validator`) by raising `ValueError` with the unresolved step IDs. This
prevents partial execution of a plan that can never complete.

### Q7: Circuit-breaker pattern

The circuit breaker wraps calls to an external tool. It has three states:

- **Closed** (normal): calls pass through. On each failure the failure counter increments.
  When the counter reaches `max_failures`, the circuit transitions to Open.
- **Open** (failing fast): calls are rejected immediately with `CircuitOpenError` — no
  attempt is made. After a configured cooldown period (`reset_timeout`), the circuit
  transitions to Half-Open.
- **Half-Open** (probing): one trial call is allowed. If it succeeds, the circuit
  closes and the counter resets. If it fails, the circuit returns to Open.

Why this is preferable to retry-with-backoff for a re-planning agent: backoff only delays
repeated attempts to the *same* operation. For a re-planning agent, a tool being Open is
a *signal* — it tells the planner "do not schedule steps using this tool in the new plan
tail." The circuit breaker converts an infrastructure state into a typed, inspectable
object that the replan hook can query. Retry-with-backoff has no equivalent mechanism.

### Q8: Failure recovery design

- **TransientError (network timeout)**: retry the same step up to N times with exponential
  backoff, then re-plan if all retries exhausted. Rationale: the tool and inputs are
  correct; only connectivity was the issue. A successful retry avoids a replan cycle.
- **DataError (bad input to the tool)**: do not retry; immediately trigger re-plan with the
  failed step ID and error message. Rationale: retrying identical bad inputs wastes budget
  and always fails. The planner must produce a step with corrected inputs.
- **FatalError (upstream service down)**: abort the execution; do not re-plan. Rationale:
  if the service is down, any plan that requires it will also fail. Re-planning would
  produce a new plan with the same unavailable dependency.

Preventing infinite re-plan loops: maintain a `replan_count` counter in the executor.
Enforce a hard cap (`max_replans`, default 3). On reaching the cap, transition to abort
regardless of error class. Additionally, deduplicate re-plans: if the new plan tail is
structurally identical to the previously failed plan (same step types, same tool, same
inputs), abort immediately rather than executing a plan guaranteed to fail again.

---

## Part 2 — Coding Challenge

### Complete implementation

```python
# challenge/models.py
from __future__ import annotations

from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field


class SearchStep(BaseModel):
    step_type: Literal["search"] = "search"
    step_id: str
    query: str = ""


class SummarizeStep(BaseModel):
    step_type: Literal["summarize"] = "summarize"
    step_id: str
    text: str = ""


class WriteReportStep(BaseModel):
    step_type: Literal["write_report"] = "write_report"
    step_id: str
    content: str = ""


Step = Annotated[
    Union[SearchStep, SummarizeStep, WriteReportStep],
    Field(discriminator="step_type"),
]


class Plan(BaseModel):
    goal: str
    steps: list[Step]


class StepResult(BaseModel):
    step_id: str
    step_type: str
    status: Literal["success", "failed", "skipped"]
    output: str | None = None
    error: str | None = None


class ExecutionReport(BaseModel):
    goal: str
    steps: list[StepResult]
    replan_count: int
    success: bool
```

```python
# challenge/errors.py
class TransientError(Exception):
    """Network timeout — retriable."""

class DataError(Exception):
    """Bad input — trigger replan."""

class FatalError(Exception):
    """Upstream service down — abort."""
```

```python
# challenge/planner.py
from __future__ import annotations

import logging
import os
from typing import Protocol, runtime_checkable

from challenge.models import Plan, SearchStep, SummarizeStep, WriteReportStep

logger = logging.getLogger(__name__)

# ANTHROPIC_API_KEY consumed from environment — never hard-coded.
_API_KEY = os.getenv("ANTHROPIC_API_KEY")


@runtime_checkable
class PlannerInterface(Protocol):
    def plan(self, goal: str) -> Plan: ...
    def replan(self, goal: str, completed_ids: list[str],
               failed_id: str, error: str) -> Plan: ...


class Planner:
    """
    Dependency-injectable planner.
    In production: calls claude-sonnet-5-5 via the Anthropic SDK.
    In tests: replaced by MockPlanner.
    """

    def plan(self, goal: str) -> Plan:
        # Production implementation would call:
        #   import anthropic
        #   client = anthropic.Anthropic(api_key=_API_KEY)
        #   ... parse JSON response ...
        raise NotImplementedError("Use MockPlanner in tests or AnthropicPlanner in production.")

    def replan(
        self,
        goal: str,
        completed_ids: list[str],
        failed_id: str,
        error: str,
    ) -> Plan:
        raise NotImplementedError


class MockPlanner:
    """
    Deterministic planner for tests.
    First call: returns a 3-step plan where step 2 (SummarizeStep) will fail.
    Second call (replan): returns a 2-step tail that succeeds.
    """

    def __init__(self) -> None:
        self._calls = 0

    def plan(self, goal: str) -> Plan:
        self._calls += 1
        return Plan(
            goal=goal,
            steps=[
                SearchStep(step_id="step_1", query="Turing Award history"),
                SummarizeStep(step_id="step_2", text="search results"),
                WriteReportStep(step_id="step_3", content="placeholder"),
            ],
        )

    def replan(
        self,
        goal: str,
        completed_ids: list[str],
        failed_id: str,
        error: str,
    ) -> Plan:
        logger.warning("MockPlanner.replan: failed=%r error=%s", failed_id, error)
        return Plan(
            goal=goal,
            steps=[
                SummarizeStep(step_id="step_2_retry", text="retry text"),
                WriteReportStep(step_id="step_3", content="final report"),
            ],
        )
```

```python
# challenge/tools.py
from challenge.errors import TransientError

_summarize_call_count = 0


def tool_search(query: str) -> str:
    return f"results for '{query}'"


def tool_summarize(text: str) -> str:
    """Fails with TransientError on the first call."""
    global _summarize_call_count
    _summarize_call_count += 1
    if _summarize_call_count == 1:
        raise TransientError("upstream LLM timeout on first call")
    return f"summary of: {text}"


def tool_write_report(content: str) -> str:
    return f"report: {content}"


def reset_tools() -> None:
    """Reset stateful tool counters (call in test setup)."""
    global _summarize_call_count
    _summarize_call_count = 0
```

```python
# challenge/executor.py
from __future__ import annotations

import logging
from typing import Callable

from challenge.errors import DataError, FatalError, TransientError
from challenge.models import (
    ExecutionReport, Plan, SearchStep, StepResult,
    SummarizeStep, WriteReportStep,
)
from challenge.planner import PlannerInterface

logger = logging.getLogger(__name__)

ToolFn = Callable[..., str]


def _classify(exc: Exception) -> str:
    if isinstance(exc, TransientError):
        return "retriable"
    if isinstance(exc, DataError):
        return "replan"
    if isinstance(exc, FatalError):
        return "abort"
    return "abort"


class PlanExecuteExecutor:
    def __init__(
        self,
        planner: PlannerInterface,
        tool_search: ToolFn,
        tool_summarize: ToolFn,
        tool_write_report: ToolFn,
        max_replans: int = 3,
        max_retries: int = 2,
    ) -> None:
        self._planner = planner
        self._tools: dict[str, ToolFn] = {
            "search":       tool_search,
            "summarize":    tool_summarize,
            "write_report": tool_write_report,
        }
        self._max_replans = max_replans
        self._max_retries = max_retries

    def run(self, goal: str) -> ExecutionReport:
        plan = self._planner.plan(goal)
        completed: list[StepResult] = []
        replan_count = 0

        while plan.steps:
            step = plan.steps.pop(0)
            tool_name, kwargs = self._resolve(step)
            tool_fn = self._tools[tool_name]
            result = self._call_with_retry(step.step_id, tool_name, tool_fn, kwargs)
            completed.append(result)

            if result.status == "failed":
                if replan_count >= self._max_replans:
                    logger.error("Max replans reached; aborting.")
                    return ExecutionReport(
                        goal=goal, steps=completed,
                        replan_count=replan_count, success=False,
                    )
                logger.warning(
                    "Re-planning triggered by failed step '%s': %s",
                    step.step_id, result.error,
                )
                replan_count += 1
                completed_ids = [r.step_id for r in completed if r.status == "success"]
                new_plan = self._planner.replan(
                    goal, completed_ids, step.step_id, result.error or ""
                )
                plan = new_plan

        return ExecutionReport(
            goal=goal, steps=completed, replan_count=replan_count, success=True
        )

    def _call_with_retry(
        self,
        step_id: str,
        tool_name: str,
        tool_fn: ToolFn,
        kwargs: dict,
    ) -> StepResult:
        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                output = tool_fn(**kwargs)
                return StepResult(
                    step_id=step_id, step_type=tool_name,
                    status="success", output=output,
                )
            except Exception as exc:
                action = _classify(exc)
                last_exc = exc
                if action == "abort":
                    break
                if action == "replan":
                    break
                # retriable — retry if attempts remain
                logger.info(
                    "Step '%s' attempt %d/%d retriable: %s",
                    step_id, attempt, self._max_retries, exc,
                )
        return StepResult(
            step_id=step_id, step_type=tool_name,
            status="failed", error=str(last_exc),
        )

    @staticmethod
    def _resolve(step: SearchStep | SummarizeStep | WriteReportStep) -> tuple[str, dict]:
        if isinstance(step, SearchStep):
            return "search", {"query": step.query}
        if isinstance(step, SummarizeStep):
            return "summarize", {"text": step.text}
        if isinstance(step, WriteReportStep):
            return "write_report", {"content": step.content}
        raise TypeError(f"Unknown step type: {type(step)}")
```

```python
# tests/test_challenge.py
from __future__ import annotations

import json
import logging

import pytest

from challenge.executor import PlanExecuteExecutor
from challenge.models import ExecutionReport
from challenge.planner import MockPlanner
from challenge.tools import reset_tools, tool_search, tool_summarize, tool_write_report


@pytest.fixture(autouse=True)
def reset_tool_state() -> None:
    reset_tools()


@pytest.fixture()
def executor() -> PlanExecuteExecutor:
    return PlanExecuteExecutor(
        planner=MockPlanner(),
        tool_search=tool_search,
        tool_summarize=tool_summarize,
        tool_write_report=tool_write_report,
        max_replans=3,
        max_retries=2,
    )


def test_execution_report_type(executor: PlanExecuteExecutor) -> None:
    report = executor.run("Research the history of the Turing Award")
    assert isinstance(report, ExecutionReport)


def test_success_true(executor: PlanExecuteExecutor) -> None:
    report = executor.run("Research the history of the Turing Award")
    assert report.success is True


def test_replan_count_is_one(executor: PlanExecuteExecutor) -> None:
    report = executor.run("Research the history of the Turing Award")
    assert report.replan_count == 1


def test_replan_logged_at_warning(
    executor: PlanExecuteExecutor, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        executor.run("Research the history of the Turing Award")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("step_2" in r.message or "Re-planning" in r.message for r in warnings)


def test_report_serialises_to_json(executor: PlanExecuteExecutor) -> None:
    report = executor.run("Research the history of the Turing Award")
    data = json.loads(report.model_dump_json())
    assert data["success"] is True
    assert data["replan_count"] == 1
    assert "steps" in data


def test_all_logical_steps_present(executor: PlanExecuteExecutor) -> None:
    report = executor.run("Research the history of the Turing Award")
    step_ids = [r.step_id for r in report.steps]
    assert "step_1" in step_ids
    assert "step_2" in step_ids        # failed step still recorded
    assert "step_2_retry" in step_ids  # replan replacement
    assert "step_3" in step_ids


def test_failed_step_status_is_failed(executor: PlanExecuteExecutor) -> None:
    report = executor.run("Research the history of the Turing Award")
    step_2 = next(r for r in report.steps if r.step_id == "step_2")
    assert step_2.status == "failed"


def test_no_hardcoded_api_key() -> None:
    """Verify no hardcoded key string appears in planner module source."""
    import inspect
    from challenge import planner
    src = inspect.getsource(planner)
    assert "sk-ant-" not in src
    assert "ANTHROPIC_API_KEY" in src   # must reference env var
```

---

## Part 3 — Grading Notes

| Dimension | Weight | What earns full marks |
|---|---|---|
| Independent implementation | 40% | All 8 modules implemented from scratch; no copy-paste from course examples. Discriminated union and topological sort coded correctly. |
| Conceptual understanding | 25% | Conceptual answers are precise: distinguish ReAct from CoT architecturally, not just definitionally; explain cycle-detection at plan-validation vs runtime. |
| Debugging & problem-solving | 20% | Re-plan loop has a max_replans guard. TransientError vs DataError vs FatalError handled separately. Cycle detection raises at validation time. |
| Code quality & testing | 15% | Full type hints. `logging` not `print`. No hardcoded secrets. `os.getenv`. pytest suite with mocked LLM, all assertions specific. |

Common partial-credit scenarios:
- Implements retry but not replan (Q4 logic not transferred to code) — loses ~15%.
- Uses `print` instead of `logging` — loses 5% of code-quality marks.
- Hard-codes `"sk-ant-..."` in any file — automatic 0% on secrets compliance; flag and rotate.
- Discriminated union replaced with `isinstance` chain — architecturally valid but loses
  5% on Pydantic usage.
- No `max_replans` guard — loses 10% (Q8 design not implemented).
