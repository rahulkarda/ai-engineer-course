# Solutions — Phase 5 · Week 03 · Day 5 — Re-planning & Failure Recovery

---

## Exercise 1 — `classify_error`

### Solution

```python
# src/plan_execute_agent/recovery.py
from __future__ import annotations

import logging
from typing import Literal

logger = logging.getLogger(__name__)


def classify_error(error: Exception) -> Literal["retriable", "replan", "abort"]:
    """
    Map an exception to a recovery action.

    Decision rationale:
    - ConnectionError / TimeoutError: transient network glitches; identical retry may
      succeed without changing the plan.
    - ValueError / KeyError: the step received bad data or a missing key; the tool
      itself worked but the inputs were wrong — re-planning with corrected context
      may produce a valid step.
    - Everything else: unknown territory; aborting is safer than silently corrupting
      state or spinning in an infinite replan loop.
    """
    if isinstance(error, (ConnectionError, TimeoutError)):
        logger.debug("classify_error: %s -> retriable", type(error).__name__)
        return "retriable"
    if isinstance(error, (ValueError, KeyError)):
        logger.debug("classify_error: %s -> replan", type(error).__name__)
        return "replan"
    logger.warning(
        "classify_error: %s -> abort (unrecognised error class)", type(error).__name__
    )
    return "abort"
```

### Tests

```python
# tests/unit/test_recovery.py
from __future__ import annotations

import pytest

from plan_execute_agent.recovery import classify_error


class CustomTransientError(ConnectionError):
    """Subclass — should still be retriable."""


class CustomFatalError(RuntimeError):
    """Unknown error class — should abort."""


@pytest.mark.parametrize(
    "exc, expected",
    [
        (ConnectionError("refused"), "retriable"),
        (TimeoutError("timed out"), "retriable"),
        (ValueError("bad value"), "replan"),
        (KeyError("missing key"), "replan"),
        (RuntimeError("unexpected"), "abort"),
        (Exception("generic"), "abort"),
        (OSError("disk full"), "abort"),
        (CustomTransientError("conn reset"), "retriable"),
        (CustomFatalError("unknown"), "abort"),
    ],
)
def test_classify_error(exc: Exception, expected: str) -> None:
    assert classify_error(exc) == expected


def test_classify_error_returns_literal() -> None:
    """Return value must be one of the three legal strings."""
    result = classify_error(ConnectionError())
    assert result in {"retriable", "replan", "abort"}
```

---

## Exercise 2 — `CircuitBreaker`

### Solution

```python
# src/plan_execute_agent/circuit_breaker.py
from __future__ import annotations

import logging
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitOpenError(Exception):
    """Raised when the circuit breaker is open and rejects a call."""


class CircuitBreaker:
    """
    Simple closed/open circuit breaker with no half-open state.

    Design decision: keeping state minimal (failure_count + is_open flag) makes
    the breaker easy to reason about and test. A production-grade breaker would
    add a cooldown / half-open probe, but for a re-planning agent the open state
    itself signals "trigger replan or escalate."
    """

    def __init__(self, max_failures: int) -> None:
        if max_failures < 1:
            raise ValueError("max_failures must be >= 1")
        self.max_failures: int = max_failures
        self._failure_count: int = 0
        self._is_open: bool = False

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def failure_count(self) -> int:
        return self._failure_count

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Invoke `fn(*args, **kwargs)`.

        - If the circuit is open: raise CircuitOpenError immediately.
        - If the call succeeds: reset failure counter, return result.
        - If the call fails: increment counter; if counter >= max_failures open
          the circuit, then re-raise the original exception.
        """
        if self._is_open:
            raise CircuitOpenError(
                f"Circuit open after {self._failure_count} consecutive failures"
            )
        try:
            result: T = fn(*args, **kwargs)
            self._failure_count = 0
            return result
        except Exception as exc:
            self._failure_count += 1
            logger.warning(
                "CircuitBreaker: failure %d/%d — %s: %s",
                self._failure_count,
                self.max_failures,
                type(exc).__name__,
                exc,
            )
            if self._failure_count >= self.max_failures:
                self._is_open = True
                logger.error(
                    "CircuitBreaker: circuit OPENED after %d consecutive failures",
                    self._failure_count,
                )
            raise
```

### Tests

```python
# tests/unit/test_circuit_breaker.py
from __future__ import annotations

import pytest

from plan_execute_agent.circuit_breaker import CircuitBreaker, CircuitOpenError


def _always_fail() -> None:
    raise ConnectionError("simulated failure")


def _always_succeed() -> str:
    return "ok"


def _fail_then_succeed(state: dict[str, int]) -> str:
    state["calls"] += 1
    if state["calls"] < 3:
        raise ConnectionError("not yet")
    return "success"


class TestCircuitBreakerInit:
    def test_invalid_max_failures(self) -> None:
        with pytest.raises(ValueError):
            CircuitBreaker(max_failures=0)

    def test_starts_closed(self) -> None:
        cb = CircuitBreaker(max_failures=3)
        assert not cb.is_open
        assert cb.failure_count == 0


class TestCircuitBreakerCall:
    def test_success_returns_value(self) -> None:
        cb = CircuitBreaker(max_failures=3)
        assert cb.call(_always_succeed) == "ok"

    def test_success_resets_counter(self) -> None:
        state = {"calls": 0}
        cb = CircuitBreaker(max_failures=5)
        # two failures, then a success
        with pytest.raises(ConnectionError):
            cb.call(_fail_then_succeed, state)
        with pytest.raises(ConnectionError):
            cb.call(_fail_then_succeed, state)
        result = cb.call(_fail_then_succeed, state)
        assert result == "success"
        assert cb.failure_count == 0
        assert not cb.is_open

    def test_opens_after_max_failures(self) -> None:
        cb = CircuitBreaker(max_failures=3)
        for _ in range(3):
            with pytest.raises(ConnectionError):
                cb.call(_always_fail)
        assert cb.is_open

    def test_open_circuit_raises_circuit_open_error(self) -> None:
        cb = CircuitBreaker(max_failures=1)
        with pytest.raises(ConnectionError):
            cb.call(_always_fail)
        assert cb.is_open
        with pytest.raises(CircuitOpenError):
            cb.call(_always_succeed)

    def test_exact_failure_threshold(self) -> None:
        cb = CircuitBreaker(max_failures=2)
        with pytest.raises(ConnectionError):
            cb.call(_always_fail)
        assert not cb.is_open  # one failure, still closed
        with pytest.raises(ConnectionError):
            cb.call(_always_fail)
        assert cb.is_open  # second failure opens circuit

    def test_original_exception_propagates(self) -> None:
        """The original exception (not CircuitOpenError) propagates while closed."""
        cb = CircuitBreaker(max_failures=5)
        with pytest.raises(ConnectionError, match="simulated failure"):
            cb.call(_always_fail)

    def test_kwargs_forwarded(self) -> None:
        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}"

        cb = CircuitBreaker(max_failures=3)
        assert cb.call(greet, "Alice", greeting="Hi") == "Hi, Alice"
```

---

## Exercise 3 — `replan_prompt`

### Solution

```python
# src/plan_execute_agent/prompts.py
from __future__ import annotations

from pydantic import BaseModel


class StepResult(BaseModel):
    step_id: str
    tool: str
    status: str  # "success" | "failed" | "skipped"
    output: str | None = None
    error: str | None = None


def replan_prompt(
    goal: str,
    completed: list[StepResult],
    failed: StepResult,
    error: str,
) -> str:
    """
    Build a structured prompt that asks the LLM to produce a new plan tail.

    Design decisions:
    - Completed steps are summarised briefly so the LLM knows what context exists.
    - The failed step and its error are presented verbatim so the LLM can diagnose
      the root cause and avoid the same mistake.
    - The instruction section is explicit about what the output format must be, which
      keeps downstream parsing deterministic.
    """
    completed_summary = "\n".join(
        f"  - [{r.step_id}] {r.tool}: {r.status}"
        + (f" — output: {r.output[:120]}" if r.output else "")
        for r in completed
    ) or "  (none)"

    prompt = f"""You are a planning assistant helping a plan-and-execute agent recover from a step failure.

ORIGINAL GOAL:
{goal}

COMPLETED STEPS (already executed successfully):
{completed_summary}

FAILED STEP:
  Step ID : {failed.step_id}
  Tool    : {failed.tool}
  Error   : {error}

TASK:
The execution pipeline has failed at the step above. Produce a revised plan tail — that is,
only the steps that still need to run to achieve the original goal, starting from just after
the last completed step. Do not repeat steps that already succeeded.

REQUIREMENTS:
1. Output a JSON array of step objects. Each object must have:
   - "step_id"  : string, unique identifier (e.g. "step_3_retry")
   - "tool"     : string, tool name to invoke
   - "input"    : object, tool-specific input parameters
   - "depends_on": list of step_ids this step requires (may be empty)
2. If the failed step can be retried with different inputs, include it with corrected inputs.
3. If the failed step is unrecoverable, skip it and adjust downstream steps accordingly.
4. Preserve any dependencies on already-completed steps by referencing their step_ids.
5. Output ONLY the JSON array — no prose, no markdown fences.

EXAMPLE OUTPUT:
[
  {{"step_id": "step_2_retry", "tool": "search", "input": {{"query": "revised query"}}, "depends_on": ["step_1"]}},
  {{"step_id": "step_3",       "tool": "summarize", "input": {{"text": "{{step_2_retry.output}}"}}, "depends_on": ["step_2_retry"]}},
  {{"step_id": "step_4",       "tool": "write_report", "input": {{"content": "{{step_3.output}}"}}, "depends_on": ["step_3"]}}
]
"""
    return prompt
```

### Example output (test / demo)

```python
# To illustrate what the function produces:

completed = [
    StepResult(step_id="step_1", tool="search", status="success",
               output="List of Turing Award winners 1966-2024"),
]
failed = StepResult(
    step_id="step_2", tool="summarize", status="failed",
    error="TimeoutError: upstream LLM did not respond within 30 s",
)

prompt_text = replan_prompt(
    goal="Research the history of the Turing Award and write a summary",
    completed=completed,
    failed=failed,
    error="TimeoutError: upstream LLM did not respond within 30 s",
)

# prompt_text will be:
#
# You are a planning assistant helping a plan-and-execute agent recover from a step failure.
#
# ORIGINAL GOAL:
# Research the history of the Turing Award and write a summary
#
# COMPLETED STEPS (already executed successfully):
#   - [step_1] search: success — output: List of Turing Award winners 1966-2024
#
# FAILED STEP:
#   Step ID : step_2
#   Tool    : summarize
#   Error   : TimeoutError: upstream LLM did not respond within 30 s
#
# TASK:
# The execution pipeline has failed at the step above. Produce a revised plan tail ...
# [etc.]
```

### Tests

```python
# tests/unit/test_prompts.py
from __future__ import annotations

from plan_execute_agent.prompts import StepResult, replan_prompt


def _make_completed() -> list[StepResult]:
    return [
        StepResult(step_id="step_1", tool="search", status="success",
                   output="some search results"),
    ]


def _make_failed() -> StepResult:
    return StepResult(step_id="step_2", tool="summarize", status="failed")


def test_replan_prompt_contains_goal() -> None:
    goal = "Research the Turing Award"
    prompt = replan_prompt(goal, _make_completed(), _make_failed(), "Timeout")
    assert goal in prompt


def test_replan_prompt_contains_failed_step_id() -> None:
    prompt = replan_prompt("goal", _make_completed(), _make_failed(), "Timeout")
    assert "step_2" in prompt


def test_replan_prompt_contains_error_message() -> None:
    error = "TimeoutError: 30s exceeded"
    prompt = replan_prompt("goal", _make_completed(), _make_failed(), error)
    assert error in prompt


def test_replan_prompt_contains_completed_step() -> None:
    prompt = replan_prompt("goal", _make_completed(), _make_failed(), "err")
    assert "step_1" in prompt
    assert "search" in prompt


def test_replan_prompt_empty_completed() -> None:
    prompt = replan_prompt("goal", [], _make_failed(), "err")
    assert "(none)" in prompt


def test_replan_prompt_returns_str() -> None:
    prompt = replan_prompt("goal", [], _make_failed(), "err")
    assert isinstance(prompt, str)
    assert len(prompt) > 100
```

---

## Exercise 4 — `ReplanningExecutor` integration test

### Solution

```python
# src/plan_execute_agent/replanning_executor.py
from __future__ import annotations

import logging
from typing import Any, Callable

from pydantic import BaseModel

from plan_execute_agent.prompts import StepResult
from plan_execute_agent.recovery import classify_error

logger = logging.getLogger(__name__)


class StepSpec(BaseModel):
    step_id: str
    tool: str
    input: dict[str, Any] = {}


class ExecutionReport(BaseModel):
    goal: str
    steps: list[StepResult]
    replan_count: int
    success: bool


# Type alias: a planner callable takes (goal, completed, failed, error) and
# returns a new list of StepSpecs to execute.
ReplanFn = Callable[[str, list[StepResult], StepResult, str], list[StepSpec]]


class ReplanningExecutor:
    """
    Execute a list of steps with automatic re-planning on failure.

    max_replans guards against infinite loops — if re-planning itself keeps
    failing we abort rather than spin forever.
    """

    def __init__(
        self,
        tool_registry: dict[str, Callable[..., str]],
        replan_fn: ReplanFn,
        max_replans: int = 3,
    ) -> None:
        self._tools = tool_registry
        self._replan_fn = replan_fn
        self._max_replans = max_replans

    def run(self, goal: str, steps: list[StepSpec]) -> ExecutionReport:
        completed: list[StepResult] = []
        replan_count = 0
        pending = list(steps)

        while pending:
            step = pending.pop(0)
            tool_fn = self._tools.get(step.tool)
            if tool_fn is None:
                result = StepResult(
                    step_id=step.step_id,
                    tool=step.tool,
                    status="failed",
                    error=f"No tool registered for '{step.tool}'",
                )
                completed.append(result)
                return ExecutionReport(
                    goal=goal, steps=completed, replan_count=replan_count, success=False
                )

            try:
                output = tool_fn(**step.input)
                completed.append(
                    StepResult(step_id=step.step_id, tool=step.tool,
                               status="success", output=output)
                )
            except Exception as exc:
                action = classify_error(exc)
                failed_result = StepResult(
                    step_id=step.step_id, tool=step.tool,
                    status="failed", error=str(exc),
                )

                if action == "abort":
                    logger.error("Step %s aborted: %s", step.step_id, exc)
                    completed.append(failed_result)
                    return ExecutionReport(
                        goal=goal, steps=completed,
                        replan_count=replan_count, success=False,
                    )

                if action == "replan":
                    if replan_count >= self._max_replans:
                        logger.error(
                            "Max replans (%d) reached; aborting.", self._max_replans
                        )
                        completed.append(failed_result)
                        return ExecutionReport(
                            goal=goal, steps=completed,
                            replan_count=replan_count, success=False,
                        )
                    logger.warning(
                        "Re-planning triggered by failed step '%s': %s",
                        step.step_id, exc,
                    )
                    replan_count += 1
                    completed.append(failed_result)
                    new_steps = self._replan_fn(
                        goal, completed, failed_result, str(exc)
                    )
                    pending = new_steps + pending
                    continue

                # retriable — put the step back at the front
                logger.info(
                    "Step %s failed with retriable error; retrying once.", step.step_id
                )
                pending.insert(0, step)

        return ExecutionReport(
            goal=goal, steps=completed, replan_count=replan_count, success=True
        )
```

### Integration test

```python
# tests/integration/test_replanning_executor.py
from __future__ import annotations

import pytest

from plan_execute_agent.replanning_executor import (
    ExecutionReport,
    ReplanningExecutor,
    StepSpec,
)
from plan_execute_agent.prompts import StepResult


# --- Tool implementations (deterministic, no network) ---

def tool_search(**kwargs: object) -> str:
    return f"search results for: {kwargs.get('query', '')}"


def tool_summarize(**kwargs: object) -> str:
    return f"summary of: {kwargs.get('text', '')}"


def tool_write(**kwargs: object) -> str:
    return f"report written: {kwargs.get('content', '')}"


def tool_publish(**kwargs: object) -> str:
    return f"published: {kwargs.get('report', '')}"


# --- Failing tool: raises ValueError on first invocation, succeeds on retry ---

class _FailOnce:
    def __init__(self) -> None:
        self._called = False

    def __call__(self, **kwargs: object) -> str:
        if not self._called:
            self._called = True
            raise ValueError("bad input on first call — needs replan")
        return f"summarized (second attempt): {kwargs.get('text', '')}"


_fail_once_instance = _FailOnce()


def tool_summarize_failing(**kwargs: object) -> str:
    return _fail_once_instance(**kwargs)


# --- Replan function ---

def mock_replan(
    goal: str,
    completed: list[StepResult],
    failed: StepResult,
    error: str,
) -> list[StepSpec]:
    """Return two replacement steps (retry summarize + write the report)."""
    return [
        StepSpec(step_id="step_2_retry", tool="summarize_ok",
                 input={"text": "Turing Award history"}),
        StepSpec(step_id="step_3",       tool="write",
                 input={"content": "{{step_2_retry.output}}"}),
        StepSpec(step_id="step_4",       tool="publish",
                 input={"report": "{{step_3.output}}"}),
    ]


# --- Fixture ---

@pytest.fixture()
def executor() -> ReplanningExecutor:
    return ReplanningExecutor(
        tool_registry={
            "search":        tool_search,
            "summarize":     tool_summarize_failing,
            "summarize_ok":  tool_summarize,
            "write":         tool_write,
            "publish":       tool_publish,
        },
        replan_fn=mock_replan,
        max_replans=3,
    )


@pytest.fixture()
def initial_steps() -> list[StepSpec]:
    return [
        StepSpec(step_id="step_1", tool="search",    input={"query": "Turing Award history"}),
        StepSpec(step_id="step_2", tool="summarize", input={"text": "search results"}),
    ]


# --- Tests ---

def test_replanning_fires_on_step_2_failure(
    executor: ReplanningExecutor, initial_steps: list[StepSpec]
) -> None:
    report = executor.run("Research Turing Award", initial_steps)
    assert report.replan_count == 1, "replan should fire exactly once"


def test_all_four_logical_steps_complete(
    executor: ReplanningExecutor, initial_steps: list[StepSpec]
) -> None:
    report = executor.run("Research Turing Award", initial_steps)
    # step_1 succeeded, step_2 failed, then step_2_retry / step_3 / step_4 succeeded
    step_ids = [r.step_id for r in report.steps]
    assert "step_1"       in step_ids
    assert "step_2"       in step_ids   # recorded as failed
    assert "step_2_retry" in step_ids
    assert "step_3"       in step_ids
    assert "step_4"       in step_ids


def test_final_report_success_true(
    executor: ReplanningExecutor, initial_steps: list[StepSpec]
) -> None:
    report = executor.run("Research Turing Award", initial_steps)
    assert report.success is True


def test_step_2_recorded_as_failed(
    executor: ReplanningExecutor, initial_steps: list[StepSpec]
) -> None:
    report = executor.run("Research Turing Award", initial_steps)
    step_2 = next(r for r in report.steps if r.step_id == "step_2")
    assert step_2.status == "failed"


def test_report_serialises_to_json(
    executor: ReplanningExecutor, initial_steps: list[StepSpec]
) -> None:
    report = executor.run("Research Turing Award", initial_steps)
    json_str = report.model_dump_json()
    assert "replan_count" in json_str
    assert "success" in json_str


def test_report_type(
    executor: ReplanningExecutor, initial_steps: list[StepSpec]
) -> None:
    report = executor.run("Research Turing Award", initial_steps)
    assert isinstance(report, ExecutionReport)
```
