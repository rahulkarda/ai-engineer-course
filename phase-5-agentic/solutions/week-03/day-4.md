# Solutions — Phase 5 · Week 03 · Day 4 — Plan Representation with Pydantic + Executor Loop

## Exercise 1

JSON round-trip: `plan_to_json` / `plan_from_json` and a pytest test.

```python
"""
planning/serialization.py — JSON round-trip helpers for Plan.
"""
from __future__ import annotations

import json
import logging

from .models import Plan

logger = logging.getLogger(__name__)


def plan_to_json(plan: Plan) -> str:
    """Serialize a Plan to a compact JSON string.

    Uses Pydantic v2's model_dump_json for guaranteed schema alignment.
    """
    data = plan.model_dump_json(indent=2)
    logger.debug("Serialized plan '%s' to JSON (%d bytes)", plan.goal, len(data))
    return data


def plan_from_json(data: str) -> Plan:
    """Deserialize a JSON string into a Plan, validating all fields.

    Raises:
        pydantic.ValidationError: if the JSON violates the Plan schema.
        json.JSONDecodeError: if the string is not valid JSON.
    """
    plan = Plan.model_validate_json(data)
    logger.debug("Deserialized plan '%s' with %d steps", plan.goal, len(plan.steps))
    return plan
```

**Test:**

```python
"""
tests/unit/test_serialization.py
"""
from __future__ import annotations

import pytest

from planning.models import Plan, Step
from planning.serialization import plan_from_json, plan_to_json


def make_full_plan() -> Plan:
    """Plan with three steps exercising all field types."""
    return Plan(
        goal="Full serialization test",
        steps=[
            Step(
                id="fetch",
                description="Fetch raw data",
                priority=1,
                depends_on=[],
            ),
            Step(
                id="transform",
                description="Transform fetched data",
                priority=3,
                depends_on=["fetch"],
            ),
            Step(
                id="store",
                description="Persist results",
                priority=5,
                depends_on=["fetch", "transform"],
            ),
        ],
    )


class TestPlanJsonRoundTrip:

    def test_round_trip_equality(self) -> None:
        """Serializing then deserializing a Plan produces an equal Plan."""
        original = make_full_plan()
        serialized = plan_to_json(original)
        restored = plan_from_json(serialized)

        assert restored == original

    def test_serialized_is_valid_json(self) -> None:
        """plan_to_json returns a parseable JSON string."""
        import json
        plan = make_full_plan()
        raw = plan_to_json(plan)
        parsed = json.loads(raw)
        assert parsed["goal"] == plan.goal
        assert len(parsed["steps"]) == 3

    def test_all_three_steps_preserved(self) -> None:
        """Each step's id, description, priority, and depends_on survive the round-trip."""
        original = make_full_plan()
        restored = plan_from_json(plan_to_json(original))

        original_map = {s.id: s for s in original.steps}
        restored_map = {s.id: s for s in restored.steps}

        assert set(original_map.keys()) == set(restored_map.keys())
        for step_id in original_map:
            orig = original_map[step_id]
            rest = restored_map[step_id]
            assert orig.description == rest.description
            assert orig.priority == rest.priority
            assert orig.depends_on == rest.depends_on

    def test_invalid_json_raises(self) -> None:
        """plan_from_json raises on malformed JSON."""
        import json
        with pytest.raises(json.JSONDecodeError):
            plan_from_json("{not valid json")

    def test_schema_violation_raises(self) -> None:
        """plan_from_json raises ValidationError when required fields are missing."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            plan_from_json('{"steps": []}')  # missing 'goal'
```

**Explanation:**
- `model_dump_json` / `model_validate_json` are Pydantic v2's native JSON methods — they bypass raw dict conversion and preserve all validators.
- Round-trip equality works because `Plan` is a `BaseModel` — Pydantic auto-generates `__eq__` based on field values.
- The test covers both the happy path and two failure modes (invalid JSON and schema violation).

---

## Exercise 2

Exponential backoff in `execute_with_retry`.

```python
"""
executor/retry.py — execute_with_retry with exponential backoff.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

from planning.models import Step

logger = logging.getLogger(__name__)

# Type alias for a tool execution callable
StepExecutor = Callable[[Step, dict[str, Any]], Any]


def execute_with_retry(
    step: Step,
    context: dict[str, Any],
    executor: StepExecutor,
    max_retries: int = 3,
    _sleep: Callable[[float], None] = time.sleep,  # injectable for testing
) -> Any:
    """Execute *step* via *executor*, retrying up to *max_retries* times.

    Backoff schedule: wait 2^attempt seconds between retries.
    - attempt 0 (first try): no wait
    - attempt 1 (first retry): wait 2s
    - attempt 2 (second retry): wait 4s
    - attempt 3 (third retry): wait 8s

    Args:
        step: The Step to execute.
        context: Dependency context injected from prior steps.
        executor: Callable that takes (step, context) -> output.
        max_retries: Maximum number of retry attempts after the first failure.
        _sleep: Sleep function — injectable to allow instant tests.

    Returns:
        The output of the executor on success.

    Raises:
        Exception: Re-raises the last exception when all retries are exhausted.
    """
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):  # attempt 0 = first try
        if attempt > 0:
            wait = 2 ** attempt
            logger.warning(
                "Step '%s': retry %d/%d in %ds (reason: %s)",
                step.id,
                attempt,
                max_retries,
                wait,
                last_exc,
            )
            _sleep(wait)

        try:
            result = executor(step, context)
            if attempt > 0:
                logger.info("Step '%s': succeeded on attempt %d", step.id, attempt)
            return result
        except Exception as exc:
            last_exc = exc
            logger.debug("Step '%s' attempt %d failed: %s", step.id, attempt, exc)

    # All retries exhausted
    logger.error(
        "Step '%s': all %d retries exhausted. Last error: %s",
        step.id,
        max_retries,
        last_exc,
    )
    raise last_exc  # type: ignore[misc]
```

**Explanation:**
- The loop runs `max_retries + 1` times (attempt 0 = first try, attempts 1..max_retries = retries).
- `wait = 2 ** attempt` gives 2s, 4s, 8s, ... — classic exponential backoff.
- `_sleep` defaults to `time.sleep` but is injectable: pass `_sleep=lambda _: None` in tests for instant execution.
- The last exception is always re-raised, preserving the original stack trace.

---

## Exercise 3

Per-step timeout via `concurrent.futures.ThreadPoolExecutor`.

```python
"""
executor/timeout.py — per-step timeout wrapper.
"""
from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Callable

from planning.models import Step

logger = logging.getLogger(__name__)

StepExecutor = Callable[[Step, dict[str, Any]], Any]


class StepTimeoutError(Exception):
    """Raised when a step exceeds its allocated execution time."""


def execute_with_timeout(
    step: Step,
    context: dict[str, Any],
    executor: StepExecutor,
    timeout_seconds: float = 30.0,
) -> Any:
    """Execute *step* with a wall-clock timeout.

    Wraps the tool call in a ThreadPoolExecutor. If it does not complete
    within *timeout_seconds*, raises StepTimeoutError (which is treated
    as a retryable failure by execute_with_retry).

    Args:
        step: The Step to execute.
        context: Injected dependency context.
        executor: Tool execution callable.
        timeout_seconds: Max seconds to wait for the tool to return.

    Returns:
        The tool output on success.

    Raises:
        StepTimeoutError: If the step exceeds timeout_seconds.
        Exception: Any other exception raised by the tool is re-raised.
    """
    with ThreadPoolExecutor(max_workers=1) as pool:
        future: Future[Any] = pool.submit(executor, step, context)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError:
            logger.error(
                "Step '%s' timed out after %.1fs", step.id, timeout_seconds
            )
            raise StepTimeoutError(
                f"Step '{step.id}' exceeded timeout of {timeout_seconds}s"
            )
        except Exception:
            raise  # propagate the tool's own exception unchanged


# ---------------------------------------------------------------------------
# Combined: timeout + retry
# ---------------------------------------------------------------------------

import time


def execute_with_timeout_and_retry(
    step: Step,
    context: dict[str, Any],
    executor: StepExecutor,
    timeout_seconds: float = 30.0,
    max_retries: int = 3,
    _sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Execute step with per-attempt timeout and exponential-backoff retries.

    StepTimeoutError counts as a retryable failure.
    """
    def timed_executor(s: Step, ctx: dict[str, Any]) -> Any:
        return execute_with_timeout(s, ctx, executor, timeout_seconds)

    from executor.retry import execute_with_retry
    return execute_with_retry(
        step=step,
        context=context,
        executor=timed_executor,
        max_retries=max_retries,
        _sleep=_sleep,
    )
```

**Explanation:**
- `ThreadPoolExecutor(max_workers=1)` runs the tool in a background thread. The main thread calls `future.result(timeout=...)`.
- On `FuturesTimeoutError`, the function raises `StepTimeoutError` — a custom exception so callers can distinguish timeouts from other tool failures.
- `StepTimeoutError` is a subclass of `Exception`, so it is caught by the retry loop's broad `except Exception` clause — timeouts are automatically retried.
- The background thread cannot be cancelled after a timeout (Python limitation), but the `with ThreadPoolExecutor` block ensures the pool shuts down cleanly.

---

## Exercise 4

Pytest test for retry logic: tool fails twice, succeeds on the third attempt.

```python
"""
tests/unit/test_retry.py
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, call

import pytest

from executor.retry import execute_with_retry
from planning.models import Step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_step(step_id: str = "test_step") -> Step:
    return Step(id=step_id, description="Test step")


def make_failing_then_succeeding_executor(fail_count: int) -> MagicMock:
    """Return a MagicMock that raises RuntimeError *fail_count* times then succeeds."""
    call_counter = {"n": 0}

    def _executor(step: Step, context: dict[str, Any]) -> str:
        call_counter["n"] += 1
        if call_counter["n"] <= fail_count:
            raise RuntimeError(f"Intentional failure #{call_counter['n']}")
        return "success"

    mock = MagicMock(side_effect=_executor)
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExecuteWithRetry:

    def test_fails_twice_then_succeeds(self) -> None:
        """Tool fails on attempts 0 and 1, succeeds on attempt 2 (retry #2)."""
        mock_executor = make_failing_then_succeeding_executor(fail_count=2)
        sleep_mock = MagicMock()  # instant — no real sleeping
        step = make_step()

        result = execute_with_retry(
            step=step,
            context={},
            executor=mock_executor,
            max_retries=3,
            _sleep=sleep_mock,
        )

        assert result == "success"
        assert mock_executor.call_count == 3  # attempt 0, 1, 2

    def test_retry_count_is_two(self) -> None:
        """Sleep is called exactly twice (before retry 1 and retry 2)."""
        mock_executor = make_failing_then_succeeding_executor(fail_count=2)
        sleep_mock = MagicMock()
        step = make_step()

        execute_with_retry(
            step=step,
            context={},
            executor=mock_executor,
            max_retries=3,
            _sleep=sleep_mock,
        )

        assert sleep_mock.call_count == 2, (
            f"Expected 2 sleep calls (2 retries), got {sleep_mock.call_count}"
        )

    def test_exponential_backoff_waits(self) -> None:
        """Sleep is called with exponentially increasing durations: 2s, 4s."""
        mock_executor = make_failing_then_succeeding_executor(fail_count=2)
        sleep_mock = MagicMock()

        execute_with_retry(
            step=make_step(),
            context={},
            executor=mock_executor,
            max_retries=3,
            _sleep=sleep_mock,
        )

        sleep_calls = [c.args[0] for c in sleep_mock.call_args_list]
        assert sleep_calls == [2, 4], f"Expected [2, 4], got {sleep_calls}"

    def test_succeeds_on_first_try_no_sleep(self) -> None:
        """No sleep when the tool succeeds immediately."""
        mock_executor = MagicMock(return_value="immediate")
        sleep_mock = MagicMock()

        result = execute_with_retry(
            step=make_step(),
            context={},
            executor=mock_executor,
            max_retries=3,
            _sleep=sleep_mock,
        )

        assert result == "immediate"
        sleep_mock.assert_not_called()
        mock_executor.assert_called_once()

    def test_all_retries_exhausted_raises(self) -> None:
        """When all retries fail, the last exception is re-raised."""
        always_fails = MagicMock(side_effect=RuntimeError("always fails"))
        sleep_mock = MagicMock()

        with pytest.raises(RuntimeError, match="always fails"):
            execute_with_retry(
                step=make_step(),
                context={},
                executor=always_fails,
                max_retries=2,
                _sleep=sleep_mock,
            )

        assert always_fails.call_count == 3  # initial + 2 retries

    def test_context_passed_through_to_executor(self) -> None:
        """The context dict is forwarded to each executor call unchanged."""
        ctx = {"dep_step": "dep_output"}
        received: list[dict] = []

        def capturing_executor(step: Step, context: dict[str, Any]) -> str:
            received.append(dict(context))
            return "done"

        execute_with_retry(
            step=make_step(),
            context=ctx,
            executor=capturing_executor,
            max_retries=0,
        )

        assert received == [ctx]
```

**Explanation:**
- `make_failing_then_succeeding_executor` uses a closure counter — no global state, each test gets a fresh instance.
- `sleep_mock` is passed as `_sleep` — no real sleeping in tests.
- `test_retry_count_is_two` and `test_exponential_backoff_waits` together verify both the count and the timing schedule.
- `test_all_retries_exhausted_raises` asserts that `call_count == max_retries + 1` (initial + all retries).
- `test_context_passed_through_to_executor` ensures context is forwarded without mutation.
