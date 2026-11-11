# Elective: Computer Use · Lesson 6 — Reliability: Verification, Retries & Evaluation

> Objective: Make browser agents reliable with post-action verification, retry logic, and metrics.
> Time: ~60 min · Prereqs: [[lesson-5.md]]

## Concept

A computer-use agent that takes an action and blindly moves on is fragile. The UI might
not have updated. The click might have missed the target. A network request might still
be in flight. **Verification** is the practice of confirming that each action produced
the expected effect before continuing.

Beyond individual actions, you need to **evaluate** the agent as a whole: does it
complete tasks? How many turns does it take? How often does it violate safety rules?
Without metrics, you're flying blind.

Three reliability practices:

1. **Post-action verification** — after every state-changing action, take a screenshot
   or check a DOM predicate and assert expected conditions.
2. **Retry with backoff** — when an action fails or verification fails, retry with
   exponential backoff up to a max attempt count.
3. **Evaluation harness** — run the agent on a suite of test tasks and track
   success rate, turn efficiency, and safety violation rate.

## How it works internally

**Verification strategies:**

| Strategy | How | When to use |
|---|---|---|
| Screenshot diff | Compare before/after screenshots (pixel hash or LLM description) | Generic; catches visible changes |
| DOM predicate | Assert `page.locator(selector).is_visible()` after action | Structured pages; reliable |
| URL change | Assert `page.url != url_before` after navigate | Navigation actions |
| Text presence | Assert expected text appears in page content | Form submissions, confirmations |
| Network idle | Wait for `networkidle` event | AJAX-heavy pages |

**Retry logic.** Naive retry (`for i in range(3): try...`) works but saturates quickly.
Use exponential backoff: wait `base * 2^attempt` seconds between retries, up to a cap.
Add jitter to avoid thundering-herd problems in parallel agents.

**Evaluation dimensions:**

| Metric | Formula | Target |
|---|---|---|
| Task success rate | tasks_completed / total_tasks | ≥ 0.7 for production |
| Avg turns per task | sum(turns) / completed_tasks | Lower = more efficient |
| Safety violation rate | blocked_actions / total_actions | Should be 0 for well-scoped tasks |
| Verification failure rate | failed_verifications / total_verifications | Indicates grounding issues |
| Turn efficiency | optimal_turns / actual_turns | Compare to human baseline |

**Evaluator design.** An evaluator runs the agent against a set of `TaskSpec` objects,
each with an expected outcome check. The check can be:
- DOM predicate: `page.locator(selector).text_content() == "..."`.
- URL check: `page.url == expected_url`.
- Screenshot + LLM judge: ask a second LLM to evaluate if the screenshot shows task completion.

## Worked example

```python
# lesson_6_demo.py
"""
Verification, retry, and evaluation harness for a computer-use agent.
All browser and LLM calls are mocked.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

class VerificationError(Exception):
    """Raised when post-action verification fails."""


@dataclass
class VerificationCondition:
    """One thing to check after an action."""
    description: str
    check: Callable[[], Awaitable[bool]]


async def verify_with_retry(
    conditions: list[VerificationCondition],
    max_attempts: int = 3,
    base_wait_s: float = 0.5,
) -> None:
    """
    Check all conditions with exponential backoff.
    Raises VerificationError if any condition fails after max_attempts.
    """
    for condition in conditions:
        for attempt in range(max_attempts):
            try:
                passed = await condition.check()
                if passed:
                    log.info("verified  condition=%r", condition.description)
                    break
                wait = base_wait_s * (2 ** attempt) + random.uniform(0, 0.1)
                log.warning(
                    "verification failed  condition=%r  attempt=%d  retrying in %.2fs",
                    condition.description, attempt + 1, wait,
                )
                await asyncio.sleep(wait)
            except Exception as exc:
                log.error("verification error  condition=%r  exc=%s", condition.description, exc)
        else:
            raise VerificationError(
                f"Condition '{condition.description}' failed after {max_attempts} attempts"
            )


# ---------------------------------------------------------------------------
# Retry wrapper for actions
# ---------------------------------------------------------------------------

async def with_retry(
    action_fn: Callable[[], Awaitable[Any]],
    label: str,
    max_attempts: int = 3,
    base_wait_s: float = 1.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Any:
    """
    Execute action_fn with exponential backoff.
    Non-retryable exceptions (safety blocks) should NOT be passed in retryable_exceptions.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            result = await action_fn()
            if attempt > 0:
                log.info("retry succeeded  label=%s  attempt=%d", label, attempt + 1)
            return result
        except retryable_exceptions as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                wait = base_wait_s * (2 ** attempt) + random.uniform(0, 0.2)
                log.warning(
                    "action failed  label=%s  attempt=%d  retrying in %.2fs  exc=%s",
                    label, attempt + 1, wait, exc,
                )
                await asyncio.sleep(wait)
            else:
                log.error("action failed permanently  label=%s  exc=%s", label, exc)
    raise RuntimeError(f"Action '{label}' failed after {max_attempts} attempts") from last_exc


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------

@dataclass
class TaskSpec:
    """One evaluation task."""
    task_id: str
    description: str
    starting_url: str
    success_check: Callable[[], Awaitable[bool]]
    optimal_turns: int = 5  # baseline for efficiency calculation


@dataclass
class TaskResult:
    task_id: str
    success: bool
    turns_used: int
    safety_violations: int
    verification_failures: int
    duration_s: float
    failure_reason: str = ""


@dataclass
class EvalSummary:
    total_tasks: int
    completed: int
    failed: int
    success_rate: float
    avg_turns: float
    avg_safety_violations: float
    avg_verification_failures: float
    results: list[TaskResult] = field(default_factory=list)


class AgentEvaluator:
    """
    Run an agent function on a suite of TaskSpecs and collect metrics.

    agent_fn signature:
        async (task: str, starting_url: str) -> dict[str, Any]
        Must return {"turns": int, "safety_violations": int, "verification_failures": int}
    """

    def __init__(self, agent_fn: Callable) -> None:
        self._agent_fn = agent_fn

    async def run_task(self, spec: TaskSpec) -> TaskResult:
        t0 = time.monotonic()
        try:
            result = await self._agent_fn(
                task=spec.description,
                starting_url=spec.starting_url,
            )
            elapsed = time.monotonic() - t0
            # Verify success condition
            success = await spec.success_check()
            return TaskResult(
                task_id=spec.task_id,
                success=success,
                turns_used=result.get("turns", 0),
                safety_violations=result.get("safety_violations", 0),
                verification_failures=result.get("verification_failures", 0),
                duration_s=elapsed,
                failure_reason="" if success else "success_check failed",
            )
        except Exception as exc:
            elapsed = time.monotonic() - t0
            log.error("task %s failed with exception: %s", spec.task_id, exc)
            return TaskResult(
                task_id=spec.task_id,
                success=False,
                turns_used=0,
                safety_violations=0,
                verification_failures=0,
                duration_s=elapsed,
                failure_reason=str(exc),
            )

    async def evaluate(self, specs: list[TaskSpec]) -> EvalSummary:
        results: list[TaskResult] = []
        for spec in specs:
            log.info("evaluating task  id=%s", spec.task_id)
            result = await self.run_task(spec)
            results.append(result)
            log.info(
                "task %s  success=%s  turns=%d  violations=%d",
                result.task_id, result.success, result.turns_used, result.safety_violations,
            )

        completed = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        avg_turns = sum(r.turns_used for r in completed) / max(len(completed), 1)
        avg_violations = sum(r.safety_violations for r in results) / max(len(results), 1)
        avg_verif_fails = sum(r.verification_failures for r in results) / max(len(results), 1)

        summary = EvalSummary(
            total_tasks=len(results),
            completed=len(completed),
            failed=len(failed),
            success_rate=len(completed) / max(len(results), 1),
            avg_turns=avg_turns,
            avg_safety_violations=avg_violations,
            avg_verification_failures=avg_verif_fails,
            results=results,
        )
        log.info(
            "eval complete  success=%.0f%%  avg_turns=%.1f  violations_per_task=%.2f",
            summary.success_rate * 100, summary.avg_turns, summary.avg_safety_violations,
        )
        return summary


# ---------------------------------------------------------------------------
# Demo: mock agent + evaluation
# ---------------------------------------------------------------------------

async def mock_agent(task: str, starting_url: str) -> dict[str, Any]:
    """Simulated agent that succeeds half the time for demo."""
    await asyncio.sleep(0.01)  # simulate work
    success = "search" in task.lower()
    return {
        "turns": 4 if success else 8,
        "safety_violations": 0,
        "verification_failures": 0 if success else 1,
        "completed": success,
    }


async def demo() -> None:
    # Example: verify that a navigation happened
    navigated_to_results = False

    async def check_navigation() -> bool:
        return navigated_to_results

    conditions = [VerificationCondition("page navigated to results", check_navigation)]

    # Fails first, then simulate success
    call_count = [0]
    async def flaky_check() -> bool:
        call_count[0] += 1
        return call_count[0] >= 2  # fails first attempt

    conditions2 = [VerificationCondition("results loaded", flaky_check)]
    await verify_with_retry(conditions2, max_attempts=3, base_wait_s=0.01)
    log.info("verification passed after retries")

    # Retry wrapper demo
    attempt_count = [0]
    async def flaky_action() -> str:
        attempt_count[0] += 1
        if attempt_count[0] < 3:
            raise ConnectionError("transient error")
        return "success"

    result = await with_retry(flaky_action, "search_click", max_attempts=3, base_wait_s=0.01)
    log.info("retry result: %s", result)

    # Evaluation harness demo
    specs = [
        TaskSpec(
            task_id="T001",
            description="search for 'playwright'",
            starting_url="https://sandbox.example.com",
            success_check=AsyncMock(return_value=True),  # type: ignore[arg-type]
            optimal_turns=3,
        ),
        TaskSpec(
            task_id="T002",
            description="navigate to profile page",
            starting_url="https://sandbox.example.com",
            success_check=AsyncMock(return_value=False),  # type: ignore[arg-type]
            optimal_turns=2,
        ),
    ]

    from unittest.mock import AsyncMock
    specs[0].success_check = AsyncMock(return_value=True)
    specs[1].success_check = AsyncMock(return_value=False)

    evaluator = AgentEvaluator(mock_agent)
    summary = await evaluator.evaluate(specs)
    log.info(
        "SUMMARY  total=%d  success_rate=%.0f%%  avg_turns=%.1f",
        summary.total_tasks, summary.success_rate * 100, summary.avg_turns,
    )


if __name__ == "__main__":
    asyncio.run(demo())
```

Key observations:
- `verify_with_retry` accepts a list of `VerificationCondition` objects — composable
  and testable independently of the agent loop.
- `with_retry` takes `retryable_exceptions` explicitly — **safety blocks should never
  be retried**; only transient failures (network errors, timing issues) should be.
- `AgentEvaluator` is a pure function over `TaskSpec` objects — easy to extend with
  new metrics without changing the agent.
- `success_rate`, `avg_turns`, and `safety_violations` are the three KPIs to track
  in every evaluation run.

## Exercises

1. **DOM predicate verification.** Write a `DOMVerification` class with:
   - `text_present(page, selector, expected_text)` — checks locator text content
   - `url_contains(page, fragment)` — checks `page.url`
   - `element_visible(page, selector)` — checks locator visibility
   
   Each method should be usable as a `VerificationCondition.check` callable.
   Write tests with mock pages for each method.

2. **Screenshot-based verification.** Add a `ScreenshotHashVerification` that:
   - Takes a screenshot before an action.
   - Takes a screenshot after.
   - Returns `True` if the two PNG byte hashes differ (page changed) and `False` otherwise.
   Write a test that injects two different byte strings and two identical ones.

3. **Per-task timeout.** Modify `AgentEvaluator.run_task` to enforce a `timeout_s`
   from `TaskSpec`. If the agent exceeds it, return a failed `TaskResult` with
   `failure_reason="timeout"`. Use `asyncio.wait_for`. Write a test that injects a
   slow agent function.

4. **LLM-judge success check.** Write a `LLMJudgeCheck` class whose `__call__` method
   sends a screenshot and task description to an LLM (mock it) and asks
   "Did the agent complete this task? Answer YES or NO." Parse the response.
   Show how to use it as the `success_check` in a `TaskSpec`.

(Solutions: solutions/lesson-6.md)

## Self-check

- [ ] I can implement `verify_with_retry` with exponential backoff from memory.
- [ ] I understand why safety blocks must not be in `retryable_exceptions`.
- [ ] I can list three verification strategies and when to use each.
- [ ] I can define `TaskSpec` and explain the three key evaluation metrics.
- [ ] I can wire a mock success check into `AgentEvaluator` for testing.

## Supplementary links

- [WebArena evaluation methodology](https://webarena.dev/)
- [VisualWebArena benchmark](https://jykoh.com/vwa)
- [Mind2Web dataset and evaluation](https://osu-nlp-group.github.io/Mind2Web/)
- [Exponential backoff best practices](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/)
- [Playwright waiting mechanisms](https://playwright.dev/python/docs/navigations)
