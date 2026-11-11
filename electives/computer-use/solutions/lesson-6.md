# Solutions: Lesson 6 — Reliability: Verification, Retries & Evaluation

## Exercise 1 — DOM predicate verification

```python
# verification_dom.py
import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Awaitable

from lesson_6_demo import VerificationCondition


class DOMVerification:
    """Factory for common DOM-based verification conditions."""

    @staticmethod
    def text_present(page: Any, selector: str, expected_text: str) -> VerificationCondition:
        async def check() -> bool:
            try:
                text = await page.locator(selector).text_content()
                return text is not None and expected_text in text
            except Exception:
                return False

        return VerificationCondition(
            description=f"'{expected_text}' present in {selector!r}",
            check=check,
        )

    @staticmethod
    def url_contains(page: Any, fragment: str) -> VerificationCondition:
        async def check() -> bool:
            return fragment in page.url

        return VerificationCondition(
            description=f"URL contains {fragment!r}",
            check=check,
        )

    @staticmethod
    def element_visible(page: Any, selector: str) -> VerificationCondition:
        async def check() -> bool:
            try:
                return await page.locator(selector).is_visible()
            except Exception:
                return False

        return VerificationCondition(
            description=f"element {selector!r} is visible",
            check=check,
        )


# Tests
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_text_present_found():
    page = MagicMock()
    locator = MagicMock()
    locator.text_content = AsyncMock(return_value="Submit your form here")
    page.locator = MagicMock(return_value=locator)

    cond = DOMVerification.text_present(page, "button", "Submit")
    assert await cond.check()


@pytest.mark.asyncio
async def test_text_present_not_found():
    page = MagicMock()
    locator = MagicMock()
    locator.text_content = AsyncMock(return_value="Cancel")
    page.locator = MagicMock(return_value=locator)

    cond = DOMVerification.text_present(page, "button", "Submit")
    assert not await cond.check()


@pytest.mark.asyncio
async def test_url_contains():
    page = MagicMock()
    page.url = "https://example.com/results?q=test"
    cond = DOMVerification.url_contains(page, "results")
    assert await cond.check()


@pytest.mark.asyncio
async def test_element_visible():
    page = MagicMock()
    locator = MagicMock()
    locator.is_visible = AsyncMock(return_value=True)
    page.locator = MagicMock(return_value=locator)

    cond = DOMVerification.element_visible(page, "#success-banner")
    assert await cond.check()
```

---

## Exercise 2 — Screenshot-based verification

```python
import hashlib


class ScreenshotHashVerification:
    """
    Verify that a page changed by comparing screenshot hashes.
    Returns True if different (page changed), False if same (no visible change).
    """

    def __init__(self, page: Any) -> None:
        self._page = page
        self._before_hash: str | None = None

    async def capture_before(self) -> None:
        raw = await self._page.screenshot(full_page=False)
        self._before_hash = hashlib.sha256(raw).hexdigest()

    def as_condition(self) -> VerificationCondition:
        async def check() -> bool:
            if self._before_hash is None:
                raise RuntimeError("call capture_before() first")
            raw = await self._page.screenshot(full_page=False)
            after_hash = hashlib.sha256(raw).hexdigest()
            return after_hash != self._before_hash

        return VerificationCondition(
            description="screenshot changed after action",
            check=check,
        )


# Tests
@pytest.mark.asyncio
async def test_screenshot_changed():
    page = MagicMock()
    call_count = [0]

    async def screenshot_side_effect(**kwargs: Any) -> bytes:
        call_count[0] += 1
        return b"before" if call_count[0] == 1 else b"after"

    page.screenshot = screenshot_side_effect
    verif = ScreenshotHashVerification(page)
    await verif.capture_before()
    cond = verif.as_condition()
    assert await cond.check()


@pytest.mark.asyncio
async def test_screenshot_unchanged():
    page = MagicMock()
    page.screenshot = AsyncMock(return_value=b"same_content")
    verif = ScreenshotHashVerification(page)
    await verif.capture_before()
    cond = verif.as_condition()
    assert not await cond.check()
```

---

## Exercise 3 — Per-task timeout

```python
@dataclass
class TaskSpec:
    task_id: str
    description: str
    starting_url: str
    success_check: Callable[[], Awaitable[bool]]
    optimal_turns: int = 5
    timeout_s: float = 60.0   # NEW


class AgentEvaluator:
    def __init__(self, agent_fn: Callable) -> None:
        self._agent_fn = agent_fn

    async def run_task(self, spec: TaskSpec) -> TaskResult:
        import time
        t0 = time.monotonic()
        try:
            result = await asyncio.wait_for(
                self._agent_fn(task=spec.description, starting_url=spec.starting_url),
                timeout=spec.timeout_s,
            )
            elapsed = time.monotonic() - t0
            success = await spec.success_check()
            return TaskResult(
                task_id=spec.task_id, success=success,
                turns_used=result.get("turns", 0),
                safety_violations=result.get("safety_violations", 0),
                verification_failures=result.get("verification_failures", 0),
                duration_s=elapsed,
            )
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - t0
            return TaskResult(
                task_id=spec.task_id, success=False, turns_used=0,
                safety_violations=0, verification_failures=0,
                duration_s=elapsed, failure_reason="timeout",
            )
        except Exception as exc:
            elapsed = time.monotonic() - t0
            return TaskResult(
                task_id=spec.task_id, success=False, turns_used=0,
                safety_violations=0, verification_failures=0,
                duration_s=elapsed, failure_reason=str(exc),
            )


# Test
@pytest.mark.asyncio
async def test_per_task_timeout():
    async def slow_agent(task: str, starting_url: str) -> dict:
        await asyncio.sleep(10)  # too slow
        return {"turns": 0, "safety_violations": 0, "verification_failures": 0}

    from unittest.mock import AsyncMock
    spec = TaskSpec(
        task_id="T001",
        description="slow task",
        starting_url="https://example.com",
        success_check=AsyncMock(return_value=True),
        timeout_s=0.05,
    )
    evaluator = AgentEvaluator(slow_agent)
    result = await evaluator.run_task(spec)
    assert not result.success
    assert result.failure_reason == "timeout"
```

---

## Exercise 4 — LLM-judge success check

```python
class LLMJudgeCheck:
    """
    Use a second LLM call to evaluate task completion from a screenshot.
    The judge model is injectable — mock it in tests.
    """

    def __init__(self, judge_client: Any, task: str, model: str = "claude-haiku-5-5") -> None:
        self._client = judge_client
        self._task = task
        self._model = model
        self._page: Any = None

    def with_page(self, page: Any) -> "LLMJudgeCheck":
        self._page = page
        return self

    async def __call__(self) -> bool:
        if self._page is None:
            raise RuntimeError("Call .with_page(page) before using as success_check")
        raw = await self._page.screenshot(full_page=False)
        import base64
        data_b64 = base64.standard_b64encode(raw).decode("ascii")

        # In production: real Anthropic call; here: mock
        response = self._client.create(
            model=self._model,
            max_tokens=10,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64",
                     "media_type": "image/png", "data": data_b64}},
                    {"type": "text", "text": (
                        f"Task: {self._task}\n"
                        "Did the agent complete this task? "
                        "Answer YES or NO only."
                    )},
                ],
            }],
        )
        text = next(
            (b.text for b in response.content if b.type == "text"), ""
        ).strip().upper()
        return text.startswith("YES")


# Test with mocked LLM
@pytest.mark.asyncio
async def test_llm_judge_yes():
    page = MagicMock()
    page.screenshot = AsyncMock(return_value=b"\x89PNG...")

    judge_client = MagicMock()
    resp = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "YES"
    resp.content = [text_block]
    judge_client.create = MagicMock(return_value=resp)

    judge = LLMJudgeCheck(judge_client, task="Search for 'test'").with_page(page)
    result = await judge()
    assert result is True


@pytest.mark.asyncio
async def test_llm_judge_no():
    page = MagicMock()
    page.screenshot = AsyncMock(return_value=b"\x89PNG...")

    judge_client = MagicMock()
    resp = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "NO"
    resp.content = [text_block]
    judge_client.create = MagicMock(return_value=resp)

    judge = LLMJudgeCheck(judge_client, task="Search for 'test'").with_page(page)
    result = await judge()
    assert result is False
```
