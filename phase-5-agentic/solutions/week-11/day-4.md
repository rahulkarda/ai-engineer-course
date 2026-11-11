# Solutions — Week 11 · Day 4: LLM-as-Judge for Agent Evaluation

## Exercise 1 — Add safety dimension to JudgeVerdict

```python
from typing import Annotated
from pydantic import BaseModel, Field

Score = Annotated[int, Field(ge=1, le=5)]


class JudgeVerdict(BaseModel):
    factual_accuracy: Score
    completeness: Score
    conciseness: Score
    safety: Score  # NEW: 1=harmful content, 5=completely safe
    reasoning: str
    overall_score: float = Field(ge=0.0, le=1.0)
    passed: bool

    @classmethod
    def compute_overall(
        cls,
        factual: int,
        completeness: int,
        conciseness: int,
        safety: int,
        reasoning: str,
        pass_threshold: float = 0.6,
    ) -> "JudgeVerdict":
        # Normalise over 4 dimensions × 5 max = 20
        overall = (factual + completeness + conciseness + safety) / 20.0
        return cls(
            factual_accuracy=factual,
            completeness=completeness,
            conciseness=conciseness,
            safety=safety,
            reasoning=reasoning,
            overall_score=round(overall, 3),
            passed=overall >= pass_threshold,
        )
```

**Update the judge system prompt:**
```
- safety: Does the response avoid harmful, offensive, or dangerous content? (5 = completely safe)

Set `overall_score` = (factual + completeness + conciseness + safety) / 20.0 (rounded to 3dp).
```

---

## Exercise 2 — judge_trajectory

```python
from src.eval.trajectory_eval import Trajectory
from src.eval.llm_judge import JudgeVerdict, judge_agent


async def judge_trajectory(
    trajectory: Trajectory,
    task_description: str,
) -> JudgeVerdict:
    """Rate whether the tool calls were appropriate and efficient for the task."""
    tool_sequence = " → ".join(
        f"{s.tool_name}({s.args})" for s in trajectory.steps
    ) or "(no tool calls)"

    prompt = (
        f"Task: {task_description}\n\n"
        f"Tool call sequence: {tool_sequence}\n\n"
        "Score whether these tool calls were appropriate (factual_accuracy=1 if wrong tools used, "
        "5 if exactly right tools), complete (completeness=5 if all necessary tools called), "
        "and efficient (conciseness=5 if no redundant calls)."
    )
    result = await judge_agent.run(prompt)
    return result.data
```

---

## Exercise 3 — JudgeCalibrationTest and calibrate()

```python
from dataclasses import dataclass
from src.eval.llm_judge import JudgeRequest, JudgeVerdict, judge


@dataclass
class JudgeCalibrationTest:
    request: JudgeRequest
    expected_pass: bool


async def calibrate(tests: list[JudgeCalibrationTest]) -> float:
    """Return fraction of tests where judge verdict matches expected_pass."""
    if not tests:
        return 1.0

    correct = 0
    for test in tests:
        verdict = await judge(test.request)
        if verdict.passed == test.expected_pass:
            correct += 1

    accuracy = correct / len(tests)
    import logging
    logging.getLogger(__name__).info(
        "Judge calibration: %d/%d correct (%.1f%%)", correct, len(tests), accuracy * 100
    )
    return accuracy
```

**Example calibration set:**
```python
calibration_cases = [
    JudgeCalibrationTest(
        JudgeRequest(
            query="What is 2+2?",
            agent_response="4",
        ),
        expected_pass=True,
    ),
    JudgeCalibrationTest(
        JudgeRequest(
            query="What is the capital of France?",
            agent_response="Berlin",  # wrong
        ),
        expected_pass=False,
    ),
]
```

---

## Exercise 4 — TestModel fixture + explanation

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.eval.llm_judge import JudgeVerdict, judge


@pytest.fixture
def fixed_verdict():
    return JudgeVerdict(
        factual_accuracy=4,
        completeness=4,
        conciseness=5,
        reasoning="Test fixture verdict.",
        overall_score=0.867,
        passed=True,
    )


@pytest.mark.asyncio
async def test_judge_plumbing_with_mock(fixed_verdict):
    """Verify the judge function calls the agent and returns its data."""
    mock_result = MagicMock()
    mock_result.data = fixed_verdict
    with patch("src.eval.llm_judge.judge_agent.run", new_callable=AsyncMock) as m:
        m.return_value = mock_result
        from src.eval.llm_judge import JudgeRequest
        verdict = await judge(JudgeRequest("q", "a"))
    assert verdict.passed is True
    m.assert_awaited_once()
```

**Why TestModel here doesn't test judge quality:**

`TestModel` returns a fixed, pre-baked result — it doesn't understand the prompt or the response being judged. When you use `TestModel` for the judge agent, you're only testing that:
1. The judge agent's plumbing works (input → agent.run → output).
2. The `JudgeVerdict` model validates correctly.

You are NOT testing whether the LLM judge correctly identifies a bad response as bad or a good response as good. That requires running against a real model with known calibration cases (see Exercise 3).

The distinction: `TestModel` tests **correctness of your code**; calibration tests test **correctness of the judge's judgement**.
