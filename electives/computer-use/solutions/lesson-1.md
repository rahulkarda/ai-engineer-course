# Solutions: Lesson 1 — The Computer-Use Paradigm

## Exercise 1 — Turn budget enforcement

```python
# tests/test_lesson1_ex1.py
import pytest
from lesson_1_demo import ActionType, Action, Observation, run_agent_loop


class TurnBudgetExceeded(Exception):
    pass


def run_agent_loop_strict(
    browser,
    task: str,
    max_turns: int = 20,
) -> dict:
    """Version that raises TurnBudgetExceeded instead of returning timeout."""
    from lesson_1_demo import take_screenshot, execute_action

    def llm_never_done(obs: Observation, task: str) -> Action:
        return Action(ActionType.CLICK, {"x": 0, "y": 0})

    history = []
    for turn in range(max_turns):
        obs = take_screenshot(browser, turn)
        action = llm_never_done(obs, task)
        history.append(action)
        if action.action_type == ActionType.DONE:
            return {"status": "done", "turns": turn + 1}
        execute_action(browser, action)

    raise TurnBudgetExceeded(f"Budget of {max_turns} turns exceeded")


def test_turn_budget_exceeded():
    with pytest.raises(TurnBudgetExceeded):
        run_agent_loop_strict(object(), task="never-ending task", max_turns=2)
```

**Key points:**
- `TurnBudgetExceeded` is a custom exception — callers can catch it specifically.
- `max_turns=2` makes the test fast.
- The mock LLM (`llm_never_done`) never returns `DONE`, so the budget always triggers.

---

## Exercise 2 — Repeat-action detection

```python
def run_agent_loop_with_repeat_detection(
    browser,
    task: str,
    max_turns: int = 20,
    repeat_window: int = 3,
) -> dict:
    from lesson_1_demo import take_screenshot, execute_action, ActionType

    def llm_stuck(obs, task):
        return Action(ActionType.CLICK, {"x": 10, "y": 10})

    history: list[Action] = []
    for turn in range(max_turns):
        obs = take_screenshot(browser, turn)
        action = llm_stuck(obs, task)
        history.append(action)

        # Repeat detection
        if len(history) >= repeat_window:
            recent = history[-repeat_window:]
            if all(
                a.action_type == recent[0].action_type
                and a.params == recent[0].params
                for a in recent
            ):
                return {"status": "error", "message": "stuck in loop"}

        if action.action_type == ActionType.DONE:
            return {"status": "done"}
        execute_action(browser, action)

    return {"status": "timeout"}


def test_repeat_action_detection():
    result = run_agent_loop_with_repeat_detection(
        object(), task="test", max_turns=10, repeat_window=3
    )
    assert result["status"] == "error"
    assert "stuck" in result["message"]
```

---

## Exercise 3 — Observation history

Modify `llm_reason` to accept `observations: list[Observation]`:

```python
def llm_reason_with_history(observations: list[Observation], task: str) -> Action:
    """
    Pass last N observations to the LLM for context.
    In production: encode all screenshots as image content blocks.
    """
    current = observations[-1]
    # Only current obs used in this mock, but all are available for real LLM
    if current.turn == 0:
        return Action(ActionType.NAVIGATE, {"url": "https://example.com"})
    return Action(ActionType.DONE, {"result": "done"})
```

**Trade-off analysis:**
- Each observation includes a base64 screenshot (~50–200 KB text).
- N=1: minimal context, may miss progression state.
- N=3–5: good balance; covers recent navigation history.
- N=10+: significant token cost; may push context length limit.
- Practical ceiling: `model_context_window / avg_screenshot_tokens`; for Claude Sonnet
  with 200K context and ~2K tokens per screenshot: ~100 images max — but cost balloons.
  In practice, use N=3 and summarize older history as text.

---

## Exercise 4 — Typed action schema

```python
from pydantic import BaseModel, field_validator
from typing import Annotated, Literal
from pydantic import Field

class ClickAction(BaseModel):
    type: Literal["click"] = "click"
    x: int = Field(ge=0)
    y: int = Field(ge=0)

class TypeAction(BaseModel):
    type: Literal["type"] = "type"
    text: str

class NavigateAction(BaseModel):
    type: Literal["navigate"] = "navigate"
    url: str

    @field_validator("url")
    @classmethod
    def require_http(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("Only http/https URLs allowed")
        return v

class DoneAction(BaseModel):
    type: Literal["done"] = "done"
    result: str = ""

# Test
import pytest
from pydantic import ValidationError

def test_click_rejects_string_x():
    with pytest.raises(ValidationError):
        ClickAction(type="click", x="not-an-int", y=0)  # type: ignore

def test_navigate_rejects_javascript_scheme():
    with pytest.raises(ValidationError):
        NavigateAction(url="javascript:alert(1)")

def test_click_valid():
    a = ClickAction(x=100, y=200)
    assert a.x == 100
```
