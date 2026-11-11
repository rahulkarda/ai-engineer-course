# Elective: Computer Use · Lesson 1 — The Computer-Use Paradigm

> Objective: Understand the perception-reason-action loop, why it works, and where it fails.
> Time: ~50 min · Prereqs: [[README.md]]

## Concept

A **computer-use agent** controls a computer the same way a human does: it looks at the
screen, decides what to do, and issues input events (mouse clicks, keystrokes, scrolls).
Unlike a REST client or a scraper, it does not need a custom API — it uses the same
interface a human would.

The loop has three phases:

```
┌─────────────┐     screenshot      ┌──────────────┐
│   Computer  │ ─────────────────▶  │     LLM      │
│  (browser / │                     │  (sees image,│
│   desktop)  │ ◀────────────────── │  emits action│
└─────────────┘     action event    └──────────────┘
       │                                   ▲
       │          new screenshot            │
       └───────────────────────────────────┘
```

Each **turn** of the loop:
1. **Perceive** — take a screenshot (or read DOM/accessibility tree).
2. **Reason** — the LLM receives the screenshot + task description + history and outputs
   a structured action (or `done` / `error`).
3. **Act** — the driver executes the action and waits for the UI to settle.
4. **Verify** — optionally take a new screenshot to confirm the action had the
   expected effect before continuing.

The loop terminates when the agent emits a terminal action (`done`), exceeds a
maximum-turn budget, or a safety gate blocks an action.

## How it works internally

**Why screenshots and not DOM?** The LLM's vision capability lets it interpret rendered
UI exactly as a human does — including canvas elements, SVG charts, custom widgets, and
anything that doesn't expose accessible markup. DOM-based approaches are more reliable
for structured pages but break on SPAs with dynamic IDs and custom components.

**Coordinate grounding.** Vision-based actions target screen coordinates `(x, y)`.
The model reasons about *where* to click by interpreting the screenshot spatially. This
is called **grounding**: mapping a high-level intent ("click the Submit button") to
a concrete pixel location. Grounding errors (clicking the wrong element) are one of the
most common failure modes.

**Action latency.** Each turn requires: screenshot capture → image encode → LLM inference →
action execution → UI settle time. On a real browser with GPT-4 class models this is
3–10 seconds per turn. A 20-step task can take several minutes. Design for this.

**State explosion.** Web pages have enormous action spaces (every pixel is a candidate
click target). The LLM narrows this via language understanding, but still makes spatial
errors. Constraining the action space (e.g., only allow clicks on detected UI elements)
dramatically improves reliability.

**Key failure modes:**

| Failure mode | Description | Mitigation |
|---|---|---|
| Grounding error | Clicks wrong element | Add DOM-based element detection as fallback |
| Stale screenshot | Acts on an outdated view | Always screenshot before acting |
| Infinite loop | Repeats same action when stuck | Turn budget + repeat-action detection |
| Safety violation | Submits sensitive data | Action allowlist + confirmation gate |
| Hallucinated completion | Claims task done when it isn't | Verification step after terminal action |

## Worked example

```python
# lesson_1_demo.py
"""
Demonstrates the perception-reason-action loop structure
using a mock LLM and mock browser — no real API calls.
"""
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class ActionType(str, Enum):
    SCREENSHOT = "screenshot"
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    NAVIGATE = "navigate"
    DONE = "done"
    ERROR = "error"


@dataclass
class Action:
    action_type: ActionType
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    screenshot_b64: str  # base64-encoded PNG
    url: str
    turn: int


def take_screenshot(browser: Any, turn: int) -> Observation:
    """Capture current screen state."""
    # In production: use Playwright page.screenshot()
    # Here: return a mock observation
    log.info("turn=%d  taking screenshot", turn)
    return Observation(
        screenshot_b64="<base64-encoded-png-placeholder>",
        url=getattr(browser, "current_url", "https://example.com"),
        turn=turn,
    )


def llm_reason(observation: Observation, task: str) -> Action:
    """
    Call LLM with screenshot + task + history; return structured action.
    In production: call Anthropic computer-use tool.
    Here: simple mock that simulates progress.
    """
    log.info("turn=%d  reasoning about screenshot", observation.turn)
    if observation.turn == 0:
        return Action(ActionType.NAVIGATE, {"url": "https://example.com/search"})
    if observation.turn == 1:
        return Action(ActionType.CLICK, {"x": 320, "y": 240})
    if observation.turn == 2:
        return Action(ActionType.TYPE, {"text": "computer use agents"})
    if observation.turn == 3:
        return Action(ActionType.CLICK, {"x": 400, "y": 300})  # Submit
    return Action(ActionType.DONE, {"result": "search completed"})


def execute_action(browser: Any, action: Action) -> bool:
    """
    Execute action against the browser.
    Returns True if successful.
    """
    log.info(
        "executing  action=%s  params=%s",
        action.action_type.value,
        action.params,
    )
    # Production: Playwright page.goto / click / fill / etc.
    return True


def run_agent_loop(
    browser: Any,
    task: str,
    max_turns: int = 20,
) -> dict[str, Any]:
    """
    Core perception-reason-action loop.

    Safety: always check max_turns to prevent runaway execution.
    """
    history: list[Action] = []

    for turn in range(max_turns):
        # 1. Perceive
        obs = take_screenshot(browser, turn)

        # 2. Reason
        action = llm_reason(obs, task)
        history.append(action)

        # 3. Terminal check
        if action.action_type == ActionType.DONE:
            log.info("agent completed task: %s", action.params.get("result"))
            return {"status": "done", "turns": turn + 1, "result": action.params}
        if action.action_type == ActionType.ERROR:
            log.error("agent error: %s", action.params.get("message"))
            return {"status": "error", "turns": turn + 1, "error": action.params}

        # 4. Act
        success = execute_action(browser, action)
        if not success:
            log.warning("action failed at turn %d", turn)

    log.error("turn budget exhausted after %d turns", max_turns)
    return {"status": "timeout", "turns": max_turns}


if __name__ == "__main__":
    mock_browser = object()  # placeholder; real: Playwright Page
    result = run_agent_loop(mock_browser, task="Search for 'computer use agents'")
    log.info("result: %s", result)
```

Key observations:
- `max_turns` is a **hard safety bound** — always set it.
- The loop is synchronous here; production loops are async to avoid blocking on I/O.
- `ActionType` uses `str, Enum` so values serialize cleanly to JSON for the LLM.
- `Observation` carries `turn` so the LLM can reason about loop position.

## Exercises

1. **Turn budget enforcement.** Modify `run_agent_loop` to raise a custom
   `TurnBudgetExceeded` exception instead of returning a `"timeout"` dict.
   Write a pytest test that asserts this exception is raised when `max_turns=2`
   and the mock LLM never returns `DONE`.

2. **Repeat-action detection.** Add logic to `run_agent_loop` that detects when the
   last three actions are identical (same type + same params) and returns an
   `"error"` result with message `"stuck in loop"`.
   Write a test that injects a mock LLM always returning the same CLICK action.

3. **Observation history.** Extend the loop to pass the last N observations
   (as a list) to `llm_reason`, not just the current one. What is the memory
   trade-off of increasing N? Where is the practical ceiling?

4. **Typed action schema.** Replace the `dict[str, Any]` params field with a proper
   Pydantic discriminated union:
   `ClickAction(x: int, y: int)`, `TypeAction(text: str)`,
   `NavigateAction(url: str)`, etc. Show that invalid params (e.g., a string for `x`)
   raise a `ValidationError` at construction time.

(Solutions: solutions/lesson-1.md)

## Self-check

- [ ] I can draw the perception-reason-action loop from memory with the three phases.
- [ ] I can name five common failure modes and their mitigations.
- [ ] I understand why `max_turns` is a safety bound, not an optimization hint.
- [ ] I can explain the coordinate-grounding problem and why it matters.
- [ ] I can describe the trade-off between DOM-based and vision-based approaches.

## Supplementary links

- [Anthropic computer use intro](https://docs.anthropic.com/en/docs/build-with-claude/computer-use)
- [SeeAct paper (vision-based web agent)](https://arxiv.org/abs/2401.01614)
- [WebArena benchmark](https://webarena.dev/)
