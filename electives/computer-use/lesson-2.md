# Elective: Computer Use · Lesson 2 — Anthropic Computer-Use Tool

> Objective: Use the Anthropic computer-use tool API correctly and drive the agentic loop.
> Time: ~60 min · Prereqs: [[lesson-1.md]], `anthropic` SDK

## Concept

Anthropic exposes computer use through **three built-in beta tools** that the model
already knows how to call:

| Tool name | Purpose |
|---|---|
| `computer` | Mouse, keyboard, screenshot operations |
| `text_editor` | View/edit file contents |
| `bash` | Execute shell commands |

This lesson focuses on the `computer` tool. The others follow the same pattern.

The model does **not** actually move your mouse — it outputs a `tool_use` content block
describing the action it wants to perform. Your code executes the action and returns a
`tool_result` block containing a new screenshot. This client-side execution is the
essential safety boundary: **you control what actually runs**.

```
Claude output:  tool_use { name="computer", input={action="screenshot"} }
Your code:      capture screenshot → encode base64
Your message:   tool_result { content=[{type="image", source={...}}] }
Claude output:  tool_use { name="computer", input={action="left_click", coordinate=[320,240]} }
Your code:      execute click
...
Claude output:  text "I have completed the task."
```

## How it works internally

**Beta header.** Computer use requires the `anthropic-beta: computer-use-2025-01-01`
header. The SDK exposes this via `betas=["computer-use-2025-01-01"]` on the
`client.beta.messages.create(...)` call.

**Tool definition.** You pass the tool definition explicitly. The `display_width_px` and
`display_height_px` fields tell the model the resolution of the screen it is operating on.
The model uses these to interpret coordinates in its action outputs.

```python
computer_tool = {
    "type": "computer_20250124",
    "name": "computer",
    "display_width_px": 1280,
    "display_height_px": 800,
}
```

**Action types the model can emit** (input dict):

| `action` | Additional fields | Effect |
|---|---|---|
| `screenshot` | — | Returns current screen state |
| `left_click` | `coordinate: [x, y]` | Single left click |
| `right_click` | `coordinate: [x, y]` | Right click |
| `double_click` | `coordinate: [x, y]` | Double click |
| `middle_click` | `coordinate: [x, y]` | Middle click |
| `left_click_drag` | `start_coordinate`, `coordinate` | Click and drag |
| `type` | `text: str` | Type text at cursor |
| `key` | `text: str` | Key press, e.g. `"Return"`, `"ctrl+c"` |
| `scroll` | `coordinate`, `direction`, `amount` | Scroll |
| `mouse_move` | `coordinate: [x, y]` | Move without click |

**Tool result format.** After executing an action you return either:
- A `screenshot` type result (for `screenshot` action, or after any action that changes state).
- A text result for errors.

```python
# After screenshot action:
{
    "type": "tool_result",
    "tool_use_id": block.id,
    "content": [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "<base64-encoded-png>",
            },
        }
    ],
}
# After a click action (no visible content change yet):
{
    "type": "tool_result",
    "tool_use_id": block.id,
    "content": [],  # or a new screenshot
}
```

## Worked example

```python
# lesson_2_demo.py
"""
Computer-use agentic loop with Anthropic API.
LLM is MOCKED here — swap mock_llm_call for real client.beta.messages.create
to run against the real API with ANTHROPIC_API_KEY set.
"""
import base64
import logging
import os
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 800
MAX_TURNS = 20

COMPUTER_TOOL = {
    "type": "computer_20250124",
    "name": "computer",
    "display_width_px": DISPLAY_WIDTH,
    "display_height_px": DISPLAY_HEIGHT,
}


# ---------------------------------------------------------------------------
# Mock LLM responses (no real API calls in this demo)
# ---------------------------------------------------------------------------

def _make_tool_use_block(tool_id: str, action: str, **kwargs: Any) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = "computer"
    block.input = {"action": action, **kwargs}
    return block


def _make_text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


class MockAnthropicClient:
    """Simulates Anthropic API for testing without real API calls."""

    def __init__(self) -> None:
        self._call_count = 0

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        tools: list[dict],
        messages: list[dict],
        betas: list[str],
        system: str | None = None,
    ) -> MagicMock:
        resp = MagicMock()
        resp.stop_reason = "tool_use"

        if self._call_count == 0:
            # First call: model requests a screenshot to see the screen
            resp.content = [_make_tool_use_block("tool_1", "screenshot")]
        elif self._call_count == 1:
            # After seeing screenshot: navigate
            resp.content = [
                _make_tool_use_block(
                    "tool_2", "left_click", coordinate=[320, 240]
                )
            ]
        elif self._call_count == 2:
            # Type text
            resp.content = [
                _make_tool_use_block("tool_3", "type", text="hello world")
            ]
        else:
            # Done
            resp.stop_reason = "end_turn"
            resp.content = [_make_text_block("Task completed successfully.")]

        self._call_count += 1
        return resp


# ---------------------------------------------------------------------------
# Browser stub (no real Playwright here)
# ---------------------------------------------------------------------------

class MockBrowser:
    """Minimal browser stub for demo purposes."""

    def screenshot(self) -> bytes:
        log.info("browser: taking screenshot")
        # Return a valid 1x1 white PNG as placeholder
        return (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
            b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
            b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )

    def left_click(self, x: int, y: int) -> None:
        log.info("browser: left_click  x=%d  y=%d", x, y)

    def type_text(self, text: str) -> None:
        log.info("browser: type  text=%r", text)

    def scroll(self, x: int, y: int, direction: str, amount: int) -> None:
        log.info("browser: scroll  direction=%s  amount=%d", direction, amount)


# ---------------------------------------------------------------------------
# Tool result builder
# ---------------------------------------------------------------------------

def build_screenshot_result(tool_use_id: str, browser: MockBrowser) -> dict:
    """Capture screenshot and wrap as tool_result content block."""
    raw = browser.screenshot()
    data = base64.standard_b64encode(raw).decode("ascii")
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": data},
            }
        ],
    }


def build_empty_result(tool_use_id: str) -> dict:
    """Acknowledge action completion (no screenshot needed)."""
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": []}


# ---------------------------------------------------------------------------
# Action executor
# ---------------------------------------------------------------------------

def execute_computer_action(
    browser: MockBrowser,
    tool_use_id: str,
    action_input: dict[str, Any],
) -> dict:
    """
    Dispatch action_input to the browser and return the appropriate tool_result.
    SAFETY: this is where allowlists and approval gates would intercept.
    """
    action = action_input.get("action")
    log.info("executing  action=%s", action)

    if action == "screenshot":
        return build_screenshot_result(tool_use_id, browser)

    if action == "left_click":
        coord = action_input["coordinate"]
        browser.left_click(coord[0], coord[1])
        # Return a new screenshot so Claude sees the result
        return build_screenshot_result(tool_use_id, browser)

    if action == "type":
        browser.type_text(action_input["text"])
        return build_screenshot_result(tool_use_id, browser)

    if action == "scroll":
        coord = action_input.get("coordinate", [640, 400])
        browser.scroll(
            coord[0], coord[1],
            action_input.get("direction", "down"),
            action_input.get("amount", 3),
        )
        return build_screenshot_result(tool_use_id, browser)

    log.warning("unhandled action type: %s", action)
    return build_empty_result(tool_use_id)


# ---------------------------------------------------------------------------
# Main agentic loop
# ---------------------------------------------------------------------------

def run_computer_use_loop(
    task: str,
    client: MockAnthropicClient,
    browser: MockBrowser,
    model: str = "claude-sonnet-5-5",
    max_turns: int = MAX_TURNS,
) -> str:
    """
    Drive the computer-use agentic loop.

    Returns the final text response from the model.
    Raises RuntimeError if max_turns is exceeded.
    """
    messages: list[dict] = [{"role": "user", "content": task}]

    for turn in range(max_turns):
        log.info("--- turn %d ---", turn)

        response = client.create(
            model=model,
            max_tokens=4096,
            tools=[COMPUTER_TOOL],
            messages=messages,
            betas=["computer-use-2025-01-01"],
            system=(
                "You are a computer-use agent. "
                "Use the computer tool to complete the given task. "
                "Always take a screenshot first to see the current state."
            ),
        )

        # Accumulate assistant turn
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Extract final text
            final_text = next(
                (b.text for b in response.content if b.type == "text"), ""
            )
            log.info("agent done after %d turns: %s", turn + 1, final_text[:80])
            return final_text

        if response.stop_reason != "tool_use":
            raise RuntimeError(f"Unexpected stop_reason: {response.stop_reason}")

        # Process all tool_use blocks in this response
        tool_results: list[dict] = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = execute_computer_action(browser, block.id, block.input)
            tool_results.append(result)

        # Return tool results to the model
        messages.append({"role": "user", "content": tool_results})

    raise RuntimeError(f"Max turns ({max_turns}) exceeded without completion")


if __name__ == "__main__":
    client = MockAnthropicClient()
    browser = MockBrowser()
    result = run_computer_use_loop(
        task="Find and click the search box, then type 'hello world'",
        client=client,
        browser=browser,
    )
    log.info("final result: %s", result)
```

Key observations:
- `messages.append({"role": "assistant", "content": response.content})` appends the
  raw content list — this preserves `tool_use` blocks in history for the model to
  reference on the next turn.
- Tool results go back as `{"role": "user", "content": [tool_result_dict, ...]}`.
- `stop_reason == "end_turn"` signals the model finished; any other value needs handling.
- The `execute_computer_action` function is the **safety interception point**.

## Exercises

1. **Real API integration.** Modify `run_computer_use_loop` to accept a real
   `anthropic.Anthropic` client (use `os.getenv("ANTHROPIC_API_KEY")`).
   Keep the mock path via a `use_mock: bool = True` parameter.
   Write a test using `MockAnthropicClient` that verifies the loop terminates on
   `end_turn` and returns a non-empty string.

2. **All action types.** Extend `execute_computer_action` to handle:
   `right_click`, `double_click`, `key`, `mouse_move`.
   Write tests for each that verify the correct browser method is called.

3. **Error handling.** What should happen if the model emits `stop_reason="max_tokens"`
   mid-loop? Add handling that logs a warning and re-sends with `max_tokens` doubled
   (up to 16384). Write a test that injects a `max_tokens` stop followed by `end_turn`.

4. **Message history size.** After 10 turns, the `messages` list can become very large
   (each turn includes base64 screenshots). Add a `max_history_images: int = 3` parameter
   that keeps only the last N screenshot images in history (drop older ones but keep text).
   Explain why this matters for cost and latency.

(Solutions: solutions/lesson-2.md)

## Self-check

- [ ] I can write the `COMPUTER_TOOL` definition dict from memory.
- [ ] I can list the 10 action types and their required input fields.
- [ ] I can construct a correct `tool_result` block for screenshot and click actions.
- [ ] I understand the message structure: assistant turn with `tool_use`, user turn with `tool_result`.
- [ ] I can explain why `execute_computer_action` is the safety interception point.

## Supplementary links

- [Anthropic computer use reference](https://docs.anthropic.com/en/docs/build-with-claude/computer-use)
- [Computer use API changelog](https://docs.anthropic.com/en/docs/build-with-claude/computer-use#computer-use-beta)
- [Anthropic beta headers](https://docs.anthropic.com/en/api/beta-headers)
