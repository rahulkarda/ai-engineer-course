# Solutions: Lesson 2 — Anthropic Computer-Use Tool

## Exercise 1 — Real API integration

```python
# lesson_2_real.py
import os
from typing import Any
import anthropic

def make_client(use_mock: bool = True) -> Any:
    if use_mock:
        return MockAnthropicClient()
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# In the loop:
def run_computer_use_loop(
    task: str,
    client: Any,
    browser: Any,
    model: str = "claude-sonnet-5-5",
    max_turns: int = 20,
    use_mock: bool = True,
) -> str:
    messages = [{"role": "user", "content": task}]
    for turn in range(max_turns):
        if use_mock:
            response = client.create(
                model=model, max_tokens=4096,
                tools=[COMPUTER_TOOL], messages=messages,
                betas=["computer-use-2025-01-01"],
            )
        else:
            response = client.beta.messages.create(
                model=model, max_tokens=4096,
                tools=[COMPUTER_TOOL], messages=messages,
                betas=["computer-use-2025-01-01"],
            )
        # ... same loop logic ...
    raise RuntimeError("max_turns exceeded")
```

**Test:**
```python
def test_loop_terminates_on_end_turn():
    client = MockAnthropicClient()  # returns end_turn after 4 calls
    browser = MockBrowser()
    result = run_computer_use_loop(
        "Find and click the search box",
        client=client, browser=browser, use_mock=True,
    )
    assert isinstance(result, str)
    assert len(result) > 0
```

---

## Exercise 2 — All action types

```python
def execute_computer_action(
    browser: MockBrowser,
    tool_use_id: str,
    action_input: dict,
) -> dict:
    action = action_input.get("action")

    if action == "screenshot":
        return build_screenshot_result(tool_use_id, browser)
    if action == "left_click":
        coord = action_input["coordinate"]
        browser.left_click(coord[0], coord[1])
        return build_screenshot_result(tool_use_id, browser)
    if action == "right_click":
        coord = action_input["coordinate"]
        browser.right_click(coord[0], coord[1])
        return build_screenshot_result(tool_use_id, browser)
    if action == "double_click":
        coord = action_input["coordinate"]
        browser.double_click(coord[0], coord[1])
        return build_screenshot_result(tool_use_id, browser)
    if action == "key":
        browser.key_press(action_input["text"])
        return build_screenshot_result(tool_use_id, browser)
    if action == "mouse_move":
        coord = action_input["coordinate"]
        browser.mouse_move(coord[0], coord[1])
        return build_empty_result(tool_use_id)
    if action == "type":
        browser.type_text(action_input["text"])
        return build_screenshot_result(tool_use_id, browser)
    if action == "scroll":
        coord = action_input.get("coordinate", [640, 400])
        browser.scroll(coord[0], coord[1],
                       action_input.get("direction", "down"),
                       action_input.get("amount", 3))
        return build_screenshot_result(tool_use_id, browser)

    return build_empty_result(tool_use_id)


# Tests
def test_right_click_calls_browser():
    browser = MockBrowser()
    browser.right_click = lambda x, y: setattr(browser, "_rc", (x, y))
    execute_computer_action(browser, "t1", {"action": "right_click", "coordinate": [100, 200]})
    assert browser._rc == (100, 200)

def test_key_calls_browser():
    browser = MockBrowser()
    pressed = []
    browser.key_press = lambda k: pressed.append(k)
    execute_computer_action(browser, "t1", {"action": "key", "text": "Return"})
    assert pressed == ["Return"]
```

---

## Exercise 3 — max_tokens handling

```python
def run_computer_use_loop_with_token_retry(
    task: str, client: Any, browser: Any,
    model: str = "claude-sonnet-5-5", max_turns: int = 20,
) -> str:
    import logging
    log = logging.getLogger(__name__)
    messages = [{"role": "user", "content": task}]
    max_tokens = 4096
    MAX_TOKENS_CAP = 16384

    for turn in range(max_turns):
        response = client.create(
            model=model, max_tokens=max_tokens,
            tools=[COMPUTER_TOOL], messages=messages,
            betas=["computer-use-2025-01-01"],
        )
        if response.stop_reason == "max_tokens":
            if max_tokens < MAX_TOKENS_CAP:
                new_max = min(max_tokens * 2, MAX_TOKENS_CAP)
                log.warning("max_tokens hit; doubling %d -> %d", max_tokens, new_max)
                max_tokens = new_max
                continue  # retry same turn
            else:
                raise RuntimeError("max_tokens cap reached; cannot continue")
        # ... rest of loop
    raise RuntimeError("max_turns exceeded")
```

**Test:**
```python
from unittest.mock import MagicMock

def test_max_tokens_retry():
    calls = []
    class SequencedMock:
        def create(self, **kwargs):
            calls.append(kwargs["max_tokens"])
            resp = MagicMock()
            if len(calls) == 1:
                resp.stop_reason = "max_tokens"
                resp.content = []
            else:
                resp.stop_reason = "end_turn"
                resp.content = [_make_text_block("done")]
            return resp

    result = run_computer_use_loop_with_token_retry(
        "test", SequencedMock(), MockBrowser(), max_turns=5
    )
    assert calls[0] == 4096
    assert calls[1] == 8192  # doubled
```

---

## Exercise 4 — Message history size

```python
def trim_screenshot_history(
    messages: list[dict], max_images: int = 3
) -> list[dict]:
    """
    Keep only the last max_images screenshot images in the message history.
    Older tool_result messages have their image content replaced with empty content.
    """
    # Collect indices of all image blocks in tool_result messages (user role)
    image_indices = []  # list of (msg_idx, content_idx)
    for mi, msg in enumerate(messages):
        if msg["role"] != "user":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for ci, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                for ii, inner in enumerate(block.get("content", [])):
                    if isinstance(inner, dict) and inner.get("type") == "image":
                        image_indices.append((mi, ci, ii))

    # Drop all but the last max_images
    to_drop = image_indices[:-max_images] if len(image_indices) > max_images else []
    for mi, ci, ii in to_drop:
        messages[mi]["content"][ci]["content"] = []  # strip image data

    return messages
```

**Why this matters:** Each base64-encoded screenshot is ~50–150 KB of text, which
translates to ~12,000–37,000 tokens. With max_history_images=3 and 1280×800 screenshots,
history adds ~36,000–111,000 tokens maximum — manageable within a 200K context window.
Without trimming, a 20-turn session can easily exceed context limits and fail mid-task.
