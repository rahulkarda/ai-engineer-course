# Solutions — Phase 5 · Week 2 · Day 3 — The tool_use / tool_result Message Cycle

---

## Exercise 1 — `build_tool_result_message`

Construct the correct `tool_result` user message that must follow a `tool_use` assistant block.

```python
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_tool_result_message(
    tool_use_id: str,
    content: str,
    is_error: bool = False,
) -> dict[str, Any]:
    """
    Build the user-role message that returns a tool result to the model.

    Args:
        tool_use_id: The `id` from the tool_use block in the assistant's response.
                     Must match exactly — this is how the model correlates results.
        content:     Serialized result string (JSON-encode dicts/lists before passing).
        is_error:    Set True when the tool execution failed. The model will see
                     this and can decide to retry, apologise, or handle gracefully.

    Returns:
        A dict representing the user-role message with a tool_result content block.
    """
    tool_result_block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }

    # is_error is optional in the API; only include when True to keep messages lean
    if is_error:
        tool_result_block["is_error"] = True

    message: dict[str, Any] = {
        "role": "user",
        "content": [tool_result_block],
    }

    logger.debug("Built tool_result message for id=%s is_error=%s", tool_use_id, is_error)
    return message


# ── Usage examples ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Successful result
    success_msg = build_tool_result_message(
        tool_use_id="toolu_01XFDUDYJgAACTvnkyqtzM5k",
        content=json.dumps({"temperature_c": 18, "condition": "Partly cloudy"}),
    )
    print("Success:", json.dumps(success_msg, indent=2))

    # Error result
    error_msg = build_tool_result_message(
        tool_use_id="toolu_01XFDUDYJgAACTvnkyqtzM5k",
        content="City 'Atlantis' not found in weather database.",
        is_error=True,
    )
    print("Error:", json.dumps(error_msg, indent=2))
```

**Key rules:**
- `tool_use_id` must match the `id` from the `tool_use` block — the API enforces this.
- The `role` of this message is always `"user"`, not `"assistant"`.
- Multiple tool results from parallel calls can share the same user message: put multiple `tool_result` blocks in the `content` list.

---

## Exercise 2 — `dispatch_tool`

Route a `tool_use` block to the correct Python function and return the `tool_result` message.

```python
import json
import logging
import traceback
from typing import Any, Callable

logger = logging.getLogger(__name__)


def dispatch_tool(
    tool_use_block: dict[str, Any],
    tools: dict[str, Callable[..., Any]],
) -> dict[str, Any]:
    """
    Dispatch a tool_use block to the registered Python function.

    Args:
        tool_use_block: A dict with keys: type, id, name, input.
                        Typically one element of response.content.
        tools:          Map of tool_name -> callable. Each callable accepts
                        keyword arguments matching the tool's input schema.

    Returns:
        A tool_result user message (see build_tool_result_message).
        Returns is_error=True if the tool name is unknown or execution raises.
    """
    tool_name: str = tool_use_block["name"]
    tool_input: dict[str, Any] = tool_use_block["input"]
    tool_use_id: str = tool_use_block["id"]

    if tool_name not in tools:
        error_content = f"Unknown tool: '{tool_name}'. Available: {list(tools.keys())}"
        logger.warning(error_content)
        return build_tool_result_message(tool_use_id, error_content, is_error=True)

    try:
        fn = tools[tool_name]
        result = fn(**tool_input)

        # Serialize result to string; keep dicts/lists as JSON
        if isinstance(result, str):
            content = result
        else:
            content = json.dumps(result)

        logger.info("Tool '%s' succeeded. Result: %s", tool_name, content[:200])
        return build_tool_result_message(tool_use_id, content)

    except Exception as exc:
        error_msg = f"Tool '{tool_name}' raised {type(exc).__name__}: {exc}"
        logger.error(error_msg)
        logger.debug(traceback.format_exc())
        return build_tool_result_message(tool_use_id, error_msg, is_error=True)


# ── Example tool implementations (stubs) ─────────────────────────────────────

def get_weather(city: str) -> dict[str, Any]:
    return {"city": city, "temperature_c": 18, "condition": "Partly cloudy"}


def search_web(query: str, max_results: int = 5) -> list[dict[str, str]]:
    return [{"title": f"Result {i}", "url": f"https://example.com/{i}"} for i in range(max_results)]


TOOL_REGISTRY: dict[str, Callable[..., Any]] = {
    "get_weather": get_weather,
    "search_web": search_web,
}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Simulate a tool_use block from the model
    tool_use = {
        "type": "tool_use",
        "id": "toolu_01abc",
        "name": "get_weather",
        "input": {"city": "Paris"},
    }

    result = dispatch_tool(tool_use, TOOL_REGISTRY)
    print(json.dumps(result, indent=2))
```

---

## Exercise 3 — `run_single_turn` + Test

Full agentic loop for one turn: send → detect tool_use → dispatch → send results → return final text.

```python
import json
import logging
import os
from typing import Any, Callable
from unittest.mock import MagicMock, patch

import anthropic
import pytest

logger = logging.getLogger(__name__)


def run_single_turn(
    client: anthropic.Anthropic,
    messages: list[dict[str, Any]],
    tools_def: list[dict[str, Any]],
    tool_fns: dict[str, Callable[..., Any]],
    model: str = "claude-sonnet-5-5",
) -> str:
    """
    Run one full agentic turn: send messages, execute any tool calls, return final text.

    If the model returns stop_reason='tool_use':
    1. Dispatch all tool_use blocks in the response.
    2. Append the assistant turn and tool_results to messages.
    3. Send a follow-up call to get the final text response.

    Args:
        client:     Anthropic client instance.
        messages:   Initial conversation history (list of role/content dicts).
        tools_def:  Tool schemas to pass to the API.
        tool_fns:   Map of tool_name -> callable for dispatch.
        model:      Claude model ID.

    Returns:
        The final text response from the model.

    Raises:
        RuntimeError: If the model stops for an unexpected reason.
    """
    # First API call
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        tools=tools_def,
        messages=messages,
    )

    logger.info("Initial stop_reason: %s", response.stop_reason)

    # Handle tool_use
    if response.stop_reason == "tool_use":
        # Collect all tool_use blocks and results
        tool_results: list[dict[str, Any]] = []
        for block in response.content:
            if hasattr(block, "type") and block.type == "tool_use":
                tool_use_block = {
                    "type": block.type,
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
                result_msg = dispatch_tool(tool_use_block, tool_fns)
                # Extract the tool_result content block
                tool_results.extend(result_msg["content"])

        # Serialize assistant turn (SDK objects → dicts)
        assistant_content = []
        for block in response.content:
            if hasattr(block, "type"):
                if block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })
                elif block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})

        # Append assistant turn + tool results to history
        updated_messages = list(messages) + [
            {"role": "assistant", "content": assistant_content},
            {"role": "user", "content": tool_results},
        ]

        # Follow-up call for final answer
        final_response = client.messages.create(
            model=model,
            max_tokens=1024,
            tools=tools_def,
            messages=updated_messages,
        )

        logger.info("Final stop_reason: %s", final_response.stop_reason)

        # Extract text from final response
        for block in final_response.content:
            if hasattr(block, "type") and block.type == "text":
                return block.text
        return ""

    # No tool call — extract text directly
    if response.stop_reason == "end_turn":
        for block in response.content:
            if hasattr(block, "type") and block.type == "text":
                return block.text
        return ""

    raise RuntimeError(f"Unexpected stop_reason: {response.stop_reason}")


# ── Tests ─────────────────────────────────────────────────────────────────────

def _make_text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_tool_use_block(tool_id: str, name: str, tool_input: dict[str, Any]) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = tool_input
    return block


def _make_response(content: list[MagicMock], stop_reason: str) -> MagicMock:
    resp = MagicMock()
    resp.content = content
    resp.stop_reason = stop_reason
    return resp


def test_run_single_turn_with_tool_call() -> None:
    """run_single_turn dispatches a tool call and returns the final text."""
    # Arrange: first response triggers tool_use, second returns text
    tool_use_block = _make_tool_use_block(
        tool_id="toolu_test_01",
        name="get_weather",
        tool_input={"city": "Paris"},
    )
    first_response = _make_response([tool_use_block], stop_reason="tool_use")
    final_response = _make_response(
        [_make_text_block("The weather in Paris is 18°C and partly cloudy.")],
        stop_reason="end_turn",
    )

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [first_response, final_response]

    tool_fns: dict[str, Callable[..., Any]] = {
        "get_weather": lambda city: {"city": city, "temperature_c": 18, "condition": "Partly cloudy"}
    }

    tools_def: list[dict[str, Any]] = [
        {
            "name": "get_weather",
            "description": "Get weather for a city.",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "What's the weather in Paris?"}
    ]

    # Act
    result = run_single_turn(mock_client, messages, tools_def, tool_fns)

    # Assert
    assert "Paris" in result
    assert "18" in result or "cloudy" in result.lower()
    assert mock_client.messages.create.call_count == 2

    # Verify the second call includes a tool_result in the messages
    second_call_kwargs = mock_client.messages.create.call_args_list[1].kwargs
    second_messages = second_call_kwargs["messages"]
    # The last user message should contain a tool_result block
    last_user_msg = second_messages[-1]
    assert last_user_msg["role"] == "user"
    assert any(
        block.get("type") == "tool_result" or
        (isinstance(block, dict) and block.get("type") == "tool_result")
        for block in last_user_msg["content"]
    )


def test_run_single_turn_no_tool() -> None:
    """run_single_turn returns text directly when no tool is called."""
    final_response = _make_response(
        [_make_text_block("Paris is in France.")],
        stop_reason="end_turn",
    )
    mock_client = MagicMock()
    mock_client.messages.create.return_value = final_response

    result = run_single_turn(mock_client, [{"role": "user", "content": "Where is Paris?"}], [], {})

    assert result == "Paris is in France."
    mock_client.messages.create.call_count == 1


def test_run_single_turn_tool_error_is_passed_to_model() -> None:
    """run_single_turn passes is_error=True tool results back to the model gracefully."""
    tool_use_block = _make_tool_use_block("toolu_err", "unknown_tool", {})
    first_response = _make_response([tool_use_block], stop_reason="tool_use")
    final_response = _make_response(
        [_make_text_block("I couldn't find that tool.")],
        stop_reason="end_turn",
    )
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [first_response, final_response]

    result = run_single_turn(mock_client, [{"role": "user", "content": "Do something."}], [], {})

    # Model received the error and responded
    assert isinstance(result, str)
    assert mock_client.messages.create.call_count == 2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__, "-v"])
```

**Conversation flow diagram:**

```
User message
     │
     ▼
[client.messages.create] ──► stop_reason="tool_use"
     │
     ▼
dispatch_tool() for each tool_use block
     │
     ▼
Append: assistant turn + tool_result user turn
     │
     ▼
[client.messages.create] ──► stop_reason="end_turn"
     │
     ▼
Return final text
```
