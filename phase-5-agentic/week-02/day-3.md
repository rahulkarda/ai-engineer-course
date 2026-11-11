# Phase 5 · Week 02 · Day 3 — The tool_use / tool_result Message Cycle

> Objective: Implement the full Anthropic multi-turn tool_use/tool_result conversation cycle, from initial request to final text response.  ·  Time: ~1 hr  ·  Prereqs: [[phase-5-agentic/week-02/day-2]]

## Concept

Tool calling is a **multi-turn conversation**. The LLM does not call tools directly — it requests them by emitting a `tool_use` content block. Your program executes the tool, then continues the conversation by appending a `tool_result` message. Only after seeing the result does the model generate its final response.

### The full message cycle

```
Turn 1 (you → API):
  messages = [{"role": "user", "content": "What's the weather in Berlin?"}]

Turn 1 (API → you):
  stop_reason = "tool_use"
  content = [tool_use block: {name: "get_weather", input: {city: "Berlin"}}]

  ← Your program executes get_weather(city="Berlin") → {"temp": 18, ...}

Turn 2 (you → API):
  messages = [
    {"role": "user",      "content": "What's the weather in Berlin?"},
    {"role": "assistant", "content": [<tool_use block from turn 1>]},
    {"role": "user",      "content": [<tool_result block with the result>]},
  ]

Turn 2 (API → you):
  stop_reason = "end_turn"
  content = [{"type": "text", "text": "It's 18°C and partly cloudy in Berlin."}]
```

Three things to notice:

1. The **assistant message** must include the original `tool_use` block verbatim — the API uses the `id` to match results.
2. The **tool_result** is sent as a `"user"` role message (not `"assistant"`), with type `"tool_result"` and the matching `tool_use_id`.
3. The cycle can repeat: if the assistant's turn 2 response contains another `tool_use` block, you dispatch again. The conversation ends only when `stop_reason == "end_turn"` (or `"max_tokens"`).

### Detecting tool_use in a response

```python
if response.stop_reason == "tool_use":
    tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
    # dispatch each block
```

A single response can contain **multiple** `tool_use` blocks (parallel tool calling). Always iterate — never assume exactly one.

### The tool_result content block

```python
{
    "type": "tool_result",
    "tool_use_id": "<same id as the tool_use block>",  # required
    "content": "<string or list of content blocks>",   # the result
    # "is_error": True,  # optional — set if the tool raised an exception
}
```

`content` is typically a JSON string: `json.dumps(your_result_dict)`. The model reads it as text.

## How it works internally

### Message list invariants

The Anthropic API enforces strict ordering rules on the message list:

| Rule | Consequence of violation |
|---|---|
| Roles must alternate user/assistant | API error |
| Every `tool_use` block must be followed by a matching `tool_result` in the next user turn | API error |
| `tool_use_id` in `tool_result` must match a `tool_use` block in the preceding assistant turn | API error |

Your loop must maintain these invariants. The safest approach: append to the message list in order, never mutate earlier entries.

### Multiple tools in one turn

If the assistant's content has two `tool_use` blocks, the next user turn must contain two `tool_result` blocks — one per `tool_use_id`. They can be dispatched concurrently (e.g. with `asyncio.gather`); both results must appear before the next API call.

### Stop reasons

| `stop_reason` | Meaning | Action |
|---|---|---|
| `"tool_use"` | Model wants to call one or more tools | Dispatch, append result, call API again |
| `"end_turn"` | Model is done | Extract text, return to caller |
| `"max_tokens"` | Context limit hit | Handle error / truncate |
| `"stop_sequence"` | A stop sequence was matched | Treat like `end_turn` |

## Worked example

Complete mock-based example showing all four message objects and a two-turn cycle.

```python
# phase-5-agentic/week-02/examples/day3_tool_cycle.py
"""
Full tool_use / tool_result message cycle — mocked LLM responses.
Run: python day3_tool_cycle.py
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any
from unittest.mock import MagicMock

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------


def get_weather(city: str, units: str = "celsius") -> dict[str, Any]:
    """Stubbed weather tool."""
    logger.info("TOOL get_weather: city=%s units=%s", city, units)
    return {
        "city": city,
        "temperature": 18,
        "units": units,
        "condition": "partly cloudy",
    }


TOOL_REGISTRY: dict[str, Any] = {
    "get_weather": get_weather,
}

# ---------------------------------------------------------------------------
# Mock factory helpers
# ---------------------------------------------------------------------------


def _make_tool_use_block(
    tool_use_id: str, name: str, tool_input: dict[str, Any]
) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_use_id
    block.name = name
    block.input = tool_input
    return block


def _make_text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_response(
    stop_reason: str, content: list[MagicMock]
) -> MagicMock:
    resp = MagicMock()
    resp.stop_reason = stop_reason
    resp.content = content
    return resp


# ---------------------------------------------------------------------------
# The agent loop
# ---------------------------------------------------------------------------


class ToolCycleAgent:
    """
    Demonstrates the full tool_use / tool_result cycle.
    Pass a list of mock responses to simulate successive API calls.
    """

    MAX_TURNS: int = 10  # safety limit

    def __init__(
        self,
        tools: list[dict[str, Any]],
        mock_responses: list[MagicMock] | None = None,
    ) -> None:
        self.tools = tools
        self._mock_responses = mock_responses or []
        self._mock_call_count = 0

    def run(self, user_message: str) -> str:
        """
        Run the agent until end_turn or MAX_TURNS.
        Returns the final text response.
        """
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_message}
        ]
        logger.info("=== Starting agent loop ===")
        logger.info("User: %s", user_message)

        for turn in range(self.MAX_TURNS):
            logger.info("--- Turn %d ---", turn + 1)
            response = self._call_api(messages)
            logger.info("stop_reason: %s", response.stop_reason)

            if response.stop_reason == "end_turn":
                # Extract the final text
                text = self._extract_text(response)
                logger.info("Final response: %s", text)
                return text

            if response.stop_reason == "tool_use":
                # 1. Append assistant message (with tool_use blocks)
                assistant_content = self._content_to_list(response.content)
                messages.append({"role": "assistant", "content": assistant_content})
                logger.info(
                    "Appended assistant message with %d content block(s)",
                    len(assistant_content),
                )

                # 2. Dispatch all tool_use blocks and collect tool_result blocks
                tool_result_blocks = self._dispatch_all(response.content)

                # 3. Append user message with tool_result blocks
                messages.append({"role": "user", "content": tool_result_blocks})
                logger.info(
                    "Appended %d tool_result block(s)", len(tool_result_blocks)
                )
                logger.info("Current message list length: %d", len(messages))
                continue

            # Unexpected stop reason
            raise RuntimeError(f"Unexpected stop_reason: {response.stop_reason!r}")

        raise RuntimeError("Exceeded MAX_TURNS without end_turn")

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _call_api(self, messages: list[dict[str, Any]]) -> MagicMock:
        """Return the next mock response (or raise if exhausted)."""
        # In production:
        # import anthropic
        # client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # return client.messages.create(
        #     model="claude-sonnet-5-5",
        #     max_tokens=1024,
        #     tools=self.tools,
        #     messages=messages,
        # )
        if self._mock_call_count >= len(self._mock_responses):
            raise RuntimeError("Ran out of mock responses")
        resp = self._mock_responses[self._mock_call_count]
        self._mock_call_count += 1
        return resp

    def _extract_text(self, response: MagicMock) -> str:
        """Pull the first text block from a response."""
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""

    def _content_to_list(self, content: list[MagicMock]) -> list[dict[str, Any]]:
        """
        Convert mock content blocks to plain dicts for the message list.
        The Anthropic SDK accepts either SDK objects or plain dicts here.
        Using plain dicts makes the structure explicit for learning purposes.
        """
        result: list[dict[str, Any]] = []
        for block in content:
            if block.type == "tool_use":
                result.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
            elif block.type == "text":
                result.append({"type": "text", "text": block.text})
        return result

    def _dispatch_all(
        self, content: list[MagicMock]
    ) -> list[dict[str, Any]]:
        """
        For each tool_use block: call the tool, build a tool_result block.
        Returns a list of tool_result dicts ready to be sent as a user message.
        """
        tool_results: list[dict[str, Any]] = []
        for block in content:
            if block.type != "tool_use":
                continue
            logger.info(
                "Dispatching tool: name=%s id=%s input=%s",
                block.name,
                block.id,
                block.input,
            )
            fn = TOOL_REGISTRY.get(block.name)
            if fn is None:
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Error: unknown tool '{block.name}'",
                        "is_error": True,
                    }
                )
                continue

            try:
                result = fn(**block.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    }
                )
                logger.info("Tool result for %s: %s", block.id, result)
            except Exception as exc:
                logger.error("Tool %s raised: %s", block.name, exc)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Error: {exc}",
                        "is_error": True,
                    }
                )

        return tool_results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

GET_WEATHER_TOOL: dict[str, Any] = {
    "name": "get_weather",
    "description": "Return current weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name."},
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit.",
            },
        },
        "required": ["city"],
    },
}

if __name__ == "__main__":
    # Build two mock API responses:
    # Response 1: model requests tool_use
    turn1_response = _make_response(
        stop_reason="tool_use",
        content=[
            _make_tool_use_block(
                "toolu_01ABC", "get_weather", {"city": "Berlin", "units": "celsius"}
            )
        ],
    )

    # Response 2: model returns final text after seeing tool_result
    turn2_response = _make_response(
        stop_reason="end_turn",
        content=[
            _make_text_block(
                "It's currently 18°C and partly cloudy in Berlin. "
                "Good weather for a walk!"
            )
        ],
    )

    agent = ToolCycleAgent(
        tools=[GET_WEATHER_TOOL],
        mock_responses=[turn1_response, turn2_response],
    )

    final = agent.run("What's the weather like in Berlin?")
    print(f"\nFinal answer: {final}")
```

**Expected output:**
```
INFO: === Starting agent loop ===
INFO: User: What's the weather like in Berlin?
INFO: --- Turn 1 ---
INFO: stop_reason: tool_use
INFO: Appended assistant message with 1 content block(s)
INFO: Dispatching tool: name=get_weather id=toolu_01ABC input={'city': 'Berlin', 'units': 'celsius'}
INFO: TOOL get_weather: city=Berlin units=celsius
INFO: Tool result for toolu_01ABC: {'city': 'Berlin', 'temperature': 18, ...}
INFO: Appended 1 tool_result block(s)
INFO: Current message list length: 3
INFO: --- Turn 2 ---
INFO: stop_reason: end_turn
INFO: Final response: It's currently 18°C and partly cloudy in Berlin. ...
Final answer: It's currently 18°C and partly cloudy in Berlin. Good weather for a walk!
```

### Minimal tests

```python
# tests/unit/phase5/week02/test_day3.py
"""Tests for day3_tool_cycle.py — all LLM calls mocked."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from phase_5_agentic.week_02.examples.day3_tool_cycle import (
    GET_WEATHER_TOOL,
    ToolCycleAgent,
    _make_response,
    _make_text_block,
    _make_tool_use_block,
    get_weather,
)


def _two_turn_mocks(city: str = "Berlin") -> list[MagicMock]:
    turn1 = _make_response(
        "tool_use",
        [_make_tool_use_block("id_001", "get_weather", {"city": city})],
    )
    turn2 = _make_response(
        "end_turn",
        [_make_text_block(f"It is sunny in {city}.")],
    )
    return [turn1, turn2]


def test_two_turn_cycle_returns_final_text() -> None:
    agent = ToolCycleAgent(
        tools=[GET_WEATHER_TOOL], mock_responses=_two_turn_mocks("Tokyo")
    )
    result = agent.run("Weather in Tokyo?")
    assert "Tokyo" in result


def test_tool_result_uses_correct_id() -> None:
    """The tool_result's tool_use_id must match the tool_use block's id."""
    agent = ToolCycleAgent(
        tools=[GET_WEATHER_TOOL], mock_responses=_two_turn_mocks("Oslo")
    )
    # Intercept _dispatch_all to inspect result
    original_dispatch = agent._dispatch_all

    captured: list[list] = []

    def capturing_dispatch(content):
        result = original_dispatch(content)
        captured.append(result)
        return result

    agent._dispatch_all = capturing_dispatch  # type: ignore[method-assign]
    agent.run("Weather?")
    assert captured[0][0]["tool_use_id"] == "id_001"


def test_unknown_tool_returns_is_error_block() -> None:
    turn1 = _make_response(
        "tool_use",
        [_make_tool_use_block("id_999", "nonexistent_tool", {})],
    )
    turn2 = _make_response("end_turn", [_make_text_block("Sorry.")])
    agent = ToolCycleAgent(tools=[GET_WEATHER_TOOL], mock_responses=[turn1, turn2])
    agent.run("Do something unknown")
    # Should not raise; error is returned as tool_result with is_error=True


def test_max_turns_raises() -> None:
    """If the model never returns end_turn, MAX_TURNS should raise."""
    infinite_tool_responses = [
        _make_response(
            "tool_use",
            [_make_tool_use_block(f"id_{i}", "get_weather", {"city": "X"})],
        )
        for i in range(ToolCycleAgent.MAX_TURNS + 1)
    ]
    agent = ToolCycleAgent(
        tools=[GET_WEATHER_TOOL], mock_responses=infinite_tool_responses
    )
    with pytest.raises(RuntimeError, match="MAX_TURNS|mock responses"):
        agent.run("Keep looping")


def test_message_list_grows_by_two_per_tool_turn() -> None:
    """After one tool turn: messages = [user, assistant, user(tool_result)] = 3."""
    agent = ToolCycleAgent(
        tools=[GET_WEATHER_TOOL], mock_responses=_two_turn_mocks()
    )
    original_call = agent._call_api
    message_lengths: list[int] = []

    def capturing_call(messages):
        message_lengths.append(len(messages))
        return original_call(messages)

    agent._call_api = capturing_call  # type: ignore[method-assign]
    agent.run("Weather?")
    assert message_lengths == [1, 3]  # turn1 had 1 message, turn2 had 3
```

## Exercises

1. **Reconstruct the message list.** By hand (no code), write out the full message list that would be sent to the API for this scenario:
   - User asks: "Search for 'LangGraph tutorial' and then summarize the top result."
   - The model calls `search_web(query="LangGraph tutorial")`.
   - The search returns `["https://langchain-ai.github.io/langgraph/"]`.
   - The model calls `fetch_url(url="https://langchain-ai.github.io/langgraph/")`.
   - The fetch returns a string of HTML.
   - The model returns a final text summary.
   Write out all four messages (role, content type, key fields) before the final response arrives.

2. **Parallel tool calls.** Modify `ToolCycleAgent` to handle a response containing two simultaneous `tool_use` blocks (e.g. `get_weather(city="Berlin")` and `get_weather(city="Tokyo")`). Ensure the returned user message contains two `tool_result` blocks with matching `tool_use_id` values. Write a test with a mock response containing two tool_use blocks.

3. **Loop termination.** The current `MAX_TURNS` guard raises an exception. Change the implementation so that when `MAX_TURNS` is reached, the agent instead returns a special string `"[AGENT_TIMEOUT]"` and logs a warning. Write a test that verifies this behaviour by feeding the agent more than `MAX_TURNS` tool_use mock responses.

(Solutions: ../solutions/week-02/day-3.md)

## Self-check

- [ ] I can draw the full two-turn message cycle (user → tool_use → tool_result → end_turn) from memory
- [ ] I can explain why the assistant's `tool_use` block must be included in the message list before sending `tool_result`
- [ ] I can implement `_dispatch_all` that handles multiple tool_use blocks in a single response
- [ ] I can explain the three conditions under which `stop_reason` is NOT `"tool_use"` and how to handle each

## Supplementary links

- [Anthropic tool use — tool use cycle](https://docs.anthropic.com/en/docs/build-with-claude/tool-use#the-tool-use-loop)
- [Anthropic API reference — tool_result content block](https://docs.anthropic.com/en/api/messages#body-messages)
- [Anthropic cookbook — tool use examples](https://github.com/anthropics/anthropic-cookbook/tree/main/tool_use)
- [Parallel tool use (Anthropic docs)](https://docs.anthropic.com/en/docs/build-with-claude/tool-use#parallel-tool-use)
