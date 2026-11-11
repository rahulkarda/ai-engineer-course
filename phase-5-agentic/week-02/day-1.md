# Phase 5 · Week 02 · Day 1 — Tool & Function Calling Fundamentals

> Objective: Understand what tool calling is, how the LLM decides to call tools, and how to read the raw Anthropic tool_use response.  ·  Time: ~1 hr  ·  Prereqs: [[phase-5-agentic/week-01/day-5]]

## Concept

**Tool calling** (also called function calling) lets an LLM request that your code execute a named function and return the result. The LLM does not run code itself — it emits a structured token block describing *which* tool to call and *with what arguments*. Your program dispatches that call, then feeds the result back into the conversation.

This is the foundational primitive for agents: without tool calling, the LLM can only produce text. With it, the LLM can search the web, query a database, send an email, or call any API you expose.

### Why it matters

| Without tools | With tools |
|---|---|
| LLM can only describe actions | LLM can trigger real actions |
| Knowledge frozen at training cutoff | Live data at inference time |
| Single-pass: output is final | Multi-turn: result feeds next reasoning step |
| No side effects possible | Side effects are explicit and auditable |

### When to use tools vs. prompting alone

Use **tools** when:
- The answer requires live or private data (weather, database, file system)
- The task requires an action with a side effect (send email, write file)
- The computation is deterministic and better handled by code (arithmetic, regex, date math)
- The output must conform to a strict schema that prose cannot guarantee

Use **prompting alone** when:
- The answer is derivable from the model's training knowledge
- You need creative, open-ended text generation
- Latency matters and one round-trip is enough
- Adding tool overhead would outweigh the benefit

### How the LLM decides to call a tool

The model is trained to recognize situations where returning structured tool-call tokens is more useful than generating prose. At inference time:

1. You pass a `tools` array alongside the user message. Each entry describes a callable function: its name, a natural-language description, and a JSON Schema for its parameters.
2. The model reads the descriptions as part of its context. If the user's request matches a tool's purpose, the model emits a `tool_use` content block instead of (or before) text.
3. `stop_reason` in the response will be `"tool_use"` — signalling your loop to dispatch the call.

The model does not have access to any special "function registry." It simply learns, during training, to produce a `tool_use` block when the context calls for it. The descriptions you write are the only signal the model has for deciding *which* tool to pick.

## How it works internally

### The API objects

When you include `tools` in your request, the Anthropic SDK attaches a special `<tools>` section to the system prompt (internally). The model's next token generation is conditioned on this system prompt, so the model "knows" what tools are available.

When the model decides to call a tool it emits content of type `tool_use`:

```
content block: {
  "type": "tool_use",
  "id":   "toolu_01XYZ",       # unique per call, echoed back in tool_result
  "name": "get_weather",       # must match a name in the tools array
  "input": { "city": "Berlin" } # JSON object matching the tool's input_schema
}
```

This block is **not executed** by the API — it is returned to your program. You look at `stop_reason == "tool_use"`, extract the block, call the Python function, and append a `tool_result` message to the conversation before calling the API again.

### The full one-tool-call cycle (high level)

```
You ──► API  [user msg + tools array]
            │
API ──► You [stop_reason=tool_use, content=[tool_use block]]
            │
You ──► (call the Python function)
            │
You ──► API  [original messages + assistant msg + tool_result msg]
            │
API ──► You [stop_reason=end_turn, final text response]
```

Day 3 covers the full multi-turn cycle in detail. Today we focus on the first half: constructing the request and reading the tool_use response.

## Worked example

The example below uses a **mocked API response** so you can run it without an `ANTHROPIC_API_KEY`. The mock returns a realistic `tool_use` response object.

```python
# phase-5-agentic/week-02/examples/day1_tool_fundamentals.py
"""
Tool calling fundamentals — mocked example.
Run: python day1_tool_fundamentals.py
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# 1. Define the tool schema (what we would send to the real API)
# ---------------------------------------------------------------------------

GET_WEATHER_TOOL: dict[str, Any] = {
    "name": "get_weather",
    "description": (
        "Return the current weather for a city. "
        "Use this when the user asks about weather conditions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name, e.g. 'Berlin' or 'New York'",
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit. Default: celsius.",
            },
        },
        "required": ["city"],
    },
}

# ---------------------------------------------------------------------------
# 2. The actual tool implementation (pure Python, no LLM involved)
# ---------------------------------------------------------------------------


def get_weather(city: str, units: str = "celsius") -> dict[str, Any]:
    """Return mock weather data. In production, call a real weather API."""
    logger.info("get_weather called: city=%s units=%s", city, units)
    # Stubbed data — replace with a real HTTP call in production
    data = {
        "city": city,
        "temperature": 18 if units == "celsius" else 64,
        "units": units,
        "condition": "partly cloudy",
    }
    logger.info("get_weather result: %s", data)
    return data


# ---------------------------------------------------------------------------
# 3. Build a mock Anthropic response that looks like a real tool_use response
# ---------------------------------------------------------------------------


def build_mock_tool_use_response(
    tool_name: str, tool_input: dict[str, Any]
) -> MagicMock:
    """
    Construct a MagicMock that mimics the object returned by:
        anthropic.Anthropic().messages.create(...)
    when the model decides to call a tool.
    """
    tool_use_block = MagicMock()
    tool_use_block.type = "tool_use"
    tool_use_block.id = "toolu_01MockId123"
    tool_use_block.name = tool_name
    tool_use_block.input = tool_input

    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [tool_use_block]
    return response


# ---------------------------------------------------------------------------
# 4. Minimal agent loop: send request, handle tool_use
# ---------------------------------------------------------------------------


@dataclass
class SimpleToolAgent:
    """One-shot agent that handles a single tool call and returns."""

    tools: list[dict[str, Any]] = field(default_factory=list)
    model: str = "claude-sonnet-5-5"

    def run(self, user_message: str, mock_response: MagicMock | None = None) -> str:
        """
        Send user_message to the model (or use mock_response in tests).
        If the model calls a tool, dispatch it and log the result.
        Returns the tool result as a JSON string for demonstration.
        """
        logger.info("User: %s", user_message)

        # --- In production you would do: ---
        # import anthropic
        # client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # response = client.messages.create(
        #     model=self.model,
        #     max_tokens=1024,
        #     tools=self.tools,
        #     messages=[{"role": "user", "content": user_message}],
        # )

        # Use the mock response supplied by the caller (or build a default one)
        response = mock_response or build_mock_tool_use_response(
            "get_weather", {"city": "Berlin", "units": "celsius"}
        )

        logger.info("API stop_reason: %s", response.stop_reason)

        if response.stop_reason == "tool_use":
            return self._handle_tool_use(response)

        # stop_reason == "end_turn" — model returned text directly
        text_content = next(
            (b for b in response.content if b.type == "text"), None
        )
        return text_content.text if text_content else ""

    def _handle_tool_use(self, response: MagicMock) -> str:
        """Extract the tool_use block and dispatch the call."""
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        results: list[str] = []
        for block in tool_use_blocks:
            logger.info(
                "Tool called: name=%s id=%s input=%s",
                block.name,
                block.id,
                block.input,
            )
            result = self._dispatch(block.name, block.input)
            results.append(json.dumps(result))

        # In a real agent you would now send the tool_result back to the API.
        # Day 3 covers that full cycle.
        return "\n".join(results)

    def _dispatch(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Route tool name to the correct Python function."""
        dispatch_table: dict[str, Any] = {
            "get_weather": get_weather,
        }
        fn = dispatch_table.get(tool_name)
        if fn is None:
            raise ValueError(f"Unknown tool: {tool_name!r}")
        return fn(**tool_input)


# ---------------------------------------------------------------------------
# 5. Run the demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent = SimpleToolAgent(tools=[GET_WEATHER_TOOL])

    # Simulate a user asking about the weather
    mock = build_mock_tool_use_response(
        "get_weather", {"city": "Tokyo", "units": "celsius"}
    )
    result = agent.run("What's the weather in Tokyo?", mock_response=mock)
    logger.info("Final tool result: %s", result)
```

**Expected output (truncated):**
```
INFO: User: What's the weather in Tokyo?
INFO: API stop_reason: tool_use
INFO: Tool called: name=get_weather id=toolu_01MockId123 input={'city': 'Tokyo', 'units': 'celsius'}
INFO: get_weather called: city=Tokyo units=celsius
INFO: get_weather result: {'city': 'Tokyo', 'temperature': 18, 'units': 'celsius', 'condition': 'partly cloudy'}
INFO: Final tool result: {"city": "Tokyo", "temperature": 18, ...}
```

### Minimal test

```python
# tests/unit/phase5/week02/test_day1.py
"""Tests for day1_tool_fundamentals.py — all LLM calls are mocked."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from phase_5_agentic.week_02.examples.day1_tool_fundamentals import (
    GET_WEATHER_TOOL,
    SimpleToolAgent,
    build_mock_tool_use_response,
    get_weather,
)


def test_get_weather_returns_expected_keys() -> None:
    result = get_weather("Berlin")
    assert {"city", "temperature", "units", "condition"} <= result.keys()
    assert result["city"] == "Berlin"


def test_get_weather_fahrenheit() -> None:
    result = get_weather("London", units="fahrenheit")
    assert result["units"] == "fahrenheit"


def test_agent_dispatches_tool_use(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = SimpleToolAgent(tools=[GET_WEATHER_TOOL])
    mock = build_mock_tool_use_response(
        "get_weather", {"city": "Paris", "units": "celsius"}
    )
    result_json = agent.run("Weather in Paris?", mock_response=mock)
    data = json.loads(result_json)
    assert data["city"] == "Paris"


def test_agent_raises_on_unknown_tool() -> None:
    agent = SimpleToolAgent(tools=[GET_WEATHER_TOOL])
    mock = build_mock_tool_use_response("nonexistent_tool", {})
    with pytest.raises(ValueError, match="Unknown tool"):
        agent.run("Do something unknown", mock_response=mock)


def test_tool_schema_has_required_fields() -> None:
    schema = GET_WEATHER_TOOL
    assert "name" in schema
    assert "description" in schema
    assert "input_schema" in schema
    assert schema["input_schema"]["required"] == ["city"]
```

## Exercises

1. **Define a new tool.** Write a `get_stock_price` tool schema (name, description, input_schema) for a tool that accepts a `ticker` symbol (string, required) and an optional `currency` (string, default `"USD"`). Do not implement the tool function yet — just the schema dict. Predict: if you send the message "What is Apple's stock price?" to the model with only this tool available, what will `stop_reason` be? What `input` dict do you expect in the `tool_use` block?

2. **Tool selection.** Given three tools: `search_web`, `get_current_time`, and `send_email`, predict which tool (if any) the model would call for each of these user messages:
   - "What time is it in Sydney?"
   - "Summarize the key themes of Shakespeare's Hamlet."
   - "Send John the meeting notes from today."
   - "Who won the 2024 US election?"
   Justify each prediction with one sentence referencing the descriptions you would write for each tool.

3. **Extend the agent.** Add a second tool `get_time(timezone: str) -> dict` to the `SimpleToolAgent` worked example. Update the `dispatch_table`, add a matching tool schema, and write a test that mocks a `tool_use` response for `get_time` and asserts the result contains a `"timezone"` key.

(Solutions: ../solutions/week-02/day-1.md)

## Self-check

- [ ] I can explain why `stop_reason == "tool_use"` does not mean the tool has been executed
- [ ] I can describe the three fields every entry in the `tools` array must have
- [ ] I can explain the difference between the `tools` array (schema) and the `tool_use` block (model output)
- [ ] I can implement a one-shot agent that reads a `tool_use` block and dispatches it to a Python function without reference

## Supplementary links

- [Anthropic tool use overview](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [Anthropic API reference — Messages](https://docs.anthropic.com/en/api/messages)
- [JSON Schema spec](https://json-schema.org/understanding-json-schema/)
- [OpenAI function calling blog (conceptual background)](https://openai.com/index/function-calling-and-other-api-updates/)
