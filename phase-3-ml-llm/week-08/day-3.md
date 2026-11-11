# Phase 3 · Week 08 · Day 3 — Tool / Function Schemas

> Objective: Define, dispatch, and loop over Claude tool-use calls to build a minimal agentic loop.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

---

## Concept

Tool calling (also called function calling) extends an LLM from a text transformer into an agent that can invoke external systems. Instead of embedding the answer in its response, the model says "I need to call this function with these arguments." Your code executes the function, returns the result, and the model integrates it into a final answer.

This matters because:
- **Factual grounding.** The model can query live data (weather, databases, APIs) rather than hallucinating from training.
- **Deterministic computation.** Arithmetic, date math, and code execution are delegated to exact tools rather than approximate language model math.
- **Composability.** A single agent can orchestrate many tools in sequence, parallelizing or chaining as the task demands.

In the Claude API, tools are defined as JSON Schema objects. The model selects a tool, emits a `tool_use` content block with typed arguments, and waits. Your code runs the tool, wraps the result in a `tool_result` content block, and sends it back. The loop continues until the model emits a `text` block with no pending tool calls.

---

## How it works internally

**Tool schema as constrained generation.** When you pass `tools=[...]` to `messages.create`, the API injects the tool definitions into the model's context. The model's output distribution is now conditioned on those schema strings. When the model decides to use a tool, it generates a structured `tool_use` block — the field names and argument types are drawn directly from the schema, for the same reason few-shot examples improve output format.

**`tool_use` and `tool_result` content blocks.** Claude uses a multi-modal content block system. A single `message.content` is a list of blocks, each with a `type`. Two new types appear in tool-use conversations:

- `tool_use` — emitted by the assistant. Has `id`, `name`, `input` (a dict matching the tool's `input_schema`).
- `tool_result` — sent by you (as the `user` role). Has `tool_use_id` (matching the `tool_use.id`), `content` (the return value as a string).

**Stop reason `tool_use`.** When the model wants to call a tool, `message.stop_reason` is `"tool_use"` rather than `"end_turn"`. This is the loop condition: keep iterating while `stop_reason == "tool_use"`.

**Multi-turn conversation accumulation.** Each API call requires the complete message history. After each tool execution, you append:
1. The assistant message containing the `tool_use` block.
2. A user message containing the `tool_result` block.

The model attends to all prior turns to decide whether to call another tool or write a final answer.

**Parallel tool calls.** Claude can emit multiple `tool_use` blocks in a single response (one per tool it wants to call simultaneously). Your loop must handle a list of `tool_use` blocks, execute each, and return all results in a single `tool_result` user message.

---

## Worked example

```python
"""
week-08/day-3-tool-calling.py

Minimal agentic loop with two tools: get_weather and calculate.
The loop runs until Claude returns a final text response.

Requirements:
    pip install anthropic
    export ANTHROPIC_API_KEY="..."
"""

import logging
import math
import os
from typing import Any

import anthropic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

MODEL = "claude-opus-5-5"
MAX_TOKENS = 1024
MAX_LOOP_ITERATIONS = 10  # safety cap to prevent infinite loops


# ---------------------------------------------------------------------------
# Tool implementations (the actual Python functions)
# ---------------------------------------------------------------------------

def get_weather(city: str) -> str:
    """
    Return a mock weather report for the given city.

    In production this would call a real weather API.
    Returns a human-readable string so Claude can incorporate it naturally.
    """
    logger.info("Tool: get_weather(city=%r)", city)
    mock_data: dict[str, str] = {
        "london": "Overcast, 12°C, wind 15 km/h from the southwest.",
        "berlin": "Sunny, 8°C, wind 5 km/h from the east.",
        "tokyo": "Partly cloudy, 18°C, humidity 65%.",
    }
    report = mock_data.get(city.lower(), f"No weather data available for '{city}'.")
    logger.info("Tool result: %s", report)
    return report


def calculate(expression: str) -> float:
    """
    Safely evaluate a mathematical expression string.

    Supports: +, -, *, /, **, sqrt, sin, cos, tan, log, pi, e.
    Raises ValueError on unsafe input.
    """
    logger.info("Tool: calculate(expression=%r)", expression)

    # Whitelist: only allow safe math characters and names
    allowed_names: dict[str, Any] = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
    }

    try:
        # eval with restricted globals — no builtins, only math names
        result: float = eval(  # noqa: S307  # restricted eval is intentional here
            expression,
            {"__builtins__": {}},
            allowed_names,
        )
    except Exception as exc:
        raise ValueError(f"Could not evaluate expression '{expression}': {exc}") from exc

    logger.info("Tool result: %s", result)
    return result


# ---------------------------------------------------------------------------
# Tool schemas (JSON Schema format required by the Claude API)
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_weather",
        "description": (
            "Get the current weather for a city. "
            "Returns a plain-text weather report including temperature and conditions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city, e.g. 'London' or 'Tokyo'.",
                },
            },
            "required": ["city"],
        },
    },
    {
        "name": "calculate",
        "description": (
            "Evaluate a mathematical expression. "
            "Supports +, -, *, /, **, sqrt, sin, cos, tan, log, pi, e. "
            "Returns a float result."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "A mathematical expression string, e.g. 'sqrt(144) + 2 * pi'."
                    ),
                },
            },
            "required": ["expression"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

def dispatch_tool(name: str, tool_input: dict[str, Any]) -> str:
    """
    Route a tool_use request to the correct Python function.

    Returns the result as a string (tool_result content must be a string).
    Raises ValueError if the tool name is unknown.
    """
    if name == "get_weather":
        return get_weather(city=tool_input["city"])
    if name == "calculate":
        result = calculate(expression=tool_input["expression"])
        return str(result)
    raise ValueError(f"Unknown tool: '{name}'")


# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------

def run_agent(user_message: str, client: anthropic.Anthropic) -> str:
    """
    Run a tool-use agentic loop until Claude returns a final text response.

    The loop:
    1. Send the current messages to Claude.
    2. If stop_reason is "tool_use": execute all tool_use blocks, append results,
       continue.
    3. If stop_reason is "end_turn": extract and return the text response.

    Args:
        user_message: The initial user query.
        client: An Anthropic client instance.

    Returns:
        Claude's final text response as a string.

    Raises:
        RuntimeError: Loop exceeded MAX_LOOP_ITERATIONS.
    """
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_message},
    ]

    for iteration in range(1, MAX_LOOP_ITERATIONS + 1):
        logger.info("Agent loop iteration %d.", iteration)

        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            tools=TOOLS,
            messages=messages,
        )

        logger.info("Stop reason: %s", response.stop_reason)

        if response.stop_reason == "end_turn":
            # Extract the text block from the response
            for block in response.content:
                if block.type == "text":
                    return block.text
            # No text block found — return empty string rather than crashing
            logger.warning("end_turn received but no text block found.")
            return ""

        if response.stop_reason == "tool_use":
            # Append the assistant's full message (may contain text + tool_use blocks)
            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                }
            )

            # Execute each tool_use block and collect results
            tool_results: list[dict[str, Any]] = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                logger.info(
                    "Executing tool '%s' with input: %s", block.name, block.input
                )

                try:
                    result_str = dispatch_tool(block.name, block.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_str,
                        }
                    )
                except (ValueError, TypeError) as exc:
                    # Return the error to Claude so it can adapt
                    logger.error("Tool '%s' raised: %s", block.name, exc)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Error: {exc}",
                            "is_error": True,
                        }
                    )

            # Append all tool results as a single user message
            messages.append(
                {
                    "role": "user",
                    "content": tool_results,
                }
            )
            continue

        # Unexpected stop reason
        logger.warning("Unexpected stop_reason: %s", response.stop_reason)
        break

    raise RuntimeError(
        f"Agent loop did not complete within {MAX_LOOP_ITERATIONS} iterations."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY environment variable is not set.")

    client = anthropic.Anthropic(api_key=api_key)

    query = (
        "What is the weather like in Tokyo right now? "
        "Also, what is the square root of 1764 plus 10 times pi? "
        "Give me both answers in a single response."
    )

    logger.info("User query: %s", query)
    final_answer = run_agent(query, client)
    print("\nFinal answer from Claude:\n")
    print(final_answer)


if __name__ == "__main__":
    main()
```

---

### Unit tests (mocked — no real API calls)

```python
"""
tests/unit/test_day3_tool_calling.py

Tests for the tool-use agentic loop.
All Anthropic API calls are mocked with unittest.mock.
"""

from typing import Any
from unittest.mock import MagicMock, call

import pytest

from week_08.day3_tool_calling import (
    calculate,
    dispatch_tool,
    get_weather,
    run_agent,
)


# ---------------------------------------------------------------------------
# Helpers: fake content blocks
# ---------------------------------------------------------------------------

def make_text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def make_tool_use_block(
    tool_id: str,
    name: str,
    input_dict: dict[str, Any],
) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = input_dict
    return block


def make_response(
    stop_reason: str,
    content: list[MagicMock],
) -> MagicMock:
    response = MagicMock()
    response.stop_reason = stop_reason
    response.content = content
    return response


# ---------------------------------------------------------------------------
# Unit tests: pure Python tool functions
# ---------------------------------------------------------------------------

class TestGetWeather:
    def test_known_city(self) -> None:
        result = get_weather("london")
        assert "12°C" in result

    def test_unknown_city(self) -> None:
        result = get_weather("atlantis")
        assert "No weather data available" in result

    def test_case_insensitive(self) -> None:
        assert get_weather("TOKYO") == get_weather("tokyo")


class TestCalculate:
    def test_simple_arithmetic(self) -> None:
        assert calculate("2 + 2") == pytest.approx(4.0)

    def test_sqrt(self) -> None:
        assert calculate("sqrt(144)") == pytest.approx(12.0)

    def test_pi_constant(self) -> None:
        import math
        assert calculate("pi") == pytest.approx(math.pi)

    def test_invalid_expression_raises(self) -> None:
        with pytest.raises(ValueError):
            calculate("import os")


# ---------------------------------------------------------------------------
# Unit tests: agentic loop
# ---------------------------------------------------------------------------

class TestRunAgent:
    def test_direct_text_response(self) -> None:
        """Claude answers with end_turn and no tool calls."""
        text_block = make_text_block("The answer is 42.")
        response = make_response("end_turn", [text_block])

        client = MagicMock()
        client.messages.create.return_value = response

        result = run_agent("What is 6 times 7?", client)

        assert result == "The answer is 42."
        client.messages.create.assert_called_once()

    def test_tool_called_then_final_response(self) -> None:
        """Claude calls get_weather once, then returns end_turn text."""
        tool_block = make_tool_use_block(
            tool_id="tu_001",
            name="get_weather",
            input_dict={"city": "Berlin"},
        )
        tool_response = make_response("tool_use", [tool_block])

        text_block = make_text_block("Berlin is sunny and 8 degrees.")
        final_response = make_response("end_turn", [text_block])

        client = MagicMock()
        client.messages.create.side_effect = [tool_response, final_response]

        result = run_agent("What is the weather in Berlin?", client)

        assert result == "Berlin is sunny and 8 degrees."
        assert client.messages.create.call_count == 2

    def test_tool_result_appended_to_messages(self) -> None:
        """Verify that the tool_result is included in the second API call."""
        tool_block = make_tool_use_block(
            tool_id="tu_002",
            name="calculate",
            input_dict={"expression": "sqrt(144)"},
        )
        tool_response = make_response("tool_use", [tool_block])

        text_block = make_text_block("The square root of 144 is 12.0.")
        final_response = make_response("end_turn", [text_block])

        client = MagicMock()
        client.messages.create.side_effect = [tool_response, final_response]

        run_agent("What is sqrt(144)?", client)

        # The second call must include the tool_result in its messages
        second_call_messages: list[dict] = (
            client.messages.create.call_args_list[1][1]["messages"]
        )
        # Last message from user should contain the tool_result
        last_user_message = second_call_messages[-1]
        assert last_user_message["role"] == "user"
        tool_results = last_user_message["content"]
        assert any(
            r.get("type") == "tool_result" and r.get("tool_use_id") == "tu_002"
            for r in tool_results
        )

    def test_raises_on_max_iterations(self) -> None:
        """If stop_reason is always tool_use, raise RuntimeError after cap."""
        tool_block = make_tool_use_block(
            tool_id="tu_inf",
            name="get_weather",
            input_dict={"city": "London"},
        )
        always_tool = make_response("tool_use", [tool_block])

        client = MagicMock()
        client.messages.create.return_value = always_tool

        with pytest.raises(RuntimeError, match="did not complete"):
            run_agent("Keep calling tools forever.", client)
```

---

### Anatomy of a tool-use conversation

```
Turn 1 — user:
  content: "What is the weather in Tokyo and sqrt(1764)?"

Turn 2 — assistant (stop_reason: "tool_use"):
  content: [
    { type: "tool_use", id: "tu_1", name: "get_weather",  input: {"city": "Tokyo"} },
    { type: "tool_use", id: "tu_2", name: "calculate",    input: {"expression": "sqrt(1764)"} }
  ]

Turn 3 — user (tool results):
  content: [
    { type: "tool_result", tool_use_id: "tu_1", content: "Partly cloudy, 18°C." },
    { type: "tool_result", tool_use_id: "tu_2", content: "42.0" }
  ]

Turn 4 — assistant (stop_reason: "end_turn"):
  content: [
    { type: "text", text: "Tokyo is partly cloudy at 18°C. The square root of 1764 is 42." }
  ]
```

The `id` on each `tool_use` block is how the model matches results back to its requests when multiple tools are called in parallel.

---

## Exercises

1. Define a complete tool schema dict for a `search_documents` tool:
   - `query: str` — the search query
   - `max_results: int` with a default of `5` and a maximum of `20`
   - `filter_by_date: bool` — optional, default `false`
   Include `description` strings on every field. The schema must be a valid Claude tool definition dict.

2. Implement a full multi-tool agent (`run_agent`-equivalent) that supports at least 3 tools of your choice. Requirements:
   - Each tool has a real Python implementation (mock data is fine)
   - The loop handles parallel tool calls (multiple `tool_use` blocks in one response)
   - Tool errors are caught, logged, and returned to the model as `is_error: true` results rather than raising
   - Typed throughout with logging

3. Add Pydantic v2 input validation to the tool dispatcher. Before executing any tool, validate the `tool_input` dict against a Pydantic model that corresponds to the tool's `input_schema`. If validation fails, return an error result to the model rather than crashing.
   Example: define `GetWeatherInput(BaseModel)` and `CalculateInput(BaseModel)`, then call `model_validate(tool_input)` before dispatching.

4. Write a test suite for the tool-use loop that covers all four scenarios:
   - Claude calls a tool and the tool succeeds (verify the result is in the next API call's messages)
   - Claude calls a tool and the tool raises an exception (verify `is_error: true` is returned)
   - Claude returns `end_turn` with a text block (verify the text is returned by `run_agent`)
   - The loop hits `MAX_LOOP_ITERATIONS` (verify `RuntimeError` is raised)
   Use `unittest.mock.MagicMock` for the Anthropic client in every test.

   (Solutions: ../solutions/week-08/day-3.md)

---

## Self-check

- [ ] I can explain the difference between `stop_reason: "end_turn"` and `stop_reason: "tool_use"` and what code should do in each case
- [ ] I can write a valid Claude tool schema dict (with `name`, `description`, `input_schema`) for a function of my own design
- [ ] I can implement the full agentic loop — appending `tool_use` assistant messages and `tool_result` user messages — without reference
- [ ] I can handle parallel tool calls (multiple `tool_use` blocks in one response) in a single iteration of the loop
- [ ] I can write a unit test that mocks the Anthropic client to simulate the two-turn tool-use flow (tool call + tool result + final answer)
