# Solutions — Phase 5 · Week 03 · Day 1 — ReAct Pattern from Scratch

## Exercise 1

Add a `fetch_weather(city: str) -> str` tool and show a sample trace.

```python
"""
react_agent.py — ReAct agent with calculator, search, and weather tools.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

def calculate(expression: str) -> str:
    """Safely evaluate a numeric expression."""
    try:
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "Error: invalid characters in expression"
        result = eval(expression)  # noqa: S307 — restricted charset above
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


def search(query: str) -> str:
    """Stub web search — replace with a real API call in production."""
    corpus = {
        "python": "Python is a high-level, interpreted programming language.",
        "weather": "Weather is the state of the atmosphere at a given time.",
    }
    for key, value in corpus.items():
        if key in query.lower():
            return value
    return f"No results found for: {query}"


def fetch_weather(city: str) -> str:
    """Return a mocked current weather report for *city*.

    In a real implementation this would call a weather API.
    """
    mock_data: dict[str, str] = {
        "london": "London: 15°C, partly cloudy, humidity 72%",
        "paris": "Paris: 18°C, sunny, humidity 55%",
        "new york": "New York: 22°C, clear skies, humidity 48%",
        "tokyo": "Tokyo: 28°C, humid, chance of rain 30%",
    }
    key = city.strip().lower()
    return mock_data.get(key, f"{city}: 20°C, data unavailable — using default")


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOLS: dict[str, Any] = {
    "calculate": calculate,
    "search": search,
    "fetch_weather": fetch_weather,
}

TOOL_DESCRIPTIONS = """
Available tools:
- calculate(expression): evaluate a numeric expression, e.g. calculate(2 + 3 * 4)
- search(query): search the web for information
- fetch_weather(city): get the current weather for a city, e.g. fetch_weather(London)

To use a tool, write:
Action: tool_name(argument)

When you have the final answer, write:
Final Answer: <your answer>
"""


# ---------------------------------------------------------------------------
# ReAct loop
# ---------------------------------------------------------------------------

def run_react_agent(question: str, max_steps: int = 10) -> str:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    system_prompt = (
        "You are a helpful assistant that reasons step by step.\n"
        f"{TOOL_DESCRIPTIONS}"
    )

    messages: list[dict[str, str]] = [{"role": "user", "content": question}]
    trace: list[str] = []

    for step in range(max_steps):
        logger.info("Step %d", step + 1)

        response = client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=512,
            system=system_prompt,
            messages=messages,
        )
        text = response.content[0].text.strip()
        trace.append(f"[Step {step + 1}] LLM: {text}")
        logger.debug("LLM output: %s", text)

        # Check for Final Answer
        if "Final Answer:" in text:
            answer = text.split("Final Answer:", 1)[1].strip()
            logger.info("Final answer reached at step %d", step + 1)
            return answer

        # Parse Action
        action_match = re.search(r"Action:\s*(\w+)\((.+?)\)", text, re.DOTALL)
        if action_match:
            tool_name = action_match.group(1).strip()
            tool_arg = action_match.group(2).strip().strip("\"'")

            observation: str
            if tool_name in TOOLS:
                observation = TOOLS[tool_name](tool_arg)
            else:
                observation = f"Error: unknown tool '{tool_name}'"

            logger.info("Tool=%s arg=%r observation=%r", tool_name, tool_arg, observation)
            trace.append(f"[Step {step + 1}] Observation: {observation}")

            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"Observation: {observation}"})
        else:
            # No action and no final answer — let the model continue
            messages.append({"role": "assistant", "content": text})

    return "Max steps reached without a final answer."


# ---------------------------------------------------------------------------
# Sample trace (run with: python react_agent.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    answer = run_react_agent("What is the weather in Paris today?")
    print(f"Answer: {answer}")
```

**Explanation:**
- `fetch_weather` is a pure function with a `dict` mock. Swap the body for a real HTTP call later.
- The registry is a plain `dict[str, Any]` — adding a tool is one line.
- Sample trace (mocked):
  ```
  [Step 1] LLM: Thought: I need to check the weather. Action: fetch_weather(Paris)
  [Step 1] Observation: Paris: 18°C, sunny, humidity 55%
  [Step 2] LLM: Final Answer: The weather in Paris is 18°C and sunny with 55% humidity.
  ```

---

## Exercise 2

Modify the ReAct loop to detect `"Final Answer: <answer>"` and break early.

```python
"""
react_agent_final_answer.py — early-exit on 'Final Answer:' pattern.
"""
from __future__ import annotations

import logging
import os
import re

import anthropic

logger = logging.getLogger(__name__)

_FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*(.+)", re.DOTALL | re.IGNORECASE)


def extract_final_answer(text: str) -> str | None:
    """Return the answer string if the LLM declared a Final Answer, else None."""
    match = _FINAL_ANSWER_RE.search(text)
    if match:
        return match.group(1).strip()
    return None


def run_react_agent(
    question: str,
    tools: dict,
    max_steps: int = 10,
) -> str:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    messages: list[dict[str, str]] = [{"role": "user", "content": question}]

    for step in range(1, max_steps + 1):
        response = client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=512,
            messages=messages,
        )
        text = response.content[0].text.strip()
        logger.info("[Step %d] %s", step, text[:120])

        # --- Early exit on Final Answer ---
        final = extract_final_answer(text)
        if final is not None:
            logger.info("Final answer at step %d: %s", step, final)
            return final  # <-- break out of loop, return just the answer string

        # Parse and execute action
        action_match = re.search(r"Action:\s*(\w+)\((.+?)\)", text, re.DOTALL)
        if action_match:
            tool_name, tool_arg = action_match.group(1), action_match.group(2).strip("\"'")
            observation = tools.get(tool_name, lambda x: f"Unknown tool: {x}")(tool_arg)
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"Observation: {observation}"})
        else:
            messages.append({"role": "assistant", "content": text})

    return "Max steps reached without a final answer."
```

**Explanation:**
- `extract_final_answer` uses a compiled regex so the pattern is defined once.
- `re.DOTALL` lets the answer span multiple lines if needed.
- The function returns `str | None` — the caller checks `None` to continue the loop, or returns the extracted string immediately.
- `re.IGNORECASE` handles `"final answer:"` casing variants from different models.

---

## Exercise 3

Add a `max_tokens` budget that tracks total tokens across steps and stops when exceeded.

```python
"""
react_agent_budget.py — token-budget-aware ReAct loop.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

import anthropic

logger = logging.getLogger(__name__)


@dataclass
class TokenBudget:
    limit: int
    used: int = field(default=0)

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.used += input_tokens + output_tokens
        logger.debug("Tokens used so far: %d / %d", self.used, self.limit)

    @property
    def exceeded(self) -> bool:
        return self.used >= self.limit

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)


def run_react_agent_budgeted(
    question: str,
    tools: dict,
    max_tokens_budget: int = 4096,
    max_steps: int = 10,
) -> str:
    """Run a ReAct agent that stops when the token budget is exhausted.

    Returns the partial answer accumulated so far if budget runs out.
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    budget = TokenBudget(limit=max_tokens_budget)
    messages: list[dict[str, str]] = [{"role": "user", "content": question}]
    partial_answer: str = ""

    for step in range(1, max_steps + 1):
        if budget.exceeded:
            logger.warning("Token budget exceeded before step %d", step)
            break

        response = client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=min(512, budget.remaining),
            messages=messages,
        )

        # Track real token usage from the API response
        budget.add(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        text = response.content[0].text.strip()
        logger.info("[Step %d | tokens %d/%d] %s", step, budget.used, budget.limit, text[:80])

        # Check for Final Answer
        final_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
        if final_match:
            return final_match.group(1).strip()

        # Capture any partial answer content for graceful degradation
        thought_match = re.search(r"Thought:\s*(.+?)(?:\n|$)", text)
        if thought_match:
            partial_answer = thought_match.group(1).strip()

        # Execute tool if present
        action_match = re.search(r"Action:\s*(\w+)\((.+?)\)", text, re.DOTALL)
        if action_match:
            tool_name, tool_arg = action_match.group(1), action_match.group(2).strip("\"'")
            observation = tools.get(tool_name, lambda x: f"Unknown tool: {x}")(tool_arg)
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"Observation: {observation}"})
        else:
            messages.append({"role": "assistant", "content": text})

        if budget.exceeded:
            logger.warning("Budget exceeded after step %d", step)
            break

    return f"[Budget exhausted after {budget.used} tokens] Partial: {partial_answer}"
```

**Explanation:**
- `TokenBudget` is a `dataclass` that encapsulates used/limit tracking — no mutable globals.
- `budget.add` is called with the real `response.usage` fields returned by the Anthropic SDK.
- `min(512, budget.remaining)` prevents requesting more output tokens than the budget allows.
- When budget is exceeded, the function returns a clearly-labelled partial answer string instead of crashing.

---

## Exercise 4

Pytest test that mocks the LLM response.

```python
"""
tests/unit/test_react_agent.py
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from react_agent import TOOLS, run_react_agent  # adjust import to actual module path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mock_response(text: str) -> MagicMock:
    """Build a minimal Anthropic Messages response mock."""
    content_block = MagicMock()
    content_block.text = text

    usage = MagicMock()
    usage.input_tokens = 10
    usage.output_tokens = 20

    response = MagicMock()
    response.content = [content_block]
    response.usage = usage
    return response


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReActAgent:
    """Unit tests for the ReAct agent loop."""

    @patch("react_agent.anthropic.Anthropic")
    def test_final_answer_returned_directly(self, mock_anthropic_cls: MagicMock) -> None:
        """Agent returns the answer string on the first 'Final Answer:' response."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_mock_response(
            "Thought: I know the answer.\nFinal Answer: 42"
        )

        result = run_react_agent("What is the answer to everything?", max_steps=5)

        assert result == "42"
        mock_client.messages.create.assert_called_once()

    @patch("react_agent.anthropic.Anthropic")
    def test_tool_action_then_final_answer(self, mock_anthropic_cls: MagicMock) -> None:
        """Agent executes one tool call, then returns a final answer on the next step."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        # First call: thought + action; second call: final answer
        mock_client.messages.create.side_effect = [
            _make_mock_response("Thought: I should calculate.\nAction: calculate(2 + 3)"),
            _make_mock_response("Thought: Result is 5.\nFinal Answer: The result is 5."),
        ]

        result = run_react_agent("What is 2 + 3?", max_steps=5)

        assert result == "The result is 5."
        assert mock_client.messages.create.call_count == 2

    @patch("react_agent.anthropic.Anthropic")
    def test_max_steps_returns_fallback(self, mock_anthropic_cls: MagicMock) -> None:
        """When max_steps is reached with no final answer, a fallback string is returned."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        # Always return a thought with no final answer
        mock_client.messages.create.return_value = _make_mock_response(
            "Thought: Still thinking..."
        )

        result = run_react_agent("Infinite question?", max_steps=3)

        assert "Max steps" in result
        assert mock_client.messages.create.call_count == 3

    @patch("react_agent.anthropic.Anthropic")
    def test_unknown_tool_returns_error_observation(self, mock_anthropic_cls: MagicMock) -> None:
        """Calling an unknown tool yields an error observation, not an exception."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_client.messages.create.side_effect = [
            _make_mock_response("Action: nonexistent_tool(arg)"),
            _make_mock_response("Final Answer: recovered"),
        ]

        result = run_react_agent("test", max_steps=5)

        assert result == "recovered"

    @patch("react_agent.anthropic.Anthropic")
    def test_fetch_weather_tool_called(self, mock_anthropic_cls: MagicMock) -> None:
        """The fetch_weather tool is invoked and its result fed back as an observation."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_client.messages.create.side_effect = [
            _make_mock_response("Action: fetch_weather(London)"),
            _make_mock_response("Final Answer: London is 15°C, partly cloudy."),
        ]

        result = run_react_agent("What is the weather in London?", max_steps=5)

        assert "London" in result
        # Verify the second call's message list contains the weather observation
        second_call_messages = mock_client.messages.create.call_args_list[1][1]["messages"]
        observations = [m["content"] for m in second_call_messages if "Observation" in m.get("content", "")]
        assert any("London" in obs for obs in observations)
```

**Explanation:**
- `_make_mock_response` builds a minimal `MagicMock` matching the shape of `anthropic.types.Message`. No real API calls.
- `@patch("react_agent.anthropic.Anthropic")` patches at the module level where `anthropic` is imported — this is the correct patch target.
- `side_effect` accepts a list, so successive calls return different responses — models a thought+action then final-answer sequence.
- Each test asserts both the return value and the call count to ensure no extra LLM calls were made.
