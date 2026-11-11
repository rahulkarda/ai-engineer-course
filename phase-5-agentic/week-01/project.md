# Week 01 Project — Minimal From-Scratch Agent Loop

> Phase 5 · Week 01  ·  Build time: ~4–5 hr (Saturday + Sunday)  ·  No framework allowed

---

## Overview

Build a minimal agent that can answer multi-step questions using two tools — `search` and `calculator` — without using any agent framework (no LangChain, no LangGraph, no CrewAI). Every component is your own code.

The agent takes a natural-language goal, calls tools as needed, and produces a final answer. All LLM calls use the Anthropic Messages API with a Claude 5-family model. Tests use `FakeLLMClient` — no real API calls in the test suite.

---

## Deliverables

```
agent/
├── __init__.py
├── client.py         # AnthropicLLMClient + FakeLLMClient + LLMClient Protocol
├── loop.py           # run_agent() — hardened observe/think/act loop
├── parser.py         # parse_action() — JSON + embedded JSON + error handling
├── tools.py          # search(), calculator() + ToolRegistry
└── exceptions.py     # AgentError, LLMError, ToolError, LoopError subclasses
tests/
├── __init__.py
├── test_client.py    # FakeLLMClient unit tests
├── test_parser.py    # parse_action unit tests
├── test_tools.py     # tool unit tests
└── test_loop.py      # integration tests for run_agent
pyproject.toml
.env.example
```

---

## Specification

### `agent/client.py`

```python
class LLMClient(Protocol):
    def __call__(self, messages: list[dict], *, system: str = "") -> str: ...

class AnthropicLLMClient:
    """
    - model: str = "claude-haiku-5-5"
    - max_tokens: int = 1024
    - reads ANTHROPIC_API_KEY from os.getenv
    - logs model, message count, response length, stop_reason
    - warns if stop_reason == "max_tokens"
    """

class FakeLLMClient:
    """
    - responses: list[str]  (iterable, raises StopIteration when empty)
    - calls: list[list[dict]]  (record every call for assertions)
    - system_prompts: list[str]
    - call_count: int property
    """
```

### `agent/parser.py`

```python
def parse_action(text: str) -> ParsedAction:
    """
    ParsedAction is a Pydantic model with:
      action: str
      args: dict[str, Any] = {}
      answer: str | None = None
      raw: str | None = None   (set on error)

    Try 1: json.loads(text.strip())
    Try 2: regex extract first {...} block, json.loads
    Fallback: ParsedAction(action="parse_error", raw=text)
    """
```

### `agent/tools.py`

```python
def search(query: str) -> str:
    """
    In production: call a real search API.
    For now: return a canned string that includes the query.
    Validates: query must be a non-empty string.
    """

def calculator(expression: str) -> float:
    """
    Evaluates a safe arithmetic expression.
    Allowed: digits, +, -, *, /, (, ), spaces, decimal points.
    Raises ValueError on any other character.
    Uses Python eval() ONLY after whitelist validation.
    """

class ToolRegistry:
    """
    tools: dict[str, Callable]
    register(name, fn)         — add a tool
    dispatch(name, args) -> str  — call tool, return str result or "ERROR: ..."
    """
```

### `agent/loop.py`

```python
def run_agent(
    goal: str,
    llm: LLMClient,
    registry: ToolRegistry,
    *,
    max_iterations: int = 10,
    max_consecutive_errors: int = 3,
    system_prompt: str = "",
) -> str:
    """
    Full observe→think→parse→act→observe loop.

    Stop conditions (in order of check):
    1. action == "finish"  → return answer
    2. iteration >= max_iterations → raise MaxIterationsError
    3. consecutive_errors >= max_consecutive_errors → raise MaxConsecutiveErrorsError

    On parse_error:
    - increment consecutive_errors
    - append nudge message to context
    - continue

    On tool error (ToolRegistry.dispatch returns "ERROR: ..."):
    - append error string to context
    - do NOT increment consecutive_errors
    - continue

    Returns: final answer string
    """
```

---

## System prompt

Use this system prompt for `AnthropicLLMClient` in the integration demo:

```
You are a research agent. You have access to two tools:

1. search(query: str) — searches the web and returns text results
2. calculator(expression: str) — evaluates a math expression

Always respond with a single JSON object:
- To call a tool: {"action": "search", "args": {"query": "..."}}
  or: {"action": "calculator", "args": {"expression": "..."}}
- To give a final answer: {"action": "finish", "answer": "..."}

Do not include any text outside the JSON object.
```

---

## Test requirements

Minimum 10 tests across all test files. Required tests:

| Test | File | What it checks |
|------|------|----------------|
| `test_fake_returns_scripted` | test_client.py | FakeLLMClient pops in order |
| `test_fake_records_system_prompts` | test_client.py | system kwarg recorded |
| `test_parse_clean_json` | test_parser.py | clean JSON parsed correctly |
| `test_parse_embedded_json` | test_parser.py | prose + JSON extracts correctly |
| `test_parse_failure_returns_error` | test_parser.py | bad text → parse_error action |
| `test_calculator_valid` | test_tools.py | `"2 + 3 * 4"` → `14.0` |
| `test_calculator_unsafe_raises` | test_tools.py | `"__import__('os')"` → ValueError |
| `test_agent_two_steps` | test_loop.py | search then finish, returns answer |
| `test_agent_max_iterations` | test_loop.py | MaxIterationsError raised |
| `test_agent_parse_error_recovery` | test_loop.py | nudge works, agent recovers |

---

## Running the demo (requires real API key)

```bash
export ANTHROPIC_API_KEY="$(cat .env | grep ANTHROPIC_API_KEY | cut -d= -f2)"
python -m agent.__main__
```

Create `agent/__main__.py`:
```python
"""Demo: ask the agent a multi-step question."""
import os
from agent.client import AnthropicLLMClient
from agent.loop import run_agent
from agent.tools import ToolRegistry, search, calculator

registry = ToolRegistry()
registry.register("search", search)
registry.register("calculator", calculator)

llm = AnthropicLLMClient(model="claude-haiku-5-5")
answer = run_agent(
    "Search for the speed of light in m/s, then calculate how far light travels in 8 minutes.",
    llm=llm,
    registry=registry,
    system_prompt=open("agent/system_prompt.txt").read(),
)
print("Answer:", answer)
```

---

## Evaluation criteria

| Criterion | Check |
|-----------|-------|
| All 10+ tests pass | `pytest -q` shows 0 failures |
| No real API calls in tests | No `anthropic.Anthropic()` instantiated in test files |
| Loop handles all stop conditions | MaxIterationsError, MaxConsecutiveErrorsError, finish |
| Tool errors don't crash the loop | ToolRegistry.dispatch catches and returns error string |
| Type hints complete | `mypy agent/` or `pyright` shows no errors on core files |
| No secrets | No literal API keys; `.env.example` present; `.env` in `.gitignore` |

---

Solution: ../solutions/week-01/project.md
