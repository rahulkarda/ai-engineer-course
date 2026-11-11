# Solutions — Week 01 Project

This solution walks through each deliverable. Study it **after** you've made your own attempt.

---

## Complete implementation

### `agent/exceptions.py`

```python
"""Agent exception hierarchy."""


class AgentError(Exception):
    pass


class LLMError(AgentError):
    pass


class LLMNetworkError(LLMError):
    pass


class LLMParseError(LLMError):
    pass


class ToolError(AgentError):
    pass


class ToolNotFoundError(ToolError):
    pass


class ToolExecutionError(ToolError):
    pass


class LoopError(AgentError):
    pass


class MaxIterationsError(LoopError):
    pass


class MaxConsecutiveErrorsError(LoopError):
    pass
```

### `agent/client.py`

```python
"""LLM client: real (Anthropic) and fake (for tests)."""

import logging
import os
from collections.abc import Iterator
from typing import Protocol

import anthropic

log = logging.getLogger(__name__)

Message = dict[str, str]


class LLMClient(Protocol):
    def __call__(self, messages: list[Message], *, system: str = "") -> str: ...


class AnthropicLLMClient:
    def __init__(self, model: str = "claude-haiku-5-5", max_tokens: int = 1024) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def __call__(self, messages: list[Message], *, system: str = "") -> str:
        log.info("LLM call model=%s messages=%d", self.model, len(messages))
        kwargs: dict = {"model": self.model, "max_tokens": self.max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system
        response = self._client.messages.create(**kwargs)
        if response.stop_reason == "max_tokens":
            log.warning("response truncated (max_tokens)")
        text = response.content[0].text
        log.info("response length=%d stop_reason=%s", len(text), response.stop_reason)
        return text


class FakeLLMClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses: Iterator[str] = iter(responses)
        self.calls: list[list[Message]] = []
        self.system_prompts: list[str] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def __call__(self, messages: list[Message], *, system: str = "") -> str:
        self.calls.append(messages)
        self.system_prompts.append(system)
        return next(self._responses)
```

### `agent/parser.py`

```python
"""LLM output parser."""

import json
import logging
import re
from typing import Any

from pydantic import BaseModel

log = logging.getLogger(__name__)


class ParsedAction(BaseModel):
    action: str
    args: dict[str, Any] = {}
    answer: str | None = None
    raw: str | None = None


def parse_action(text: str) -> ParsedAction:
    try:
        data = json.loads(text.strip())
        return ParsedAction(**data)
    except (json.JSONDecodeError, Exception):
        pass

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            log.warning("extracted embedded JSON from prose")
            return ParsedAction(**data)
        except (json.JSONDecodeError, Exception):
            pass

    log.error("parse failed for: %r", text[:80])
    return ParsedAction(action="parse_error", raw=text)
```

### `agent/tools.py`

```python
"""Built-in tools and tool registry."""

import logging
import re
from typing import Any, Callable

log = logging.getLogger(__name__)


def search(query: str) -> str:
    if not query or not isinstance(query, str):
        raise ValueError("query must be a non-empty string")
    log.info("search query=%r", query)
    return f"Search results for '{query}': [placeholder — wire a real search API here]"


def calculator(expression: str) -> float:
    if not re.fullmatch(r'[\d\s\+\-\*/\(\)\.]+', expression):
        raise ValueError(f"unsafe expression: {expression!r}")
    result: float = eval(expression)  # noqa: S307
    log.info("calculator %r = %s", expression, result)
    return result


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        self._tools[name] = fn
        log.info("registered tool: %s", name)

    def dispatch(self, name: str, args: dict[str, Any]) -> str:
        if name not in self._tools:
            return f"ERROR: unknown tool '{name}'"
        try:
            result = self._tools[name](**args)
            return str(result)
        except Exception as exc:
            log.error("tool '%s' raised: %s", name, exc)
            return f"ERROR: {exc}"
```

### `agent/loop.py`

```python
"""Hardened agent loop."""

import logging
from dataclasses import dataclass, field

from agent.client import LLMClient, Message
from agent.exceptions import MaxConsecutiveErrorsError, MaxIterationsError
from agent.parser import ParsedAction, parse_action
from agent.tools import ToolRegistry

log = logging.getLogger(__name__)


@dataclass
class LoopState:
    goal: str
    messages: list[Message] = field(default_factory=list)
    iteration: int = 0
    consecutive_errors: int = 0
    done: bool = False
    final_answer: str | None = None


def run_agent(
    goal: str,
    llm: LLMClient,
    registry: ToolRegistry,
    *,
    max_iterations: int = 10,
    max_consecutive_errors: int = 3,
    system_prompt: str = "",
) -> str:
    state = LoopState(goal=goal)
    state.messages.append({"role": "user", "content": goal})

    for _ in range(max_iterations):
        # OBSERVE
        ctx = list(state.messages)

        # THINK
        raw = llm(ctx, system=system_prompt)
        log.info("[loop] it=%d raw=%r", state.iteration, raw[:60])

        # PARSE
        parsed: ParsedAction = parse_action(raw)

        if parsed.action == "parse_error":
            state.consecutive_errors += 1
            log.warning("[loop] parse error #%d", state.consecutive_errors)
            if state.consecutive_errors >= max_consecutive_errors:
                raise MaxConsecutiveErrorsError(
                    f"{max_consecutive_errors} consecutive parse errors"
                )
            nudge = f"Respond only with a JSON object. Your last output was: {parsed.raw}"
            state.messages.append({"role": "user", "content": nudge})
            continue

        state.consecutive_errors = 0

        # STOP CHECK
        if parsed.action == "finish":
            state.done = True
            state.final_answer = parsed.answer or ""
            log.info("[loop] finished after %d iterations", state.iteration)
            break

        # ACT
        tool_output = registry.dispatch(parsed.action, parsed.args)
        log.info("[loop] tool=%s output=%r", parsed.action, tool_output[:60])

        # OBSERVE result
        state.messages.append({"role": "assistant", "content": f"called {parsed.action}"})
        state.messages.append({"role": "tool", "content": tool_output})
        state.iteration += 1

    else:
        raise MaxIterationsError(f"reached {max_iterations} iterations without finishing")

    return state.final_answer or ""
```

---

## Self-check for the full project

- [ ] `pytest -q` shows 0 failures and ≥ 10 tests
- [ ] No test file imports `anthropic` or instantiates `AnthropicLLMClient`
- [ ] `agent/loop.py` is under 100 lines
- [ ] All Pydantic models validate on construction (try passing a bad type)
- [ ] `calculator("__import__('os')")` raises `ValueError`
- [ ] Running `python -m agent.__main__` with a real key produces a sensible answer
