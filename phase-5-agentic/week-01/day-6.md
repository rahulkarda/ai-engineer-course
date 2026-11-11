# Phase 5 · Week 01 · Day 6 — Error Handling & Max-Iteration Guards (Saturday Deep Practice)

> Objective: Harden the agent loop with comprehensive error handling, then spend 4 hours integrating everything into a fully working minimal agent.  ·  Time: ~4 hr  ·  Prereqs: [[day-5]]

## Concept

Production agents fail in three distinct layers:

```
Layer 1 — LLM layer    : network errors, rate limits, malformed responses, max_tokens
Layer 2 — Tool layer   : tool raises exception, returns bad type, times out
Layer 3 — Loop layer   : infinite loop, parse error cascade, context overflow
```

Each layer needs its own error type, recovery strategy, and test.

### Layer 1 — LLM errors

| Error | Cause | Recovery |
|-------|-------|----------|
| `anthropic.APIConnectionError` | Network | Retry with exponential backoff |
| `anthropic.RateLimitError` | Too many requests | Backoff + jitter |
| `anthropic.APIStatusError` (5xx) | Server error | Retry up to N times |
| `stop_reason == "max_tokens"` | Response truncated | Warn + continue (may cause parse error) |
| Parse failure | Model ignored formatting instruction | Nudge + retry |

### Layer 2 — Tool errors

Tools are untrusted: they might raise, return wrong types, or block. Wrap every tool call:

```python
try:
    result = tool(**args)
except Exception as exc:
    result = f"ERROR: {exc}"
    mark_error()
```

Return the error string as the tool result. Let the LLM decide what to do with it.
Do **not** silently swallow the error — the LLM needs to know it failed.

### Layer 3 — Loop guards

```python
MAX_ITERATIONS = 10          # hard stop — always
MAX_CONSECUTIVE_ERRORS = 3   # parse-error cascade stop
MAX_CONTEXT_MESSAGES = 50    # context overflow guard (trim or raise)
```

These are constants, not prompts. The LLM cannot override them.

### Retry decorator pattern

```python
import time, random

def with_retry(fn, *, retries=3, base_delay=1.0):
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(delay)
```

## How it works internally

### Why exponential backoff?

Rate limiters use token buckets. If every client retries immediately, they all hammer the server at the same time. Exponential backoff + jitter spreads retries over time, reducing thundering-herd effects.

### Context overflow

The context window is finite (200k tokens for Claude). At 50 messages × ~200 tokens each = 10k tokens — rarely an issue for a single agent run. But long-running agents hit the limit. Strategies:
- **Trim oldest messages** (loses early context)
- **Summarise** (one LLM call → condense N messages into 1)
- **Hard stop** at a message count threshold

### Structured error hierarchy

```
AgentError
├── LLMError         (Layer 1)
│   ├── LLMNetworkError
│   └── LLMParseError
├── ToolError         (Layer 2)
│   ├── ToolNotFoundError
│   └── ToolExecutionError
└── LoopError         (Layer 3)
    ├── MaxIterationsError
    └── MaxErrorsError
```

## Worked example — Hardened loop

```python
"""
day6_hardened_agent.py — production-grade error handling in the agent loop.
All LLM calls are mocked; no network calls required.
"""

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------- Exception hierarchy ----------

class AgentError(Exception):
    pass

class LLMError(AgentError):
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


# ---------- Retry helper ----------

def with_retry(
    fn: Callable[[], Any],
    *,
    retries: int = 3,
    base_delay: float = 1.0,
    jitter: float = 0.3,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Any:
    """Exponential backoff with jitter. Only retries the given exception types."""
    for attempt in range(retries):
        try:
            return fn()
        except exceptions as exc:
            if attempt == retries - 1:
                raise
            delay = base_delay * (2**attempt) + random.uniform(0, jitter)
            log.warning("attempt %d/%d failed (%s), retrying in %.2fs", attempt + 1, retries, exc, delay)
            time.sleep(delay)


# ---------- Message + state ----------

class Message(BaseModel):
    role: str
    content: str


@dataclass
class AgentConfig:
    max_iterations: int = 10
    max_consecutive_errors: int = 3
    max_context_messages: int = 50
    llm_retries: int = 3


@dataclass
class LoopState:
    goal: str
    config: AgentConfig
    messages: list[Message] = field(default_factory=list)
    iteration: int = 0
    consecutive_errors: int = 0
    done: bool = False
    final_answer: str | None = None


# ---------- Output parser (from day-5) ----------

import json
import re

def parse_output(text: str) -> dict[str, Any]:
    """Returns parsed dict or raises LLMParseError."""
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    raise LLMParseError(f"cannot parse: {text!r}")


# ---------- Hardened tool dispatch ----------

def dispatch_tool(
    action: str,
    args: dict[str, Any],
    registry: dict[str, Callable[..., Any]],
) -> str:
    if action not in registry:
        raise ToolNotFoundError(f"no tool named '{action}'")
    try:
        result = registry[action](**args)
        return str(result)
    except ToolNotFoundError:
        raise
    except Exception as exc:
        raise ToolExecutionError(f"tool '{action}' raised: {exc}") from exc


# ---------- Main hardened loop ----------

def run_hardened_agent(
    goal: str,
    llm: Callable[[list[dict]], str],
    registry: dict[str, Callable[..., Any]],
    config: AgentConfig | None = None,
) -> str:
    cfg = config or AgentConfig()
    state = LoopState(goal=goal, config=cfg)
    state.messages.append(Message(role="user", content=goal))

    for _ in range(cfg.max_iterations):
        # Guard: context overflow
        if len(state.messages) > cfg.max_context_messages:
            log.warning("context overflow — trimming oldest messages")
            state.messages = state.messages[-cfg.max_context_messages:]

        # THINK (with retry for transient LLM errors)
        ctx = [{"role": m.role, "content": m.content} for m in state.messages]
        try:
            raw = with_retry(
                lambda: llm(ctx),
                retries=cfg.llm_retries,
                base_delay=0.01,   # tiny for tests
                exceptions=(LLMError,),
            )
        except LLMError as exc:
            raise LLMError("LLM failed after retries") from exc

        # PARSE
        try:
            data = parse_output(raw)
            state.consecutive_errors = 0
        except LLMParseError as exc:
            log.error("parse error: %s", exc)
            state.consecutive_errors += 1
            if state.consecutive_errors >= cfg.max_consecutive_errors:
                raise MaxConsecutiveErrorsError(
                    f"{cfg.max_consecutive_errors} consecutive parse errors"
                ) from exc
            nudge = f"Your response could not be parsed. Respond only with JSON. Raw: {raw}"
            state.messages.append(Message(role="user", content=nudge))
            continue

        # STOP CHECK
        if data.get("action") == "finish":
            state.final_answer = data.get("answer", "")
            state.done = True
            log.info("agent finished after %d iterations", state.iteration)
            break

        # ACT
        action = data.get("action", "")
        args = data.get("args", {})
        try:
            tool_output = dispatch_tool(action, args, registry)
        except ToolNotFoundError as exc:
            tool_output = f"ERROR: {exc}"
            log.error("unknown tool: %s", action)
        except ToolExecutionError as exc:
            tool_output = f"ERROR: {exc}"
            log.error("tool error: %s", exc)

        # OBSERVE
        state.messages.append(Message(role="assistant", content=f"called {action}"))
        state.messages.append(Message(role="tool", content=tool_output))
        state.iteration += 1

    else:
        raise MaxIterationsError(f"reached {cfg.max_iterations} iterations without finishing")

    return state.final_answer or ""


# ---------- Demo ----------

_step = 0

def scripted_llm(context: list[dict]) -> str:
    global _step
    _step += 1
    if _step == 1:
        return '{"action": "search", "args": {"query": "error handling"}}'
    return '{"action": "finish", "answer": "Error handling makes agents robust."}'


def search_tool(query: str) -> str:
    return f"results for: {query}"


if __name__ == "__main__":
    answer = run_hardened_agent(
        "How does error handling work in agents?",
        llm=scripted_llm,
        registry={"search": search_tool},
    )
    print("Answer:", answer)
```

## Exercises

1. Write a pytest test that feeds `run_hardened_agent` an LLM that returns non-JSON on the first two calls and valid JSON on the third. Assert it completes successfully (does not raise) and returns the correct answer.

2. Write a pytest test that verifies `MaxConsecutiveErrorsError` is raised when the LLM returns non-JSON on every call and `max_consecutive_errors=2`.

3. Write a pytest test that verifies `MaxIterationsError` is raised when the LLM always returns a search action (never "finish") and `max_iterations=3`.

4. Write a pytest test that verifies `ToolNotFoundError` is converted to an error string (not raised) and the agent continues to the next iteration.

5. Add a `total_tool_errors: int` counter to `LoopState`. Increment it whenever a tool raises. Write a test asserting it equals 1 after one tool failure.

(Solutions: ../solutions/week-01/day-6.md)

## Saturday build session — 4 hours

Integrate days 1–6 into a complete agent:

**Goal**: an agent that can answer multi-step questions using a `search` and `calculator` tool, with full error handling, a mockable LLM client, and a test suite.

**Deliverables**:
- `agent/loop.py` — hardened loop
- `agent/client.py` — `AnthropicLLMClient` + `FakeLLMClient`
- `agent/tools.py` — `search` and `calculator` tools with input validation
- `agent/parser.py` — `parse_output`
- `tests/test_agent.py` — at least 8 tests, all LLM calls mocked

See `project.md` for full spec.

## Self-check

- [ ] I can name the three error layers in an agent and give one example of each.
- [ ] I can implement exponential backoff with jitter from scratch.
- [ ] I can explain why tool errors should be returned as strings, not raised.
- [ ] I can implement all four loop guards (max_iter, max_errors, context overflow, finish).
- [ ] I have passing tests for each failure scenario without any real API calls.

## Supplementary links

- Anthropic error codes: https://docs.anthropic.com/en/api/errors
- Python `tenacity` library (production retry): https://tenacity.readthedocs.io/
