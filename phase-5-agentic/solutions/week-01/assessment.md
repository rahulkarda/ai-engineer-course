# Solutions — Week 01 Assessment

## Part 1 — Conceptual answers

**Q1 — Agent definition**

An AI agent is a system that perceives its environment (observe), decides what to do (think), takes an action (act), observes the outcome, and repeats until a goal is reached or a stop condition fires. Four mandatory components: LLM (reasoning), tools (actuation), loop (control flow), memory (context). Without the loop, a single LLM call is a chain — it cannot adapt to tool results or multi-step tasks.

**Q2 — ReAct**

ReAct = Reasoning + Acting. Two-turn trace:
- User goal arrives → Observe phase assembles context list.
- LLM call → Think phase; model outputs `{"action": "search", "args": {"query": "..."}}`.
- Act phase dispatches search, gets result.
- Observe phase appends result to context, increments iteration.
- LLM call again → Think; model outputs `{"action": "finish", "answer": "..."}`.
- Loop exits; answer returned.

**Q3 — Workflow vs agent**

Workflow: developer defines every edge at design time; LLM never controls flow. Agent: LLM output determines next step. Choose workflow when: steps are fixed, auditability is required, cost/latency must be tightly bounded (e.g., a fixed 3-step RAG pipeline that always retrieve → rerank → generate).

**Q4 — Tool registry**

A dict mapping action names to callables. Tool execution lives in host code because the LLM only produces text — it cannot call functions. The host parses the text and dispatches the real function. This separation makes tools testable, auditable, and safely sandboxed.

**Q5 — Rate limit handling**

Exponential backoff with jitter: `delay = base * 2^attempt + random(0, jitter)`. Governed by `retries` (number of attempts), `base_delay` (initial delay in seconds). If all retries exhausted: re-raise as `LLMError("LLM failed after retries")` so the caller can decide to abort or surface the error.

**Q6 — Three stop conditions**

| Condition | Missing consequence |
|-----------|---------------------|
| `action == "finish"` | Agent ignores final answer, loops forever |
| `iteration >= max_iterations` | Infinite loop, unbounded cost |
| `consecutive_errors >= max_consecutive_errors` | Parse-error cascade fills context, exponential cost, no useful output |

---

## Part 2 — Coding challenge reference implementation

```python
"""agent_challenge.py"""

import json
import logging
import re
from collections.abc import Iterator
from typing import Any, Callable

from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

Message = dict[str, str]


# ---------- FakeLLMClient ----------

class FakeLLMClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses: Iterator[str] = iter(responses)
        self.calls: list[list[Message]] = []

    def __call__(self, messages: list[Message], *, system: str = "") -> str:
        self.calls.append(messages)
        return next(self._responses)


# ---------- ParsedAction ----------

class ParsedAction(BaseModel):
    action: str
    args: dict[str, Any] = {}
    answer: str | None = None
    raw: str | None = None


# ---------- Parser ----------

def parse_action(text: str) -> dict:
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"action": "parse_error", "raw": text}


# ---------- Registry ----------

def build_registry() -> dict[str, Callable]:
    def search(query: str) -> str:
        return f"result:{query}"

    def add(a: float, b: float) -> float:
        return a + b

    return {"search": search, "add": add}


# ---------- run_agent ----------

def run_agent(
    goal: str,
    llm: Callable,
    registry: dict[str, Callable],
    *,
    max_iterations: int = 10,
    max_errors: int = 3,
) -> str:
    messages: list[Message] = [{"role": "user", "content": goal}]
    consecutive_errors = 0

    for iteration in range(max_iterations):
        raw = llm(messages)
        data = parse_action(raw)

        if data.get("action") == "parse_error":
            consecutive_errors += 1
            log.warning("parse error #%d", consecutive_errors)
            if consecutive_errors >= max_errors:
                raise RuntimeError("max_errors")
            nudge = f"Respond only with JSON. Your output was: {data.get('raw', '')}"
            messages.append({"role": "user", "content": nudge})
            continue

        consecutive_errors = 0

        if data.get("action") == "finish":
            return data.get("answer", "")

        action = data.get("action", "")
        args = data.get("args", {})
        if action in registry:
            try:
                result = str(registry[action](**args))
            except Exception as exc:
                result = f"ERROR: {exc}"
                log.error("tool error: %s", exc)
        else:
            result = f"ERROR: unknown tool '{action}'"

        messages.append({"role": "assistant", "content": f"called {action}"})
        messages.append({"role": "tool", "content": result})

    raise RuntimeError("max_iterations")
```

```python
"""tests/test_challenge.py"""

import pytest
from agent_challenge import FakeLLMClient, build_registry, run_agent


def test_two_step_finish():
    llm = FakeLLMClient([
        '{"action": "search", "args": {"query": "x"}}',
        '{"action": "finish", "answer": "42"}',
    ])
    answer = run_agent("find 42", llm, build_registry())
    assert answer == "42"


def test_parse_error_recovery():
    llm = FakeLLMClient([
        "not json",
        '{"action": "finish", "answer": "recovered"}',
    ])
    answer = run_agent("test", llm, build_registry(), max_errors=3)
    assert answer == "recovered"


def test_max_iterations_raises():
    llm = FakeLLMClient(['{"action": "search", "args": {"query": "x"}}'] * 20)
    with pytest.raises(RuntimeError, match="max_iterations"):
        run_agent("test", llm, build_registry(), max_iterations=3)


def test_max_errors_raises():
    llm = FakeLLMClient(["bad"] * 10)
    with pytest.raises(RuntimeError, match="max_errors"):
        run_agent("test", llm, build_registry(), max_errors=2)


def test_tool_exception_continues():
    def bad_search(query: str) -> str:
        raise RuntimeError("network down")

    registry = {"search": bad_search}
    llm = FakeLLMClient([
        '{"action": "search", "args": {"query": "x"}}',
        '{"action": "finish", "answer": "survived"}',
    ])
    answer = run_agent("test", llm, registry)
    assert answer == "survived"


def test_fake_records_calls():
    llm = FakeLLMClient(['{"action": "finish", "answer": "ok"}'])
    run_agent("test", llm, build_registry())
    assert len(llm.calls) == 1
```

---

## Part 3 — Debugging answers

**Bug 1 — Line A: silent `continue` on parse error**

`continue` skips the rest of the loop body but does NOT inform the model that its response was unparseable. The model never receives a nudge, so it will keep producing the same malformed output indefinitely (until max_iterations).

Fix: append a nudge message before `continue`:
```python
messages.append({"role": "user", "content": "Respond only with JSON."})
continue
```

**Bug 2 — Line B: unguarded registry lookup**

`registry[data["action"]]` raises `KeyError` if the model produces an action name not in the registry. This crashes the agent rather than recovering.

Fix: check membership before dispatch:
```python
if data["action"] not in registry:
    result = f"ERROR: unknown tool '{data['action']}'"
else:
    result = str(registry[data["action"]](**data.get("args", {})))
messages.append({"role": "tool", "content": result})
```

**Bug 3 — Line C: tool result not coerced to string**

`tool(**data["args"])` may return a non-string (e.g., `float` from a calculator). `{"role": "tool", "content": result}` with a non-string `content` value will either be rejected by the Messages API or produce type errors downstream.

Fix: `messages.append({"role": "tool", "content": str(result)})`
