# Phase 5 · Week 01 · Day 3 — The Agent Loop Anatomy

> Objective: Dissect every phase of the observe→think→act→observe cycle and implement each phase as a named function.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

## Concept

The agent loop has four canonical phases, sometimes called **ReAct** (Reasoning + Acting):

```
┌──────────────────────────────────────────────────────┐
│                   AGENT LOOP                          │
│                                                       │
│  ┌──────────┐   ┌────────┐   ┌──────┐   ┌────────┐  │
│  │ OBSERVE  │──►│ THINK  │──►│ ACT  │──►│OBSERVE │  │
│  └──────────┘   └────────┘   └──────┘   └────────┘  │
│       ▲                                      │       │
│       └──────────────────────────────────────┘       │
└──────────────────────────────────────────────────────┘
```

### Phase 1 — OBSERVE

Assemble the current state of the world into the LLM's context:
- The original user goal
- All previous messages (assistant turns + tool results)
- Any injected context (system prompt, retrieved documents)

Output: a `list[dict]` in the Messages API format.

### Phase 2 — THINK

Send the context to the LLM and receive a structured response.
The LLM "thinks" by producing tokens. Its output can be:
- A **tool call** — `{"action": "tool_name", "args": {...}}`
- A **final answer** — `{"action": "finish", "answer": "..."}`
- **Free text** — less structured; requires robust parsing

Output: a parsed `ThinkResult` object.

### Phase 3 — ACT

Dispatch the chosen action:
- Look up the tool by name in a registry
- Call it with the provided arguments
- Capture the return value (and any exception)

Output: a `ToolResult` with the raw value or error message.

### Phase 4 — OBSERVE (again)

Append the tool result to the context and increment the iteration counter.
Check stop conditions:
- `done == True` (LLM chose "finish")
- `iteration >= max_iterations`
- Any domain-specific stopping criterion

If not done, go back to Phase 2.

## How it works internally

### Why the loop must be in your code, not the LLM

The LLM cannot "call" a function. It produces text that *looks like* a function call. Your loop reads that text, actually calls the function, and feeds the result back. This indirection is what makes agents testable and safe — you control every execution.

### The scratchpad

Between THINK and ACT the model often produces a **scratchpad** — natural language reasoning ("I need to search for X because..."). Some implementations strip the scratchpad before adding it to the persistent context to save tokens. Others keep it for debugging. Both are valid.

### Tool registry

A tool registry is a dict mapping action names to callables:

```python
registry: dict[str, Callable[..., Any]] = {
    "search": search_fn,
    "calculator": calc_fn,
    "finish": finish_fn,
}
```

Dispatching is just `registry[action_name](**args)`.

## Worked example

```python
"""
day3_loop_anatomy.py — four-phase loop, no real LLM.
Each phase is its own function with a typed return value.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------- Data models ----------

class Message(BaseModel):
    role: str
    content: str


class ThinkResult(BaseModel):
    action: str
    args: dict[str, Any] = {}
    answer: str | None = None
    scratchpad: str | None = None


class ToolResult(BaseModel):
    tool: str
    output: str
    error: bool = False


@dataclass
class LoopState:
    goal: str
    messages: list[Message] = field(default_factory=list)
    iteration: int = 0
    done: bool = False
    final_answer: str | None = None


# ---------- Phase functions ----------

def observe(state: LoopState) -> list[dict[str, str]]:
    """Assemble context for the LLM."""
    context: list[dict[str, str]] = [{"role": m.role, "content": m.content} for m in state.messages]
    log.info("[observe] %d messages in context", len(context))
    return context


def think(context: list[dict[str, str]], llm: Callable[[list[dict]], str]) -> ThinkResult:
    """Call LLM and parse its output into a ThinkResult."""
    raw = llm(context)
    log.info("[think] raw output: %s", raw)
    import json
    data = json.loads(raw)
    return ThinkResult(**data)


def act(result: ThinkResult, registry: dict[str, Callable[..., Any]]) -> ToolResult:
    """Dispatch the chosen action."""
    if result.action not in registry:
        return ToolResult(tool=result.action, output=f"unknown action: {result.action}", error=True)
    try:
        output = registry[result.action](**result.args)
        log.info("[act] tool=%s output=%s", result.action, output)
        return ToolResult(tool=result.action, output=str(output))
    except Exception as exc:
        log.error("[act] tool=%s error=%s", result.action, exc)
        return ToolResult(tool=result.action, output=str(exc), error=True)


def observe_result(state: LoopState, tool_result: ToolResult) -> None:
    """Append tool result to context and increment iteration."""
    state.messages.append(Message(role="tool", content=f"{tool_result.tool}: {tool_result.output}"))
    state.iteration += 1
    log.info("[observe_result] iteration=%d", state.iteration)


# ---------- Fake LLM + tools for demo ----------

_turn = 0

def fake_llm(context: list[dict]) -> str:
    global _turn
    _turn += 1
    if _turn == 1:
        return '{"action": "search", "args": {"query": "agent loop"}}'
    return '{"action": "finish", "answer": "An agent loop is observe-think-act-observe."}'


def fake_search(query: str) -> str:
    return f"Results for '{query}': observe-think-act is the ReAct pattern."


# ---------- Full loop ----------

def run_agent(goal: str, max_iterations: int = 10) -> str:
    registry: dict[str, Callable[..., Any]] = {"search": fake_search}
    state = LoopState(goal=goal)
    state.messages.append(Message(role="user", content=goal))

    for _ in range(max_iterations):
        ctx = observe(state)
        think_result = think(ctx, fake_llm)

        if think_result.action == "finish":
            state.done = True
            state.final_answer = think_result.answer
            log.info("[loop] done after %d iterations", state.iteration)
            break

        tool_result = act(think_result, registry)
        observe_result(state, tool_result)
        state.messages.append(
            Message(role="assistant", content=f"called {think_result.action}")
        )
    else:
        state.final_answer = "max iterations reached without answer"

    return state.final_answer or ""


if __name__ == "__main__":
    answer = run_agent("Explain the agent loop.")
    print("Answer:", answer)
```

## Exercises

1. Add a `"calculator"` tool to the `registry` in `run_agent` that accepts `expression: str` and evaluates it with Python's `eval()`. Then add a check: if `expression` contains anything other than digits, operators (`+−*/`), spaces, and parentheses, raise `ValueError`. Write two unit tests — one passing calculation, one blocked unsafe expression.

2. Modify `think` to handle `json.JSONDecodeError` gracefully: if parsing fails, return a `ThinkResult(action="error", args={"raw": raw})`. Write a pytest test that passes a non-JSON string to `think` and asserts it gets back `action="error"`.

3. Add a `history: list[ThinkResult]` to `LoopState`. Record every `ThinkResult` (except finish). Write a test that verifies `history` length equals number of tool calls made.

(Solutions: ../solutions/week-01/day-3.md)

## Self-check

- [ ] I can name all four phases of the agent loop and describe what each one does.
- [ ] I can explain why tool execution happens in the host program, not inside the LLM.
- [ ] I can implement each phase as a standalone function with typed inputs/outputs.
- [ ] I can describe two ways the scratchpad is handled and the trade-off between them.

## Supplementary links

- ReAct paper: https://arxiv.org/abs/2210.03629
- Anthropic tool use overview: https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview
