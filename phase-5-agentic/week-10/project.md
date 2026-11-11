# Week 10 Project — Re-implement Plan-Execute Agent in LangGraph

> Phase 5 · Agentic AI with LangGraph  
> Implementation: ~3 hr  ·  Assessment: ~1 hr  ·  Total: ~4 hr

---

## Goal

Rebuild the plan-execute multi-agent you built from scratch in weeks 1–9 using idiomatic LangGraph. The result must be functionally equivalent — given a task, it plans steps, executes them with tools, and replans if needed — but uses LangGraph state, checkpointing, HITL, and streaming throughout.

---

## Requirements

### Functional requirements

1. **Planner node** — calls `claude-sonnet-5-5` to decompose `task` into a `list[str]` of steps. Writes to `AgentState.plan`.
2. **Executor node** — pops `plan[current_step]`, calls `claude-haiku-5-5` with available tools, writes result to `past_steps`. Increments `current_step`.
3. **Replanner conditional node** — calls `claude-sonnet-5-5` with `task + past_steps`. Returns either a new plan (overwrite `plan`, reset `current_step`) or sets `done: bool = True`.
4. **HITL interrupt** — `interrupt_before=["executor"]` so a human can approve/edit the plan before any execution starts. Uses `MemorySaver` + `thread_id`.
5. **Streaming** — caller uses `graph.stream()` and logs each completed node name.
6. **Tools** — at minimum two mock tools decorated with `@tool`: `web_search(query: str) -> str` and `calculate(expression: str) -> str`. Real implementations are stubs; tests mock them.
7. **ToolNode** — executor uses `langgraph.prebuilt.ToolNode` for tool dispatch (not manual `if/else`).

### Non-functional requirements

- `AgentState` is a `TypedDict` — no untyped dicts.
- All Pydantic models for structured LLM output use Pydantic v2 `BaseModel`.
- `ANTHROPIC_API_KEY` via `os.getenv` only — no hardcoded keys.
- `logging` not `print`.
- At least **4 pytest tests** with mocked LLM (use `FakeListChatModel` or `unittest.mock.patch`):
  - Test 1: planner produces 2 steps, executor completes both, replanner sets `done=True`, final state correct.
  - Test 2: replanner produces a new plan after step 1 fails (adds a recovery step).
  - Test 3: HITL flow — graph pauses at interrupt, `update_state()` injects modified plan, graph resumes and completes.
  - Test 4: `current_step` never exceeds `len(plan)` (boundary guard test).
- Code is `ruff`-clean.

---

## Acceptance Criteria

- [ ] `AgentState` TypedDict defined with all required fields (see starter code).
- [ ] `planner_node` function signature: `(state: AgentState) -> dict` — pure function.
- [ ] `executor_node` function signature: `(state: AgentState) -> dict` — pure function.
- [ ] `replanner_node` function signature: `(state: AgentState) -> dict` — pure function.
- [ ] `should_continue` edge function returns `"executor"` or `"__end__"` (string literals).
- [ ] `ToolNode([web_search, calculate])` added as `"tools"` node.
- [ ] Graph compiled with `MemorySaver` and `interrupt_before=["executor"]`.
- [ ] `graph.stream()` used in `main()` — each step logged by node name.
- [ ] Human approval simulated with `graph.update_state()` and `graph.invoke(None, config)` resume.
- [ ] All 4 tests pass: `pytest tests/ -q` exits 0.
- [ ] No `print()` calls. No hardcoded secrets. `os.getenv("ANTHROPIC_API_KEY")` used at LLM init.
- [ ] File is under 300 lines or split into modules if over (per global CLAUDE.md rule).

---

## Starter Code Scaffold

```python
"""
week10/project_plan_execute.py

Plan-Execute agent in LangGraph.

Dependencies (uv add):
  langchain-anthropic
  langgraph
  langchain-core

Usage:
  ANTHROPIC_API_KEY=... python week10/project_plan_execute.py
"""

import logging
import os
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel
from typing_extensions import TypedDict

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models for structured LLM output
# ---------------------------------------------------------------------------

class PlannerOutput(BaseModel):
    """TODO: Define fields — steps list and any metadata."""
    # steps: list[str]
    ...


class ReplannerOutput(BaseModel):
    """TODO: Define fields — new plan or done flag."""
    # new_plan: list[str] | None
    # done: bool
    ...


# ---------------------------------------------------------------------------
# Typed state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """TODO: Add all required fields with correct types and reducers."""
    # task: str
    # plan: list[str]
    # current_step: int
    # past_steps: list[str]
    # done: bool
    # messages: Annotated[list, add_messages]
    ...


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def web_search(query: str) -> str:
    """Search the web for information. Returns a stub result."""
    # TODO: implement stub (return f"Search result for: {query}")
    raise NotImplementedError


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Returns a stub result."""
    # TODO: implement stub (safe eval or return f"Result of: {expression}")
    raise NotImplementedError


# ---------------------------------------------------------------------------
# LLM clients — initialise lazily so tests can patch os.getenv
# ---------------------------------------------------------------------------

def get_planner_llm() -> ChatAnthropic:
    """TODO: return ChatAnthropic with model=claude-sonnet-5-5, api_key from env."""
    raise NotImplementedError


def get_executor_llm() -> ChatAnthropic:
    """TODO: return ChatAnthropic with model=claude-haiku-5-5 bound to tools, api_key from env."""
    raise NotImplementedError


def get_replanner_llm() -> ChatAnthropic:
    """TODO: return ChatAnthropic with model=claude-sonnet-5-5, api_key from env."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def planner_node(state: AgentState) -> dict:
    """
    TODO:
    1. Call get_planner_llm() with state["task"].
    2. Parse response into PlannerOutput.
    3. Return dict updating plan and resetting current_step to 0.
    """
    raise NotImplementedError


def executor_node(state: AgentState) -> dict:
    """
    TODO:
    1. Get current step: state["plan"][state["current_step"]].
    2. Call get_executor_llm() with the step as a HumanMessage.
    3. Append (step, result) to past_steps.
    4. Increment current_step.
    5. Return dict.
    """
    raise NotImplementedError


def replanner_node(state: AgentState) -> dict:
    """
    TODO:
    1. Call get_replanner_llm() with task + past_steps summary.
    2. Parse into ReplannerOutput.
    3. If done: return {"done": True}.
    4. If new_plan: return {"plan": new_plan, "current_step": 0}.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------

def should_continue(state: AgentState) -> str:
    """
    TODO:
    Return "executor" if not done and current_step < len(plan).
    Return "__end__" otherwise.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    TODO:
    1. Create StateGraph(AgentState).
    2. Add nodes: planner_node, executor_node, ToolNode([web_search, calculate]), replanner_node.
    3. Set entry point to planner_node.
    4. Add edges: planner → executor, executor → tools, tools → executor (loop back), executor → replanner.
    5. Add conditional edge from replanner using should_continue.
    6. Return builder (NOT compiled — compile in main so tests can compile with different options).
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    TODO:
    1. Build and compile graph with MemorySaver + interrupt_before=["executor"].
    2. Set config with a fixed thread_id.
    3. Stream first pass (planner runs, graph pauses at executor interrupt).
    4. Log the generated plan from graph.get_state(config).
    5. Simulate human approval: graph.update_state(config, {"approved": True}) — add approved field to state.
    6. Resume with graph.stream(None, config) and log each step.
    7. Log final state.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
```

```python
"""
tests/test_project_plan_execute.py

Pytest tests for week10/project_plan_execute.py.
All LLM calls must be mocked — no real API calls.

Dependencies (uv add --dev):
  pytest
  pytest-mock  (optional, unittest.mock is fine too)
"""

import pytest

# TODO: Import your graph builder and state types here.
# from week10.project_plan_execute import build_graph, AgentState

# ---------------------------------------------------------------------------
# Test 1: Happy path — 2 steps, no replan, done=True
# ---------------------------------------------------------------------------

def test_happy_path_two_steps(monkeypatch):
    """
    TODO:
    - Mock get_planner_llm() to return FakeListChatModel with a 2-step plan.
    - Mock get_executor_llm() to return a stub response for each step.
    - Mock get_replanner_llm() to return done=True after both steps.
    - Compile graph WITHOUT interrupt (to run headless in tests).
    - Invoke with a test task.
    - Assert final state: done=True, len(past_steps)==2, current_step==2.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Test 2: Replan path — replanner produces new plan after step 1
# ---------------------------------------------------------------------------

def test_replan_after_first_step(monkeypatch):
    """
    TODO:
    - Mock planner to return a 1-step plan.
    - Mock executor to return a result.
    - Mock replanner to first return a new 1-step plan, then return done=True.
    - Assert final state: done=True, len(past_steps)==2 (original + recovery step).
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Test 3: HITL interrupt and resume
# ---------------------------------------------------------------------------

def test_hitl_interrupt_and_resume(monkeypatch):
    """
    TODO:
    - Compile graph WITH interrupt_before=["executor"] and MemorySaver.
    - Stream first pass — assert graph pauses (snapshot.next != ()).
    - Inject a modified plan via graph.update_state().
    - Resume with graph.stream(None, config).
    - Assert final plan in state matches the injected plan.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Test 4: Boundary guard — current_step never exceeds len(plan)
# ---------------------------------------------------------------------------

def test_current_step_boundary(monkeypatch):
    """
    TODO:
    - Mock planner to return a 2-step plan.
    - Mock executor to complete both steps.
    - Mock replanner to return done=True.
    - After graph completes, assert state["current_step"] == 2 (== len(plan)).
    - Assert should_continue would return "__end__" given this state.
    """
    raise NotImplementedError
```

---

## Grading Rubric

| Dimension | Weight | Indicators |
|---|---|---|
| Independent implementation | 40% | Graph runs end-to-end without starter code scaffolding; all five nodes implemented; HITL resume works; stream output shows each node |
| Conceptual understanding | 25% | Correct use of TypedDict state; Pydantic v2 models for LLM output; ToolNode used (not manual dispatch); `should_continue` returns correct string literals |
| Debugging & problem-solving | 20% | Retry/replan logic terminates correctly; boundary guard prevents off-by-one on `current_step`; mocks accurately simulate LLM sequences; `get_state()` used to inspect interrupt snapshot |
| Code quality & testing | 15% | All 4 tests pass; LLM calls mocked (no real API calls in tests); `logging` not `print`; `ruff`-clean; no hardcoded secrets; file under 300 lines or split |

> Answer key and reference implementation: ../solutions/week-10/project.md

---

## Hints

- `FakeListChatModel` from `langchain_community.llms.fake` simulates sequential LLM responses. Each call pops the next response from the list.
- For structured output (PlannerOutput, ReplannerOutput), use `llm.with_structured_output(PlannerOutput)` — this chains the LLM with a Pydantic parser automatically.
- When `interrupt_before` is set and you stream with `graph.stream(input, config)`, the generator completes after the interrupt. Check `graph.get_state(config).next` — if it is non-empty, the graph is paused.
- `graph.stream(None, config)` (input=None, same config) resumes from the checkpoint. Do not pass the original input again — that would restart from the entry node.
- `ToolNode` expects `tool_calls` on the last `AIMessage`. If your executor LLM returns a `ToolMessage` directly (mock scenario), you may need to wrap it.

---

## Supplementary Links

- [Plan-execute tutorial](https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/plan-and-execute/)
- [`ToolNode` reference](https://langchain-ai.github.io/langgraph/reference/prebuilt/#toolnode)
- [LangGraph HITL guide](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)
- [`with_structured_output` guide](https://python.langchain.com/docs/how_to/structured_output/)
- [`FakeListChatModel`](https://python.langchain.com/docs/integrations/chat/fake/)
- [Pydantic v2 `BaseModel`](https://docs.pydantic.dev/latest/concepts/models/)
- [LangGraph streaming](https://langchain-ai.github.io/langgraph/concepts/streaming/)
