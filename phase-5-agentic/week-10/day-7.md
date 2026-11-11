# Phase 5 · Week 10 · Day 7 — Mapping From-Scratch Patterns onto LangGraph

> Objective: Translate every agent pattern built from scratch in weeks 1–9 into idiomatic LangGraph.  ·  Time: ~4 hr  ·  Prereqs: [[day-1]], [[day-2]], [[day-3]], [[day-4]], [[day-5]], [[day-6]]

---

## Concept

Weeks 1–9 built agent patterns by hand: raw Python loops, hand-rolled tool dispatchers, custom memory stores. That work was intentional — you understand exactly what every line does. Today you close the loop: map each hand-built pattern onto its LangGraph equivalent, understand what the library buys you, and practice the mechanical refactoring step by step.

### Patterns Learned Weeks 1–9

| Pattern | What you built by hand |
|---|---|
| ReAct loop | `while True` loop — think, pick tool, call tool, observe, repeat or stop |
| Plan-Execute | Planner LLM → list of steps → executor loop → optional replanner |
| Multi-agent coordinator | Supervisor LLM routes messages to specialist sub-agents |
| Tool executor | `if tool_name == "X": call_X()` dispatch table |
| Conversational memory | Append to list, slice window, or embed+retrieve |

Each pattern maps cleanly onto LangGraph primitives. The mapping is mechanical once you recognise the structural equivalence.

---

## How It Works Internally

### ReAct Loop → LangGraph Nodes

A hand-rolled ReAct loop:

```python
while True:
    thought = llm.invoke(messages)        # think
    if thought.tool_calls:
        result = execute_tool(thought)    # act
        messages.append(result)           # observe
    else:
        break                             # done
```

LangGraph equivalent — three nodes connected by a conditional edge:

```
think_node → [conditional] → act_node → observe_node → back to think_node
                           ↘ END (no tool calls)
```

- **`think_node`**: calls LLM, writes `AIMessage` with optional `tool_calls` to state.
- **`act_node`**: executes the tool call. In LangGraph use `ToolNode` from `langgraph.prebuilt` — it reads `tool_calls` from the last `AIMessage` automatically.
- **`observe_node`**: (often merged into `ToolNode`) writes `ToolMessage` result to state.
- **Conditional edge** after `think_node`: `if last_message.tool_calls → "act"` else `→ END`.

`langgraph.prebuilt.create_react_agent` packages exactly this topology. Use it for standard ReAct; build manually when you need non-standard routing.

### Plan-Execute → LangGraph

Hand-rolled plan-execute:

```python
plan = planner_llm.invoke(task)           # list of steps
for step in plan.steps:
    result = executor_llm.invoke(step)
    if needs_replan(result):
        plan = replanner_llm.invoke(...)
```

LangGraph equivalent:

```
planner_node → executor_node → replanner_node → [conditional]
                                                 ↙         ↘
                                         executor_node    END
```

- **`planner_node`**: calls LLM, writes `Plan` (list of steps) to state.
- **`executor_node`**: pops the next step, runs it, appends result to `past_steps`.
- **`replanner_node`**: calls LLM with original task + past steps, decides `continue` or `done`.
- **Conditional edge** after replanner: if `done` → END, else → executor.

### Multi-Agent Coordinator → LangGraph

Hand-rolled supervisor:

```python
while True:
    next_agent = supervisor_llm.invoke(state)
    if next_agent == "FINISH":
        break
    result = agents[next_agent].invoke(state)
    state.update(result)
```

LangGraph equivalent:

```
supervisor_node → [conditional route] → specialist_A_subgraph
                                      → specialist_B_subgraph
                                      → END
```

- **`supervisor_node`**: LLM picks next agent name or `FINISH`.
- **Conditional edge**: routes to the named subgraph node.
- **Specialist nodes**: can be full subgraphs compiled separately and added as nodes via `builder.add_node("specialist_a", specialist_a_graph)`.
- State flows back to supervisor after each specialist returns.

### Tool Execution → ToolNode

Hand-rolled dispatcher:

```python
tool_map = {"search": search_fn, "calculator": calc_fn}
result = tool_map[tool_name](tool_args)
```

LangGraph equivalent:

```python
from langgraph.prebuilt import ToolNode

tools = [search_fn, calc_fn]  # LangChain @tool-decorated functions
tool_node = ToolNode(tools)
builder.add_node("tools", tool_node)
```

`ToolNode` reads `tool_calls` from the last `AIMessage` in state, dispatches, and writes `ToolMessage` results back. Handles errors, multiple parallel tool calls, and type coercion automatically.

### Memory → LangGraph Checkpointing + MessagesState

Hand-rolled memory:

```python
messages: list[BaseMessage] = []
messages.append(HumanMessage(content=user_input))
# ... later
messages = messages[-10:]  # window trim
```

LangGraph equivalent:

```python
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # append-only reducer
```

- `add_messages` is a **reducer**: when a node returns `{"messages": [new_msg]}`, it appends rather than overwrites.
- `MemorySaver` persists the full state across turns via `thread_id`.
- For windowed memory: use `trim_messages` from `langchain_core.messages` inside a node before the LLM call.

### LangGraph Built-ins vs DIY

| Built-in | DIY equivalent | When to use built-in |
|---|---|---|
| `create_react_agent` | Manual think/act/observe graph | Standard ReAct, no custom routing |
| `ToolNode` | Hand-written dispatcher | Any tool execution; handles parallel calls |
| `MessagesState` | `TypedDict` with `add_messages` | Simple message-list state |
| `MemorySaver` | In-memory dict store | Dev/test; swap for SqliteSaver in prod |

Use built-ins until you need to deviate. Deviation signals = custom routing logic, non-message state, side effects between tool calls.

---

## Worked Example

**Refactoring a hand-rolled ReAct loop into LangGraph — step by step.**

### Starting point: hand-rolled ReAct

```python
"""
week10/day7_before_refactor.py — hand-rolled ReAct loop.

Dependencies (uv add):
  langchain-anthropic
  langchain-core
"""

import logging
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


TOOLS = {t.name: t for t in [multiply, add]}

llm = ChatAnthropic(
    model="claude-sonnet-5-5",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
).bind_tools(list(TOOLS.values()))


def react_loop(user_input: str) -> str:
    messages: list[BaseMessage] = [HumanMessage(content=user_input)]
    for _ in range(10):  # safety cap
        response: AIMessage = llm.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return response.content
        for call in response.tool_calls:
            result = TOOLS[call["name"]].invoke(call["args"])
            messages.append(
                ToolMessage(content=str(result), tool_call_id=call["id"])
            )
    return "Max iterations reached."


if __name__ == "__main__":
    answer = react_loop("What is (3 * 4) + 7?")
    logger.info("Answer: %s", answer)
```

### Step 1 — Define typed state

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class ReActState(TypedDict):
    messages: Annotated[list, add_messages]
```

### Step 2 — Lift the LLM call into a node

```python
def think(state: ReActState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}
```

### Step 3 — Replace the tool dispatch with ToolNode

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode([multiply, add])
```

### Step 4 — Wire the conditional edge

```python
from langchain_core.messages import AIMessage

def should_continue(state: ReActState) -> str:
    last: AIMessage = state["messages"][-1]
    return "tools" if last.tool_calls else "__end__"
```

### Step 5 — Build the graph

```python
"""
week10/day7_after_refactor.py — LangGraph ReAct.

Dependencies (uv add):
  langchain-anthropic
  langgraph
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
from typing_extensions import TypedDict

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


class ReActState(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatAnthropic(
    model="claude-sonnet-5-5",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
).bind_tools([multiply, add])


def think(state: ReActState) -> dict:
    logger.info("Node: think")
    response: AIMessage = llm.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: ReActState) -> str:
    last: AIMessage = state["messages"][-1]
    return "tools" if last.tool_calls else "__end__"


def build_graph() -> StateGraph:
    builder = StateGraph(ReActState)
    builder.add_node("think", think)
    builder.add_node("tools", ToolNode([multiply, add]))
    builder.set_entry_point("think")
    builder.add_conditional_edges("think", should_continue, {"tools": "tools", "__end__": END})
    builder.add_edge("tools", "think")
    return builder


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    graph = build_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "react-001"}}

    for step in graph.stream(
        {"messages": [HumanMessage(content="What is (3 * 4) + 7?")]},
        config=config,
    ):
        node_name = list(step.keys())[0]
        logger.info("Step: %s", node_name)

    final = graph.get_state(config).values["messages"][-1]
    logger.info("Answer: %s", final.content)
```

**What changed, what stayed the same:**

| Aspect | Before | After |
|---|---|---|
| Loop control | `for _ in range(10)` | Conditional edge, LangGraph handles cycle |
| Tool dispatch | Manual `TOOLS[call["name"]].invoke(...)` | `ToolNode` |
| State | Local `messages` list | `ReActState` TypedDict with `add_messages` |
| Persistence | None | `MemorySaver` + `thread_id` |
| Observability | `logger.info` manually | LangSmith auto-traces every node |

---

## Exercises

*(Solutions: ../solutions/week-10/day-7.md)*

Exercises build on each other. Work through them in order. Each exercise has a starter file path in the comment header.

**Exercise 1 — State Audit**

Take any hand-rolled agent from weeks 1–5 in your course files. List every piece of mutable state it maintains (e.g., `messages`, `plan`, `step_index`, `results`). For each, write the corresponding `TypedDict` field with the correct reducer annotation (`add_messages` for message lists, default overwrite for scalar values, `operator.add` for accumulating lists). Do not implement the graph yet.

**Exercise 2 — ReAct Refactor**

Complete the refactor started in the worked example. Add a third tool `subtract(a: int, b: int) -> int`. Add a safety-cap node `safety_check` that reads `len(state["messages"])` and routes to `END` if over 20 messages. Wire it as an additional conditional check inside `should_continue`. Write one `pytest` test that runs the graph with a mocked LLM returning a fixed sequence: one `AIMessage` with a tool call, one `ToolMessage` from `ToolNode`, one `AIMessage` with no tool calls.

**Exercise 3 — Plan-Execute in LangGraph**

Implement the plan-execute pattern as a LangGraph graph with these nodes: `planner`, `executor`, `replanner`. Use `claude-sonnet-5-5` for planner and replanner, `claude-haiku-5-5` for the executor (cheaper for individual steps). State must include: `task: str`, `plan: list[str]`, `current_step: int`, `past_steps: list[str]`, `final_answer: str`. Add a conditional edge after `replanner`: if `current_step >= len(plan)` and no replan needed, route to END. Write two tests: one for the happy path (planner produces 2 steps, executor completes both, replanner says done), one for the replan path (replanner produces a new plan after step 1).

**Exercise 4 — Supervisor Multi-Agent**

Implement a supervisor that routes to two specialists: `researcher` (uses a `web_search` mock tool) and `writer` (uses no tools, just drafts text). The supervisor node calls `claude-opus-5-5` to decide routing or `FINISH`. Use a `Literal["researcher", "writer", "FINISH"]` return type for the routing function. State: `messages`, `task`, `draft`. Wire so that after `writer` completes, control returns to `supervisor` (which then outputs FINISH). Write a test with mocked supervisor always routing `researcher → writer → FINISH`.

**Exercise 5 — Memory and Checkpointing**

Take the ReAct graph from Exercise 2. Add a second `thread_id` and verify via `graph.get_state()` that the two threads maintain independent state. Then add `interrupt_before=["think"]` with a checkpointer. Simulate: thread A starts, interrupts, human injects a correction via `update_state()`, resumes. Thread B runs independently without interruption. Assert that thread A's final answer reflects the injected correction and thread B is unaffected.

**Exercise 6 — create_react_agent vs Manual**

Use `langgraph.prebuilt.create_react_agent` to build the same ReAct agent from Exercise 2 in fewer lines. Then list three concrete limitations of `create_react_agent` that would force you to build the graph manually. For each limitation, write a one-sentence description of the scenario that exposes it. (No code needed for the limitation list — prose is fine.)

---

## Self-check

- What is the structural difference between a conditional edge and a regular edge in LangGraph?
- Why does `add_messages` exist as a reducer instead of simple list overwrite?
- When does `ToolNode` fail silently vs raise an exception?
- What does `create_react_agent` add over a manual think/act graph that uses `ToolNode`?
- Name the three nodes in a plan-execute LangGraph and their LLM roles.
- How do you access state between two independent threads using the same compiled graph?

---

## Supplementary Links

- [LangGraph conceptual overview](https://langchain-ai.github.io/langgraph/concepts/)
- [`create_react_agent` reference](https://langchain-ai.github.io/langgraph/reference/prebuilt/#create_react_agent)
- [`ToolNode` reference](https://langchain-ai.github.io/langgraph/reference/prebuilt/#toolnode)
- [Plan-execute tutorial](https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/plan-and-execute/)
- [Multi-agent supervisor tutorial](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/)
- [`add_messages` reducer](https://langchain-ai.github.io/langgraph/reference/graphs/#add_messages)
- [LangGraph Send API (fan-out)](https://langchain-ai.github.io/langgraph/concepts/low_level/#send)
