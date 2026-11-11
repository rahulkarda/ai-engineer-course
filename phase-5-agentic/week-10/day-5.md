# Phase 5 · Week 10 · Day 5 — LangGraph Conditional Edges, Cycles & Checkpointing

> Objective: Add conditional routing, cycles, and persistent checkpointing to a LangGraph agent.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

---

## Concept

Day 4 covered linear graphs: every edge was static and the graph ran top-to-bottom once. Real agents need:

- **Conditional routing** — choose the next node at runtime based on state
- **Cycles** — loop back to an earlier node (retry, reflection, planning iteration)
- **Checkpointing** — serialize state between steps so runs can be paused and resumed

These three features together are what elevate LangGraph above a simple chain and make it suitable for long-running, resumable agent workflows.

---

## How It Works Internally

### Conditional edges

```python
graph.add_conditional_edges(
    source="verify",          # the node whose output we're routing from
    path=router_fn,           # callable(state) -> str  —  returns next node name
    path_map={                # optional: map router return values to node names
        "retry": "plan",
        "done": END,
    },
)
```

`router_fn` receives the full current state and returns a string — either a node name or the `END` sentinel. `path_map` is optional; if omitted, the return value is used directly as the next node name.

`END` is imported from `langgraph.graph`:

```python
from langgraph.graph import END
```

Routing to `END` terminates the graph and returns final state to the caller.

### Cycles

A cycle is just a conditional edge that routes back to a previous node. The graph executor will keep running until a node routes to `END`. Without a termination condition this is an infinite loop — always wire a guard.

Typical pattern: a `verify` node inspects quality and routes to `plan` (retry) or `END` (done):

```
plan → execute → verify → plan   (cycle: quality too low)
                        ↓
                       END        (quality threshold met)
```

### MemorySaver and SqliteSaver

LangGraph ships two built-in checkpointers:

| Checkpointer | Storage | Use case |
|---|---|---|
| `MemorySaver` | In-process dict | Development, testing, short sessions |
| `SqliteSaver` | SQLite file | Persistent across process restarts |

Both implement the `BaseCheckpointSaver` interface. Swap them without changing graph logic.

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

# In-memory (default for dev)
checkpointer = MemorySaver()

# Persistent SQLite
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
```

### The checkpoint tuple

Internally, each checkpoint is a `(config, state, next)` tuple:

- `config` — the `RunnableConfig` dict including `thread_id`
- `state` — full serialized state at that step
- `next` — list of nodes scheduled to run next (empty at completion)

You can inspect stored checkpoints via `app.get_state_history(config)`.

### Thread IDs

Every invocation that should share state belongs to a **thread**. Pass a `thread_id` via the `config` argument:

```python
config = {"configurable": {"thread_id": "run-42"}}
result = app.invoke(initial_state, config=config)
```

Two invocations with the same `thread_id` share the same checkpoint history. Different `thread_id` values are completely isolated.

### Resuming from a checkpoint

To resume a paused or interrupted run, pass `None` as input (LangGraph uses the last checkpoint as state) and re-pass the same `config`:

```python
# First invocation (runs until interrupt or END)
app.invoke(initial_state, config=config)

# Resume from last checkpoint
app.invoke(None, config=config)
```

If the graph ran to `END` the previous time, resuming is a no-op. Use `app.get_state(config)` to inspect where execution stopped.

---

## Worked Example

Build a **plan → execute → verify** cycle. The `verify` node checks output quality against a threshold and routes back to `plan` (retry) or to `END` (done). State carries an iteration counter as a cycle-termination guard.

### Dependencies

```
uv add langchain langgraph langchain-anthropic
```

### Full example

```python
import logging
import os
from typing import Annotated, TypedDict
import operator

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QUALITY_THRESHOLD = 0.7
MAX_ITERATIONS = 3

# --- State ---

class PlanExecuteState(TypedDict):
    goal: str
    plan: str
    result: str
    quality: float
    iterations: int
    messages: Annotated[list[str], operator.add]

# --- LLM ---

llm = ChatAnthropic(
    model="claude-sonnet-5-5",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

# --- Nodes ---

def plan_node(state: PlanExecuteState) -> dict:
    """Generate or refine a plan for the goal."""
    iteration = state["iterations"] + 1
    logger.info("plan_node: iteration=%d", iteration)

    context = ""
    if state["result"]:
        context = f"\nPrevious attempt result: {state['result']}\nQuality score: {state['quality']:.2f}. Improve on this."

    prompt = f"Goal: {state['goal']}{context}\n\nWrite a concise step-by-step plan (3 steps max)."
    response = llm.invoke([HumanMessage(content=prompt)])
    plan = response.content.strip()
    logger.info("plan_node: plan generated (%d chars)", len(plan))

    return {
        "plan": plan,
        "iterations": iteration,
        "messages": [f"plan: iteration {iteration}"],
    }


def execute_node(state: PlanExecuteState) -> dict:
    """Execute the current plan and produce a result."""
    logger.info("execute_node: executing plan")
    prompt = (
        f"Goal: {state['goal']}\n"
        f"Plan:\n{state['plan']}\n\n"
        "Execute this plan and produce the final output."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    result = response.content.strip()
    logger.info("execute_node: result length=%d", len(result))

    return {
        "result": result,
        "messages": [f"execute: {len(result)} chars"],
    }


def verify_node(state: PlanExecuteState) -> dict:
    """Score the result quality as a float 0.0–1.0."""
    logger.info("verify_node: scoring result")
    prompt = (
        f"Goal: {state['goal']}\n"
        f"Result:\n{state['result']}\n\n"
        "Rate how well this result achieves the goal. "
        "Reply with ONLY a decimal number between 0.0 and 1.0."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        quality = float(response.content.strip())
        quality = max(0.0, min(1.0, quality))
    except ValueError:
        logger.warning("verify_node: could not parse quality score, defaulting to 0.5")
        quality = 0.5

    logger.info("verify_node: quality=%.2f (threshold=%.2f)", quality, QUALITY_THRESHOLD)

    return {
        "quality": quality,
        "messages": [f"verify: quality={quality:.2f}"],
    }

# --- Router ---

def quality_router(state: PlanExecuteState) -> str:
    """Route to 'plan' for retry or END if quality threshold met or max iterations reached."""
    if state["quality"] >= QUALITY_THRESHOLD:
        logger.info("quality_router: quality threshold met — routing to END")
        return "done"
    if state["iterations"] >= MAX_ITERATIONS:
        logger.info("quality_router: max iterations reached — routing to END")
        return "done"
    logger.info("quality_router: quality too low — routing back to plan")
    return "retry"

# --- Graph construction ---

def build_graph(checkpointer: MemorySaver | None = None) -> StateGraph:
    graph = StateGraph(PlanExecuteState)

    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("verify", verify_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "verify")

    graph.add_conditional_edges(
        source="verify",
        path=quality_router,
        path_map={
            "retry": "plan",   # cycle back
            "done": END,       # terminate
        },
    )

    return graph.compile(checkpointer=checkpointer)


# --- Entrypoint ---

def main() -> None:
    checkpointer = MemorySaver()
    app = build_graph(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "demo-run-1"}}

    initial_state: PlanExecuteState = {
        "goal": "Write a one-paragraph summary of transformer architecture.",
        "plan": "",
        "result": "",
        "quality": 0.0,
        "iterations": 0,
        "messages": [],
    }

    logger.info("Starting graph invocation")
    final_state = app.invoke(initial_state, config=config)

    print("=== Final result ===")
    print(final_state["result"])
    print(f"\nQuality: {final_state['quality']:.2f}  |  Iterations: {final_state['iterations']}")
    print("\n--- message log ---")
    for msg in final_state["messages"]:
        print(" ", msg)

    # Show checkpoint history
    print("\n--- checkpoint history ---")
    for checkpoint in app.get_state_history(config):
        next_nodes = checkpoint.next
        step_msgs = checkpoint.values.get("messages", [])
        print(f"  step messages={step_msgs[-1] if step_msgs else '(none)'}  next={next_nodes}")

    # Demonstrate resumption (graph already at END, so this is a no-op that returns final state)
    logger.info("Attempting resume (should be no-op — graph already at END)")
    resumed_state = app.invoke(None, config=config)
    assert resumed_state["result"] == final_state["result"], "Resumed state should match"
    logger.info("Resume confirmed — state consistent")


if __name__ == "__main__":
    main()
```

**Key points:**

- `quality_router` is the only place routing logic lives — nodes stay pure.
- `path_map` maps router return strings to node names; `"done"` maps to the `END` sentinel.
- `MAX_ITERATIONS` guard prevents infinite cycles — always include one.
- `MemorySaver` is passed to `compile()`, not to individual nodes.
- `thread_id` in `config` is what ties invocations together for checkpointing.
- `app.get_state_history(config)` returns all checkpoints for the thread in reverse chronological order.
- `app.invoke(None, config=config)` resumes from the last checkpoint.

---

## Exercises

(Solutions: ../solutions/week-10/day-5.md)

**Exercise 1 — Hard interrupt**

Add a `graph.compile(..., interrupt_before=["execute"])` to pause execution before the `execute` node. Run the graph with `stream()` instead of `invoke()` and observe that it stops before execution. Then call `app.invoke(None, config=config)` to resume. Print the state at the interruption point using `app.get_state(config)`.

**Exercise 2 — SqliteSaver persistence**

Replace `MemorySaver` with `SqliteSaver.from_conn_string("checkpoints.db")`. Run the graph in one Python process, then in a fresh process (new script invocation) pass the same `thread_id` and call `app.invoke(None, config=config)` with no initial state. Verify that the final state is correctly retrieved from the database.

**Exercise 3 — Multiple threads**

Run the same `build_graph()` for three different goals simultaneously (sequential calls, different `thread_id` values: `"run-a"`, `"run-b"`, `"run-c"`). After all three complete, use `app.get_state_history(config)` for each thread to print a per-thread summary: number of checkpoints, final quality, iterations used. Confirm threads are fully isolated.

**Exercise 4 — Dynamic cycle limit via state**

Remove the `MAX_ITERATIONS` module-level constant. Instead, add a `max_iterations: int` field to `PlanExecuteState` with a default of `3`. Update `quality_router` to read `state["max_iterations"]` for the guard. Pass `max_iterations=2` in the initial state for one run and `max_iterations=5` for another. Compare the number of retry cycles in each run's message log.

---

## Self-check

- What is the return type of a router function passed to `add_conditional_edges()`?
- Why must every cycle have a termination condition that routes to `END`?
- What does `MemorySaver` store and where does the data live?
- What is a thread ID and what happens if two runs share the same thread ID?
- When you call `app.invoke(None, config=config)`, what state does LangGraph use as input?
- What does the `next` field of a checkpoint tuple represent?

---

## Supplementary Links

- [LangGraph concepts — Conditional edges](https://langchain-ai.github.io/langgraph/concepts/low_level/#conditional-edges)
- [LangGraph concepts — Cycles](https://langchain-ai.github.io/langgraph/concepts/low_level/#cycles)
- [LangGraph how-to — Persistence / checkpointing](https://langchain-ai.github.io/langgraph/how-tos/persistence/)
- [LangGraph how-to — Human-in-the-loop (interrupt)](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/wait-user-input/)
- [LangGraph API — MemorySaver](https://langchain-ai.github.io/langgraph/reference/checkpoints/#memorysaver)
- [LangGraph API — SqliteSaver](https://langchain-ai.github.io/langgraph/reference/checkpoints/#sqlitesaver)
- [LangGraph how-to — Time-travel (state history)](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/time-travel/)
