# Phase 5 · Week 10 · Day 4 — LangGraph Basics: State, Nodes, Edges

> Objective: Build a compiled, stateful 3-node LangGraph pipeline with typed state and pure node functions.  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

---

## Concept

**LangGraph** is a library built on top of LangChain for constructing stateful, cyclic agent graphs. Where LangChain Expression Language (LCEL) is a linear, chainable pipeline, LangGraph lets you define:

- **Shared, typed state** that persists across every node invocation
- **Explicit edges** (static or conditional) controlling execution flow
- **Cycles** — a node can route back to a previous node, enabling retry loops, reflective agents, and multi-step planning

The core primitive is a `StateGraph`: you describe nodes, wire them with edges, compile the graph, then invoke it.

### StateGraph vs MessageGraph

| | `StateGraph` | `MessageGraph` |
|---|---|---|
| State type | Any `TypedDict` or Pydantic model you define | Hardcoded `list[BaseMessage]` |
| Flexibility | Full — carry any typed fields | Narrow — message list only |
| Use case | Agents with complex intermediate state | Simple chat pipelines |

Prefer `StateGraph` for anything non-trivial. `MessageGraph` is a convenience wrapper around `StateGraph` with a pre-wired message accumulator.

---

## How It Works Internally

### State definition

LangGraph state is a `TypedDict`. Fields annotated with `Annotated[list[X], operator.add]` are **reducers**: instead of replacing the field on each node update, new values are appended to the existing list.

```python
import operator
from typing import Annotated, TypedDict

class AgentState(TypedDict):
    query: str
    classification: str
    answer: str
    messages: Annotated[list[str], operator.add]  # reducer — appended, not replaced
```

### Graph construction API

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(AgentState)

graph.add_node("classify", classify_node)   # name → callable(state) -> dict
graph.add_node("answer", answer_node)
graph.add_node("format", format_node)

graph.set_entry_point("classify")           # first node invoked
graph.add_edge("classify", "answer")        # static edge
graph.add_edge("answer", "format")
graph.set_finish_point("format")            # equivalent to add_edge("format", END)

app = graph.compile()
```

Each node is a **pure function** with signature `(state: AgentState) -> dict`. It returns only the keys it wants to update — LangGraph merges the dict into the current state (using reducers where defined).

### Execution model

When you call `app.invoke(initial_state)`:

1. Executor starts at the entry point node.
2. Node function receives the full current state dict.
3. Node returns a partial dict; executor merges it into state.
4. Executor follows the outgoing edge to find the next node.
5. Repeats until a node routes to `END` (or `set_finish_point` is reached).
6. Final state is returned to the caller.

State is passed by value between steps — nodes cannot hold side effects in closures that depend on mutable shared objects.

### LCEL vs LangGraph

| Dimension | LCEL | LangGraph |
|---|---|---|
| Flow shape | Strictly linear (DAG, no back-edges) | Cyclic graphs allowed |
| State | Input → output piped via `|` | Explicit shared `TypedDict` |
| Flow control | Implicit (chain order) | Explicit edges + conditional routers |
| Use case | Transforms, retrieval, simple chains | Agents, retry loops, planning cycles |

---

## Worked Example

Build a 3-node linear graph: `classify` → `answer` → `format`.

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- State ---

class AgentState(TypedDict):
    query: str
    classification: str
    answer: str
    messages: Annotated[list[str], operator.add]

# --- LLM ---

llm = ChatAnthropic(
    model="claude-haiku-5-5",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

# --- Nodes (pure functions) ---

def classify_node(state: AgentState) -> dict:
    """Classify the query as factual, opinion, or procedural."""
    logger.info("classify_node: classifying query")
    prompt = f"Classify this query in one word (factual/opinion/procedural): {state['query']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    classification = response.content.strip().lower()
    logger.info("classify_node: classification=%s", classification)
    return {
        "classification": classification,
        "messages": [f"classify: {classification}"],
    }


def answer_node(state: AgentState) -> dict:
    """Generate an answer given the classification."""
    logger.info("answer_node: generating answer (type=%s)", state["classification"])
    prompt = (
        f"Query type: {state['classification']}\n"
        f"Query: {state['query']}\n"
        "Provide a concise answer."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    answer = response.content.strip()
    logger.info("answer_node: answer generated (%d chars)", len(answer))
    return {
        "answer": answer,
        "messages": [f"answer: {len(answer)} chars"],
    }


def format_node(state: AgentState) -> dict:
    """Wrap the answer in a structured markdown response."""
    logger.info("format_node: formatting output")
    formatted = (
        f"**Type:** {state['classification'].capitalize()}\n\n"
        f"**Answer:** {state['answer']}"
    )
    return {
        "answer": formatted,
        "messages": ["format: done"],
    }

# --- Graph construction ---

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("classify", classify_node)
    graph.add_node("answer", answer_node)
    graph.add_node("format", format_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "answer")
    graph.add_edge("answer", "format")
    graph.set_finish_point("format")

    return graph.compile()


# --- Entrypoint ---

def main() -> None:
    app = build_graph()

    initial_state: AgentState = {
        "query": "How does photosynthesis work?",
        "classification": "",
        "answer": "",
        "messages": [],
    }

    logger.info("Invoking graph")
    final_state = app.invoke(initial_state)

    print(final_state["answer"])
    print("\n--- message log ---")
    for msg in final_state["messages"]:
        print(" ", msg)


if __name__ == "__main__":
    main()
```

**Key points:**

- Each node returns only the keys it modifies — LangGraph merges them.
- `messages` uses `operator.add` as reducer, so each node's list entry is appended.
- Nodes are pure functions; no global mutable state.
- `compile()` validates the graph (checks for unreachable nodes, missing edges) before any execution.

---

## Exercises

(Solutions: ../solutions/week-10/day-4.md)

**Exercise 1 — Add a validation node**

Insert a `validate` node between `classify` and `answer`. The node should check that `state["classification"]` is one of `{"factual", "opinion", "procedural"}`. If it is not, set `classification` to `"factual"` as a fallback and log a warning. Wire the updated graph and verify the message log includes a `validate:` entry.

**Exercise 2 — Richer state with Pydantic**

Replace the `TypedDict`-based `AgentState` with a Pydantic v2 `BaseModel`. Add a `confidence: float` field (default `1.0`) to carry an estimated confidence score. Update `classify_node` to parse a confidence value from the LLM response (prompt the model to reply as `<label>:<0-1 score>`) and populate the field. Use `model_validator` or a plain `@field_validator` for any validation needed.

**Exercise 3 — Parallel fan-out (conceptual + code)**

Sketch and implement a graph where `classify` fans out to two parallel nodes — `answer_short` (one-sentence answer) and `answer_detailed` (three-sentence answer) — and a final `merge` node combines both into a single `answer` field. Note: LangGraph does not run nodes truly in parallel within a single thread by default; use `add_edge` from `classify` to both nodes and observe execution order.

**Exercise 4 — Graph introspection**

After compiling the graph, use `app.get_graph().draw_mermaid()` (or `draw_ascii()`) to print a visual representation of the graph. Extend the graph from the worked example with a no-op `log_node` at the start and re-draw. Document what changes in the diagram.

---

## Self-check

- Can you explain the difference between a reducer field and a plain field in `AgentState`?
- What does `graph.compile()` validate, and when does it raise an error?
- Why must node functions be pure (no side effects on shared mutable objects)?
- What is the `END` sentinel and why does `set_finish_point` not strictly require it?
- How does LangGraph's execution model differ from LCEL's `|` operator chain?

---

## Supplementary Links

- [LangGraph documentation — Quickstart](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
- [LangGraph concepts — State](https://langchain-ai.github.io/langgraph/concepts/low_level/#state)
- [LangGraph concepts — Nodes](https://langchain-ai.github.io/langgraph/concepts/low_level/#nodes)
- [LangGraph concepts — Edges](https://langchain-ai.github.io/langgraph/concepts/low_level/#edges)
- [Python `operator` module — `operator.add`](https://docs.python.org/3/library/operator.html#operator.add)
- [Pydantic v2 — TypedDict compatibility](https://docs.pydantic.dev/latest/concepts/postponed_annotations/)
- [LangChain Anthropic integration](https://python.langchain.com/docs/integrations/chat/anthropic/)
