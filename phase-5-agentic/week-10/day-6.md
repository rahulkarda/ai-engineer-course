# Phase 5 · Week 10 · Day 6 — LangGraph HITL Interrupts, Streaming & Framework vs DIY

> Objective: Add human-in-the-loop interrupts and streaming to LangGraph graphs; decide when LangGraph is worth the overhead.  ·  Time: ~1 hr  ·  Prereqs: [[day-5]]

---

## Concept

By the end of week 10, you know how to build agents two ways: hand-rolled Python loops and LangGraph graphs. Today covers the two features that tip the scales toward LangGraph for production work—**human-in-the-loop (HITL) interrupts** and **streaming**—then closes with a decision framework for when to reach for the library at all.

### Human-in-the-Loop (HITL)

Most agents are fully autonomous. But many real workflows require a human checkpoint: approve a database migration, review a generated email, confirm a financial trade. LangGraph handles this through **interrupt signals** embedded in the compiled graph.

Two interrupt directives exist:

| Directive | Meaning |
|---|---|
| `interrupt_before=["node_name"]` | Pause **before** the listed node runs |
| `interrupt_after=["node_name"]` | Pause **after** the listed node runs |

Both are passed to `graph.compile()`, not to individual nodes. They are declarative: you name which nodes trigger a pause, and the framework handles the rest.

### Streaming

LangGraph graphs are synchronous or async step machines. Streaming lets callers observe each step as it completes rather than waiting for the whole graph to finish.

Two streaming modes:

| Method | Granularity |
|---|---|
| `graph.stream(input, config)` | One dict per completed node: `{"node_name": state_delta}` |
| `graph.astream_events(input, config, version="v2")` | Token-level and tool-level events via the AsyncIterator protocol |

`stream()` is synchronous and sufficient for most UI progress indicators. `astream_events()` is async and needed when you want to forward LLM token deltas to a front-end in real time.

---

## How It Works Internally

### Interrupt Mechanism

1. `graph.compile(checkpointer=..., interrupt_before=["node_name"])` inserts a sentinel edge before the named node.
2. When execution reaches that sentinel, LangGraph serialises the current state into the checkpointer and raises an `Interrupt` internally.
3. The `.invoke()` or `.stream()` call returns early, yielding the state snapshot rather than the final output.
4. The caller inspects and/or modifies state, then calls `.invoke()` again **with the same `thread_id`** in the config. LangGraph resumes from the checkpoint.

State injection uses `graph.update_state(config, new_values)` before the resume call:

```python
# After interrupt, human edits the draft
graph.update_state(
    config={"configurable": {"thread_id": "thread-1"}},
    values={"feedback": "Please tone down the criticism in paragraph 2."},
)
# Resume — graph picks up where it left off
final = graph.invoke(None, config={"configurable": {"thread_id": "thread-1"}})
```

Passing `None` as input signals "resume from checkpoint, do not re-inject a new HumanMessage."

### Checkpoint Lifecycle

```
invoke(input) → node_A → node_B → [INTERRUPT sentinel] → checkpoint saved → return partial state
                                                           ^
                                        update_state() modifies checkpoint here
invoke(None)  → [resume from checkpoint] → node_C → END → return final state
```

The checkpointer (e.g., `MemorySaver`) stores a full snapshot of the typed state after every node. Resuming replays from the most recent snapshot, not from the beginning.

### stream() Internals

`graph.stream()` is a generator. After each node completes, LangGraph yields a dict:

```python
{"classify_document": {"doc_type": "pull_request", "confidence": 0.94}}
```

The key is the node name; the value is the **delta** written to state during that node's execution (not the full state). To get the full state at any point, use `graph.get_state(config)`.

### astream_events() Internals

`astream_events()` wraps `astream()` and emits structured event objects conforming to LangChain's `BaseCallbackHandler` event protocol:

- `on_chat_model_start` / `on_chat_model_stream` / `on_chat_model_end` — LLM token deltas
- `on_tool_start` / `on_tool_end` — tool invocations
- `on_chain_start` / `on_chain_end` — node boundaries

Filter by `event` field and `name` field to isolate the stream you care about.

### Framework vs DIY Decision Matrix

| Factor | Prefer LangGraph | Prefer DIY (LCEL / raw Python) |
|---|---|---|
| **Cycles / retries** | Yes — conditional edges are first-class | No — cycles in LCEL require manual while loops |
| **HITL checkpoints** | Yes — built-in interrupt + resume | Manual — must serialise/deserialise state yourself |
| **Persistence / multi-turn** | Yes — MemorySaver / SqliteSaver | Manual — need your own store |
| **Parallel branches** | Yes — `Send` API / fan-out edges | Possible with asyncio, but no built-in merge |
| **Team familiarity** | High LangGraph familiarity → prefer it | Low familiarity → DIY may ship faster |
| **Observability** | LangSmith traces nodes automatically | Must instrument manually |
| **Simple linear chains** | Overkill | LCEL pipe (`|`) is cleaner |
| **Single-shot inference** | Overkill | Direct `ChatModel.invoke()` is clearest |

Rule of thumb: if your control flow has **cycles, checkpointing, or HITL**, LangGraph pays for itself. If it is a straight pipeline with no branching, raw LCEL or a simple function is easier to read and test.

---

## Worked Example

**Scenario:** A code-review graph that drafts feedback with an LLM, interrupts for human approval, then sends the final review. Streaming shows progress node by node.

```python
"""
week10/day6_hitl_code_review.py

Dependencies (uv add):
  langchain-anthropic
  langgraph
"""

import logging
import os
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ReviewState(TypedDict):
    code_snippet: str
    messages: Annotated[list, add_messages]
    human_approved: bool
    final_review: str


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def draft_feedback(state: ReviewState) -> dict:
    """Call LLM to produce initial code-review feedback."""
    logger.info("Node: draft_feedback")
    llm = ChatAnthropic(
        model="claude-sonnet-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    prompt = (
        f"You are a senior engineer. Review this code snippet and give constructive feedback:\n\n"
        f"```\n{state['code_snippet']}\n```"
    )
    response: AIMessage = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [response], "final_review": response.content}


def send_review(state: ReviewState) -> dict:
    """Finalize and 'send' the review (log it)."""
    logger.info("Node: send_review — final review sent")
    logger.info("REVIEW:\n%s", state["final_review"])
    return {}


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    builder = StateGraph(ReviewState)
    builder.add_node("draft_feedback", draft_feedback)
    builder.add_node("send_review", send_review)

    builder.set_entry_point("draft_feedback")
    builder.add_edge("draft_feedback", "send_review")
    builder.add_edge("send_review", END)
    return builder


def main() -> None:
    checkpointer = MemorySaver()
    graph = build_graph().compile(
        checkpointer=checkpointer,
        interrupt_after=["draft_feedback"],   # pause after draft, before send
    )

    config = {"configurable": {"thread_id": "cr-001"}}
    initial_input: ReviewState = {
        "code_snippet": "def add(a, b):\n    return a+b",
        "messages": [],
        "human_approved": False,
        "final_review": "",
    }

    # --- First pass: streams until interrupt ---
    logger.info("=== Starting graph (will interrupt after draft_feedback) ===")
    for step in graph.stream(initial_input, config=config):
        node_name = list(step.keys())[0]
        logger.info("Step completed: %s", node_name)

    # --- Inspect state at interrupt point ---
    snapshot = graph.get_state(config)
    draft = snapshot.values.get("final_review", "")
    logger.info("\n--- INTERRUPT: Human review point ---")
    logger.info("Draft feedback:\n%s", draft)

    # Simulate human editing the draft
    human_edit = draft + "\n\n[Human addition: Also add a docstring.]"

    # Inject human edit into state
    graph.update_state(
        config=config,
        values={"final_review": human_edit, "human_approved": True},
    )

    # --- Resume ---
    logger.info("=== Resuming graph ===")
    for step in graph.stream(None, config=config):
        node_name = list(step.keys())[0]
        logger.info("Step completed: %s", node_name)

    logger.info("Graph complete.")


if __name__ == "__main__":
    main()
```

**Key observations:**

- `interrupt_after=["draft_feedback"]` is the only change to make a graph interruptible — no node code changes needed.
- `graph.stream(None, config=config)` resumes because `None` input + same `thread_id` = "continue from checkpoint."
- Each `step` dict in the stream loop contains exactly the state delta written by that node.
- `update_state()` merges into the checkpoint. Fields not listed are unchanged.

---

## Exercises

*(Solutions: ../solutions/week-10/day-6.md)*

**Exercise 1 — Interrupt Before**

Convert the worked example to use `interrupt_before=["send_review"]` instead of `interrupt_after=["draft_feedback"]`. Verify that the human can still modify `final_review` before `send_review` runs. What is the practical difference in UX between the two interrupt points for this workflow?

**Exercise 2 — Token Streaming**

Rewrite `draft_feedback` to use an async LLM call and expose token deltas via `astream_events()`. Print each token chunk to stdout as it arrives. Use `event["event"] == "on_chat_model_stream"` to filter. The rest of the graph (interrupt, resume) should still work.

**Exercise 3 — Multi-Point Interrupts**

Add a third node `escalate_to_lead` that fires only when the human sets a flag `escalate: bool = True` in state. Add a conditional edge after `draft_feedback`: if `escalate`, route to `escalate_to_lead`, then `send_review`; otherwise go directly to `send_review`. Place an interrupt **before** `escalate_to_lead`. Write a test that verifies the graph resumes correctly when `escalate=True`.

**Exercise 4 — Framework vs DIY Analysis**

You are reviewing a PR for a colleague. They implemented a document-summarisation pipeline using LangGraph. The pipeline is: load document → chunk → summarise each chunk → combine summaries. There are no cycles, no HITL, no shared state between chunks. Write a one-page technical note (as a Python docstring or markdown comment block) arguing whether LangGraph was the right tool. Reference at least three factors from the decision matrix. Propose the simpler alternative if applicable.

---

## Self-check

- Can you explain why `graph.invoke(None, config=config)` resumes rather than re-running the graph?
- What does `update_state()` write to? (answer: the checkpointer snapshot for the given thread)
- When would you choose `astream_events()` over `stream()`?
- Name two real scenarios where raw LCEL is a better choice than LangGraph.
- What happens if you call `compile()` with `interrupt_after` but no `checkpointer`?

---

## Supplementary Links

- [LangGraph HITL conceptual guide](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)
- [LangGraph streaming guide](https://langchain-ai.github.io/langgraph/concepts/streaming/)
- [`astream_events` reference](https://python.langchain.com/docs/concepts/streaming/#astream_events)
- [LangGraph vs LCEL discussion](https://langchain-ai.github.io/langgraph/concepts/faq/#how-is-langgraph-different-from-langchain)
- [MemorySaver API](https://langchain-ai.github.io/langgraph/reference/checkpoints/#memorysaver)
