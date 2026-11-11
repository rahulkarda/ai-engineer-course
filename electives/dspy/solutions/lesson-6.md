# Solutions — Lesson 6: DSPy vs Manual Prompting · Agent Integration

---

## Exercise 1 — Swap backbone explanation

Change:

```python
lm = dspy.LM(
    model="anthropic/claude-sonnet-5-5",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=128,
)
```

**Would the compiled JSON still work if loaded against `haiku`?**

Yes. A compiled program's JSON stores only the prompt instructions and few-shot
demonstration texts — it contains no model-specific parameters, weights, or embeddings.
Loading it against a different model simply injects those same demos and instructions
into the new model's prompt. Performance may differ (a prompt optimized for Sonnet
may not be equally effective for Haiku), but the file loads without error and the
program runs. For best results, compile against the model you intend to run in
production.

---

## Exercise 2 — Async classify node

DSPy `dspy.Module` subclasses expose `aforward()` (async forward) if the underlying
LM client supports async. For Anthropic through DSPy, async is available via
`dspy.LM(..., async_mode=True)`.

```python
import asyncio
import dspy

async def classify_node_async(
    state: TicketState,
    compiled_router: dspy.Module,
) -> TicketState:
    # DSPy async forward
    result = await compiled_router.aforward(ticket=state["ticket"])
    return {**state, "category": result.category, "priority": result.priority}
```

In LangGraph, wrap in `asyncio.run()` at the graph level, or use an async graph:

```python
from langgraph.graph import StateGraph

async_builder = StateGraph(TicketState)
# add_node accepts async callables directly
async_builder.add_node("classify", lambda s: classify_node_async(s, compiled_router))
```

---

## Exercise 3 — Pydantic state in LangGraph

Replace `TypedDict`:

```python
from pydantic import BaseModel

class TicketState(BaseModel):
    ticket: str
    category: str = ""
    priority: str = ""
    response: str = ""
```

Changes:
- LangGraph validates state transitions against the Pydantic model — invalid field
  types raise `ValidationError` at graph edges, not silently.
- Default values in Pydantic make partial state updates cleaner.
- Node functions return a dict; LangGraph merges it into the model via
  `TicketState(**{**current.model_dump(), **update})`.

---

## Exercise 4 — Subgraph

```python
from langgraph.graph import END, START, StateGraph


def build_routing_subgraph(compiled_router: dspy.Module) -> StateGraph:
    """Subgraph: classify → route → specialist node."""
    classify_fn = make_classify_node(compiled_router)

    builder = StateGraph(TicketState)
    builder.add_node("classify", classify_fn)
    builder.add_node("billing", billing_node)
    builder.add_node("technical", technical_node)
    builder.add_node("account", account_node)
    builder.add_node("other", other_node)
    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route_by_category)
    for node in ["billing", "technical", "account", "other"]:
        builder.add_edge(node, END)
    return builder.compile()


def intake_node(state: TicketState) -> TicketState:
    """Normalize raw ticket text (strip, lowercase subject line, etc.)."""
    return {**state, "ticket": state["ticket"].strip()}


def build_parent_graph(compiled_router: dspy.Module) -> StateGraph:
    routing_subgraph = build_routing_subgraph(compiled_router)

    builder = StateGraph(TicketState)
    builder.add_node("intake", intake_node)
    builder.add_node("routing", routing_subgraph)
    builder.add_edge(START, "intake")
    builder.add_edge("intake", "routing")
    builder.add_edge("routing", END)
    return builder.compile()
```

Test:

```python
def test_parent_graph_runs() -> None:
    mock_router = MagicMock()
    mock_router.return_value = MagicMock(category="billing", priority="high")

    graph = build_parent_graph(mock_router)
    state: TicketState = {"ticket": "  I was charged twice.  ", "category": "", "priority": "", "response": ""}
    result = graph.invoke(state)
    assert result["category"] == "billing"
    assert "Billing" in result["response"]
```

---

## Exercise 5 — Latency benchmark

```python
import time
import logging

logger = logging.getLogger(__name__)


def benchmark_graph(graph, initial_state: TicketState, n: int = 5) -> float:
    times = []
    for _ in range(n):
        start = time.perf_counter()
        graph.invoke(initial_state)
        times.append(time.perf_counter() - start)
    mean_ms = (sum(times) / len(times)) * 1000
    logger.info("Mean latency: %.1f ms over %d runs", mean_ms, n)
    return mean_ms
```

**Expected finding**: The compiled (3-demo) variant is typically *slower* per call
than the uncompiled variant because each LM call now includes 3 complete
demonstration examples in the prompt, increasing input tokens. However, the
compiled variant may require fewer retries (fewer `dspy.Assert` failures) and fewer
reasoning steps, so end-to-end task completion time can be lower on complex tickets.

The latency trade-off: more tokens per call ↔ fewer calls needed for correctness.
For simple classification tasks, uncompiled + CoT is often fast enough.
