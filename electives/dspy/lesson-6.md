# Lesson 6 — DSPy vs Manual Prompting · Agent Integration

> Objective: Decide when DSPy is worth the overhead; integrate a compiled DSPy module into a LangGraph agent.
> Time: ~60 min · Prereqs: lesson-5.md, Phase 5 (LangGraph basics)

---

## Concept

### When DSPy helps

DSPy earns its cost when at least two of these are true:

| Condition | Why DSPy wins |
|---|---|
| You have labeled examples (even 20–50) | Optimizer has signal to work with |
| The task has a measurable metric | Without a metric there is nothing to optimize |
| You will change models | Compiled programs transfer; hand-written prompts don't |
| You have multiple reasoning steps | Each hop can be independently optimized |
| Prompt brittleness is burning dev time | Let the optimizer find robust phrasing |

### When manual prompting is better

- One-off scripts or demos (setup overhead not justified)
- No labeled data and no cheap metric (optimizer can't fire)
- Ultra-low latency — compiled programs can add LM calls (assertions, CoT)
- Creative tasks where "right answer" is subjective

### Integrating DSPy with LangGraph

A compiled DSPy module is just a Python callable — `module(field=value)`. LangGraph
nodes are also Python callables. The integration is straightforward:

```
LangGraph node → calls compiled DSPy module → returns structured result
```

The key is to compile the DSPy module once at startup (outside the graph), then
reference the compiled instance inside the node function.

---

## How it works internally

### Compilation at startup

```python
# startup.py — run once, cache the compiled module
compiled_classifier = SentimentProgram()
compiled_classifier.load("compiled_sentiment.json")
```

Keep the compiled module in a module-level variable or dependency-inject it into the
graph state. Do not re-compile on every request — compilation is expensive.

### Typed state in LangGraph

```python
from typing import TypedDict

class AgentState(TypedDict):
    ticket: str
    category: str
    priority: str
    response: str
```

### Node wrapping a DSPy module

```python
def classify_node(state: AgentState) -> AgentState:
    result = compiled_router(ticket=state["ticket"])
    return {**state, "category": result.category, "priority": result.priority}
```

### Full wiring example

```
START → classify_node → route_node → [billing_node | technical_node | ...] → END
```

`route_node` reads `state["category"]` and returns the next node name. The DSPy
module does the cognitive work; LangGraph does the control flow.

---

## Worked example

```python
"""lesson6_agent_integration.py — DSPy compiled module inside a LangGraph agent."""
import logging
import os
from typing import Literal, TypedDict

import dspy
from dspy.teleprompt import BootstrapFewShot
from langgraph.graph import END, START, StateGraph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DSPy module (ticket classifier — same as lesson 5)
# ---------------------------------------------------------------------------

class TicketRoute(dspy.Signature):
    """Route a support ticket to the correct team."""
    ticket: str = dspy.InputField()
    category: Literal["billing", "technical", "account", "other"] = dspy.OutputField()
    priority: Literal["low", "medium", "high"] = dspy.OutputField()


class TicketRouter(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.route = dspy.ChainOfThought(TicketRoute)

    def forward(self, ticket: str) -> dspy.Prediction:
        return self.route(ticket=ticket)


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class TicketState(TypedDict):
    ticket: str
    category: str
    priority: str
    response: str


# ---------------------------------------------------------------------------
# Build and compile DSPy module
# ---------------------------------------------------------------------------

def build_compiled_router(lm: dspy.LM) -> dspy.Module:
    """Compile the ticket router on a small synthetic trainset."""
    dspy.configure(lm=lm)
    trainset = [
        dspy.Example(ticket="I was charged twice this month.", category="billing", priority="high").with_inputs("ticket"),
        dspy.Example(ticket="The API returns 500 on POST /orders.", category="technical", priority="high").with_inputs("ticket"),
        dspy.Example(ticket="I can't log in.", category="account", priority="medium").with_inputs("ticket"),
        dspy.Example(ticket="Can I get a refund?", category="billing", priority="medium").with_inputs("ticket"),
        dspy.Example(ticket="Dashboard is slow.", category="technical", priority="low").with_inputs("ticket"),
    ]

    def exact_both(ex: dspy.Example, pred: dspy.Prediction, trace=None) -> bool:
        return (
            pred.category.strip().lower() == ex.category.strip().lower()
            and pred.priority.strip().lower() == ex.priority.strip().lower()
        )

    optimizer = BootstrapFewShot(metric=exact_both, max_bootstrapped_demos=2)
    return optimizer.compile(TicketRouter(), trainset=trainset)


# ---------------------------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------------------------

def make_classify_node(compiled_router: dspy.Module):
    """Closure: captures the compiled module."""
    def classify_node(state: TicketState) -> TicketState:
        result = compiled_router(ticket=state["ticket"])
        logger.info("Classified as %s / %s", result.category, result.priority)
        return {**state, "category": result.category, "priority": result.priority}
    return classify_node


def billing_node(state: TicketState) -> TicketState:
    response = f"[Billing] Ticket received with priority {state['priority']}. Escalating to billing team."
    return {**state, "response": response}


def technical_node(state: TicketState) -> TicketState:
    response = f"[Technical] Ticket received with priority {state['priority']}. Assigning to on-call engineer."
    return {**state, "response": response}


def account_node(state: TicketState) -> TicketState:
    response = f"[Account] Ticket received with priority {state['priority']}. Account recovery initiated."
    return {**state, "response": response}


def other_node(state: TicketState) -> TicketState:
    response = f"[General] Ticket received. A team member will follow up shortly."
    return {**state, "response": response}


def route_by_category(state: TicketState) -> str:
    return {"billing": "billing", "technical": "technical",
            "account": "account", "other": "other"}.get(state["category"], "other")


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_graph(compiled_router: dspy.Module) -> StateGraph:
    classify_node = make_classify_node(compiled_router)

    builder = StateGraph(TicketState)
    builder.add_node("classify", classify_node)
    builder.add_node("billing", billing_node)
    builder.add_node("technical", technical_node)
    builder.add_node("account", account_node)
    builder.add_node("other", other_node)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route_by_category)
    for node in ["billing", "technical", "account", "other"]:
        builder.add_edge(node, END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    lm = dspy.LM(
        model="anthropic/claude-haiku-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=128,
    )
    compiled_router = build_compiled_router(lm)
    graph = build_graph(compiled_router)

    test_tickets = [
        "My credit card was charged but my subscription wasn't activated.",
        "Getting ImportError on your Python SDK version 2.1.",
        "I forgot my password and the reset email never arrived.",
    ]
    for ticket in test_tickets:
        initial_state: TicketState = {
            "ticket": ticket,
            "category": "",
            "priority": "",
            "response": "",
        }
        final = graph.invoke(initial_state)
        logger.info("Ticket: %s...", ticket[:50])
        logger.info("Response: %s\n", final["response"])
```

### Testing the integration without LM calls

```python
"""test_lesson6.py — integration tests with mocked DSPy."""
from unittest.mock import MagicMock, patch

import dspy
import pytest

from lesson6_agent_integration import (
    TicketState,
    billing_node,
    make_classify_node,
    route_by_category,
)


def test_classify_node_calls_router() -> None:
    mock_router = MagicMock()
    mock_router.return_value = MagicMock(category="billing", priority="high")

    node = make_classify_node(mock_router)
    state: TicketState = {"ticket": "charged twice", "category": "", "priority": "", "response": ""}
    result = node(state)

    mock_router.assert_called_once_with(ticket="charged twice")
    assert result["category"] == "billing"
    assert result["priority"] == "high"


def test_route_by_category() -> None:
    assert route_by_category({"ticket": "", "category": "billing", "priority": "high", "response": ""}) == "billing"
    assert route_by_category({"ticket": "", "category": "technical", "priority": "low", "response": ""}) == "technical"
    assert route_by_category({"ticket": "", "category": "unknown", "priority": "low", "response": ""}) == "other"


def test_billing_node() -> None:
    state: TicketState = {"ticket": "t", "category": "billing", "priority": "high", "response": ""}
    result = billing_node(state)
    assert "Billing" in result["response"]
    assert "high" in result["response"]
```

---

## Exercises

1. **Swap the backbone** — Change `build_compiled_router` to use
   `anthropic/claude-sonnet-5-5` instead of `claude-haiku-5-5`. The compiled program
   (JSON) is model-agnostic — would the saved `.json` still work if loaded against
   `haiku`? Write a one-paragraph explanation.

2. **Streaming node** — LangGraph supports async graphs. Rewrite `classify_node` as an
   `async def` that calls `await compiled_router.aforward(ticket=...)`. What does DSPy
   require for async forward calls?

3. **State schema validation** — Replace the `TypedDict` for `TicketState` with a
   Pydantic `BaseModel`. Update the graph. What changes in how LangGraph handles the
   state?

4. **Subgraph** — Move all ticket-routing nodes into a subgraph builder function
   `build_routing_subgraph()`. Wire it into a parent graph that first runs an
   `intake_node` (populates `ticket` from raw user input). Test the parent graph with
   the subgraph mock.

5. **Benchmark** — Compare the latency of:
   (a) `TicketRouter` uncompiled (0 demos)
   (b) `TicketRouter` compiled with 3 demos
   Use `time.perf_counter()` around the `graph.invoke` call. Log mean over 5 runs.
   Explain the latency difference.

(Solutions: [solutions/lesson-6.md](solutions/lesson-6.md))

---

## Self-check

- [ ] I can list three conditions where DSPy provides clear value over manual prompting
- [ ] I can list two scenarios where manual prompting is more practical
- [ ] I can wrap a compiled DSPy module in a LangGraph node using a closure
- [ ] I know to compile once at startup, not per request
- [ ] I can write integration tests for a DSPy-backed graph node without LM calls
