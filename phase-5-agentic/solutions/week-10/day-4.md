# Solutions — Week 10 · Day 4 — LangGraph Basics: State, Nodes, Edges

> These are reference solutions. Your implementation may differ and still be correct.

## Exercise 1

**Add a validation node between `classify` and `answer` that checks if the classification is one of `["question", "task", "chitchat"]` and raises `ValueError` otherwise; wire it into the graph.**

Design: insert a `validate_node` between `classify` and `answer`. The node checks `state["classification"]` against the allowed set. If valid, it passes through unchanged; if invalid, it raises `ValueError` with a descriptive message. The lesson example used `factual/opinion/procedural`; the exercise specifies `question/task/chitchat` — the prompt to `classify_node` is updated accordingly, and `validate_node` enforces the contract.

```python
"""
day4_ex1.py — LangGraph with a validation node between classify and answer.

Run: python day4_ex1.py
Env: ANTHROPIC_API_KEY
"""
import logging
import operator
import os
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

VALID_CLASSIFICATIONS: frozenset[str] = frozenset({"question", "task", "chitchat"})


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    query: str
    classification: str
    answer: str
    messages: Annotated[list[str], operator.add]


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

llm = ChatAnthropic(
    model="claude-haiku-5-5",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def classify_node(state: AgentState) -> dict:
    """Classify the query as question, task, or chitchat."""
    logger.info("classify_node: classifying query")
    prompt = (
        f"Classify this message with EXACTLY ONE word from: question, task, chitchat.\n"
        f"Message: {state['query']}\n"
        "Reply with only the single classification word."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    classification = response.content.strip().lower()
    logger.info("classify_node: raw classification=%r", classification)
    return {
        "classification": classification,
        "messages": [f"classify: {classification}"],
    }


def validate_node(state: AgentState) -> dict:
    """
    Validate that classification is one of the allowed values.

    Raises ValueError if the classification is unexpected.
    LangGraph propagates the exception to the caller — the graph run aborts
    and the exception is visible in the traceback.
    """
    classification = state["classification"]
    logger.info("validate_node: checking classification=%r", classification)

    if classification not in VALID_CLASSIFICATIONS:
        raise ValueError(
            f"Invalid classification {classification!r}. "
            f"Expected one of {sorted(VALID_CLASSIFICATIONS)}."
        )

    logger.info("validate_node: classification is valid")
    return {"messages": [f"validate: ok ({classification})"]}


def answer_node(state: AgentState) -> dict:
    """Generate an answer given the validated classification."""
    logger.info("answer_node: generating answer (type=%s)", state["classification"])
    prompt = (
        f"Query type: {state['classification']}\n"
        f"Query: {state['query']}\n"
        "Provide a concise, helpful response."
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


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("classify", classify_node)
    graph.add_node("validate", validate_node)   # NEW — inserted between classify and answer
    graph.add_node("answer", answer_node)
    graph.add_node("format", format_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "validate")      # classify → validate (not directly to answer)
    graph.add_edge("validate", "answer")        # validate → answer
    graph.add_edge("answer", "format")
    graph.set_finish_point("format")

    return graph.compile()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = build_graph()

    for query in [
        "What is the capital of France?",
        "Please summarise this document for me.",
        "Hey, how's your day going?",
    ]:
        initial_state: AgentState = {
            "query": query,
            "classification": "",
            "answer": "",
            "messages": [],
        }
        final_state = app.invoke(initial_state)
        print(f"\nQuery: {query}")
        print(final_state["answer"])
        print("Message log:", final_state["messages"])


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

import unittest.mock
from unittest.mock import MagicMock, patch


def _make_llm_response(content: str) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.content = content
    return mock_resp


def test_validate_node_passes_for_valid_classification() -> None:
    state: AgentState = {
        "query": "test",
        "classification": "question",
        "answer": "",
        "messages": [],
    }
    result = validate_node(state)
    assert "validate: ok (question)" in result["messages"]


def test_validate_node_passes_for_all_valid_values() -> None:
    for valid in ["question", "task", "chitchat"]:
        state: AgentState = {
            "query": "test",
            "classification": valid,
            "answer": "",
            "messages": [],
        }
        result = validate_node(state)
        assert result["messages"][0].startswith("validate: ok")


def test_validate_node_raises_for_invalid_classification() -> None:
    import pytest
    state: AgentState = {
        "query": "test",
        "classification": "unknown_label",
        "answer": "",
        "messages": [],
    }
    try:
        validate_node(state)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "unknown_label" in str(exc)
        assert "Expected one of" in str(exc)


def test_validate_node_raises_for_empty_string() -> None:
    state: AgentState = {
        "query": "test",
        "classification": "",
        "answer": "",
        "messages": [],
    }
    try:
        validate_node(state)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_full_graph_valid_path() -> None:
    """Graph runs end-to-end when classify returns a valid label."""
    app = build_graph()

    with patch("__main__.llm") as mock_llm:
        mock_llm.invoke.side_effect = [
            _make_llm_response("question"),         # classify_node
            _make_llm_response("Paris is the capital of France."),  # answer_node
        ]
        initial: AgentState = {
            "query": "What is the capital of France?",
            "classification": "",
            "answer": "",
            "messages": [],
        }
        final = app.invoke(initial)

    assert final["classification"] == "question"
    assert "Paris" in final["answer"]
    assert any("validate: ok" in m for m in final["messages"])


def test_graph_edges_include_validate_node() -> None:
    """Introspect compiled graph to confirm validate is wired between classify and answer."""
    app = build_graph()
    graph_data = app.get_graph()

    node_names = set(graph_data.nodes.keys())
    assert "validate" in node_names, "validate node must be present"

    edges = [(e.source, e.target) for e in graph_data.edges]
    assert ("classify", "validate") in edges, "classify must edge to validate"
    assert ("validate", "answer") in edges, "validate must edge to answer"
```

**What changed and why:**
- `validate_node` inserted between `classify` and `answer`; it has no LLM call — pure state inspection.
- `add_edge("classify", "validate")` and `add_edge("validate", "answer")` replace the single direct edge.
- Raising `ValueError` inside a node propagates to the caller of `app.invoke()` — no special LangGraph error handling needed; the graph simply aborts.
- The node returns `{"messages": [...]}` even on the happy path so the message log shows the validate step ran.


## Exercise 2

**Convert the `TypedDict` state to use a Pydantic v2 `BaseModel` as the state schema (show how LangGraph handles Pydantic v2 state).**

Design: LangGraph accepts a Pydantic v2 `BaseModel` subclass as the state type directly. Fields with `Annotated[list[X], operator.add]` reducers work identically. The key difference from `TypedDict` is that LangGraph wraps the model class for internal merging — node functions still receive a dict-like object and still return plain `dict`s. A `confidence: float` field (defaulting to `1.0`) is added as specified; `classify_node` prompts the LLM for `label:score` format and parses it.

```python
"""
day4_ex2.py — LangGraph with Pydantic v2 BaseModel as state schema.

Run: python day4_ex2.py
Env: ANTHROPIC_API_KEY
"""
import logging
import operator
import os
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State — Pydantic v2 BaseModel
# ---------------------------------------------------------------------------

class AgentState(BaseModel):
    """
    LangGraph state as a Pydantic v2 model.

    LangGraph calls model_validate() on each merged state dict, so validators
    run after every node update. Fields with operator.add reducers are still
    appended (LangGraph handles reducer semantics before passing to Pydantic).
    """

    query: str = ""
    classification: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    answer: str = ""
    messages: Annotated[list[str], operator.add] = Field(default_factory=list)

    @field_validator("classification")
    @classmethod
    def classification_must_be_lowercase(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        """Clamp any out-of-range float that the LLM might return."""
        return max(0.0, min(1.0, v))


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

llm = ChatAnthropic(
    model="claude-haiku-5-5",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def classify_node(state: AgentState) -> dict:
    """
    Classify the query and return a confidence score.

    Prompt asks for <label>:<score> format, e.g. "question:0.92".
    Falls back to ("question", 0.5) on parse error.
    """
    logger.info("classify_node: classifying query")
    prompt = (
        "Classify this message with EXACTLY ONE word from: question, task, chitchat.\n"
        "Then rate your confidence as a decimal 0.0–1.0.\n"
        f"Message: {state.query}\n"
        "Reply in the format: <label>:<confidence>  — for example: question:0.92"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip().lower()
    logger.info("classify_node: raw response=%r", raw)

    label, confidence = "question", 0.5  # safe defaults
    if ":" in raw:
        parts = raw.split(":", 1)
        label = parts[0].strip()
        try:
            confidence = float(parts[1].strip())
        except ValueError:
            logger.warning("classify_node: could not parse confidence from %r", raw)
    else:
        label = raw  # no score provided — use default confidence

    logger.info("classify_node: label=%s confidence=%.2f", label, confidence)
    return {
        "classification": label,
        "confidence": confidence,
        "messages": [f"classify: {label} (confidence={confidence:.2f})"],
    }


def answer_node(state: AgentState) -> dict:
    """Generate an answer, noting the classification and confidence."""
    logger.info(
        "answer_node: type=%s confidence=%.2f",
        state.classification,
        state.confidence,
    )
    prompt = (
        f"Query type: {state.classification}\n"
        f"Query: {state.query}\n"
        "Provide a concise, helpful response."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    answer = response.content.strip()
    return {
        "answer": answer,
        "messages": [f"answer: {len(answer)} chars"],
    }


def format_node(state: AgentState) -> dict:
    """Format the final response with metadata."""
    formatted = (
        f"**Type:** {state.classification.capitalize()}  "
        f"(confidence: {state.confidence:.0%})\n\n"
        f"**Answer:** {state.answer}"
    )
    return {
        "answer": formatted,
        "messages": ["format: done"],
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    # Pass the Pydantic BaseModel class — not an instance — to StateGraph
    graph = StateGraph(AgentState)

    graph.add_node("classify", classify_node)
    graph.add_node("answer", answer_node)
    graph.add_node("format", format_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "answer")
    graph.add_edge("answer", "format")
    graph.set_finish_point("format")

    return graph.compile()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = build_graph()

    initial = AgentState(query="How do I reverse a Python list?")
    final_state = app.invoke(initial)

    # LangGraph returns a dict when invoked; reconstruct for type safety
    if isinstance(final_state, dict):
        final = AgentState(**final_state)
    else:
        final = final_state

    print(final.answer)
    print(f"\nConfidence: {final.confidence:.0%}")
    print("Message log:", final.messages)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

import unittest.mock
from unittest.mock import MagicMock, patch


def _make_llm_response(content: str) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.content = content
    return mock_resp


def test_pydantic_state_field_defaults() -> None:
    state = AgentState(query="hello")
    assert state.confidence == 1.0
    assert state.classification == ""
    assert state.messages == []


def test_pydantic_field_validator_lowercases_classification() -> None:
    state = AgentState(query="x", classification="  Question  ")
    assert state.classification == "question"


def test_pydantic_field_validator_clamps_confidence() -> None:
    state = AgentState(query="x", confidence=1.5)
    assert state.confidence == 1.0

    state2 = AgentState(query="x", confidence=-0.3)
    assert state2.confidence == 0.0


def test_classify_node_parses_label_and_confidence() -> None:
    state = AgentState(query="What is 2+2?")
    with patch("__main__.llm") as mock_llm:
        mock_llm.invoke.return_value = _make_llm_response("question:0.95")
        result = classify_node(state)

    assert result["classification"] == "question"
    assert abs(result["confidence"] - 0.95) < 1e-6


def test_classify_node_falls_back_on_missing_score() -> None:
    state = AgentState(query="What is 2+2?")
    with patch("__main__.llm") as mock_llm:
        mock_llm.invoke.return_value = _make_llm_response("task")
        result = classify_node(state)

    assert result["classification"] == "task"
    assert result["confidence"] == 0.5  # default fallback


def test_classify_node_falls_back_on_bad_score() -> None:
    state = AgentState(query="test")
    with patch("__main__.llm") as mock_llm:
        mock_llm.invoke.return_value = _make_llm_response("question:not_a_number")
        result = classify_node(state)

    assert result["confidence"] == 0.5


def test_messages_reducer_accumulates() -> None:
    """Pydantic state with operator.add reducer appends across nodes."""
    app = build_graph()

    with patch("__main__.llm") as mock_llm:
        mock_llm.invoke.side_effect = [
            _make_llm_response("chitchat:0.8"),
            _make_llm_response("Hi there!"),
        ]
        final = app.invoke(AgentState(query="Hey!"))

    if isinstance(final, dict):
        msgs = final["messages"]
    else:
        msgs = final.messages

    # All three nodes should have appended to messages
    assert any("classify" in m for m in msgs)
    assert any("answer" in m for m in msgs)
    assert any("format" in m for m in msgs)
```

**What changed and why:**
- `AgentState` inherits from `pydantic.BaseModel` instead of `TypedDict`. LangGraph's `StateGraph` accepts either.
- Node functions still receive the state as an object with attribute access (`state.query`, `state.confidence`). Return dicts are merged the same way.
- `@field_validator` runs after every merge — useful for normalization (lowercase) and clamping (confidence range).
- `Annotated[list[str], operator.add]` reducer syntax is identical to the `TypedDict` version — LangGraph handles it the same way regardless of state schema type.
- The initial state passed to `app.invoke()` can be the model instance directly or a plain dict — both work.


## Exercise 3

**Add a parallel fan-out: after `classify`, run both an `expand_context` node and a `check_safety` node in parallel before feeding results to `answer`.**

Design: LangGraph executes nodes with multiple incoming edges in parallel when you add edges from a single source to multiple targets. The trick is the `answer` node needs both `expand_context` and `check_safety` to have written their fields before it runs — add edges from both to a `merge` node (or directly to `answer` if the state fields are independent). Since LangGraph's default execution is single-threaded, parallel "fan-out" means both nodes run in sequence but the graph structure makes the intent explicit; with `stream_mode="values"` or an async executor they can run concurrently.

```python
"""
day4_ex3.py — LangGraph parallel fan-out: classify → {expand_context, check_safety} → answer.

Note: LangGraph executes both branches before advancing to the merge node.
With a sync executor the branches run sequentially in topological order;
with an async executor (ainvoke) they run concurrently.

Run: python day4_ex3.py
Env: ANTHROPIC_API_KEY
"""
import logging
import operator
import os
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State — carries fields for both parallel branches
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    query: str
    classification: str
    context: str          # populated by expand_context_node
    safety_flag: str      # populated by check_safety_node ("safe" | "unsafe")
    answer: str
    messages: Annotated[list[str], operator.add]


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

llm = ChatAnthropic(
    model="claude-haiku-5-5",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def classify_node(state: AgentState) -> dict:
    """Classify the query as question, task, or chitchat."""
    logger.info("classify_node: classifying query")
    prompt = (
        "Classify this message with EXACTLY ONE word from: question, task, chitchat.\n"
        f"Message: {state['query']}\n"
        "Reply with only the single classification word."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    classification = response.content.strip().lower()
    logger.info("classify_node: classification=%s", classification)
    return {
        "classification": classification,
        "messages": [f"classify: {classification}"],
    }


def expand_context_node(state: AgentState) -> dict:
    """
    Branch A: expand the query with background context.

    Runs in parallel with check_safety_node after classify.
    """
    logger.info("expand_context_node: generating context for query type=%s", state["classification"])
    prompt = (
        f"Query: {state['query']}\n"
        "Provide one sentence of relevant background context that would help answer this query."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    context = response.content.strip()
    logger.info("expand_context_node: context generated (%d chars)", len(context))
    return {
        "context": context,
        "messages": [f"expand_context: {len(context)} chars"],
    }


def check_safety_node(state: AgentState) -> dict:
    """
    Branch B: check if the query is safe to answer.

    Runs in parallel with expand_context_node after classify.
    """
    logger.info("check_safety_node: evaluating safety")
    prompt = (
        f"Is this query safe and appropriate to answer? Query: {state['query']}\n"
        "Reply with exactly one word: safe or unsafe."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    safety_flag = response.content.strip().lower()
    if safety_flag not in {"safe", "unsafe"}:
        safety_flag = "safe"  # default to safe on ambiguous response
    logger.info("check_safety_node: safety_flag=%s", safety_flag)
    return {
        "safety_flag": safety_flag,
        "messages": [f"check_safety: {safety_flag}"],
    }


def answer_node(state: AgentState) -> dict:
    """
    Merge node: uses context + safety_flag produced by the parallel branches.

    Only runs after both expand_context_node and check_safety_node complete.
    """
    logger.info(
        "answer_node: generating answer (safety=%s, has_context=%s)",
        state["safety_flag"],
        bool(state["context"]),
    )

    if state["safety_flag"] == "unsafe":
        answer = "I'm sorry, I can't help with that query."
        logger.warning("answer_node: query flagged as unsafe — returning refusal")
    else:
        prompt = (
            f"Query type: {state['classification']}\n"
            f"Background context: {state['context']}\n"
            f"Query: {state['query']}\n"
            "Provide a concise, helpful response using the context."
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = response.content.strip()

    logger.info("answer_node: answer generated (%d chars)", len(answer))
    return {
        "answer": answer,
        "messages": [f"answer: {len(answer)} chars"],
    }


# ---------------------------------------------------------------------------
# Graph construction — fan-out then fan-in
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("classify", classify_node)
    graph.add_node("expand_context", expand_context_node)   # parallel branch A
    graph.add_node("check_safety", check_safety_node)       # parallel branch B
    graph.add_node("answer", answer_node)                   # merge node

    graph.set_entry_point("classify")

    # Fan-out: classify → both branches simultaneously
    graph.add_edge("classify", "expand_context")
    graph.add_edge("classify", "check_safety")

    # Fan-in: both branches → answer (LangGraph waits for both before running answer)
    graph.add_edge("expand_context", "answer")
    graph.add_edge("check_safety", "answer")

    graph.set_finish_point("answer")

    return graph.compile()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = build_graph()

    initial_state: AgentState = {
        "query": "How does photosynthesis work?",
        "classification": "",
        "context": "",
        "safety_flag": "",
        "answer": "",
        "messages": [],
    }

    final_state = app.invoke(initial_state)

    print("=== Answer ===")
    print(final_state["answer"])
    print(f"\nSafety: {final_state['safety_flag']}  |  Context snippet: {final_state['context'][:80]}...")
    print("\n--- message log ---")
    for msg in final_state["messages"]:
        print(" ", msg)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

import unittest.mock
from unittest.mock import MagicMock, patch


def _make_llm_response(content: str) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.content = content
    return mock_resp


def test_graph_contains_parallel_nodes() -> None:
    """Both parallel branch nodes must be present in the compiled graph."""
    app = build_graph()
    node_names = set(app.get_graph().nodes.keys())
    assert "expand_context" in node_names
    assert "check_safety" in node_names
    assert "answer" in node_names


def test_graph_fan_out_edges() -> None:
    """classify must have edges to both expand_context and check_safety."""
    app = build_graph()
    edges = [(e.source, e.target) for e in app.get_graph().edges]
    assert ("classify", "expand_context") in edges
    assert ("classify", "check_safety") in edges


def test_graph_fan_in_edges() -> None:
    """Both branches must converge into answer."""
    app = build_graph()
    edges = [(e.source, e.target) for e in app.get_graph().edges]
    assert ("expand_context", "answer") in edges
    assert ("check_safety", "answer") in edges


def test_expand_context_node_writes_context() -> None:
    state: AgentState = {
        "query": "What is DNA?",
        "classification": "question",
        "context": "",
        "safety_flag": "",
        "answer": "",
        "messages": [],
    }
    with patch("__main__.llm") as mock_llm:
        mock_llm.invoke.return_value = _make_llm_response("DNA carries genetic information.")
        result = expand_context_node(state)

    assert result["context"] == "DNA carries genetic information."
    assert any("expand_context" in m for m in result["messages"])


def test_check_safety_node_safe() -> None:
    state: AgentState = {
        "query": "What is DNA?",
        "classification": "question",
        "context": "",
        "safety_flag": "",
        "answer": "",
        "messages": [],
    }
    with patch("__main__.llm") as mock_llm:
        mock_llm.invoke.return_value = _make_llm_response("safe")
        result = check_safety_node(state)

    assert result["safety_flag"] == "safe"


def test_check_safety_node_defaults_to_safe_on_ambiguous() -> None:
    state: AgentState = {
        "query": "What is 2+2?",
        "classification": "question",
        "context": "",
        "safety_flag": "",
        "answer": "",
        "messages": [],
    }
    with patch("__main__.llm") as mock_llm:
        mock_llm.invoke.return_value = _make_llm_response("maybe it depends")
        result = check_safety_node(state)

    assert result["safety_flag"] == "safe"


def test_answer_node_refuses_unsafe_query() -> None:
    state: AgentState = {
        "query": "harmful query",
        "classification": "task",
        "context": "some context",
        "safety_flag": "unsafe",
        "answer": "",
        "messages": [],
    }
    result = answer_node(state)
    assert "can't help" in result["answer"].lower()


def test_full_graph_safe_path() -> None:
    app = build_graph()
    with patch("__main__.llm") as mock_llm:
        mock_llm.invoke.side_effect = [
            _make_llm_response("question"),                  # classify
            _make_llm_response("Relevant background here."),  # expand_context
            _make_llm_response("safe"),                      # check_safety
            _make_llm_response("Photosynthesis is the process..."),  # answer
        ]
        initial: AgentState = {
            "query": "How does photosynthesis work?",
            "classification": "",
            "context": "",
            "safety_flag": "",
            "answer": "",
            "messages": [],
        }
        final = app.invoke(initial)

    assert final["safety_flag"] == "safe"
    assert final["context"] == "Relevant background here."
    assert "Photosynthesis" in final["answer"]
    # All four nodes must appear in message log
    assert any("classify" in m for m in final["messages"])
    assert any("expand_context" in m for m in final["messages"])
    assert any("check_safety" in m for m in final["messages"])
    assert any("answer" in m for m in final["messages"])
```

**What changed and why:**
- Two edges are added from `classify` to `expand_context` AND from `classify` to `check_safety`. LangGraph detects the fan-out and schedules both nodes before running `answer`.
- `answer` has two incoming edges (`expand_context` and `check_safety`). LangGraph treats this as a fan-in: `answer` only runs once both branches have written their state fields.
- State carries `context` and `safety_flag` — separate fields ensure the two branches don't step on each other.
- With a sync executor, branches execute in topological order (expand_context then check_safety, or vice versa, depending on insertion order); with `ainvoke()` they run concurrently.


## Exercise 4

**Use `graph.get_graph().draw_mermaid()` to print the graph structure and show how to inspect node names and edges programmatically.**

Design: `app.get_graph()` returns a `DrawableGraph` object. `draw_mermaid()` returns a Mermaid diagram string. `draw_ascii()` returns an ASCII diagram. The graph object also exposes `.nodes` (a dict of node name → node metadata) and `.edges` (a list of edge objects with `.source` and `.target` attributes) for programmatic inspection. This exercise extends the worked example with a `log_node` at the start and re-draws.

```python
"""
day4_ex4.py — Graph introspection: draw_mermaid(), draw_ascii(), nodes and edges.

Run: python day4_ex4.py
Env: ANTHROPIC_API_KEY (only needed if you uncomment app.invoke at the bottom)
"""
import logging
import operator
import os
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    query: str
    classification: str
    answer: str
    messages: Annotated[list[str], operator.add]


# ---------------------------------------------------------------------------
# LLM (only used if running live; mocked in tests)
# ---------------------------------------------------------------------------

llm = ChatAnthropic(
    model="claude-haiku-5-5",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)


# ---------------------------------------------------------------------------
# Nodes — original worked-example nodes
# ---------------------------------------------------------------------------

def log_node(state: AgentState) -> dict:
    """No-op node prepended to the graph to demonstrate structure changes."""
    logger.info("log_node: query received, length=%d", len(state["query"]))
    return {"messages": [f"log: query_len={len(state['query'])}"]}


def classify_node(state: AgentState) -> dict:
    logger.info("classify_node: classifying query")
    prompt = (
        "Classify this message with EXACTLY ONE word from: question, task, chitchat.\n"
        f"Message: {state['query']}\n"
        "Reply with only the single classification word."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    classification = response.content.strip().lower()
    return {
        "classification": classification,
        "messages": [f"classify: {classification}"],
    }


def answer_node(state: AgentState) -> dict:
    logger.info("answer_node: generating answer")
    prompt = (
        f"Query type: {state['classification']}\n"
        f"Query: {state['query']}\n"
        "Provide a concise answer."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    answer = response.content.strip()
    return {
        "answer": answer,
        "messages": [f"answer: {len(answer)} chars"],
    }


def format_node(state: AgentState) -> dict:
    formatted = (
        f"**Type:** {state['classification'].capitalize()}\n\n"
        f"**Answer:** {state['answer']}"
    )
    return {
        "answer": formatted,
        "messages": ["format: done"],
    }


# ---------------------------------------------------------------------------
# Graph A — original 3-node graph (without log_node)
# ---------------------------------------------------------------------------

def build_original_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("classify", classify_node)
    graph.add_node("answer", answer_node)
    graph.add_node("format", format_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "answer")
    graph.add_edge("answer", "format")
    graph.set_finish_point("format")

    return graph.compile()


# ---------------------------------------------------------------------------
# Graph B — extended graph with log_node at the start
# ---------------------------------------------------------------------------

def build_extended_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("log", log_node)           # NEW — entry point
    graph.add_node("classify", classify_node)
    graph.add_node("answer", answer_node)
    graph.add_node("format", format_node)

    graph.set_entry_point("log")              # log is now first
    graph.add_edge("log", "classify")
    graph.add_edge("classify", "answer")
    graph.add_edge("answer", "format")
    graph.set_finish_point("format")

    return graph.compile()


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------

def print_graph_info(app, label: str) -> None:
    """Print Mermaid diagram, ASCII diagram, and programmatic node/edge data."""
    graph_data = app.get_graph()

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # --- Mermaid diagram ---
    print("\n[Mermaid diagram]")
    print(graph_data.draw_mermaid())

    # --- ASCII diagram ---
    print("\n[ASCII diagram]")
    print(graph_data.draw_ascii())

    # --- Programmatic inspection: nodes ---
    print("\n[Node names]")
    for name in graph_data.nodes:
        print(f"  {name}")

    # --- Programmatic inspection: edges ---
    print("\n[Edges]")
    for edge in graph_data.edges:
        target_label = edge.target if edge.target else "END"
        print(f"  {edge.source} --> {target_label}")

    # --- Counts ---
    print(f"\nTotal nodes: {len(graph_data.nodes)}")
    print(f"Total edges: {len(graph_data.edges)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app_original = build_original_graph()
    app_extended = build_extended_graph()

    print_graph_info(app_original, "Original 3-node graph (classify → answer → format)")
    print_graph_info(app_extended, "Extended graph with log_node (log → classify → answer → format)")

    # Diff: what changed?
    orig_nodes = set(app_original.get_graph().nodes.keys())
    ext_nodes  = set(app_extended.get_graph().nodes.keys())
    added_nodes = ext_nodes - orig_nodes

    orig_edges = {(e.source, e.target) for e in app_original.get_graph().edges}
    ext_edges  = {(e.source, e.target) for e in app_extended.get_graph().edges}
    added_edges   = ext_edges - orig_edges
    removed_edges = orig_edges - ext_edges

    print("\n[Structural diff: extended vs original]")
    print(f"  Added nodes:   {sorted(added_nodes)}")
    print(f"  Added edges:   {sorted(str(e) for e in added_edges)}")
    print(f"  Removed edges: {sorted(str(e) for e in removed_edges)}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_draw_mermaid_returns_string() -> None:
    app = build_original_graph()
    diagram = app.get_graph().draw_mermaid()
    assert isinstance(diagram, str)
    assert len(diagram) > 0


def test_mermaid_contains_all_node_names() -> None:
    app = build_original_graph()
    diagram = app.get_graph().draw_mermaid()
    for node_name in ["classify", "answer", "format"]:
        assert node_name in diagram, f"Node {node_name!r} missing from mermaid diagram"


def test_mermaid_extended_contains_log_node() -> None:
    app = build_extended_graph()
    diagram = app.get_graph().draw_mermaid()
    assert "log" in diagram, "log_node must appear in extended graph mermaid diagram"


def test_programmatic_node_names_original() -> None:
    app = build_original_graph()
    node_names = set(app.get_graph().nodes.keys())
    assert {"classify", "answer", "format"}.issubset(node_names)


def test_programmatic_node_names_extended() -> None:
    app = build_extended_graph()
    node_names = set(app.get_graph().nodes.keys())
    assert "log" in node_names


def test_programmatic_edges_original() -> None:
    app = build_original_graph()
    edges = {(e.source, e.target) for e in app.get_graph().edges}
    assert ("classify", "answer") in edges
    assert ("answer", "format") in edges


def test_programmatic_edges_extended_adds_log() -> None:
    app = build_extended_graph()
    edges = {(e.source, e.target) for e in app.get_graph().edges}
    assert ("log", "classify") in edges
    # original edges still present
    assert ("classify", "answer") in edges
    assert ("answer", "format") in edges


def test_extended_graph_has_one_more_node_than_original() -> None:
    orig_nodes = app_original = build_original_graph()
    ext_nodes  = build_extended_graph()

    orig_count = len(build_original_graph().get_graph().nodes)
    ext_count  = len(build_extended_graph().get_graph().nodes)

    # Extended adds log_node — at minimum 1 more node
    assert ext_count >= orig_count + 1


def test_draw_ascii_returns_non_empty_string() -> None:
    app = build_original_graph()
    ascii_diagram = app.get_graph().draw_ascii()
    assert isinstance(ascii_diagram, str)
    assert len(ascii_diagram) > 0
```

**What changed and why:**
- `app.get_graph()` returns a `DrawableGraph`; calling `.draw_mermaid()` on it produces a Mermaid flowchart string you can paste into any Mermaid renderer or GitHub markdown.
- `.draw_ascii()` produces a compact terminal-friendly diagram — useful for quick inspection during development.
- `.nodes` is a dict keyed by node name; `.edges` is a list of edge objects with `.source` and `.target` string attributes — iterate these for structural analysis or test assertions.
- Adding `log_node` at the start produces one additional node and replaces the `__start__ → classify` edge with `__start__ → log → classify`, which is visible in both the Mermaid output and the programmatic edge list.
- The structural diff pattern (set difference on nodes/edges between two compiled graphs) is a useful technique for verifying graph modifications in tests.
