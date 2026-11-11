# Week 10 Assessment — Answer Key

---

## Part 1: Conceptual Questions (8 questions)

---

### Q1: What is LCEL and what problem does it solve vs hand-written Python?

**LCEL (LangChain Expression Language)** is a declarative composition framework that lets you build LLM pipelines using the `|` pipe operator: `prompt | llm | parser`. The problem it solves is the **wiring boilerplate** that accumulates in hand-written Python chains: manually passing outputs between components, handling streaming vs. batch consistently, adding retries/fallbacks, and enabling async execution all require separate code paths. LCEL abstracts these concerns behind the **Runnable protocol** — every component exposes the same interface (`invoke`, `stream`, `batch`, `ainvoke`, etc.), so composition, streaming, batching, and async all work automatically without changing your code. Hand-written Python gives full control but forces you to re-implement the plumbing (retry logic, streaming passthrough, parallel batching) on every project.

---

### Q2: Explain the Runnable protocol — what methods must a Runnable implement?

The **Runnable protocol** is an interface (abstract base class) that every LCEL component must satisfy. The required methods are:
- `invoke(input, config=None)` — synchronous single-item execution
- `stream(input, config=None)` — synchronous generator yielding incremental outputs
- `batch(inputs, config=None)` — synchronous execution over a list of inputs (default: calls invoke in a loop)
- `ainvoke(input, config=None)` — async version of invoke
- `astream(input, config=None)` — async generator version of stream
- `abatch(inputs, config=None)` — async batch

The power is that when you compose `A | B`, the resulting `RunnableSequence` is itself a Runnable — composability is recursive. Optional methods like `get_input_schema()` and `get_output_schema()` enable runtime introspection. `RunnableConfig` carries cross-cutting concerns (callbacks, tags, metadata, recursion limits) so components don't need to accept them explicitly.

---

### Q3: What is the difference between StateGraph and a simple function pipeline?

A **function pipeline** (e.g., `f3(f2(f1(x)))` or an LCEL chain) is a **linear, stateless** composition — output of one step is the input of the next, there are no loops, no branching based on runtime values, and no persistent intermediate state. A **StateGraph** is a **directed graph with shared mutable state**: every node reads from and writes to a shared `TypedDict` state, edges can be **conditional** (routing decisions made at runtime based on state values), and the graph can contain **cycles** (loops for retry, refinement, or multi-turn dialogue). StateGraph also integrates with a **checkpointer** for persistence and supports `interrupt_before`/`interrupt_after` for human-in-the-loop pauses — none of which are possible in a linear pipeline without significant custom infrastructure.

---

### Q4: How does LangGraph's checkpointing enable human-in-the-loop workflows?

LangGraph checkpointing works by **serializing the complete graph state to a store** (memory, SQLite, Postgres) after each node execution, keyed by `thread_id`. When a node calls `interrupt()`, execution suspends immediately and the current state is persisted. Later, the application calls `.invoke(Command(resume=<value>))` with the same `thread_id`, and LangGraph **re-reads the checkpoint**, restores the full state (message history, all node outputs, current position in the graph), and continues from the interrupted node with the human's input. This means the graph can pause for seconds, minutes, or days — it's not held in memory. The human's response becomes a first-class value in the graph state, indistinguishable from a tool output or LLM response, so the rest of the graph logic doesn't need to know whether a value came from a human or an automated source.

---

### Q5: When would you choose conditional edges over a linear graph?

Use conditional edges whenever the **next step depends on a runtime value** rather than being fixed at design time. Concrete scenarios: (1) **routing** — send user queries to different specialized agents (researcher vs. writer) based on the query type; (2) **retry loops** — re-run an LLM node if the output fails validation, up to N times; (3) **quality gates** — route low-quality outputs back for revision, high-quality ones forward to output; (4) **ReAct loops** — route to tool execution if the LLM produced a tool call, or to END if it produced a final answer; (5) **plan completion checks** — a replanner node decides whether to continue executing plan steps or declare the task done. If your logic is genuinely sequential with no branches or loops, a linear graph or LCEL chain is simpler and clearer — don't add conditional complexity you don't need.

---

### Q6: What is the role of `Annotated[list[T], operator.add]` in LangGraph state?

In a `TypedDict` state definition, `Annotated[list[T], operator.add]` declares a field as an **accumulating list**: instead of the latest node's return value overwriting the field, LangGraph **calls `operator.add` (list concatenation)** to merge the node's return value with the existing state. For example, `results: Annotated[list[str], operator.add]` means that when a node returns `{"results": ["new item"]}`, LangGraph appends `"new item"` to the existing `results` list rather than replacing it. This is the mechanism behind `MessagesState`'s `messages` field — `add_messages` is a more sophisticated reducer that also deduplicates by message ID. Without this annotation, every node would overwrite the field; with it, outputs from parallel branches or sequential nodes accumulate correctly without any node needing to read the current state first.

---

### Q7: Compare `MemorySaver` vs `SqliteSaver` — when to use each?

| | `MemorySaver` | `SqliteSaver` |
|---|---|---|
| **Storage** | In-process Python dict | SQLite file on disk |
| **Persistence** | Lost when process ends | Survives restarts |
| **Setup** | Zero configuration | Requires a file path |
| **Concurrency** | Thread-safe for single process | File-level locking (single writer) |
| **Scale** | Limited by RAM | Limited by disk + SQLite locks |

**Use `MemorySaver`** for: unit tests (fast, no cleanup needed), interactive notebooks, demos, single-request workflows where you don't need cross-request state. **Use `SqliteSaver`** for: development/staging environments where you need to pause and resume across process restarts, local CLI tools, single-user apps, and any workflow where the human-in-the-loop pause might span minutes or hours. For production multi-user systems, use `PostgresSaver` (or a cloud-managed equivalent) — SQLite doesn't handle concurrent writers from multiple processes.

---

### Q8: Give 3 concrete signals that indicate you should use LangGraph rather than raw LCEL

1. **The workflow has loops or retries.** If any step can re-run based on output quality, validation failure, or tool results (e.g., ReAct's think→act→observe cycle, a retry-on-bad-output loop), you need a graph with back-edges. LCEL chains are DAGs — they have no mechanism for cycles without hand-coding a `while` loop that loses all graph features.

2. **Human-in-the-loop pauses are required.** If a human needs to approve, edit, or provide input mid-workflow and that pause might last minutes/hours (or the process might restart between the pause and the resume), LangGraph's `interrupt()` + checkpointing + `Command(resume=...)` pattern is the only clean solution. Implementing this in LCEL requires custom state persistence, serialization, and resume logic — effectively reinventing LangGraph.

3. **Multiple agents or branches need to share structured state.** When more than one agent reads from and writes to shared state (supervisor routing, parallel research branches merging their results, a planner and executor both updating the same plan list), you need typed reducers, conditional routing, and a single serializable state object. An LCEL chain passes outputs as positional arguments and has no shared state concept — you'd have to manually thread dicts through every component.

---

## Part 2: Coding Challenge — 5-Node Document Analysis Agent

### Solution

```python
"""
Week 10 Assessment — Coding Challenge Solution
5-node document analysis agent with classify → extract → grade → approve → format.
"""

import logging
import operator
import os
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Pydantic models ────────────────────────────────────────────────────────

class DocInfo(BaseModel):
    """Structured document metadata extracted by the LLM."""
    title: str = Field(description="Document title or best-guess heading")
    date: str = Field(description="Document date (ISO 8601 or 'unknown')")
    key_points: list[str] = Field(description="3-5 key points from the document")
    quality_score: int = Field(ge=1, le=10, description="Document quality score 1-10")


# ── State ─────────────────────────────────────────────────────────────────

class DocAnalysisState(TypedDict):
    document_text: str
    doc_type: str                          # "contract" | "report" | "email"
    doc_info: DocInfo | None
    retry_count: int
    approved: bool
    output: str


# ── Nodes ─────────────────────────────────────────────────────────────────

def classify_doc(state: DocAnalysisState) -> DocAnalysisState:
    """Node 1: Classify the document as contract, report, or email."""
    logger.info("classify_doc: classifying document")
    llm = ChatAnthropic(
        model="claude-haiku-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    response = llm.invoke([
        SystemMessage(content=(
            "Classify the document as exactly one of: contract, report, email. "
            "Reply with only the single word."
        )),
        HumanMessage(content=state["document_text"][:2000]),
    ])
    doc_type = response.content.strip().lower()
    if doc_type not in {"contract", "report", "email"}:
        doc_type = "report"  # safe fallback
    logger.info("classify_doc: type='%s'", doc_type)
    return {"doc_type": doc_type}


def extract_info(state: DocAnalysisState) -> DocAnalysisState:
    """Node 2: Extract structured DocInfo using PydanticOutputParser."""
    logger.info("extract_info: extracting (attempt %d)", state.get("retry_count", 0) + 1)
    parser = PydanticOutputParser(pydantic_object=DocInfo)
    llm = ChatAnthropic(
        model="claude-sonnet-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    prompt = (
        f"Extract structured information from this {state['doc_type']}.\n\n"
        f"Document:\n{state['document_text'][:3000]}\n\n"
        f"{parser.get_format_instructions()}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        doc_info = parser.parse(response.content)
    except Exception as e:
        logger.warning("extract_info: parse failed (%s) — using defaults", e)
        doc_info = DocInfo(
            title="Unknown",
            date="unknown",
            key_points=["Extraction failed"],
            quality_score=1,
        )
    logger.info("extract_info: quality_score=%d", doc_info.quality_score)
    return {
        "doc_info": doc_info,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def grade_quality(state: DocAnalysisState) -> DocAnalysisState:
    """
    Node 3: Conditional routing based on quality_score.
    This node itself does nothing — routing is handled by the conditional edge.
    """
    logger.info(
        "grade_quality: score=%d, retries=%d",
        state["doc_info"].quality_score if state.get("doc_info") else 0,
        state.get("retry_count", 0),
    )
    return {}  # no state change — routing function does the work


def approve_interrupt(state: DocAnalysisState) -> DocAnalysisState:
    """Node 4: HITL — pause for human approval."""
    logger.info("approve_interrupt: pausing for human approval")
    decision = interrupt({
        "question": "Approve this document analysis?",
        "doc_type": state["doc_type"],
        "title": state["doc_info"].title if state.get("doc_info") else "unknown",
        "quality_score": state["doc_info"].quality_score if state.get("doc_info") else 0,
    })
    approved = str(decision).strip().lower() in {"yes", "y", "approve", "true"}
    logger.info("approve_interrupt: approved=%s", approved)
    return {"approved": approved}


def format_output(state: DocAnalysisState) -> DocAnalysisState:
    """Node 5: Format the final DocInfo as a JSON string."""
    logger.info("format_output: formatting")
    if not state.get("doc_info"):
        output = '{"error": "no doc_info available"}'
    else:
        output = state["doc_info"].model_dump_json(indent=2)
    logger.info("format_output: output length=%d", len(output))
    return {"output": output}


# ── Conditional routing ────────────────────────────────────────────────────

def route_grade(state: DocAnalysisState) -> str:
    """
    After grade_quality: if score < 7, retry extract_info (up to 2 times);
    otherwise proceed to approve_interrupt.
    """
    doc_info = state.get("doc_info")
    retry_count = state.get("retry_count", 0)

    if doc_info is None or (doc_info.quality_score < 7 and retry_count < 3):
        logger.info("route_grade: low quality, routing to extract_info (retry #%d)", retry_count)
        return "extract_info"

    logger.info("route_grade: quality sufficient or max retries reached, routing to approve_interrupt")
    return "approve_interrupt"


# ── Graph assembly ─────────────────────────────────────────────────────────

def build_doc_analysis_graph() -> StateGraph:
    builder = StateGraph(DocAnalysisState)

    builder.add_node("classify_doc", classify_doc)
    builder.add_node("extract_info", extract_info)
    builder.add_node("grade_quality", grade_quality)
    builder.add_node("approve_interrupt", approve_interrupt)
    builder.add_node("format_output", format_output)

    builder.set_entry_point("classify_doc")
    builder.add_edge("classify_doc", "extract_info")
    builder.add_edge("extract_info", "grade_quality")
    builder.add_conditional_edges(
        "grade_quality",
        route_grade,
        {"extract_info": "extract_info", "approve_interrupt": "approve_interrupt"},
    )
    builder.add_edge("approve_interrupt", "format_output")
    builder.add_edge("format_output", END)

    return builder.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["approve_interrupt"],
    )


# ── Tests ──────────────────────────────────────────────────────────────────

"""
tests/test_doc_analysis.py

Run with: pytest tests/test_doc_analysis.py -v
"""
import json
import unittest
from unittest.mock import MagicMock, patch


def test_classify_doc_returns_contract() -> None:
    """
    Test that classify_doc correctly sets doc_type from LLM response.
    All LLM calls are mocked — no API key required.
    """
    # Arrange
    mock_response = MagicMock()
    mock_response.content = "contract"

    state: DocAnalysisState = {
        "document_text": "This Agreement is entered into between...",
        "doc_type": "",
        "doc_info": None,
        "retry_count": 0,
        "approved": False,
        "output": "",
    }

    # Act — patch the ChatAnthropic class so no real API call is made
    with patch("langchain_anthropic.ChatAnthropic") as MockLLM:
        mock_instance = MockLLM.return_value
        mock_instance.invoke.return_value = mock_response

        result = classify_doc(state)

    # Assert
    assert result["doc_type"] == "contract", f"Expected 'contract', got '{result['doc_type']}'"


def test_extract_info_populates_doc_info() -> None:
    """
    Test that extract_info parses a valid DocInfo from a mocked LLM response.
    Uses a realistic JSON string that PydanticOutputParser can parse.
    """
    # Arrange: mock LLM returns valid JSON matching DocInfo schema
    mock_doc_info_json = json.dumps({
        "title": "Q3 Sales Report",
        "date": "2025-09-30",
        "key_points": ["Revenue up 15%", "New markets entered", "Headcount stable"],
        "quality_score": 8,
    })
    mock_response = MagicMock()
    mock_response.content = mock_doc_info_json

    state: DocAnalysisState = {
        "document_text": "Q3 Sales Report — Revenue up 15% year-over-year...",
        "doc_type": "report",
        "doc_info": None,
        "retry_count": 0,
        "approved": False,
        "output": "",
    }

    # Act
    with patch("langchain_anthropic.ChatAnthropic") as MockLLM:
        mock_instance = MockLLM.return_value
        mock_instance.invoke.return_value = mock_response

        result = extract_info(state)

    # Assert
    assert result["doc_info"] is not None, "doc_info should be populated"
    assert isinstance(result["doc_info"], DocInfo), "doc_info should be a DocInfo instance"
    assert result["doc_info"].quality_score == 8
    assert result["doc_info"].title == "Q3 Sales Report"
    assert result["retry_count"] == 1


# ── Grading rubric ─────────────────────────────────────────────────────────
```

### Grading Rubric Notes

| Component | Points | What to check |
|---|---|---|
| `DocInfo` Pydantic model | 10 | All 4 fields with types, `ge/le` constraint on `quality_score` |
| `DocAnalysisState` TypedDict | 10 | All 6 fields typed correctly |
| `classify_doc` node | 10 | Calls LLM, returns valid `doc_type`, handles fallback |
| `extract_info` node | 15 | Uses `PydanticOutputParser`, handles parse failure gracefully |
| `grade_quality` routing | 15 | `score < 7` → retry; `score >= 7` → approve; max retry guard |
| `approve_interrupt` + `interrupt()` | 15 | Calls `interrupt()`, interprets human response as bool |
| `format_output` node | 5 | Calls `.model_dump_json()` on DocInfo |
| `MemorySaver` + `interrupt_before` | 10 | Correct compile call with both args |
| Tests mock LLM correctly | 10 | `patch("langchain_anthropic.ChatAnthropic")`, no real keys |
| **Total** | **100** | |

**Common mistakes to deduct:**
- `-10`: Using `FakeListLLM` or hardcoded API key in tests
- `-10`: Missing `interrupt_before=["approve_interrupt"]` in compile
- `-5`: `quality_score` field has no `ge=1, le=10` constraint
- `-5`: `grade_quality` routing allows infinite retries (no max guard)
- `-5`: `extract_info` crashes on parse failure instead of using fallback
