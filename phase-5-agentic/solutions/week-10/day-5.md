# Solutions — Week 10 · Day 5 — LangGraph Conditional Edges, Cycles & Checkpointing

> These are reference solutions. Your implementation may differ and still be correct.

## Exercise 1

**Implement a hard interrupt via `interrupt_before=["execute"]` on `compile()` so the graph pauses before every execution step — show how to inspect paused state and resume.**

Design: `interrupt_before=["execute"]` tells the compiled graph to pause (raise a `GraphInterrupt`) just before entering the `execute` node. The run must be driven with `stream()` rather than `invoke()` to observe the interruption without an exception propagating to the caller. After inspecting the paused state via `app.get_state(config)`, pass `None` as input to `app.invoke()` with the same `config` to resume from the checkpoint.

```python
"""
day5_ex1.py — Hard interrupt via interrupt_before=["execute"].

Demonstrates:
  1. Compiling with interrupt_before=["execute"]
  2. Driving the graph with stream() to reach the interruption point
  3. Inspecting paused state with app.get_state(config)
  4. Resuming by calling app.invoke(None, config=config)

Run: python day5_ex1.py
Env: ANTHROPIC_API_KEY
"""
import logging
import operator
import os
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

QUALITY_THRESHOLD = 0.7
MAX_ITERATIONS = 3


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class PlanExecuteState(TypedDict):
    goal: str
    plan: str
    result: str
    quality: float
    iterations: int
    messages: Annotated[list[str], operator.add]


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

llm = ChatAnthropic(
    model="claude-sonnet-5-5",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def plan_node(state: PlanExecuteState) -> dict:
    iteration = state["iterations"] + 1
    logger.info("plan_node: iteration=%d", iteration)
    context = ""
    if state["result"]:
        context = f"\nPrevious result: {state['result']}\nScore: {state['quality']:.2f}. Improve."

    prompt = f"Goal: {state['goal']}{context}\n\nWrite a concise 3-step plan."
    response = llm.invoke([HumanMessage(content=prompt)])
    plan = response.content.strip()
    return {
        "plan": plan,
        "iterations": iteration,
        "messages": [f"plan: iteration {iteration}"],
    }


def execute_node(state: PlanExecuteState) -> dict:
    """
    This node is preceded by a hard interrupt.
    The graph pauses before entering here on every cycle.
    """
    logger.info("execute_node: executing plan (graph was interrupted before this)")
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
        quality = 0.5
    logger.info("verify_node: quality=%.2f", quality)
    return {
        "quality": quality,
        "messages": [f"verify: quality={quality:.2f}"],
    }


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def quality_router(state: PlanExecuteState) -> str:
    if state["quality"] >= QUALITY_THRESHOLD:
        logger.info("quality_router: threshold met — routing to END")
        return "done"
    if state["iterations"] >= MAX_ITERATIONS:
        logger.info("quality_router: max iterations reached — routing to END")
        return "done"
    logger.info("quality_router: quality too low — retrying")
    return "retry"


# ---------------------------------------------------------------------------
# Graph construction — with interrupt_before=["execute"]
# ---------------------------------------------------------------------------

def build_graph(checkpointer: MemorySaver) -> StateGraph:
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
        path_map={"retry": "plan", "done": END},
    )

    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["execute"],   # <-- hard interrupt before every execute step
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    checkpointer = MemorySaver()
    app = build_graph(checkpointer)

    config = {"configurable": {"thread_id": "interrupt-demo-1"}}
    initial_state: PlanExecuteState = {
        "goal": "Write a one-paragraph summary of the transformer architecture.",
        "plan": "",
        "result": "",
        "quality": 0.0,
        "iterations": 0,
        "messages": [],
    }

    # --- Phase 1: stream until the interrupt fires ---
    print("=== Phase 1: running until interrupt ===")
    for event in app.stream(initial_state, config=config):
        for node_name, output in event.items():
            logger.info("streamed event from node=%s", node_name)

    # --- Inspect paused state ---
    paused = app.get_state(config)
    print(f"\n[Paused state]")
    print(f"  Next nodes to run: {paused.next}")
    print(f"  Plan so far:       {paused.values.get('plan', '')[:120]}...")
    print(f"  Iterations:        {paused.values.get('iterations', 0)}")
    print(f"  Messages:          {paused.values.get('messages', [])}")

    assert paused.next == ("execute",), (
        f"Expected graph to pause before 'execute', got: {paused.next}"
    )

    # --- Phase 2: resume by passing None as input ---
    print("\n=== Phase 2: resuming from checkpoint ===")
    # The graph continues from the execute node through to the next interrupt
    # (or END if quality threshold is met).  For brevity, drive one more resume cycle.
    resumed_state = app.invoke(None, config=config)

    print("\n=== Resumed — current result ===")
    print(resumed_state["result"][:300])
    print(f"\nQuality: {resumed_state.get('quality', 'N/A')}  |  Iterations: {resumed_state['iterations']}")


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


def test_graph_pauses_before_execute() -> None:
    """After streaming through plan_node, the graph must pause before execute."""
    checkpointer = MemorySaver()
    app = build_graph(checkpointer)
    config = {"configurable": {"thread_id": "test-interrupt-1"}}

    with patch("__main__.llm") as mock_llm:
        mock_llm.invoke.return_value = _make_llm_response("Step 1. Step 2. Step 3.")
        initial: PlanExecuteState = {
            "goal": "Test goal",
            "plan": "",
            "result": "",
            "quality": 0.0,
            "iterations": 0,
            "messages": [],
        }
        # Drain the stream — the interrupt fires after plan_node completes
        for _ in app.stream(initial, config=config):
            pass

    paused = app.get_state(config)
    assert paused.next == ("execute",), (
        f"Graph should pause before execute, got next={paused.next}"
    )


def test_paused_state_contains_plan() -> None:
    """Paused state should have the plan from plan_node."""
    checkpointer = MemorySaver()
    app = build_graph(checkpointer)
    config = {"configurable": {"thread_id": "test-interrupt-2"}}

    with patch("__main__.llm") as mock_llm:
        mock_llm.invoke.return_value = _make_llm_response("My three-step plan.")
        initial: PlanExecuteState = {
            "goal": "Test goal",
            "plan": "",
            "result": "",
            "quality": 0.0,
            "iterations": 0,
            "messages": [],
        }
        for _ in app.stream(initial, config=config):
            pass

    paused = app.get_state(config)
    assert paused.values.get("plan") == "My three-step plan."


def test_resume_runs_execute_node() -> None:
    """After pausing, app.invoke(None, config) must run execute_node."""
    checkpointer = MemorySaver()
    app = build_graph(checkpointer)
    config = {"configurable": {"thread_id": "test-interrupt-3"}}

    plan_response = "Step 1. Step 2. Step 3."
    execute_response = "Execution result here."
    verify_response = "0.9"  # above threshold → done on first iteration

    with patch("__main__.llm") as mock_llm:
        mock_llm.invoke.side_effect = [
            _make_llm_response(plan_response),    # plan_node
            _make_llm_response(execute_response), # execute_node (after resume)
            _make_llm_response(verify_response),  # verify_node
        ]
        initial: PlanExecuteState = {
            "goal": "Test goal",
            "plan": "",
            "result": "",
            "quality": 0.0,
            "iterations": 0,
            "messages": [],
        }
        # Phase 1: stream to interrupt
        for _ in app.stream(initial, config=config):
            pass

        # Phase 2: resume
        final = app.invoke(None, config=config)

    assert final["result"] == execute_response
    assert final["quality"] == 0.9


def test_interrupt_before_not_in_non_interrupt_graph() -> None:
    """Without interrupt_before, the graph should run to completion."""
    checkpointer = MemorySaver()
    # Build without interrupt
    graph = StateGraph(PlanExecuteState)
    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("verify", verify_node)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "verify")
    graph.add_conditional_edges("verify", quality_router, {"retry": "plan", "done": END})
    app = graph.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "no-interrupt-test"}}
    with patch("__main__.llm") as mock_llm:
        mock_llm.invoke.side_effect = [
            _make_llm_response("Plan."),
            _make_llm_response("Result."),
            _make_llm_response("0.95"),
        ]
        initial: PlanExecuteState = {
            "goal": "Test",
            "plan": "",
            "result": "",
            "quality": 0.0,
            "iterations": 0,
            "messages": [],
        }
        final = app.invoke(initial, config=config)

    # Ran to completion — result is populated and quality above threshold
    assert final["result"] == "Result."
    paused = app.get_state(config)
    assert paused.next == ()  # nothing left to run
```

**What changed and why:**
- `interrupt_before=["execute"]` is passed to `graph.compile()`. This installs a checkpoint just before the named node runs; when the executor reaches it, execution stops and control returns to the caller.
- `app.stream()` is used instead of `app.invoke()` to drive Phase 1 — `invoke()` would block until the graph is done or raise `GraphInterruptException`; `stream()` yields events per node and returns cleanly when the interrupt fires.
- `app.get_state(config)` retrieves the `StateSnapshot` at the interruption point. `.next` is a tuple of node names scheduled to run next — `("execute",)` confirms the graph is paused where expected.
- `app.invoke(None, config=config)` resumes: passing `None` as input tells LangGraph to load the last checkpoint and continue from there instead of starting fresh.


## Exercise 2

**Replace `MemorySaver` with `SqliteSaver` writing to `/tmp/week10.db` — show the connection and cleanup.**

Design: `SqliteSaver` persists checkpoints to an SQLite file. The canonical pattern uses `SqliteSaver.from_conn_string("/tmp/week10.db")` as a context manager so the connection is closed on exit. Without a context manager, call `checkpointer.conn.close()` explicitly. The graph logic is identical to the worked example — only the checkpointer construction changes.

```python
"""
day5_ex2.py — SqliteSaver persistence writing to /tmp/week10.db.

Demonstrates:
  1. Creating SqliteSaver with from_conn_string()
  2. Running the graph in "Process A" (first invocation)
  3. Reconnecting in "Process B" (second invocation, same thread_id)
  4. Verifying state is retrieved from disk

Run: python day5_ex2.py
Env: ANTHROPIC_API_KEY
"""
import logging
import operator
import os
import sqlite3
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path("/tmp/week10.db")
QUALITY_THRESHOLD = 0.7
MAX_ITERATIONS = 3


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class PlanExecuteState(TypedDict):
    goal: str
    plan: str
    result: str
    quality: float
    iterations: int
    messages: Annotated[list[str], operator.add]


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

llm = ChatAnthropic(
    model="claude-sonnet-5-5",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)


# ---------------------------------------------------------------------------
# Nodes (same as worked example)
# ---------------------------------------------------------------------------

def plan_node(state: PlanExecuteState) -> dict:
    iteration = state["iterations"] + 1
    logger.info("plan_node: iteration=%d", iteration)
    context = ""
    if state["result"]:
        context = f"\nPrevious result: {state['result']}\nScore: {state['quality']:.2f}. Improve."
    prompt = f"Goal: {state['goal']}{context}\n\nWrite a concise 3-step plan."
    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "plan": response.content.strip(),
        "iterations": iteration,
        "messages": [f"plan: iteration {iteration}"],
    }


def execute_node(state: PlanExecuteState) -> dict:
    logger.info("execute_node: executing plan")
    prompt = (
        f"Goal: {state['goal']}\n"
        f"Plan:\n{state['plan']}\n\n"
        "Execute this plan and produce the final output."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    result = response.content.strip()
    return {
        "result": result,
        "messages": [f"execute: {len(result)} chars"],
    }


def verify_node(state: PlanExecuteState) -> dict:
    logger.info("verify_node: scoring result")
    prompt = (
        f"Goal: {state['goal']}\n"
        f"Result:\n{state['result']}\n\n"
        "Rate quality 0.0–1.0. Reply with ONLY a decimal number."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        quality = max(0.0, min(1.0, float(response.content.strip())))
    except ValueError:
        quality = 0.5
    logger.info("verify_node: quality=%.2f", quality)
    return {
        "quality": quality,
        "messages": [f"verify: quality={quality:.2f}"],
    }


def quality_router(state: PlanExecuteState) -> str:
    if state["quality"] >= QUALITY_THRESHOLD or state["iterations"] >= MAX_ITERATIONS:
        return "done"
    return "retry"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(checkpointer: SqliteSaver) -> StateGraph:
    graph = StateGraph(PlanExecuteState)
    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("verify", verify_node)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "verify")
    graph.add_conditional_edges(
        "verify", quality_router, {"retry": "plan", "done": END}
    )
    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Entry point — simulates two separate process invocations
# ---------------------------------------------------------------------------

def run_process_a(thread_id: str) -> dict:
    """
    Process A: first invocation — runs the graph and persists to SQLite.
    Returns the final state dict.
    """
    logger.info("=== Process A: first run, thread_id=%s ===", thread_id)

    # Context manager ensures the connection is closed cleanly
    with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        app = build_graph(checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        initial_state: PlanExecuteState = {
            "goal": "Write a one-paragraph summary of the transformer architecture.",
            "plan": "",
            "result": "",
            "quality": 0.0,
            "iterations": 0,
            "messages": [],
        }
        final_state = app.invoke(initial_state, config=config)

    logger.info("Process A complete. DB written to %s", DB_PATH)
    return final_state


def run_process_b(thread_id: str) -> dict:
    """
    Process B: reconnects to the same DB file, passes None as input,
    and retrieves the completed state from the persisted checkpoint.
    """
    logger.info("=== Process B: reconnecting, thread_id=%s ===", thread_id)

    with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        app = build_graph(checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        # app.invoke(None, ...) loads the last checkpoint for this thread_id.
        # The graph is already at END, so it returns the final persisted state.
        retrieved_state = app.invoke(None, config=config)

    return retrieved_state


def main() -> None:
    # Clean up any previous run for a clean demo
    if DB_PATH.exists():
        DB_PATH.unlink()
        logger.info("Deleted existing DB at %s", DB_PATH)

    thread_id = "sqlite-demo-run-1"

    final_state_a = run_process_a(thread_id)
    print("\n=== Process A — Final result ===")
    print(final_state_a["result"][:300])
    print(f"Quality: {final_state_a['quality']:.2f}  |  Iterations: {final_state_a['iterations']}")

    retrieved_state_b = run_process_b(thread_id)
    print("\n=== Process B — Retrieved from SQLite ===")
    print(retrieved_state_b["result"][:300])

    assert retrieved_state_b["result"] == final_state_a["result"], (
        "Process B must retrieve the same result as Process A persisted"
    )
    print("\nAssertion passed: Process B retrieved identical state from SQLite.")

    # Cleanup
    if DB_PATH.exists():
        DB_PATH.unlink()
        logger.info("Cleaned up DB at %s", DB_PATH)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

import unittest.mock
from unittest.mock import MagicMock, patch
import tempfile


def _make_llm_response(content: str) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.content = content
    return mock_resp


def test_sqlite_saver_persists_state() -> None:
    """State written in one connection is readable in a fresh connection."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    thread_id = "test-sqlite-1"
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # First connection — run the graph
        with patch("__main__.llm") as mock_llm:
            mock_llm.invoke.side_effect = [
                _make_llm_response("Plan here."),
                _make_llm_response("Result here."),
                _make_llm_response("0.9"),
            ]
            with SqliteSaver.from_conn_string(db_path) as checkpointer:
                app = build_graph(checkpointer)
                initial: PlanExecuteState = {
                    "goal": "Test",
                    "plan": "",
                    "result": "",
                    "quality": 0.0,
                    "iterations": 0,
                    "messages": [],
                }
                final_a = app.invoke(initial, config=config)

        # Second connection — retrieve from disk
        with SqliteSaver.from_conn_string(db_path) as checkpointer2:
            app2 = build_graph(checkpointer2)
            final_b = app2.invoke(None, config=config)

        assert final_b["result"] == "Result here."
        assert final_b["quality"] == 0.9

    finally:
        Path(db_path).unlink(missing_ok=True)


def test_sqlite_db_file_is_created() -> None:
    """SqliteSaver creates the DB file on disk."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as tmp:
        db_path = tmp.name  # deleted by context manager

    try:
        with patch("__main__.llm") as mock_llm:
            mock_llm.invoke.side_effect = [
                _make_llm_response("Plan."),
                _make_llm_response("Result."),
                _make_llm_response("0.95"),
            ]
            with SqliteSaver.from_conn_string(db_path) as checkpointer:
                app = build_graph(checkpointer)
                config = {"configurable": {"thread_id": "db-file-test"}}
                initial: PlanExecuteState = {
                    "goal": "Test",
                    "plan": "",
                    "result": "",
                    "quality": 0.0,
                    "iterations": 0,
                    "messages": [],
                }
                app.invoke(initial, config=config)

        assert Path(db_path).exists(), "SQLite DB file should exist after graph run"

    finally:
        Path(db_path).unlink(missing_ok=True)


def test_different_thread_ids_are_isolated_in_sqlite() -> None:
    """Two thread IDs must produce independently retrievable states."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        with patch("__main__.llm") as mock_llm:
            mock_llm.invoke.side_effect = [
                # thread-a
                _make_llm_response("Plan A."),
                _make_llm_response("Result A."),
                _make_llm_response("0.9"),
                # thread-b
                _make_llm_response("Plan B."),
                _make_llm_response("Result B."),
                _make_llm_response("0.9"),
            ]
            with SqliteSaver.from_conn_string(db_path) as checkpointer:
                app = build_graph(checkpointer)

                config_a = {"configurable": {"thread_id": "thread-a"}}
                config_b = {"configurable": {"thread_id": "thread-b"}}

                initial: PlanExecuteState = {
                    "goal": "Test",
                    "plan": "",
                    "result": "",
                    "quality": 0.0,
                    "iterations": 0,
                    "messages": [],
                }
                final_a = app.invoke(initial, config=config_a)
                final_b = app.invoke(initial, config=config_b)

        assert final_a["result"] == "Result A."
        assert final_b["result"] == "Result B."
        assert final_a["result"] != final_b["result"]

    finally:
        Path(db_path).unlink(missing_ok=True)
```

**What changed and why:**
- `MemorySaver()` is replaced with `SqliteSaver.from_conn_string(str(DB_PATH))`. The interface is identical — both implement `BaseCheckpointSaver`.
- Using `SqliteSaver` as a context manager (`with SqliteSaver.from_conn_string(...) as checkpointer`) ensures the underlying SQLite connection is closed after the block, preventing file locks on the `.db` file.
- "Process B" calls `app.invoke(None, config=config)` — `None` input tells LangGraph to load the last checkpoint for the thread from the database. Because the graph already reached `END`, it returns the persisted final state.
- The only prerequisite is the same `thread_id` in both connections; the graph definition itself need not be aware of persistence.


## Exercise 3

**Run the same graph with two different `thread_id`s simultaneously (simulating two independent users) and show their states are isolated.**

Design: Use two distinct `thread_id` values with the same `MemorySaver`-backed graph. After both runs complete, call `app.get_state_history(config)` for each thread to print per-thread checkpoint summaries. Because checkpoints are keyed by `thread_id`, one thread's state never bleeds into the other's.

```python
"""
day5_ex3.py — Two concurrent users on the same graph, isolated by thread_id.

Demonstrates:
  1. Running the same compiled graph with thread_id="user-alice" and thread_id="user-bob"
  2. Confirming state isolation via app.get_state_history()
  3. Per-thread summary: checkpoints, final quality, iterations

Run: python day5_ex3.py
Env: ANTHROPIC_API_KEY
"""
import logging
import operator
import os
from dataclasses import dataclass
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

QUALITY_THRESHOLD = 0.7
MAX_ITERATIONS = 3


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class PlanExecuteState(TypedDict):
    goal: str
    plan: str
    result: str
    quality: float
    iterations: int
    messages: Annotated[list[str], operator.add]


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

llm = ChatAnthropic(
    model="claude-sonnet-5-5",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def plan_node(state: PlanExecuteState) -> dict:
    iteration = state["iterations"] + 1
    logger.info("plan_node: iteration=%d", iteration)
    context = ""
    if state["result"]:
        context = f"\nPrevious result: {state['result']}\nScore: {state['quality']:.2f}. Improve."
    prompt = f"Goal: {state['goal']}{context}\n\nWrite a concise 3-step plan."
    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "plan": response.content.strip(),
        "iterations": iteration,
        "messages": [f"plan: iteration {iteration}"],
    }


def execute_node(state: PlanExecuteState) -> dict:
    logger.info("execute_node: executing plan")
    prompt = (
        f"Goal: {state['goal']}\n"
        f"Plan:\n{state['plan']}\n\n"
        "Execute this plan and produce the final output."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    result = response.content.strip()
    return {
        "result": result,
        "messages": [f"execute: {len(result)} chars"],
    }


def verify_node(state: PlanExecuteState) -> dict:
    logger.info("verify_node: scoring result")
    prompt = (
        f"Goal: {state['goal']}\n"
        f"Result:\n{state['result']}\n\n"
        "Rate quality 0.0–1.0. Reply with ONLY a decimal number."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        quality = max(0.0, min(1.0, float(response.content.strip())))
    except ValueError:
        quality = 0.5
    return {
        "quality": quality,
        "messages": [f"verify: quality={quality:.2f}"],
    }


def quality_router(state: PlanExecuteState) -> str:
    if state["quality"] >= QUALITY_THRESHOLD or state["iterations"] >= MAX_ITERATIONS:
        return "done"
    return "retry"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph(checkpointer: MemorySaver) -> StateGraph:
    graph = StateGraph(PlanExecuteState)
    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("verify", verify_node)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "verify")
    graph.add_conditional_edges("verify", quality_router, {"retry": "plan", "done": END})
    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Thread summary helper
# ---------------------------------------------------------------------------

@dataclass
class ThreadSummary:
    thread_id: str
    checkpoint_count: int
    final_quality: float
    final_iterations: int
    final_goal: str


def get_thread_summary(app, config: dict) -> ThreadSummary:
    """
    Summarise a completed thread using get_state_history().

    get_state_history() returns StateSnapshot objects in reverse chronological order.
    The first snapshot is the most recent (final) state.
    """
    history = list(app.get_state_history(config))
    if not history:
        raise RuntimeError(f"No checkpoints found for config={config}")

    final_snapshot = history[0]  # most recent = index 0 (reverse chron order)
    final_values = final_snapshot.values

    return ThreadSummary(
        thread_id=config["configurable"]["thread_id"],
        checkpoint_count=len(history),
        final_quality=final_values.get("quality", 0.0),
        final_iterations=final_values.get("iterations", 0),
        final_goal=final_values.get("goal", ""),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # One shared checkpointer — thread isolation is by thread_id, not by saver instance
    checkpointer = MemorySaver()
    app = build_graph(checkpointer)

    # --- Run for user Alice ---
    config_alice = {"configurable": {"thread_id": "user-alice"}}
    initial_alice: PlanExecuteState = {
        "goal": "Write a haiku about machine learning.",
        "plan": "",
        "result": "",
        "quality": 0.0,
        "iterations": 0,
        "messages": [],
    }
    logger.info("Running for user-alice")
    final_alice = app.invoke(initial_alice, config=config_alice)

    # --- Run for user Bob ---
    config_bob = {"configurable": {"thread_id": "user-bob"}}
    initial_bob: PlanExecuteState = {
        "goal": "Explain gradient descent in two sentences.",
        "plan": "",
        "result": "",
        "quality": 0.0,
        "iterations": 0,
        "messages": [],
    }
    logger.info("Running for user-bob")
    final_bob = app.invoke(initial_bob, config=config_bob)

    # --- Per-thread summaries ---
    summary_alice = get_thread_summary(app, config_alice)
    summary_bob   = get_thread_summary(app, config_bob)

    print("\n=== Thread Summaries ===")
    for summary in (summary_alice, summary_bob):
        print(
            f"  thread_id={summary.thread_id!r}  "
            f"checkpoints={summary.checkpoint_count}  "
            f"quality={summary.final_quality:.2f}  "
            f"iterations={summary.final_iterations}  "
            f"goal={summary.final_goal!r}"
        )

    # --- Verify isolation ---
    print("\n=== Results ===")
    print(f"Alice ({final_alice['goal']!r}):\n  {final_alice['result'][:200]}\n")
    print(f"Bob   ({final_bob['goal']!r}):\n  {final_bob['result'][:200]}\n")

    assert final_alice["goal"] == "Write a haiku about machine learning.", \
        "Alice's state must contain Alice's goal"
    assert final_bob["goal"] == "Explain gradient descent in two sentences.", \
        "Bob's state must contain Bob's goal"
    assert final_alice["result"] != final_bob["result"], \
        "Different threads must produce different results"

    print("Isolation confirmed: Alice's and Bob's states are fully independent.")


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


def test_two_threads_have_isolated_goals() -> None:
    """Alice's and Bob's goals must not cross-contaminate."""
    checkpointer = MemorySaver()
    app = build_graph(checkpointer)

    config_a = {"configurable": {"thread_id": "test-alice"}}
    config_b = {"configurable": {"thread_id": "test-bob"}}

    initial_a: PlanExecuteState = {
        "goal": "Alice's goal",
        "plan": "",
        "result": "",
        "quality": 0.0,
        "iterations": 0,
        "messages": [],
    }
    initial_b: PlanExecuteState = {
        "goal": "Bob's goal",
        "plan": "",
        "result": "",
        "quality": 0.0,
        "iterations": 0,
        "messages": [],
    }

    with patch("__main__.llm") as mock_llm:
        mock_llm.invoke.side_effect = [
            _make_llm_response("Plan for Alice."),
            _make_llm_response("Result for Alice."),
            _make_llm_response("0.9"),
            _make_llm_response("Plan for Bob."),
            _make_llm_response("Result for Bob."),
            _make_llm_response("0.9"),
        ]
        final_a = app.invoke(initial_a, config=config_a)
        final_b = app.invoke(initial_b, config=config_b)

    assert final_a["goal"] == "Alice's goal"
    assert final_b["goal"] == "Bob's goal"


def test_two_threads_produce_independent_results() -> None:
    checkpointer = MemorySaver()
    app = build_graph(checkpointer)

    config_a = {"configurable": {"thread_id": "iso-thread-a"}}
    config_b = {"configurable": {"thread_id": "iso-thread-b"}}

    initial: PlanExecuteState = {
        "goal": "Any goal",
        "plan": "",
        "result": "",
        "quality": 0.0,
        "iterations": 0,
        "messages": [],
    }

    with patch("__main__.llm") as mock_llm:
        mock_llm.invoke.side_effect = [
            _make_llm_response("Plan A."),
            _make_llm_response("Result A."),
            _make_llm_response("0.9"),
            _make_llm_response("Plan B."),
            _make_llm_response("Result B."),
            _make_llm_response("0.9"),
        ]
        final_a = app.invoke(initial, config=config_a)
        final_b = app.invoke(initial, config=config_b)

    assert final_a["result"] != final_b["result"]


def test_get_state_history_returns_checkpoints_per_thread() -> None:
    """Each thread must have its own independent checkpoint history."""
    checkpointer = MemorySaver()
    app = build_graph(checkpointer)

    config_a = {"configurable": {"thread_id": "hist-thread-a"}}
    config_b = {"configurable": {"thread_id": "hist-thread-b"}}

    initial: PlanExecuteState = {
        "goal": "Test",
        "plan": "",
        "result": "",
        "quality": 0.0,
        "iterations": 0,
        "messages": [],
    }

    with patch("__main__.llm") as mock_llm:
        mock_llm.invoke.side_effect = [
            # thread-a
            _make_llm_response("Plan A."),
            _make_llm_response("Result A."),
            _make_llm_response("0.9"),
            # thread-b (needs 2 iterations before passing)
            _make_llm_response("Plan B1."),
            _make_llm_response("Result B1."),
            _make_llm_response("0.3"),   # below threshold → retry
            _make_llm_response("Plan B2."),
            _make_llm_response("Result B2."),
            _make_llm_response("0.9"),
        ]
        app.invoke(initial, config=config_a)
        app.invoke(initial, config=config_b)

    history_a = list(app.get_state_history(config_a))
    history_b = list(app.get_state_history(config_b))

    # thread-b ran more iterations so must have more checkpoints
    assert len(history_b) > len(history_a), (
        f"thread-b ({len(history_b)} checkpoints) should have more than "
        f"thread-a ({len(history_a)} checkpoints)"
    )


def test_thread_summary_extracts_correct_fields() -> None:
    checkpointer = MemorySaver()
    app = build_graph(checkpointer)
    config = {"configurable": {"thread_id": "summary-test"}}

    initial: PlanExecuteState = {
        "goal": "Summary test goal",
        "plan": "",
        "result": "",
        "quality": 0.0,
        "iterations": 0,
        "messages": [],
    }

    with patch("__main__.llm") as mock_llm:
        mock_llm.invoke.side_effect = [
            _make_llm_response("My plan."),
            _make_llm_response("My result."),
            _make_llm_response("0.85"),
        ]
        app.invoke(initial, config=config)

    summary = get_thread_summary(app, config)
    assert summary.thread_id == "summary-test"
    assert summary.checkpoint_count >= 1
    assert summary.final_quality == 0.85
    assert summary.final_iterations == 1
    assert summary.final_goal == "Summary test goal"
```

**What changed and why:**
- A single `MemorySaver` instance is shared across both users. Thread isolation is entirely by `thread_id` key — the saver stores checkpoints under `(thread_id, checkpoint_id)` tuples.
- `get_state_history(config)` returns `StateSnapshot` objects in **reverse chronological order** — the first element is the most recent (final) state. Consuming it as a `list` gives the full history; `len(list(...))` gives the checkpoint count.
- `ThreadSummary` is a plain dataclass — a convenient carrier for per-thread stats without adding runtime complexity.
- The test for checkpoint count divergence confirms isolation: thread-b's extra retry cycle produces additional checkpoints that thread-a's history doesn't contain.


## Exercise 4

**Make the cycle limit dynamic: add a `max_iterations: int` field to the state and respect it in the routing function instead of a hardcoded constant.**

Design: add `max_iterations: int` to `PlanExecuteState` (default `3` when passed in initial state). Remove the `MAX_ITERATIONS` module-level constant. Update `quality_router` to read `state["max_iterations"]` instead. Demonstrate the effect by running with `max_iterations=2` vs `max_iterations=5` and comparing iteration counts.

```python
"""
day5_ex4.py — Dynamic cycle limit via max_iterations field in state.

Demonstrates:
  1. max_iterations in TypedDict state (no module constant)
  2. quality_router reads state["max_iterations"] — no global reference
  3. Two runs with max_iterations=2 and max_iterations=5 to compare cycles

Run: python day5_ex4.py
Env: ANTHROPIC_API_KEY
"""
import logging
import operator
import os
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

QUALITY_THRESHOLD = 0.7
# MAX_ITERATIONS is REMOVED — now a state field


# ---------------------------------------------------------------------------
# State — max_iterations is now carried in state, not a module constant
# ---------------------------------------------------------------------------

class PlanExecuteState(TypedDict):
    goal: str
    plan: str
    result: str
    quality: float
    iterations: int
    max_iterations: int          # NEW — dynamic limit, set by the caller
    messages: Annotated[list[str], operator.add]


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

llm = ChatAnthropic(
    model="claude-sonnet-5-5",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def plan_node(state: PlanExecuteState) -> dict:
    iteration = state["iterations"] + 1
    logger.info(
        "plan_node: iteration=%d / max=%d",
        iteration,
        state["max_iterations"],
    )
    context = ""
    if state["result"]:
        context = f"\nPrevious result: {state['result']}\nScore: {state['quality']:.2f}. Improve."
    prompt = f"Goal: {state['goal']}{context}\n\nWrite a concise 3-step plan."
    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "plan": response.content.strip(),
        "iterations": iteration,
        "messages": [f"plan: iteration {iteration}/{state['max_iterations']}"],
    }


def execute_node(state: PlanExecuteState) -> dict:
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
    logger.info("verify_node: scoring result")
    prompt = (
        f"Goal: {state['goal']}\n"
        f"Result:\n{state['result']}\n\n"
        "Rate quality 0.0–1.0. Reply with ONLY a decimal number."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        quality = max(0.0, min(1.0, float(response.content.strip())))
    except ValueError:
        quality = 0.5
    logger.info(
        "verify_node: quality=%.2f threshold=%.2f",
        quality,
        QUALITY_THRESHOLD,
    )
    return {
        "quality": quality,
        "messages": [f"verify: quality={quality:.2f}"],
    }


# ---------------------------------------------------------------------------
# Router — reads max_iterations from state, not from a module constant
# ---------------------------------------------------------------------------

def quality_router(state: PlanExecuteState) -> str:
    """
    Route to 'plan' for retry or END if:
      - quality >= threshold, OR
      - iterations >= state["max_iterations"]  (dynamic limit)
    """
    if state["quality"] >= QUALITY_THRESHOLD:
        logger.info("quality_router: threshold met — routing to END")
        return "done"
    if state["iterations"] >= state["max_iterations"]:
        logger.info(
            "quality_router: reached dynamic limit max_iterations=%d — routing to END",
            state["max_iterations"],
        )
        return "done"
    logger.info("quality_router: quality too low — retrying")
    return "retry"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph(checkpointer: MemorySaver | None = None) -> StateGraph:
    graph = StateGraph(PlanExecuteState)
    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("verify", verify_node)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "verify")
    graph.add_conditional_edges(
        "verify", quality_router, {"retry": "plan", "done": END}
    )
    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_initial_state(goal: str, max_iterations: int) -> PlanExecuteState:
    return {
        "goal": goal,
        "plan": "",
        "result": "",
        "quality": 0.0,
        "iterations": 0,
        "max_iterations": max_iterations,
        "messages": [],
    }


def count_plan_iterations(messages: list[str]) -> int:
    """Count how many 'plan: iteration N/...' entries appear in the message log."""
    return sum(1 for m in messages if m.startswith("plan: iteration"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    goal = "Write a one-paragraph summary of the transformer architecture."

    # --- Run with max_iterations=2 ---
    print("=== Run A: max_iterations=2 ===")
    checkpointer_a = MemorySaver()
    app_a = build_graph(checkpointer_a)
    config_a = {"configurable": {"thread_id": "dynamic-limit-a"}}
    final_a = app_a.invoke(make_initial_state(goal, max_iterations=2), config=config_a)

    iterations_a = count_plan_iterations(final_a["messages"])
    print(f"  Iterations used: {final_a['iterations']} (max allowed: {final_a['max_iterations']})")
    print(f"  Final quality:   {final_a['quality']:.2f}")
    print(f"  Result snippet:  {final_a['result'][:120]}...")

    # --- Run with max_iterations=5 ---
    print("\n=== Run B: max_iterations=5 ===")
    checkpointer_b = MemorySaver()
    app_b = build_graph(checkpointer_b)
    config_b = {"configurable": {"thread_id": "dynamic-limit-b"}}
    final_b = app_b.invoke(make_initial_state(goal, max_iterations=5), config=config_b)

    iterations_b = count_plan_iterations(final_b["messages"])
    print(f"  Iterations used: {final_b['iterations']} (max allowed: {final_b['max_iterations']})")
    print(f"  Final quality:   {final_b['quality']:.2f}")
    print(f"  Result snippet:  {final_b['result'][:120]}...")

    print(f"\nRun A capped at {final_a['max_iterations']}, "
          f"Run B allowed up to {final_b['max_iterations']} — "
          f"iteration counts: A={final_a['iterations']}, B={final_b['iterations']}")


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


def test_router_respects_max_iterations_from_state() -> None:
    """Router must route to 'done' when iterations >= state max_iterations."""
    state: PlanExecuteState = {
        "goal": "Test",
        "plan": "",
        "result": "",
        "quality": 0.0,       # below threshold
        "iterations": 2,       # == max_iterations
        "max_iterations": 2,
        "messages": [],
    }
    assert quality_router(state) == "done"


def test_router_retries_when_below_max_iterations() -> None:
    """Router must retry when quality is low and iterations < max_iterations."""
    state: PlanExecuteState = {
        "goal": "Test",
        "plan": "",
        "result": "",
        "quality": 0.0,       # below threshold
        "iterations": 1,
        "max_iterations": 3,
        "messages": [],
    }
    assert quality_router(state) == "retry"


def test_router_done_on_quality_threshold_regardless_of_max() -> None:
    """If quality >= threshold, router returns 'done' even with iterations remaining."""
    state: PlanExecuteState = {
        "goal": "Test",
        "plan": "",
        "result": "",
        "quality": 0.9,       # above threshold
        "iterations": 1,
        "max_iterations": 10,
        "messages": [],
    }
    assert quality_router(state) == "done"


def test_run_with_max_iterations_2_caps_cycles() -> None:
    """Graph with max_iterations=2 must not exceed 2 plan invocations."""
    app = build_graph()
    # Always score below threshold to force retries — limit is the only stop
    with patch("__main__.llm") as mock_llm:
        mock_llm.invoke.side_effect = [
            # iteration 1
            _make_llm_response("Plan 1."),
            _make_llm_response("Result 1."),
            _make_llm_response("0.1"),   # below threshold
            # iteration 2
            _make_llm_response("Plan 2."),
            _make_llm_response("Result 2."),
            _make_llm_response("0.1"),   # below threshold — but max_iterations=2 reached
        ]
        initial = make_initial_state("Test goal", max_iterations=2)
        final = app.invoke(initial)

    assert final["iterations"] == 2
    assert count_plan_iterations(final["messages"]) == 2


def test_run_with_max_iterations_1_stops_after_first_cycle() -> None:
    """max_iterations=1 must terminate after a single plan/execute/verify cycle."""
    app = build_graph()
    with patch("__main__.llm") as mock_llm:
        mock_llm.invoke.side_effect = [
            _make_llm_response("Plan."),
            _make_llm_response("Result."),
            _make_llm_response("0.1"),   # below threshold — hits max_iterations=1
        ]
        initial = make_initial_state("Test goal", max_iterations=1)
        final = app.invoke(initial)

    assert final["iterations"] == 1


def test_max_iterations_preserved_in_final_state() -> None:
    """The max_iterations value set in initial state must survive to final state unchanged."""
    app = build_graph()
    with patch("__main__.llm") as mock_llm:
        mock_llm.invoke.side_effect = [
            _make_llm_response("Plan."),
            _make_llm_response("Result."),
            _make_llm_response("0.9"),  # above threshold — done on first iteration
        ]
        initial = make_initial_state("Test goal", max_iterations=7)
        final = app.invoke(initial)

    assert final["max_iterations"] == 7, (
        "max_iterations must not be mutated by any node — it is set only in initial state"
    )


def test_dynamic_limit_allows_more_iterations_than_low_limit() -> None:
    """A run with max_iterations=5 may use more cycles than one with max_iterations=2."""
    app = build_graph()
    with patch("__main__.llm") as mock_llm:
        # Low limit run (max=2) — 2 full cycles, quality never passes
        mock_llm.invoke.side_effect = [
            _make_llm_response("Plan 1."),
            _make_llm_response("Result 1."),
            _make_llm_response("0.1"),
            _make_llm_response("Plan 2."),
            _make_llm_response("Result 2."),
            _make_llm_response("0.1"),
        ]
        final_low = app.invoke(make_initial_state("Test", max_iterations=2))

    with patch("__main__.llm") as mock_llm:
        # High limit run (max=5) — 4 cycles before quality passes
        mock_llm.invoke.side_effect = [
            _make_llm_response("Plan 1."),
            _make_llm_response("Result 1."),
            _make_llm_response("0.1"),
            _make_llm_response("Plan 2."),
            _make_llm_response("Result 2."),
            _make_llm_response("0.1"),
            _make_llm_response("Plan 3."),
            _make_llm_response("Result 3."),
            _make_llm_response("0.1"),
            _make_llm_response("Plan 4."),
            _make_llm_response("Result 4."),
            _make_llm_response("0.9"),  # passes on iteration 4
        ]
        final_high = app.invoke(make_initial_state("Test", max_iterations=5))

    assert final_high["iterations"] > final_low["iterations"], (
        "High limit run must have used more iterations than low limit run"
    )
```

**What changed and why:**
- `MAX_ITERATIONS` module constant removed; `max_iterations: int` is a first-class field in `PlanExecuteState`.
- `quality_router` reads `state["max_iterations"]` — the routing function is now stateless with respect to module globals. This makes the graph fully deterministic given only its input state.
- `make_initial_state(goal, max_iterations)` is a factory helper that makes the intent of each invocation explicit.
- No node modifies `max_iterations` — it is set once in the initial state and read by the router. This is the clean LangGraph pattern: configuration lives in state, routing logic is pure.
- Tests verify both the router function directly (unit) and the full graph (integration) with controlled mock responses to force exactly N iterations.
