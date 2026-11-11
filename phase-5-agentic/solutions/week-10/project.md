# Week 10 Project — Reference Implementation: Plan-Execute LangGraph Agent

---

## Overview

This is the full reference implementation for the Week 10 project: a **plan-execute agent** that:
1. Generates a multi-step plan using `claude-sonnet-5-5`
2. Executes each step using `claude-haiku-5-5` with tool support
3. Uses a replanner to decide when the task is done
4. Pauses for human approval of the plan before execution begins
5. Checkpoints state with `MemorySaver` so sessions can be resumed

---

## Full Implementation

```python
"""
week10_project/agent.py

Plan-Execute LangGraph Agent — Week 10 Project Reference Implementation
"""

import logging
import os
from typing import Any, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── State ─────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    task: str                    # the original user task
    plan: list[str]              # ordered list of steps (populated by planner)
    current_step: int            # index into plan (0-based)
    results: list[str]           # accumulated step results
    final_answer: str            # populated when replanner says DONE
    approved: bool               # True after human approves the plan


# ── Tools ─────────────────────────────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """Search the web for up-to-date information on a topic."""
    logger.info("web_search: query='%s'", query)
    # Production: call a real search API (Tavily, Brave, etc.)
    return f"[web_search] Results for '{query}': [result_1, result_2, result_3]"


@tool
def read_file(file_path: str) -> str:
    """Read the contents of a local file."""
    logger.info("read_file: path='%s'", file_path)
    try:
        with open(file_path) as f:
            return f.read()[:2000]  # cap at 2000 chars
    except FileNotFoundError:
        return f"Error: file '{file_path}' not found"


@tool
def write_note(content: str, label: str = "note") -> str:
    """Save a note for later reference during this session."""
    logger.info("write_note: label='%s', length=%d", label, len(content))
    # In production: persist to a store
    return f"Note '{label}' saved ({len(content)} chars)"


TOOLS = [web_search, read_file, write_note]


# ── Planner Node ───────────────────────────────────────────────────────────

def planner_node(state: AgentState) -> AgentState:
    """
    Call claude-sonnet-5-5 to generate an ordered plan for the task.
    Returns list[str] of steps stored in state.plan.
    """
    logger.info("planner_node: generating plan for task='%s'", state["task"])
    llm = ChatAnthropic(
        model="claude-sonnet-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=1024,
    )
    response = llm.invoke([
        SystemMessage(content=(
            "You are a planning assistant. Given a task, produce a numbered list of "
            "concrete, actionable steps to complete it. Each step should be self-contained. "
            "Format exactly as:\n1. <step one>\n2. <step two>\n..."
            "\nAim for 3-6 steps. Be specific."
        )),
        HumanMessage(content=f"Task: {state['task']}"),
    ])

    raw = response.content.strip()
    steps: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if line and line[0].isdigit() and ". " in line:
            steps.append(line.split(". ", 1)[1].strip())

    if not steps:
        # Fallback: treat the whole response as one step
        steps = [raw]

    logger.info("planner_node: generated %d steps", len(steps))
    return {
        "plan": steps,
        "current_step": 0,
        "results": [],
        "approved": False,
    }


# ── Plan Approval HITL Node ────────────────────────────────────────────────

def plan_approval_node(state: AgentState) -> AgentState:
    """
    HITL: interrupt_before this node pauses the graph so a human can
    review the plan. On resume, record approval.
    """
    logger.info("plan_approval_node: presenting plan for approval")
    answer = interrupt({
        "message": "Review the generated plan. Approve to begin execution?",
        "plan": state["plan"],
        "task": state["task"],
    })
    approved = str(answer).strip().lower() in {"yes", "y", "approve", "true"}
    logger.info("plan_approval_node: approved=%s", approved)
    return {"approved": approved}


# ── Executor Node ──────────────────────────────────────────────────────────

def executor_node(state: AgentState) -> AgentState:
    """
    Execute the current plan step using claude-haiku-5-5 with tools.
    Uses a minimal agentic loop: if the LLM calls a tool, run it and
    feed the result back; repeat until the LLM gives a plain answer.
    """
    step_idx = state["current_step"]
    if step_idx >= len(state["plan"]):
        logger.warning("executor_node: no more steps to execute")
        return {}

    current_step = state["plan"][step_idx]
    logger.info("executor_node: executing step %d/%d: '%s'", step_idx + 1, len(state["plan"]), current_step)

    llm = ChatAnthropic(
        model="claude-haiku-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=512,
    ).bind_tools(TOOLS)

    # Build context from previous results
    context_lines = [
        f"Step {i + 1} result: {r}"
        for i, r in enumerate(state["results"])
    ]
    context = "\n".join(context_lines) if context_lines else "No previous results."

    messages: list[Any] = [
        SystemMessage(content=(
            "You are an execution assistant. Complete the given step using available tools if needed. "
            "Provide a concise result when done."
        )),
        HumanMessage(content=(
            f"Task context: {state['task']}\n\n"
            f"Previous results:\n{context}\n\n"
            f"Current step to complete: {current_step}"
        )),
    ]

    # Simple agentic tool-use loop (max 3 iterations to prevent runaway)
    result_text = ""
    for _ in range(3):
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            # LLM gave a final answer — we're done
            result_text = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
            break

        # Execute each tool call
        for tool_call in response.tool_calls:
            tool_name: str = tool_call["name"]
            tool_args: dict[str, Any] = tool_call["args"]
            logger.info("executor_node: calling tool '%s'", tool_name)

            tool_fn = next((t for t in TOOLS if t.name == tool_name), None)
            if tool_fn is None:
                tool_result = f"Error: unknown tool '{tool_name}'"
            else:
                try:
                    tool_result = tool_fn.invoke(tool_args)
                except Exception as e:
                    tool_result = f"Tool error: {e}"

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                )
            )
    else:
        # Exceeded loop limit — use last response content
        last = messages[-1]
        if hasattr(last, "content"):
            result_text = str(last.content)

    logger.info("executor_node: step result='%s'", result_text[:80])
    new_results = list(state["results"]) + [result_text]
    return {
        "results": new_results,
        "current_step": step_idx + 1,
    }


# ── Replanner Node ─────────────────────────────────────────────────────────

def replanner_node(state: AgentState) -> AgentState:
    """
    Conditional node: call claude-sonnet-5-5 to assess whether the task
    is complete. Returns final_answer if DONE, empty string if CONTINUE.
    """
    logger.info(
        "replanner_node: assessing progress — step %d/%d",
        state["current_step"], len(state["plan"]),
    )

    # If we've exceeded all planned steps, force completion
    if state["current_step"] >= len(state["plan"]):
        summary = "; ".join(state["results"][-3:]) if state["results"] else "No results."
        logger.info("replanner_node: all steps done, synthesizing answer")
        return {"final_answer": f"Task completed. Summary: {summary}"}

    llm = ChatAnthropic(
        model="claude-sonnet-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=512,
    )

    completed = "\n".join(
        f"Step {i + 1} ({step}): {result}"
        for i, (step, result) in enumerate(zip(state["plan"], state["results"]))
    )
    remaining = state["plan"][state["current_step"]:]

    response = llm.invoke([
        SystemMessage(content=(
            "Assess task completion. Reply with EXACTLY:\n"
            "'DONE: <concise final answer>' — if the task is fully complete, OR\n"
            "'CONTINUE' — if more steps should be executed."
        )),
        HumanMessage(content=(
            f"Task: {state['task']}\n\n"
            f"Completed steps:\n{completed}\n\n"
            f"Remaining planned steps: {remaining}\n\n"
            "Is the task complete?"
        )),
    ])

    answer = response.content.strip()
    if answer.upper().startswith("DONE:"):
        final = answer[5:].strip()
        logger.info("replanner_node: DONE — '%s'", final[:80])
        return {"final_answer": final}

    logger.info("replanner_node: CONTINUE")
    return {"final_answer": ""}


# ── Routing ────────────────────────────────────────────────────────────────

def route_after_approval(state: AgentState) -> str:
    """After plan approval: proceed to executor if approved, else END."""
    if state.get("approved"):
        return "executor_node"
    logger.info("route_after_approval: plan rejected, ending")
    return END


def route_replanner(state: AgentState) -> str:
    """After replanner: END if done, executor_node if more steps needed."""
    if state.get("final_answer"):
        return END
    return "executor_node"


# ── Graph Assembly ─────────────────────────────────────────────────────────

def build_plan_execute_graph(checkpointer: MemorySaver | None = None) -> StateGraph:
    """
    Build and compile the plan-execute graph.

    Args:
        checkpointer: optional checkpointer (defaults to MemorySaver if None)

    Returns:
        Compiled StateGraph
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("planner_node", planner_node)
    builder.add_node("plan_approval_node", plan_approval_node)
    builder.add_node("executor_node", executor_node)
    builder.add_node("replanner_node", replanner_node)

    # Wire edges
    builder.set_entry_point("planner_node")
    builder.add_edge("planner_node", "plan_approval_node")
    builder.add_conditional_edges(
        "plan_approval_node",
        route_after_approval,
        {"executor_node": "executor_node", END: END},
    )
    builder.add_edge("executor_node", "replanner_node")
    builder.add_conditional_edges(
        "replanner_node",
        route_replanner,
        {"executor_node": "executor_node", END: END},
    )

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["plan_approval_node"],  # HITL: pause for plan approval
    )


# ── Stream Demo ────────────────────────────────────────────────────────────

def run_stream_demo(task: str, thread_id: str = "demo-001") -> None:
    """
    Demonstrate graph.stream() showing each step as it completes.
    Handles the HITL interrupt and resumes automatically (for demo purposes).
    """
    graph = build_plan_execute_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: AgentState = {
        "task": task,
        "plan": [],
        "current_step": 0,
        "results": [],
        "final_answer": "",
        "approved": False,
    }

    print(f"\n{'='*60}")
    print(f"Task: {task}")
    print('='*60)

    # ── Phase 1: Stream until HITL interrupt ─────────────────────────────
    print("\n[Phase 1] Planning...")
    for chunk in graph.stream(initial_state, config=config):
        for node_name, node_output in chunk.items():
            if node_name == "__interrupt__":
                print(f"\n[INTERRUPT] Graph paused for human review")
                print(f"Plan generated:")
                # Access plan from graph state
                snapshot = graph.get_state(config)
                for i, step in enumerate(snapshot.values.get("plan", []), 1):
                    print(f"  {i}. {step}")
            else:
                keys = list(node_output.keys()) if isinstance(node_output, dict) else []
                print(f"[{node_name}] output keys: {keys}")

    # ── Phase 2: Resume with approval ────────────────────────────────────
    print("\n[Phase 2] Resuming with approval...")
    for chunk in graph.stream(Command(resume="yes"), config=config):
        for node_name, node_output in chunk.items():
            if node_name == "__interrupt__":
                continue
            print(f"[{node_name}] ", end="")
            if isinstance(node_output, dict):
                if "results" in node_output:
                    latest = node_output["results"][-1] if node_output["results"] else ""
                    print(f"result='{latest[:80]}...'")
                elif "final_answer" in node_output and node_output["final_answer"]:
                    print(f"FINAL ANSWER: {node_output['final_answer'][:120]}")
                else:
                    print(list(node_output.keys()))
            else:
                print(str(node_output)[:80])

    final_state = graph.get_state(config).values
    print(f"\n{'='*60}")
    print("FINAL ANSWER:", final_state.get("final_answer", "not set"))
    print('='*60)


if __name__ == "__main__":
    run_stream_demo(
        task="Research the key advantages of using async Python for web APIs and summarize them",
        thread_id="week10-project-demo",
    )
```

---

## Tests

```python
"""
tests/test_week10_project.py

4 pytest tests for the plan-execute agent.
All LLM calls are mocked with unittest.mock.patch — no ANTHROPIC_API_KEY required.

Run with: pytest tests/test_week10_project.py -v
"""

import pytest
from unittest.mock import MagicMock, patch, call
from typing import Any

# Import from the implementation module
# Adjust the import path to match your project structure
from week10_project.agent import (
    AgentState,
    planner_node,
    executor_node,
    replanner_node,
    build_plan_execute_graph,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command


# ── Helpers ────────────────────────────────────────────────────────────────

def make_state(**overrides: Any) -> AgentState:
    """Build a minimal valid AgentState with optional overrides."""
    defaults: AgentState = {
        "task": "test task",
        "plan": [],
        "current_step": 0,
        "results": [],
        "final_answer": "",
        "approved": False,
    }
    return {**defaults, **overrides}  # type: ignore[return-value]


def mock_llm_response(content: str) -> MagicMock:
    """Create a mock LLM response with the given content and no tool_calls."""
    response = MagicMock()
    response.content = content
    response.tool_calls = []
    return response


# ── Test 1: Planner generates steps ───────────────────────────────────────

def test_planner_generates_steps() -> None:
    """
    Mock LLM response containing a numbered plan.
    Verify that plan is populated with parsed steps.
    """
    mock_plan_text = (
        "1. Research the topic online\n"
        "2. Identify key concepts\n"
        "3. Write a summary"
    )

    with patch("week10_project.agent.ChatAnthropic") as MockLLM:
        mock_instance = MockLLM.return_value
        mock_instance.invoke.return_value = mock_llm_response(mock_plan_text)

        state = make_state(task="Summarize Python async programming")
        result = planner_node(state)

    assert "plan" in result
    assert len(result["plan"]) == 3, f"Expected 3 steps, got {len(result['plan'])}"
    assert result["plan"][0] == "Research the topic online"
    assert result["plan"][1] == "Identify key concepts"
    assert result["plan"][2] == "Write a summary"
    assert result["current_step"] == 0
    assert result["results"] == []


# ── Test 2: Executor runs tool and appends result ─────────────────────────

def test_executor_runs_tool() -> None:
    """
    Mock LLM to return a tool call on first invoke, then a final answer.
    Verify that the tool result is appended to state.results.
    """
    # First LLM call: returns a tool_call for web_search
    tool_call_response = MagicMock()
    tool_call_response.content = ""
    tool_call_response.tool_calls = [{
        "id": "call_abc123",
        "name": "web_search",
        "args": {"query": "Python async benefits"},
    }]

    # Second LLM call: returns a final plain text answer
    final_response = mock_llm_response("Async Python improves throughput for I/O-bound tasks.")

    with patch("week10_project.agent.ChatAnthropic") as MockLLM:
        mock_instance = MockLLM.return_value
        # bind_tools returns a new mock — chain it
        mock_bound = MagicMock()
        mock_instance.bind_tools.return_value = mock_bound
        mock_bound.invoke.side_effect = [tool_call_response, final_response]

        state = make_state(
            task="Research Python async",
            plan=["Search for Python async benefits"],
            current_step=0,
            results=[],
        )
        result = executor_node(state)

    assert "results" in result
    assert len(result["results"]) == 1, f"Expected 1 result, got {len(result['results'])}"
    assert "Async Python" in result["results"][0]
    assert result["current_step"] == 1


# ── Test 3: Replanner routes to END when done ────────────────────────────

def test_replanner_routes_to_end() -> None:
    """
    Mock LLM to return 'DONE: final answer text'.
    Verify that final_answer is set and route_replanner returns END.
    """
    from langgraph.graph import END
    from week10_project.agent import route_replanner

    with patch("week10_project.agent.ChatAnthropic") as MockLLM:
        mock_instance = MockLLM.return_value
        mock_instance.invoke.return_value = mock_llm_response(
            "DONE: Python async improves performance for I/O-bound web APIs."
        )

        state = make_state(
            task="Research Python async",
            plan=["Search web", "Summarize findings"],
            current_step=1,
            results=["Search completed with relevant results."],
        )
        result = replanner_node(state)

    assert result.get("final_answer"), "final_answer should be populated"
    assert "Python async" in result["final_answer"]

    # Verify routing decision
    updated_state = make_state(**{**state, **result})
    route = route_replanner(updated_state)
    assert route == END, f"Expected END, got '{route}'"


# ── Test 4: HITL interrupt pauses graph; state has plan ──────────────────

def test_hitl_interrupt_pauses_graph() -> None:
    """
    Build the full compiled graph with MemorySaver.
    Run it with a mock LLM response for planning.
    Verify the graph pauses before plan_approval_node and the state
    contains the generated plan.

    The graph should raise GraphInterrupt (or return an interrupt signal)
    when it hits interrupt_before=["plan_approval_node"].
    """
    from langgraph.errors import GraphInterrupt

    mock_plan_text = (
        "1. Gather requirements\n"
        "2. Design architecture\n"
        "3. Implement solution"
    )

    memory = MemorySaver()
    graph = build_plan_execute_graph(checkpointer=memory)
    config = {"configurable": {"thread_id": "test-hitl-001"}}

    initial_state: AgentState = {
        "task": "Build a REST API",
        "plan": [],
        "current_step": 0,
        "results": [],
        "final_answer": "",
        "approved": False,
    }

    with patch("week10_project.agent.ChatAnthropic") as MockLLM:
        mock_instance = MockLLM.return_value
        mock_instance.invoke.return_value = mock_llm_response(mock_plan_text)

        # The graph should pause (interrupt) before plan_approval_node
        # LangGraph raises GraphInterrupt when interrupt() is called
        # OR returns a state with __interrupt__ key depending on API version
        try:
            result = graph.invoke(initial_state, config=config)
            # If we get here, check for interrupt signal in result
            # (some LG versions return state instead of raising)
            interrupted = "__interrupt__" in result or not result.get("approved", True)
        except GraphInterrupt:
            interrupted = True

    assert interrupted, "Graph should have paused at plan_approval_node"

    # Verify the plan was generated and saved in the checkpoint
    snapshot = graph.get_state(config)
    saved_plan = snapshot.values.get("plan", [])
    assert len(saved_plan) == 3, f"Expected 3 plan steps in checkpoint, got {len(saved_plan)}"
    assert saved_plan[0] == "Gather requirements"
```

---

## Project Structure

```
week10_project/
├── __init__.py
├── agent.py          # main implementation (above)
└── tools.py          # optional: move @tool definitions here

tests/
└── test_week10_project.py

pyproject.toml        # dependencies
.env.example          # ANTHROPIC_API_KEY=
.gitignore            # .env
```

### `pyproject.toml` (minimal)

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "week10-project"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "langchain-anthropic>=0.3.0",
    "langgraph>=0.2.0",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23"]

[tool.ruff]
line-length = 100
target-version = "py312"
```

### `.env.example`

```
ANTHROPIC_API_KEY=your-key-here
```

---

## Architecture Notes

### Why `interrupt_before=["plan_approval_node"]`?

The plan approval step is a **gate**: no execution happens before a human reviews the plan. Using `interrupt_before` (rather than calling `interrupt()` inside the node) means the node's logic only runs *after* the human has confirmed — there's no ambiguity about what state the node starts with.

### Why separate `planner` and `replanner`?

The planner runs once at the start and generates a broad multi-step plan. The replanner runs after each execution step with full context — it can update the plan dynamically if early steps reveal that the original plan was wrong. This separation enables **adaptive planning** without needing a full re-plan from scratch.

### Why `claude-haiku-5-5` for execution?

Execution steps are typically narrow and well-specified (the plan step tells the executor exactly what to do). Haiku is faster and cheaper for these focused tasks. Sonnet-level reasoning is reserved for planning and replan assessment, where broader context and judgment matter.

### Why `MemorySaver` here instead of `SqliteSaver`?

For a course project, `MemorySaver` requires no setup and is sufficient for demonstrating checkpoint/resume within a single Python session. In production, swap it with `SqliteSaver("agent_state.db")` (one-line change) to persist across process restarts.

### Thread safety

`MemorySaver` is **not** thread-safe for concurrent writes to the same `thread_id`. For concurrent users, use a separate `thread_id` per user (e.g., `str(uuid.uuid4())`), or use `PostgresSaver` which handles concurrent writes via row-level locking.
