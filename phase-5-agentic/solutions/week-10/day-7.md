# Week 10 Day 7 — Solutions: Mapping From-Scratch Patterns onto LangGraph

---

## Exercise 1: ReAct Loop as LangGraph

**Goal:** Implement think → act → observe with a conditional edge from act to either observe (tool call) or END (final answer).

### Explanation

ReAct (Reasoning + Acting) alternates between "think" (produce reasoning + decide action) and "act" (execute the action). The conditional edge inspects whether the LLM produced a tool call or a final answer — this is the core routing logic.

```python
import logging
import os
from typing import TypedDict, Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── fake tool for demonstration ───────────────────────────────────────────

def search_web(query: str) -> str:
    """Simulated web search tool."""
    return f"Search results for '{query}': [result 1, result 2, result 3]"

TOOLS = {"search_web": search_web}


class ReActState(TypedDict):
    messages: list[Any]          # full message history
    final_answer: str            # populated when LLM is done


# ── nodes ──────────────────────────────────────────────────────────────────

def think_node(state: ReActState) -> ReActState:
    """
    Think: call LLM with the current message history.
    LLM may return a tool_call (act) or a plain text answer (done).
    """
    logger.info("think_node: invoking LLM")
    llm = ChatAnthropic(
        model="claude-haiku-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    ).bind_tools([
        {
            "name": "search_web",
            "description": "Search the web for current information",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "search query"}},
                "required": ["query"],
            },
        }
    ])
    response: AIMessage = llm.invoke(state["messages"])
    logger.info("think_node: got response, has tool_calls=%s", bool(response.tool_calls))
    return {"messages": state["messages"] + [response]}


def act_node(state: ReActState) -> ReActState:
    """
    Act: execute the tool call from the last AI message.
    Appends a ToolMessage with the result.
    """
    last_message: AIMessage = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    tool_name: str = tool_call["name"]
    tool_args: dict[str, Any] = tool_call["args"]

    logger.info("act_node: calling tool '%s' with args %s", tool_name, tool_args)

    if tool_name not in TOOLS:
        result = f"Error: unknown tool '{tool_name}'"
    else:
        result = TOOLS[tool_name](**tool_args)

    tool_message = ToolMessage(
        content=result,
        tool_call_id=tool_call["id"],
        name=tool_name,
    )
    logger.info("act_node: tool returned '%s'", result[:80])
    return {"messages": state["messages"] + [tool_message]}


def observe_node(state: ReActState) -> ReActState:
    """
    Observe: no-op pass-through — the ToolMessage is already in state.
    In a real system this could parse/validate the tool output.
    """
    logger.info("observe_node: tool result is in messages, looping back to think")
    return {}


def extract_final_answer(state: ReActState) -> ReActState:
    """Extract the final answer from the last AI message."""
    last_message = state["messages"][-1]
    answer = last_message.content if isinstance(last_message.content, str) else str(last_message.content)
    logger.info("Final answer extracted: %s", answer[:80])
    return {"final_answer": answer}


# ── conditional routing ────────────────────────────────────────────────────

def route_from_think(state: ReActState) -> str:
    """Route: tool call → act; plain answer → extract_final_answer."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "act"
    return "extract_final_answer"


# ── graph ─────────────────────────────────────────────────────────────────

def build_react_graph() -> StateGraph:
    builder = StateGraph(ReActState)

    builder.add_node("think", think_node)
    builder.add_node("act", act_node)
    builder.add_node("observe", observe_node)
    builder.add_node("extract_final_answer", extract_final_answer)

    builder.set_entry_point("think")
    builder.add_conditional_edges(
        "think",
        route_from_think,
        {"act": "act", "extract_final_answer": "extract_final_answer"},
    )
    builder.add_edge("act", "observe")
    builder.add_edge("observe", "think")  # loop back
    builder.add_edge("extract_final_answer", END)

    return builder.compile()


if __name__ == "__main__":
    graph = build_react_graph()
    result = graph.invoke({
        "messages": [HumanMessage(content="What are the top 3 Python web frameworks?")],
        "final_answer": "",
    })
    print("Answer:", result["final_answer"])
```

---

## Exercise 2: ReAct with `ToolNode` from `langgraph.prebuilt`

**Goal:** Replace the manual act+observe pair with LangGraph's `ToolNode`.

### Explanation

`ToolNode` is a prebuilt node that automatically: (1) finds tool calls in the last `AIMessage`, (2) executes the matching tools, and (3) appends `ToolMessage` results to the state. This eliminates the manual `act_node` and `observe_node` boilerplate.

```python
import logging
import os
from typing import TypedDict, Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── define tools as LangChain @tool functions ─────────────────────────────

@tool
def search_web(query: str) -> str:
    """Search the web for current information."""
    return f"Search results for '{query}': [result 1, result 2, result 3]"


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    try:
        # NOTE: in production use a proper math parser, not eval
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expression):
            return "Error: invalid characters in expression"
        return str(eval(expression))  # noqa: S307 — restricted to safe chars above
    except Exception as e:
        return f"Error: {e}"


TOOL_LIST = [search_web, calculate]


# ── nodes ──────────────────────────────────────────────────────────────────

def think_node(state: MessagesState) -> MessagesState:
    """LLM reasoning node — bound to tools."""
    logger.info("think_node: invoking LLM")
    llm = ChatAnthropic(
        model="claude-haiku-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    ).bind_tools(TOOL_LIST)
    response = llm.invoke(state["messages"])
    logger.info("think_node: has tool_calls=%s", bool(response.tool_calls))
    return {"messages": [response]}


# ── ToolNode replaces act + observe ───────────────────────────────────────
tool_node = ToolNode(TOOL_LIST)  # handles execution of all registered tools


# ── conditional routing ────────────────────────────────────────────────────

def route_from_think(state: MessagesState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ── graph ─────────────────────────────────────────────────────────────────

def build_react_graph_v2() -> StateGraph:
    builder = StateGraph(MessagesState)

    builder.add_node("think", think_node)
    builder.add_node("tools", tool_node)  # ToolNode handles act+observe

    builder.set_entry_point("think")
    builder.add_conditional_edges(
        "think",
        route_from_think,
        {"tools": "tools", END: END},
    )
    builder.add_edge("tools", "think")  # observe → loop back to think

    return builder.compile()


if __name__ == "__main__":
    graph = build_react_graph_v2()
    result = graph.invoke({
        "messages": [HumanMessage(content="What is 123 * 456? Also search for Python tutorials.")]
    })
    print("Final messages:", [m.content for m in result["messages"]])
```

**Key points:**
- `ToolNode` requires tools defined with `@tool` (not raw dicts) so it can match by name.
- `MessagesState` uses the `add_messages` reducer automatically — no manual list concatenation needed.
- The graph is now 2 nodes instead of 4, with identical behavior.

---

## Exercise 3: Plan-Execute as LangGraph

**Goal:** planner → executor → replanner; replanner routes to executor (more steps) or END (done).

### Explanation

Plan-execute separates strategic planning from step-by-step execution. The replanner checks after each step whether the goal is achieved — this is a more reliable pattern than one-shot planning because the agent can adapt.

```python
import logging
import os
from typing import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlanExecuteState(TypedDict):
    task: str
    plan: list[str]
    current_step: int
    results: list[str]
    final_answer: str


# ── nodes ──────────────────────────────────────────────────────────────────

def planner_node(state: PlanExecuteState) -> PlanExecuteState:
    """Call LLM to produce an ordered list of steps for the task."""
    logger.info("planner_node: planning task '%s'", state["task"])
    llm = ChatAnthropic(
        model="claude-sonnet-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    response = llm.invoke([
        SystemMessage(content=(
            "You are a planning assistant. Given a task, produce a numbered list of "
            "concrete steps to complete it. Be specific and actionable. "
            "Format: '1. step one\\n2. step two\\n...'"
        )),
        HumanMessage(content=f"Task: {state['task']}"),
    ])
    raw_plan = response.content
    # Parse numbered list into individual steps
    steps = [
        line.split(". ", 1)[1].strip()
        for line in raw_plan.strip().splitlines()
        if line.strip() and line[0].isdigit()
    ]
    if not steps:
        steps = [raw_plan.strip()]  # fallback: treat whole response as single step
    logger.info("planner_node: created %d steps", len(steps))
    return {"plan": steps, "current_step": 0, "results": []}


def executor_node(state: PlanExecuteState) -> PlanExecuteState:
    """Execute the current plan step using the LLM (or tools)."""
    step_idx = state["current_step"]
    if step_idx >= len(state["plan"]):
        logger.warning("executor_node: no more steps")
        return {}

    current_step = state["plan"][step_idx]
    logger.info("executor_node: executing step %d: '%s'", step_idx + 1, current_step)

    llm = ChatAnthropic(
        model="claude-haiku-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    context = "\n".join(
        f"Step {i+1} result: {r}" for i, r in enumerate(state["results"])
    )
    prompt = (
        f"Task: {state['task']}\n\n"
        f"Previous results:\n{context or 'None'}\n\n"
        f"Current step to execute: {current_step}\n\n"
        "Execute this step and provide the result."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    result = response.content
    logger.info("executor_node: step result: %s", result[:80])

    new_results = list(state["results"]) + [result]
    return {
        "results": new_results,
        "current_step": step_idx + 1,
    }


def replanner_node(state: PlanExecuteState) -> PlanExecuteState:
    """
    Assess progress and decide: continue executing or declare done.
    Returns final_answer if done; otherwise leaves it empty.
    """
    logger.info("replanner_node: assessing progress at step %d/%d", state["current_step"], len(state["plan"]))
    llm = ChatAnthropic(
        model="claude-sonnet-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    steps_done = "\n".join(
        f"Step {i+1}: {s}\nResult: {r}"
        for i, (s, r) in enumerate(zip(state["plan"], state["results"]))
    )
    remaining = state["plan"][state["current_step"]:]
    response = llm.invoke([
        SystemMessage(content=(
            "You assess task completion. Reply with exactly:\n"
            "'DONE: <final answer>' if the task is complete, or\n"
            "'CONTINUE' if more steps are needed."
        )),
        HumanMessage(content=(
            f"Task: {state['task']}\n\n"
            f"Steps completed:\n{steps_done}\n\n"
            f"Remaining steps: {remaining}\n\n"
            "Is the task done?"
        )),
    ])
    answer = response.content.strip()
    if answer.startswith("DONE:"):
        final = answer[5:].strip()
        logger.info("replanner_node: DONE — '%s'", final[:80])
        return {"final_answer": final}
    logger.info("replanner_node: CONTINUE")
    return {"final_answer": ""}


# ── routing ────────────────────────────────────────────────────────────────

def route_replanner(state: PlanExecuteState) -> str:
    if state.get("final_answer"):
        return END
    if state["current_step"] >= len(state["plan"]):
        # Ran out of steps — force end
        logger.warning("route_replanner: no steps left, forcing END")
        return END
    return "executor"


# ── graph ─────────────────────────────────────────────────────────────────

def build_plan_execute_graph() -> StateGraph:
    builder = StateGraph(PlanExecuteState)

    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("replanner", replanner_node)

    builder.set_entry_point("planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "replanner")
    builder.add_conditional_edges(
        "replanner",
        route_replanner,
        {"executor": "executor", END: END},
    )

    return builder.compile()


if __name__ == "__main__":
    graph = build_plan_execute_graph()
    result = graph.invoke({
        "task": "Research the key differences between REST and GraphQL APIs",
        "plan": [], "current_step": 0, "results": [], "final_answer": "",
    })
    print("Final Answer:", result["final_answer"])
```

---

## Exercise 4: 2-Agent Supervisor Pattern

**Goal:** Supervisor node routes to either "researcher" or "writer" subgraph nodes based on `state.next_agent`.

### Explanation

The supervisor pattern uses a central routing node that delegates to specialized worker nodes. The supervisor maintains control flow — workers don't talk to each other directly, they return to the supervisor. This mirrors a manager-employee hierarchy.

```python
import logging
import os
from typing import TypedDict, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SupervisorState(TypedDict):
    task: str
    research_notes: str
    written_content: str
    next_agent: Literal["researcher", "writer", "FINISH"]
    final_output: str


# ── supervisor ────────────────────────────────────────────────────────────

def supervisor_node(state: SupervisorState) -> SupervisorState:
    """
    Decides which agent to call next, or declares FINISH.
    Examines current state to determine what's needed.
    """
    logger.info("supervisor_node: deciding next agent")
    llm = ChatAnthropic(
        model="claude-sonnet-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    has_research = bool(state.get("research_notes"))
    has_content = bool(state.get("written_content"))

    prompt = (
        f"Task: {state['task']}\n"
        f"Research notes available: {has_research}\n"
        f"Written content available: {has_content}\n\n"
        "Which agent should work next? Reply with exactly one word:\n"
        "- 'researcher' if we need to gather information first\n"
        "- 'writer' if we have enough research and need to write\n"
        "- 'FINISH' if the task is complete"
    )
    response = llm.invoke([
        SystemMessage(content="You are a supervisor. Respond with exactly one word: researcher, writer, or FINISH."),
        HumanMessage(content=prompt),
    ])
    next_agent = response.content.strip().lower()
    if next_agent not in {"researcher", "writer", "finish"}:
        next_agent = "researcher"  # safe fallback
    next_agent = next_agent if next_agent != "finish" else "FINISH"
    logger.info("supervisor_node: next_agent='%s'", next_agent)
    return {"next_agent": next_agent}


# ── worker nodes ───────────────────────────────────────────────────────────

def researcher_node(state: SupervisorState) -> SupervisorState:
    """Research agent: gathers information for the task."""
    logger.info("researcher_node: researching task '%s'", state["task"])
    llm = ChatAnthropic(
        model="claude-haiku-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    response = llm.invoke([
        SystemMessage(content="You are a researcher. Gather key facts and insights for the given task."),
        HumanMessage(content=f"Research task: {state['task']}"),
    ])
    logger.info("researcher_node: completed research")
    return {"research_notes": response.content}


def writer_node(state: SupervisorState) -> SupervisorState:
    """Writer agent: produces final written content from research."""
    logger.info("writer_node: writing based on research notes")
    llm = ChatAnthropic(
        model="claude-haiku-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    response = llm.invoke([
        SystemMessage(content="You are a writer. Produce polished content based on research notes."),
        HumanMessage(content=f"Task: {state['task']}\n\nResearch notes:\n{state.get('research_notes', '')}"),
    ])
    logger.info("writer_node: writing complete")
    return {"written_content": response.content, "final_output": response.content}


# ── routing ────────────────────────────────────────────────────────────────

def route_supervisor(state: SupervisorState) -> str:
    next_agent = state.get("next_agent", "researcher")
    if next_agent == "FINISH":
        return END
    return next_agent  # "researcher" or "writer"


# ── graph ─────────────────────────────────────────────────────────────────

def build_supervisor_graph() -> StateGraph:
    builder = StateGraph(SupervisorState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("writer", writer_node)

    builder.set_entry_point("supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {"researcher": "researcher", "writer": "writer", END: END},
    )
    # Both workers return to supervisor for the next routing decision
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("writer", "supervisor")

    return builder.compile()


if __name__ == "__main__":
    graph = build_supervisor_graph()
    result = graph.invoke({
        "task": "Write a short blog post about the benefits of type hints in Python",
        "research_notes": "", "written_content": "", "next_agent": "researcher", "final_output": "",
    })
    print("Final output:", result["final_output"][:200])
```

---

## Exercise 5: Add `MessagesState` and `add_messages` Reducer

**Goal:** Use `MessagesState` as the base for the supervisor state so all messages accumulate automatically.

### Explanation

`MessagesState` is a prebuilt TypedDict with `messages: Annotated[list[AnyMessage], add_messages]`. The `add_messages` reducer handles deduplication and ordering — you don't need to manually append. Subclassing it adds custom fields while inheriting the message accumulation behavior.

```python
import logging
import os
from typing import TypedDict, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, END

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Extend MessagesState with our custom fields
class SupervisorStateV2(MessagesState):
    # MessagesState already provides: messages: Annotated[list[AnyMessage], add_messages]
    task: str
    research_notes: str
    written_content: str
    next_agent: Literal["researcher", "writer", "FINISH"]
    final_output: str


def supervisor_node_v2(state: SupervisorStateV2) -> SupervisorStateV2:
    logger.info("supervisor_node_v2: %d messages in history", len(state["messages"]))
    llm = ChatAnthropic(
        model="claude-sonnet-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    has_research = bool(state.get("research_notes"))
    has_content = bool(state.get("written_content"))

    # Add supervisor's reasoning to the message history
    supervisor_message = HumanMessage(
        content=(
            f"Supervisor check — research_ready={has_research}, content_ready={has_content}. "
            "Next: researcher/writer/FINISH?"
        ),
        name="supervisor",
    )
    response: AIMessage = llm.invoke([
        SystemMessage(content="Reply with exactly: researcher, writer, or FINISH."),
        supervisor_message,
    ])
    next_agent = response.content.strip().lower()
    next_agent = next_agent if next_agent != "finish" else "FINISH"
    logger.info("supervisor: next='%s'", next_agent)

    # add_messages reducer will append these without duplicating
    return {
        "messages": [supervisor_message, response],  # type: ignore[list-item]
        "next_agent": next_agent,
    }


def researcher_node_v2(state: SupervisorStateV2) -> SupervisorStateV2:
    logger.info("researcher_node_v2: running")
    llm = ChatAnthropic(
        model="claude-haiku-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    research_request = HumanMessage(content=f"Research: {state['task']}", name="researcher")
    response = llm.invoke([
        SystemMessage(content="You are a researcher. Return key facts."),
        research_request,
    ])
    return {
        "messages": [research_request, response],  # type: ignore[list-item]
        "research_notes": response.content,
    }


def writer_node_v2(state: SupervisorStateV2) -> SupervisorStateV2:
    logger.info("writer_node_v2: running, %d messages so far", len(state["messages"]))
    llm = ChatAnthropic(
        model="claude-haiku-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    write_request = HumanMessage(
        content=f"Write content for: {state['task']}\nResearch: {state.get('research_notes', '')}",
        name="writer",
    )
    response = llm.invoke([
        SystemMessage(content="You are a writer. Produce polished output."),
        write_request,
    ])
    return {
        "messages": [write_request, response],  # type: ignore[list-item]
        "written_content": response.content,
        "final_output": response.content,
    }


def route_supervisor_v2(state: SupervisorStateV2) -> str:
    next_agent = state.get("next_agent", "researcher")
    return END if next_agent == "FINISH" else next_agent


def build_supervisor_graph_v2() -> StateGraph:
    builder = StateGraph(SupervisorStateV2)
    builder.add_node("supervisor", supervisor_node_v2)
    builder.add_node("researcher", researcher_node_v2)
    builder.add_node("writer", writer_node_v2)
    builder.set_entry_point("supervisor")
    builder.add_conditional_edges("supervisor", route_supervisor_v2,
                                   {"researcher": "researcher", "writer": "writer", END: END})
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("writer", "supervisor")
    return builder.compile()
```

**Key points:**
- `MessagesState` is a `TypedDict` with `messages: Annotated[list[AnyMessage], add_messages]` — subclass it to add fields.
- The `add_messages` reducer deduplicates by message ID and handles ordering — return `{"messages": [new_msg]}`, not the full accumulated list.
- All agent messages end up in one chronological history, enabling full conversation replay.

---

## Exercise 6: Add `MemorySaver` Checkpointing and Demonstrate Resume

**Goal:** Add checkpointing, show that a conversation can be paused and resumed across two separate `.invoke()` calls.

### Explanation

`MemorySaver` stores the full graph state after each node execution, keyed by `thread_id`. A second `.invoke()` with the same `thread_id` resumes exactly where it left off — the graph re-reads the checkpoint and continues from the last completed node.

```python
import logging
import os
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SupervisorStateV3(MessagesState):
    task: str
    research_notes: str
    written_content: str
    next_agent: Literal["researcher", "writer", "FINISH"]
    final_output: str
    approved: bool


# ── nodes (same logic as v2, abbreviated) ────────────────────────────────

def supervisor_node_v3(state: SupervisorStateV3) -> SupervisorStateV3:
    logger.info("supervisor_node_v3: %d messages", len(state["messages"]))
    llm = ChatAnthropic(
        model="claude-sonnet-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    has_research = bool(state.get("research_notes"))
    has_content = bool(state.get("written_content"))
    response: AIMessage = llm.invoke([
        SystemMessage(content="Reply with exactly: researcher, writer, or FINISH."),
        HumanMessage(content=f"research_ready={has_research}, content_ready={has_content}. Next?"),
    ])
    next_agent = response.content.strip().lower()
    next_agent = "FINISH" if next_agent == "finish" else next_agent
    return {"messages": [response], "next_agent": next_agent}  # type: ignore[list-item]


def researcher_node_v3(state: SupervisorStateV3) -> SupervisorStateV3:
    logger.info("researcher_node_v3: researching")
    llm = ChatAnthropic(model="claude-haiku-5-5", api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = llm.invoke([
        SystemMessage(content="Return key facts as bullet points."),
        HumanMessage(content=f"Research: {state['task']}"),
    ])
    return {"messages": [response], "research_notes": response.content}  # type: ignore[list-item]


def writer_node_v3(state: SupervisorStateV3) -> SupervisorStateV3:
    logger.info("writer_node_v3: writing")
    llm = ChatAnthropic(model="claude-haiku-5-5", api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = llm.invoke([
        SystemMessage(content="Write polished content from the research notes."),
        HumanMessage(content=f"Task: {state['task']}\nNotes: {state.get('research_notes', '')}"),
    ])
    return {
        "messages": [response],  # type: ignore[list-item]
        "written_content": response.content,
        "final_output": response.content,
    }


def human_approval_node(state: SupervisorStateV3) -> SupervisorStateV3:
    """HITL node: pause for human to approve the final output."""
    logger.info("human_approval_node: pausing for approval")
    decision = interrupt({
        "question": "Approve this output? (yes/no)",
        "output": state.get("final_output", ""),
    })
    approved = str(decision).strip().lower() in {"yes", "y", "approve", "true"}
    logger.info("human_approval_node: approved=%s", approved)
    return {"approved": approved}


def route_supervisor_v3(state: SupervisorStateV3) -> str:
    next_agent = state.get("next_agent", "researcher")
    if next_agent == "FINISH":
        return "human_approval"
    return next_agent


def build_supervisor_graph_v3() -> StateGraph:
    memory = MemorySaver()  # in-memory checkpoint store

    builder = StateGraph(SupervisorStateV3)
    builder.add_node("supervisor", supervisor_node_v3)
    builder.add_node("researcher", researcher_node_v3)
    builder.add_node("writer", writer_node_v3)
    builder.add_node("human_approval", human_approval_node)

    builder.set_entry_point("supervisor")
    builder.add_conditional_edges(
        "supervisor", route_supervisor_v3,
        {"researcher": "researcher", "writer": "writer", "human_approval": "human_approval"},
    )
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("writer", "supervisor")
    builder.add_edge("human_approval", END)

    return builder.compile(
        checkpointer=memory,
        interrupt_before=["human_approval"],  # pause before human approval
    )


# ── demonstration: save and resume across two separate invocations ─────────

def demo_checkpoint_resume() -> None:
    graph = build_supervisor_graph_v3()
    thread_config = {"configurable": {"thread_id": "supervisor-session-42"}}

    initial_state: SupervisorStateV3 = {
        "messages": [HumanMessage(content="Write a blog post about Python type hints")],
        "task": "Write a blog post about Python type hints",
        "research_notes": "",
        "written_content": "",
        "next_agent": "researcher",
        "final_output": "",
        "approved": False,
    }

    # ── First invocation: runs until interrupt at human_approval ──────────
    logger.info("=== First invocation ===")
    state_after_first = graph.invoke(initial_state, config=thread_config)
    logger.info(
        "Paused. Messages so far: %d, final_output length: %d",
        len(state_after_first.get("messages", [])),
        len(state_after_first.get("final_output", "")),
    )
    print("Content preview:", state_after_first.get("final_output", "")[:200])

    # ── Second invocation: resume with human approval ─────────────────────
    # The graph reads the checkpoint from MemorySaver using the same thread_id.
    # Execution continues from where it paused (human_approval node).
    logger.info("=== Second invocation (resume) ===")
    final_state = graph.invoke(
        Command(resume="yes"),
        config=thread_config,
    )
    logger.info("Resume complete. approved=%s", final_state.get("approved"))
    print("Final approved output:", final_state.get("final_output", "")[:300])


if __name__ == "__main__":
    demo_checkpoint_resume()
```

**Key points:**
- `MemorySaver` stores checkpoints in memory keyed by `(thread_id, checkpoint_id)`. The same `thread_id` in config links both `.invoke()` calls.
- The graph automatically reads the latest checkpoint on the second call — you don't need to pass the state again, only `Command(resume=...)`.
- In production, replace `MemorySaver` with `SqliteSaver` or `PostgresSaver` so state persists across process restarts.
- The full message history (`MessagesState`) is preserved in the checkpoint, so the second invocation "remembers" everything from the first.
