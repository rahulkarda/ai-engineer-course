# Week 10 Day 6 — Solutions: HITL, Streaming, Framework vs DIY

---

## Exercise 1: Multi-Turn HITL with Two Interrupts

**Goal:** After human edits feedback, interrupt again to confirm "Is this final?" before continuing.

### Explanation

LangGraph's `interrupt_before` accepts a list of node names. By adding a second node — `confirm_final` — between the human-edit step and the final send step, you get a second pause point. The graph resumes when you call `.invoke()` with `Command(resume=...)` again.

```python
import logging
import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReviewState(TypedDict):
    code: str
    feedback: str
    confirmed: bool
    final_feedback: str


# ── nodes ────────────────────────────────────────────────────────────────────

def generate_feedback(state: ReviewState) -> ReviewState:
    """LLM generates initial code-review feedback."""
    logger.info("Generating LLM feedback for code")
    # In production: call claude-sonnet-5-5 here
    feedback = f"Suggested improvements for:\n{state['code']}"
    return {"feedback": feedback}


def human_edit(state: ReviewState) -> ReviewState:
    """
    First interrupt: human can edit the feedback.
    interrupt() suspends execution; resume value becomes the new feedback.
    """
    logger.info("Pausing for human edit of feedback")
    edited = interrupt(
        {
            "question": "Please review and edit the feedback below.",
            "current_feedback": state["feedback"],
        }
    )
    logger.info("Received edited feedback from human")
    return {"feedback": edited}


def confirm_final(state: ReviewState) -> ReviewState:
    """
    Second interrupt: ask 'Is this final?'
    Human must reply True/False (or yes/no string).
    """
    logger.info("Pausing for final confirmation")
    answer = interrupt(
        {
            "question": "Is this feedback final?",
            "feedback": state["feedback"],
        }
    )
    # Accept bool or yes/no string
    if isinstance(answer, bool):
        confirmed = answer
    else:
        confirmed = str(answer).strip().lower() in {"yes", "y", "true", "1"}
    logger.info("Confirmation received: %s", confirmed)
    return {"confirmed": confirmed, "final_feedback": state["feedback"] if confirmed else ""}


def send_feedback(state: ReviewState) -> ReviewState:
    """Deliver confirmed feedback (e.g. post to PR)."""
    if state.get("confirmed"):
        logger.info("Sending final feedback: %s", state["final_feedback"])
    else:
        logger.info("Feedback not confirmed — discarding")
    return {}


# ── conditional routing ────────────────────────────────────────────────────

def route_after_confirm(state: ReviewState) -> str:
    if state.get("confirmed"):
        return "send_feedback"
    # Not confirmed → loop back for another human edit
    return "human_edit"


# ── graph wiring ───────────────────────────────────────────────────────────

def build_code_review_graph() -> StateGraph:
    builder = StateGraph(ReviewState)

    builder.add_node("generate_feedback", generate_feedback)
    builder.add_node("human_edit", human_edit)
    builder.add_node("confirm_final", confirm_final)
    builder.add_node("send_feedback", send_feedback)

    builder.set_entry_point("generate_feedback")
    builder.add_edge("generate_feedback", "human_edit")
    builder.add_edge("human_edit", "confirm_final")
    builder.add_conditional_edges(
        "confirm_final",
        route_after_confirm,
        {"send_feedback": "send_feedback", "human_edit": "human_edit"},
    )
    builder.add_edge("send_feedback", END)

    return builder.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["human_edit", "confirm_final"],
    )


# ── demo run ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    graph = build_code_review_graph()
    config = {"configurable": {"thread_id": "review-001"}}

    # Step 1: run until first interrupt (human_edit)
    state = graph.invoke(
        {"code": "def add(a, b): return a+b", "feedback": "", "confirmed": False, "final_feedback": ""},
        config=config,
    )
    logger.info("Graph paused at human_edit. Current state: %s", state)

    # Step 2: human provides edited feedback → resumes, pauses at confirm_final
    state = graph.invoke(
        Command(resume="Rename parameters to x and y for clarity."),
        config=config,
    )
    logger.info("Graph paused at confirm_final. State: %s", state)

    # Step 3: human confirms → graph runs to END
    state = graph.invoke(
        Command(resume=True),
        config=config,
    )
    logger.info("Final state: %s", state)
```

**Key points:**
- Both `human_edit` and `confirm_final` call `interrupt()` internally — these are the actual suspension points.
- `interrupt_before=["human_edit", "confirm_final"]` ensures the graph pauses *before* entering each node on each pass.
- The conditional edge after `confirm_final` loops back to `human_edit` if the human says "no", enabling unlimited revision cycles.
- Each `.invoke(Command(resume=...))` continues from the exact checkpoint.

---

## Exercise 2: Token-Level Streaming with `astream_events()`

**Goal:** Filter `astream_events()` to `on_chat_model_stream` events and reconstruct the full response.

### Explanation

`astream_events()` emits a structured stream of typed events. The `on_chat_model_stream` event carries `AIMessageChunk` objects. Accumulating their `.content` fields reconstructs the full response without buffering the whole message.

```python
import asyncio
import logging
import os
from typing import AsyncIterator

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessageChunk
from langgraph.graph import StateGraph, MessagesState, END

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_simple_chat_graph():
    """Minimal one-node graph to demonstrate token streaming."""
    llm = ChatAnthropic(
        model="claude-sonnet-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        streaming=True,
    )

    def chat_node(state: MessagesState) -> MessagesState:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("chat", chat_node)
    builder.set_entry_point("chat")
    builder.add_edge("chat", END)
    return builder.compile()


async def stream_tokens(prompt: str) -> str:
    """
    Stream a prompt through the graph, printing each token as it arrives.
    Returns the fully reconstructed response string.
    """
    graph = build_simple_chat_graph()
    initial_state = {"messages": [HumanMessage(content=prompt)]}

    full_response_parts: list[str] = []

    async for event in graph.astream_events(initial_state, version="v2"):
        event_type: str = event["event"]

        # Filter to token-level chat model stream events only
        if event_type != "on_chat_model_stream":
            continue

        chunk: AIMessageChunk = event["data"]["chunk"]

        # content may be a string or a list of content blocks
        if isinstance(chunk.content, str):
            token = chunk.content
        elif isinstance(chunk.content, list):
            # Anthropic returns list of dicts with "type"/"text" keys
            token = "".join(
                block.get("text", "")
                for block in chunk.content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        else:
            token = ""

        if token:
            print(token, end="", flush=True)  # live display
            full_response_parts.append(token)

    print()  # newline after streaming completes
    full_response = "".join(full_response_parts)
    logger.info("Reconstructed response length: %d chars", len(full_response))
    return full_response


# ── alternative: standalone LLM streaming (no graph) ─────────────────────

async def stream_llm_direct(prompt: str) -> str:
    """
    Direct LLM token streaming without a graph wrapper.
    Useful when you don't need graph orchestration.
    """
    llm = ChatAnthropic(
        model="claude-sonnet-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        streaming=True,
    )

    full_response_parts: list[str] = []

    async for event in llm.astream_events(
        [HumanMessage(content=prompt)],
        version="v2",
    ):
        if event["event"] == "on_chat_model_stream":
            chunk: AIMessageChunk = event["data"]["chunk"]
            if isinstance(chunk.content, str):
                token = chunk.content
            else:
                token = ""
            if token:
                print(token, end="", flush=True)
                full_response_parts.append(token)

    print()
    return "".join(full_response_parts)


if __name__ == "__main__":
    response = asyncio.run(stream_tokens("Explain async generators in Python in 3 sentences."))
    logger.info("Full response: %s", response)
```

**Key points:**
- `version="v2"` is required for the current `astream_events()` API.
- Filter on `event["event"] == "on_chat_model_stream"` — all other event types (`on_chain_start`, `on_tool_start`, etc.) are skipped.
- Anthropic's streaming returns `content` as a list of typed blocks; always handle both `str` and `list[dict]` formats.
- The reconstructed string equals what a non-streaming `.invoke()` would return.

---

## Exercise 3: Framework Decision Function

**Goal:** Given graph complexity signals, return a reasoned framework recommendation.

### Explanation

The decision logic encodes real engineering heuristics: LangGraph shines for cyclic, stateful, HITL workflows; raw LCEL is better for linear chains; a DIY loop is the right choice for teams unfamiliar with LangGraph who have simple needs.

```python
import logging
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FrameworkChoice(str, Enum):
    LANGGRAPH = "use LangGraph"
    LCEL = "use raw LCEL"
    DIY_LOOP = "DIY loop"


@dataclass(frozen=True)
class FrameworkDecision:
    choice: FrameworkChoice
    reasoning: str
    confidence: float  # 0.0 – 1.0


def decide_framework(
    num_nodes: int,
    has_cycles: bool,
    hitl_needed: bool,
    team_langgraph_experience: int,  # 0–10
) -> FrameworkDecision:
    """
    Recommend a framework based on workflow complexity signals.

    Args:
        num_nodes: Number of distinct processing steps/nodes.
        has_cycles: Whether the workflow needs any loops or retries.
        hitl_needed: Whether human approval or editing is required.
        team_langgraph_experience: Team's self-rated LG experience (0=none, 10=expert).

    Returns:
        FrameworkDecision with choice and reasoning.
    """
    if not 0 <= team_langgraph_experience <= 10:
        raise ValueError("team_langgraph_experience must be in [0, 10]")

    logger.info(
        "Evaluating: nodes=%d, cycles=%s, hitl=%s, experience=%d",
        num_nodes, has_cycles, hitl_needed, team_langgraph_experience,
    )

    reasons: list[str] = []
    langgraph_score = 0
    lcel_score = 0
    diy_score = 0

    # ── signal: cycles ─────────────────────────────────────────────────────
    if has_cycles:
        langgraph_score += 3
        reasons.append("cycles/loops present → LangGraph manages state across iterations")
    else:
        lcel_score += 1
        diy_score += 1

    # ── signal: HITL ──────────────────────────────────────────────────────
    if hitl_needed:
        langgraph_score += 3
        reasons.append("HITL required → LangGraph interrupt/resume is purpose-built for this")
    else:
        lcel_score += 1

    # ── signal: node count ────────────────────────────────────────────────
    if num_nodes >= 5:
        langgraph_score += 2
        reasons.append(f"{num_nodes} nodes → LangGraph's explicit graph wiring prevents spaghetti")
    elif num_nodes <= 3:
        lcel_score += 2
        diy_score += 1
        reasons.append(f"only {num_nodes} nodes → LCEL chain or simple loop is sufficient")

    # ── signal: team experience ───────────────────────────────────────────
    if team_langgraph_experience < 3:
        diy_score += 2
        lcel_score += 1
        reasons.append(
            f"low LG experience ({team_langgraph_experience}/10) → "
            "DIY loop avoids framework learning curve for simple cases"
        )
    elif team_langgraph_experience >= 7:
        langgraph_score += 1
        reasons.append(f"high LG experience ({team_langgraph_experience}/10) → team can leverage full LG feature set")

    # ── decision ───────────────────────────────────────────────────────────
    scores = {
        FrameworkChoice.LANGGRAPH: langgraph_score,
        FrameworkChoice.LCEL: lcel_score,
        FrameworkChoice.DIY_LOOP: diy_score,
    }
    total = sum(scores.values()) or 1  # avoid div-by-zero
    choice = max(scores, key=lambda k: scores[k])
    confidence = round(scores[choice] / total, 2)

    reasoning = "; ".join(reasons) + f". Scores: LG={langgraph_score}, LCEL={lcel_score}, DIY={diy_score}."

    logger.info("Decision: %s (confidence=%.2f)", choice.value, confidence)
    return FrameworkDecision(choice=choice, reasoning=reasoning, confidence=confidence)


# ── examples ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cases = [
        dict(num_nodes=7, has_cycles=True,  hitl_needed=True,  team_langgraph_experience=8),
        dict(num_nodes=2, has_cycles=False, hitl_needed=False, team_langgraph_experience=4),
        dict(num_nodes=3, has_cycles=False, hitl_needed=False, team_langgraph_experience=1),
        dict(num_nodes=5, has_cycles=True,  hitl_needed=False, team_langgraph_experience=2),
    ]
    for case in cases:
        result = decide_framework(**case)
        print(f"\nInput:  {case}")
        print(f"Choice: {result.choice.value}")
        print(f"Reason: {result.reasoning}")
        print(f"Conf:   {result.confidence}")
```

**Key points:**
- The function is pure — no side effects, fully testable.
- A score-based system with explicit weights makes the reasoning transparent.
- `FrameworkDecision` is a frozen dataclass: immutable, hashable, safe to pass between layers.
- Cycles and HITL are the strongest signals for LangGraph; low experience is the strongest signal for DIY.

---

## Exercise 4: Async Nodes with `astream()`

**Goal:** Convert sync code-review graph to use `async def` nodes and `astream()`.

### Explanation

`async def` nodes let you use `await` for I/O-bound calls (LLM, DB, HTTP) inside nodes without blocking the event loop. `astream()` is the async analogue of `stream()` — it yields state diffs as an async generator.

```python
import asyncio
import logging
import os
from typing import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeReviewState(TypedDict):
    code: str
    feedback: str
    approved: bool
    revision_count: int


# ── async nodes ────────────────────────────────────────────────────────────

async def async_generate_feedback(state: CodeReviewState) -> CodeReviewState:
    """Async node: awaits LLM call to generate code review feedback."""
    logger.info("Generating feedback (async), revision #%d", state.get("revision_count", 0))

    llm = ChatAnthropic(
        model="claude-sonnet-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    prompt = f"Review this code and provide concise feedback:\n\n```python\n{state['code']}\n```"
    response = await llm.ainvoke([HumanMessage(content=prompt)])

    logger.info("Feedback generated")
    return {
        "feedback": response.content,
        "revision_count": state.get("revision_count", 0) + 1,
    }


async def async_human_review(state: CodeReviewState) -> CodeReviewState:
    """Async HITL node: suspends for human approval."""
    logger.info("Pausing for human review (async)")
    decision = interrupt(
        {
            "feedback": state["feedback"],
            "question": "Approve this feedback? (yes/no or edited feedback)",
        }
    )
    approved = str(decision).strip().lower() in {"yes", "y", "approve", "true"}
    logger.info("Human decision: approved=%s", approved)
    return {"approved": approved, "feedback": decision if not approved else state["feedback"]}


async def async_send_review(state: CodeReviewState) -> CodeReviewState:
    """Async node: posts approved feedback (e.g. to GitHub PR API)."""
    logger.info("Sending approved feedback")
    # await github_api.post_review(state["feedback"])  # production
    return {}


def route_after_human(state: CodeReviewState) -> str:
    if state.get("approved"):
        return "send_review"
    if state.get("revision_count", 0) >= 3:
        logger.warning("Max revisions reached — forcing send")
        return "send_review"
    return "generate_feedback"  # regenerate with human's edits


# ── graph ─────────────────────────────────────────────────────────────────

def build_async_review_graph():
    builder = StateGraph(CodeReviewState)

    builder.add_node("generate_feedback", async_generate_feedback)
    builder.add_node("human_review", async_human_review)
    builder.add_node("send_review", async_send_review)

    builder.set_entry_point("generate_feedback")
    builder.add_edge("generate_feedback", "human_review")
    builder.add_conditional_edges(
        "human_review",
        route_after_human,
        {
            "send_review": "send_review",
            "generate_feedback": "generate_feedback",
        },
    )
    builder.add_edge("send_review", END)

    return builder.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["human_review"],
    )


# ── async streaming demo ───────────────────────────────────────────────────

async def run_async_review(code: str) -> None:
    graph = build_async_review_graph()
    config = {"configurable": {"thread_id": "async-review-001"}}

    initial_state: CodeReviewState = {
        "code": code,
        "feedback": "",
        "approved": False,
        "revision_count": 0,
    }

    logger.info("Starting async code review stream")

    # Stream until first interrupt (human_review node)
    async for chunk in graph.astream(initial_state, config=config):
        logger.info("Stream chunk: %s", list(chunk.keys()))
        for node_name, node_output in chunk.items():
            print(f"[{node_name}] output keys: {list(node_output.keys())}")

    logger.info("Graph paused — waiting for human input")

    # Resume with approval
    async for chunk in graph.astream(
        Command(resume="yes"),
        config=config,
    ):
        logger.info("Resume chunk: %s", list(chunk.keys()))

    logger.info("Async review complete")


if __name__ == "__main__":
    sample_code = """
def calculate_average(numbers):
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers)
"""
    asyncio.run(run_async_review(sample_code))
```

**Key points:**
- Every node is `async def` — this allows `await llm.ainvoke()`, `await db.fetch()`, etc. inside nodes.
- `astream()` returns an `AsyncGenerator` — iterate with `async for`.
- `interrupt()` works identically in async nodes; LangGraph handles the suspension/resume plumbing.
- The sync graph (`stream()`) and async graph (`astream()`) share the same `StateGraph` definition — only the compile and iteration differ.
- `MemorySaver` is thread-safe and works with both sync and async.
