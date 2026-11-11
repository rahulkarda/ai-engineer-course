# Phase 5 · Week 09 · Day 2 — The A2A Protocol Concept

> Objective: Understand what the A2A protocol is, how agents discover and call each other, and the design principles behind it.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

## Concept

**A2A (Agent-to-Agent)** is an open protocol specification (originally proposed by Google, now community-governed) that defines how AI agents from different vendors, frameworks, and organizations can discover and collaborate with each other. Think of it as HTTP for agent cooperation.

**Core design principles:**

1. **Opaque agents** — An A2A server does not expose its internal reasoning, tools, or model. It exposes only what it *can do* and how to ask it to do things. Callers don't need to know if the agent uses Claude, GPT-4, or a deterministic rule engine.

2. **Long-running tasks first** — Unlike REST (designed for sub-second responses), A2A assumes tasks may take seconds to hours. The protocol has built-in states, streaming, and push notifications.

3. **Structured communication** — Messages carry typed `Parts` (text, files, structured data) rather than raw strings, enabling rich multi-modal exchanges.

4. **HTTP-based transport** — A2A runs over HTTPS, meaning existing infrastructure (load balancers, API gateways, auth proxies) works without modification.

**Key actors:**
- **A2A Client** — the agent (or orchestrator) that wants to delegate work
- **A2A Server** — the agent offering a service
- **Agent Card** — a JSON document (served at `/.well-known/agent.json`) that describes what the server can do

## How it works internally

A minimal A2A exchange has four steps:

```
Client                        Server
  |                              |
  |  GET /.well-known/agent.json |  1. Discovery
  |<-----------------------------|
  |                              |
  |  POST /tasks/send            |  2. Task submission
  |  { "task_id": ...,           |
  |    "message": {...} }        |
  |------------------------------>|
  |  { "task_id": ...,           |
  |    "status": "submitted" }   |  3. Ack
  |<-----------------------------|
  |                              |
  |  GET /tasks/{id}             |  4. Poll / stream
  |  or SSE stream               |
  |<-----------------------------|
```

The server is free to spawn sub-agents, call tools, run long chains—the client doesn't know or care. Only the *interface* is standardized.

**Task lifecycle states:**
```
submitted → working → (input-required) → completed
                    ↘ failed
                    ↘ cancelled
```

`input-required` is a critical state: the server is blocked waiting for more information from the client (e.g., a clarifying question). This enables interactive, multi-turn delegation without breaking the protocol.

## Worked example

```python
"""
day_2_a2a_concept.py
Pure-Python simulation of the A2A exchange pattern.
No external libraries required. Demonstrates discovery, submission, lifecycle.
"""

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Protocol types
# ──────────────────────────────────────────────

class TaskStatus(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TextPart:
    type: str = "text"
    text: str = ""


@dataclass
class Message:
    role: str  # "user" or "agent"
    parts: list[TextPart] = field(default_factory=list)


@dataclass
class Task:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = TaskStatus.SUBMITTED
    messages: list[Message] = field(default_factory=list)
    result: Message | None = None
    error: str | None = None


@dataclass
class AgentCard:
    """Served at /.well-known/agent.json"""
    name: str
    description: str
    version: str
    capabilities: list[str]
    url: str
    auth_required: bool = False


# ──────────────────────────────────────────────
# Simulated A2A server (the "remote" agent)
# ──────────────────────────────────────────────

class SummarizationAgent:
    """A minimal A2A-compliant server implementation."""

    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="summarization-agent",
            description="Summarizes documents and extracts key points",
            version="1.0.0",
            capabilities=["text-summarization", "key-point-extraction", "sentiment-analysis"],
            url="https://agents.example.com/summarizer",
        )

    def submit_task(self, message: Message) -> Task:
        task = Task(messages=[message])
        task.status = TaskStatus.WORKING
        logger.info("[Server] Task %s submitted, now working", task.task_id[:8])
        return task

    def process_task(self, task: Task) -> Task:
        """Simulate task processing (would call LLM in production)."""
        input_text = task.messages[0].parts[0].text if task.messages else ""

        # Simulate: if text is very short, ask for more context
        if len(input_text) < 20:
            task.status = TaskStatus.INPUT_REQUIRED
            task.result = Message(
                role="agent",
                parts=[TextPart(text="Please provide more context — the input is too short to summarize meaningfully.")],
            )
            logger.info("[Server] Task %s needs more input", task.task_id[:8])
            return task

        # Normal completion
        summary = f"Summary of '{input_text[:40]}...': Key themes identified, 3 main points extracted."
        task.status = TaskStatus.COMPLETED
        task.result = Message(
            role="agent",
            parts=[TextPart(text=summary)],
        )
        logger.info("[Server] Task %s completed", task.task_id[:8])
        return task


# ──────────────────────────────────────────────
# Simulated A2A client (the "calling" agent)
# ──────────────────────────────────────────────

class OrchestratorAgent:
    """Discovers and delegates to A2A servers."""

    def __init__(self) -> None:
        self._servers: dict[str, SummarizationAgent] = {}

    def discover(self, server_url: str, server: SummarizationAgent) -> AgentCard:
        """Step 1: Fetch agent card (GET /.well-known/agent.json)."""
        card = server.get_agent_card()
        self._servers[server_url] = server
        logger.info("[Client] Discovered '%s' at %s — capabilities: %s",
                    card.name, card.url, card.capabilities)
        return card

    def delegate(self, server_url: str, text: str) -> str:
        """Steps 2–4: Submit task, poll until done."""
        server = self._servers[server_url]
        msg = Message(role="user", parts=[TextPart(text=text)])

        # Step 2: Submit
        task = server.submit_task(msg)
        logger.info("[Client] Submitted task %s", task.task_id[:8])

        # Step 3 & 4: Process and poll
        task = server.process_task(task)

        if task.status == TaskStatus.INPUT_REQUIRED:
            # Send additional context
            logger.info("[Client] Agent needs more input, sending follow-up")
            follow_up = Message(
                role="user",
                parts=[TextPart(text=text + " — This is a detailed quarterly financial report.")],
            )
            task.messages.append(follow_up)
            # Resubmit with full context
            full_task = server.submit_task(follow_up)
            full_task = server.process_task(full_task)
            task = full_task

        if task.status == TaskStatus.COMPLETED and task.result:
            return task.result.parts[0].text
        else:
            raise RuntimeError(f"Task ended in unexpected state: {task.status}")


# ──────────────────────────────────────────────
# Main demonstration
# ──────────────────────────────────────────────

def main() -> None:
    orchestrator = OrchestratorAgent()
    summarizer = SummarizationAgent()

    # Discovery
    card = orchestrator.discover("https://agents.example.com/summarizer", summarizer)
    assert "text-summarization" in card.capabilities

    # Normal delegation
    result = orchestrator.delegate(
        "https://agents.example.com/summarizer",
        "This is a comprehensive annual report covering financial performance, "
        "strategic initiatives, risk factors, and market outlook for fiscal year 2024.",
    )
    logger.info("[Client] Final result: %s", result)

    # Short input — triggers input-required flow
    result2 = orchestrator.delegate(
        "https://agents.example.com/summarizer",
        "Q4 report",  # Too short
    )
    logger.info("[Client] Final result (short input): %s", result2)


if __name__ == "__main__":
    main()
```

## Exercises

1. Add a `DataAnalysisAgent` A2A server with capabilities `["csv-analysis", "chart-generation", "statistical-summary"]`. Give it an agent card and a `process_task` that returns mock results. Connect it to `OrchestratorAgent` and write a pytest test for the discovery + delegation flow.

2. Implement a `TaskStore` (in-memory dict keyed by `task_id`) that persists task state across calls. Add a `get_task(task_id)` method so the client can poll status separately from result retrieval.

3. Extend `TaskStatus` with a `CANCELLATION_REQUESTED` intermediate state and add a `cancel_task(task_id)` method to the server. The server should check this state before doing expensive work. Write a test that cancels a task mid-flight.

4. A2A uses `input-required` for multi-turn interactions. Design a `ConversationBuffer` that holds the full message history for a task and allows the server to read all prior messages when resuming after `input-required`.

(Solutions: ../solutions/week-09/day-2.md)

## Self-check
- [ ] I can describe the four steps of an A2A exchange from memory
- [ ] I can list all six task lifecycle states and explain when `input-required` triggers
- [ ] I can implement a minimal A2A client-server pair in Python
- [ ] I understand why A2A uses HTTP rather than a custom binary protocol

## Supplementary links
- [A2A Protocol specification](https://a2aproject.github.io/A2A/specification/)
- [A2A GitHub repository](https://github.com/a2aproject/A2A)
- [Google's original A2A announcement blog post](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)
