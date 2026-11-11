# Phase 5 · Week 09 · Day 5 — Message Formats & Task Lifecycle

> Objective: Master A2A message structure, Part types, and the full task lifecycle including streaming and push notifications.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

## Concept

A2A messages carry structured `Parts`, not raw strings. This allows agents to exchange text, files, binary data, and structured JSON in the same message envelope — enabling multi-modal delegation.

**Message structure:**

```json
{
  "role": "user" | "agent",
  "parts": [
    { "type": "text", "text": "Analyze the attached file" },
    { "type": "file", "file": { "name": "q4.csv", "mimeType": "text/csv", "bytes": "<base64>" } },
    { "type": "data", "data": { "format": "json", "value": {"rows": 500} } }
  ]
}
```

**Part types:**
| Type | Use case |
|---|---|
| `text` | Plain or markdown text |
| `file` | Binary files (documents, images, CSVs) via base64 or URI |
| `data` | Structured JSON payload (parameters, configs, results) |

**Task object** (the full envelope):

```json
{
  "id": "task-uuid",
  "sessionId": "session-uuid",
  "status": { "state": "working", "timestamp": "2024-01-15T10:30:00Z" },
  "history": [ <Message list> ],
  "artifacts": [ <Artifact list> ],
  "metadata": {}
}
```

**Artifacts** are the outputs produced by a task — distinct from the conversation `history`. An artifact might be a generated PDF, a dataset, or a code file. This separation matters: history is ephemeral conversation; artifacts are durable outputs.

## How it works internally

**Streaming** uses Server-Sent Events (SSE). The server sends `TaskStatusUpdateEvent` and `TaskArtifactUpdateEvent` frames as work progresses. This lets the client render partial results without polling:

```
data: {"type":"TaskStatusUpdateEvent","taskId":"...","status":{"state":"working"}}

data: {"type":"TaskArtifactUpdateEvent","taskId":"...","artifact":{"name":"draft","parts":[{"type":"text","text":"Here is the first paragraph..."}]}}

data: {"type":"TaskStatusUpdateEvent","taskId":"...","status":{"state":"completed"}}
```

**Push notifications** invert control: instead of the client polling, the server POSTs to a client-provided webhook URL when the task changes state. Useful for long-running tasks (hours) where polling is wasteful.

The client registers a webhook in the task submission:
```json
{
  "pushNotificationConfig": {
    "url": "https://my-agent.example.com/callbacks",
    "authentication": { "schemes": ["bearer"] }
  }
}
```

## Worked example

```python
"""
day_5_message_formats.py
A2A message types, artifact handling, SSE streaming simulation,
and task lifecycle state machine.
"""

import asyncio
import base64
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Part types
# ──────────────────────────────────────────────

class PartType(str, Enum):
    TEXT = "text"
    FILE = "file"
    DATA = "data"


@dataclass
class FilePart:
    type: str = PartType.FILE
    name: str = ""
    mime_type: str = "application/octet-stream"
    bytes_b64: str = ""    # base64-encoded content
    uri: str | None = None  # alternative: URI reference

    def decode(self) -> bytes:
        return base64.b64decode(self.bytes_b64)

    @classmethod
    def from_bytes(cls, name: str, mime_type: str, data: bytes) -> "FilePart":
        return cls(name=name, mime_type=mime_type, bytes_b64=base64.b64encode(data).decode())


@dataclass
class TextPart:
    type: str = PartType.TEXT
    text: str = ""


@dataclass
class DataPart:
    type: str = PartType.DATA
    data: dict[str, Any] = field(default_factory=dict)


Part = TextPart | FilePart | DataPart


@dataclass
class Message:
    role: str   # "user" or "agent"
    parts: list[Part] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


# ──────────────────────────────────────────────
# Artifact
# ──────────────────────────────────────────────

@dataclass
class Artifact:
    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    parts: list[Part] = field(default_factory=list)
    index: int = 0        # position in multi-artifact responses
    last_chunk: bool = True  # streaming: False while chunks arriving


# ──────────────────────────────────────────────
# Task state machine
# ──────────────────────────────────────────────

class TaskState(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL = {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED}

# Valid transitions
TRANSITIONS: dict[TaskState, set[TaskState]] = {
    TaskState.SUBMITTED: {TaskState.WORKING, TaskState.CANCELLED},
    TaskState.WORKING: {TaskState.COMPLETED, TaskState.FAILED, TaskState.INPUT_REQUIRED, TaskState.CANCELLED},
    TaskState.INPUT_REQUIRED: {TaskState.WORKING, TaskState.CANCELLED},
    TaskState.COMPLETED: set(),
    TaskState.FAILED: set(),
    TaskState.CANCELLED: set(),
}


@dataclass
class TaskStatus:
    state: TaskState = TaskState.SUBMITTED
    timestamp: float = field(default_factory=time.time)
    message: str = ""


@dataclass
class Task:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = field(default_factory=TaskStatus)
    history: list[Message] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def transition(self, new_state: TaskState, message: str = "") -> None:
        allowed = TRANSITIONS[self.status.state]
        if new_state not in allowed:
            raise ValueError(
                f"Invalid transition {self.status.state} → {new_state}. "
                f"Allowed: {[s.value for s in allowed]}"
            )
        logger.debug("Task %s: %s → %s", self.id[:8], self.status.state.value, new_state.value)
        self.status = TaskStatus(state=new_state, message=message)

    def add_message(self, msg: Message) -> None:
        self.history.append(msg)

    def add_artifact(self, artifact: Artifact) -> None:
        self.artifacts.append(artifact)


# ──────────────────────────────────────────────
# SSE event types
# ──────────────────────────────────────────────

@dataclass
class TaskStatusUpdateEvent:
    event_type: str = "TaskStatusUpdateEvent"
    task_id: str = ""
    status: TaskStatus = field(default_factory=TaskStatus)
    final: bool = False  # True when no more events will follow


@dataclass
class TaskArtifactUpdateEvent:
    event_type: str = "TaskArtifactUpdateEvent"
    task_id: str = ""
    artifact: Artifact = field(default_factory=Artifact)


SSEEvent = TaskStatusUpdateEvent | TaskArtifactUpdateEvent


# ──────────────────────────────────────────────
# Streaming agent (generates SSE events)
# ──────────────────────────────────────────────

class StreamingDocumentAgent:
    """
    Generates a long document in chunks, emitting SSE events.
    Simulates a real LLM streaming response.
    """

    async def handle(self, task: Task) -> AsyncIterator[SSEEvent]:
        task.transition(TaskState.WORKING)
        yield TaskStatusUpdateEvent(task_id=task.id, status=task.status)

        # Simulate: need clarification if no topic given
        initial_msg = task.history[0] if task.history else None
        topic = ""
        if initial_msg:
            text_parts = [p for p in initial_msg.parts if isinstance(p, TextPart)]
            topic = text_parts[0].text if text_parts else ""

        if not topic:
            task.transition(TaskState.INPUT_REQUIRED, "Please specify the document topic.")
            yield TaskStatusUpdateEvent(task_id=task.id, status=task.status)
            return  # Wait for client to send more messages

        # Generate document in 3 chunks (simulating streaming)
        chunks = [
            f"# Report on {topic}\n\n## Introduction\nThis report covers key aspects of {topic}.",
            f"\n\n## Analysis\nThe primary findings regarding {topic} indicate significant growth.",
            f"\n\n## Conclusion\nBased on our analysis of {topic}, we recommend further investigation.",
        ]

        combined_text = ""
        for i, chunk in enumerate(chunks):
            await asyncio.sleep(0.02)  # Simulate generation delay
            combined_text += chunk
            artifact = Artifact(
                name="report.md",
                description=f"Generated report on {topic}",
                parts=[TextPart(text=combined_text)],
                index=0,
                last_chunk=(i == len(chunks) - 1),
            )
            yield TaskArtifactUpdateEvent(task_id=task.id, artifact=artifact)

        # Add final artifact to task
        final_artifact = Artifact(
            name="report.md",
            parts=[TextPart(text=combined_text)],
            last_chunk=True,
        )
        task.add_artifact(final_artifact)
        task.transition(TaskState.COMPLETED)
        yield TaskStatusUpdateEvent(task_id=task.id, status=task.status, final=True)


# ──────────────────────────────────────────────
# Client consuming the SSE stream
# ──────────────────────────────────────────────

class StreamingClient:
    async def run(self, agent: StreamingDocumentAgent, topic: str) -> str:
        task = Task()
        task.add_message(Message(
            role="user",
            parts=[TextPart(text=topic)],
        ))

        logger.info("[Client] Submitting task for topic: '%s'", topic)
        last_artifact_text = ""

        async for event in agent.handle(task):
            if isinstance(event, TaskStatusUpdateEvent):
                logger.info("[Client] Status → %s", event.status.state.value)
                if event.status.state == TaskState.INPUT_REQUIRED:
                    logger.warning("[Client] Input required: %s", event.status.message)
                    # Respond to clarification
                    task.add_message(Message(
                        role="user",
                        parts=[TextPart(text="Q4 2024 Financial Performance")],
                    ))
                    # Re-run (simplified — real impl resumes the same task)
                    async for sub_event in agent.handle(task):
                        if isinstance(sub_event, TaskArtifactUpdateEvent):
                            last_artifact_text = sub_event.artifact.parts[0].text  # type: ignore[union-attr]
                    break

            elif isinstance(event, TaskArtifactUpdateEvent):
                last_artifact_text = event.artifact.parts[0].text  # type: ignore[union-attr]
                logger.info(
                    "[Client] Artifact chunk received (%d chars, last=%s)",
                    len(last_artifact_text), event.artifact.last_chunk,
                )

        return last_artifact_text


# ──────────────────────────────────────────────
# File attachment demo
# ──────────────────────────────────────────────

def demo_file_attachment() -> None:
    """Show how a client attaches a CSV file to a task message."""
    csv_data = b"date,revenue,units\n2024-01-01,10000,250\n2024-01-02,12000,300\n"
    file_part = FilePart.from_bytes("sales.csv", "text/csv", csv_data)
    msg = Message(
        role="user",
        parts=[
            TextPart(text="Analyze the attached sales data and identify trends."),
            file_part,
        ],
    )
    recovered = file_part.decode()
    assert recovered == csv_data, "Round-trip file encoding failed"
    logger.info("File attachment: %s (%d bytes, base64 len=%d)",
                file_part.name, len(recovered), len(file_part.bytes_b64))


# ──────────────────────────────────────────────
# State machine validation demo
# ──────────────────────────────────────────────

def demo_state_machine() -> None:
    task = Task()
    assert task.status.state == TaskState.SUBMITTED

    task.transition(TaskState.WORKING)
    assert task.status.state == TaskState.WORKING

    try:
        task.transition(TaskState.SUBMITTED)   # Invalid
    except ValueError as exc:
        logger.info("Correctly blocked invalid transition: %s", exc)

    task.transition(TaskState.COMPLETED)
    assert task.status.state == TaskState.COMPLETED
    logger.info("State machine demo passed")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

async def main() -> None:
    demo_file_attachment()
    demo_state_machine()

    agent = StreamingDocumentAgent()
    client = StreamingClient()

    # Normal flow
    result = await client.run(agent, "Cloud Computing Trends 2025")
    logger.info("[Client] Final document length: %d chars", len(result))

    # Empty topic — triggers input-required
    result2 = await client.run(agent, "")
    logger.info("[Client] After clarification, document length: %d chars", len(result2))


if __name__ == "__main__":
    asyncio.run(main())
```

## Exercises

1. Add a `DataPart` handler to `StreamingDocumentAgent`: when the user includes a `DataPart` with `{"word_count": N}`, limit the generated report to approximately N words. Write a test verifying this constraint.

2. Implement a `PushNotificationClient` that registers a webhook URL in task metadata and receives a mock callback when the task completes. Use `asyncio.Queue` to simulate the webhook delivery (no real HTTP needed). Write a test that confirms the callback fires exactly once on completion.

3. Build a `TaskHistory` class wrapping a list of `Message` objects. Add methods: `last_user_message()`, `last_agent_message()`, `messages_since(timestamp)`. Unit-test all three methods.

4. The `TRANSITIONS` dict encodes the valid state machine. Write a function `validate_state_machine(transitions)` that checks: (1) all states are reachable from `SUBMITTED`, (2) all terminal states have no outgoing transitions, (3) no self-loops. Assert it passes for the definition above.

(Solutions: ../solutions/week-09/day-5.md)

## Self-check
- [ ] I can construct a multi-part A2A message with text, file, and data parts
- [ ] I can implement the A2A task state machine with valid-transition enforcement
- [ ] I understand how SSE streaming events (`TaskStatusUpdateEvent`, `TaskArtifactUpdateEvent`) work
- [ ] I can explain the difference between task `history` and task `artifacts`

## Supplementary links
- [A2A message specification](https://a2aproject.github.io/A2A/specification/#message)
- [Server-Sent Events (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
- [Python asyncio generators](https://docs.python.org/3/library/asyncio-task.html#asyncio.ensure_future)
