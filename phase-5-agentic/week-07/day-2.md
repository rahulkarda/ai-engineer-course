# Phase 5 · Week 07 · Day 2 — Agent Communication & Message Passing

> Objective: Design and implement typed inter-agent messaging with routing, acknowledgement, and error propagation.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

## Concept

Agents in a multi-agent system cannot call each other directly as functions — that creates tight coupling. Instead they **pass messages**: serialisable, typed envelopes that travel through a bus or queue. This is the same principle as microservices talking over HTTP or a message broker.

### Message anatomy

```
┌─────────────────────────────────────────┐
│ AgentMessage                             │
│  id             : UUID (unique)         │
│  correlation_id : UUID (links req→resp) │
│  sender         : str                   │
│  receiver       : str (or "broadcast")  │
│  message_type   : MessageType enum      │
│  payload        : BaseModel subclass    │
│  timestamp      : datetime              │
│  priority       : int (0=low, 9=high)   │
│  ttl            : int (hops remaining)  │
└─────────────────────────────────────────┘
```

### Message types

| Type | Direction | Semantics |
|---|---|---|
| `REQUEST` | sender → receiver | "do this task, return result" |
| `RESPONSE` | receiver → sender | "here is the result" |
| `BROADCAST` | sender → all | "FYI to every agent" |
| `ERROR` | any → any | "something went wrong" |
| `HEARTBEAT` | agent → bus | "I am alive" |

### Message bus patterns

**Direct channel**: one-to-one queue per pair. Simplest, lowest latency, O(n²) queues.

**Topic bus**: agents subscribe to topics. O(n) queues. Decoupled — sender doesn't know receivers.

**Mailbox**: each agent has one inbox queue. Sender puts message in receiver's inbox. Clean ownership. This is what we implement today.

### Routing strategies

1. **Name-based**: `receiver="researcher"` → deliver to agent named "researcher"
2. **Role-based**: `receiver_role="analyst"` → any agent with that role can pick up
3. **Capability-based**: `required_capability="code_execution"` → first free agent with that cap
4. **Priority queue**: high-priority messages jump the queue

## How it works internally

A `MessageBus` is essentially a dict of `{agent_name: asyncio.Queue}`. When an agent wants to send:
1. Bus validates the message (schema, TTL, known receiver)
2. Bus puts message in receiver's queue
3. Receiver's `run_loop` dequeues and dispatches

For synchronous code we use `queue.Queue`. For async we use `asyncio.Queue`. The interface is the same; choose based on your I/O pattern.

**Acknowledgement**: after a REQUEST, the sender either blocks (sync RPC) or registers a callback keyed on `correlation_id`. When the RESPONSE arrives with a matching `correlation_id`, the promise resolves.

**Error propagation**: if a worker fails, it sends an ERROR message with `original_id` pointing to the request. The supervisor catches it via the same correlation map.

## Worked example

```python
"""
day2_messaging.py — In-process message bus with mailbox pattern.
"""
import logging
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Message schema ─────────────────────────────────────────────────────────────

class MessageType(str, Enum):
    REQUEST   = "REQUEST"
    RESPONSE  = "RESPONSE"
    BROADCAST = "BROADCAST"
    ERROR     = "ERROR"
    HEARTBEAT = "HEARTBEAT"


class AgentMessage(BaseModel):
    id:             str = Field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str = ""
    sender:         str
    receiver:       str
    message_type:   MessageType
    payload:        dict[str, Any]
    timestamp:      datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    priority:       int = 0
    ttl:            int = 10  # max hops


# ── Message bus ────────────────────────────────────────────────────────────────

class MessageBus:
    """Simple mailbox bus. Thread-safe."""

    def __init__(self) -> None:
        self._mailboxes: dict[str, queue.PriorityQueue[tuple[int, AgentMessage]]] = {}
        self._lock = threading.Lock()

    def register(self, agent_name: str) -> None:
        with self._lock:
            if agent_name not in self._mailboxes:
                self._mailboxes[agent_name] = queue.PriorityQueue()
                logger.info("bus | registered agent: %s", agent_name)

    def send(self, message: AgentMessage) -> None:
        if message.ttl <= 0:
            logger.warning("bus | TTL expired, dropping message %s", message.id)
            return
        message = message.model_copy(update={"ttl": message.ttl - 1})

        if message.receiver == "broadcast":
            with self._lock:
                targets = list(self._mailboxes.keys())
            for name in targets:
                if name != message.sender:
                    self._mailboxes[name].put((-message.priority, message))
            logger.info("bus | broadcast from %s to %d agents", message.sender, len(targets) - 1)
        else:
            with self._lock:
                box = self._mailboxes.get(message.receiver)
            if box is None:
                raise ValueError(f"Unknown receiver: {message.receiver}")
            box.put((-message.priority, message))
            logger.debug("bus | routed %s → %s", message.sender, message.receiver)

    def receive(self, agent_name: str, timeout: float = 1.0) -> AgentMessage | None:
        with self._lock:
            box = self._mailboxes.get(agent_name)
        if box is None:
            raise ValueError(f"Unknown agent: {agent_name}")
        try:
            _, msg = box.get(timeout=timeout)
            return msg
        except queue.Empty:
            return None


# ── Base agent ─────────────────────────────────────────────────────────────────

class BaseAgent(threading.Thread):
    def __init__(self, name: str, bus: MessageBus) -> None:
        super().__init__(name=name, daemon=True)
        self.name = name
        self._bus = bus
        self._running = True
        self._bus.register(name)
        self._log = logging.getLogger(f"agent.{name}")

    def handle(self, message: AgentMessage) -> AgentMessage | None:
        """Override to handle a received message. Return a reply or None."""
        raise NotImplementedError

    def send(self, receiver: str, msg_type: MessageType, payload: dict[str, Any],
             correlation_id: str = "") -> None:
        msg = AgentMessage(
            sender=self.name,
            receiver=receiver,
            message_type=msg_type,
            payload=payload,
            correlation_id=correlation_id,
        )
        self._bus.send(msg)

    def run(self) -> None:
        self._log.info("started")
        while self._running:
            msg = self._bus.receive(self.name, timeout=0.5)
            if msg is None:
                continue
            self._log.debug("received %s from %s", msg.message_type, msg.sender)
            reply = self.handle(msg)
            if reply is not None:
                self._bus.send(reply)
        self._log.info("stopped")

    def stop(self) -> None:
        self._running = False


# ── Concrete agents ────────────────────────────────────────────────────────────

class SummarizerAgent(BaseAgent):
    """Accepts REQUEST with {"text": str}, returns RESPONSE with {"summary": str}."""

    def handle(self, message: AgentMessage) -> AgentMessage | None:
        if message.message_type != MessageType.REQUEST:
            return None
        text: str = message.payload.get("text", "")
        summary = text[:80] + ("..." if len(text) > 80 else "")  # mock LLM
        self._log.info("summarised %d chars → %d chars", len(text), len(summary))
        return AgentMessage(
            sender=self.name,
            receiver=message.sender,
            message_type=MessageType.RESPONSE,
            correlation_id=message.id,
            payload={"summary": summary},
        )


class AnalyserAgent(BaseAgent):
    """Accepts REQUEST with {"summary": str}, returns RESPONSE with {"sentiment": str}."""

    def handle(self, message: AgentMessage) -> AgentMessage | None:
        if message.message_type != MessageType.REQUEST:
            return None
        summary = message.payload.get("summary", "")
        sentiment = "positive" if len(summary) % 2 == 0 else "negative"  # mock
        return AgentMessage(
            sender=self.name,
            receiver=message.sender,
            message_type=MessageType.RESPONSE,
            correlation_id=message.id,
            payload={"sentiment": sentiment},
        )


class OrchestratorAgent(BaseAgent):
    """Orchestrates: summarise → analyse, then prints result."""

    def __init__(self, bus: MessageBus) -> None:
        super().__init__("orchestrator", bus)
        self._pending: dict[str, str] = {}  # correlation_id → stage

    def handle(self, message: AgentMessage) -> AgentMessage | None:
        cid = message.correlation_id
        stage = self._pending.get(cid)

        if stage == "summarise" and message.message_type == MessageType.RESPONSE:
            summary = message.payload["summary"]
            self._log.info("got summary, sending to analyser")
            new_msg = AgentMessage(
                sender=self.name,
                receiver="analyser",
                message_type=MessageType.REQUEST,
                payload={"summary": summary},
            )
            self._pending[new_msg.id] = "analyse"
            return new_msg

        if stage == "analyse" and message.message_type == MessageType.RESPONSE:
            sentiment = message.payload["sentiment"]
            self._log.info("FINAL RESULT — sentiment: %s", sentiment)
            del self._pending[cid]

        return None

    def kick_off(self, text: str) -> None:
        msg = AgentMessage(
            sender=self.name,
            receiver="summariser",
            message_type=MessageType.REQUEST,
            payload={"text": text},
        )
        self._pending[msg.id] = "summarise"
        self._bus.send(msg)


# ── Driver ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    bus = MessageBus()
    orch    = OrchestratorAgent(bus)
    summ    = SummarizerAgent("summariser", bus)
    analyser = AnalyserAgent("analyser", bus)

    for agent in (orch, summ, analyser):
        agent.start()

    orch.kick_off(
        "Python's asyncio event loop provides single-threaded concurrent I/O "
        "using coroutines and event callbacks, enabling scalable network servers "
        "without the overhead of OS threads."
    )

    time.sleep(3)
    for agent in (orch, summ, analyser):
        agent.stop()
    for agent in (orch, summ, analyser):
        agent.join(timeout=2)
    logger.info("done")
```

## Exercises

1. Add a `HEARTBEAT` handler to `BaseAgent` that logs a heartbeat whenever the bus broadcasts one.
2. Implement a `DeadLetterQueue` in `MessageBus`: messages for unknown receivers go there instead of raising an exception. Add a `drain_dead_letters()` method.
3. Add `priority` to the `kick_off` call so high-priority tasks jump ahead of queued low-priority ones. Write a test that verifies ordering.
4. Extend `OrchestratorAgent` to handle `ERROR` message_type from `SummarizerAgent` — log the error and retry once.
5. Replace the `threading.Thread` base with `asyncio` coroutines. How does the `run` loop change?

(Solutions: ../solutions/week-07/day-2.md)

## Self-check

- [ ] I can describe the five message types and their directionality
- [ ] I can implement a mailbox message bus with priority queueing
- [ ] I can correlate requests to responses using `correlation_id`
- [ ] I can propagate errors between agents without crashing the system

## Supplementary links

- [Python queue module docs](https://docs.python.org/3/library/queue.html)
- [Pydantic BaseModel](https://docs.pydantic.dev/latest/concepts/models/)
- [Actor model overview (Wikipedia)](https://en.wikipedia.org/wiki/Actor_model)
