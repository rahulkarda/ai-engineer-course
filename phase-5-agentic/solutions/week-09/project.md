# Solutions — Week 09 Project — A2A Interop Layer

This is the complete reference implementation. Study it *after* attempting the project yourself.

## Complete project implementation

### pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "a2a-interop"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.0",
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

### a2a_interop/exceptions.py

```python
class A2AError(Exception):
    """Base exception for A2A interop layer."""

class AgentNotFoundError(A2AError):
    """No agent registered for the requested skill."""

class InvalidTransitionError(A2AError):
    """Attempted an illegal task state transition."""

class TaskNotFoundError(A2AError):
    """Task ID not found in the store."""

class TransportError(A2AError):
    """Transport layer failure (network, timeout, no handler)."""
```

### a2a_interop/models.py

```python
import time
import uuid
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field


class TaskState(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATES: frozenset[TaskState] = frozenset({
    TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED
})

VALID_TRANSITIONS: dict[TaskState, frozenset[TaskState]] = {
    TaskState.SUBMITTED: frozenset({TaskState.WORKING, TaskState.CANCELLED}),
    TaskState.WORKING: frozenset({
        TaskState.COMPLETED, TaskState.FAILED,
        TaskState.INPUT_REQUIRED, TaskState.CANCELLED,
    }),
    TaskState.INPUT_REQUIRED: frozenset({TaskState.WORKING, TaskState.CANCELLED}),
    TaskState.COMPLETED: frozenset(),
    TaskState.FAILED: frozenset(),
    TaskState.CANCELLED: frozenset(),
}


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class FilePart(BaseModel):
    type: Literal["file"] = "file"
    name: str
    mime_type: str = "application/octet-stream"
    bytes_b64: str = ""
    uri: str | None = None


class DataPart(BaseModel):
    type: Literal["data"] = "data"
    data: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    role: Literal["user", "agent"]
    parts: list[TextPart | FilePart | DataPart] = Field(default_factory=list)
    timestamp: float = Field(default_factory=time.time)


class TaskStatus(BaseModel):
    state: TaskState = TaskState.SUBMITTED
    timestamp: float = Field(default_factory=time.time)
    message: str = ""


class Artifact(BaseModel):
    artifact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    parts: list[TextPart | FilePart | DataPart] = Field(default_factory=list)
    last_chunk: bool = True


class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = Field(default_factory=TaskStatus)
    history: list[Message] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentSkill(BaseModel):
    id: str
    name: str
    description: str
    input_modes: list[str] = Field(default_factory=lambda: ["text/plain"])
    output_modes: list[str] = Field(default_factory=lambda: ["text/plain"])


class AgentCapabilities(BaseModel):
    streaming: bool = False
    push_notifications: bool = False


class AgentCard(BaseModel):
    name: str
    description: str
    version: str
    url: str
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    skills: list[AgentSkill] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    reliability: float = Field(default=1.0, ge=0.0, le=1.0)

    def skill_ids(self) -> list[str]:
        return [s.id for s in self.skills]

    def has_skill(self, skill_id: str) -> bool:
        return skill_id in self.skill_ids()
```

### a2a_interop/task_store.py

```python
import logging
from typing import Iterator
from a2a_interop.exceptions import InvalidTransitionError, TaskNotFoundError
from a2a_interop.models import (
    Artifact, Message, Task, TaskState, TaskStatus,
    VALID_TRANSITIONS, TERMINAL_STATES,
)

logger = logging.getLogger(__name__)


class TaskStore:
    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    def create(self, task: Task) -> Task:
        self._tasks[task.id] = task
        return task

    def get(self, task_id: str) -> Task:
        if task_id not in self._tasks:
            raise TaskNotFoundError(f"Task {task_id!r} not found")
        return self._tasks[task_id]

    def transition(self, task_id: str, new_state: TaskState, message: str = "") -> Task:
        task = self.get(task_id)
        current = task.status.state
        if new_state not in VALID_TRANSITIONS[current]:
            raise InvalidTransitionError(
                f"{current.value} → {new_state.value} not allowed"
            )
        task.status = TaskStatus(state=new_state, message=message)
        logger.info("Task %s: %s → %s", task_id[:8], current.value, new_state.value)
        return task

    def add_message(self, task_id: str, message: Message) -> Task:
        task = self.get(task_id)
        task.history.append(message)
        return task

    def add_artifact(self, task_id: str, artifact: Artifact) -> Task:
        task = self.get(task_id)
        task.artifacts.append(artifact)
        return task

    def pending_tasks(self) -> list[Task]:
        return [t for t in self._tasks.values() if t.status.state not in TERMINAL_STATES]

    def all_tasks(self) -> Iterator[Task]:
        yield from self._tasks.values()

    def __len__(self) -> int:
        return len(self._tasks)
```

### a2a_interop/registry.py

```python
import logging
from a2a_interop.exceptions import AgentNotFoundError
from a2a_interop.models import AgentCard

logger = logging.getLogger(__name__)


class AgentRegistry:
    def __init__(self) -> None:
        self._cards: dict[str, AgentCard] = {}

    def register(self, card: AgentCard) -> None:
        self._cards[card.name] = card
        logger.info("Registered '%s' skills=%s", card.name, card.skill_ids())

    def deregister(self, name: str) -> None:
        self._cards.pop(name, None)

    def get(self, name: str) -> AgentCard | None:
        return self._cards.get(name)

    def find_by_skill(self, skill_id: str, min_reliability: float = 0.0) -> list[AgentCard]:
        return sorted(
            [c for c in self._cards.values()
             if c.has_skill(skill_id) and c.reliability >= min_reliability],
            key=lambda c: c.reliability,
            reverse=True,
        )

    def find_by_multiple_skills(
        self, skill_ids: list[str], require_all: bool = True
    ) -> list[AgentCard]:
        required = set(skill_ids)
        def matches(card: AgentCard) -> bool:
            card_skills = set(card.skill_ids())
            return required.issubset(card_skills) if require_all else bool(required & card_skills)
        return sorted(
            [c for c in self._cards.values() if matches(c)],
            key=lambda c: c.reliability,
            reverse=True,
        )

    def route(self, skill_id: str, min_reliability: float = 0.0) -> AgentCard:
        candidates = self.find_by_skill(skill_id, min_reliability)
        if not candidates:
            raise AgentNotFoundError(f"No agent for skill '{skill_id}'")
        return candidates[0]

    def skill_index(self) -> dict[str, list[str]]:
        idx: dict[str, list[str]] = {}
        for card in self._cards.values():
            for sid in card.skill_ids():
                idx.setdefault(sid, []).append(card.name)
        return idx

    def __len__(self) -> int:
        return len(self._cards)
```

### a2a_interop/transport.py

```python
import logging
from abc import ABC, abstractmethod
from typing import Awaitable, Callable
from a2a_interop.exceptions import TransportError
from a2a_interop.models import Task

logger = logging.getLogger(__name__)

Handler = Callable[[Task], Awaitable[Task]]


class Transport(ABC):
    @abstractmethod
    async def send(self, agent_url: str, task: Task) -> Task: ...

    @abstractmethod
    def register_handler(self, agent_url: str, handler: Handler) -> None: ...

    @abstractmethod
    async def health(self, agent_url: str) -> bool: ...


class InProcessTransport(Transport):
    def __init__(self) -> None:
        self._handlers: dict[str, Handler] = {}

    def register_handler(self, agent_url: str, handler: Handler) -> None:
        self._handlers[agent_url] = handler

    async def send(self, agent_url: str, task: Task) -> Task:
        if agent_url not in self._handlers:
            raise TransportError(f"No handler for {agent_url!r}")
        return await self._handlers[agent_url](task)

    async def health(self, agent_url: str) -> bool:
        return agent_url in self._handlers


class MockTransport(Transport):
    def __init__(self) -> None:
        self._responses: dict[str, list[Task]] = {}
        self._handlers: dict[str, Handler] = {}
        self._calls: list[tuple[str, Task]] = []
        self._health_map: dict[str, bool] = {}

    def add_response(self, agent_url: str, task: Task) -> None:
        self._responses.setdefault(agent_url, []).append(task)

    def set_health(self, agent_url: str, healthy: bool) -> None:
        self._health_map[agent_url] = healthy

    def register_handler(self, agent_url: str, handler: Handler) -> None:
        self._handlers[agent_url] = handler

    async def send(self, agent_url: str, task: Task) -> Task:
        self._calls.append((agent_url, task))
        if agent_url in self._handlers:
            return await self._handlers[agent_url](task)
        if self._responses.get(agent_url):
            return self._responses[agent_url].pop(0)
        raise TransportError(f"No response for {agent_url!r}")

    async def health(self, agent_url: str) -> bool:
        return self._health_map.get(agent_url, True)

    @property
    def call_count(self) -> int:
        return len(self._calls)

    def calls_to(self, agent_url: str) -> list[Task]:
        return [t for url, t in self._calls if url == agent_url]
```

### a2a_interop/server.py

```python
import logging
from abc import ABC, abstractmethod
from a2a_interop.models import AgentCard, Task, TaskState
from a2a_interop.task_store import TaskStore

logger = logging.getLogger(__name__)


class A2AServer(ABC):
    def __init__(self, store: TaskStore) -> None:
        self._store = store

    @abstractmethod
    def get_card(self) -> AgentCard: ...

    @abstractmethod
    async def handle(self, task: Task) -> Task: ...

    async def process(self, task: Task) -> Task:
        """Public entry — wraps handle with lifecycle management."""
        self._store.transition(task.id, TaskState.WORKING)
        try:
            return await self.handle(task)
        except Exception as exc:
            logger.exception("Task %s failed", task.id[:8])
            self._store.transition(task.id, TaskState.FAILED, str(exc))
            return self._store.get(task.id)
```

### a2a_interop/client.py

```python
import asyncio
import logging
from a2a_interop.exceptions import A2AError, TransportError
from a2a_interop.models import AgentCard, Message, Task, TextPart, TaskState
from a2a_interop.registry import AgentRegistry
from a2a_interop.task_store import TaskStore
from a2a_interop.transport import Transport

logger = logging.getLogger(__name__)


class A2AClient:
    MAX_RETRIES = 2
    RETRY_DELAY = 0.1

    def __init__(self, registry: AgentRegistry, store: TaskStore, transport: Transport) -> None:
        self._registry = registry
        self._store = store
        self._transport = transport

    def discover(self, agent_url: str, card: AgentCard) -> None:
        self._registry.register(card)
        logger.info("Discovered '%s' at %s", card.name, agent_url)

    async def delegate(self, skill_id: str, text: str) -> str:
        card = self._registry.route(skill_id)
        task = self._store.create(Task())
        task.history.append(Message(role="user", parts=[TextPart(text=text)]))

        for attempt in range(1, self.MAX_RETRIES + 2):
            try:
                result = await self._transport.send(card.url, task)
                if result.status.state == TaskState.COMPLETED:
                    for artifact in result.artifacts:
                        for part in artifact.parts:
                            if hasattr(part, "text"):
                                return part.text  # type: ignore[union-attr]
                    return ""
                raise A2AError(f"Task in state {result.status.state.value}: {result.status.message}")
            except TransportError as exc:
                if attempt <= self.MAX_RETRIES:
                    logger.warning("Retry %d/%d: %s", attempt, self.MAX_RETRIES, exc)
                    await asyncio.sleep(self.RETRY_DELAY)
                else:
                    self._store.transition(task.id, TaskState.FAILED, str(exc))
                    raise A2AError(f"Max retries exceeded") from exc
        raise A2AError("Delegation failed")

    async def delegate_parallel(self, tasks: list[tuple[str, str]]) -> list[str]:
        return list(await asyncio.gather(*[self.delegate(s, t) for s, t in tasks]))
```

### agents/summarizer.py

```python
import logging
from a2a_interop.models import AgentCard, AgentSkill, Artifact, Task, TaskState, TextPart
from a2a_interop.server import A2AServer

logger = logging.getLogger(__name__)


class SummarizationAgent(A2AServer):
    def get_card(self) -> AgentCard:
        return AgentCard(
            name="summarization-agent",
            description="Summarizes documents and extracts key points",
            version="1.0.0",
            url="http://summarizer",
            skills=[AgentSkill(id="text-summarization", name="Text Summarization",
                               description="Summarize long documents")],
        )

    async def handle(self, task: Task) -> Task:
        text_parts = [p for m in task.history for p in m.parts if hasattr(p, "text")]
        if not text_parts:
            self._store.transition(task.id, TaskState.INPUT_REQUIRED, "Provide text to summarize")
            return self._store.get(task.id)

        input_text = text_parts[0].text  # type: ignore[union-attr]
        summary = f"Summary ({len(input_text)} chars): {input_text[:60]}... [3 key points identified]"
        self._store.add_artifact(task.id, Artifact(name="summary", parts=[TextPart(text=summary)]))
        self._store.transition(task.id, TaskState.COMPLETED)
        return self._store.get(task.id)
```

### agents/sql_agent.py

```python
import logging
from a2a_interop.models import AgentCard, AgentSkill, Artifact, Task, TaskState, TextPart
from a2a_interop.server import A2AServer

logger = logging.getLogger(__name__)


class SQLGenerationAgent(A2AServer):
    def get_card(self) -> AgentCard:
        return AgentCard(
            name="sql-generation-agent",
            description="Translates natural language questions to SQL",
            version="1.0.0",
            url="http://sql-agent",
            skills=[AgentSkill(id="nl-to-sql", name="NL to SQL",
                               description="Generate SQL from plain English")],
        )

    async def handle(self, task: Task) -> Task:
        text_parts = [p for m in task.history for p in m.parts if hasattr(p, "text")]
        if not text_parts:
            self._store.transition(task.id, TaskState.INPUT_REQUIRED, "Provide a question")
            return self._store.get(task.id)

        question = text_parts[0].text  # type: ignore[union-attr]
        sql = f"SELECT * FROM data WHERE description LIKE '%{question[:20]}%' LIMIT 100;"
        self._store.add_artifact(task.id, Artifact(name="query.sql", parts=[TextPart(text=sql)]))
        self._store.transition(task.id, TaskState.COMPLETED)
        return self._store.get(task.id)
```

### agents/orchestrator.py

```python
import asyncio
import logging
from a2a_interop.client import A2AClient

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    def __init__(self, client: A2AClient) -> None:
        self._client = client

    async def run(self, document: str, question: str) -> str:
        logger.info("Running orchestration: summarize + generate SQL")
        summary, sql = await self._client.delegate_parallel([
            ("text-summarization", document),
            ("nl-to-sql", question),
        ])
        return f"=== Summary ===\n{summary}\n\n=== SQL Query ===\n{sql}"
```

### main.py

```python
"""
Week 09 project demo — A2A interop layer end-to-end.
"""

import asyncio
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Optional: real LLM path (mocked by default; set ANTHROPIC_API_KEY to use real)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


async def main() -> None:
    from a2a_interop.registry import AgentRegistry
    from a2a_interop.task_store import TaskStore
    from a2a_interop.transport import InProcessTransport
    from a2a_interop.client import A2AClient
    from agents.summarizer import SummarizationAgent
    from agents.sql_agent import SQLGenerationAgent
    from agents.orchestrator import OrchestratorAgent

    store = TaskStore()
    registry = AgentRegistry()
    transport = InProcessTransport()
    client = A2AClient(registry, store, transport)

    # Register agents
    for AgentClass in [SummarizationAgent, SQLGenerationAgent]:
        agent = AgentClass(store)
        card = agent.get_card()
        transport.register_handler(card.url, agent.process)
        client.discover(card.url, card)

    # Run orchestrator
    orchestrator = OrchestratorAgent(client)
    result = await orchestrator.run(
        document=(
            "This quarterly report covers financial performance, market expansion, "
            "new product launches, and risk factors for Q4 2024."
        ),
        question="How many products were launched in Q4 2024?",
    )
    print("\n" + result)
    logger.info("Demo complete. Tasks in store: %d", len(store))


if __name__ == "__main__":
    asyncio.run(main())
```
