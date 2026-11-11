# Solutions — Week 09 · Day 2 — The A2A Protocol Concept

## Exercise 1: DataAnalysisAgent + tests

```python
from dataclasses import dataclass, field
from enum import Enum
import uuid
import pytest


class TaskStatus(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"
    INPUT_REQUIRED = "input-required"
    CANCELLED = "cancelled"


@dataclass
class TextPart:
    type: str = "text"
    text: str = ""


@dataclass
class Message:
    role: str
    parts: list[TextPart] = field(default_factory=list)


@dataclass
class Task:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = TaskStatus.SUBMITTED
    messages: list[Message] = field(default_factory=list)
    result: Message | None = None


@dataclass
class AgentCard:
    name: str
    description: str
    version: str
    capabilities: list[str]
    url: str


class DataAnalysisAgent:
    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="data-analysis-agent",
            description="Analyzes CSV data and generates statistical summaries",
            version="1.0.0",
            capabilities=["csv-analysis", "chart-generation", "statistical-summary"],
            url="https://agents.example.com/data-analysis",
        )

    def submit_task(self, message: Message) -> Task:
        task = Task(messages=[message], status=TaskStatus.WORKING)
        return task

    def process_task(self, task: Task) -> Task:
        input_text = task.messages[0].parts[0].text if task.messages else ""
        if "csv" in input_text.lower() or "data" in input_text.lower():
            task.status = TaskStatus.COMPLETED
            task.result = Message(
                role="agent",
                parts=[TextPart(text="Analysis complete: 3 columns, 100 rows, mean=42.5, std=8.2")],
            )
        else:
            task.status = TaskStatus.INPUT_REQUIRED
            task.result = Message(
                role="agent",
                parts=[TextPart(text="Please provide a CSV file or data description.")],
            )
        return task


class OrchestratorAgent:
    def __init__(self) -> None:
        self._servers: dict[str, DataAnalysisAgent] = {}

    def discover(self, url: str, server: DataAnalysisAgent) -> AgentCard:
        self._servers[url] = server
        return server.get_agent_card()

    def delegate(self, url: str, text: str) -> str:
        server = self._servers[url]
        msg = Message(role="user", parts=[TextPart(text=text)])
        task = server.submit_task(msg)
        task = server.process_task(task)
        if task.status == TaskStatus.COMPLETED and task.result:
            return task.result.parts[0].text
        raise RuntimeError(f"Task in state: {task.status}")


def test_data_analysis_discovery():
    orchestrator = OrchestratorAgent()
    agent = DataAnalysisAgent()
    card = orchestrator.discover("https://agents.example.com/data-analysis", agent)
    assert "csv-analysis" in card.capabilities
    assert "statistical-summary" in card.capabilities


def test_data_analysis_delegation():
    orchestrator = OrchestratorAgent()
    agent = DataAnalysisAgent()
    orchestrator.discover("https://agents.example.com/data-analysis", agent)
    result = orchestrator.delegate(
        "https://agents.example.com/data-analysis",
        "Analyze the sales CSV data",
    )
    assert "Analysis complete" in result


def test_data_analysis_input_required():
    agent = DataAnalysisAgent()
    msg = Message(role="user", parts=[TextPart(text="help me")])
    task = agent.submit_task(msg)
    task = agent.process_task(task)
    assert task.status == TaskStatus.INPUT_REQUIRED
```

## Exercise 2: TaskStore

```python
from dataclasses import dataclass, field
import uuid
from enum import Enum


class TaskStatus(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"
    INPUT_REQUIRED = "input-required"
    CANCELLED = "cancelled"


@dataclass
class Task:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = TaskStatus.SUBMITTED


class TaskStore:
    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    def create(self, task: Task) -> Task:
        self._tasks[task.task_id] = task
        return task

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def update_status(self, task_id: str, status: TaskStatus) -> Task | None:
        task = self._tasks.get(task_id)
        if task:
            task.status = status
        return task

    def __len__(self) -> int:
        return len(self._tasks)


def test_task_store_create_and_get():
    store = TaskStore()
    task = Task()
    store.create(task)
    retrieved = store.get_task(task.task_id)
    assert retrieved is not None
    assert retrieved.task_id == task.task_id


def test_task_store_missing_returns_none():
    store = TaskStore()
    assert store.get_task("nonexistent") is None


def test_task_store_update_status():
    store = TaskStore()
    task = store.create(Task())
    updated = store.update_status(task.task_id, TaskStatus.COMPLETED)
    assert updated is not None
    assert updated.status == TaskStatus.COMPLETED
```

## Exercise 3: cancel_task with CANCELLATION_REQUESTED

```python
from enum import Enum
from dataclasses import dataclass, field
import uuid
import pytest


class TaskStatus(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    CANCELLATION_REQUESTED = "cancellation-requested"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"
    INPUT_REQUIRED = "input-required"


@dataclass
class Task:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = TaskStatus.SUBMITTED
    _work_done: bool = False  # simulates mid-flight state


class CancellableAgent:
    def submit_task(self, task: Task) -> Task:
        task.status = TaskStatus.WORKING
        return task

    def cancel_task(self, task: Task) -> Task:
        if task.status == TaskStatus.WORKING:
            task.status = TaskStatus.CANCELLATION_REQUESTED
        return task

    def process_task(self, task: Task) -> Task:
        # Check before doing expensive work
        if task.status == TaskStatus.CANCELLATION_REQUESTED:
            task.status = TaskStatus.CANCELLED
            return task
        # Do work...
        task._work_done = True
        task.status = TaskStatus.COMPLETED
        return task


def test_cancel_mid_flight():
    agent = CancellableAgent()
    task = Task()
    agent.submit_task(task)
    agent.cancel_task(task)
    assert task.status == TaskStatus.CANCELLATION_REQUESTED
    agent.process_task(task)
    assert task.status == TaskStatus.CANCELLED
    assert not task._work_done


def test_no_cancel_before_work_starts():
    agent = CancellableAgent()
    task = Task()
    agent.submit_task(task)
    # Don't cancel — should complete normally
    agent.process_task(task)
    assert task.status == TaskStatus.COMPLETED
    assert task._work_done
```

## Exercise 4: ConversationBuffer

```python
from dataclasses import dataclass, field
import time


@dataclass
class TextPart:
    type: str = "text"
    text: str = ""


@dataclass
class Message:
    role: str
    parts: list[TextPart] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class ConversationBuffer:
    def __init__(self) -> None:
        self._messages: list[Message] = []

    def add(self, message: Message) -> None:
        self._messages.append(message)

    def all_messages(self) -> list[Message]:
        return list(self._messages)

    def last_user_message(self) -> Message | None:
        for msg in reversed(self._messages):
            if msg.role == "user":
                return msg
        return None

    def last_agent_message(self) -> Message | None:
        for msg in reversed(self._messages):
            if msg.role == "agent":
                return msg
        return None

    def messages_since(self, timestamp: float) -> list[Message]:
        return [m for m in self._messages if m.timestamp > timestamp]


def test_conversation_buffer():
    buf = ConversationBuffer()
    t0 = time.time()
    buf.add(Message(role="user", parts=[TextPart(text="hello")]))
    buf.add(Message(role="agent", parts=[TextPart(text="hi there")]))
    buf.add(Message(role="user", parts=[TextPart(text="follow up")]))

    assert buf.last_user_message().parts[0].text == "follow up"  # type: ignore[union-attr]
    assert buf.last_agent_message().parts[0].text == "hi there"  # type: ignore[union-attr]
    since = buf.messages_since(t0)
    assert len(since) == 3
```
