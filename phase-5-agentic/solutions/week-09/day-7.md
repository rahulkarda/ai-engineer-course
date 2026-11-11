# Solutions — Week 09 · Day 7 — Building the A2A Interop Layer (Part 1)

## Exercise 1: deregister with test

```python
# Add to AgentRegistry in registry.py

def deregister(self, name: str) -> None:
    """Remove an agent from the registry."""
    removed = self._cards.pop(name, None)
    if removed:
        logger.info("Deregistered agent '%s'", name)


# Test
def test_deregister():
    from a2a_interop.models import AgentCard, AgentSkill
    from a2a_interop.registry import AgentRegistry
    from a2a_interop.exceptions import AgentNotFoundError

    reg = AgentRegistry()
    card = AgentCard(
        name="agent-a",
        description="test",
        version="1.0.0",
        url="http://agent-a",
        skills=[AgentSkill(id="summarize", name="Summarize", description="Summarize text")],
    )
    reg.register(card)
    assert len(reg.find_by_skill("summarize")) == 1

    reg.deregister("agent-a")
    assert len(reg.find_by_skill("summarize")) == 0

    with pytest.raises(AgentNotFoundError):
        reg.route("summarize")
```

## Exercise 2: TaskStore.add_message and add_artifact

```python
# Add to TaskStore in task_store.py

from a2a_interop.models import Message, Artifact

def add_message(self, task_id: str, message: Message) -> Task:
    task = self.get(task_id)
    task.history.append(message)
    logger.debug("Added message to task %s (total: %d)", task_id[:8], len(task.history))
    return task

def add_artifact(self, task_id: str, artifact: Artifact) -> Task:
    task = self.get(task_id)
    task.artifacts.append(artifact)
    logger.debug("Added artifact '%s' to task %s", artifact.name, task_id[:8])
    return task


# Tests
def test_add_message():
    from a2a_interop.models import Task, Message, TextPart
    from a2a_interop.task_store import TaskStore

    store = TaskStore()
    task = store.create(Task())
    msg = Message(role="user", parts=[TextPart(text="hello")])
    updated = store.add_message(task.id, msg)
    assert len(updated.history) == 1
    assert updated.history[0].role == "user"


def test_add_artifact():
    from a2a_interop.models import Task, Artifact, TextPart
    from a2a_interop.task_store import TaskStore

    store = TaskStore()
    task = store.create(Task())
    art = Artifact(name="report.md", parts=[TextPart(text="# Report\nContent here")])
    updated = store.add_artifact(task.id, art)
    assert len(updated.artifacts) == 1
    assert updated.artifacts[0].name == "report.md"
```

## Exercise 3: HealthCheck ABC

```python
# Add to transport.py

from abc import abstractmethod


class Transport(ABC):
    @abstractmethod
    async def send(self, agent_url: str, task: "Task") -> "Task": ...

    @abstractmethod
    def register_handler(self, agent_url: str, handler: "Handler") -> None: ...

    @abstractmethod
    async def health(self, agent_url: str) -> bool:
        """Return True if the agent at agent_url is reachable."""


class InProcessTransport(Transport):
    # ... existing code ...

    async def health(self, agent_url: str) -> bool:
        return agent_url in self._handlers


class MockTransport(Transport):
    def __init__(self) -> None:
        super().__init__()
        # ... existing ...
        self._health_map: dict[str, bool] = {}

    def set_health(self, agent_url: str, healthy: bool) -> None:
        self._health_map[agent_url] = healthy

    async def health(self, agent_url: str) -> bool:
        return self._health_map.get(agent_url, True)  # Default healthy


# Tests
import asyncio
import pytest


def test_inprocess_health_registered():
    from a2a_interop.transport import InProcessTransport
    from a2a_interop.models import Task, TaskState, TaskStatus

    transport = InProcessTransport()

    async def handler(task):
        return task

    transport.register_handler("http://agent", handler)
    result = asyncio.get_event_loop().run_until_complete(
        transport.health("http://agent")
    )
    assert result is True


def test_inprocess_health_unregistered():
    from a2a_interop.transport import InProcessTransport

    transport = InProcessTransport()
    result = asyncio.get_event_loop().run_until_complete(
        transport.health("http://nobody")
    )
    assert result is False


def test_mock_transport_health_configurable():
    from a2a_interop.transport import MockTransport

    transport = MockTransport()
    transport.set_health("http://agent", False)
    result = asyncio.get_event_loop().run_until_complete(
        transport.health("http://agent")
    )
    assert result is False
```

## Exercise 4: find_by_multiple_skills

```python
# Add to AgentRegistry in registry.py

def find_by_multiple_skills(
    self,
    skill_ids: list[str],
    require_all: bool = True,
) -> list[AgentCard]:
    """
    If require_all=True: return agents that have ALL listed skills.
    If require_all=False: return agents that have ANY listed skill.
    Sorted by reliability descending.
    """
    def matches(card: AgentCard) -> bool:
        card_skills = set(card.skill_ids())
        required = set(skill_ids)
        if require_all:
            return required.issubset(card_skills)
        else:
            return bool(required & card_skills)

    return sorted(
        [c for c in self._cards.values() if matches(c)],
        key=lambda c: c.reliability,
        reverse=True,
    )


# Tests
def test_find_by_multiple_skills_require_all():
    from a2a_interop.models import AgentCard, AgentSkill
    from a2a_interop.registry import AgentRegistry

    def make(name, skills):
        return AgentCard(
            name=name, description="", version="1.0.0", url=f"http://{name}",
            skills=[AgentSkill(id=s, name=s, description=s) for s in skills],
        )

    reg = AgentRegistry()
    reg.register(make("a1", ["s1", "s2", "s3"]))
    reg.register(make("a2", ["s1", "s2"]))
    reg.register(make("a3", ["s1"]))

    all_three = reg.find_by_multiple_skills(["s1", "s2", "s3"], require_all=True)
    assert [c.name for c in all_three] == ["a1"]

    any_two = reg.find_by_multiple_skills(["s1", "s2"], require_all=False)
    assert {c.name for c in any_two} == {"a1", "a2", "a3"}  # a3 has s1


def test_find_by_multiple_skills_any():
    from a2a_interop.models import AgentCard, AgentSkill
    from a2a_interop.registry import AgentRegistry

    def make(name, skills):
        return AgentCard(
            name=name, description="", version="1.0.0", url=f"http://{name}",
            skills=[AgentSkill(id=s, name=s, description=s) for s in skills],
        )

    reg = AgentRegistry()
    reg.register(make("sql", ["sql-generation"]))
    reg.register(make("nlp", ["summarization"]))
    reg.register(make("full", ["sql-generation", "summarization"]))

    results = reg.find_by_multiple_skills(["sql-generation", "summarization"], require_all=False)
    assert len(results) == 3
```
