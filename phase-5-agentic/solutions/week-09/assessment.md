# Solutions — Week 09 Assessment

## Conceptual answers

**Q1. Why can't agents just call each other's REST APIs directly?**

They technically can, but three contract gaps emerge immediately:
1. **Semantic mismatch** — each agent has its own JSON schema; callers must know every detail of the callee's interface.
2. **Capability opacity** — the caller can't discover what the callee does without reading external documentation.
3. **Lifecycle gap** — HTTP 200/404/500 doesn't express running/completed/failed/input-required for long-running tasks.
A2A adds a standard message envelope (typed Parts), a discovery contract (AgentCard at `/.well-known/agent.json`), and a six-state task lifecycle on top of HTTP.

**Q2. Capability vs. skill (code-review agent example)**

- **Capability:** a protocol feature — e.g., `streaming: true` means the agent can return partial results via SSE.
- **Skill:** a domain function — e.g., `id: "pr-review"` means the agent can review pull requests.
Capabilities tell the client *how* to talk to the agent; skills tell it *what* to ask for.

**Q3. input-required state**

The server is blocked, waiting for more information from the client. The *client* acts next — it must send a follow-up message with the requested information. If the client never responds, the task remains in `input-required` indefinitely (servers should implement a timeout that transitions it to `failed` or `cancelled`).

**Q4. Agent B returns HTTP 503 — error-handling strategy**

1. Task stays in `working` — the HTTP failure is transport-level, not a task state change yet.
2. Retry with exponential backoff (e.g., 1s, 2s, 4s) — 503 is a transient error.
3. After max retries: check registry for a fallback agent with the same skill. If found, create a new task and submit.
4. If no fallback: transition original task to `failed` with error message. Surface to user: "Specialist unavailable, please retry."
5. User sees a structured error, not a raw HTTP exception.

**Q5. What A2A gives over a REST wrapper**

- **Lifecycle management** — submitted/working/input-required/completed/failed/cancelled, all standard.
- **Multi-turn interaction** — `input-required` enables the agent to ask clarifying questions without breaking the contract.
- **Discovery** — Agent B's card tells Agent A exactly what it can do without prior coordination.
- **Transport agnosticism** — swap HTTP for a message queue without changing either agent's business logic.
- **Structured artifact outputs** — results are distinct from conversation history; rich Part types.

**Q6. MCP vs. A2A**

1. MCP connects agents to *deterministic* tools and data sources (databases, file systems, APIs). The responder is not intelligent.
2. A2A connects agents to *other intelligent agents* that may themselves reason, plan, and delegate further.
3. The combination: specialist agents use MCP internally for tool access; orchestrators use A2A to delegate to specialists. They operate at different layers and are complementary.

**Q7. `.well-known/agent.json` and zero-config discovery**

Any HTTP client can GET `{base_url}/.well-known/agent.json` to retrieve the full capability manifest without prior knowledge. This is borrowed from RFC 8615 (Well-Known URIs), which reserves the `/.well-known/` path prefix for service metadata. Discovery requires only the agent's base URL — no service registry, DNS records, or manual configuration.

**Q8. Direct access to SQL specialist**

No code changes are needed in the SQL specialist — it is already A2A compliant and can accept tasks from any client. Risks of direct access:
- **Auth bypass risk** — clients calling directly may have broader scope than intended; the orchestrator normally gates access.
- **Missing context** — the orchestrator may inject enriched context (correlation IDs, user scoping) that direct callers omit.
- **No fallback orchestration** — if the SQL specialist fails, there is no retry/fallback layer.
Mitigation: enforce the same auth at the specialist level regardless of caller.

## Coding challenge — reference implementation

Key design decisions and reference patterns (not a complete copy-paste solution):

```python
# a2a_interop/server.py
import logging
from abc import ABC, abstractmethod
from a2a_interop.models import AgentCard, Task, TaskState
from a2a_interop.task_store import TaskStore

logger = logging.getLogger(__name__)


class A2AServer(ABC):
    """Base class for all A2A-compliant agent servers."""

    def __init__(self, store: TaskStore) -> None:
        self._store = store

    @abstractmethod
    def get_card(self) -> AgentCard:
        """Return the agent's capability card."""

    @abstractmethod
    async def handle(self, task: Task) -> Task:
        """Process a task. Must transition it to a terminal state."""

    async def process(self, task: Task) -> Task:
        """Entry point — wraps handle with lifecycle management."""
        self._store.transition(task.id, TaskState.WORKING)
        try:
            result = await self.handle(task)
            return result
        except Exception as exc:
            logger.exception("Task %s failed: %s", task.id[:8], exc)
            self._store.transition(task.id, TaskState.FAILED, str(exc))
            return self._store.get(task.id)
```

```python
# a2a_interop/client.py
import asyncio
import logging
from a2a_interop.exceptions import A2AError, AgentNotFoundError, TransportError
from a2a_interop.models import AgentCard, Message, Task, TextPart, TaskState, TERMINAL_STATES
from a2a_interop.registry import AgentRegistry
from a2a_interop.task_store import TaskStore
from a2a_interop.transport import Transport

logger = logging.getLogger(__name__)


class A2AClient:
    MAX_RETRIES = 2
    RETRY_DELAY = 0.1

    def __init__(
        self,
        registry: AgentRegistry,
        store: TaskStore,
        transport: Transport,
    ) -> None:
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
                    if result.artifacts:
                        parts = result.artifacts[0].parts
                        text_parts = [p for p in parts if hasattr(p, "text")]
                        return text_parts[0].text if text_parts else ""  # type: ignore[union-attr]
                    return ""
                raise A2AError(f"Task ended in {result.status.state.value}: {result.status.message}")
            except TransportError as exc:
                if attempt <= self.MAX_RETRIES:
                    logger.warning("Transport error (attempt %d): %s — retrying", attempt, exc)
                    await asyncio.sleep(self.RETRY_DELAY)
                else:
                    self._store.transition(task.id, TaskState.FAILED, str(exc))
                    raise A2AError(f"Max retries exceeded: {exc}") from exc

        raise A2AError("Delegation failed")

    async def delegate_parallel(self, tasks: list[tuple[str, str]]) -> list[str]:
        return list(await asyncio.gather(*[self.delegate(skill, text) for skill, text in tasks]))
```

```python
# agents/summarizer.py
import logging
from a2a_interop.models import AgentCard, AgentSkill, Artifact, Task, TaskState, TextPart
from a2a_interop.server import A2AServer
from a2a_interop.task_store import TaskStore

logger = logging.getLogger(__name__)


class SummarizationAgent(A2AServer):
    def get_card(self) -> AgentCard:
        return AgentCard(
            name="summarization-agent",
            description="Summarizes documents",
            version="1.0.0",
            url="http://summarizer",
            skills=[AgentSkill(id="text-summarization", name="Text Summarization",
                               description="Summarize long documents")],
        )

    async def handle(self, task: Task) -> Task:
        text_parts = [
            p for msg in task.history for p in msg.parts if hasattr(p, "text")
        ]
        if not text_parts:
            self._store.transition(task.id, TaskState.INPUT_REQUIRED, "Provide text to summarize")
            return self._store.get(task.id)

        input_text = text_parts[0].text  # type: ignore[union-attr]
        summary = f"Summary: {input_text[:50]}... [key points identified]"
        artifact = Artifact(name="summary", parts=[TextPart(text=summary)])
        self._store.add_artifact(task.id, artifact)
        self._store.transition(task.id, TaskState.COMPLETED)
        return self._store.get(task.id)
```

### Integration test reference

```python
# tests/test_integration.py
import asyncio
import pytest
from a2a_interop.registry import AgentRegistry
from a2a_interop.task_store import TaskStore
from a2a_interop.transport import InProcessTransport
from a2a_interop.client import A2AClient
from agents.summarizer import SummarizationAgent
from agents.sql_agent import SQLGenerationAgent


@pytest.fixture
def setup():
    store = TaskStore()
    registry = AgentRegistry()
    transport = InProcessTransport()
    client = A2AClient(registry, store, transport)

    # Wire up agents
    for AgentClass in [SummarizationAgent, SQLGenerationAgent]:
        agent = AgentClass(store)
        card = agent.get_card()
        transport.register_handler(card.url, agent.process)
        client.discover(card.url, card)

    return client, store


@pytest.mark.asyncio
async def test_full_pipeline_inprocess(setup):
    client, _ = setup
    result = await client.delegate(
        "text-summarization",
        "This is a long document about quarterly earnings performance and market outlook.",
    )
    assert "Summary:" in result


@pytest.mark.asyncio
async def test_parallel_delegation(setup):
    client, _ = setup
    results = await client.delegate_parallel([
        ("text-summarization", "Earnings report text here"),
        ("nl-to-sql", "How many users signed up last month?"),
    ])
    assert len(results) == 2
    assert all(r for r in results)   # Both non-empty
```

## Rubric scoring guide

| Score | Criteria |
|---|---|
| 100% | All 18+ tests pass; all FRs implemented; typed; clean architecture; retry + fallback |
| 90% | 15+ tests pass; minor missing FR (e.g., no `delegate_parallel`); good types |
| 75% (pass) | Core layer works (registry + transport + task store + at least one agent end-to-end); ≥12 tests |
| 60% | Partial implementation; task store OR registry missing; <10 tests |
| Below pass | Core loop doesn't run; major structural issues |
