# Solutions: Lesson 5 — Forgetting, Decay, and Mem0 / Letta

---

## Exercise 1 — `expires_at` field

```python
from datetime import datetime, timezone
from uuid import uuid4

# In MemoryRecord (or as a subclass)
@dataclass
class MemoryRecord:
    id: UUID
    content: str
    vector: list[float]
    importance: float
    created_at: datetime
    last_accessed: datetime
    archived: bool = False
    expires_at: datetime | None = None   # <-- new field
    metadata: dict = field(default_factory=dict)
```

In `MemoryManager.run_forgetting_pass`:

```python
def run_forgetting_pass(self) -> dict[str, int]:
    now = datetime.now(timezone.utc)
    active = self.active_records()

    # Step 0: expire TTL records before decay-score pass
    expired = [r for r in active if r.expires_at is not None and r.expires_at <= now]
    for r in expired:
        r.archived = True
        log.info("Expired record archived: %s", r.content[:40])
    active = [r for r in active if not r.archived]

    # ... rest of existing decay-score logic ...
```

Test:

```python
def test_expires_at_archival():
    manager = MemoryManager(archive_threshold=0.0)  # disable decay archival
    now = datetime.now(timezone.utc)

    expired_record = MemoryRecord(
        id=uuid4(),
        content="TTL record",
        vector=mock_embed("TTL record"),
        importance=0.9,
        created_at=now,
        last_accessed=now,
        expires_at=now - timedelta(minutes=1),   # already expired
    )
    active_record = MemoryRecord(
        id=uuid4(),
        content="Active record",
        vector=mock_embed("Active record"),
        importance=0.9,
        created_at=now,
        last_accessed=now,
        expires_at=now + timedelta(days=7),     # future expiry
    )

    manager.add(expired_record)
    manager.add(active_record)
    manager.run_forgetting_pass()

    active = manager.active_records()
    ids = [r.id for r in active]
    assert expired_record.id not in ids
    assert active_record.id in ids
```

---

## Exercise 2 — Cluster test

```python
import pytest
from lesson_5_decay_forgetting import MemoryManager, MemoryRecord, mock_embed
from datetime import datetime, timedelta, timezone
from uuid import uuid4


def test_clustering_two_pairs():
    manager = MemoryManager(cluster_similarity=0.85)

    now = datetime.now(timezone.utc)

    # Two records that will produce the same mock_embed hash (same text)
    # and two others that are different
    # Use different texts to ensure they cluster by similarity

    records = [
        MemoryRecord(
            id=uuid4(),
            content="User prefers PDF reports",
            vector=mock_embed("User prefers PDF reports"),
            importance=0.3,
            created_at=now - timedelta(hours=200),
            last_accessed=now - timedelta(hours=200),
        ),
        MemoryRecord(
            id=uuid4(),
            content="User prefers PDF reports",   # same embed vector
            vector=mock_embed("User prefers PDF reports"),
            importance=0.3,
            created_at=now - timedelta(hours=200),
            last_accessed=now - timedelta(hours=200),
        ),
        MemoryRecord(
            id=uuid4(),
            content="dark mode enabled",
            vector=mock_embed("dark mode enabled"),
            importance=0.3,
            created_at=now - timedelta(hours=200),
            last_accessed=now - timedelta(hours=200),
        ),
        MemoryRecord(
            id=uuid4(),
            content="dark mode enabled",  # same embed vector
            vector=mock_embed("dark mode enabled"),
            importance=0.3,
            created_at=now - timedelta(hours=200),
            last_accessed=now - timedelta(hours=200),
        ),
    ]

    clusters = manager._cluster(records)
    assert len(clusters) == 2
    assert all(len(c) == 2 for c in clusters)
```

---

## Exercise 3 — Mem0 vs Letta architectural difference

**Mem0** handles memory automatically: at each turn, an LLM call extracts facts from
the conversation and writes them to the store without the agent explicitly requesting it.
The agent never "thinks" about memory — it is managed externally.

**Letta** makes memory management explicit: the agent (LLM) decides when to write or
retrieve memories by calling internal tools (`core_memory_append`, `archival_memory_search`).
This produces auditable, reasoned memory operations — the agent can explain *why* it
stored something.

**Choose Letta when**: the agent must reason over long-horizon tasks where memory
decisions are non-trivial and auditability matters (e.g., a research assistant that
must track contradictions across months of notes).

---

## Exercise 4 — `core_memory_append` as a LangGraph tool

```python
from typing import Any
from langgraph.prebuilt import ToolNode

# The core memory is stored in agent state
class AgentState(TypedDict):
    messages: list[dict]
    core_memory: dict[str, str]   # e.g. {"human": "...", "persona": "..."}


def core_memory_append(
    state: AgentState,
    section: str,
    content: str,
) -> AgentState:
    """
    Tool the LLM calls to append a fact to a named core memory section.

    Args:
        section: memory section name, e.g. "human" or "persona"
        content: the fact to append
    """
    existing = state["core_memory"].get(section, "")
    separator = "\n" if existing else ""
    state["core_memory"][section] = existing + separator + content
    log.info("core_memory_append section=%s content=%s", section, content[:60])
    return state
```

This can be registered as a LangGraph tool by wrapping the state mutation in a
`ToolNode`. The LLM produces a tool call with `{"section": "human", "content": "..."}`,
and the node applies the update to the state dict.
