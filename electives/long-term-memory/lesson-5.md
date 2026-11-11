# Elective: Long-Term Memory · Lesson 5 — Forgetting, Decay, and Mem0 / Letta

> Objective: Implement decay and forgetting strategies, then understand Mem0 and Letta/MemGPT's memory architectures at the concept and API level.
> Time: ~75 min · Prereqs: [[lesson-4.md]]

---

## Concept

### Why forgetting is necessary

A memory store that only grows has three failure modes:

1. **Noise accumulation**: outdated facts contradict new ones and confuse the retriever
2. **Latency creep**: larger stores mean slower candidate retrieval
3. **Stale personalisation**: a user's preferences from 3 years ago may no longer apply

Forgetting is **selective amnesia**: remove or down-weight memories that are unlikely
to be useful again.

### Decay strategies

#### 1. Exponential decay (time-based)

We already use this in recency scoring. As a *forgetting* mechanism, memories whose
composite retrieval score drops below a threshold are marked for archival or deletion.

```python
if composite_score(record, ...) < ARCHIVE_THRESHOLD:
    archive(record)
```

#### 2. Access-based reinforcement

Memories retrieved frequently are "strengthened" — their decay is slowed by resetting
`last_accessed`. Memories never retrieved decay faster.

This mirrors the **spacing effect** in human memory: recalling something reinforces it.

#### 3. TTL (time-to-live)

Explicit expiry on records that are inherently transient:

```python
record.expires_at = datetime.now() + timedelta(days=7)
```

TTL is appropriate for: meeting reminders, short-term tasks, session state.

#### 4. Consolidation (merge and summarise)

Instead of deleting, **consolidate** related memories into a higher-level summary.
Example: 30 episodic records about "User asked report questions" → 1 semantic record
"User frequently requests PDF reports". This is analogous to how human episodic memories
consolidate into semantic knowledge during sleep.

### The forgetting pipeline

```
Scheduled job (e.g., nightly)
    │
    ├─[score all records]─────────────────► identify low-score candidates
    │
    ├─[cluster low-score records]─────────► group related low-importance memories
    │
    ├─[consolidate cluster]───────────────► single summary memory
    │
    └─[archive originals]─────────────────► soft-delete or move to cold storage
```

---

## Mem0

**Mem0** (mem0ai) is an open-source memory layer for LLM applications. It provides:

- A unified API over multiple memory types
- Automatic extraction of facts from conversations using an LLM
- Vector + KV + graph backends
- Built-in deduplication and merging

### Mem0 API shape

```python
# Installation: uv add mem0ai
# (Do not install for tests — mock instead)

from mem0 import Memory

# Initialise with default in-memory store
m = Memory()

# Add memories from a conversation
messages = [
    {"role": "user", "content": "I like Python and hate Java."},
    {"role": "assistant", "content": "Noted!"},
]
result = m.add(messages, user_id="alice")
# result = [{"id": "...", "memory": "Alice likes Python", ...}, ...]

# Search for relevant memories
hits = m.search("programming language preference", user_id="alice")
# hits = [{"id": "...", "memory": "Alice likes Python", "score": 0.92}, ...]

# Get all memories for a user
all_mems = m.get_all(user_id="alice")

# Delete a memory
m.delete(memory_id="...", user_id="alice")

# Update a memory
m.update(memory_id="...", data="Alice prefers Python and Rust", user_id="alice")
```

Key design decisions in Mem0:
- LLM-powered extraction: uses a cheap model (configurable) to extract facts
- Dedup by merging: when a new fact contradicts an existing one, the LLM merges them
- Pluggable backends: swap vector store (Chroma, Qdrant, Pinecone), graph store, LLM

Mem0 is suitable when you want an out-of-the-box memory layer and are comfortable
with the LLM extraction cost on every write.

---

## Letta / MemGPT

**Letta** (formerly MemGPT) implements a **memory hierarchy** with explicit paging,
inspired by virtual memory in operating systems.

### The MemGPT insight

LLM context windows are like CPU registers — limited but fast. Long-term storage is
like disk — unlimited but requires an explicit I/O operation to page in.

MemGPT gives the LLM **tools to manage its own memory**:

```
┌─────────────────────────────────────────┐
│  Main context (context window = "RAM")  │
│  ┌─────────────┐  ┌──────────────────┐  │
│  │ core memory │  │  conversation    │  │
│  │ (always in) │  │  (recent turns)  │  │
│  └─────────────┘  └──────────────────┘  │
└─────────────────────────────────────────┘
         ▲                    ▲
         │ page in            │ summarise out
         ▼                    ▼
┌─────────────────────────────────────────┐
│  Archival storage ("disk")              │
│  (vector-indexed, unlimited capacity)  │
└─────────────────────────────────────────┘
```

### Memory tiers

| Tier | Contents | Size | LLM access |
|------|----------|------|-----------|
| **Core memory** | Name, key facts, persona | ~1–2k tokens | Always in context |
| **Recall memory** | Recent conversation history | ~5–10k tokens | Auto-appended |
| **Archival memory** | Past conversations, knowledge | Unlimited | Via tool call (`archival_memory_search`) |

### Self-editing memory

The LLM agent can call internal tools to modify its own memory:

```
core_memory_append(name="human", content="Prefers Rust over Python")
core_memory_replace(name="human", old_content="...", new_content="...")
archival_memory_insert("User completed onboarding on 2024-03-01")
archival_memory_search("user programming preferences")
```

This makes memory management **explicit and auditable**: you can inspect exactly what
the agent knows and when it wrote it.

### Letta API shape

```python
# Installation: uv add letta
# (Do not install for tests — mock instead)

from letta import create_client

client = create_client()

# Create an agent with a memory persona
agent = client.create_agent(
    name="my_agent",
    system="You are a helpful assistant. Use your memory tools.",
)

# Send a message — Letta handles all memory I/O internally
response = client.send_message(
    agent_id=agent.id,
    role="user",
    message="My name is Alice and I prefer Python.",
)

# Inspect core memory
core = client.get_core_memory(agent_id=agent.id)
print(core.human)     # what the agent knows about the user
print(core.persona)   # agent's own self-description

# Inspect archival memory
archival = client.get_archival_memory(agent_id=agent.id)
```

Letta is suitable when you want the agent to explicitly reason about its own memory
and you need fine-grained auditability of what is stored and why.

---

## Mem0 vs Letta — decision guide

| Dimension | Mem0 | Letta |
|-----------|------|-------|
| Memory management | Automatic (LLM extracts) | Explicit (agent uses tools) |
| Auditability | Medium | High |
| Customisation | Moderate | High |
| Latency | Lower (background) | Higher (in-context tool calls) |
| Best for | Simple apps, fast integration | Complex agents, long-running tasks |

---

## Worked example

```python
"""
lesson_5_decay_forgetting.py

Implements time-based decay with archival threshold,
plus consolidation of related low-importance memories.
LLM consolidation is mocked.
"""
from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable
from uuid import UUID, uuid4

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class MemoryRecord:
    id: UUID
    content: str
    vector: list[float]
    importance: float
    created_at: datetime
    last_accessed: datetime
    archived: bool = False
    metadata: dict = field(default_factory=dict)

    def touch(self) -> None:
        self.last_accessed = datetime.now(timezone.utc)


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na * nb > 1e-9 else 0.0


def decay_score(record: MemoryRecord, decay_rate: float = 0.01) -> float:
    """Combined score used to decide archival. Uses recency × importance."""
    now = datetime.now(timezone.utc)
    hours_old = (now - record.last_accessed).total_seconds() / 3600
    recency = math.exp(-decay_rate * hours_old)
    return recency * record.importance


# ── Consolidation ─────────────────────────────────────────────────────────────

ConsolidateFn = Callable[[list[MemoryRecord]], str]


def mock_consolidate(records: list[MemoryRecord]) -> str:
    """Mock consolidation: join first 40 chars of each record."""
    snippets = [r.content[:40] for r in records]
    return "Consolidated: " + "; ".join(snippets)


# ── Memory manager with forgetting ───────────────────────────────────────────

class MemoryManager:
    """
    Maintains a memory store and runs periodic forgetting passes.
    """

    def __init__(
        self,
        archive_threshold: float = 0.05,
        cluster_similarity: float = 0.80,
        consolidate_fn: ConsolidateFn = mock_consolidate,
        decay_rate: float = 0.01,
    ) -> None:
        self._records: list[MemoryRecord] = []
        self.archive_threshold = archive_threshold
        self.cluster_similarity = cluster_similarity
        self.consolidate_fn = consolidate_fn
        self.decay_rate = decay_rate

    def add(self, record: MemoryRecord) -> None:
        self._records.append(record)

    def active_records(self) -> list[MemoryRecord]:
        return [r for r in self._records if not r.archived]

    def run_forgetting_pass(self) -> dict[str, int]:
        """
        1. Score all active records.
        2. Identify candidates below archive_threshold.
        3. Cluster candidates by similarity.
        4. Consolidate each cluster → new record.
        5. Archive originals.
        Returns stats dict.
        """
        active = self.active_records()
        scores = {r.id: decay_score(r, self.decay_rate) for r in active}

        candidates = [r for r in active if scores[r.id] < self.archive_threshold]
        log.info("Forgetting pass: %d/%d records below threshold", len(candidates), len(active))

        if not candidates:
            return {"archived": 0, "consolidated": 0}

        # Cluster candidates
        clusters = self._cluster(candidates)
        log.info("Forgetting pass: %d cluster(s) formed", len(clusters))

        consolidated = 0
        for cluster in clusters:
            if len(cluster) < 2:
                # Single-record cluster: archive directly
                cluster[0].archived = True
            else:
                # Consolidate into a new record
                summary = self.consolidate_fn(cluster)
                max_importance = max(r.importance for r in cluster)
                new_record = MemoryRecord(
                    id=uuid4(),
                    content=summary,
                    vector=cluster[0].vector,  # use first record's vector as proxy
                    importance=max_importance * 0.8,  # consolidated = slightly less important
                    created_at=datetime.now(timezone.utc),
                    last_accessed=datetime.now(timezone.utc),
                    metadata={"consolidated_from": [str(r.id) for r in cluster]},
                )
                self._records.append(new_record)
                for r in cluster:
                    r.archived = True
                consolidated += 1
                log.info("Consolidated %d records → '%s'", len(cluster), summary[:60])

        archived_count = sum(1 for r in self._records if r.archived)
        return {"archived": archived_count, "consolidated": consolidated}

    def _cluster(self, records: list[MemoryRecord]) -> list[list[MemoryRecord]]:
        """Simple greedy cosine-similarity clustering."""
        clusters: list[list[MemoryRecord]] = []
        used = set()
        for i, r in enumerate(records):
            if r.id in used:
                continue
            cluster = [r]
            used.add(r.id)
            for j, r2 in enumerate(records):
                if j <= i or r2.id in used:
                    continue
                if cosine_sim(r.vector, r2.vector) >= self.cluster_similarity:
                    cluster.append(r2)
                    used.add(r2.id)
            clusters.append(cluster)
        return clusters


# ── Mock embedder ─────────────────────────────────────────────────────────────

def mock_embed(text: str, dim: int = 8) -> list[float]:
    digest = hashlib.md5(text.encode()).digest()
    raw = [float(digest[i % 16]) / 255.0 for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


# ── Demo ──────────────────────────────────────────────────────────────────────

def main() -> None:
    manager = MemoryManager(archive_threshold=0.15, decay_rate=0.5)

    now = datetime.now(timezone.utc)

    records = [
        MemoryRecord(
            id=uuid4(),
            content="User prefers PDF reports",
            vector=mock_embed("User prefers PDF reports"),
            importance=0.9,
            created_at=now - timedelta(hours=1),
            last_accessed=now - timedelta(hours=1),
        ),
        MemoryRecord(
            id=uuid4(),
            content="User asked about PDFs once",
            vector=mock_embed("User asked about PDF exports"),
            importance=0.2,
            created_at=now - timedelta(hours=100),
            last_accessed=now - timedelta(hours=100),
        ),
        MemoryRecord(
            id=uuid4(),
            content="User mentioned PDF as format",
            vector=mock_embed("User mentioned PDF format preference"),
            importance=0.15,
            created_at=now - timedelta(hours=120),
            last_accessed=now - timedelta(hours=120),
        ),
    ]

    for r in records:
        manager.add(r)

    log.info("Before forgetting: %d active records", len(manager.active_records()))
    stats = manager.run_forgetting_pass()
    log.info("After forgetting: %d active records | stats=%s",
             len(manager.active_records()), stats)


if __name__ == "__main__":
    main()
```

---

## Exercises

1. Add an `expires_at: datetime | None = None` field to `MemoryRecord`. Extend
   `MemoryManager.run_forgetting_pass()` to archive any record whose `expires_at` is in
   the past, before the decay-score pass.

2. The `_cluster` method uses cosine similarity. Write a pytest test that adds 4 records
   — two semantically similar pairs — and verifies that `_cluster` returns exactly 2
   clusters of 2.

3. Describe (max 100 words) the key architectural difference between how Mem0 and Letta
   handle memory writes. In which scenario would you choose Letta over Mem0?

4. Sketch (code or prose) how you would implement `core_memory_append` as a LangGraph
   tool that the LLM can call to write a fact to its own core memory dict.

(Solutions: [solutions/lesson-5.md](solutions/lesson-5.md))

---

## Self-check

- [ ] I can implement exponential decay scoring and identify archival candidates
- [ ] I can implement greedy clustering for consolidation
- [ ] I can describe Mem0's extraction-based and Letta's tool-based memory approaches
- [ ] I understand the memory hierarchy (core / recall / archival) in Letta
