# Elective: Long-Term Memory · Lesson 1 — Why Agents Need Persistent Memory & Memory Types

> Objective: Understand the memory problem in LLM agents and map the four cognitive memory types to agent needs.
> Time: ~60 min · Prereqs: [[README.md]]

---

## Concept

### The context-window trap

Every LLM call is stateless. The model receives tokens, returns tokens, and forgets
everything. The "context window" (128k, 200k tokens) is not memory — it is a
**working scratchpad that is destroyed at the end of each call**.

This creates concrete failure modes in agents:

| Failure | Root cause |
|---------|-----------|
| Agent asks a returning user their name again | No episodic store |
| Agent contradicts advice it gave last week | No semantic consistency |
| Agent relearns a custom tool on every session | No procedural store |
| Agent cannot summarise a 6-month project thread | Context limit exceeded |

### Four memory types (from cognitive science → agents)

Cognitive science identifies four memory systems. Each maps cleanly to an agent need.

#### 1. Working memory
**What it is:** The immediately active mental workspace. Limited capacity, fast access.  
**In agents:** The current context window — messages, tool results, scratchpad variables.  
**Lifespan:** Single request or conversation turn.  
**Agent use:** Reasoning steps, intermediate tool outputs, current task state.

#### 2. Episodic memory
**What it is:** Memory of specific past events — *what happened, when, with whom*.  
**In agents:** Records of past conversations, decisions, actions, outcomes.  
**Lifespan:** Persistent across sessions.  
**Agent use:** "Last time we talked, you mentioned X." Personalisation, continuity.

#### 3. Semantic memory
**What it is:** General facts, concepts, and world knowledge — *what things mean*.  
**In agents:** Extracted facts about users, entities, relationships. Knowledge graphs.  
**Lifespan:** Long-lived; updated when contradicted.  
**Agent use:** "Your preferred currency is EUR." "The project deadline is Q3."

#### 4. Procedural memory
**What it is:** How to do things — skills encoded as automatic processes.  
**In agents:** System prompts, few-shot examples, tool schemas, refined workflows.  
**Lifespan:** Persistent; versioned.  
**Agent use:** "When the user asks for a summary, use the following format…"

### Why this taxonomy matters for engineering

Each type needs a different storage strategy:

| Type | Access pattern | Best store |
|------|---------------|-----------|
| Working | In-flight, must be fast | In-memory / context |
| Episodic | Sequential + similarity search | Vector store |
| Semantic | Structured lookup + graph traversal | KV store + graph |
| Procedural | Exact key lookup, versioned | KV store / file |

---

## How it works internally

### Memory as a pipeline

Storing and retrieving memory is not one operation — it is a pipeline:

```
Conversation turn
    │
    ▼
[Working memory] ── summarise ──▶ [Episodic store]
    │                                      │
    ▼                                      ▼
[Extract facts] ─────────────────▶ [Semantic store]
    │
    ▼
[Detect workflow pattern] ────────▶ [Procedural store]
```

### The three forces on memory design

1. **Capacity vs. recall**: More stored = more noise at retrieval time.
2. **Staleness vs. forgetting**: Old memories may be wrong; deleting too early loses context.
3. **Latency vs. completeness**: Querying 100k memories takes time; querying 10 may miss the best match.

All memory system designs are trade-offs between these three forces.

---

## Worked example

```python
"""
lesson_1_memory_types.py

Demonstrates the four memory types as typed Pydantic models.
No LLM calls — pure data modelling.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ── Memory type enum ──────────────────────────────────────────────────────────

class MemoryType(StrEnum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


# ── Base memory record ────────────────────────────────────────────────────────

class MemoryRecord(BaseModel):
    """A single memory entry regardless of type."""

    id: UUID = Field(default_factory=uuid4)
    memory_type: MemoryType
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def touch(self) -> None:
        """Update access stats on retrieval."""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1


# ── Specialised subtypes ──────────────────────────────────────────────────────

class EpisodicMemory(MemoryRecord):
    """A past event: conversation, action, or outcome."""

    memory_type: MemoryType = MemoryType.EPISODIC
    session_id: str = ""
    participants: list[str] = Field(default_factory=list)


class SemanticMemory(MemoryRecord):
    """A durable fact or belief about the world."""

    memory_type: MemoryType = MemoryType.SEMANTIC
    subject: str = ""       # entity the fact is about
    predicate: str = ""     # relationship
    object_: str = ""       # value
    confidence: float = 1.0  # 0.0–1.0


class ProceduralMemory(MemoryRecord):
    """A skill, workflow, or reusable instruction."""

    memory_type: MemoryType = MemoryType.PROCEDURAL
    trigger: str = ""       # condition that activates this procedure
    version: int = 1


# ── Simple in-memory store (working memory for this example) ──────────────────

class WorkingMemory:
    """Represents the agent's live context — cleared each session."""

    def __init__(self, max_items: int = 20) -> None:
        self._store: list[MemoryRecord] = []
        self.max_items = max_items

    def add(self, record: MemoryRecord) -> None:
        if len(self._store) >= self.max_items:
            evicted = self._store.pop(0)
            log.info("Working memory full — evicted: %s", evicted.content[:40])
        self._store.append(record)
        log.info("Working memory: added %s record", record.memory_type)

    def all(self) -> list[MemoryRecord]:
        return list(self._store)

    def clear(self) -> None:
        self._store.clear()
        log.info("Working memory cleared")


# ── Demo ──────────────────────────────────────────────────────────────────────

def main() -> None:
    wm = WorkingMemory(max_items=3)

    # Episodic: something that happened
    ep = EpisodicMemory(
        content="User asked about quarterly sales report",
        session_id="sess_001",
        participants=["user", "agent"],
    )

    # Semantic: a durable fact extracted from the conversation
    sm = SemanticMemory(
        content="User's preferred report format is PDF",
        subject="user",
        predicate="prefers_format",
        object_="PDF",
    )

    # Procedural: a reusable workflow
    pm = ProceduralMemory(
        content="When asked for a report, always include an executive summary first.",
        trigger="generate report",
        version=1,
    )

    for record in (ep, sm, pm):
        wm.add(record)

    log.info("Working memory contents:")
    for r in wm.all():
        log.info("  [%s] %s", r.memory_type, r.content[:60])


if __name__ == "__main__":
    main()
```

Run it:

```bash
python lesson_1_memory_types.py
```

Expected output (timestamps vary):

```
INFO:__main__:Working memory: added episodic record
INFO:__main__:Working memory: added semantic record
INFO:__main__:Working memory: added procedural record
INFO:__main__:Working memory contents:
INFO:__main__:  [episodic] User asked about quarterly sales report
INFO:__main__:  [semantic] User's preferred report format is PDF
INFO:__main__:  [procedural] When asked for a report, always include an executive
```

---

## Exercises

1. Add a `source` field to `MemoryRecord` (e.g. `"llm_extraction"`, `"user_explicit"`,
   `"tool_output"`). Update `EpisodicMemory` to set `source = "llm_extraction"` by default.

2. `WorkingMemory` currently evicts the oldest item (FIFO). Implement an LRU eviction
   strategy instead: when capacity is exceeded, remove the record with the oldest
   `last_accessed` timestamp.

3. Write a function `classify_memory(text: str) -> MemoryType` that uses simple keyword
   heuristics (no LLM) to classify a raw string into one of the four types. For example,
   text containing "last time" or "yesterday" → `EPISODIC`; "always" or "procedure" →
   `PROCEDURAL`. Write three pytest cases.

4. Sketch (prose or diagram) how you would persist `EpisodicMemory` records across process
   restarts. What file format or database would you choose? Why?

(Solutions: [solutions/lesson-1.md](solutions/lesson-1.md))

---

## Self-check

- [ ] I can name the four memory types and give one agent use case for each
- [ ] I understand why the context window is *not* long-term memory
- [ ] I can model a memory record as a Pydantic type with appropriate fields
- [ ] I can explain the three engineering trade-offs in memory system design
