# Solutions: Lesson 1 — Why Agents Need Persistent Memory & Memory Types

---

## Exercise 1 — `source` field

```python
from enum import StrEnum

class MemorySource(StrEnum):
    LLM_EXTRACTION = "llm_extraction"
    USER_EXPLICIT = "user_explicit"
    TOOL_OUTPUT = "tool_output"
    UNKNOWN = "unknown"


class MemoryRecord(BaseModel):
    ...
    source: MemorySource = MemorySource.UNKNOWN


class EpisodicMemory(MemoryRecord):
    memory_type: MemoryType = MemoryType.EPISODIC
    source: MemorySource = MemorySource.LLM_EXTRACTION   # default override
    session_id: str = ""
    participants: list[str] = Field(default_factory=list)
```

---

## Exercise 2 — LRU eviction

Replace `self._store.pop(0)` (FIFO) with:

```python
def add(self, record: MemoryRecord) -> None:
    if len(self._store) >= self.max_items:
        # LRU: evict the record with the oldest last_accessed
        lru = min(self._store, key=lambda r: r.last_accessed)
        self._store.remove(lru)
        log.info("Working memory LRU evict: %s", lru.content[:40])
    self._store.append(record)
```

Test:

```python
import time
from datetime import datetime, timezone, timedelta

def test_lru_eviction():
    wm = WorkingMemory(max_items=2)
    now = datetime.now(timezone.utc)

    old = EpisodicMemory(content="old record")
    old.last_accessed = now - timedelta(hours=2)

    recent = EpisodicMemory(content="recent record")
    recent.last_accessed = now

    wm._store = [old, recent]

    new_record = EpisodicMemory(content="new record that triggers eviction")
    wm.add(new_record)

    contents = [r.content for r in wm.all()]
    assert "old record" not in contents
    assert "recent record" in contents
    assert "new record that triggers eviction" in contents
```

---

## Exercise 3 — `classify_memory`

```python
from memory_types import MemoryType


def classify_memory(text: str) -> MemoryType:
    lower = text.lower()
    if any(kw in lower for kw in ["last time", "yesterday", "previously", "last week", "last session"]):
        return MemoryType.EPISODIC
    if any(kw in lower for kw in ["always", "procedure", "step 1", "workflow", "whenever", "follow these steps"]):
        return MemoryType.PROCEDURAL
    if any(kw in lower for kw in ["prefers", "likes", "is called", "works at", "the deadline is", "born in"]):
        return MemoryType.SEMANTIC
    return MemoryType.WORKING


# Tests
import pytest

@pytest.mark.parametrize("text,expected", [
    ("Last time we talked about reports", MemoryType.EPISODIC),
    ("Always start with an executive summary", MemoryType.PROCEDURAL),
    ("The user prefers PDF format", MemoryType.SEMANTIC),
    ("Hello there!", MemoryType.WORKING),
])
def test_classify_memory(text: str, expected: MemoryType):
    assert classify_memory(text) == expected
```

---

## Exercise 4 — Persistence sketch

**Format**: JSON Lines (`.jsonl`), one `MemoryRecord` JSON object per line.  
**Rationale**: Human-readable, appendable without loading the full file, easy to parse
with standard Python (`json` module), no external database dependency.

```python
# Writing
import json
from pathlib import Path

def persist(records: list[EpisodicMemory], path: Path) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")

# Loading
def load(path: Path) -> list[EpisodicMemory]:
    records = []
    with path.open() as f:
        for line in f:
            records.append(EpisodicMemory.model_validate_json(line))
    return records
```

For a production system with concurrent writes, a SQLite database with
`journal_mode=WAL` would be more appropriate than a flat file.
