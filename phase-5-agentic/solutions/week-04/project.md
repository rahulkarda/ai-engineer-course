# Week 04 Project — Reference Solution: Pluggable Agent Memory System

## File Layout

```
src/
  memory_system/
    __init__.py
    models.py          # AgentMemoryState (Pydantic)
    buffer.py          # BufferMemory
    summary.py         # SummaryMemory
    vector.py          # VectorSemanticMemory
    episodic.py        # EpisodicMemory
    system.py          # PluggableMemorySystem (orchestrator)
    persistence.py     # StatePersistenceManager
tests/
  conftest.py
  test_state_round_trip.py
  test_buffer_eviction.py
  test_summarization_trigger.py
  test_semantic_search.py
  test_session_persistence.py
pyproject.toml
.env.example
```

---

## src/memory_system/models.py

```python
"""Shared Pydantic models for the memory system."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class AgentMemoryState(BaseModel):
    """
    Complete typed state for one agent session.

    All fields have defaults so construction is ergonomic:
        state = AgentMemoryState()
    """

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: int = 1
    messages: list[dict[str, Any]] = Field(default_factory=list)
    summary: str | None = None
    facts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
```

---

## src/memory_system/buffer.py

```python
"""Short-term buffer memory with FIFO eviction."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class BufferMemory:
    """
    Fixed-size ring buffer of recent messages.

    Eviction policy: FIFO (oldest message dropped when max_size exceeded).
    """

    def __init__(self, max_size: int = 10) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self._max_size = max_size
        self._messages: list[dict[str, Any]] = []

    def add(self, message: dict[str, Any]) -> None:
        self._messages.append(message)
        while len(self._messages) > self._max_size:
            evicted = self._messages.pop(0)
            logger.debug("Buffer evicted: %r", str(evicted)[:60])

    def messages(self) -> list[dict[str, Any]]:
        return list(self._messages)

    def size(self) -> int:
        return len(self._messages)

    def clear(self) -> None:
        self._messages.clear()
```

---

## src/memory_system/summary.py

```python
"""Summarization memory: batches messages and stores compressed summaries."""
from __future__ import annotations

import logging
import os
from typing import Callable

logger = logging.getLogger(__name__)

SummarizeFn = Callable[[list[dict]], str]


def _build_default_summarize_fn() -> SummarizeFn:
    """
    Return a real or mock summarizer based on ANTHROPIC_API_KEY presence.

    In production: calls claude-sonnet-5-5 via the Anthropic SDK.
    In tests: no API key is set, so a deterministic fallback is used.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning(
            "ANTHROPIC_API_KEY not set; using offline summarizer. "
            "Set the key for production summarization."
        )

        def _offline_summarize(messages: list[dict]) -> str:
            parts = [f"{m.get('role','?')}: {m.get('content','')[:40]}" for m in messages]
            return "[OFFLINE SUMMARY] " + " | ".join(parts)

        return _offline_summarize

    def _llm_summarize(messages: list[dict]) -> str:
        import anthropic  # only imported if key present

        client = anthropic.Anthropic(api_key=api_key)
        conversation = "\n".join(
            f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in messages
        )
        response = client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Summarize the following conversation in 2-3 sentences:\n\n{conversation}"
                    ),
                }
            ],
        )
        return response.content[0].text

    return _llm_summarize


class SummaryMemory:
    """
    Accumulates messages and produces summaries when a trigger count is reached.

    Design: once trigger is hit, pending messages are summarized and cleared.
    Summaries accumulate in _summaries indefinitely (or until cleared).
    """

    def __init__(
        self,
        trigger: int = 5,
        summarize_fn: SummarizeFn | None = None,
    ) -> None:
        self._trigger = trigger
        self._summarize_fn: SummarizeFn = summarize_fn or _build_default_summarize_fn()
        self._pending: list[dict] = []
        self._summaries: list[str] = []

    def add(self, message: dict) -> None:
        self._pending.append(message)
        if len(self._pending) >= self._trigger:
            self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        summary = self._summarize_fn(list(self._pending))
        self._summaries.append(summary)
        logger.info("SummaryMemory: flushed %d messages to summary", len(self._pending))
        self._pending.clear()

    def summaries(self) -> list[str]:
        return list(self._summaries)

    def summary_count(self) -> int:
        return len(self._summaries)

    def pending_count(self) -> int:
        return len(self._pending)
```

---

## src/memory_system/vector.py

```python
"""Vector-based semantic memory with cosine similarity retrieval."""
from __future__ import annotations

import logging
import math
import os
from typing import Callable

logger = logging.getLogger(__name__)

EmbedFn = Callable[[str], list[float]]


def _hash_embed(text: str) -> list[float]:
    """
    Deterministic 8-dim embedding from hash.

    Used when no real embedding model is configured.
    NOT for production — lacks semantic meaning beyond hash collisions.
    """
    h = abs(hash(text)) % (10 ** 12)
    vec = [(h >> (i * 8)) % 256 / 255.0 for i in range(8)]
    mag = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / mag for v in vec]


def _build_default_embed_fn() -> EmbedFn:
    """Return a real or mock embed function based on environment."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning(
            "ANTHROPIC_API_KEY not set; using hash-based mock embeddings. "
            "Set the key for semantic similarity."
        )
        return _hash_embed

    def _real_embed(text: str) -> list[float]:
        # Production: use an embedding model (e.g. sentence-transformers or
        # a third-party embeddings API). Claude does not expose an embed endpoint,
        # so use e.g. openai or cohere embeddings, or sentence-transformers locally.
        # Example placeholder:
        raise NotImplementedError(
            "Provide a real embed function; Anthropic does not expose embeddings."
        )

    return _real_embed


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a)) or 1e-9
    mag_b = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (mag_a * mag_b)


class VectorSemanticMemory:
    """
    Stores text as embeddings and retrieves the top-k most similar results.

    In production: swap _embed_fn for a real sentence embedding model.
    In tests: use the default hash-based embedding for determinism.
    """

    def __init__(self, embed_fn: EmbedFn | None = None) -> None:
        self._embed_fn: EmbedFn = embed_fn or _build_default_embed_fn()
        self._store: list[tuple[str, list[float]]] = []

    def add(self, text: str) -> None:
        emb = self._embed_fn(text)
        self._store.append((text, emb))
        logger.debug("VectorMemory: stored %r", text[:50])

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        if not self._store:
            return []
        q_emb = self._embed_fn(query)
        ranked = sorted(
            self._store,
            key=lambda item: cosine_similarity(q_emb, item[1]),
            reverse=True,
        )
        return [text for text, _ in ranked[:k]]

    def count(self) -> int:
        return len(self._store)
```

---

## src/memory_system/episodic.py

```python
"""Episodic memory: time-stamped events with optional tags."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Episode(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tags: list[str] = Field(default_factory=list)


class EpisodicMemory:
    """
    Ordered store of episodes (events with timestamps).

    Supports recency-based retrieval and tag filtering.
    """

    def __init__(self) -> None:
        self._episodes: list[Episode] = []

    def add(self, content: str, tags: list[str] | None = None) -> Episode:
        ep = Episode(content=content, tags=tags or [])
        self._episodes.append(ep)
        logger.debug("EpisodicMemory: added episode id=%s", ep.id)
        return ep

    def get_recent(self, n: int) -> list[Episode]:
        """Return the N most recent episodes, newest first."""
        return list(reversed(self._episodes))[:n]

    def search_by_tag(self, tag: str) -> list[Episode]:
        tag_lower = tag.lower()
        return [ep for ep in self._episodes if tag_lower in (t.lower() for t in ep.tags)]

    def all(self) -> list[Episode]:
        return list(self._episodes)

    def count(self) -> int:
        return len(self._episodes)
```

---

## src/memory_system/system.py

```python
"""PluggableMemorySystem: orchestrates all memory backends."""
from __future__ import annotations

import logging

from memory_system.buffer import BufferMemory
from memory_system.episodic import EpisodicMemory
from memory_system.summary import SummaryMemory
from memory_system.vector import VectorSemanticMemory

logger = logging.getLogger(__name__)


class PluggableMemorySystem:
    """
    Central orchestrator for agent memory.

    Routes each add() call to the appropriate backends based on priority.
    Provides a unified retrieve() interface that queries the vector store.

    Backends:
    - buffer:   short-term, bounded, fast
    - summary:  batch compression of older messages
    - vector:   long-term semantic retrieval
    - episodic: time-stamped event log
    """

    def __init__(
        self,
        buffer: BufferMemory,
        summary: SummaryMemory,
        vector: VectorSemanticMemory,
        episodic: EpisodicMemory,
    ) -> None:
        self._buffer = buffer
        self._summary = summary
        self._vector = vector
        self._episodic = episodic
        self._retrieval_count = 0

    # ------------------------------------------------------------------ write

    def add(self, text: str, priority: int = 0) -> None:
        """
        Add a memory to the system.

        priority > 0: skip buffer/summary, route directly to vector.
        priority == 0 (default): add to all backends.
        """
        if priority > 0:
            self._vector.add(text)
            logger.debug("High-priority add to vector: %r", text[:50])
        else:
            message = {"role": "system", "content": text}
            self._buffer.add(message)
            self._summary.add(message)
            self._vector.add(text)
            logger.debug("Normal add to all backends: %r", text[:50])

    def record_episode(self, content: str, tags: list[str] | None = None) -> None:
        """Add a time-stamped episodic event."""
        self._episodic.add(content, tags=tags)

    # ------------------------------------------------------------------ read

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        self._retrieval_count += 1
        return self._vector.retrieve(query, k=k)

    def recent_buffer(self) -> list[dict]:
        return self._buffer.messages()

    def recent_episodes(self, n: int = 5):
        return self._episodic.get_recent(n)

    # ------------------------------------------------------------------ stats

    def stats(self) -> dict:
        return {
            "buffer_size": self._buffer.size(),
            "summary_count": self._summary.summary_count(),
            "vector_count": self._vector.count(),
            "episodic_count": self._episodic.count(),
            "total_retrievals": self._retrieval_count,
        }

    # ------------------------------------------------------------------ snapshot

    def export_snapshot(self) -> dict:
        return {
            "buffer": list(self._buffer._messages),
            "summaries": self._summary.summaries(),
            "pending_summary": list(self._summary._pending),
            "vector": [
                {"text": t, "embedding": e} for t, e in self._vector._store
            ],
            "episodic": [ep.model_dump(mode="json") for ep in self._episodic.all()],
            "retrieval_count": self._retrieval_count,
        }

    def import_snapshot(self, data: dict) -> None:
        from memory_system.episodic import Episode

        self._buffer._messages = list(data.get("buffer", []))
        self._summary._summaries = list(data.get("summaries", []))
        self._summary._pending = list(data.get("pending_summary", []))
        self._vector._store = [
            (entry["text"], entry["embedding"]) for entry in data.get("vector", [])
        ]
        self._episodic._episodes = [
            Episode.model_validate(ep) for ep in data.get("episodic", [])
        ]
        self._retrieval_count = data.get("retrieval_count", 0)
```

---

## src/memory_system/persistence.py

```python
"""StatePersistenceManager: JSON-based session storage."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memory_system.models import AgentMemoryState

logger = logging.getLogger(__name__)


class StatePersistenceManager:
    """
    Saves and loads AgentMemoryState to a directory of per-session JSON files.

    Uses atomic write (tmp + rename) to avoid partial-write corruption.
    Maintains a _index.json for quick session enumeration.
    """

    def __init__(self, storage_dir: Path) -> None:
        self._dir = storage_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ paths

    def _session_path(self, session_id: str) -> Path:
        return self._dir / f"{session_id}.json"

    @property
    def _index_path(self) -> Path:
        return self._dir / "_index.json"

    # ------------------------------------------------------------------ index

    def _read_index(self) -> dict[str, str]:
        if not self._index_path.exists():
            return {}
        return json.loads(self._index_path.read_text(encoding="utf-8"))

    def _write_index(self, index: dict[str, str]) -> None:
        self._index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    def _update_index(self, session_id: str, updated_at: datetime) -> None:
        index = self._read_index()
        index[session_id] = updated_at.isoformat()
        self._write_index(index)

    # ------------------------------------------------------------------ core

    def save(self, state: AgentMemoryState) -> Path:
        state = state.model_copy(update={"updated_at": datetime.now(timezone.utc)})
        final = self._session_path(state.session_id)
        tmp = final.with_suffix(".tmp")
        try:
            tmp.write_text(state.model_dump_json(indent=2), encoding="utf-8")
            tmp.replace(final)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        self._update_index(state.session_id, state.updated_at)
        logger.debug("Saved session %s", state.session_id)
        return final

    def load(self, session_id: str) -> AgentMemoryState:
        path = self._session_path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"Session {session_id!r} not found")
        data = json.loads(path.read_text(encoding="utf-8"))
        return AgentMemoryState.model_validate(data)

    def delete(self, session_id: str) -> None:
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()
        index = self._read_index()
        index.pop(session_id, None)
        self._write_index(index)
        logger.debug("Deleted session %s", session_id)

    def list_sessions(self) -> list[str]:
        return [p.stem for p in self._dir.glob("*.json") if p.stem != "_index"]
```

---

## tests/conftest.py

```python
"""Shared fixtures for Week 04 project tests."""
from __future__ import annotations

import math

import pytest

from memory_system.buffer import BufferMemory
from memory_system.episodic import EpisodicMemory
from memory_system.summary import SummaryMemory
from memory_system.vector import VectorSemanticMemory
from memory_system.system import PluggableMemorySystem


def mock_embed(text: str) -> list[float]:
    """Deterministic 8-dim hash embedding. No API calls."""
    h = abs(hash(text)) % (10 ** 12)
    vec = [(h >> (i * 8)) % 256 / 255.0 for i in range(8)]
    mag = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / mag for v in vec]


def mock_summarize(messages: list[dict]) -> str:
    parts = [m.get("content", "")[:30] for m in messages]
    return "[MOCK SUMMARY] " + " | ".join(parts)


@pytest.fixture
def memory_system() -> PluggableMemorySystem:
    return PluggableMemorySystem(
        buffer=BufferMemory(max_size=5),
        summary=SummaryMemory(trigger=3, summarize_fn=mock_summarize),
        vector=VectorSemanticMemory(embed_fn=mock_embed),
        episodic=EpisodicMemory(),
    )
```

---

## tests/test_state_round_trip.py

```python
"""Test 1: AgentMemoryState serializes and deserializes correctly."""
from __future__ import annotations

import json
from memory_system.models import AgentMemoryState


def test_state_round_trip():
    """State must survive a JSON round-trip with all fields intact."""
    original = AgentMemoryState(
        messages=[{"role": "user", "content": "hello"}],
        summary="prior summary",
        facts=["fact A", "fact B"],
        metadata={"env": "test"},
    )

    serialized = original.model_dump_json()
    restored = AgentMemoryState.model_validate_json(serialized)

    assert restored.session_id == original.session_id
    assert restored.messages == original.messages
    assert restored.summary == original.summary
    assert restored.facts == original.facts
    assert restored.metadata == original.metadata
    assert restored.schema_version == original.schema_version


def test_state_default_fields():
    """Default construction must not raise and IDs must be unique."""
    s1 = AgentMemoryState()
    s2 = AgentMemoryState()
    assert s1.session_id != s2.session_id
    assert s1.messages == []
    assert s1.summary is None
```

---

## tests/test_buffer_eviction.py

```python
"""Test 2: BufferMemory evicts oldest messages when max_size exceeded."""
from __future__ import annotations

from memory_system.buffer import BufferMemory


def test_buffer_eviction():
    """Buffer must not exceed max_size; oldest messages are evicted (FIFO)."""
    buf = BufferMemory(max_size=3)
    for i in range(5):
        buf.add({"role": "user", "content": f"msg {i}"})

    assert buf.size() == 3

    contents = [m["content"] for m in buf.messages()]
    # Most recent 3 messages should remain
    assert "msg 4" in contents
    assert "msg 3" in contents
    assert "msg 2" in contents
    # Oldest two should be gone
    assert "msg 0" not in contents
    assert "msg 1" not in contents


def test_buffer_eviction_single_capacity():
    buf = BufferMemory(max_size=1)
    buf.add({"role": "user", "content": "first"})
    buf.add({"role": "user", "content": "second"})
    assert buf.size() == 1
    assert buf.messages()[0]["content"] == "second"
```

---

## tests/test_summarization_trigger.py

```python
"""Test 3: SummaryMemory triggers summarization when pending >= trigger."""
from __future__ import annotations

from unittest.mock import MagicMock
from memory_system.summary import SummaryMemory


def test_summarization_trigger():
    """Summary must fire exactly when pending count reaches trigger."""
    mock_fn = MagicMock(return_value="[SUMMARY]")
    mem = SummaryMemory(trigger=3, summarize_fn=mock_fn)

    mem.add({"role": "user", "content": "one"})
    mem.add({"role": "user", "content": "two"})
    assert mock_fn.call_count == 0  # not yet triggered

    mem.add({"role": "user", "content": "three"})  # triggers at exactly 3
    assert mock_fn.call_count == 1
    assert mem.summary_count() == 1
    assert mem.pending_count() == 0  # pending cleared after flush


def test_summarization_no_trigger_below_threshold():
    mock_fn = MagicMock(return_value="[SUMMARY]")
    mem = SummaryMemory(trigger=5, summarize_fn=mock_fn)
    for _ in range(4):
        mem.add({"role": "user", "content": "x"})
    assert mock_fn.call_count == 0


def test_summarization_multiple_flushes():
    mock_fn = MagicMock(return_value="[SUMMARY]")
    mem = SummaryMemory(trigger=2, summarize_fn=mock_fn)
    for i in range(6):
        mem.add({"role": "user", "content": f"msg {i}"})
    assert mock_fn.call_count == 3
    assert mem.summary_count() == 3
```

---

## tests/test_semantic_search.py

```python
"""Test 4: VectorSemanticMemory retrieves the most relevant stored fact."""
from __future__ import annotations

import math
from memory_system.vector import VectorSemanticMemory


def _controlled_embed(text: str) -> list[float]:
    """
    Two semantic clusters:
    - texts containing 'python' → [1, 0, 0, 0]
    - texts containing 'database' → [0, 1, 0, 0]
    - others → hash-based
    """
    if "python" in text.lower():
        return [1.0, 0.0, 0.0, 0.0]
    if "database" in text.lower():
        return [0.0, 1.0, 0.0, 0.0]
    h = abs(hash(text)) % (10 ** 8)
    vec = [(h >> (i * 8)) % 256 / 255.0 for i in range(4)]
    mag = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / mag for v in vec]


def test_semantic_search():
    """Top-1 retrieval should return the semantically closest fact."""
    mem = VectorSemanticMemory(embed_fn=_controlled_embed)
    mem.add("Python is a programming language")
    mem.add("PostgreSQL is a relational database")
    mem.add("The sky is blue")

    # Query in the 'python' cluster
    results = mem.retrieve("python programming", k=1)
    assert results == ["Python is a programming language"]

    # Query in the 'database' cluster
    results_db = mem.retrieve("SQL database", k=1)
    assert results_db == ["PostgreSQL is a relational database"]


def test_semantic_search_empty_returns_empty():
    mem = VectorSemanticMemory(embed_fn=_controlled_embed)
    assert mem.retrieve("anything") == []
```

---

## tests/test_session_persistence.py

```python
"""Test 5: StatePersistenceManager saves, loads, and deletes sessions."""
from __future__ import annotations

import pytest
from pathlib import Path

from memory_system.models import AgentMemoryState
from memory_system.persistence import StatePersistenceManager


def test_session_persistence(tmp_path: Path):
    """Full save → load round-trip preserves all state fields."""
    mgr = StatePersistenceManager(tmp_path / "store")

    original = AgentMemoryState(
        messages=[{"role": "user", "content": "persistent message"}],
        summary="saved summary",
        facts=["fact 1", "fact 2"],
        metadata={"env": "pytest"},
    )
    mgr.save(original)

    loaded = mgr.load(original.session_id)

    assert loaded.session_id == original.session_id
    assert loaded.messages == original.messages
    assert loaded.summary == original.summary
    assert loaded.facts == original.facts


def test_session_persistence_delete(tmp_path: Path):
    """Deleted sessions must not be loadable."""
    mgr = StatePersistenceManager(tmp_path / "store")
    state = AgentMemoryState()
    mgr.save(state)
    mgr.delete(state.session_id)

    with pytest.raises(FileNotFoundError):
        mgr.load(state.session_id)


def test_session_persistence_list(tmp_path: Path):
    """list_sessions() must reflect current disk state."""
    mgr = StatePersistenceManager(tmp_path / "store")
    s1 = AgentMemoryState()
    s2 = AgentMemoryState()
    mgr.save(s1)
    mgr.save(s2)

    ids = set(mgr.list_sessions())
    assert s1.session_id in ids
    assert s2.session_id in ids


def test_session_persistence_atomic_no_tmp_leftover(tmp_path: Path):
    """After a successful save, no .tmp file must remain on disk."""
    mgr = StatePersistenceManager(tmp_path / "store")
    state = AgentMemoryState()
    final = mgr.save(state)
    tmp = final.with_suffix(".tmp")
    assert final.exists()
    assert not tmp.exists()
```

---

## .env.example

```
# Anthropic API key — required for real LLM summarization and future embedding support.
# Leave unset to run in offline/mock mode (suitable for tests).
ANTHROPIC_API_KEY=your_api_key_here
```

---

## pyproject.toml (minimal)

```toml
[project]
name = "memory-system"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.0",
    "anthropic>=0.40",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "ruff>=0.6",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
```

---

## Notes: Extending to Production

### Real Embeddings
Replace `_hash_embed` in `vector.py` with a proper embedding model:
- **sentence-transformers** (local, offline): `SentenceTransformer("all-MiniLM-L6-v2").encode(text).tolist()`
- **OpenAI / Azure OpenAI** embed endpoint
- **Cohere embed v3**

The `embed_fn: EmbedFn` parameter keeps the rest of the system unchanged.

### Real LLM Summarization
Set `ANTHROPIC_API_KEY` and the `_llm_summarize` branch in `summary.py` activates. The model is `claude-sonnet-5-5` for cost/quality balance; swap to `claude-opus-5-5` for highest quality or `claude-haiku-5-5` for lowest latency.

### Production Vector Store
Replace `VectorSemanticMemory._store` (in-memory list) with:
- **Chroma** (local, zero-ops): `chromadb.Client()`
- **Qdrant** (self-hosted or cloud)
- **pgvector** (Postgres extension) if you already run Postgres

The `VectorSemanticMemory` interface (`add(text)`, `retrieve(query, k)`) remains stable; only `_store` internals change.

### Persistence at Scale
`StatePersistenceManager` uses one JSON file per session. For more than ~1000 sessions, switch to:
- SQLite (via `sqlite3` or `sqlmodel`) for indexed queries
- Redis for hot-path session state with TTL
- S3-compatible object store for archival snapshots

### Observability
Wire `LoggingMiddleware` (from day-7 exercises) into `PluggableMemorySystem` and emit structured JSON logs. Route to your observability stack (Datadog, CloudWatch, etc.) by configuring a `logging.handlers.SocketHandler` or a JSON formatter.
