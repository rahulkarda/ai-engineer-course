# Phase 5 · Week 04 · Day 7 — Pluggable Memory System: Integration & Patterns

> Objective: Design and build a pluggable, tiered memory system that integrates buffer, summary, vector, episodic, and semantic memory with a unified interface.  ·  Time: ~4 hr  ·  Prereqs: [[day-1.md]] through [[day-6.md]]

## Concept

### Why pluggability matters

Real agents evolve. You start with a buffer, add summarization, then need vector search for long-term recall, then episodic replay for debugging. If each memory type is hardcoded into the agent, every addition is a refactor. A pluggable system allows you to swap or add backends without touching the agent core.

The pattern: define an abstract interface (`MemoryBackend` protocol), write concrete implementations, and compose them in an orchestrator (`PluggableMemorySystem`).

### Tiered memory architecture

Borrowed from CPU cache design:

```
Short-Term Memory (STM)
  └── Buffer Memory    — raw, recent, fast, bounded
  └── Summary Memory   — compressed STM, bounded
Long-Term Memory (LTM)
  └── Vector/Semantic  — embedded facts, unbounded, similarity search
  └── Episodic         — timestamped events, chronological
```

Information flows **down** (raw → compressed → vectorized) during consolidation and **up** (retrieve from all tiers, merge) during recall.

### Memory lifecycle

```
Write (new input)
  → append to buffer (STM)
  → if buffer full: trigger consolidation
    → summarize buffer → store summary in SummaryMemory
    → extract key facts → store in VectorSemanticMemory
    → flush buffer

Retrieve (query)
  → query buffer (exact recent context)
  → query summary (compressed recent history)
  → query vector store (semantic similarity)
  → merge and rank results

Consolidate (scheduled or on-demand)
  → move old summaries → LTM
  → compress episodic events → semantic facts
```

### Memory routing

Not every write should go to every tier. Route by information type:

| Information | Tier |
|---|---|
| Raw conversation turn | Buffer |
| Summarized context | SummaryMemory |
| Domain fact extracted from conversation | VectorSemantic |
| Time-anchored event | Episodic |

The orchestrator decides routing based on the `write_type` parameter or content heuristics.

### Performance considerations

- Vector search is O(n) with naive cosine scan. For >10k facts, use an HNSW index (e.g., `hnswlib`, `faiss`).
- Summarization requires an LLM call — batch or debounce when possible.
- JSON persistence (Day 6) is fine for <10k episodes. SQLite for larger stores.
- Keep STM small (last 20 turns max); LTM can grow unboundedly.

## How it works internally

The `PluggableMemorySystem` holds references to all backends. On `add(text)`, it routes to the buffer and optionally the episodic store. When the buffer exceeds its threshold, `consolidate()` is called automatically:

1. The buffer contents are passed to `SummaryMemory.summarize()` (calls a mock or real LLM).
2. The summary is stored in `SummaryMemory`.
3. Key facts are extracted (mock: use the summary itself) and stored in `VectorSemanticMemory`.
4. The buffer is cleared.

On `retrieve(query)`, results from all tiers are collected, deduplicated, and ranked by recency + relevance.

The `save()` / `load()` methods delegate to `StatePersistenceManager` from Day 6 — the pluggable system serializes each backend's state into a combined `AgentMemoryState`.

## Worked example

```python
"""
day7_pluggable_memory.py

Pluggable, tiered memory system.
Integrates: BufferMemory, SummaryMemory (mock LLM), VectorSemanticMemory.
Includes a mock agent loop running 10 turns.
No real LLM or API key required.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

EMBED_DIM = 16


def _fake_embed(text: str) -> list[float]:
    seed = int(hashlib.md5(text.lower().encode()).hexdigest(), 16)
    raw = []
    for i in range(EMBED_DIM):
        seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
        raw.append((seed / 0xFFFFFFFF) * 2 - 1)
    magnitude = math.sqrt(sum(v * v for v in raw))
    return [v / magnitude for v in raw]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _mock_llm_summarize(texts: list[str]) -> str:
    """Deterministic mock summarizer — concatenates with truncation."""
    joined = " | ".join(texts)
    return f"[Summary of {len(texts)} items]: {joined[:120]}"


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class MemoryBackend(ABC):
    """Protocol for all memory backends."""

    @abstractmethod
    def add(self, text: str, **kwargs: Any) -> None: ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[str]: ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryBackend": ...


# ---------------------------------------------------------------------------
# BufferMemory
# ---------------------------------------------------------------------------

class BufferMemory(MemoryBackend):
    """
    Fixed-size ring buffer of raw text entries.
    Retrieval: returns the most recent top_k entries.
    """

    def __init__(self, max_size: int = 10) -> None:
        self._buffer: list[str] = []
        self._max = max_size
        self._log = logging.getLogger(self.__class__.__name__)

    def add(self, text: str, **kwargs: Any) -> None:
        self._buffer.append(text)
        if len(self._buffer) > self._max:
            self._buffer.pop(0)
        self._log.debug("Buffer size: %d/%d", len(self._buffer), self._max)

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        return self._buffer[-top_k:]

    def clear(self) -> None:
        self._buffer.clear()

    def is_full(self) -> bool:
        return len(self._buffer) >= self._max

    def all_entries(self) -> list[str]:
        return list(self._buffer)

    def to_dict(self) -> dict[str, Any]:
        return {"max_size": self._max, "buffer": self._buffer}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BufferMemory":
        obj = cls(max_size=data["max_size"])
        obj._buffer = data["buffer"]
        return obj


# ---------------------------------------------------------------------------
# SummaryMemory
# ---------------------------------------------------------------------------

class SummaryMemory(MemoryBackend):
    """
    Stores LLM-generated summaries of buffer contents.
    Each summary is paired with a timestamp for recency ranking.
    """

    def __init__(self, summarize_fn=_mock_llm_summarize) -> None:
        self._summaries: list[dict[str, str]] = []
        self._summarize = summarize_fn
        self._log = logging.getLogger(self.__class__.__name__)

    def add(self, text: str, **kwargs: Any) -> None:
        """Add a pre-formed summary string directly."""
        entry = {
            "text": text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._summaries.append(entry)
        self._log.info("Stored summary: %s", text[:60])

    def summarize_and_store(self, texts: list[str]) -> str:
        """Summarize a list of texts and store the result."""
        summary = self._summarize(texts)
        self.add(summary)
        return summary

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        # Return the most recent summaries (simple recency; extend with similarity if needed)
        recent = self._summaries[-top_k:]
        return [s["text"] for s in recent]

    def clear(self) -> None:
        self._summaries.clear()

    def to_dict(self) -> dict[str, Any]:
        return {"summaries": self._summaries}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SummaryMemory":
        obj = cls()
        obj._summaries = data.get("summaries", [])
        return obj


# ---------------------------------------------------------------------------
# VectorSemanticMemory
# ---------------------------------------------------------------------------

class VectorEntry(BaseModel):
    text: str
    embedding: list[float]
    source: str = "unknown"
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class VectorSemanticMemory(MemoryBackend):
    """Vector-embedded semantic memory with cosine similarity retrieval."""

    def __init__(self, embed_fn=_fake_embed) -> None:
        self._entries: list[VectorEntry] = []
        self._embed = embed_fn
        self._log = logging.getLogger(self.__class__.__name__)

    def add(self, text: str, source: str = "unknown", **kwargs: Any) -> None:
        embedding = self._embed(text)
        self._entries.append(
            VectorEntry(text=text, embedding=embedding, source=source)
        )
        self._log.info("Stored vector fact: %s", text[:60])

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        if not self._entries:
            return []
        q_vec = self._embed(query)
        scored = sorted(
            self._entries,
            key=lambda e: _cosine(q_vec, e.embedding),
            reverse=True,
        )
        return [e.text for e in scored[:top_k]]

    def clear(self) -> None:
        self._entries.clear()

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [e.model_dump() for e in self._entries]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VectorSemanticMemory":
        obj = cls()
        obj._entries = [VectorEntry.model_validate(e) for e in data.get("entries", [])]
        return obj


# ---------------------------------------------------------------------------
# PluggableMemorySystem
# ---------------------------------------------------------------------------

class MemorySystemState(BaseModel):
    session_id: str
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    buffer: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    vector: dict[str, Any] = Field(default_factory=dict)


class PluggableMemorySystem:
    """
    Orchestrates BufferMemory, SummaryMemory, and VectorSemanticMemory.

    Memory lifecycle:
      add(text) → buffer; if buffer full → consolidate()
      retrieve(query) → merge results from all tiers
      consolidate() → summarize buffer → store in summary + vector; clear buffer
      save(path) / load(path) → JSON persistence
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        buffer_size: int = 5,
        consolidate_threshold: int = 5,
    ) -> None:
        self.session_id = session_id or str(uuid.uuid4())
        self._buffer = BufferMemory(max_size=buffer_size)
        self._summary = SummaryMemory()
        self._vector = VectorSemanticMemory()
        self._consolidate_threshold = consolidate_threshold
        self._log = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, text: str, store_as_fact: bool = False) -> None:
        """
        Add text to memory.
        - Always written to buffer.
        - If store_as_fact=True, also written directly to vector store.
        - Auto-consolidates if buffer reaches threshold.
        """
        self._buffer.add(text)
        if store_as_fact:
            self._vector.add(text, source="explicit-fact")
        if len(self._buffer.all_entries()) >= self._consolidate_threshold:
            self._log.info("Buffer full — triggering consolidation.")
            self.consolidate()

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """
        Retrieve from all tiers, deduplicate, return merged list.
        Order: buffer (most recent) → summary → vector.
        """
        results: list[str] = []
        seen: set[str] = set()

        for tier_name, tier in [
            ("buffer", self._buffer),
            ("summary", self._summary),
            ("vector", self._vector),
        ]:
            candidates = tier.retrieve(query, top_k=top_k)
            for c in candidates:
                if c not in seen:
                    results.append(c)
                    seen.add(c)
            self._log.debug("Tier '%s' returned %d candidates", tier_name, len(candidates))

        return results[:top_k]

    # ------------------------------------------------------------------
    # Consolidate
    # ------------------------------------------------------------------

    def consolidate(self) -> Optional[str]:
        """
        Compress buffer into summary and vector store, then clear buffer.
        Returns the generated summary string, or None if buffer was empty.
        """
        entries = self._buffer.all_entries()
        if not entries:
            return None
        summary = self._summary.summarize_and_store(entries)
        # Store the summary as a semantic fact for future retrieval
        self._vector.add(summary, source="consolidation")
        self._buffer.clear()
        self._log.info(
            "Consolidated %d entries → summary stored, buffer cleared.", len(entries)
        )
        return summary

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, storage_dir: Path | str = ".memory_state") -> Path:
        storage = Path(storage_dir)
        storage.mkdir(parents=True, exist_ok=True)
        state = MemorySystemState(
            session_id=self.session_id,
            buffer=self._buffer.to_dict(),
            summary=self._summary.to_dict(),
            vector=self._vector.to_dict(),
        )
        path = storage / f"{self.session_id}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(state.model_dump(mode="json"), indent=2), encoding="utf-8"
        )
        tmp.rename(path)
        self._log.info("Saved memory system to %s", path)
        return path

    @classmethod
    def load(
        cls,
        session_id: str,
        storage_dir: Path | str = ".memory_state",
    ) -> Optional["PluggableMemorySystem"]:
        path = Path(storage_dir) / f"{session_id}.json"
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        state = MemorySystemState.model_validate(raw)
        obj = cls(session_id=state.session_id)
        obj._buffer = BufferMemory.from_dict(state.buffer)
        obj._summary = SummaryMemory.from_dict(state.summary)
        obj._vector = VectorSemanticMemory.from_dict(state.vector)
        return obj


# ---------------------------------------------------------------------------
# Mock agent loop — 10 turns
# ---------------------------------------------------------------------------

AGENT_TURNS = [
    ("user", "What is Python's GIL?"),
    ("agent", "The GIL is a mutex that protects CPython's memory manager."),
    ("user", "Does it affect multiprocessing?"),
    ("agent", "No, multiprocessing bypasses the GIL by using separate processes."),
    ("user", "What about asyncio?"),
    ("agent", "asyncio is single-threaded; it sidesteps the GIL entirely."),
    ("user", "When should I use threading vs multiprocessing?"),
    ("agent", "Threading for I/O-bound; multiprocessing for CPU-bound tasks."),
    ("user", "Any libraries to watch out for?"),
    ("agent", "NumPy releases the GIL for heavy computations."),
    ("user", "So Python is not truly parallel?"),
]


def run_mock_agent_loop(memory: PluggableMemorySystem) -> None:
    for role, text in AGENT_TURNS:
        entry = f"[{role}] {text}"
        memory.add(entry)
        logger.info("Turn added: %s", entry[:60])

    # Final query
    query = "Python concurrency and threading"
    results = memory.retrieve(query, top_k=4)
    print("\n=== Memory retrieval for:", query, "===")
    for r in results:
        print(" ", r[:100])


def main() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        memory = PluggableMemorySystem(
            session_id="demo-session",
            buffer_size=5,
            consolidate_threshold=5,
        )

        # Seed a few semantic facts
        memory.add("Python was created by Guido van Rossum.", store_as_fact=True)
        memory.add("Pydantic v2 uses Rust-based validation.", store_as_fact=True)

        run_mock_agent_loop(memory)

        # Persist
        save_path = memory.save(storage_dir=tmp)
        print(f"\nSaved to: {save_path}")

        # Reload
        loaded = PluggableMemorySystem.load("demo-session", storage_dir=tmp)
        assert loaded is not None
        print(f"Reloaded session: {loaded.session_id}")
        print("Reload retrieve:", loaded.retrieve("Python GIL", top_k=2))


if __name__ == "__main__":
    main()
```

### Example tests

```python
# tests/unit/test_day7_pluggable.py
import pytest
from pathlib import Path
from day7_pluggable_memory import (
    BufferMemory,
    SummaryMemory,
    VectorSemanticMemory,
    PluggableMemorySystem,
)


def test_buffer_evicts_oldest() -> None:
    buf = BufferMemory(max_size=3)
    for i in range(5):
        buf.add(f"entry-{i}")
    assert len(buf.all_entries()) == 3
    assert buf.all_entries()[0] == "entry-2"


def test_summary_stores_and_retrieves() -> None:
    sm = SummaryMemory(summarize_fn=lambda texts: f"SUMMARY:{len(texts)}")
    sm.summarize_and_store(["a", "b", "c"])
    results = sm.retrieve("anything", top_k=1)
    assert results[0].startswith("SUMMARY:3")


def test_vector_retrieves_similar() -> None:
    vm = VectorSemanticMemory()
    vm.add("Python is dynamically typed")
    vm.add("Java is statically typed")
    vm.add("Rust uses ownership for memory safety")
    results = vm.retrieve("Python typing system", top_k=1)
    assert "Python" in results[0]


def test_pluggable_consolidates_on_full_buffer() -> None:
    mem = PluggableMemorySystem(buffer_size=3, consolidate_threshold=3)
    for i in range(3):
        mem.add(f"message {i}")
    # After 3 adds, buffer should be cleared (auto-consolidation)
    assert len(mem._buffer.all_entries()) == 0
    # Summary should have one entry
    assert len(mem._summary._summaries) == 1


def test_pluggable_save_load_round_trip(tmp_path: Path) -> None:
    mem = PluggableMemorySystem(session_id="rt-test", buffer_size=10)
    mem.add("Fact about persistence", store_as_fact=True)
    mem.save(storage_dir=tmp_path)

    loaded = PluggableMemorySystem.load("rt-test", storage_dir=tmp_path)
    assert loaded is not None
    assert len(loaded._vector._entries) == 1
```

## Exercises

1. **Tier weight in retrieval.** Currently `retrieve()` returns results in tier order (buffer first). Implement a scored merge: buffer results get a recency bonus of `+1.0`, summary results `+0.5`, vector results get their raw cosine score. Return the top-k by combined score. Write a test that verifies a very recent buffer entry outranks an older vector fact.

2. **Router implementation.** Add a `MemoryRouter` class that decides which backends to write to based on keywords in the text. Rule: if the text contains `"fact:"`, write to `VectorSemanticMemory` only; if it starts with `"[user]"`, write to buffer and episodic; otherwise write to buffer only. Write tests for each routing rule.

3. **EpisodicMemory integration.** Import `EpisodicMemory` from Day 5. Add it as a fourth tier in `PluggableMemorySystem`. In `add()`, always write to episodic with the appropriate metadata. In `retrieve()`, include the 2 most recent episodic events. Extend `save()` / `load()` to include episodic state. Write a test verifying episodic state survives a round-trip.

4. **Pluggable summarizer.** `SummaryMemory` currently takes a `summarize_fn`. Create a `LLMSummarizer` class with a `summarize(texts: list[str]) -> str` method that calls `os.getenv("ANTHROPIC_API_KEY")` in production but returns a mock in tests. Wire it into `PluggableMemorySystem` via constructor injection. Show the injection pattern. Tests must use the mock path (no real API calls in tests).

5. **Memory health metrics.** Add a `health() -> dict[str, Any]` method to `PluggableMemorySystem` that returns: `buffer_size`, `summary_count`, `vector_count`, `consolidation_count` (how many times `consolidate()` was called), `last_consolidated_at` (ISO timestamp or `None`). Write tests that verify the counters increment correctly through a sequence of adds and consolidations.

6. **Benchmarking.** Write a standalone script (not a test) that inserts 1000 entries into `VectorSemanticMemory` and times 100 `retrieve()` calls. Print mean and p95 latency. Note: with the naive O(n) scan this should still be sub-millisecond at 1000 entries; at what n would you expect it to become a bottleneck?

7. **Backend swappability.** Define a `MemoryBackendRegistry` that maps string keys (`"buffer"`, `"summary"`, `"vector"`) to backend classes. `PluggableMemorySystem` should accept a `registry` param and instantiate backends from it. This allows users to drop in a `RedisBufferMemory` or `PineconeVectorMemory` without changing the orchestrator. Write a test that registers a `SpyMemory(MemoryBackend)` that records every `add()` call, injects it as the buffer, and verifies it received all expected texts.

(Solutions: ../solutions/week-04/day-7.md)

## Self-check

- [ ] I can explain the STM → LTM tiered architecture and why it mirrors CPU cache design
- [ ] I can implement a `MemoryBackend` abstract base class and at least two concrete implementations
- [ ] I can describe the memory lifecycle: write → consolidate → retrieve
- [ ] I can implement `consolidate()` — summarize buffer, store in summary + vector, clear buffer
- [ ] I can implement `retrieve()` that merges results from multiple tiers with deduplication
- [ ] I can serialize and restore the complete pluggable memory system state via JSON
- [ ] I can inject a mock summarizer/embedder for testing without touching production code paths
