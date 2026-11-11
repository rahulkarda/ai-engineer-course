# Week 04 Project — Pluggable Agent Memory System

## Goal

Build a production-ready pluggable memory system for an AI agent. The system must be modular, fully typed, serializable, and thoroughly tested. It demonstrates every memory concept from the week: short-term buffering, summarization, vector semantic search, episodic logging, and durable session persistence.

This is a design-and-build project. You start from scratch (no starter code). The acceptance tests define the contract; everything else is your design decision.

---

## Background

An agent that cannot remember is limited to single-turn interactions. An agent with naive memory (full history in context) fails at scale. A well-engineered memory system balances:

- **Recency** (buffer): keep the immediate context available without retrieval overhead.
- **Compression** (summary): preserve meaning when the buffer overflows, without ballooning token cost.
- **Long-term recall** (vector semantic): retrieve relevant facts regardless of when they were stored.
- **Event logging** (episodic): track *what happened, when* for debugging, personalization, and replay.
- **Durability** (persistence): survive restarts, support multi-session agents.

Your system must satisfy all five.

---

## Requirements

### Functional

**1. Typed state** — all internal state managed via Pydantic models. No untyped `dict[str, Any]` at system boundaries.

**2. Buffer memory** — fixed-size FIFO buffer (configurable max size). When full, oldest entry is evicted automatically.

**3. Summary memory with mock LLM** — when the buffer exceeds a configurable threshold, trigger summarization. Use a mock summarizer in tests. In production path, use:
```python
api_key = os.getenv("ANTHROPIC_API_KEY")
# call claude-sonnet-5-5 or claude-haiku-5-5 here
```
The mock must be injectable via constructor (no monkeypatching in tests).

**4. Vector semantic memory with mock embeddings** — store facts as vector embeddings; retrieve by cosine similarity. Use a mock embedder (deterministic hash-based) in tests. In production path, use an embedding API or model. Mock must be injectable.

**5. Episodic memory** — append-only log of timestamped events. Support retrieval by recency and by time range.

**6. JSON persistence** — save and load the complete system state to/from a JSON file using `pathlib.Path`. Use atomic writes (write-to-tmp, rename). Support multiple sessions (one file per session ID).

**7. Pluggable via ABC/Protocol** — each memory backend implements a common abstract base or Protocol. The orchestrator depends on the abstraction, not the concrete class. Swapping a backend must not require changing the orchestrator.

### Non-functional

- Python 3.12+. All functions and classes fully type-annotated.
- `logging` module for all diagnostics. No `print` statements in library code.
- No hardcoded secrets. API keys via `os.getenv` only. Maintain a `.env.example`.
- `ruff` clean (no linting errors).
- Tests in `tests/` mirroring `src/`. At minimum: the 5 acceptance tests below.

---

## Acceptance Tests

Your implementation must pass all five. Write them in `tests/test_project.py`.

```python
"""
Week 04 Project — Acceptance Tests
All 5 must pass. Use mock_llm() and mock_embed() fixtures — no real API calls.
"""
import pytest
from pathlib import Path


# -------------------------------------------------------------------------
# Test 1: State round-trip
# -------------------------------------------------------------------------
def test_state_round_trip(tmp_path: Path) -> None:
    """
    Save a PluggableMemorySystem with known content, reload it,
    and verify all tiers are restored exactly.
    """
    # YOUR IMPORTS HERE
    mem = PluggableMemorySystem(session_id="rt-test", storage_dir=tmp_path)
    mem.add("User asked about Python GIL", store_as_fact=True)
    mem.add("Agent explained the GIL mutex")
    mem.save()

    loaded = PluggableMemorySystem.load("rt-test", storage_dir=tmp_path)
    assert loaded is not None
    # Vector store has the explicit fact
    results = loaded.retrieve("Python GIL", top_k=1)
    assert any("Python" in r or "GIL" in r for r in results)


# -------------------------------------------------------------------------
# Test 2: Buffer eviction
# -------------------------------------------------------------------------
def test_buffer_eviction() -> None:
    """Buffer never exceeds max_size; oldest entry is dropped."""
    mem = PluggableMemorySystem(buffer_size=3)
    for i in range(5):
        mem.add(f"entry-{i}")
    buffer_contents = mem.buffer_contents()  # expose current buffer as list[str]
    assert len(buffer_contents) <= 3
    assert "entry-0" not in buffer_contents
    assert "entry-4" in buffer_contents


# -------------------------------------------------------------------------
# Test 3: Summarization trigger
# -------------------------------------------------------------------------
def test_summarization_trigger() -> None:
    """
    When buffer exceeds consolidate_threshold, consolidate() is triggered,
    summary memory receives an entry, and buffer is cleared.
    """
    mem = PluggableMemorySystem(buffer_size=10, consolidate_threshold=3)
    for i in range(3):
        mem.add(f"message-{i}")
    # After 3 adds at threshold=3, buffer cleared, summary populated
    assert len(mem.buffer_contents()) == 0
    assert mem.summary_count() >= 1  # expose count of stored summaries


# -------------------------------------------------------------------------
# Test 4: Semantic search
# -------------------------------------------------------------------------
def test_semantic_search() -> None:
    """Top-1 semantic retrieval returns the most relevant stored fact."""
    mem = PluggableMemorySystem()
    mem.add("Python uses reference counting for memory management.", store_as_fact=True)
    mem.add("Java uses a mark-and-sweep garbage collector.", store_as_fact=True)
    mem.add("Rust enforces memory safety through ownership.", store_as_fact=True)

    results = mem.retrieve("Python memory management", top_k=1)
    assert len(results) == 1
    assert "Python" in results[0] or "reference" in results[0]


# -------------------------------------------------------------------------
# Test 5: Session persistence
# -------------------------------------------------------------------------
def test_session_persistence(tmp_path: Path) -> None:
    """
    Two separate PluggableMemorySystem instances (simulating separate process runs)
    sharing the same session_id correctly continue the same conversation.
    """
    # First "run"
    run1 = PluggableMemorySystem(session_id="persist-test", storage_dir=tmp_path)
    run1.add("Session started.", store_as_fact=True)
    run1.save()

    # Second "run" — fresh object, same session
    run2 = PluggableMemorySystem.load("persist-test", storage_dir=tmp_path)
    assert run2 is not None
    run2.add("Session resumed.")
    results = run2.retrieve("session", top_k=2)
    # The fact from run1 should be retrievable in run2
    assert any("Session" in r or "session" in r for r in results)
```

---

## Deliverables

```
src/
  memory/
    __init__.py
    backends/
      __init__.py
      base.py          # MemoryBackend ABC or Protocol
      buffer.py        # BufferMemory
      summary.py       # SummaryMemory
      vector.py        # VectorSemanticMemory
      episodic.py      # EpisodicMemory
    persistence.py     # StatePersistenceManager
    system.py          # PluggableMemorySystem (orchestrator)
    models.py          # Pydantic models for state
tests/
  test_project.py      # 5 acceptance tests + your own unit tests
.env.example           # ANTHROPIC_API_KEY=
```

You may restructure within reason as long as the acceptance tests pass and the pluggable-via-ABC requirement holds.

---

## Implementation Notes

**LLM calls mocked — production path**

In `SummaryMemory`, the summarizer is injected:
```python
class SummaryMemory:
    def __init__(self, summarize_fn: Callable[[list[str]], str] = default_summarizer):
        ...
```

Production path (not called in tests):
```python
import os
api_key = os.getenv("ANTHROPIC_API_KEY")
# use claude-sonnet-5-5 or claude-haiku-5-5 with the Anthropic SDK
```

Tests always pass `mock_llm()` as a pytest fixture returning a deterministic lambda. Never use a real API key in tests.

Same pattern for `VectorSemanticMemory`:
```python
class VectorSemanticMemory:
    def __init__(self, embed_fn: Callable[[str], list[float]] = default_embedder):
        ...
```

Tests inject `mock_embed()` (hash-based deterministic unit vector). Production path uses an embedding model.

---

## Extension Ideas

These are optional stretch goals. Complete the core requirements first.

1. **HNSW vector index** — replace the O(n) cosine scan in `VectorSemanticMemory` with `hnswlib`. Benchmark the speedup at 10k, 100k entries.

2. **SQLite persistence backend** — implement `SQLiteStatePersistenceManager` as a drop-in replacement for the JSON backend. Same interface, ACID guarantees, concurrent read support.

3. **Memory health dashboard** — `PluggableMemorySystem.health()` returns a `MemoryHealthReport` Pydantic model with per-tier stats (size, hit rate, last consolidation timestamp). Expose as a FastAPI endpoint.

4. **Automatic fact extraction** — after each consolidation, run a second mock LLM call that extracts `n` key facts from the summary and stores them in the vector store. Implement as a post-consolidation hook.

5. **LangGraph integration** — wire `PluggableMemorySystem` as a tool node in a minimal LangGraph graph. The graph has three nodes: `retrieve_memory`, `generate_response`, `update_memory`. State is your `AgentMemoryState` Pydantic model.

6. **Episodic replay** — implement `EpisodicMemory.replay(start: datetime, end: datetime) -> list[str]` that returns all events in the time window as a formatted string. Write a test with fixed timestamps.

---

Reference solution: ../solutions/week-04/project.md
