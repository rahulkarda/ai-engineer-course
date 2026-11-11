# Elective: Long-Term Memory · Project — Long-Term Memory Service for an Agent

> Capstone project · Estimated time: 4–5 hours  
> Prereqs: All 6 lessons complete

---

## Overview

Build a **pluggable long-term memory service** and wire it into a LangGraph agent.
The service implements the full memory lifecycle:

- **Write pipeline**: filter → summarise → embed → dedup → importance score → store
- **Retrieval**: composite scoring (recency + importance + relevance), two-stage, top-k
- **Decay & forgetting**: scheduled consolidation pass with configurable thresholds
- **Pluggable backend**: abstract storage interface with two implementations (in-memory for tests, file-backed for development)
- **Agent integration**: LangGraph node that injects relevant memories into the agent's context

All LLM and embedding calls must be **mockable via dependency injection** — the test
suite must run without any API key.

---

## Acceptance criteria

- [ ] `MemoryBackend` protocol with `upsert`, `query`, `delete`, `all`, `count` methods
- [ ] `InMemoryBackend` implementation (for tests)
- [ ] `FileBackend` implementation (JSON Lines persistence, no external DB)
- [ ] `MemoryService` class with `write(messages)`, `retrieve(query, top_k)`, `consolidate()` methods
- [ ] Composite retrieval score: recency (exponential decay) + importance + relevance, configurable weights
- [ ] Decay/forgetting: `consolidate()` archives records below threshold, merges clusters
- [ ] LangGraph `memory_node` that calls `retrieve` and injects results into agent state
- [ ] Agent responds differently (uses memory) when relevant records are present
- [ ] Full test suite: all LLM/embedding calls mocked, no API key needed
- [ ] `MemoryEvaluator` with Precision@k and MRR, exercised in at least one test
- [ ] Type hints throughout, `logging` not `print`, Pydantic models for all boundaries

---

## Specification

### Project structure

```
memory_service/
├── src/
│   └── memory_service/
│       ├── __init__.py
│       ├── models.py          # Pydantic models: MemoryRecord, MemoryConfig, etc.
│       ├── backends.py        # MemoryBackend protocol + InMemoryBackend + FileBackend
│       ├── scoring.py         # cosine_sim, compute_recency, composite_score
│       ├── pipeline.py        # MemoryWritePipeline
│       ├── retrieval.py       # MemoryRetriever
│       ├── consolidation.py   # MemoryConsolidator (decay + forgetting)
│       ├── service.py         # MemoryService (facade over all components)
│       ├── agent.py           # LangGraph agent with memory_node
│       └── evaluation.py      # MemoryEvaluator
├── tests/
│   ├── conftest.py            # shared fixtures: MockEmbedder, MockSummariser, mock records
│   ├── test_backends.py
│   ├── test_scoring.py
│   ├── test_pipeline.py
│   ├── test_retrieval.py
│   ├── test_consolidation.py
│   ├── test_service.py
│   ├── test_agent.py
│   └── test_evaluation.py
├── pyproject.toml
└── README.md
```

### `models.py`

```python
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MemoryType(StrEnum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemoryRecord(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    content: str
    vector: list[float] = Field(default_factory=list)
    importance: float = Field(ge=0.0, le=1.0)
    memory_type: MemoryType = MemoryType.EPISODIC
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    content_hash: str = Field(default="")
    archived: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if not self.content_hash:
            object.__setattr__(
                self, "content_hash",
                hashlib.sha256(self.content.encode()).hexdigest()
            )

    def touch(self) -> None:
        object.__setattr__(self, "last_accessed", datetime.now(timezone.utc))

    class Config:
        frozen = False


class MemoryConfig(BaseModel):
    """Tuneable parameters for the memory service."""
    decay_rate: float = 0.01
    archive_threshold: float = 0.05
    dedup_threshold: float = 0.92
    cluster_similarity: float = 0.80
    alpha: float = 1 / 3   # relevance weight
    beta: float = 1 / 3    # recency weight
    gamma: float = 1 / 3   # importance weight
    max_retrieval_candidates: int = 50
    top_k: int = 5
```

### `backends.py`

Define a `MemoryBackend` Protocol (structural typing) and two implementations:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class MemoryBackend(Protocol):
    def upsert(self, record: MemoryRecord) -> None: ...
    def query(
        self,
        vector: list[float],
        top_k: int,
        include_archived: bool = False,
    ) -> list[MemoryRecord]: ...
    def delete(self, record_id: UUID) -> bool: ...
    def all(self, include_archived: bool = False) -> list[MemoryRecord]: ...
    def count(self, include_archived: bool = False) -> int: ...
```

`InMemoryBackend`: stores records in a `list[MemoryRecord]`, queries by cosine similarity.

`FileBackend`: persists records as JSON Lines in a file. On `__init__`, load existing
records from file if it exists. On every `upsert`/`delete`, rewrite the file.
Use `pathlib.Path` for file I/O.

### `scoring.py`

Implement:

```python
def cosine_sim(a: list[float], b: list[float]) -> float: ...

def compute_recency(record: MemoryRecord, decay_rate: float = 0.01) -> float:
    """exp(-decay_rate * hours_since_last_access)"""
    ...

def composite_score(
    record: MemoryRecord,
    query_vector: list[float],
    config: MemoryConfig,
) -> tuple[float, dict[str, float]]:
    """Returns (score, breakdown). Breakdown keys: relevance, recency, importance, composite."""
    ...
```

### `pipeline.py`

`MemoryWritePipeline`:

- Takes `summariser: Summariser` and `embedder: Embedder` as injected dependencies
- `process_turn(messages: list[Message]) -> MemoryRecord | None`
- Steps: summarise → embed → exact dedup → semantic dedup → score importance → store via backend

### `retrieval.py`

`MemoryRetriever`:

- Takes `backend: MemoryBackend`, `embedder: Embedder`, `config: MemoryConfig`
- `retrieve(query: str, top_k: int | None = None) -> list[RetrievalResult]`
  - `RetrievalResult`: dataclass with `record`, `score`, `breakdown`
- Two-stage: embed query → candidate fetch → composite re-rank → touch accessed

### `consolidation.py`

`MemoryConsolidator`:

- Takes `backend: MemoryBackend`, `consolidate_fn: ConsolidateFn`, `config: MemoryConfig`
- `run() -> ConsolidationStats` (dataclass: `archived: int`, `consolidated: int`, `new_records: int`)
- Algorithm:
  1. Score all active records with `composite_score(record, record.vector, config)` (self-similarity = 1.0 relevance)
  2. Identify candidates below `archive_threshold`
  3. Cluster by cosine similarity >= `cluster_similarity`
  4. For clusters of size >= 2: consolidate and add summary record, archive originals
  5. For single-record clusters: archive directly

### `service.py`

`MemoryService` — facade:

```python
class MemoryService:
    def __init__(
        self,
        backend: MemoryBackend,
        summariser: Summariser,
        embedder: Embedder,
        config: MemoryConfig | None = None,
    ) -> None: ...

    def write(self, messages: list[Message]) -> MemoryRecord | None:
        """Run write pipeline and return stored record or None."""
        ...

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Retrieve top-k memories relevant to query."""
        ...

    def consolidate(self) -> ConsolidationStats:
        """Run forgetting pass."""
        ...

    def stats(self) -> dict[str, Any]:
        """Return {total, active, archived, avg_importance}."""
        ...
```

### `agent.py`

LangGraph agent with memory integration:

```python
from typing import TypedDict
import anthropic
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    messages: list[dict]
    memory_context: str        # injected memory string
    response: str

def memory_node(state: AgentState, service: MemoryService) -> AgentState:
    """Retrieve relevant memories and inject into state."""
    query = state["messages"][-1]["content"]
    results = service.retrieve(query, top_k=3)
    if results:
        snippets = [f"- {r.record.content}" for r in results]
        state["memory_context"] = "Relevant memories:\n" + "\n".join(snippets)
    else:
        state["memory_context"] = ""
    return state

def agent_node(state: AgentState, client: anthropic.Anthropic, model: str) -> AgentState:
    """Call LLM with memory context prepended."""
    system = "You are a helpful assistant."
    if state["memory_context"]:
        system += f"\n\n{state['memory_context']}"
    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=system,
        messages=state["messages"],
    )
    state["response"] = response.content[0].text
    return state

def build_agent(service: MemoryService, client: anthropic.Anthropic, model: str):
    graph = StateGraph(AgentState)
    graph.add_node("memory", lambda s: memory_node(s, service))
    graph.add_node("agent", lambda s: agent_node(s, client, model))
    graph.set_entry_point("memory")
    graph.add_edge("memory", "agent")
    graph.add_edge("agent", END)
    return graph.compile()
```

### `evaluation.py`

Port `MemoryEvaluator` from Lesson 6 with the `map@k` bug fixed. Add:

```python
def run_eval(
    service: MemoryService,
    cases: list[QueryCase],
    k: int = 5,
) -> EvalSummary:
    """High-level evaluation entry point."""
    ...
```

---

## Test requirements

### `conftest.py`

```python
import pytest
from memory_service.models import MemoryConfig

class MockEmbedder:
    def embed(self, text: str) -> list[float]:
        # deterministic 8-D based on hash
        ...

class MockSummariser:
    def summarise(self, messages) -> str | None:
        combined = " ".join(m.content for m in messages)
        if "prefer" in combined.lower() or "like" in combined.lower():
            return combined[:100]
        return None

class MockLLMClient:
    """Replaces anthropic.Anthropic in agent tests."""
    def __init__(self, fixed_response: str = "Mock response") -> None:
        self.fixed_response = fixed_response
        self.call_count = 0

    # Implement .messages.create(...) returning a mock response object

@pytest.fixture
def config() -> MemoryConfig:
    return MemoryConfig()

@pytest.fixture
def embedder() -> MockEmbedder:
    return MockEmbedder()

@pytest.fixture
def summariser() -> MockSummariser:
    return MockSummariser()
```

### Required test cases (minimum)

| Test file | Cases |
|-----------|-------|
| `test_backends.py` | upsert + query returns record; delete removes it; count is correct; FileBackend persists across instantiation |
| `test_scoring.py` | cosine_sim orthogonal=0, identical=1; recency at t=0 ≈ 1.0, at t=large ≈ 0; composite weights sum correctly |
| `test_pipeline.py` | filter discards chitchat; exact duplicate skipped; semantic duplicate boosts importance; new fact stored |
| `test_retrieval.py` | top-k is respected; touch is called on retrieved records; higher composite score ranks first |
| `test_consolidation.py` | records below threshold archived; cluster of 2 produces 1 consolidated record; single-record cluster archived directly |
| `test_service.py` | write → retrieve round-trip; consolidate reduces active count; stats returns expected keys |
| `test_agent.py` | memory_node injects memory string; agent_node receives system prompt with memory; no real API calls |
| `test_evaluation.py` | precision@k, recall@k, mrr correct; summarise returns expected keys |

---

## Deliverables

1. All source files under `src/memory_service/`
2. Full test suite under `tests/` — `pytest` passes with no API key
3. `pyproject.toml` with correct dependencies (no `mem0ai`/`letta` in test deps)
4. Brief docstring on every public class and method

---

## Extension challenges (optional)

- **ChromaDB backend**: implement `ChromaBackend` using `chromadb` (mock in tests with `chromadb.EphemeralClient`)
- **Async API**: make `MemoryService.write` and `retrieve` async; use `asyncio` in tests
- **Real LLM summariser**: implement `AnthropicSummariser` using `claude-haiku-5-5` via `os.getenv("ANTHROPIC_API_KEY")`; skip integration tests if key is absent with `pytest.mark.skipif`
- **Memory streaming**: yield partial retrieval results as they are scored, using async generators
- **Graph backend**: implement a `GraphBackend` that stores entity triples, tested with `InMemoryGraphStore` from Lesson 2

(Solutions: [solutions/project.md](solutions/project.md))
