# Solutions: Project — Long-Term Memory Service

This is a **complete reference implementation**. Study the structure; implement it
yourself first.

---

## `src/memory_service/models.py`

```python
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


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
    content_hash: str = ""
    archived: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _set_hash(self) -> "MemoryRecord":
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()
        return self

    def touch(self) -> None:
        self.last_accessed = datetime.now(timezone.utc)

    model_config = {"frozen": False}


class MemoryConfig(BaseModel):
    decay_rate: float = 0.01
    archive_threshold: float = 0.05
    dedup_threshold: float = 0.92
    cluster_similarity: float = 0.80
    alpha: float = 1 / 3
    beta: float = 1 / 3
    gamma: float = 1 / 3
    max_retrieval_candidates: int = 50
    top_k: int = 5


class Message(BaseModel):
    role: str
    content: str
```

---

## `src/memory_service/backends.py`

```python
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Protocol, runtime_checkable
from uuid import UUID

from .models import MemoryRecord

log = logging.getLogger(__name__)


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na * nb > 1e-9 else 0.0


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


class InMemoryBackend:
    def __init__(self) -> None:
        self._records: dict[UUID, MemoryRecord] = {}

    def upsert(self, record: MemoryRecord) -> None:
        self._records[record.id] = record
        log.debug("InMemoryBackend.upsert id=%s", record.id)

    def query(
        self,
        vector: list[float],
        top_k: int,
        include_archived: bool = False,
    ) -> list[MemoryRecord]:
        candidates = [
            r for r in self._records.values()
            if include_archived or not r.archived
        ]
        if not vector:
            return candidates[:top_k]
        scored = sorted(
            candidates,
            key=lambda r: _cosine_sim(vector, r.vector) if r.vector else 0.0,
            reverse=True,
        )
        return scored[:top_k]

    def delete(self, record_id: UUID) -> bool:
        existed = record_id in self._records
        self._records.pop(record_id, None)
        return existed

    def all(self, include_archived: bool = False) -> list[MemoryRecord]:
        return [
            r for r in self._records.values()
            if include_archived or not r.archived
        ]

    def count(self, include_archived: bool = False) -> int:
        return len(self.all(include_archived))


class FileBackend:
    """JSON Lines persistence. Loads on init, rewrites on every mutation."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._records: dict[UUID, MemoryRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        with self._path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    r = MemoryRecord.model_validate_json(line)
                    self._records[r.id] = r
        log.info("FileBackend loaded %d records from %s", len(self._records), self._path)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w") as f:
            for r in self._records.values():
                f.write(r.model_dump_json() + "\n")

    def upsert(self, record: MemoryRecord) -> None:
        self._records[record.id] = record
        self._save()

    def query(
        self,
        vector: list[float],
        top_k: int,
        include_archived: bool = False,
    ) -> list[MemoryRecord]:
        candidates = [
            r for r in self._records.values()
            if include_archived or not r.archived
        ]
        if not vector:
            return candidates[:top_k]
        scored = sorted(
            candidates,
            key=lambda r: _cosine_sim(vector, r.vector) if r.vector else 0.0,
            reverse=True,
        )
        return scored[:top_k]

    def delete(self, record_id: UUID) -> bool:
        existed = record_id in self._records
        self._records.pop(record_id, None)
        if existed:
            self._save()
        return existed

    def all(self, include_archived: bool = False) -> list[MemoryRecord]:
        return [
            r for r in self._records.values()
            if include_archived or not r.archived
        ]

    def count(self, include_archived: bool = False) -> int:
        return len(self.all(include_archived))
```

---

## `src/memory_service/scoring.py`

```python
from __future__ import annotations

import math
from datetime import datetime, timezone

from .models import MemoryConfig, MemoryRecord


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na * nb > 1e-9 else 0.0


def compute_recency(record: MemoryRecord, decay_rate: float = 0.01) -> float:
    now = datetime.now(timezone.utc)
    delta_hours = (now - record.last_accessed).total_seconds() / 3600
    return math.exp(-decay_rate * delta_hours)


def composite_score(
    record: MemoryRecord,
    query_vector: list[float],
    config: MemoryConfig,
) -> tuple[float, dict[str, float]]:
    relevance = max(0.0, cosine_sim(query_vector, record.vector)) if record.vector else 0.0
    recency = compute_recency(record, config.decay_rate)
    importance = record.importance
    score = config.alpha * relevance + config.beta * recency + config.gamma * importance
    breakdown = {
        "relevance": round(relevance, 4),
        "recency": round(recency, 4),
        "importance": round(importance, 4),
        "composite": round(score, 4),
    }
    return score, breakdown
```

---

## `src/memory_service/pipeline.py`

```python
from __future__ import annotations

import hashlib
import logging
from typing import Callable, Protocol

from .backends import MemoryBackend
from .models import MemoryConfig, MemoryRecord, Message
from .scoring import cosine_sim

log = logging.getLogger(__name__)


class Summariser(Protocol):
    def summarise(self, messages: list[Message]) -> str | None: ...


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


def _score_importance(content: str, existing_count: int = 0) -> float:
    score = 0.25
    lower = content.lower()
    if any(kw in lower for kw in ["prefer", "always", "never", "must", "required"]):
        score += 0.35
    if existing_count > 0:
        score += min(0.25, existing_count * 0.1)
    score += min(0.15, len(content) / 500)
    return min(1.0, score)


class MemoryWritePipeline:
    DEDUP_THRESHOLD = 0.92

    def __init__(
        self,
        summariser: Summariser,
        embedder: Embedder,
        backend: MemoryBackend,
        config: MemoryConfig | None = None,
    ) -> None:
        self._summariser = summariser
        self._embedder = embedder
        self._backend = backend
        self._config = config or MemoryConfig()

    def process_turn(self, messages: list[Message]) -> MemoryRecord | None:
        summary = self._summariser.summarise(messages)
        if summary is None:
            log.info("Pipeline: filtered (no storable content)")
            return None

        vector = self._embedder.embed(summary)
        content_hash = hashlib.sha256(summary.encode()).hexdigest()

        existing = self._backend.all()
        if any(r.content_hash == content_hash for r in existing):
            log.info("Pipeline: exact duplicate skipped")
            return None

        near_dups = sum(
            1 for r in existing
            if r.vector and cosine_sim(vector, r.vector) >= self._config.dedup_threshold
            and r.content != summary
        )

        importance = _score_importance(summary, near_dups)
        record = MemoryRecord(content=summary, vector=vector, importance=importance)
        self._backend.upsert(record)
        log.info("Pipeline: stored id=%s importance=%.2f", record.id, record.importance)
        return record
```

---

## `src/memory_service/retrieval.py`

```python
from __future__ import annotations

import logging
from dataclasses import dataclass

from .backends import MemoryBackend
from .models import MemoryConfig, MemoryRecord
from .pipeline import Embedder
from .scoring import composite_score, cosine_sim

log = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    record: MemoryRecord
    score: float
    breakdown: dict[str, float]


class MemoryRetriever:
    def __init__(
        self,
        backend: MemoryBackend,
        embedder: Embedder,
        config: MemoryConfig | None = None,
    ) -> None:
        self._backend = backend
        self._embedder = embedder
        self._config = config or MemoryConfig()

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        k = top_k or self._config.top_k
        query_vector = self._embedder.embed(query)

        # Stage 1: candidate fetch
        candidates = self._backend.query(
            query_vector,
            top_k=self._config.max_retrieval_candidates,
        )

        # Stage 2: composite re-rank
        scored: list[RetrievalResult] = []
        for rec in candidates:
            score, breakdown = composite_score(rec, query_vector, self._config)
            scored.append(RetrievalResult(record=rec, score=score, breakdown=breakdown))

        scored.sort(key=lambda r: r.score, reverse=True)
        results = scored[:k]

        for r in results:
            r.record.touch()
            self._backend.upsert(r.record)  # persist updated last_accessed

        log.info("Retriever: returned %d results for query='%s'", len(results), query[:40])
        return results
```

---

## `src/memory_service/consolidation.py`

```python
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable
from uuid import uuid4

from .backends import MemoryBackend
from .models import MemoryConfig, MemoryRecord
from .scoring import composite_score, cosine_sim

log = logging.getLogger(__name__)

ConsolidateFn = Callable[[list[MemoryRecord]], str]


@dataclass
class ConsolidationStats:
    archived: int = 0
    consolidated: int = 0
    new_records: int = 0


class MemoryConsolidator:
    def __init__(
        self,
        backend: MemoryBackend,
        consolidate_fn: ConsolidateFn,
        config: MemoryConfig | None = None,
    ) -> None:
        self._backend = backend
        self._consolidate_fn = consolidate_fn
        self._config = config or MemoryConfig()

    def run(self) -> ConsolidationStats:
        stats = ConsolidationStats()
        active = self._backend.all()
        candidates = [
            r for r in active
            if composite_score(r, r.vector or [], self._config)[0] < self._config.archive_threshold
        ]
        log.info("Consolidation: %d/%d below threshold", len(candidates), len(active))
        if not candidates:
            return stats

        clusters = self._cluster(candidates)
        for cluster in clusters:
            if len(cluster) < 2:
                cluster[0].archived = True
                self._backend.upsert(cluster[0])
                stats.archived += 1
            else:
                summary = self._consolidate_fn(cluster)
                new_record = MemoryRecord(
                    content=summary,
                    vector=cluster[0].vector,
                    importance=max(r.importance for r in cluster) * 0.8,
                    metadata={"consolidated_from": [str(r.id) for r in cluster]},
                )
                self._backend.upsert(new_record)
                stats.new_records += 1
                for r in cluster:
                    r.archived = True
                    self._backend.upsert(r)
                stats.archived += len(cluster)
                stats.consolidated += 1

        return stats

    def _cluster(self, records: list[MemoryRecord]) -> list[list[MemoryRecord]]:
        clusters: list[list[MemoryRecord]] = []
        used: set = set()
        for i, r in enumerate(records):
            if r.id in used:
                continue
            cluster = [r]
            used.add(r.id)
            for j, r2 in enumerate(records):
                if j <= i or r2.id in used:
                    continue
                if r.vector and r2.vector and cosine_sim(r.vector, r2.vector) >= self._config.cluster_similarity:
                    cluster.append(r2)
                    used.add(r2.id)
            clusters.append(cluster)
        return clusters
```

---

## `src/memory_service/service.py`

```python
from __future__ import annotations

import logging
from typing import Any

from .backends import MemoryBackend
from .consolidation import ConsolidationStats, ConsolidateFn, MemoryConsolidator
from .models import MemoryConfig, MemoryRecord, Message
from .pipeline import Embedder, MemoryWritePipeline, Summariser
from .retrieval import MemoryRetriever, RetrievalResult

log = logging.getLogger(__name__)


def _default_consolidate(records: list[MemoryRecord]) -> str:
    snippets = [r.content[:60] for r in records]
    return "Consolidated: " + "; ".join(snippets)


class MemoryService:
    def __init__(
        self,
        backend: MemoryBackend,
        summariser: Summariser,
        embedder: Embedder,
        config: MemoryConfig | None = None,
        consolidate_fn: ConsolidateFn | None = None,
    ) -> None:
        self._backend = backend
        self._config = config or MemoryConfig()
        self._pipeline = MemoryWritePipeline(summariser, embedder, backend, self._config)
        self._retriever = MemoryRetriever(backend, embedder, self._config)
        self._consolidator = MemoryConsolidator(
            backend, consolidate_fn or _default_consolidate, self._config
        )

    def write(self, messages: list[Message]) -> MemoryRecord | None:
        return self._pipeline.process_turn(messages)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        return self._retriever.retrieve(query, top_k)

    def consolidate(self) -> ConsolidationStats:
        return self._consolidator.run()

    def stats(self) -> dict[str, Any]:
        active = self._backend.all(include_archived=False)
        all_records = self._backend.all(include_archived=True)
        archived = [r for r in all_records if r.archived]
        avg_importance = (
            sum(r.importance for r in active) / len(active) if active else 0.0
        )
        return {
            "total": len(all_records),
            "active": len(active),
            "archived": len(archived),
            "avg_importance": round(avg_importance, 4),
        }
```

---

## `src/memory_service/agent.py`

```python
from __future__ import annotations

import logging
import os
from typing import Any, TypedDict

import anthropic
from langgraph.graph import END, StateGraph

from .service import MemoryService

log = logging.getLogger(__name__)

MODEL = "claude-sonnet-5-5"


class AgentState(TypedDict):
    messages: list[dict[str, str]]
    memory_context: str
    response: str


def memory_node(state: AgentState, service: MemoryService) -> AgentState:
    query = state["messages"][-1]["content"]
    results = service.retrieve(query, top_k=3)
    if results:
        snippets = [f"- {r.record.content}" for r in results]
        state["memory_context"] = "Relevant memories:\n" + "\n".join(snippets)
        log.info("memory_node: injected %d memories", len(results))
    else:
        state["memory_context"] = ""
    return state


def agent_node(
    state: AgentState,
    client: anthropic.Anthropic,
    model: str = MODEL,
) -> AgentState:
    system = "You are a helpful assistant with persistent memory."
    if state["memory_context"]:
        system += f"\n\n{state['memory_context']}"
    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=system,
        messages=state["messages"],
    )
    state["response"] = response.content[0].text
    log.info("agent_node: response length=%d", len(state["response"]))
    return state


def build_agent(
    service: MemoryService,
    client: anthropic.Anthropic | None = None,
    model: str = MODEL,
) -> Any:
    if client is None:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    graph = StateGraph(AgentState)
    graph.add_node("memory", lambda s: memory_node(s, service))
    graph.add_node("agent", lambda s: agent_node(s, client, model))
    graph.set_entry_point("memory")
    graph.add_edge("memory", "agent")
    graph.add_edge("agent", END)
    return graph.compile()
```

---

## `src/memory_service/evaluation.py`

```python
from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import UUID

from .service import MemoryService

log = logging.getLogger(__name__)


@dataclass
class QueryCase:
    query: str
    relevant_ids: set[UUID]


@dataclass
class EvalResult:
    query: str
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    retrieved_ids: list[UUID]


@dataclass
class EvalSummary:
    mean_precision: float
    mean_recall: float
    mrr: float
    map_at_k: float
    k: int


def precision_at_k(retrieved: list[UUID], relevant: set[UUID], k: int) -> float:
    hits = sum(1 for uid in retrieved[:k] if uid in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved: list[UUID], relevant: set[UUID], k: int) -> float:
    if not relevant:
        return 1.0
    hits = len(set(retrieved[:k]) & relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: list[UUID], relevant: set[UUID]) -> float:
    for i, uid in enumerate(retrieved, 1):
        if uid in relevant:
            return 1.0 / i
    return 0.0


def mean_average_precision(
    retrieved_lists: list[list[UUID]],
    relevant_sets: list[set[UUID]],
    k: int,
) -> float:
    aps: list[float] = []
    for retrieved, relevant in zip(retrieved_lists, relevant_sets):
        if not relevant:
            continue
        hits, precision_sum = 0, 0.0
        for rank, uid in enumerate(retrieved[:k], 1):
            if uid in relevant:
                hits += 1
                precision_sum += hits / rank
        aps.append(precision_sum / len(relevant))
    return sum(aps) / len(aps) if aps else 0.0


class MemoryEvaluator:
    def __init__(self, service: MemoryService, k: int = 5) -> None:
        self._service = service
        self.k = k
        self._last_cases: list[QueryCase] = []

    def evaluate(self, cases: list[QueryCase]) -> list[EvalResult]:
        self._last_cases = cases
        results: list[EvalResult] = []
        for case in cases:
            hits = self._service.retrieve(case.query, top_k=self.k)
            retrieved_ids = [r.record.id for r in hits]
            results.append(EvalResult(
                query=case.query,
                precision_at_k=precision_at_k(retrieved_ids, case.relevant_ids, self.k),
                recall_at_k=recall_at_k(retrieved_ids, case.relevant_ids, self.k),
                reciprocal_rank=reciprocal_rank(retrieved_ids, case.relevant_ids),
                retrieved_ids=retrieved_ids,
            ))
        return results

    def summarise(self, results: list[EvalResult]) -> EvalSummary:
        n = len(results)
        if n == 0:
            return EvalSummary(0, 0, 0, 0, self.k)
        return EvalSummary(
            mean_precision=sum(r.precision_at_k for r in results) / n,
            mean_recall=sum(r.recall_at_k for r in results) / n,
            mrr=sum(r.reciprocal_rank for r in results) / n,
            map_at_k=mean_average_precision(
                [r.retrieved_ids for r in results],
                [c.relevant_ids for c in self._last_cases],
                self.k,
            ),
            k=self.k,
        )
```

---

## `tests/conftest.py`

```python
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from memory_service.backends import InMemoryBackend
from memory_service.models import MemoryConfig, Message


class MockEmbedder:
    DIM = 8

    def embed(self, text: str) -> list[float]:
        digest = hashlib.md5(text.encode()).digest()
        raw = [float(digest[i % 16]) / 255.0 for i in range(self.DIM)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]


class MockSummariser:
    def summarise(self, messages: list[Message]) -> str | None:
        combined = " ".join(m.content for m in messages).lower()
        if any(kw in combined for kw in ["prefer", "like", "want", "need", "always"]):
            user_msgs = [m.content for m in messages if m.role == "user"]
            return user_msgs[-1] if user_msgs else None
        return None


class MockLLMClient:
    def __init__(self, fixed_response: str = "Mock LLM response") -> None:
        self.fixed_response = fixed_response
        self.call_count = 0
        self.messages = self  # for client.messages.create(...)

    def create(self, **kwargs) -> MagicMock:
        self.call_count += 1
        mock = MagicMock()
        mock.content = [MagicMock(text=self.fixed_response)]
        return mock


@pytest.fixture
def config() -> MemoryConfig:
    return MemoryConfig()


@pytest.fixture
def backend() -> InMemoryBackend:
    return InMemoryBackend()


@pytest.fixture
def embedder() -> MockEmbedder:
    return MockEmbedder()


@pytest.fixture
def summariser() -> MockSummariser:
    return MockSummariser()
```

---

## Key test patterns

### `test_service.py` — write-retrieve round-trip

```python
def test_write_retrieve_round_trip(backend, embedder, summariser, config):
    from memory_service.service import MemoryService
    from memory_service.models import Message

    svc = MemoryService(backend, summariser, embedder, config)
    msgs = [
        Message(role="user", content="I prefer PDF reports"),
        Message(role="assistant", content="Got it."),
    ]
    record = svc.write(msgs)
    assert record is not None

    results = svc.retrieve("report format PDF")
    assert len(results) > 0
    assert results[0].record.id == record.id
```

### `test_agent.py` — no real API calls

```python
def test_memory_node_injects_context(backend, embedder, summariser, config):
    from memory_service.service import MemoryService
    from memory_service.agent import memory_node, AgentState
    from memory_service.models import Message, MemoryRecord

    svc = MemoryService(backend, summariser, embedder, config)
    # Pre-populate a memory
    record = MemoryRecord(
        content="User prefers PDF",
        vector=embedder.embed("User prefers PDF"),
        importance=0.9,
    )
    backend.upsert(record)

    state: AgentState = {
        "messages": [{"role": "user", "content": "what format do you use"}],
        "memory_context": "",
        "response": "",
    }
    new_state = memory_node(state, svc)
    assert "PDF" in new_state["memory_context"]


def test_agent_node_uses_mock_client(backend, embedder, summariser, config):
    from memory_service.service import MemoryService
    from memory_service.agent import agent_node, AgentState

    svc = MemoryService(backend, summariser, embedder, config)
    mock_client = MockLLMClient(fixed_response="Here is your PDF.")

    state: AgentState = {
        "messages": [{"role": "user", "content": "give me a report"}],
        "memory_context": "Relevant memories:\n- User prefers PDF",
        "response": "",
    }
    new_state = agent_node(state, mock_client)
    assert new_state["response"] == "Here is your PDF."
    assert mock_client.call_count == 1
```

### `test_backends.py` — FileBackend persistence

```python
def test_file_backend_persists(tmp_path):
    from memory_service.backends import FileBackend
    from memory_service.models import MemoryRecord

    path = tmp_path / "memories.jsonl"
    b1 = FileBackend(path)
    r = MemoryRecord(content="persistent fact", importance=0.8)
    b1.upsert(r)
    del b1

    b2 = FileBackend(path)
    assert b2.count() == 1
    loaded = b2.all()[0]
    assert loaded.content == "persistent fact"
    assert loaded.id == r.id
```
