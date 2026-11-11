# Elective: Long-Term Memory · Lesson 2 — Memory Architectures: Vector, Graph, KV

> Objective: Understand the three storage primitives and how to combine them for a complete memory backend.
> Time: ~60 min · Prereqs: [[lesson-1.md]]

---

## Concept

No single database technology handles all memory types well. A production memory system
combines three storage primitives, each handling what it does best:

| Store type | Best for | Query model | Weakness |
|-----------|---------|-------------|---------|
| **Vector store** | Semantic similarity search | Nearest-neighbour (ANN) | No structured filtering |
| **Graph store** | Entity relationships + multi-hop traversal | Graph query (Cypher/Gremlin) | Slow on unstructured text |
| **KV store** | Exact lookup, session state, procedures | Key → value | No similarity |

### Vector stores

A vector store embeds content as high-dimensional float vectors, then retrieves records
by **cosine similarity** (or dot product, L2 distance).

Core operations:
- `upsert(id, vector, metadata)` — add or replace a record
- `query(vector, top_k, filter)` — return k nearest neighbours, optionally filtered by metadata

Used for: episodic memory (past conversations), semantic facts (when you want "similar
facts" not exact match).

Popular backends: ChromaDB, Pinecone, Qdrant, Weaviate, pgvector.

### Graph stores

A graph store holds **nodes** (entities) and **edges** (relationships) with properties
on both. It answers questions like:

- "What projects does this user own?"
- "Which users share a preference for Python?"
- "What did the agent recommend two hops from topic X?"

Used for: semantic memory with entity relationships, knowledge graphs.

Popular backends: Neo4j, Amazon Neptune, Memgraph.

### KV stores

A KV store maps a string key to an arbitrary value (string, JSON, bytes). Lightning fast
for exact lookup.

Used for: procedural memory (keyed by trigger), session state, configuration, counters.

Popular backends: Redis, DynamoDB, SQLite (as a KV via JSON column), plain files.

### Combining the three

A production memory service uses all three in a **layered architecture**:

```
Agent
  │
  ▼
Memory Service
  ├── Vector store  ← episodic + semantic similarity retrieval
  ├── Graph store   ← entity relationships + structured semantic facts
  └── KV store      ← procedures, session state, user prefs (exact)
```

The memory service presents a single interface; callers never touch individual stores.

---

## How it works internally

### Embedding pipeline

Before storing in a vector store, text must be embedded:

```
raw_text
    │
    ▼
Embedding model (e.g. text-embedding-3-small)
    │
    ▼
float32 vector [d=1536]
    │
    ▼
Vector store (id, vector, metadata={...})
```

Embedding latency is the dominant cost in write paths. Batch embedding amortises this.

### Approximate nearest-neighbour (ANN)

Vector stores do not scan every vector at query time. They use **ANN indices**:

- **HNSW** (Hierarchical Navigable Small World) — graph-based, fast inserts, fast queries
- **IVF** (Inverted File) — cluster-based, slower build, faster large-scale query
- **Flat** — exact scan, no index, only viable under ~10k records

ChromaDB and most managed services default to HNSW.

### Cosine similarity

For two vectors **a** and **b**:

```
cos_sim(a, b) = (a · b) / (||a|| * ||b||)
```

Range: −1 (opposite) to 1 (identical). In practice, embedding similarity scores for
semantically related text land in 0.7–0.95.

---

## Worked example

```python
"""
lesson_2_architectures.py

A minimal in-process implementation of all three store types
using only numpy (vector) and plain dicts (KV/graph).
No external databases required — swap the backends in the project.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    dot = float(np.dot(va, vb))
    norm = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return dot / norm if norm > 1e-9 else 0.0


# ── In-process vector store ───────────────────────────────────────────────────

@dataclass
class VectorRecord:
    id: UUID
    vector: list[float]
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class InMemoryVectorStore:
    """Simple cosine-similarity vector store for development and tests."""

    def __init__(self) -> None:
        self._records: list[VectorRecord] = []

    def upsert(
        self,
        content: str,
        vector: list[float],
        metadata: dict[str, Any] | None = None,
        id: UUID | None = None,
    ) -> UUID:
        record_id = id or uuid4()
        # Replace existing record with same id
        self._records = [r for r in self._records if r.id != record_id]
        self._records.append(
            VectorRecord(id=record_id, vector=vector, content=content, metadata=metadata or {})
        )
        log.debug("VectorStore.upsert id=%s", record_id)
        return record_id

    def query(
        self,
        query_vector: list[float],
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[VectorRecord, float]]:
        candidates = self._records
        if metadata_filter:
            candidates = [
                r for r in candidates
                if all(r.metadata.get(k) == v for k, v in metadata_filter.items())
            ]
        scored = [
            (r, cosine_similarity(query_vector, r.vector))
            for r in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def delete(self, id: UUID) -> bool:
        before = len(self._records)
        self._records = [r for r in self._records if r.id != id]
        return len(self._records) < before

    def __len__(self) -> int:
        return len(self._records)


# ── In-process graph store ────────────────────────────────────────────────────

@dataclass
class GraphNode:
    id: str
    label: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str
    properties: dict[str, Any] = field(default_factory=dict)


class InMemoryGraphStore:
    """Directed property graph — minimal API for semantic memory."""

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._edges: list[GraphEdge] = []

    def add_node(self, id: str, label: str, **props: Any) -> GraphNode:
        node = GraphNode(id=id, label=label, properties=props)
        self._nodes[id] = node
        log.debug("GraphStore.add_node id=%s label=%s", id, label)
        return node

    def add_edge(self, source: str, target: str, relation: str, **props: Any) -> GraphEdge:
        edge = GraphEdge(source=source, target=target, relation=relation, properties=props)
        self._edges.append(edge)
        log.debug("GraphStore.add_edge %s -[%s]-> %s", source, relation, target)
        return edge

    def get_neighbours(self, node_id: str, relation: str | None = None) -> list[GraphNode]:
        """Return nodes reachable from node_id via optional relation filter."""
        targets = [
            e.target for e in self._edges
            if e.source == node_id and (relation is None or e.relation == relation)
        ]
        return [self._nodes[t] for t in targets if t in self._nodes]

    def find_nodes(self, label: str) -> list[GraphNode]:
        return [n for n in self._nodes.values() if n.label == label]


# ── In-process KV store ───────────────────────────────────────────────────────

class InMemoryKVStore:
    """Key-value store for procedures, session state, exact lookups."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value
        log.debug("KVStore.set key=%s", key)

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def delete(self, key: str) -> bool:
        existed = key in self._store
        self._store.pop(key, None)
        return existed

    def keys_with_prefix(self, prefix: str) -> list[str]:
        return [k for k in self._store if k.startswith(prefix)]


# ── Demo: using all three together ────────────────────────────────────────────

def main() -> None:
    vs = InMemoryVectorStore()
    gs = InMemoryGraphStore()
    kv = InMemoryKVStore()

    # --- Vector store: episodic memory ---
    # In production, vectors come from an embedding model.
    # Here we use fixed mock vectors (unit-normalised for clean cosine scores).
    def mock_embed(text: str) -> list[float]:
        """Deterministic 4-D mock embedding based on text length."""
        n = len(text)
        raw = [math.sin(n), math.cos(n), math.sin(n * 2), math.cos(n * 2)]
        norm = math.sqrt(sum(x * x for x in raw))
        return [x / norm for x in raw]

    id1 = vs.upsert("User asked about Q3 report", mock_embed("User asked about Q3 report"))
    id2 = vs.upsert("User prefers dark mode UI", mock_embed("User prefers dark mode UI"))
    id3 = vs.upsert("Agent summarised meeting notes", mock_embed("Agent summarised meeting notes"))

    query = mock_embed("quarterly report question")
    results = vs.query(query, top_k=2)
    log.info("Vector query top-2:")
    for rec, score in results:
        log.info("  %.3f — %s", score, rec.content)

    # --- Graph store: semantic memory as entity graph ---
    gs.add_node("user_42", "User", name="Alice")
    gs.add_node("proj_crm", "Project", name="CRM Upgrade")
    gs.add_node("pref_pdf", "Preference", format="PDF")
    gs.add_edge("user_42", "proj_crm", "OWNS")
    gs.add_edge("user_42", "pref_pdf", "PREFERS")

    neighbours = gs.get_neighbours("user_42")
    log.info("Graph neighbours of user_42: %s", [n.id for n in neighbours])

    # --- KV store: procedural memory ---
    kv.set("proc:summarise", "Always start with a TL;DR bullet list.")
    kv.set("proc:greet_returning", "Reference last session's main topic.")

    procs = kv.keys_with_prefix("proc:")
    log.info("Procedures: %s", procs)


if __name__ == "__main__":
    main()
```

---

## Exercises

1. Extend `InMemoryVectorStore.query` to accept a `min_score: float = 0.0` parameter
   and filter out results below that threshold. Write two pytest cases: one where all
   results pass, one where some are filtered.

2. Add a `get_path(start: str, end: str) -> list[str] | None` method to
   `InMemoryGraphStore` that does a BFS and returns the node IDs on the shortest path,
   or `None` if no path exists.

3. Implement `InMemoryKVStore.get_all_with_prefix(prefix: str) -> dict[str, Any]` and
   write a test that stores three procedures and retrieves all of them at once.

4. The worked example uses `mock_embed` as a 4-dimensional deterministic function.
   Explain why this mock is acceptable for unit tests but would fail in an integration
   test. What would a better integration-test mock look like?

(Solutions: [solutions/lesson-2.md](solutions/lesson-2.md))

---

## Self-check

- [ ] I can explain which store type is best for each memory type
- [ ] I can implement a cosine similarity search in pure Python/numpy
- [ ] I understand the role of ANN indices and why exact search doesn't scale
- [ ] I can model entity relationships in a simple graph store
