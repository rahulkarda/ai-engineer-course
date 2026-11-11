# Phase 5 · Week 04 · Day 4 — Long-Term Semantic Memory: Vector-Backed

> Objective: Build a vector memory store that retrieves relevant past facts by semantic similarity  ·  Time: ~1 hr  ·  Prereqs: [[day-3]], week-4 RAG concepts (vector stores, embeddings, similarity search)

## Concept

Short-term memory (days 2–3) is bounded by the context window. Long-term semantic memory breaks that constraint: facts are stored outside the context and retrieved on demand based on **meaning**, not recency.

How it works:
1. When the agent observes a fact worth remembering, embed it as a dense vector and store it.
2. At the start of each turn, embed the current query and retrieve the top-k most similar stored facts.
3. Inject the retrieved facts into the context as a `[Relevant memories]` block.

**Cosine similarity** measures the angle between two unit vectors. It is 1.0 for identical direction (most similar), 0.0 for orthogonal, −1.0 for opposite. For memory retrieval, anything above ~0.75 is typically relevant.

```
cosine_similarity(a, b) = (a · b) / (||a|| * ||b||)
```

Why this matters:
- An agent can "remember" facts from thousands of turns ago without keeping them in the context window.
- Retrieval is selective — only relevant facts are injected, not the entire history.
- The vector store is queryable at sub-millisecond latency with in-memory numpy for small datasets (<100K entries).

For production, replace the in-memory store with a vector database (pgvector, Chroma, Pinecone, Weaviate).

## How it works internally

**Embedding:** A text string is passed to an embedding model (e.g., `voyage-3` via Anthropic) which returns a fixed-size float vector (e.g., 1024 dimensions). Similar texts produce vectors pointing in similar directions.

**Storage:** Each memory entry stores:
- The original text (for injection into prompts).
- Its embedding vector (for similarity search).
- Metadata (source, timestamp, tags) for filtering.

**Retrieval:**
1. Embed the query.
2. Compute cosine similarity between the query vector and every stored vector.
3. Return top-k entries by similarity score.

With numpy, this is a single matrix multiply — O(n · d) where n = number of entries, d = embedding dimension. For n < 10K this is fast enough without an index. For larger stores, use HNSW indexing (e.g., `hnswlib` or `faiss`).

**Mock embedding for reproducibility:** In tests and this lesson's worked example, the embedding function is mocked: it hashes the text and uses the hash as a seed for `numpy.random.default_rng(seed).standard_normal(dim)`, then normalises to unit length. This means:
- Same text → same vector (deterministic).
- Different texts → different vectors (good enough for testing retrieval logic).

## Worked example

```python
"""
day_4_vector_memory.py

VectorMemoryStore — in-memory vector store using numpy cosine similarity.
Embedding is mocked for reproducibility; swap mock_embed() for a real
embedding call when ANTHROPIC_API_KEY is set.

No external vector DB required.
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Embedding dimension for the mock; real voyage-3 uses 1024.
EMBED_DIM = 128
MODEL = "claude-opus-5-5"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class MemoryEntry(BaseModel):
    """A single stored memory fact with its embedding."""

    id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"arbitrary_types_allowed": True}


class SearchResult(BaseModel):
    """A retrieved memory with its similarity score."""

    entry: MemoryEntry
    score: float


# ---------------------------------------------------------------------------
# Embedding functions
# ---------------------------------------------------------------------------

def mock_embed(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    """
    Deterministic mock embedding.

    Uses SHA-256 of the text as an RNG seed so identical text always
    produces the identical unit vector. Different texts → different vectors.
    """
    digest = hashlib.sha256(text.encode()).digest()
    seed = int.from_bytes(digest[:4], "big")
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        vec = np.ones(dim, dtype=np.float32)
        norm = np.sqrt(dim)
    return vec / norm


def live_embed(text: str) -> np.ndarray:
    """
    Real embedding via Anthropic's voyage-3 model.
    Requires: pip install anthropic
    Only called when ANTHROPIC_API_KEY is set.
    """
    import anthropic  # type: ignore[import]

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    # Anthropic embedding endpoint (voyage-3 via client)
    response = client.embeddings.create(
        model="voyage-3",
        input=text,
    )
    vec = np.array(response.embeddings[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / max(norm, 1e-10)


def embed(text: str) -> np.ndarray:
    """Dispatch to mock or live embedding based on env."""
    if os.getenv("ANTHROPIC_API_KEY"):
        logger.info("embed: using live Anthropic embedding")
        return live_embed(text)
    logger.debug("embed: ANTHROPIC_API_KEY not set — using mock")
    return mock_embed(text)


# ---------------------------------------------------------------------------
# VectorMemoryStore
# ---------------------------------------------------------------------------

class VectorMemoryStore:
    """
    In-memory vector store for semantic memory retrieval.

    Uses numpy for cosine similarity — suitable for up to ~10K entries
    before requiring an indexed store (e.g., faiss, hnswlib).
    """

    def __init__(self) -> None:
        self._entries: list[MemoryEntry] = []
        self._matrix: np.ndarray | None = None  # (n, dim) float32, unit-normalised
        self._dirty: bool = False                 # recompute matrix on next search
        logger.info("VectorMemoryStore init")

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def add_memory(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        memory_id: str | None = None,
    ) -> MemoryEntry:
        """
        Embed text and store as a new memory entry.

        Returns the created MemoryEntry (embedding stored as list[float]
        for Pydantic serialisability; numpy array used internally).
        """
        vec = embed(text)
        entry_id = memory_id or f"mem-{len(self._entries):06d}"
        entry = MemoryEntry(
            id=entry_id,
            text=text,
            embedding=vec.tolist(),
            metadata=metadata or {},
        )
        self._entries.append(entry)
        self._dirty = True
        logger.info(
            "add_memory: id=%s, text='%s...'",
            entry_id,
            text[:50],
        )
        return entry

    def _rebuild_matrix(self) -> None:
        """Rebuild the (n, dim) embedding matrix from stored entries."""
        if not self._entries:
            self._matrix = None
            self._dirty = False
            return
        self._matrix = np.array(
            [e.embedding for e in self._entries], dtype=np.float32
        )
        self._dirty = False
        logger.debug("_rebuild_matrix: shape=%s", self._matrix.shape)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 3,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """
        Retrieve top-k memories most similar to the query.

        Steps:
          1. Embed the query.
          2. Compute cosine similarity against all stored embeddings.
          3. Return top-k results sorted by descending similarity.

        Args:
            query:     Natural language query string.
            k:         Maximum number of results to return.
            min_score: Discard results with similarity below this threshold.

        Returns:
            List of SearchResult, sorted by score descending.
        """
        if not self._entries:
            logger.info("search: store is empty")
            return []

        if self._dirty:
            self._rebuild_matrix()

        query_vec = embed(query).reshape(1, -1)  # (1, dim)

        # cosine similarity: both query_vec and stored vecs are already unit-normalised
        # so dot product == cosine similarity
        scores: np.ndarray = (self._matrix @ query_vec.T).flatten()  # (n,)

        # top-k indices by descending score
        k_capped = min(k, len(self._entries))
        top_indices = np.argsort(scores)[::-1][:k_capped]

        results: list[SearchResult] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < min_score:
                continue
            results.append(SearchResult(entry=self._entries[idx], score=score))

        logger.info(
            "search: query='%s...', returned %d/%d results",
            query[:40],
            len(results),
            len(self._entries),
        )
        return results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def delete(self, memory_id: str) -> bool:
        """Remove a memory by id. Returns True if found."""
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.id != memory_id]
        removed = len(self._entries) < before
        if removed:
            self._dirty = True
            logger.info("delete: removed id=%s", memory_id)
        return removed

    def render_retrieved(self, results: list[SearchResult]) -> str:
        """Format retrieved memories as a prompt-ready block."""
        if not results:
            return ""
        lines = ["[Relevant memories]"]
        for r in results:
            lines.append(f"- (score={r.score:.3f}) {r.entry.text}")
        return "\n".join(lines)

    @property
    def size(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

def simulate_agent_with_memory() -> None:
    """Store facts from a session and query them semantically."""
    store = VectorMemoryStore()

    # Populate memory with facts the agent learned during earlier sessions
    facts = [
        "The user prefers concise answers without unnecessary detail.",
        "The user is working on a LangGraph agent for SAP data processing.",
        "The user wants all code in Python 3.12 with strict type hints.",
        "The project uses Pydantic v2 for all data models.",
        "The user's team uses pytest for all testing; no unittest allowed.",
        "The user's primary language is English but reads German fluently.",
        "The agent should log with logging module, never with print statements.",
        "The project deadline is end of Q1.",
        "The user prefers Claude models over other LLM providers.",
        "The codebase lives on SAP GitHub Enterprise.",
    ]

    for i, fact in enumerate(facts):
        store.add_memory(fact, metadata={"source": "session-0", "index": i})

    print(f"\nStored {store.size} memories.\n")

    # Simulate a new turn: the agent retrieves relevant memories for a query
    queries = [
        "What testing framework does the user prefer?",
        "How should I format code in this project?",
        "When is the project due?",
    ]

    for query in queries:
        print(f"Query: {query}")
        results = store.search(query, k=3, min_score=0.0)
        print(store.render_retrieved(results))
        print()


if __name__ == "__main__":
    simulate_agent_with_memory()
```

**Run:**
```bash
pip install numpy pydantic
python day_4_vector_memory.py
```

With the mock embedder the similarity scores are low (random vectors are nearly orthogonal) — this is expected. With a real embedding model scores will reflect true semantic similarity.

**Run with real embeddings (optional):**
```bash
ANTHROPIC_API_KEY=your_key python day_4_vector_memory.py
```

## Exercises

1. Write a pytest test for `VectorMemoryStore.search` that:
   - Adds three memory entries with `mock_embed`.
   - Asserts that searching with the exact text of the first entry returns that entry at rank 1 with `score == 1.0`.
   - (Hint: `mock_embed(text)` is deterministic, so querying with the same text gives the same vector.)

2. Implement `VectorMemoryStore.update_memory(memory_id, new_text)` that re-embeds and replaces an existing entry in-place. Write a pytest test verifying that the updated entry is returned by search with the new text and not with the old text.

3. Add a `metadata_filter` parameter to `VectorMemoryStore.search(query, k=3, metadata_filter=None)`. When provided, `metadata_filter` is a `dict[str, Any]` and only entries whose metadata is a superset of it are candidates. Write a pytest test with two entries sharing one source tag and one not, and assert the filter correctly restricts results.

4. Implement `VectorMemoryStore.batch_add(texts, metadata_list)` that embeds all texts in a single loop and adds them. Add a test checking that batch-adding 50 entries and individually-adding 50 identical entries produce the same embeddings (same mock vectors).

(Solutions: ../solutions/week-04/day-4.md)

## Self-check

- [ ] I can explain why short-term memory alone is insufficient for long-running agents.
- [ ] I can explain cosine similarity, write the formula, and describe what high/low scores mean for retrieval.
- [ ] I can implement `VectorMemoryStore` with `add_memory()`, `search()`, and `render_retrieved()` from scratch using numpy.
- [ ] I can explain why both the stored embeddings and query vector must be unit-normalised before a dot-product similarity search.
- [ ] I can explain the mock embedding design (hash-seeded RNG → deterministic vectors) and why it is suitable for tests but not for production.
