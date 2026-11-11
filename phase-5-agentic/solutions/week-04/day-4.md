# Solutions — Phase 5 · Week 04 · Day 4 — Long-Term Semantic Memory: Vector-Backed

---

> All tests use **deterministic mock embeddings** derived from a hash of the text string.
> No real API calls are made. Cosine similarity is computed locally.

---

## Shared Infrastructure

The exercises below all build on a common `VectorMemoryStore`. Paste this once at the top of your
implementation file (or import it from a shared module).

```python
"""
vector_memory.py — Vector-backed long-term memory with mock embeddings.
"""
from __future__ import annotations

import hashlib
import logging
import math
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mock embedding — deterministic, hash-based, no API call
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 16


def _mock_embed(text: str) -> list[float]:
    """Produce a deterministic unit-normalised embedding for *text*.

    Uses SHA-256 of the text to fill a fixed-dimension vector, then L2-normalises.
    Texts that share common substrings will NOT necessarily be similar — this
    is purely for testing that the plumbing works correctly.
    """
    digest = hashlib.sha256(text.encode()).digest()
    # Take the first EMBEDDING_DIM bytes as floats in [0, 1)
    raw = [b / 255.0 for b in digest[:EMBEDDING_DIM]]
    magnitude = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / magnitude for x in raw]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a)) or 1.0
    mag_b = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class MemoryEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Base store used by all exercises
# ---------------------------------------------------------------------------

EmbedFn = Any  # Callable[[str], list[float]]


class VectorMemoryStore:
    """In-memory vector store with cosine-similarity search."""

    def __init__(self, embed_fn: EmbedFn = _mock_embed) -> None:
        self._embed_fn = embed_fn
        self._entries: dict[str, MemoryEntry] = {}

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> MemoryEntry:
        embedding = self._embed_fn(text)
        entry = MemoryEntry(
            text=text,
            embedding=embedding,
            metadata=metadata or {},
        )
        self._entries[entry.id] = entry
        logger.debug("Added memory %s: %s", entry.id, text[:60])
        return entry

    def search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
    ) -> list[tuple[MemoryEntry, float]]:
        """Return top-k entries by cosine similarity above *similarity_threshold*."""
        q_emb = self._embed_fn(query)
        scored = [
            (entry, cosine_similarity(q_emb, entry.embedding))
            for entry in self._entries.values()
        ]
        filtered = [(e, s) for e, s in scored if s >= similarity_threshold]
        filtered.sort(key=lambda x: x[1], reverse=True)
        return filtered[:top_k]

    def all_entries(self) -> list[MemoryEntry]:
        return list(self._entries.values())
```

---

## Exercise 1

Add `delete_memory(memory_id: str)`. Deleted memories must not appear in search.

```python
"""
vector_memory_delete.py — VectorMemoryStore extended with delete_memory.
"""
from __future__ import annotations

import logging

from vector_memory import VectorMemoryStore, MemoryEntry, _mock_embed

logger = logging.getLogger(__name__)


class VectorMemoryStore(VectorMemoryStore):  # type: ignore[no-redef]
    """Extends base store with memory deletion."""

    def delete_memory(self, memory_id: str) -> None:
        """Remove the entry with *memory_id* from the store.

        Silently ignores IDs that do not exist.
        """
        removed = self._entries.pop(memory_id, None)
        if removed:
            logger.debug("Deleted memory %s", memory_id)
        else:
            logger.debug("delete_memory called for unknown id: %s", memory_id)
```

```python
"""
tests/test_vector_delete.py — Tests for delete_memory.
"""
from __future__ import annotations

import pytest

from vector_memory_delete import VectorMemoryStore


def test_deleted_memory_not_in_search() -> None:
    store = VectorMemoryStore()
    entry = store.add("The sky is blue")
    store.delete_memory(entry.id)
    results = store.search("sky")
    ids = [e.id for e, _ in results]
    assert entry.id not in ids


def test_delete_nonexistent_id_silent() -> None:
    store = VectorMemoryStore()
    store.delete_memory("nonexistent-id-xyz")  # must not raise


def test_delete_only_removes_target() -> None:
    store = VectorMemoryStore()
    e1 = store.add("Memory one")
    e2 = store.add("Memory two")
    store.delete_memory(e1.id)
    remaining = [e.id for e in store.all_entries()]
    assert e1.id not in remaining
    assert e2.id in remaining


def test_deleted_entry_gone_from_all_entries() -> None:
    store = VectorMemoryStore()
    entry = store.add("Delete me")
    store.delete_memory(entry.id)
    assert all(e.id != entry.id for e in store.all_entries())
```

---

## Exercise 2

`filter_by_metadata(key: str, value: str) -> list[MemoryEntry]`.

```python
"""
vector_memory_filter.py — VectorMemoryStore extended with filter_by_metadata.
"""
from __future__ import annotations

import logging
from typing import Any

from vector_memory_delete import VectorMemoryStore

logger = logging.getLogger(__name__)


class VectorMemoryStore(VectorMemoryStore):  # type: ignore[no-redef]
    """Adds metadata filtering to the store."""

    def filter_by_metadata(self, key: str, value: Any) -> list:
        """Return all entries where metadata[key] == value."""
        results = [
            entry
            for entry in self._entries.values()
            if entry.metadata.get(key) == value
        ]
        logger.debug(
            "filter_by_metadata(%s=%s): %d results", key, value, len(results)
        )
        return results
```

```python
"""
tests/test_vector_filter.py — Tests for filter_by_metadata.
"""
from __future__ import annotations

import pytest

from vector_memory_filter import VectorMemoryStore


def test_filter_returns_matching_entries() -> None:
    store = VectorMemoryStore()
    store.add("SAP BTP intro", metadata={"source": "docs", "topic": "btp"})
    store.add("Python basics", metadata={"source": "book", "topic": "python"})
    store.add("SAP HANA overview", metadata={"source": "docs", "topic": "hana"})

    results = store.filter_by_metadata("source", "docs")
    assert len(results) == 2
    topics = {e.metadata["topic"] for e in results}
    assert topics == {"btp", "hana"}


def test_filter_no_match_returns_empty() -> None:
    store = VectorMemoryStore()
    store.add("Something", metadata={"type": "fact"})
    assert store.filter_by_metadata("type", "opinion") == []


def test_filter_empty_store() -> None:
    store = VectorMemoryStore()
    assert store.filter_by_metadata("any", "value") == []


def test_filter_exact_value_match() -> None:
    store = VectorMemoryStore()
    store.add("Entry A", metadata={"priority": 1})
    store.add("Entry B", metadata={"priority": 2})
    results = store.filter_by_metadata("priority", 1)
    assert len(results) == 1
    assert results[0].text == "Entry A"
```

---

## Exercise 3

`similarity_threshold` parameter in `search()` filters low-similarity results.

The `VectorMemoryStore.search()` in `vector_memory.py` already accepts `similarity_threshold`.
Here we extend it with explicit tests and a helper to control the mock embeddings for
threshold-testing purposes.

```python
"""
vector_memory_threshold.py — Controlled embeddings for threshold testing.
"""
from __future__ import annotations

import math
import logging
from typing import Any

from vector_memory import VectorMemoryStore, MemoryEntry, cosine_similarity

logger = logging.getLogger(__name__)


def _fixed_embed(value: list[float]):
    """Return an embed function that always returns *value* (unit-normalised)."""
    mag = math.sqrt(sum(x * x for x in value)) or 1.0
    normalised = [x / mag for x in value]

    def embed(text: str) -> list[float]:  # noqa: ARG001
        return normalised

    return embed


def _rotating_embed(vectors: list[list[float]]):
    """Returns an embed function cycling through *vectors* for successive calls."""
    state = {"idx": 0}
    normalised_vecs = []
    for v in vectors:
        mag = math.sqrt(sum(x * x for x in v)) or 1.0
        normalised_vecs.append([x / mag for x in v])

    def embed(text: str) -> list[float]:  # noqa: ARG001
        vec = normalised_vecs[state["idx"] % len(normalised_vecs)]
        state["idx"] += 1
        return vec

    return embed
```

```python
"""
tests/test_similarity_threshold.py — Tests for similarity_threshold in search().
"""
from __future__ import annotations

import pytest

from vector_memory import VectorMemoryStore
from vector_memory_threshold import _fixed_embed, _rotating_embed


def test_threshold_zero_returns_all() -> None:
    store = VectorMemoryStore()
    store.add("alpha")
    store.add("beta")
    results = store.search("query", top_k=10, similarity_threshold=0.0)
    assert len(results) == 2


def test_threshold_one_returns_near_identical_only() -> None:
    """Entries with same text -> same hash embedding -> similarity == 1.0."""
    store = VectorMemoryStore()
    store.add("exact text")
    store.add("different content")
    results = store.search("exact text", top_k=10, similarity_threshold=0.99)
    # Only "exact text" should have similarity ~1.0
    assert len(results) >= 1
    assert results[0][0].text == "exact text"


def test_threshold_filters_low_similarity() -> None:
    """Use controlled embeddings to set precise similarity values."""
    # Two orthogonal unit vectors -> similarity = 0
    high_vec = [1.0, 0.0]
    low_vec = [0.0, 1.0]
    query_vec = [1.0, 0.0]

    call_vecs = [high_vec, low_vec]
    embed = _rotating_embed(call_vecs)
    query_embed = _fixed_embed(query_vec)

    store = VectorMemoryStore(embed_fn=embed)
    store.add("high similarity entry")   # gets high_vec
    store.add("low similarity entry")    # gets low_vec

    # Query with high_vec via a fresh fixed embedder
    store._embed_fn = query_embed
    results = store.search("ignored", top_k=10, similarity_threshold=0.5)
    texts = [e.text for e, _ in results]
    assert "high similarity entry" in texts
    assert "low similarity entry" not in texts


def test_threshold_returns_empty_when_all_below() -> None:
    store = VectorMemoryStore()
    store.add("some memory")
    # Threshold of 1.0 requires exact match; query differs -> filtered out
    results = store.search("completely different query xyz", top_k=5, similarity_threshold=1.0)
    # May be empty if no exact match
    assert all(score == 1.0 for _, score in results)
```

---

## Exercise 4

`batch_add(memories: list[str])` adds all entries in one call.

```python
"""
vector_memory_batch.py — VectorMemoryStore with batch_add.
"""
from __future__ import annotations

import logging
from typing import Any

from vector_memory_filter import VectorMemoryStore
from vector_memory import MemoryEntry

logger = logging.getLogger(__name__)


class VectorMemoryStore(VectorMemoryStore):  # type: ignore[no-redef]
    """Adds batch_add to the store."""

    def batch_add(
        self,
        memories: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> list[MemoryEntry]:
        """Add all *memories* strings to the store.

        An optional shared *metadata* dict is applied to every entry.
        Returns the list of created MemoryEntry objects.
        """
        entries: list[MemoryEntry] = []
        for text in memories:
            entry = self.add(text, metadata=dict(metadata) if metadata else {})
            entries.append(entry)
        logger.debug("batch_add: inserted %d entries", len(entries))
        return entries
```

```python
"""
tests/test_vector_batch.py — Tests for batch_add.
"""
from __future__ import annotations

import pytest

from vector_memory_batch import VectorMemoryStore


def test_batch_add_inserts_all() -> None:
    store = VectorMemoryStore()
    texts = ["Memory A", "Memory B", "Memory C"]
    entries = store.batch_add(texts)
    assert len(entries) == 3
    stored_texts = {e.text for e in store.all_entries()}
    assert stored_texts == set(texts)


def test_batch_add_returns_entries() -> None:
    store = VectorMemoryStore()
    entries = store.batch_add(["X", "Y"])
    assert all(hasattr(e, "id") for e in entries)
    assert all(hasattr(e, "embedding") for e in entries)


def test_batch_add_empty_list() -> None:
    store = VectorMemoryStore()
    entries = store.batch_add([])
    assert entries == []
    assert store.all_entries() == []


def test_batch_add_with_shared_metadata() -> None:
    store = VectorMemoryStore()
    store.batch_add(["fact one", "fact two"], metadata={"source": "wiki"})
    for entry in store.all_entries():
        assert entry.metadata.get("source") == "wiki"


def test_batch_add_single_item() -> None:
    store = VectorMemoryStore()
    entries = store.batch_add(["Solo memory"])
    assert len(entries) == 1
    assert entries[0].text == "Solo memory"
```

---

## Exercise 5

`MemoryConsolidator`: deduplicate memories with similarity > 0.95, keeping the most recently added.

```python
"""
memory_consolidator.py — Deduplicates near-duplicate memories by similarity.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Callable

from vector_memory import MemoryEntry, VectorMemoryStore, cosine_similarity, _mock_embed

logger = logging.getLogger(__name__)


class MemoryConsolidator:
    """Wraps a VectorMemoryStore and removes near-duplicate entries.

    Two entries are considered duplicates when their cosine similarity exceeds
    *dedup_threshold*. When duplicates are found, the older entry is removed.
    """

    def __init__(
        self,
        store: VectorMemoryStore | None = None,
        dedup_threshold: float = 0.95,
    ) -> None:
        self._store = store or VectorMemoryStore()
        self.dedup_threshold = dedup_threshold

    def add(self, text: str) -> MemoryEntry:
        """Add *text* and remove any near-duplicates that already exist."""
        new_entry = self._store.add(text)
        self._consolidate(new_entry)
        return new_entry

    def _consolidate(self, new_entry: MemoryEntry) -> None:
        """Remove existing entries that are near-duplicates of *new_entry*."""
        to_remove: list[str] = []
        for entry in self._store.all_entries():
            if entry.id == new_entry.id:
                continue
            sim = cosine_similarity(new_entry.embedding, entry.embedding)
            if sim > self.dedup_threshold:
                logger.debug(
                    "Consolidating duplicate: %s (sim=%.3f) -> keeping newer %s",
                    entry.id, sim, new_entry.id,
                )
                to_remove.append(entry.id)

        for eid in to_remove:
            self._store._entries.pop(eid, None)

    def search(self, query: str, top_k: int = 5) -> list[tuple[MemoryEntry, float]]:
        return self._store.search(query, top_k=top_k)

    def all_entries(self) -> list[MemoryEntry]:
        return self._store.all_entries()
```

```python
"""
tests/test_memory_consolidator.py — Tests for MemoryConsolidator deduplication.
"""
from __future__ import annotations

import math

import pytest

from vector_memory import VectorMemoryStore, cosine_similarity
from memory_consolidator import MemoryConsolidator


# ---------------------------------------------------------------------------
# Deterministic near-identical embeddings for dedup testing
# ---------------------------------------------------------------------------

_CALL_COUNT: dict[str, int] = {}
_TEXTS_TO_VEC: dict[str, list[float]] = {}


def _controlled_embed(text: str) -> list[float]:
    """Return same vector for near-duplicate texts (sharing a key prefix).

    Texts prefixed with "NEAR:" share an embedding (similarity == 1.0).
    All other texts get the default hash-based mock embedding.
    """
    from vector_memory import _mock_embed

    if text.startswith("NEAR:"):
        # All NEAR: texts get the same vector -> similarity == 1.0
        base = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        return base
    return _mock_embed(text)


def test_near_duplicate_removed() -> None:
    store = VectorMemoryStore(embed_fn=_controlled_embed)
    consolidator = MemoryConsolidator(store=store, dedup_threshold=0.95)

    consolidator.add("NEAR: first version of the memory")
    consolidator.add("NEAR: second version of the memory")

    entries = consolidator.all_entries()
    # Only one should survive (the newer one)
    assert len(entries) == 1


def test_newer_entry_kept() -> None:
    store = VectorMemoryStore(embed_fn=_controlled_embed)
    consolidator = MemoryConsolidator(store=store, dedup_threshold=0.95)

    consolidator.add("NEAR: old memory")
    entries_before = consolidator.all_entries()

    consolidator.add("NEAR: new memory")
    entries_after = consolidator.all_entries()

    assert len(entries_after) == 1
    assert entries_after[0].text == "NEAR: new memory"


def test_distinct_memories_all_kept() -> None:
    store = VectorMemoryStore(embed_fn=_controlled_embed)
    consolidator = MemoryConsolidator(store=store, dedup_threshold=0.95)

    consolidator.add("Unique memory about Python")
    consolidator.add("Unique memory about LangGraph")
    consolidator.add("Unique memory about FastAPI")

    # These have hash-based embeddings — unlikely to be near-duplicates
    entries = consolidator.all_entries()
    assert len(entries) == 3


def test_threshold_respected() -> None:
    """With threshold=0.0 every entry is a duplicate — only newest survives."""
    store = VectorMemoryStore(embed_fn=_controlled_embed)
    consolidator = MemoryConsolidator(store=store, dedup_threshold=0.0)

    consolidator.add("NEAR: alpha")
    consolidator.add("NEAR: beta")
    consolidator.add("NEAR: gamma")

    entries = consolidator.all_entries()
    assert len(entries) == 1
    assert entries[0].text == "NEAR: gamma"


def test_single_entry_no_consolidation() -> None:
    store = VectorMemoryStore(embed_fn=_controlled_embed)
    consolidator = MemoryConsolidator(store=store, dedup_threshold=0.95)
    consolidator.add("Only memory")
    assert len(consolidator.all_entries()) == 1
```
