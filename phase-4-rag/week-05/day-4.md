# Phase 4 · Week 05 · Day 4 — Query Expansion & Parent-Document Retrieval

> Objective: Improve recall with query expansion and retrieve richer context with parent-document / chunk-window expansion.  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

## Concept

Two separate problems, one day:

### Problem 1: Vocabulary mismatch (query expansion)

A user asks "fast text search engine" but your corpus uses "BM25 keyword retrieval". The terms don't overlap — BM25 scores zero, the embedding may be close but not confident. **Query expansion** generates alternative phrasings and retrieves against all of them, then merges results.

Strategies:
- **LLM-based HyDE (Hypothetical Document Embedding)**: ask the LLM to write a short document that would answer the query; embed that document instead of the query. Particularly effective when the user question is short but the answer is verbose.
- **LLM sub-question decomposition**: decompose a complex question into 2–4 simpler sub-questions; retrieve independently; merge.
- **Synonym expansion**: expand named entities or domain terms with known aliases.

### Problem 2: Chunk granularity mismatch (parent-document retrieval)

Chunks are small for precise retrieval (128–256 tokens). But 128 tokens often lacks the surrounding context the LLM needs to generate a good answer. **Parent-document retrieval** stores two representations of each document:

1. **Child chunks** — small, used for retrieval (high precision).
2. **Parent chunks or full documents** — large, returned to the LLM.

When a child chunk matches, you look up and return its parent. The LLM sees rich context; the retriever saw a concise signal.

**Chunk-window expansion** is a simpler variant: given a matching chunk at index i, also return chunks i-1 and i+1 (the surrounding window). No parent store needed — just an index lookup.

## How it works internally

### HyDE mechanics

```
query ──► LLM ──► "hypothetical document" (a plausible answer)
                       │
                  bi-encoder embed
                       │
              ANN search in vector store
                       │
             actual retrieved documents
```

The hypothesis embedding sits in the answer space rather than the question space, reducing the semantic gap to indexed documents.

### Parent-document store

```
Source document
│
├── child chunk 0  (index: doc_id=A, chunk=0)
├── child chunk 1  (index: doc_id=A, chunk=1)   ← matched by retriever
├── child chunk 2  (index: doc_id=A, chunk=2)
└── child chunk 3  (index: doc_id=A, chunk=3)

lookup(doc_id=A) ──► full parent document
```

The parent store is a simple dict keyed by `doc_id`. No vector similarity needed for the parent lookup — it is a deterministic fetch.

### Deduplication after expansion

Multiple sub-queries or window expansion can return the same chunk. Deduplicate by content hash before passing to the reranker.

## Worked example

```python
"""
day4_query_expansion.py

Two retrieval optimisations:
  1. LLM-based query expansion (sub-questions)
  2. Parent-document / window expansion

LLM calls are abstracted behind a callable so they can be mocked in tests.
"""

import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Callable

log = logging.getLogger(__name__)

LLMCallable = Callable[[str], str]   # prompt -> completion text


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_index: int
    text: str

    def content_hash(self) -> str:
        return hashlib.md5(self.text.encode()).hexdigest()


@dataclass
class Document:
    doc_id: str
    text: str
    chunks: list[Chunk] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------
_EXPANSION_PROMPT = """\
Generate {n} alternative search queries for the following question.
Each query should approach the topic from a different angle.
Return ONLY the queries, one per line, no numbering.

Question: {query}
"""


def expand_query(
    query: str,
    llm: LLMCallable,
    n_expansions: int = 2,
) -> list[str]:
    """
    Use an LLM to generate alternative query phrasings.

    Returns the original query plus n_expansions alternatives.
    """
    prompt = _EXPANSION_PROMPT.format(query=query, n=n_expansions)
    raw: str = llm(prompt).strip()
    expansions = [line.strip() for line in raw.splitlines() if line.strip()]
    all_queries = [query] + expansions[:n_expansions]
    log.info("Query expansion: %d variants generated", len(all_queries))
    for q in all_queries:
        log.debug("  variant: %s", q)
    return all_queries


def deduplicate_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Remove duplicate chunks by content hash, preserving first occurrence."""
    seen: set[str] = set()
    unique: list[Chunk] = []
    for chunk in chunks:
        h = chunk.content_hash()
        if h not in seen:
            seen.add(h)
            unique.append(chunk)
    removed = len(chunks) - len(unique)
    if removed:
        log.debug("Deduplicated %d duplicate chunks", removed)
    return unique


# ---------------------------------------------------------------------------
# Parent-document store
# ---------------------------------------------------------------------------
class ParentDocumentStore:
    """
    Stores full parent documents keyed by doc_id.
    Child chunks are retrieved from the vector store; parents fetched here.
    """

    def __init__(self) -> None:
        self._store: dict[str, Document] = {}

    def add_document(self, doc: Document) -> None:
        self._store[doc.doc_id] = doc
        log.debug("ParentDocumentStore: added doc_id=%s", doc.doc_id)

    def get_parent(self, doc_id: str) -> Document | None:
        doc = self._store.get(doc_id)
        if doc is None:
            log.warning("ParentDocumentStore: doc_id=%s not found", doc_id)
        return doc

    def fetch_parents_for_chunks(self, chunks: list[Chunk]) -> list[Document]:
        """Return unique parent documents for a list of matched child chunks."""
        seen_ids: set[str] = set()
        parents: list[Document] = []
        for chunk in chunks:
            if chunk.doc_id not in seen_ids:
                parent = self.get_parent(chunk.doc_id)
                if parent is not None:
                    parents.append(parent)
                    seen_ids.add(chunk.doc_id)
        log.info(
            "Fetched %d parent docs for %d chunks", len(parents), len(chunks)
        )
        return parents


# ---------------------------------------------------------------------------
# Window expansion
# ---------------------------------------------------------------------------
def expand_window(
    matched_chunk: Chunk,
    all_chunks: list[Chunk],
    window_size: int = 1,
) -> list[Chunk]:
    """
    Return the matched chunk plus up to window_size neighbours on each side,
    filtering to the same doc_id.

    Parameters
    ----------
    matched_chunk: the chunk that matched the query.
    all_chunks: all chunks in document order.
    window_size: how many chunks to include on each side.
    """
    same_doc = [c for c in all_chunks if c.doc_id == matched_chunk.doc_id]
    # Find position of matched chunk
    try:
        pos = next(i for i, c in enumerate(same_doc) if c.chunk_index == matched_chunk.chunk_index)
    except StopIteration:
        log.warning("Matched chunk not found in same_doc list — returning as-is")
        return [matched_chunk]

    start = max(0, pos - window_size)
    end = min(len(same_doc), pos + window_size + 1)
    window = same_doc[start:end]
    log.debug(
        "Window expansion: chunk_index=%d → indices %d..%d",
        matched_chunk.chunk_index,
        start,
        end - 1,
    )
    return window


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def _mock_llm(prompt: str) -> str:
    """Fake LLM for demo purposes."""
    return (
        "techniques to improve document search relevance\n"
        "how retrieval systems handle query vocabulary mismatch"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Query expansion demo
    query = "best practices for RAG retrieval quality"
    variants = expand_query(query, _mock_llm, n_expansions=2)
    log.info("Expanded queries: %s", variants)

    # Parent-document store demo
    doc = Document(
        doc_id="doc-001",
        text="Full parent document text with rich context about RAG systems.",
        chunks=[
            Chunk("doc-001", 0, "Chunking affects retrieval recall."),
            Chunk("doc-001", 1, "Reranking improves precision over recall stage."),
            Chunk("doc-001", 2, "MMR selects diverse relevant documents."),
        ],
    )
    store = ParentDocumentStore()
    store.add_document(doc)

    matched = [doc.chunks[1]]  # pretend retriever matched chunk 1
    parents = store.fetch_parents_for_chunks(matched)
    log.info("Parent text: %s", parents[0].text[:60])

    # Window expansion demo
    window = expand_window(doc.chunks[1], doc.chunks, window_size=1)
    log.info("Window chunks: %s", [c.chunk_index for c in window])


if __name__ == "__main__":
    main()
```

### Tests

```python
"""
tests/unit/test_query_expansion.py
"""

from unittest.mock import MagicMock

import pytest

from day4_query_expansion import (
    Chunk,
    Document,
    ParentDocumentStore,
    deduplicate_chunks,
    expand_query,
    expand_window,
)


class TestExpandQuery:
    def test_original_query_always_included(self):
        llm = MagicMock(return_value="alt1\nalt2")
        result = expand_query("my query", llm, n_expansions=2)
        assert result[0] == "my query"

    def test_correct_number_of_variants(self):
        llm = MagicMock(return_value="alt1\nalt2\nalt3")
        result = expand_query("q", llm, n_expansions=2)
        assert len(result) == 3  # original + 2

    def test_empty_llm_response_returns_only_original(self):
        llm = MagicMock(return_value="   ")
        result = expand_query("q", llm, n_expansions=2)
        assert result == ["q"]


class TestDeduplicateChunks:
    def test_removes_exact_duplicates(self):
        c = Chunk("d1", 0, "same text")
        chunks = [c, Chunk("d1", 1, "same text"), Chunk("d2", 0, "other")]
        result = deduplicate_chunks(chunks)
        assert len(result) == 2

    def test_preserves_order(self):
        a = Chunk("d", 0, "alpha")
        b = Chunk("d", 1, "beta")
        result = deduplicate_chunks([a, b, a])
        assert [r.text for r in result] == ["alpha", "beta"]


class TestParentDocumentStore:
    def _make_store(self):
        store = ParentDocumentStore()
        doc = Document("doc-1", "parent text", [Chunk("doc-1", 0, "child")])
        store.add_document(doc)
        return store, doc

    def test_get_parent_returns_correct_doc(self):
        store, doc = self._make_store()
        assert store.get_parent("doc-1") is doc

    def test_missing_doc_returns_none(self):
        store, _ = self._make_store()
        assert store.get_parent("missing") is None

    def test_fetch_parents_deduplicates(self):
        store, doc = self._make_store()
        chunks = [doc.chunks[0], doc.chunks[0]]
        parents = store.fetch_parents_for_chunks(chunks)
        assert len(parents) == 1


class TestWindowExpansion:
    def _chunks(self):
        return [Chunk("d", i, f"chunk {i}") for i in range(5)]

    def test_middle_chunk_includes_neighbours(self):
        chunks = self._chunks()
        result = expand_window(chunks[2], chunks, window_size=1)
        indices = [c.chunk_index for c in result]
        assert indices == [1, 2, 3]

    def test_start_chunk_clamped(self):
        chunks = self._chunks()
        result = expand_window(chunks[0], chunks, window_size=1)
        indices = [c.chunk_index for c in result]
        assert indices == [0, 1]

    def test_end_chunk_clamped(self):
        chunks = self._chunks()
        result = expand_window(chunks[4], chunks, window_size=1)
        indices = [c.chunk_index for c in result]
        assert indices == [3, 4]

    def test_only_same_doc_included(self):
        doc1 = [Chunk("A", i, f"A{i}") for i in range(3)]
        doc2 = [Chunk("B", i, f"B{i}") for i in range(3)]
        all_chunks = doc1 + doc2
        result = expand_window(doc1[1], all_chunks, window_size=2)
        assert all(c.doc_id == "A" for c in result)
```

## Exercises

1. Implement HyDE: write a `hyde_expand` function that prompts an LLM to produce a "hypothetical answer document" and returns its embedding (use a mock LLM + mock encoder). Write a test.
2. Add a `max_chunks_per_doc: int` parameter to `fetch_parents_for_chunks` — when the same doc_id appears more than this many times in the matched list, skip fetching the parent (signal of over-indexing). Write tests for the boundary conditions.
3. Wire query expansion into a retrieval loop: given a `retrieve: Callable[[str], list[Chunk]]` function, call it for each expanded query, deduplicate the results, and return a flat list. Test with mocked retrieve calls.

(Solutions: ../solutions/week-05/day-4.md)

## Self-check

- [ ] I can explain HyDE and why it improves recall.
- [ ] I can implement parent-document retrieval without reference.
- [ ] I can implement chunk-window expansion with correct boundary clamping.
- [ ] I can write deduplication that is stable (preserves insertion order).
- [ ] I understand the difference between the child index (for retrieval) and parent store (for LLM context).

## Supplementary links

- [HyDE paper: Precise Zero-Shot Dense Retrieval](https://arxiv.org/abs/2212.10496)
- [LangChain ParentDocumentRetriever](https://python.langchain.com/docs/how_to/parent_document_retriever/)
- [RAG survey (Gao et al. 2023)](https://arxiv.org/abs/2312.10997)
