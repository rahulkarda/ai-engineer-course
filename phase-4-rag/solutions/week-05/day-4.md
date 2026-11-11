# Solutions — Week 05 · Day 4

## Exercise 1: HyDE expand

```python
"""HyDE (Hypothetical Document Embedding) query expansion."""

import logging
import numpy as np
from typing import Callable

log = logging.getLogger(__name__)

_HYDE_PROMPT = """\
Write a short paragraph (3-5 sentences) that would directly answer the following question.
Write as if it is an excerpt from a technical document.

Question: {query}
"""


def hyde_expand(
    query: str,
    llm: Callable[[str], str],
    encode: Callable[[list[str]], np.ndarray],
) -> np.ndarray:
    """
    Generate a hypothetical answer document and return its embedding.

    Parameters
    ----------
    query: the original query string.
    llm: callable that takes a prompt and returns a text completion.
    encode: callable that takes a list of strings and returns unit-normalised embeddings.

    Returns
    -------
    Unit-normalised embedding of the hypothetical document, shape (D,).
    """
    prompt = _HYDE_PROMPT.format(query=query)
    hypothesis = llm(prompt).strip()
    log.debug("HyDE hypothesis: %s", hypothesis[:100])
    emb: np.ndarray = encode([hypothesis])[0]
    return emb
```

```python
# tests/test_hyde.py
from unittest.mock import MagicMock
import numpy as np
from day4_hyde import hyde_expand

def test_hyde_returns_embedding_shape():
    llm = MagicMock(return_value="Cross-encoders improve precision by attending jointly.")
    def mock_encode(texts):
        return np.array([[0.6, 0.8]])  # 1 text → shape (1, 2)
    emb = hyde_expand("q", llm, mock_encode)
    assert emb.shape == (2,)

def test_hyde_calls_llm_once():
    llm = MagicMock(return_value="answer text")
    encode = MagicMock(return_value=np.array([[1.0, 0.0]]))
    hyde_expand("what is X?", llm, encode)
    llm.assert_called_once()
```

## Exercise 2: max_chunks_per_doc

```python
def fetch_parents_for_chunks(
    self,
    chunks: list[Chunk],
    max_chunks_per_doc: int | None = None,
) -> list[Document]:
    doc_hit_count: dict[str, int] = {}
    seen_ids: set[str] = set()
    parents: list[Document] = []

    for chunk in chunks:
        doc_hit_count[chunk.doc_id] = doc_hit_count.get(chunk.doc_id, 0) + 1

        if max_chunks_per_doc is not None and doc_hit_count[chunk.doc_id] > max_chunks_per_doc:
            log.debug(
                "Skipping doc_id=%s: hit count %d > max_chunks_per_doc %d",
                chunk.doc_id,
                doc_hit_count[chunk.doc_id],
                max_chunks_per_doc,
            )
            continue

        if chunk.doc_id not in seen_ids:
            parent = self.get_parent(chunk.doc_id)
            if parent is not None:
                parents.append(parent)
                seen_ids.add(chunk.doc_id)

    return parents
```

```python
# Tests
def test_max_chunks_per_doc_filters_over_indexed():
    store = ParentDocumentStore()
    doc = Document("d1", "parent text", [Chunk("d1", i, f"chunk {i}") for i in range(5)])
    store.add_document(doc)
    # 3 hits for same doc → exceeds max_chunks_per_doc=2 on 3rd hit
    chunks = doc.chunks[:3]
    parents = store.fetch_parents_for_chunks(chunks, max_chunks_per_doc=2)
    # doc is first added when hit count == 1 (<=2), then at count 2 (<=2); count 3 exceeds
    # But since doc was already added at count 1, we just get 1 parent total
    assert len(parents) == 1

def test_max_chunks_per_doc_none_allows_all():
    store = ParentDocumentStore()
    doc = Document("d1", "text", [Chunk("d1", i, f"c{i}") for i in range(5)])
    store.add_document(doc)
    parents = store.fetch_parents_for_chunks(doc.chunks, max_chunks_per_doc=None)
    assert len(parents) == 1  # still 1 because dedup by doc_id
```

## Exercise 3: Retrieve loop with query expansion

```python
from typing import Callable
from day4_query_expansion import Chunk, deduplicate_chunks, expand_query

def retrieve_with_expansion(
    query: str,
    retrieve: Callable[[str], list[Chunk]],
    llm: Callable[[str], str],
    n_expansions: int = 2,
) -> list[Chunk]:
    """Call retrieve for each expanded query and return deduplicated results."""
    queries = expand_query(query, llm, n_expansions=n_expansions)
    all_chunks: list[Chunk] = []
    for q in queries:
        all_chunks.extend(retrieve(q))
    return deduplicate_chunks(all_chunks)
```

```python
# Test
from unittest.mock import MagicMock

def test_retrieve_with_expansion_deduplicates():
    llm = MagicMock(return_value="variant1\nvariant2")
    chunk = Chunk("d", 0, "same text")
    retrieve = MagicMock(return_value=[chunk])
    result = retrieve_with_expansion("query", retrieve, llm, n_expansions=2)
    # retrieve called 3 times (original + 2 expansions), all return same chunk
    assert retrieve.call_count == 3
    assert len(result) == 1  # deduplicated
```
