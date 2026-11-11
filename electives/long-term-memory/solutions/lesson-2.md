# Solutions: Lesson 2 — Memory Architectures: Vector, Graph, KV

---

## Exercise 1 — `min_score` parameter

```python
def query(
    self,
    query_vector: list[float],
    top_k: int = 5,
    metadata_filter: dict[str, Any] | None = None,
    min_score: float = 0.0,
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
    scored = [(r, s) for r, s in scored if s >= min_score]  # filter by min_score
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
```

Tests:

```python
import pytest
from lesson_2_architectures import InMemoryVectorStore, mock_embed


def _store_with_records() -> tuple[InMemoryVectorStore, list]:
    vs = InMemoryVectorStore()
    texts = ["report format PDF", "dark mode setting", "meeting Tuesday"]
    ids = [vs.upsert(t, mock_embed(t)) for t in texts]
    return vs, ids


def test_min_score_all_pass():
    vs, _ = _store_with_records()
    q = mock_embed("report format PDF")
    results = vs.query(q, top_k=3, min_score=0.0)
    assert len(results) == 3


def test_min_score_filters_low():
    vs, _ = _store_with_records()
    q = mock_embed("report format PDF")
    # Use a very high threshold to filter most results
    results = vs.query(q, top_k=3, min_score=0.999)
    # Only the identical record should pass (score ≈ 1.0)
    assert len(results) <= 1
```

---

## Exercise 2 — BFS `get_path`

```python
from collections import deque

def get_path(self, start: str, end: str) -> list[str] | None:
    if start not in self._nodes or end not in self._nodes:
        return None
    if start == end:
        return [start]
    
    visited = {start}
    queue: deque[list[str]] = deque([[start]])
    
    while queue:
        path = queue.popleft()
        current = path[-1]
        for edge in self._edges:
            if edge.source == current and edge.target not in visited:
                new_path = path + [edge.target]
                if edge.target == end:
                    return new_path
                visited.add(edge.target)
                queue.append(new_path)
    
    return None
```

Test:

```python
def test_get_path_found():
    gs = InMemoryGraphStore()
    gs.add_node("a", "X")
    gs.add_node("b", "X")
    gs.add_node("c", "X")
    gs.add_edge("a", "b", "LINKS")
    gs.add_edge("b", "c", "LINKS")
    assert gs.get_path("a", "c") == ["a", "b", "c"]


def test_get_path_not_found():
    gs = InMemoryGraphStore()
    gs.add_node("a", "X")
    gs.add_node("b", "X")
    # no edge between a and b
    assert gs.get_path("a", "b") is None
```

---

## Exercise 3 — `get_all_with_prefix`

```python
def get_all_with_prefix(self, prefix: str) -> dict[str, Any]:
    return {k: v for k, v in self._store.items() if k.startswith(prefix)}
```

Test:

```python
def test_get_all_with_prefix():
    kv = InMemoryKVStore()
    kv.set("proc:summarise", "TL;DR first")
    kv.set("proc:greet", "Reference last topic")
    kv.set("proc:format", "Use markdown")
    kv.set("session:user_42", "active")

    procs = kv.get_all_with_prefix("proc:")
    assert len(procs) == 3
    assert "proc:summarise" in procs
    assert "session:user_42" not in procs
```

---

## Exercise 4 — Why the mock works for unit tests but not integration tests

`mock_embed` produces a **deterministic but semantically meaningless** embedding based
on text length, not semantic content. Two unrelated texts of the same length get
identical vectors, and two semantically identical texts with different lengths get
different vectors. This is fine for unit tests because:

- Cosine similarity calculations are exercised
- Store/retrieve round-trips are tested
- No external calls are made

A better integration-test mock would use a **fixed lookup table** of pre-computed
real embeddings for specific test strings, ensuring semantic relationships hold
(e.g. "PDF report" and "PDF summary" genuinely cluster together). These can be
generated once with the real API and committed to the test fixtures file.

```python
# Better integration mock
EMBED_FIXTURES: dict[str, list[float]] = {
    "PDF report": [0.12, -0.34, ...],   # pre-computed real embedding
    "PDF summary": [0.11, -0.33, ...],  # similar to above
    "dark mode": [-0.45, 0.67, ...],    # different cluster
}

class FixtureEmbedder:
    def embed(self, text: str) -> list[float]:
        if text in EMBED_FIXTURES:
            return EMBED_FIXTURES[text]
        # fallback for unknown texts
        return mock_embed(text)
```
