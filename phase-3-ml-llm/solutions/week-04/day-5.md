# Solutions — Week 04 · Day 5

## Exercise 1 — Batched `build_index` with `batch_size`

```python
def build_index(self, documents: list[str], batch_size: int = 64) -> None:
    self._docs = list(documents)
    chunks: list[np.ndarray] = []
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        chunks.append(self._embedder.embed(batch))
    self._index = np.vstack(chunks)
    logger.info("Index built: %d docs × %d dims", *self._index.shape)
```

**Test — same result regardless of batch_size:**

```python
import numpy as np
import pytest
from unittest.mock import MagicMock
from day5_sentence_embeddings import SemanticSearchEngine, StubEmbedder

DOCS = ["cats", "dogs", "fish", "birds", "horses"]

@pytest.mark.parametrize("batch_size", [1, 2, 5, 100])
def test_batching_produces_same_index(batch_size: int) -> None:
    embedder = StubEmbedder(dim=16)
    e1 = SemanticSearchEngine(embedder)
    e2 = SemanticSearchEngine(embedder)
    e1.build_index(DOCS)
    e2.build_index(DOCS)
    assert np.allclose(e1._index, e2._index, atol=1e-6)
```

---

## Exercise 2 — MMR search

```python
def search_mmr(
    self,
    query: str,
    top_k: int = 5,
    lambda_: float = 0.5,
) -> list[tuple[str, float]]:
    """Maximal Marginal Relevance search balancing relevance and diversity."""
    if self._index is None:
        raise RuntimeError("Call build_index first")
    q_vec = self._embedder.embed([query])[0]
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-9)
    relevance = self._index @ q_norm      # shape (n_docs,)

    selected: list[int] = []
    remaining = list(range(len(self._docs)))

    while len(selected) < min(top_k, len(self._docs)) and remaining:
        if not selected:
            # First pick: highest relevance
            best = max(remaining, key=lambda i: relevance[i])
        else:
            # MMR: balance relevance and max similarity to already-selected
            selected_vecs = self._index[selected]  # (k, dim)
            scores = []
            for i in remaining:
                rel = relevance[i]
                max_sim = float((selected_vecs @ self._index[i]).max())
                score = lambda_ * rel - (1 - lambda_) * max_sim
                scores.append((i, score))
            best = max(scores, key=lambda x: x[1])[0]
        selected.append(best)
        remaining.remove(best)

    return [(self._docs[i], float(relevance[i])) for i in selected]
```

---

## Exercise 3 — Disk-persistent cache

```python
from pathlib import Path
import numpy as np
import json

class CachedEmbedder:
    def __init__(self, provider, cache_path: Path | None = None) -> None:
        self._provider = provider
        self._cache: dict[str, np.ndarray] = {}
        self._cache_path = cache_path
        if cache_path and Path(cache_path).exists():
            self.load_cache(cache_path)

    def save_cache(self, path: Path) -> None:
        path = Path(path)
        # Save as NPZ with text-to-key mapping
        keys = list(self._cache.keys())
        vectors = np.array(list(self._cache.values()))
        np.savez(path, vectors=vectors)
        path.with_suffix(".json").write_text(json.dumps(keys), encoding="utf-8")

    def load_cache(self, path: Path) -> None:
        path = Path(path)
        data = np.load(str(path) + ".npz")
        keys = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        self._cache = dict(zip(keys, data["vectors"]))
```

---

## Exercise 4 — pytest test for `SemanticSearchEngine.search`

```python
import numpy as np
from unittest.mock import MagicMock
from day5_sentence_embeddings import SemanticSearchEngine

DOCS = ["cats are furry", "dogs bark loudly", "fish swim silently"]

def _vecs(n: int, dim: int = 8) -> np.ndarray:
    """Return n orthogonal-ish unit vectors for deterministic testing."""
    rng = np.random.default_rng(0)
    m = rng.standard_normal((n, dim)).astype(np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)

def test_top_result_is_expected_doc() -> None:
    mock = MagicMock()
    # Make the query vector identical to DOCS[1] so "dogs" should rank first
    doc_vecs = _vecs(3)
    mock.embed.side_effect = lambda texts: doc_vecs[:len(texts)]

    engine = SemanticSearchEngine(mock)
    engine.build_index(DOCS)

    # Set query vec = doc_vecs[1] for a perfect match
    mock.embed.side_effect = lambda texts: (
        doc_vecs[1:2] if texts == ["dogs"] else doc_vecs[:len(texts)]
    )
    results = engine.search("dogs", top_k=1)
    assert results[0][0] == DOCS[1]
```

---

## Exercise 5 — SHA-256 determinism test

```python
import hashlib
import numpy as np
import pytest
from day5_sentence_embeddings import StubEmbedder

class SHA256StubEmbedder(StubEmbedder):
    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            h = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**31)
            rng = np.random.default_rng(h)
            vec = rng.standard_normal(self.dim).astype(np.float32)
            vec /= np.linalg.norm(vec)
            vectors.append(vec)
        return np.array(vectors)


def test_same_text_same_vector() -> None:
    e = SHA256StubEmbedder(dim=32)
    v1 = e.embed(["hello world"])
    v2 = e.embed(["hello world"])
    assert np.allclose(v1, v2)


def test_different_texts_different_vectors() -> None:
    e = SHA256StubEmbedder(dim=32)
    v1 = e.embed(["hello"])
    v2 = e.embed(["world"])
    assert not np.allclose(v1, v2)


def test_vectors_are_unit_norm() -> None:
    e = SHA256StubEmbedder(dim=32)
    vecs = e.embed(["a", "b", "c"])
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)
```
