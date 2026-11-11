# Solutions — Week 04 · Day 1

## Exercise 1 — Swap encoder

```python
retriever = SemanticRetriever(model_name="BAAI/bge-small-en-v1.5")
retriever.index_documents(CORPUS)
```

Expected: rankings are similar but not identical. BGE models are trained with asymmetric retrieval (query prefix `Represent this sentence: `), so for best results:

```python
from sentence_transformers import SentenceTransformer

class BGESemanticRetriever(SemanticRetriever):
    QUERY_PREFIX = "Represent this sentence: "

    def retrieve(self, query: str, top_k: int = 3):
        prefixed = self.QUERY_PREFIX + query
        return super().retrieve(prefixed, top_k)
```

Rankings may change, especially for the BM25/keyword query — BGE has stronger semantic discrimination at sentence level.

## Exercise 2 — Add document at runtime

```python
def add_document(self, text: str) -> None:
    """Encode and append one document without rebuilding the index."""
    self._docs.append(text)
    vec: np.ndarray = self._model.encode([text], normalize_embeddings=True)
    self._index.add(vec)
```

Test:

```python
def test_add_document_grows_count():
    r = SemanticRetriever()
    r.index_documents(["doc one", "doc two"])
    assert r._index.ntotal == 2
    r.add_document("doc three")
    assert r._index.ntotal == 3
    assert r._docs[-1] == "doc three"
```

## Exercise 3 — Cosine vs. dot-product

Without normalisation + `IndexFlatL2`:
- Results degrade for queries on short documents because L2 distance penalises magnitude differences, not just direction.
- Use `faiss.IndexFlatIP` with unit-normalised vectors (cosine) for text embeddings — magnitude is meaningless in text space.

## Exercise 4 — Batch retrieval

```python
def retrieve_batch(self, queries: list[str], top_k: int = 3) -> list[list[tuple[str, float]]]:
    q_vecs: np.ndarray = self._model.encode(
        queries, normalize_embeddings=True, show_progress_bar=False
    )
    scores_batch, indices_batch = self._index.search(q_vecs, top_k)
    return [
        [(self._docs[idx], float(scores_batch[qi][rank])) for rank, idx in enumerate(indices_batch[qi])]
        for qi in range(len(queries))
    ]
```

## Exercise 5 — Persist index

```python
import json
import faiss

def save(self, path: str) -> None:
    p = Path(path)
    faiss.write_index(self._index, str(p.with_suffix(".faiss")))
    p.with_suffix(".docs.json").write_text(json.dumps(self._docs))

@classmethod
def load(cls, path: str, model_name: str = "all-MiniLM-L6-v2") -> "SemanticRetriever":
    p = Path(path)
    obj = cls(model_name=model_name)
    obj._index = faiss.read_index(str(p.with_suffix(".faiss")))
    obj._docs = json.loads(p.with_suffix(".docs.json").read_text())
    return obj
```

Test:

```python
def test_save_load_top1(tmp_path):
    r = SemanticRetriever()
    r.index_documents(CORPUS)
    save_path = tmp_path / "index"
    r.save(str(save_path))
    r2 = SemanticRetriever.load(str(save_path))
    query = "probabilistic ranking algorithm"
    top1_orig = r.retrieve(query, top_k=1)[0][0]
    top1_loaded = r2.retrieve(query, top_k=1)[0][0]
    assert top1_orig == top1_loaded
```
