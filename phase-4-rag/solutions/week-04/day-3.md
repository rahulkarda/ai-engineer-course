# Solutions — Week 04 · Day 3

## Exercise 1 — Custom tokeniser (suffix stripping)

```python
SUFFIXES = ["ing", "tion", "ed", "er", "ly", "s"]

def stem(word: str) -> str:
    for suffix in sorted(SUFFIXES, key=len, reverse=True):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word

def _tokenize_stemmed(self, text: str) -> list[str]:
    tokens = re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
    return [stem(t) for t in tokens]
```

Retrieval quality on "indexing vectors": stemming maps "indexing" → "index" and "vectors" → "vector", which may match more documents. For exact-match use cases (CVE IDs, version numbers), stemming hurts precision.

## Exercise 2 — Multi-field BM25

```python
@dataclass
class MultiFieldBM25:
    title_weight: float = 2.0
    body_weight: float = 1.0
    _title_retriever: BM25Retriever = field(default_factory=BM25Retriever, init=False)
    _body_retriever: BM25Retriever = field(default_factory=BM25Retriever, init=False)
    _docs: list[dict] = field(default_factory=list, init=False)

    def add_documents(self, docs: list[dict]) -> None:
        """Each doc: {"title": str, "body": str}"""
        self._docs = docs
        self._title_retriever.add_document.__func__  # ensure method exists
        for d in docs:
            self._title_retriever.add_document(d["title"])
            self._body_retriever.add_document(d["body"])

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        t_results = {did: s * self.title_weight for did, s in self._title_retriever.retrieve(query, top_k=len(self._docs))}
        b_results = {did: s * self.body_weight for did, s in self._body_retriever.retrieve(query, top_k=len(self._docs))}
        all_ids = set(t_results) | set(b_results)
        combined = {did: t_results.get(did, 0) + b_results.get(did, 0) for did in all_ids}
        return sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
```

## Exercise 3 — Inverted index stats

```python
def vocabulary_stats(self) -> dict:
    posting_lens = [len(v) for v in self._inverted.values()]
    sorted_by_df = sorted(self._inverted.items(), key=lambda x: len(x[1]), reverse=True)
    return {
        "vocab_size": len(self._inverted),
        "avg_posting_len": sum(posting_lens) / len(posting_lens) if posting_lens else 0,
        "top_10_terms": [term for term, _ in sorted_by_df[:10]],
    }
```

## Exercise 4 — Stress test

```python
import random, string, time

VOCAB = ["".join(random.choices(string.ascii_lowercase, k=random.randint(3,8))) for _ in range(500)]

def random_doc() -> str:
    return " ".join(random.choices(VOCAB, k=random.randint(5, 15)))

docs = [random_doc() for _ in range(10_000)]

# Per-document adds
r1 = BM25Retriever()
t0 = time.perf_counter()
for d in docs:
    r1.add_document(d)
log.info("Per-doc: %.3fs", time.perf_counter() - t0)

# Batch
r2 = BM25Retriever()
t0 = time.perf_counter()
r2.add_documents(docs)
log.info("Batch: %.3fs", time.perf_counter() - t0)
```

The bottleneck is Python object creation overhead per call — batch is faster mainly by avoiding repeated method call overhead. The IDF recomputation (currently eager) also fires per-document; batching it would help further.

## Exercise 5 — Test suite

```python
# tests/test_bm25.py
import json, os, tempfile
import pytest
from day3_bm25_from_scratch import BM25Retriever, BM25Config, CORPUS


@pytest.fixture
def retriever() -> BM25Retriever:
    r = BM25Retriever()
    r.add_documents(CORPUS)
    return r


def test_document_count(retriever):
    assert len(retriever) == len(CORPUS)


def test_retrieve_top1_relevant(retriever):
    results = retriever.retrieve("probabilistic ranking retrieval", top_k=1)
    assert len(results) == 1
    doc_id, score, text = results[0]
    assert "BM25" in text or "probabilistic" in text.lower() or "retrieval" in text.lower()


def test_retrieve_top_k_length(retriever):
    results = retriever.retrieve("Python web framework", top_k=3)
    assert len(results) <= 3


def test_empty_query_returns_empty(retriever):
    results = retriever.retrieve("zzzzquux_nonexistent_term")
    assert results == []


def test_save_load_roundtrip(retriever):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    try:
        retriever.save(tmp)
        r2 = BM25Retriever.load(tmp)
        q = "vector similarity search"
        assert retriever.retrieve(q, top_k=1)[0][0] == r2.retrieve(q, top_k=1)[0][0]
    finally:
        os.unlink(tmp)


def test_score_positive_for_exact_term(retriever):
    results = retriever.retrieve("BM25", top_k=5)
    assert any(score > 0 for _, score, _ in results)
```
