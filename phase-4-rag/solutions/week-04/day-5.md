# Solutions — Week 04 · Day 5

## Exercise 1 — FilterBuilder

```python
class FilterBuilder:
    def __init__(self) -> None:
        self._filters: list[Filter] = []

    def eq(self, field: str, value: Any) -> "FilterBuilder":
        self._filters.append(lambda doc, f=field, v=value: doc.metadata.get(f) == v)
        return self

    def gte(self, field: str, value: Any) -> "FilterBuilder":
        self._filters.append(
            lambda doc, f=field, v=value: doc.metadata.get(f) is not None and doc.metadata[f] >= v
        )
        return self

    def lte(self, field: str, value: Any) -> "FilterBuilder":
        self._filters.append(
            lambda doc, f=field, v=value: doc.metadata.get(f) is not None and doc.metadata[f] <= v
        )
        return self

    def build(self) -> Filter:
        filters = list(self._filters)
        return lambda doc: all(f(doc) for f in filters)

# Usage
f = FilterBuilder().eq("category", "agents").gte("year", 2024).build()
results = retriever.retrieve("workflow orchestration", top_k=5, filter_fn=f)
```

Note the default-argument capture (`f=field, v=value`) in the lambdas — this avoids the classic Python loop-closure bug where all lambdas capture the last loop value.

## Exercise 2 — HyDE stub

```python
from typing import Callable

def hyde_query(question: str, generate_fn: Callable[[str], str]) -> str:
    """Generate a hypothetical answer and return it for embedding."""
    prompt = f"Write a short paragraph answering: {question}"
    return generate_fn(prompt)
```

Test:

```python
def test_hyde_query_calls_generate():
    mock_gen = MagicMock(return_value="Transformers use self-attention to encode sequences.")
    result = hyde_query("How do transformers work?", mock_gen)
    mock_gen.assert_called_once()
    assert "attention" in result.lower() or len(result) > 0
```

HyDE helps when the question and relevant documents use different vocabulary (e.g. question: "What causes heart failure?" → HyDE answer uses medical terminology matching document language). Risk: if the generated answer is factually wrong, it can steer retrieval to irrelevant documents.

## Exercise 3 — Multi-query with RRF

```python
from collections import defaultdict

def merge_sub_query_results_rrf(
    sub_results: list[list[tuple[Document, float]]],
    k: int = 60,
) -> list[tuple[int, float]]:
    rrf: dict[int, float] = defaultdict(float)
    for ranked in sub_results:
        for rank, (doc, _) in enumerate(ranked, start=1):
            rrf[doc.doc_id] += 1.0 / (k + rank)
    return sorted(rrf.items(), key=lambda x: x[1], reverse=True)

sub_results = [retriever.retrieve(sq, top_k=5) for sq in sub_queries]
merged_rrf = merge_sub_query_results_rrf(sub_results)
```

RRF tends to surface documents that appear across multiple sub-queries — more stable than max-score which can be dominated by one high-scoring sub-query.

## Exercise 4 — Dynamic top-k

```python
def auto_top_k(score_list: list[float], threshold: float = 0.5) -> int:
    """Return smallest k where score[k] / score[0] < threshold."""
    if not score_list or score_list[0] == 0:
        return 1
    for k, s in enumerate(score_list, start=1):
        if s / score_list[0] < threshold:
            return k
    return len(score_list)

# Usage
results = retriever.retrieve("semantic search embeddings", top_k=20)
scores = [score for _, score in results]
k = auto_top_k(scores, threshold=0.5)
log.info("Auto top-k: %d", k)
```

## Exercise 5 — Filter + hybrid

```python
# In HybridRetriever.retrieve:
def retrieve(self, query: str, top_k: int = 5, filter_fn=None):
    bm25_results = self.bm25.retrieve(query, top_k=self.candidate_k, filter_fn=filter_fn)
    dense_results = self.semantic.retrieve(query, top_k=self.candidate_k, filter_fn=filter_fn)
    ...
```

Test:

```python
def test_hybrid_filter_respected():
    # Build hybrid with 5 docs, 2 categories
    docs = [
        Document(doc_id=i, text=f"doc {i}", metadata={"cat": "a" if i < 3 else "b"})
        for i in range(5)
    ]
    hybrid = HybridRetriever(bm25=BM25Retriever(), semantic=mock_semantic())
    hybrid.add_documents(docs)
    f = lambda doc: doc.metadata.get("cat") == "a"
    results = hybrid.retrieve("doc", top_k=10, filter_fn=f)
    assert all(r[0] < 3 for r, _, _ in results)  # only doc_ids 0,1,2
```
