# Week 03 Assessment — Vector Databases & Indexing

## Conceptual (interview-style)

1. Explain the difference between exact nearest-neighbor search and approximate nearest-neighbor (ANN) search. What is the key trade-off, and how is it quantified?

2. Describe the IVF index build process in four steps. What is the role of the `nlist` and `nprobe` parameters, and how do they affect the recall–latency trade-off?

3. Walk through HNSW query-phase operation from top layer to layer 0. What does `efSearch` control?

4. A colleague proposes using a KD-tree for 1536-dimensional text embeddings. Explain why this is a poor choice.

5. You have a 10M-vector FAISS index using `IndexFlatIP`. A product manager asks you to cut memory usage by 16×. What technique do you apply, and what metric do you expect to degrade?

6. Describe two scenarios where you would choose pgvector over a dedicated vector database like Qdrant.

7. What is the difference between pre-filtering and post-filtering in vector search? Give one advantage and one disadvantage of each.

8. Your FAISS `IndexIVFFlat.search()` returns arrays filled with `-1`. What are the two most likely causes?

## Coding challenge

### Spec

Implement a `SmartVectorStore` class that:

1. Wraps your `VectorStore` ABC from the week project (use the in-memory backend)
2. Adds a `smart_query` method with this signature:

```python
def smart_query(
    self,
    vector: np.ndarray,
    k: int = 5,
    filter: dict | None = None,
    fallback_without_filter: bool = True,
) -> list[SearchResult]:
```

Behaviour:
- Run a filtered query first (if filter provided)
- If results < k AND `fallback_without_filter=True`, pad with un-filtered results (no duplicates)
- Always return at most k results, sorted by score descending

3. Write at least 3 pytest tests for `smart_query`:
   - Returns exactly k results when corpus is large enough and no filter
   - Pads to k when filter is restrictive
   - Does NOT pad when `fallback_without_filter=False`

### Acceptance criteria

- Code is typed, uses `logging` not `print`
- Tests pass with `pytest -v`
- No secrets or hardcoded credentials
- `smart_query` handles edge case: corpus smaller than k

## Rubric (pass ≥ 75 %)

| Dimension | Weight | Indicators |
|-----------|--------|-----------|
| Independent implementation | 40 % | `SmartVectorStore` works correctly without reference to solutions; logic for padding is correct and deduplicates |
| Conceptual understanding | 25 % | Conceptual answers are precise; uses correct terminology (recall\@k, nlist, nprobe, efSearch, PQ) |
| Debugging & problem-solving | 20 % | Correctly diagnoses both IVF -1 scenario and Chroma zero-result scenario; explains root causes |
| Code quality & testing | 15 % | Type hints, logging, pytest markers, parametrize used; no bare excepts |

Answer key: ../solutions/week-03/assessment.md
