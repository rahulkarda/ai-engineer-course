# Solutions — Week 03 · Day 7

Day 7 is a project work and review day. This file contains self-check answers for the review questions posed in the lesson.

## Conceptual self-check answers

### Why KD-trees degrade in high dimensions

KD-trees partition space with axis-aligned hyperplanes. In d dimensions, the number of hyperplane cuts needed to isolate a region grows exponentially. The fraction of the tree that must be visited to find the nearest neighbor approaches 1.0 as d increases — the tree degenerates to a linear scan. This is sometimes called the "curse of dimensionality." Above ~20 dims, KD-tree search is slower than brute force due to overhead.

### What recall@10 = 0.95 means

Of the 10 true nearest neighbors (as returned by exact search), 9.5 on average are found by the ANN search. In practice, for RAG retrieval where you pass the top-k chunks to an LLM, missing 1 in 10 candidates rarely affects final answer quality — the LLM still receives 9 highly relevant contexts.

### IVF vs HNSW selection

**Choose IVF when**:
- n > 5M and RAM is the bottleneck (IVF + PQ compresses aggressively)
- Vectors are batch-loaded (no streaming inserts needed)
- You need predictable, tunable latency/recall via nprobe

**Choose HNSW when**:
- Streaming inserts are required
- You want the highest recall at moderate M values
- RAM is not constrained

## Coding quick-reference (from memory)

```python
# FAISS HNSWFlat: build, add, query (5 lines)
import faiss, numpy as np
index = faiss.IndexHNSWFlat(128, 32, faiss.METRIC_INNER_PRODUCT)
index.add(corpus)                      # corpus: (n, 128) float32, normalised
D, I = index.search(query, k=5)        # query: (1, 128), returns (1, 5) each

# Chroma compound filter
{"$and": [{"category": {"$eq": "legal"}}, {"year": {"$gte": 2023}}]}
```

## Debugging answers

### FAISS IVFFlat returns all -1 IDs

Two most likely causes:
1. **Index was never trained**: `index.is_trained` is False. Call `index.train(training_vectors)` before `index.add()`.
2. **No vectors added**: `index.ntotal == 0`. Add vectors before searching.

### Chroma zero results with filter

Two most likely causes:
1. **Metadata key mismatch**: the field name in the `where` clause doesn't exactly match the key stored in metadata (case-sensitive).
2. **Wrong operator format**: passing `{"category": "legal"}` instead of `{"category": {"$eq": "legal"}}` — Chroma requires the operator dict form.
