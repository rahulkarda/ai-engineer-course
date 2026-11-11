# Phase 4 · Week 04 · Day 7 — Week Review & Assessment

> Objective: Consolidate retrieval knowledge; complete the week assessment.  ·  Time: ~3 hr project review + 1 hr assessment  ·  Prereqs: [[day-1]] through [[day-6]]

## Week 4 summary

| Day | Topic | Key takeaway |
|---|---|---|
| 1 | Semantic / dense retrieval | FAISS + encoder; cosine similarity; HNSW internals |
| 2 | TF-IDF & BM25 theory | TF saturation (k₁), length normalisation (b), IDF formula |
| 3 | BM25 from scratch | Inverted index, incremental add, serialisation |
| 4 | Hybrid retrieval & RRF | RRF formula, score-invariance, fusion strategies |
| 5 | Filtering & query construction | Pre/post filter, HyDE, expansion, decomposition, top-k tuning |
| 6 | Project build | End-to-end hybrid retriever with eval metrics |

## Retrieval tradeoffs — decision guide

```
Is the query a paraphrase / synonym?       → Semantic retrieval favoured
Does the query contain exact IDs / codes?   → BM25 favoured
Is the corpus multilingual?                 → Semantic retrieval favoured
Is latency critical (< 10ms)?               → BM25 favoured (no encoder)
No labelled data to tune alpha?             → Use RRF (not score fusion)
Both paraphrase and exact-match queries?    → Hybrid with RRF
Large corpus (> 1M docs)?                   → ANN index required; BM25 inverted index OK
```

## Review tasks (do before assessment)

1. Re-read your `hybrid.py` from day-6. Can you explain every line?
2. Run `pytest tests/` — all tests should pass.
3. Run `compare_retrievers.py` — check the MRR table. If hybrid doesn't beat both solo retrievers on at least 3/5 queries, debug `rrf_k` or `candidate_k`.
4. Without looking at your code, re-derive the BM25 formula on paper. Check k₁ saturation with f=1 and f=100 (k₁=1.5).
5. Explain to yourself why RRF doesn't need score normalisation (the key insight: it's rank-based, not score-based).

## Practice problems

1. Given a BM25 inverted index, show the complete scoring computation for query "fast retrieval" against a document "fast information retrieval algorithms" (5 tokens). Use k₁=1.5, b=0.75, N=100, df("fast")=10, df("retrieval")=20, avgdl=8.
2. A hybrid system returns:  
   BM25 list: [A, B, C, D, E]  
   Dense list: [C, A, D, F, G]  
   Compute RRF scores (k=60) for A, B, C, D, E, F, G and rank them.
3. You add a pre-filter that restricts to 10% of the corpus. Estimate the speedup for (a) BM25 with inverted index, (b) exact k-NN dense retrieval, (c) HNSW ANN.

## Assessment

See [[assessment.md]] for the full graded assessment (90 min).

## Supplementary links

- [BEIR benchmark — cross-domain retrieval evaluation](https://github.com/beir-cellar/beir)
- [Pinecone "Hybrid Search Explained"](https://www.pinecone.io/learn/hybrid-search-intro/)
- [Weaviate BM25 docs](https://weaviate.io/developers/weaviate/search/bm25)
