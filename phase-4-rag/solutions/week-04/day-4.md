# Solutions — Week 04 · Day 4

## Exercise 1 — RRF k sensitivity

```python
for k in [1, 10, 30, 100]:
    fused = reciprocal_rank_fusion([bm25_results, dense_results], k=k)
    log.info("k=%d top-3: %s", k, [doc_id for doc_id, _ in fused[:3]])
```

With k=1: small-rank differences dominate (rank 1 vs 2 is huge). With k=100: all ranks compressed — top-100 all look similar. k=60 gives stable rankings across both lists. The mixed query shows least variation between k=30 and k=60.

## Exercise 2 — Score fusion baseline

```python
def normalise(scores: list[tuple[int, float]]) -> dict[int, float]:
    if not scores:
        return {}
    vals = [s for _, s in scores]
    mn, mx = min(vals), max(vals)
    rng = mx - mn if mx != mn else 1.0
    return {did: (s - mn) / rng for did, s in scores}

def score_fusion(
    bm25_results: list[tuple[int, float]],
    dense_results: list[tuple[int, float]],
    alpha: float = 0.5,
) -> list[tuple[int, float]]:
    bm25_norm = normalise(bm25_results)
    dense_norm = normalise(dense_results)
    all_ids = set(bm25_norm) | set(dense_norm)
    fused = {
        did: alpha * bm25_norm.get(did, 0.0) + (1 - alpha) * dense_norm.get(did, 0.0)
        for did in all_ids
    }
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)
```

Score fusion is less stable than RRF for the CVE query because dense scores for off-target docs span a wide range, distorting the normalisation — one high-scoring doc compresses all others to near 0. RRF is immune.

## Exercise 3 — Missing-list penalty

```python
def reciprocal_rank_fusion_with_penalty(
    ranked_lists: list[list[tuple[int, float]]],
    k: int = 60,
    corpus_size: int = 1000,
) -> list[tuple[int, float]]:
    rrf: dict[int, float] = defaultdict(float)
    all_ids: set[int] = set()
    for rl in ranked_lists:
        for did, _ in rl:
            all_ids.add(did)
    for ranked in ranked_lists:
        present = {did for did, _ in ranked}
        rank_map = {did: r for r, (did, _) in enumerate(ranked, 1)}
        for did in all_ids:
            rank = rank_map.get(did, corpus_size + 1)
            rrf[did] += 1.0 / (k + rank)
    return sorted(rrf.items(), key=lambda x: x[1], reverse=True)
```

For the CVE query: the CVE doc is rank 1 in BM25 but rank 50+ in dense. Without penalty: it still ranks first via RRF. With penalty (corpus_size=12): the penalty contribution is small (1/(60+13) ≈ 0.014) — barely changes output.

## Exercise 4 — Retrieval overlap

```python
def compute_overlap(bm25_top10, dense_top10):
    bm25_ids = {did for did, _ in bm25_top10}
    dense_ids = {did for did, _ in dense_top10}
    return len(bm25_ids & dense_ids) / 10

for label, q in QUERIES:
    overlap = compute_overlap(
        bm25.retrieve(q, top_k=10),
        sem.retrieve(q, top_k=10),
    )
    log.info("%s overlap=%.2f", label, overlap)
```

Typical results: paraphrase query ≈ 0.3 (low overlap — both needed), exact-term queries ≈ 0.1 (very different — hybrid crucial), mixed ≈ 0.5.

## Exercise 5 — Unit tests

```python
# tests/test_hybrid.py
import pytest
from unittest.mock import MagicMock
from day4_hybrid_rrf import reciprocal_rank_fusion, HybridRetriever, BM25Retriever

def test_rrf_both_lists_score_higher():
    list1 = [(0, 5.0), (1, 3.0), (2, 1.0)]
    list2 = [(0, 4.0), (2, 2.0), (3, 1.0)]  # doc 0 in both
    fused = dict(reciprocal_rank_fusion([list1, list2]))
    # doc 0 is rank 1 in both → highest RRF
    assert fused[0] > fused[1]
    assert fused[0] > fused[3]

def test_rrf_one_list_only_doc():
    list1 = [(0, 5.0)]
    list2 = [(1, 5.0)]
    fused = dict(reciprocal_rank_fusion([list1, list2]))
    # both have same RRF score (rank 1 in one list, absent from other)
    assert abs(fused[0] - fused[1]) < 1e-9

def test_fused_list_length():
    list1 = [(i, float(i)) for i in range(10)]
    list2 = [(i + 5, float(i)) for i in range(10)]
    fused = reciprocal_rank_fusion([list1, list2])
    assert len(fused) == 15  # union of 0-9 and 5-14

def test_hybrid_retrieve_returns_text():
    bm25 = BM25Retriever()
    bm25.add_documents(["doc about Python", "doc about BM25"])
    sem = MagicMock()
    sem.retrieve.return_value = [(0, 0.9), (1, 0.8)]
    sem._docs = ["doc about Python", "doc about BM25"]
    hybrid = HybridRetriever(bm25=bm25, semantic=sem, candidate_k=10)
    results = hybrid.retrieve("Python", top_k=2)
    assert len(results) == 2
    assert all(isinstance(text, str) for _, _, text in results)

def test_rrf_k60_default():
    list1 = [(0, 1.0)]
    fused = reciprocal_rank_fusion([list1])
    assert abs(fused[0][1] - 1.0 / (60 + 1)) < 1e-9
```
