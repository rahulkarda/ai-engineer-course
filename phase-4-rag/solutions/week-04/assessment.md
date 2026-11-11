# Solutions — Week 04 Assessment

## Conceptual answers

### Q1 — TF saturation in BM25

With k₁=1.5: as f → ∞, the BM25 TF contribution approaches:

```
lim_{f→∞}  f(k₁+1) / (f + k₁·norm)  =  k₁+1 = 2.5
```

So the maximum possible TF contribution per term is `(k₁+1) · IDF = 2.5 · IDF`, regardless of how many times the term repeats. This is desirable because the 50th occurrence of "the" in a document adds no real evidence of relevance beyond the 5th — the term simply dominates the document, not the topic. TF-IDF has no such ceiling, so a document mentioning "Python" 100 times in unrelated context beats a 10-token expert answer about Python.

### Q2 — Reciprocal Rank Fusion

RRF assigns each document a score = Σ 1/(k + rank_r) across retriever lists. It ignores raw scores entirely — only rank positions matter. This makes it scale-invariant: BM25 scores (0–10) and cosine scores (0–1) are incomparable, but ranks 1 through N are directly comparable across any retriever. Without labelled data we cannot tune the α weight in score fusion reliably, but RRF works well with default k=60 across diverse retrieval benchmarks. Its smoothing constant k prevents rank-1 documents from dominating by an unreasonable margin over rank-2.

### Q3 — GDPR article 17 hybrid scenario

Hybrid retrieval runs both dense (returns summaries) and BM25 (returns exact text containing "GDPR article 17") in parallel. RRF fusion places the exact statutory text high in the fused list because BM25 ranks it rank 1, contributing 1/(60+1) ≈ 0.016. The semantic summaries also contribute. The cost: two retrieval operations + encoder latency for the dense leg. For exact statutory text, a filter (`source="legislation"`) would further improve precision by excluding summaries outright.

### Q4 — HNSW vs. flat FAISS

| | Build time | Query time | Accuracy |
|---|---|---|---|
| Flat | O(1) | O(n·d) | 100% exact |
| HNSW | O(n log n) | O(log n) | 95–99% (ANN) |

Choose flat when: (a) n < ~100k (flat is fast enough), (b) correctness is critical and recall@k must be 100%, (c) embeddings change frequently (HNSW rebuild is expensive).

### Q5 — HyDE

HyDE (Hypothetical Document Embedding) generates a hypothetical answer to the query using an LLM, then embeds that answer rather than the raw question. The hypothesis is usually closer in vocabulary and style to the actual relevant documents than the original question. This helps when questions are very short or use different terminology from the corpus. Risk: if the LLM generates a plausible-sounding but factually wrong answer, the embedding will steer retrieval toward documents matching the wrong content — amplifying hallucinations rather than correcting them. HyDE requires an LLM call per query, adding latency and cost.

### Q6 — Memory footprint

**(a) BM25 inverted index:**  
500k terms × 200 postings × 8 bytes = 800 MB. Plus overhead for doc lengths, vocab strings (~10 bytes each) = 5 MB extra. Total ≈ **~800 MB**. Fits in 32GB RAM easily.

**(b) FAISS dense index (dim=384, float32):**  
10M documents × 384 × 4 bytes = 15.36 GB. This fits in 32GB RAM but is tight — no room for the application, OS, or other data structures. An HNSW index adds ~20–60 bytes per vector for graph edges, pushing to ~18–22 GB. A flat index at exactly 15.4 GB fits. In practice, for 10M docs you would use a quantised IVF index (PQ compression) to reduce memory to 1–3 GB at a small accuracy cost.

---

## Coding challenge — bugs in bm25_retrieve

```python
# Bug 1 (line: for term in tokens):
# Counts a term multiple times if it appears in a document more than once,
# inflating df. Fix: use `for term in set(tokens):`
for term in set(tokens):
    df[term] += 1

# Bug 2 (idf function: log(N / df.get(t, 1))):
# Classic IDF formula returns log(1) = 0 when term appears in every document,
# and can return 0 for very common terms (df close to N), giving no penalty.
# More critically: if df > N this would go negative. The BM25 Lucene IDF
# formula should be used instead:
def idf(t):
    d = df.get(t, 0)
    return math.log((N - d + 0.5) / (d + 0.5) + 1)

# Bug 3 (denominator: missing parentheses grouping):
# The formula f + k1 * (1 - b + b * dl / avgdl) is correct in isolation,
# but the code computes:  f + k1 * 1 - b + b * dl / avgdl
# which due to Python operator precedence is: f + k1 - b + (b*dl/avgdl).
# Fix: explicit parentheses:
s += idf(t) * f * (k1 + 1) / (f + k1 * (1 - b + b * (dl / avgdl)))

# Bug 4 (BONUS — sort direction):
# `sorted(..., key=lambda x: x[1])` sorts ascending (lowest score first).
# Should be descending:
return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
```

**Summary:**
1. `set(tokens)` missing → df inflated → IDF scores wrong.
2. IDF formula is vanilla TF-IDF, not BM25 Lucene — no "+0.5" buffer → edge cases fail.
3. Missing parentheses → wrong denominator → wrong TF normalisation.
4. (Bonus) Sort ascending → returns worst matches, not best.

---

## Code quality checklist

Full marks require:
- Every function/method has type annotations including return type.
- `import logging; log = logging.getLogger(__name__)` at top; all output via `log.*`.
- No `os.getenv` needed here (local model), but the pattern must be used for any hypothetical API key.
- `Document`, `SearchResult`, `RetrieverConfig` are Pydantic models with field validation.
- `SentenceTransformer` is patched via `unittest.mock.patch` — no real network/GPU calls in tests.
- All file I/O uses `pathlib.Path` (not `os.path.join` or bare string concatenation).
