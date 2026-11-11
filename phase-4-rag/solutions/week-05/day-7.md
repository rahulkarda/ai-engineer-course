# Solutions — Week 05 · Day 7 (Project Day)

This is a project completion and self-review day. Solutions here cover the self-review questions from the lesson.

## Self-review answers

**1. Best metric for "first relevant result should appear early"**

MRR (Mean Reciprocal Rank). It gives 1/rank of the *first* relevant result. A system that always puts the best document at rank 1 gets MRR=1.0; one that buries it at rank 10 gets MRR=0.1. NDCG also captures this but weights all relevant positions, not just the first.

**2. MMR formula, λ=0 and λ=1**

Formula: `score = λ · relevance − (1 − λ) · max_redundancy`

- λ=0: `score = −max_redundancy` → pure diversity. The algorithm selects the document *least similar to already-selected documents*, regardless of relevance. In the extreme, results may be irrelevant but maximally spread across the embedding space.
- λ=1: `score = relevance` → pure relevance. MMR degenerates to standard relevance ranking. Diversity is ignored.
- Maximum diversity: λ=0. Maximum relevance: λ=1.

**3. HyDE hurts on factual questions**

HyDE embeds a *hypothetical answer*, not the query. For "What is the capital of France?", the hypothesis is "Paris is the capital of France." This is short and specific — the embedding may be fine. But the problem is that factual Q&A corpora often contain the answer verbatim ("The capital of France is Paris."). The query embedding itself ("What is the capital of France?") is semantically close to that sentence. HyDE adds noise (the LLM may hallucinate) without adding value. Fix: gate HyDE on question type — use it for complex "how/why" questions, not simple factual lookups.

**4. Handling parent docs larger than context window**

- Sliding-window summarisation: split the parent into 4 096-token segments and summarise each; pass summaries to the LLM.
- Map-reduce retrieval: retrieve at sub-doc level (smaller parents), not full-document level. Adjust `ParentDocumentStore` to store medium-sized chunks (1 000–2 000 tokens) as "parents", not full documents.

**5. Interpreting NDCG@5 = 0.65**

You need a baseline to interpret it: What is the NDCG@5 of a random retriever (lower bound)? A BM25 baseline? State of the art on similar data? Without comparison, 0.65 is meaningless. You also need to know whether the eval set reflects real-world query distribution.

**6. Near-duplicate vs exact-duplicate**

- Exact: identical content hash. Catches copy-pasted or re-indexed chunks.
- Near-duplicate: cosine similarity > threshold. Catches paraphrased or overlapping chunks (e.g., overlapping sliding windows from a 256-token stride on a 512-token window).

Real-world scenario: a PDF is indexed twice (once as full text, once via an OCR pass). The OCR introduces minor differences (line breaks, ligature characters). Content hashes differ; cosine similarity ≈ 0.98 → caught by near-duplicate deduplication.

**7. P@5 improves, R@10 drops: is this an improvement?**

Depends on the use case. If the goal is to provide a short, precise context to an LLM (top-5 results), then the improvement in P@5 is what matters. If the goal is to ensure no relevant document is missed (e.g., for legal or medical search), the recall drop at k=10 is a regression. Most RAG pipelines optimise for precision in a fixed context window — so yes, this is likely an improvement. Document the trade-off clearly.
