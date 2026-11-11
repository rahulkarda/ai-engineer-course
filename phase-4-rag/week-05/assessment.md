# Week 05 Assessment — Reranking & Retrieval Optimization

## Conceptual (interview-style)

Answer each question concisely (2–5 sentences).

1. A bi-encoder returns the top-10 results for a query in 8 ms. A cross-encoder reranking those 10 results takes 120 ms. A user says this is too slow. What are three engineering options to reduce latency without removing the cross-encoder entirely?

2. Explain the MMR formula. What happens at λ=0? What happens at λ=1? What value of λ maximises diversity?

3. Your retrieval pipeline uses HyDE query expansion. A user reports that for factual questions like "What is the capital of France?", HyDE hurts precision. Why might this be? What would you do?

4. You are building a parent-document retrieval system. A document is 10 000 tokens. Your LLM context window is 4 096 tokens. What are two options for handling this mismatch?

5. NDCG@5 for your pipeline is 0.65. Your manager asks "is that good?". What information do you need to answer this question properly?

6. What is the difference between near-duplicate deduplication and exact-duplicate deduplication? Give one real-world scenario where you would need near-duplicate deduplication.

7. You run a before/after evaluation. Precision@5 improves from 0.40 to 0.55 but Recall@10 drops from 0.80 to 0.70. Is this an improvement? Justify your answer.

## Coding challenge

### Specification

Implement a `RetrievalEvaluator` class that:

1. Accepts a list of `(query, relevant_doc_ids)` tuples at construction time.
2. Has a method `evaluate(retrieve_fn, k=5)` that:
   - Calls `retrieve_fn(query) -> list[str]` for each query.
   - Computes Precision@k, Recall@k, MRR, and NDCG@k (binary relevance for NDCG).
   - Returns a `dict[str, float]` with keys `"precision"`, `"recall"`, `"mrr"`, `"ndcg"`.
3. Has a method `compare(baseline_fn, improved_fn, k=5)` that calls `evaluate` twice and returns a dict with keys `"baseline"`, `"improved"`, and `"delta"` (each a metric dict).

### Acceptance criteria

- All metrics are correctly computed (verified against hand-calculated examples in the rubric).
- `compare` returns a `"delta"` dict where each value = improved − baseline.
- A `retrieve_fn` that returns all relevant docs first must produce P@k=1.0 and R@k = min(k, |relevant|)/|relevant|.
- All external calls are mockable.
- Type hints throughout. Uses `logging`, not `print`.
- `pytest` tests: at least 6 test cases covering normal operation and edge cases.

### Constraints

- Use only the standard library and `numpy`. No external IR libraries.
- Code must pass `ruff check`.

## Rubric (pass ≥ 75 points)

| Dimension | Weight | Full marks criteria |
|---|---|---|
| Independent implementation | 40 pts | Evaluator implemented correctly; metrics match hand-calculations; compare returns correct delta |
| Conceptual understanding | 25 pts | All 7 conceptual questions answered accurately |
| Debugging & problem-solving | 20 pts | Edge cases handled (empty retrieved list, query with no relevant docs, k > list length); tests demonstrate these |
| Code quality & testing | 15 pts | Type hints, logging, ruff-clean, ≥6 meaningful tests |

**Pass gate: 75 / 100**

Answer key: ../solutions/week-05/assessment.md
