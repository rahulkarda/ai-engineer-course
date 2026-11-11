# Phase 4 · Week 05 · Day 7 — Project: Reranking & Optimization Layer

> Objective: Complete, test, and evaluate the full week-5 reranking pipeline; prepare for the assessment.  ·  Time: ~3 hr project + 1 hr review  ·  Prereqs: [[day-6]]

## Project completion checklist

Work through the following in order. Each item should be a testable deliverable.

### Part 1 — Pipeline completeness (1 hr)

- [ ] `pipeline.py` passes all unit tests from day-6.
- [ ] All five stages (query expansion, retrieval, dedup, reranking, MMR, parent-doc) are wired.
- [ ] `PipelineConfig` is loaded from environment variables via `pydantic-settings`:
  ```python
  from pydantic_settings import BaseSettings
  class PipelineSettings(BaseSettings):
      reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
      mmr_k: int = 5
      mmr_lambda: float = 0.7
      # ... etc.
      model_config = {"env_prefix": "RAG_"}
  ```
- [ ] No hardcoded model names or thresholds outside of `PipelineConfig` defaults.

### Part 2 — Evaluation (1 hr)

Build a `before_after_eval.py` script that:

1. Defines a 20-query eval set over a synthetic or real corpus (at least 30 documents about a topic of your choice — Python best practices, RAG concepts, software architecture, etc.).
2. Implements a `BaselineRetriever` that uses BM25 (via `rank_bm25`) or simple TF-IDF cosine similarity — no cross-encoder, no MMR.
3. Implements a `FullPipeline` wrapper that calls `RerankingPipeline.retrieve` and returns doc_ids.
4. Calls `evaluate_retrieval` for both. Prints a comparison table:

```
Metric      Baseline    Full Pipeline    Delta
P@5         0.320       0.480            +0.160
R@5         0.450       0.610            +0.160
MRR         0.520       0.720            +0.200
NDCG@5      0.410       0.590            +0.180
```

5. Logs: "Improvement observed: YES/NO" based on whether NDCG@5 improved.

### Part 3 — Test coverage (45 min)

Minimum test coverage for submission:

| Module | Tests required |
|---|---|
| `reranker.py` | ordering, threshold, empty input, score in [0,1] |
| `mmr.py` | diversity with λ<0.5, pure relevance with λ=1, k clamping |
| `query_expansion.py` | expand includes original, dedup removes dupes, window clamping |
| `metrics.py` | precision, recall, MRR, NDCG (all from day-5 tests) |
| `pipeline.py` | happy path, empty retriever, parent expansion, no-llm guard |

Run: `pytest tests/ -v --tb=short`

### Part 4 — Code quality (15 min)

- [ ] `ruff check src/ tests/` — zero violations.
- [ ] All functions have type hints.
- [ ] All modules use `logging`, not `print`.
- [ ] No API keys, model weights, or hardcoded paths in committed files.

## Self-review questions

Answer these before the assessment:

1. What metric best captures "the first relevant result should appear early"? Why?
2. At what λ does MMR produce identical results to reranking by relevance alone?
3. Why does HyDE work better than raw query embedding for "how does X work?" questions?
4. If your NDCG@5 is 0.40 on the baseline and 0.38 on the full pipeline, what could explain the regression?
5. When would you *not* use parent-document expansion?

## Submission criteria

A complete week-5 submission contains:
- All source files in `src/`
- All tests passing (`pytest tests/ -v`)
- `before_after_eval.py` producing a metrics table
- No ruff violations
- A `pyproject.toml` with `[project.dependencies]` listing `sentence-transformers`, `pydantic`, `pydantic-settings`, `numpy`, `rank-bm25` (if used), `pytest`

## Supplementary links

- [sentence-transformers: usage & models](https://www.sbert.net/)
- [rank-bm25 Python library](https://github.com/dorianbrown/rank_bm25)
- [pydantic-settings docs](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [BEIR evaluation framework](https://github.com/beir-cellar/beir)
