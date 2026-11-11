# Week 05 Project — Reranking & Optimization Layer

## Summary

Build a modular reranking and retrieval optimisation pipeline on top of the week-4 hybrid retriever. The pipeline must be production-grade: typed, tested, configurable via environment variables, and evaluated with before/after quality metrics.

## Deliverables

| File | Description |
|---|---|
| `src/models.py` | `Chunk`, `Document`, `RetrievedContext` dataclasses |
| `src/reranker.py` | `CrossEncoderReranker` with sigmoid normalisation |
| `src/mmr.py` | `mmr_select` with λ parameter |
| `src/query_expansion.py` | `expand_query`, `deduplicate_chunks`, `expand_window` |
| `src/metrics.py` | `precision_at_k`, `recall_at_k`, `reciprocal_rank`, `ndcg_at_k`, `evaluate_retrieval` |
| `src/pipeline.py` | `RerankingPipeline` with `PipelineConfig` (pydantic-settings) |
| `src/before_after_eval.py` | Script producing a metrics comparison table |
| `tests/unit/test_*.py` | Full test suite, all external calls mocked |
| `pyproject.toml` | Dependencies and project metadata |

## Requirements

### Functional
- `RerankingPipeline.retrieve(query)` returns `list[RetrievedContext]` in ranked order.
- Query expansion is toggled via `PipelineConfig.use_query_expansion`. When enabled, `llm` must be provided.
- Parent-document expansion is toggled via `PipelineConfig.use_parent_docs`.
- All config values are overridable via `RAG_` prefixed environment variables.
- `before_after_eval.py` prints a 4-row comparison table (P, R, MRR, NDCG).

### Non-functional
- Zero `ruff` violations.
- All functions typed.
- `logging` not `print`.
- No hardcoded model names outside config defaults.
- No secrets in code (use `os.getenv` or `pydantic-settings`).
- `pytest tests/ -v` passes with 100% of tests green.

## Evaluation criteria

Same rubric as the assessment:

| Dimension | Weight |
|---|---|
| Independent implementation | 40% |
| Conceptual understanding | 25% |
| Debugging & problem-solving | 20% |
| Code quality & testing | 15% |

Pass gate: 75%.

## Tips

- Start with `models.py` and `metrics.py` (no ML dependencies — easiest to test first).
- Add `CrossEncoderReranker` next, using mocks to verify ordering before downloading any model.
- Wire the pipeline last, once each component is independently verified.
- For the eval set: use 20 hardcoded (query, relevant_ids) pairs about a narrow topic so you can manually verify which documents are relevant.
- If NDCG does not improve on your synthetic data, that is fine — document *why* in a comment (synthetic data without a relevance gap shows no improvement by construction).

## Submission

Push to your course repo under `phase-4-rag/week-05-project/`. Tag the commit `p4w5-complete`.
