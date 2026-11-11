# Week 02 Project — Configurable Chunking & Embedding Module

## Overview

Build a production-quality `rag_pipeline` Python package that implements five chunking strategies, a pluggable embedding provider interface, and a strategy comparison harness. The deliverable is a fully tested, documented module that a colleague could drop into any RAG project.

## Goals

By the end of this project you will have:

- A `ChunkerProtocol` satisfied by five distinct chunker implementations.
- An `EmbeddingProvider` protocol satisfied by a mock provider, an OpenAI provider, and a sentence-transformers provider (last two with graceful fallback when unavailable).
- A `ChunkEmbedPipeline` that runs end-to-end.
- A `compare_strategies` harness producing quantitative metrics.
- ≥ 80% test coverage on the `rag_pipeline` package.
- A written comparison report (3–5 sentences per strategy).

## Requirements

### Functional

| ID | Requirement |
|---|---|
| F1 | `FixedSizeChunker(chunk_size, overlap)` — raises `ValueError` on invalid params |
| F2 | `SentenceChunker(max_chars, sentence_overlap)` — never cuts mid-sentence |
| F3 | `RecursiveChunker(chunk_size, separators)` — no chunk exceeds `chunk_size + max_sep_len` |
| F4 | `MarkdownChunker(min_chunk_chars, max_chunk_chars)` — each chunk carries `heading` metadata |
| F5 | `SemanticChunker(provider, breakpoint_percentile, window_size)` — uses `provider.embed` |
| F6 | `MockEmbeddingProvider` — deterministic, no network, normalised output |
| F7 | `OpenAIEmbeddingProvider` — reads `OPENAI_API_KEY` from env; falls back to mock when key absent |
| F8 | `SentenceTransformerProvider` — falls back to mock when package absent |
| F9 | `ChunkEmbedPipeline.run(text, metadata)` — embeds in batches, normalises all vectors |
| F10 | `compare_strategies(strategies, provider, text)` — returns `list[StrategyMetrics]` sorted by `mean_intra_sim` |
| F11 | `estimate_cost(texts, model)` — returns `CostEstimate` with correct `estimated_usd` |

### Non-functional

| ID | Requirement |
|---|---|
| NF1 | All public functions and classes have type hints |
| NF2 | `logging` used throughout; no bare `print` statements |
| NF3 | `.env.example` documents all required env vars |
| NF4 | `ruff check .` passes with no errors |
| NF5 | `pytest tests/` passes with 0 failures |
| NF6 | No secrets, no hardcoded API keys |

## Deliverables

```
rag_pipeline/
├── __init__.py
├── chunking/
│   ├── __init__.py
│   ├── base.py
│   ├── fixed.py
│   ├── sentence.py
│   ├── recursive.py
│   ├── markdown.py
│   └── semantic.py
├── embeddings/
│   ├── __init__.py
│   ├── providers.py
│   ├── similarity.py
│   └── cost.py
├── pipeline.py
└── compare.py
tests/
├── unit/         (≥ 7 test files)
└── integration/  (≥ 1 test file)
scripts/
└── run_comparison.py
pyproject.toml
.env.example
COMPARISON_REPORT.md   ← your analysis of the strategy comparison output
```

## Grading

| Criterion | Points |
|---|---|
| All 11 functional requirements implemented | 40 |
| All 6 non-functional requirements met | 15 |
| Unit tests cover each chunker's edge cases | 20 |
| Integration test covers pipeline + metadata | 10 |
| `COMPARISON_REPORT.md` with substantive analysis | 10 |
| Stretch: `query_top_k` method on pipeline | 5 (bonus) |

**Pass gate:** ≥ 75 / 100.

## Stretch goals

- Implement `ChunkerFactory` (from the assessment coding challenge) and wire it into `run_comparison.py` so strategies are configured from a JSON file.
- Add tiktoken-based token counting to replace the 4-chars heuristic in `estimate_cost`.
- Add `SemanticChunker` to the comparison run and add a 4th column to the report comparing it against the structural strategies.

## Timeline suggestion

| Time block | Focus |
|---|---|
| 0–45 min | Scaffold package, implement `base.py` and `FixedSizeChunker` |
| 45–90 min | Implement remaining four chunkers |
| 90–150 min | Implement `providers.py`, `similarity.py`, `cost.py` |
| 150–195 min | Implement `pipeline.py` and `compare.py` |
| 195–225 min | Write all unit and integration tests |
| 225–240 min | Run comparison, write `COMPARISON_REPORT.md` |

## Hints

- Start with `MockEmbeddingProvider`. Every test uses it — get it right first.
- Verify `ChunkerProtocol` satisfaction with `assert isinstance(chunker, ChunkerProtocol)` — Python's runtime check uses structural typing.
- For `RecursiveChunker`, test the boundary: a piece exactly equal to `chunk_size` must NOT trigger recursion.
- In `compare_strategies`, add a guard: if `len(indexed) < 2`, set `mean_intra_sim=0.0`.

## Reference solution

Solutions: [[../solutions/week-02/project.md]]
