# Phase 4 · Week 02 · Day 7 — Project & Strategy Comparison Report

> Objective: Complete the chunking+embedding module, run the strategy comparison, interpret results, and prepare for the week assessment.  ·  Time: ~3 hr project + 1 hr assessment prep

## Project brief

See [[project]] for the full specification. Today you complete any remaining implementation, run the comparison harness on a real corpus, interpret the metrics, and write a brief analysis.

## Completion checklist

Work through this list. Check each item only when you can verify it — run tests, not just "I think it works."

### Implementation

- [ ] All five chunker classes implemented and satisfying `ChunkerProtocol`.
- [ ] `ChunkEmbedPipeline.run(text, metadata)` works end-to-end.
- [ ] `compare_strategies` produces `StrategyMetrics` for every strategy.
- [ ] `MockEmbeddingProvider` is the default in all tests — no `OPENAI_API_KEY` required.
- [ ] `estimate_cost` returns correct values for a known input.

### Tests

Run `pytest tests/ -v` — all tests should pass.

- [ ] Unit tests for every chunker.
- [ ] Unit tests for `similarity.py` (normalise, cosine, top_k).
- [ ] Unit tests for `providers.py` (mock, batch, retry path).
- [ ] Integration test for pipeline + metadata propagation.
- [ ] Integration test for `compare_strategies` with ≥ 2 strategies.

### Code quality

- [ ] All functions have type hints.
- [ ] All modules use `logging`, not `print`.
- [ ] No hardcoded secrets. `.env.example` present.
- [ ] `ruff check .` passes (or equivalent linter).

## Running the comparison

Use the sample corpus below (or substitute a document you care about):

```python
# scripts/run_comparison.py
"""Run strategy comparison on a sample corpus and print a summary table."""
from __future__ import annotations

import logging
import textwrap

from rag_pipeline.chunking.fixed import FixedSizeChunker
from rag_pipeline.chunking.sentence import SentenceChunker
from rag_pipeline.chunking.recursive import RecursiveChunker
from rag_pipeline.chunking.markdown import MarkdownChunker
from rag_pipeline.embeddings.providers import MockEmbeddingProvider
from rag_pipeline.compare import compare_strategies

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")

CORPUS = textwrap.dedent("""
    # Introduction to Retrieval-Augmented Generation

    Retrieval-Augmented Generation (RAG) is a technique that improves the factual
    accuracy of large language model outputs by injecting retrieved context into
    the prompt at inference time. Unlike fine-tuning, RAG does not update model
    weights; instead, it augments the model's knowledge dynamically.

    ## The Retrieval Pipeline

    The pipeline has three stages: indexing, retrieval, and generation.

    During indexing, source documents are split into chunks, embedded into dense
    vectors, and stored in a vector database. Good chunking is critical: chunks
    that are too large produce diluted embeddings; chunks that are too small may
    miss cross-sentence context.

    ## Chunking Strategies

    Fixed-size chunking splits on character count with optional overlap. It is
    simple, predictable, and fast. The main risk is splitting sentences mid-way.

    Sentence chunking respects natural language boundaries. It produces chunks of
    variable length that typically align better with semantic units.

    Recursive chunking tries structural separators in order, falling back to
    finer splits only when needed. It preserves paragraph and section structure
    while still enforcing a maximum chunk size.

    Structure-aware chunking parses document format (Markdown headings, HTML
    tags) and aligns chunk boundaries with those structural elements.

    Semantic chunking embeds sentences and finds topic-shift boundaries using
    cosine similarity drops. It is the most expensive but produces the highest
    intra-chunk coherence.

    ## Embedding Models

    Embedding models convert text to dense float vectors. Cloud APIs like OpenAI
    text-embedding-3-small offer high quality with per-token cost. Local models
    like sentence-transformers/all-MiniLM-L6-v2 are free after download and
    suitable for private data.

    Vector dimension ranges from 384 (MiniLM) to 3072 (text-embedding-3-large).
    Higher dimensions capture finer semantic distinctions but increase storage and
    retrieval latency.

    ## Summary

    Choose your chunking strategy based on document structure and retrieval needs.
    Always measure: run a comparison harness on representative data and examine
    chunk size distribution, adjacent-chunk similarity, and retrieval precision on
    held-out queries.
""").strip()

provider = MockEmbeddingProvider()

strategies = {
    "fixed-500-50": FixedSizeChunker(chunk_size=500, overlap=50),
    "fixed-200-20": FixedSizeChunker(chunk_size=200, overlap=20),
    "sentence-600": SentenceChunker(max_chars=600, sentence_overlap=1),
    "recursive-500": RecursiveChunker(chunk_size=500),
    "markdown": MarkdownChunker(min_chunk_chars=100, max_chunk_chars=1000),
}

metrics = compare_strategies(strategies, provider, CORPUS)

print(f"\n{'Strategy':<20} {'Chunks':>7} {'Mean±Std chars':>18} {'Adj sim':>9} {'Coverage':>9}")
print("-" * 70)
for m in metrics:
    print(
        f"{m.strategy_name:<20} {m.num_chunks:>7} "
        f"{m.mean_chars:>7.0f}±{m.stdev_chars:>6.0f}   "
        f"{m.mean_intra_sim:>9.4f} {m.coverage_ratio:>9.3f}"
    )
```

## Interpreting the output

After running `uv run python scripts/run_comparison.py`, examine the table:

| Metric | What it means | Better direction |
|---|---|---|
| `Chunks` | Index size | Depends on use-case |
| `Mean chars` | Average retrieval unit size | Match to expected answer length |
| `Std chars` | Uniformity | Lower = more predictable retrieval |
| `Adj sim` | How similar adjacent chunks are | Lower = clearer topic separation |
| `Coverage` | % of source text indexed | Should be ≥ 0.95 |

Write 3–5 sentences in a comment in `scripts/run_comparison.py` answering:

1. Which strategy produced the lowest adjacent-chunk similarity? Why might that be?
2. Which strategy had the most uniform chunk sizes? What are the implications?
3. For a Markdown-heavy technical knowledge base, which strategy would you recommend and why?

## Pre-assessment review

Re-read Days 1–5 notes for these concepts before the assessment:

- Fixed-size overlap calculation (where exactly does chunk N+1 start?).
- Recursive separator hierarchy — what happens when a piece exceeds `chunk_size` and there are no more separators?
- Semantic chunking: percentile vs. fixed threshold — when does each fail?
- Embedding normalisation: why must you normalise before a dot-product search?
- Cosine similarity formula: `dot(a,b)/(||a||*||b||)`.
- Embedding cost formula: tokens → cost.

## Self-check

- [ ] `pytest tests/ -v` passes with 0 failures.
- [ ] `scripts/run_comparison.py` runs and prints a readable table.
- [ ] I can explain each metric in the output table.
- [ ] I have written my 3-sentence interpretation in the script.
- [ ] I can answer all six pre-assessment bullets without looking at notes.

## Supplementary links

- [Full RAG survey paper (2024)](https://arxiv.org/abs/2312.10997)
- [Evaluating RAG — RAGAS](https://docs.ragas.io/)
- [Practical evaluation of chunking strategies](https://www.pinecone.io/learn/chunking-strategies/)
