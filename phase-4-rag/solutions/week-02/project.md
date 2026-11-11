# Solutions — Week 02 Project

This is a review-and-pointer file. The full project implementation spans all five daily lessons. Check your work against the criteria below.

## Self-check: all 11 functional requirements

Work through each requirement and verify it against your code.

| ID | Requirement | Verification |
|---|---|---|
| F1 | `FixedSizeChunker(chunk_size, overlap)` raises `ValueError` on invalid params | `pytest tests/unit/test_fixed.py::test_invalid_overlap_raises` |
| F2 | `SentenceChunker` never cuts mid-sentence | Inspect any chunk — confirm it ends with `.`, `!`, or `?` |
| F3 | `RecursiveChunker`: no chunk exceeds `chunk_size + max_sep_len` | `pytest tests/unit/test_recursive.py::test_no_chunk_exceeds_limit` |
| F4 | `MarkdownChunker` carries `heading` metadata | `pytest tests/unit/test_markdown.py::test_headings_propagate` |
| F5 | `SemanticChunker` uses `provider.embed` | Check constructor injection; mock provider must receive the `embed` call |
| F6 | `MockEmbeddingProvider`: deterministic, normalised | `pytest tests/unit/test_providers.py::test_mock_provider_normalised` |
| F7 | `OpenAIEmbeddingProvider` reads env var, falls back gracefully | Remove `OPENAI_API_KEY` from env; provider should fall back, not crash |
| F8 | `SentenceTransformerProvider` falls back to mock when absent | Uninstall `sentence-transformers`; provider logs a warning, returns mock vectors |
| F9 | `ChunkEmbedPipeline.run` embeds in batches, normalises | `pytest tests/integration/test_pipeline.py::test_pipeline_produces_indexed_chunks` |
| F10 | `compare_strategies` returns sorted `list[StrategyMetrics]` | Run `scripts/run_comparison.py` — verify table renders with all strategies |
| F11 | `estimate_cost` returns correct `estimated_usd` | `pytest tests/unit/test_providers.py::test_cost_estimate` |

## Common mistakes to check

**Overlap off by one:** `step = chunk_size - overlap`. If you wrote `step = chunk_size + overlap`, chunks will have gaps not overlaps — adjacent chunks won't share text.

**Not normalising after `embed`:** Some providers return unnormalised vectors. Always apply `normalise()` in `ChunkEmbedPipeline.run` regardless of what the provider claims.

**Empty text edge case:** Every chunker should return `[]` for empty input, not raise an exception.

**SemanticChunker with < 2 sentences:** The algorithm requires at least 2 sentences to compute similarities. Return a single chunk immediately if `len(sentences) < 2`.

## COMPARISON_REPORT.md structure

Your report should answer:

1. Which strategy produced the fewest chunks on the sample corpus? Why?
2. Which produced the lowest adjacent-chunk cosine similarity? Interpret this.
3. Which would you recommend for: (a) a Markdown wiki, (b) raw scraped HTML, (c) a transcript of a meeting?
4. One limitation of the `MockEmbeddingProvider` that could affect the validity of the comparison.

## Stretch goal: `ChunkerFactory`

See the full implementation in `solutions/week-02/assessment.md`.

The key design decision: `ChunkerFactory` takes the `EmbeddingProvider` at construction time (not per-call) because `SemanticChunker` needs it and most strategies do not. Passing it per-call would add noise to the interface.

## Stretch goal: tiktoken token counting

```python
# embeddings/cost.py  (extended)
import os

def _count_tokens_tiktoken(texts: list[str], model: str) -> int:
    """Count tokens using tiktoken (accurate, requires 'tiktoken' package)."""
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return sum(len(enc.encode(t)) for t in texts)
    except ImportError:
        # Fall back to heuristic
        return sum(len(t) for t in texts) // CHARS_PER_TOKEN


def estimate_cost(
    texts: list[str],
    model: str = "text-embedding-3-small",
    use_tiktoken: bool = False,
) -> CostEstimate:
    total_chars = sum(len(t) for t in texts)
    if use_tiktoken:
        total_tokens = _count_tokens_tiktoken(texts, model)
    else:
        total_tokens = total_chars // CHARS_PER_TOKEN
    price_per_million = PRICING.get(model, 0.0)
    cost = (total_tokens / 1_000_000) * price_per_million
    return CostEstimate(
        model=model,
        total_tokens=total_tokens,
        total_chars=total_chars,
        estimated_usd=cost,
    )
```
