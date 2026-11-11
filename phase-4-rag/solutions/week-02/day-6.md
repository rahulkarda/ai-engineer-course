# Solutions — Week 02 · Day 6

## Exercise answers

### Exercise 1 — Add RecursiveChunker and MarkdownChunker to comparison

```python
# tests/integration/test_pipeline.py  (extended)

from rag_pipeline.chunking.recursive import RecursiveChunker
from rag_pipeline.chunking.markdown import MarkdownChunker


def test_compare_all_strategies(provider):
    strategies = {
        "fixed": FixedSizeChunker(chunk_size=150, overlap=30),
        "sentence": SentenceChunker(max_chars=200, sentence_overlap=1),
        "recursive": RecursiveChunker(chunk_size=150),
        "markdown": MarkdownChunker(min_chunk_chars=50, max_chunk_chars=400),
    }
    metrics = compare_strategies(strategies, provider, SAMPLE_TEXT)
    assert len(metrics) == 4
    # MarkdownChunker should have lower stdev than FixedSizeChunker on markdown text
    md_m = next(m for m in metrics if m.strategy_name == "markdown")
    fixed_m = next(m for m in metrics if m.strategy_name == "fixed")
    # MarkdownChunker groups by section — fewer, larger chunks
    assert md_m.num_chunks <= fixed_m.num_chunks
```

**Observation on `stdev_chars`:** `FixedSizeChunker` has near-zero stdev (all chunks ≈ same length). `MarkdownChunker` has high stdev because sections vary greatly in length.

### Exercise 2 — `query_top_k` method

```python
# rag_pipeline/pipeline.py  (addition to ChunkEmbedPipeline)

from rag_pipeline.embeddings.similarity import top_k as _top_k


def query_top_k(
    self,
    query_text: str,
    indexed: list[IndexedChunk],
    k: int = 5,
) -> list[tuple[IndexedChunk, float]]:
    """Embed *query_text* and return the top-k closest IndexedChunks."""
    if not indexed:
        return []

    raw = self.provider.embed([query_text])
    query_vec = normalise(np.array(raw[0], dtype=np.float32))

    corpus = np.stack([ic.embedding for ic in indexed])  # (N, D)
    top = _top_k(query_vec, corpus, k=k)

    return [(indexed[i], sim) for i, sim in top]
```

### Exercise 3 — Coverage ratio assertion

```python
def test_coverage_ratio(provider):
    chunker = FixedSizeChunker(chunk_size=100, overlap=20)
    pipeline = ChunkEmbedPipeline(chunker, provider)
    result = pipeline.run(SAMPLE_TEXT)
    total_embedded_chars = sum(len(ic.chunk.text) for ic in result)
    original_chars = len(SAMPLE_TEXT)
    # With overlap, embedded chars >= original; ratio should be between 0.9 and 2.0
    ratio = total_embedded_chars / original_chars
    assert 0.9 <= ratio <= 2.0, f"coverage ratio out of range: {ratio:.3f}"
```

### Exercise 4 — `__repr__` for StrategyMetrics

```python
@dataclass
class StrategyMetrics:
    ...

    def __repr__(self) -> str:
        return (
            f"StrategyMetrics(strategy={self.strategy_name!r}, "
            f"chunks={self.num_chunks}, "
            f"mean_chars={self.mean_chars:.0f}, "
            f"adj_sim={self.mean_intra_sim:.4f})"
        )
```

### Exercise 5 — Profiling

```python
import cProfile
import pstats

from rag_pipeline.chunking.fixed import FixedSizeChunker
from rag_pipeline.embeddings.providers import MockEmbeddingProvider
from rag_pipeline.pipeline import ChunkEmbedPipeline

text = "word " * 10_000  # ~50,000 chars

def run():
    p = ChunkEmbedPipeline(FixedSizeChunker(500, 50), MockEmbeddingProvider())
    p.run(text)

pr = cProfile.Profile()
pr.enable()
run()
pr.disable()
stats = pstats.Stats(pr)
stats.sort_stats("cumulative")
stats.print_stats(10)
```

**Expected finding:** With `MockEmbeddingProvider`, both stages are fast. With a real provider, the `embed` call (network I/O or model inference) dominates. Chunking is CPU-bound string operations — negligible compared to embedding.
