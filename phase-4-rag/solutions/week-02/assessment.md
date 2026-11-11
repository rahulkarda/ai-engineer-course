# Solutions — Week 02 Assessment

## Conceptual answers

**Q1 — Chunk size precision/recall tradeoff**

At 100 characters: very high precision (each chunk contains one focused idea), low recall (an answer spanning multiple sentences may not fit in any single chunk; the relevant text is split across chunk boundaries and no individual chunk scores high enough to retrieve it).

At 5,000 characters: high recall (the answer is almost certainly inside one chunk), low precision (the embedding is diluted by many irrelevant surrounding sentences — the retriever returns the correct chunk but also many false positives because their embeddings are pulled toward a similar "topic average").

The sweet spot for most dense technical text is 200–600 characters (roughly 50–150 tokens), tuned empirically.

**Q2 — Recursive splitting trace**

Input: one 150-char paragraph with no blank lines.

1. Try `\n\n` — no match; piece stays as-is (150 chars > 100 limit).
2. Try `\n` — no match (single paragraph, no newlines); piece stays 150 chars.
3. Try `" "` — split on spaces. Produces words. Merge adjacent words until adding the next word would exceed 100; emit chunk, start new one.
4. Result: 2–3 word-based chunks of ≤ 100 chars each.

`""` (character-level) is never reached because space splitting resolves the problem.

**Q3 — Mean pooling vs. CLS token**

Mean pooling averages all non-padding token vectors. It captures the contribution of every token in the sequence. The `[CLS]` token was designed for classification tasks (e.g., BERT NSP); it attends to the full sequence but its representation is biased toward classification objectives. Sentence-transformers empirically show mean pooling produces better semantic similarity scores on MTEB benchmarks.

**Q4 — Why normalise before storing**

After normalisation, `||v||=1`. Cosine similarity then equals the dot product: `cos(a,b) = a·b`. Vector databases implement ANN (approximate nearest neighbours) using highly optimised BLAS dot-product kernels (e.g., FAISS GEMM). Storing unnormalised vectors forces a division at query time for every candidate — slower, and many libraries assume unit vectors.

**Q5 — Euclidean vs. cosine**

Euclidean is equivalent to cosine distance when all vectors are unit-normalised. Proof:
```
||a - b||² = ||a||² + ||b||² - 2·(a·b) = 1 + 1 - 2·cos(θ) = 2·(1 - cos(θ))
```
So `L2(a,b) = sqrt(2·(1-cos_sim(a,b)))` — monotone transform. On unnormalised vectors, a long paraphrase vector has large magnitude; its Euclidean distance to a short paraphrase is large even though they mean the same thing — retrieval quality degrades.

**Q6 — Semantic chunking failure modes**

1. **Domain-specific corpora**: a general embedding model sees all technical paragraphs as high-similarity; no valleys appear, producing very few or no breakpoints → one giant chunk. Mitigation: use a domain-tuned model or fall back to recursive chunking.

2. **Very short documents**: with only 3–4 sentences, percentile-based breakpoints force artificial splits regardless of semantic coherence. Mitigation: add a `min_chunks=1` guard that skips breakpoint detection if the document has fewer than `min_chunk_sentences * 2` sentences.

**Q7 — Cost switch**

Original cost with `text-embedding-3-large`: $5 at $0.13/1M tokens → 5 / 0.13 = 38.46 million tokens.  
Same 38.46M tokens with `text-embedding-3-small` at $0.02/1M: 38.46 × 0.02 = **$0.77**.

Saving: ~85% cost reduction.

**Q8 — `sentence_overlap` example**

Text: "The transformer architecture uses attention. Attention is computed as QKV. The output is fed to FFN layers."

With `sentence_overlap=0`, chunk 0 = "The transformer architecture uses attention. Attention is computed as QKV." and chunk 1 = "The output is fed to FFN layers."

A query about "QKV and FFN connection" has no single chunk that contains both. With `sentence_overlap=1`, chunk 1 = "Attention is computed as QKV. The output is fed to FFN layers." — both concepts appear together.

---

## Coding challenge — reference implementation

```python
# rag_pipeline/chunking/factory.py
from __future__ import annotations

import logging
from typing import Any

from rag_pipeline.chunking.base import Chunk, ChunkerProtocol
from rag_pipeline.chunking.fixed import FixedSizeChunker
from rag_pipeline.chunking.sentence import SentenceChunker
from rag_pipeline.chunking.recursive import RecursiveChunker
from rag_pipeline.chunking.markdown import MarkdownChunker
from rag_pipeline.chunking.semantic import SemanticChunker
from rag_pipeline.embeddings.providers import EmbeddingProvider

logger = logging.getLogger(__name__)


class ChunkerFactory:
    """Construct chunkers from configuration dicts."""

    def __init__(self, provider: EmbeddingProvider) -> None:
        self.provider = provider

    def create(self, config: dict[str, Any]) -> ChunkerProtocol:
        strategy = config.get("strategy")
        logger.debug("ChunkerFactory.create: strategy=%s", strategy)

        if strategy == "fixed":
            return FixedSizeChunker(
                chunk_size=config.get("chunk_size", 500),
                overlap=config.get("overlap", 50),
            )
        elif strategy == "sentence":
            return SentenceChunker(
                max_chars=config.get("max_chars", 600),
                sentence_overlap=config.get("sentence_overlap", 1),
            )
        elif strategy == "recursive":
            return RecursiveChunker(
                chunk_size=config.get("chunk_size", 500),
                separators=config.get("separators", None),
            )
        elif strategy == "markdown":
            return MarkdownChunker(
                min_chunk_chars=config.get("min_chunk_chars", 100),
                max_chunk_chars=config.get("max_chunk_chars", 1500),
            )
        elif strategy == "semantic":
            return SemanticChunker(
                provider=self.provider,
                breakpoint_percentile=config.get("breakpoint_percentile", 25.0),
                window_size=config.get("window_size", 1),
                min_chunk_sentences=config.get("min_chunk_sentences", 2),
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}. Choose from: fixed, sentence, recursive, markdown, semantic")
```

```python
# tests/unit/test_factory.py
import pytest
from rag_pipeline.chunking.factory import ChunkerFactory
from rag_pipeline.chunking.base import ChunkerProtocol
from rag_pipeline.embeddings.providers import MockEmbeddingProvider


@pytest.fixture
def factory():
    return ChunkerFactory(provider=MockEmbeddingProvider())


def test_create_fixed(factory):
    c = factory.create({"strategy": "fixed", "chunk_size": 300, "overlap": 30})
    assert isinstance(c, ChunkerProtocol)


def test_create_sentence(factory):
    c = factory.create({"strategy": "sentence", "max_chars": 400})
    assert isinstance(c, ChunkerProtocol)


def test_create_recursive(factory):
    c = factory.create({"strategy": "recursive", "chunk_size": 200})
    assert isinstance(c, ChunkerProtocol)


def test_create_semantic(factory):
    c = factory.create({"strategy": "semantic", "breakpoint_percentile": 25.0})
    assert isinstance(c, ChunkerProtocol)


def test_unknown_strategy_raises(factory):
    with pytest.raises(ValueError, match="Unknown strategy"):
        factory.create({"strategy": "magic"})


def test_extra_keys_ignored(factory):
    # Should not raise; extra keys are silently ignored
    c = factory.create({"strategy": "fixed", "chunk_size": 100, "overlap": 10, "unknown_key": "x"})
    assert isinstance(c, ChunkerProtocol)


def test_fixed_chunker_produces_chunks(factory):
    c = factory.create({"strategy": "fixed", "chunk_size": 50, "overlap": 5})
    text = "word " * 100
    chunks = c(text)
    assert len(chunks) > 1
    for ch in chunks:
        assert len(ch.text) <= 50
```
