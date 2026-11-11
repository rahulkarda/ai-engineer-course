# Phase 4 · Week 02 · Day 1 — Why Chunk? Fixed-Size & Sentence Chunking

> Objective: Understand why raw text must be chunked before embedding, implement fixed-size and sentence chunking strategies with configurable overlap.  ·  Time: ~1 hr  ·  Prereqs: [[week-01/day-7]]

## Concept

LLMs and embedding models have **context windows** — hard token limits. A 30-page PDF exceeds every practical limit. Chunking solves three problems simultaneously:

1. **Context window fit** — each chunk must fit inside the embedding model's max input length (typically 512–8192 tokens).
2. **Retrieval granularity** — a 10,000-word document embedded as one vector loses specificity; the right paragraph scores better than "somewhere in this book."
3. **Noise reduction** — a chunk focused on one idea produces a cleaner, more discriminative embedding.

The tradeoff is always precision vs. recall:

- **Too small** — high precision (relevant text only), low recall (answer spans chunk boundary, gets cut off).
- **Too large** — high recall, poor precision (embeddings diluted by irrelevant surrounding text).

The sweet spot depends on the embedding model, the query length, and document structure. You will learn to measure this empirically in the project.

## How it works internally

### Fixed-size chunking

Split on character count. Overlap copies a tail of the previous chunk into the head of the next so sentences crossing the boundary appear in both.

```
[chunk 0]  chars 0..499
[chunk 1]  chars 400..899   ← 100-char overlap with chunk 0
[chunk 2]  chars 800..1299
```

Overlap means each chunk independently contains some cross-boundary context. The cost is redundant text and slightly inflated index size.

### Sentence chunking

Detect sentence boundaries (`.`, `!`, `?` followed by whitespace), accumulate sentences until you reach the target token/character budget, then start a new chunk. Overlap works at sentence granularity: carry the last *N* sentences into the next chunk.

Sentence chunking respects natural language structure — no embedding of a half-sentence. It produces variable-length chunks (fine for embedding models).

## Worked example

```python
# chunking/fixed.py
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    text: str
    index: int
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.text)


def fixed_size_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """Split *text* into fixed-character chunks with optional overlap.

    Args:
        text: Raw input string.
        chunk_size: Maximum characters per chunk.
        overlap: Characters of overlap between consecutive chunks.

    Returns:
        Ordered list of Chunk objects.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(f"overlap must be in [0, chunk_size), got {overlap}")

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    start = 0
    idx = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        chunks.append(Chunk(text=chunk_text, index=idx))
        logger.debug("chunk %d: chars %d–%d (%d chars)", idx, start, end, len(chunk_text))
        start += step
        idx += 1

    logger.info("fixed_size_chunks: produced %d chunks from %d chars", len(chunks), len(text))
    return chunks
```

```python
# chunking/sentence.py
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Sentence boundary: period/exclamation/question followed by whitespace or end-of-string.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Naively split text into sentences using punctuation boundaries."""
    parts = _SENTENCE_END.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


@dataclass
class Chunk:
    text: str
    index: int
    metadata: dict = field(default_factory=dict)


def sentence_chunks(
    text: str,
    max_chars: int = 600,
    sentence_overlap: int = 1,
) -> list[Chunk]:
    """Accumulate sentences into chunks up to *max_chars*, with sentence-level overlap.

    Args:
        text: Raw input string.
        max_chars: Soft character limit per chunk.
        sentence_overlap: Number of trailing sentences from the previous chunk
                          to prepend to the next.

    Returns:
        Ordered list of Chunk objects.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    idx = 0

    for sent in sentences:
        sent_len = len(sent)
        # If adding this sentence would exceed the limit AND we already have content, flush.
        if current_len + sent_len > max_chars and current:
            chunk_text = " ".join(current)
            chunks.append(Chunk(text=chunk_text, index=idx))
            logger.debug("sentence chunk %d: %d chars, %d sentences", idx, len(chunk_text), len(current))
            idx += 1
            # Carry over the overlap tail.
            overlap_tail = current[-sentence_overlap:] if sentence_overlap > 0 else []
            current = overlap_tail + [sent]
            current_len = sum(len(s) for s in current)
        else:
            current.append(sent)
            current_len += sent_len

    if current:
        chunk_text = " ".join(current)
        chunks.append(Chunk(text=chunk_text, index=idx))

    logger.info("sentence_chunks: produced %d chunks", len(chunks))
    return chunks
```

```python
# Demo
if __name__ == "__main__":
    import logging
    from chunking.fixed import fixed_size_chunks
    from chunking.sentence import sentence_chunks

    logging.basicConfig(level=logging.INFO)

    sample = (
        "Retrieval-Augmented Generation (RAG) is a technique that improves LLM accuracy "
        "by injecting retrieved context into the prompt. The retrieval step depends critically "
        "on how well documents are chunked and embedded. Fixed-size chunking is simple but "
        "can cut sentences in half. Sentence-aware chunking respects natural boundaries. "
        "Both strategies support configurable overlap to avoid losing cross-boundary information."
    )

    fixed = fixed_size_chunks(sample, chunk_size=150, overlap=30)
    for c in fixed:
        print(f"[fixed {c.index}] {c.text[:60]}...")

    sent = sentence_chunks(sample, max_chars=200, sentence_overlap=1)
    for c in sent:
        print(f"[sentence {c.index}] {c.text[:60]}...")
```

## Exercises

1. Modify `fixed_size_chunks` to accept `chunk_size` in **tokens** (approx: 1 token ≈ 4 characters). Add a `unit: Literal["chars", "tokens"]` parameter.
2. Add `source` and `page` keys to each `Chunk.metadata` dict in both chunkers. The caller should pass them in.
3. Write a `chunking/base.py` that defines a `ChunkerProtocol` (using `typing.Protocol`) that both chunkers satisfy.
4. Write `tests/unit/test_fixed.py` covering: empty string input, `chunk_size=10 overlap=0`, overlap boundary correctness, and a `ValueError` on bad overlap.
5. Measure average chunk size for fixed vs. sentence chunking on a 5,000-character text. Which produces more uniform sizes?

(Solutions: ../solutions/week-02/day-1.md)

## Self-check

- [ ] I can explain the precision/recall tradeoff of chunk size in plain English.
- [ ] I can implement fixed-size chunking with overlap from scratch without reference.
- [ ] I can implement sentence chunking and explain where the overlap boundary falls.
- [ ] I can write a `ChunkerProtocol` and verify both chunkers satisfy it.
- [ ] My tests cover edge cases: empty input, exact boundary, overlap > chunk.

## Supplementary links

- [LangChain text splitters docs](https://python.langchain.com/docs/how_to/split_by_token/)
- [OpenAI cookbook — chunking strategies](https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb)
- [Pinecone — chunking strategies guide](https://www.pinecone.io/learn/chunking-strategies/)
