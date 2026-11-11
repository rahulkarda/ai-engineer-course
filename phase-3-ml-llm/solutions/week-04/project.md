# Solutions — Week 04 · Project

## Complete reference implementation

This solution mirrors the spec in `project.md`. All components are tested and runnable.

---

## `src/bpe_tokenizer.py`

See `day-6.md` — the reference implementation there is the complete, production-quality solution.

Key things the solution must have:
- `fit` → `encode` → `decode` round-trip
- `encode_ids` returning integer IDs
- `save` / `load` via JSON
- Edge case guards on empty string and punctuation-only input

---

## `src/semantic_search.py`

See `day-6.md` and `day-5.md` — the reference implementation there is the complete solution.

Key things:
- `EmbeddingProvider` Protocol
- Batched `build_index` with `batch_size`
- L2 normalisation before storing index
- `save_index` / `load_index` using `numpy.save` + plain text doc list

---

## `demo.py` — complete reference

```python
"""demo.py — End-to-end demonstration of BPE tokenizer and semantic search."""

import logging
import numpy as np
from pathlib import Path

from src.bpe_tokenizer import BPETokenizer
from src.semantic_search import SemanticSearchEngine, StubEmbedder, CachedEmbedder

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ARTEFACTS = Path("artefacts")
ARTEFACTS.mkdir(exist_ok=True)

TRAIN_CORPUS: list[str] = [
    "low lower lowest",
    "newer newest news",
    "old older oldest",
    "wide wider widest",
    "fast faster fastest",
    "slow slower slowest",
    "high higher highest",
    "deep deeper deepest",
    "large larger largest",
    "small smaller smallest",
    "bright brighter brightest",
    "dark darker darkest",
    "the quick brown fox jumps over the lazy dog",
    "machine learning is a subfield of artificial intelligence",
    "neural networks learn from labelled training data",
    "attention mechanisms are core to transformer models",
    "embeddings map tokens to dense numeric vectors",
    "cosine similarity measures directional agreement",
    "tokenization splits raw text into discrete units",
    "byte pair encoding is a greedy subword algorithm",
]

CORPUS: list[str] = [
    "Machine learning models learn patterns from large datasets.",
    "Python is widely used for data science and automation.",
    "Neural networks are inspired by biological neurons.",
    "Gradient descent is a first-order optimisation algorithm.",
    "Transformers rely on self-attention over the full input sequence.",
    "Random forests are an ensemble of decision trees.",
    "Natural language processing enables text understanding.",
    "Embeddings encode semantic meaning in vector space.",
    "Tokenization is the first step in any NLP pipeline.",
    "Cosine similarity measures the angle between two vectors.",
]


def run_bpe_demo() -> None:
    logger.info("=== BPE Tokenizer Demo ===")
    tok = BPETokenizer()
    tok.fit(TRAIN_CORPUS, num_merges=30)

    test_phrases = ["lower", "newest", "fastest slower", "unknown xray", ""]
    for phrase in test_phrases:
        tokens = tok.encode(phrase)
        decoded = tok.decode(tokens)
        logger.info("  encode(%r) = %s", phrase, tokens)
        if phrase:
            logger.info("  decode(tokens) = %r  [match=%s]", decoded, decoded == phrase)

    # Save and reload
    tok_path = ARTEFACTS / "tokenizer.json"
    tok.save(tok_path)
    tok2 = BPETokenizer.load(tok_path)
    assert tok2.encode("lower") == tok.encode("lower"), "Save/load mismatch"
    logger.info("Save/load verified OK")


def run_search_demo() -> None:
    logger.info("=== Semantic Search Demo ===")
    embedder = CachedEmbedder(StubEmbedder(dim=64))
    engine = SemanticSearchEngine(embedder, batch_size=4)
    engine.build_index(CORPUS)

    queries = [
        "How do language models represent words numerically?",
        "What optimisation algorithms are used to train deep learning?",
        "How is text split before being fed to a model?",
    ]
    for q in queries:
        results = engine.search(q, top_k=3)
        logger.info("\nQuery: %r", q)
        for doc, score in results:
            logger.info("  [%.3f] %s", score, doc)

    # Save and reload
    idx_path = ARTEFACTS / "search_index"
    engine.save_index(idx_path)
    engine2 = SemanticSearchEngine(embedder, batch_size=4)
    engine2.load_index(idx_path)
    r1 = engine.search(queries[0], top_k=1)
    r2 = engine2.search(queries[0], top_k=1)
    assert r1[0][0] == r2[0][0], "Index save/load mismatch"
    logger.info("Index save/load verified OK")


if __name__ == "__main__":
    run_bpe_demo()
    run_search_demo()
```

---

## Additional test — `test_bpe_save_load_preserves_encode_ids`

```python
def test_bpe_save_load_preserves_encode_ids(
    trained_tokenizer: BPETokenizer, tmp_path: Path
) -> None:
    path = tmp_path / "tok.json"
    trained_tokenizer.save(path)
    loaded = BPETokenizer.load(path)
    text = "lower newest"
    assert loaded.encode_ids(text) == trained_tokenizer.encode_ids(text)
```

---

## Additional test — `test_search_top_k_respects_corpus_size`

```python
def test_search_top_k_larger_than_corpus() -> None:
    mock = MagicMock()
    mock.embed.side_effect = lambda texts: _fixed_vectors(texts)
    small_docs = ["a", "b"]
    engine = SemanticSearchEngine(mock)
    engine.build_index(small_docs)
    results = engine.search("query", top_k=100)
    assert len(results) == 2  # capped at corpus size
```

---

## Stretch goal — MMR search

See day-5 solution for full `search_mmr` implementation. The key invariant to test:

```python
def test_mmr_results_are_diverse() -> None:
    """MMR top-2 results should not be more similar to each other
    than a plain top-2 search would be."""
    # Build a corpus where docs 0 and 1 are near-identical,
    # docs 2 and 3 are different.
    # MMR top-2 should prefer docs {0, 2} or {0, 3} over {0, 1}.
    ...
```

---

## Rubric self-assessment guide

Run this before submitting:

```bash
pytest tests/ -v --tb=short        # all tests green?
python demo.py                     # no errors, artefacts/ created?
ruff check src/ tests/             # zero violations?
grep -r "print(" src/ tests/       # should return nothing
grep -r "api_key.*=" src/          # should return nothing (no hardcoded keys)
```
