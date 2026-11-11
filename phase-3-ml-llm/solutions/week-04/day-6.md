# Solutions — Phase 3 · Week 04 · Day 6 — BPE Tokenizer + Semantic Search

## Overview

Day 6 is a full build day: the deliverable IS the Week 4 project. The complete reference
implementations for both components are embedded directly in the lesson file (`week-04/day-6.md`).
This solutions file provides supplementary notes, test completions, and common-mistake guidance.

---

## Part 1 — BPE Tokenizer

The complete `BPETokenizer` class (with `fit`, `encode`, `encode_ids`, `decode`, `save`, `load`)
is given verbatim in `week-04/day-6.md`. Study it; do not copy without understanding.

### Complete `tests/test_bpe_tokenizer.py`

All 10 required tests from `day-6.md` plus two extras:

```python
"""tests/test_bpe_tokenizer.py — Full test suite for BPETokenizer."""
import pytest
from pathlib import Path

from src.bpe_tokenizer import BPETokenizer

CORPUS = [
    "low lower lowest",
    "newer newest news",
    "old older oldest",
    "new new newer newest",
    "low low lower lowest",
]


@pytest.fixture
def trained_tokenizer() -> BPETokenizer:
    tok = BPETokenizer()
    tok.fit(CORPUS, num_merges=20)
    return tok


def test_encode_returns_nonempty(trained_tokenizer: BPETokenizer) -> None:
    assert trained_tokenizer.encode("low") != []


def test_encode_unknown_word_does_not_crash(trained_tokenizer: BPETokenizer) -> None:
    tokens = trained_tokenizer.encode("zyxwvuts")
    assert len(tokens) > 0


def test_decode_reconstructs_word(trained_tokenizer: BPETokenizer) -> None:
    original = "low"
    tokens = trained_tokenizer.encode(original)
    decoded = trained_tokenizer.decode(tokens)
    assert decoded == original


def test_round_trip(trained_tokenizer: BPETokenizer) -> None:
    text = "newer low older"
    decoded = trained_tokenizer.decode(trained_tokenizer.encode(text))
    assert decoded == text


def test_encode_ids_length_matches_encode(trained_tokenizer: BPETokenizer) -> None:
    tokens = trained_tokenizer.encode("newest lower")
    ids = trained_tokenizer.encode_ids("newest lower")
    assert len(tokens) == len(ids)


def test_vocab_size_positive(trained_tokenizer: BPETokenizer) -> None:
    assert trained_tokenizer.vocab_size > 0


def test_save_and_load(trained_tokenizer: BPETokenizer, tmp_path: Path) -> None:
    save_path = tmp_path / "tokenizer.json"
    trained_tokenizer.save(save_path)
    loaded = BPETokenizer.load(save_path)
    assert loaded.merges == trained_tokenizer.merges
    assert loaded.encode("lower") == trained_tokenizer.encode("lower")


def test_empty_string_returns_empty(trained_tokenizer: BPETokenizer) -> None:
    assert trained_tokenizer.encode("") == []


def test_punctuation_only_returns_empty(trained_tokenizer: BPETokenizer) -> None:
    assert trained_tokenizer.encode("!!!???...") == []


def test_num_merges_limits(trained_tokenizer: BPETokenizer) -> None:
    assert len(trained_tokenizer.merges) <= 20


# --- Extra tests ---------------------------------------------------------

def test_save_load_preserves_encode_ids(
    trained_tokenizer: BPETokenizer, tmp_path: Path
) -> None:
    path = tmp_path / "tok.json"
    trained_tokenizer.save(path)
    loaded = BPETokenizer.load(path)
    text = "lower newest"
    assert loaded.encode_ids(text) == trained_tokenizer.encode_ids(text)


def test_more_merges_reduces_token_count() -> None:
    corpus = ["low lower lowest" * 10]
    tok_few  = BPETokenizer().fit(corpus, num_merges=5)
    tok_many = BPETokenizer().fit(corpus, num_merges=50)
    # More merges → longer merged tokens → fewer tokens per word
    n_few  = len(tok_few.encode("lower lowest"))
    n_many = len(tok_many.encode("lower lowest"))
    assert n_many <= n_few
```

---

## Part 2 — Semantic Search Engine

The complete `SemanticSearchEngine` class (with `EmbeddingProvider` Protocol, `build_index`,
`search`, `save_index`, `load_index`) is given verbatim in `week-04/day-6.md`.

### Complete `tests/test_semantic_search.py`

All scaffold tests from `day-6.md` plus one additional test:

```python
"""tests/test_semantic_search.py — Tests for SemanticSearchEngine."""
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.semantic_search import SemanticSearchEngine

DOCS = ["cats are furry", "dogs bark loudly", "fish swim silently"]
DIM  = 8


def _fixed_vectors(texts: list[str], dim: int = DIM) -> np.ndarray:
    """Deterministic fake embeddings keyed by content hash."""
    rng = np.random.default_rng(0)
    mat = rng.standard_normal((len(texts), dim)).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / norms


def _make_engine(dim: int = DIM) -> tuple[SemanticSearchEngine, MagicMock]:
    mock_embedder = MagicMock()
    engine = SemanticSearchEngine(mock_embedder)
    return engine, mock_embedder


def test_build_index_calls_embed() -> None:
    engine, mock = _make_engine()
    mock.embed.return_value = _fixed_vectors(DOCS)
    engine.build_index(DOCS)
    mock.embed.assert_called_once_with(DOCS)


def test_search_returns_top_k() -> None:
    engine, mock = _make_engine()
    mock.embed.side_effect = lambda texts: _fixed_vectors(texts)
    engine.build_index(DOCS)
    results = engine.search("query text", top_k=2)
    assert len(results) == 2


def test_search_scores_between_minus_one_and_one() -> None:
    engine, mock = _make_engine()
    mock.embed.side_effect = lambda texts: _fixed_vectors(texts)
    engine.build_index(DOCS)
    results = engine.search("animals", top_k=3)
    for _, score in results:
        assert -1.01 <= score <= 1.01


def test_search_without_index_raises() -> None:
    engine, _ = _make_engine()
    with pytest.raises(RuntimeError):
        engine.search("anything")


def test_save_load_index(tmp_path: Path) -> None:
    engine, mock = _make_engine()
    mock.embed.return_value = _fixed_vectors(DOCS)
    engine.build_index(DOCS)
    engine.save_index(tmp_path / "idx")
    engine2, mock2 = _make_engine()
    mock2.embed.side_effect = lambda texts: _fixed_vectors(texts)
    engine2.load_index(tmp_path / "idx")
    r1 = engine.search("fish", top_k=1)
    r2 = engine2.search("fish", top_k=1)
    assert r1[0][0] == r2[0][0]


def test_batch_size_respected() -> None:
    mock_embedder = MagicMock()
    big_corpus = [f"document {i}" for i in range(10)]
    mock_embedder.embed.side_effect = lambda texts: _fixed_vectors(texts)
    engine = SemanticSearchEngine(mock_embedder, batch_size=3)
    engine.build_index(big_corpus)
    # ceil(10 / 3) = 4 batches
    assert mock_embedder.embed.call_count == 4


def test_search_top_k_larger_than_corpus() -> None:
    engine, mock = _make_engine()
    mock.embed.side_effect = lambda texts: _fixed_vectors(texts)
    small_docs = ["a", "b"]
    engine.build_index(small_docs)
    results = engine.search("query", top_k=100)
    assert len(results) == 2  # capped at corpus size
```

---

## Common mistakes

**`_apply_merge` loops over the wrong index range**
When applying a BPE merge in a for-loop, ensure `i` skips by 2 after a merge — else a merged
symbol can be partially merged again on the same pass.

**Encoding OOV chars**
Unknown characters not seen during `fit` are not in `_token2id`. The `encode_ids` method should
fall back to the `<unk>` ID; the `encode` method returns the raw character symbol (graceful
degradation to char tokens).

**L2 normalisation before storing**
Store unit-norm vectors so that dot product equals cosine similarity. If you skip normalisation,
`scores = index @ q_vec` computes unnormalised inner products — results are biased toward long
documents.

**Forgetting to re-normalise query**
Both corpus vectors AND query vectors must be L2-normalised before the dot product. Missing one
side breaks cosine similarity entirely.

---

## Self-check

```bash
pytest tests/ -v --tb=short         # all tests green?
python -m src.bpe_tokenizer         # (if you add a __main__ demo)
ruff check src/ tests/              # zero violations?
grep -r "print(" src/               # should return nothing
grep -r "api_key" src/              # should return nothing
```
