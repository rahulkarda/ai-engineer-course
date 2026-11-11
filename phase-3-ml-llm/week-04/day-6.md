# Phase 3 · Week 04 · Day 6 (Saturday) — Build: Tiny BPE Tokenizer + Semantic Search

> Objective: Implement a production-quality BPE tokenizer and a semantic search engine, both fully tested.  ·  Time: ~4 hr  ·  Prereqs: [[day-3]], [[day-5]]

## Overview

Today's build integrates the week's concepts into two deliverables that will serve as the Week 4 project:

1. **`bpe_tokenizer`** — a save/load-capable BPE tokenizer that tokenises and decodes arbitrary text.
2. **`semantic_search`** — a corpus-indexed semantic search engine with a mocked embedding provider, tested with `pytest`.

The two components integrate: the tokenizer pre-processes text before embedding; the search engine operates on the resulting vectors.

Work in a fresh directory:

```
week04_project/
├── src/
│   ├── bpe_tokenizer.py
│   └── semantic_search.py
└── tests/
    ├── test_bpe_tokenizer.py
    └── test_semantic_search.py
```

---

## Part 1 — BPE Tokenizer (2 hr)

### Spec

Build a `BPETokenizer` class that supports:

| Method | Signature | Description |
|---|---|---|
| `fit` | `(corpus: list[str], num_merges: int) -> Self` | Learn BPE merge rules |
| `encode` | `(text: str) -> list[str]` | Tokenise text into BPE tokens |
| `encode_ids` | `(text: str) -> list[int]` | Tokenise to integer IDs |
| `decode` | `(tokens: list[str]) -> str` | Reconstruct text from tokens |
| `save` | `(path: Path) -> None` | Persist vocabulary + merge rules (JSON) |
| `load` | `classmethod (path: Path) -> Self` | Restore from file |

Acceptance criteria:
- `decode(encode(text))` round-trips any whitespace-separated ASCII input.
- Unknown words (not in training vocabulary) degrade gracefully to character tokens.
- Saving and loading produces an identical tokeniser.
- All public methods have type hints and docstrings.
- At least 10 unit tests in `tests/test_bpe_tokenizer.py`.

### Reference implementation

```python
"""src/bpe_tokenizer.py — Production-quality BPE tokenizer."""

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Self

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

END = "</w>"
UNK = "<unk>"

Word = tuple[str, ...]
Vocab = dict[Word, int]
MergeRule = tuple[str, str]


def _to_word(surface: str) -> Word:
    """Split surface form into char sequence with end marker."""
    return tuple(list(surface[:-1]) + [surface[-1] + END])


def _get_pairs(vocab: Vocab) -> Counter[MergeRule]:
    counts: Counter[MergeRule] = Counter()
    for word, freq in vocab.items():
        for a, b in zip(word, word[1:]):
            counts[(a, b)] += freq
    return counts


def _apply_merge(vocab: Vocab, pair: MergeRule) -> Vocab:
    merged = "".join(pair)
    new: Vocab = {}
    for word, freq in vocab.items():
        out: list[str] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                out.append(merged)
                i += 2
            else:
                out.append(word[i])
                i += 1
        new[tuple(out)] = freq
    return new


class BPETokenizer:
    """Byte-Pair Encoding tokenizer with save/load support."""

    def __init__(self) -> None:
        self.merges: list[MergeRule] = []
        self._vocab: Vocab = {}
        self._token2id: dict[str, int] = {}
        self._id2token: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, corpus: list[str], num_merges: int = 50) -> Self:
        """Learn BPE merge rules from corpus."""
        word_freqs: Counter[str] = Counter()
        for sentence in corpus:
            for raw_word in sentence.lower().split():
                word = re.sub(r"[^a-z']+", "", raw_word)
                if word:
                    word_freqs[word] += 1

        self._vocab = {_to_word(w): f for w, f in word_freqs.items()}

        for step in range(num_merges):
            pairs = _get_pairs(self._vocab)
            if not pairs:
                break
            best = pairs.most_common(1)[0][0]
            self.merges.append(best)
            self._vocab = _apply_merge(self._vocab, best)

        # Build token ↔ id mappings
        all_symbols: set[str] = {sym for word in self._vocab for sym in word}
        all_symbols.add(UNK)
        self._token2id = {t: i for i, t in enumerate(sorted(all_symbols))}
        self._id2token = {i: t for t, i in self._token2id.items()}

        logger.info(
            "BPE fit: %d merges, %d symbol types", len(self.merges), len(self._token2id)
        )
        return self

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _tokenise_word(self, surface: str) -> list[str]:
        symbols: list[str] = list(surface[:-1]) + [surface[-1] + END]
        for pair in self.merges:
            out: list[str] = []
            i = 0
            while i < len(symbols):
                if (
                    i < len(symbols) - 1
                    and symbols[i] == pair[0]
                    and symbols[i + 1] == pair[1]
                ):
                    out.append("".join(pair))
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            symbols = out
        return symbols

    def encode(self, text: str) -> list[str]:
        """Return BPE token strings for text."""
        tokens: list[str] = []
        for raw_word in text.lower().split():
            word = re.sub(r"[^a-z']+", "", raw_word)
            if word:
                tokens.extend(self._tokenise_word(word))
        return tokens

    def encode_ids(self, text: str) -> list[int]:
        """Return BPE token IDs for text."""
        unk_id = self._token2id.get(UNK, 0)
        return [self._token2id.get(t, unk_id) for t in self.encode(text)]

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, tokens: list[str]) -> str:
        """Reconstruct text from BPE tokens."""
        words: list[str] = []
        current: list[str] = []
        for tok in tokens:
            if tok.endswith(END):
                current.append(tok[: -len(END)])
                words.append("".join(current))
                current = []
            else:
                current.append(tok)
        if current:
            words.append("".join(current))
        return " ".join(words)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Serialise tokenizer to JSON."""
        path = Path(path)
        data = {
            "merges": self.merges,
            "token2id": self._token2id,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Saved tokenizer to %s", path)

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        """Load tokenizer from JSON."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        tok = cls()
        tok.merges = [tuple(pair) for pair in data["merges"]]
        tok._token2id = data["token2id"]
        tok._id2token = {int(i): t for t, i in tok._token2id.items()}
        logger.info("Loaded tokenizer from %s (%d tokens)", path, len(tok._token2id))
        return tok

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self._token2id)
```

---

## Part 2 — Semantic Search Engine (2 hr)

### Spec

Build a `SemanticSearchEngine` that:
- Accepts any `EmbeddingProvider` (Protocol).
- Indexes a corpus of strings.
- Answers queries with ranked `(document, score)` pairs.
- Supports `save_index` / `load_index` to avoid re-embedding.

Acceptance criteria:
- All embedding calls go through the provider — engine never imports `openai` or `sentence_transformers` directly.
- Tests mock the provider and verify search ranking logic.
- At least 8 unit tests.

### Reference implementation

```python
"""src/semantic_search.py — Embedding-based semantic search engine."""

import logging
from pathlib import Path
from typing import Protocol

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class SemanticSearchEngine:
    """Corpus-indexed semantic search using cosine similarity."""

    def __init__(self, embedder: EmbeddingProvider, batch_size: int = 64) -> None:
        self._embedder = embedder
        self._batch_size = batch_size
        self._docs: list[str] = []
        self._index: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def build_index(self, documents: list[str]) -> None:
        """Embed all documents and store L2-normalised index."""
        self._docs = list(documents)
        vectors: list[np.ndarray] = []
        for start in range(0, len(documents), self._batch_size):
            batch = documents[start : start + self._batch_size]
            vecs = self._embedder.embed(batch)
            vectors.append(self._l2_normalise(vecs))
        self._index = np.vstack(vectors)
        logger.info("Index built: %d docs × %d dims", *self._index.shape)

    @staticmethod
    def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms == 0, 1.0, norms)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Return top-k (document, cosine_score) pairs."""
        if self._index is None:
            raise RuntimeError("Call build_index before search.")
        q_vec = self._l2_normalise(self._embedder.embed([query]))[0]
        scores: np.ndarray = self._index @ q_vec
        top_k = min(top_k, len(self._docs))
        idx = scores.argsort()[::-1][:top_k]
        return [(self._docs[i], float(scores[i])) for i in idx]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_index(self, path: Path) -> None:
        """Save document list and embedding matrix."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "index.npy", self._index)
        (path / "docs.txt").write_text("\n".join(self._docs), encoding="utf-8")
        logger.info("Index saved to %s", path)

    def load_index(self, path: Path) -> None:
        """Load document list and embedding matrix."""
        path = Path(path)
        self._index = np.load(path / "index.npy")
        self._docs = (path / "docs.txt").read_text(encoding="utf-8").splitlines()
        logger.info("Index loaded from %s: %d docs", path, len(self._docs))
```

---

## Part 3 — Tests

### `tests/test_bpe_tokenizer.py` (scaffold — complete it)

```python
"""Tests for BPETokenizer."""
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
    # Punctuation is stripped; no alphabetic chars remain
    assert trained_tokenizer.encode("!!!???...") == []


def test_num_merges_limits(trained_tokenizer: BPETokenizer) -> None:
    assert len(trained_tokenizer.merges) <= 20
```

### `tests/test_semantic_search.py` (scaffold — complete it)

```python
"""Tests for SemanticSearchEngine (all embedding calls mocked)."""
import numpy as np
import pytest
from unittest.mock import MagicMock
from src.semantic_search import SemanticSearchEngine


def make_engine(dim: int = 8) -> tuple[SemanticSearchEngine, MagicMock]:
    mock_embedder = MagicMock()
    engine = SemanticSearchEngine(mock_embedder)
    return engine, mock_embedder


DOCS = ["cats are furry", "dogs bark loudly", "fish swim silently"]


def _fixed_vectors(texts: list[str], dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(0)
    mat = rng.standard_normal((len(texts), dim)).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / norms


def test_build_index_calls_embed(make_engine=make_engine) -> None:
    engine, mock = make_engine()
    mock.embed.return_value = _fixed_vectors(DOCS)
    engine.build_index(DOCS)
    mock.embed.assert_called_once_with(DOCS)


def test_search_returns_top_k(make_engine=make_engine) -> None:
    engine, mock = make_engine()
    all_texts = DOCS + ["query text"]
    mock.embed.side_effect = lambda texts: _fixed_vectors(texts)
    engine.build_index(DOCS)
    results = engine.search("query text", top_k=2)
    assert len(results) == 2


def test_search_scores_between_minus_one_and_one() -> None:
    engine, mock = make_engine()
    mock.embed.side_effect = lambda texts: _fixed_vectors(texts)
    engine.build_index(DOCS)
    results = engine.search("animals", top_k=3)
    for _, score in results:
        assert -1.01 <= score <= 1.01


def test_search_without_index_raises() -> None:
    engine, _ = make_engine()
    with pytest.raises(RuntimeError):
        engine.search("anything")


def test_save_load_index(tmp_path) -> None:
    engine, mock = make_engine()
    mock.embed.return_value = _fixed_vectors(DOCS)
    engine.build_index(DOCS)
    engine.save_index(tmp_path / "idx")
    engine2, mock2 = make_engine()
    mock2.embed.side_effect = lambda texts: _fixed_vectors(texts)
    engine2.load_index(tmp_path / "idx")
    r1 = engine.search("fish", top_k=1)
    r2 = engine2.search("fish", top_k=1)
    assert r1[0][0] == r2[0][0]


def test_batch_size_respected() -> None:
    engine, mock = make_engine()
    big_corpus = [f"document {i}" for i in range(10)]
    mock.embed.side_effect = lambda texts: _fixed_vectors(texts)
    engine = SemanticSearchEngine(mock, batch_size=3)
    engine.build_index(big_corpus)
    # embed should have been called ceil(10/3) = 4 times
    assert mock.embed.call_count == 4
```

---

## Checklist

- [ ] `BPETokenizer.fit` + `encode` + `decode` pass all 10 tests.
- [ ] `SemanticSearchEngine` passes all 6 tests with mocked embedder.
- [ ] `decode(encode(text)) == text` for multi-word strings.
- [ ] Save/load round-trips for both classes.
- [ ] No test makes a real network call.
- [ ] All code has type hints.
- [ ] `logging` used, no bare `print`.

## Supplementary links
- [Karpathy's minBPE walkthrough](https://github.com/karpathy/minbpe)
- [pytest fixtures docs](https://docs.pytest.org/en/stable/reference/fixtures.html)
- [numpy save/load](https://numpy.org/doc/stable/reference/generated/numpy.save.html)
