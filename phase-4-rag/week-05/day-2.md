# Phase 4 · Week 05 · Day 2 — Cross-encoder Reranking in Practice

> Objective: Build a production-ready, mockable cross-encoder reranker with error handling, scoring normalisation, and pytest coverage.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

## Concept

Day 1 covered the conceptual gap between bi-encoders and cross-encoders. Today we implement a *reusable reranker component* you can drop into any pipeline. Three engineering concerns dominate:

1. **Model loading is expensive.** Load once, reuse many times. Wrap in a class.
2. **Raw cross-encoder logits are unbounded.** Normalise to [0, 1] with sigmoid so downstream code can treat scores as pseudo-probabilities.
3. **Tests must not hit the network.** Mock `CrossEncoder.predict` so CI passes without downloading models.

### Score normalisation

`cross-encoder/ms-marco-*` models output raw logits. A logit of 5.0 vs −2.0 is meaningful but hard to interpret. Sigmoid maps any real value to (0, 1):

```
sigmoid(x) = 1 / (1 + exp(-x))
```

This preserves ranking order while giving interpretable scores.

### Batch sizing

`CrossEncoder.predict` accepts a list of pairs. Batch all candidates in a single call — it is faster than calling `predict` once per pair because the model processes a padded batch in one GPU/CPU kernel launch.

### Threshold filtering

After reranking you often want to drop anything below a relevance threshold (e.g., 0.3) rather than always returning a fixed count. Expose both `top_n` and `score_threshold` as optional parameters.

## How it works internally

**`CrossEncoder.predict` internals (sentence-transformers)**

Internally `sentence_transformers.CrossEncoder` wraps a `transformers.AutoModelForSequenceClassification`. When you call `predict(pairs)`:

1. Pairs are tokenised with `AutoTokenizer`, padded to a common length.
2. A single forward pass produces logits of shape `(batch_size, num_labels)`.
3. For MS-MARCO models `num_labels=1`, so logits are squeezed to shape `(batch_size,)`.
4. The library optionally applies `activation_fct` (default none for MARCO models — we apply sigmoid ourselves).

Key config knobs:
- `max_length` — truncate long documents (default 512 tokens).
- `batch_size` — controls memory vs throughput trade-off.
- `show_progress_bar` — disable in production.

## Worked example

```python
"""
day2_reranker.py

Production-ready cross-encoder reranker with:
- lazy model loading
- sigmoid score normalisation
- configurable top_n and score_threshold
- full type hints
"""

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RerankResult:
    """A single reranked document with its normalised relevance score."""
    original_index: int   # position in the input candidates list
    score: float          # sigmoid-normalised relevance in [0, 1]
    text: str


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------
class CrossEncoderReranker:
    """
    Wraps a sentence-transformers CrossEncoder with lazy loading,
    sigmoid normalisation, and configurable filtering.

    Parameters
    ----------
    model_name:
        HuggingFace model id. Defaults to a fast MS-MARCO MiniLM model.
    max_length:
        Token truncation limit passed to the underlying model.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_length: int = 512,
    ) -> None:
        self._model_name = model_name
        self._max_length = max_length
        self._model: Optional[object] = None  # lazy

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _load(self) -> None:
        """Load the model on first use."""
        if self._model is None:
            from sentence_transformers import CrossEncoder  # local import → mockable
            log.info("Loading cross-encoder model: %s", self._model_name)
            self._model = CrossEncoder(
                self._model_name,
                max_length=self._max_length,
            )

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def rerank(
        self,
        query: str,
        candidates: list[str],
        top_n: int | None = None,
        score_threshold: float = 0.0,
    ) -> list[RerankResult]:
        """
        Rerank *candidates* against *query*.

        Parameters
        ----------
        query:
            The search query.
        candidates:
            List of document strings to rerank.
        top_n:
            Return at most this many results. None = return all above threshold.
        score_threshold:
            Drop results whose normalised score is below this value.

        Returns
        -------
        List of RerankResult ordered from most to least relevant.
        """
        if not candidates:
            log.warning("rerank called with empty candidates list")
            return []

        self._load()

        pairs = [[query, doc] for doc in candidates]
        log.debug("Scoring %d pairs with cross-encoder", len(pairs))
        raw: np.ndarray = np.asarray(
            self._model.predict(pairs, show_progress_bar=False)
        )
        scores = self._sigmoid(raw)

        results = [
            RerankResult(original_index=i, score=float(scores[i]), text=doc)
            for i, doc in enumerate(candidates)
            if float(scores[i]) >= score_threshold
        ]
        results.sort(key=lambda r: r.score, reverse=True)

        if top_n is not None:
            results = results[:top_n]

        log.info(
            "Reranked %d candidates → %d results (threshold=%.2f)",
            len(candidates),
            len(results),
            score_threshold,
        )
        return results
```

### Test file

```python
"""
tests/unit/test_reranker.py

Unit tests for CrossEncoderReranker.
All CrossEncoder calls are mocked — no network, no model download.
"""

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from day2_reranker import CrossEncoderReranker, RerankResult


@pytest.fixture()
def reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()


def make_mock_model(raw_scores: list[float]) -> MagicMock:
    mock = MagicMock()
    mock.predict.return_value = np.array(raw_scores)
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestRerankOrdering:
    def test_returns_sorted_by_score_descending(self, reranker):
        # Logits: doc 2 has highest logit → should rank first
        raw = [-2.0, 5.0, 1.0]
        candidates = ["doc A", "doc B", "doc C"]
        with patch("day2_reranker.CrossEncoder", return_value=make_mock_model(raw)):
            results = reranker.rerank("query", candidates)
        assert [r.original_index for r in results] == [1, 2, 0]

    def test_original_index_preserved(self, reranker):
        raw = [3.0, 1.0]
        candidates = ["first", "second"]
        with patch("day2_reranker.CrossEncoder", return_value=make_mock_model(raw)):
            results = reranker.rerank("q", candidates)
        assert results[0].original_index == 0
        assert results[0].text == "first"


class TestTopN:
    def test_top_n_limits_results(self, reranker):
        raw = [4.0, 3.0, 2.0, 1.0]
        candidates = ["a", "b", "c", "d"]
        with patch("day2_reranker.CrossEncoder", return_value=make_mock_model(raw)):
            results = reranker.rerank("q", candidates, top_n=2)
        assert len(results) == 2

    def test_top_n_none_returns_all(self, reranker):
        raw = [1.0, 2.0]
        candidates = ["x", "y"]
        with patch("day2_reranker.CrossEncoder", return_value=make_mock_model(raw)):
            results = reranker.rerank("q", candidates, top_n=None)
        assert len(results) == 2


class TestThreshold:
    def test_threshold_filters_low_scores(self, reranker):
        # sigmoid(-5) ≈ 0.0067 — below 0.3 threshold
        raw = [5.0, -5.0]
        candidates = ["relevant", "irrelevant"]
        with patch("day2_reranker.CrossEncoder", return_value=make_mock_model(raw)):
            results = reranker.rerank("q", candidates, score_threshold=0.3)
        assert len(results) == 1
        assert results[0].text == "relevant"


class TestEdgeCases:
    def test_empty_candidates_returns_empty_list(self, reranker):
        results = reranker.rerank("query", [])
        assert results == []

    def test_scores_in_zero_one_range(self, reranker):
        raw = [-10.0, 0.0, 10.0]
        candidates = ["a", "b", "c"]
        with patch("day2_reranker.CrossEncoder", return_value=make_mock_model(raw)):
            results = reranker.rerank("q", candidates)
        for r in results:
            assert 0.0 <= r.score <= 1.0
```

## Exercises

1. Add a `batch_size: int = 32` parameter to `CrossEncoderReranker.__init__` and pass it through to `model.predict`. Write a test that verifies the value is forwarded.
2. Implement a `rerank_with_metadata` method that accepts `list[dict]` (each dict has a `"text"` key plus arbitrary fields) and returns a new list with an added `"rerank_score"` field, preserving all original fields.
3. The current `_load` is not thread-safe. Sketch (in comments) how you would add a `threading.Lock` to make it safe for concurrent calls. What is the double-checked locking pattern?

(Solutions: ../solutions/week-05/day-2.md)

## Self-check

- [ ] I can explain why lazy model loading is preferable for rerankers.
- [ ] I can implement sigmoid normalisation without a library.
- [ ] I can write pytest mocks for sentence-transformers without network access.
- [ ] I understand how `CrossEncoder.predict` batches inputs internally.
- [ ] I can add threshold and top-N filtering to a reranker.

## Supplementary links

- [sentence-transformers CrossEncoder API](https://www.sbert.net/docs/package_reference/cross_encoder.html)
- [MS-MARCO dataset and models](https://microsoft.github.io/msmarco/)
- [pytest `unittest.mock` docs](https://docs.pytest.org/en/stable/how-to/monkeypatch.html)
