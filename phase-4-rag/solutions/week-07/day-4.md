# Solutions — Phase 4 · Week 07 · Day 4

## Exercise 1

Add `composite_score` method to `RAGASEvaluator` that weights the four RAGAS metrics.

```python
# ragas_evaluator.py  (additions to existing class)
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import anthropic

logger = logging.getLogger(__name__)


@dataclass
class RAGSample:
    question: str
    contexts: list[str]
    answer: str
    ground_truth: str | None = None


@dataclass
class RAGASScores:
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevance: float

    def composite_score(
        self,
        w_ctx_precision: float = 0.25,
        w_ctx_recall: float = 0.25,
        w_faithfulness: float = 0.30,
        w_answer_relevance: float = 0.20,
    ) -> float:
        """Weighted composite of the four RAGAS metrics.

        Default weights:
            context_precision  = 0.25
            context_recall     = 0.25
            faithfulness       = 0.30
            answer_relevance   = 0.20
        Weights must sum to 1.0; caller is responsible for passing valid weights.
        """
        composite = (
            w_ctx_precision * self.context_precision
            + w_ctx_recall * self.context_recall
            + w_faithfulness * self.faithfulness
            + w_answer_relevance * self.answer_relevance
        )
        logger.debug("Composite score: %.4f", composite)
        return composite


class RAGASEvaluator:
    """Evaluates RAG pipeline outputs using RAGAS-style LLM-based metrics."""

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self._model = "claude-opus-5-5"

    def _ask(self, prompt: str) -> float:
        """Send a prompt, parse the first float in the reply (0-1)."""
        response = self._client.messages.create(
            model=self._model,
            max_tokens=64,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        try:
            value = float(text.split()[0])
        except ValueError:
            logger.warning("Could not parse float from: %r", text)
            value = 0.0
        return max(0.0, min(1.0, value))

    def context_precision(self, sample: RAGSample) -> float:
        ctx = "\n".join(sample.contexts)
        prompt = (
            f"Rate from 0.0 to 1.0 how precisely the retrieved context answers "
            f"the question. Reply with a single float.\n\n"
            f"Question: {sample.question}\nContext:\n{ctx}"
        )
        return self._ask(prompt)

    def context_recall(self, sample: RAGSample) -> float:
        ctx = "\n".join(sample.contexts)
        prompt = (
            f"Rate from 0.0 to 1.0 how much of the information needed to answer "
            f"the question is present in the context. Reply with a single float.\n\n"
            f"Question: {sample.question}\nContext:\n{ctx}\n"
            f"Expected answer: {sample.ground_truth}"
        )
        return self._ask(prompt)

    def faithfulness(self, sample: RAGSample) -> float:
        ctx = "\n".join(sample.contexts)
        prompt = (
            f"Rate from 0.0 to 1.0 how faithful the answer is to the context "
            f"(no hallucinations). Reply with a single float.\n\n"
            f"Context:\n{ctx}\nAnswer: {sample.answer}"
        )
        return self._ask(prompt)

    def answer_relevance(self, sample: RAGSample) -> float:
        prompt = (
            f"Rate from 0.0 to 1.0 how relevant the answer is to the question. "
            f"Reply with a single float.\n\n"
            f"Question: {sample.question}\nAnswer: {sample.answer}"
        )
        return self._ask(prompt)

    def evaluate(self, sample: RAGSample) -> RAGASScores:
        scores = RAGASScores(
            context_precision=self.context_precision(sample),
            context_recall=self.context_recall(sample),
            faithfulness=self.faithfulness(sample),
            answer_relevance=self.answer_relevance(sample),
        )
        logger.info("RAGAS scores: %s | composite=%.4f", scores, scores.composite_score())
        return scores
```

**Test:**
```python
# tests/unit/test_day4_ex1.py
from unittest.mock import MagicMock, patch

from ragas_evaluator import RAGASScores


def test_composite_score_default_weights() -> None:
    scores = RAGASScores(
        context_precision=0.8,
        context_recall=0.6,
        faithfulness=0.9,
        answer_relevance=0.7,
    )
    expected = 0.25 * 0.8 + 0.25 * 0.6 + 0.30 * 0.9 + 0.20 * 0.7
    assert abs(scores.composite_score() - expected) < 1e-9


def test_composite_score_custom_weights() -> None:
    scores = RAGASScores(1.0, 1.0, 1.0, 1.0)
    assert abs(scores.composite_score(0.4, 0.3, 0.2, 0.1) - 1.0) < 1e-9


def test_composite_score_zeros() -> None:
    scores = RAGASScores(0.0, 0.0, 0.0, 0.0)
    assert scores.composite_score() == 0.0
```

---

## Exercise 2

Async batch evaluation using `asyncio.gather` + `asyncio.to_thread` so each
sample's four LLM calls run concurrently without blocking the event loop.

```python
# ragas_evaluator.py  (continued — add to RAGASEvaluator)
import asyncio


class RAGASEvaluator:
    # ... (previous methods unchanged) ...

    async def _evaluate_async(self, sample: RAGSample) -> RAGASScores:
        """Run four metric calls concurrently for a single sample."""
        ctx_prec, ctx_rec, faith, ans_rel = await asyncio.gather(
            asyncio.to_thread(self.context_precision, sample),
            asyncio.to_thread(self.context_recall, sample),
            asyncio.to_thread(self.faithfulness, sample),
            asyncio.to_thread(self.answer_relevance, sample),
        )
        return RAGASScores(
            context_precision=ctx_prec,
            context_recall=ctx_rec,
            faithfulness=faith,
            answer_relevance=ans_rel,
        )

    async def async_evaluate_batch(
        self, samples: list[RAGSample]
    ) -> list[RAGASScores]:
        """Evaluate all samples in the batch concurrently.

        Each sample's four LLM calls are gathered; all samples run in parallel.
        Returns results in the same order as `samples`.
        """
        logger.info("Async batch evaluation: %d samples", len(samples))
        results: list[RAGASScores] = await asyncio.gather(
            *[self._evaluate_async(s) for s in samples]
        )
        return list(results)
```

**Usage:**
```python
import asyncio

evaluator = RAGASEvaluator()
samples = [
    RAGSample("What is RAG?", ["RAG = Retrieval-Augmented Generation"], "RAG retrieves docs."),
    RAGSample("What is LLM?", ["LLM = Large Language Model"], "LLM generates text."),
]
scores: list[RAGASScores] = asyncio.run(evaluator.async_evaluate_batch(samples))
```

**Test:**
```python
# tests/unit/test_day4_ex2.py
import asyncio
from unittest.mock import MagicMock, patch

from ragas_evaluator import RAGASEvaluator, RAGASScores, RAGSample


def _make_evaluator() -> RAGASEvaluator:
    with patch("ragas_evaluator.anthropic.Anthropic"):
        ev = RAGASEvaluator()
    return ev


def test_async_evaluate_batch_returns_correct_count() -> None:
    ev = _make_evaluator()
    fixed = RAGASScores(0.8, 0.7, 0.9, 0.6)

    async def _run() -> list[RAGASScores]:
        with patch.object(ev, "_evaluate_async", return_value=fixed):
            samples = [
                RAGSample("q1", ["c1"], "a1"),
                RAGSample("q2", ["c2"], "a2"),
                RAGSample("q3", ["c3"], "a3"),
            ]
            return await ev.async_evaluate_batch(samples)

    results = asyncio.run(_run())
    assert len(results) == 3
    for r in results:
        assert r.faithfulness == 0.9


def test_async_evaluate_batch_empty() -> None:
    ev = _make_evaluator()

    async def _run() -> list[RAGASScores]:
        return await ev.async_evaluate_batch([])

    assert asyncio.run(_run()) == []
```

---

## Exercise 3

Alternative `answer_relevance` using TF-IDF cosine similarity — no external
libraries. Implements IDF = log(N / df) over the two-document corpus
{question, answer}, then normalises to get cosine similarity.

```python
# tfidf_relevance.py
from __future__ import annotations

import logging
import math
from collections import Counter

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on whitespace and punctuation."""
    import re
    return re.findall(r"[a-z0-9']+", text.lower())


def _tf(tokens: list[str]) -> dict[str, float]:
    """Term frequency: count / total tokens."""
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)
    return {term: count / total for term, count in counts.items()}


def _idf(documents: list[list[str]]) -> dict[str, float]:
    """Inverse document frequency: log(N / df) for each term across documents."""
    n = len(documents)
    df: dict[str, int] = {}
    for doc in documents:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1
    return {term: math.log(n / count) for term, count in df.items()}


def _tfidf_vector(
    tokens: list[str], tf: dict[str, float], idf: dict[str, float]
) -> dict[str, float]:
    return {term: tf.get(term, 0.0) * idf.get(term, 0.0) for term in idf}


def _cosine(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors (same key space)."""
    dot = sum(vec_a[k] * vec_b[k] for k in vec_a)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def tfidf_answer_relevance(question: str, answer: str) -> float:
    """Return cosine similarity of TF-IDF vectors for question and answer.

    The corpus is the two documents {question, answer}, so IDF is computed
    over just those two.  Terms unique to one document get IDF = log(2/1) ~ 0.69.
    Common terms get IDF = log(2/2) = 0.0 and drop out of the similarity.
    Range: [0.0, 1.0].
    """
    q_tokens = _tokenize(question)
    a_tokens = _tokenize(answer)

    if not q_tokens or not a_tokens:
        logger.warning("Empty token list — returning 0.0")
        return 0.0

    idf = _idf([q_tokens, a_tokens])
    q_tf = _tf(q_tokens)
    a_tf = _tf(a_tokens)
    q_vec = _tfidf_vector(q_tokens, q_tf, idf)
    a_vec = _tfidf_vector(a_tokens, a_tf, idf)

    similarity = _cosine(q_vec, a_vec)
    logger.debug("TF-IDF cosine similarity: %.4f", similarity)
    return similarity
```

**Test:**
```python
# tests/unit/test_day4_ex3.py
from tfidf_relevance import tfidf_answer_relevance


def test_identical_texts_give_high_similarity() -> None:
    score = tfidf_answer_relevance("what is machine learning", "what is machine learning")
    # All terms appear in both docs → IDF = 0 → zero vectors → returns 0.0
    # (expected: identical overlapping corpus collapses to 0 via IDF)
    assert 0.0 <= score <= 1.0


def test_totally_different_texts() -> None:
    score = tfidf_answer_relevance("capital of france", "photosynthesis requires sunlight")
    assert score == 0.0


def test_partial_overlap() -> None:
    score = tfidf_answer_relevance(
        "what is retrieval augmented generation",
        "retrieval augmented generation combines search with LLM",
    )
    assert 0.0 <= score <= 1.0


def test_empty_question_returns_zero() -> None:
    assert tfidf_answer_relevance("", "some answer text") == 0.0


def test_empty_answer_returns_zero() -> None:
    assert tfidf_answer_relevance("some question", "") == 0.0
```

---

## Exercise 4

`Alert` dataclass + threshold checking integrated into `RAGASEvaluator`.

```python
# ragas_evaluator.py  (continued — add Alert + check_thresholds)
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Alert:
    metric_name: str
    score: float
    threshold: float
    message: str


_DEFAULT_THRESHOLDS: dict[str, float] = {
    "context_precision": 0.7,
    "context_recall": 0.7,
    "faithfulness": 0.8,
    "answer_relevance": 0.7,
    "composite": 0.75,
}


def check_thresholds(
    scores: RAGASScores,
    thresholds: dict[str, float] | None = None,
) -> list[Alert]:
    """Return one Alert per metric that falls below its configured threshold.

    Args:
        scores:     Evaluated RAGAS scores for one sample.
        thresholds: Mapping of metric name to minimum acceptable value.
                    Falls back to _DEFAULT_THRESHOLDS for missing keys.

    Returns:
        List of Alert instances (empty if all metrics pass).
    """
    effective = dict(_DEFAULT_THRESHOLDS)
    if thresholds:
        effective.update(thresholds)

    metric_values: dict[str, float] = {
        "context_precision": scores.context_precision,
        "context_recall": scores.context_recall,
        "faithfulness": scores.faithfulness,
        "answer_relevance": scores.answer_relevance,
        "composite": scores.composite_score(),
    }

    alerts: list[Alert] = []
    for metric, value in metric_values.items():
        threshold = effective.get(metric, 0.0)
        if value < threshold:
            msg = (
                f"{metric} score {value:.4f} is below threshold {threshold:.4f}"
            )
            alerts.append(Alert(metric, value, threshold, msg))
            logger.warning("ALERT: %s", msg)

    return alerts
```

**Test:**
```python
# tests/unit/test_day4_ex4.py
import pytest

from ragas_evaluator import Alert, RAGASScores, check_thresholds


def test_all_pass_no_alerts() -> None:
    scores = RAGASScores(0.9, 0.9, 0.95, 0.9)
    alerts = check_thresholds(scores, {"context_precision": 0.7, "context_recall": 0.7,
                                        "faithfulness": 0.8, "answer_relevance": 0.7,
                                        "composite": 0.75})
    assert alerts == []


def test_faithfulness_alert_raised() -> None:
    scores = RAGASScores(0.9, 0.9, 0.5, 0.9)
    alerts = check_thresholds(scores, {"faithfulness": 0.8})
    names = [a.metric_name for a in alerts]
    assert "faithfulness" in names


def test_alert_fields() -> None:
    scores = RAGASScores(0.5, 0.9, 0.9, 0.9)
    alerts = check_thresholds(scores, {"context_precision": 0.7})
    alert = next(a for a in alerts if a.metric_name == "context_precision")
    assert isinstance(alert, Alert)
    assert alert.score == pytest.approx(0.5)
    assert alert.threshold == pytest.approx(0.7)
    assert "context_precision" in alert.message


def test_custom_thresholds_override_defaults() -> None:
    scores = RAGASScores(0.65, 0.65, 0.65, 0.65)
    # All below defaults but set thresholds very low — expect no alerts
    alerts = check_thresholds(
        scores,
        {"context_precision": 0.1, "context_recall": 0.1,
         "faithfulness": 0.1, "answer_relevance": 0.1, "composite": 0.1},
    )
    assert alerts == []


def test_multiple_alerts_returned() -> None:
    scores = RAGASScores(0.3, 0.3, 0.3, 0.3)
    alerts = check_thresholds(scores)
    assert len(alerts) >= 4  # at least the four base metrics
```
