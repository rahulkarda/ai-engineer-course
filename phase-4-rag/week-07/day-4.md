# Phase 4 · Week 07 · Day 4 — RAGAS-style Metrics: Implementing Core Metrics

> Objective: Implement simplified versions of RAGAS metrics: context precision, context recall, faithfulness, and answer relevance.  ·  Time: ~1 hr  ·  Prereqs: [[week-07/day-3]]

## Concept

RAGAS (Retrieval Augmented Generation Assessment) provides a framework for evaluating RAG pipelines without requiring annotated ground-truth answers for every response. Four pillars:

**Context Precision** — Are the retrieved documents relevant to the question? Measures the fraction of retrieved chunks that actually contain information useful for answering. High precision = low noise in the context window.

**Context Recall** — Are all facts needed to answer the question present in the retrieved context? Measures coverage: if the ground-truth answer requires five facts, how many of those facts appear in the retrieved chunks?

**Faithfulness** — Does the generated answer only make claims that are supported by the retrieved context? An unfaithful answer hallucinates beyond what the context says, even if the answer is factually correct in the real world.

**Answer Relevance** — Does the generated answer actually address the question asked? A faithful, context-grounded answer can still be off-topic or excessively verbose.

Each metric is LLM-scored: no human labels needed per query. The LLM acts as an evaluator, not a generator. This makes RAGAS practical at scale but introduces LLM-as-judge variance — multiple runs may give slightly different scores.

## How it works internally

**Context Precision** — For each retrieved chunk, prompt the LLM: "Given this ground-truth answer, is this chunk relevant?" Count relevant chunks / total chunks. Binary relevance per chunk keeps the prompt simple and scores interpretable.

**Context Recall** — Decompose the ground-truth answer into atomic claims (one sentence = one claim). For each claim, ask the LLM whether the claim can be attributed to the retrieved context. Score = attributable claims / total claims. The decomposition step is itself LLM-driven.

**Faithfulness** — Decompose the generated answer into atomic claims. For each claim, ask whether it is supported by the retrieved context (NLI-style). Score = supported claims / total claims. This is the key hallucination detector.

**Answer Relevance** — Reverse question generation: prompt the LLM to generate N questions that the given answer would answer. Compute cosine similarity between these generated questions and the original question. High similarity = answer is on-topic. Requires an embedding model for similarity.

All four scores are in [0, 1]. The composite score is a simple mean (or weighted mean if you want to emphasize a particular pillar).

## Worked example

```python
"""
ragas_evaluator.py — Simplified RAGAS-style metrics using Anthropic claude-opus-5-5.

Run:
    ANTHROPIC_API_KEY=<key> python ragas_evaluator.py

Requires:
    pip install anthropic numpy
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import anthropic
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class RAGSample:
    """One evaluation sample."""
    question: str
    retrieved_contexts: list[str]          # chunks returned by retriever
    generated_answer: str                  # LLM answer
    ground_truth: str                      # reference answer for precision/recall


@dataclass
class RAGASScores:
    """Scores for a single sample, all in [0, 1]."""
    context_precision: float = 0.0
    context_recall: float = 0.0
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    composite: float = field(init=False)

    def __post_init__(self) -> None:
        self.composite = np.mean([
            self.context_precision,
            self.context_recall,
            self.faithfulness,
            self.answer_relevance,
        ]).item()


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class RAGASEvaluator:
    """
    Simplified RAGAS-style evaluator backed by Anthropic claude-opus-5-5.

    Each metric uses a structured JSON prompt so scores are machine-readable
    without fragile regex parsing.
    """

    _MODEL = "claude-opus-5-5"

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.Anthropic(api_key=key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, sample: RAGSample) -> RAGASScores:
        """Run all four metrics and return a RAGASScores object."""
        logger.info("Evaluating sample: %s", sample.question[:60])

        cp = self._context_precision(sample)
        cr = self._context_recall(sample)
        fa = self._faithfulness(sample)
        ar = self._answer_relevance(sample)

        scores = RAGASScores(
            context_precision=cp,
            context_recall=cr,
            faithfulness=fa,
            answer_relevance=ar,
        )
        logger.info(
            "Scores — precision=%.2f recall=%.2f faithfulness=%.2f relevance=%.2f composite=%.2f",
            scores.context_precision,
            scores.context_recall,
            scores.faithfulness,
            scores.answer_relevance,
            scores.composite,
        )
        return scores

    def evaluate_dataset(self, samples: list[RAGSample]) -> dict[str, float]:
        """
        Evaluate a list of samples and return mean scores across the dataset.
        """
        all_scores = [self.evaluate(s) for s in samples]
        return {
            "context_precision": float(np.mean([s.context_precision for s in all_scores])),
            "context_recall": float(np.mean([s.context_recall for s in all_scores])),
            "faithfulness": float(np.mean([s.faithfulness for s in all_scores])),
            "answer_relevance": float(np.mean([s.answer_relevance for s in all_scores])),
            "composite": float(np.mean([s.composite for s in all_scores])),
        }

    # ------------------------------------------------------------------
    # Metric implementations
    # ------------------------------------------------------------------

    def _context_precision(self, sample: RAGSample) -> float:
        """
        For each retrieved chunk, ask: is this chunk relevant to the ground truth?
        Score = relevant chunks / total chunks.
        """
        if not sample.retrieved_contexts:
            return 0.0

        relevant = 0
        for i, chunk in enumerate(sample.retrieved_contexts):
            prompt = (
                "You are a relevance judge. Given a question and a reference answer, "
                "decide whether the provided text chunk contains information that is "
                "useful for arriving at that answer.\n\n"
                f"Question: {sample.question}\n\n"
                f"Reference answer: {sample.ground_truth}\n\n"
                f"Text chunk:\n{chunk}\n\n"
                'Reply ONLY with valid JSON: {"relevant": true} or {"relevant": false}'
            )
            result = self._call_llm(prompt)
            if result.get("relevant") is True:
                relevant += 1
            logger.debug("Chunk %d relevant=%s", i, result.get("relevant"))

        return relevant / len(sample.retrieved_contexts)

    def _context_recall(self, sample: RAGSample) -> float:
        """
        Decompose ground truth into claims; check each claim against context.
        Score = attributable claims / total claims.
        """
        # Step 1: decompose ground truth into atomic claims
        decompose_prompt = (
            "Break the following answer into a list of standalone atomic claims. "
            "Each claim should be a single fact that can be verified independently.\n\n"
            f"Answer: {sample.ground_truth}\n\n"
            'Reply ONLY with valid JSON: {"claims": ["claim1", "claim2", ...]}'
        )
        decomposed = self._call_llm(decompose_prompt)
        claims: list[str] = decomposed.get("claims", [])

        if not claims:
            logger.warning("No claims extracted from ground truth")
            return 0.0

        context_blob = "\n\n---\n\n".join(sample.retrieved_contexts)

        # Step 2: check attribution for each claim
        attributable = 0
        for claim in claims:
            attr_prompt = (
                "Given the following context, decide whether the claim below can be "
                "attributed to information present in the context.\n\n"
                f"Context:\n{context_blob}\n\n"
                f"Claim: {claim}\n\n"
                'Reply ONLY with valid JSON: {"attributable": true} or {"attributable": false}'
            )
            result = self._call_llm(attr_prompt)
            if result.get("attributable") is True:
                attributable += 1

        return attributable / len(claims)

    def _faithfulness(self, sample: RAGSample) -> float:
        """
        Decompose generated answer into claims; check each against context (NLI-style).
        Score = supported claims / total claims.
        """
        # Step 1: decompose generated answer
        decompose_prompt = (
            "Break the following answer into a list of standalone atomic claims.\n\n"
            f"Answer: {sample.generated_answer}\n\n"
            'Reply ONLY with valid JSON: {"claims": ["claim1", "claim2", ...]}'
        )
        decomposed = self._call_llm(decompose_prompt)
        claims: list[str] = decomposed.get("claims", [])

        if not claims:
            logger.warning("No claims extracted from generated answer")
            return 0.0

        context_blob = "\n\n---\n\n".join(sample.retrieved_contexts)

        # Step 2: NLI-style support check
        supported = 0
        for claim in claims:
            nli_prompt = (
                "Given the following context, decide whether the claim is fully supported "
                "by the information in the context. 'Supported' means the context contains "
                "sufficient evidence for the claim — do not rely on outside knowledge.\n\n"
                f"Context:\n{context_blob}\n\n"
                f"Claim: {claim}\n\n"
                'Reply ONLY with valid JSON: {"supported": true} or {"supported": false}'
            )
            result = self._call_llm(nli_prompt)
            if result.get("supported") is True:
                supported += 1

        return supported / len(claims)

    def _answer_relevance(self, sample: RAGSample) -> float:
        """
        Generate N questions that the answer would address.
        Score = average cosine similarity between generated questions and original question.
        Uses simple token-overlap similarity (no embedding model required).
        """
        n_questions = 3
        gen_prompt = (
            f"Given the following answer, generate {n_questions} different questions "
            "that this answer directly addresses. The questions should be diverse.\n\n"
            f"Answer: {sample.generated_answer}\n\n"
            f'Reply ONLY with valid JSON: {{"questions": ["q1", "q2", "q3"]}}'
        )
        result = self._call_llm(gen_prompt)
        generated_qs: list[str] = result.get("questions", [])

        if not generated_qs:
            logger.warning("No questions generated for answer relevance")
            return 0.0

        # Token-overlap similarity (Jaccard) as a lightweight proxy for cosine similarity
        # In production, replace with an embedding model for better accuracy
        original_tokens = set(sample.question.lower().split())
        similarities = []
        for gq in generated_qs:
            gq_tokens = set(gq.lower().split())
            if not original_tokens and not gq_tokens:
                similarities.append(1.0)
                continue
            union = original_tokens | gq_tokens
            inter = original_tokens & gq_tokens
            similarities.append(len(inter) / len(union) if union else 0.0)

        return float(np.mean(similarities))

    # ------------------------------------------------------------------
    # LLM helper
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> dict[str, Any]:
        """
        Call claude-opus-5-5 and parse the JSON response.
        Returns empty dict on parse failure so callers degrade gracefully.
        """
        try:
            message = self._client.messages.create(
                model=self._MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("JSON parse failure: %s — raw: %.80s", exc, raw)
            return {}
        except anthropic.APIError as exc:
            logger.error("Anthropic API error: %s", exc)
            return {}


# ---------------------------------------------------------------------------
# Quick demo (run directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = RAGSample(
        question="What is the capital of France and what is it known for?",
        retrieved_contexts=[
            "Paris is the capital city of France. It is located in northern France on the "
            "River Seine and has a population of over 2 million in the city proper.",
            "France is a country in Western Europe. It is known for its cuisine, wine, and "
            "fashion. The Eiffel Tower is one of the most visited monuments in the world.",
            "The history of ancient Rome spans several centuries.",  # irrelevant chunk
        ],
        generated_answer=(
            "The capital of France is Paris. It is known for landmarks such as the Eiffel Tower, "
            "its world-famous cuisine, fashion industry, and its location on the River Seine."
        ),
        ground_truth=(
            "Paris is the capital of France. It is known for the Eiffel Tower, French cuisine, "
            "fashion, and the River Seine."
        ),
    )

    evaluator = RAGASEvaluator()
    scores = evaluator.evaluate(sample)
    print(f"\nComposite RAGAS score: {scores.composite:.3f}")
```

```python
# tests/test_ragas_evaluator.py — pytest with mocked LLM responses

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from ragas_evaluator import RAGASEvaluator, RAGSample


@pytest.fixture()
def evaluator() -> RAGASEvaluator:
    """Create evaluator with a dummy API key (LLM is mocked)."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        with patch("anthropic.Anthropic"):
            return RAGASEvaluator(api_key="test-key")


def _make_message(payload: dict) -> MagicMock:
    """Build a mock Anthropic message object returning JSON payload."""
    msg = MagicMock()
    msg.content = [MagicMock(text=json.dumps(payload))]
    return msg


@pytest.fixture()
def sample() -> RAGSample:
    return RAGSample(
        question="What is the capital of France?",
        retrieved_contexts=["Paris is the capital of France.", "Rome is in Italy."],
        generated_answer="The capital of France is Paris.",
        ground_truth="Paris is the capital of France.",
    )


class TestContextPrecision:
    def test_all_relevant(self, evaluator: RAGASEvaluator, sample: RAGSample) -> None:
        evaluator._client.messages.create.side_effect = [
            _make_message({"relevant": True}),
            _make_message({"relevant": True}),
        ]
        score = evaluator._context_precision(sample)
        assert score == pytest.approx(1.0)

    def test_partial_relevant(self, evaluator: RAGASEvaluator, sample: RAGSample) -> None:
        evaluator._client.messages.create.side_effect = [
            _make_message({"relevant": True}),
            _make_message({"relevant": False}),
        ]
        score = evaluator._context_precision(sample)
        assert score == pytest.approx(0.5)

    def test_empty_contexts(self, evaluator: RAGASEvaluator, sample: RAGSample) -> None:
        sample.retrieved_contexts = []
        assert evaluator._context_precision(sample) == 0.0


class TestContextRecall:
    def test_full_recall(self, evaluator: RAGASEvaluator, sample: RAGSample) -> None:
        evaluator._client.messages.create.side_effect = [
            _make_message({"claims": ["Paris is the capital of France."]}),
            _make_message({"attributable": True}),
        ]
        score = evaluator._context_recall(sample)
        assert score == pytest.approx(1.0)

    def test_partial_recall(self, evaluator: RAGASEvaluator, sample: RAGSample) -> None:
        evaluator._client.messages.create.side_effect = [
            _make_message({"claims": ["claim A", "claim B"]}),
            _make_message({"attributable": True}),
            _make_message({"attributable": False}),
        ]
        score = evaluator._context_recall(sample)
        assert score == pytest.approx(0.5)


class TestFaithfulness:
    def test_fully_faithful(self, evaluator: RAGASEvaluator, sample: RAGSample) -> None:
        evaluator._client.messages.create.side_effect = [
            _make_message({"claims": ["The capital of France is Paris."]}),
            _make_message({"supported": True}),
        ]
        score = evaluator._faithfulness(sample)
        assert score == pytest.approx(1.0)

    def test_hallucination(self, evaluator: RAGASEvaluator, sample: RAGSample) -> None:
        evaluator._client.messages.create.side_effect = [
            _make_message({"claims": ["claim A", "unsupported hallucination"]}),
            _make_message({"supported": True}),
            _make_message({"supported": False}),
        ]
        score = evaluator._faithfulness(sample)
        assert score == pytest.approx(0.5)


class TestAnswerRelevance:
    def test_relevant_answer(self, evaluator: RAGASEvaluator, sample: RAGSample) -> None:
        # Generated questions that closely match the original question
        evaluator._client.messages.create.return_value = _make_message(
            {"questions": [
                "What is the capital of France?",
                "Which city is the capital of France?",
                "Name the capital city of France.",
            ]}
        )
        score = evaluator._answer_relevance(sample)
        # Jaccard similarity with near-identical questions should be > 0.3
        assert score > 0.3

    def test_irrelevant_answer(self, evaluator: RAGASEvaluator, sample: RAGSample) -> None:
        evaluator._client.messages.create.return_value = _make_message(
            {"questions": [
                "How do penguins swim?",
                "What is the boiling point of water?",
                "Who wrote Hamlet?",
            ]}
        )
        score = evaluator._answer_relevance(sample)
        # Very low overlap expected
        assert score < 0.3
```

## Exercises

1. **Weighted composite**: The `RAGASScores.composite` currently uses a simple mean. Modify `__post_init__` to accept optional per-metric weights and compute a weighted mean. Write a pytest parametrize test covering equal weights, faithfulness-heavy weights (0.5), and edge case where all weights are 0.

2. **Batch efficiency**: The current `evaluate_dataset` calls `evaluate` sequentially, resulting in many synchronous LLM calls. Refactor using `asyncio` and `anthropic.AsyncAnthropic` so all per-chunk calls within a single metric run concurrently. Benchmark the speedup with `time.perf_counter` on a dataset of 5 samples.

3. **Embedding-based answer relevance**: Replace the Jaccard token-overlap similarity in `_answer_relevance` with real cosine similarity using Anthropic's embeddings API (`client.embeddings.create` or a lightweight local model via `sentence-transformers`). Test that the embedding-based version ranks a paraphrase of the original question higher than a completely unrelated question.

4. **Threshold alerting**: Add a `threshold` parameter to `RAGASEvaluator.evaluate_dataset`. If any of the five keys in the returned dict falls below its threshold, raise a `MetricBelowThreshold` exception (custom exception class) with a message listing which metrics failed. Write pytest tests for both the passing and failing cases.

(Solutions: ../solutions/week-07/day-4.md)

## Self-check

- [ ] I can explain the difference between context precision and context recall in my own words, with a concrete example of a retrieval failure that would hurt one but not the other
- [ ] I can explain why faithfulness and answer relevance are orthogonal — give an example answer that scores high on faithfulness but low on answer relevance
- [ ] I can implement all four metric prompts from memory, including the JSON contract each prompt enforces
- [ ] I can explain why claim decomposition is needed for faithfulness and recall, and what goes wrong if you skip it
- [ ] I can implement a pytest fixture that mocks the Anthropic client and returns deterministic scores without making real API calls

## Supplementary links

- RAGAS paper: https://arxiv.org/abs/2309.15217
- RAGAS Python library: https://docs.ragas.io/en/latest/
- Anthropic Messages API reference: https://docs.anthropic.com/en/api/messages
- LLM-as-judge survey: https://arxiv.org/abs/2306.05685
