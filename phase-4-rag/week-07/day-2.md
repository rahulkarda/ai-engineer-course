# Phase 4 · Week 07 · Day 2 — Generation Metrics: Faithfulness and Answer Relevance

> Objective: Measure whether generated answers are faithful to retrieved context and relevant to the question.  ·  Time: ~1 hr  ·  Prereqs: [[week-07/day-1]]

## Concept

Retrieval quality is necessary but not sufficient. Even with perfect recall, a language model can produce answers that:

1. **Hallucinate** — assert facts not present in the retrieved context.
2. **Wander off-topic** — retrieve the right docs but answer a different question.

Two distinct signals catch these failure modes:

- **Faithfulness** (groundedness): every claim in the answer is supported by the retrieved context. Measures whether the LLM "made things up."
- **Answer relevance**: the answer actually addresses the question asked. Measures whether the LLM "stayed on topic."

These are orthogonal: an answer can be fully grounded but completely irrelevant ("The sky is blue. Photons scatter." for "What is the capital of France?"), or highly relevant but unfaithful (correct conclusion, invented supporting detail).

**String-overlap vs LLM-as-judge**

| Approach | Speed | Cost | Semantic depth |
|---|---|---|---|
| Token-overlap (ROUGE, F1) | Fast | Free | Low — misses paraphrase |
| LLM-as-judge | Slow | API cost | High — catches semantics |

In practice, use token-overlap as a fast filter and LLM-judge for the cases that matter most or for calibration.

## How it works internally

**Token F1 (faithfulness baseline)**

Tokenise both the answer and the context. Compute precision and recall of answer tokens against context tokens, then take the F1. This is a proxy: a high-F1 answer reuses context vocabulary, which correlates with groundedness.

```
precision = |tokens(answer) ∩ tokens(context)| / |tokens(answer)|
recall    = |tokens(answer) ∩ tokens(context)| / |tokens(context)|
token_f1  = 2 * precision * recall / (precision + recall)
```

Limitations: order-insensitive, synonym-blind, context length dominates recall.

**ROUGE-L (faithfulness baseline alternative)**

ROUGE-L measures the longest common subsequence (LCS) between answer and context. Slightly better than F1 for multi-word phrase matching, still fully lexical.

**LLM-as-judge (richer faithfulness)**

Send a structured prompt to an LLM with:
- The retrieved context
- The generated answer
- A rubric (e.g. "Does every claim in the answer appear in the context? Score 1-5.")

The LLM returns a JSON score + reasoning. Key design choices:
- Use chain-of-thought before the score to reduce inconsistency.
- Ask for explicit evidence quotation so the score is auditable.
- Always request structured JSON output to make parsing deterministic.

**Answer relevance**

Use a similar LLM-judge prompt focused on: "Does the answer directly address the question?" independent of whether it is grounded. Can also be estimated by embedding the question and answer, computing cosine similarity — cheap but misses pragmatic relevance.

## Worked example

```python
"""
week_07/day2_generation_metrics.py

Faithfulness and answer relevance metrics for RAG evaluation.
Requires: pip install anthropic
Set ANTHROPIC_API_KEY in environment before running.
Run: python week_07/day2_generation_metrics.py
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class GenerationSample:
    """One RAG generation to evaluate."""
    question: str
    context: str       # concatenated retrieved chunks
    answer: str        # LLM-generated answer


@dataclass
class FaithfulnessScore:
    token_f1: float
    llm_score: int | None         # 1-5 or None if LLM call skipped
    llm_reasoning: str | None


@dataclass
class RelevanceScore:
    llm_score: int | None         # 1-5
    llm_reasoning: str | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> Counter[str]:
    """Lowercase whitespace-split tokens, no punctuation stripping for simplicity."""
    return Counter(text.lower().split())


# ---------------------------------------------------------------------------
# Metrics class
# ---------------------------------------------------------------------------

class GenerationMetrics:
    """
    Compute faithfulness and answer relevance for a GenerationSample.

    LLM-based methods require ANTHROPIC_API_KEY in environment.
    """

    MODEL_ID = "claude-opus-5-5"

    def __init__(self) -> None:
        self._client: object | None = None  # lazy-init

    def _get_client(self) -> object:
        """Lazy-initialise Anthropic client; raises if key missing."""
        if self._client is None:
            try:
                import anthropic  # type: ignore[import]
            except ImportError as exc:
                raise ImportError("pip install anthropic") from exc
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise EnvironmentError("ANTHROPIC_API_KEY not set")
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    # ------------------------------------------------------------------
    # String-overlap faithfulness
    # ------------------------------------------------------------------

    def token_f1_faithfulness(self, sample: GenerationSample) -> float:
        """
        Token-overlap F1 between answer and context.
        High score = answer tokens appear in context (proxy for groundedness).
        """
        answer_tokens = _tokenise(sample.answer)
        context_tokens = _tokenise(sample.context)

        if not answer_tokens or not context_tokens:
            return 0.0

        # intersection counts minimum of each token's frequency
        common: Counter[str] = Counter()
        for token, count in answer_tokens.items():
            common[token] = min(count, context_tokens[token])
        common_count = sum(common.values())

        precision = common_count / sum(answer_tokens.values())
        recall = common_count / sum(context_tokens.values())

        if precision + recall == 0.0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    # ------------------------------------------------------------------
    # LLM-judge faithfulness
    # ------------------------------------------------------------------

    _FAITHFULNESS_PROMPT = """\
You are a rigorous evaluator. Given a CONTEXT and an ANSWER, assess whether \
every factual claim in the ANSWER is directly supported by the CONTEXT.

CONTEXT:
{context}

ANSWER:
{answer}

Instructions:
1. Identify each factual claim in the ANSWER.
2. For each claim, determine whether it is explicitly supported by the CONTEXT.
3. Score faithfulness on a scale of 1-5:
   1 = Most claims contradict or are absent from the context (hallucination).
   2 = Some claims supported, but significant unsupported content.
   3 = About half supported.
   4 = Most claims supported; minor gaps.
   5 = Every claim is directly supported by the context.

Respond with ONLY a JSON object in this exact format:
{{
  "score": <integer 1-5>,
  "reasoning": "<brief chain-of-thought, max 3 sentences>"
}}"""

    def llm_faithfulness(self, sample: GenerationSample) -> tuple[int, str]:
        """
        Use Claude as judge to score faithfulness (1-5).
        Returns (score, reasoning).
        Raises EnvironmentError if ANTHROPIC_API_KEY not set.
        """
        client = self._get_client()
        prompt = self._FAITHFULNESS_PROMPT.format(
            context=sample.context,
            answer=sample.answer,
        )
        response = client.messages.create(  # type: ignore[union-attr]
            model=self.MODEL_ID,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)
        return int(parsed["score"]), str(parsed["reasoning"])

    # ------------------------------------------------------------------
    # LLM-judge answer relevance
    # ------------------------------------------------------------------

    _RELEVANCE_PROMPT = """\
You are a rigorous evaluator. Given a QUESTION and an ANSWER, assess whether \
the ANSWER directly and completely addresses the QUESTION.

QUESTION:
{question}

ANSWER:
{answer}

Instructions:
1. Determine whether the ANSWER addresses what the QUESTION is asking.
2. Score relevance on a scale of 1-5:
   1 = Completely off-topic or evasive.
   2 = Tangentially related but does not answer the question.
   3 = Partially answers the question.
   4 = Mostly answers the question with minor gaps.
   5 = Directly and completely answers the question.

Respond with ONLY a JSON object in this exact format:
{{
  "score": <integer 1-5>,
  "reasoning": "<brief chain-of-thought, max 3 sentences>"
}}"""

    def llm_relevance(self, sample: GenerationSample) -> tuple[int, str]:
        """
        Use Claude as judge to score answer relevance (1-5).
        Returns (score, reasoning).
        """
        client = self._get_client()
        prompt = self._RELEVANCE_PROMPT.format(
            question=sample.question,
            answer=sample.answer,
        )
        response = client.messages.create(  # type: ignore[union-attr]
            model=self.MODEL_ID,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)
        return int(parsed["score"]), str(parsed["reasoning"])

    def evaluate(
        self,
        sample: GenerationSample,
        use_llm: bool = False,
    ) -> tuple[FaithfulnessScore, RelevanceScore]:
        """
        Full evaluation pass. Set use_llm=True to call Claude API.
        """
        f1 = self.token_f1_faithfulness(sample)
        llm_faith_score: int | None = None
        llm_faith_reason: str | None = None
        llm_rel_score: int | None = None
        llm_rel_reason: str | None = None

        if use_llm:
            llm_faith_score, llm_faith_reason = self.llm_faithfulness(sample)
            llm_rel_score, llm_rel_reason = self.llm_relevance(sample)

        faithfulness = FaithfulnessScore(
            token_f1=f1,
            llm_score=llm_faith_score,
            llm_reasoning=llm_faith_reason,
        )
        relevance = RelevanceScore(
            llm_score=llm_rel_score,
            llm_reasoning=llm_rel_reason,
        )
        return faithfulness, relevance


# ---------------------------------------------------------------------------
# Demo (string-overlap only — no API key needed)
# ---------------------------------------------------------------------------

def main() -> None:
    samples = [
        GenerationSample(
            question="What is photosynthesis?",
            context=(
                "Photosynthesis is the process by which green plants convert "
                "sunlight, water, and carbon dioxide into glucose and oxygen."
            ),
            answer=(
                "Photosynthesis converts sunlight, water, and carbon dioxide "
                "into glucose and oxygen in green plants."
            ),
        ),
        GenerationSample(
            question="What is photosynthesis?",
            context=(
                "Photosynthesis is the process by which green plants convert "
                "sunlight, water, and carbon dioxide into glucose and oxygen."
            ),
            answer=(
                # Hallucination: ATP and mitochondria not in context
                "Photosynthesis uses ATP produced by mitochondria to convert "
                "sunlight into glucose."
            ),
        ),
    ]

    metrics = GenerationMetrics()

    for i, sample in enumerate(samples, start=1):
        f1 = metrics.token_f1_faithfulness(sample)
        logger.info(
            "Sample %d — token_f1_faithfulness=%.3f  question=%r",
            i, f1, sample.question[:40],
        )

    logger.info(
        "Sample 1 has higher F1 (%.3f) because answer tokens appear in context.",
        metrics.token_f1_faithfulness(samples[0]),
    )
    logger.info(
        "Sample 2 is lower (%.3f) because 'ATP' and 'mitochondria' are absent from context.",
        metrics.token_f1_faithfulness(samples[1]),
    )
    logger.info(
        "To run LLM judge: set ANTHROPIC_API_KEY and call metrics.evaluate(sample, use_llm=True)",
    )


if __name__ == "__main__":
    main()
```

**pytest with mock (no real API key needed):**

```python
"""
tests/unit/test_day2_generation_metrics.py

Unit tests for GenerationMetrics. LLM calls are fully mocked.
Run: pytest tests/unit/test_day2_generation_metrics.py -v
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from week_07.day2_generation_metrics import GenerationMetrics, GenerationSample


@pytest.fixture()
def metrics() -> GenerationMetrics:
    return GenerationMetrics()


@pytest.fixture()
def grounded_sample() -> GenerationSample:
    return GenerationSample(
        question="What is photosynthesis?",
        context="Photosynthesis converts sunlight water co2 into glucose oxygen.",
        answer="Photosynthesis converts sunlight water co2 into glucose oxygen.",
    )


@pytest.fixture()
def hallucinated_sample() -> GenerationSample:
    return GenerationSample(
        question="What is photosynthesis?",
        context="Photosynthesis converts sunlight water co2 into glucose oxygen.",
        answer="Photosynthesis uses nuclear fusion to produce ATP in ribosomes.",
    )


class TestTokenF1Faithfulness:
    def test_identical_answer_and_context_gives_high_score(
        self, metrics: GenerationMetrics, grounded_sample: GenerationSample
    ) -> None:
        score = metrics.token_f1_faithfulness(grounded_sample)
        assert score > 0.8

    def test_unrelated_answer_gives_low_score(
        self, metrics: GenerationMetrics, hallucinated_sample: GenerationSample
    ) -> None:
        score = metrics.token_f1_faithfulness(hallucinated_sample)
        assert score < 0.3

    def test_empty_answer_returns_zero(self, metrics: GenerationMetrics) -> None:
        sample = GenerationSample(question="q", context="some context", answer="")
        assert metrics.token_f1_faithfulness(sample) == 0.0

    def test_empty_context_returns_zero(self, metrics: GenerationMetrics) -> None:
        sample = GenerationSample(question="q", context="", answer="some answer")
        assert metrics.token_f1_faithfulness(sample) == 0.0


class TestLLMFaithfulness:
    def _make_mock_client(self, score: int, reasoning: str) -> MagicMock:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({"score": score, "reasoning": reasoning}))]
        mock_client.messages.create.return_value = mock_response
        return mock_client

    def test_llm_faithfulness_returns_score_and_reasoning(
        self, metrics: GenerationMetrics, grounded_sample: GenerationSample
    ) -> None:
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            mock_client = self._make_mock_client(5, "All claims are present in context.")
            metrics._client = mock_client  # inject mock directly
            score, reasoning = metrics.llm_faithfulness(grounded_sample)

        assert score == 5
        assert "claims" in reasoning.lower()

    def test_llm_faithfulness_raises_without_api_key(
        self, metrics: GenerationMetrics, grounded_sample: GenerationSample
    ) -> None:
        with patch.dict("os.environ", {}, clear=True):
            # Remove key if present
            import os
            os.environ.pop("ANTHROPIC_API_KEY", None)
            metrics._client = None  # force re-init
            with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
                metrics.llm_faithfulness(grounded_sample)
```

## Exercises

1. **ROUGE-L faithfulness**: Implement a `rouge_l_faithfulness(sample)` method that computes the F1 of the Longest Common Subsequence between `sample.answer` and `sample.context` (split on whitespace). Compare its scores to `token_f1_faithfulness` on three samples of your choice.

2. **Embedding-based relevance**: Use the `sentence-transformers` library (or any embedding API) to compute the cosine similarity between the question embedding and the answer embedding as an answer relevance proxy. Add `embedding_relevance(sample)` to `GenerationMetrics`. When does this agree or disagree with LLM-judge relevance?

3. **Batch evaluation**: Write a function `batch_evaluate(samples: list[GenerationSample], metrics: GenerationMetrics) -> list[dict]` that runs `token_f1_faithfulness` on each sample and returns a list of dicts with keys `question`, `token_f1`, `is_grounded` (True if f1 > 0.5). Write a pytest that covers at least two samples.

4. **Calibration check**: Given a list of LLM-judge faithfulness scores (integers 1-5) and corresponding human labels (0=not faithful, 1=faithful), compute the Pearson correlation and find the threshold on the 1-5 scale that maximises accuracy. Use only the standard library (`statistics` module for correlation).

(Solutions: ../solutions/week-07/day-2.md)

## Self-check

- [ ] I can explain the difference between faithfulness and answer relevance without confusing the two.
- [ ] I can implement token F1 faithfulness from scratch using only Python built-ins.
- [ ] I can write an LLM-judge prompt with a rubric and structured JSON output requirement.
- [ ] I can mock an Anthropic API call in pytest so tests run without a real API key.

---

## Supplementary links

- RAGAS framework (automated RAG evaluation): https://github.com/explodinggradients/ragas
- TruLens evaluation library: https://github.com/truera/trulens
- Anthropic Messages API docs: https://docs.anthropic.com/en/api/messages
- "LLM as a Judge" paper (Zheng et al. 2023): https://arxiv.org/abs/2306.05685
- ROUGE paper (Lin 2004): https://aclanthology.org/W04-1013/
