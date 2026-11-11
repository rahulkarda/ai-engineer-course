# Phase 4 · Week 07 · Day 3 — Golden Datasets and LLM-as-Judge Evaluators

> Objective: Build a golden evaluation dataset and implement an LLM-as-judge pipeline for RAG evaluation.  ·  Time: ~1 hr  ·  Prereqs: [[week-07/day-2]]

## Concept

**Golden datasets**

A golden dataset is a curated collection of `(question, context, expected_answer)` triples where correctness has been verified — by humans, by subject matter experts, or by a trusted LLM pass followed by human spot-check.

Why you need them:
- Ad-hoc testing ("it looks fine") does not scale and does not catch regressions.
- Without a fixed eval set, every model or retriever change is a gamble.
- They are the ground truth your automated judges are calibrated against.

Building them:
1. Sample representative questions from real user traffic or from your document corpus.
2. Run your best current pipeline to generate candidate answers.
3. Human-review a sample; LLM-review the rest; flag low-confidence cases for human review.
4. Store as structured JSON — not spreadsheets, not docs.

**LLM-as-judge**

An LLM judge replaces or supplements human annotation. You send the judge:
- The question
- The retrieved context
- The generated answer (and optionally the expected answer)
- A rubric with a numeric scale

The judge returns a structured score + chain-of-thought reasoning. The reasoning is load-bearing: it makes the score auditable and helps you diagnose failure patterns.

**Trade-offs vs human evaluation**

| | Human eval | LLM-as-judge |
|---|---|---|
| Cost | High | Low (API tokens) |
| Speed | Days | Seconds |
| Consistency | Variable across annotators | Consistent within a model version |
| Semantic depth | Highest | High |
| Bias | Human bias | Model bias (hallucination self-evaluation) |
| Calibration | Ground truth | Must be calibrated against human labels |

Best practice: calibrate LLM-judge scores against a small human-labelled set, then use the judge at scale.

## How it works internally

**Dataset schema with Pydantic**

Pydantic enforces types, provides validation, and serialises cleanly to/from JSON. A `GoldenSample` holds one evaluation example. An `EvalDataset` wraps a list of samples with metadata (version, creation date, owner).

**Generating candidates**

Typical pipeline:
1. For each document chunk, ask an LLM: "What questions can this chunk answer?"
2. Run your RAG pipeline on each generated question to produce a candidate answer.
3. Store question + retrieved context + candidate answer as a `GoldenSample`.
4. Human-review or run a quality-filter LLM to flag weak samples.

**LLM judge prompt design**

Key principles:
- **Rubric first**: define the scale (1-5) with explicit anchors before presenting the content.
- **Chain-of-thought before score**: ask the judge to reason before committing to a number. This reduces bias from anchoring.
- **JSON output**: always request structured output so parsing is deterministic.
- **Separate faithfulness from relevance**: one prompt per dimension to avoid conflation.

**Calibration against human labels**

Collect 50-100 human-labelled samples. Compute Pearson or Spearman correlation between LLM scores and human scores. If correlation < 0.7, investigate: the rubric may be ambiguous, or the model may self-favour certain styles. Use the calibration set to choose a threshold (e.g. "LLM score >= 4 = pass") that matches human accept/reject rate.

## Worked example

```python
"""
week_07/day3_golden_dataset.py

Golden evaluation dataset schema and LLM-as-judge pipeline for RAG.
Requires: pip install anthropic pydantic
Set ANTHROPIC_API_KEY in environment before running LLM judge.
Run: python week_07/day3_golden_dataset.py
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class GoldenSample(BaseModel):
    """One evaluation example in a golden dataset."""

    sample_id: str = Field(description="Unique identifier, e.g. 'q001'")
    question: str = Field(description="The question posed to the RAG system")
    context: str = Field(description="Retrieved context used to generate the answer")
    expected_answer: str = Field(description="Ground-truth answer, human-verified")
    generated_answer: str | None = Field(
        default=None,
        description="Answer produced by the RAG pipeline under evaluation",
    )
    source_doc_ids: list[str] = Field(
        default_factory=list,
        description="IDs of documents from which this sample was derived",
    )
    difficulty: str = Field(
        default="medium",
        description="Estimated difficulty: easy | medium | hard",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Topic tags for stratified analysis",
    )


class JudgeResult(BaseModel):
    """Score and reasoning from the LLM judge for one sample."""

    sample_id: str
    faithfulness_score: int = Field(ge=1, le=5)
    relevance_score: int = Field(ge=1, le=5)
    faithfulness_reasoning: str
    relevance_reasoning: str
    passed: bool = Field(description="True if both scores >= threshold")


class EvalDataset(BaseModel):
    """A versioned, serialisable collection of golden samples."""

    version: str = Field(default="1.0.0")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    owner: str = Field(default="unknown")
    description: str = Field(default="")
    samples: list[GoldenSample] = Field(default_factory=list)

    def add(self, sample: GoldenSample) -> None:
        """Append a sample, raising ValueError on duplicate sample_id."""
        existing_ids = {s.sample_id for s in self.samples}
        if sample.sample_id in existing_ids:
            raise ValueError(f"Duplicate sample_id: {sample.sample_id}")
        self.samples.append(sample)

    def save(self, path: Path) -> None:
        """Serialise dataset to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Saved %d samples to %s", len(self.samples), path)

    @classmethod
    def load(cls, path: Path) -> "EvalDataset":
        """Load dataset from JSON file."""
        raw = path.read_text(encoding="utf-8")
        dataset = cls.model_validate_json(raw)
        logger.info("Loaded %d samples from %s", len(dataset.samples), path)
        return dataset

    def filter_by_tag(self, tag: str) -> list[GoldenSample]:
        """Return samples that have the given tag."""
        return [s for s in self.samples if tag in s.tags]

    def summary(self) -> dict[str, object]:
        """Return a dict with aggregate statistics."""
        difficulty_counts: dict[str, int] = {}
        for s in self.samples:
            difficulty_counts[s.difficulty] = difficulty_counts.get(s.difficulty, 0) + 1
        return {
            "total": len(self.samples),
            "with_generated_answer": sum(1 for s in self.samples if s.generated_answer),
            "difficulty_distribution": difficulty_counts,
            "version": self.version,
        }


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

class LLMJudge:
    """
    Scores a GoldenSample's generated_answer against context and question
    using Claude as the evaluator.

    Requires ANTHROPIC_API_KEY in environment.
    """

    MODEL_ID = "claude-opus-5-5"
    PASS_THRESHOLD = 4  # scores >= this are considered passing

    _JUDGE_PROMPT = """\
You are a strict but fair RAG evaluation judge. Evaluate the GENERATED ANSWER \
against the CONTEXT and QUESTION using two dimensions.

QUESTION:
{question}

CONTEXT:
{context}

GENERATED ANSWER:
{generated_answer}

EXPECTED ANSWER (for reference only — do not penalise paraphrase):
{expected_answer}

Scoring rubrics:

FAITHFULNESS (is every claim in the generated answer supported by the context?):
  1 = Most claims unsupported or contradicted by context (hallucination).
  2 = Significant unsupported content.
  3 = About half of claims supported.
  4 = Most claims supported; minor unsupported details.
  5 = Every claim directly supported by context.

RELEVANCE (does the generated answer address the question?):
  1 = Completely off-topic.
  2 = Tangentially related; does not answer the question.
  3 = Partially answers the question.
  4 = Mostly answers the question with minor gaps.
  5 = Directly and completely answers the question.

Instructions:
1. Reason through faithfulness step by step (2-3 sentences).
2. Assign a faithfulness score.
3. Reason through relevance step by step (2-3 sentences).
4. Assign a relevance score.

Respond with ONLY a JSON object in this exact format:
{{
  "faithfulness_reasoning": "<your reasoning>",
  "faithfulness_score": <integer 1-5>,
  "relevance_reasoning": "<your reasoning>",
  "relevance_score": <integer 1-5>
}}"""

    def __init__(self) -> None:
        self._client: object | None = None

    def _get_client(self) -> object:
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

    def score(self, sample: GoldenSample) -> JudgeResult:
        """
        Score a single sample. Requires sample.generated_answer to be set.
        Raises ValueError if generated_answer is None.
        """
        if sample.generated_answer is None:
            raise ValueError(f"sample {sample.sample_id} has no generated_answer")

        client = self._get_client()
        prompt = self._JUDGE_PROMPT.format(
            question=sample.question,
            context=sample.context,
            generated_answer=sample.generated_answer,
            expected_answer=sample.expected_answer,
        )
        response = client.messages.create(  # type: ignore[union-attr]
            model=self.MODEL_ID,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)

        faith_score = int(parsed["faithfulness_score"])
        rel_score = int(parsed["relevance_score"])
        return JudgeResult(
            sample_id=sample.sample_id,
            faithfulness_score=faith_score,
            relevance_score=rel_score,
            faithfulness_reasoning=parsed["faithfulness_reasoning"],
            relevance_reasoning=parsed["relevance_reasoning"],
            passed=(faith_score >= self.PASS_THRESHOLD and rel_score >= self.PASS_THRESHOLD),
        )

    def score_dataset(self, dataset: EvalDataset) -> list[JudgeResult]:
        """Score all samples in a dataset that have a generated_answer."""
        results = []
        for sample in dataset.samples:
            if sample.generated_answer is None:
                logger.warning("Skipping %s — no generated_answer", sample.sample_id)
                continue
            result = self.score(sample)
            logger.info(
                "%s  faith=%d  rel=%d  passed=%s",
                result.sample_id,
                result.faithfulness_score,
                result.relevance_score,
                result.passed,
            )
            results.append(result)
        return results


# ---------------------------------------------------------------------------
# Demo (dataset creation only — no API call)
# ---------------------------------------------------------------------------

def build_demo_dataset() -> EvalDataset:
    """Build a small demo dataset with pre-filled generated answers."""
    dataset = EvalDataset(
        version="0.1.0",
        owner="ai-engineer-course",
        description="Week 07 demo golden dataset — photosynthesis domain",
    )

    dataset.add(GoldenSample(
        sample_id="q001",
        question="What inputs does photosynthesis require?",
        context=(
            "Photosynthesis is the process by which green plants use sunlight, "
            "water, and carbon dioxide to produce glucose and oxygen."
        ),
        expected_answer="Photosynthesis requires sunlight, water, and carbon dioxide.",
        generated_answer="Photosynthesis requires sunlight, water, and carbon dioxide.",
        source_doc_ids=["doc_001"],
        difficulty="easy",
        tags=["biology", "photosynthesis"],
    ))

    dataset.add(GoldenSample(
        sample_id="q002",
        question="What inputs does photosynthesis require?",
        context=(
            "Photosynthesis is the process by which green plants use sunlight, "
            "water, and carbon dioxide to produce glucose and oxygen."
        ),
        expected_answer="Photosynthesis requires sunlight, water, and carbon dioxide.",
        # Hallucination: nitrogen and minerals not in context
        generated_answer="Photosynthesis requires sunlight, nitrogen, and minerals from the soil.",
        source_doc_ids=["doc_001"],
        difficulty="easy",
        tags=["biology", "photosynthesis"],
    ))

    return dataset


def main() -> None:
    dataset = build_demo_dataset()
    summary = dataset.summary()
    logger.info("Dataset summary: %s", summary)

    # Save and reload round-trip
    out_path = Path("/tmp/week07_golden_dataset.json")
    dataset.save(out_path)
    reloaded = EvalDataset.load(out_path)
    logger.info("Round-trip OK: %d samples", len(reloaded.samples))

    # Filter demonstration
    bio_samples = reloaded.filter_by_tag("biology")
    logger.info("Samples tagged 'biology': %d", len(bio_samples))

    logger.info(
        "To run LLM judge: set ANTHROPIC_API_KEY and call "
        "LLMJudge().score_dataset(dataset)"
    )


if __name__ == "__main__":
    main()
```

**pytest with mock:**

```python
"""
tests/unit/test_day3_golden_dataset.py

Unit tests for EvalDataset and LLMJudge. LLM calls are fully mocked.
Run: pytest tests/unit/test_day3_golden_dataset.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from week_07.day3_golden_dataset import (
    EvalDataset,
    GoldenSample,
    JudgeResult,
    LLMJudge,
)


@pytest.fixture()
def sample() -> GoldenSample:
    return GoldenSample(
        sample_id="q001",
        question="What is photosynthesis?",
        context="Photosynthesis converts sunlight, water, and CO2 into glucose.",
        expected_answer="Photosynthesis converts sunlight into glucose.",
        generated_answer="Photosynthesis converts sunlight and CO2 into glucose.",
        tags=["biology"],
    )


@pytest.fixture()
def dataset(sample: GoldenSample) -> EvalDataset:
    ds = EvalDataset(version="0.1.0", owner="test")
    ds.add(sample)
    return ds


class TestEvalDataset:
    def test_add_and_retrieve(self, dataset: EvalDataset, sample: GoldenSample) -> None:
        assert len(dataset.samples) == 1
        assert dataset.samples[0].sample_id == sample.sample_id

    def test_duplicate_raises(self, dataset: EvalDataset, sample: GoldenSample) -> None:
        with pytest.raises(ValueError, match="Duplicate"):
            dataset.add(sample)

    def test_save_and_load_round_trip(self, dataset: EvalDataset) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_dataset.json"
            dataset.save(path)
            reloaded = EvalDataset.load(path)
        assert len(reloaded.samples) == len(dataset.samples)
        assert reloaded.samples[0].sample_id == dataset.samples[0].sample_id

    def test_filter_by_tag(self, dataset: EvalDataset) -> None:
        bio = dataset.filter_by_tag("biology")
        assert len(bio) == 1
        empty = dataset.filter_by_tag("chemistry")
        assert len(empty) == 0

    def test_summary_keys(self, dataset: EvalDataset) -> None:
        s = dataset.summary()
        assert "total" in s
        assert s["total"] == 1


class TestLLMJudge:
    def _make_mock_client(self, faith: int, rel: int) -> MagicMock:
        mock_client = MagicMock()
        payload = {
            "faithfulness_reasoning": "Claim supported by context.",
            "faithfulness_score": faith,
            "relevance_reasoning": "Directly answers the question.",
            "relevance_score": rel,
        }
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(payload))]
        mock_client.messages.create.return_value = mock_response
        return mock_client

    def test_score_returns_judge_result(self, sample: GoldenSample) -> None:
        judge = LLMJudge()
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            judge._client = self._make_mock_client(faith=5, rel=4)
            result = judge.score(sample)

        assert isinstance(result, JudgeResult)
        assert result.faithfulness_score == 5
        assert result.relevance_score == 4
        assert result.passed is True

    def test_score_below_threshold_fails(self, sample: GoldenSample) -> None:
        judge = LLMJudge()
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            judge._client = self._make_mock_client(faith=2, rel=5)
            result = judge.score(sample)
        assert result.passed is False

    def test_score_raises_without_generated_answer(self) -> None:
        judge = LLMJudge()
        sample_no_answer = GoldenSample(
            sample_id="qX",
            question="q",
            context="c",
            expected_answer="e",
        )
        with pytest.raises(ValueError, match="generated_answer"):
            judge.score(sample_no_answer)

    def test_score_raises_without_api_key(self, sample: GoldenSample) -> None:
        judge = LLMJudge()
        judge._client = None
        import os
        os.environ.pop("ANTHROPIC_API_KEY", None)
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
                judge.score(sample)
```

## Exercises

1. **Dataset generation from docs**: Write a function `generate_samples_from_chunks(chunks: list[str], n_per_chunk: int) -> list[GoldenSample]` that, for each chunk, calls an LLM to generate `n_per_chunk` questions. The LLM call must be mockable (accept an optional `client` parameter). Use `os.getenv("ANTHROPIC_API_KEY")`. Write a pytest that mocks the LLM and checks the returned list length equals `len(chunks) * n_per_chunk`.

2. **Calibration metric**: Given a list of `JudgeResult` objects and a parallel list of binary human labels (1 = good answer, 0 = bad answer), write a function `calibrate(results, human_labels, threshold)` that returns a dict with keys `precision`, `recall`, `f1` where "positive" is `judge_result.passed`. Sweep the threshold from 3 to 5 and print the metrics for each.

3. **Stratified analysis**: Extend `EvalDataset` with a method `results_by_difficulty(judge_results: list[JudgeResult]) -> dict[str, dict]` that returns, for each difficulty level, the average faithfulness score, average relevance score, and pass rate. Use only the standard library.

4. **Adversarial sample**: Design a `GoldenSample` where `expected_answer` and `generated_answer` differ in wording but are semantically equivalent, and another where they agree in wording but the generated answer adds one hallucinated claim. Confirm that your `LLMJudge` (mocked with appropriate scores) correctly scores the hallucination case lower on faithfulness.

(Solutions: ../solutions/week-07/day-3.md)

## Self-check

- [ ] I can explain why golden datasets need to be versioned and stored as structured JSON rather than spreadsheets.
- [ ] I can implement `GoldenSample` and `EvalDataset` Pydantic models with save/load round-trip from scratch.
- [ ] I can write an LLM-judge prompt that returns structured JSON scores for two separate rubric dimensions.
- [ ] I can mock an Anthropic API response in pytest so that `LLMJudge.score` tests run without a real API key.
- [ ] I can describe at least two failure modes of LLM-as-judge evaluation and how to detect them.

---

## Supplementary links

- RAGAS automated RAG evaluation (uses LLM-as-judge internally): https://github.com/explodinggradients/ragas
- "LLM as a Judge" paper (Zheng et al. 2023): https://arxiv.org/abs/2306.05685
- Anthropic Messages API structured output guide: https://docs.anthropic.com/en/docs/test-and-evaluate/strengthen-guardrails/increase-consistency
- Pydantic v2 serialisation docs: https://docs.pydantic.dev/latest/concepts/serialization/
- "Building Reliable LLM Applications" — Eugene Yan: https://eugeneyan.com/writing/llm-patterns/
