# Solutions — Phase 4 · Week 07 · Day 3

## Exercise 1

```python
import json
import logging
import os
from typing import Any

import anthropic
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

JUDGE_MODEL = "claude-opus-5-5"


class GoldenSample(BaseModel):
    """A single ground-truth evaluation sample."""

    question: str
    context: str
    expected_answer: str
    metadata: dict[str, Any] = {}


def generate_golden_samples(
    docs: list[str],
    llm_client: anthropic.Anthropic,
) -> list[GoldenSample]:
    """
    Use an LLM to generate (question, expected_answer) pairs from documents.

    Each document yields 1–2 GoldenSamples. The document text is used as
    the context field so evaluators can later verify answer grounding.

    Args:
        docs:       List of document strings (raw passage text).
        llm_client: Instantiated Anthropic client.

    Returns:
        List of GoldenSample instances, one or two per document.
    """
    samples: list[GoldenSample] = []

    for idx, doc in enumerate(docs):
        logger.info("Generating questions for doc %d/%d", idx + 1, len(docs))
        prompt = f"""Given the following document, generate 1 or 2 question-answer pairs that:
- Can be answered directly from the document
- Are specific and factual (not opinion or inference)
- Have concise expected answers (under 100 words each)

Document:
{doc}

Return ONLY a JSON array with this structure:
[
  {{
    "question": "...",
    "expected_answer": "..."
  }}
]
Generate either 1 or 2 objects depending on how much factual content the document contains."""

        response = llm_client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=512,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        logger.debug("doc %d raw response: %s", idx, raw)

        try:
            pairs: list[dict[str, str]] = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Skipping doc %d — invalid JSON: %s", idx, exc)
            continue

        for pair in pairs:
            sample = GoldenSample(
                question=pair["question"],
                context=doc,
                expected_answer=pair["expected_answer"],
                metadata={"source_doc_index": idx},
            )
            samples.append(sample)
            logger.debug("Added sample: %s", sample.question)

    logger.info("Generated %d golden samples from %d docs", len(samples), len(docs))
    return samples


# --- Test with mocked LLM ---

def test_generate_golden_samples() -> None:
    import unittest.mock as mock

    docs = [
        "Python was created by Guido van Rossum and first released in 1991.",
        "The Eiffel Tower is 330 metres tall and located in Paris, France.",
    ]

    mock_client = mock.MagicMock()

    mock_responses = [
        json.dumps([
            {"question": "Who created Python?", "expected_answer": "Guido van Rossum"},
            {"question": "When was Python first released?", "expected_answer": "1991"},
        ]),
        json.dumps([
            {"question": "How tall is the Eiffel Tower?", "expected_answer": "330 metres"},
        ]),
    ]

    def make_response(text: str) -> mock.MagicMock:
        resp = mock.MagicMock()
        resp.content[0].text = text
        return resp

    mock_client.messages.create.side_effect = [
        make_response(r) for r in mock_responses
    ]

    samples = generate_golden_samples(docs, mock_client)

    assert len(samples) == 3
    assert samples[0].question == "Who created Python?"
    assert samples[0].context == docs[0]
    assert samples[0].metadata["source_doc_index"] == 0
    assert isinstance(samples[0], GoldenSample)
    print("Exercise 1 test passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_generate_golden_samples()
```

## Exercise 2

```python
import logging
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class GoldenSample(BaseModel):
    question: str
    context: str
    expected_answer: str
    metadata: dict[str, Any] = {}


class EvalDataset(BaseModel):
    """A validated collection of golden evaluation samples."""

    samples: list[GoldenSample]

    def validate_samples(self) -> None:
        """
        Validate dataset integrity. Raises ValueError with a specific
        message on the first failing condition.

        Checks:
        1. All samples have non-empty question, context, expected_answer.
        2. No duplicate question strings (case-sensitive).
        3. expected_answer is <= 500 characters.
        """
        seen_questions: set[str] = set()

        for i, sample in enumerate(self.samples):
            # Check 1: non-empty required fields
            if not sample.question.strip():
                raise ValueError(
                    f"Sample {i}: 'question' is empty or whitespace-only"
                )
            if not sample.context.strip():
                raise ValueError(
                    f"Sample {i}: 'context' is empty or whitespace-only"
                )
            if not sample.expected_answer.strip():
                raise ValueError(
                    f"Sample {i}: 'expected_answer' is empty or whitespace-only"
                )

            # Check 2: duplicate questions
            if sample.question in seen_questions:
                raise ValueError(
                    f"Sample {i}: duplicate question '{sample.question}'"
                )
            seen_questions.add(sample.question)

            # Check 3: expected_answer length
            if len(sample.expected_answer) > 500:
                raise ValueError(
                    f"Sample {i}: 'expected_answer' exceeds 500 chars "
                    f"(got {len(sample.expected_answer)})"
                )

        logger.info("EvalDataset.validate_samples: all %d samples valid", len(self.samples))


# --- Tests ---

def test_validate_passes_clean_dataset() -> None:
    dataset = EvalDataset(samples=[
        GoldenSample(question="Q1?", context="ctx1", expected_answer="ans1"),
        GoldenSample(question="Q2?", context="ctx2", expected_answer="ans2"),
    ])
    dataset.validate_samples()  # should not raise
    print("test_validate_passes_clean_dataset passed")


def test_validate_rejects_empty_question() -> None:
    dataset = EvalDataset(samples=[
        GoldenSample(question="  ", context="ctx", expected_answer="ans"),
    ])
    try:
        dataset.validate_samples()
        assert False, "Should have raised"
    except ValueError as e:
        assert "question" in str(e).lower()
    print("test_validate_rejects_empty_question passed")


def test_validate_rejects_empty_context() -> None:
    dataset = EvalDataset(samples=[
        GoldenSample(question="Q?", context="", expected_answer="ans"),
    ])
    try:
        dataset.validate_samples()
        assert False, "Should have raised"
    except ValueError as e:
        assert "context" in str(e).lower()
    print("test_validate_rejects_empty_context passed")


def test_validate_rejects_duplicate_question() -> None:
    dataset = EvalDataset(samples=[
        GoldenSample(question="Same?", context="c1", expected_answer="a1"),
        GoldenSample(question="Same?", context="c2", expected_answer="a2"),
    ])
    try:
        dataset.validate_samples()
        assert False, "Should have raised"
    except ValueError as e:
        assert "duplicate" in str(e).lower()
    print("test_validate_rejects_duplicate_question passed")


def test_validate_rejects_long_answer() -> None:
    dataset = EvalDataset(samples=[
        GoldenSample(question="Q?", context="ctx", expected_answer="x" * 501),
    ])
    try:
        dataset.validate_samples()
        assert False, "Should have raised"
    except ValueError as e:
        assert "500" in str(e)
    print("test_validate_rejects_long_answer passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_validate_passes_clean_dataset()
    test_validate_rejects_empty_question()
    test_validate_rejects_empty_context()
    test_validate_rejects_duplicate_question()
    test_validate_rejects_long_answer()
    print("All Exercise 2 tests passed.")
```

## Exercise 3

```python
import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any

import anthropic
from pydantic import BaseModel

logger = logging.getLogger(__name__)

JUDGE_MODEL = "claude-opus-5-5"


class GoldenSample(BaseModel):
    question: str
    context: str
    expected_answer: str
    metadata: dict[str, Any] = {}


@dataclass
class LLMJudge:
    """LLM-as-judge for faithfulness scoring."""

    _client: anthropic.Anthropic = field(init=False)

    def __post_init__(self) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.Anthropic(api_key=api_key)

    def score_faithfulness(self, sample: GoldenSample, answer: str) -> float:
        """Return faithfulness score 1–5 for the given answer."""
        prompt = f"""Rate how faithful the ANSWER is to the CONTEXT on a scale of 1–5.
5=fully grounded, 1=contradicts or ignores context.

QUESTION: {sample.question}
CONTEXT: {sample.context}
ANSWER: {answer}

Return ONLY JSON: {{"faithfulness": <integer 1-5>}}"""

        response = self._client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=64,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        data = json.loads(response.content[0].text.strip())
        return float(data["faithfulness"])


def calibrate_judge(
    annotated_pairs: list[tuple[GoldenSample, float]],
    judge: LLMJudge,
    answers: list[str],
) -> float:
    """
    Measure alignment between an LLM judge and human annotations.

    Args:
        annotated_pairs: List of (GoldenSample, human_score) tuples.
        judge:           Instantiated LLMJudge.
        answers:         Parallel list of model answers to score, same
                         length as annotated_pairs.

    Returns:
        Pearson correlation coefficient between human scores and LLM scores.
        Range [-1, 1]. Values >= 0.7 indicate good calibration.

    Raises:
        ValueError: If fewer than 2 samples are provided (cannot compute r).
    """
    if len(annotated_pairs) < 2:
        raise ValueError("Need at least 2 samples to compute Pearson correlation")
    if len(annotated_pairs) != len(answers):
        raise ValueError("annotated_pairs and answers must have the same length")

    human_scores: list[float] = []
    llm_scores: list[float] = []

    for (sample, human_score), answer in zip(annotated_pairs, answers):
        llm_score = judge.score_faithfulness(sample, answer)
        human_scores.append(human_score)
        llm_scores.append(llm_score)
        logger.debug(
            "calibrate_judge: human=%.2f llm=%.2f q=%r",
            human_score,
            llm_score,
            sample.question,
        )

    r = _pearson(human_scores, llm_scores)
    logger.info("Pearson correlation (human vs LLM): %.4f", r)
    return r


def _pearson(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient without scipy dependency."""
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denom_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    denom_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    if denom_x == 0 or denom_y == 0:
        logger.warning("_pearson: zero variance in one series — returning 0.0")
        return 0.0
    return numerator / (denom_x * denom_y)


# --- Tests ---

def test_calibrate_judge_high_correlation() -> None:
    import unittest.mock as mock

    judge = LLMJudge.__new__(LLMJudge)
    mock_client = mock.MagicMock()
    judge._client = mock_client  # type: ignore[attr-defined]

    # LLM scores closely track human scores -> high correlation
    human_scores_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    llm_scores_data   = [1.0, 2.0, 3.0, 4.0, 5.0]

    side_effects = []
    for score in llm_scores_data:
        resp = mock.MagicMock()
        resp.content[0].text = json.dumps({"faithfulness": int(score)})
        side_effects.append(resp)
    mock_client.messages.create.side_effect = side_effects

    samples = [
        GoldenSample(question=f"Q{i}?", context="ctx", expected_answer="ans")
        for i in range(5)
    ]
    pairs = list(zip(samples, human_scores_data))
    answers = ["answer"] * 5

    r = calibrate_judge(pairs, judge, answers)
    assert r == 1.0, f"Expected r=1.0, got {r}"
    print("test_calibrate_judge_high_correlation passed")


def test_calibrate_judge_low_correlation() -> None:
    import unittest.mock as mock

    judge = LLMJudge.__new__(LLMJudge)
    mock_client = mock.MagicMock()
    judge._client = mock_client  # type: ignore[attr-defined]

    human_scores_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    llm_scores_data   = [5.0, 4.0, 3.0, 2.0, 1.0]  # inverted -> r = -1.0

    side_effects = []
    for score in llm_scores_data:
        resp = mock.MagicMock()
        resp.content[0].text = json.dumps({"faithfulness": int(score)})
        side_effects.append(resp)
    mock_client.messages.create.side_effect = side_effects

    samples = [
        GoldenSample(question=f"Q{i}?", context="ctx", expected_answer="ans")
        for i in range(5)
    ]
    pairs = list(zip(samples, human_scores_data))
    answers = ["answer"] * 5

    r = calibrate_judge(pairs, judge, answers)
    assert abs(r - (-1.0)) < 1e-9, f"Expected r=-1.0, got {r}"
    print("test_calibrate_judge_low_correlation passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_calibrate_judge_high_correlation()
    test_calibrate_judge_low_correlation()
    print("All Exercise 3 tests passed.")
```

## Exercise 4

```python
import json
import logging
import os
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import anthropic
from pydantic import BaseModel

logger = logging.getLogger(__name__)

JUDGE_MODEL = "claude-opus-5-5"


class HallucinationSeverity(IntEnum):
    """Hallucination severity levels."""

    NONE = 0    # All claims grounded in context
    MINOR = 1   # Small unsupported additions; no contradictions
    MAJOR = 2   # Contradicts context or invents significant facts


class GoldenSample(BaseModel):
    question: str
    context: str
    expected_answer: str
    metadata: dict[str, Any] = {}


class JudgeResult(BaseModel):
    """Structured output from the dual-dimension LLM judge."""

    faithfulness: float                          # 1–5 scale
    hallucination_severity: HallucinationSeverity
    reasoning: str


@dataclass
class LLMJudge:
    """
    LLM-as-judge scoring two dimensions:

    - faithfulness (1–5): how well the answer is grounded in context
    - hallucination_severity (0/1/2): degree of unsupported or contradicting claims

    The two dimensions are related but complementary:
    - faithfulness measures how much of the answer IS supported
    - hallucination_severity classifies the worst offending claim type
    """

    _client: anthropic.Anthropic = field(init=False)

    def __post_init__(self) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.Anthropic(api_key=api_key)

    def evaluate(self, sample: GoldenSample, answer: str) -> JudgeResult:
        """
        Score answer on faithfulness and hallucination severity.

        Args:
            sample: GoldenSample providing question and context.
            answer: The generated answer to evaluate.

        Returns:
            JudgeResult with both dimensions and a brief reasoning string.
        """
        prompt = f"""You are an expert RAG evaluator. Score the ANSWER on two dimensions.

QUESTION: {sample.question}
CONTEXT: {sample.context}
ANSWER: {answer}

Return ONLY valid JSON:
{{
  "faithfulness": <integer 1-5>,
  "hallucination_severity": <0, 1, or 2>,
  "reasoning": "<one concise sentence covering both scores>"
}}

Scoring rubrics:

faithfulness (1-5):
  5 = Every claim is directly supported by context
  4 = Mostly grounded, trivial additions only
  3 = Some supported, some unsupported claims
  2 = Mostly unsupported
  1 = Contradicts or ignores context entirely

hallucination_severity (0/1/2):
  0 = NONE    — all claims traceable to context; no invented facts
  1 = MINOR   — adds plausible-sounding but unsupported details;
                does NOT contradict context
  2 = MAJOR   — invents significant facts or directly contradicts context"""

        response = self._client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=192,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        logger.debug("LLMJudge.evaluate raw: %s", raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned invalid JSON: {raw}") from exc

        result = JudgeResult(
            faithfulness=float(data["faithfulness"]),
            hallucination_severity=HallucinationSeverity(int(data["hallucination_severity"])),
            reasoning=data.get("reasoning", ""),
        )
        logger.info(
            "evaluate: faithfulness=%.1f severity=%s",
            result.faithfulness,
            result.hallucination_severity.name,
        )
        return result


# --- Tests ---

def test_evaluate_none_hallucination() -> None:
    import unittest.mock as mock

    judge = LLMJudge.__new__(LLMJudge)
    mock_client = mock.MagicMock()
    judge._client = mock_client  # type: ignore[attr-defined]

    resp = mock.MagicMock()
    resp.content[0].text = json.dumps({
        "faithfulness": 5,
        "hallucination_severity": 0,
        "reasoning": "All claims are fully grounded in context.",
    })
    mock_client.messages.create.return_value = resp

    sample = GoldenSample(
        question="What does ATP do?",
        context="ATP provides energy for cellular processes.",
        expected_answer="ATP provides cellular energy.",
    )
    result = judge.evaluate(sample, "ATP provides energy for cellular processes.")

    assert result.faithfulness == 5.0
    assert result.hallucination_severity == HallucinationSeverity.NONE
    print("test_evaluate_none_hallucination passed")


def test_evaluate_major_hallucination() -> None:
    import unittest.mock as mock

    judge = LLMJudge.__new__(LLMJudge)
    mock_client = mock.MagicMock()
    judge._client = mock_client  # type: ignore[attr-defined]

    resp = mock.MagicMock()
    resp.content[0].text = json.dumps({
        "faithfulness": 1,
        "hallucination_severity": 2,
        "reasoning": "Answer directly contradicts the provided context.",
    })
    mock_client.messages.create.return_value = resp

    sample = GoldenSample(
        question="What does ATP do?",
        context="ATP provides energy for cellular processes.",
        expected_answer="ATP provides cellular energy.",
    )
    result = judge.evaluate(sample, "ATP inhibits all cellular activity.")

    assert result.faithfulness == 1.0
    assert result.hallucination_severity == HallucinationSeverity.MAJOR
    print("test_evaluate_major_hallucination passed")


def test_evaluate_minor_hallucination() -> None:
    import unittest.mock as mock

    judge = LLMJudge.__new__(LLMJudge)
    mock_client = mock.MagicMock()
    judge._client = mock_client  # type: ignore[attr-defined]

    resp = mock.MagicMock()
    resp.content[0].text = json.dumps({
        "faithfulness": 3,
        "hallucination_severity": 1,
        "reasoning": "Mostly grounded but adds an unsupported claim about efficiency.",
    })
    mock_client.messages.create.return_value = resp

    sample = GoldenSample(
        question="What does ATP do?",
        context="ATP provides energy for cellular processes.",
        expected_answer="ATP provides cellular energy.",
    )
    result = judge.evaluate(sample, "ATP provides energy and also regulates pH levels.")

    assert result.faithfulness == 3.0
    assert result.hallucination_severity == HallucinationSeverity.MINOR
    print("test_evaluate_minor_hallucination passed")


def test_judge_result_uses_temperature_zero() -> None:
    """Confirm temperature=0 is always passed to avoid inconsistency."""
    import unittest.mock as mock

    judge = LLMJudge.__new__(LLMJudge)
    mock_client = mock.MagicMock()
    judge._client = mock_client  # type: ignore[attr-defined]

    resp = mock.MagicMock()
    resp.content[0].text = json.dumps({
        "faithfulness": 4,
        "hallucination_severity": 0,
        "reasoning": "Fine.",
    })
    mock_client.messages.create.return_value = resp

    sample = GoldenSample(question="Q?", context="ctx", expected_answer="ans")
    judge.evaluate(sample, "ans")

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["temperature"] == 0
    print("test_judge_result_uses_temperature_zero passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_evaluate_none_hallucination()
    test_evaluate_major_hallucination()
    test_evaluate_minor_hallucination()
    test_judge_result_uses_temperature_zero()
    print("All Exercise 4 tests passed.")
```
