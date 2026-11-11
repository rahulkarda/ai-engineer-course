# Solutions — Phase 4 · Week 07 · Day 2

## Exercise 1

```python
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric boundaries."""
    return re.findall(r"\b\w+\b", text.lower())


@dataclass
class GenerationMetrics:
    """Token-level and LLM-judge metrics for RAG generation quality."""

    def token_f1_faithfulness(self, answer: str, context: str) -> float:
        """
        Token-level F1 measuring overlap between answer and context.

        Precision: fraction of answer tokens that appear in context.
        Recall:    fraction of context tokens that appear in answer.
        F1:        harmonic mean of precision and recall.

        Returns 0.0 when either answer or context is empty.
        """
        answer_tokens = tokenize(answer)
        context_tokens = tokenize(context)

        if not answer_tokens or not context_tokens:
            logger.warning("token_f1_faithfulness: empty input — returning 0.0")
            return 0.0

        answer_set = set(answer_tokens)
        context_set = set(context_tokens)

        # Precision: answer tokens that appear in context
        precision_hits = sum(1 for t in answer_tokens if t in context_set)
        precision = precision_hits / len(answer_tokens)

        # Recall: context tokens that appear in answer
        recall_hits = sum(1 for t in context_tokens if t in answer_set)
        recall = recall_hits / len(context_tokens)

        if precision + recall == 0.0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        logger.debug(
            "token_f1: precision=%.4f recall=%.4f f1=%.4f", precision, recall, f1
        )
        return f1


# --- Quick manual check ---

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    gm = GenerationMetrics()

    context = "The quick brown fox jumps over the lazy dog"
    answer_full = "The quick brown fox jumps over the lazy dog"
    answer_none = "cats eat fish and sleep all day"
    answer_partial = "The quick fox jumped over a dog"

    print(f"Full overlap:    {gm.token_f1_faithfulness(answer_full, context):.4f}")
    print(f"No overlap:      {gm.token_f1_faithfulness(answer_none, context):.4f}")
    print(f"Partial overlap: {gm.token_f1_faithfulness(answer_partial, context):.4f}")
```

## Exercise 2

```python
import json
import logging
import os
from dataclasses import dataclass, field

import anthropic

logger = logging.getLogger(__name__)

JUDGE_MODEL = "claude-opus-5-5"


@dataclass
class JudgeResult:
    faithfulness: float        # 1.0–5.0
    answer_completeness: float  # 1.0–5.0
    reasoning: str


@dataclass
class GenerationMetrics:
    """LLM-as-judge with faithfulness and answer completeness dimensions."""

    _client: anthropic.Anthropic = field(init=False)

    def __post_init__(self) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.Anthropic(api_key=api_key)

    def judge(self, question: str, context: str, answer: str) -> JudgeResult:
        """
        Score answer on faithfulness (grounded in context) and completeness
        (covers all key points needed to answer the question).

        Both dimensions use a 1–5 integer scale.
        """
        prompt = f"""You are an expert RAG evaluator. Score the assistant answer on TWO dimensions.

QUESTION: {question}

CONTEXT: {context}

ANSWER: {answer}

Return ONLY valid JSON with these keys:
{{
  "faithfulness": <integer 1-5>,
  "answer_completeness": <integer 1-5>,
  "reasoning": "<one sentence explaining both scores>"
}}

Scoring rubrics:
- faithfulness: 5=all claims grounded in context, 1=major unsupported claims
- answer_completeness: 5=covers all key points from context relevant to question,
  1=misses most key points"""

        response = self._client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=256,
            temperature=0,  # deterministic — see Exercise 3
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        logger.debug("judge raw response: %s", raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned invalid JSON: {raw}") from exc

        return JudgeResult(
            faithfulness=float(data["faithfulness"]),
            answer_completeness=float(data["answer_completeness"]),
            reasoning=data.get("reasoning", ""),
        )


# --- Test with mocked LLM ---

def test_judge_completeness_field() -> None:
    import unittest.mock as mock

    gm = GenerationMetrics.__new__(GenerationMetrics)
    mock_client = mock.MagicMock()
    gm._client = mock_client  # type: ignore[attr-defined]

    mock_response = mock.MagicMock()
    mock_response.content[0].text = json.dumps({
        "faithfulness": 4,
        "answer_completeness": 3,
        "reasoning": "Answer is mostly grounded but misses one key point.",
    })
    mock_client.messages.create.return_value = mock_response

    result = gm.judge(
        question="What is Python?",
        context="Python is a high-level, interpreted programming language.",
        answer="Python is interpreted.",
    )

    assert result.faithfulness == 4.0
    assert result.answer_completeness == 3.0
    assert "key point" in result.reasoning
    print("Exercise 2 test passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_judge_completeness_field()
```

## Exercise 3

```python
import json
import logging
import os
import statistics
from dataclasses import dataclass, field
from collections import Counter

import anthropic

logger = logging.getLogger(__name__)

JUDGE_MODEL = "claude-opus-5-5"


# -----------------------------------------------------------------------
# Approach 1 (recommended): temperature=0
# -----------------------------------------------------------------------
# LLM judges are inconsistent because sampling introduces randomness at
# each token. Setting temperature=0 makes the model deterministic: it
# always picks the highest-probability token. This is the simplest and
# cheapest fix and is sufficient for most evaluation pipelines.
#
# Limitation: temperature=0 removes diversity, so if the model has a
# systematic bias for a score it will always return that bias. Majority
# vote (Approach 2) can help detect this but does not fix it.

# -----------------------------------------------------------------------
# Approach 2: majority vote over N calls (shown below)
# -----------------------------------------------------------------------
# Run the judge N times (e.g. N=5) and return the mode score. This
# averages out sampling noise and provides a confidence estimate via the
# vote distribution. More expensive (N x LLM calls).


@dataclass
class StableGenerationMetrics:
    """LLM judge with two consistency strategies."""

    n_votes: int = 5  # for majority-vote mode

    _client: anthropic.Anthropic = field(init=False)

    def __post_init__(self) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.Anthropic(api_key=api_key)

    def _build_prompt(self, question: str, context: str, answer: str) -> str:
        return f"""Rate the faithfulness of the ANSWER to the CONTEXT on a scale of 1–5.

QUESTION: {question}
CONTEXT: {context}
ANSWER: {answer}

Return ONLY valid JSON: {{"faithfulness": <integer 1-5>, "reasoning": "<one sentence>"}}"""

    def judge_temperature_zero(
        self, question: str, context: str, answer: str
    ) -> dict[str, float | str]:
        """Deterministic judge via temperature=0 (recommended approach)."""
        response = self._client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=128,
            temperature=0,
            messages=[{"role": "user", "content": self._build_prompt(question, context, answer)}],
        )
        data = json.loads(response.content[0].text.strip())
        logger.debug("temperature=0 judge result: %s", data)
        return data

    def judge_majority_vote(
        self, question: str, context: str, answer: str
    ) -> dict[str, float | str]:
        """
        Majority-vote judge: run N times at temperature>0, return modal score.
        Also returns vote distribution for transparency.
        """
        scores: list[int] = []
        reasonings: list[str] = []

        for i in range(self.n_votes):
            response = self._client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=128,
                temperature=1,  # allow sampling diversity across votes
                messages=[
                    {"role": "user", "content": self._build_prompt(question, context, answer)}
                ],
            )
            data = json.loads(response.content[0].text.strip())
            scores.append(int(data["faithfulness"]))
            reasonings.append(data.get("reasoning", ""))
            logger.debug("vote %d/%d: score=%d", i + 1, self.n_votes, scores[-1])

        vote_counts = Counter(scores)
        modal_score = vote_counts.most_common(1)[0][0]
        confidence = vote_counts[modal_score] / self.n_votes

        logger.info(
            "majority_vote: modal=%d confidence=%.2f distribution=%s",
            modal_score,
            confidence,
            dict(vote_counts),
        )
        return {
            "faithfulness": float(modal_score),
            "confidence": confidence,
            "vote_distribution": dict(vote_counts),  # type: ignore[dict-item]
            "reasoning": reasonings[scores.index(modal_score)],
        }


# --- Test with mocked LLM ---

def test_majority_vote_returns_mode() -> None:
    import unittest.mock as mock

    sm = StableGenerationMetrics.__new__(StableGenerationMetrics)
    sm.n_votes = 5
    mock_client = mock.MagicMock()
    sm._client = mock_client  # type: ignore[attr-defined]

    # Votes: 4, 3, 4, 4, 5 -> mode = 4
    vote_scores = [4, 3, 4, 4, 5]
    side_effects = []
    for score in vote_scores:
        resp = mock.MagicMock()
        resp.content[0].text = json.dumps({"faithfulness": score, "reasoning": "ok"})
        side_effects.append(resp)
    mock_client.messages.create.side_effect = side_effects

    result = sm.judge_majority_vote("Q?", "ctx", "ans")
    assert result["faithfulness"] == 4.0
    assert result["confidence"] == 3 / 5
    print("Exercise 3 majority-vote test passed.")


def test_temperature_zero_judge() -> None:
    import unittest.mock as mock

    sm = StableGenerationMetrics.__new__(StableGenerationMetrics)
    mock_client = mock.MagicMock()
    sm._client = mock_client  # type: ignore[attr-defined]

    resp = mock.MagicMock()
    resp.content[0].text = json.dumps({"faithfulness": 5, "reasoning": "fully grounded"})
    mock_client.messages.create.return_value = resp

    result = sm.judge_temperature_zero("Q?", "ctx", "ans")
    assert result["faithfulness"] == 5

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["temperature"] == 0, "Must use temperature=0"
    print("Exercise 3 temperature=0 test passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_majority_vote_returns_mode()
    test_temperature_zero_judge()
    print("All Exercise 3 tests passed.")
```

## Exercise 4

```python
import logging
import re
import pytest

logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def token_f1_faithfulness(answer: str, context: str) -> float:
    """Token-level F1 between answer and context (binary overlap)."""
    answer_tokens = tokenize(answer)
    context_tokens = tokenize(context)

    if not answer_tokens or not context_tokens:
        logger.warning("token_f1_faithfulness: empty input — returning 0.0")
        return 0.0

    answer_set = set(answer_tokens)
    context_set = set(context_tokens)

    precision = sum(1 for t in answer_tokens if t in context_set) / len(answer_tokens)
    recall = sum(1 for t in context_tokens if t in answer_set) / len(context_tokens)

    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Parametrized pytest tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "answer, context, expected_range",
    [
        # (a) Answer fully copied from context -> score near 1.0
        (
            "The mitochondria is the powerhouse of the cell.",
            "The mitochondria is the powerhouse of the cell.",
            (0.95, 1.0),
        ),
        # (b) Answer with no word overlap with context -> score near 0.0
        (
            "Jupiter Saturn Neptune Uranus",
            "Paris London Berlin Rome",
            (0.0, 0.05),
        ),
        # (c) Partial overlap -> score between 0.3 and 0.7
        (
            "The sun rises in the east every morning.",
            "The moon rises in the west and the sun sets in the east.",
            (0.3, 0.7),
        ),
    ],
    ids=["full_overlap", "no_overlap", "partial_overlap"],
)
def test_token_f1_faithfulness(answer: str, context: str, expected_range: tuple[float, float]) -> None:
    score = token_f1_faithfulness(answer, context)
    low, high = expected_range
    assert low <= score <= high, (
        f"Expected score in [{low}, {high}] but got {score:.4f}\n"
        f"  answer:  {answer!r}\n"
        f"  context: {context!r}"
    )


# --- Additional edge-case tests ---

def test_token_f1_empty_answer() -> None:
    assert token_f1_faithfulness("", "some context words") == 0.0


def test_token_f1_empty_context() -> None:
    assert token_f1_faithfulness("some answer words", "") == 0.0


def test_token_f1_symmetric_when_identical() -> None:
    text = "hello world foo bar"
    assert token_f1_faithfulness(text, text) == pytest.approx(1.0, abs=1e-9)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # Run the parametrized cases manually without pytest
    cases = [
        (
            "The mitochondria is the powerhouse of the cell.",
            "The mitochondria is the powerhouse of the cell.",
            (0.95, 1.0),
            "full_overlap",
        ),
        (
            "Jupiter Saturn Neptune Uranus",
            "Paris London Berlin Rome",
            (0.0, 0.05),
            "no_overlap",
        ),
        (
            "The sun rises in the east every morning.",
            "The moon rises in the west and the sun sets in the east.",
            (0.3, 0.7),
            "partial_overlap",
        ),
    ]
    for answer, context, (low, high), label in cases:
        score = token_f1_faithfulness(answer, context)
        status = "PASS" if low <= score <= high else "FAIL"
        print(f"[{status}] {label}: score={score:.4f}, expected=[{low}, {high}]")
```
