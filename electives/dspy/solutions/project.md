# Solutions — Project: Optimize a Classification/Extraction Pipeline

> Reference implementation. Attempt the project independently before reading.

---

## project_data.py

```python
"""project_data.py — 30 labeled support ticket examples."""
import dspy

_RAW: list[tuple[str, str, str, str, str]] = [
    # (ticket, category, urgency, sentiment, recommendation)
    ("I was billed twice for my annual subscription.", "billing", "high", "negative",
     "Escalate to billing team for duplicate charge investigation."),
    ("The API returns HTTP 500 on all POST /orders requests.", "technical", "high", "negative",
     "Assign immediately to on-call backend engineer."),
    ("I cannot log in; the page just reloads.", "account", "medium", "negative",
     "Route to account team for authentication debugging."),
    ("How do I update my billing address?", "billing", "low", "neutral",
     "Send self-service link to billing settings page."),
    ("SDK throws AttributeError on version 3.2.1.", "technical", "medium", "negative",
     "Assign to SDK team with version and stack trace."),
    ("I love the new dashboard — really intuitive.", "other", "low", "positive",
     "Log as positive feedback; no action required."),
    ("My payment failed three times today.", "billing", "high", "negative",
     "Escalate to billing team to inspect payment gateway errors."),
    ("Webhook events are arriving 10 minutes late.", "technical", "high", "negative",
     "Assign to infrastructure team for latency investigation."),
    ("I need a receipt for my last invoice.", "billing", "low", "neutral",
     "Send download link to invoice portal."),
    ("Password reset email never arrives.", "account", "medium", "negative",
     "Route to account team to check email delivery pipeline."),
    ("Your docs are really clear, great job!", "other", "low", "positive",
     "Log as positive feedback; share with documentation team."),
    ("Dashboard is extremely slow since the update.", "technical", "medium", "negative",
     "Assign to frontend team to investigate regression."),
    ("I was charged for a plan I downgraded from.", "billing", "high", "negative",
     "Escalate to billing for refund and plan correction."),
    ("Two-factor authentication is not working.", "account", "high", "negative",
     "Route to security team — potential auth service outage."),
    ("Can I add a second user to my account?", "account", "low", "neutral",
     "Send link to team management documentation."),
    ("Rate limiting is triggering at 50% of our advertised quota.", "technical", "high", "negative",
     "Assign to API team to audit rate limit configuration."),
    ("The mobile app crashes on iOS 17.", "technical", "high", "negative",
     "Assign to mobile team with device and OS details."),
    ("I'd like to cancel my subscription.", "billing", "medium", "neutral",
     "Route to retention team per cancellation policy."),
    ("SSO login fails with SAML error code 40.", "account", "high", "negative",
     "Route to enterprise account team for SAML configuration review."),
    ("The CSV export is missing three columns.", "technical", "medium", "negative",
     "Assign to data team to fix export schema."),
    ("Renewed my plan and very happy with the service!", "billing", "low", "positive",
     "Log as positive feedback; no action required."),
    ("I can't find where to download my data.", "account", "low", "neutral",
     "Send link to data export documentation."),
    ("Refund request for unused annual prepayment.", "billing", "medium", "neutral",
     "Route to billing for pro-rated refund assessment."),
    ("Notifications are not sending to Slack.", "technical", "low", "neutral",
     "Route to integrations team to check Slack webhook config."),
    ("Account locked after too many login attempts.", "account", "high", "negative",
     "Route to account team for immediate lockout resolution."),
    ("Your pricing page has a broken link.", "other", "low", "neutral",
     "Route to web team to fix broken link."),
    ("I want to upgrade to the enterprise plan.", "billing", "low", "positive",
     "Route to sales team for enterprise onboarding."),
    ("GraphQL mutations return null for nested objects.", "technical", "medium", "negative",
     "Assign to API team with query and schema reproduction case."),
    ("How do I set up OAuth2 with your API?", "technical", "low", "neutral",
     "Send link to OAuth2 integration guide."),
    ("I've been waiting 3 weeks for my invoice to be corrected.", "billing", "high", "negative",
     "Escalate to billing manager — SLA breach."),
]


def build_dataset() -> tuple[list[dspy.Example], list[dspy.Example]]:
    examples = [
        dspy.Example(
            ticket=ticket,
            category=category,
            urgency=urgency,
            sentiment=sentiment,
            recommendation=recommendation,
        ).with_inputs("ticket")
        for ticket, category, urgency, sentiment, recommendation in _RAW
    ]
    return examples[:20], examples[20:]   # trainset, testset
```

---

## pipeline_dspy.py

```python
"""pipeline_dspy.py — TriagePipeline DSPy module."""
import logging
from typing import Literal

import dspy

logger = logging.getLogger(__name__)


class ExtractTicketFields(dspy.Signature):
    """Extract routing fields from a support ticket."""
    ticket: str = dspy.InputField(desc="raw support ticket text")
    category: Literal["billing", "technical", "account", "other"] = dspy.OutputField()
    urgency: Literal["low", "medium", "high"] = dspy.OutputField()
    sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField()


class GenerateRecommendation(dspy.Signature):
    """Generate a one-sentence routing recommendation."""
    ticket: str = dspy.InputField()
    category: str = dspy.InputField()
    urgency: str = dspy.InputField()
    sentiment: str = dspy.InputField()
    recommendation: str = dspy.OutputField(
        desc="one sentence describing how to route and handle this ticket"
    )


class TriagePipeline(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.extract = dspy.ChainOfThought(ExtractTicketFields)
        self.recommend = dspy.Predict(GenerateRecommendation)

    def forward(self, ticket: str) -> dspy.Prediction:
        extracted = self.extract(ticket=ticket)
        logger.debug(
            "Extracted: category=%s urgency=%s sentiment=%s",
            extracted.category, extracted.urgency, extracted.sentiment,
        )
        rec = self.recommend(
            ticket=ticket,
            category=extracted.category,
            urgency=extracted.urgency,
            sentiment=extracted.sentiment,
        )
        return dspy.Prediction(
            category=extracted.category,
            urgency=extracted.urgency,
            sentiment=extracted.sentiment,
            recommendation=rec.recommendation,
        )
```

---

## pipeline_baseline.py

```python
"""pipeline_baseline.py — Manual prompt baseline using Anthropic SDK directly."""
import json
import logging
import os

import anthropic

logger = logging.getLogger(__name__)

_CLIENT: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _CLIENT
    if _CLIENT is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set")
        _CLIENT = anthropic.Anthropic(api_key=api_key)
    return _CLIENT


_SYSTEM = """You are a support ticket router. Given a ticket, extract:
- category: exactly one of billing, technical, account, other
- urgency: exactly one of low, medium, high
- sentiment: exactly one of positive, negative, neutral
- recommendation: one sentence describing how to route this ticket

Respond ONLY with valid JSON matching:
{"category": "...", "urgency": "...", "sentiment": "...", "recommendation": "..."}"""


def classify_baseline(ticket: str) -> dict[str, str]:
    client = _get_client()
    message = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=256,
        system=_SYSTEM,
        messages=[{"role": "user", "content": ticket}],
    )
    raw = message.content[0].text.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON from baseline: %s", raw[:100])
        data = {"category": "other", "urgency": "low", "sentiment": "neutral", "recommendation": ""}
    return data
```

---

## pipeline_metric.py

```python
"""pipeline_metric.py — Composite triage metric."""
from typing import Any

import dspy


def triage_metric(
    example: dspy.Example,
    pred: dspy.Prediction,
    trace: Any = None,
) -> float:
    """Weighted composite: category(0.4) + urgency(0.3) + sentiment(0.2) + rec_nonempty(0.1)."""
    def match(a: str, b: str) -> float:
        return 1.0 if a.strip().lower() == b.strip().lower() else 0.0

    category_score = match(getattr(pred, "category", ""), example.category)
    urgency_score = match(getattr(pred, "urgency", ""), example.urgency)
    sentiment_score = match(getattr(pred, "sentiment", ""), example.sentiment)
    rec = getattr(pred, "recommendation", "") or ""
    rec_score = 1.0 if rec.strip() else 0.0

    return (
        0.4 * category_score
        + 0.3 * urgency_score
        + 0.2 * sentiment_score
        + 0.1 * rec_score
    )
```

---

## pipeline_compile.py

```python
"""pipeline_compile.py — Compile and save TriagePipeline."""
import logging
from pathlib import Path

import dspy
from dspy.teleprompt import BootstrapFewShot

from pipeline_dspy import TriagePipeline
from pipeline_metric import triage_metric

logger = logging.getLogger(__name__)

COMPILED_PATH = Path("compiled_triage.json")


def compile_pipeline(trainset: list[dspy.Example]) -> dspy.Module:
    optimizer = BootstrapFewShot(
        metric=triage_metric,
        max_bootstrapped_demos=3,
    )
    compiled = optimizer.compile(TriagePipeline(), trainset=trainset)
    compiled.save(str(COMPILED_PATH))
    logger.info("Saved compiled pipeline to %s", COMPILED_PATH)
    return compiled


def load_pipeline() -> dspy.Module:
    if not COMPILED_PATH.exists():
        raise FileNotFoundError(f"Compiled pipeline not found: {COMPILED_PATH}")
    program = TriagePipeline()
    program.load(str(COMPILED_PATH))
    logger.info("Loaded compiled pipeline from %s", COMPILED_PATH)
    return program
```

---

## pipeline_eval.py

```python
"""pipeline_eval.py — Compare baseline, uncompiled DSPy, and compiled DSPy."""
import logging
import os

import dspy
from dspy.evaluate import Evaluate

from pipeline_baseline import classify_baseline
from pipeline_compile import compile_pipeline, load_pipeline
from pipeline_dspy import TriagePipeline
from pipeline_metric import triage_metric
from project_data import build_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _score_baseline(testset: list[dspy.Example]) -> float:
    total = 0.0
    for example in testset:
        raw = classify_baseline(example.ticket)
        pred = dspy.Prediction(**raw)
        total += triage_metric(example, pred)
    return (total / len(testset)) * 100


def run_comparison(testset: list[dspy.Example]) -> dict[str, float]:
    evaluator = Evaluate(
        devset=testset,
        metric=triage_metric,
        num_threads=1,
        display_progress=True,
        display_table=False,
    )

    baseline_score = _score_baseline(testset)
    logger.info("Baseline (manual prompt): %.1f%%", baseline_score)

    uncompiled = TriagePipeline()
    uncompiled_score = evaluator(uncompiled)
    logger.info("DSPy uncompiled: %.1f%%", uncompiled_score)

    compiled = load_pipeline()
    compiled_score = evaluator(compiled)
    logger.info("DSPy compiled: %.1f%%", compiled_score)

    results = {
        "baseline": baseline_score,
        "dspy_uncompiled": uncompiled_score,
        "dspy_compiled": compiled_score,
    }
    logger.info(
        "\n%-25s %s\n%-25s %.1f%%\n%-25s %.1f%%\n%-25s %.1f%%",
        "Method", "Score",
        "Baseline (manual prompt)", baseline_score,
        "DSPy uncompiled", uncompiled_score,
        "DSPy compiled", compiled_score,
    )
    return results


if __name__ == "__main__":
    lm = dspy.LM(
        model="anthropic/claude-haiku-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=256,
    )
    dspy.configure(lm=lm)

    trainset, testset = build_dataset()

    # Compile if not already saved
    try:
        load_pipeline()
    except FileNotFoundError:
        compile_pipeline(trainset)

    results = run_comparison(testset)
    assert results["dspy_compiled"] >= results["dspy_uncompiled"], (
        "Compiled should be >= uncompiled"
    )
```

---

## tests/test_pipeline.py

```python
"""tests/test_pipeline.py — Full pytest suite, no real LM calls."""
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import dspy
import pytest

from pipeline_dspy import TriagePipeline
from pipeline_metric import triage_metric


# ---------------------------------------------------------------------------
# triage_metric tests
# ---------------------------------------------------------------------------

def _make_example(**kwargs: str) -> dspy.Example:
    return dspy.Example(
        ticket="test",
        category=kwargs.get("category", "billing"),
        urgency=kwargs.get("urgency", "high"),
        sentiment=kwargs.get("sentiment", "negative"),
        recommendation="Escalate.",
    ).with_inputs("ticket")


def _make_pred(**kwargs: str) -> dspy.Prediction:
    return dspy.Prediction(
        category=kwargs.get("category", "billing"),
        urgency=kwargs.get("urgency", "high"),
        sentiment=kwargs.get("sentiment", "negative"),
        recommendation=kwargs.get("recommendation", "Escalate."),
    )


def test_metric_perfect_score() -> None:
    ex = _make_example()
    pred = _make_pred()
    assert triage_metric(ex, pred) == pytest.approx(1.0)


def test_metric_wrong_category() -> None:
    ex = _make_example(category="billing")
    pred = _make_pred(category="technical")
    score = triage_metric(ex, pred)
    assert score == pytest.approx(0.6)  # urgency(0.3)+sentiment(0.2)+rec(0.1)


def test_metric_empty_recommendation() -> None:
    ex = _make_example()
    pred = _make_pred(recommendation="")
    score = triage_metric(ex, pred)
    assert score == pytest.approx(0.9)  # rec penalty: -0.1


def test_metric_all_wrong() -> None:
    ex = _make_example(category="billing", urgency="high", sentiment="negative")
    pred = _make_pred(category="other", urgency="low", sentiment="positive", recommendation="")
    assert triage_metric(ex, pred) == pytest.approx(0.0)


def test_metric_case_insensitive() -> None:
    ex = _make_example(category="billing")
    pred = _make_pred(category="BILLING")
    assert triage_metric(ex, pred) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TriagePipeline tests
# ---------------------------------------------------------------------------

def test_triage_pipeline_forward() -> None:
    mock_extract_result = MagicMock()
    mock_extract_result.category = "billing"
    mock_extract_result.urgency = "high"
    mock_extract_result.sentiment = "negative"

    mock_recommend_result = MagicMock()
    mock_recommend_result.recommendation = "Escalate to billing."

    with (
        patch.object(dspy.ChainOfThought, "__call__", return_value=mock_extract_result),
        patch.object(dspy.Predict, "__call__", return_value=mock_recommend_result),
    ):
        pipe = TriagePipeline()
        result = pipe(ticket="I was charged twice.")

    assert result.category == "billing"
    assert result.urgency == "high"
    assert result.sentiment == "negative"
    assert result.recommendation == "Escalate to billing."


# ---------------------------------------------------------------------------
# compile_pipeline tests
# ---------------------------------------------------------------------------

def test_compile_pipeline_calls_bootstrap(tmp_path: Path) -> None:
    from pipeline_compile import compile_pipeline
    import pipeline_compile

    mock_compiled = MagicMock(spec=dspy.Module)
    mock_compiled.save = MagicMock()

    with (
        patch("pipeline_compile.BootstrapFewShot") as MockBSFS,
        patch.object(Path, "exists", return_value=False),
    ):
        instance = MagicMock()
        instance.compile.return_value = mock_compiled
        MockBSFS.return_value = instance

        original_path = pipeline_compile.COMPILED_PATH
        pipeline_compile.COMPILED_PATH = tmp_path / "compiled.json"

        trainset = [MagicMock(spec=dspy.Example) for _ in range(5)]
        result = compile_pipeline(trainset)

        pipeline_compile.COMPILED_PATH = original_path

    MockBSFS.assert_called_once_with(metric=triage_metric, max_bootstrapped_demos=3)
    instance.compile.assert_called_once()
    assert result is mock_compiled


# ---------------------------------------------------------------------------
# load_pipeline tests
# ---------------------------------------------------------------------------

def test_load_pipeline_from_json(tmp_path: Path) -> None:
    from pipeline_compile import load_pipeline
    import pipeline_compile

    # Write a minimal valid JSON that DSPy can load
    fake_json = {"predict": {}, "demos": []}
    json_path = tmp_path / "compiled.json"
    json_path.write_text(json.dumps(fake_json))

    original_path = pipeline_compile.COMPILED_PATH
    pipeline_compile.COMPILED_PATH = json_path

    with patch.object(TriagePipeline, "load", return_value=None) as mock_load:
        program = load_pipeline()
        mock_load.assert_called_once_with(str(json_path))

    pipeline_compile.COMPILED_PATH = original_path


def test_load_pipeline_missing_file(tmp_path: Path) -> None:
    from pipeline_compile import load_pipeline
    import pipeline_compile

    original_path = pipeline_compile.COMPILED_PATH
    pipeline_compile.COMPILED_PATH = tmp_path / "nonexistent.json"

    with pytest.raises(FileNotFoundError):
        load_pipeline()

    pipeline_compile.COMPILED_PATH = original_path


# ---------------------------------------------------------------------------
# run_comparison tests
# ---------------------------------------------------------------------------

def test_run_comparison_returns_all_keys() -> None:
    from pipeline_eval import run_comparison

    testset = [
        dspy.Example(
            ticket="test", category="billing", urgency="low",
            sentiment="neutral", recommendation="Route to billing."
        ).with_inputs("ticket")
    ]

    mock_pred = dspy.Prediction(
        category="billing", urgency="low", sentiment="neutral",
        recommendation="Route to billing."
    )

    with (
        patch("pipeline_eval.classify_baseline", return_value={
            "category": "billing", "urgency": "low",
            "sentiment": "neutral", "recommendation": "Route.",
        }),
        patch("pipeline_eval.Evaluate") as MockEvaluate,
        patch("pipeline_eval.load_pipeline") as mock_load,
    ):
        eval_instance = MagicMock()
        eval_instance.return_value = 85.0
        MockEvaluate.return_value = eval_instance

        mock_load.return_value = MagicMock(spec=dspy.Module)

        results = run_comparison(testset)

    assert "baseline" in results
    assert "dspy_uncompiled" in results
    assert "dspy_compiled" in results
```

---

## Key design decisions

**Why ChainOfThought for extraction and Predict for recommendation?**
Extraction benefits from scratchpad reasoning to decide between ambiguous categories.
Recommendation is straightforward given the extracted fields — no multi-step reasoning
needed, so plain Predict saves tokens.

**Why triage_metric weights category highest (0.4)?**
Routing to the wrong team is the most costly error — it delays resolution and
frustrates the customer. Urgency (0.3) determines SLA. Sentiment (0.2) affects
communication tone. Recommendation (0.1) is a free-text bonus.

**Why compile once and load, not compile per request?**
Compilation runs the training examples through the program dozens of times (for
BootstrapFewShot) or hundreds of times (for MIPRO). Running at request time would add
seconds to latency and burn API credits. Treat the compiled JSON like a trained model
artifact — compile in CI, deploy the artifact.
