# Solutions — Week 06 · Day 5 — Confidence Thresholds & Escalation

> These are reference solutions. Your implementation may differ and still be correct.

---

## Exercise 1 — Four-Tier Escalation by Confidence

Add a fourth tier below the original three: immediate rejection under 0.2.

```python
# src/agents/escalation.py
from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Any

import anthropic
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class EscalationTier(str, Enum):
    AUTONOMOUS = "autonomous"         # 0.60 – 1.00
    SENIOR_AGENT = "senior_agent"     # 0.40 – 0.59
    HUMAN = "human"                   # 0.20 – 0.39
    REJECTED = "rejected"             # 0.00 – 0.19


class AgentRequest(BaseModel):
    query: str
    context: dict[str, Any] = Field(default_factory=dict)


class EscalationDecision(BaseModel):
    tier: EscalationTier
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    request: AgentRequest

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {v}")
        return v


def classify_confidence(confidence: float) -> EscalationTier:
    """Map a confidence score to the appropriate escalation tier."""
    if confidence >= 0.60:
        return EscalationTier.AUTONOMOUS
    if confidence >= 0.40:
        return EscalationTier.SENIOR_AGENT
    if confidence >= 0.20:
        return EscalationTier.HUMAN
    return EscalationTier.REJECTED


def make_escalation_decision(
    request: AgentRequest,
    confidence: float,
) -> EscalationDecision:
    tier = classify_confidence(confidence)

    reason_map = {
        EscalationTier.AUTONOMOUS: "Confidence sufficient for autonomous handling.",
        EscalationTier.SENIOR_AGENT: "Moderate confidence; routing to senior agent for review.",
        EscalationTier.HUMAN: "Low confidence; human escalation required.",
        EscalationTier.REJECTED: (
            f"Confidence {confidence:.2f} is below minimum threshold 0.20. "
            "Request rejected — insufficient information to proceed safely."
        ),
    }

    decision = EscalationDecision(
        tier=tier,
        confidence=confidence,
        reason=reason_map[tier],
        request=request,
    )
    logger.info(
        "escalation_decision",
        extra={"tier": tier.value, "confidence": confidence, "query": request.query},
    )
    return decision


def handle_request(
    request: AgentRequest,
    confidence: float,
) -> dict[str, Any]:
    """Route request through four-tier escalation and return outcome."""
    decision = make_escalation_decision(request, confidence)

    if decision.tier == EscalationTier.AUTONOMOUS:
        logger.info("Handling autonomously: %s", request.query)
        return {"status": "handled", "tier": decision.tier.value, "reason": decision.reason}

    if decision.tier == EscalationTier.SENIOR_AGENT:
        logger.info("Routing to senior agent: %s", request.query)
        return {"status": "escalated", "tier": decision.tier.value, "reason": decision.reason}

    if decision.tier == EscalationTier.HUMAN:
        logger.warning("Human escalation required: %s", request.query)
        return {"status": "escalated", "tier": decision.tier.value, "reason": decision.reason}

    # REJECTED
    logger.error("Request rejected (confidence too low): %s", request.query)
    return {
        "status": "rejected",
        "tier": decision.tier.value,
        "reason": decision.reason,
        "explanation": (
            "This request cannot be processed automatically. "
            "Please provide more context or contact support directly."
        ),
    }
```

### Tests

```python
# tests/unit/test_escalation_tiers.py
from unittest.mock import patch

import pytest

from src.agents.escalation import (
    AgentRequest,
    EscalationTier,
    classify_confidence,
    handle_request,
    make_escalation_decision,
)


@pytest.fixture()
def sample_request() -> AgentRequest:
    return AgentRequest(query="What is the refund policy?")


@pytest.mark.parametrize(
    "confidence,expected_tier",
    [
        (1.00, EscalationTier.AUTONOMOUS),
        (0.80, EscalationTier.AUTONOMOUS),
        (0.60, EscalationTier.AUTONOMOUS),
        (0.59, EscalationTier.SENIOR_AGENT),
        (0.50, EscalationTier.SENIOR_AGENT),
        (0.40, EscalationTier.SENIOR_AGENT),
        (0.39, EscalationTier.HUMAN),
        (0.25, EscalationTier.HUMAN),
        (0.20, EscalationTier.HUMAN),
        (0.19, EscalationTier.REJECTED),
        (0.10, EscalationTier.REJECTED),
        (0.00, EscalationTier.REJECTED),
    ],
)
def test_classify_confidence_boundaries(confidence: float, expected_tier: EscalationTier) -> None:
    assert classify_confidence(confidence) == expected_tier


def test_autonomous_returns_handled(sample_request: AgentRequest) -> None:
    result = handle_request(sample_request, confidence=0.75)
    assert result["status"] == "handled"
    assert result["tier"] == EscalationTier.AUTONOMOUS.value


def test_senior_agent_returns_escalated(sample_request: AgentRequest) -> None:
    result = handle_request(sample_request, confidence=0.50)
    assert result["status"] == "escalated"
    assert result["tier"] == EscalationTier.SENIOR_AGENT.value


def test_human_returns_escalated(sample_request: AgentRequest) -> None:
    result = handle_request(sample_request, confidence=0.30)
    assert result["status"] == "escalated"
    assert result["tier"] == EscalationTier.HUMAN.value


def test_rejected_returns_rejected_with_explanation(sample_request: AgentRequest) -> None:
    result = handle_request(sample_request, confidence=0.10)
    assert result["status"] == "rejected"
    assert result["tier"] == EscalationTier.REJECTED.value
    assert "explanation" in result
    assert len(result["explanation"]) > 0


def test_decision_reason_contains_confidence_for_rejection(sample_request: AgentRequest) -> None:
    decision = make_escalation_decision(sample_request, confidence=0.05)
    assert "0.05" in decision.reason
```

---

## Exercise 2 — Confidence Calibration with ECE

Track predicted-vs-actual pairs, bucket them, compute Expected Calibration Error.

```python
# src/agents/calibration.py
from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import NamedTuple

logger = logging.getLogger(__name__)

NUM_BUCKETS = 10  # equal-width buckets: [0,0.1), [0.1,0.2), ..., [0.9,1.0]


class CalibrationSample(NamedTuple):
    predicted_confidence: float   # agent's stated confidence
    actual_success: bool          # did the task actually succeed?


@dataclass
class BucketStats:
    count: int = 0
    success_count: int = 0
    confidence_sum: float = 0.0

    @property
    def actual_success_rate(self) -> float:
        return self.success_count / self.count if self.count > 0 else 0.0

    @property
    def mean_confidence(self) -> float:
        return self.confidence_sum / self.count if self.count > 0 else 0.0


@dataclass
class CalibrationTracker:
    samples: list[CalibrationSample] = field(default_factory=list)

    def record(self, predicted_confidence: float, actual_success: bool) -> None:
        sample = CalibrationSample(
            predicted_confidence=max(0.0, min(1.0, predicted_confidence)),
            actual_success=actual_success,
        )
        self.samples.append(sample)
        logger.debug(
            "calibration_sample recorded",
            extra={"predicted": sample.predicted_confidence, "success": actual_success},
        )

    def _bucket_index(self, confidence: float) -> int:
        # Clamp to [0, NUM_BUCKETS-1]
        return min(int(confidence * NUM_BUCKETS), NUM_BUCKETS - 1)

    def compute_ece(self) -> float:
        """Expected Calibration Error = mean over buckets of |mean_conf - actual_rate|,
        weighted by fraction of samples in each bucket."""
        if not self.samples:
            return 0.0

        buckets: dict[int, BucketStats] = defaultdict(BucketStats)
        for sample in self.samples:
            idx = self._bucket_index(sample.predicted_confidence)
            b = buckets[idx]
            b.count += 1
            b.success_count += int(sample.actual_success)
            b.confidence_sum += sample.predicted_confidence

        n = len(self.samples)
        ece = sum(
            (stats.count / n) * abs(stats.mean_confidence - stats.actual_success_rate)
            for stats in buckets.values()
        )
        return ece

    def calibration_report(self) -> dict[str, object]:
        if not self.samples:
            return {"ece": 0.0, "n_samples": 0, "buckets": []}

        buckets: dict[int, BucketStats] = defaultdict(BucketStats)
        for sample in self.samples:
            idx = self._bucket_index(sample.predicted_confidence)
            b = buckets[idx]
            b.count += 1
            b.success_count += int(sample.actual_success)
            b.confidence_sum += sample.predicted_confidence

        ece = self.compute_ece()
        bucket_rows = [
            {
                "bucket": i,
                "range": f"[{i/NUM_BUCKETS:.1f}, {(i+1)/NUM_BUCKETS:.1f})",
                "n": stats.count,
                "mean_predicted": round(stats.mean_confidence, 4),
                "actual_success_rate": round(stats.actual_success_rate, 4),
                "abs_error": round(abs(stats.mean_confidence - stats.actual_success_rate), 4),
            }
            for i, stats in sorted(buckets.items())
        ]

        report = {
            "ece": round(ece, 6),
            "n_samples": len(self.samples),
            "buckets": bucket_rows,
        }
        logger.info("calibration_report: ECE=%.6f n=%d", ece, len(self.samples))
        return report
```

### Tests

```python
# tests/unit/test_calibration.py
import math

import pytest

from src.agents.calibration import CalibrationTracker


def test_ece_perfectly_calibrated() -> None:
    tracker = CalibrationTracker()
    # Predict 0.8, always succeed → ECE near 0.2 (overconfident bucket)
    # Perfect calibration: predict 1.0, always succeed
    for _ in range(100):
        tracker.record(1.0, actual_success=True)
    assert tracker.compute_ece() == pytest.approx(0.0, abs=1e-6)


def test_ece_worst_case() -> None:
    tracker = CalibrationTracker()
    # Predict 1.0, always fail → ECE = 1.0
    for _ in range(100):
        tracker.record(1.0, actual_success=False)
    assert tracker.compute_ece() == pytest.approx(1.0, abs=1e-6)


def test_ece_moderate() -> None:
    tracker = CalibrationTracker()
    # Predict 0.7, succeed 70% of the time → near zero ECE for that bucket
    for i in range(100):
        tracker.record(0.75, actual_success=(i < 75))
    ece = tracker.compute_ece()
    assert ece < 0.1


def test_empty_tracker_returns_zero() -> None:
    tracker = CalibrationTracker()
    assert tracker.compute_ece() == 0.0
    report = tracker.calibration_report()
    assert report["ece"] == 0.0
    assert report["n_samples"] == 0


def test_calibration_report_structure() -> None:
    tracker = CalibrationTracker()
    tracker.record(0.65, True)
    tracker.record(0.65, False)
    report = tracker.calibration_report()
    assert "ece" in report
    assert "n_samples" in report
    assert report["n_samples"] == 2
    assert isinstance(report["buckets"], list)


def test_confidence_clamped() -> None:
    tracker = CalibrationTracker()
    tracker.record(1.5, True)   # clamped to 1.0
    tracker.record(-0.1, False) # clamped to 0.0
    assert len(tracker.samples) == 2
    assert tracker.samples[0].predicted_confidence == 1.0
    assert tracker.samples[1].predicted_confidence == 0.0
```

---

## Exercise 3 — FallbackChain with Pydantic v2

Primary → Secondary → Human, each hop guarded by a confidence threshold.

```python
# src/agents/fallback_chain.py
from __future__ import annotations

import logging
import os
from typing import Any, Protocol

import anthropic
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class AgentResponse(BaseModel):
    agent_name: str
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FallbackResult(BaseModel):
    final_response: AgentResponse | None = None
    escalated_to_human: bool = False
    attempts: list[AgentResponse] = Field(default_factory=list)
    reason: str = ""


# ---------------------------------------------------------------------------
# Agent protocol — enables dependency injection / mocking in tests
# ---------------------------------------------------------------------------

class AgentProtocol(Protocol):
    @property
    def name(self) -> str: ...

    def run(self, query: str) -> AgentResponse: ...


# ---------------------------------------------------------------------------
# Concrete LLM-backed agent (real implementation)
# ---------------------------------------------------------------------------

class LLMAgent:
    """Thin wrapper around Anthropic API — mocked in tests."""

    def __init__(self, agent_name: str, model: str = "claude-sonnet-5-5") -> None:
        self._name = agent_name
        self._model = model
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    @property
    def name(self) -> str:
        return self._name

    def run(self, query: str) -> AgentResponse:
        logger.info("LLMAgent '%s' running query: %s", self._name, query)
        message = self._client.messages.create(
            model=self._model,
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Answer the following query and rate your confidence "
                        f"(0-1) in JSON: {{\"answer\": \"...\", \"confidence\": 0.0}}\n\n{query}"
                    ),
                }
            ],
        )
        # Minimal parsing — real impl would use structured output
        import json
        raw = message.content[0].text
        try:
            parsed = json.loads(raw)
            return AgentResponse(
                agent_name=self._name,
                answer=parsed["answer"],
                confidence=float(parsed["confidence"]),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return AgentResponse(
                agent_name=self._name,
                answer=raw,
                confidence=0.3,  # default low confidence on parse error
            )


# ---------------------------------------------------------------------------
# FallbackChain
# ---------------------------------------------------------------------------

class FallbackChain:
    """
    Try primary agent; if confidence < threshold, try secondary.
    If secondary also < threshold, escalate to human.
    """

    def __init__(
        self,
        primary: AgentProtocol,
        secondary: AgentProtocol,
        confidence_threshold: float = 0.60,
    ) -> None:
        if not 0.0 < confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in (0, 1]")
        self.primary = primary
        self.secondary = secondary
        self.confidence_threshold = confidence_threshold

    def run(self, query: str) -> FallbackResult:
        result = FallbackResult()

        # --- Primary attempt ---
        primary_response = self.primary.run(query)
        result.attempts.append(primary_response)
        logger.info(
            "Primary '%s' confidence=%.2f",
            self.primary.name,
            primary_response.confidence,
        )

        if primary_response.confidence >= self.confidence_threshold:
            result.final_response = primary_response
            result.reason = f"Primary agent '{self.primary.name}' confidence sufficient."
            return result

        # --- Secondary attempt ---
        logger.info(
            "Primary confidence %.2f below threshold %.2f; trying secondary '%s'",
            primary_response.confidence,
            self.confidence_threshold,
            self.secondary.name,
        )
        secondary_response = self.secondary.run(query)
        result.attempts.append(secondary_response)
        logger.info(
            "Secondary '%s' confidence=%.2f",
            self.secondary.name,
            secondary_response.confidence,
        )

        if secondary_response.confidence >= self.confidence_threshold:
            result.final_response = secondary_response
            result.reason = f"Secondary agent '{self.secondary.name}' confidence sufficient."
            return result

        # --- Human escalation ---
        logger.warning(
            "Both agents below threshold; escalating to human. "
            "Primary=%.2f, Secondary=%.2f",
            primary_response.confidence,
            secondary_response.confidence,
        )
        result.escalated_to_human = True
        result.reason = (
            f"Both primary (conf={primary_response.confidence:.2f}) and "
            f"secondary (conf={secondary_response.confidence:.2f}) below "
            f"threshold {self.confidence_threshold}. Human review required."
        )
        return result
```

### Tests

```python
# tests/unit/test_fallback_chain.py
from unittest.mock import MagicMock

import pytest

from src.agents.fallback_chain import (
    AgentResponse,
    FallbackChain,
    FallbackResult,
)


def _make_agent(name: str, confidence: float) -> MagicMock:
    agent = MagicMock()
    agent.name = name
    agent.run.return_value = AgentResponse(
        agent_name=name, answer=f"Answer from {name}", confidence=confidence
    )
    return agent


def test_primary_succeeds() -> None:
    primary = _make_agent("primary", confidence=0.90)
    secondary = _make_agent("secondary", confidence=0.50)
    chain = FallbackChain(primary, secondary, confidence_threshold=0.60)

    result = chain.run("What is the refund policy?")

    assert result.final_response is not None
    assert result.final_response.agent_name == "primary"
    assert not result.escalated_to_human
    assert len(result.attempts) == 1
    secondary.run.assert_not_called()


def test_primary_fails_secondary_succeeds() -> None:
    primary = _make_agent("primary", confidence=0.40)
    secondary = _make_agent("secondary", confidence=0.75)
    chain = FallbackChain(primary, secondary, confidence_threshold=0.60)

    result = chain.run("Complex billing question")

    assert result.final_response is not None
    assert result.final_response.agent_name == "secondary"
    assert not result.escalated_to_human
    assert len(result.attempts) == 2


def test_both_fail_human_escalation() -> None:
    primary = _make_agent("primary", confidence=0.30)
    secondary = _make_agent("secondary", confidence=0.25)
    chain = FallbackChain(primary, secondary, confidence_threshold=0.60)

    result = chain.run("Ambiguous legal question")

    assert result.final_response is None
    assert result.escalated_to_human is True
    assert len(result.attempts) == 2
    assert "Human review required" in result.reason


def test_threshold_boundary_exact_match() -> None:
    primary = _make_agent("primary", confidence=0.60)
    secondary = _make_agent("secondary", confidence=0.50)
    chain = FallbackChain(primary, secondary, confidence_threshold=0.60)

    result = chain.run("Boundary test query")
    assert result.final_response is not None
    assert result.final_response.agent_name == "primary"


def test_invalid_threshold_raises() -> None:
    primary = _make_agent("primary", 0.5)
    secondary = _make_agent("secondary", 0.5)
    with pytest.raises(ValueError):
        FallbackChain(primary, secondary, confidence_threshold=0.0)


def test_attempts_recorded_on_human_escalation() -> None:
    primary = _make_agent("primary", 0.10)
    secondary = _make_agent("secondary", 0.15)
    chain = FallbackChain(primary, secondary, confidence_threshold=0.60)
    result = chain.run("query")
    assert len(result.attempts) == 2
    assert result.attempts[0].agent_name == "primary"
    assert result.attempts[1].agent_name == "secondary"
```
