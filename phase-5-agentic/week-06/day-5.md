# Phase 5 · Week 06 · Day 5 — Confidence Thresholds & Escalation to Human

> Objective: Build a router that extracts structured confidence scores from LLM output and routes requests autonomously, to a senior agent, or to a human based on configurable thresholds.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

## Concept

Agents make mistakes. The question is not whether an agent will be wrong, but whether it knows when it is wrong. **Confidence-based routing** is the mechanism that converts this uncertainty into a decision: handle it, escalate to a smarter agent, or hand off to a human.

A confidence score is a number (0.0–1.0) that the LLM produces alongside its answer, representing its self-assessed certainty. Three design principles matter here:

1. **Structured extraction beats free-text parsing.** Ask the model to return JSON with an explicit `confidence` field. Regex-parsing prose like "I'm fairly sure..." is brittle.
2. **Thresholds must be configurable.** A billing agent and a content-moderation agent have very different stakes. Hard-coded thresholds couple your code to your current risk tolerance.
3. **Calibration matters more than raw values.** A model that always returns 0.95 has zero routing utility. Prompt engineering (asking for confidence on a specific criterion, e.g. policy coverage) improves calibration.

### Routing decision tree

```
confidence >= AUTO_THRESHOLD      → handle autonomously
confidence >= ESCALATE_THRESHOLD  → escalate to senior agent
else                               → escalate to human
```

## How it works internally

Each call is a regular Claude `messages.create` with a structured-output prompt. The model returns a JSON object; `ConfidenceScore` (Pydantic) validates and parses it.

Pydantic v2 `model_validate` rejects invalid JSON at the boundary — no silent failures. If the LLM garbles the JSON, the router catches `ValidationError` and defaults to human escalation (fail-safe direction).

Threshold config lives in a `pydantic-settings` `Settings` class. Values come from environment variables (`CONFIDENCE_AUTO_THRESHOLD`, `CONFIDENCE_ESCALATE_THRESHOLD`), so the same binary works in staging (permissive) and production (strict) with no code change.

The `EscalationLevel` enum makes branching exhaustive: the Python type system forces you to handle every case, preventing silent fall-throughs when you add a new level later.

## Worked example

```python
"""
day5_confidence_router.py — Confidence-based escalation router using Claude.

Run:  python day5_confidence_router.py
Env:  ANTHROPIC_API_KEY
      CONFIDENCE_AUTO_THRESHOLD       (default: 0.85)
      CONFIDENCE_ESCALATE_THRESHOLD   (default: 0.60)
"""
import json
import logging
import os
from enum import Enum

import anthropic
from pydantic import BaseModel, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class ThresholdSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CONFIDENCE_")

    auto_threshold: float = 0.85      # >= this → handle autonomously
    escalate_threshold: float = 0.60  # >= this → senior agent; below → human

    @field_validator("auto_threshold", "escalate_threshold")
    @classmethod
    def must_be_unit_interval(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        return v


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class EscalationLevel(str, Enum):
    AUTONOMOUS = "autonomous"
    SENIOR_AGENT = "senior_agent"
    HUMAN = "human"


class ConfidenceScore(BaseModel):
    answer: str
    confidence: float          # 0.0–1.0
    reasoning: str             # why this confidence level
    missing_context: list[str] # what would raise confidence

    @field_validator("confidence")
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence {v} outside [0, 1]")
        return v


class RoutingDecision(BaseModel):
    level: EscalationLevel
    confidence_score: ConfidenceScore
    rationale: str


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

STRUCTURED_PROMPT = """\
You are a helpful assistant. Answer the user's question, then assess your own confidence.

Return ONLY valid JSON matching this schema (no markdown fences):
{{
  "answer": "<your answer>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<why you assigned this confidence>",
  "missing_context": ["<item1>", "<item2>"]
}}

confidence = 1.0 means you are certain; 0.0 means you are guessing.
Be calibrated: if you lack key context, score accordingly.

User question: {question}
"""


class ConfidenceRouter:
    def __init__(self, settings: ThresholdSettings | None = None) -> None:
        self.settings = settings or ThresholdSettings()
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = "claude-sonnet-5-5"
        log.info(
            "Router thresholds — auto: %.2f  escalate: %.2f",
            self.settings.auto_threshold,
            self.settings.escalate_threshold,
        )

    def _query_llm(self, question: str) -> ConfidenceScore:
        """Call the LLM and parse structured confidence output."""
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": STRUCTURED_PROMPT.format(question=question),
                }
            ],
        )
        raw = resp.content[0].text.strip()
        try:
            data = json.loads(raw)
            return ConfidenceScore.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            log.error("Failed to parse LLM response: %s — raw: %s", exc, raw[:200])
            # Fail safe: return a minimal score that triggers human escalation
            return ConfidenceScore(
                answer="[parse error — see logs]",
                confidence=0.0,
                reasoning="LLM response could not be parsed as structured JSON.",
                missing_context=["valid JSON response from model"],
            )

    def _decide_level(self, confidence: float) -> EscalationLevel:
        if confidence >= self.settings.auto_threshold:
            return EscalationLevel.AUTONOMOUS
        if confidence >= self.settings.escalate_threshold:
            return EscalationLevel.SENIOR_AGENT
        return EscalationLevel.HUMAN

    def route(self, question: str) -> RoutingDecision:
        """Main entry point: query LLM, score, and decide routing."""
        score = self._query_llm(question)
        level = self._decide_level(score.confidence)

        rationale_map = {
            EscalationLevel.AUTONOMOUS: f"Confidence {score.confidence:.2f} >= {self.settings.auto_threshold:.2f}; handling autonomously.",
            EscalationLevel.SENIOR_AGENT: (
                f"Confidence {score.confidence:.2f} in [{self.settings.escalate_threshold:.2f}, {self.settings.auto_threshold:.2f}); "
                "escalating to senior agent."
            ),
            EscalationLevel.HUMAN: (
                f"Confidence {score.confidence:.2f} < {self.settings.escalate_threshold:.2f}; "
                "escalating to human. Missing: " + ", ".join(score.missing_context or ["(unspecified)"])
            ),
        }

        decision = RoutingDecision(
            level=level,
            confidence_score=score,
            rationale=rationale_map[level],
        )
        log.info("Routing decision: %s (confidence=%.2f)", level.value, score.confidence)
        return decision


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _handle_autonomous(decision: RoutingDecision) -> None:
    print(f"[AUTO] Answer: {decision.confidence_score.answer}")


def _handle_senior_agent(decision: RoutingDecision) -> None:
    print(f"[SENIOR AGENT] Forwarding: {decision.confidence_score.answer}")
    print(f"  Context gaps: {decision.confidence_score.missing_context}")


def _handle_human(decision: RoutingDecision) -> None:
    print(f"[HUMAN ESCALATION] Please review manually.")
    print(f"  Tentative answer: {decision.confidence_score.answer}")
    print(f"  Reason for low confidence: {decision.confidence_score.reasoning}")


DISPATCH = {
    EscalationLevel.AUTONOMOUS: _handle_autonomous,
    EscalationLevel.SENIOR_AGENT: _handle_senior_agent,
    EscalationLevel.HUMAN: _handle_human,
}

if __name__ == "__main__":
    router = ConfidenceRouter()

    questions = [
        "What is the capital of France?",                          # should be high-confidence
        "What was the exact revenue of Acme Corp in Q3 2019?",     # should be low-confidence
        "What are common causes of a 500 error in a Flask app?",   # should be medium-confidence
    ]

    for q in questions:
        print(f"\n--- Question: {q}")
        decision = router.route(q)
        DISPATCH[decision.level](decision)
        print(f"  Rationale: {decision.rationale}")
```

Key design decisions:
- `ConfidenceScore` Pydantic model validates at parse time; `model_validate` rejects on schema mismatch.
- `ThresholdSettings` uses `pydantic-settings` `env_prefix` — zero code changes to reconfigure.
- Parse failure defaults to `confidence=0.0`, guaranteeing human escalation (fail-safe default).
- `EscalationLevel` is a `str` enum so it serialises cleanly to JSON for audit logs.
- `DISPATCH` dict replaces a cascade of `if/elif` — exhaustiveness is obvious at a glance.

## Exercises

1. The current router makes a single LLM call. Add an `ensemble_size: int = 1` parameter to `ConfidenceRouter`. When `ensemble_size > 1`, make that many independent calls and average the confidence scores before routing. Write a test that mocks three calls returning confidences `[0.9, 0.7, 0.5]` and asserts the mean `0.7` routes correctly.

2. The `_decide_level` method uses a flat threshold. Implement **hysteresis**: the system only upgrades its routing level if confidence exceeds `auto_threshold + hysteresis_margin`, and only downgrades below `escalate_threshold - hysteresis_margin`. Add `hysteresis_margin: float = 0.05` to `ThresholdSettings` and update the logic.

3. The router currently treats all questions the same. Add a `domain: str` parameter to `route()` and a `DomainThresholds` mapping (`dict[str, ThresholdSettings]`) to `ConfidenceRouter`. If the domain has a custom threshold, use it; otherwise fall back to the default. Write a test covering domain-specific override and the fallback.

4. Add a `RoutingAuditLogger` class that writes each `RoutingDecision` as a JSON line to a file. Wire it into `ConfidenceRouter.route()`. Write a test that captures the log output, routes two questions (mocked), and asserts both lines are valid JSON with the expected `level` field.

(Solutions: ../solutions/week-06/day-5.md)

## Self-check

- [ ] I can explain why structured JSON output is preferable to free-text confidence extraction
- [ ] I can implement a `ConfidenceScore` Pydantic model with field validation
- [ ] I can configure routing thresholds via environment variables using `pydantic-settings`
- [ ] I understand the fail-safe direction (parse error → human escalation) and why it matters
- [ ] I can write pytest tests that mock LLM responses to exercise each routing branch
- [ ] I can explain the difference between confidence scoring and confidence calibration
