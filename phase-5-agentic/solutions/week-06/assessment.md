# Solutions — Week 06 Assessment

> These are reference solutions. Your implementation may differ and still be correct.

---

## Conceptual Questions

### Q1 — Rule-based router vs. LLM-based router

A **rule-based router** dispatches messages using deterministic logic: keyword matching, regex patterns, or decision trees. No model call is made; the mapping is explicit and auditable. Latency is near-zero; accuracy depends on how well the rules cover the domain.

An **LLM-based router** asks a language model to classify the intent of a free-text message. It generalises to phrasing not anticipated at design time and handles ambiguous or multilingual input. The cost is latency, token spend, and non-determinism.

**Choose rule-based when:** the intent space is small, enumerable, and stable (e.g., `"cancel"` / `"status"` / `"help"`); regulatory requirements demand full auditability; sub-millisecond routing matters.

**Choose LLM-based when:** user phrasing is unpredictable, the intent taxonomy is large, or you need to extract structured data (entities, slots) alongside classification.

**Hybrid pattern (common in production):** fast rule-based first pass for known high-traffic intents; LLM fallback only for the unmatched tail.

---

### Q2 — How does the orchestrator know which worker to delegate to?

The orchestrator maintains a **specialist registry** — a mapping from intent label (or capability tag) to a worker/agent reference. After the router produces a `RoutingResult`, the orchestrator looks up the intent in the registry and dispatches. Three common strategies:

1. **Static registry:** populated at startup; key = intent string.
2. **Capability-advertised registry:** workers register themselves with a set of capability tags; orchestrator scores each worker against the sub-task and picks the highest scorer.
3. **LLM-planned dispatch:** the orchestrator asks an LLM to produce a delegation plan as structured output (e.g., a list of `{worker, task}` pairs), then executes it.

For simple systems, strategy 1 is sufficient. Strategy 3 is used in ReAct-style multi-agent loops.

---

### Q3 — Parallel vs. sequential delegation

**Sequential delegation:** Worker B starts only after Worker A finishes (and A's output may be B's input). Use when tasks have data dependencies — e.g., "fetch order details" must complete before "issue refund" begins, because the refund needs the order ID.

**Parallel delegation:** Multiple workers start simultaneously. Use when tasks are independent — e.g., an orchestrator answering "compare shipping costs from three carriers" can dispatch to Carrier-A, Carrier-B, and Carrier-C agents concurrently and aggregate results. This cuts total latency from `O(n)` to `O(max)`.

---

### Q4 — Dynamic tool selection and ToolRegistry vs. if/else dispatch

In **dynamic tool selection**, the agent (or orchestrator) decides at runtime which tools are appropriate for the current task, rather than having that decision baked into code.

A **ToolRegistry** stores tools as first-class objects with metadata (name, description, input schema, cost/latency estimates). At runtime the agent can:
- enumerate all registered tools,
- embed tool descriptions into an LLM prompt and let the model pick,
- or apply capability matching rules.

**Hardcoded if/else dispatch:**
```python
if intent == "billing":
    billing_agent.run(msg)
elif intent == "technical":
    tech_agent.run(msg)
```
This is brittle: adding a new tool requires a code change, no capability metadata exists, and there is no mechanism for the agent to reason about tool suitability.

A registry decouples the tool inventory from dispatch logic, enables runtime introspection, and allows tools to be added, removed, or toggled without touching orchestration code.

---

### Q5 — Human-in-the-loop approval gate

A HITL approval gate is a checkpoint in an agent workflow where execution pauses and a human must explicitly approve or reject before the agent continues. It is used for actions whose consequences are difficult or impossible to reverse.

**Three categories warranting HITL gates:**

1. **Financial transactions** — issuing refunds, processing payments, applying credits, modifying billing.
2. **Destructive data operations** — deleting user accounts, bulk-deleting records, exporting sensitive PII.
3. **External integrations with side effects** — sending customer-facing emails, posting to social media, triggering deployments to production infrastructure.

---

### Q6 — Confidence threshold

A confidence threshold is a scalar cut-off: if the agent's self-reported confidence in a decision falls below it, the action is not taken autonomously — the request is escalated, deferred, or rejected.

**Too low a threshold** → the agent escalates or rejects too many requests that it could handle correctly; human review becomes the bottleneck; throughput collapses.

**Too high a threshold** → the agent autonomously handles requests for which it is actually unreliable; error rate climbs; trust is damaged. Miscalibrated agents (overconfident) are especially dangerous here because their stated confidence is inflated relative to actual accuracy.

The correct threshold is calibrated empirically against historical (predicted confidence, actual outcome) pairs — this is what the Expected Calibration Error metric measures.

---

### Q7 — Escalation in agent systems

Escalation is the process of transferring a request to a handler with greater authority, expertise, or accountability when the current handler cannot or should not resolve it alone.

**Example three-level escalation path:**

| Level | Handler | Trigger |
|---|---|---|
| L1 | Autonomous agent | Confidence >= 0.60 |
| L2 | Senior agent / specialist | 0.30 <= Confidence < 0.60 |
| L3 | Human reviewer | Confidence < 0.30, or action is in high-risk category |

Each level has a corresponding timeout; if the L2/L3 handler does not respond within the SLA, the request escalates further or auto-expires.

---

### Q8 — Audit log vs. application logging

**Application logging** records operational events for debugging and monitoring: exceptions, performance timings, service health. Entries may be dropped, rotated, or sampled under load. The audience is the on-call engineer.

**Audit logging** records who requested what action, when, what decision was made, and by whom. Entries must be tamper-evident, complete, and retained for compliance. The audience is compliance officers, security teams, and legal. Audit logs cannot be sampled or silently dropped.

**Why both are needed in an approval workflow:**

- Application logs tell you *why* the workflow engine crashed or was slow.
- Audit logs tell you *that* a refund of $5,000 was approved by `approver@company.com` at 14:32 UTC on a specific date — the kind of record required for SOX, GDPR, or ISO 27001 compliance.

Neither replaces the other: operational and compliance concerns are orthogonal.

---

## Coding Challenge — Mini Routing + HITL System

```python
# challenge/mini_routing_hitl.py
"""
Mini routing + HITL system.
All LLM calls mocked in tests.
"""
from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Any

import anthropic
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Intent(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    REFUND = "refund"
    GENERAL = "general"


class RoutedRequest(BaseModel):
    message: str
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)


class HITLDecision(BaseModel):
    requires_human: bool
    reason: str


class FinalResponse(BaseModel):
    answer: str
    intent: Intent
    confidence: float
    human_involved: bool


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.60
SENSITIVE_INTENTS = {Intent.REFUND}


def route_message(message: str, client: anthropic.Anthropic) -> RoutedRequest:
    import json
    response = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=128,
        system='Classify intent as one of billing/technical/refund/general. Reply JSON: {"intent":"...","confidence":0.0-1.0}',
        messages=[{"role": "user", "content": message}],
    )
    parsed = json.loads(response.content[0].text)
    return RoutedRequest(
        message=message,
        intent=Intent(parsed["intent"]),
        confidence=float(parsed["confidence"]),
    )


def hitl_gate(routed: RoutedRequest) -> HITLDecision:
    if routed.confidence < CONFIDENCE_THRESHOLD:
        return HITLDecision(
            requires_human=True,
            reason=f"Confidence {routed.confidence:.2f} below threshold {CONFIDENCE_THRESHOLD}",
        )
    if routed.intent in SENSITIVE_INTENTS:
        return HITLDecision(
            requires_human=True,
            reason=f"Intent '{routed.intent.value}' requires human approval.",
        )
    return HITLDecision(requires_human=False, reason="Autonomous handling approved.")


def specialist_answer(routed: RoutedRequest, client: anthropic.Anthropic) -> str:
    response = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=256,
        system=f"You are a {routed.intent.value} specialist.",
        messages=[{"role": "user", "content": routed.message}],
    )
    return response.content[0].text


def handle(message: str, client: anthropic.Anthropic) -> FinalResponse:
    routed = route_message(message, client)
    gate = hitl_gate(routed)

    if gate.requires_human:
        logger.warning("HITL required: %s", gate.reason)
        return FinalResponse(
            answer=f"Your request has been forwarded for human review. Reason: {gate.reason}",
            intent=routed.intent,
            confidence=routed.confidence,
            human_involved=True,
        )

    answer = specialist_answer(routed, client)
    return FinalResponse(
        answer=answer,
        intent=routed.intent,
        confidence=routed.confidence,
        human_involved=False,
    )
```

### Tests

```python
# tests/unit/test_mini_routing_hitl.py
import json
from unittest.mock import MagicMock, call

import pytest

from challenge.mini_routing_hitl import (
    Intent,
    FinalResponse,
    handle,
    hitl_gate,
    route_message,
    RoutedRequest,
)


def _mock_client(router_payload: dict, specialist_text: str = "Answer here") -> MagicMock:
    def _msg(text: str) -> MagicMock:
        b = MagicMock()
        b.text = text
        m = MagicMock()
        m.content = [b]
        return m

    client = MagicMock()
    client.messages.create.side_effect = [
        _msg(json.dumps(router_payload)),
        _msg(specialist_text),
    ]
    return client


def test_billing_high_confidence_autonomous() -> None:
    client = _mock_client({"intent": "billing", "confidence": 0.90}, "Billing info here.")
    result = handle("What is my bill?", client)
    assert result.intent == Intent.BILLING
    assert not result.human_involved
    assert result.answer == "Billing info here."


def test_low_confidence_triggers_hitl() -> None:
    client = _mock_client({"intent": "general", "confidence": 0.35})
    result = handle("I dunno what I need", client)
    assert result.human_involved is True
    assert "human review" in result.answer.lower()


def test_refund_always_requires_human() -> None:
    client = _mock_client({"intent": "refund", "confidence": 0.95})
    result = handle("Refund my last purchase", client)
    assert result.human_involved is True
    assert "refund" in result.answer.lower() or "human review" in result.answer.lower()


def test_hitl_gate_logic_directly() -> None:
    req_low = RoutedRequest(message="?", intent=Intent.GENERAL, confidence=0.30)
    assert hitl_gate(req_low).requires_human is True

    req_high = RoutedRequest(message="?", intent=Intent.BILLING, confidence=0.85)
    assert hitl_gate(req_high).requires_human is False

    req_refund = RoutedRequest(message="?", intent=Intent.REFUND, confidence=0.95)
    assert hitl_gate(req_refund).requires_human is True
```

---

## Grading Notes

### Rubric dimensions

| Dimension | Weight | What to look for |
|---|---|---|
| Conceptual accuracy | 40% | Correct definitions, appropriate use of terminology, meaningful distinctions (not just restating the question) |
| Coding challenge — correctness | 25% | Routes correctly for all three scenarios; HITL gate fires on both low-confidence AND sensitive intent; mocked LLM means no real API calls |
| Coding challenge — code quality | 20% | Pydantic v2 models, type hints throughout, `logging` not `print`, `os.getenv` for API key, no hardcoded secrets |
| Testing | 15% | Tests cover happy path, low-confidence path, refund/sensitive path; mocks via `unittest.mock.patch`; assertions on `human_involved` field |

**Common deductions:**

- Using `print()` instead of `logging` (-3 pts per occurrence)
- Hardcoding `ANTHROPIC_API_KEY` in source (-10 pts, security violation)
- Using `claude-3-*` model IDs instead of `claude-sonnet-5-5` or `claude-opus-5-5` (-5 pts)
- No tests for the low-confidence escalation path (-8 pts)
- Missing type hints on function signatures (-2 pts per function)
- Pydantic v1 syntax (`@validator` instead of `@field_validator`) (-3 pts)
