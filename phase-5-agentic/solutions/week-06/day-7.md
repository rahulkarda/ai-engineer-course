# Solutions — Week 06 · Day 7 — Week Project Integration

> These are reference solutions. Your implementation may differ and still be correct.

---

## 1 — Wiring the Router to the Orchestrator

The router classifies intent; the orchestrator dispatches to specialists. They connect through a shared `RoutingResult` model.

```python
# src/orchestrator/glue.py
"""
Integration glue: router → orchestrator → specialist.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import anthropic
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared models
# ---------------------------------------------------------------------------

class RoutingResult(BaseModel):
    intent: str          # e.g. "billing", "technical", "refund"
    confidence: float    # 0.0 – 1.0
    raw_message: str


class OrchestratorResponse(BaseModel):
    intent: str
    confidence: float
    specialist_used: str
    answer: str
    escalated: bool = False


# ---------------------------------------------------------------------------
# Router (classifies intent)
# ---------------------------------------------------------------------------

class IntentRouter:
    """
    Calls LLM to classify user intent.
    Returns (intent, confidence).
    Replace client with a mock in tests.
    """

    def __init__(self, client: anthropic.Anthropic | None = None) -> None:
        self._client = client or anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

    def route(self, message: str) -> RoutingResult:
        logger.info("IntentRouter.route: %s", message[:80])
        response = self._client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=128,
            system=(
                "Classify the user message into one of: billing, technical, refund, general. "
                'Reply with JSON: {"intent": "<intent>", "confidence": <0-1>}'
            ),
            messages=[{"role": "user", "content": message}],
        )
        import json
        raw = response.content[0].text
        parsed = json.loads(raw)
        return RoutingResult(
            intent=parsed["intent"],
            confidence=float(parsed["confidence"]),
            raw_message=message,
        )


# ---------------------------------------------------------------------------
# Specialist registry
# ---------------------------------------------------------------------------

class SpecialistRegistry:
    def __init__(self) -> None:
        self._specialists: dict[str, "Specialist"] = {}

    def register(self, intent: str, specialist: "Specialist") -> None:
        self._specialists[intent] = specialist

    def get(self, intent: str) -> "Specialist | None":
        return self._specialists.get(intent) or self._specialists.get("general")


class Specialist:
    def __init__(self, name: str, client: anthropic.Anthropic | None = None) -> None:
        self.name = name
        self._client = client or anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

    def answer(self, message: str) -> str:
        logger.info("Specialist '%s' answering: %s", self.name, message[:80])
        response = self._client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=256,
            system=f"You are a {self.name} specialist. Answer concisely.",
            messages=[{"role": "user", "content": message}],
        )
        return response.content[0].text


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    def __init__(
        self,
        router: IntentRouter,
        registry: SpecialistRegistry,
        confidence_threshold: float = 0.60,
    ) -> None:
        self._router = router
        self._registry = registry
        self._confidence_threshold = confidence_threshold

    def handle(self, message: str) -> OrchestratorResponse:
        routing = self._router.route(message)
        logger.info(
            "Routed intent=%s confidence=%.2f", routing.intent, routing.confidence
        )

        if routing.confidence < self._confidence_threshold:
            logger.warning("Low confidence (%.2f) — escalating", routing.confidence)
            return OrchestratorResponse(
                intent=routing.intent,
                confidence=routing.confidence,
                specialist_used="none",
                answer="Escalated to human due to low routing confidence.",
                escalated=True,
            )

        specialist = self._registry.get(routing.intent)
        if specialist is None:
            raise ValueError(f"No specialist for intent '{routing.intent}'")

        answer = specialist.answer(message)
        return OrchestratorResponse(
            intent=routing.intent,
            confidence=routing.confidence,
            specialist_used=specialist.name,
            answer=answer,
        )
```

---

## 2 — Integrating Confidence Routing with HITL Gates

Decision tree that combines confidence tier classification and approval gate logic.

```python
# src/orchestrator/hitl_gate.py
from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ConfidenceTier(str, Enum):
    HIGH = "high"         # >= 0.80: autonomous, no gate
    MEDIUM = "medium"     # 0.60 – 0.79: autonomous, but log for audit
    LOW = "low"           # 0.40 – 0.59: soft HITL (notify human, proceed if no veto)
    VERY_LOW = "very_low" # 0.20 – 0.39: hard HITL (block until approved)
    REJECT = "reject"     # < 0.20: immediate rejection


SENSITIVE_ACTIONS = {"refund", "account_delete", "data_export", "credit_issue"}


class GateDecision(BaseModel):
    tier: ConfidenceTier
    requires_approval: bool
    hard_block: bool        # True = must wait; False = can proceed optimistically
    reason: str


def classify_tier(confidence: float) -> ConfidenceTier:
    if confidence >= 0.80:
        return ConfidenceTier.HIGH
    if confidence >= 0.60:
        return ConfidenceTier.MEDIUM
    if confidence >= 0.40:
        return ConfidenceTier.LOW
    if confidence >= 0.20:
        return ConfidenceTier.VERY_LOW
    return ConfidenceTier.REJECT


def evaluate_gate(
    intent: str,
    confidence: float,
    action_type: str | None = None,
) -> GateDecision:
    """
    Combine confidence tier + sensitivity of the action to decide
    whether HITL approval is needed and whether to hard-block.
    """
    tier = classify_tier(confidence)
    is_sensitive = action_type in SENSITIVE_ACTIONS

    if tier == ConfidenceTier.REJECT:
        return GateDecision(
            tier=tier,
            requires_approval=False,
            hard_block=True,
            reason="Confidence below minimum threshold. Request rejected.",
        )

    if tier == ConfidenceTier.VERY_LOW:
        return GateDecision(
            tier=tier,
            requires_approval=True,
            hard_block=True,
            reason=f"Low confidence ({confidence:.2f}) — hard HITL block until approved.",
        )

    if tier == ConfidenceTier.LOW or (tier == ConfidenceTier.MEDIUM and is_sensitive):
        return GateDecision(
            tier=tier,
            requires_approval=True,
            hard_block=False,
            reason=(
                f"Soft HITL gate — {'sensitive action' if is_sensitive else 'medium confidence'}. "
                "Human notified; proceeding optimistically."
            ),
        )

    # HIGH or MEDIUM non-sensitive
    return GateDecision(
        tier=tier,
        requires_approval=False,
        hard_block=False,
        reason="Confidence sufficient; proceeding autonomously.",
    )
```

---

## 3 — End-to-End Integration Test (all LLM calls mocked)

```python
# tests/integration/test_e2e_routing.py
"""
Full path: user message → classify intent → route → confidence check
           → approval gate → specialist → response.
All Anthropic API calls mocked with unittest.mock.patch.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.orchestrator.glue import (
    IntentRouter,
    Orchestrator,
    Specialist,
    SpecialistRegistry,
)
from src.orchestrator.hitl_gate import evaluate_gate, GateDecision
from src.workflow.engine import WorkflowEngine
from src.workflow.sequential_approval import ApprovalRequest, ApprovalState


# ---------------------------------------------------------------------------
# Helpers — build mock Anthropic message objects
# ---------------------------------------------------------------------------

def _mock_message(text: str) -> MagicMock:
    content_block = MagicMock()
    content_block.text = text
    msg = MagicMock()
    msg.content = [content_block]
    return msg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestE2ERouting:

    @patch("src.orchestrator.glue.anthropic.Anthropic")
    def test_high_confidence_billing_query_handled_autonomously(
        self, mock_anthropic_cls: MagicMock
    ) -> None:
        # --- Mock setup ---
        client = MagicMock()
        mock_anthropic_cls.return_value = client

        # First call: router returns billing intent, high confidence
        router_reply = _mock_message(json.dumps({"intent": "billing", "confidence": 0.92}))
        # Second call: specialist answers
        specialist_reply = _mock_message("Your billing cycle resets on the 1st of each month.")
        client.messages.create.side_effect = [router_reply, specialist_reply]

        # --- Build system ---
        router = IntentRouter(client=client)
        registry = SpecialistRegistry()
        registry.register("billing", Specialist("billing", client=client))
        orchestrator = Orchestrator(router, registry, confidence_threshold=0.60)

        # --- Execute ---
        result = orchestrator.handle("When does my billing cycle reset?")

        # --- Assert ---
        assert result.intent == "billing"
        assert result.confidence == pytest.approx(0.92)
        assert result.specialist_used == "billing"
        assert not result.escalated
        assert "billing" in result.answer.lower() or len(result.answer) > 0

    @patch("src.orchestrator.glue.anthropic.Anthropic")
    def test_low_confidence_triggers_escalation(
        self, mock_anthropic_cls: MagicMock
    ) -> None:
        client = MagicMock()
        mock_anthropic_cls.return_value = client
        # Router returns very low confidence
        client.messages.create.return_value = _mock_message(
            json.dumps({"intent": "general", "confidence": 0.25})
        )

        router = IntentRouter(client=client)
        registry = SpecialistRegistry()
        registry.register("general", Specialist("general", client=client))
        orchestrator = Orchestrator(router, registry, confidence_threshold=0.60)

        result = orchestrator.handle("I have some kind of issue maybe?")

        assert result.escalated is True
        assert result.specialist_used == "none"

    @patch("src.orchestrator.glue.anthropic.Anthropic")
    def test_refund_requires_approval_gate(
        self, mock_anthropic_cls: MagicMock
    ) -> None:
        """Refund with medium confidence should trigger soft HITL gate."""
        client = MagicMock()
        mock_anthropic_cls.return_value = client

        router_reply = _mock_message(json.dumps({"intent": "refund", "confidence": 0.72}))
        specialist_reply = _mock_message("Refund issued for order #123.")
        client.messages.create.side_effect = [router_reply, specialist_reply]

        router = IntentRouter(client=client)
        registry = SpecialistRegistry()
        registry.register("refund", Specialist("refund", client=client))
        orchestrator = Orchestrator(router, registry, confidence_threshold=0.60)

        result = orchestrator.handle("I need a refund for order #123")
        # Check gate separately
        gate = evaluate_gate(result.intent, result.confidence, action_type="refund")

        assert result.intent == "refund"
        assert gate.requires_approval is True   # sensitive action
        assert gate.hard_block is False         # medium confidence → soft gate

    @patch("src.orchestrator.glue.anthropic.Anthropic")
    def test_full_approval_workflow_integration(
        self, mock_anthropic_cls: MagicMock
    ) -> None:
        """Simulate end-to-end with an approval workflow for a flagged action."""
        client = MagicMock()
        mock_anthropic_cls.return_value = client

        # Router
        client.messages.create.return_value = _mock_message(
            json.dumps({"intent": "account_delete", "confidence": 0.35})
        )
        router = IntentRouter(client=client)
        routing = router.route("Please delete my account")

        # Gate
        gate = evaluate_gate(routing.intent, routing.confidence, action_type="account_delete")
        assert gate.hard_block is True   # very low confidence → hard block

        # Approval workflow
        engine = WorkflowEngine()
        req = ApprovalRequest(
            description=f"account_delete requested by user — confidence={routing.confidence}",
            requester="user@example.com",
        )
        with patch("src.workflow.engine.notify_approver"):
            rid = engine.submit(req, approver1_id="mgr1", approver2_id="mgr2")
            engine.decide(rid, "mgr1", "approve")
            final_state = engine.decide(rid, "mgr2", "approve")

        assert final_state == ApprovalState.APPROVED
```

---

## 4 — Common Integration Bugs and Fixes

### Bug 1 — Router and orchestrator use different intent label conventions

**Symptom:** Router returns `"billing_inquiry"`, registry key is `"billing"` → `get()` returns None → crash.

**Fix:** Normalise intent labels in `RoutingResult` validation:
```python
# In RoutingResult validator
@field_validator("intent")
@classmethod
def normalise_intent(cls, v: str) -> str:
    return v.lower().strip().replace(" ", "_").split("_")[0]  # "billing_inquiry" → "billing"
```

---

### Bug 2 — Confidence threshold checked in two places with different values

**Symptom:** Orchestrator escalates at 0.60, but HITL gate hardcodes 0.40 — requests in the 0.40–0.60 range never reach the gate.

**Fix:** Store the single threshold in a shared config and pass it to both components:
```python
# src/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    confidence_threshold: float = 0.60
    model_config = {"env_file": ".env"}

settings = Settings()
```

---

### Bug 3 — asyncio expiry task never started in production but runs in tests

**Symptom:** Tests pass because `_check_and_expire()` is called directly; production workflows never expire.

**Fix:** Call `manager.start()` inside the application lifespan, and add an integration test that boots the full app stack:
```python
# In FastAPI lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    expiry_manager.start()
    yield
    expiry_manager.stop()
```

---

### Bug 4 — Sequential approval chain allows any approver to decide

**Symptom:** An attacker or misconfigured client calls `decide_approver2` at stage 1 and the chain moves forward.

**Fix:** The `WorkflowEngine.decide()` method validates `approver_id` before calling the chain. Ensure this check is present and covered by tests (see Exercise 2 tests above).

---

### Bug 5 — Mocked LLM returns dict but code expects Pydantic model

**Symptom:** `mock_client.messages.create.return_value = {"content": [...]}` — then `response.content[0].text` raises `AttributeError: 'dict' object has no attribute 'content'`.

**Fix:** Always use `MagicMock` with `.content` attribute set, or build a proper stub:
```python
def _mock_message(text: str) -> MagicMock:
    block = MagicMock()
    block.text = text
    msg = MagicMock()
    msg.content = [block]
    return msg
```
