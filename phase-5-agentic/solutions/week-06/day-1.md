# Solutions — Week 06 · Day 1 — Router Agents: Intent Classification

> These are reference solutions. Your implementation may differ and still be correct.

## Exercise 1

**Add a new intent category `"escalation"` to the router with a matching specialist handler.**

Key changes: add `"escalation"` to `VALID_INTENTS`, update the tool schema enum, add the system-prompt description, write a handler, and register it. The `@field_validator` on `IntentResult` guards automatically once the set is updated.

```python
"""
day1_ex1.py — Router with escalation intent added.

Run: python day1_ex1.py
Env: ANTHROPIC_API_KEY
"""
import logging
import os
import unittest.mock
from collections.abc import Callable
from typing import Any

import anthropic
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

VALID_INTENTS = {"billing", "technical", "general", "escalation"}  # <-- added


class IntentResult(BaseModel):
    intent: str = Field(..., description="Classified intent category")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str

    @field_validator("intent")
    @classmethod
    def intent_must_be_valid(cls, v: str) -> str:
        if v not in VALID_INTENTS:
            raise ValueError(f"intent must be one of {VALID_INTENTS}, got '{v}'")
        return v


class RouterResponse(BaseModel):
    original_message: str
    intent_result: IntentResult
    handler_response: str


# ---------------------------------------------------------------------------
# Tool schema — enum updated to include escalation
# ---------------------------------------------------------------------------

CLASSIFICATION_TOOL: dict[str, Any] = {
    "name": "classify_intent",
    "description": "Classify the user message into exactly one intent category.",
    "input_schema": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": sorted(VALID_INTENTS),  # always sorted for determinism
                "description": "The intent category",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence score between 0.0 and 1.0",
            },
            "reasoning": {"type": "string", "description": "One-sentence rationale"},
        },
        "required": ["intent", "confidence", "reasoning"],
    },
}

# System prompt updated with escalation description
SYSTEM_PROMPT = """\
You are an intent classifier for a customer support system.
Classify user messages into one of four categories:
- billing: questions about invoices, payments, subscriptions, refunds, pricing
- technical: questions about bugs, errors, configuration, API usage, integrations
- escalation: requests to speak with a manager, complaints about service quality,
  legal threats, or any message expressing severe frustration or urgency
- general: greetings, company info, FAQs, anything that doesn't fit the above

Always call the classify_intent tool. Be conservative with confidence — use < 0.70
when the message is genuinely ambiguous.\
"""


def classify_intent(message: str, client: anthropic.Anthropic) -> IntentResult:
    logger.debug("Classifying: %r", message)
    try:
        response = client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=256,
            system=SYSTEM_PROMPT,
            tools=[CLASSIFICATION_TOOL],  # type: ignore[list-item]
            tool_choice={"type": "tool", "name": "classify_intent"},
            messages=[{"role": "user", "content": message}],
        )
        tool_block = next(
            (b for b in response.content if b.type == "tool_use"), None
        )
        if tool_block is None:
            raise ValueError("No tool_use block in response")
        result = IntentResult.model_validate(tool_block.input)
        logger.info("Intent=%s confidence=%.2f", result.intent, result.confidence)
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("Classification failed (%s) — defaulting to general", exc)
        return IntentResult(
            intent="general",
            confidence=0.0,
            reasoning="Classification error — routed to general as safe fallback",
        )


# ---------------------------------------------------------------------------
# Specialist handlers — including new escalation handler
# ---------------------------------------------------------------------------

def handle_billing(message: str) -> str:
    logger.info("Billing handler: %r", message)
    return (
        "Billing specialist: I can help with invoices, refunds, and subscription changes. "
        "Please provide your account ID."
    )


def handle_technical(message: str) -> str:
    logger.info("Technical handler: %r", message)
    return (
        "Technical specialist: I can help debug issues. "
        "Please share any error messages or logs."
    )


def handle_general(message: str) -> str:
    logger.info("General handler: %r", message)
    return "General support: Happy to help! Could you give me more detail?"


def handle_escalation(message: str) -> str:
    """New escalation handler — routes to senior support staff."""
    logger.info("Escalation handler: %r", message)
    return (
        "Escalation team: Your concern has been flagged as high-priority. "
        "A senior support manager will contact you within 2 hours. "
        "We apologise for any inconvenience."
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

HandlerFn = Callable[[str], str]
CONFIDENCE_THRESHOLD = 0.70


class Router:
    def __init__(self, client: anthropic.Anthropic) -> None:
        self._client = client
        self._handlers: dict[str, HandlerFn] = {}
        self._fallback: HandlerFn = handle_general

    def register(self, intent: str, handler: HandlerFn) -> "Router":
        if intent not in VALID_INTENTS:
            raise ValueError(f"Unknown intent '{intent}'. Valid: {VALID_INTENTS}")
        self._handlers[intent] = handler
        return self

    def set_fallback(self, handler: HandlerFn) -> "Router":
        self._fallback = handler
        return self

    def route(self, message: str) -> RouterResponse:
        intent_result = classify_intent(message, self._client)
        if intent_result.confidence < CONFIDENCE_THRESHOLD:
            logger.info(
                "Confidence %.2f < threshold — using fallback",
                intent_result.confidence,
            )
            handler = self._fallback
        else:
            handler = self._handlers.get(intent_result.intent, self._fallback)
        return RouterResponse(
            original_message=message,
            intent_result=intent_result,
            handler_response=handler(message),
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(api_key=api_key)
    router = (
        Router(client)
        .register("billing", handle_billing)
        .register("technical", handle_technical)
        .register("general", handle_general)
        .register("escalation", handle_escalation)
    )
    msgs = [
        "My invoice shows a double charge.",
        "I want to speak to your manager RIGHT NOW — this is unacceptable!",
        "I'm getting a 401 error on the API.",
        "What are your business hours?",
    ]
    for msg in msgs:
        r = router.route(msg)
        print(f"\nMessage : {r.original_message}")
        print(f"Intent  : {r.intent_result.intent} ({r.intent_result.confidence:.0%})")
        print(f"Response: {r.handler_response}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _make_mock_response(intent: str, confidence: float, reasoning: str) -> Any:
    tool_block = unittest.mock.MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = {"intent": intent, "confidence": confidence, "reasoning": reasoning}
    mock_response = unittest.mock.MagicMock()
    mock_response.content = [tool_block]
    return mock_response


def test_escalation_intent_valid() -> None:
    """IntentResult accepts 'escalation' after adding it to VALID_INTENTS."""
    result = IntentResult(intent="escalation", confidence=0.9, reasoning="Angry customer")
    assert result.intent == "escalation"


def test_escalation_handler_dispatched() -> None:
    """Router dispatches to escalation handler when LLM returns escalation intent."""
    mock_client = unittest.mock.MagicMock(spec=anthropic.Anthropic)
    mock_client.messages.create.return_value = _make_mock_response(
        "escalation", 0.92, "Customer demands manager"
    )
    escalation_handler = unittest.mock.MagicMock(return_value="Escalation response")
    router = Router(mock_client).register("escalation", escalation_handler)

    result = router.route("I demand to speak to a manager!")

    escalation_handler.assert_called_once_with("I demand to speak to a manager!")
    assert result.handler_response == "Escalation response"
    assert result.intent_result.intent == "escalation"


def test_escalation_handler_returns_priority_message() -> None:
    """handle_escalation returns a message mentioning 'manager' or 'priority'."""
    response = handle_escalation("I am furious")
    assert any(kw in response.lower() for kw in ("manager", "priority", "senior"))


def test_unknown_intent_raises() -> None:
    """IntentResult still rejects intents outside VALID_INTENTS."""
    from pydantic import ValidationError
    try:
        IntentResult(intent="unknown_xyz", confidence=0.9, reasoning="test")
        assert False, "Should have raised"
    except ValidationError:
        pass
```

**What changes when you add a new category?**
- `VALID_INTENTS` — add the string.
- Tool schema `enum` list — add the string (auto-derived from the set here).
- `SYSTEM_PROMPT` — describe the new category so the LLM knows when to use it.
- Handler function — implement the specialist logic.
- `Router.register()` call — wire the handler at construction time.
- Tests — cover the new path (dispatch + handler output).


## Exercise 2

**Make the router return a confidence score alongside the intent, using structured JSON output from the LLM; reject routing if confidence < 0.7.**

The worked example already returns confidence via `tool_use`. This exercise makes that explicit by building a standalone version from scratch that uses `tool_use` for structured output, validates with Pydantic, and enforces the threshold as a hard gate that raises `LowConfidenceError`.

```python
"""
day1_ex2.py — Router with explicit confidence gate.

Run: python day1_ex2.py
Env: ANTHROPIC_API_KEY
"""
import logging
import os
import unittest.mock
from typing import Any

import anthropic
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.70
VALID_INTENTS = {"billing", "technical", "general", "escalation"}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class IntentResult(BaseModel):
    intent: str = Field(..., description="Classified intent category")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str

    @field_validator("intent")
    @classmethod
    def intent_must_be_valid(cls, v: str) -> str:
        if v not in VALID_INTENTS:
            raise ValueError(f"intent must be one of {VALID_INTENTS}")
        return v


class LowConfidenceError(RuntimeError):
    """Raised when LLM classification confidence is below the threshold."""

    def __init__(self, result: IntentResult, threshold: float) -> None:
        self.result = result
        self.threshold = threshold
        super().__init__(
            f"Confidence {result.confidence:.2f} < threshold {threshold:.2f} "
            f"for intent '{result.intent}'"
        )


# ---------------------------------------------------------------------------
# Classifier using tool_use for guaranteed structured output
# ---------------------------------------------------------------------------

CLASSIFICATION_TOOL: dict[str, Any] = {
    "name": "classify_intent",
    "description": "Classify the user message into exactly one intent category.",
    "input_schema": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": sorted(VALID_INTENTS),
            },
            "confidence": {
                "type": "number",
                "description": "Confidence score 0.0–1.0. Use <0.70 for ambiguous messages.",
            },
            "reasoning": {"type": "string"},
        },
        "required": ["intent", "confidence", "reasoning"],
    },
}

SYSTEM_PROMPT = """\
You are an intent classifier for a customer support system.
- billing: invoices, payments, subscriptions, refunds, pricing
- technical: bugs, errors, configuration, API, integrations
- escalation: manager requests, complaints, legal threats, severe frustration
- general: greetings, FAQs, anything else

Call classify_intent with confidence < 0.70 when the message is ambiguous.\
"""


def classify_intent(
    message: str,
    client: anthropic.Anthropic,
    *,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> IntentResult:
    """
    Classify message intent via LLM tool_use.

    Raises:
        LowConfidenceError: when confidence < threshold (caller decides fallback).
        RuntimeError: on LLM or parse failure.
    """
    response = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=256,
        system=SYSTEM_PROMPT,
        tools=[CLASSIFICATION_TOOL],  # type: ignore[list-item]
        tool_choice={"type": "tool", "name": "classify_intent"},
        messages=[{"role": "user", "content": message}],
    )

    tool_block = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_block is None:
        raise RuntimeError("No tool_use block in LLM response")

    result = IntentResult.model_validate(tool_block.input)
    logger.info(
        "classified intent=%s confidence=%.2f", result.intent, result.confidence
    )

    if result.confidence < threshold:
        raise LowConfidenceError(result, threshold)

    return result


# ---------------------------------------------------------------------------
# Router with confidence gate
# ---------------------------------------------------------------------------

class Router:
    def __init__(
        self,
        client: anthropic.Anthropic,
        *,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        self._client = client
        self._threshold = confidence_threshold
        self._handlers: dict[str, Any] = {}

    def register(self, intent: str, handler: Any) -> "Router":
        self._handlers[intent] = handler
        return self

    def dispatch(self, message: str) -> str:
        """
        Classify and dispatch. Routes to clarification handler on low confidence.
        """
        try:
            result = classify_intent(
                message, self._client, threshold=self._threshold
            )
        except LowConfidenceError as exc:
            logger.warning("Low confidence routing rejected: %s", exc)
            return (
                f"I'm not sure I understood your request "
                f"(confidence {exc.result.confidence:.0%}). "
                "Could you please clarify what you need help with?"
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Classification error: %s", exc)
            return "An error occurred. Please try again."

        handler = self._handlers.get(result.intent)
        if handler is None:
            logger.warning("No handler registered for intent=%s", result.intent)
            return f"Unhandled intent: {result.intent}"
        return handler(message)


def main() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)
    router = (
        Router(client, confidence_threshold=0.70)
        .register("billing", lambda m: "Billing: " + m)
        .register("technical", lambda m: "Tech: " + m)
        .register("general", lambda m: "General: " + m)
        .register("escalation", lambda m: "Escalation: " + m)
    )

    messages = [
        "My payment failed",           # clear billing
        "I keep getting 500 errors",   # clear technical
        "hmm maybe something about X", # likely ambiguous → low confidence
    ]
    for msg in messages:
        print(f"\nMessage: {msg}")
        print(f"Response: {router.dispatch(msg)}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _make_tool_response(intent: str, confidence: float) -> Any:
    block = unittest.mock.MagicMock()
    block.type = "tool_use"
    block.input = {"intent": intent, "confidence": confidence, "reasoning": "test"}
    resp = unittest.mock.MagicMock()
    resp.content = [block]
    return resp


def test_classify_returns_intent_result_above_threshold() -> None:
    mock_client = unittest.mock.MagicMock(spec=anthropic.Anthropic)
    mock_client.messages.create.return_value = _make_tool_response("billing", 0.90)

    result = classify_intent("Invoice question", mock_client, threshold=0.70)

    assert result.intent == "billing"
    assert result.confidence == 0.90


def test_classify_raises_low_confidence_error() -> None:
    mock_client = unittest.mock.MagicMock(spec=anthropic.Anthropic)
    mock_client.messages.create.return_value = _make_tool_response("billing", 0.55)

    try:
        classify_intent("Hmm maybe", mock_client, threshold=0.70)
        assert False, "Should raise LowConfidenceError"
    except LowConfidenceError as exc:
        assert exc.result.intent == "billing"
        assert exc.result.confidence == 0.55


def test_router_returns_clarification_on_low_confidence() -> None:
    mock_client = unittest.mock.MagicMock(spec=anthropic.Anthropic)
    mock_client.messages.create.return_value = _make_tool_response("billing", 0.45)

    router = Router(mock_client, confidence_threshold=0.70)
    router.register("billing", lambda m: "Billing response")

    response = router.dispatch("I dunno something about money maybe?")

    assert "clarify" in response.lower() or "confidence" in response.lower()


def test_router_dispatches_above_threshold() -> None:
    mock_client = unittest.mock.MagicMock(spec=anthropic.Anthropic)
    mock_client.messages.create.return_value = _make_tool_response("technical", 0.88)

    handler = unittest.mock.MagicMock(return_value="Tech response")
    router = Router(mock_client).register("technical", handler)

    response = router.dispatch("API is returning 404")

    handler.assert_called_once()
    assert response == "Tech response"


def test_confidence_threshold_boundary() -> None:
    """Exactly at threshold (0.70) should pass; one tick below should fail."""
    mock_client = unittest.mock.MagicMock(spec=anthropic.Anthropic)

    mock_client.messages.create.return_value = _make_tool_response("general", 0.70)
    result = classify_intent("test", mock_client, threshold=0.70)
    assert result.confidence == 0.70

    mock_client.messages.create.return_value = _make_tool_response("general", 0.699)
    try:
        classify_intent("test", mock_client, threshold=0.70)
        assert False, "Should raise"
    except LowConfidenceError:
        pass
```


## Exercise 3

**Implement a fallback handler for when the LLM is unavailable, using simple keyword-matching as a rule-based classifier.**

Design: `RuleBasedClassifier` is called first if the primary LLM call raises a connectivity/availability exception. If it returns a hit, routing proceeds with a synthetic `IntentResult(confidence=1.0)` (rules are deterministic). If no keyword matches, fall back to `"general"`.

```python
"""
day1_ex3.py — Router with LLM-unavailable fallback using keyword classifier.

Run: python day1_ex3.py
Env: ANTHROPIC_API_KEY
"""
import logging
import os
import re
import unittest.mock
from typing import Any

import anthropic
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

VALID_INTENTS = {"billing", "technical", "general", "escalation"}
CONFIDENCE_THRESHOLD = 0.70


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class IntentResult(BaseModel):
    intent: str = Field(..., description="Classified intent category")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    source: str = Field(default="llm", description="'llm' or 'rule_based'")

    @field_validator("intent")
    @classmethod
    def intent_must_be_valid(cls, v: str) -> str:
        if v not in VALID_INTENTS:
            raise ValueError(f"intent must be one of {VALID_INTENTS}")
        return v


# ---------------------------------------------------------------------------
# Rule-based fallback classifier
# ---------------------------------------------------------------------------

# Each entry: (intent, list_of_patterns). Patterns are case-insensitive.
_RULES: list[tuple[str, list[str]]] = [
    (
        "billing",
        [
            r"\binvoice\b",
            r"\bpayment\b",
            r"\brefund\b",
            r"\bcharge\b",
            r"\bsubscription\b",
            r"\bpricing\b",
            r"\bbill\b",
        ],
    ),
    (
        "technical",
        [
            r"\berror\b",
            r"\bbug\b",
            r"\bapi\b",
            r"\b4[0-9]{2}\b",   # HTTP 4xx
            r"\b5[0-9]{2}\b",   # HTTP 5xx
            r"\bcrash\b",
            r"\btimeout\b",
            r"\bintegration\b",
            r"\bconfiguration\b",
        ],
    ),
    (
        "escalation",
        [
            r"\bmanager\b",
            r"\blegal\b",
            r"\blawsuit\b",
            r"\bunacceptable\b",
            r"\bescalat\b",
            r"\bdemand\b",
        ],
    ),
]


class RuleBasedClassifier:
    """
    Keyword-matching fallback when the LLM is unavailable.

    Returns an IntentResult with source='rule_based' and confidence=1.0 when
    a keyword matches (deterministic), or intent='general' when nothing matches.
    """

    def classify(self, message: str) -> IntentResult:
        lower = message.lower()
        for intent, patterns in _RULES:
            for pattern in patterns:
                if re.search(pattern, lower):
                    logger.info(
                        "Rule match: intent=%s pattern=%r", intent, pattern
                    )
                    return IntentResult(
                        intent=intent,
                        confidence=1.0,
                        reasoning=f"Keyword match on pattern '{pattern}'",
                        source="rule_based",
                    )
        logger.info("No rule matched — defaulting to general")
        return IntentResult(
            intent="general",
            confidence=1.0,
            reasoning="No keyword matched; routed to general",
            source="rule_based",
        )


# ---------------------------------------------------------------------------
# Tool schema + LLM classifier
# ---------------------------------------------------------------------------

CLASSIFICATION_TOOL: dict[str, Any] = {
    "name": "classify_intent",
    "description": "Classify the user message into exactly one intent category.",
    "input_schema": {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": sorted(VALID_INTENTS)},
            "confidence": {"type": "number"},
            "reasoning": {"type": "string"},
        },
        "required": ["intent", "confidence", "reasoning"],
    },
}

SYSTEM_PROMPT = """\
You are an intent classifier for a customer support system.
- billing: invoices, payments, subscriptions, refunds, pricing
- technical: bugs, errors, configuration, API, integrations
- escalation: manager requests, complaints, legal threats, severe frustration
- general: greetings, FAQs, anything else
Call classify_intent. Use confidence < 0.70 for ambiguous messages.\
"""

# Exceptions that indicate the LLM service is down (not a logic error)
_LLM_UNAVAILABLE = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
)


class Router:
    """
    Classifies with the LLM and automatically falls back to keyword rules
    when the LLM service raises a connectivity or server-error exception.
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        *,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        self._client = client
        self._threshold = confidence_threshold
        self._rule_classifier = RuleBasedClassifier()
        self._handlers: dict[str, Any] = {}

    def register(self, intent: str, handler: Any) -> "Router":
        self._handlers[intent] = handler
        return self

    def classify_intent(self, message: str) -> IntentResult:
        """
        Try LLM classification first; fall back to rule-based on LLM errors.
        """
        try:
            response = self._client.messages.create(
                model="claude-sonnet-5-5",
                max_tokens=256,
                system=SYSTEM_PROMPT,
                tools=[CLASSIFICATION_TOOL],  # type: ignore[list-item]
                tool_choice={"type": "tool", "name": "classify_intent"},
                messages=[{"role": "user", "content": message}],
            )
            tool_block = next(
                (b for b in response.content if b.type == "tool_use"), None
            )
            if tool_block is None:
                raise RuntimeError("No tool_use block")
            result = IntentResult.model_validate(
                {**tool_block.input, "source": "llm"}
            )
            logger.info(
                "LLM classified intent=%s confidence=%.2f",
                result.intent,
                result.confidence,
            )
            return result

        except _LLM_UNAVAILABLE as exc:
            logger.warning(
                "LLM unavailable (%s) — switching to rule-based classifier", exc
            )
            return self._rule_classifier.classify(message)

        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected classification error: %s — using rules", exc)
            return self._rule_classifier.classify(message)

    def dispatch(self, message: str) -> str:
        result = self.classify_intent(message)

        if result.source == "llm" and result.confidence < self._threshold:
            return (
                f"I'm not confident about your request "
                f"(confidence {result.confidence:.0%}). Please clarify."
            )

        handler = self._handlers.get(result.intent, self._handlers.get("general"))
        if handler is None:
            return f"No handler registered for intent: {result.intent}"
        return handler(message)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)
    router = (
        Router(client)
        .register("billing", lambda m: "[Billing] " + m)
        .register("technical", lambda m: "[Technical] " + m)
        .register("escalation", lambda m: "[Escalation] " + m)
        .register("general", lambda m: "[General] " + m)
    )

    msgs = [
        "I need a refund for my subscription",
        "Getting a 500 error on your API",
        "I want to speak to a manager",
        "What are your opening hours?",
    ]
    for msg in msgs:
        print(f"\nMessage: {msg}")
        print(f"Response: {router.dispatch(msg)}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_rule_classifier_billing_keyword() -> None:
    clf = RuleBasedClassifier()
    result = clf.classify("I need a refund for my invoice")
    assert result.intent == "billing"
    assert result.source == "rule_based"
    assert result.confidence == 1.0


def test_rule_classifier_technical_keyword() -> None:
    clf = RuleBasedClassifier()
    result = clf.classify("Getting a 500 error on your API")
    assert result.intent == "technical"
    assert result.source == "rule_based"


def test_rule_classifier_escalation_keyword() -> None:
    clf = RuleBasedClassifier()
    result = clf.classify("I want to speak to a manager immediately")
    assert result.intent == "escalation"
    assert result.source == "rule_based"


def test_rule_classifier_no_match_returns_general() -> None:
    clf = RuleBasedClassifier()
    result = clf.classify("Hello, how are you?")
    assert result.intent == "general"
    assert result.source == "rule_based"


def test_router_falls_back_to_rules_on_connection_error() -> None:
    """When LLM raises APIConnectionError, rule-based classifier is used."""
    mock_client = unittest.mock.MagicMock(spec=anthropic.Anthropic)
    mock_client.messages.create.side_effect = anthropic.APIConnectionError(
        request=unittest.mock.MagicMock()
    )

    clf = RuleBasedClassifier()
    router = Router(mock_client)
    router._rule_classifier = clf  # use real rule classifier

    result = router.classify_intent("My payment failed")

    assert result.source == "rule_based"
    assert result.intent == "billing"
    mock_client.messages.create.assert_called_once()


def test_router_falls_back_to_rules_on_timeout() -> None:
    mock_client = unittest.mock.MagicMock(spec=anthropic.Anthropic)
    mock_client.messages.create.side_effect = anthropic.APITimeoutError(
        request=unittest.mock.MagicMock()
    )

    router = Router(mock_client)
    result = router.classify_intent("invoice double charge")

    assert result.source == "rule_based"
    assert result.intent == "billing"


def test_router_uses_llm_when_available() -> None:
    mock_client = unittest.mock.MagicMock(spec=anthropic.Anthropic)
    block = unittest.mock.MagicMock()
    block.type = "tool_use"
    block.input = {"intent": "technical", "confidence": 0.91, "reasoning": "API error"}
    resp = unittest.mock.MagicMock()
    resp.content = [block]
    mock_client.messages.create.return_value = resp

    router = Router(mock_client)
    result = router.classify_intent("I keep getting 404 errors")

    assert result.source == "llm"
    assert result.intent == "technical"


def test_rule_classifier_case_insensitive() -> None:
    clf = RuleBasedClassifier()
    result = clf.classify("I WANT A REFUND")
    assert result.intent == "billing"
```
