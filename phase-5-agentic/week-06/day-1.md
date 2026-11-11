# Phase 5 · Week 06 · Day 1 — Router Agents: Intent Classification

> Objective: Build a router agent that classifies user intent and dispatches to specialist handlers  ·  Time: ~1 hr  ·  Prereqs: [[phase-5-agentic/week-05/day-5]]

## Concept

A **router agent** sits at the front of a multi-agent system. Its job is narrow: decide *who* should handle this message, then pass it on. Specialist agents do the actual work.

Two routing strategies exist:

| Strategy | When to use |
|---|---|
| Rule-based | Fast, deterministic, low-cost; works when intent maps cleanly to keywords or regex |
| LLM-based | Handles ambiguous, multi-lingual, or complex inputs; higher latency and cost |

Hybrid is common: fast rule-check first, LLM fallback when rules miss.

The router itself should be **stateless** — classification is a pure function of the input. State lives in the specialist agents downstream.

## How it works internally

**LLM-based classification flow:**

1. Router receives raw user message.
2. Constructs a prompt that lists valid intent categories with descriptions.
3. Sends to LLM with structured output (JSON). The model returns `{"intent": "billing", "confidence": 0.92, "reasoning": "..."}`.
4. Router validates with Pydantic, falls back to `"general"` if parse fails or confidence is below threshold.
5. Dispatcher looks up the registered handler for that intent and calls it.

**Why structured output matters:** Asking for JSON in the prompt is brittle — the model may wrap it in markdown. Better approach: use the Anthropic API's `tool_use` feature where the tool schema enforces the output shape. The model is forced to call the tool, giving you a guaranteed-valid JSON payload you can parse directly.

**Confidence threshold:** Keep a minimum (e.g. 0.70). Below it, route to a fallback or ask the user to clarify. Routing with low confidence silently degrades quality.

## Worked example

```python
"""
Router agent with LLM-based intent classification.

Dispatches to billing / technical / general specialist handlers.
Run: python day-1.py
Requires: ANTHROPIC_API_KEY in environment
"""

import json
import logging
import os
from collections.abc import Callable
from typing import Any

import anthropic
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

VALID_INTENTS = {"billing", "technical", "general"}


class IntentResult(BaseModel):
    intent: str = Field(..., description="Classified intent category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    reasoning: str = Field(..., description="Short rationale for the classification")

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
# Intent classification via LLM tool_use
# ---------------------------------------------------------------------------

CLASSIFICATION_TOOL: dict[str, Any] = {
    "name": "classify_intent",
    "description": "Classify the user message into exactly one intent category.",
    "input_schema": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": list(VALID_INTENTS),
                "description": "The intent category",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence score between 0.0 and 1.0",
            },
            "reasoning": {
                "type": "string",
                "description": "One-sentence rationale",
            },
        },
        "required": ["intent", "confidence", "reasoning"],
    },
}

SYSTEM_PROMPT = """\
You are an intent classifier for a customer support system.
Classify user messages into one of three categories:
- billing: questions about invoices, payments, subscriptions, refunds, pricing
- technical: questions about bugs, errors, configuration, API usage, integrations
- general: greetings, company info, FAQs, anything that doesn't fit the above

Always call the classify_intent tool. Be conservative with confidence — use < 0.70
when the message is genuinely ambiguous.\
"""


def classify_intent(message: str, client: anthropic.Anthropic) -> IntentResult:
    """
    Call the LLM to classify message intent.

    Uses tool_use to guarantee structured JSON output.
    Falls back to IntentResult(intent='general', confidence=0.0, ...) on any failure.
    """
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

        # With tool_choice forced, first content block is always tool_use
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
# Specialist handler functions
# ---------------------------------------------------------------------------

def handle_billing(message: str) -> str:
    logger.info("Billing handler received: %r", message)
    return (
        "Billing specialist: I can help with invoices, refunds, and subscription changes. "
        "Please provide your account ID and I will look up your billing history."
    )


def handle_technical(message: str) -> str:
    logger.info("Technical handler received: %r", message)
    return (
        "Technical specialist: I can help debug issues and walk you through configuration. "
        "Please share any error messages or logs you are seeing."
    )


def handle_general(message: str) -> str:
    logger.info("General handler received: %r", message)
    return (
        "General support: Happy to help! Could you give me a bit more detail "
        "about what you need today?"
    )


# ---------------------------------------------------------------------------
# Router class
# ---------------------------------------------------------------------------

HandlerFn = Callable[[str], str]

CONFIDENCE_THRESHOLD = 0.70


class Router:
    """
    Classifies inbound messages and dispatches to registered specialist handlers.

    Handlers are registered by intent name. If no handler is registered for
    a classified intent, or confidence is below threshold, the fallback handler
    is used instead.
    """

    def __init__(self, client: anthropic.Anthropic) -> None:
        self._client = client
        self._handlers: dict[str, HandlerFn] = {}
        self._fallback: HandlerFn = handle_general

    def register(self, intent: str, handler: HandlerFn) -> "Router":
        """Register a handler function for an intent. Chainable."""
        if intent not in VALID_INTENTS:
            raise ValueError(f"Unknown intent '{intent}'. Valid: {VALID_INTENTS}")
        self._handlers[intent] = handler
        logger.debug("Registered handler for intent=%s", intent)
        return self

    def set_fallback(self, handler: HandlerFn) -> "Router":
        """Override the default fallback handler."""
        self._fallback = handler
        return self

    def route(self, message: str) -> RouterResponse:
        """
        Classify message, select handler, and return the specialist response.

        Low-confidence classifications are routed to the fallback regardless
        of which intent was predicted.
        """
        intent_result = classify_intent(message, self._client)

        if intent_result.confidence < CONFIDENCE_THRESHOLD:
            logger.info(
                "Confidence %.2f < threshold %.2f — using fallback",
                intent_result.confidence,
                CONFIDENCE_THRESHOLD,
            )
            handler = self._fallback
        else:
            handler = self._handlers.get(intent_result.intent, self._fallback)

        response_text = handler(message)
        return RouterResponse(
            original_message=message,
            intent_result=intent_result,
            handler_response=response_text,
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
    )

    test_messages = [
        "My invoice from last month shows a double charge.",
        "I'm getting a 401 error when calling the REST API.",
        "What are your business hours?",
    ]

    for msg in test_messages:
        result = router.route(msg)
        print(f"\nMessage : {result.original_message}")
        print(f"Intent  : {result.intent_result.intent} ({result.intent_result.confidence:.0%})")
        print(f"Reason  : {result.intent_result.reasoning}")
        print(f"Response: {result.handler_response}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Tests (pytest-style) — run: pytest day-1.py -v
# ---------------------------------------------------------------------------

import unittest.mock  # noqa: E402 — keep tests at bottom for readability


def _make_mock_response(intent: str, confidence: float, reasoning: str) -> Any:
    """Build a fake anthropic.types.Message with a tool_use block."""
    tool_block = unittest.mock.MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = {"intent": intent, "confidence": confidence, "reasoning": reasoning}

    mock_response = unittest.mock.MagicMock()
    mock_response.content = [tool_block]
    return mock_response


def test_classify_intent_billing() -> None:
    """LLM returns billing intent — IntentResult parsed correctly."""
    mock_client = unittest.mock.MagicMock(spec=anthropic.Anthropic)
    mock_client.messages.create.return_value = _make_mock_response(
        "billing", 0.95, "Mentions invoice and charge"
    )

    result = classify_intent("My invoice is wrong", mock_client)

    assert result.intent == "billing"
    assert result.confidence == 0.95
    mock_client.messages.create.assert_called_once()


def test_classify_intent_fallback_on_error() -> None:
    """When LLM call raises, fallback IntentResult with intent=general returned."""
    mock_client = unittest.mock.MagicMock(spec=anthropic.Anthropic)
    mock_client.messages.create.side_effect = RuntimeError("network error")

    result = classify_intent("anything", mock_client)

    assert result.intent == "general"
    assert result.confidence == 0.0


def test_router_dispatches_to_correct_handler() -> None:
    """Router calls the technical handler for technical intent."""
    mock_client = unittest.mock.MagicMock(spec=anthropic.Anthropic)
    mock_client.messages.create.return_value = _make_mock_response(
        "technical", 0.88, "API error mentioned"
    )

    technical_handler = unittest.mock.MagicMock(return_value="Tech response")
    router = Router(mock_client).register("technical", technical_handler)

    result = router.route("I keep getting 500 errors")

    technical_handler.assert_called_once_with("I keep getting 500 errors")
    assert result.handler_response == "Tech response"
    assert result.intent_result.intent == "technical"


def test_router_uses_fallback_on_low_confidence() -> None:
    """Low-confidence classification routes to fallback, not the predicted handler."""
    mock_client = unittest.mock.MagicMock(spec=anthropic.Anthropic)
    mock_client.messages.create.return_value = _make_mock_response(
        "billing", 0.50, "Unclear context"
    )

    billing_handler = unittest.mock.MagicMock(return_value="Billing response")
    fallback_handler = unittest.mock.MagicMock(return_value="Fallback response")

    router = (
        Router(mock_client)
        .register("billing", billing_handler)
        .set_fallback(fallback_handler)
    )

    result = router.route("hmm I have a question")

    fallback_handler.assert_called_once()
    billing_handler.assert_not_called()
    assert result.handler_response == "Fallback response"


def test_intent_result_rejects_unknown_intent() -> None:
    """IntentResult raises ValidationError for unknown intent string."""
    from pydantic import ValidationError

    try:
        IntentResult(intent="unknown_xyz", confidence=0.9, reasoning="test")
        assert False, "Should have raised"
    except ValidationError:
        pass  # expected
```

## Exercises

1. **Extend intents**: Add a `"returns"` intent category. Update the tool schema, system prompt, and register a new handler. What changes when you add a new category?

2. **Rule-based pre-filter**: Before calling the LLM, add a `rule_classify(message: str) -> str | None` function that returns an intent for obvious cases (e.g. message contains "refund" or "invoice" → `"billing"`). Only call the LLM when the rule returns `None`. Measure how often the fast path fires on a batch of 20 sample messages.

3. **Confidence logging**: Log every classification to a local JSON file (`classification_log.jsonl`), one JSON object per line. Include timestamp, message hash (SHA-256 of message), intent, confidence. Write a script that reads the log and prints the distribution of intents and the average confidence per intent.

(Solutions: ../solutions/week-06/day-1.md)

## Self-check

- [ ] I can explain the difference between rule-based and LLM-based routing and when each is appropriate
- [ ] I can implement a `classify_intent()` function using Anthropic's `tool_use` to enforce structured output without additional parsing
- [ ] I can describe what a confidence threshold does and what happens at the boundary
- [ ] I can implement `Router.route()` with fallback logic from memory, without reference
- [ ] I can write a unit test that mocks the Anthropic client and verifies dispatch behaviour
