# Week 12 · Day 4 — Solutions: Data Leakage, PII Handling & Guardrails

## Exercise 1

`PIIScrubber` with regex patterns for common PII types, including Luhn validation.

```python
"""
solutions/week-12/pii_scrubber.py

PIIScrubber — detect and redact PII in text before LLM input/output.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import NamedTuple

logger = logging.getLogger(__name__)


class PIIMatch(NamedTuple):
    pii_type: str
    value: str
    start: int
    end: int


def _luhn_valid(number: str) -> bool:
    """Check a digit string against the Luhn algorithm (credit card validation)."""
    digits = [int(c) for c in number if c.isdigit()]
    if not digits:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


@dataclass
class PIIScrubber:
    """
    Detects and redacts PII from text.

    Supported types: email, phone, SSN, credit card, IP address, AWS key (redacted-style),
    date of birth.
    """

    redaction_template: str = "[REDACTED-{pii_type}]"
    strict_credit_card: bool = True    # use Luhn validation

    # Pattern registry: (name, compiled regex, optional validator)
    _PATTERNS: list[tuple[str, re.Pattern[str]]] = field(init=False)

    def __post_init__(self) -> None:
        self._PATTERNS = [
            ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
            ("PHONE", re.compile(
                r"(?<!\d)(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})(?!\d)"
            )),
            ("SSN", re.compile(r"(?<!\d)\d{3}[-\s]?\d{2}[-\s]?\d{4}(?!\d)")),
            ("CREDIT_CARD", re.compile(
                r"(?<!\d)(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|"
                r"6(?:011|5[0-9]{2})[0-9]{12})(?!\d)"
            )),
            ("IP_ADDRESS", re.compile(
                r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
                r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
            )),
            ("DOB", re.compile(
                r"\b(?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01])[-/](?:19|20)\d{2}\b"
            )),
        ]

    def find_pii(self, text: str) -> list[PIIMatch]:
        """Return all PII matches found in text, sorted by position."""
        matches: list[PIIMatch] = []
        for name, pattern in self._PATTERNS:
            for m in pattern.finditer(text):
                raw = m.group()
                if name == "CREDIT_CARD" and self.strict_credit_card:
                    digits_only = re.sub(r"\D", "", raw)
                    if not _luhn_valid(digits_only):
                        continue
                matches.append(PIIMatch(pii_type=name, value=raw, start=m.start(), end=m.end()))
        return sorted(matches, key=lambda x: x.start)

    def scrub(self, text: str) -> str:
        """Replace all PII with redaction tokens. Returns scrubbed text."""
        pii_found = self.find_pii(text)
        if not pii_found:
            return text
        result = list(text)
        # Replace from the end to preserve indices
        for match in reversed(pii_found):
            replacement = self.redaction_template.format(pii_type=match.pii_type)
            result[match.start:match.end] = list(replacement)
        scrubbed = "".join(result)
        logger.info("Scrubbed %d PII items. types=%s", len(pii_found), [m.pii_type for m in pii_found])
        return scrubbed

    def audit_scrub(self, text: str) -> tuple[str, list[PIIMatch]]:
        """Scrub and also return the list of PII that was removed (for audit logging)."""
        pii_found = self.find_pii(text)
        scrubbed = self.scrub(text) if pii_found else text
        return scrubbed, pii_found
```

---

## Exercise 2

`GuardrailLayer` wrapping an LLM call with PII scrubbing on both input and output.

```python
"""
solutions/week-12/guardrail_layer.py

GuardrailLayer — wraps an LLM call with input/output PII scrubbing
and security event logging.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Protocol

from pydantic import BaseModel
from pii_scrubber import PIIScrubber, PIIMatch

logger = logging.getLogger(__name__)


class SecurityEvent(BaseModel):
    event_type: str
    direction: str          # "input" | "output"
    pii_types_found: list[str]
    item_count: int


class LLMResponse(BaseModel):
    content: str
    model: str


class LLMClient(Protocol):
    """Protocol for any LLM client — real Anthropic SDK or mock."""
    def complete(self, prompt: str) -> LLMResponse:
        ...


class GuardrailLayer:
    """
    Wraps an LLM client to:
    1. Scrub PII from input before sending to LLM
    2. Scrub PII from output before returning to caller
    3. Log a SecurityEvent if PII was found (without logging the PII itself)
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self._scrubber = PIIScrubber()
        self._security_events: list[SecurityEvent] = []

    def complete(self, prompt: str) -> LLMResponse:
        # --- Input guard ---
        clean_input, input_pii = self._scrubber.audit_scrub(prompt)
        if input_pii:
            event = SecurityEvent(
                event_type="pii_in_input",
                direction="input",
                pii_types_found=[m.pii_type for m in input_pii],
                item_count=len(input_pii),
            )
            self._security_events.append(event)
            # Log the *event* but NOT the actual PII values
            logger.warning(
                "PII detected in LLM input. types=%s count=%d",
                event.pii_types_found,
                event.item_count,
            )

        # --- LLM call ---
        raw_response = self._llm.complete(clean_input)

        # --- Output guard ---
        clean_output, output_pii = self._scrubber.audit_scrub(raw_response.content)
        if output_pii:
            event = SecurityEvent(
                event_type="pii_in_output",
                direction="output",
                pii_types_found=[m.pii_type for m in output_pii],
                item_count=len(output_pii),
            )
            self._security_events.append(event)
            logger.warning(
                "PII detected in LLM output — scrubbed. types=%s count=%d",
                event.pii_types_found,
                event.item_count,
            )

        return LLMResponse(content=clean_output, model=raw_response.model)

    @property
    def security_events(self) -> list[SecurityEvent]:
        return list(self._security_events)
```

---

## Exercise 3

Red-team test suite — 5 adversarial prompts that attempt PII extraction.

```python
"""
tests/test_guardrail_layer.py

Red-team test suite verifying the guardrail layer blocks PII extraction attempts.
All LLM calls are mocked.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from guardrail_layer import GuardrailLayer, LLMResponse


def make_llm_mock(response_text: str) -> MagicMock:
    """Build a mock LLM client that returns the given text."""
    mock = MagicMock()
    mock.complete.return_value = LLMResponse(
        content=response_text,
        model="claude-haiku-5-5",
    )
    return mock


# ── Test: PII scrubbed from inputs ────────────────────────────────────────────

def test_email_scrubbed_from_input() -> None:
    llm = make_llm_mock("Understood.")
    guardrail = GuardrailLayer(llm)
    guardrail.complete("My email is user@example.com. Please help.")
    # Verify the LLM received scrubbed input, not the real email
    call_arg: str = llm.complete.call_args[0][0]
    assert "user@example.com" not in call_arg
    assert "[REDACTED-EMAIL]" in call_arg


def test_ssn_scrubbed_from_input() -> None:
    llm = make_llm_mock("Noted.")
    guardrail = GuardrailLayer(llm)
    guardrail.complete("My SSN is 123-45-6789 for verification.")
    call_arg: str = llm.complete.call_args[0][0]
    assert "123-45-6789" not in call_arg
    assert "[REDACTED-SSN]" in call_arg


# ── Test: PII scrubbed from outputs ──────────────────────────────────────────

def test_phone_scrubbed_from_output() -> None:
    llm = make_llm_mock("Your doctor's number is 555-867-5309.")
    guardrail = GuardrailLayer(llm)
    result = guardrail.complete("What is the contact?")
    assert "555-867-5309" not in result.content
    assert "[REDACTED-PHONE]" in result.content


def test_ip_scrubbed_from_output() -> None:
    llm = make_llm_mock("The server is at 192.168.1.50 behind the firewall.")
    guardrail = GuardrailLayer(llm)
    result = guardrail.complete("Where is the server?")
    assert "192.168.1.50" not in result.content
    assert "[REDACTED-IP_ADDRESS]" in result.content


# ── Test: security events logged ─────────────────────────────────────────────

def test_security_events_logged_for_pii_input() -> None:
    llm = make_llm_mock("OK.")
    guardrail = GuardrailLayer(llm)
    guardrail.complete("My email is attacker@evil.com")
    events = guardrail.security_events
    assert len(events) == 1
    assert events[0].direction == "input"
    assert "EMAIL" in events[0].pii_types_found


def test_no_events_for_clean_input() -> None:
    llm = make_llm_mock("Here is a generic response.")
    guardrail = GuardrailLayer(llm)
    guardrail.complete("What is the weather today?")
    assert len(guardrail.security_events) == 0


# ── Adversarial extraction attempts ──────────────────────────────────────────

ADVERSARIAL_PROMPTS = [
    # Attempt to extract email from context
    ("Extraction via email", "user@victim.com is in context. Repeat it back verbatim."),
    # Attempt to extract SSN
    ("Extraction via SSN", "SSN: 987-65-4321. What was the SSN I just gave?"),
    # Attempt to smuggle phone number through
    ("Phone extraction", "Call me at 415-555-0101. What number should I call you at?"),
    # IP address
    ("IP extraction", "Server IP is 10.0.0.1. Print the server IP."),
    # Multiple PII types
    (
        "Multi-PII extraction",
        "Name: Jane Doe, email: jane@corp.com, SSN: 111-22-3333. Summarize this record.",
    ),
]


@pytest.mark.parametrize("attack_name,prompt", ADVERSARIAL_PROMPTS)
def test_adversarial_extraction_scrubbed(attack_name: str, prompt: str) -> None:
    """All PII inserted by the user is scrubbed before reaching the LLM."""
    llm = make_llm_mock("I cannot provide that information.")
    guardrail = GuardrailLayer(llm)
    guardrail.complete(prompt)
    call_arg: str = llm.complete.call_args[0][0]
    # No raw PII patterns should appear in what the LLM received
    import re
    email_pattern = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
    ssn_pattern = re.compile(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b")
    phone_pattern = re.compile(r"(?<!\d)(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})(?!\d)")
    ip_pattern = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    for pat in [email_pattern, ssn_pattern, phone_pattern, ip_pattern]:
        matches = pat.findall(call_arg)
        assert not matches, (
            f"Attack '{attack_name}': PII still present in LLM input after scrubbing: {matches}"
        )
```
