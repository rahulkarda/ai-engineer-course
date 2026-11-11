# Solutions — Week 08 · Project — Structured Extraction Service

## Complete Reference Implementation

---

### `src/extraction_service/models.py`

```python
"""Data models for the Structured Extraction Service.

Deps: pydantic>=2.0
"""
from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------

class TokenUsage(BaseModel):
    """Token usage statistics for a single LLM call or accumulated across retries."""

    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: "TokenUsage") -> "TokenUsage":
        """Return a new TokenUsage that is the sum of self and other."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ExtractionResult(BaseModel, Generic[T]):
    """Typed extraction result with token usage and attempt count."""

    data: T
    usage: TokenUsage
    attempts: int = Field(ge=1, le=3)


# ---------------------------------------------------------------------------
# Domain schemas
# ---------------------------------------------------------------------------

class ParsedResume(BaseModel):
    name: str
    email: str
    skills: list[str]
    years_experience: int
    education: list[str]


class ParsedInvoice(BaseModel):
    vendor: str
    invoice_number: str
    total_amount: float
    line_items: list[str]
    due_date: str   # ISO 8601 (YYYY-MM-DD)
```

---

### `src/extraction_service/llm_client.py`

```python
"""LLM client abstraction and implementations.

Production client uses os.getenv("ANTHROPIC_API_KEY") and
os.getenv("LLM_MODEL", "claude-opus-5-5"). No hardcoded keys or model IDs.

Deps: anthropic>=0.40, pydantic>=2.0
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

from extraction_service.models import TokenUsage

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """Injectable LLM client interface."""

    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
    ) -> tuple[str, TokenUsage]:
        """Send a prompt and return (response_text, token_usage).

        Args:
            system: system prompt text.
            user: user message text.
            max_tokens: maximum tokens in the response.

        Returns:
            Tuple of (response string, TokenUsage for this call).
        """


# ---------------------------------------------------------------------------
# Production client (Anthropic SDK)
# ---------------------------------------------------------------------------

class AnthropicClient(LLMClient):
    """Concrete client wrapping the Anthropic SDK.

    API key:  os.getenv("ANTHROPIC_API_KEY")   — never hardcoded.
    Model:    os.getenv("LLM_MODEL", "claude-opus-5-5")
    """

    def __init__(self) -> None:
        import anthropic  # imported lazily so tests never need the SDK
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = os.getenv("LLM_MODEL", "claude-opus-5-5")
        log.info("AnthropicClient initialised with model=%s", self._model)

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
    ) -> tuple[str, TokenUsage]:
        response = self._client.messages.create(
            model=self._model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
        )
        text = response.content[0].text
        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        log.debug("AnthropicClient.complete: %d in / %d out tokens",
                  usage.input_tokens, usage.output_tokens)
        return text, usage


# ---------------------------------------------------------------------------
# Mock client (for tests)
# ---------------------------------------------------------------------------

class MockLLMClient(LLMClient):
    """Test double that returns pre-scripted responses in order.

    Usage:
        client = MockLLMClient(['{"name": "Alice", ...}', 'bad json'])
        # First call returns first response, second call returns second, etc.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses: list[str] = list(responses)
        self._index = 0
        self.calls: list[tuple[str, str]] = []   # (system, user) per call

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
    ) -> tuple[str, TokenUsage]:
        if self._index >= len(self._responses):
            raise RuntimeError("MockLLMClient exhausted all scripted responses")
        self.calls.append((system, user))
        response = self._responses[self._index]
        self._index += 1
        return response, TokenUsage(input_tokens=100, output_tokens=50)
```

---

### `src/extraction_service/prompts.py`

```python
"""Prompt builders for the Structured Extraction Service.

ALL prompt strings live here. No f-strings spread through service.py.

Deps: pydantic>=2.0
"""
from __future__ import annotations

import json

from pydantic import BaseModel


def build_extraction_prompt(schema: type[BaseModel]) -> str:
    """Build a system prompt that instructs the model to extract structured JSON.

    Embeds the JSON schema and a one-shot example so the model understands
    the expected output format.

    Args:
        schema: Pydantic model class defining the extraction target.

    Returns:
        System prompt string.
    """
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    schema_name = schema.__name__

    # One-shot example using the schema's field names
    example_fields: dict[str, object] = {}
    for field_name, field_info in schema.model_fields.items():
        annotation = field_info.annotation
        if annotation is str:
            example_fields[field_name] = f"<{field_name}_value>"
        elif annotation is int:
            example_fields[field_name] = 0
        elif annotation is float:
            example_fields[field_name] = 0.0
        elif hasattr(annotation, "__origin__"):  # list[...] etc.
            example_fields[field_name] = []
        else:
            example_fields[field_name] = None

    example_json = json.dumps(example_fields, indent=2)

    return (
        f"You are a structured data extraction assistant.\n\n"
        f"Extract information from the user's text and return it as a single JSON object "
        f"that conforms to the following JSON schema for `{schema_name}`:\n\n"
        f"```json\n{schema_json}\n```\n\n"
        f"Rules:\n"
        f"- Return ONLY a valid JSON object. No markdown, no explanation, no code fences.\n"
        f"- All required fields must be present.\n"
        f"- If a field cannot be determined from the text, use a sensible default "
        f"(empty string, 0, or empty list).\n\n"
        f"Example output format:\n{example_json}"
    )


def build_repair_prompt(previous_output: str, error: str) -> str:
    """Build a repair user message that feeds the model its own bad output and the error.

    Args:
        previous_output: the raw string returned by the model on the previous attempt.
        error: the validation or parse error message.

    Returns:
        User message asking the model to fix its previous output.
    """
    return (
        f"Your previous response could not be parsed or validated.\n\n"
        f"Error:\n{error}\n\n"
        f"Your previous response was:\n{previous_output}\n\n"
        f"Please return a corrected JSON object that fixes the error above. "
        f"Return ONLY the JSON object — no markdown, no explanation."
    )
```

---

### `src/extraction_service/config.py`

```python
"""Service configuration via pydantic-settings.

Environment variables (set in .env or shell):
    ANTHROPIC_API_KEY  — Anthropic API key (never hardcode)
    LLM_MODEL          — model ID (default: claude-opus-5-5)
    MAX_RETRIES        — max repair attempts (default: 3)
    MAX_OUTPUT_TOKENS  — max tokens per LLM call (default: 1024)
    LOG_LEVEL          — logging level (default: INFO)

Deps: pydantic-settings>=2.0
"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    llm_model: str = Field(default="claude-opus-5-5", alias="LLM_MODEL")
    max_retries: int = Field(default=3, alias="MAX_RETRIES")
    max_output_tokens: int = Field(default=1024, alias="MAX_OUTPUT_TOKENS")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    model_config = {"populate_by_name": True}
```

---

### `src/extraction_service/service.py`

```python
"""ExtractionService — core extraction logic with repair loop.

Deps: pydantic>=2.0
"""
from __future__ import annotations

import json
import logging
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from extraction_service.llm_client import LLMClient
from extraction_service.models import ExtractionResult, TokenUsage
from extraction_service.prompts import build_extraction_prompt, build_repair_prompt

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ExtractionError(Exception):
    """Raised when all retry attempts are exhausted without a valid extraction."""


class ExtractionService:
    """Extract typed structured data from unstructured text.

    Uses an injectable LLMClient and a repair loop (up to max_retries attempts).
    Token usage is accumulated across all attempts and returned in the result.

    Args:
        client: any LLMClient implementation (real or mock).
        max_retries: maximum total attempts (including initial attempt).
    """

    def __init__(self, client: LLMClient, max_retries: int = 3) -> None:
        self._client = client
        self._max_retries = max_retries

    def extract(
        self,
        raw_text: str,
        schema: type[T],
    ) -> ExtractionResult[T]:
        """Extract typed data from raw_text using the given Pydantic schema.

        1. Build a system prompt that embeds schema.model_json_schema().
        2. Call client.complete with raw_text as the user message.
        3. Attempt json.loads + schema.model_validate on the response.
        4. On failure, call client.complete with a repair prompt that includes
           the previous bad output and the Pydantic/JSON error.
        5. Repeat up to max_retries total attempts.
        6. Accumulate TokenUsage across all attempts.
        7. Raise ExtractionError (not ValueError) if all attempts fail.

        Returns:
            ExtractionResult[T] with validated data, usage, and attempt count.

        Raises:
            ExtractionError: if all max_retries attempts fail.
        """
        system_prompt = build_extraction_prompt(schema)
        cumulative_usage = TokenUsage()
        previous_output: str | None = None
        last_error: str | None = None

        for attempt in range(1, self._max_retries + 1):
            if attempt == 1:
                user_message = raw_text
            else:
                assert previous_output is not None
                assert last_error is not None
                user_message = build_repair_prompt(previous_output, last_error)

            log.debug("Attempt %d/%d — calling LLM", attempt, self._max_retries)
            response_text, usage = self._client.complete(
                system=system_prompt,
                user=user_message,
            )
            cumulative_usage = cumulative_usage.add(usage)

            try:
                parsed = json.loads(response_text)
                validated = schema.model_validate(parsed)
                log.info(
                    "Extraction succeeded on attempt %d — schema=%s",
                    attempt, schema.__name__,
                )
                return ExtractionResult(
                    data=validated,
                    usage=cumulative_usage,
                    attempts=attempt,
                )
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = str(exc)
                previous_output = response_text
                log.debug(
                    "Attempt %d failed: %s", attempt, last_error[:120]
                )

        log.error(
            "ExtractionService exhausted %d attempts for schema=%s. Last error: %s",
            self._max_retries, schema.__name__, last_error,
        )
        raise ExtractionError(
            f"Failed to extract {schema.__name__} after {self._max_retries} attempts. "
            f"Last error: {last_error}"
        )
```

---

### `src/extraction_service/__init__.py`

```python
"""Structured Extraction Service — public API."""
from .models import TokenUsage, ExtractionResult, ParsedResume, ParsedInvoice
from .llm_client import LLMClient, AnthropicClient, MockLLMClient
from .service import ExtractionService, ExtractionError
from .config import Settings

__all__ = [
    "TokenUsage",
    "ExtractionResult",
    "ParsedResume",
    "ParsedInvoice",
    "LLMClient",
    "AnthropicClient",
    "MockLLMClient",
    "ExtractionService",
    "ExtractionError",
    "Settings",
]
```

---

### `tests/unit/test_models.py`

```python
"""Unit tests for data models."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from extraction_service.models import TokenUsage, ExtractionResult, ParsedResume


def test_token_usage_add() -> None:
    a = TokenUsage(input_tokens=100, output_tokens=50)
    b = TokenUsage(input_tokens=200, output_tokens=75)
    c = a.add(b)
    assert c.input_tokens == 300
    assert c.output_tokens == 125
    assert c.total_tokens == 425


def test_token_usage_total_tokens() -> None:
    u = TokenUsage(input_tokens=10, output_tokens=5)
    assert u.total_tokens == 15


def test_extraction_result_rejects_attempts_zero() -> None:
    with pytest.raises(ValidationError):
        ExtractionResult(
            data=ParsedResume(name="A", email="a@b.com",
                              skills=[], years_experience=0, education=[]),
            usage=TokenUsage(),
            attempts=0,
        )


def test_extraction_result_rejects_attempts_four() -> None:
    with pytest.raises(ValidationError):
        ExtractionResult(
            data=ParsedResume(name="A", email="a@b.com",
                              skills=[], years_experience=0, education=[]),
            usage=TokenUsage(),
            attempts=4,
        )
```

---

### `tests/unit/test_prompts.py`

```python
"""Unit tests for prompt builders."""
from __future__ import annotations

from extraction_service.models import ParsedResume
from extraction_service.prompts import build_extraction_prompt, build_repair_prompt


def test_extraction_prompt_contains_schema_name() -> None:
    prompt = build_extraction_prompt(ParsedResume)
    assert "ParsedResume" in prompt


def test_extraction_prompt_contains_json_schema() -> None:
    prompt = build_extraction_prompt(ParsedResume)
    assert "properties" in prompt or "name" in prompt


def test_repair_prompt_contains_previous_output() -> None:
    bad = '{"broken json'
    error = "Expecting property name"
    prompt = build_repair_prompt(bad, error)
    assert bad in prompt


def test_repair_prompt_contains_error() -> None:
    bad = '{"broken json'
    error = "Expecting property name"
    prompt = build_repair_prompt(bad, error)
    assert error in prompt
```

---

### `tests/unit/test_service.py`

```python
"""Unit tests for ExtractionService (all with MockLLMClient — no real API calls)."""
from __future__ import annotations

import json

import pytest

from extraction_service.llm_client import MockLLMClient
from extraction_service.models import ParsedResume, ParsedInvoice, TokenUsage
from extraction_service.service import ExtractionService, ExtractionError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_RESUME = json.dumps({
    "name": "Alice Smith",
    "email": "alice@example.com",
    "skills": ["Python", "ML"],
    "years_experience": 5,
    "education": ["BSc Computer Science"],
})

VALID_INVOICE = json.dumps({
    "vendor": "Acme Corp",
    "invoice_number": "INV-001",
    "total_amount": 1500.00,
    "line_items": ["Widget x10"],
    "due_date": "2026-12-01",
})


# ---------------------------------------------------------------------------
# Required tests
# ---------------------------------------------------------------------------

def test_extract_success_first_attempt() -> None:
    client = MockLLMClient([VALID_RESUME])
    service = ExtractionService(client, max_retries=3)
    result = service.extract("hire Alice Smith...", ParsedResume)
    assert result.attempts == 1
    assert result.data.name == "Alice Smith"


def test_extract_success_after_repair() -> None:
    client = MockLLMClient(["not valid json !!!", VALID_RESUME])
    service = ExtractionService(client, max_retries=3)
    result = service.extract("hire Alice Smith...", ParsedResume)
    assert result.attempts == 2
    assert result.data.name == "Alice Smith"


def test_extract_exhausts_retries() -> None:
    client = MockLLMClient(["bad", "also bad", "still bad"])
    service = ExtractionService(client, max_retries=3)
    with pytest.raises(ExtractionError):
        service.extract("some text", ParsedResume)


def test_token_usage_accumulates() -> None:
    """Each MockLLMClient call returns 100 input + 50 output tokens."""
    client = MockLLMClient(["bad json", VALID_RESUME])
    service = ExtractionService(client, max_retries=3)
    result = service.extract("text", ParsedResume)
    assert result.usage.input_tokens == 200   # 2 calls × 100
    assert result.usage.output_tokens == 100  # 2 calls × 50
    assert result.usage.total_tokens == 300


def test_repair_prompt_includes_previous_output() -> None:
    bad_output = "this is not json at all"
    client = MockLLMClient([bad_output, VALID_RESUME])
    service = ExtractionService(client, max_retries=3)
    service.extract("text", ParsedResume)
    # Second call (repair) is calls[1]; user message should contain the bad output
    _, repair_user = client.calls[1]
    assert bad_output in repair_user


def test_schema_agnostic_resume() -> None:
    client = MockLLMClient([VALID_RESUME])
    service = ExtractionService(client)
    result = service.extract("resume text", ParsedResume)
    assert isinstance(result.data, ParsedResume)
    assert result.data.email == "alice@example.com"


def test_schema_agnostic_invoice() -> None:
    client = MockLLMClient([VALID_INVOICE])
    service = ExtractionService(client)
    result = service.extract("invoice text", ParsedInvoice)
    assert isinstance(result.data, ParsedInvoice)
    assert result.data.vendor == "Acme Corp"


def test_extraction_result_model_rejects_invalid_attempts() -> None:
    from pydantic import ValidationError
    from extraction_service.models import ExtractionResult
    with pytest.raises(ValidationError):
        ExtractionResult(data=ParsedResume(
            name="X", email="x@x.com", skills=[], years_experience=0, education=[]
        ), usage=TokenUsage(), attempts=0)
    with pytest.raises(ValidationError):
        ExtractionResult(data=ParsedResume(
            name="X", email="x@x.com", skills=[], years_experience=0, education=[]
        ), usage=TokenUsage(), attempts=4)
```

---

### `tests/integration/test_anthropic_client.py`

```python
"""Integration test for AnthropicClient — skipped unless ANTHROPIC_API_KEY is set.

Run with:  ANTHROPIC_API_KEY=... pytest tests/integration/ -v
"""
from __future__ import annotations

import os

import pytest

from extraction_service.llm_client import AnthropicClient


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping live API test",
)
def test_anthropic_client_smoke() -> None:
    client = AnthropicClient()
    response, usage = client.complete(
        system="Respond with exactly one word.",
        user="Say: hello",
        max_tokens=10,
    )
    assert isinstance(response, str) and len(response) > 0
    assert usage.total_tokens > 0
```

---

### `pyproject.toml`

```toml
[project]
name = "extraction-service"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "anthropic>=0.40",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-cov",
    "ruff",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py312"
```

---

### `.env.example`

```
ANTHROPIC_API_KEY=
LLM_MODEL=claude-opus-5-5
MAX_RETRIES=3
MAX_OUTPUT_TOKENS=1024
LOG_LEVEL=INFO
```

---

## Design decisions

**`LLMClient` as an ABC, not a Protocol**: An ABC with `@abstractmethod` raises `TypeError`
on instantiation if `complete` is not implemented, giving a clear error message. A Protocol
is structurally typed (duck-typed), which allows accidental compatibility. For an injectable
service boundary, the explicit ABC contract is preferable.

**`ExtractionError` vs `ValueError`**: raising a domain-specific exception lets callers
`except ExtractionError` without accidentally catching unrelated `ValueError`s from Pydantic
or the standard library. This is the open/closed principle applied to error handling.

**All prompt strings in `prompts.py`**: prompt text has its own lifecycle — it needs to be
reviewed, version-controlled, and tested independently of the service logic. Centralising it
means a prompt change never requires touching `service.py`.

**`MockLLMClient.calls`**: storing `(system, user)` per call enables tests to assert on
repair prompt content without monkey-patching or inspecting internal state.

**No `print()` anywhere**: all diagnostic output goes through `log.debug` / `log.info` /
`log.error` so callers can control verbosity via `logging.basicConfig` or the `LOG_LEVEL`
env var.

---

## Verification checklist

```bash
# Run unit tests (no API key needed)
unset ANTHROPIC_API_KEY
pytest tests/unit/ -v --cov=src/extraction_service --cov-report=term-missing

# Lint
ruff check src/ tests/

# No print in src/
grep -r "print(" src/

# .env in .gitignore
grep ".env" .gitignore
```
