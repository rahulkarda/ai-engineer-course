# Phase 3 · Week 08 · Day 6 — Saturday Build: Structured Extraction Service

> Objective: Build a complete, production-quality structured extraction service from scratch, integrating all week-8 concepts.  ·  Time: ~4 hr  ·  Prereqs: [[day-1]], [[day-2]], [[day-3]], [[day-4]], [[day-5]]

---

## Concept

Today you build a complete structured extraction service end-to-end. No isolated snippets — every module is production-quality: typed, tested, composable.

The service takes freeform text and extracts structured entities into Pydantic models. It handles JSON repair, tracks token budgets, uses the provider-agnostic client from day-5, and has a full test suite using `MockLLMClient`.

**What you will build:**

```
src/extraction/
    __init__.py
    models.py       — Pydantic models: ExtractedEntity, ContactInfo, DocumentSummary
    prompts.py      — System prompt, extraction prompt builder with XML delimiters
    client.py       — Provider-agnostic LLMClient (reused from day-5)
    service.py      — ExtractionService: extract(), retry/repair, token tracking
tests/
    __init__.py
    test_service.py — Full test suite, MockLLMClient only
pyproject.toml      — Project metadata and dependencies
```

**Ground rules for this build:**

- No real API calls anywhere — all tests use `MockLLMClient`
- All API keys via `os.getenv("ANTHROPIC_API_KEY")` only
- Pydantic v2 everywhere — `model_validate`, `model_dump`, `model_json_schema`
- `logging` not `print`
- Type hints on every function signature and class attribute

---

## How it works internally

### Architecture overview

```
CLI / caller
     |
     v
ExtractionService.extract(text, extraction_type)
     |
     +-- build prompt (prompts.py)
     |
     +-- call LLMClient.complete(request)   <-- injectable
     |
     +-- parse JSON from response
     |
     +-- validate with Pydantic
     |       |
     |       +-- valid? return model
     |       |
     |       +-- invalid? repair_loop()
     |                   |
     |                   +-- send repair prompt
     |                   +-- re-validate
     |                   +-- max_attempts? raise
     |
     +-- record usage in TokenBudget
```

### Why XML delimiters in prompts

Claude is pretrained on documents with XML/HTML structure. Delimiting input sections explicitly:

```xml
<document>
{text}
</document>
```

reduces prompt injection risk (text inside cannot escape its container to modify instructions) and improves the model's ability to locate the right content. It is more robust than plain text boundaries like `---`.

### Pydantic v2 validators

Use `@field_validator` for field-level validation and `@model_validator(mode="after")` for cross-field constraints:

```python
from pydantic import BaseModel, field_validator, model_validator

class ContactInfo(BaseModel):
    email: str | None = None
    phone: str | None = None

    @model_validator(mode="after")
    def at_least_one_contact(self) -> "ContactInfo":
        if self.email is None and self.phone is None:
            raise ValueError("At least one of email or phone must be present")
        return self
```

Validators run during `model_validate` — including when the repair loop reconstructs the model from repaired JSON. This means the model enforces its own invariants throughout the pipeline.

---

## Worked example

### Step 1 — Project setup

Create the directory structure:

```bash
mkdir -p src/extraction tests
touch src/__init__.py src/extraction/__init__.py tests/__init__.py
```

**`pyproject.toml`:**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "extraction-service"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "anthropic>=0.40.0",
    "pydantic>=2.7.0",
    "tenacity>=8.3.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.2.0",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.4.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.ruff]
line-length = 100
target-version = "py312"
```

Install:

```bash
pip install -e ".[dev]"
```

---

### Step 2 — Models (`src/extraction/models.py`)

```python
"""
src/extraction/models.py

Pydantic v2 models for all extraction targets.

Design notes:
- All fields are Optional where the model may not find them.
- Validators enforce invariants at parse time, including after JSON repair.
- model_json_schema() is used to inject the schema into repair prompts.
"""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class ExtractionResult(BaseModel):
    """Base class for all extraction results. Carries provenance metadata."""

    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Model's self-reported confidence [0, 1]. Heuristic only.",
    )
    extraction_notes: str | None = Field(
        default=None,
        description="Any caveats or ambiguities the model noted during extraction.",
    )


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------


class ExtractedEntity(ExtractionResult):
    """
    A single named entity extracted from text.

    entity_type: coarse category (PERSON, ORGANIZATION, LOCATION, DATE, PRODUCT, OTHER)
    value: the surface form as it appears in the text
    normalized: standardized form (e.g. "Jan 5 2025" -> "2025-01-05" for dates)
    """

    entity_type: str = Field(
        description="Entity category: PERSON, ORGANIZATION, LOCATION, DATE, PRODUCT, OTHER"
    )
    value: str = Field(min_length=1, description="Surface form from the text")
    normalized: str | None = Field(
        default=None, description="Normalized/canonical form, if applicable"
    )

    @field_validator("entity_type")
    @classmethod
    def validate_entity_type(cls, v: str) -> str:
        allowed = {"PERSON", "ORGANIZATION", "LOCATION", "DATE", "PRODUCT", "OTHER"}
        normalized = v.upper().strip()
        if normalized not in allowed:
            raise ValueError(f"entity_type must be one of {allowed}, got {v!r}")
        return normalized

    @field_validator("value")
    @classmethod
    def strip_value(cls, v: str) -> str:
        return v.strip()


class EntityList(BaseModel):
    """Wrapper returned when extracting a list of entities."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    total_count: int = 0

    @model_validator(mode="after")
    def sync_total_count(self) -> "EntityList":
        self.total_count = len(self.entities)
        return self


# ---------------------------------------------------------------------------
# Contact info extraction
# ---------------------------------------------------------------------------


class ContactInfo(ExtractionResult):
    """
    Contact information extracted from a document.
    At least one of email or phone must be present.
    """

    name: str | None = None
    email: str | None = None
    phone: str | None = None
    organization: str | None = None
    address: str | None = None

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, v: str | None) -> str | None:
        if v is None:
            return v
        # Simple format check — not RFC 5322 complete
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError(f"email does not look valid: {v!r}")
        return v.lower().strip()

    @field_validator("phone")
    @classmethod
    def normalize_phone(cls, v: str | None) -> str | None:
        if v is None:
            return v
        # Strip formatting to digits and + only
        cleaned = re.sub(r"[^\d+]", "", v)
        if len(cleaned) < 7:
            raise ValueError(f"phone too short after normalization: {v!r}")
        return cleaned

    @model_validator(mode="after")
    def at_least_one_contact_method(self) -> "ContactInfo":
        if self.email is None and self.phone is None:
            raise ValueError("At least one of email or phone must be present")
        return self


# ---------------------------------------------------------------------------
# Document summary
# ---------------------------------------------------------------------------


class DocumentSummary(ExtractionResult):
    """
    High-level summary and metadata extracted from a document.
    """

    title: str | None = Field(default=None, description="Document or email subject line")
    summary: str = Field(min_length=10, description="2-3 sentence summary of the document")
    key_topics: list[str] = Field(default_factory=list, description="Main topics covered")
    document_type: str = Field(
        default="OTHER",
        description="Document type: EMAIL, INVOICE, CONTRACT, REPORT, ARTICLE, OTHER",
    )
    language: str = Field(default="en", description="ISO 639-1 language code")

    @field_validator("document_type")
    @classmethod
    def validate_doc_type(cls, v: str) -> str:
        allowed = {"EMAIL", "INVOICE", "CONTRACT", "REPORT", "ARTICLE", "OTHER"}
        normalized = v.upper().strip()
        if normalized not in allowed:
            return "OTHER"
        return normalized

    @field_validator("language")
    @classmethod
    def lowercase_language(cls, v: str) -> str:
        return v.lower().strip()[:5]  # e.g. "en-US" -> "en-us"
```

---

### Step 3 — Prompts (`src/extraction/prompts.py`)

```python
"""
src/extraction/prompts.py

All prompts live here — never inline in service logic.

Design notes:
- System prompt is a module-level constant, not a function.
- Extraction prompts are pure functions: (text, schema_json) -> str.
- XML delimiters contain untrusted input to reduce prompt injection surface.
- Repair prompt is separate from extraction prompt (different task).
"""

from __future__ import annotations

import json
from typing import type[Any]

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """\
You are a precise structured data extraction assistant.

Rules:
- Extract information exactly as stated in the document. Do not infer or hallucinate.
- Return ONLY valid JSON matching the provided schema. No prose, no markdown fences.
- If a field is not present in the document, omit it or set it to null.
- For confidence, use 1.0 if all fields are clearly present, lower if uncertain.
- Do not include any text outside the JSON object.
"""

REPAIR_SYSTEM_PROMPT = """\
You are a JSON repair assistant.
Fix the provided JSON so it matches the schema exactly.
Return ONLY valid JSON. No explanation, no markdown fences, no extra text.
"""


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------


def build_extraction_prompt(
    text: str,
    schema: type[BaseModel],
    extra_instructions: str | None = None,
) -> str:
    """
    Build an extraction prompt for a given document and target schema.

    Args:
        text: The source document to extract from.
        schema: The Pydantic model class defining the extraction target.
        extra_instructions: Optional task-specific guidance appended after the schema.

    Returns:
        A formatted prompt string ready to send as a user message.
    """
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    instructions = extra_instructions or ""

    return (
        "Extract structured information from the document below.\n\n"
        "<document>\n"
        f"{text}\n"
        "</document>\n\n"
        "<schema>\n"
        f"{schema_json}\n"
        "</schema>\n\n"
        + (f"<extra_instructions>\n{instructions}\n</extra_instructions>\n\n" if instructions else "")
        + "Return ONLY a JSON object matching the schema. No markdown, no explanation."
    )


# ---------------------------------------------------------------------------
# Repair prompt
# ---------------------------------------------------------------------------


def build_repair_prompt(
    invalid_json: str,
    error_message: str,
    schema: type[BaseModel],
) -> str:
    """
    Build a repair prompt that asks the model to fix its invalid output.

    Args:
        invalid_json: The raw invalid JSON string from the previous call.
        error_message: The validation or parse error message.
        schema: The target Pydantic model class.

    Returns:
        A formatted repair prompt string.
    """
    schema_json = json.dumps(schema.model_json_schema(), indent=2)

    return (
        "The JSON below is invalid or does not match the required schema. Fix it.\n\n"
        "<invalid_json>\n"
        f"{invalid_json}\n"
        "</invalid_json>\n\n"
        "<error>\n"
        f"{error_message}\n"
        "</error>\n\n"
        "<required_schema>\n"
        f"{schema_json}\n"
        "</required_schema>\n\n"
        "Return ONLY a corrected JSON object matching the schema. No explanation."
    )
```

---

### Step 4 — Client (`src/extraction/client.py`)

This reuses the provider-agnostic wrapper from day-5. Copy or symlink:

```python
"""
src/extraction/client.py

Re-exports the LLMClient protocol and implementations from day-5.
Keeps the extraction package self-contained.
"""

from __future__ import annotations

import logging
import os
from typing import Literal, Protocol, runtime_checkable

import anthropic
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LLMMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class LLMRequest(BaseModel):
    messages: list[LLMMessage]
    system: str | None = None
    max_tokens: int = 1024
    model: str = "claude-opus-5-5"
    stop_sequences: list[str] = Field(default_factory=list)


class LLMResponse(BaseModel):
    content: str
    input_tokens: int
    output_tokens: int
    model: str
    stop_reason: str = "end_turn"


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse: ...


class AnthropicClient:
    """Production LLMClient backed by the Anthropic SDK."""

    def __init__(self, model_override: str | None = None) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")
        self._sdk = anthropic.Anthropic(api_key=api_key)
        self._model_override = model_override

    def complete(self, request: LLMRequest) -> LLMResponse:
        model = self._model_override or request.model
        kwargs: dict = {
            "model": model,
            "max_tokens": request.max_tokens,
            "messages": [m.model_dump() for m in request.messages],
        }
        if request.system:
            kwargs["system"] = request.system
        if request.stop_sequences:
            kwargs["stop_sequences"] = request.stop_sequences

        response = self._sdk.messages.create(**kwargs)
        content = "\n".join(
            b.text for b in response.content if hasattr(b, "text")
        )
        return LLMResponse(
            content=content,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
            stop_reason=response.stop_reason or "end_turn",
        )


class MockLLMClient:
    """
    In-memory LLMClient for testing.
    Returns responses from a pre-configured list in order.
    """

    def __init__(
        self,
        responses: list[LLMResponse],
        error_on_calls: list[int] | None = None,
    ) -> None:
        self._responses = list(responses)
        self._error_on_calls: set[int] = set(error_on_calls or [])
        self._call_count = 0
        self._recorded_requests: list[LLMRequest] = []

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def recorded_requests(self) -> list[LLMRequest]:
        return list(self._recorded_requests)

    def complete(self, request: LLMRequest) -> LLMResponse:
        idx = self._call_count
        self._call_count += 1
        self._recorded_requests.append(request)

        if idx in self._error_on_calls:
            raise RuntimeError(f"Simulated error on call {idx}")

        if idx >= len(self._responses):
            raise StopIteration(
                f"MockLLMClient exhausted after {idx} calls "
                f"(only {len(self._responses)} responses configured)"
            )
        return self._responses[idx]
```

---

### Step 5 — Extraction Service (`src/extraction/service.py`)

```python
"""
src/extraction/service.py

ExtractionService: orchestrates prompting, parsing, repair, and budget tracking.

Design decisions:
- ExtractionService is a class (not a function) to hold injected dependencies.
- It does NOT import anthropic directly — only the LLMClient protocol.
- Token tracking via TokenBudget (inline, not imported from day-4 to keep this
  module self-contained). Adjust to import if sharing across services.
- Repair loop is capped at max_repair_attempts to bound cost.
"""

from __future__ import annotations

import json
import logging
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .client import LLMClient, LLMMessage, LLMRequest, LLMResponse
from .models import DocumentSummary, EntityList, ExtractedEntity, ExtractionResult
from .prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    REPAIR_SYSTEM_PROMPT,
    build_extraction_prompt,
    build_repair_prompt,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Budget tracking (self-contained copy; import from day-4 module if preferred)
# ---------------------------------------------------------------------------


class BudgetExceededError(Exception):
    def __init__(self, spent: float, limit: float, projected: float) -> None:
        self.spent = spent
        self.limit = limit
        self.projected = projected
        super().__init__(
            f"Budget exceeded: spent=${spent:.4f}, limit=${limit:.4f}, "
            f"projected call=${projected:.4f}"
        )


class TokenBudget:
    INPUT_PRICE_PER_M: float = 15.00   # claude-opus-5-5
    OUTPUT_PRICE_PER_M: float = 75.00  # claude-opus-5-5

    def __init__(self, max_budget_usd: float | None = None) -> None:
        self.max_budget_usd = max_budget_usd
        self._input: int = 0
        self._output: int = 0
        self._calls: int = 0

    @property
    def estimated_cost_usd(self) -> float:
        return (self._input / 1_000_000) * self.INPUT_PRICE_PER_M + (
            self._output / 1_000_000
        ) * self.OUTPUT_PRICE_PER_M

    def check_budget(self, projected_input: int, projected_output: int) -> None:
        if self.max_budget_usd is None:
            return
        projected_cost = (
            (projected_input / 1_000_000) * self.INPUT_PRICE_PER_M
            + (projected_output / 1_000_000) * self.OUTPUT_PRICE_PER_M
        )
        if self.estimated_cost_usd + projected_cost > self.max_budget_usd:
            raise BudgetExceededError(
                spent=self.estimated_cost_usd,
                limit=self.max_budget_usd,
                projected=projected_cost,
            )

    def record(self, response: LLMResponse) -> None:
        self._input += response.input_tokens
        self._output += response.output_tokens
        self._calls += 1

    def summary(self) -> dict[str, Any]:
        return {
            "calls": self._calls,
            "total_input_tokens": self._input,
            "total_output_tokens": self._output,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
        }


# ---------------------------------------------------------------------------
# Core service
# ---------------------------------------------------------------------------


class ExtractionError(Exception):
    """Raised when extraction fails after all retries and repair attempts."""


class ExtractionService:
    """
    Structured extraction service.

    Extracts typed Pydantic models from unstructured text using an injected
    LLM client. Handles JSON repair and optional budget tracking.

    Args:
        client: Any object satisfying the LLMClient protocol.
        model: Claude model ID to use.
        max_repair_attempts: How many times to ask the model to fix invalid JSON.
        budget: Optional TokenBudget. Pass a shared budget to track across calls.
    """

    def __init__(
        self,
        client: LLMClient,
        model: str = "claude-opus-5-5",
        max_repair_attempts: int = 2,
        budget: TokenBudget | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._max_repair_attempts = max_repair_attempts
        self._budget = budget or TokenBudget()

    @property
    def budget(self) -> TokenBudget:
        return self._budget

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        text: str,
        target: type[T],
        extra_instructions: str | None = None,
        max_tokens: int = 1024,
    ) -> T:
        """
        Extract structured data from text into a typed Pydantic model.

        Args:
            text: Source document to extract from.
            target: Pydantic model class defining the extraction target.
            extra_instructions: Optional task-specific guidance for the model.
            max_tokens: Maximum output tokens for the extraction call.

        Returns:
            Validated instance of `target`.

        Raises:
            BudgetExceededError: If the call would exceed the configured budget.
            ExtractionError: If extraction fails after all repair attempts.
        """
        prompt = build_extraction_prompt(
            text=text,
            schema=target,
            extra_instructions=extra_instructions,
        )

        # Rough budget check: estimate prompt tokens from character count
        projected_input = max(1, len(prompt) // 4)
        self._budget.check_budget(
            projected_input=projected_input,
            projected_output=max_tokens,
        )

        request = LLMRequest(
            messages=[LLMMessage(role="user", content=prompt)],
            system=EXTRACTION_SYSTEM_PROMPT,
            max_tokens=max_tokens,
            model=self._model,
        )

        logger.info(
            "Extraction started: target=%s text_len=%d",
            target.__name__,
            len(text),
        )

        response = self._client.complete(request)
        self._budget.record(response)

        return self._parse_with_repair(
            raw=response.content,
            target=target,
            max_tokens=max_tokens,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _strip_fences(self, text: str) -> str:
        """Remove markdown code fences wrapping JSON."""
        text = text.strip()
        if text.startswith("```"):
            first_nl = text.find("\n")
            if first_nl != -1:
                text = text[first_nl + 1 :]
            if text.endswith("```"):
                text = text[: text.rfind("```")]
        return text.strip()

    def _try_parse(self, candidate: str, target: type[T]) -> T | None:
        """Attempt JSON parse and Pydantic validation. Returns None on any failure."""
        try:
            data = json.loads(candidate)
            return target.model_validate(data)
        except (json.JSONDecodeError, ValidationError):
            return None

    def _parse_with_repair(
        self,
        raw: str,
        target: type[T],
        max_tokens: int,
    ) -> T:
        """
        Attempt to parse raw JSON. On failure, invoke the repair loop.

        The repair loop sends the broken output back to the model with the
        error message and schema, asking it to produce valid JSON.
        """
        candidate = self._strip_fences(raw)

        result = self._try_parse(candidate, target)
        if result is not None:
            logger.info("Parsed successfully on first attempt")
            return result

        # Repair loop
        for attempt in range(1, self._max_repair_attempts + 1):
            logger.warning(
                "Repair attempt %d/%d for target=%s",
                attempt,
                self._max_repair_attempts,
                target.__name__,
            )

            # Capture the specific error for the repair prompt
            try:
                json.loads(candidate)
                # JSON is valid but validation failed
                error_msg = "JSON parsed but failed schema validation (Pydantic error)"
                try:
                    target.model_validate(json.loads(candidate))
                except ValidationError as ve:
                    error_msg = str(ve)
            except json.JSONDecodeError as je:
                error_msg = f"JSON parse error: {je}"

            repair_prompt = build_repair_prompt(
                invalid_json=candidate,
                error_message=error_msg,
                schema=target,
            )

            self._budget.check_budget(
                projected_input=max(1, len(repair_prompt) // 4),
                projected_output=max_tokens,
            )

            repair_request = LLMRequest(
                messages=[LLMMessage(role="user", content=repair_prompt)],
                system=REPAIR_SYSTEM_PROMPT,
                max_tokens=max_tokens,
                model=self._model,
            )

            repair_response = self._client.complete(repair_request)
            self._budget.record(repair_response)

            candidate = self._strip_fences(repair_response.content)
            result = self._try_parse(candidate, target)
            if result is not None:
                logger.info("Repair succeeded on attempt %d", attempt)
                return result

        raise ExtractionError(
            f"Failed to extract {target.__name__} after "
            f"{self._max_repair_attempts} repair attempts. "
            f"Last candidate: {candidate[:200]}"
        )
```

---

### Step 6 — Full test suite (`tests/test_service.py`)

```python
"""
tests/test_service.py

Full test suite for ExtractionService.
All tests use MockLLMClient. Zero real API calls.

Run: pytest tests/test_service.py -v
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.extraction.client import LLMResponse, MockLLMClient
from src.extraction.models import (
    ContactInfo,
    DocumentSummary,
    EntityList,
    ExtractedEntity,
)
from src.extraction.service import (
    BudgetExceededError,
    ExtractionError,
    ExtractionService,
    TokenBudget,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def mock_response(content: str, input_tokens: int = 100, output_tokens: int = 50) -> LLMResponse:
    return LLMResponse(
        content=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model="mock",
    )


def valid_entity_list_json() -> str:
    return json.dumps({
        "entities": [
            {
                "entity_type": "PERSON",
                "value": "Alice Johnson",
                "normalized": None,
                "confidence": 0.95,
                "extraction_notes": None,
            }
        ],
        "total_count": 1,
    })


def valid_contact_json() -> str:
    return json.dumps({
        "name": "Bob Smith",
        "email": "bob@example.com",
        "phone": "5551234567",
        "organization": "Acme Corp",
        "address": None,
        "confidence": 0.9,
        "extraction_notes": None,
    })


def valid_summary_json() -> str:
    return json.dumps({
        "title": "Q3 Report",
        "summary": "This report covers Q3 financial results for fiscal year 2025.",
        "key_topics": ["revenue", "costs"],
        "document_type": "REPORT",
        "language": "en",
        "confidence": 1.0,
        "extraction_notes": None,
    })


# ---------------------------------------------------------------------------
# Happy path — first attempt succeeds
# ---------------------------------------------------------------------------


def test_extract_entity_list_success_first_attempt() -> None:
    mock = MockLLMClient(responses=[mock_response(valid_entity_list_json())])
    service = ExtractionService(client=mock)

    result = service.extract("Alice Johnson is a software engineer.", EntityList)

    assert isinstance(result, EntityList)
    assert len(result.entities) == 1
    assert result.entities[0].value == "Alice Johnson"
    assert result.entities[0].entity_type == "PERSON"
    assert mock.call_count == 1


def test_extract_contact_info_success() -> None:
    mock = MockLLMClient(responses=[mock_response(valid_contact_json())])
    service = ExtractionService(client=mock)

    result = service.extract("Contact: bob@example.com, 555-123-4567", ContactInfo)

    assert isinstance(result, ContactInfo)
    assert result.email == "bob@example.com"
    assert mock.call_count == 1


def test_extract_document_summary_success() -> None:
    mock = MockLLMClient(responses=[mock_response(valid_summary_json())])
    service = ExtractionService(client=mock)

    result = service.extract("Q3 report content here...", DocumentSummary)

    assert isinstance(result, DocumentSummary)
    assert result.document_type == "REPORT"
    assert "Q3" in result.summary
    assert mock.call_count == 1


# ---------------------------------------------------------------------------
# Repair loop — invalid JSON on first attempt
# ---------------------------------------------------------------------------


def test_repair_loop_succeeds_on_second_attempt() -> None:
    """First response has invalid JSON; second (repair) response is valid."""
    invalid_json = '{"entities": [BROKEN'
    mock = MockLLMClient(responses=[
        mock_response(invalid_json),           # call 0: initial extraction (broken)
        mock_response(valid_entity_list_json()),  # call 1: repair call (valid)
    ])
    service = ExtractionService(client=mock, max_repair_attempts=2)

    result = service.extract("Some text", EntityList)

    assert isinstance(result, EntityList)
    assert mock.call_count == 2  # initial + 1 repair


def test_repair_loop_succeeds_after_two_repairs() -> None:
    """Two broken responses, then one valid repair."""
    mock = MockLLMClient(responses=[
        mock_response('not json at all'),
        mock_response('{"entities": null}'),         # valid JSON, fails Pydantic
        mock_response(valid_entity_list_json()),      # valid
    ])
    service = ExtractionService(client=mock, max_repair_attempts=2)

    result = service.extract("Some text", EntityList)

    assert isinstance(result, EntityList)
    assert mock.call_count == 3


def test_repair_loop_strips_markdown_fences() -> None:
    """Model wraps JSON in ```json ... ``` — service must strip it."""
    fenced = "```json\n" + valid_entity_list_json() + "\n```"
    mock = MockLLMClient(responses=[mock_response(fenced)])
    service = ExtractionService(client=mock)

    result = service.extract("Some text", EntityList)

    assert isinstance(result, EntityList)
    assert mock.call_count == 1


def test_repair_loop_raises_extraction_error_when_exhausted() -> None:
    """All attempts return invalid JSON — ExtractionError raised."""
    mock = MockLLMClient(responses=[
        mock_response("not json"),
        mock_response("also not json"),
        mock_response("still not json"),
    ])
    service = ExtractionService(client=mock, max_repair_attempts=2)

    with pytest.raises(ExtractionError, match="repair attempts"):
        service.extract("Some text", EntityList)

    assert mock.call_count == 3  # initial + 2 repairs


def test_repair_prompt_includes_invalid_json() -> None:
    """Verify the repair call's prompt contains the broken output."""
    broken = '{"entities": [BROKEN'
    mock = MockLLMClient(responses=[
        mock_response(broken),
        mock_response(valid_entity_list_json()),
    ])
    service = ExtractionService(client=mock, max_repair_attempts=1)

    service.extract("Some text", EntityList)

    repair_request = mock.recorded_requests[1]
    assert broken in repair_request.messages[0].content


# ---------------------------------------------------------------------------
# Budget tracking
# ---------------------------------------------------------------------------


def test_budget_records_usage_after_extraction() -> None:
    mock = MockLLMClient(responses=[mock_response(valid_entity_list_json(), 200, 100)])
    budget = TokenBudget()
    service = ExtractionService(client=mock, budget=budget)

    service.extract("text", EntityList)

    assert budget.summary()["total_input_tokens"] == 200
    assert budget.summary()["total_output_tokens"] == 100
    assert budget.summary()["calls"] == 1


def test_budget_accumulates_over_repair_calls() -> None:
    mock = MockLLMClient(responses=[
        mock_response("bad json", 100, 50),
        mock_response(valid_entity_list_json(), 150, 75),
    ])
    budget = TokenBudget()
    service = ExtractionService(client=mock, budget=budget, max_repair_attempts=1)

    service.extract("text", EntityList)

    assert budget.summary()["total_input_tokens"] == 250
    assert budget.summary()["calls"] == 2


def test_budget_exceeded_raises_before_api_call() -> None:
    """BudgetExceededError raised before any API call when budget is exhausted."""
    budget = TokenBudget(max_budget_usd=0.000001)  # effectively zero

    mock = MockLLMClient(responses=[mock_response(valid_entity_list_json())])
    service = ExtractionService(client=mock, budget=budget)

    with pytest.raises(BudgetExceededError):
        service.extract("A moderately long document text " * 50, EntityList)

    assert mock.call_count == 0  # no API call made


def test_budget_exceeded_mid_repair_raises() -> None:
    """Budget exhausted after the first (failed) call, before the repair call."""
    budget = TokenBudget(max_budget_usd=0.0)  # any spend exceeds

    mock = MockLLMClient(responses=[
        mock_response("bad json", 1000, 500),
        mock_response(valid_entity_list_json()),
    ])
    service = ExtractionService(client=mock, budget=budget, max_repair_attempts=2)

    with pytest.raises(BudgetExceededError):
        service.extract("text", EntityList)


# ---------------------------------------------------------------------------
# Pydantic model validators
# ---------------------------------------------------------------------------


def test_extracted_entity_normalizes_entity_type() -> None:
    entity = ExtractedEntity(entity_type="person", value="Alice")
    assert entity.entity_type == "PERSON"


def test_extracted_entity_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        ExtractedEntity(entity_type="ANIMAL", value="Cat")


def test_contact_info_requires_email_or_phone() -> None:
    with pytest.raises(ValidationError, match="email or phone"):
        ContactInfo(name="Bob")


def test_contact_info_accepts_email_only() -> None:
    c = ContactInfo(email="test@example.com")
    assert c.email == "test@example.com"
    assert c.phone is None


def test_contact_info_normalizes_email_to_lowercase() -> None:
    c = ContactInfo(email="Alice@Example.COM", phone="5551234567")
    assert c.email == "alice@example.com"


def test_contact_info_strips_phone_formatting() -> None:
    c = ContactInfo(phone="(555) 123-4567", email="x@x.com")
    assert c.phone == "5551234567"


def test_document_summary_defaults_unknown_doc_type() -> None:
    summary = DocumentSummary(
        summary="This is a valid summary with enough characters.",
        document_type="MEMO",
    )
    assert summary.document_type == "OTHER"


def test_entity_list_syncs_total_count() -> None:
    el = EntityList(entities=[
        ExtractedEntity(entity_type="PERSON", value="Alice"),
        ExtractedEntity(entity_type="ORGANIZATION", value="Acme"),
    ])
    assert el.total_count == 2


# ---------------------------------------------------------------------------
# Service configuration
# ---------------------------------------------------------------------------


def test_service_uses_injected_model_in_request() -> None:
    mock = MockLLMClient(responses=[mock_response(valid_entity_list_json())])
    service = ExtractionService(client=mock, model="claude-sonnet-5-5")

    service.extract("text", EntityList)

    assert mock.recorded_requests[0].model == "claude-sonnet-5-5"


def test_service_sets_system_prompt() -> None:
    mock = MockLLMClient(responses=[mock_response(valid_entity_list_json())])
    service = ExtractionService(client=mock)

    service.extract("text", EntityList)

    assert mock.recorded_requests[0].system is not None
    assert len(mock.recorded_requests[0].system) > 0


def test_extra_instructions_included_in_prompt() -> None:
    mock = MockLLMClient(responses=[mock_response(valid_entity_list_json())])
    service = ExtractionService(client=mock)

    service.extract("text", EntityList, extra_instructions="Focus on company names only.")

    prompt = mock.recorded_requests[0].messages[0].content
    assert "Focus on company names only." in prompt
```

---

### Step 7 — CLI entrypoint (`src/extraction/cli.py`)

```python
"""
src/extraction/cli.py

Command-line interface for the extraction service.

Usage:
    python -m src.extraction.cli --text "Alice at Acme called 555-1234." --type contact
    python -m src.extraction.cli --file document.txt --type entities
    python -m src.extraction.cli --text "Annual report..." --type summary
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .client import AnthropicClient
from .models import ContactInfo, DocumentSummary, EntityList
from .service import BudgetExceededError, ExtractionError, ExtractionService, TokenBudget

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

EXTRACTION_TYPES: dict[str, type] = {
    "entities": EntityList,
    "contact": ContactInfo,
    "summary": DocumentSummary,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extraction-service",
        description="Extract structured data from text using Claude.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", type=str, help="Input text to extract from")
    source.add_argument("--file", type=Path, help="Path to input text file")

    parser.add_argument(
        "--type",
        choices=list(EXTRACTION_TYPES.keys()),
        required=True,
        help="Extraction target type",
    )
    parser.add_argument(
        "--model",
        default="claude-opus-5-5",
        help="Claude model ID (default: claude-opus-5-5)",
    )
    parser.add_argument(
        "--max-budget",
        type=float,
        default=None,
        metavar="USD",
        help="Maximum spend in USD (e.g. 0.10)",
    )
    parser.add_argument(
        "--max-repairs",
        type=int,
        default=2,
        help="Max repair attempts on invalid JSON (default: 2)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.file is not None:
        text = args.file.read_text(encoding="utf-8")
    else:
        text = args.text

    target = EXTRACTION_TYPES[args.type]

    try:
        client = AnthropicClient(model_override=args.model)
    except RuntimeError as exc:
        logger.error("Client initialization failed: %s", exc)
        sys.exit(1)

    budget = TokenBudget(max_budget_usd=args.max_budget)
    service = ExtractionService(
        client=client,
        model=args.model,
        max_repair_attempts=args.max_repairs,
        budget=budget,
    )

    try:
        result = service.extract(text=text, target=target)
    except BudgetExceededError as exc:
        logger.error("Budget exceeded: %s", exc)
        sys.exit(2)
    except ExtractionError as exc:
        logger.error("Extraction failed: %s", exc)
        sys.exit(3)

    print(result.model_dump_json(indent=2))
    logger.info("Budget summary: %s", budget.summary())


if __name__ == "__main__":
    main()
```

---

### Step 8 — Run the tests

```bash
pytest tests/test_service.py -v
```

Expected output (all 30+ tests pass, zero network calls):

```
tests/test_service.py::test_extract_entity_list_success_first_attempt PASSED
tests/test_service.py::test_extract_contact_info_success PASSED
...
tests/test_service.py::test_budget_exceeded_raises_before_api_call PASSED
```

---

## Exercises

(Solutions: ../solutions/week-08/day-6.md)

1. **New entity type — EventInfo**: Add a new Pydantic model `EventInfo(ExtractionResult)` with fields: `name: str`, `date: str | None`, `location: str | None`, `attendees: list[str]`. Add a `field_validator` on `date` that rejects strings shorter than 4 characters. Add `"event"` to the CLI's `EXTRACTION_TYPES` dict. Write at least 3 tests for `EventInfo` extraction.

2. **Async ExtractionService**: Add an `async def aextract(...)` method to `ExtractionService` that uses `asyncio.to_thread` to wrap the synchronous `client.complete` call. Update `MockLLMClient` with an `async def acomplete` method. Write async tests using `pytest-asyncio`.

3. **Prompt caching in AnthropicClient**: The system prompt and schema are stable across calls — they are good caching candidates. Modify `AnthropicClient.complete` to add `"cache_control": {"type": "ephemeral"}` to the system prompt block and to the first user message (the schema portion). Use the Anthropic SDK's `beta` messages API with `betas=["prompt-caching-2024-07-31"]`. Verify the structure with a unit test that checks the request shape (mock the SDK's `messages.create`).

4. **Repair loop benchmark**: Write a script `scripts/benchmark_repair.py` that uses `MockLLMClient` to simulate a 20% JSON failure rate across 100 extraction calls. Measure: average number of LLM calls per extraction, percentage that required at least one repair, percentage that failed completely (exhausted all repair attempts). Use `random.seed(42)` for reproducibility. Print a summary table.

---

## Self-check

- [ ] I can describe the architecture: which module handles what responsibility, and why.
- [ ] I can explain why `ExtractionService` depends on `LLMClient` (a protocol) rather than `AnthropicClient` directly.
- [ ] I can explain why XML delimiters (`<document>`, `<schema>`) reduce prompt injection risk.
- [ ] I can trace a repair loop call: what the repair prompt contains, why the error message is included.
- [ ] I can run `pytest tests/test_service.py` and explain what each test group is verifying.
- [ ] I can extend the service with a new extraction target without modifying existing code (open-closed principle).
- [ ] I can calculate how many tokens the test suite "costs" in a real run (hint: zero — all mocked).

---

## Supplementary reading

- [Pydantic v2 validators](https://docs.pydantic.dev/latest/concepts/validators/)
- [Pydantic v2 model_json_schema](https://docs.pydantic.dev/latest/concepts/json_schema/)
- [Anthropic structured output patterns](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [Python typing.TypeVar](https://docs.python.org/3/library/typing.html#typing.TypeVar)
- [argparse official docs](https://docs.python.org/3/library/argparse.html)
