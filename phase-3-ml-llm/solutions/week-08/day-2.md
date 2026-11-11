# Day 2 Solutions — Structured Outputs

## Exercise 1: Pydantic v2 Model for Job Posting Extraction

```python
# models.py
"""Pydantic v2 models for structured job posting extraction."""

from __future__ import annotations

import logging
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class ExperienceLevel(str, Enum):
    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    PRINCIPAL = "principal"


class SalaryRange(BaseModel):
    """Salary information, all amounts in USD per year."""

    min_usd: int | None = Field(None, ge=0, description="Minimum annual salary in USD")
    max_usd: int | None = Field(None, ge=0, description="Maximum annual salary in USD")
    currency: str = Field(default="USD", max_length=3)

    @model_validator(mode="after")
    def max_gte_min(self) -> "SalaryRange":
        if self.min_usd is not None and self.max_usd is not None:
            if self.max_usd < self.min_usd:
                raise ValueError(
                    f"max_usd ({self.max_usd}) must be >= min_usd ({self.min_usd})"
                )
        return self


class JobPosting(BaseModel):
    """Structured representation of a job posting extracted from raw text."""

    title: str = Field(..., min_length=1, max_length=200, description="Job title")
    company: str = Field(..., min_length=1, max_length=200, description="Hiring company name")
    location: str | None = Field(None, description="City, State or 'Remote'")
    remote_ok: bool = Field(False, description="True if remote work is explicitly offered")
    experience_level: ExperienceLevel | None = Field(
        None, description="Seniority level inferred from posting"
    )
    years_experience_required: int | None = Field(
        None, ge=0, le=40, description="Minimum years of experience stated"
    )
    required_skills: list[str] = Field(
        default_factory=list,
        description="Hard skills explicitly listed as required",
    )
    preferred_skills: list[str] = Field(
        default_factory=list,
        description="Skills listed as nice-to-have or preferred",
    )
    salary: SalaryRange | None = Field(None, description="Compensation range if stated")
    employment_type: str | None = Field(
        None, description="e.g. Full-time, Part-time, Contract"
    )
    description_summary: str = Field(
        ..., max_length=500, description="One-paragraph summary of role and responsibilities"
    )

    @field_validator("required_skills", "preferred_skills", mode="before")
    @classmethod
    def deduplicate_skills(cls, v: list[str]) -> list[str]:
        """Remove duplicates while preserving order."""
        seen: set[str] = set()
        result: list[str] = []
        for skill in v:
            normalized = skill.strip()
            if normalized and normalized.lower() not in seen:
                seen.add(normalized.lower())
                result.append(normalized)
        return result

    @field_validator("location", mode="before")
    @classmethod
    def normalize_location(cls, v: str | None) -> str | None:
        if v is None:
            return None
        stripped = v.strip()
        return stripped if stripped else None
```

---

## Exercise 2: Prompt Instructing LLM to Return JSON Matching Schema

```python
# prompts.py
"""Prompts for structured extraction tasks."""

from __future__ import annotations

import json


# Embed the schema inline so the LLM can see exactly what to produce.
JOB_EXTRACTION_SCHEMA = {
    "title": "string (required) — job title",
    "company": "string (required) — company name",
    "location": "string or null — city/state or 'Remote'",
    "remote_ok": "boolean — true if remote is offered",
    "experience_level": "one of: 'entry', 'mid', 'senior', 'lead', 'principal', or null",
    "years_experience_required": "integer or null — minimum years stated",
    "required_skills": "array of strings — hard required skills",
    "preferred_skills": "array of strings — nice-to-have skills",
    "salary": {
        "min_usd": "integer or null",
        "max_usd": "integer or null",
        "currency": "string, default 'USD'",
    },
    "employment_type": "string or null — e.g. 'Full-time', 'Contract'",
    "description_summary": "string (required) — 1-paragraph summary, max 500 chars",
}


def build_job_extraction_prompt(raw_text: str) -> str:
    """Build the user-turn prompt for job posting extraction.

    Args:
        raw_text: The raw job posting text to extract information from.

    Returns:
        A prompt string instructing the LLM to output valid JSON.
    """
    schema_str = json.dumps(JOB_EXTRACTION_SCHEMA, indent=2)

    return f"""Extract structured information from the job posting below.

<schema>
{schema_str}
</schema>

Rules:
1. Return ONLY a valid JSON object matching the schema above — no markdown fences, no extra text.
2. Use null for fields not mentioned in the posting.
3. Normalize salary figures to annual USD; if only hourly is given, multiply by 2080.
4. Infer experience_level from title and years stated (entry=0-2y, mid=3-5y, senior=5-8y, lead/principal=8y+).
5. Keep description_summary under 500 characters.

<job_posting>
{raw_text}
</job_posting>

JSON output:"""


SYSTEM_PROMPT_EXTRACTOR = (
    "You are a data extraction assistant. "
    "You read job postings and return structured JSON. "
    "You never include markdown code fences, preamble, or explanation — only the JSON object."
)
```

---

## Exercise 3: `parse_llm_json` with Error Handling

```python
# parser.py
"""Parse and validate JSON responses from LLM into Pydantic models."""

from __future__ import annotations

import json
import logging
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMParseError(Exception):
    """Raised when an LLM response cannot be parsed into the target model."""

    def __init__(self, message: str, raw_response: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.cause = cause


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers if present."""
    pattern = r"^```(?:json)?\s*\n?(.*?)\n?```\s*$"
    match = re.match(pattern, text.strip(), re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _extract_json_object(text: str) -> str:
    """Extract the first complete JSON object from text that may have surrounding prose."""
    # Find first '{' and match to its closing '}'
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response")
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError("Unterminated JSON object in response")


def parse_llm_json(response: str, model: type[T]) -> T:
    """Parse a raw LLM string response into a validated Pydantic model instance.

    Handles common LLM output quirks:
    - Markdown code fences (```json ... ```)
    - Leading/trailing prose before/after the JSON object
    - JSON decode errors (raises LLMParseError with the raw text)
    - Pydantic validation errors (raises LLMParseError with field details)

    Args:
        response: Raw string returned by the LLM.
        model: The Pydantic v2 BaseModel subclass to validate against.

    Returns:
        A validated instance of ``model``.

    Raises:
        LLMParseError: If parsing or validation fails for any reason.
    """
    logger.debug("Parsing LLM response into %s (length=%d)", model.__name__, len(response))

    # Step 1: strip markdown fences
    cleaned = _strip_markdown_fences(response)

    # Step 2: extract the JSON object even if prose surrounds it
    try:
        json_str = _extract_json_object(cleaned)
    except ValueError as exc:
        logger.warning("Could not locate JSON object in response: %s", exc)
        raise LLMParseError(
            f"No JSON object found in LLM response: {exc}",
            raw_response=response,
            cause=exc,
        ) from exc

    # Step 3: parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.warning("JSON decode error at position %d: %s", exc.pos, exc.msg)
        raise LLMParseError(
            f"JSON decode error: {exc.msg} (line {exc.lineno}, col {exc.colno})",
            raw_response=response,
            cause=exc,
        ) from exc

    # Step 4: validate with Pydantic
    try:
        instance = model.model_validate(data)
    except ValidationError as exc:
        logger.warning("Pydantic validation failed: %s", exc)
        raise LLMParseError(
            f"Validation failed for {model.__name__}: {exc.error_count()} error(s)",
            raw_response=response,
            cause=exc,
        ) from exc

    logger.debug("Successfully parsed %s", model.__name__)
    return instance
```

---

## Exercise 4: Unit Test with Mocked Anthropic Client

```python
# tests/test_extraction.py
"""Unit tests for job posting extraction using mocked Anthropic client."""

from __future__ import annotations

import json
from unittest import mock

import pytest

from models import JobPosting, SalaryRange
from parser import LLMParseError, parse_llm_json
from prompts import SYSTEM_PROMPT_EXTRACTOR, build_job_extraction_prompt


# ---------------------------------------------------------------------------
# Sample fixture data
# ---------------------------------------------------------------------------

SAMPLE_JOB_TEXT = """
Senior Python Engineer — Acme Corp (San Francisco, CA)

We are hiring a Senior Python Engineer with 5+ years of experience.
Required: Python, FastAPI, PostgreSQL.
Preferred: Kubernetes, Terraform.
Salary: $140,000 — $170,000/year. Full-time, hybrid remote OK.
"""

VALID_JOB_JSON = {
    "title": "Senior Python Engineer",
    "company": "Acme Corp",
    "location": "San Francisco, CA",
    "remote_ok": True,
    "experience_level": "senior",
    "years_experience_required": 5,
    "required_skills": ["Python", "FastAPI", "PostgreSQL"],
    "preferred_skills": ["Kubernetes", "Terraform"],
    "salary": {"min_usd": 140000, "max_usd": 170000, "currency": "USD"},
    "employment_type": "Full-time",
    "description_summary": "Senior Python Engineer role at Acme Corp in San Francisco.",
}


def _make_mock_message(content: str) -> mock.MagicMock:
    """Build a mock object that mimics anthropic.types.Message structure."""
    mock_message = mock.MagicMock()
    mock_message.content = [mock.MagicMock(text=content)]
    mock_message.stop_reason = "end_turn"
    return mock_message


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExtractJobPosting:
    """Tests for a hypothetical extract_job_posting() function that calls the API."""

    def _run_extraction(self, raw_text: str, api_response: str) -> JobPosting:
        """Helper: run extraction with a mocked Anthropic client."""
        # This is the exact mocking pattern to use:
        with mock.patch("anthropic.Anthropic") as mock_client_class:
            mock_client = mock_client_class.return_value
            mock_client.messages.create.return_value = _make_mock_message(api_response)

            # Import here to allow the patch to intercept the constructor
            import anthropic

            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-opus-5-5",
                max_tokens=1024,
                system=SYSTEM_PROMPT_EXTRACTOR,
                messages=[
                    {"role": "user", "content": build_job_extraction_prompt(raw_text)}
                ],
            )
            raw_llm_output = response.content[0].text
            return parse_llm_json(raw_llm_output, JobPosting)

    def test_successful_extraction(self) -> None:
        """Happy path: valid JSON returned by LLM parses into JobPosting."""
        api_response = json.dumps(VALID_JOB_JSON)
        posting = self._run_extraction(SAMPLE_JOB_TEXT, api_response)

        assert posting.title == "Senior Python Engineer"
        assert posting.company == "Acme Corp"
        assert posting.remote_ok is True
        assert "Python" in posting.required_skills
        assert posting.salary is not None
        assert posting.salary.min_usd == 140000

    def test_client_called_with_correct_model(self) -> None:
        """Verify the API call uses the expected model id."""
        with mock.patch("anthropic.Anthropic") as mock_client_class:
            mock_client = mock_client_class.return_value
            mock_client.messages.create.return_value = _make_mock_message(
                json.dumps(VALID_JOB_JSON)
            )

            import anthropic

            client = anthropic.Anthropic()
            client.messages.create(
                model="claude-opus-5-5",
                max_tokens=1024,
                system=SYSTEM_PROMPT_EXTRACTOR,
                messages=[{"role": "user", "content": "test"}],
            )

            call_kwargs = mock_client.messages.create.call_args
            assert call_kwargs.kwargs["model"] == "claude-opus-5-5"

    def test_markdown_fenced_json_is_handled(self) -> None:
        """LLM wrapping output in ```json fences should still parse."""
        fenced = "```json\n" + json.dumps(VALID_JOB_JSON) + "\n```"
        posting = self._run_extraction(SAMPLE_JOB_TEXT, fenced)
        assert posting.title == "Senior Python Engineer"

    def test_invalid_json_raises_parse_error(self) -> None:
        """Malformed JSON from LLM raises LLMParseError."""
        with pytest.raises(LLMParseError):
            self._run_extraction(SAMPLE_JOB_TEXT, "This is not JSON at all.")

    def test_validation_failure_raises_parse_error(self) -> None:
        """JSON that fails Pydantic validation raises LLMParseError."""
        bad_data = {**VALID_JOB_JSON, "title": ""}  # min_length=1 violated
        with pytest.raises(LLMParseError, match="Validation failed"):
            self._run_extraction(SAMPLE_JOB_TEXT, json.dumps(bad_data))

    def test_no_real_network_calls(self) -> None:
        """Ensure the mock intercepts; no real HTTP is made."""
        with mock.patch("anthropic.Anthropic") as mock_client_class:
            mock_client = mock_client_class.return_value
            mock_client.messages.create.return_value = _make_mock_message(
                json.dumps(VALID_JOB_JSON)
            )

            import anthropic

            client = anthropic.Anthropic()
            client.messages.create(
                model="claude-opus-5-5",
                max_tokens=1024,
                system=SYSTEM_PROMPT_EXTRACTOR,
                messages=[{"role": "user", "content": "test"}],
            )

            # Verify mock was used (real calls would raise without API key)
            mock_client.messages.create.assert_called_once()


class TestParseLLMJson:
    """Unit tests for parse_llm_json directly, no API involved."""

    def test_valid_json_parses(self) -> None:
        result = parse_llm_json(json.dumps(VALID_JOB_JSON), JobPosting)
        assert isinstance(result, JobPosting)

    def test_salary_range_validation(self) -> None:
        bad = {**VALID_JOB_JSON, "salary": {"min_usd": 200000, "max_usd": 100000}}
        with pytest.raises(LLMParseError):
            parse_llm_json(json.dumps(bad), JobPosting)

    def test_strips_fences_before_parse(self) -> None:
        fenced = f"```json\n{json.dumps(VALID_JOB_JSON)}\n```"
        result = parse_llm_json(fenced, JobPosting)
        assert result.company == "Acme Corp"

    def test_extracts_json_from_prose(self) -> None:
        prose = "Sure! Here is the JSON:\n" + json.dumps(VALID_JOB_JSON) + "\nHope that helps."
        result = parse_llm_json(prose, JobPosting)
        assert result.title == "Senior Python Engineer"
```

### Key Mocking Pattern (Reference)

```python
# The exact pattern for mocking the Anthropic client:

with mock.patch("anthropic.Anthropic") as mock_client_class:
    # mock_client_class is the patched constructor
    mock_client = mock_client_class.return_value  # the instance returned by Anthropic()

    # Set up the return value for messages.create
    mock_response = mock.MagicMock()
    mock_response.content = [mock.MagicMock(text='{"key": "value"}')]
    mock_response.stop_reason = "end_turn"
    mock_client.messages.create.return_value = mock_response

    # Now any code that does `anthropic.Anthropic().messages.create(...)` uses the mock
    import anthropic
    client = anthropic.Anthropic()           # returns mock_client
    resp = client.messages.create(...)       # returns mock_response
    text = resp.content[0].text             # returns '{"key": "value"}'
```
