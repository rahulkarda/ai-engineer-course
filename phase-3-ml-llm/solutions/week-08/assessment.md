# Week 8 Assessment — Complete Answer Key

## Section 1: Conceptual Questions (8 questions)

---

**Q1. What is the core problem that the repair loop pattern solves, and why is it
preferable to simply raising an exception on the first parse failure?**

LLMs are probabilistic: a model that produces valid JSON 95% of the time will fail
~5% of calls, and that failure is usually fixable with a second prompt that includes
the specific error. Raising immediately on the first failure propagates a transient
error to the caller; the repair loop handles it transparently. The key insight is that
the model already "knows" the correct output schema — it just made a formatting mistake
that it can self-correct given feedback. The cost is one extra LLM call (~2x tokens),
which is almost always cheaper than a human retry or a failed pipeline run.

---

**Q2. Why must `max_retries` be a finite constant, and what happens if you omit it?**

Without a bound, a model that consistently returns malformed output (wrong schema,
hallucinated fields, refusal) will loop forever, consuming tokens and incurring
unbounded cost. A finite `max_retries` (typically 3) converts an infinite loop into a
`RuntimeError` that the caller can catch and handle — log, alert, or queue for human
review. Three retries covers the vast majority of transient failures without excessive
cost: with a 10% per-call failure rate, three attempts succeed with probability
`1 - 0.1^3 = 99.9%`.

---

**Q3. Explain the Dependency Injection pattern as applied to `LLMClient`. What
testability benefit does it provide over instantiating the client inside the service?**

Dependency Injection means the `LLMClient` is passed into the service at construction
time rather than created internally. The service depends on the *interface* (Protocol),
not on a specific implementation. In tests, you pass a `MockLLMClient` that returns
hardcoded responses with no network calls. In production, you pass `AnthropicClient`.
If the client were instantiated inside the service, tests would require `mock.patch`
to intercept the constructor call — fragile, tightly coupled to implementation details.
DI makes the seam explicit and the test code identical in structure to production code.

---

**Q4. What is a Protocol in Python's type system, and how does it differ from an
abstract base class (ABC)?**

A `Protocol` defines a structural interface — any class that has the required methods
and signatures satisfies it, without inheriting from it (duck typing with static
checking). An ABC requires explicit inheritance (`class MyClass(MyABC)`). Protocols
enable retroactive compatibility: you can make an existing third-party class satisfy
your interface without modifying it. ABCs enforce inheritance at runtime via
`isinstance` checks (unless `@runtime_checkable` is added to the Protocol too).
For LLM clients from different SDKs, Protocol is the right choice: you cannot modify
`anthropic.Anthropic` to inherit from your ABC, but you can wrap it in a class that
satisfies your Protocol.

---

**Q5. Describe two ways to reduce the cost of running the same system prompt against
1000 documents. Quantify the savings for each.**

(1) **Prompt caching**: Mark the system prompt block with `cache_control: ephemeral`.
After the first call (which pays 125% of normal input cost to write the cache entry),
subsequent calls pay ~10% for the cached tokens. For a 500-token system prompt at
$15/1M input, uncached cost for 1000 calls = 1000 × 500 × $15/1M = $7.50. With
caching: $0.009375 (write) + 999 × 500 × $1.50/1M (read) ≈ $0.76. Saving: ~90%.

(2) **Batch API / bulk processing**: Anthropic's Batch API processes requests
asynchronously at 50% of the normal per-token price. For 1000 × 2000 input tokens at
$15/1M, the Batch API costs $15 instead of $30 — a flat 50% reduction. Suitable when
latency tolerance is hours rather than seconds.

---

**Q6. What is the difference between `temperature=0.0` and `temperature=1.0` in an
extraction pipeline, and which should you use?**

Temperature controls the randomness of token sampling. At `temperature=0.0` the model
always picks the highest-probability next token (greedy decoding), producing deterministic
output. At `temperature=1.0` sampling is proportional to the full probability distribution,
introducing creative variation. For structured extraction, use `temperature=0.0`:
you want the single most-likely JSON object, not creative alternatives. High temperature
increases the probability of JSON syntax errors and hallucinated field values. The only
case for temperature > 0 in extraction is generating diverse examples for a dataset.

---

**Q7. Explain the middleware (decorator) pattern. Why is stacking three middlewares
`LoggingMiddleware(CostTrackingMiddleware(RateLimitMiddleware(client)))` preferable to
putting all three concerns in a single class?**

The Decorator pattern wraps an object behind the same interface, adding behaviour
before/after delegation without inheritance. Stacking three single-responsibility
middlewares keeps each class small (~30 lines), independently testable, and reorderable.
A monolithic class combining logging, cost tracking, and rate limiting violates the
Single Responsibility Principle, is harder to unit test (you cannot test cost tracking
without triggering rate limiting), and makes reuse impossible (you cannot add just logging
to a different client). The stack is also reconfigurable at runtime: swap out
`RateLimitMiddleware` in tests without touching the other two.

---

**Q8. A colleague proposes storing the ANTHROPIC_API_KEY in a config file committed to
git so "the CI pipeline can find it". What is wrong with this proposal, and what is
the correct solution?**

Committing a secret to git means it exists permanently in the repository's history even
after deletion — anyone with repo access (current or future) can retrieve it via `git log`.
If the repo is public or later leaked, the key is compromised. The correct solution is:
(1) Store the key in the CI platform's secret store (GitHub Actions Secrets, GitLab CI
variables, etc.), which injects it as an environment variable at runtime. (2) In code,
read it exclusively via `os.getenv("ANTHROPIC_API_KEY")`. (3) Maintain a `.env.example`
with a placeholder value — never a `.env` file in git. (4) Add `.env` to `.gitignore`
and add a pre-commit hook that rejects files containing `sk-ant-` patterns.

---

## Section 2: Implementation — ResumeParser

### Models

```python
# models.py
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


class WorkExperience(BaseModel):
    company: str
    title: str
    start_date: str = Field(..., description="YYYY-MM or 'present'")
    end_date: str = Field(..., description="YYYY-MM or 'present'")
    description: str | None = None

    @field_validator("start_date", "end_date")
    @classmethod
    def date_format(cls, v: str) -> str:
        v = v.strip()
        if v.lower() == "present":
            return "present"
        parts = v.split("-")
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            raise ValueError(f"Date must be YYYY-MM or 'present', got {v!r}")
        return v


class Education(BaseModel):
    institution: str
    degree: str
    field_of_study: str | None = None
    graduation_year: int | None = Field(default=None, ge=1950, le=2100)


class ParsedResume(BaseModel):
    """Structured output of resume parsing."""

    full_name: str = Field(..., min_length=1)
    email: str | None = Field(default=None)
    phone: str | None = Field(default=None)
    location: str | None = Field(default=None)
    summary: str | None = Field(default=None)
    skills: list[str] = Field(default_factory=list)
    work_experience: list[WorkExperience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    total_years_experience: float | None = Field(
        default=None, ge=0, description="Approximate total years of experience"
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    extraction_notes: list[str] = Field(
        default_factory=list, description="Warnings about missing or ambiguous fields"
    )

    @field_validator("email")
    @classmethod
    def email_looks_valid(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError(f"Email does not look valid: {v!r}")
        return v.lower().strip()

    @model_validator(mode="after")
    def warn_if_empty(self) -> ParsedResume:
        if not self.work_experience and not self.education:
            if "No work experience or education found" not in self.extraction_notes:
                self.extraction_notes.append("No work experience or education found")
        return self
```

---

### MockLLMClient

```python
# mock_client.py
from __future__ import annotations

import json
import logging
from collections.abc import Iterator

from week08.protocol import LLMClient, LLMResponse

logger = logging.getLogger(__name__)


class MockLLMClient:
    """Injectable test double for LLMClient.

    Supports multiple response modes:
    - responses: list of strings returned in sequence (last is repeated)
    - raise_on_call: list of call indices (0-based) where an exception is raised
    - token_counts: (input, output) tuple applied to all responses
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        raise_on_call: list[int] | None = None,
        model: str = "claude-opus-5-5",
        input_tokens: int = 200,
        output_tokens: int = 100,
    ) -> None:
        self._responses: list[str] = list(responses or ['{"error": "no response configured"}'])
        self._raise_on_call: set[int] = set(raise_on_call or [])
        self._model = model
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._call_count: int = 0
        self.calls: list[dict[str, str]] = []

    @property
    def call_count(self) -> int:
        return self._call_count

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        call_idx = self._call_count
        self._call_count += 1
        self.calls.append({"system": system, "user": user})

        if call_idx in self._raise_on_call:
            raise RuntimeError(f"Simulated network error on call {call_idx}")

        idx = min(call_idx, len(self._responses) - 1)
        content = self._responses[idx]
        logger.debug("MockLLMClient call %d -> %d chars", call_idx, len(content))

        return LLMResponse(
            content=content,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            model=self._model,
            stop_reason="end_turn",
        )

    def stream_complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Iterator[str]:
        response = self.complete(system, user, max_tokens=max_tokens, temperature=temperature)
        for word in response.content.split():
            yield word + " "
```

---

### ResumeParser

```python
# resume_parser.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from pydantic import ValidationError

from week08.models import ParsedResume
from week08.protocol import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a resume parsing assistant. Extract structured information from the resume
text provided by the user.

Return ONLY valid JSON matching this exact schema (no markdown, no explanation):
{
  "full_name": "string",
  "email": "string or null",
  "phone": "string or null",
  "location": "string or null",
  "summary": "string or null",
  "skills": ["list", "of", "skill", "strings"],
  "work_experience": [
    {
      "company": "string",
      "title": "string",
      "start_date": "YYYY-MM",
      "end_date": "YYYY-MM or present",
      "description": "string or null"
    }
  ],
  "education": [
    {
      "institution": "string",
      "degree": "string",
      "field_of_study": "string or null",
      "graduation_year": integer or null
    }
  ],
  "total_years_experience": number or null,
  "confidence": 0.0-1.0,
  "extraction_notes": ["list of warnings"]
}

Rules:
- full_name must not be empty
- Dates must be YYYY-MM format or the string "present"
- Skills must be individual skill strings, not sentences
- If a field is absent: use null (never omit the key)
- confidence reflects clarity of the source document (0=illegible, 1=perfect)
"""


@dataclass
class TokenUsage:
    """Accumulated token usage across all LLM calls for one parse operation."""

    input_tokens: int = 0
    output_tokens: int = 0
    num_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ResumeParser:
    """Parses unstructured resume text into a ParsedResume using an LLM.

    Uses a repair loop (up to MAX_RETRIES attempts) so transient JSON errors
    are handled transparently. Token usage is tracked and returned.
    """

    MAX_RETRIES: int = 3

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def extract(self, resume_text: str) -> tuple[ParsedResume, TokenUsage]:
        """Extract structured resume data from raw text.

        Returns:
            (ParsedResume, TokenUsage) on success.

        Raises:
            RuntimeError: if all retries are exhausted.
        """
        usage = TokenUsage()
        original_user = f"Parse this resume:\n\n{resume_text}"
        last_error = ""

        for attempt in range(1, self.MAX_RETRIES + 1):
            if attempt == 1:
                user = original_user
            else:
                user = self._repair_prompt(original_user, last_error)

            logger.debug("ResumeParser attempt %d/%d", attempt, self.MAX_RETRIES)
            response = self._client.complete(
                system=_SYSTEM_PROMPT,
                user=user,
                max_tokens=2048,
                temperature=0.0,
            )

            usage.input_tokens += response.input_tokens
            usage.output_tokens += response.output_tokens
            usage.num_calls += 1

            content = response.content.strip()
            # Strip markdown fences if present
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(
                    line for line in lines if not line.startswith("```")
                ).strip()

            try:
                data = json.loads(content)
                resume = ParsedResume.model_validate(data)
                logger.info(
                    "ResumeParser succeeded on attempt %d (tokens: %d in + %d out)",
                    attempt,
                    usage.input_tokens,
                    usage.output_tokens,
                )
                return resume, usage
            except json.JSONDecodeError as exc:
                last_error = f"JSON parse error: {exc}"
                logger.warning("attempt %d: %s", attempt, last_error)
            except ValidationError as exc:
                last_error = f"Schema validation error: {exc}"
                logger.warning("attempt %d: %s", attempt, last_error)

        raise RuntimeError(
            f"ResumeParser failed after {self.MAX_RETRIES} attempts. "
            f"Last error: {last_error}"
        )

    @staticmethod
    def _repair_prompt(original_user: str, error: str) -> str:
        return (
            f"{original_user}\n\n"
            f"Your previous response had an error:\n{error}\n\n"
            "Return corrected JSON only. No markdown fences, no explanation."
        )
```

---

### Full Test Suite

```python
# tests/test_resume_parser.py
from __future__ import annotations

import json
import logging

import pytest
from pydantic import ValidationError

from week08.mock_client import MockLLMClient
from week08.models import ParsedResume, WorkExperience, Education
from week08.protocol import LLMResponse
from week08.resume_parser import ResumeParser, TokenUsage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GOOD_RESUME_JSON = json.dumps(
    {
        "full_name": "Jane Doe",
        "email": "jane.doe@example.com",
        "phone": "+1-555-0100",
        "location": "San Francisco, CA",
        "summary": "Senior software engineer with 8 years in distributed systems.",
        "skills": ["Python", "Kubernetes", "PostgreSQL", "Go"],
        "work_experience": [
            {
                "company": "Acme Corp",
                "title": "Senior Engineer",
                "start_date": "2020-03",
                "end_date": "present",
                "description": "Led platform migration to Kubernetes.",
            },
            {
                "company": "Beta Inc",
                "title": "Software Engineer",
                "start_date": "2017-06",
                "end_date": "2020-02",
                "description": None,
            },
        ],
        "education": [
            {
                "institution": "MIT",
                "degree": "B.Sc.",
                "field_of_study": "Computer Science",
                "graduation_year": 2017,
            }
        ],
        "total_years_experience": 8.0,
        "confidence": 0.95,
        "extraction_notes": [],
    }
)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_returns_parsed_resume(self) -> None:
        mock = MockLLMClient(responses=[GOOD_RESUME_JSON])
        parser = ResumeParser(mock)
        resume, usage = parser.extract("Jane Doe resume text...")
        assert isinstance(resume, ParsedResume)
        assert resume.full_name == "Jane Doe"

    def test_fields_extracted_correctly(self) -> None:
        mock = MockLLMClient(responses=[GOOD_RESUME_JSON])
        parser = ResumeParser(mock)
        resume, _ = parser.extract("any text")
        assert resume.email == "jane.doe@example.com"
        assert "Python" in resume.skills
        assert len(resume.work_experience) == 2
        assert resume.work_experience[0].company == "Acme Corp"
        assert resume.education[0].institution == "MIT"

    def test_confidence_field(self) -> None:
        mock = MockLLMClient(responses=[GOOD_RESUME_JSON])
        parser = ResumeParser(mock)
        resume, _ = parser.extract("any text")
        assert resume.confidence == 0.95

    def test_markdown_fences_stripped(self) -> None:
        """Parser should handle ```json ... ``` wrapping."""
        fenced = f"```json\n{GOOD_RESUME_JSON}\n```"
        mock = MockLLMClient(responses=[fenced])
        parser = ResumeParser(mock)
        resume, _ = parser.extract("any text")
        assert resume.full_name == "Jane Doe"

    def test_only_one_llm_call_on_success(self) -> None:
        mock = MockLLMClient(responses=[GOOD_RESUME_JSON])
        parser = ResumeParser(mock)
        parser.extract("any text")
        assert mock.call_count == 1


# ---------------------------------------------------------------------------
# Repair loop
# ---------------------------------------------------------------------------

class TestRepairLoop:
    def test_retries_on_bad_json(self) -> None:
        """First response is broken JSON; second is valid."""
        bad = "Here is the JSON: {oops not valid}"
        mock = MockLLMClient(responses=[bad, GOOD_RESUME_JSON])
        parser = ResumeParser(mock)
        resume, usage = parser.extract("any text")
        assert resume.full_name == "Jane Doe"
        assert mock.call_count == 2

    def test_repair_prompt_includes_error(self) -> None:
        """The second call's user prompt must include the parse error."""
        bad = "not json at all"
        mock = MockLLMClient(responses=[bad, GOOD_RESUME_JSON])
        parser = ResumeParser(mock)
        parser.extract("original resume text")
        # Second call (index 1) user message should reference the error
        second_user = mock.calls[1]["user"]
        assert "JSON parse error" in second_user or "error" in second_user.lower()

    def test_repair_prompt_includes_original_text(self) -> None:
        """Original resume text must survive into the repair prompt."""
        bad = "not json"
        mock = MockLLMClient(responses=[bad, GOOD_RESUME_JSON])
        parser = ResumeParser(mock)
        original = "Jane Doe senior engineer resume"
        parser.extract(original)
        second_user = mock.calls[1]["user"]
        assert "Jane Doe" in second_user

    def test_retries_on_schema_validation_failure(self) -> None:
        """Valid JSON but wrong schema triggers repair."""
        wrong_schema = json.dumps({"name": "Jane", "job": "Engineer"})  # missing full_name
        mock = MockLLMClient(responses=[wrong_schema, GOOD_RESUME_JSON])
        parser = ResumeParser(mock)
        resume, _ = parser.extract("any text")
        assert resume.full_name == "Jane Doe"
        assert mock.call_count == 2

    def test_succeeds_on_third_attempt(self) -> None:
        bad = "not json"
        also_bad = "{missing: quotes}"
        mock = MockLLMClient(responses=[bad, also_bad, GOOD_RESUME_JSON])
        parser = ResumeParser(mock)
        resume, _ = parser.extract("any text")
        assert resume.full_name == "Jane Doe"
        assert mock.call_count == 3


# ---------------------------------------------------------------------------
# Max retries exceeded
# ---------------------------------------------------------------------------

class TestMaxRetriesExceeded:
    def test_raises_runtime_error_after_max_retries(self) -> None:
        always_bad = MockLLMClient(responses=["not json ever"])
        parser = ResumeParser(always_bad)
        with pytest.raises(RuntimeError, match="failed after"):
            parser.extract("any text")

    def test_exhausts_exactly_max_retries_calls(self) -> None:
        mock = MockLLMClient(responses=["bad json"])
        parser = ResumeParser(mock)
        with pytest.raises(RuntimeError):
            parser.extract("any text")
        assert mock.call_count == ResumeParser.MAX_RETRIES

    def test_error_message_includes_last_error(self) -> None:
        mock = MockLLMClient(responses=["definitely not json !!!"])
        parser = ResumeParser(mock)
        with pytest.raises(RuntimeError) as exc_info:
            parser.extract("any text")
        assert "JSON parse error" in str(exc_info.value) or "error" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Token tracking
# ---------------------------------------------------------------------------

class TestTokenTracking:
    def test_tokens_tracked_for_single_call(self) -> None:
        mock = MockLLMClient(
            responses=[GOOD_RESUME_JSON],
            input_tokens=500,
            output_tokens=200,
        )
        parser = ResumeParser(mock)
        _, usage = parser.extract("any text")
        assert usage.input_tokens == 500
        assert usage.output_tokens == 200
        assert usage.total_tokens == 700
        assert usage.num_calls == 1

    def test_tokens_accumulate_across_retries(self) -> None:
        """Two calls (one bad, one good) should sum their token counts."""
        mock = MockLLMClient(
            responses=["bad json", GOOD_RESUME_JSON],
            input_tokens=300,
            output_tokens=100,
        )
        parser = ResumeParser(mock)
        _, usage = parser.extract("any text")
        assert usage.input_tokens == 600   # 300 * 2
        assert usage.output_tokens == 200  # 100 * 2
        assert usage.total_tokens == 800
        assert usage.num_calls == 2

    def test_token_usage_type(self) -> None:
        mock = MockLLMClient(responses=[GOOD_RESUME_JSON])
        parser = ResumeParser(mock)
        _, usage = parser.extract("any text")
        assert isinstance(usage, TokenUsage)
        assert isinstance(usage.input_tokens, int)
        assert isinstance(usage.output_tokens, int)


# ---------------------------------------------------------------------------
# ParsedResume model validation (unit tests, no LLM needed)
# ---------------------------------------------------------------------------

class TestParsedResumeModel:
    def test_full_name_required(self) -> None:
        with pytest.raises(ValidationError):
            ParsedResume(full_name="")

    def test_email_validated(self) -> None:
        with pytest.raises(ValidationError):
            ParsedResume(full_name="Alice", email="not-an-email")

    def test_email_lowercased(self) -> None:
        r = ParsedResume(full_name="Alice", email="Alice@Example.COM")
        assert r.email == "alice@example.com"

    def test_defaults_are_empty_lists(self) -> None:
        r = ParsedResume(full_name="Bob")
        assert r.skills == []
        assert r.work_experience == []
        assert r.education == []

    def test_warn_if_no_experience_or_education(self) -> None:
        r = ParsedResume(full_name="Charlie")
        assert any("No work experience" in note for note in r.extraction_notes)

    def test_work_experience_date_format(self) -> None:
        with pytest.raises(ValidationError):
            WorkExperience(
                company="X",
                title="Y",
                start_date="January 2020",  # wrong format
                end_date="present",
            )

    def test_work_experience_present_accepted(self) -> None:
        we = WorkExperience(
            company="X", title="Y", start_date="2020-01", end_date="present"
        )
        assert we.end_date == "present"
```

---

## Section 3: Grading Notes

### Conceptual Questions (40 points, 5 pts each)

| Q | What to look for |
|---|-----------------|
| Q1 | Mentions probabilistic nature of LLMs; repair vs raise tradeoff; cost justification |
| Q2 | Infinite loop / unbounded cost risk; specific probability calculation or reasoning |
| Q3 | Interface vs implementation; DI = test without mock.patch; structural explanation |
| Q4 | Structural vs nominal typing; retroactive compatibility; runtime_checkable mention |
| Q5 | Two distinct techniques; quantitative reasoning (even if approximate numbers differ) |
| Q6 | Determinism at 0.0; specific risk for extraction (JSON errors, hallucination); correct recommendation |
| Q7 | Single Responsibility; independent testability; reorderability/composability |
| Q8 | Git history persistence; correct solution (CI secret store + env var); .gitignore; pre-commit hook |

**Deduct 2 pts** for vague answers with no technical specifics.
**Deduct 1 pt** for correct concept with no justification.

---

### Implementation (60 points)

| Dimension | Points | What to look for |
|-----------|--------|-----------------|
| `ParsedResume` model | 12 | Pydantic v2 with `BaseModel`; field validators; correct field types; `model_validator`; no untyped dicts |
| `MockLLMClient` | 8 | Satisfies `LLMClient` Protocol; configurable responses; call introspection (`calls` list); no network |
| `ResumeParser.extract()` | 15 | Repair loop with correct retry count; error message passed to repair prompt; Pydantic validation catch; `RuntimeError` on exhaustion |
| Token tracking | 5 | `TokenUsage` dataclass; accumulates across retries; returned from `extract()` |
| Test: `test_happy_path` | 5 | No mock.patch; DI pattern; asserts on field values AND call count |
| Test: `test_repair_loop` | 7 | Asserts repair prompt contains original text AND error; verifies call count = 2 |
| Test: `test_max_retries_exceeded` | 5 | Asserts `RuntimeError`; verifies exactly `MAX_RETRIES` calls made |
| Test: `test_token_tracking` | 3 | Asserts accumulation across retries; checks `total_tokens` property |

**Common deductions:**
- `-5` Using `mock.patch` instead of DI (tests no longer verify the real contract)
- `-5` Repair prompt does not include `last_error` (the core bug from Day 7 Q2)
- `-3` `MAX_RETRIES` hardcoded as a literal `3` inside the loop instead of a class constant
- `-3` Missing type hints on any method signature
- `-2` `print()` instead of `logging`
- `-2` API key as a literal string or `"your-api-key-here"` placeholder
- `-1` per missing test case from the required four
- `-1` Tests pass but contain no assertions (only check that no exception is raised)

**Full credit requires all four test classes present.** Partial implementations accepted
with proportional deductions. Award half credit for tests that demonstrate the correct
concept but have a minor syntax error (e.g. wrong pytest fixture name).
