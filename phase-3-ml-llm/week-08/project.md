# Week 08 Project: Structured Extraction Service

## Overview

Build a production-ready **Structured Extraction Service** that accepts arbitrary unstructured text, calls an LLM to extract typed data, validates the result with Pydantic v2, repairs failures automatically, and tracks token cost — all behind an injectable client interface that never requires a real API key in tests.

This project synthesises every technique from Week 8: prompt engineering, structured output, repair loops, provider-agnostic clients, token tracking, and prompt evaluation.

---

## Learning goals

- Design a clean provider-agnostic LLM client abstraction and understand why it matters for testability and vendor portability.
- Write prompts that reliably produce JSON-parseable structured output from Claude.
- Implement a repair loop that feeds the model's own bad output back as a correction signal.
- Track and report token usage accurately across multi-attempt requests.
- Achieve 100% test coverage of extraction logic with no real API calls.
- (Stretch) Evaluate prompt variants against a labelled dataset using a lightweight eval harness.

---

## Full specification

### System boundaries

```
raw text (str)
    |
    v
[ExtractionService]
    |-- LLMClient (injectable)
    |-- ExtractionPromptBuilder
    |-- RepairLoop (max 3 attempts)
    |-- Pydantic model validation
    |
    v
Extracted[T] (typed result + token usage)
```

### Data models

Define all models in `src/extraction_service/models.py`:

```python
from pydantic import BaseModel, Field
from typing import TypeVar, Generic

T = TypeVar("T", bound=BaseModel)

class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

class ExtractionResult(BaseModel, Generic[T]):
    data: T
    usage: TokenUsage
    attempts: int = Field(ge=1, le=3)
```

Define at least two domain schemas to prove the service is schema-agnostic:

```python
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
    due_date: str   # ISO 8601
```

### LLM client interface

Define in `src/extraction_service/llm_client.py`:

```python
from abc import ABC, abstractmethod
from extraction_service.models import TokenUsage

class LLMClient(ABC):
    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
    ) -> tuple[str, TokenUsage]:
        """Return (response_text, token_usage)."""

class AnthropicClient(LLMClient):
    """Concrete implementation wrapping the Anthropic SDK."""
    # API key: os.getenv("ANTHROPIC_API_KEY")
    # Model:   os.getenv("LLM_MODEL", "claude-opus-5-5")
```

### Extraction service

Define in `src/extraction_service/service.py`:

```python
class ExtractionService:
    def __init__(self, client: LLMClient, max_retries: int = 3) -> None: ...

    def extract(
        self,
        raw_text: str,
        schema: type[T],
    ) -> ExtractionResult[T]:
        """
        1. Build a system prompt that embeds schema.model_json_schema().
        2. Call client.complete with the raw_text as user message.
        3. Attempt json.loads + schema.model_validate on the response.
        4. On failure, call client.complete again with a repair prompt
           that includes the previous bad output and the Pydantic error.
        5. Repeat up to max_retries total attempts.
        6. Accumulate TokenUsage across all attempts.
        7. Raise ExtractionError (custom exception) if all attempts fail.
        """
```

### Prompt builder

Define in `src/extraction_service/prompts.py`:

- `build_extraction_prompt(schema: type[BaseModel]) -> str` — returns the system prompt, embedding the JSON schema and a one-shot example.
- `build_repair_prompt(previous_output: str, error: str) -> str` — returns a correction user message.
- Keep all prompt strings in this module; no f-strings spread across `service.py`.

### Configuration

Define in `src/extraction_service/config.py` using `pydantic-settings`:

```python
class Settings(BaseSettings):
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    llm_model: str = Field(default="claude-opus-5-5", alias="LLM_MODEL")
    max_retries: int = Field(default=3, alias="MAX_RETRIES")
    max_output_tokens: int = Field(default=1024, alias="MAX_OUTPUT_TOKENS")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
```

---

## Detailed requirements

### Must-have (required for passing grade)

| # | Requirement | Acceptance criterion |
|---|---|---|
| R1 | Accept arbitrary text | `extract(raw_text, schema)` works for any Pydantic model |
| R2 | Return validated Pydantic instance | `result.data` is an instance of the passed `schema` type |
| R3 | Repair loop | Up to 3 attempts; each repair prompt includes the previous bad output and the error |
| R4 | Token tracking | `result.usage` sums `input_tokens` + `output_tokens` across all attempts |
| R5 | Injectable client | Constructor accepts any `LLMClient` implementation |
| R6 | No real API calls in tests | All tests pass with `MockLLMClient`; `ANTHROPIC_API_KEY` never read during `pytest` |
| R7 | Custom exception | `ExtractionError` raised (not generic `ValueError`) after all retries exhausted |
| R8 | Logging | Every attempt logged at DEBUG; extraction result at INFO; errors at ERROR |
| R9 | Prompt isolation | All prompt strings live in `prompts.py` only |
| R10 | Type annotations | Every function/method fully annotated; no bare `Any` |

### Must-not

- No `print()` anywhere — use `logging`.
- No hardcoded API key or model name outside `config.py` / `os.getenv`.
- No real HTTP calls in the test suite — any test that touches the network fails CI.
- No god files — if a module exceeds 300 lines, split it.

---

## Suggested architecture

```
src/
  extraction_service/
    __init__.py
    config.py          # pydantic-settings Settings
    models.py          # TokenUsage, ExtractionResult, domain schemas
    llm_client.py      # LLMClient ABC, AnthropicClient, MockLLMClient
    prompts.py         # build_extraction_prompt, build_repair_prompt
    service.py         # ExtractionService, ExtractionError
    main.py            # CLI entrypoint (stretch goal)

tests/
  unit/
    test_models.py
    test_prompts.py
    test_service.py        # core logic, MockLLMClient, repair loop
  integration/
    test_anthropic_client.py   # skipped unless ANTHROPIC_API_KEY set
```

`pyproject.toml` dependencies (minimum):

```toml
[project]
requires-python = ">=3.12"

[project.dependencies]
anthropic = ">=0.40"
pydantic = ">=2.0"
pydantic-settings = ">=2.0"

[project.optional-dependencies]
dev = ["pytest", "pytest-cov", "ruff"]
```

---

## Testing requirements

### Unit tests (`tests/unit/test_service.py`)

All tests use `MockLLMClient`. Never instantiate `AnthropicClient` in a unit test.

Required test cases:

| Test | What it verifies |
|---|---|
| `test_extract_success_first_attempt` | Valid JSON on attempt 1 → `result.attempts == 1`, correct data |
| `test_extract_success_after_repair` | Invalid JSON on attempt 1, valid on attempt 2 → `result.attempts == 2` |
| `test_extract_exhausts_retries` | All 3 attempts return invalid JSON → raises `ExtractionError` |
| `test_token_usage_accumulates` | Mock returns known usage per call; assert cumulative total |
| `test_repair_prompt_includes_previous_output` | Capture calls to `MockLLMClient`; assert repair user message contains the bad output |
| `test_schema_agnostic_resume` | Extract `ParsedResume` |
| `test_schema_agnostic_invoice` | Extract `ParsedInvoice` |
| `test_extraction_result_model` | `ExtractionResult` rejects `attempts=0` and `attempts=4` |

**`MockLLMClient` pattern** (define in `llm_client.py` or a test fixture):

```python
class MockLLMClient(LLMClient):
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.calls: list[tuple[str, str]] = []   # (system, user)

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> tuple[str, TokenUsage]:
        self.calls.append((system, user))
        return next(self._responses), TokenUsage(input_tokens=100, output_tokens=50)
```

### Integration tests (`tests/integration/test_anthropic_client.py`)

- Decorated with `@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="no key")`.
- One smoke test: send a trivial prompt, assert response is non-empty string.
- Not run in CI unless the key is explicitly injected.

### Coverage target

`pytest --cov=extraction_service --cov-report=term-missing` must report >= 90% for `service.py` and `prompts.py`.

---

## Stretch goals

### S1 — Async support

Add `AsyncLLMClient(ABC)` with `async def complete(...)`. Implement `AsyncAnthropicClient` using `anthropic.AsyncAnthropic`. Add `AsyncExtractionService` with `async def extract(...)`. Test with `pytest-asyncio` and an `AsyncMockLLMClient`.

### S2 — Multiple schemas via registry

```python
SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {
    "resume": ParsedResume,
    "invoice": ParsedInvoice,
}

service.extract_by_name(raw_text, schema_name="resume")
```

Raise `SchemaNotFoundError` for unknown names.

### S3 — CLI interface

```bash
python -m extraction_service extract --schema resume --file resume.txt
python -m extraction_service extract --schema invoice --stdin
```

Use `argparse` or `click`. Output JSON to stdout. Respect `LOG_LEVEL` env var.

### S4 — Prompt evaluation harness

```python
class EvalCase(BaseModel):
    raw_text: str
    expected: dict[str, Any]
    schema_name: str

class EvalResult(BaseModel):
    case_id: str
    schema_valid: bool
    field_accuracy: float   # fraction of fields matching expected
    attempts: int
    usage: TokenUsage
```

Write `scripts/run_eval.py` that loads a JSONL dataset of `EvalCase` records, runs `ExtractionService` against each, and prints a summary table (schema_valid rate, mean field_accuracy, mean attempts, total tokens).

### S5 — Prompt caching

Add `cache_control` breakpoints to the system prompt in `AnthropicClient.complete` to enable Anthropic prompt caching. Add a `cache_read_tokens` field to `TokenUsage` and report it separately.

---

## Submission checklist

- [ ] `src/extraction_service/` package present with all required modules
- [ ] `tests/unit/test_service.py` covers all required test cases
- [ ] `pytest` exits 0 with no real API calls (run `unset ANTHROPIC_API_KEY && pytest`)
- [ ] `pytest --cov=extraction_service` reports >= 90% for `service.py` and `prompts.py`
- [ ] `ruff check src/ tests/` exits 0
- [ ] `ExtractionError` raised on exhausted retries (not `ValueError` or generic exception)
- [ ] `TokenUsage` sums correctly across repair attempts
- [ ] All prompt strings isolated to `prompts.py`
- [ ] No `print()` in `src/`
- [ ] `.env.example` present with `ANTHROPIC_API_KEY=`, `LLM_MODEL=`, `MAX_RETRIES=`
- [ ] `.env` in `.gitignore`
- [ ] (Stretch) Any stretch goals attempted are documented in `README.md`

---

## Grading guide

| Area | Points | Notes |
|---|---|---|
| `ExtractionService` core logic | 30 | Correct repair loop, token accumulation, exception type |
| Pydantic models | 15 | `TokenUsage.add`, `ExtractionResult` generic, domain schemas |
| `LLMClient` abstraction | 15 | ABC with correct signature, `AnthropicClient` uses env vars only |
| Prompt quality | 10 | System prompt embeds JSON schema, repair prompt includes bad output + error |
| Unit test coverage | 20 | All 8 required cases present, no real API calls |
| Code quality | 10 | Type annotations, logging, no god files, ruff clean |
| **Stretch goals** | +10 each | Async, registry, CLI, eval harness, caching — up to +20 total |

**Pass threshold: 75 / 100 base points.**

Stretch goal points are additive; the maximum grade is 120.
