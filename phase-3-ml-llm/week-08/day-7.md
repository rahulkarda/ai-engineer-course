# Phase 3 · Week 08 · Day 7 — Week Review: Prompt Engineering & Structured Outputs

> Objective: Consolidate the week's concepts, identify gaps, and prepare for assessment with evaluation frameworks and cheatsheets.  ·  Time: ~3 hr  ·  Prereqs: [[day-6]]

---

## Concept

Week 8 built a complete structured extraction pipeline. Before moving forward, consolidate the concepts into reusable mental models and cheatsheets you can reach for during any LLM engineering task.

This day covers:

1. **What we built and why** — the big picture across all five days
2. **Prompt engineering patterns cheatsheet** — zero-shot, few-shot, CoT, system prompt, delimiters
3. **Structured output patterns cheatsheet** — JSON schema, Pydantic validation, repair loop
4. **Common failure modes and fixes** — a diagnostic reference
5. **Prompt evaluation** — how to measure whether a prompt is good

---

## How it works internally

### The week in one diagram

```
Day 1: Prompt engineering fundamentals
  |  Zero-shot / few-shot / CoT / system prompts
  v
Day 2: Structured outputs — JSON schema in prompts
  |  model_json_schema(), response format, delimiters
  v
Day 3: Tool use and function calling
  |  tools= parameter, tool_use blocks, tool_result
  v
Day 4: Reliability
  |  Retries (tenacity), repair loops, TokenBudget, BudgetExceededError
  v
Day 5: Provider abstraction
  |  LLMClient Protocol, AnthropicClient, MockLLMClient, DI
  v
Day 6: Full build — ExtractionService
       models.py / prompts.py / client.py / service.py / tests
```

Each day added one vertical slice: day-4 (reliability) wraps day-5 (abstraction), which the service (day-6) consumes. The test suite (day-6) validates the whole stack without touching the API.

### Why each piece matters

**Pydantic v2 at every boundary**: catches errors where they originate (parsing), not downstream (weird AttributeError ten function calls later). Validators enforce business rules at parse time, including after repair.

**Protocol/DI for the client**: the service has no import of `anthropic` — only the protocol. Changing providers requires changing one file (`client.py`), zero changes to service logic. Test isolation is guaranteed: `MockLLMClient` is not a patch, it is a first-class implementation.

**Separate prompts module**: prompts are code. They change frequently, they need tests, they should not be scattered across functions. Centralizing them in `prompts.py` makes prompt iteration fast and auditable.

**Repair loop**: structured output extraction fails silently without it. The model produces valid-looking text that fails JSON parse or Pydantic validation roughly 5-15% of the time in practice (higher for complex schemas). A repair loop converts most of these failures to successes at modest extra cost.

**Token budget**: cost is observable only if you instrument it. A `TokenBudget` shared across a pipeline gives you real-time cost feedback. The `BudgetExceededError` guard ensures no runaway batch job.

---

## Worked example

### Prompt evaluation framework

When you change a prompt, you need a way to measure whether it got better or worse. The framework below runs N test cases and computes precision and recall for structured extraction.

```python
"""
week-08/day-7: Prompt evaluation framework.

Tests whether a prompt+model combination correctly extracts structured fields.
All tests use MockLLMClient — evaluation logic is model-agnostic.

Run: pytest tests/test_day7_evaluation.py -v
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from src.extraction.client import LLMClient, LLMMessage, LLMRequest, LLMResponse, MockLLMClient
from src.extraction.models import EntityList, ExtractedEntity
from src.extraction.service import ExtractionError, ExtractionService, TokenBudget

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")


# ---------------------------------------------------------------------------
# Evaluation data types
# ---------------------------------------------------------------------------


@dataclass
class TestCase:
    """A single evaluation example with input text and expected extraction."""

    input_text: str
    expected: dict[str, Any]
    description: str = ""


@dataclass
class EvalResult:
    """Result for a single test case."""

    test_case: TestCase
    actual: dict[str, Any] | None
    passed: bool
    fields_correct: int
    fields_total: int
    error: str | None = None


@dataclass
class EvalReport:
    """Aggregated evaluation metrics across all test cases."""

    results: list[EvalResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    @property
    def field_precision(self) -> float:
        """
        Of all fields we extracted, what fraction matched the expected value?
        Measures false positives — did we hallucinate values?
        """
        total_extracted = sum(r.fields_correct + (r.fields_total - r.fields_correct) for r in self.results if r.actual is not None)
        total_correct = sum(r.fields_correct for r in self.results)
        return total_correct / total_extracted if total_extracted > 0 else 0.0

    @property
    def field_recall(self) -> float:
        """
        Of all expected fields, what fraction did we correctly extract?
        Measures false negatives — did we miss values?
        """
        total_expected = sum(r.fields_total for r in self.results)
        total_correct = sum(r.fields_correct for r in self.results)
        return total_correct / total_expected if total_expected > 0 else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "total_cases": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": round(self.pass_rate, 3),
            "field_precision": round(self.field_precision, 3),
            "field_recall": round(self.field_recall, 3),
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class PromptEvaluator:
    """
    Evaluates an extraction prompt against a set of labeled test cases.

    Usage:
        evaluator = PromptEvaluator(service=service, target=EntityList)
        report = evaluator.run(test_cases)
        print(report.summary())

    The evaluator compares actual extraction output against expected dicts
    at the field level. Fields that are lists (like `entities`) are compared
    by length and by spot-checking the first element's key fields.

    This is a lightweight framework — for production, use a proper eval
    harness (LangSmith, RAGAS, etc.) with richer comparison strategies.
    """

    def __init__(
        self,
        service: ExtractionService,
        target: type[BaseModel],
    ) -> None:
        self._service = service
        self._target = target

    def _compare_fields(
        self,
        actual: dict[str, Any],
        expected: dict[str, Any],
    ) -> tuple[int, int]:
        """
        Compare actual vs expected dicts at top level.
        Returns (correct_fields, total_expected_fields).

        List fields: correct if length matches and first element matches.
        Scalar fields: correct if values match (case-insensitive strings).
        """
        correct = 0
        total = len(expected)

        for key, exp_val in expected.items():
            act_val = actual.get(key)
            if act_val is None:
                continue

            if isinstance(exp_val, list):
                if isinstance(act_val, list) and len(act_val) == len(exp_val):
                    correct += 1
            elif isinstance(exp_val, str):
                if isinstance(act_val, str) and exp_val.lower() == act_val.lower():
                    correct += 1
            elif exp_val == act_val:
                correct += 1

        return correct, total

    def run(self, test_cases: list[TestCase]) -> EvalReport:
        """Run all test cases and return an aggregated report."""
        report = EvalReport()

        for i, tc in enumerate(test_cases):
            logger.info("Evaluating case %d/%d: %s", i + 1, len(test_cases), tc.description)

            try:
                result = self._service.extract(tc.input_text, self._target)
                actual = result.model_dump()
                correct, total = self._compare_fields(actual, tc.expected)
                passed = correct == total

                report.results.append(EvalResult(
                    test_case=tc,
                    actual=actual,
                    passed=passed,
                    fields_correct=correct,
                    fields_total=total,
                ))
            except ExtractionError as exc:
                report.results.append(EvalResult(
                    test_case=tc,
                    actual=None,
                    passed=False,
                    fields_correct=0,
                    fields_total=len(tc.expected),
                    error=str(exc),
                ))

        return report


# ---------------------------------------------------------------------------
# Deterministic tests using MockLLMClient
# ---------------------------------------------------------------------------
# The key insight: MockLLMClient makes LLM behavior fully deterministic.
# We can write unit tests for evaluation logic itself.
# ---------------------------------------------------------------------------


def make_entity_response(entities_json: list[dict[str, Any]]) -> LLMResponse:
    payload = {
        "entities": entities_json,
        "total_count": len(entities_json),
    }
    return LLMResponse(
        content=json.dumps(payload),
        input_tokens=80,
        output_tokens=40,
        model="mock",
    )


def test_evaluator_pass_rate_all_correct() -> None:
    """All 3 test cases produce correct output — pass_rate should be 1.0."""
    entity_json = [{"entity_type": "PERSON", "value": "Alice", "normalized": None,
                    "confidence": 1.0, "extraction_notes": None}]

    mock = MockLLMClient(responses=[
        make_entity_response(entity_json),
        make_entity_response(entity_json),
        make_entity_response(entity_json),
    ])
    service = ExtractionService(client=mock)
    evaluator = PromptEvaluator(service=service, target=EntityList)

    test_cases = [
        TestCase(
            input_text="Alice is here.",
            expected={"entities": entity_json, "total_count": 1},
            description="basic person extraction",
        ),
        TestCase(
            input_text="Alice arrived.",
            expected={"entities": entity_json, "total_count": 1},
            description="same entity, different context",
        ),
        TestCase(
            input_text="Mention of Alice.",
            expected={"entities": entity_json, "total_count": 1},
            description="third variation",
        ),
    ]

    report = evaluator.run(test_cases)

    assert report.total == 3
    assert report.passed == 3
    assert report.pass_rate == 1.0
    assert mock.call_count == 3


def test_evaluator_pass_rate_partial_failure() -> None:
    """One test case uses MockLLMClient error injection to simulate ExtractionError."""
    entity_json = [{"entity_type": "PERSON", "value": "Bob", "normalized": None,
                    "confidence": 1.0, "extraction_notes": None}]

    # Call 0 returns valid JSON, call 1 returns broken JSON (error after max repairs)
    mock = MockLLMClient(
        responses=[
            make_entity_response(entity_json),  # case 0: success
            LLMResponse(content="BROKEN", input_tokens=10, output_tokens=5, model="mock"),  # case 1: initial
            LLMResponse(content="STILL BROKEN", input_tokens=10, output_tokens=5, model="mock"),  # case 1: repair 1
            LLMResponse(content="STILL BROKEN", input_tokens=10, output_tokens=5, model="mock"),  # case 1: repair 2
        ]
    )
    service = ExtractionService(client=mock, max_repair_attempts=2)
    evaluator = PromptEvaluator(service=service, target=EntityList)

    test_cases = [
        TestCase(input_text="Bob here.", expected={"total_count": 1}, description="success case"),
        TestCase(input_text="Test.", expected={"total_count": 1}, description="failure case"),
    ]

    report = evaluator.run(test_cases)

    assert report.total == 2
    assert report.passed == 1
    assert report.failed == 1
    assert report.pass_rate == 0.5
    assert report.results[1].error is not None


def test_evaluator_report_summary_shape() -> None:
    """EvalReport.summary() returns the expected keys."""
    entity_json = [{"entity_type": "ORGANIZATION", "value": "Acme", "normalized": None,
                    "confidence": 0.9, "extraction_notes": None}]

    mock = MockLLMClient(responses=[make_entity_response(entity_json)])
    service = ExtractionService(client=mock)
    evaluator = PromptEvaluator(service=service, target=EntityList)

    report = evaluator.run([
        TestCase(input_text="Acme Corp.", expected={"total_count": 1}, description="org test")
    ])

    summary = report.summary()
    assert "total_cases" in summary
    assert "pass_rate" in summary
    assert "field_precision" in summary
    assert "field_recall" in summary


def test_mock_llm_client_enables_deterministic_eval() -> None:
    """
    Demonstrates the core pattern: MockLLMClient makes LLM calls deterministic.
    The same test will always produce the same result — no flakiness, no cost.
    """
    expected_content = json.dumps({
        "entities": [
            {"entity_type": "PERSON", "value": "Carol", "normalized": None,
             "confidence": 1.0, "extraction_notes": None}
        ],
        "total_count": 1,
    })

    mock = MockLLMClient(responses=[
        LLMResponse(content=expected_content, input_tokens=50, output_tokens=30, model="mock")
    ])
    service = ExtractionService(client=mock)

    result = service.extract("Carol is here.", EntityList)

    assert result.entities[0].value == "Carol"
    assert mock.call_count == 1
    assert mock.recorded_requests[0].system is not None
```

---

## Prompt Engineering Cheatsheet

### Pattern reference

| Pattern | When to use | Key structure |
|---|---|---|
| **Zero-shot** | Task is clear, model knows it | `Task: {task}\nInput: {input}\nOutput:` |
| **Few-shot** | Format is ambiguous or domain-specific | 3-5 `Input: ... Output: ...` examples before the real one |
| **Chain-of-Thought** | Multi-step reasoning, math, logic | Add `Think step by step before answering.` or show reasoning in examples |
| **System prompt** | Persistent role, rules, format constraints | Keep it stable (cache it); separate from user turn |
| **XML delimiters** | Untrusted input, complex multi-section prompts | `<document>...</document>`, `<schema>...</schema>` |
| **Negative constraints** | Stop hallucination of absent fields | `If a field is not present, use null. Do not infer.` |
| **Self-repair** | Fix model's own output | Include broken output + error + schema in a second call |

### Zero-shot vs few-shot: the decision rule

Use zero-shot first. Switch to few-shot when:
- The output format is non-standard (unusual JSON structure, specific field names)
- The task requires domain knowledge the model may not have (medical codes, internal taxonomies)
- Zero-shot produces correct answers with wrong formatting consistently

Few-shot examples must match your real distribution. Bad examples are worse than none — the model learns the wrong pattern.

### What makes a good system prompt

1. **Role**: who the model is (`You are a data extraction assistant.`)
2. **Task scope**: what it should and should not do (`Extract only explicitly stated information. Do not infer.`)
3. **Output format**: what format is expected (`Return only valid JSON. No explanation.`)
4. **Edge case handling**: what to do when data is absent (`Use null for missing fields.`)
5. **Safety constraints** (if applicable): what to refuse (`Do not extract or repeat PII unless in the schema.`)

Keep system prompts short and scannable. Every word costs tokens on every call.

---

## Structured Output Patterns Cheatsheet

### Three approaches in order of reliability

| Approach | Mechanism | Pros | Cons |
|---|---|---|---|
| **JSON schema in prompt** | Paste `model.model_json_schema()` into prompt | Works with any model | Model may ignore it |
| **Repair loop** | On parse/validation failure, ask model to fix output | Recovers from 5-15% failure rate | Extra tokens, latency |
| **Tool use (function calling)** | Use `tools=` parameter to force structured output | Model cannot deviate from schema | Only works with tool-use-capable models |

For production extraction at scale: use tool use as the primary mechanism (day-3), with a repair loop as fallback.

### Common failure modes

| Failure | Symptom | Fix |
|---|---|---|
| Model wraps JSON in markdown fences | `json.JSONDecodeError` on ` ``` ` | `_strip_fences()` before parsing |
| Model truncates JSON mid-object | `json.JSONDecodeError: Expecting value` | Increase `max_tokens`; add `stop_sequences=["```"]` |
| Model adds explanatory prose before JSON | `json.JSONDecodeError` on first line | Use stricter system prompt; add `stop_sequences=["\n\n"]` |
| Model uses string for int field | `ValidationError: int parsing` | Add type coercion in system prompt; use repair loop |
| Model hallucinates absent fields | Extraction contains data not in source | Add `If a field is not present, use null. Do not infer.` to system prompt |
| Model uses wrong enum value | `ValidationError: enum` | Include enum values explicitly in schema description |
| Repair loop runs forever | Infinite cost | Cap `max_repair_attempts`; always check budget before repair call |

---

## What makes a good prompt: evaluation criteria

A prompt is not good because it "works on examples." It is good when:

1. **Coverage**: it produces correct output across diverse inputs (different document types, edge cases, multilingual input)
2. **Stability**: small changes in input don't produce large changes in output format
3. **Parsability**: output parses successfully on first attempt (high rate, measured, not assumed)
4. **Faithfulness**: extracted values appear verbatim in the source (no hallucination)
5. **Refusal rate**: for out-of-scope inputs, the model says so cleanly rather than making up data

**How to measure coverage and parsability:**

```
parse_success_rate = successful_first_parse / total_calls
repair_rate = calls_requiring_repair / total_calls
failure_rate = ExtractionErrors / total_calls
```

Track these metrics in production. A rising repair rate signals prompt drift (the document distribution changed) or model version change.

---

## Exercises

(Solutions: ../solutions/week-08/day-7.md)

1. **Explain the distinction**: In your own words (no code), explain the difference between zero-shot and few-shot prompting. Under what specific conditions would you switch from zero-shot to few-shot? Give a concrete example from the extraction service context.

2. **Debug the broken repair loop**: The following code has a bug that causes the repair loop to run forever even when repair succeeds. Identify the bug and explain why it causes the infinite loop.

```python
def buggy_repair_loop(
    raw: str,
    schema: type[BaseModel],
    client: LLMClient,
    max_repair_attempts: int = 3,
) -> BaseModel:
    candidate = raw
    for attempt in range(max_repair_attempts):
        try:
            return schema.model_validate(json.loads(candidate))
        except (json.JSONDecodeError, ValidationError) as exc:
            repair_prompt = build_repair_prompt(candidate, str(exc), schema)
            response = client.complete(
                LLMRequest(messages=[LLMMessage(role="user", content=repair_prompt)])
            )
            candidate = raw  # BUG HERE
    raise ExtractionError("All repair attempts failed")
```

3. **System prompt design**: Design a system prompt for a medical document extractor that must:
   - Extract patient name, diagnosis codes (ICD-10 format), medication names, and dosages
   - Never extract or repeat information not explicitly present in the source document
   - Handle documents that contain no diagnoses (return empty list, not null)
   - Refuse to process documents that appear to be non-medical
   Write the system prompt as a Python string constant. No code needed beyond the string — focus on completeness, conciseness, and safety constraints.

4. **Cost calculation**: A batch job processes 1000 documents. Each document averages 2000 input tokens for the extraction prompt and 500 output tokens for the JSON response. 12% of documents require one repair call; each repair averages 500 input tokens and 300 output tokens.

   Using `claude-opus-5-5` pricing (input: $15.00/1M tokens, output: $75.00/1M tokens):

   a. Calculate the total cost for the 1000 successful first-attempt extractions.
   b. Calculate the additional cost for the 120 repair calls.
   c. Calculate the total cost.
   d. If you switched to `claude-sonnet-5-5` for repair calls only (research current pricing), what is the new total cost?
   e. What `max_budget_usd` would you set for a `TokenBudget` covering this entire batch?

---

## Self-check

All topics from the full week:

**Prompt engineering (days 1-3):**
- [ ] I can describe zero-shot, few-shot, and chain-of-thought prompting and give a use case for each.
- [ ] I can explain why XML delimiters reduce prompt injection risk.
- [ ] I can write a system prompt with role, scope, format, edge cases, and safety constraints.
- [ ] I can explain how tool use forces structured output more reliably than JSON schema in prompt.

**Reliability (day 4):**
- [ ] I can implement exponential backoff with tenacity on API calls.
- [ ] I can write a self-repair prompt that includes the broken output, error message, and schema.
- [ ] I can calculate the USD cost of a call from `input_tokens`, `output_tokens`, and per-million pricing.
- [ ] I can implement `BudgetExceededError` and a pre-call budget check.

**Client abstraction (day 5):**
- [ ] I can define a `Protocol` interface for an LLM client and explain structural vs nominal subtyping.
- [ ] I can implement `MockLLMClient` with response sequencing and request recording.
- [ ] I can implement a middleware client (logging, cost tracking) that composes with any `LLMClient`.
- [ ] I can explain why DI + MockLLMClient is preferable to `unittest.mock.patch` for testing LLM calls.

**Full build (day 6):**
- [ ] I can describe the ExtractionService architecture: which module handles what.
- [ ] I can trace a complete extraction call from `service.extract()` through prompts, client, parse, repair.
- [ ] I can run the test suite and explain what each test group verifies.
- [ ] I can extend the service with a new extraction target without modifying existing code.

**Evaluation (day 7):**
- [ ] I can implement a `PromptEvaluator` that runs test cases and reports precision/recall.
- [ ] I can explain parse success rate, repair rate, and failure rate as production metrics.
- [ ] I can list 5 common structured output failure modes and their fixes.

---

## Supplementary reading

- [Anthropic prompt engineering guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)
- [Pydantic v2 validators reference](https://docs.pydantic.dev/latest/concepts/validators/)
- [RAGAS: evaluation framework for LLM pipelines](https://docs.ragas.io/)
- [LangSmith evaluation docs](https://docs.smith.langchain.com/evaluation)
- [Prompt injection: Simon Willison's overview](https://simonwillison.net/2022/Sep/12/prompt-injection/)
