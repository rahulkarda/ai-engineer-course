# Week 08 Assessment

## Conceptual (interview-style)

1. **Zero-shot vs few-shot prompting — when do you choose each?**
   Zero-shot: well-defined task, clear instruction suffices, or latency/cost matters and examples add little. Few-shot: when the model needs to learn output format, style, or a non-obvious mapping from the examples; particularly effective when zero-shot produces inconsistent structure or wrong tone. Rule of thumb: start zero-shot, add examples only when output quality is measurably insufficient.

2. **What is chain-of-thought and why does it improve reasoning tasks?**
   Chain-of-thought (CoT) prompts the model to emit intermediate reasoning steps before the final answer (either via an explicit "think step by step" instruction or few-shot examples that include scratchpad text). It improves performance on multi-step arithmetic, logic, and planning tasks because the model's own intermediate tokens act as working memory — each generated token conditions subsequent tokens, so exposing the reasoning path reduces the probability of the model jumping to an incorrect shortcut.

3. **Explain how Claude's tool_use content block works in a multi-turn conversation.**
   When Claude decides to call a tool it emits a `tool_use` content block (role: `assistant`) containing an `id`, `name`, and `input` dict. The caller executes the tool and returns the result as a `tool_result` content block (role: `user`) referencing the same `id`. Claude then continues generation with that result in context. A single assistant turn can contain multiple parallel `tool_use` blocks; the caller must return a `tool_result` for each before Claude proceeds. The conversation therefore alternates: `system → user → assistant(tool_use) → user(tool_result) → assistant(final)`.

4. **Why use a provider-agnostic LLM client interface in production systems?**
   Decouples business logic from a specific SDK, enabling: vendor switching without rewriting call sites, easy mocking in tests (inject a `MockLLMClient` that never touches the network), A/B testing across providers, and cost/latency routing at runtime. The interface also becomes the single place to enforce cross-cutting concerns — retry policy, token tracking, logging — rather than scattering them across every call site.

5. **What is a repair loop and when is it preferable to a simple retry?**
   A repair loop re-invokes the LLM with the original prompt plus the bad output and a targeted correction instruction (e.g. "Your previous response was not valid JSON. Fix it."). A simple retry just re-sends the original prompt. Prefer a repair loop when: (a) the model's output was structurally close but had a minor error a correction message can fix; (b) you want to avoid burning tokens regenerating content that was mostly correct; (c) the error is deterministic (e.g. schema mismatch) and the correction can be precisely specified. Simple retry is fine for transient network/rate-limit failures.

6. **How do you embed a Pydantic schema in an LLM prompt? What are the trade-offs of different approaches?**
   Three common approaches:
   - **JSON Schema dump** (`Model.model_json_schema()`): machine-precise, verbose, unfamiliar to models trained on prose.
   - **Field-annotated docstring / inline description**: human-readable, concise, less precise — model may infer structure incorrectly.
   - **Example instance** (`Model.model_validate({...}).model_dump_json(indent=2)`): shows shape and value types concretely; risk of the model copying example values literally.
   Best practice: combine JSON Schema with a one-shot example and ask the model to return only the JSON object. Wrap with `response_format` / tool schema when the provider supports it to get structured output guarantees at the API level.

7. **Explain token budget management: input_tokens vs output_tokens, and how prompt caching affects cost.**
   `input_tokens` = tokens in the request (system prompt + conversation history + tool definitions). `output_tokens` = tokens the model generates. Cost = `input_tokens * price_in + output_tokens * price_out`; output is typically 3-5x more expensive per token. Prompt caching (Anthropic's cache-control breakpoints) stores a prefix of the input on the server; subsequent requests that share that prefix are charged at a reduced cache-read rate instead of full input rate. Effective when: system prompt is long and reused across many requests (RAG instructions, tool schemas). Cache misses on first request are billed at the standard write rate.

8. **What makes a prompt evaluation framework useful? What metrics matter for structured extraction?**
   A prompt eval framework lets you systematically compare prompt variants against a labelled dataset without manual inspection. Key properties: reproducible runs (fixed seed/temperature), versioned prompts, automated scoring. Metrics for structured extraction: (a) **field-level accuracy** — per-field exact match or semantic similarity vs. ground truth; (b) **schema validity rate** — fraction of responses that parse without Pydantic error; (c) **repair rate** — fraction needing >0 retries; (d) **token cost per call** — tracks efficiency regressions; (e) **latency p50/p95**. Aggregate into a single score only after confirming individual metrics are healthy, to avoid masking regressions in one field with wins in another.

---

## Coding challenge

### Spec

Build a `ResumeParser` service in a single module (`resume_parser.py`).

**Data model** — define with Pydantic v2:

```python
class ParsedResume(BaseModel):
    name: str
    email: str
    skills: list[str]
    years_experience: int
    education: list[str]
```

**`LLMClient` interface** — define an abstract base (or `Protocol`) with at least:

```python
def complete(self, system: str, user: str, max_tokens: int) -> tuple[str, dict[str, int]]:
    ...
    # returns (response_text, {"input_tokens": int, "output_tokens": int})
```

**`ResumeParser` class**:

- Constructor accepts an `LLMClient` instance.
- `parse(raw_text: str) -> ParsedResume` method:
  - Sends a structured extraction prompt to the LLM.
  - Attempts `json.loads` + `ParsedResume.model_validate` on the response.
  - On failure, re-prompts with a repair message (up to 3 total attempts).
  - Accumulates `input_tokens` and `output_tokens` across all attempts.
  - Exposes cumulative usage via a `token_usage` property returning `dict[str, int]`.
  - Raises `ValueError` if all attempts fail.

**`AnthropicClient`** — concrete `LLMClient` wrapping the Anthropic SDK:

- Reads API key via `os.getenv("ANTHROPIC_API_KEY")`.
- Uses model `claude-opus-5-5` (or env-overridable via `os.getenv("LLM_MODEL", "claude-opus-5-5")`).
- Maps the `complete` return signature to `anthropic.Anthropic().messages.create(...)`.

### Acceptance criteria

- [ ] `ParsedResume` validates: missing required field raises `ValidationError`.
- [ ] `ResumeParser.parse` returns a `ParsedResume` on a well-formed LLM response.
- [ ] Repair loop: if attempts 1-2 return invalid JSON/schema, attempt 3 with a corrective prompt succeeds (or raises `ValueError` after 3 failures).
- [ ] `token_usage` accumulates tokens across all attempts in a session.
- [ ] Full `pytest` suite in `tests/test_resume_parser.py` using a `MockLLMClient` — no real API calls.
- [ ] `ruff check` passes with no errors.
- [ ] All type annotations present; no `Any` without explicit justification.

---

## Rubric (pass >= 75%)

| Dimension | Weight |
|---|---|
| Independent implementation | 40% |
| Conceptual understanding | 25% |
| Debugging & problem-solving | 20% |
| Code quality & testing | 15% |

Answer key: ../solutions/week-08/assessment.md
