# Phase 3 · Week 08 · Day 4 — Reliability: Retries, Validation, Repair Loops, Cost Management

> Objective: Build production-grade LLM call infrastructure with retries, self-repair, and token budget tracking.  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

---

## Concept

LLM API calls fail. Not rarely — at scale, every failure mode occurs:

- **Transient API errors**: rate limits (429), server overload (529), timeouts
- **Malformed output**: the model returns text instead of JSON, truncates mid-object, adds markdown fences around JSON
- **Validation failures**: JSON parses but fails Pydantic validation — wrong types, missing fields, enum violations
- **Budget overruns**: cost accumulates silently until a billing surprise

Reliability engineering for LLM systems requires three layers:

1. **Retry layer** — handles transient API errors with exponential backoff
2. **Repair layer** — handles semantic failures by asking the model to fix its own output
3. **Budget layer** — tracks cumulative token usage and enforces cost ceilings

These are separable concerns. Build them as composable components, not a monolithic function.

---

## How it works internally

### Why exponential backoff matters

Rate limit errors are not random. They cluster: if you hit a 429, retrying immediately guarantees another 429. Exponential backoff adds `2^attempt * base_delay` wait between retries, spreading requests over time. Adding jitter (`+ random(0, base_delay)`) prevents thundering herd when multiple clients retry simultaneously.

`tenacity` implements this declaratively. Key parameters:

```
@retry(
    wait=wait_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(APIStatusError),
)
```

`retry_if_exception_type` means only retryable errors trigger retries. A `ValidationError` should not be retried by the backoff layer — it goes to the repair layer.

### The self-repair loop

When a model returns invalid JSON, the standard approach is to ask it again — with the invalid output included:

```
System: You are a JSON repair assistant.
User: This JSON is invalid: {broken output}
Schema: {schema}
Error: {validation error message}
Return only valid JSON matching the schema.
```

This works because the model can see exactly what went wrong. The repair prompt is a targeted second call, not a full re-run of the original prompt. Limit repair attempts (typically 2-3) to bound cost.

### Token counting and cost estimation

The Anthropic API returns `usage` in every response:

```python
response.usage.input_tokens   # tokens in the prompt
response.usage.output_tokens  # tokens in the completion
```

Cost calculation:
```
cost = (input_tokens / 1_000_000) * input_price_per_million
     + (output_tokens / 1_000_000) * output_price_per_million
```

Track cumulative totals across calls. Before each call, estimate the likely cost from the prompt length (use `anthropic.count_tokens` or estimate 1 token ≈ 4 chars) and reject if it would exceed budget.

### Prompt caching (conceptual)

Anthropic supports prompt caching via the `cache_control` parameter on message blocks. When you mark a block as `{"type": "ephemeral"}`, the API caches the KV computation for that block for up to 5 minutes. Subsequent calls that share the same cached prefix pay only the output token cost for the cached portion (roughly 10% of normal input cost).

Use cases:
- Long system prompts reused across many calls
- Few-shot examples that don't change between requests
- Large document context shared across multiple extraction calls

Cache hits show as `cache_read_input_tokens` in `usage`. Cache creation shows as `cache_creation_input_tokens`.

### Budget tracking

Three enforcement points:

1. **Before call**: check `projected_cost <= remaining_budget`; raise if not
2. **After call**: record actual usage; update cumulative totals
3. **Globally**: expose a budget object that can be shared across a pipeline

`max_tokens` controls output length. Set it explicitly — never let the model run unbounded. Use `stop_sequences` to terminate at logical endpoints (`["```", "</output>"]`) rather than burning tokens on the end-of-context marker.

---

## Worked example

```python
"""
week-08/day-4: Reliability — retries, validation, repair loops, token budgets.

Requirements:
    pip install anthropic tenacity pydantic
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import anthropic
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
    before_sleep_log,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------


class RetryConfig(BaseModel):
    """Tenacity retry configuration for LLM API calls."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: float = 0.5


# ---------------------------------------------------------------------------
# 2. Token budget
# ---------------------------------------------------------------------------


class BudgetExceededError(Exception):
    """Raised when a call would exceed the configured USD budget."""

    def __init__(self, current_usd: float, limit_usd: float, projected_usd: float) -> None:
        self.current_usd = current_usd
        self.limit_usd = limit_usd
        self.projected_usd = projected_usd
        super().__init__(
            f"Budget exceeded: spent=${current_usd:.4f}, limit=${limit_usd:.4f}, "
            f"projected call cost=${projected_usd:.4f}"
        )


class TokenBudget:
    """
    Tracks cumulative input/output tokens and estimated cost across multiple calls.

    Pricing defaults are for claude-opus-5-5 (November 2025):
        input:  $15.00 per 1M tokens
        output: $75.00 per 1M tokens
    """

    INPUT_PRICE_PER_M = 15.00
    OUTPUT_PRICE_PER_M = 75.00

    def __init__(
        self,
        max_budget_usd: float | None = None,
        input_price_per_m: float = INPUT_PRICE_PER_M,
        output_price_per_m: float = OUTPUT_PRICE_PER_M,
    ) -> None:
        self.max_budget_usd = max_budget_usd
        self.input_price_per_m = input_price_per_m
        self.output_price_per_m = output_price_per_m

        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._call_count: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_input_tokens(self) -> int:
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._total_output_tokens

    @property
    def total_tokens(self) -> int:
        return self._total_input_tokens + self._total_output_tokens

    @property
    def estimated_cost_usd(self) -> float:
        return self._compute_cost(self._total_input_tokens, self._total_output_tokens)

    @property
    def call_count(self) -> int:
        return self._call_count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens / 1_000_000) * self.input_price_per_m + (
            output_tokens / 1_000_000
        ) * self.output_price_per_m

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_budget(self, projected_input_tokens: int, projected_output_tokens: int) -> None:
        """Raise BudgetExceededError if this call would exceed max_budget_usd."""
        if self.max_budget_usd is None:
            return
        projected_call_cost = self._compute_cost(projected_input_tokens, projected_output_tokens)
        if self.estimated_cost_usd + projected_call_cost > self.max_budget_usd:
            raise BudgetExceededError(
                current_usd=self.estimated_cost_usd,
                limit_usd=self.max_budget_usd,
                projected_usd=projected_call_cost,
            )

    def record_usage(self, usage: anthropic.types.Usage) -> None:
        """Record actual usage from an API response."""
        self._total_input_tokens += usage.input_tokens
        self._total_output_tokens += usage.output_tokens
        self._call_count += 1
        logger.debug(
            "Usage recorded: input=%d output=%d | cumulative cost=$%.4f",
            usage.input_tokens,
            usage.output_tokens,
            self.estimated_cost_usd,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "calls": self._call_count,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
        }


# ---------------------------------------------------------------------------
# 3. Retry-wrapped call
# ---------------------------------------------------------------------------


def _build_retry_decorator(config: RetryConfig):
    """Build a tenacity retry decorator from a RetryConfig."""
    return retry(
        retry=retry_if_exception_type(
            (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APITimeoutError)
        ),
        wait=wait_exponential(
            multiplier=config.base_delay,
            min=config.base_delay,
            max=config.max_delay,
        )
        + wait_random(0, config.jitter),
        stop=stop_after_attempt(config.max_retries),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


def call_with_retry(
    client: anthropic.Anthropic,
    messages: list[dict[str, str]],
    retry_config: RetryConfig | None = None,
    budget: TokenBudget | None = None,
    **kwargs: Any,
) -> anthropic.types.Message:
    """
    Call client.messages.create with exponential backoff on transient errors.

    Args:
        client: Anthropic client instance.
        messages: Message list in Anthropic format.
        retry_config: Retry parameters. Uses defaults if not provided.
        budget: Optional TokenBudget for cost tracking.
        **kwargs: Forwarded to messages.create (model, max_tokens, system, etc.).

    Returns:
        The raw Anthropic Message response.
    """
    config = retry_config or RetryConfig()
    decorator = _build_retry_decorator(config)

    @decorator
    def _call() -> anthropic.types.Message:
        response = client.messages.create(messages=messages, **kwargs)
        if budget is not None:
            budget.record_usage(response.usage)
        return response

    return _call()


# ---------------------------------------------------------------------------
# 4. Repair loop
# ---------------------------------------------------------------------------


def _extract_json_from_text(text: str) -> str:
    """
    Strip markdown fences and leading/trailing whitespace from model output.

    The model sometimes wraps JSON in ```json ... ``` blocks even when
    instructed not to. This handles the common cases.
    """
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        # Remove closing fence
        if text.endswith("```"):
            text = text[: text.rfind("```")]
    return text.strip()


def repair_json_output(
    raw: str,
    schema: type[BaseModel],
    client: anthropic.Anthropic,
    model: str = "claude-opus-5-5",
    max_repair_attempts: int = 2,
    budget: TokenBudget | None = None,
) -> BaseModel:
    """
    Attempt to parse raw as schema. If it fails, ask Claude to repair the JSON.

    Strategy:
      1. Try direct parse (strip fences first).
      2. On failure, send a repair prompt with the broken output + error message.
      3. Repeat up to max_repair_attempts times.
      4. Raise ValueError if all attempts fail.

    Args:
        raw: The raw text output from a previous LLM call.
        schema: Pydantic model class to validate against.
        client: Anthropic client for repair calls.
        model: Model to use for repair calls.
        max_repair_attempts: Max number of repair calls before raising.
        budget: Optional TokenBudget for cost tracking.

    Returns:
        Validated Pydantic model instance.

    Raises:
        ValueError: If parsing fails after all repair attempts.
    """
    schema_json = json.dumps(schema.model_json_schema(), indent=2)

    # Attempt 0: direct parse of original output
    candidate = _extract_json_from_text(raw)
    for attempt in range(max_repair_attempts + 1):
        try:
            parsed = json.loads(candidate)
            return schema.model_validate(parsed)
        except (json.JSONDecodeError, ValidationError) as exc:
            if attempt == max_repair_attempts:
                raise ValueError(
                    f"JSON repair failed after {max_repair_attempts} attempts. "
                    f"Last error: {exc}"
                ) from exc

            error_msg = str(exc)
            logger.warning(
                "Repair attempt %d/%d — parsing error: %s",
                attempt + 1,
                max_repair_attempts,
                error_msg[:200],
            )

            repair_prompt = (
                "The following JSON is invalid or does not match the required schema.\n\n"
                "<invalid_json>\n"
                f"{candidate}\n"
                "</invalid_json>\n\n"
                "<error>\n"
                f"{error_msg}\n"
                "</error>\n\n"
                "<schema>\n"
                f"{schema_json}\n"
                "</schema>\n\n"
                "Return ONLY valid JSON that matches the schema exactly. "
                "No explanation, no markdown fences, no extra text."
            )

            repair_kwargs: dict[str, Any] = {
                "model": model,
                "max_tokens": 1024,
                "system": "You are a JSON repair assistant. Output only valid JSON.",
            }

            response = call_with_retry(
                client=client,
                messages=[{"role": "user", "content": repair_prompt}],
                budget=budget,
                **repair_kwargs,
            )
            candidate = _extract_json_from_text(response.content[0].text)

    # Unreachable, but makes type checkers happy
    raise ValueError("Repair loop exhausted unexpectedly")


# ---------------------------------------------------------------------------
# 5. Usage demo (not run in tests)
# ---------------------------------------------------------------------------


def demo() -> None:
    """
    Minimal end-to-end demo. Requires ANTHROPIC_API_KEY in the environment.
    Run: python day-4.py
    """

    class PersonInfo(BaseModel):
        name: str
        age: int
        role: str

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")

    client = anthropic.Anthropic(api_key=api_key)
    budget = TokenBudget(max_budget_usd=0.10)

    prompt = (
        "Extract person info from this text as JSON:\n"
        "Alice is a 34-year-old software engineer.\n\n"
        "Return JSON with keys: name, age, role."
    )

    response = call_with_retry(
        client=client,
        messages=[{"role": "user", "content": prompt}],
        model="claude-opus-5-5",
        max_tokens=256,
        budget=budget,
    )
    raw = response.content[0].text

    person = repair_json_output(raw=raw, schema=PersonInfo, client=client, budget=budget)
    logger.info("Extracted: %s", person)
    logger.info("Budget summary: %s", budget.summary())


if __name__ == "__main__":
    demo()
```

---

## Exercises

(Solutions: ../solutions/week-08/day-4.md)

1. **Repair with retries**: Extend `repair_json_output` so it wraps the repair call itself in `call_with_retry`. The repair call should use a `RetryConfig` with `max_retries=2` and `base_delay=0.5`. Ensure the retry only fires on transient API errors, not on continued `ValidationError`.

2. **Cost estimator**: Write a function `estimate_call_cost(prompt: str, max_tokens: int, model: str) -> float` that estimates the USD cost of a call before making it. Use the heuristic 1 token ≈ 4 characters for the prompt. Implement pricing for `claude-opus-5-5` (input $15/M, output $75/M) and `claude-sonnet-5-5` (research current pricing). Return the worst-case estimate assuming all `max_tokens` are consumed.

3. **Budget guard**: Add a `max_budget_usd` guard to `call_with_retry` that calls `budget.check_budget(projected_input, projected_output)` before making the API call. Use the character-based token estimator from exercise 2 for `projected_input`, and `kwargs.get("max_tokens", 1024)` for `projected_output`. Raise `BudgetExceededError` before any network call if the budget would be exceeded.

4. **Test suite**: Write a `pytest` test module `tests/test_reliability.py` that covers:
   - `repair_json_output` succeeds on first try when output is valid JSON
   - `repair_json_output` succeeds after one repair when the first output is broken JSON
   - `repair_json_output` raises `ValueError` after exceeding `max_repair_attempts`
   - `TokenBudget.check_budget` raises `BudgetExceededError` when cumulative cost would exceed the limit
   - All Anthropic client calls must be mocked with `unittest.mock.MagicMock` — no real API calls

---

## Self-check

- [ ] I can explain why exponential backoff with jitter reduces thundering herd on rate limit errors.
- [ ] I can describe the self-repair loop: what goes in the repair prompt, why including the error message helps.
- [ ] I can distinguish retry-eligible errors (transient API) from repair-eligible errors (semantic/validation).
- [ ] I can calculate the USD cost of a call given `input_tokens`, `output_tokens`, and per-million pricing.
- [ ] I can explain prompt caching: which content is worth caching, what `cache_creation_input_tokens` vs `cache_read_input_tokens` mean.
- [ ] I can explain what `max_tokens` and `stop_sequences` control and why both matter for cost.

---

## Supplementary reading

- [Tenacity documentation](https://tenacity.readthedocs.io/en/latest/)
- [Anthropic API errors reference](https://docs.anthropic.com/en/api/errors)
- [Anthropic prompt caching guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
- [Pydantic v2 model_json_schema](https://docs.pydantic.dev/latest/concepts/json_schema/)
