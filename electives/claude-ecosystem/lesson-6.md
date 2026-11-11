# Elective: Claude Ecosystem · Lesson 6 — Token Counting & Usage Tracking

> Objective: Count tokens accurately before sending requests; track cumulative costs in production.
> Time: ~1 hr · Prereqs: [[lesson-1.md]], [[lesson-2.md]]

## Concept

Token counting is the foundation of cost control. Two use cases:

1. **Pre-flight estimation:** Count before sending to decide model tier, chunk size, or
   whether to warn a user their input is too large.
2. **Post-call tracking:** Accumulate `response.usage` to build per-user, per-job,
   or per-session cost dashboards.

**Never use `tiktoken`** (OpenAI's tokenizer). It undercounts Claude tokens by 15–20%
on typical text and by much more on code or non-English input.

**The `count_tokens` endpoint is model-specific.** Pass the same model ID you'll use
for inference. Token counts differ between models.

## How it works internally

`client.messages.count_tokens()` takes the same `model`, `messages`, `system`, and `tools`
parameters as `messages.create()` but returns only `input_tokens: int`. No tokens are
generated; no cost is incurred.

After a real `messages.create()` call, `response.usage` carries:

| Field | Meaning |
|-------|---------|
| `input_tokens` | Uncached tokens billed at full input price |
| `cache_creation_input_tokens` | Tokens written to cache (~1.25x input price) |
| `cache_read_input_tokens` | Tokens served from cache (~0.10x input price) |
| `output_tokens` | Generated tokens billed at output price |

**Effective cost formula:**

```
cost = (input_tokens * input_price
        + cache_creation_input_tokens * input_price * 1.25
        + cache_read_input_tokens * input_price * 0.10
        + output_tokens * output_price) / 1_000_000
```

## Worked example

```python
# lesson_6_demo.py
import logging
import os
from dataclasses import dataclass, field

import anthropic

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Prices per 1M tokens (as of the skill reference, Nov 2026)
MODEL_PRICES: dict[str, dict[str, float]] = {
    "claude-opus-5-5":   {"input": 4.00,  "output": 20.00},
    "claude-sonnet-5-5": {"input": 2.00,  "output": 10.00},
    "claude-haiku-5-5":  {"input": 0.10,  "output": 0.50},
}


def count_tokens(
    model: str,
    messages: list[dict],
    system: str | list[dict] | None = None,
    tools: list[dict] | None = None,
) -> int:
    """Return the input token count for a hypothetical request."""
    kwargs: dict = {"model": model, "messages": messages}
    if system is not None:
        kwargs["system"] = system
    if tools is not None:
        kwargs["tools"] = tools

    result = client.messages.count_tokens(**kwargs)
    log.info("Token count for model=%s: %d", model, result.input_tokens)
    return result.input_tokens


def calculate_cost(usage: anthropic.types.Usage, model: str) -> float:
    """
    Calculate the USD cost of a single API response.
    Handles caching fields gracefully (they may be 0 or absent).
    """
    prices = MODEL_PRICES.get(model, MODEL_PRICES["claude-opus-5-5"])
    m = 1_000_000

    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    regular_input = usage.input_tokens

    cost = (
        regular_input * prices["input"] / m
        + cache_write * prices["input"] * 1.25 / m
        + cache_read * prices["input"] * 0.10 / m
        + usage.output_tokens * prices["output"] / m
    )
    return cost


@dataclass
class CostTracker:
    """Accumulate token usage and cost across multiple API calls."""

    model: str
    total_input_tokens: int = 0
    total_cache_write_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_output_tokens: int = 0
    call_count: int = 0
    _call_costs: list[float] = field(default_factory=list)

    def record(self, usage: anthropic.types.Usage) -> float:
        """Record usage from one response; return the cost of this call."""
        self.total_input_tokens += usage.input_tokens
        self.total_cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.total_cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.total_output_tokens += usage.output_tokens
        self.call_count += 1

        cost = calculate_cost(usage, self.model)
        self._call_costs.append(cost)
        return cost

    @property
    def total_cost(self) -> float:
        return sum(self._call_costs)

    @property
    def cache_hit_rate(self) -> float:
        total_in = self.total_input_tokens + self.total_cache_read_tokens + self.total_cache_write_tokens
        if total_in == 0:
            return 0.0
        return self.total_cache_read_tokens / total_in

    def summary(self) -> dict:
        return {
            "calls": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "cache_write_tokens": self.total_cache_write_tokens,
            "cache_read_tokens": self.total_cache_read_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 6),
            "cache_hit_rate_pct": round(self.cache_hit_rate * 100, 1),
        }


def safe_to_send(
    model: str,
    messages: list[dict],
    system: str | None = None,
    max_input_tokens: int = 100_000,
) -> bool:
    """Pre-flight check: return False if the request would exceed `max_input_tokens`."""
    actual = count_tokens(model, messages, system=system)
    if actual > max_input_tokens:
        log.warning(
            "Request too large: %d tokens > %d limit for model %s",
            actual, max_input_tokens, model,
        )
        return False
    return True


def token_budget_pipeline(
    texts: list[str],
    model: str = "claude-haiku-5-5",
    max_tokens_per_request: int = 50_000,
) -> tuple[list[str], dict]:
    """
    Process texts; skip any that exceed the per-request token budget.
    Returns (answers, cost_summary).
    """
    tracker = CostTracker(model=model)
    answers: list[str] = []

    for text in texts:
        messages = [{"role": "user", "content": text}]
        if not safe_to_send(model, messages, max_input_tokens=max_tokens_per_request):
            answers.append("[SKIPPED: too long]")
            continue

        response = client.messages.create(
            model=model,
            max_tokens=256,
            messages=messages,
        )
        cost = tracker.record(response.usage)
        answer = next(b.text for b in response.content if b.type == "text")
        answers.append(answer)
        log.info("Cost this call: $%.6f", cost)

    log.info("Pipeline summary: %s", tracker.summary())
    return answers, tracker.summary()


if __name__ == "__main__":
    # Pre-flight token count
    messages = [{"role": "user", "content": "Explain gradient descent in two sentences."}]
    n = count_tokens("claude-haiku-5-5", messages)
    log.info("Pre-flight token count: %d", n)

    # Run pipeline on a small batch
    texts = [
        "What is the capital of France?",
        "Summarize the history of Python in one sentence.",
    ]
    results, summary = token_budget_pipeline(texts)
    log.info("Results: %s", results)
    log.info("Summary: %s", summary)
```

Key observations:
- `count_tokens` uses the same parameter names as `messages.create` — easy to reuse
  your request-building logic.
- `CostTracker` is a pure data accumulator; no API calls inside it — easy to test.
- `cache_creation_input_tokens` / `cache_read_input_tokens` may be absent (None) on
  models or requests without caching; guard with `or 0`.

## Exercises

1. **Per-model cost comparison.** Given a fixed prompt (200 input tokens, 100 output tokens),
   calculate and print the cost for all three 5.5 model tiers.
   Write as a pure function with a pytest test.

2. **Budget enforcement.** Extend `token_budget_pipeline` with a `total_budget_usd` parameter.
   Stop processing new texts once the accumulated cost exceeds the budget.
   Test the budget cutoff with mocked usage objects.

3. **CostTracker serialisation.** Add a `to_json() -> str` method to `CostTracker` that
   emits the `summary()` dict as JSON (use `json.dumps`). Write a round-trip test.

4. **Token count vs. actual.** Write a test that uses mocked `count_tokens` (pre-flight)
   and mocked `messages.create` (post-call) to verify:
   (a) `safe_to_send` returns `False` when mocked count exceeds the limit.
   (b) The tracker's `total_input_tokens` matches the mocked `usage.input_tokens`.

(Solutions: solutions/lesson-6.md)

## Self-check

- [ ] I can call `client.messages.count_tokens()` with the same parameters as a real request.
- [ ] I understand why `tiktoken` gives wrong results for Claude.
- [ ] I can calculate USD cost from `response.usage` including cache fields.
- [ ] I can build a `CostTracker` that accumulates usage across multiple calls.
- [ ] I can implement a pre-flight token budget check.

## Supplementary links

- [Token counting guide](https://docs.anthropic.com/en/docs/build-with-claude/token-counting)
- [Usage object reference](https://docs.anthropic.com/en/api/messages#response-usage)
- [Pricing page](https://www.anthropic.com/pricing)
