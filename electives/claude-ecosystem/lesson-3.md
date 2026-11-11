# Elective: Claude Ecosystem · Lesson 3 — Prompt Caching

> Objective: Design stable prompt prefixes and use `cache_control` to cut costs by up to 90%.
> Time: ~1 hr · Prereqs: [[lesson-1.md]], [[lesson-2.md]]

## Concept

**Prompt caching stores the KV-cache** of a rendered prompt prefix on Anthropic's servers
for a TTL (default 5 minutes, or set `"ttl": "1h"` for one hour). On a cache hit, those
tokens cost ~10% of the normal input price instead of 100%.

Cache write: tokens in the prefix are billed at ~125% of normal input price (one-time cost).
Cache read: those same tokens cost ~10% of normal input price on every subsequent hit.
Break-even: approximately 2 requests for a cold-start prefix to pay for itself.

**This is the single highest-leverage cost optimization.** Before reaching for a smaller
model, add caching to your stable context.

## How it works internally

**Prefix match rule:** The cache key is derived from the exact bytes of the rendered prompt
up to each `cache_control` marker. **A single byte change anywhere in the prefix invalidates
everything downstream.**

Render order: `tools` → `system` → `messages`.

A breakpoint on the last `system` block caches: tools + system.
A breakpoint on the last `messages` turn caches: tools + system + all earlier turns.

**Maximum 4 breakpoints per request.** Place them at stability boundaries.

**Minimum cacheable prefix:** approximately 1024–4096 tokens depending on model.
Shorter prefixes don't cache silently — no error, just a miss.

**Silent invalidators** — the most common bugs:
- `datetime.now()` injected into `system` → prefix changes every second.
- `uuid4()` or random IDs in stable content → every request is unique.
- `json.dumps(d)` without `sort_keys=True` → non-deterministic key order.
- Changing the tool list between requests → tools render at position 0, everything invalidates.

**Verify with `response.usage`:**
```
cache_creation_input_tokens  # tokens written to cache this request
cache_read_input_tokens      # tokens served from cache this request
input_tokens                 # uncached tokens billed at full price
```
If `cache_read_input_tokens` stays zero across repeated identical requests,
a silent invalidator is at work.

## Worked example

```python
# lesson_3_demo.py
import logging
import os
from anthropic import Anthropic

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Simulated large shared document (in production: read from file)
LARGE_DOCUMENT = "=" * 2000 + "\n# Important Technical Specification\n" + "x" * 4000


def build_system_with_cache() -> list[dict]:
    """System prompt structured for maximum cache efficiency."""
    return [
        {
            "type": "text",
            "text": (
                "You are an expert technical analyst. "
                "Your job is to answer questions about the specification below.\n\n"
                + LARGE_DOCUMENT
            ),
            # Mark the end of the stable prefix — everything above will be cached.
            "cache_control": {"type": "ephemeral"},
        }
    ]


def query_document(question: str) -> tuple[str, dict]:
    """
    Query the document; return (answer, usage_dict).
    First call: cache write (slightly expensive).
    Subsequent calls: cache read (cheap).
    """
    response = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=512,
        system=build_system_with_cache(),
        messages=[{"role": "user", "content": question}],
    )

    usage = {
        "input_tokens": response.usage.input_tokens,
        "cache_creation_input_tokens": response.usage.cache_creation_input_tokens,
        "cache_read_input_tokens": response.usage.cache_read_input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    log.info("Usage: %s", usage)

    answer = next(b.text for b in response.content if b.type == "text")
    return answer, usage


def multi_turn_with_cache(questions: list[str]) -> list[str]:
    """
    Multi-turn conversation: cache the growing history prefix.
    The last user message breakpoint caches all prior turns.
    """
    history: list[dict] = []
    answers: list[str] = []

    system = [{"type": "text", "text": "You are a helpful assistant."}]

    for question in questions:
        # Add cache breakpoint on the last content block of the latest user turn.
        user_content = [
            {"type": "text", "text": question, "cache_control": {"type": "ephemeral"}}
        ]
        history.append({"role": "user", "content": user_content})

        response = client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=256,
            system=system,
            messages=history,
        )

        assistant_text = next(b.text for b in response.content if b.type == "text")
        # Append full content blocks (important for thinking blocks in other models)
        history.append({"role": "assistant", "content": response.content})
        answers.append(assistant_text)

        log.info(
            "Q%d cache_read=%d  cache_write=%d",
            len(answers),
            response.usage.cache_read_input_tokens,
            response.usage.cache_creation_input_tokens,
        )

    return answers


def calculate_cache_savings(
    uncached_input_tokens: int,
    cached_input_tokens: int,
    cache_write_tokens: int,
    output_tokens: int,
    model: str = "claude-haiku-5-5",
) -> dict:
    """
    Compare cost with and without caching.
    Haiku 5.5: $0.10 input / $0.50 output per 1M tokens.
    Cache write: 1.25x input price. Cache read: 0.1x input price.
    """
    rates = {
        "claude-haiku-5-5":   {"input": 0.10, "output": 0.50},
        "claude-sonnet-5-5":  {"input": 2.00, "output": 10.00},
        "claude-opus-5-5":    {"input": 4.00, "output": 20.00},
    }
    r = rates[model]
    m = 1_000_000

    cost_no_cache = (uncached_input_tokens / m) * r["input"] + (output_tokens / m) * r["output"]
    cost_with_cache = (
        (cache_write_tokens / m) * r["input"] * 1.25
        + (cached_input_tokens / m) * r["input"] * 0.10
        + ((uncached_input_tokens - cache_write_tokens) / m) * r["input"]
        + (output_tokens / m) * r["output"]
    )
    savings_pct = (1 - cost_with_cache / cost_no_cache) * 100 if cost_no_cache > 0 else 0.0

    return {
        "cost_no_cache_usd": round(cost_no_cache, 6),
        "cost_with_cache_usd": round(cost_with_cache, 6),
        "savings_pct": round(savings_pct, 1),
    }


if __name__ == "__main__":
    # Two questions about the same document — second call should hit cache
    a1, u1 = query_document("What is this document about?")
    a2, u2 = query_document("Give me a one-line summary.")
    log.info("Answer 1: %s", a1[:100])
    log.info("Answer 2: %s", a2[:100])

    # Savings estimate
    savings = calculate_cache_savings(
        uncached_input_tokens=6000,
        cached_input_tokens=5800,
        cache_write_tokens=5800,
        output_tokens=200,
    )
    log.info("Cache savings: %s", savings)
```

## Exercises

1. **Silent invalidator audit.** The following system prompt builder has a bug that
   prevents caching. Identify it and fix it:
   ```python
   import datetime
   def build_system(user_name: str) -> str:
       return f"Today is {datetime.date.today()}. You are helping {user_name}."
   ```

2. **Cache hit verifier.** Write a function `verify_cache_hit(usage) -> bool` that
   returns `True` when at least 80% of input tokens were served from cache.
   Test it with mocked usage objects.

3. **Multi-document caching.** Design a system prompt structure for a RAG pipeline
   that serves 10 different documents. Use multiple `cache_control` breakpoints
   (up to 4 allowed) to maximize prefix reuse across questions about the same doc.
   Draw the breakpoint layout as comments before the code.

4. **Savings calculator test.** Write parametrized pytest tests for `calculate_cache_savings`
   covering: (a) 0% cache hit rate, (b) 100% cache hit rate on a 10k-token prefix,
   (c) the break-even point (savings ≈ 0 on the first call).

(Solutions: solutions/lesson-3.md)

## Self-check

- [ ] I can explain the byte-exact prefix match rule and its implications.
- [ ] I know the render order (`tools` → `system` → `messages`) and where to place breakpoints.
- [ ] I can identify and fix the five most common silent invalidators.
- [ ] I can read `response.usage` to verify cache hits.
- [ ] I can calculate the cost savings of caching for a given request pattern.

## Supplementary links

- [Prompt caching guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
- [cache_control API reference](https://docs.anthropic.com/en/api/messages#body-system-cache_control)
- [Usage object reference](https://docs.anthropic.com/en/api/messages#response-usage)
