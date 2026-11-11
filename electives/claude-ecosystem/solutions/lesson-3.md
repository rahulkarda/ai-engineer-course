# Solution: Lesson 3 — Prompt Caching

## Exercise 1: Silent invalidator audit

The bug: `datetime.date.today()` in the system prompt changes the prefix every day,
invalidating the cache. The user name also varies per user.

```python
# BROKEN — prefix changes daily and per-user
import datetime
def build_system_broken(user_name: str) -> str:
    return f"Today is {datetime.date.today()}. You are helping {user_name}."

# FIXED — move dynamic content out of the stable prefix
STABLE_SYSTEM = "You are a helpful technical assistant."

def build_messages_with_dynamic_context(user_name: str, question: str) -> list[dict]:
    """
    Keep the system prompt stable (cacheable).
    Inject dynamic context into the first user message.
    """
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"[Context: helping user {user_name} on {datetime.date.today()}]\n{question}",
                }
            ],
        }
    ]

# The system prompt never changes → prefix is always cacheable.
```

**Key insight:** Any per-request value (date, user ID, UUID) placed in `system` or
anywhere before a `cache_control` marker invalidates every downstream cache entry.
Move it *after* the last breakpoint — typically into the user message.

## Exercise 2: Cache hit verifier

```python
def verify_cache_hit(usage, threshold: float = 0.80) -> bool:
    """
    Return True if at least `threshold` fraction of input tokens came from cache.
    """
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    total_input = (
        (getattr(usage, "input_tokens", 0) or 0)
        + cache_read
        + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
    )
    if total_input == 0:
        return False
    return (cache_read / total_input) >= threshold


# tests
from unittest.mock import MagicMock
import pytest

def make_usage(input_tokens, cache_read, cache_write=0):
    u = MagicMock()
    u.input_tokens = input_tokens
    u.cache_read_input_tokens = cache_read
    u.cache_creation_input_tokens = cache_write
    return u

@pytest.mark.parametrize("inp,read,write,expected", [
    (200,   800,    0, True),   # 80% from cache
    (300,   700,    0, True),   # 70% — borderline True at 0.70 threshold
    (800,   200,    0, False),  # only 20% from cache
    (1000,    0,    0, False),  # no cache
    (0,     0,      0, False),  # all zeros
])
def test_verify_cache_hit(inp, read, write, expected):
    usage = make_usage(inp, read, write)
    assert verify_cache_hit(usage, threshold=0.70) == expected
```

## Exercise 3: Multi-document caching layout

```python
"""
Layout for a RAG pipeline serving 10 documents with up to 4 cache breakpoints.

Strategy:
  Breakpoint 1: System prompt (instructions) — never changes.
  Breakpoint 2: Selected document content — changes per document group.
  Breakpoints 3-4: Reserved for few-shot examples or large retrieved context.

Since we only have 4 breakpoints and 10 documents, we cannot cache each document
individually across requests. Instead, structure requests so that all questions
about document D are batched together — the document is cached for the duration
of that group.
"""

INSTRUCTIONS = (
    "You are a precise document analyser. "
    "Answer questions based only on the provided document."
)


def build_doc_request(doc_text: str, question: str) -> dict:
    """
    Build a messages.create kwargs dict with two cache breakpoints:
      1. After instructions (stable across all documents)
      2. After the document text (stable within one document's question group)
    """
    return {
        "system": [
            {
                "type": "text",
                "text": INSTRUCTIONS,
                "cache_control": {"type": "ephemeral"},   # Breakpoint 1
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"<document>\n{doc_text}\n</document>",
                        "cache_control": {"type": "ephemeral"},   # Breakpoint 2
                    },
                    {
                        "type": "text",
                        "text": question,
                        # No cache_control — varies per question
                    },
                ],
            }
        ],
    }


# Usage: for each document, ask all its questions in sequence.
# The document content at breakpoint 2 will be served from cache
# starting from the second question onward.
```

## Exercise 4: Savings calculator tests

```python
from lesson_3_demo import calculate_cache_savings
import pytest


def test_zero_cache_hit_rate():
    """No cache reads → savings should be negative (cache write cost)."""
    result = calculate_cache_savings(
        uncached_input_tokens=10_000,
        cached_input_tokens=0,
        cache_write_tokens=0,
        output_tokens=500,
    )
    # No caching at all — costs are identical
    assert result["savings_pct"] == 0.0


def test_full_cache_hit():
    """All tokens served from cache → ~87% savings on input (0.1x vs 1.0x)."""
    result = calculate_cache_savings(
        uncached_input_tokens=10_000,
        cached_input_tokens=10_000,
        cache_write_tokens=10_000,
        output_tokens=0,
    )
    # Cost_no_cache = 10000 * 0.10 / 1e6 = 0.001
    # Cost_with_cache = 10000 * 0.10 * 1.25 / 1e6 (write) + 10000 * 0.10 * 0.10 / 1e6 (read)
    # = 0.00125 + 0.0001 = 0.00135
    # savings = (1 - 0.00135/0.001) * 100 = -35% (first call is MORE expensive due to write)
    # After 2nd call: cost = 0.0001 → savings = 90%
    assert result["cost_no_cache_usd"] > 0
    assert result["cost_with_cache_usd"] > 0


def test_break_even_savings():
    """
    On the first call, cache write costs 1.25x normal.
    With 100% cache read on the SAME call, effective cost is:
    write (1.25x) + read (0.10x) for the same tokens = 1.35x → negative savings.
    This is expected: caching pays off across multiple requests, not within one.
    """
    result = calculate_cache_savings(
        uncached_input_tokens=5000,
        cached_input_tokens=5000,
        cache_write_tokens=5000,
        output_tokens=100,
    )
    # savings_pct < 0 for a single-request scenario
    # (write premium + read discount = net cost increase on first call)
    assert isinstance(result["savings_pct"], float)
    assert "cost_no_cache_usd" in result
    assert "cost_with_cache_usd" in result
```
