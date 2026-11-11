# Phase 6 · Week 10 · Day 5 — Streaming Responses + Token/Cost Reduction + Caching

> Objective: Implement streaming clients, understand caching layers, and apply cost-reduction techniques at the API level.  ·  Time: ~1 hr  ·  Prereqs: [[day-4.md]]

## Concept

Three orthogonal levers reduce inference cost and latency:

### 1. Streaming (latency perception)
LLMs generate tokens sequentially. **Server-Sent Events (SSE)** let the server push each token to the client as it is generated, so time-to-first-token (TTFT) feels fast even when total generation is slow. The wire protocol is plain HTTP chunked transfer with `Content-Type: text/event-stream`.

### 2. Prompt compression (cost reduction)
Every input token costs money. Common waste:
- Verbose system prompts with obvious instructions ("You are a helpful AI assistant...")
- Redundant chain-of-thought instructions on every request
- Boilerplate in few-shot examples

Strategies: prune filler, reuse prompts via caching, route simple queries to smaller models, use structured output to avoid generation verbosity.

### 3. Caching (cost + latency reduction)

**Exact caching**: hash the full prompt; return stored response on collision. Works for repeated identical calls (batch jobs, retries). Cache key = SHA256(prompt + model + params).

**Semantic caching**: embed the query; find nearest cached query by cosine similarity; return its response if similarity > threshold. Works for paraphrased queries. Requires a vector store.

**Anthropic prompt caching**: mark stable prompt prefixes with `cache_control: {"type": "ephemeral"}`. Anthropic caches the KV activations for 5 minutes. Cache hits cost ~10% of normal input token price — 90% savings. Requires the prefix to be at least 1024 tokens (Haiku: 2048). The cached prefix must be the same across requests.

**KV-cache warming**: send a "warm-up" request with just the system prompt during server startup so the first real user request gets a cache hit.

## How it works internally

### SSE protocol

```
HTTP/1.1 200 OK
Content-Type: text/event-stream
Transfer-Encoding: chunked
Cache-Control: no-cache

data: {"type": "content_block_delta", "delta": {"text": "Hello"}}

data: {"type": "content_block_delta", "delta": {"text": " world"}}

data: [DONE]
```

Each chunk is sent as soon as it is ready. The client reads the stream and appends text to the UI. TTFT = time from request send to first `data:` line. Per-token latency = time between successive `data:` lines.

### Exact caching (hash-based)

```
key = sha256(model + system_prompt + user_message + temperature + max_tokens)
cache[key] → stored response
```

Hit rate is brittle: any parameter change = miss. Good for idempotent batch jobs.

### Semantic caching (embedding similarity)

```
query_embedding = embed(user_message)          # e.g., 1536-dim vector
for cached_embedding, cached_response in cache:
    sim = cosine_similarity(query_embedding, cached_embedding)
    if sim > threshold:  # e.g., 0.92
        return cached_response
# Cache miss: call LLM, store (query_embedding, response)
```

Threshold tuning is critical: too low = wrong cached answers; too high = low hit rate.

### Anthropic cache_control

```python
messages=[
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": long_stable_system_context,   # >= 1024 tokens
                "cache_control": {"type": "ephemeral"}  # mark for caching
            },
            {
                "type": "text",
                "text": user_question                  # varies per request
            }
        ]
    }
]
```

The response includes `usage.cache_read_input_tokens` and `usage.cache_creation_input_tokens`. On first call, tokens are written to cache (slightly more expensive). On subsequent calls within 5 min, `cache_read_input_tokens` is populated.

## Worked example

```python
"""
day5_streaming_and_caching.py

Demonstrates:
  1. Streaming client with TTFT and per-token timing
  2. SemanticCache mock (in-memory, cosine similarity)
  3. Anthropic prompt caching with cache_control
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Generator, Iterator

import anthropic
import numpy as np
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL = "claude-haiku-5-5"


# ---------------------------------------------------------------------------
# Pydantic models for stream events and timing
# ---------------------------------------------------------------------------

class StreamEvent(BaseModel):
    """One token event from the stream."""
    token: str
    elapsed_ms: float = Field(description="ms since stream start")
    is_first: bool = False


class StreamSummary(BaseModel):
    """Aggregated stats for a completed stream."""
    full_text: str
    ttft_ms: float = Field(description="Time to first token, ms")
    total_ms: float
    token_count: int
    avg_inter_token_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


# ---------------------------------------------------------------------------
# 1. Streaming client
# ---------------------------------------------------------------------------

class StreamingClient:
    """
    Wraps Anthropic streaming API.
    Yields StreamEvent objects and produces a StreamSummary.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._client = anthropic.Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )

    def stream(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 256,
    ) -> tuple[Generator[StreamEvent, None, None], StreamSummary]:
        """
        Returns (event_generator, summary).
        summary is populated after the generator is exhausted.
        """
        summary_holder: dict = {}

        def _generate() -> Generator[StreamEvent, None, None]:
            messages = [{"role": "user", "content": prompt}]
            kwargs: dict = {
                "model": MODEL,
                "max_tokens": max_tokens,
                "messages": messages,
            }
            if system:
                kwargs["system"] = system

            start = time.perf_counter()
            first_token_time: float | None = None
            tokens: list[str] = []
            token_times: list[float] = []

            with self._client.messages.stream(**kwargs) as stream:
                for text_chunk in stream.text_stream:
                    now = time.perf_counter()
                    elapsed_ms = (now - start) * 1000
                    is_first = first_token_time is None
                    if is_first:
                        first_token_time = elapsed_ms

                    event = StreamEvent(
                        token=text_chunk,
                        elapsed_ms=elapsed_ms,
                        is_first=is_first,
                    )
                    tokens.append(text_chunk)
                    token_times.append(elapsed_ms)
                    yield event

                final_msg = stream.get_final_message()
                usage = final_msg.usage
                total_ms = (time.perf_counter() - start) * 1000
                inter_token_gaps = [
                    token_times[i] - token_times[i - 1]
                    for i in range(1, len(token_times))
                ]
                summary_holder["summary"] = StreamSummary(
                    full_text="".join(tokens),
                    ttft_ms=first_token_time or 0.0,
                    total_ms=total_ms,
                    token_count=len(tokens),
                    avg_inter_token_ms=(
                        sum(inter_token_gaps) / len(inter_token_gaps)
                        if inter_token_gaps else 0.0
                    ),
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0),
                    cache_creation_tokens=getattr(
                        usage, "cache_creation_input_tokens", 0
                    ),
                )

        gen = _generate()
        # Return a proxy that, once exhausted, makes summary available
        return gen, summary_holder

    def stream_and_print(self, prompt: str, system: str = "") -> StreamSummary:
        """Convenience: stream to stdout, return summary."""
        gen, holder = self.stream(prompt, system)
        print("Response: ", end="", flush=True)
        for event in gen:
            print(event.token, end="", flush=True)
        print()
        summary = holder.get("summary")
        if summary:
            log.info(
                "TTFT=%.1fms | total=%.1fms | tokens=%d | avg_ITL=%.1fms",
                summary.ttft_ms,
                summary.total_ms,
                summary.token_count,
                summary.avg_inter_token_ms,
            )
            if summary.cache_read_tokens:
                log.info("Cache hit: %d tokens read from cache", summary.cache_read_tokens)
        return summary


# ---------------------------------------------------------------------------
# 2. Semantic cache (mock — no real embeddings)
# ---------------------------------------------------------------------------

def _mock_embed(text: str, dim: int = 64) -> np.ndarray:
    """
    Deterministic mock embedding: hash text → seed → random unit vector.
    Similar strings get dissimilar embeddings (intentional — this is a mock).
    In production: use a real embedding model (e.g., text-embedding-3-small).
    """
    seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim)
    return vec / np.linalg.norm(vec)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # both unit vectors


@dataclass
class CacheEntry:
    query: str
    embedding: np.ndarray
    response: str
    hit_count: int = 0


class SemanticCache:
    """
    In-memory semantic cache using cosine similarity.
    Uses mock embeddings — swap _mock_embed for a real model.
    """

    def __init__(self, threshold: float = 0.92) -> None:
        self.threshold = threshold
        self._entries: list[CacheEntry] = []
        self.total_lookups: int = 0
        self.total_hits: int = 0

    def get(self, query: str) -> str | None:
        """Return cached response if similar query exists, else None."""
        self.total_lookups += 1
        q_emb = _mock_embed(query)
        best_sim = -1.0
        best_entry: CacheEntry | None = None

        for entry in self._entries:
            sim = cosine_similarity(q_emb, entry.embedding)
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_entry and best_sim >= self.threshold:
            best_entry.hit_count += 1
            self.total_hits += 1
            log.info(
                "Cache HIT (sim=%.4f >= %.2f): '%s'",
                best_sim,
                self.threshold,
                query[:60],
            )
            return best_entry.response

        log.info("Cache MISS (best_sim=%.4f): '%s'", best_sim, query[:60])
        return None

    def put(self, query: str, response: str) -> None:
        embedding = _mock_embed(query)
        self._entries.append(CacheEntry(query=query, embedding=embedding, response=response))

    @property
    def hit_rate(self) -> float:
        if self.total_lookups == 0:
            return 0.0
        return self.total_hits / self.total_lookups

    def stats(self) -> dict:
        return {
            "entries": len(self._entries),
            "lookups": self.total_lookups,
            "hits": self.total_hits,
            "hit_rate": round(self.hit_rate, 3),
        }


# ---------------------------------------------------------------------------
# 3. Anthropic prompt caching demo
# ---------------------------------------------------------------------------

LONG_SYSTEM_CONTEXT = """
You are an expert Python code reviewer. You have deep knowledge of:
- PEP 8 style guidelines and modern Python idioms
- Type annotations and mypy compatibility
- Performance optimization techniques
- Security best practices including input validation and injection prevention
- Testing with pytest including fixtures, parametrize, and mock
- Async programming with asyncio
- Memory management and garbage collection in CPython
- Common anti-patterns: mutable default arguments, bare excepts, global state
- Design patterns: dependency injection, factory, observer, strategy
- Documentation standards: Google-style, NumPy-style, Sphinx
""" * 5  # Repeat to exceed 1024 token minimum for caching demo


def demo_prompt_caching(client: anthropic.Anthropic, question: str) -> dict:
    """
    Show prompt caching with cache_control.
    The long system context is marked for caching.
    Second call should show cache_read_input_tokens > 0.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": LONG_SYSTEM_CONTEXT,
                    "cache_control": {"type": "ephemeral"},  # Mark for caching
                },
                {
                    "type": "text",
                    "text": question,
                },
            ],
        }
    ]

    response = client.messages.create(
        model=MODEL,
        max_tokens=128,
        messages=messages,
    )
    usage = response.usage
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_creation_tokens": getattr(usage, "cache_creation_input_tokens", 0),
        "cache_read_tokens": getattr(usage, "cache_read_input_tokens", 0),
        "answer": response.content[0].text,
    }


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("ANTHROPIC_API_KEY not set — set it in your environment")
        return

    # --- Semantic cache demo (no API calls needed) ---
    log.info("=== Semantic Cache Demo ===")
    cache = SemanticCache(threshold=0.92)

    # Populate cache
    cache.put("What is the capital of France?", "The capital of France is Paris.")
    cache.put("Explain async/await in Python", "async/await enables non-blocking I/O...")

    # Exact hit (same string → same embedding → sim = 1.0)
    result = cache.get("What is the capital of France?")
    log.info("Retrieved: %s", result)

    # Near-miss (different string → different embedding in mock → sim < threshold)
    result = cache.get("France capital city?")
    log.info("Retrieved (near-miss): %s", result)

    log.info("Cache stats: %s", cache.stats())

    # --- Streaming demo ---
    log.info("=== Streaming Demo ===")
    streaming_client = StreamingClient(api_key=api_key)
    summary = streaming_client.stream_and_print(
        prompt="List 3 benefits of type hints in Python in one sentence each.",
    )
    log.info("Summary: %s", summary.model_dump())

    # --- Prompt caching demo ---
    log.info("=== Prompt Caching Demo ===")
    raw_client = anthropic.Anthropic(api_key=api_key)

    log.info("First call (cache creation):")
    result1 = demo_prompt_caching(raw_client, "What is a mutable default argument?")
    log.info("cache_creation=%d, cache_read=%d, answer: %s",
             result1["cache_creation_tokens"], result1["cache_read_tokens"],
             result1["answer"][:80])

    log.info("Second call within 5 min (should hit cache):")
    result2 = demo_prompt_caching(raw_client, "What is a bare except clause?")
    log.info("cache_creation=%d, cache_read=%d, answer: %s",
             result2["cache_creation_tokens"], result2["cache_read_tokens"],
             result2["answer"][:80])

    if result2["cache_read_tokens"] > 0:
        savings_pct = 90  # cache reads cost ~10% of normal input price
        log.info(
            "Cache HIT confirmed. Saved ~%d%% on %d tokens.",
            savings_pct,
            result2["cache_read_tokens"],
        )


if __name__ == "__main__":
    main()
```

## Exercises

1. **TTFT benchmark**: Extend `StreamingClient` to accept a `mock_mode: bool` parameter. In mock mode, yield pre-set tokens with configurable delays (e.g., 50ms TTFT, 20ms per token) instead of calling the API. Write a pytest test that verifies TTFT is within 10% of the expected mock delay.

2. **Semantic cache threshold sweep**: Write a function `evaluate_cache(queries: list[str], threshold: float) -> dict` that populates a `SemanticCache` with the first half of queries, then looks up the second half, returning hit rate and precision (no false hits). Run it for thresholds [0.80, 0.85, 0.90, 0.95, 0.99]. What threshold minimizes false positives while keeping hit rate > 0?

3. **Cost comparison — caching economics**: Write a `CostCalculator` that, given `n_requests`, `system_prompt_tokens`, `user_tokens`, `output_tokens`, and `cache_hit_rate`, computes total cost with and without prompt caching using Haiku pricing (input: $0.80/MTok, cache_read: $0.08/MTok, cache_creation: $1.00/MTok, output: $4.00/MTok). At what `cache_hit_rate` does caching break even vs. not caching?

(Solutions: ../solutions/week-10/day-5.md)

## Self-check

- [ ] I can explain the SSE wire format and why it feels faster than waiting for the full response.
- [ ] I can implement a streaming client that measures TTFT and inter-token latency.
- [ ] I can explain the difference between exact caching and semantic caching and when each is appropriate.
- [ ] I can use `cache_control: {"type": "ephemeral"}` in an Anthropic API call and read cache hit metrics from `usage`.
- [ ] I can calculate the break-even cache hit rate for Anthropic prompt caching.

## References

- Anthropic streaming API: https://docs.anthropic.com/en/api/messages-streaming
- Anthropic prompt caching: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
