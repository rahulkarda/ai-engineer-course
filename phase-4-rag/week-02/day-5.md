# Phase 4 · Week 02 · Day 5 — Embedding Providers, Batching & Cost

> Objective: Wire mocked embedding providers (OpenAI, sentence-transformers), implement batch embedding with retry logic, and calculate embedding cost.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

## Concept

There are two broad categories of embedding providers:

| Category | Examples | Pros | Cons |
|---|---|---|---|
| **Cloud API** | OpenAI, Cohere, Voyage AI | High quality, no GPU, managed | API cost per token, latency, rate limits |
| **Local model** | sentence-transformers, Ollama | Free after download, private, fast at scale | Requires machine resources, self-managed |

For production RAG, you choose based on:
- **Data sensitivity** — PII/confidential → local model.
- **Scale** — millions of chunks/day → cost calculator matters.
- **Quality** — MTEB leaderboard shows local models often match API quality.

### Batching

Embedding APIs (and local models) are optimised for batch requests: sending 100 texts in one call is faster and cheaper than 100 individual calls. Standard batch size: 100–2048 texts depending on provider.

### Cost (OpenAI pricing example)

`text-embedding-3-small`: ~$0.02 / 1M tokens  
`text-embedding-3-large`: ~$0.13 / 1M tokens

For 10,000 chunks averaging 200 tokens each:
- Total tokens = 2,000,000
- Cost (`3-small`) = $0.04
- Cost (`3-large`) = $0.26

Always estimate cost before running large embedding jobs.

## How it works internally

### Provider pattern

Use a `Protocol` so your chunking/embedding pipeline is agnostic:

```python
class EmbeddingProvider(Protocol):
    model: str
    dimensions: int
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def embed_batch(self, texts: list[str], batch_size: int) -> list[list[float]]: ...
```

Your application code calls `embed_batch`; the implementation decides whether to call OpenAI or run a local sentence-transformer.

### Retry with exponential backoff

Cloud APIs rate-limit. Implement retries with jitter so concurrent processes don't all retry at the same instant:

```
wait = base_delay * (2 ** attempt) + random_jitter
```

## Worked example

```python
# embeddings/providers.py
from __future__ import annotations

import logging
import os
import random
import time
from typing import Protocol, runtime_checkable

import numpy as np

from embeddings.similarity import normalise

logger = logging.getLogger(__name__)

Vector = list[float]


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Common interface for any embedding backend."""

    model: str
    dimensions: int

    def embed(self, texts: list[str]) -> list[Vector]:
        """Embed a list of texts in a single call."""
        ...

    def embed_batch(self, texts: list[str], batch_size: int = 100) -> list[Vector]:
        """Embed texts in batches, with retry on transient errors."""
        ...


# ---------------------------------------------------------------------------
# Mock provider — deterministic, no network
# ---------------------------------------------------------------------------

class MockEmbeddingProvider:
    """Deterministic embedding provider for testing and offline development.

    Returns normalised vectors whose direction is seeded by the text hash,
    so similar-looking texts get similar vectors.
    """

    model: str = "mock-embed-v1"
    dimensions: int = 384

    def embed(self, texts: list[str]) -> list[Vector]:
        results: list[Vector] = []
        for t in texts:
            seed = hash(t) % (2**32)
            rng = np.random.default_rng(seed)
            raw = rng.standard_normal(self.dimensions).astype(np.float32)
            results.append(normalise(raw).tolist())
        logger.debug("MockEmbeddingProvider.embed: %d texts → %d vectors", len(texts), len(results))
        return results

    def embed_batch(self, texts: list[str], batch_size: int = 100) -> list[Vector]:
        return _batch_embed(self, texts, batch_size)


# ---------------------------------------------------------------------------
# OpenAI provider — mocked (swap with real client when key is present)
# ---------------------------------------------------------------------------

class OpenAIEmbeddingProvider:
    """OpenAI embeddings via the official SDK.

    Requires:
        OPENAI_API_KEY environment variable.
        ``uv add openai`` (not installed here — mocked below).

    Usage:
        provider = OpenAIEmbeddingProvider(model="text-embedding-3-small")
        vectors = provider.embed_batch(texts)
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set. Add it to your .env file."
            )
        # Lazy import — only needed at runtime, not during tests.
        try:
            import openai  # noqa: F401
            self._client_available = True
        except ImportError:
            logger.warning("openai package not installed — using mock fallback.")
            self._client_available = False

    def embed(self, texts: list[str]) -> list[Vector]:
        if not self._client_available:
            return MockEmbeddingProvider().embed(texts)
        import openai
        client = openai.OpenAI()
        response = client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]

    def embed_batch(self, texts: list[str], batch_size: int = 100) -> list[Vector]:
        return _batch_embed(self, texts, batch_size)


# ---------------------------------------------------------------------------
# sentence-transformers provider (local)
# ---------------------------------------------------------------------------

class SentenceTransformerProvider:
    """Local embedding via sentence-transformers.

    Requires:
        ``uv add sentence-transformers``

    Usage:
        provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
        vectors = provider.embed_batch(texts)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = model_name
        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(model_name)
            self.dimensions = self._st_model.get_sentence_embedding_dimension()
            self._available = True
        except ImportError:
            logger.warning("sentence-transformers not installed — using mock fallback.")
            self._available = False
            self._mock = MockEmbeddingProvider()
            self.dimensions = self._mock.dimensions

    def embed(self, texts: list[str]) -> list[Vector]:
        if not self._available:
            return self._mock.embed(texts)
        embeddings = self._st_model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[Vector]:
        return _batch_embed(self, texts, batch_size)


# ---------------------------------------------------------------------------
# Shared batching + retry logic
# ---------------------------------------------------------------------------

def _batch_embed(
    provider: EmbeddingProvider,
    texts: list[str],
    batch_size: int,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> list[Vector]:
    """Batch-embed *texts* in chunks of *batch_size* with exponential backoff."""
    results: list[Vector] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_num in range(total_batches):
        batch = texts[batch_num * batch_size : (batch_num + 1) * batch_size]
        logger.debug("embedding batch %d/%d (%d texts)", batch_num + 1, total_batches, len(batch))

        for attempt in range(max_retries):
            try:
                vecs = provider.embed(batch)
                results.extend(vecs)
                break
            except Exception as exc:
                wait = base_delay * (2**attempt) + random.uniform(0, 0.5)
                logger.warning(
                    "embed attempt %d/%d failed: %s — retrying in %.1fs",
                    attempt + 1,
                    max_retries,
                    exc,
                    wait,
                )
                if attempt == max_retries - 1:
                    raise
                time.sleep(wait)

    logger.info("_batch_embed: embedded %d texts in %d batches", len(texts), total_batches)
    return results
```

```python
# embeddings/cost.py
from __future__ import annotations

from dataclasses import dataclass

# Approximate token count: 1 token ≈ 4 characters (OpenAI tiktoken baseline).
CHARS_PER_TOKEN = 4

# Prices per 1M tokens as of mid-2024 (USD). Check provider docs for current rates.
PRICING: dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}


@dataclass
class CostEstimate:
    model: str
    total_tokens: int
    total_chars: int
    estimated_usd: float

    def __str__(self) -> str:
        return (
            f"Model: {self.model}\n"
            f"Chars: {self.total_chars:,}\n"
            f"Est. tokens: {self.total_tokens:,}\n"
            f"Est. cost: ${self.estimated_usd:.4f}"
        )


def estimate_cost(
    texts: list[str],
    model: str = "text-embedding-3-small",
) -> CostEstimate:
    """Estimate embedding cost for *texts* using *model*.

    Uses the 4-chars-per-token heuristic. For exact token counts use tiktoken.
    """
    total_chars = sum(len(t) for t in texts)
    total_tokens = total_chars // CHARS_PER_TOKEN
    price_per_million = PRICING.get(model, 0.0)
    cost = (total_tokens / 1_000_000) * price_per_million
    return CostEstimate(
        model=model,
        total_tokens=total_tokens,
        total_chars=total_chars,
        estimated_usd=cost,
    )
```

```python
# tests/unit/test_providers.py
import pytest
from embeddings.providers import MockEmbeddingProvider, _batch_embed
import numpy as np


def test_mock_provider_dimensions():
    p = MockEmbeddingProvider()
    vecs = p.embed(["hello world"])
    assert len(vecs) == 1
    assert len(vecs[0]) == p.dimensions


def test_mock_provider_normalised():
    p = MockEmbeddingProvider()
    v = np.array(p.embed(["test text"])[0])
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5


def test_mock_provider_deterministic():
    p = MockEmbeddingProvider()
    v1 = p.embed(["same text"])
    v2 = p.embed(["same text"])
    assert v1 == v2


def test_batch_embed_returns_all():
    p = MockEmbeddingProvider()
    texts = [f"sentence {i}" for i in range(25)]
    vecs = p.embed_batch(texts, batch_size=10)
    assert len(vecs) == 25


def test_cost_estimate():
    from embeddings.cost import estimate_cost
    texts = ["a" * 400] * 100  # 100 texts × 400 chars = 40,000 chars = ~10,000 tokens
    est = estimate_cost(texts, model="text-embedding-3-small")
    assert est.total_tokens == 10_000
    assert abs(est.estimated_usd - 0.0002) < 1e-6
```

## Exercises

1. Implement a `VoyageAIProvider` stub that raises `NotImplementedError` for `embed` and documents the expected API key env var (`VOYAGE_API_KEY`). Add it to a `__all__` list in `providers.py`.
2. Add a `rate_limit_rpm: int` parameter to `_batch_embed`. After each batch, sleep `60 / rate_limit_rpm` seconds if the batch completed faster than that.
3. Write a `log_cost_before_embed` decorator that calls `estimate_cost` and logs the estimate at `WARNING` level before calling the wrapped `embed_batch` method.
4. Extend `test_providers.py`: assert that `embed_batch` on an empty list returns an empty list without raising.
5. Calculate: how many chunks averaging 300 tokens can you embed per dollar with `text-embedding-3-large`?

(Solutions: ../solutions/week-02/day-5.md)

## Self-check

- [ ] I can explain the tradeoff between cloud API and local model embedding.
- [ ] I can implement exponential backoff with jitter from scratch.
- [ ] I can calculate embedding cost for a given corpus and model.
- [ ] My `MockEmbeddingProvider` is deterministic and the tests never make real API calls.
- [ ] I understand why `sentence-transformers` models need `normalize_embeddings=True`.

## Supplementary links

- [OpenAI embeddings API reference](https://platform.openai.com/docs/api-reference/embeddings)
- [sentence-transformers usage docs](https://www.sbert.net/docs/usage/computing_sentence_embeddings.html)
- [Voyage AI embeddings](https://docs.voyageai.com/docs/embeddings)
- [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
