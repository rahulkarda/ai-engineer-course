# Day 2 Solutions — HyDE + Contextual Retrieval

## Exercise 1: Implement `HyDERetriever.retrieve`

```python
import logging
import os
from dataclasses import dataclass, field

import anthropic

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    id: str
    text: str
    embedding: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class HyDERetriever:
    """Hypothetical Document Embeddings retriever."""

    chunks: list[Chunk]
    model: str = "claude-sonnet-5-5"
    embed_fn: callable = None  # injected; signature: (text: str) -> list[float]

    def __post_init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def _generate_hypothetical_answer(self, query: str) -> str:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Write a concise, factual passage that would directly answer "
                        "the following question as if retrieved from a document.\n\n"
                        f"Question: {query}"
                    ),
                }
            ],
        )
        return msg.content[0].text.strip()

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x**2 for x in a) ** 0.5
        norm_b = sum(x**2 for x in b) ** 0.5
        return dot / (norm_a * norm_b + 1e-9)

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        """Generate hypothetical answer, embed it, return top-k similar chunks."""
        hypothetical = self._generate_hypothetical_answer(query)
        logger.debug("Hypothetical answer: %s", hypothetical[:80])

        hyp_embedding = self.embed_fn(hypothetical)

        scored: list[tuple[float, Chunk]] = [
            (self._cosine_similarity(hyp_embedding, chunk.embedding), chunk)
            for chunk in self.chunks
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        top_k = [chunk for _, chunk in scored[:k]]
        logger.info("HyDE retrieved %d chunks for query '%s'", len(top_k), query)
        return top_k
```

---

## Exercise 2: Multiple Hypothetical Docs — Average Embeddings

```python
import logging
import os
from dataclasses import dataclass, field

import anthropic

logger = logging.getLogger(__name__)


@dataclass
class HyDERetrieverMulti:
    chunks: list[Chunk]
    model: str = "claude-sonnet-5-5"
    num_hypothetical_docs: int = 3
    embed_fn: callable = None

    def __post_init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def _generate_hypothetical_answer(self, query: str, variation: int) -> str:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=256,
            temperature=0.8,  # higher temp for diversity
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Write hypothetical document #{variation + 1} that answers: {query}\n"
                        "Be concise and factual."
                    ),
                }
            ],
        )
        return msg.content[0].text.strip()

    @staticmethod
    def _average_embeddings(embeddings: list[list[float]]) -> list[float]:
        n = len(embeddings)
        dim = len(embeddings[0])
        return [sum(emb[i] for emb in embeddings) / n for i in range(dim)]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x**2 for x in a) ** 0.5
        norm_b = sum(x**2 for x in b) ** 0.5
        return dot / (norm_a * norm_b + 1e-9)

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        hypotheticals = [
            self._generate_hypothetical_answer(query, i)
            for i in range(self.num_hypothetical_docs)
        ]
        embeddings = [self.embed_fn(h) for h in hypotheticals]
        avg_embedding = self._average_embeddings(embeddings)

        scored = [
            (self._cosine_similarity(avg_embedding, chunk.embedding), chunk)
            for chunk in self.chunks
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        logger.info(
            "Multi-HyDE (%d docs) retrieved %d chunks", self.num_hypothetical_docs, k
        )
        return [chunk for _, chunk in scored[:k]]
```

---

## Exercise 3: Implement `ContextualChunkBuilder.build_context`

```python
import logging
import os
from dataclasses import dataclass

import anthropic

logger = logging.getLogger(__name__)


@dataclass
class ContextualChunkBuilder:
    """Prepend a short context prefix to each chunk at index time (Anthropic contextual retrieval)."""

    model: str = "claude-sonnet-5-5"

    def __post_init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def build_context(self, chunk_text: str, doc_summary: str) -> str:
        """Return `<context prefix>\n\n<original chunk_text>`."""
        prompt = (
            "Given the following document summary and a chunk from that document, "
            "write a 1-2 sentence context prefix that explains what this chunk is about "
            "and how it relates to the broader document. "
            "Return only the context prefix — no extra formatting.\n\n"
            f"Document summary:\n{doc_summary}\n\n"
            f"Chunk:\n{chunk_text}"
        )
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=128,
            messages=[{"role": "user", "content": prompt}],
        )
        context_prefix = msg.content[0].text.strip()
        contextualized = f"{context_prefix}\n\n{chunk_text}"
        logger.debug("Context prefix added (%d chars)", len(context_prefix))
        return contextualized
```

---

## Exercise 4: `build_context_batch` with Rate-Limit Delay

```python
import logging
import time
import os
from dataclasses import dataclass

import anthropic

logger = logging.getLogger(__name__)


@dataclass
class ContextualChunkBuilderBatch(ContextualChunkBuilder):

    def build_context_batch(
        self,
        chunks: list[str],
        doc_summary: str,
        rate_limit_delay: float = 0.1,
    ) -> list[str]:
        """Process all chunks sequentially with inter-call delay."""
        results: list[str] = []
        for i, chunk in enumerate(chunks):
            logger.info("Building context for chunk %d/%d", i + 1, len(chunks))
            contextualized = self.build_context(chunk, doc_summary)
            results.append(contextualized)
            if i < len(chunks) - 1:
                time.sleep(rate_limit_delay)
        logger.info("Batch complete: %d chunks contextualized", len(results))
        return results
```

---

## Exercise 5: Unit Tests — Mocking Anthropic + Embeddings

```python
# tests/unit/test_day2.py
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, call, patch


def _make_message(text: str) -> MagicMock:
    cb = MagicMock()
    cb.text = text
    m = MagicMock()
    m.content = [cb]
    return m


# ---------------------------------------------------------------------------
# HyDERetriever tests
# ---------------------------------------------------------------------------
class TestHyDERetriever(unittest.TestCase):
    @patch("anthropic.Anthropic")
    def test_retrieve_returns_k_chunks(self, mock_cls: MagicMock) -> None:
        from day2_solutions import Chunk, HyDERetriever

        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_message("hypothetical answer")

        def fake_embed(text: str) -> list[float]:
            return [1.0, 0.0, 0.0]

        chunks = [
            Chunk(id=str(i), text=f"chunk {i}", embedding=[1.0, 0.0, 0.0])
            for i in range(10)
        ]

        retriever = HyDERetriever(chunks=chunks, embed_fn=fake_embed)
        result = retriever.retrieve("test query", k=3)

        self.assertEqual(len(result), 3)

    @patch("anthropic.Anthropic")
    def test_hypothetical_answer_used_for_embedding(
        self, mock_cls: MagicMock
    ) -> None:
        from day2_solutions import Chunk, HyDERetriever

        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_message("hyp doc text")

        embedded_texts: list[str] = []

        def tracking_embed(text: str) -> list[float]:
            embedded_texts.append(text)
            return [1.0, 0.0]

        chunks = [Chunk(id="1", text="c", embedding=[1.0, 0.0])]
        retriever = HyDERetriever(chunks=chunks, embed_fn=tracking_embed)
        retriever.retrieve("query")

        self.assertIn("hyp doc text", embedded_texts)


# ---------------------------------------------------------------------------
# ContextualChunkBuilder tests
# ---------------------------------------------------------------------------
class TestContextualChunkBuilder(unittest.TestCase):
    @patch("anthropic.Anthropic")
    def test_build_context_prepends_prefix(self, mock_cls: MagicMock) -> None:
        from day2_solutions import ContextualChunkBuilder

        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_message(
            "This chunk discusses Python GC."
        )

        builder = ContextualChunkBuilder()
        result = builder.build_context("del obj; gc.collect()", "Python memory guide")

        self.assertTrue(result.startswith("This chunk discusses Python GC."))
        self.assertIn("del obj; gc.collect()", result)

    @patch("anthropic.Anthropic")
    def test_build_context_batch_length(self, mock_cls: MagicMock) -> None:
        from day2_solutions import ContextualChunkBuilderBatch

        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_message("prefix")

        builder = ContextualChunkBuilderBatch()
        results = builder.build_context_batch(
            ["chunk A", "chunk B", "chunk C"], "summary", rate_limit_delay=0.0
        )
        self.assertEqual(len(results), 3)

    @patch("time.sleep")
    @patch("anthropic.Anthropic")
    def test_rate_limit_delay_called(
        self, mock_cls: MagicMock, mock_sleep: MagicMock
    ) -> None:
        from day2_solutions import ContextualChunkBuilderBatch

        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_message("p")

        builder = ContextualChunkBuilderBatch()
        builder.build_context_batch(
            ["a", "b", "c"], "summary", rate_limit_delay=0.05
        )
        # sleep called between chunks: n-1 times
        self.assertEqual(mock_sleep.call_count, 2)


if __name__ == "__main__":
    unittest.main()
```
