# Day 4 Solutions — Self-RAG & Corrective RAG

## Exercise 1: Implement `RelevanceScorer.score`

```python
import logging
import os
from dataclasses import dataclass

import anthropic

logger = logging.getLogger(__name__)


@dataclass
class RelevanceScorer:
    model: str = "claude-sonnet-5-5"
    temperature: float = 0.0

    def __post_init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def score(self, query: str, chunk_text: str) -> float:
        """Return relevance score in [0.0, 1.0] via LLM judge."""
        prompt = (
            "Rate the relevance of the following text passage to the given query.\n"
            "Respond with ONLY a decimal number between 0.0 (completely irrelevant) "
            "and 1.0 (perfectly relevant). No explanation.\n\n"
            f"Query: {query}\n\nPassage:\n{chunk_text}"
        )
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=8,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        try:
            score = float(raw)
        except ValueError:
            logger.warning("Could not parse score '%s', defaulting to 0.0", raw)
            score = 0.0
        score = max(0.0, min(1.0, score))
        logger.debug("Relevance score for chunk: %.2f", score)
        return score
```

---

## Exercise 2: Implement `SelfRAGPipeline.generate`

```python
import logging
import os
from dataclasses import dataclass, field
from pydantic import BaseModel

import anthropic

logger = logging.getLogger(__name__)


class Chunk(BaseModel):
    id: str
    text: str
    metadata: dict = {}


@dataclass
class SelfRAGPipeline:
    scorer: RelevanceScorer
    model: str = "claude-sonnet-5-5"
    relevance_threshold: float = 0.5

    def __post_init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def generate(self, query: str, chunks: list[Chunk]) -> str:
        """Filter chunks by relevance then generate answer from surviving chunks."""
        relevant: list[Chunk] = []
        for chunk in chunks:
            score = self.scorer.score(query, chunk.text)
            if score >= self.relevance_threshold:
                relevant.append(chunk)
                logger.debug("Chunk %s accepted (score=%.2f)", chunk.id, score)
            else:
                logger.debug("Chunk %s rejected (score=%.2f)", chunk.id, score)

        if not relevant:
            logger.warning("No relevant chunks found for query: %s", query)
            return "I could not find relevant information to answer this question."

        context = "\n\n".join(f"[{c.id}] {c.text}" for c in relevant)
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Answer the question using only the provided context.\n\n"
                        f"Context:\n{context}\n\n"
                        f"Question: {query}"
                    ),
                }
            ],
        )
        answer = msg.content[0].text.strip()
        logger.info(
            "SelfRAG generated answer from %d/%d relevant chunks",
            len(relevant),
            len(chunks),
        )
        return answer
```

---

## Exercise 3: Implement `CorrectionRouter.route`

```python
import logging
import os
from dataclasses import dataclass
from typing import Literal

import anthropic

logger = logging.getLogger(__name__)


@dataclass
class CorrectionRouter:
    model: str = "claude-sonnet-5-5"
    low_threshold: float = 0.4
    high_threshold: float = 0.75

    def __post_init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def route(
        self, query: str, confidence: float
    ) -> Literal["correct", "incorrect", "ambiguous"]:
        """Route based on confidence score, with LLM-assisted boundary resolution."""
        if confidence >= self.high_threshold:
            logger.debug("Route=correct (conf=%.2f)", confidence)
            return "correct"
        if confidence <= self.low_threshold:
            logger.debug("Route=incorrect (conf=%.2f)", confidence)
            return "incorrect"

        # Ambiguous range: ask LLM
        prompt = (
            f"A retrieval system has confidence score {confidence:.2f} (range 0-1) "
            f"for the query: '{query}'\n"
            "Classify the retrieval as: correct, incorrect, or ambiguous.\n"
            "Return only one of those three words."
        )
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=8,
            messages=[{"role": "user", "content": prompt}],
        )
        verdict = msg.content[0].text.strip().lower()
        if verdict not in {"correct", "incorrect", "ambiguous"}:
            verdict = "ambiguous"
        logger.info("CorrectionRouter verdict: %s", verdict)
        return verdict  # type: ignore[return-value]
```

---

## Exercise 4: Web Fallback on `incorrect` Route

```python
import logging
import os
from dataclasses import dataclass

import anthropic

logger = logging.getLogger(__name__)


def web_search_stub(query: str) -> list[Chunk]:
    """Stub: replace with real web search integration (e.g. Tavily, SerpAPI)."""
    logger.info("Web search stub called for: %s", query)
    return [
        Chunk(
            id="web-1",
            text=f"Web result placeholder for: {query}",
            metadata={"source": "web"},
        )
    ]


@dataclass
class CorrectiveRAGPipeline:
    scorer: RelevanceScorer
    router: CorrectionRouter
    model: str = "claude-sonnet-5-5"
    relevance_threshold: float = 0.5

    def __post_init__(self) -> None:
        self._pipeline = SelfRAGPipeline(
            scorer=self.scorer,
            model=self.model,
            relevance_threshold=self.relevance_threshold,
        )
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def _estimate_confidence(self, query: str, chunks: list[Chunk]) -> float:
        if not chunks:
            return 0.0
        scores = [self.scorer.score(query, c.text) for c in chunks]
        return sum(scores) / len(scores)

    def generate(self, query: str, chunks: list[Chunk]) -> str:
        confidence = self._estimate_confidence(query, chunks)
        route = self.router.route(query, confidence)
        logger.info("Route=%s for query='%s'", route, query)

        if route == "incorrect":
            logger.info("Falling back to web search")
            web_chunks = web_search_stub(query)
            chunks = web_chunks  # replace with web results

        return self._pipeline.generate(query, chunks)
```

---

## Exercise 5: Tests for SelfRAGPipeline

```python
# tests/unit/test_self_rag.py
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from pydantic import BaseModel


def _make_message(text: str) -> MagicMock:
    cb = MagicMock()
    cb.text = text
    m = MagicMock()
    m.content = [cb]
    return m


class FakeChunk(BaseModel):
    id: str
    text: str
    metadata: dict = {}


class TestSelfRAGPipeline(unittest.TestCase):

    @patch("anthropic.Anthropic")
    def test_filters_low_relevance_chunks(self, mock_cls: MagicMock) -> None:
        from day4_solutions import RelevanceScorer, SelfRAGPipeline

        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_message("Generated answer.")

        mock_scorer = MagicMock(spec=RelevanceScorer)
        # chunk-a: high score, chunk-b: low score
        mock_scorer.score.side_effect = lambda q, text: (
            0.9 if "relevant" in text else 0.3
        )

        pipeline = SelfRAGPipeline(scorer=mock_scorer, relevance_threshold=0.5)
        chunks = [
            FakeChunk(id="a", text="relevant content here"),
            FakeChunk(id="b", text="unrelated noise"),
        ]
        result = pipeline.generate("test query", chunks)

        self.assertEqual(result, "Generated answer.")

        # Verify only the relevant chunk made it into the prompt
        call_args = mock_client.messages.create.call_args[1]
        prompt_content: str = call_args["messages"][0]["content"]
        self.assertIn("[a]", prompt_content)
        self.assertNotIn("[b]", prompt_content)

    @patch("anthropic.Anthropic")
    def test_no_relevant_chunks_returns_fallback(self, mock_cls: MagicMock) -> None:
        from day4_solutions import RelevanceScorer, SelfRAGPipeline

        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_scorer = MagicMock(spec=RelevanceScorer)
        mock_scorer.score.return_value = 0.1  # all below threshold

        pipeline = SelfRAGPipeline(scorer=mock_scorer, relevance_threshold=0.5)
        result = pipeline.generate(
            "query", [FakeChunk(id="1", text="irrelevant")]
        )

        self.assertIn("could not find", result.lower())
        mock_client.messages.create.assert_not_called()

    @patch("anthropic.Anthropic")
    def test_threshold_boundary(self, mock_cls: MagicMock) -> None:
        from day4_solutions import RelevanceScorer, SelfRAGPipeline

        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_message("answer")

        mock_scorer = MagicMock(spec=RelevanceScorer)
        mock_scorer.score.return_value = 0.5  # exactly at threshold

        pipeline = SelfRAGPipeline(scorer=mock_scorer, relevance_threshold=0.5)
        result = pipeline.generate("q", [FakeChunk(id="1", text="text")])

        # 0.5 >= 0.5 → accepted
        mock_client.messages.create.assert_called_once()
        self.assertEqual(result, "answer")


if __name__ == "__main__":
    unittest.main()
```
