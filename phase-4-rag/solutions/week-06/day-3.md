# Day 3 Solutions — Multi-hop & Iterative Retrieval

## Exercise 1: Implement `MultiHopRetriever.retrieve`

```python
import logging
import os
from dataclasses import dataclass, field
from typing import Protocol

import anthropic
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Chunk(BaseModel):
    id: str
    text: str
    metadata: dict = {}


class BaseRetriever(Protocol):
    def retrieve(self, query: str, k: int = 5) -> list[Chunk]: ...


class MultiHopResult(BaseModel):
    final_answer: str
    hops: int
    queries_used: list[str]
    all_chunks: list[Chunk]


@dataclass
class MultiHopRetriever:
    retriever: BaseRetriever
    model: str = "claude-sonnet-5-5"
    max_hops: int = 3

    def __post_init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def _build_context(self, chunks: list[Chunk]) -> str:
        return "\n\n".join(f"[{c.id}] {c.text}" for c in chunks)

    def _generate_answer(self, query: str, context: str) -> str:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Answer the question based on the context below.\n\n"
                        f"Context:\n{context}\n\n"
                        f"Question: {query}"
                    ),
                }
            ],
        )
        return msg.content[0].text.strip()

    def retrieve(self, initial_query: str) -> MultiHopResult:
        current_query = initial_query
        all_chunks: list[Chunk] = []
        queries_used: list[str] = []

        for hop in range(self.max_hops):
            logger.info("Hop %d: query='%s'", hop + 1, current_query)
            queries_used.append(current_query)

            new_chunks = self.retriever.retrieve(current_query)
            all_chunks.extend(new_chunks)

            context = self._build_context(all_chunks)
            answer = self._generate_answer(initial_query, context)

            # Use SufficiencyChecker (see exercise 2) to decide if we should stop
            sufficiency = SufficiencyChecker().check(initial_query, context)
            logger.debug(
                "Hop %d sufficiency: %s (confidence=%.2f)",
                hop + 1,
                sufficiency.sufficient,
                sufficiency.confidence,
            )

            if sufficiency.sufficient:
                logger.info("Stopping at hop %d: sufficient context", hop + 1)
                return MultiHopResult(
                    final_answer=answer,
                    hops=hop + 1,
                    queries_used=queries_used,
                    all_chunks=all_chunks,
                )

            # Refine query for next hop
            if hop < self.max_hops - 1:
                current_query = query_refiner(initial_query, context)

        # Exhausted hops — return best available answer
        context = self._build_context(all_chunks)
        final_answer = self._generate_answer(initial_query, context)
        return MultiHopResult(
            final_answer=final_answer,
            hops=self.max_hops,
            queries_used=queries_used,
            all_chunks=all_chunks,
        )
```

---

## Exercise 2: `SufficiencyChecker` with Pydantic Result

```python
import logging
import os
import json
from dataclasses import dataclass

import anthropic
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SufficiencyResult(BaseModel):
    sufficient: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


@dataclass
class SufficiencyChecker:
    model: str = "claude-sonnet-5-5"

    def __post_init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def check(self, query: str, context: str) -> SufficiencyResult:
        prompt = (
            "Assess whether the provided context is sufficient to answer the question.\n"
            "Return JSON with keys: sufficient (bool), confidence (float 0-1), reasoning (str).\n\n"
            f"Question: {query}\n\nContext:\n{context}"
        )
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        data = json.loads(raw)
        result = SufficiencyResult(**data)
        logger.debug("Sufficiency: %s (conf=%.2f)", result.sufficient, result.confidence)
        return result
```

---

## Exercise 3: `query_refiner`

```python
import logging
import os

import anthropic

logger = logging.getLogger(__name__)


def query_refiner(original_query: str, context_so_far: str) -> str:
    """Generate a follow-up query to fill gaps in context_so_far."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    prompt = (
        "You are helping a retrieval system find missing information.\n"
        "Given the original question and the context retrieved so far, "
        "generate ONE follow-up search query to find the missing information.\n"
        "Return only the follow-up query — no explanation.\n\n"
        f"Original question: {original_query}\n\n"
        f"Context so far:\n{context_so_far}"
    )

    msg = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=128,
        messages=[{"role": "user", "content": prompt}],
    )
    refined = msg.content[0].text.strip()
    logger.info("Refined query: '%s'", refined)
    return refined
```

---

## Exercise 4: Confidence Threshold Early Stopping

Already integrated in Exercise 1 via `SufficiencyChecker`. The threshold logic:

```python
# In MultiHopRetriever.retrieve, after sufficiency check:
CONFIDENCE_THRESHOLD = 0.85

if sufficiency.sufficient or sufficiency.confidence > CONFIDENCE_THRESHOLD:
    logger.info(
        "Early stop at hop %d: confidence=%.2f (threshold=%.2f)",
        hop + 1,
        sufficiency.confidence,
        CONFIDENCE_THRESHOLD,
    )
    return MultiHopResult(
        final_answer=answer,
        hops=hop + 1,
        queries_used=queries_used,
        all_chunks=all_chunks,
    )
```

Stand-alone version showing the threshold as a configurable parameter:

```python
@dataclass
class MultiHopRetriever:
    retriever: BaseRetriever
    model: str = "claude-sonnet-5-5"
    max_hops: int = 3
    confidence_threshold: float = 0.85  # stop early if confidence exceeds this

    def retrieve(self, initial_query: str) -> MultiHopResult:
        ...
        for hop in range(self.max_hops):
            ...
            sufficiency = SufficiencyChecker().check(initial_query, context)
            if sufficiency.sufficient or sufficiency.confidence > self.confidence_threshold:
                return MultiHopResult(...)
            ...
```

---

## Exercise 5: Tests for MultiHopRetriever

```python
# tests/unit/test_multi_hop.py
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


class FakeRetriever:
    def __init__(self, chunks: list[FakeChunk]) -> None:
        self._chunks = chunks
        self.call_count = 0

    def retrieve(self, query: str, k: int = 5) -> list[FakeChunk]:
        self.call_count += 1
        return self._chunks


class TestMultiHopRetriever(unittest.TestCase):

    @patch("anthropic.Anthropic")
    def test_stops_when_sufficient(self, mock_cls: MagicMock) -> None:
        from day3_solutions import MultiHopResult, MultiHopRetriever, SufficiencyResult

        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        # LLM answer call
        mock_client.messages.create.return_value = _make_message("Final answer here.")

        chunks = [FakeChunk(id="1", text="relevant info")]
        fake_retriever = FakeRetriever(chunks)

        mock_sufficiency = MagicMock()
        mock_sufficiency.check.return_value = SufficiencyResult(
            sufficient=True, confidence=0.95, reasoning="enough info"
        )

        retriever = MultiHopRetriever(retriever=fake_retriever, max_hops=5)

        with patch.object(retriever, "_get_sufficiency_checker", return_value=mock_sufficiency):
            # Manually patch SufficiencyChecker inside module
            pass

        # Direct test: only 1 hop used when sufficient immediately
        # We patch SufficiencyChecker at module level
        with patch("day3_solutions.SufficiencyChecker") as mock_sc_cls:
            mock_sc_cls.return_value.check.return_value = SufficiencyResult(
                sufficient=True, confidence=0.95, reasoning="enough"
            )
            result = retriever.retrieve("What is X?")

        self.assertEqual(result.hops, 1)
        self.assertIsInstance(result.final_answer, str)
        self.assertEqual(fake_retriever.call_count, 1)

    @patch("anthropic.Anthropic")
    def test_max_hops_respected(self, mock_cls: MagicMock) -> None:
        from day3_solutions import MultiHopRetriever, SufficiencyResult

        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_message("answer")

        fake_retriever = FakeRetriever([FakeChunk(id="1", text="partial")])

        with patch("day3_solutions.SufficiencyChecker") as mock_sc_cls:
            with patch("day3_solutions.query_refiner", return_value="follow-up"):
                mock_sc_cls.return_value.check.return_value = SufficiencyResult(
                    sufficient=False, confidence=0.4, reasoning="insufficient"
                )
                retriever = MultiHopRetriever(retriever=fake_retriever, max_hops=3)
                result = retriever.retrieve("complex question")

        self.assertEqual(result.hops, 3)
        self.assertEqual(fake_retriever.call_count, 3)

    @patch("anthropic.Anthropic")
    def test_confidence_threshold_triggers_early_stop(
        self, mock_cls: MagicMock
    ) -> None:
        from day3_solutions import MultiHopRetriever, SufficiencyResult

        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_message("answer")

        fake_retriever = FakeRetriever([FakeChunk(id="1", text="info")])

        with patch("day3_solutions.SufficiencyChecker") as mock_sc_cls:
            mock_sc_cls.return_value.check.return_value = SufficiencyResult(
                sufficient=False,
                confidence=0.90,  # above 0.85 threshold
                reasoning="high confidence even if not technically sufficient",
            )
            retriever = MultiHopRetriever(
                retriever=fake_retriever, max_hops=5, confidence_threshold=0.85
            )
            result = retriever.retrieve("query")

        self.assertEqual(result.hops, 1)


if __name__ == "__main__":
    unittest.main()
```
