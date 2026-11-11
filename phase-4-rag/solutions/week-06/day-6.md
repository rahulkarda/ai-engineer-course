# Day 6 Solutions — Citations, Grounding & Graph RAG

## Exercise 1: Implement `CitationExtractor.extract`

```python
import logging
import os
import re
from dataclasses import dataclass

import anthropic
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Chunk(BaseModel):
    id: str
    text: str
    metadata: dict = {}


class CitedAnswer(BaseModel):
    answer: str                    # answer text with inline [N] markers
    citations: list[dict]          # [{"index": N, "chunk_id": str, "excerpt": str}]
    sources: list[Chunk]           # full source chunks in citation order


@dataclass
class CitationExtractor:
    model: str = "claude-sonnet-5-5"

    def __post_init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def extract(self, answer_text: str, source_chunks: list[Chunk]) -> CitedAnswer:
        """
        Ask the LLM to rewrite answer_text with inline [N] citations
        referencing the provided source_chunks (1-indexed).
        """
        sources_block = "\n".join(
            f"[{i+1}] (id={c.id}): {c.text[:300]}"
            for i, c in enumerate(source_chunks)
        )
        prompt = (
            "Rewrite the answer below, inserting inline citations in the format [N] "
            "where N is the source number from the list provided. "
            "Only cite sources that directly support the claim. "
            "Return only the rewritten answer — no explanation.\n\n"
            f"Sources:\n{sources_block}\n\n"
            f"Answer:\n{answer_text}"
        )
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        cited_text = msg.content[0].text.strip()

        # Parse inline citation numbers from text
        citation_indices = sorted(
            set(int(m) for m in re.findall(r"\[(\d+)\]", cited_text))
        )
        citations: list[dict] = []
        for idx in citation_indices:
            if 1 <= idx <= len(source_chunks):
                chunk = source_chunks[idx - 1]
                citations.append(
                    {
                        "index": idx,
                        "chunk_id": chunk.id,
                        "excerpt": chunk.text[:150],
                    }
                )

        cited_sources = [
            source_chunks[c["index"] - 1] for c in citations
        ]
        logger.info(
            "CitationExtractor: %d citations in answer", len(citations)
        )
        return CitedAnswer(
            answer=cited_text, citations=citations, sources=cited_sources
        )
```

---

## Exercise 2: Implement `GroundingChecker.check`

```python
import logging
import os
from dataclasses import dataclass

import anthropic

logger = logging.getLogger(__name__)


@dataclass
class GroundingChecker:
    model: str = "claude-sonnet-5-5"

    def __post_init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def check(self, claim: str, chunks: list[Chunk]) -> bool:
        """Return True if the claim is supported by at least one chunk."""
        context = "\n\n".join(f"[{c.id}] {c.text}" for c in chunks)
        prompt = (
            "Does the following claim appear to be supported by the provided context?\n"
            "Answer with exactly one word: YES or NO.\n\n"
            f"Claim: {claim}\n\nContext:\n{context}"
        )
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=4,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        verdict = msg.content[0].text.strip().upper()
        grounded = verdict.startswith("YES")
        logger.debug("Grounding check: '%s...' → %s", claim[:60], grounded)
        return grounded
```

---

## Exercise 3: `faithfulness_score`

```python
import logging
import re

logger = logging.getLogger(__name__)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on common sentence-ending punctuation."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if len(s.strip()) > 10]


def faithfulness_score(
    answer: str,
    chunks: list[Chunk],
    checker: GroundingChecker | None = None,
) -> float:
    """
    Fraction of answer sentences that are grounded in at least one chunk.
    Returns 0.0 if answer has no sentences.
    """
    if checker is None:
        checker = GroundingChecker()

    sentences = split_sentences(answer)
    if not sentences:
        return 0.0

    grounded_count = sum(
        1 for sent in sentences if checker.check(sent, chunks)
    )
    score = grounded_count / len(sentences)
    logger.info(
        "Faithfulness: %d/%d sentences grounded → %.2f",
        grounded_count,
        len(sentences),
        score,
    )
    return score
```

---

## Exercise 4: `GraphRetriever.local_query`

```python
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GraphRetriever:
    """
    Simplified graph retriever over an adjacency list.
    Node values are text passages; edges represent entity co-occurrence or links.
    """

    # adjacency: {entity_id: [neighbor_entity_id, ...]}
    adjacency: dict[str, list[str]] = field(default_factory=dict)
    # node_texts: {entity_id: passage_text}
    node_texts: dict[str, str] = field(default_factory=dict)

    def _entity_match(self, query: str, entity_id: str) -> bool:
        """Simple substring match — replace with embedding similarity in production."""
        return query.lower() in entity_id.lower() or entity_id.lower() in query.lower()

    def local_query(self, query: str, k: int = 5) -> list[str]:
        """
        Find seed nodes matching query, then traverse 1-hop neighbors.
        Return up to k passage texts.
        """
        seed_ids: list[str] = [
            eid for eid in self.node_texts
            if self._entity_match(query, eid)
        ]

        candidate_ids: set[str] = set(seed_ids)
        for seed in seed_ids:
            neighbors = self.adjacency.get(seed, [])
            candidate_ids.update(neighbors)

        results: list[str] = []
        for eid in candidate_ids:
            if eid in self.node_texts:
                results.append(self.node_texts[eid])
            if len(results) >= k:
                break

        logger.info(
            "GraphRetriever: %d seeds, %d candidates, %d results",
            len(seed_ids),
            len(candidate_ids),
            len(results),
        )
        return results
```

---

## Exercise 5: Wire `CitationExtractor` into `AgenticRAGAgent`

```python
import logging
import os
from dataclasses import dataclass
from typing import Any

import anthropic

logger = logging.getLogger(__name__)


@dataclass
class AgenticRAGAgentWithCitations:
    """AgenticRAGAgent extended to run every answer through CitationExtractor."""

    retriever_fn: Any
    model: str = "claude-sonnet-5-5"
    max_iterations: int = 5

    def __post_init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self._extractor = CitationExtractor(model=self.model)

    def run(self, user_query: str) -> CitedAnswer:
        # Re-use base agent logic — inline for clarity
        from day5_solutions import AgenticRAGAgent

        base_agent = AgenticRAGAgent(
            retriever_fn=self.retriever_fn,
            model=self.model,
            max_iterations=self.max_iterations,
        )
        result = base_agent.run(user_query)

        # Post-process: extract citations
        cited = self._extractor.extract(result.answer, result.sources)
        logger.info(
            "Citations extracted: %d references in final answer",
            len(cited.citations),
        )
        return cited
```

---

## Exercise 6: End-to-End Test — Mock Retriever + Mock LLM → `CitedAnswer` with Citation

```python
# tests/unit/test_citations_e2e.py
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


def _make_tool_use_response(tool_id: str, tool_input: dict) -> MagicMock:
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = tool_id
    tool_block.name = "retrieve"
    tool_block.input = tool_input
    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [tool_block]
    return response


def _make_end_turn_response(text: str) -> MagicMock:
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [text_block]
    return response


class FakeChunk(BaseModel):
    id: str
    text: str
    metadata: dict = {}


class TestCitedAnswerE2E(unittest.TestCase):

    @patch("anthropic.Anthropic")
    def test_run_returns_cited_answer_with_citation(
        self, mock_cls: MagicMock
    ) -> None:
        from day6_solutions import AgenticRAGAgentWithCitations, CitedAnswer

        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        # First call: agent decides to retrieve
        # Second call: agent generates raw answer
        # Third call: CitationExtractor rewrites with citations
        mock_client.messages.create.side_effect = [
            _make_tool_use_response(
                "t1", {"query": "GIL", "source": "vector_store"}
            ),
            _make_end_turn_response("The GIL prevents true parallelism."),
            _make_message(
                "The GIL prevents true parallelism [1]. It is a mutex [1]."
            ),
        ]

        retrieved_chunks = [
            FakeChunk(id="c1", text="GIL is a mutex protecting CPython internals.")
        ]

        def fake_retriever(query: str, source: str, k: int) -> list[FakeChunk]:
            return retrieved_chunks

        agent = AgenticRAGAgentWithCitations(retriever_fn=fake_retriever)
        result = agent.run("What is the Python GIL?")

        self.assertIsInstance(result, CitedAnswer)
        self.assertGreaterEqual(len(result.citations), 1)
        self.assertIn("[1]", result.answer)
        self.assertEqual(result.citations[0]["chunk_id"], "c1")

    @patch("anthropic.Anthropic")
    def test_cited_answer_sources_not_empty(self, mock_cls: MagicMock) -> None:
        from day6_solutions import AgenticRAGAgentWithCitations

        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_tool_use_response(
                "t2", {"query": "something", "source": "vector_store"}
            ),
            _make_end_turn_response("Answer based on retrieved content."),
            _make_message("Answer based on retrieved content [1]."),
        ]

        chunks = [FakeChunk(id="s1", text="source text")]
        agent = AgenticRAGAgentWithCitations(
            retriever_fn=lambda q, s, k: chunks
        )
        result = agent.run("question")

        self.assertGreater(len(result.sources), 0)


if __name__ == "__main__":
    unittest.main()
```
