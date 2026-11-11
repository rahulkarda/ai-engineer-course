# Assessment Answer Key — Week 06

## Conceptual Questions

**Q1. Why does HyDE improve retrieval over direct query embedding?**
Queries are short and semantically sparse; document passages are dense. Embedding a hypothetical answer that looks like a document puts the query vector in the same semantic neighborhood as actual document embeddings, reducing the distribution mismatch. The hypothesis doesn't need to be accurate — it just needs to be stylistically similar to the target passages.

**Q2. What is the key difference between Self-RAG and Corrective RAG?**
Self-RAG filters retrieved chunks per-generation using a relevance judge trained as part of the model itself (or an LLM judge call), discarding irrelevant context before generating. Corrective RAG evaluates the overall retrieval quality post-retrieval and routes to an alternative source (web search) if the primary retrieval is deemed insufficient. Self-RAG acts on chunk granularity; Corrective RAG acts on pipeline routing.

**Q3. Why is contextual prefixing done at index time rather than query time?**
Index-time prefixing pays the LLM cost once per chunk, not once per query. A document with 1,000 chunks queried 10,000 times would require 10M LLM calls at query time vs. 1,000 at index time. The prefix is baked into the stored embedding, improving retrieval without adding query-time latency.

**Q4. What stops the agentic tool-use loop from running forever?**
Three safeguards: (1) `stop_reason == "end_turn"` — the model decides it has enough information; (2) `max_iterations` counter — the harness forces a final generation after N tool calls; (3) SufficiencyChecker confidence threshold — multi-hop retrieval exits early when confidence > threshold, preventing unnecessary hops.

**Q5. How does Step-Back prompting complement query decomposition?**
Decomposition splits a complex question into narrow sub-questions (breadth). Step-Back generalizes a specific question to its underlying principle (depth/abstraction). Used together: decompose for multi-fact questions, step back to retrieve foundational context that underpins all sub-answers.

**Q6. What is the purpose of the `SufficiencyChecker` confidence score vs the boolean `sufficient` flag?**
The boolean handles the clear-cut case (enough context / not enough). The confidence float allows the retriever to stop even when the model says "not technically sufficient" but is very close (e.g., confidence=0.90 with `sufficient=False`). It prevents diminishing-returns hops where the 4th hop adds <5% information but costs 2 API calls.

**Q7. Why mock all LLM calls in unit tests?**
LLM calls are non-deterministic, slow, costly, and require network + API key. Unit tests must be fast, reproducible, and runnable in CI without secrets. Mocking replaces the call with a deterministic stub, testing the surrounding logic (parsing, routing, state transitions) without coupling test correctness to model behavior.

**Q8. What does faithfulness score measure, and why is it different from relevance?**
Faithfulness measures whether the generated answer is supported by the retrieved context (hallucination detection). Relevance measures whether the retrieved chunks match the query. High relevance with low faithfulness means the model ignored the context and hallucinated. Both dimensions are required for a trustworthy RAG system.

---

## Coding Challenge: Full Implementation

### `CitedAnswer` Pydantic Model

```python
from pydantic import BaseModel, Field


class Citation(BaseModel):
    index: int = Field(ge=1, description="1-based citation index as it appears in answer text")
    chunk_id: str
    excerpt: str = Field(max_length=300)


class CitedAnswer(BaseModel):
    answer: str = Field(description="Answer text with inline [N] citation markers")
    citations: list[Citation]
    sources: list["Chunk"]

    model_config = {"arbitrary_types_allowed": True}
```

### `AgenticRAGSystem` with `run()`

```python
import logging
import os
from dataclasses import dataclass
from typing import Any

import anthropic
from pydantic import BaseModel

logger = logging.getLogger(__name__)

RETRIEVE_TOOL: dict = {
    "name": "retrieve",
    "description": "Retrieve relevant document chunks from the knowledge base.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "source": {
                "type": "string",
                "enum": ["vector_store", "web", "sql"],
            },
            "k": {"type": "integer", "default": 5},
        },
        "required": ["query", "source"],
    },
}


class Chunk(BaseModel):
    id: str
    text: str
    metadata: dict = {}


@dataclass
class AgenticRAGSystem:
    retriever_fn: Any  # (query: str, source: str, k: int) -> list[Chunk]
    model: str = "claude-sonnet-5-5"
    max_iterations: int = 5

    def __post_init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def _process_tool(self, tool_input: dict) -> list[Chunk]:
        return self.retriever_fn(
            tool_input["query"],
            tool_input.get("source", "vector_store"),
            tool_input.get("k", 5),
        )

    def _add_citations(self, raw_answer: str, sources: list[Chunk]) -> CitedAnswer:
        if not sources:
            return CitedAnswer(answer=raw_answer, citations=[], sources=[])

        sources_text = "\n".join(
            f"[{i+1}] {c.text[:200]}" for i, c in enumerate(sources)
        )
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Rewrite with inline [N] citations.\nSources:\n{sources_text}"
                        f"\n\nAnswer:\n{raw_answer}"
                    ),
                }
            ],
        )
        import re

        cited_text = msg.content[0].text.strip()
        indices = sorted(set(int(m) for m in re.findall(r"\[(\d+)\]", cited_text)))
        citations = [
            Citation(
                index=idx,
                chunk_id=sources[idx - 1].id,
                excerpt=sources[idx - 1].text[:150],
            )
            for idx in indices
            if 1 <= idx <= len(sources)
        ]
        return CitedAnswer(
            answer=cited_text,
            citations=citations,
            sources=[sources[c.index - 1] for c in citations],
        )

    def run(self, user_query: str) -> CitedAnswer:
        messages: list[dict] = [{"role": "user", "content": user_query}]
        all_sources: list[Chunk] = []

        for iteration in range(self.max_iterations):
            response = self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                tools=[RETRIEVE_TOOL],
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                raw_answer = next(
                    (b.text for b in response.content if hasattr(b, "text")), ""
                )
                return self._add_citations(raw_answer, all_sources)

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        chunks = self._process_tool(block.input)
                        all_sources.extend(chunks)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": "\n".join(
                                    f"[{c.id}] {c.text}" for c in chunks
                                ),
                            }
                        )
                messages.append({"role": "user", "content": tool_results})

        # Forced final answer
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=messages
            + [{"role": "user", "content": "Provide your best answer now."}],
        )
        raw_answer = resp.content[0].text.strip()
        return self._add_citations(raw_answer, all_sources)
```

### Test File — Mocked LLM Calls

```python
# tests/unit/test_assessment.py
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from pydantic import BaseModel


def _make_tool_use_response(tool_id: str, query: str) -> MagicMock:
    tb = MagicMock()
    tb.type = "tool_use"
    tb.id = tool_id
    tb.name = "retrieve"
    tb.input = {"query": query, "source": "vector_store", "k": 5}
    r = MagicMock()
    r.stop_reason = "tool_use"
    r.content = [tb]
    return r


def _make_end_turn(text: str) -> MagicMock:
    tb = MagicMock()
    tb.type = "text"
    tb.text = text
    r = MagicMock()
    r.stop_reason = "end_turn"
    r.content = [tb]
    return r


def _make_citation_response(text: str) -> MagicMock:
    cb = MagicMock()
    cb.text = text
    m = MagicMock()
    m.content = [cb]
    return m


class FakeChunk(BaseModel):
    id: str
    text: str
    metadata: dict = {}


class TestAgenticRAGSystem(unittest.TestCase):

    @patch("anthropic.Anthropic")
    def test_run_returns_cited_answer(self, mock_cls: MagicMock) -> None:
        from assessment_solution import AgenticRAGSystem, CitedAnswer

        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_client.messages.create.side_effect = [
            _make_tool_use_response("t1", "Python GIL"),
            _make_end_turn("The GIL is a mutex protecting CPython memory."),
            _make_citation_response(
                "The GIL is a mutex [1] protecting CPython memory [1]."
            ),
        ]

        chunks = [FakeChunk(id="c1", text="GIL protects CPython internals.")]
        system = AgenticRAGSystem(retriever_fn=lambda q, s, k: chunks)
        result = system.run("What is the Python GIL?")

        self.assertIsInstance(result, CitedAnswer)
        self.assertGreaterEqual(len(result.citations), 1)
        self.assertIn("[1]", result.answer)

    @patch("anthropic.Anthropic")
    def test_run_direct_answer_no_tool(self, mock_cls: MagicMock) -> None:
        from assessment_solution import AgenticRAGSystem, CitedAnswer

        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_end_turn("Simple answer with no retrieval needed."),
            _make_citation_response("Simple answer with no retrieval needed."),
        ]

        system = AgenticRAGSystem(retriever_fn=lambda q, s, k: [])
        result = system.run("Easy question")

        self.assertIsInstance(result, CitedAnswer)

    @patch("anthropic.Anthropic")
    def test_max_iterations_respected(self, mock_cls: MagicMock) -> None:
        from assessment_solution import AgenticRAGSystem

        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        tool_resp = _make_tool_use_response("t", "q")
        final = _make_citation_response("final")
        mock_client.messages.create.side_effect = [
            tool_resp,
            tool_resp,
            tool_resp,
            _make_end_turn("forced"),  # forced stop
            final,
        ]

        system = AgenticRAGSystem(
            retriever_fn=lambda q, s, k: [], max_iterations=3
        )
        result = system.run("looping query")
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
```

---

## Grading Guidance

| Dimension | Full Marks | Partial | Zero |
|---|---|---|---|
| **Pydantic Models** | All fields typed, validators present, nested models correct | Missing validator or optional type hint | Untyped dicts used instead |
| **Tool-Use Loop** | Handles end_turn, tool_use, max_iterations, forced stop | Missing one branch | Single-shot only |
| **HyDE** | Multi-doc averaging, embedding delegation injected | Single doc only | Direct query embedding |
| **Self-RAG Filtering** | Per-chunk scoring with configurable threshold | Fixed filter | No filtering |
| **Citations** | Inline [N] markers, parsed citation list, source mapping | Markers present but no parsed list | No citations |
| **Test Coverage** | All LLM calls mocked, edge cases covered (empty chunks, threshold boundary) | Mocked but no edge cases | Real API calls in tests |
| **Logging** | `logger.info/debug` throughout, no `print` | Partial logging | `print` statements only |
| **Error Handling** | JSON parse fallback, score clamp to [0,1], max_iter guard | One guard missing | No guards |
