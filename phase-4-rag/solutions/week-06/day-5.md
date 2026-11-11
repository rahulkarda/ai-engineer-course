# Day 5 Solutions — Agentic RAG Tool Use

## Exercise 1: Complete `RETRIEVE_TOOL` Definition

```python
# Matches Anthropic tool JSON schema format exactly.

RETRIEVE_TOOL: dict = {
    "name": "retrieve",
    "description": (
        "Retrieve relevant document chunks from a knowledge base. "
        "Use this tool whenever you need factual information to answer a question. "
        "You may call it multiple times with different queries to gather complete information."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to retrieve relevant documents for.",
            },
            "source": {
                "type": "string",
                "enum": ["vector_store", "web", "sql"],
                "description": (
                    "The source to retrieve from. "
                    "'vector_store' for internal documents, "
                    "'web' for live internet search, "
                    "'sql' for structured database queries."
                ),
            },
            "k": {
                "type": "integer",
                "description": "Number of results to return (default 5, max 20).",
                "default": 5,
            },
        },
        "required": ["query", "source"],
    },
}
```

---

## Exercise 2: Implement `AgenticRAGAgent.run` — Full Tool-Use Loop

```python
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import anthropic
from pydantic import BaseModel

logger = logging.getLogger(__name__)

RETRIEVE_TOOL: dict = { ... }  # as defined in Exercise 1


class Chunk(BaseModel):
    id: str
    text: str
    metadata: dict = {}


class AgentResult(BaseModel):
    answer: str
    sources: list[Chunk]
    iterations: int


@dataclass
class AgenticRAGAgent:
    retriever_fn: Any  # callable(query, source, k) -> list[Chunk]
    model: str = "claude-sonnet-5-5"
    max_iterations: int = 5

    def __post_init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def _process_tool_call(
        self, tool_name: str, tool_input: dict
    ) -> list[Chunk]:
        if tool_name != "retrieve":
            raise ValueError(f"Unknown tool: {tool_name}")
        query = tool_input["query"]
        source = tool_input.get("source", "vector_store")
        k = tool_input.get("k", 5)
        chunks = self.retriever_fn(query, source, k)
        logger.info(
            "Tool retrieve: query='%s' source=%s → %d chunks", query, source, len(chunks)
        )
        return chunks

    def _chunks_to_tool_result(self, chunks: list[Chunk]) -> str:
        if not chunks:
            return "No relevant documents found."
        return "\n\n".join(f"[{c.id}] {c.text}" for c in chunks)

    def run(self, user_query: str) -> AgentResult:
        messages: list[dict] = [{"role": "user", "content": user_query}]
        all_sources: list[Chunk] = []
        iterations = 0

        while iterations < self.max_iterations:
            iterations += 1
            logger.debug("Agentic RAG iteration %d", iterations)

            response = self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                tools=[RETRIEVE_TOOL],
                messages=messages,
            )

            # Append assistant response to history
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                # Extract text answer
                answer = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        answer = block.text.strip()
                        break
                logger.info(
                    "AgenticRAG finished in %d iteration(s)", iterations
                )
                return AgentResult(
                    answer=answer, sources=all_sources, iterations=iterations
                )

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        chunks = self._process_tool_call(block.name, block.input)
                        all_sources.extend(chunks)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": self._chunks_to_tool_result(chunks),
                            }
                        )
                messages.append({"role": "user", "content": tool_results})
            else:
                logger.warning(
                    "Unexpected stop_reason: %s", response.stop_reason
                )
                break

        # Forced stop — generate answer from what we have
        logger.warning("Max iterations (%d) reached", self.max_iterations)
        messages.append(
            {
                "role": "user",
                "content": "Please provide your best answer based on the information retrieved so far.",
            }
        )
        final_response = self._client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=messages,
        )
        answer = final_response.content[0].text.strip()
        return AgentResult(
            answer=answer, sources=all_sources, iterations=iterations
        )
```

---

## Exercise 3: `classify_source` — Keyword Router

```python
from typing import Literal


def classify_source(query: str) -> Literal["vector_store", "web", "sql"]:
    """Simple keyword-based source classifier."""
    q = query.lower()

    sql_keywords = {"count", "sum", "average", "total", "how many", "sales", "revenue",
                    "database", "table", "records", "rows", "aggregate"}
    web_keywords = {"today", "latest", "current", "news", "price", "now", "recent",
                    "2024", "2025", "2026", "live", "real-time"}

    if any(kw in q for kw in sql_keywords):
        return "sql"
    if any(kw in q for kw in web_keywords):
        return "web"
    return "vector_store"


# Usage in retriever_fn:
def route_and_retrieve(query: str, source: str, k: int) -> list[Chunk]:
    if source == "auto":
        source = classify_source(query)
    # dispatch to actual retriever based on source...
    return []
```

---

## Exercise 4: Multi-turn with Max 5 Iterations

Already handled in `AgenticRAGAgent.run` above via the `while iterations < self.max_iterations` loop.

Key design points:
- Each tool_use response appends assistant content + tool results to `messages`
- Full conversation history preserved across hops
- Forced-stop message prompts graceful answer generation at iteration limit
- `max_iterations=5` is the default, configurable per instance

---

## Exercise 5: Tests — Mocking Anthropic tool_use Response

```python
# tests/unit/test_agentic_rag.py
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, call, patch

from pydantic import BaseModel


class FakeChunk(BaseModel):
    id: str
    text: str
    metadata: dict = {}


def _make_tool_use_response(
    tool_id: str, tool_name: str, tool_input: dict
) -> MagicMock:
    """Build a mock response where stop_reason='tool_use'."""
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = tool_id
    tool_block.name = tool_name
    tool_block.input = tool_input

    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [tool_block]
    return response


def _make_end_turn_response(text: str) -> MagicMock:
    """Build a mock response where stop_reason='end_turn'."""
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text

    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [text_block]
    return response


class TestAgenticRAGAgent(unittest.TestCase):

    @patch("anthropic.Anthropic")
    def test_direct_answer_no_tool_call(self, mock_cls: MagicMock) -> None:
        from day5_solutions import AgenticRAGAgent, AgentResult

        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_end_turn_response(
            "Direct answer without retrieval."
        )

        agent = AgenticRAGAgent(retriever_fn=lambda q, s, k: [])
        result = agent.run("Simple question")

        self.assertIsInstance(result, AgentResult)
        self.assertEqual(result.answer, "Direct answer without retrieval.")
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.sources, [])

    @patch("anthropic.Anthropic")
    def test_single_tool_call_then_answer(self, mock_cls: MagicMock) -> None:
        from day5_solutions import AgenticRAGAgent, AgentResult

        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_client.messages.create.side_effect = [
            _make_tool_use_response(
                "tool-abc",
                "retrieve",
                {"query": "Python GIL", "source": "vector_store", "k": 5},
            ),
            _make_end_turn_response("The GIL is a mutex..."),
        ]

        retrieved_chunks = [FakeChunk(id="c1", text="GIL explanation")]

        def fake_retriever(query: str, source: str, k: int) -> list[FakeChunk]:
            return retrieved_chunks

        agent = AgenticRAGAgent(retriever_fn=fake_retriever)
        result = agent.run("What is the Python GIL?")

        self.assertEqual(result.answer, "The GIL is a mutex...")
        self.assertEqual(result.iterations, 2)
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0].id, "c1")

    @patch("anthropic.Anthropic")
    def test_max_iterations_enforced(self, mock_cls: MagicMock) -> None:
        from day5_solutions import AgenticRAGAgent

        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        # Always returns tool_use → forces max_iterations
        tool_response = _make_tool_use_response(
            "t1", "retrieve", {"query": "q", "source": "vector_store"}
        )
        end_response = _make_end_turn_response("forced answer")
        # max_iterations=3 tool calls + 1 forced final
        mock_client.messages.create.side_effect = [
            tool_response,
            tool_response,
            tool_response,
            end_response,  # forced stop message
        ]

        agent = AgenticRAGAgent(retriever_fn=lambda q, s, k: [], max_iterations=3)
        result = agent.run("looping question")

        self.assertEqual(result.iterations, 3)

    @patch("anthropic.Anthropic")
    def test_tool_result_passed_back_to_model(self, mock_cls: MagicMock) -> None:
        from day5_solutions import AgenticRAGAgent

        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_tool_use_response(
                "tid1", "retrieve", {"query": "test", "source": "vector_store"}
            ),
            _make_end_turn_response("answer"),
        ]

        chunks = [FakeChunk(id="x", text="retrieved content")]
        agent = AgenticRAGAgent(retriever_fn=lambda q, s, k: chunks)
        agent.run("query")

        # Second call to messages.create should contain tool_result in messages
        second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]
        # Find tool_result block
        has_tool_result = any(
            isinstance(m.get("content"), list)
            and any(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in m["content"]
            )
            for m in messages
        )
        self.assertTrue(has_tool_result)


if __name__ == "__main__":
    unittest.main()
```
