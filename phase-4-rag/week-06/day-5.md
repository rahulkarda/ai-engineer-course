# Phase 4 · Week 06 · Day 5 — Agentic RAG: Retrieval as a Tool + Source Routing

> Objective: Build an agentic RAG loop where the model decides when to retrieve, what to retrieve, and which source to use — using the Anthropic tool_use API.  ·  Time: ~1 hr  ·  Prereqs: [[week-06/day-4]]

---

## Concept

Day 4 added gates around retrieval, but the pipeline structure was still fixed: retrieve → score → generate. **Agentic RAG** removes that fixed order. The model itself decides:

- **Whether** to retrieve (or answer from parametric knowledge).
- **What** to retrieve (the query it sends to the retrieval tool).
- **Which source** to retrieve from (vector store, web, SQL, API).
- **Whether to retrieve again** after reading the first results.

This is retrieval-as-a-tool: the model treats each retrieval source as a callable function in the same way it calls any other tool. The generation loop continues until the model emits a final answer with no more tool calls.

### Why agentic over pipeline RAG?

| Concern | Pipeline RAG | Agentic RAG |
|---|---|---|
| Multi-hop questions | One retrieval pass often insufficient | Agent can retrieve, read, re-query |
| Source selection | Hard-coded | Model picks the right tool |
| When not to retrieve | Requires a separate gate | Model simply doesn't call retrieve |
| Explainability | Implicit | Tool call log shows reasoning chain |

### The Anthropic tool_use pattern

The Anthropic messages API lets you define tools as JSON schemas. When the model needs a tool result, it emits a `tool_use` content block. You execute the tool, add the result as a `tool_result` message, and continue the loop until the model emits a `text` block with no pending tool calls.

```
messages = [{"role": "user", "content": query}]
loop:
    response = client.messages.create(..., tools=tools, messages=messages)
    if response.stop_reason == "tool_use":
        # extract tool_use blocks, execute, append tool_result
        messages = updated_messages_with_results
    elif response.stop_reason == "end_turn":
        return extract_text(response)
```

---

## How it works internally

### Tool definition format

Each tool is a dict with `name`, `description`, and `input_schema` (JSON Schema):

```python
{
    "name": "retrieve",
    "description": "Retrieve relevant documents from a source.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "source": {
                "type": "string",
                "enum": ["vector_store", "web", "sql", "api"],
                "description": "Which retrieval source to use",
            },
        },
        "required": ["query", "source"],
    },
}
```

The `description` field is the most important. The model uses it to decide when and how to call the tool. Be specific: describe what the source contains, not just its name.

### Handling tool_use content blocks

A response with `stop_reason="tool_use"` has content blocks of type `"tool_use"`. Each has:
- `id` — a unique call identifier you must echo back in the `tool_result`
- `name` — the tool name
- `input` — a dict matching your `input_schema`

The `tool_result` message must echo the `tool_use_id`. Structure:

```python
{
    "role": "user",
    "content": [
        {
            "type": "tool_result",
            "tool_use_id": tool_use_block.id,
            "content": json_serialised_result,
        }
    ],
}
```

### Source router logic

The model picks the source via the `source` enum in the tool schema. You can bias routing by writing clear source descriptions. Alternatively, add a separate `classify_query` tool that the model calls first — but in practice, a well-described `source` enum is sufficient for most use cases.

### Multi-step retrieval

The agentic loop runs until `stop_reason == "end_turn"`. This means the model can call `retrieve` multiple times across different turns — first to get background context, then to answer a follow-up detail. Limit iterations with a `max_iterations` guard to prevent runaway loops.

---

## Worked example

```python
"""
day5_agentic_rag.py — Agentic RAG with retrieval as a tool.

Install:
    uv add anthropic pydantic

Run:
    python day5_agentic_rag.py
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import anthropic
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

# ── Data models ───────────────────────────────────────────────────────────

class Chunk(BaseModel):
    id: str
    text: str
    source: str
    score: float = 1.0


class RetrievalResult(BaseModel):
    chunks: list[Chunk]
    source: str
    query: str


class AgentStep(BaseModel):
    step_type: str          # "tool_call" | "tool_result" | "answer"
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    content: str = ""


class AgentTrace(BaseModel):
    query: str
    steps: list[AgentStep] = Field(default_factory=list)
    answer: str = ""
    iterations: int = 0


# ── Tool definitions ──────────────────────────────────────────────────────

RETRIEVE_TOOL: dict[str, Any] = {
    "name": "retrieve",
    "description": (
        "Retrieve relevant documents from a data source. "
        "Use 'vector_store' for internal company documentation and product knowledge. "
        "Use 'web' for current events, recent releases, or anything after your training cutoff. "
        "Use 'sql' to query structured data like sales figures, user counts, or metrics. "
        "Use 'api' for structured third-party data such as weather, stock prices, or geocoding."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query. Be specific — include key entities and constraints.",
            },
            "source": {
                "type": "string",
                "enum": ["vector_store", "web", "sql", "api"],
                "description": "Retrieval source to use.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return. Default 3.",
                "default": 3,
            },
        },
        "required": ["query", "source"],
    },
}

TOOLS: list[dict[str, Any]] = [RETRIEVE_TOOL]


# ── Retrieval backends (stubs) ────────────────────────────────────────────

def _retrieve_vector_store(query: str, top_k: int) -> list[Chunk]:
    log.info("  [vector_store] query=%r top_k=%d", query, top_k)
    return [
        Chunk(
            id="vs-1",
            text="Python 3.12 introduced PEP 695 type parameter syntax: `type Point[T] = tuple[T, T]`.",
            source="vector_store",
            score=0.91,
        ),
        Chunk(
            id="vs-2",
            text="PEP 695 also adds `class Stack[T]:` syntax replacing `Generic[T]`.",
            source="vector_store",
            score=0.87,
        ),
    ][:top_k]


def _retrieve_web(query: str, top_k: int) -> list[Chunk]:
    log.info("  [web] query=%r top_k=%d", query, top_k)
    return [
        Chunk(
            id="web-1",
            text=f"[Simulated web result] Latest information on: {query}",
            source="web",
            score=0.75,
        )
    ][:top_k]


def _retrieve_sql(query: str, top_k: int) -> list[Chunk]:
    log.info("  [sql] query=%r top_k=%d", query, top_k)
    return [
        Chunk(
            id="sql-1",
            text="SELECT result: monthly_active_users=142000, date='2024-11'",
            source="sql",
            score=1.0,
        )
    ][:top_k]


def _retrieve_api(query: str, top_k: int) -> list[Chunk]:
    log.info("  [api] query=%r top_k=%d", query, top_k)
    return [
        Chunk(
            id="api-1",
            text=f"[API stub] Structured response for query: {query}",
            source="api",
            score=0.8,
        )
    ][:top_k]


# ── Retrieval dispatcher ──────────────────────────────────────────────────

def retrieve(query: str, source: str, top_k: int = 3) -> RetrievalResult:
    """Dispatch to the correct retrieval backend."""
    dispatch = {
        "vector_store": _retrieve_vector_store,
        "web": _retrieve_web,
        "sql": _retrieve_sql,
        "api": _retrieve_api,
    }
    if source not in dispatch:
        raise ValueError(f"Unknown source: {source!r}. Valid: {list(dispatch)}")

    chunks = dispatch[source](query, top_k)
    return RetrievalResult(chunks=chunks, source=source, query=query)


def execute_tool(name: str, tool_input: dict[str, Any]) -> str:
    """Execute a tool by name and return a JSON string result."""
    if name == "retrieve":
        result = retrieve(
            query=tool_input["query"],
            source=tool_input["source"],
            top_k=tool_input.get("top_k", 3),
        )
        return json.dumps(result.model_dump(), indent=2)
    raise ValueError(f"Unknown tool: {name!r}")


# ── Agentic RAG loop ──────────────────────────────────────────────────────

class AgenticRAG:
    """
    Runs a retrieval-augmented agentic loop.

    The model decides when to call `retrieve` and which source to use.
    The loop continues until stop_reason == "end_turn" or max_iterations is hit.
    """

    SYSTEM = (
        "You are a helpful assistant with access to retrieval tools. "
        "When answering questions:\n"
        "1. First decide if you need to retrieve information.\n"
        "2. If yes, call the retrieve tool with the appropriate source.\n"
        "3. You may call retrieve multiple times if the first results are insufficient.\n"
        "4. Cite your sources in the format [source:chunk_id] when using retrieved content.\n"
        "5. Answer concisely based on retrieved context."
    )

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str,
        max_iterations: int = 6,
    ) -> None:
        self._client = client
        self._model = model
        self._max_iterations = max_iterations

    def run(self, query: str) -> AgentTrace:
        trace = AgentTrace(query=query)
        messages: list[dict[str, Any]] = [{"role": "user", "content": query}]

        for iteration in range(self._max_iterations):
            trace.iterations = iteration + 1

            response = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=self.SYSTEM,
                tools=TOOLS,
                messages=messages,
            )

            log.info("Iteration %d — stop_reason=%s", iteration + 1, response.stop_reason)

            if response.stop_reason == "end_turn":
                # Extract final text answer
                text_blocks = [b for b in response.content if b.type == "text"]
                trace.answer = text_blocks[0].text if text_blocks else ""
                trace.steps.append(AgentStep(step_type="answer", content=trace.answer))
                break

            if response.stop_reason == "tool_use":
                # Process all tool_use blocks in this response
                tool_results: list[dict[str, Any]] = []

                for block in response.content:
                    if block.type != "tool_use":
                        continue

                    log.info("  Tool call: %s(%s)", block.name, json.dumps(block.input))
                    trace.steps.append(
                        AgentStep(
                            step_type="tool_call",
                            tool_name=block.name,
                            tool_input=block.input,
                        )
                    )

                    result_str = execute_tool(block.name, block.input)
                    log.info("  Tool result length: %d chars", len(result_str))

                    trace.steps.append(
                        AgentStep(
                            step_type="tool_result",
                            tool_name=block.name,
                            content=result_str,
                        )
                    )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_str,
                        }
                    )

                # Append assistant response + tool results to message history
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})

            else:
                log.warning("Unexpected stop_reason: %s", response.stop_reason)
                break

        else:
            log.warning("max_iterations=%d reached", self._max_iterations)

        return trace


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    agent = AgenticRAG(client, model="claude-sonnet-5-5")

    queries = [
        "What is the new type parameter syntax introduced in Python 3.12?",
        "What are our monthly active users for November 2024?",
        "What is 2 + 2?",  # Should not trigger retrieval
    ]

    for q in queries:
        log.info("=" * 60)
        log.info("Query: %s", q)
        trace = agent.run(q)
        log.info("Iterations: %d", trace.iterations)
        log.info("Steps: %d", len(trace.steps))
        for step in trace.steps:
            if step.step_type == "tool_call":
                log.info("  → retrieve(%s, source=%s)", step.tool_input.get("query"), step.tool_input.get("source"))
        log.info("Answer: %s", trace.answer)


if __name__ == "__main__":
    main()
```

---

## Exercises

1. **Tool schema improvement**: The current `retrieve` tool description doesn't mention what the `vector_store` contains. Extend the description to say it holds "internal Python documentation and PEP summaries". Write a unit test that mocks `client.messages.create` to verify the exact tool definition dict passed in the `tools` parameter matches the expected schema (use `assert tools[0]["name"] == "retrieve"`).

2. **Multi-source routing test**: Write a pytest test that creates a mock `client.messages.create` returning a `tool_use` block with `source="sql"` on the first call, then an `end_turn` response on the second call. Assert that `execute_tool("retrieve", {"query": "...", "source": "sql"})` calls `_retrieve_sql` (use `unittest.mock.patch`).

3. **Iteration cap**: Add a test that mocks `client.messages.create` to always return `stop_reason="tool_use"`. Assert that `AgenticRAG.run` returns after exactly `max_iterations` iterations and logs a warning. Set `max_iterations=3` for the test.

4. **Citation extraction**: After `trace.answer` is set, write a function `extract_citations(answer: str) -> list[str]` that uses a regex to find all `[source:chunk_id]` patterns. Add a unit test that parses `"See [source:vs-1] and [source:web-2]."` and returns `["vs-1", "web-2"]`.

5. **Parallel tool calls**: The Anthropic API can return multiple `tool_use` blocks in a single response. Modify `AgenticRAG.run` to collect all tool call results first, then append them all in a single `tool_result` user message (the current code already does this — verify it and write a test that asserts two simultaneous `tool_use` blocks are both executed and both results appear in the same reply message).

(Solutions: ../solutions/week-06/day-5.md)

---

## Self-check

- [ ] I can explain why agentic RAG is more flexible than pipeline RAG for multi-hop questions.
- [ ] I can write a valid Anthropic tool definition dict with `name`, `description`, and `input_schema`.
- [ ] I can implement the tool_use loop: detect `stop_reason`, extract `tool_use` blocks, execute, append `tool_result`.
- [ ] I can explain what `tool_use_id` is for and what happens if it's missing.
- [ ] I can describe how the model chooses between sources using the tool description text.
- [ ] I can write a mocked pytest test for `AgenticRAG.run` without making real API calls.

---

## Supplementary links

- Anthropic tool use guide: https://docs.anthropic.com/en/docs/build-with-claude/tool-use
- Anthropic tool use examples: https://docs.anthropic.com/en/docs/build-with-claude/tool-use/implement-tool-use
- Agentic RAG survey: https://arxiv.org/abs/2501.09136
- LangGraph tool node reference: https://langchain-ai.github.io/langgraph/how-tos/tool-calling/
- Self-RAG day notes: ./day-4.md
