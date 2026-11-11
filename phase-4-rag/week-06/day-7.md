# Phase 4 · Week 06 · Day 7 — Agentic RAG: Integration & Review

> Objective: Integrate all week-6 components into a complete agentic RAG system, validate against acceptance criteria, and complete the week assessment.  ·  Time: ~3 hr project + 1 hr assessment  ·  Prereqs: [[day-6]]

## Week recap

**Query rewriting & decomposition** (Day 1): Raw user queries often fail due to vocabulary mismatch, ambiguity, or multi-faceted intent. Query rewriting uses an LLM to rephrase the query toward corpus vocabulary. Multi-query expansion generates N variants and union-retrieves across all of them. Decomposition breaks complex questions into atomic sub-questions each independently retrievable. Step-back prompting generalises a specific query to the underlying concept first.

**HyDE + contextual retrieval** (Day 2): HyDE transforms the query into a hypothetical answer before embedding, placing the search vector in "answer space" rather than "question space". Contextual retrieval enriches chunks at index time with an LLM-generated prefix that situates each chunk within its source document — a one-time cost that improves every subsequent retrieval.

**Multi-hop retrieval** (Day 3): Some questions require chaining multiple retrievals — each hop uses the partial answer from the previous to refine the next query. A SufficiencyChecker (LLM-based, structured JSON output) decides at each hop whether to continue or stop. Three stopping criteria: max_hops exceeded, LLM declares sufficient, or confidence threshold passed.

**Self-RAG & Corrective RAG** (Day 4): Self-RAG introduces four decision tokens (IsRAGRequired, IsChunkRelevant, IsAnswerGrounded, IsAnswerUseful) that gate retrieval and generation at inference time. CRAG adds a three-branch routing: if retrieved chunks are correct, use them; if incorrect, query the web; if ambiguous, blend both. Both techniques reduce hallucination by explicitly checking grounding before and after generation.

**Agentic RAG with tool_use** (Day 5): Retrieval-as-a-tool: the LLM decides when to retrieve, with what query, and from which source. The agent calls `retrieve_internal(query)` or `retrieve_web(query)` as tools, iterates until it has enough context, then generates a grounded answer. The Anthropic `tool_use` API provides the interface — tool definitions, tool_use content blocks, tool_result injection.

**Citations & grounding** (Day 6): Inline citations `[N]` link answer claims to source chunks. Faithfulness measures whether every claim in the answer is supported by at least one cited chunk. Relevance measures whether cited chunks actually contain the claim. A grounding checker LLM call evaluates both dimensions per claim.

## Integration walkthrough

The following shows the complete wiring of all week-6 components into a single runnable system. Components are stubbed for the example — replace stubs with real implementations from each day's file.

```python
"""
week06_agentic_rag_system.py

Complete agentic RAG system integrating all Week 6 components.
All LLM calls use os.getenv("ANTHROPIC_API_KEY").
Model: claude-sonnet-5-5
Python 3.12+

Run with:
    ANTHROPIC_API_KEY=... python week06_agentic_rag_system.py
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import anthropic
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

MODEL = "claude-sonnet-5-5"


def _client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# Shared Pydantic models
# ---------------------------------------------------------------------------
class CitedAnswer(BaseModel):
    text: str
    sources: list[str]


class Chunk(BaseModel):
    id: str
    text: str
    score: float = 0.0


# ---------------------------------------------------------------------------
# Stub components (replace with real implementations from each day)
# ---------------------------------------------------------------------------

class QueryRewriter:
    """Day 1: rewrites query to match corpus vocabulary."""
    def rewrite(self, query: str) -> str:
        log.info("QueryRewriter.rewrite: %s", query)
        # Stub — real implementation calls LLM
        return query


class MultiQueryExpander:
    """Day 1: generates N query variants."""
    def expand(self, query: str, n: int = 3) -> list[str]:
        log.info("MultiQueryExpander.expand: %s", query)
        # Stub — real implementation calls LLM
        return [query]


class ContextualChunkBuilder:
    """Day 2: adds LLM-generated context prefix to each chunk at index time."""
    def build_contextual_chunk(
        self, chunk_id: str, chunk_text: str, document_text: str
    ) -> Chunk:
        # Stub
        return Chunk(id=chunk_id, text=chunk_text)


class MultiHopRetriever:
    """Day 3: iterative retrieval with LLM sufficiency checking."""
    def retrieve(self, question: str, max_hops: int = 3) -> list[Chunk]:
        log.info("MultiHopRetriever.retrieve: %s", question)
        # Stub — real implementation loops up to max_hops times
        return [
            Chunk(id="stub-1", text=f"Stub context for: {question}", score=0.9)
        ]


class SelfRAGPipeline:
    """Day 4: relevance filtering + corrective routing."""
    def filter_relevant(
        self, chunks: list[Chunk], threshold: float = 0.7
    ) -> list[Chunk]:
        log.info("SelfRAGPipeline.filter_relevant: %d chunks in", len(chunks))
        return [c for c in chunks if c.score >= threshold]


class CitationExtractor:
    """Day 6: extracts inline [N] citations from generated text."""
    def extract(self, text: str, chunks: list[Chunk]) -> CitedAnswer:
        log.info("CitationExtractor.extract")
        sources = [c.id for c in chunks]
        # Stub — real implementation parses [N] markers and maps to chunk IDs
        return CitedAnswer(text=text, sources=sources)


# ---------------------------------------------------------------------------
# Tool definitions for Anthropic tool_use (Day 5)
# ---------------------------------------------------------------------------
TOOLS: list[dict[str, Any]] = [
    {
        "name": "retrieve_internal",
        "description": "Search the internal knowledge base for relevant chunks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "retrieve_web",
        "description": "Search the public web for recent information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"],
        },
    },
]


# ---------------------------------------------------------------------------
# AgenticRAGSystem — orchestrates all components
# ---------------------------------------------------------------------------
class AgenticRAGSystem:
    """
    End-to-end agentic RAG pipeline.

    Flow:
      user_query
        → QueryRewriter (vocabulary alignment)
        → MultiQueryExpander (recall boost)
        → MultiHopRetriever (multi-hop loop)
        → SelfRAGPipeline.filter_relevant (relevance gate)
        → AgenticRAGAgent tool_use loop (routing + generation)
        → CitationExtractor (inline citations)
        → CitedAnswer
    """

    def __init__(self) -> None:
        self._rewriter = QueryRewriter()
        self._expander = MultiQueryExpander()
        self._multihop = MultiHopRetriever()
        self._self_rag = SelfRAGPipeline()
        self._citations = CitationExtractor()
        self._client = _client()

    # ------------------------------------------------------------------
    # Internal tool dispatch (called when agent emits a tool_use block)
    # ------------------------------------------------------------------
    def _dispatch_tool(
        self, tool_name: str, tool_input: dict[str, str]
    ) -> list[Chunk]:
        query = tool_input.get("query", "")
        if tool_name == "retrieve_internal":
            log.info("Tool: retrieve_internal(%s)", query)
            return self._multihop.retrieve(query)
        if tool_name == "retrieve_web":
            log.info("Tool: retrieve_web(%s) — stub returns empty", query)
            # Wire a real web retriever here (e.g. Brave Search API)
            return []
        log.warning("Unknown tool: %s", tool_name)
        return []

    # ------------------------------------------------------------------
    # Agentic loop (Anthropic tool_use)
    # ------------------------------------------------------------------
    def _run_agent_loop(
        self, rewritten_query: str, seed_chunks: list[Chunk]
    ) -> tuple[str, list[Chunk]]:
        """
        Runs the Anthropic tool_use loop.
        Returns (generated_answer_text, all_chunks_used).
        """
        seed_context = "\n\n".join(
            f"[{i + 1}] {c.text}" for i, c in enumerate(seed_chunks)
        )
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    f"Question: {rewritten_query}\n\n"
                    f"Available context:\n{seed_context}\n\n"
                    "If the context is sufficient, answer directly with inline citations [N]. "
                    "If you need more information, call retrieve_internal or retrieve_web."
                ),
            }
        ]

        all_chunks = list(seed_chunks)
        answer_text = ""

        for _ in range(5):  # max agentic loop iterations
            response = self._client.messages.create(
                model=MODEL,
                max_tokens=1024,
                tools=TOOLS,
                messages=messages,
            )
            log.info("Agent stop_reason: %s", response.stop_reason)

            if response.stop_reason == "end_turn":
                # Extract text from final response
                for block in response.content:
                    if hasattr(block, "text"):
                        answer_text = block.text
                break

            if response.stop_reason == "tool_use":
                # Process all tool calls in this turn
                tool_results: list[dict[str, Any]] = []
                for block in response.content:
                    if block.type == "tool_use":
                        chunks = self._dispatch_tool(block.name, block.input)
                        all_chunks.extend(chunks)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps(
                                    [{"id": c.id, "text": c.text} for c in chunks]
                                ),
                            }
                        )
                # Append assistant turn + tool results
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})

        return answer_text, all_chunks

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def run(self, user_query: str) -> CitedAnswer:
        """
        Full pipeline: user_query → CitedAnswer with inline citations.
        """
        log.info("=== AgenticRAGSystem.run ===")
        log.info("Query: %s", user_query)

        # 1. Query transformation
        rewritten = self._rewriter.rewrite(user_query)
        expanded = self._expander.expand(rewritten)
        log.info("Rewritten: %s | Expanded variants: %d", rewritten, len(expanded))

        # 2. Multi-hop retrieval on rewritten query
        raw_chunks = self._multihop.retrieve(rewritten)
        log.info("Multi-hop retrieved %d chunks", len(raw_chunks))

        # 3. Relevance filtering (Self-RAG gate)
        relevant_chunks = self._self_rag.filter_relevant(raw_chunks, threshold=0.7)
        log.info("After relevance filter: %d chunks", len(relevant_chunks))

        # 4. Agentic tool_use loop
        answer_text, all_chunks = self._run_agent_loop(rewritten, relevant_chunks)

        # 5. Citation extraction
        cited = self._citations.extract(answer_text, all_chunks)
        log.info(
            "Final answer: %d chars, %d sources", len(cited.text), len(cited.sources)
        )
        return cited


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    system = AgenticRAGSystem()
    query = "What Python version does Pydantic v2 require and which SAP framework uses it?"
    answer = system.run(query)
    log.info("Answer: %s", answer.text)
    log.info("Sources: %s", answer.sources)


if __name__ == "__main__":
    main()
```

## Project build steps

**1. Set up project structure**

```
week_06_project/
├── src/
│   ├── query_transform.py
│   ├── contextual_index.py
│   ├── multihop.py
│   ├── self_rag.py
│   ├── agent.py
│   ├── citations.py
│   └── models.py
├── tests/
│   ├── test_query_transform.py
│   ├── test_multihop.py
│   ├── test_self_rag.py
│   ├── test_agent.py
│   └── test_citations.py
├── pyproject.toml
└── .env.example
```

Create `pyproject.toml` with `anthropic`, `pydantic`, `pytest`. Create `.env.example` with `ANTHROPIC_API_KEY=`. Decision: keep all Pydantic models in `models.py` to avoid circular imports across modules.

**2. Implement QueryRewriter + MultiQueryExpander**

File: `src/query_transform.py`. Import `anthropic`, define `MODEL = "claude-sonnet-5-5"`, implement `QueryRewriter.rewrite(query) -> RewrittenQuery` and `MultiQueryExpander.expand(query, n=3) -> ExpandedQueries`. Key decision: should `expand` call `rewrite` first? Prefer composing at the pipeline level, not inside the class. Common pitfall: not stripping blank lines from the LLM's variant list — always filter with `[l.strip() for l in lines if l.strip()]`.

**3. Wire ContextualChunkBuilder for index-time prefixing**

File: `src/contextual_index.py`. The builder must receive the full document text, not just the chunk. Key decision: how much of the document to pass? Cap at 6000 chars to stay within context window while leaving room for the chunk and response. Common pitfall: calling `build_all` at query time instead of index time — this is a one-time indexing step, not a per-query step.

**4. Build MultiHopRetriever**

File: `src/multihop.py`. Implement `SufficiencyChecker.check(question, chunks) -> SufficiencyCheck` with structured JSON output via the Anthropic SDK. Implement `MultiHopRetriever.retrieve(question) -> MultiHopResult` with the three stopping criteria. Key decision: accumulate all chunks across hops or only keep the latest hop's chunks? Accumulate — the final synthesis needs the full chain. Common pitfall: forgetting to deduplicate by chunk ID across hops.

**5. Implement SelfRAGPipeline**

File: `src/self_rag.py`. Implement `RelevanceChecker.score(query, chunk) -> float` (LLM-based) and `SelfRAGPipeline.filter_relevant(chunks, threshold=0.7) -> list[Chunk]`. Also implement `CorrectionRouter.route(chunks) -> Literal["correct", "ambiguous", "incorrect"]` for CRAG-style routing. Key decision: score individually or batch? Individual is simpler and easier to test; batch is faster — implement individual first, optimise later. Common pitfall: using a fixed threshold without considering corpus-specific score distributions.

**6. Build AgenticRAGAgent with Anthropic tool_use**

File: `src/agent.py`. Define TOOLS list with `retrieve_internal` and `retrieve_web` schemas. Implement the `while stop_reason == "tool_use"` loop, dispatch tool calls, inject `tool_result` messages. Key decision: max iterations cap — 5 is safe for most queries; set as a configurable parameter. Common pitfall: not injecting tool results as `{"role": "user", "content": [{"type": "tool_result", ...}]}` — the content must be a list, not a string.

**7. Add CitationExtractor**

File: `src/citations.py`. Parse `[N]` markers from generated text with `re.findall(r'\[(\d+)\]', text)`. Map citation numbers to chunk IDs. Implement `GroundingChecker.check_faithfulness(claim, chunk) -> bool`. Return `CitedAnswer(text=..., sources=...)`. Common pitfall: off-by-one when mapping 1-indexed citation numbers to 0-indexed chunk lists.

**8. Write comprehensive test suite**

One test file per module. Mock all LLM calls using `unittest.mock.patch`. For each class:
- Unit test the happy path with a known mocked response
- Unit test edge cases (empty chunks, max hops hit, malformed JSON from LLM)
- Assert `CitedAnswer` structure in integration-level tests

Use `pytest.mark.unit` and `pytest.mark.integration` markers. Run with `pytest tests/ -v -m unit` for fast local checks.

## Self-check (full week)

- [ ] I can explain vocabulary mismatch and why query rewriting helps
- [ ] I can implement HyDE from scratch
- [ ] I can implement contextual chunk prefixing at index time
- [ ] I can build a multi-hop retriever with configurable stopping criteria
- [ ] I can explain self-RAG's four decision points
- [ ] I can implement an agentic RAG loop using Anthropic tool_use API
- [ ] I can extract inline citations and verify grounding
- [ ] I can explain Graph RAG's entity-relationship approach vs. vector similarity

## Supplementary links

- [Anthropic contextual retrieval blog](https://www.anthropic.com/news/contextual-retrieval)
- [Self-RAG paper](https://arxiv.org/abs/2310.11511)
- [CRAG paper](https://arxiv.org/abs/2401.15884)
- [Microsoft GraphRAG](https://github.com/microsoft/graphrag)
- [Anthropic tool_use docs](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
