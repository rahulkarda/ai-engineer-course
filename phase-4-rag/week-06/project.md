# Week 06 Project — Agentic RAG System

## Overview

Build a production-quality agentic RAG system that integrates query transformation, contextual retrieval, multi-hop reasoning, self-correction, and citation generation. The agent decides when and how to retrieve, chains multiple retrievals when needed, and returns grounded cited answers.

## Requirements (functional)

1. **QueryRewriter**: rewrite the user query to match corpus vocabulary + multi-query expansion generating 3 distinct phrasings per query.
2. **ContextualChunkBuilder**: LLM-generated context prefix for each chunk at index time; prefix + original text stored as the embedded unit.
3. **MultiHopRetriever**: iterative retrieval loop up to 3 hops, with LLM-based sufficiency check at each hop; stops at sufficiency or max_hops, whichever comes first.
4. **SelfRAGPipeline**: relevance filter discarding chunks with score < 0.7; CRAG-style corrective routing (correct/ambiguous/incorrect) determining whether to use internal chunks, web results, or both.
5. **AgenticRAGAgent**: Anthropic `tool_use` loop with `retrieve_internal` and `retrieve_web` tool definitions; agent decides when to retrieve and from which source; max 5 agentic loop iterations.
6. **CitationExtractor**: parse inline `[N]` markers from generated answer text; map each citation number to the corresponding chunk's source ID; return `CitedAnswer(text, sources)`.
7. **Full test suite**: all LLM calls mocked with `unittest.mock.patch`; pytest with markers (`unit`, `integration`); ≥ 80% coverage measured with `pytest-cov`.

## Project structure

```
week_06_project/
├── src/
│   ├── query_transform.py    # QueryRewriter, MultiQueryExpander, QueryDecomposer
│   ├── contextual_index.py   # ContextualChunkBuilder
│   ├── multihop.py           # MultiHopRetriever, SufficiencyChecker
│   ├── self_rag.py           # SelfRAGPipeline, CorrectionRouter
│   ├── agent.py              # AgenticRAGAgent (tool_use loop)
│   ├── citations.py          # CitationExtractor, GroundingChecker
│   └── models.py             # Pydantic models: Chunk, CitedAnswer, HopResult, etc.
├── tests/
│   ├── test_query_transform.py
│   ├── test_multihop.py
│   ├── test_self_rag.py
│   ├── test_agent.py
│   └── test_citations.py
├── pyproject.toml
└── .env.example
```

## Acceptance criteria

All 10 must pass before submission:

1. `AgenticRAGSystem.run(query)` returns a `CitedAnswer` with non-empty `text` field.
2. Multi-hop retriever stops at sufficiency or max 3 hops (whichever comes first) — verified by unit test asserting `len(result.hops) <= 3`.
3. Irrelevant chunks (relevance score < 0.7) are filtered before generation — verified by unit test confirming filtered-out chunks do not appear in the final context.
4. Every answer contains at least one `[N]` citation — verified with `re.search(r'\[\d+\]', answer.text)`.
5. `CitedAnswer.sources` length matches the number of distinct citation indices found in `answer.text`.
6. All pytest tests pass with no real API calls — CI must run cleanly with `ANTHROPIC_API_KEY` unset.
7. No `os.environ["ANTHROPIC_API_KEY"] = ...` assignment in any source file — only `os.getenv("ANTHROPIC_API_KEY")` reads are permitted.
8. All public function signatures and class `__init__` methods have type hints on all parameters and return types.
9. `logging` used throughout all `src/` files; no bare `print(` calls in `src/` (grep check in CI).
10. `models.py` defines all shared Pydantic models: `Chunk`, `CitedAnswer`, `HopResult`, `SufficiencyCheck`, `RewrittenQuery`, `ExpandedQueries`, `ContextualChunk`.

## Stretch goals

- **GraphRAG stub**: implement entity extraction from chunks using an LLM call that returns `list[Entity]` with `name` and `type` fields; build a simple adjacency dict; implement `graph_retrieve(entity_name) -> list[Chunk]` that returns chunks mentioning directly connected entities. No full graph DB required — in-memory dict is sufficient for the stub.
- **Async multi-query**: replace sequential retrieval across expanded query variants with `asyncio.gather` — issue all N retrieval calls in parallel and merge results. Benchmark latency improvement with `time.perf_counter`.
- **Evaluation harness**: define a 10-query test set with known relevant chunk IDs; compute precision@3, recall@3, and faithfulness score (LLM judge: does the answer text contradict any cited chunk?) for each query; log a summary table.
- **Prompt caching**: add `"cache_control": {"type": "ephemeral"}` to the document excerpt in `ContextualChunkBuilder` system message; log `usage.cache_read_input_tokens` vs. `usage.cache_creation_input_tokens` to confirm caching is active across a 20-chunk indexing run.

## Submission checklist

- [ ] All tests pass (`pytest tests/ -v`)
- [ ] No hardcoded API keys (grep for `sk-ant` returns nothing)
- [ ] Type hints on all public functions and classes
- [ ] `logging` used (grep for bare `print(` in `src/` returns nothing)
- [ ] `.env.example` with `ANTHROPIC_API_KEY=` placeholder
- [ ] `models.py` has all shared Pydantic models
