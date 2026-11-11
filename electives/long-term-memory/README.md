# Elective: Long-Term Memory Systems for Agents

> Module overview · Self-contained · Estimated time: 10–12 hours

## What this module teaches

LLM context windows are ephemeral: every conversation starts blank. Production agents
need **persistent memory** — the ability to recall past interactions, facts, preferences,
and procedures across sessions, users, and time.

This module builds that capability from first principles: memory taxonomies, storage
architectures, write/retrieval pipelines, scoring, decay, and the leading open-source
frameworks (Mem0, Letta/MemGPT). You will ship a working memory service that plugs into
a LangGraph agent, with mocked LLM/embedding calls in tests.

After this module you will be able to:

- [ ] Explain the four memory types (working / episodic / semantic / procedural) and when each applies
- [ ] Design a memory architecture using vector store + graph + KV store
- [ ] Implement write pipelines with summarization, dedup, and importance scoring
- [ ] Retrieve memories using recency + importance + relevance composite scoring
- [ ] Apply forgetting / decay strategies (exponential decay, access-based reinforcement)
- [ ] Use Mem0 (concept + API shape) and Letta/MemGPT (memory hierarchy, paging)
- [ ] Evaluate a memory system (precision@k, recall@k, staleness, latency)
- [ ] Build and test a pluggable memory service for a LangGraph agent

## Prerequisites

- Python 3.12+
- Phase 5 (Agentic) complete, or equivalent LangGraph experience
- Familiarity with Pydantic, pytest, vector embeddings (Phase 4)
- `ANTHROPIC_API_KEY` set in environment (mocked in tests — not required for tests)

## Dependencies

```bash
uv add anthropic
uv add pydantic pydantic-settings
uv add numpy                        # cosine similarity without full ML stack
uv add pytest pytest-asyncio
```

Optional (concept exploration only — do not install for tests):

```bash
uv add mem0ai          # Mem0 — pip name is mem0ai, import as mem0
uv add letta           # Letta (was MemGPT) — import as letta
uv add chromadb        # vector backend option
uv add neo4j           # graph backend option
```

> `mem0ai` import: `from mem0 import Memory`
> `letta` import: `from letta import create_client`

## Lesson index

| # | File | Topic | Time |
|---|------|--------|------|
| 1 | [lesson-1.md](lesson-1.md) | Why agents need persistent memory & memory types | ~60 min |
| 2 | [lesson-2.md](lesson-2.md) | Memory architectures: vector, graph, KV | ~60 min |
| 3 | [lesson-3.md](lesson-3.md) | Writing & consolidating memories | ~60 min |
| 4 | [lesson-4.md](lesson-4.md) | Retrieval & relevance scoring | ~60 min |
| 5 | [lesson-5.md](lesson-5.md) | Forgetting, decay, and Mem0 / Letta | ~75 min |
| 6 | [lesson-6.md](lesson-6.md) | Evaluating memory systems | ~60 min |
| P | [project.md](project.md) | Capstone: pluggable memory service for an agent | ~4–5 hr |

## Solutions

All exercise answers live under [solutions/](solutions/). Open them only after attempting
the exercises yourself.

## Supplementary links

- Mem0 docs: <https://docs.mem0.ai>
- Mem0 GitHub: <https://github.com/mem0ai/mem0>
- Letta (MemGPT) docs: <https://docs.letta.com>
- Letta GitHub: <https://github.com/letta-ai/letta>
- MemGPT paper (Packer et al., 2023): <https://arxiv.org/abs/2310.08560>
- Cognitive Architectures for Language Agents (CoALA) survey: <https://arxiv.org/abs/2309.02427>
- Generative Agents (Park et al., 2023): <https://arxiv.org/abs/2304.03442>
- ChromaDB docs: <https://docs.trychroma.com>
- Neo4j Python driver: <https://neo4j.com/docs/python-manual/current/>
- Anthropic API docs: <https://docs.anthropic.com>
