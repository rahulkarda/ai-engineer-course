# Day 7 — Project Review Day

Review your implementation against the `project.md` acceptance criteria before demo.

---

## Architecture: All Components Connected

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  AgenticRAGAgentWithCitations                           │
│                                                         │
│  1. QueryRewriter ──────────────────────┐               │
│     (temperature=0.7)                   │               │
│                                         ▼               │
│  2. Tool-use loop (max 5 iters)  ◄── Claude LLM         │
│     ├─ classify_source(query)           │               │
│     ├─ vector_store ──► HyDERetriever   │               │
│     │  (multi-doc, avg embeddings)      │               │
│     ├─ web ──────────► web_search_stub  │               │
│     └─ sql ──────────► sql_stub         │               │
│                                         │               │
│  3. SelfRAGPipeline.generate            │               │
│     └─ RelevanceScorer filters chunks   │               │
│                                         │               │
│  4. CorrectionRouter.route              │               │
│     └─ if incorrect → web fallback      │               │
│                                         │               │
│  5. SufficiencyChecker (multi-hop)      │               │
│     └─ confidence > 0.85 → early stop  │               │
│                                         │               │
│  6. CitationExtractor.extract ◄─────────┘               │
│     └─ returns CitedAnswer with [N] refs                │
└─────────────────────────────────────────────────────────┘
    │
    ▼
CitedAnswer { answer, citations, sources }
```

---

## Integration Snippet

```python
import logging
import os
from dataclasses import dataclass
from typing import Any

import anthropic
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class CitedAnswer(BaseModel):
    answer: str
    citations: list[dict]
    sources: list[Any]


@dataclass
class AgenticRAGSystem:
    """Ties all week-06 components into one cohesive pipeline."""

    embed_fn: Any           # (text: str) -> list[float]
    chunks: list[Any]       # pre-indexed Chunk objects with embeddings
    model: str = "claude-sonnet-5-5"

    def __post_init__(self) -> None:
        from day1_solutions import QueryRewriter
        from day2_solutions import HyDERetrieverMulti
        from day4_solutions import RelevanceScorer, SelfRAGPipeline
        from day5_solutions import AgenticRAGAgent, classify_source
        from day6_solutions import AgenticRAGAgentWithCitations

        self._rewriter = QueryRewriter(model=self.model, temperature=0.7)
        self._hyde = HyDERetrieverMulti(
            chunks=self.chunks,
            model=self.model,
            num_hypothetical_docs=3,
            embed_fn=self.embed_fn,
        )
        self._scorer = RelevanceScorer(model=self.model)
        self._agent = AgenticRAGAgentWithCitations(
            retriever_fn=self._retriever_fn,
            model=self.model,
        )

    def _retriever_fn(self, query: str, source: str, k: int) -> list[Any]:
        if source == "vector_store":
            return self._hyde.retrieve(query, k=k)
        if source == "web":
            from day4_solutions import web_search_stub
            return web_search_stub(query)
        return []

    def run(self, user_query: str) -> CitedAnswer:
        rewritten = self._rewriter.rewrite(user_query)
        logger.info("Rewritten query: %s", rewritten)
        return self._agent.run(rewritten)
```

---

## Top 5 Pitfalls

1. **LLM returns malformed JSON** — always wrap `json.loads()` in try/except; strip markdown fences before parsing.

2. **Embedding dimension mismatch** — averaging embeddings from different text lengths is fine, but confirm all use the same embedding model/dimension; assert `len(emb) == expected_dim` at index time.

3. **Tool-use loop never terminates** — always enforce `max_iterations`; the Anthropic API will not stop the loop for you if the model keeps calling tools.

4. **Relevance threshold too high** — a threshold of 0.8+ filters out nearly everything on noisy corpora; start at 0.5 and tune against a held-out eval set.

5. **Citation index off-by-one** — sources are 1-indexed in prompts (`[1]`, `[2]`...) but 0-indexed in Python lists; always convert with `chunk = source_chunks[idx - 1]`.
