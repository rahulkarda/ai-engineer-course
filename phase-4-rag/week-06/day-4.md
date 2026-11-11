# Phase 4 · Week 06 · Day 4 — Self-RAG & Corrective RAG

> Objective: Implement retrieval quality gates that decide whether retrieved context should be used, corrected, or replaced before generation.  ·  Time: ~1 hr  ·  Prereqs: [[week-05/day-2]]

---

## Concept

Standard RAG has a blind spot: it always injects retrieved chunks into the prompt regardless of whether those chunks are relevant. If the top-k results are off-topic, you inject noise — and noise degrades output quality.

Two papers address this directly:

**Self-RAG** (Asai et al., 2023) trains a model to emit special reflection tokens at generation time. The model critiques its own retrieval and decides whether to use each retrieved passage.

**Corrective RAG** (Yan et al., 2024) adds a *pre-generation* relevance check. If retrieved documents score too low, the pipeline rewrites the query and searches again (e.g., using web search) before generating.

Both ideas share a core insight: **retrieval confidence should gate context injection**, not just passage rank.

### Self-RAG: four reflection tokens

Self-RAG defines four binary special tokens that the model can emit as part of its generation:

| Token | Question it answers | When emitted |
|---|---|---|
| `[Retrieve]` | Should I retrieve at all? | Before generating; `YES` triggers retrieval, `NO` skips it |
| `[IsRel]` | Is this retrieved passage relevant to my query? | After receiving each passage |
| `[IsSup]` | Does my generated segment match facts in this passage? | After generating a segment |
| `[IsUse]` | Should I use this passage in my final answer? | Final gate per passage |

In the original paper the model is fine-tuned to emit these tokens. In practice you can approximate all four gates with lightweight LLM calls (or a cross-encoder) without fine-tuning.

### When retrieval helps vs. hurts

Retrieval helps when:
- The query requires factual knowledge the model may hallucinate.
- The answer is grounded in a specific document corpus.

Retrieval hurts when:
- Retrieved passages are topically adjacent but factually irrelevant.
- The model's parametric knowledge is more accurate than the retrieved content.
- Chunk overlap introduces contradictory statements.

Injecting irrelevant context is worse than no context. Studies show 15–30% accuracy drops when random passages are prepended to prompts.

### Corrective RAG: three branches

After computing a relevance score for retrieved documents, CRAG routes to one of three branches:

```
relevance score
    ≥ HIGH_THRESH  →  CORRECT    — use retrieved docs as-is
    ≤ LOW_THRESH   →  INCORRECT  — discard, rewrite query, web search
    in between     →  AMBIGUOUS  — keep + supplement with web results
```

The key operations:
1. **Relevance scoring**: lightweight LLM call or cross-encoder per (query, chunk) pair.
2. **Query rewriting**: ask the LLM to rephrase the query for better search coverage.
3. **Source fallback**: switch to web search or a different retrieval source.

---

## How it works internally

### Relevance scoring with an LLM

For each (query, chunk) pair, make a small structured call:

```
System: You are a relevance judge. Reply with JSON {"relevant": true/false, "score": 0.0-1.0}.
User: Query: {query}
      Passage: {chunk_text}
```

Use a low-cost model (haiku-class) for this. Score every retrieved chunk before deciding the branch.

Alternatively use a cross-encoder (sentence-transformers `ms-marco-MiniLM`) for lower latency — same approach as week-05/day-2. The LLM approach is preferred when your domain is narrow and your cross-encoder was trained on general web data.

### State machine

Self-RAG and CRAG both implement a **linear state machine** over the generation pipeline. Each stage receives the current state dict and returns an updated state dict. This maps naturally to a LangGraph `StateGraph` or a plain `dict`-based loop.

```
query
  ↓ should_retrieve?
  ↓ retrieve_chunks()
  ↓ score_relevance()  ← IsRel gate
  ↓ route_branch()     ← CRAG routing
  ↓ [optionally rewrite + re-retrieve]
  ↓ generate_answer()
  ↓ check_support()    ← IsSup gate
  ↓ return answer
```

### Why not always re-retrieve?

Re-retrieval is expensive (extra latency, extra API calls). The LOW_THRESH gate ensures re-retrieval only fires when initial retrieval is clearly poor. In practice, ~10–15% of queries trigger the INCORRECT branch; the rest proceed normally.

---

## Worked example

```python
"""
day4_self_crag.py — Self-RAG relevance gates + Corrective RAG routing.

Install:
    uv add anthropic pydantic

Run:
    python day4_self_crag.py
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

# ── Models ──────────────────────────────────────────────────────────────────

class Chunk(BaseModel):
    id: str
    text: str
    source: str


class RelevanceScore(BaseModel):
    chunk_id: str
    relevant: bool
    score: float = Field(ge=0.0, le=1.0)


class RAGState(BaseModel):
    query: str
    chunks: list[Chunk] = Field(default_factory=list)
    relevance_scores: list[RelevanceScore] = Field(default_factory=list)
    branch: Literal["correct", "incorrect", "ambiguous"] = "ambiguous"
    rewritten_query: str | None = None
    answer: str = ""
    should_retrieve: bool = True


# ── Thresholds ────────────────────────────────────────────────────────────

HIGH_THRESH: float = 0.7   # CORRECT branch
LOW_THRESH: float = 0.3    # INCORRECT branch


# ── Relevance scorer ──────────────────────────────────────────────────────

class RelevanceScorer:
    """Scores (query, chunk) relevance using a lightweight LLM call."""

    SYSTEM = (
        "You are a relevance judge. Given a query and a passage, "
        "respond with JSON only: {\"relevant\": <bool>, \"score\": <0.0-1.0>}. "
        "score 1.0 = perfectly on-topic, 0.0 = completely off-topic."
    )

    def __init__(self, client: anthropic.Anthropic, model: str) -> None:
        self._client = client
        self._model = model

    def score(self, query: str, chunk: Chunk) -> RelevanceScore:
        import json

        prompt = f"Query: {query}\n\nPassage: {chunk.text}"
        response = self._client.messages.create(
            model=self._model,
            max_tokens=64,
            system=self.SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        try:
            data = json.loads(raw)
            return RelevanceScore(
                chunk_id=chunk.id,
                relevant=bool(data["relevant"]),
                score=float(data["score"]),
            )
        except (json.JSONDecodeError, KeyError) as exc:
            log.warning("Relevance parse failed for chunk %s: %s", chunk.id, exc)
            return RelevanceScore(chunk_id=chunk.id, relevant=False, score=0.0)

    def score_all(self, query: str, chunks: list[Chunk]) -> list[RelevanceScore]:
        return [self.score(query, c) for c in chunks]


# ── Correction router ─────────────────────────────────────────────────────

class CorrectionRouter:
    """Decides the CRAG branch based on mean relevance score."""

    def route(
        self, scores: list[RelevanceScore]
    ) -> Literal["correct", "incorrect", "ambiguous"]:
        if not scores:
            return "incorrect"
        mean = sum(s.score for s in scores) / len(scores)
        log.info("Mean relevance score: %.3f", mean)
        if mean >= HIGH_THRESH:
            return "correct"
        if mean <= LOW_THRESH:
            return "incorrect"
        return "ambiguous"


# ── Query rewriter ────────────────────────────────────────────────────────

class QueryRewriter:
    """Rewrites a query to improve retrieval coverage."""

    SYSTEM = (
        "Rewrite the user's query to make it clearer and more specific "
        "for a web search engine. Return only the rewritten query, nothing else."
    )

    def __init__(self, client: anthropic.Anthropic, model: str) -> None:
        self._client = client
        self._model = model

    def rewrite(self, query: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=128,
            system=self.SYSTEM,
            messages=[{"role": "user", "content": query}],
        )
        return response.content[0].text.strip()


# ── Retrieval stubs ───────────────────────────────────────────────────────

def retrieve_from_vector_store(query: str) -> list[Chunk]:
    """Stub: returns fake chunks from an internal vector store."""
    log.info("Retrieving from vector store for: %r", query)
    return [
        Chunk(id="vs-1", text="Python 3.12 introduced the new type parameter syntax for generics.", source="internal"),
        Chunk(id="vs-2", text="The walrus operator := was introduced in Python 3.8.", source="internal"),
    ]


def retrieve_from_web(query: str) -> list[Chunk]:
    """Stub: simulates web search retrieval."""
    log.info("Web retrieval for: %r", query)
    return [
        Chunk(
            id="web-1",
            text=f"[Web result] Current information about: {query}",
            source="web",
        )
    ]


# ── Self-RAG: should-retrieve gate ───────────────────────────────────────

class ShouldRetrieveGate:
    """[Retrieve] token gate — decides if retrieval is needed at all."""

    SYSTEM = (
        "Decide if the following question requires retrieving documents. "
        "Reply JSON only: {\"retrieve\": <bool>, \"reason\": \"<short reason>\"}. "
        "retrieve=false for simple factual questions the model can answer from memory, "
        "retrieve=true for domain-specific, recent, or corpus-dependent questions."
    )

    def __init__(self, client: anthropic.Anthropic, model: str) -> None:
        self._client = client
        self._model = model

    def check(self, query: str) -> bool:
        import json

        response = self._client.messages.create(
            model=self._model,
            max_tokens=64,
            system=self.SYSTEM,
            messages=[{"role": "user", "content": query}],
        )
        raw = response.content[0].text.strip()
        try:
            data = json.loads(raw)
            log.info("ShouldRetrieve: %s — %s", data["retrieve"], data.get("reason", ""))
            return bool(data["retrieve"])
        except (json.JSONDecodeError, KeyError):
            return True  # default to retrieving if unsure


# ── Answer generator ──────────────────────────────────────────────────────

class AnswerGenerator:
    """Generates a final answer given a query and (filtered) context chunks."""

    def __init__(self, client: anthropic.Anthropic, model: str) -> None:
        self._client = client
        self._model = model

    def generate(self, query: str, chunks: list[Chunk]) -> str:
        if chunks:
            context = "\n\n".join(
                f"[{c.source}] {c.text}" for c in chunks
            )
            user_msg = f"Context:\n{context}\n\nQuestion: {query}"
        else:
            user_msg = query

        response = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system="Answer the question using only the provided context. Be concise.",
            messages=[{"role": "user", "content": user_msg}],
        )
        return response.content[0].text.strip()


# ── Self-RAG pipeline ─────────────────────────────────────────────────────

class SelfRAGPipeline:
    """
    Implements Self-RAG gates + Corrective RAG branching.

    Flow:
        query
          → [Retrieve] gate  (skip retrieval if not needed)
          → retrieve_from_vector_store
          → [IsRel] score each chunk
          → CorrectionRouter picks branch
          → CORRECT: use chunks as-is
          → INCORRECT: rewrite + web retrieve
          → AMBIGUOUS: keep relevant chunks + supplement with web
          → generate answer
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        *,
        fast_model: str,
        gen_model: str,
    ) -> None:
        self._retrieve_gate = ShouldRetrieveGate(client, fast_model)
        self._scorer = RelevanceScorer(client, fast_model)
        self._router = CorrectionRouter()
        self._rewriter = QueryRewriter(client, fast_model)
        self._generator = AnswerGenerator(client, gen_model)

    def run(self, query: str) -> RAGState:
        state = RAGState(query=query)

        # [Retrieve] gate
        state.should_retrieve = self._retrieve_gate.check(query)
        if not state.should_retrieve:
            log.info("[Retrieve]=NO — generating from parametric knowledge")
            state.answer = self._generator.generate(query, [])
            return state

        # Initial retrieval
        log.info("[Retrieve]=YES — retrieving from vector store")
        state.chunks = retrieve_from_vector_store(query)

        # [IsRel] score each chunk
        state.relevance_scores = self._scorer.score_all(query, state.chunks)
        for rs in state.relevance_scores:
            log.info("  chunk=%s  relevant=%s  score=%.3f", rs.chunk_id, rs.relevant, rs.score)

        # CRAG routing
        state.branch = self._router.route(state.relevance_scores)
        log.info("CRAG branch: %s", state.branch)

        if state.branch == "correct":
            # Use all retrieved chunks
            context_chunks = state.chunks

        elif state.branch == "incorrect":
            # Discard retrieved chunks, rewrite + web search
            state.rewritten_query = self._rewriter.rewrite(query)
            log.info("Rewritten query: %r", state.rewritten_query)
            context_chunks = retrieve_from_web(state.rewritten_query)
            state.chunks = context_chunks

        else:  # ambiguous
            # Keep chunks scoring above 0.4, supplement with web
            relevant_chunks = [
                c for c, rs in zip(state.chunks, state.relevance_scores) if rs.score >= 0.4
            ]
            web_chunks = retrieve_from_web(query)
            context_chunks = relevant_chunks + web_chunks
            state.chunks = context_chunks

        # Generate answer with filtered context
        state.answer = self._generator.generate(query, context_chunks)
        return state


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Use a fast model for gates; slightly stronger model for final generation
    pipeline = SelfRAGPipeline(
        client,
        fast_model="claude-sonnet-5-5",
        gen_model="claude-sonnet-5-5",
    )

    queries = [
        "What type parameter syntax was introduced in Python 3.12?",
        "What is the current interest rate set by the Federal Reserve?",
    ]

    for q in queries:
        log.info("=" * 60)
        log.info("Query: %s", q)
        state = pipeline.run(q)
        log.info("Branch: %s", state.branch)
        log.info("Answer: %s", state.answer)


if __name__ == "__main__":
    main()
```

---

## Exercises

1. **Relevance threshold tuning**: Modify `SelfRAGPipeline` to accept `high_thresh` and `low_thresh` as constructor parameters instead of module-level constants. Write a pytest parametrize test that verifies `CorrectionRouter.route` returns `"correct"` / `"ambiguous"` / `"incorrect"` for scores `[0.9]`, `[0.5]`, and `[0.1]` respectively. Mock the Anthropic client.

2. **IsSup gate**: Implement an `IsSupportedGate` class that takes a (generated answer segment, list of chunks) and returns `bool`. It should make a single LLM call asking: "Is this claim supported by the passages?". Add it to `SelfRAGPipeline.run` as a post-generation check. If not supported, log a warning and append a disclaimer to the answer.

3. **Batch relevance scoring**: The current `score_all` method calls `score()` in a loop (one API call per chunk). Rewrite it to send all (query, chunk) pairs in a single prompt (one API call total), parsing the JSON array response. Benchmark the latency difference against the loop version using `time.perf_counter`.

4. **Query rewriting quality**: Add a unit test for `QueryRewriter` that mocks `client.messages.create` to return the text `"  Python 3.12 type aliases and generics syntax\n"`. Assert that `rewrite()` strips the whitespace and returns the clean string.

5. **Full pipeline integration test**: Write a pytest integration test (marked `@pytest.mark.integration`, skipped unless `ANTHROPIC_API_KEY` is set) that runs `SelfRAGPipeline.run` end-to-end on the query `"What is 2 + 2?"`. Assert that `state.should_retrieve` is `False` (the model recognises it doesn't need retrieval) and `state.answer` contains `"4"`.

(Solutions: ../solutions/week-06/day-4.md)

---

## Self-check

- [ ] I can explain all four Self-RAG reflection tokens and what question each answers.
- [ ] I can describe the three CRAG branches and what triggers each.
- [ ] I can explain why injecting irrelevant context degrades LLM output quality.
- [ ] I can implement a relevance scorer that makes a single structured LLM call per chunk.
- [ ] I can wire a state machine (dict-based or LangGraph) that gates context injection on relevance score.
- [ ] I can write a mocked pytest test for `CorrectionRouter.route` without any real API calls.

---

## Supplementary links

- Self-RAG paper: https://arxiv.org/abs/2310.11511
- Corrective RAG paper: https://arxiv.org/abs/2401.15884
- LangChain CRAG tutorial (reference implementation): https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag/
- Cross-encoder relevance scoring (week-05/day-2 notes): ../week-05/day-2.md
- Anthropic messages API: https://docs.anthropic.com/en/api/messages
