# Phase 4 · Week 06 · Day 1 — Query Rewriting & Decomposition

> Objective: Transform weak user queries into retrieval-optimised forms using rewriting, multi-query expansion, decomposition, and step-back prompting.  ·  Time: ~1 hr  ·  Prereqs: [[week-05/day-2]] (reranking pipeline)

## Concept

Naive RAG passes the user's raw query directly to the retriever. This breaks in three predictable ways:

**Vocabulary mismatch.** The user asks "how do I stop my app from crashing on startup?" but the indexed docs say "handle application initialisation exceptions". Embedding similarity is non-zero but far from optimal — different words, same meaning.

**Ambiguity.** "Python context manager" could mean the `with` statement, `contextlib`, async context managers, or the `__enter__`/`__exit__` protocol. A single embedding averages across all meanings.

**Multi-faceted questions.** "Compare LangGraph and CrewAI for production agentic systems" is really three questions bundled: what is LangGraph, what is CrewAI, how do they differ in production. Embedding the whole sentence produces a centroid that retrieves nothing well.

Query transformation techniques break these problems before retrieval starts.

**The four main techniques**

| Technique | Problem it solves | Mechanism |
|---|---|---|
| Query rewriting | Vocabulary mismatch | LLM rephrases query to match expected corpus language |
| Multi-query expansion | Single point-of-failure retrieval | LLM generates N phrasings; union retrieved sets |
| Decomposition | Multi-faceted questions | LLM breaks question into atomic sub-questions |
| Step-back prompting | Over-specific queries | LLM generalises to the underlying concept first |

## How it works internally

**Query rewriting** is an LLM call at query time: `system = "Rephrase the user query to be more likely to match technical documentation."`. The original query is replaced or augmented. Adds one LLM round-trip (~200–500 ms) but can dramatically lift recall.

**Multi-query expansion** calls the LLM once to produce N rephrased variants (N = 3–5 is typical). For each variant, a separate retrieval is issued. Results are merged with deduplication by chunk ID (set union). Recall improves because different phrasings hit different vocabulary clusters in the index. Cost: N retrieval calls, though these are cheap (ANN lookups).

**Decomposition** is structurally different: the LLM produces a list of atomic sub-questions. Each sub-question is independently retrievable. Results may be used in a sequential chain (answer Q1 → inject answer into Q2) or in parallel (retrieve all, merge context). This is the foundation of multi-hop retrieval (Day 3).

**Step-back prompting** (Google DeepMind, 2023) first generalises the specific query — "What causes `KeyError` in Python defaultdict" → "How does Python dictionary key lookup work?" — then retrieves on the broader concept. The intuition: the corpus is more likely to contain general principle explanations than highly specific edge-case descriptions.

**Deduplication** matters in all multi-retrieval strategies. Use a `dict[str, Chunk]` keyed on chunk ID; insertion order is preserved in Python 3.7+ so you get implicit score-ranked dedup.

## Worked example

```python
"""
day1_query_rewriting.py

Query transformation pipeline: rewriting, multi-query, decomposition,
and step-back prompting. LLM calls use the Anthropic SDK.
Run with ANTHROPIC_API_KEY set in environment.
Mock all LLM calls in tests — see exercises.
"""

import logging
import os
from dataclasses import dataclass, field

import anthropic
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared Anthropic client — key from env, never hardcoded
# ---------------------------------------------------------------------------
def _client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=api_key)


MODEL = "claude-sonnet-5-5"  # or claude-opus-5-5 for higher quality

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
class RewrittenQuery(BaseModel):
    original: str
    rewritten: str


class ExpandedQueries(BaseModel):
    original: str
    variants: list[str]


class DecomposedQuery(BaseModel):
    original: str
    sub_questions: list[str]


class StepBackQuery(BaseModel):
    original: str
    general_query: str


@dataclass
class Chunk:
    id: str
    text: str
    score: float = 0.0


# ---------------------------------------------------------------------------
# Simulated retriever (replace with week_05.retriever.VectorRetriever in prod)
# ---------------------------------------------------------------------------
from week_05.retriever import VectorRetriever  # type: ignore[import]


# ---------------------------------------------------------------------------
# QueryRewriter: single-pass LLM rewriting
# ---------------------------------------------------------------------------
class QueryRewriter:
    """Rewrites a user query to better match technical documentation vocabulary."""

    SYSTEM_PROMPT = (
        "You are a search query optimiser. "
        "Rewrite the given user query so it better matches the vocabulary "
        "used in technical documentation and code comments. "
        "Return only the rewritten query, no explanation."
    )

    def __init__(self, model: str = MODEL) -> None:
        self._model = model
        self._client = _client()

    def rewrite(self, query: str) -> RewrittenQuery:
        log.info("Rewriting query: %s", query)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=256,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": query}],
        )
        rewritten = response.content[0].text.strip()
        log.info("Rewritten → %s", rewritten)
        return RewrittenQuery(original=query, rewritten=rewritten)


# ---------------------------------------------------------------------------
# MultiQueryExpander: generate N variant phrasings, deduplicate retrieved chunks
# ---------------------------------------------------------------------------
class MultiQueryExpander:
    """Generates N query variants and retrieves chunks for each, merging results."""

    SYSTEM_PROMPT = (
        "You are a search query generator. "
        "Given a user question, produce {n} distinct rephrased versions that each "
        "approach the same information need from a different angle. "
        "Return one query per line, no numbering, no explanations."
    )

    def __init__(self, retriever: VectorRetriever, n: int = 3, model: str = MODEL) -> None:
        self._retriever = retriever
        self._n = n
        self._model = model
        self._client = _client()

    def expand(self, query: str) -> ExpandedQueries:
        log.info("Expanding query into %d variants …", self._n)
        prompt = self.SYSTEM_PROMPT.format(n=self._n)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system=prompt,
            messages=[{"role": "user", "content": query}],
        )
        lines = [l.strip() for l in response.content[0].text.strip().splitlines() if l.strip()]
        variants = lines[: self._n]
        log.info("Variants: %s", variants)
        return ExpandedQueries(original=query, variants=variants)

    def retrieve_union(self, query: str, top_k: int = 5) -> list[Chunk]:
        """Retrieve for each variant; deduplicate by chunk ID; return ordered list."""
        expanded = self.expand(query)
        all_queries = [query] + expanded.variants
        seen: dict[str, Chunk] = {}
        for q in all_queries:
            chunks: list[Chunk] = self._retriever.retrieve(q, top_k=top_k)
            for chunk in chunks:
                if chunk.id not in seen:
                    seen[chunk.id] = chunk
        log.info("Union retrieved %d unique chunks across %d queries", len(seen), len(all_queries))
        return list(seen.values())


# ---------------------------------------------------------------------------
# QueryDecomposer: break multi-faceted question into atomic sub-questions
# ---------------------------------------------------------------------------
class QueryDecomposer:
    """Decomposes a complex question into independently-retrievable sub-questions."""

    SYSTEM_PROMPT = (
        "You are a question decomposition expert. "
        "Break the following complex question into simple, atomic sub-questions "
        "that can each be answered by retrieving a single document chunk. "
        "Return one sub-question per line. No numbering. No explanations."
    )

    def __init__(self, model: str = MODEL) -> None:
        self._model = model
        self._client = _client()

    def decompose(self, query: str) -> DecomposedQuery:
        log.info("Decomposing: %s", query)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": query}],
        )
        lines = [l.strip() for l in response.content[0].text.strip().splitlines() if l.strip()]
        log.info("Sub-questions: %s", lines)
        return DecomposedQuery(original=query, sub_questions=lines)


# ---------------------------------------------------------------------------
# StepBackPrompter: generalise a specific query before retrieval
# ---------------------------------------------------------------------------
class StepBackPrompter:
    """Generalises a specific question to a broader concept for improved retrieval."""

    SYSTEM_PROMPT = (
        "You are a query abstraction specialist. "
        "Given a specific technical question, produce a more general version "
        "that asks about the underlying principle or concept. "
        "Return only the generalised question, no explanation."
    )

    def __init__(self, model: str = MODEL) -> None:
        self._model = model
        self._client = _client()

    def step_back(self, query: str) -> StepBackQuery:
        log.info("Step-back from: %s", query)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=256,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": query}],
        )
        general = response.content[0].text.strip()
        log.info("General query → %s", general)
        return StepBackQuery(original=query, general_query=general)


# ---------------------------------------------------------------------------
# Demo: run all four techniques on the same query (requires ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------
def main() -> None:
    query = "Why does LangGraph agent keep looping without stopping?"

    log.info("=== Query Rewriting ===")
    rewriter = QueryRewriter()
    rewritten = rewriter.rewrite(query)
    log.info("Original : %s", rewritten.original)
    log.info("Rewritten: %s", rewritten.rewritten)

    log.info("\n=== Step-Back Prompting ===")
    step_back = StepBackPrompter()
    sb = step_back.step_back(query)
    log.info("Original : %s", sb.original)
    log.info("General  : %s", sb.general_query)

    log.info("\n=== Query Decomposition ===")
    decomposer = QueryDecomposer()
    decomposed = decomposer.decompose(
        "Compare LangGraph and CrewAI for building production-grade agentic systems"
    )
    for i, sq in enumerate(decomposed.sub_questions, 1):
        log.info("  Sub-Q %d: %s", i, sq)


if __name__ == "__main__":
    main()
```

**Key observations to note when running:**

- `QueryRewriter` changes vocabulary: "keep looping" → "infinite loop / missing termination condition" — closer to how docs phrase it.
- `StepBackPrompter` generalises: specific LangGraph loop → general "how do LLM agent loop termination conditions work?"
- `QueryDecomposer` turns one complex question into 3–5 atomic ones, each independently retrievable.
- `MultiQueryExpander.retrieve_union` deduplicates by chunk ID — same chunk retrieved by two variants only appears once.

## Exercises

1. **Rewriting ablation.** Wire `QueryRewriter` to your `VectorRetriever` from Week 5. Retrieve top-5 chunks with the original query, then with the rewritten query. Log both result sets. Do the chunk IDs differ? Which result set contains more relevant chunks (score them manually)?

2. **Multi-query dedup test.** Write a `pytest` unit test for `MultiQueryExpander.retrieve_union`. Use `unittest.mock.patch` to mock `_client()` so the LLM returns a fixed list of 3 variants. Assert that when the mock retriever returns overlapping chunks across variants, each chunk ID appears exactly once in the output. Hint: use `unittest.mock.patch("__main__.MultiQueryExpander._client", return_value=mock_client)`.

3. **Decompose and retrieve in parallel.** Extend `QueryDecomposer` with a method `decompose_and_retrieve(query, retriever, top_k=3) -> dict[str, list[Chunk]]` that returns a mapping from each sub-question to its retrieved chunks. Use `concurrent.futures.ThreadPoolExecutor` to issue all retrievals in parallel.

4. **Step-back + original fusion.** Implement a `StepBackRetriever` that retrieves on both the step-back query and the original query, then merges and deduplicates results (step-back first for ranking priority). Write a unit test that mocks both LLM and retriever calls.

5. **Benchmark all four.** Create a mini evaluation: define 5 questions and a toy corpus of 20 chunks where you know which chunks are relevant. Compare Recall@5 for: raw query, rewritten query, multi-query expansion, and decomposed queries. Log a summary table. This is the foundation for systematic RAG evaluation (Week 7).

(Solutions: ../solutions/week-06/day-1.md)

## Self-check

- [ ] I can explain why vocabulary mismatch causes retrieval failure.
- [ ] I can describe what multi-query expansion does differently from a single rewrite.
- [ ] I can implement `QueryDecomposer` from memory using the Anthropic SDK.
- [ ] I can explain the step-back prompting intuition with a concrete example.
- [ ] I can write a pytest test that mocks the Anthropic client for `QueryRewriter`.

## Supplementary links

- [Step-Back Prompting (DeepMind, 2023)](https://arxiv.org/abs/2310.06025)
- [RAG-Fusion: multi-query + reciprocal rank fusion](https://arxiv.org/abs/2402.03367)
- [Anthropic Messages API reference](https://docs.anthropic.com/en/api/messages)
- [LangChain multi-query retriever (for comparison)](https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever)
