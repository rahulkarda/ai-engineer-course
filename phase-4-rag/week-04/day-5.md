# Phase 4 · Week 04 · Day 5 — Filtering, Query Construction & Top-k Tuning

> Objective: Add metadata filtering to retrieval, construct structured queries, and tune top-k for quality vs. cost.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

## Concept

Real RAG systems rarely do unfiltered full-corpus search. They add:

1. **Metadata filters** — pre-filter by date, category, language, tenant, etc. before scoring.
2. **Query construction** — transform raw user input into a richer structured query (rewriting, expansion, decomposition).
3. **Top-k tuning** — balance recall, LLM context cost, and latency.

### Metadata filtering

Documents carry structured fields (e.g. `{"source": "internal", "date": "2024-01", "lang": "en"}`). Filtering happens *before* or *during* retrieval:

- **Pre-filter (hard filter):** restrict the candidate set to documents matching a predicate. Reduces ANN search space → faster.
- **Post-filter:** retrieve top-k then discard non-matching docs. Simpler but wastes ANN computation; can return < k results.
- **Filtered ANN (hybrid index):** some vector DBs (Qdrant, Weaviate, Pinecone) push filters into the ANN index, avoiding both downsides.

### Query construction

Raw queries are often ambiguous, short, or query-term-poor. Strategies:

| Technique | Input | Output |
|---|---|---|
| **HyDE** (Hypothetical Document Embedding) | Question | Embed a *generated* answer, not the question |
| **Query expansion** | Short query | Add synonyms / related terms |
| **Multi-query** | Complex question | Decompose into N sub-queries, merge results |
| **Step-back prompting** | Specific question | Generalise to broader topic query first |

We implement **query expansion** (LLM-free, synonym-based) and **sub-query decomposition** (mock LLM call) today.

### Top-k selection

| k | Recall | Context tokens | Latency |
|---|---|---|---|
| 1 | low | low | fast |
| 5 | medium | ~500–2000 | moderate |
| 20 | high | ~4000–8000 | slower |
| 100 | very high | too large for most LLMs | slow |

**Rule of thumb:** k = 5–10 for a single-turn QA chain; k = 20–50 for a candidate pool before a reranker step.

**Marginal utility** of k falls quickly after rank 5–10 (measured by NDCG@k plateau in most benchmarks).

## How it works internally

```
User query
    │
    ▼ QueryConstructor
  [metadata filter predicates]  +  [expanded/decomposed query text]
    │
    ▼ FilteredRetriever
  Pre-filter docs by metadata predicates
    │
    ▼ BM25 + Dense (on filtered subset)
    │
    ▼ RRF fusion
    │
    ▼ Top-k selection
  Results → LLM context
```

## Worked example

```python
"""
day5_filtering_query_construction.py — metadata filters + query construction.

Run: python day5_filtering_query_construction.py
Deps: uv add sentence-transformers faiss-cpu numpy pydantic
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Document model
# ---------------------------------------------------------------------------
class Document(BaseModel):
    doc_id: int
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Metadata filter
# ---------------------------------------------------------------------------
Filter = Callable[[Document], bool]


def eq_filter(field_name: str, value: Any) -> Filter:
    return lambda doc: doc.metadata.get(field_name) == value


def gte_filter(field_name: str, value: Any) -> Filter:
    return lambda doc: doc.metadata.get(field_name, None) is not None and doc.metadata[field_name] >= value


def and_filter(*filters: Filter) -> Filter:
    return lambda doc: all(f(doc) for f in filters)


def or_filter(*filters: Filter) -> Filter:
    return lambda doc: any(f(doc) for f in filters)


# ---------------------------------------------------------------------------
# Minimal BM25 with Document support
# ---------------------------------------------------------------------------
@dataclass
class FilteredBM25Retriever:
    k1: float = 1.5
    b: float = 0.75
    _docs: list[Document] = field(default_factory=list, init=False)
    _doc_lengths: list[int] = field(default_factory=list, init=False)
    _total_len: int = field(default=0, init=False)
    _inverted: dict[str, dict[int, int]] = field(
        default_factory=lambda: defaultdict(dict), init=False
    )

    def _tokenize(self, text: str) -> list[str]:
        return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()

    def add_documents(self, docs: list[Document]) -> None:
        for doc in docs:
            self._docs.append(doc)
            tokens = self._tokenize(doc.text)
            self._doc_lengths.append(len(tokens))
            self._total_len += len(tokens)
            for term, freq in Counter(tokens).items():
                self._inverted[term][doc.doc_id] = freq

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filter_fn: Filter | None = None,
    ) -> list[tuple[Document, float]]:
        terms = self._tokenize(query)
        n = len(self._docs)
        avgdl = self._total_len / n if n else 1.0

        # Pre-filter candidate docs
        candidates: list[Document] = [
            d for d in self._docs if filter_fn is None or filter_fn(d)
        ]
        candidate_ids = {d.doc_id for d in candidates}

        scores: dict[int, float] = {}
        for t in terms:
            for doc_id, freq in self._inverted.get(t, {}).items():
                if doc_id not in candidate_ids:
                    continue
                dl = self._doc_lengths[doc_id]
                df = len(self._inverted.get(t, {}))
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
                tf_norm = freq * (self.k1 + 1) / (
                    freq + self.k1 * (1 - self.b + self.b * dl / avgdl)
                )
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf_norm

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        doc_map = {d.doc_id: d for d in candidates}
        return [(doc_map[did], score) for did, score in ranked if did in doc_map]


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------
SYNONYM_MAP: dict[str, list[str]] = {
    "fast": ["quick", "rapid", "high-speed"],
    "vector": ["embedding", "dense", "float array"],
    "search": ["retrieval", "lookup", "query"],
    "model": ["llm", "language model", "neural network"],
}


def expand_query(query: str) -> str:
    """Append synonyms for known terms."""
    tokens = query.lower().split()
    extras: list[str] = []
    for tok in tokens:
        if tok in SYNONYM_MAP:
            extras.extend(SYNONYM_MAP[tok])
    expanded = query + " " + " ".join(extras) if extras else query
    log.debug("Expanded query: '%s' → '%s'", query, expanded)
    return expanded


def decompose_query(query: str) -> list[str]:
    """
    Mock LLM-based decomposition.
    In production, call an LLM with a decomposition prompt.
    """
    # Hard-coded decomposition for demo — replace with LLM call.
    decompositions: dict[str, list[str]] = {
        "compare BM25 and dense retrieval for short technical queries": [
            "BM25 retrieval performance on technical queries",
            "dense retrieval performance on short queries",
            "comparison of keyword vs semantic retrieval",
        ],
    }
    sub_queries = decompositions.get(query, [query])
    log.debug("Decomposed '%s' → %s", query, sub_queries)
    return sub_queries


# ---------------------------------------------------------------------------
# Top-k analysis
# ---------------------------------------------------------------------------
def marginal_recall(
    retriever: FilteredBM25Retriever,
    query: str,
    relevant_ids: set[int],
    max_k: int = 20,
) -> dict[int, float]:
    """Compute recall@k for k in 1..max_k."""
    results = retriever.retrieve(query, top_k=max_k)
    recall_at_k: dict[int, float] = {}
    retrieved_relevant = 0
    for k, (doc, _) in enumerate(results, start=1):
        if doc.doc_id in relevant_ids:
            retrieved_relevant += 1
        recall_at_k[k] = retrieved_relevant / len(relevant_ids)
    return recall_at_k


# ---------------------------------------------------------------------------
# Demo corpus with metadata
# ---------------------------------------------------------------------------
CORPUS_WITH_META: list[dict[str, Any]] = [
    {"text": "Python is a high-level language used in data science.", "category": "programming", "year": 2020, "lang": "en"},
    {"text": "FastAPI is a fast Python web framework.", "category": "programming", "year": 2022, "lang": "en"},
    {"text": "BM25 ranks documents using probabilistic term weighting.", "category": "retrieval", "year": 2009, "lang": "en"},
    {"text": "Dense embeddings are used in semantic search systems.", "category": "retrieval", "year": 2021, "lang": "en"},
    {"text": "FAISS provides approximate nearest-neighbor search.", "category": "retrieval", "year": 2019, "lang": "en"},
    {"text": "Pydantic validates Python data using type hints.", "category": "programming", "year": 2023, "lang": "en"},
    {"text": "LangGraph orchestrates stateful multi-step agent workflows.", "category": "agents", "year": 2024, "lang": "en"},
    {"text": "FastAPI supports async request handling natively.", "category": "programming", "year": 2022, "lang": "en"},
    {"text": "Semantic retrieval finds documents by meaning not keywords.", "category": "retrieval", "year": 2022, "lang": "en"},
    {"text": "Agent frameworks manage tool calls and conversation state.", "category": "agents", "year": 2024, "lang": "en"},
]


def main() -> None:
    docs = [
        Document(doc_id=i, text=d["text"], metadata={"category": d["category"], "year": d["year"], "lang": d["lang"]})
        for i, d in enumerate(CORPUS_WITH_META)
    ]

    retriever = FilteredBM25Retriever()
    retriever.add_documents(docs)

    # --- 1. Metadata filtering ---
    log.info("=== Filtered retrieval: category=retrieval ===")
    retrieval_filter = eq_filter("category", "retrieval")
    results = retriever.retrieve("semantic search vectors", top_k=5, filter_fn=retrieval_filter)
    for doc, score in results:
        log.info("  [%.4f] [%s] %s", score, doc.metadata.get("category"), doc.text)

    log.info("=== Filtered: category=programming AND year>=2022 ===")
    prog_recent = and_filter(eq_filter("category", "programming"), gte_filter("year", 2022))
    results = retriever.retrieve("Python framework", top_k=5, filter_fn=prog_recent)
    for doc, score in results:
        log.info("  [%.4f] [%d] %s", score, doc.metadata.get("year"), doc.text)

    # --- 2. Query expansion ---
    log.info("=== Query expansion ===")
    raw_q = "fast vector search"
    expanded_q = expand_query(raw_q)
    log.info("Raw:      %s", raw_q)
    log.info("Expanded: %s", expanded_q)
    for doc, score in retriever.retrieve(expanded_q, top_k=3):
        log.info("  [%.4f] %s", score, doc.text)

    # --- 3. Sub-query decomposition ---
    log.info("=== Sub-query decomposition ===")
    complex_q = "compare BM25 and dense retrieval for short technical queries"
    sub_queries = decompose_query(complex_q)
    all_ids: dict[int, float] = {}
    for sq in sub_queries:
        for doc, score in retriever.retrieve(sq, top_k=5):
            all_ids[doc.doc_id] = max(all_ids.get(doc.doc_id, 0), score)
    merged = sorted(all_ids.items(), key=lambda x: x[1], reverse=True)
    log.info("Merged results from %d sub-queries:", len(sub_queries))
    for doc_id, score in merged[:5]:
        log.info("  [%.4f] %s", score, docs[doc_id].text)

    # --- 4. Top-k recall analysis ---
    log.info("=== Recall@k analysis ===")
    relevant = {2, 3, 4, 8}  # known relevant retrieval docs
    recall = marginal_recall(retriever, "semantic search embeddings", relevant, max_k=10)
    for k, r in sorted(recall.items()):
        log.info("  Recall@%d = %.2f", k, r)


if __name__ == "__main__":
    main()
```

## Exercises

1. **Compound filter builder.** Implement a `FilterBuilder` class with chainable `.eq(field, val)`, `.gte(field, val)`, `.lte(field, val)`, `.build() -> Filter` methods. Use it to filter "agents category, year >= 2024".
2. **HyDE stub.** Write `hyde_query(question: str, generate_fn: Callable[[str], str]) -> str` that calls `generate_fn` (a mock) with a "write a short paragraph answering: {question}" prompt and returns the generated text for embedding. Write a pytest test with a mock `generate_fn`.
3. **Multi-query deduplication.** Modify the sub-query demo to merge results using RRF instead of max-score. Does the top-3 change?
4. **Dynamic top-k.** Implement `auto_top_k(score_list: list[float], threshold: float = 0.5) -> int` that returns the smallest k where score[k] / score[0] < threshold (score drop-off). Apply it to a retrieval result and log the chosen k.
5. **Filter + hybrid.** Extend the `HybridRetriever` from day-4 to accept an optional `filter_fn: Filter`. Apply it to both BM25 and dense retrieval legs. Write a test asserting no document outside the filter appears in results.

(Solutions: ../solutions/week-04/day-5.md)

## Self-check

- [ ] I can implement pre-filter, post-filter, and explain their tradeoffs.
- [ ] I can build composable filter predicates (eq, gte, and, or) in Python.
- [ ] I can explain HyDE and when it helps (high-specificity questions with sparse corpora).
- [ ] I understand the marginal-recall curve and why k > 10 often adds little value.
- [ ] I can implement query expansion and sub-query decomposition.

## Supplementary links

- [HyDE paper (Gao et al., 2022)](https://arxiv.org/abs/2212.10496)
- [Qdrant metadata filtering](https://qdrant.tech/documentation/concepts/filtering/)
- [LangChain self-query retriever](https://python.langchain.com/docs/how_to/self_query/)
