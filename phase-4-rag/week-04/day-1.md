# Phase 4 · Week 04 · Day 1 — Semantic (Dense) Retrieval

> Objective: Understand dense vector retrieval and implement a semantic search pipeline end-to-end.  ·  Time: ~1 hr  ·  Prereqs: [[week-03/day-7]]

## Concept

**Semantic retrieval** finds documents by meaning, not exact word overlap. A neural encoder (embedding model) maps text into a high-dimensional vector space where semantically similar texts land close together. At query time the query is embedded and the nearest document vectors are returned.

Key characteristics:
- **Dense vectors** — every dimension carries signal (contrast: sparse BM25 vectors are mostly zero).
- **Approximate nearest-neighbor (ANN)** — exact k-NN is O(n·d); ANN structures (HNSW, IVF) trade tiny accuracy loss for sub-linear speed.
- **Recall vs. precision tradeoff** — semantic retrieval has high recall across paraphrases and synonyms; it can miss exact-keyword matches on rare technical terms.

### When to use semantic retrieval alone

- Queries are natural-language questions (paraphrases, synonyms common).
- Corpus language is messy or multilingual.
- Documents are long-form prose, not short keyword-rich fields.

## How it works internally

```
Query text
    │
    ▼  encoder (e.g. sentence-transformers, OpenAI text-embedding-3-small)
Query vector  q ∈ ℝ^d
    │
    ▼  cosine similarity  (or dot-product if vectors are L2-normalised)
     score(q, dᵢ) = (q · dᵢ) / (‖q‖ ‖dᵢ‖)
    │
    ▼  ANN index (HNSW in FAISS / ChromaDB / Qdrant)
Top-k document IDs  →  fetch payloads
```

**HNSW (Hierarchical Navigable Small World):** layered graph where each node connects to its nearest neighbors. Search starts at the top (sparse) layer and greedily descends. O(log n) per query; build is O(n log n).

**Cosine vs. dot-product:** cosine normalises magnitude — use it when documents vary greatly in length. Dot-product rewards both direction *and* magnitude; used when embeddings are already unit-normalised (same result then).

## Worked example

```python
"""
day1_semantic_retrieval.py — semantic search over a small corpus.

Run:  python day1_semantic_retrieval.py
Deps: uv add sentence-transformers faiss-cpu numpy
"""

import logging
import os
from dataclasses import dataclass, field

import faiss  # type: ignore[import-untyped]
import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
CORPUS: list[str] = [
    "Python is a high-level, interpreted programming language.",
    "FastAPI is a modern web framework for building APIs with Python.",
    "Transformers are a type of neural network architecture based on attention.",
    "Retrieval-Augmented Generation combines retrieval systems with LLMs.",
    "BM25 is a probabilistic keyword-based ranking function.",
    "Dense embeddings encode semantic meaning into fixed-length vectors.",
    "FAISS is a library for efficient similarity search over dense vectors.",
    "A vector database stores embeddings and supports ANN queries.",
    "Fine-tuning adapts a pre-trained model to a specific task.",
    "Prompt engineering crafts inputs to elicit desired LLM behaviour.",
]


@dataclass
class SemanticRetriever:
    model_name: str = "all-MiniLM-L6-v2"
    _model: SentenceTransformer = field(init=False, repr=False)
    _index: faiss.IndexFlatIP = field(init=False, repr=False)  # inner-product (unit vecs → cosine)
    _docs: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._model = SentenceTransformer(self.model_name)
        self._index = faiss.IndexFlatIP(self._model.get_sentence_embedding_dimension())
        log.info("Retriever ready. Embedding dim=%d", self._model.get_sentence_embedding_dimension())

    # ------------------------------------------------------------------
    def index_documents(self, docs: list[str]) -> None:
        self._docs = docs
        embeddings: np.ndarray = self._model.encode(
            docs, normalize_embeddings=True, show_progress_bar=False
        )
        self._index.add(embeddings)
        log.info("Indexed %d documents.", len(docs))

    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        q_vec: np.ndarray = self._model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )
        scores, indices = self._index.search(q_vec, top_k)
        results = [
            (self._docs[idx], float(scores[0][rank]))
            for rank, idx in enumerate(indices[0])
        ]
        return results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def main() -> None:
    retriever = SemanticRetriever()
    retriever.index_documents(CORPUS)

    queries = [
        "How do transformers process sequences?",
        "fast web framework Python",
        "keyword search ranking algorithm",
    ]

    for q in queries:
        log.info("Query: %s", q)
        for doc, score in retriever.retrieve(q, top_k=3):
            log.info("  [%.4f] %s", score, doc)


if __name__ == "__main__":
    main()
```

### What to observe

Run it and note:
- "How do transformers process sequences?" returns the transformers + RAG docs (semantic match), not "FastAPI".
- "fast web framework Python" returns the FastAPI doc even though "fast" ≠ "FastAPI" (substring, but the semantic overlap is strong enough).
- "keyword search ranking algorithm" surfaces the BM25 doc — good semantic recall.

## Exercises

1. **Swap the encoder.** Replace `all-MiniLM-L6-v2` with `BAAI/bge-small-en-v1.5`. Compare top-3 results for the same three queries. Do rankings change?
2. **Add documents at runtime.** Implement an `add_document(text: str) -> None` method that encodes and appends a new vector to the index without rebuilding it. Write a pytest test asserting the document count grows.
3. **Cosine vs. dot-product.** Remove `normalize_embeddings=True` and switch to `faiss.IndexFlatL2`. Re-run queries. Do results degrade? Explain why.
4. **Batch retrieval.** Refactor `retrieve` to accept `list[str]` queries and return `list[list[tuple[str, float]]]`. Run all three demo queries in a single encoder call.
5. **Persist the index.** Add `save(path: str)` and `load(path: str)` classmethods using `faiss.write_index` / `faiss.read_index` + `json` for the doc list. Write a test that saves, reloads, and checks a query returns the same top-1.

(Solutions: ../solutions/week-04/day-1.md)

## Self-check

- [ ] I can explain why cosine similarity is preferred over Euclidean distance for text embeddings.
- [ ] I can describe what HNSW does and its time complexity.
- [ ] I can implement a FAISS-backed semantic retriever from scratch without reference.
- [ ] I understand the difference between dense and sparse vector representations.
- [ ] I can articulate two scenarios where semantic retrieval alone is insufficient.

## Supplementary links

- [FAISS wiki](https://github.com/facebookresearch/faiss/wiki)
- [Sentence Transformers docs](https://www.sbert.net/docs/sentence_transformer/usage/usage.html)
- [BEIR benchmark — retrieval evaluation](https://github.com/beir-cellar/beir)
