# Phase 4 · Week 04 · Day 3 — Implement BM25 from Scratch

> Objective: Build a production-ready BM25 retriever class with incremental indexing and serialisation.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

## Concept

Today moves from theory to implementation. A production BM25 retriever needs:

1. **Inverted index** — map from term → posting list (doc_id, term_frequency pairs).
2. **Incremental add** — append documents without full reindex.
3. **Serialisation** — save/load without re-processing the corpus.
4. **Batch scoring** — score many queries efficiently.
5. **Type safety** — Pydantic for config, dataclasses for internals.

Building it yourself (rather than using `rank-bm25`) cements the formula, lets you customise tokenisation, and is a common interview task.

## How it works internally

### Inverted index layout

```
term → [(doc_id, freq), (doc_id, freq), ...]
```

At query time, intersect/union posting lists for query terms, compute BM25, accumulate per doc_id, sort.

### Incremental indexing

Track `doc_lengths[]` and a running sum for `avgdl`. When a new document is added:
- Tokenize → update inverted index with new `doc_id`.
- Append to `doc_lengths`.
- Recompute `avgdl` (cheap: O(1) with running sum).
- **IDF changes** — recompute lazily (on next query) or eagerly.

For simplicity we recompute IDF eagerly on each `add_document`. For large corpora, batch with `add_documents(batch)`.

### Serialisation

Store as JSON: `{docs, doc_lengths, inverted_index, config}`. Large corpora could use msgpack or pickle, but JSON is debuggable.

## Worked example

```python
"""
day3_bm25_from_scratch.py — production-ready BM25 retriever.

Run: python day3_bm25_from_scratch.py
Deps: uv add pydantic   (no rank-bm25)
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class BM25Config(BaseModel):
    k1: float = Field(default=1.5, gt=0, description="TF saturation parameter")
    b: float = Field(default=0.75, ge=0, le=1, description="Length normalisation parameter")
    lowercase: bool = True
    strip_punctuation: bool = True


# ---------------------------------------------------------------------------
# BM25 retriever
# ---------------------------------------------------------------------------
@dataclass
class BM25Retriever:
    config: BM25Config = field(default_factory=BM25Config)
    # internals
    _docs: list[str] = field(default_factory=list, init=False)
    _doc_lengths: list[int] = field(default_factory=list, init=False)
    _total_len: int = field(default=0, init=False)
    # term → {doc_id: freq}
    _inverted: dict[str, dict[int, int]] = field(
        default_factory=lambda: defaultdict(dict), init=False
    )

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------
    def _tokenize(self, text: str) -> list[str]:
        if self.config.lowercase:
            text = text.lower()
        if self.config.strip_punctuation:
            text = re.sub(r"[^a-z0-9\s]", " ", text)
        return text.split()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def add_document(self, text: str) -> int:
        """Append one document; return its doc_id."""
        doc_id = len(self._docs)
        self._docs.append(text)
        tokens = self._tokenize(text)
        self._doc_lengths.append(len(tokens))
        self._total_len += len(tokens)
        for term, freq in Counter(tokens).items():
            self._inverted[term][doc_id] = freq
        log.debug("Indexed doc_id=%d (%d tokens)", doc_id, len(tokens))
        return doc_id

    def add_documents(self, texts: list[str]) -> list[int]:
        return [self.add_document(t) for t in texts]

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------
    @property
    def _avgdl(self) -> float:
        n = len(self._docs)
        return self._total_len / n if n else 1.0

    def _idf(self, term: str) -> float:
        n = len(self._docs)
        df = len(self._inverted.get(term, {}))
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def _score_term(self, term: str, doc_id: int) -> float:
        f = self._inverted.get(term, {}).get(doc_id, 0)
        if f == 0:
            return 0.0
        k1, b = self.config.k1, self.config.b
        dl = self._doc_lengths[doc_id]
        tf_norm = f * (k1 + 1) / (f + k1 * (1 - b + b * dl / self._avgdl))
        return self._idf(term) * tf_norm

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def retrieve(
        self, query: str, top_k: int = 5
    ) -> list[tuple[int, float, str]]:
        """Return list of (doc_id, score, text) sorted descending."""
        query_terms = self._tokenize(query)
        # gather candidate doc_ids from posting lists
        candidates: set[int] = set()
        for term in query_terms:
            candidates.update(self._inverted.get(term, {}).keys())

        scores: dict[int, float] = {}
        for doc_id in candidates:
            scores[doc_id] = sum(self._score_term(t, doc_id) for t in query_terms)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(doc_id, score, self._docs[doc_id]) for doc_id, score in ranked]

    def retrieve_batch(
        self, queries: list[str], top_k: int = 5
    ) -> list[list[tuple[int, float, str]]]:
        return [self.retrieve(q, top_k) for q in queries]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        payload: dict[str, Any] = {
            "config": self.config.model_dump(),
            "docs": self._docs,
            "doc_lengths": self._doc_lengths,
            "total_len": self._total_len,
            # JSON requires string keys; cast int → str
            "inverted": {
                term: {str(did): freq for did, freq in postings.items()}
                for term, postings in self._inverted.items()
            },
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info("Saved index to %s (%d docs)", path, len(self._docs))

    @classmethod
    def load(cls, path: str | Path) -> "BM25Retriever":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = BM25Config(**payload["config"])
        obj = cls(config=config)
        obj._docs = payload["docs"]
        obj._doc_lengths = payload["doc_lengths"]
        obj._total_len = payload["total_len"]
        obj._inverted = defaultdict(
            dict,
            {
                term: {int(did): freq for did, freq in postings.items()}
                for term, postings in payload["inverted"].items()
            },
        )
        log.info("Loaded index from %s (%d docs)", path, len(obj._docs))
        return obj

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._docs)

    def __repr__(self) -> str:
        return (
            f"BM25Retriever(docs={len(self._docs)}, "
            f"vocab={len(self._inverted)}, "
            f"config={self.config})"
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
CORPUS = [
    "Python is a high-level, interpreted programming language.",
    "FastAPI is a modern web framework for building APIs with Python.",
    "Transformers are a neural network architecture based on self-attention.",
    "Retrieval-Augmented Generation combines retrieval systems with language models.",
    "BM25 is a probabilistic ranking function used in information retrieval.",
    "Dense embeddings encode semantic meaning into fixed-length float vectors.",
    "FAISS provides efficient similarity search over large sets of dense vectors.",
    "A vector database stores embeddings and supports approximate nearest-neighbor queries.",
    "Fine-tuning adapts a pre-trained language model to a specific downstream task.",
    "Prompt engineering crafts input instructions to guide LLM behaviour.",
]


def main() -> None:
    retriever = BM25Retriever()
    retriever.add_documents(CORPUS)
    log.info("Index: %r", retriever)

    queries = [
        "probabilistic ranking retrieval",
        "Python web framework",
        "vector similarity search",
    ]
    for q in queries:
        log.info("Query: %s", q)
        for doc_id, score, text in retriever.retrieve(q, top_k=3):
            log.info("  [%.4f] doc=%d %s", score, doc_id, text[:60])

    # round-trip serialisation
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name
    retriever.save(tmp_path)
    reloaded = BM25Retriever.load(tmp_path)
    assert len(reloaded) == len(retriever)
    log.info("Serialisation round-trip OK. doc_count=%d", len(reloaded))
    os.unlink(tmp_path)


if __name__ == "__main__":
    main()
```

## Exercises

1. **Custom tokeniser.** Add support for stemming via a simple suffix-stripping function (`"running" → "run"`, `"indexing" → "index"`). Does retrieval quality improve or degrade for "indexing vectors"?
2. **Multi-field BM25.** Extend the retriever to accept `{"title": str, "body": str}` documents. Score title terms with weight 2× body terms. Hint: index title and body separately, then combine.
3. **Inverted index stats.** Add a method `vocabulary_stats() -> dict` returning: `vocab_size`, `avg_posting_len`, `top_10_terms` (by document frequency). Call it after indexing CORPUS.
4. **Stress test.** Generate 10 000 random short documents (each 5–15 words drawn from a 500-word vocabulary). Benchmark `add_documents` vs. per-document adds using `time.perf_counter`. What's the bottleneck?
5. **Test suite.** Write a full pytest test file (`tests/test_bm25.py`) with at least 6 tests covering: indexing, retrieve top-1, retrieve returns ≤ top_k results, empty query returns empty list, save/load round-trip, score > 0 for exact term match.

(Solutions: ../solutions/week-04/day-3.md)

## Self-check

- [ ] I can implement an inverted index with incremental document adds.
- [ ] I can serialise and deserialise the index to/from JSON.
- [ ] I understand how IDF changes as documents are added.
- [ ] I can explain TF saturation with a numeric example (k₁=1.5, f=1 vs f=10).
- [ ] My BM25 implementation passes all 6 unit tests without reference.

## Supplementary links

- [Original Okapi BM25 paper (Robertson et al., 1994)](https://trec.nist.gov/pubs/trec3/papers/city.ps.gz)
- [rank-bm25 source (reference implementation)](https://github.com/dorianbrown/rank_bm25/blob/master/rank_bm25.py)
