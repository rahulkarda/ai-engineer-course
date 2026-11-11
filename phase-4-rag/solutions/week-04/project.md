# Solutions — Week 04 Project (Hybrid Retriever System)

This is the reference solution for the week-4 project. Compare against your own implementation. Key design decisions are called out.

## Self-check against acceptance criteria

Before reading further, verify your own implementation against each criterion:

- [ ] `pytest tests/` passes with zero failures (≥ 25 tests across 5 files)
- [ ] `python scripts/compare_retrievers.py` prints a results table
- [ ] Hybrid MRR ≥ max(BM25, Dense) for ≥ 3/5 eval queries
- [ ] All source files: type hints, `logging`, no hardcoded secrets
- [ ] No module exceeds 300 lines
- [ ] `RetrieverConfig` uses Pydantic with field validation

---

## Architecture notes

### Why the inverted index maps `doc_id → freq` rather than a list of `(doc_id, freq)` tuples

A dict (`{doc_id: freq}`) gives O(1) lookup when checking if a candidate doc contains a term. A list requires O(n) scan. For top-k retrieval the dict is strictly better.

### Why `candidate_k >> top_k`

Each retriever returns `candidate_k` results. The union can be up to `2 × candidate_k` documents. RRF then re-ranks this pool. If `candidate_k = top_k = 5`, the hybrid has no diversity to improve upon. Setting `candidate_k = 50` gives the fusion step material to work with.

### Why `save` uses a directory, not a single file

FAISS requires writing a binary index file. The BM25 index is JSON. They cannot be combined in one file without custom serialisation. A directory is the natural container:

```
save_dir/
├── bm25.json
├── dense.faiss
├── dense_docs.json
└── config.json
```

---

## Reference hybrid.py (abbreviated)

```python
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from hybrid_retriever.models import Document, RetrieverConfig, SearchResult
from hybrid_retriever.bm25 import BM25Retriever
from hybrid_retriever.dense import SemanticRetriever
from hybrid_retriever.fusion import reciprocal_rank_fusion, score_fusion

log = logging.getLogger(__name__)


@dataclass
class HybridRetriever:
    config: RetrieverConfig = field(default_factory=RetrieverConfig)
    _bm25: BM25Retriever = field(init=False)
    _dense: SemanticRetriever = field(init=False)

    def __post_init__(self) -> None:
        self._bm25 = BM25Retriever(self.config)
        self._dense = SemanticRetriever(self.config)

    def add_documents(self, docs: list[Document]) -> None:
        self._bm25.add_documents(docs)
        self._dense.add_documents(docs)
        log.info("Indexed %d documents.", len(docs))

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter_fn: Callable[[Document], bool] | None = None,
        fusion: Literal["rrf", "score"] = "rrf",
    ) -> list[SearchResult]:
        k = top_k or self.config.top_k
        ck = self.config.candidate_k

        bm25_results = self._bm25.retrieve(query, top_k=ck, filter_fn=filter_fn)
        dense_results = self._dense.retrieve(query, top_k=ck, filter_fn=filter_fn)

        if fusion == "rrf":
            fused = reciprocal_rank_fusion(
                [bm25_results, dense_results], k=self.config.rrf_k
            )
        else:
            fused = score_fusion([bm25_results, dense_results])

        # Rebuild SearchResult from fused [(doc_id, score)] list
        all_docs: dict[str, Document] = {
            r.document.doc_id: r.document
            for r in bm25_results + dense_results
        }
        results = [
            SearchResult(document=all_docs[doc_id], score=score, retriever="hybrid")
            for doc_id, score in fused[:k]
            if doc_id in all_docs
        ]
        return results

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._bm25.save(directory / "bm25.json")
        self._dense.save(directory / "dense")
        (directory / "config.json").write_text(self.config.model_dump_json(indent=2))
        log.info("Saved HybridRetriever to %s", directory)

    @classmethod
    def load(cls, directory: Path) -> "HybridRetriever":
        directory = Path(directory)
        config = RetrieverConfig.model_validate_json(
            (directory / "config.json").read_text()
        )
        obj = cls(config=config)
        obj._bm25 = BM25Retriever.load(directory / "bm25.json")
        obj._dense = SemanticRetriever.load(directory / "dense", config)
        log.info("Loaded HybridRetriever from %s", directory)
        return obj
```

## Common mistakes to check for

1. **IDF recalculated per-query vs. cached** — recomputing for every query is slow at scale. Cache as a property invalidated on `add_documents`.
2. **Missing `normalize_embeddings=True`** in the dense retriever — produces incorrect cosine similarity via `IndexFlatIP`.
3. **JSON key type mismatch** — inverted index keys are strings in JSON, must be cast to `int` on load.
4. **`filter_fn` applied to dense retriever** — dense retriever must also respect the filter, not just BM25.
5. **No mock in tests** — any test that instantiates `SemanticRetriever` without patching `SentenceTransformer` will attempt a network download.

## Expected evaluation results

Hybrid should win or tie on MRR for ≥ 3/5 queries. The "how does RRF work" query is the hardest — dense likely ranks it first because "RRF fuses multiple ranked lists" is the closest semantic match; BM25 might rank it lower if "RRF" is a rare term with high IDF but only one doc has it.

If hybrid underperforms, check:
- `candidate_k` — increase to 100.
- `rrf_k` — try 30 instead of 60 for small corpora.
- Tokenisation — ensure the BM25 tokeniser lowercases consistently.
