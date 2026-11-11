# Phase 4 · Week 03 · Day 7 (Sunday) — Project & Review

> Objective: Complete and polish the week project; review all concepts; prepare for assessment.  ·  Time: ~3 hr project + 1 hr review  ·  Prereqs: [[day-6]]

## Today's agenda

| Time | Task |
|------|------|
| 0:00–0:30 | Review any open items from Day 6 build |
| 0:30–1:30 | Extend the project with two of the extension tasks below |
| 1:30–2:30 | Write a short "RAG integration" wrapper that uses your vector store |
| 2:30–3:00 | Week concept review |
| 3:00–4:00 | Assessment preparation |

## Project extension tasks (pick two)

### Extension A — pgvector backend

Add `PgVectorStore` to the factory. It should:
- Accept `DATABASE_URL` from `os.getenv("DATABASE_URL")`
- On init, `CREATE EXTENSION IF NOT EXISTS vector` and `CREATE TABLE IF NOT EXISTS`
- Implement `upsert` using `INSERT ... ON CONFLICT DO UPDATE`
- Implement `query` using `ORDER BY embedding <=> %s::vector LIMIT k`
- Skip gracefully (log a warning) if `DATABASE_URL` is not set

### Extension B — RAG integration stub

Create `src/rag/retriever.py`:

```python
"""Minimal RAG retriever that wraps VectorStore."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

from src.vector_store.base import SearchResult, VectorRecord, VectorStore
from src.vector_store.factory import create_store

log = logging.getLogger(__name__)


@dataclass
class Document:
    id: str
    content: str
    metadata: dict


class Retriever:
    """
    Wraps a VectorStore to support:
    - index(docs, embed_fn) — embed and upsert documents
    - retrieve(query, k, filter) — embed query and search
    """

    def __init__(self, store: VectorStore) -> None:
        self._store = store

    def index(
        self,
        docs: list[Document],
        embed_fn,   # callable: list[str] → np.ndarray of shape (n, d)
    ) -> None:
        texts = [d.content for d in docs]
        embeddings = embed_fn(texts)
        records = [
            VectorRecord(id=d.id, vector=embeddings[i], metadata={**d.metadata, "content": d.content})
            for i, d in enumerate(docs)
        ]
        self._store.upsert(records)
        log.info("Indexed %d documents", len(records))

    def retrieve(
        self,
        query: str,
        embed_fn,
        k: int = 5,
        filter: dict | None = None,
    ) -> list[SearchResult]:
        q_vec = embed_fn([query])[0]
        return self._store.query(q_vec, k=k, filter=filter)
```

Test it with a dummy embed function (random vectors):

```python
import numpy as np
from src.rag.retriever import Document, Retriever
from src.vector_store.factory import create_store

DIM = 32

def dummy_embed(texts: list[str]) -> np.ndarray:
    rng = np.random.default_rng(sum(hash(t) for t in texts) % 2**32)
    return rng.standard_normal((len(texts), DIM)).astype(np.float32)

store = create_store("memory", dim=DIM)
ret = Retriever(store)
docs = [Document(id=f"d{i}", content=f"text {i}", metadata={"src": "test"}) for i in range(20)]
ret.index(docs, dummy_embed)
results = ret.retrieve("query text", dummy_embed, k=3)
print([r.id for r in results])
```

### Extension C — Benchmark script

Create `scripts/benchmark.py` that:
1. Builds an index with 50 000 vectors at DIM=128 for each backend
2. Runs 100 queries
3. Reports median latency and recall\@10 vs flat exact search
4. Outputs a markdown table

## Week concept review

### Core vocabulary

| Term | Definition |
|------|-----------|
| ANN | Approximate nearest-neighbor; trades recall for speed |
| Recall\@k | Fraction of exact top-k results found by ANN |
| IVF | Inverted file index; Voronoi cell partitioning |
| nlist | Number of IVF cells (√n to 4√n is typical) |
| nprobe | IVF cells probed at query time |
| HNSW | Hierarchical navigable small world graph |
| M | HNSW edges per node per layer |
| efConstruction | HNSW build-time beam width |
| efSearch | HNSW query-time beam width |
| PQ | Product quantization; compresses vectors for scale |
| IDMap | FAISS wrapper for custom int64 IDs |
| Metadata filter | Restrict search to a subset of records by attributes |
| Pre-filter | Filter before ANN (correct but can hurt recall on small sets) |
| Post-filter | Filter after ANN (simple but wastes compute) |

### Decision tree

```
Need vector search?
  ├─ n < 50k AND correctness critical → Flat (exact)
  ├─ n < 1M AND streaming inserts → HNSW
  ├─ n > 1M AND batch-loaded AND RAM tight → IVF + PQ
  ├─ Already on Postgres → pgvector
  ├─ Prototyping / local dev → Chroma
  └─ High-throughput production microservice → Qdrant / Weaviate
```

### Common mistakes

1. **Forgetting to normalise** before inner-product search → results are wrong
2. **Building IVF without training** → `assert index.is_trained` catches this
3. **Setting efSearch < k** → HNSW silently returns fewer results
4. **Post-filtering too aggressively** → k results may be < k
5. **Not persisting the ID map** alongside the FAISS index → IDs are lost on reload

## Assessment preparation

Work through these without looking at solutions:

**Conceptual**
- Explain why KD-trees degrade in high dimensions
- What does recall\@10 = 0.95 mean in practice?
- When would you choose IVF over HNSW?

**Coding**
- From memory, write the 5 lines needed to build, add to, and query an `IndexHNSWFlat` in FAISS
- Write a Chroma `where` clause for: "category is 'legal' AND year >= 2023"

**Debugging**
- A FAISS `IndexIVFFlat` search returns all -1 IDs. What is the likely cause?
- Chroma returns 0 results for a filtered query that should have matches. What do you check?

## Self-check

- [ ] I finished the Day 6 project with at least 2 backends working and passing tests
- [ ] I completed at least one extension task
- [ ] I can state the IVF and HNSW trade-offs without notes
- [ ] I feel confident about the assessment

## Supplementary

- [ANN Benchmarks — compare HNSW vs IVF](https://ann-benchmarks.com/)
- [FAISS wiki — FAQ](https://github.com/facebookresearch/faiss/wiki/FAQ)
- [Chroma cookbook](https://docs.trychroma.com/guides)
