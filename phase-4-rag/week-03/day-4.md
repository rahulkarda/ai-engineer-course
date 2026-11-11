# Phase 4 · Week 03 · Day 4 — Chroma & pgvector Hands-On

> Objective: Use Chroma for in-process vector storage and pgvector for SQL-native vector search; compare developer experience and trade-offs.  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

## Concept

### Chroma

Chroma is an embedded, open-source vector database written in Python/Rust. Its main selling points:
- Zero-config in-memory or persistent mode (SQLite + parquet backend)
- Built-in metadata filtering (structured queries on attached key-value metadata)
- Native collections with automatic ID management
- Optional client/server mode for multi-process access

Chroma stores vectors, documents (text), metadata, and IDs together in a **collection**. It handles embedding internally if you pass an embedding function, or you can supply pre-computed vectors.

### pgvector

pgvector is a PostgreSQL extension that adds a `vector` column type and two index types:
- `ivfflat` — IVF with flat quantization; requires `ANALYZE` after bulk load
- `hnsw` — HNSW (added in v0.5); better recall and streaming-insert friendly

Advantages of pgvector:
- Unified relational + vector store — join vectors with SQL tables
- ACID transactions, connection pooling, row-level security
- Scales to tens of millions of vectors in a single Postgres instance
- Familiar tooling (psql, SQLAlchemy, Alembic)

Trade-offs: heavier than an embedded store; requires a running Postgres instance.

### When to choose each

| Scenario | Recommendation |
|----------|---------------|
| Local dev, prototyping, notebook | Chroma (persistent or in-memory) |
| Production app already on Postgres | pgvector |
| High-throughput dedicated vector service | Qdrant or Weaviate |
| Benchmarking / offline batch | FAISS |

## How it works internally

### Chroma internals

Chroma v0.4+ uses DuckDB + Parquet for persistent storage and HNSW (via `hnswlib`) for the vector index. Collections are lazy-loaded. Each `add()` call updates both the metadata store and the HNSW graph. The on-disk format is a directory; copying it is sufficient for backup.

### pgvector internals

The `vector` type is stored as a variable-length binary column. An `ivfflat` index clusters vectors into lists (specified by `lists = N`) and uses Postgres' buffer manager for the index pages. At query time `SET ivfflat.probes = N` controls how many lists are scanned.

For HNSW: `CREATE INDEX ... USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64)`.

## Worked example

```python
"""
day4_chroma_pgvector.py — Chroma in-process demo + pgvector connection template.

Install: uv add chromadb numpy psycopg
Chroma demo runs without Postgres.
pgvector section requires: a running Postgres with pgvector extension.
  Set env var: DATABASE_URL=postgresql://user:pass@localhost:5432/dbname
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from chromadb.config import Settings

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

DIM = 64
N = 500


def make_vecs(n: int, d: int, seed: int = 42) -> tuple[np.ndarray, list[str], list[dict]]:
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, d)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = [str(uuid.uuid4()) for _ in range(n)]
    metadata = [{"source": f"doc_{i}", "category": ("science" if i % 2 == 0 else "tech")}
                for i in range(n)]
    return vecs, ids, metadata


# ── Chroma demo ─────────────────────────────────────────────────────────────

def demo_chroma_inmemory(vecs: np.ndarray, ids: list[str], meta: list[dict]) -> None:
    client = chromadb.Client(Settings(anonymized_telemetry=False))
    col = client.create_collection(name="demo", metadata={"hnsw:space": "cosine"})

    # Add in one shot (Chroma accepts list[list[float]])
    col.add(
        embeddings=vecs.tolist(),
        ids=ids,
        metadatas=meta,
    )
    log.info("Chroma in-memory collection count: %d", col.count())

    query_vec = vecs[0].tolist()
    results = col.query(
        query_embeddings=[query_vec],
        n_results=5,
        include=["distances", "metadatas"],
    )
    log.info("Top-5 ids: %s", results["ids"][0])
    log.info("Top-5 distances: %s", [f"{d:.4f}" for d in results["distances"][0]])


def demo_chroma_persistent(
    vecs: np.ndarray,
    ids: list[str],
    meta: list[dict],
    persist_dir: Path,
) -> None:
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    # get_or_create is idempotent across restarts
    col = client.get_or_create_collection(
        name="persistent_demo",
        metadata={"hnsw:space": "cosine"},
    )
    if col.count() == 0:
        col.add(embeddings=vecs.tolist(), ids=ids, metadatas=meta)
        log.info("Inserted %d vectors into persistent Chroma", col.count())
    else:
        log.info("Loaded existing persistent collection, count=%d", col.count())

    # Metadata filter: only 'science' category
    filtered = col.query(
        query_embeddings=[vecs[0].tolist()],
        n_results=5,
        where={"category": {"$eq": "science"}},
        include=["metadatas"],
    )
    log.info("Filtered (science) top-5 ids: %s", filtered["ids"][0])


# ── pgvector demo (requires DATABASE_URL) ──────────────────────────────────

def demo_pgvector(vecs: np.ndarray, ids: list[str]) -> None:
    """
    Demonstrates pgvector usage. Skipped if DATABASE_URL is not set.
    Run against a local Postgres with pgvector:
      docker run -e POSTGRES_PASSWORD=pass -p 5432:5432 pgvector/pgvector:pg16
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        log.warning("DATABASE_URL not set — skipping pgvector demo")
        return

    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError:
        log.warning("psycopg not installed — skipping pgvector demo")
        return

    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("DROP TABLE IF EXISTS embeddings")
        conn.execute(
            f"CREATE TABLE embeddings (id TEXT PRIMARY KEY, embedding vector({DIM}))"
        )
        # Bulk insert
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO embeddings (id, embedding) VALUES (%s, %s::vector)",
                [(id_, vec.tolist()) for id_, vec in zip(ids[:100], vecs[:100])],
            )
        conn.commit()
        log.info("Inserted 100 rows into pgvector table")

        # Create HNSW index
        conn.execute(
            "CREATE INDEX ON embeddings USING hnsw (embedding vector_cosine_ops) "
            "WITH (m = 16, ef_construction = 64)"
        )
        conn.commit()
        log.info("HNSW index created")

        # Query nearest 5
        q = vecs[0].tolist()
        rows = conn.execute(
            "SELECT id, 1 - (embedding <=> %s::vector) AS cosine_sim "
            "FROM embeddings ORDER BY embedding <=> %s::vector LIMIT 5",
            (q, q),
        ).fetchall()
        for row in rows:
            log.info("  id=%s  cosine_sim=%.4f", row["id"], row["cosine_sim"])


def main() -> None:
    vecs, ids, meta = make_vecs(N, DIM)
    persist_dir = Path("/tmp/chroma_demo_week03")
    persist_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== Chroma in-memory ===")
    demo_chroma_inmemory(vecs, ids, meta)

    log.info("=== Chroma persistent ===")
    demo_chroma_persistent(vecs, ids, meta, persist_dir)

    log.info("=== pgvector ===")
    demo_pgvector(vecs, ids)


if __name__ == "__main__":
    main()
```

Run: `uv run python day4_chroma_pgvector.py`

## Exercises

1. Add a `upsert` to the Chroma persistent collection: call `col.upsert(embeddings=[...], ids=[existing_id], metadatas=[{"category": "updated"}])` and verify the metadata changed on re-query.
2. Delete a vector from the Chroma collection by ID and verify `col.count()` decrements.
3. Query a Chroma collection with a compound metadata filter: `{"$and": [{"category": {"$eq": "science"}}, {"source": {"$ne": "doc_0"}}]}`. How many results come back?
4. (pgvector) Write the SQL to create an `ivfflat` index on `embeddings` with `lists=50` for cosine distance. When must you run `VACUUM ANALYZE` after bulk load?
5. Compare Chroma vs pgvector: write a 100-word comparison covering persistence, metadata filtering, and production readiness. No code needed.

(Solutions: ../solutions/week-03/day-4.md)

## Self-check

- [ ] I can create, populate, and query a Chroma collection with metadata filters
- [ ] I can explain the difference between `client.Client()` and `PersistentClient()`
- [ ] I can write the SQL DDL and DML for pgvector insert and cosine-similarity search
- [ ] I know when pgvector beats a dedicated vector DB and vice versa

## Supplementary

- [Chroma docs](https://docs.trychroma.com/)
- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [pgvector HNSW announcement](https://github.com/pgvector/pgvector/releases/tag/v0.5.0)
- [Postgres with pgvector Docker image](https://hub.docker.com/r/pgvector/pgvector)
