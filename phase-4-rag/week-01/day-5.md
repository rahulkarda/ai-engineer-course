# Phase 4 · Week 01 · Day 5 — Ingestion Pipeline Architecture

> Objective: Design and implement a stage-based document ingestion pipeline with typed protocols, error isolation, and deduplication.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

## Concept

A production ingestion pipeline is not a single script — it is a sequence of independent, composable stages. Each stage has one job, accepts documents, and returns documents (or raises clearly).

**Canonical stages:**

| Stage | Input | Output | Concerns |
|---|---|---|---|
| Load | file paths / URLs | raw `Document` objects | I/O, format dispatch |
| Extract | raw `Document` | text-only `Document` | parsing, encoding |
| Clean | text `Document` | normalized `Document` | unicode, whitespace, noise |
| Chunk | `Document` | `list[Chunk]` | size strategy, overlap |
| Embed | `list[Chunk]` | `list[Chunk]` with vectors | model I/O, batching |
| Store | `list[Chunk]` | receipts / IDs | vector DB writes |

**Why this decomposition?**

- **Independent testability**: mock one stage, test the rest. No monolith to stub.
- **Replaceability**: swap the PDF extractor without touching the cleaner.
- **Observability**: log timing and error counts per stage, not per file.
- **Idempotency**: if you re-run a pipeline on the same input, the output should be the same. Achieve this with content hashing at the load/dedup stage — already-seen documents never re-enter downstream stages.

**Pipeline vs DAG mental model:**

A linear pipeline is a degenerate DAG (single path). Start with a pipeline; promote to a DAG (LangGraph, Prefect, Airflow) when you need branching, fan-out, or conditional routing. For ingestion pipelines under ~100 k documents, a sequential pipeline is almost always the right level of complexity.

**Error handling strategies:**

| Strategy | When to use |
|---|---|
| **Skip-and-log** | Best-effort pipelines; bad docs are acceptable losses |
| **Fail-fast** | Data integrity is critical; partial output is worse than none |
| **Retry** | Transient failures (network, rate limits); use exponential back-off |

Choose per stage, not per pipeline: load/extract can skip-and-log; embed/store often retry; chunk/clean typically fail-fast (logic bugs, not I/O).

## How it works internally

### Stage Protocol

Python `Protocol` gives structural typing without inheritance. Any class with a matching `process` method satisfies the Protocol — no `register`, no ABC boilerplate.

```python
from typing import Protocol, runtime_checkable
from ingestion.models import Document

@runtime_checkable
class PipelineStage(Protocol):
    """Contract: transform a batch of documents."""

    def process(self, docs: list[Document]) -> list[Document]: ...
```

`@runtime_checkable` lets you use `isinstance(stage, PipelineStage)` in tests or guards.

### Data flow

```
list[Document]
  → Stage A → list[Document]
  → Stage B → list[Document]
  → ...
  → list[Document]  # final output
```

Each stage receives the *full current batch* and returns a (possibly smaller) batch. A stage that filters documents (e.g., deduplication) returns fewer documents than it received. A stage that chunks returns more (if Chunk is a subtype of Document) — or you switch the generic type parameter at that boundary.

### Deduplication via content hashing

```python
import hashlib

def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
```

Store seen hashes in a `set[str]`. At pipeline start, optionally pre-populate from a persistent store (shelve, Redis, DB) to skip documents processed in previous runs — this is what makes the pipeline **resumable**.

### Progress tracking and resumability

- Log at stage entry/exit with document count and elapsed time.
- Persist the set of processed hashes to disk after each successful run.
- On restart, load that set before the dedup stage — previously successful documents are skipped automatically.
- Pair with a dead-letter log (append-only file of failed document paths + error) to retry only failures.

## Worked example

```python
# pipeline.py
from __future__ import annotations

import hashlib
import logging
import time
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------

class Document:
    """Minimal document used across pipeline stages."""

    def __init__(self, text: str, metadata: dict[str, str] | None = None) -> None:
        self.text = text
        self.metadata: dict[str, str] = metadata or {}

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Stage contract
# ---------------------------------------------------------------------------

@runtime_checkable
class PipelineStage(Protocol):
    def process(self, docs: list[Document]) -> list[Document]: ...


# ---------------------------------------------------------------------------
# Concrete stages
# ---------------------------------------------------------------------------

class HashDeduplicator:
    """Remove documents whose content hash has already been seen."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def process(self, docs: list[Document]) -> list[Document]:
        unique: list[Document] = []
        for doc in docs:
            h = doc.content_hash
            if h in self._seen:
                logger.debug("Duplicate skipped: hash=%s", h[:8])
            else:
                self._seen.add(h)
                unique.append(doc)
        logger.info("Dedup: %d → %d documents", len(docs), len(unique))
        return unique


class CleaningStage:
    """Normalize unicode and collapse whitespace (mirrors TextCleaner from day-3)."""

    def process(self, docs: list[Document]) -> list[Document]:
        import unicodedata
        import re

        cleaned: list[Document] = []
        for doc in docs:
            text = unicodedata.normalize("NFC", doc.text)
            text = re.sub(r"\s+", " ", text).strip()
            cleaned.append(Document(text=text, metadata=doc.metadata))
        logger.info("Cleaning: processed %d documents", len(docs))
        return cleaned


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

class Pipeline:
    """Run a sequence of PipelineStage instances, logging timing per stage."""

    def __init__(self, stages: list[PipelineStage]) -> None:
        self._stages = stages

    def run(self, docs: list[Document]) -> list[Document]:
        logger.info("Pipeline start: %d documents, %d stages", len(docs), len(self._stages))
        current = docs
        for stage in self._stages:
            name = type(stage).__name__
            t0 = time.perf_counter()
            current = stage.process(current)
            elapsed = time.perf_counter() - t0
            logger.info("Stage %-25s → %d docs  (%.3fs)", name, len(current), elapsed)
        logger.info("Pipeline complete: %d documents out", len(current))
        return current


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    raw_docs = [
        Document(text="  Hello   world  ", metadata={"source": "a.txt"}),
        Document(text="Hello world", metadata={"source": "b.txt"}),   # duplicate after cleaning
        Document(text="Café au lait’s recipe", metadata={"source": "c.txt"}),
        Document(text="  Hello   world  ", metadata={"source": "d.txt"}),  # exact duplicate
    ]

    pipeline = Pipeline(stages=[CleaningStage(), HashDeduplicator()])
    result = pipeline.run(raw_docs)

    for doc in result:
        print(f"[{doc.metadata.get('source')}] {doc.text!r}")
```

**Expected output (illustrative):**

```
INFO: Pipeline start: 4 documents, 2 stages
INFO: Stage CleaningStage            → 4 docs  (0.000s)
INFO: Stage HashDeduplicator         → 2 docs  (0.000s)
INFO: Pipeline complete: 2 documents out
[a.txt] 'Hello world'
[c.txt] "Café au lait's recipe"
```

Note: `b.txt` and `d.txt` are deduped — after cleaning they match `a.txt`.

## Exercises

1. **Error isolation**: Modify `Pipeline.run` so that a stage raising an exception on a single document does not abort the whole batch. Instead, catch the exception per-document inside each stage, log the error with the document source, and continue. Hint: you may need to change the stage interface or add a wrapper class.

2. **CachingPipeline**: Implement a `CachingPipeline` subclass (or wrapper) that persists processed hashes to a `shelve` file. On construction it loads existing hashes; after `run` it saves the updated set. Verify that running the pipeline twice on the same documents processes them only once.

3. **Pytest stage ordering**: Write a pytest test that creates two mock stages (using `unittest.mock.MagicMock` or a stub class), wires them into a `Pipeline`, calls `run`, and asserts that `stage_1.process` was called before `stage_2.process` and that the output of `stage_1` was passed as the input to `stage_2`.

(Solutions: ../solutions/week-01/day-5.md)

## Self-check

- [ ] Can explain why each ingestion stage should be independently testable and replaceable.
- [ ] Can implement a `PipelineStage` Protocol and verify a class satisfies it with `isinstance`.
- [ ] Can describe the three error-handling strategies (skip-and-log, fail-fast, retry) and choose appropriately per stage.
- [ ] Can implement content-hash-based deduplication and explain how it enables pipeline resumability.
