# Solutions — Phase 4 · Week 01 · Day 5

## Exercise 1 — Error Isolation

Add per-document try/except so one bad document does not abort the entire pipeline.

```python
"""
solutions/week-01/day-5-ex1.py

Pipeline with per-document error isolation.
Assumes the PipelineStage Protocol and Document/DocumentMetadata models
from the day-5 lesson are importable from doc_ingestion.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from doc_ingestion.models import Document

logger = logging.getLogger(__name__)


@runtime_checkable
class PipelineStage(Protocol):
    """A single processing step in the ingestion pipeline."""

    def process(self, documents: list[Document]) -> list[Document]:
        ...


class Pipeline:
    """Orchestrates a sequence of PipelineStage instances.

    Error isolation: if any stage raises on a document it is removed from the
    batch and processing continues with the remaining documents.
    """

    def __init__(self, stages: list[PipelineStage]) -> None:
        self._stages = stages

    def run(self, documents: list[Document]) -> list[Document]:
        """Run all stages; isolate per-document errors at each stage.

        Args:
            documents: Initial list of raw Document objects.

        Returns:
            Documents that survived every stage without error.
        """
        current: list[Document] = list(documents)

        for stage in self._stages:
            survived: list[Document] = []
            for doc in current:
                try:
                    result = stage.process([doc])
                    survived.extend(result)
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Stage %s failed on document %s — skipping. Error: %s",
                        type(stage).__name__,
                        doc.metadata.source_path,
                        exc,
                    )
            current = survived
            logger.info(
                "Stage %s complete: %d/%d documents survived.",
                type(stage).__name__,
                len(current),
                len(documents),
            )

        return current
```

Key points:
- Each document is processed individually so a single failure only drops that document.
- The original list size is captured before the loop so the log message always shows `survived/total`.
- `except Exception` is intentional here: loaders can raise `OSError`, `UnicodeDecodeError`, `ValueError`, etc. Log the full error; do not swallow silently.

---

## Exercise 2 — CachingPipeline

Skip documents whose content has already been processed in a previous run by persisting a SHA-256 content-hash cache with `shelve`.

```python
"""
solutions/week-01/day-5-ex2.py

CachingPipeline — idempotent re-runs via SHA-256 content hashing.
"""

from __future__ import annotations

import hashlib
import logging
import shelve
from pathlib import Path

from doc_ingestion.models import Document
from day_5_ex1 import Pipeline, PipelineStage  # reuse Pipeline from Ex 1

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_PATH = Path(".pipeline_cache")


def _content_hash(doc: Document) -> str:
    """Return hex SHA-256 of the document's text content."""
    return hashlib.sha256(doc.text.encode("utf-8")).hexdigest()


class CachingPipeline(Pipeline):
    """Pipeline that skips already-processed documents across runs.

    Cache is stored as a shelve database at *cache_path* so it survives
    process restarts.  The cache key is the SHA-256 of the document text;
    the value is a boolean sentinel.

    Args:
        stages:     Ordered list of PipelineStage instances.
        cache_path: Base path for the shelve file (no extension needed).
    """

    def __init__(
        self,
        stages: list[PipelineStage],
        cache_path: Path = _DEFAULT_CACHE_PATH,
    ) -> None:
        super().__init__(stages)
        self._cache_path = str(cache_path)

    def run(self, documents: list[Document]) -> list[Document]:
        """Filter already-cached documents before delegating to Pipeline.run.

        Args:
            documents: Incoming documents (may include duplicates from prior runs).

        Returns:
            Documents that were newly processed this run.
        """
        with shelve.open(self._cache_path) as cache:
            new_docs: list[Document] = []
            for doc in documents:
                h = _content_hash(doc)
                if h in cache:
                    logger.debug(
                        "Skipping cached document: %s (hash=%s)",
                        doc.metadata.source_path,
                        h[:12],
                    )
                else:
                    new_docs.append(doc)

            logger.info(
                "Cache filter: %d new / %d total documents.",
                len(new_docs),
                len(documents),
            )

            processed = super().run(new_docs)

            # Record successfully processed hashes so they are skipped next run.
            for doc in processed:
                cache[_content_hash(doc)] = True

        return processed
```

Design notes:
- `shelve` is stdlib — no extra dependencies.
- The cache is written *after* `super().run()` succeeds, so a crash mid-pipeline does not mark the document as done.
- To invalidate, delete the shelve files (`*.db` / `*.dir` / `*.bak` depending on the backend).
- For in-memory-only use (e.g. within a single process) replace `shelve.open` with a plain `set` stored on `self`.

---

## Exercise 3 — pytest for Stage Call Order

Verify that stages are called in the correct order using `unittest.mock`.

```python
"""
solutions/week-01/day-5-ex3.py  (run as a test file: pytest day-5-ex3.py)

Tests that Pipeline.run calls stages in the declared order.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from doc_ingestion.models import Document, DocumentMetadata
from day_5_ex1 import Pipeline  # adjust import to match your project layout


def _make_doc(text: str = "hello world", source: str = "test.txt") -> Document:
    return Document(
        text=text,
        metadata=DocumentMetadata(source_path=source, format="txt"),
    )


def _passthrough_stage(name: str) -> MagicMock:
    """Create a mock PipelineStage that returns its input unchanged."""
    stage = MagicMock()
    stage.__class__.__name__ = name  # used in logger output
    stage.process.side_effect = lambda docs: docs  # passthrough
    return stage


class TestPipelineStageCallOrder:
    """Verify that Pipeline.run invokes stages in declaration order."""

    def test_two_stages_called_in_order(self) -> None:
        """Stages must be called in the order they were passed to Pipeline."""
        doc = _make_doc()
        stage_a = _passthrough_stage("StageA")
        stage_b = _passthrough_stage("StageB")

        pipeline = Pipeline(stages=[stage_a, stage_b])
        pipeline.run([doc])

        # Each stage's process() must have been called exactly once.
        stage_a.process.assert_called_once()
        stage_b.process.assert_called_once()

        # Verify order: StageA before StageB using a shared manager mock.
        manager = MagicMock()
        manager.attach_mock(stage_a.process, "stage_a_process")
        manager.attach_mock(stage_b.process, "stage_b_process")

        # Re-run to record calls on the manager.
        pipeline.run([doc])

        assert manager.mock_calls[0][0] == "stage_a_process", (
            "StageA must be called before StageB"
        )
        assert manager.mock_calls[1][0] == "stage_b_process", (
            "StageB must be called after StageA"
        )

    def test_three_stages_called_in_order(self) -> None:
        """Order must be preserved for three or more stages."""
        doc = _make_doc()
        stages = [_passthrough_stage(f"Stage{c}") for c in "ABC"]
        manager = MagicMock()
        for i, stage in enumerate(stages):
            manager.attach_mock(stage.process, f"stage_{i}_process")

        Pipeline(stages=stages).run([doc])

        call_names = [c[0] for c in manager.mock_calls]
        assert call_names == ["stage_0_process", "stage_1_process", "stage_2_process"]

    def test_no_stages_returns_input(self) -> None:
        """Pipeline with no stages must return documents unchanged."""
        docs = [_make_doc("a"), _make_doc("b")]
        result = Pipeline(stages=[]).run(docs)
        assert result == docs

    def test_error_in_stage_skips_document(self) -> None:
        """Per-document error isolation: bad doc is dropped, good doc passes."""
        good_doc = _make_doc("good content", "good.txt")
        bad_doc = _make_doc("bad content", "bad.txt")

        failing_stage = MagicMock()
        failing_stage.__class__.__name__ = "FailingStage"

        def selective_fail(docs: list[Document]) -> list[Document]:
            if docs[0].metadata.source_path == "bad.txt":
                raise ValueError("Simulated parse error")
            return docs

        failing_stage.process.side_effect = selective_fail

        pipeline = Pipeline(stages=[failing_stage])
        result = pipeline.run([good_doc, bad_doc])

        assert len(result) == 1
        assert result[0].metadata.source_path == "good.txt"

    def test_stage_receives_output_of_previous_stage(self) -> None:
        """Each stage must receive the output of the preceding stage, not the original input."""
        original = _make_doc("original")
        transformed = _make_doc("transformed")

        stage_a = MagicMock()
        stage_a.__class__.__name__ = "TransformStage"
        stage_a.process.return_value = [transformed]  # transforms the document

        stage_b = MagicMock()
        stage_b.__class__.__name__ = "VerifyStage"
        stage_b.process.side_effect = lambda docs: docs  # passthrough

        Pipeline(stages=[stage_a, stage_b]).run([original])

        # StageB must see the *output* of StageA (transformed), not original.
        stage_b.process.assert_called_once_with([transformed])
```

Run with:
```bash
pytest day-5-ex3.py -v
```

Expected output: 5 tests, all green.
