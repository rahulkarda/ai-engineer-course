# Phase 4 · Week 01 · Day 7 — Project Review & Assessment Prep

> Objective: Consolidate week-01 understanding, debug common pipeline failure modes, and prepare for the assessment.  ·  Time: ~3 hr project + 1 hr assessment prep  ·  Prereqs: [[day-6]]

## Concept

### Week 01 at a glance

This week covered the full ingestion layer of a RAG system — the part that transforms raw files into clean, structured documents ready for chunking and embedding. Key concepts:

| Day | Topic | Key idea |
|---|---|---|
| Day 1 | RAG overview | Retrieval-augmented generation: why it works, when to use it |
| Day 2 | Document loading | Format dispatch, loader abstraction, LoadError |
| Day 3 | Text cleaning | Unicode normalization, whitespace collapse, noise removal |
| Day 4 | Metadata & tables | DocumentMetadata Pydantic model, table serialization to Markdown |
| Day 5 | Pipeline architecture | PipelineStage Protocol, Pipeline runner, dedup, error strategies |
| Day 6 | Build day | Full multi-format IngestionPipeline end-to-end |

### Understanding checklist

Before the assessment, verify you can answer each without notes:

- [ ] What is RAG? Why does retrieving relevant context improve LLM output quality?
- [ ] What are the trade-offs between RAG, fine-tuning, and long-context prompting?
- [ ] Why does text cleaning matter before chunking and embedding?
- [ ] What does `unicodedata.normalize("NFC", text)` do, and why does it matter for embeddings?
- [ ] What is a `PipelineStage` Protocol? Why use `Protocol` over `ABC`?
- [ ] What is pipeline idempotency and how does content hashing enable it?
- [ ] What metadata fields are essential on a `Document` for downstream retrieval?
- [ ] What is the difference between skip-and-log and fail-fast error handling?

## How it works internally

### Common failure modes in ingestion pipelines

**1. Encoding errors (`UnicodeDecodeError`)**

```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 42
```

Cause: file is not UTF-8 (Latin-1, Windows-1252, or binary). Fix: use `chardet` to detect encoding, or open with `errors="replace"`. Always specify encoding explicitly — never rely on the system default.

```python
# Defensive read
text = path.read_text(encoding="utf-8", errors="replace")
```

**2. Malformed / scanned PDFs**

Symptom: `pypdf` returns empty strings from `page.extract_text()`. Cause: PDF is a scanned image with no text layer — it is a picture, not text.

Diagnosis: check `len(text.strip()) == 0` after extraction. Fix: fall back to OCR (Tesseract via `pytesseract`) or flag for manual review. For today's pipeline, log a warning and return the document with empty text — do not silently discard.

**3. Table extraction failures**

Symptom: rows are extracted but column alignment is wrong; the Markdown table renders incorrectly. Cause: merged cells, nested tables, or irregular column counts in DOCX/PDF tables.

Fix: detect irregular rows (row length != header length) and fall back to raw text serialization instead of Markdown table format.

**4. Memory issues with large files**

Symptom: `MemoryError` or OOM kill when processing a 500 MB PDF. Cause: loading the entire file into memory at once.

Fix: stream page-by-page for PDFs; use chunked reading for large CSVs (`csv.reader` is already a streaming iterator). For very large files, consider a size guard:

```python
MAX_FILE_BYTES = 100 * 1024 * 1024  # 100 MB
if path.stat().st_size > MAX_FILE_BYTES:
    logger.warning("File exceeds size limit, skipping: %s", path)
    return []
```

**5. Silent data loss**

Symptom: pipeline runs successfully but output document count is much lower than expected. Cause: exceptions are being swallowed silently (bare `except: pass`).

Fix: always log at `ERROR` level when skipping a document. Add a `skipped_count` counter to the pipeline and include it in the summary output.

**6. Non-deterministic content hashes**

Symptom: the same document produces a different hash on re-run. Cause: metadata (e.g., `ingested_at` timestamp) was included in the hash computation.

Fix: hash only the document **text**, not the metadata. The hash identifies the content, not the ingestion event.

## Worked example

A diagnostic script to run against your pipeline and produce a health report.

```python
# diagnostic.py
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FormatStats:
    total: int = 0
    success: int = 0
    failed: int = 0
    total_chars: int = 0
    elapsed_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 0.0

    @property
    def avg_chars(self) -> float:
        return self.total_chars / self.success if self.success else 0.0


@dataclass
class PipelineHealthReport:
    total_files: int = 0
    total_documents: int = 0
    total_failed: int = 0
    total_elapsed_ms: float = 0.0
    by_format: dict[str, FormatStats] = field(default_factory=dict)

    def print_summary(self) -> None:
        print(f"\n{'='*50}")
        print(f"Pipeline Health Report")
        print(f"{'='*50}")
        print(f"Files discovered : {self.total_files}")
        print(f"Documents output : {self.total_documents}")
        print(f"Failed           : {self.total_failed}")
        print(f"Total time       : {self.total_elapsed_ms:.1f} ms")
        print(f"\nPer-format breakdown:")
        for fmt, stats in self.by_format.items():
            print(
                f"  {fmt:<10} total={stats.total}  ok={stats.success}"
                f"  failed={stats.failed}  avg_chars={stats.avg_chars:.0f}"
                f"  success_rate={stats.success_rate:.0%}"
            )
        print(f"{'='*50}\n")


class PipelineDiagnostic:
    """Run the ingestion pipeline on a directory and produce a health report."""

    def __init__(self, input_dir: Path) -> None:
        self._input_dir = input_dir

    def run(self) -> PipelineHealthReport:
        from doc_ingestion.pipeline import IngestionPipeline
        from doc_ingestion.models import Document

        report = PipelineHealthReport()
        extensions = {".md", ".csv", ".pdf", ".html", ".htm", ".docx", ".txt"}

        paths = [p for p in self._input_dir.rglob("*") if p.suffix.lower() in extensions]
        report.total_files = len(paths)

        t0 = time.perf_counter()
        pipeline = IngestionPipeline(input_dir=self._input_dir)

        # Instrument: run per-format to collect granular stats
        for path in paths:
            fmt = path.suffix.lower().lstrip(".")
            if fmt not in report.by_format:
                report.by_format[fmt] = FormatStats()
            stats = report.by_format[fmt]
            stats.total += 1

            try:
                # Run a single-file mini-pipeline for isolation
                from doc_ingestion.dispatcher import FileDispatcher, LoadError
                from doc_ingestion.models import DocumentMetadata

                stub = Document(
                    text="",
                    metadata=DocumentMetadata(source_path=str(path), format="unknown"),
                )
                t_fmt = time.perf_counter()
                result = FileDispatcher().process([stub])
                stats.elapsed_ms += (time.perf_counter() - t_fmt) * 1000

                if result:
                    stats.success += 1
                    stats.total_chars += len(result[0].text)
                    report.total_documents += 1
                else:
                    stats.failed += 1
                    report.total_failed += 1

            except Exception as exc:
                logger.error("Diagnostic error on %s: %s", path, exc)
                stats.failed += 1
                report.total_failed += 1

        report.total_elapsed_ms = (time.perf_counter() - t0) * 1000
        return report


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.WARNING)
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./test-docs")
    report = PipelineDiagnostic(target).run()
    report.print_summary()
```

Run it: `python diagnostic.py ./test-docs`

## Exercises

1. **Trace the scanned-PDF path**: Starting from `PdfLoader.load()`, trace step-by-step what happens when `page.extract_text()` returns `""` for every page. What does the current implementation return? What should it return? What metadata field could signal "possible scan — no text extracted" to downstream stages?

2. **Retry/fallback strategy**: Design a `RetryingDispatcher` that wraps `FileDispatcher`. For each failed document it should: (a) wait 100 ms and retry once, (b) if still failing, attempt a fallback loader (e.g., read the file as plain text with `errors="replace"`), (c) if the fallback also fails, log at `ERROR` and skip. Write the class signature with full type hints.

3. **Assessment practice question**: A production team is deciding between three approaches for a customer support bot:
   - **RAG**: embed 50 000 historical support tickets; retrieve top-5 at query time.
   - **Fine-tuning**: fine-tune a base LLM on the same 50 000 tickets.
   - **Long context**: stuff 200 relevant tickets into a 200k-token context window at each request.

   For each approach, identify one concrete scenario where it outperforms the others, and one failure mode. When would you combine approaches?

(Solutions: ../solutions/week-01/day-7.md)

## Self-check

- [ ] Can describe all six stages of a production ingestion pipeline (load, extract, clean, chunk, embed, store) and the concerns of each.
- [ ] Can diagnose the three most common ingestion failures: encoding errors, empty-text PDFs, and memory issues.
- [ ] Can explain RAG, fine-tuning, and long-context trade-offs for at least two contrasting scenarios.
- [ ] Can read a `PipelineHealthReport` output and identify which format has the highest failure rate.
- [ ] Day-6 project has ≥ 5 passing pytest tests and handles corrupt files without crashing.

## Assessment prep

Topics to review before taking `assessment.md`:

1. **RAG fundamentals**: how retrieval improves generation, semantic search vs keyword search, why vector embeddings capture meaning.
2. **RAG vs fine-tuning vs long context**: cost, latency, knowledge update speed, hallucination risk for each.
3. **Document cleaning**: why `NFC` normalization matters, what whitespace collapse does to downstream tokenization.
4. **Metadata design**: which fields are needed for filtering vs attribution vs dedup; why `source_path` and `format` are non-negotiable.
5. **Pipeline idempotency**: definition, how content hashing implements it, why it matters for incremental updates.
6. **Debugging**: given a traceback from a loader or cleaner, identify the root cause and fix.
7. **Code quality**: type hints on all function signatures, `logging` vs `print`, Pydantic for data models, `os.getenv` for secrets.

Time box for the coding challenge: **90 minutes**. Focus on correctness and test coverage over feature completeness.
