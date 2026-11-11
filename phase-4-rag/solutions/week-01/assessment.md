# Answer Key — Week 01 Assessment

## Conceptual Questions — Model Answers

### Q1. What is Retrieval-Augmented Generation (RAG) and why is it preferred over fine-tuning for frequently-updated knowledge bases?

RAG retrieves relevant documents at query time and injects them into the prompt, grounding the model's answer in factual, up-to-date content. Fine-tuning bakes knowledge into model weights: updating it requires a full retraining cycle (hours to days, significant compute cost), and the updated model still cannot cite its source.

For a knowledge base that changes weekly, RAG is preferred because:
- **Freshness**: re-ingesting changed documents and updating the vector index takes minutes, not days.
- **Cost**: re-embedding only changed documents is cheap; retraining is not.
- **Traceability**: retrieved chunks can be returned alongside the answer so users can verify the source.
- **No hallucination amplification**: the model is constrained to retrieved context, reducing confabulation.

Fine-tuning is more appropriate when the goal is to change the model's style, reasoning pattern, or to encode task-specific skills — not to inject domain facts.

---

### Q2. Describe the four main stages of a document ingestion pipeline and what each does.

| Stage | Responsibility |
|---|---|
| **Load** | Read raw bytes from disk (or a remote source) and produce a `Document` with extracted plain text and source metadata. Format-specific logic lives here (PDF, DOCX, CSV, HTML, etc.). |
| **Clean** | Normalise and sanitise text: Unicode normalisation (NFKC), whitespace collapse, control-character removal. Does not change meaning — only representation. |
| **Chunk** | Split long documents into fixed-size or semantically coherent fragments that fit within an embedding model's context window. Overlap between chunks preserves boundary context. |
| **Embed** | Convert each chunk to a dense vector representation using an embedding model. Vectors are stored in a vector database for similarity search. |

An optional **Deduplicate** stage between Load and Clean removes identical or near-identical documents to avoid redundant embeddings.

---

### Q3. What is the role of the `PipelineStage` Protocol, and what advantage does it offer over an abstract base class?

`PipelineStage` is a structural type (defined with `typing.Protocol`) that specifies a single required method: `process(documents: list[Document]) -> list[Document]`.

Advantages over ABC:
1. **No inheritance required** — any class with the right method signature automatically satisfies the Protocol. Third-party or legacy classes integrate without modification.
2. **Static type checking** — mypy and pyright verify conformance at type-check time, not at runtime.
3. **Loose coupling** — stages can be unit-tested in isolation; the pipeline does not need to know the concrete type.
4. **Composability** — trivial to wrap, chain, or mock stages.

---

### Q4. Explain what `unicodedata.normalize('NFKC', text)` does and give one concrete example of why it matters for RAG.

NFKC (Compatibility Decomposition followed by Canonical Composition) converts characters that are visually similar but encoded differently into a single canonical form.

**Example:** The ligature `ﬁ` (U+FB01 LATIN SMALL LIGATURE FI, a single code point) is normalised to `fi` (two separate code points). Without normalisation, a query for "configuration" would not match a document containing "conﬁguration" typed with the ligature, causing a retrieval miss.

Other cases: full-width Latin characters (`Ａ` → `A`), superscript digits (`²` → `2`), and typographic apostrophes (`'` → `'`).

---

### Q5. What is content-hash deduplication and when does it fail?

Content-hash deduplication computes SHA-256 of each document's text and discards documents whose hash has already been seen. It guarantees that byte-for-byte identical texts are not indexed twice.

**When it fails:**
- **Near-duplicates** — a document with one extra space or a changed date hash differently and both are indexed. Near-duplicate detection (MinHash / SimHash) is needed for this case.
- **Reformatted content** — the same article exported as `.md` vs `.txt` may have different whitespace, producing different hashes. Apply cleaning *before* hashing to mitigate.
- **Partial updates** — a document where only one paragraph changed gets a new hash; the old version remains in the index. Requires a source-path keyed update strategy, not just content hashing.

---

### Q6. A user reports that after ingestion, some PDF documents return empty text. Walk through the diagnostic steps you would take.

1. **Reproduce** — load the specific PDF in isolation with `PDFLoader` and check `doc.is_empty` on the result.
2. **Check if text is extractable** — open the PDF in a viewer. If it displays text that you can select and copy, `pypdf` should extract it. If you cannot select text, the PDF is scanned.
3. **Inspect `pypdf` directly** — `PdfReader(path).pages[0].extract_text()` — if this returns `""`, the page has no text layer.
4. **Check pypdf version** — some older versions have bugs with certain font encodings; `pip show pypdf`.
5. **Check for password protection** — `PdfReader.is_encrypted` — if True, decryption is needed before extraction.
6. **OCR path** — if the file is a scanned image, integrate `pdf2image` + `pytesseract` as a fallback stage.
7. **Metadata flag** — set `doc.metadata.extra["needs_ocr"] = "true"` for documents that return empty, enabling a follow-up batch OCR job.

---

## Coding Challenge — Reference Solution

```python
"""
Week 01 Assessment — Coding Challenge Reference Solution

DocumentIngestionPipeline: discovers, loads, cleans, and returns Documents
from a directory. No external dependencies required.
"""

from __future__ import annotations

import logging
import unicodedata
from pathlib import Path
from typing import Literal
import re

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class DocumentMetadata(BaseModel):
    source_path: str
    format: Literal["md", "csv", "txt", "unknown"] = "unknown"


class Document(BaseModel):
    text: str
    metadata: DocumentMetadata

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")
_SUPPORTED_EXTENSIONS: dict[str, Literal["md", "csv", "txt"]] = {
    ".md": "md",
    ".markdown": "md",
    ".csv": "csv",
    ".txt": "txt",
}


def _clean_text(text: str) -> str:
    """NFKC normalisation + whitespace collapse."""
    text = unicodedata.normalize("NFKC", text)
    paragraphs = re.split(r"\n{2,}", text)
    cleaned = [_WHITESPACE_RE.sub(" ", p).strip() for p in paragraphs]
    return "\n\n".join(p for p in cleaned if p)


def _load_file(path: Path) -> list[Document]:
    """Load a single file; return list of Documents (CSV = one per row)."""
    fmt = _SUPPORTED_EXTENSIONS.get(path.suffix.lower(), "unknown")

    if fmt == "csv":
        import csv
        documents: list[Document] = []
        with path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                text = "\n".join(f"{k}: {v}" for k, v in row.items())
                documents.append(
                    Document(
                        text=text,
                        metadata=DocumentMetadata(source_path=str(path), format="csv"),
                    )
                )
        return documents

    # md and txt: read entire file as one document
    text = path.read_text(encoding="utf-8")
    return [
        Document(
            text=text,
            metadata=DocumentMetadata(source_path=str(path), format=fmt),
        )
    ]


class DocumentIngestionPipeline:
    """Discover, load, and clean documents from a directory.

    Args:
        input_dir: Root directory to scan recursively.
    """

    def __init__(self, input_dir: Path) -> None:
        self._input_dir = input_dir

    def run(self) -> list[Document]:
        """Execute the pipeline and return cleaned documents.

        Returns:
            Cleaned, non-empty Document objects from all supported files.
        """
        files = sorted(
            p
            for p in self._input_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
        )
        logger.info("Found %d files in %s.", len(files), self._input_dir)

        raw: list[Document] = []
        for path in files:
            try:
                docs = _load_file(path)
                raw.extend(docs)
            except OSError as exc:
                logger.error("Cannot read %s: %s — skipping.", path, exc)

        cleaned: list[Document] = []
        for doc in raw:
            try:
                clean_text = _clean_text(doc.text)
                cleaned_doc = Document(text=clean_text, metadata=doc.metadata)
                if not cleaned_doc.is_empty:
                    cleaned.append(cleaned_doc)
                else:
                    logger.warning("Empty document after cleaning: %s", doc.metadata.source_path)
            except Exception as exc:  # noqa: BLE001
                logger.error("Cleaning failed for %s: %s — skipping.", doc.metadata.source_path, exc)

        logger.info("Pipeline complete: %d documents.", len(cleaned))
        return cleaned
```

### Pytest tests for the coding challenge

```python
"""Tests for DocumentIngestionPipeline (assessment coding challenge)."""

from pathlib import Path
import pytest

# Adjust import path to match how you run the assessment:
# from assessment_solution import DocumentIngestionPipeline, Document


class TestDocumentIngestionPipeline:

    def test_loads_txt_file(self, tmp_path: Path) -> None:
        (tmp_path / "hello.txt").write_text("Hello world", encoding="utf-8")
        pipeline = DocumentIngestionPipeline(tmp_path)
        docs = pipeline.run()
        assert len(docs) == 1
        assert "Hello world" in docs[0].text
        assert docs[0].metadata.format == "txt"

    def test_loads_markdown_file(self, tmp_path: Path) -> None:
        (tmp_path / "doc.md").write_text("# Title\n\nContent here.", encoding="utf-8")
        pipeline = DocumentIngestionPipeline(tmp_path)
        docs = pipeline.run()
        assert len(docs) == 1
        assert docs[0].metadata.format == "md"

    def test_loads_csv_one_doc_per_row(self, tmp_path: Path) -> None:
        (tmp_path / "data.csv").write_text("name,age\nAlice,30\nBob,25\n", encoding="utf-8")
        pipeline = DocumentIngestionPipeline(tmp_path)
        docs = pipeline.run()
        assert len(docs) == 2

    def test_skips_empty_after_cleaning(self, tmp_path: Path) -> None:
        (tmp_path / "empty.txt").write_text("   \n\n   ", encoding="utf-8")
        (tmp_path / "real.txt").write_text("Real content", encoding="utf-8")
        pipeline = DocumentIngestionPipeline(tmp_path)
        docs = pipeline.run()
        assert len(docs) == 1
        assert "Real content" in docs[0].text

    def test_unreadable_file_is_skipped(self, tmp_path: Path) -> None:
        good = tmp_path / "good.txt"
        good.write_text("good content", encoding="utf-8")
        bad = tmp_path / "bad.txt"
        bad.write_text("bad content", encoding="utf-8")
        bad.chmod(0o000)  # remove read permissions
        try:
            pipeline = DocumentIngestionPipeline(tmp_path)
            docs = pipeline.run()
            # Only the readable file should be returned.
            assert all(d.metadata.source_path == str(good) for d in docs)
        finally:
            bad.chmod(0o644)  # restore for cleanup
```

---

## Rubric Guidance

### Conceptual Questions (30 points, 5 per question)

| Marks | Criterion |
|---|---|
| 5/5 | Core concept correct, trade-offs articulated, concrete example or counterexample given. |
| 3–4/5 | Core concept correct, trade-offs partially addressed, no example. |
| 1–2/5 | Partially correct or vague. |
| 0/5 | Missing or wrong. |

**Q1 RAG vs fine-tuning**: Full marks require mentioning freshness, cost, and grounding. Penalise answers that say fine-tuning is "never useful."

**Q2 Four stages**: Full marks require Load, Clean, Chunk, Embed with a one-sentence explanation of each. Deduplication as optional bonus.

**Q3 Protocol vs ABC**: Must mention structural subtyping / no-inheritance-required to get full marks.

**Q4 NFKC**: Must give a concrete character example (ligature, full-width, etc.).

**Q5 Content-hash deduplication**: Must identify at least two failure modes for full marks.

**Q6 PDF diagnosis**: Full marks require at least 4 distinct diagnostic steps and the OCR path mentioned.

---

### Coding Challenge (40 points)

| Dimension | Points | Full-marks criteria |
|---|---|---|
| Correctness | 15 | Discovers `.md`, `.csv`, `.txt`; CSV produces one doc per row; text/md produce one doc per file. |
| Type annotations | 5 | All function signatures annotated; no bare `dict` as return type. |
| Error handling | 10 | `OSError` caught per file with `logger.error`; empty docs after cleaning are dropped with a log. |
| Code quality | 5 | No `print`; `logging` used; no hardcoded strings for supported extensions. |
| Tests | 5 | At least 2 tests use `tmp_path`; tests are independent; no real filesystem paths. |

---

### Reflection (30 points)

| Dimension | Points | Full-marks criteria |
|---|---|---|
| Chunking strategy justification | 10 | Explains why chunk size matters for embedding models (context window), mentions overlap. |
| Deduplication across runs | 10 | Describes persisting hashes (shelve/db) between runs; distinguishes from in-run deduplication. |
| Production hardening | 10 | Mentions at least 3 of: retry/backoff, OCR fallback, metrics/observability, schema validation, rate limiting. |
