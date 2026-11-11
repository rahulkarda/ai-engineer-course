# Phase 4 · Week 01 · Day 6 — Build Session: Multi-Format Ingestion Pipeline

> Objective: Build a complete, production-quality document ingestion pipeline supporting PDF, HTML, Markdown, DOCX, and CSV.  ·  Time: ~4 hr  ·  Prereqs: [[day-1]], [[day-2]], [[day-3]], [[day-4]], [[day-5]]

## Concept

Today you build the full pipeline end-to-end. By the end of 4 hours you will have a runnable Python package, `doc_ingestion`, that:

- Accepts a directory of mixed-format documents.
- Dispatches each file to the right loader by extension.
- Extracts clean text and structured tables.
- Normalizes text (unicode, whitespace) and enriches metadata.
- Outputs a list of typed `Document` objects ready for chunking and embedding.

Supported formats: `.md`, `.html`, `.txt`, `.csv`, `.pdf` (via `pypdf`), `.docx` (via `python-docx`).

Heavy dependencies (install once before you start):

```bash
uv add pypdf python-docx
```

Everything else uses stdlib: `csv`, `html.parser`, `pathlib`, `logging`, `hashlib`, `argparse`.

The output of today's build is the project deliverable for Week 01. Keep tests alongside implementation — you need at least 5 passing pytest tests by end of day.

## Build plan

### Hour 1 — Project setup, Document model, dispatcher, Markdown + CSV loaders

1. Create the package structure (see skeleton below).
2. Define `Document` and `DocumentMetadata` Pydantic models in `models.py`.
3. Implement `FileDispatcher` — maps `{".md": MarkdownLoader, ".csv": CsvLoader, ...}`.
4. Implement `MarkdownLoader`: read file with `pathlib.Path.read_text(encoding="utf-8")`, populate metadata.
5. Implement `CsvLoader`: use `csv.DictReader`; serialize each row to a `key: value` string; join rows.
6. Write first 2 pytest tests: one for `MarkdownLoader`, one for `CsvLoader` (use `tmp_path` fixture).

### Hour 2 — PDF, HTML, DOCX loaders

1. Implement `PdfLoader` using `pypdf.PdfReader`. Extract text page by page; join with `\n`. Store `page_count` in metadata. Handle `pypdf.errors.PdfReadError` — log and re-raise as a custom `LoadError`.
2. Implement `HtmlLoader` using `html.parser.HTMLParser`. Strip tags, extract visible text. Note: for production use `BeautifulSoup(html, "html.parser")` — but today use stdlib to avoid the dependency.
3. Implement `DocxLoader` using `python_docx.Document`. Extract all paragraphs; join with `\n`. Handle `docx.oxml.exceptions.InvalidXmlError` as `LoadError`.
4. Add tests for `PdfLoader` and `HtmlLoader` (mock `pypdf.PdfReader` and file I/O).

### Hour 3 — TextCleaner, TableExtractor, metadata enrichment

1. Port `TextCleaner` from day-3: `unicodedata.normalize("NFC", text)` + `re.sub(r"\s+", " ", text).strip()`.
2. Port `TableExtractor` from day-4: detect CSV-like content, serialize to Markdown table string.
3. Implement `MetadataEnricher` stage: add `ingested_at` (ISO timestamp), `char_count`, `word_count`, `content_hash` (sha256) to each document's metadata.
4. Write a test for `MetadataEnricher` verifying the four metadata fields are present and correct.

### Hour 4 — Pipeline wiring, error handling, tests, CLI

1. Implement `IngestionPipeline` using the `PipelineStage` Protocol from day-5.
2. Wire stages: `FileDispatcher → CleaningStage → MetadataEnricher → HashDeduplicator`.
3. Add skip-and-log error isolation at the dispatcher level (one bad file → logged, rest continue).
4. Implement CLI entry point in `__main__.py` using `argparse`: takes `--input-dir` and `--output-json`, runs pipeline, writes JSON summary.
5. Final test count check — must be ≥ 5. Run `pytest -v`.

## Worked example

### Project structure

```
doc_ingestion/
├── __init__.py
├── __main__.py          # CLI: python -m doc_ingestion --input-dir ./docs
├── models.py            # Document, DocumentMetadata (Pydantic)
├── pipeline.py          # PipelineStage Protocol, Pipeline, HashDeduplicator
├── dispatcher.py        # FileDispatcher — maps extension → loader
├── loaders/
│   ├── __init__.py
│   ├── markdown.py
│   ├── csv_loader.py
│   ├── pdf.py
│   ├── html.py
│   └── docx.py
├── cleaning.py          # TextCleaner, CleaningStage
├── extraction.py        # TableExtractor
└── enrichment.py        # MetadataEnricher
tests/
├── conftest.py
├── test_loaders.py
├── test_pipeline.py
└── test_enrichment.py
pyproject.toml
```

### Core models (`models.py`)

```python
from __future__ import annotations

from pydantic import BaseModel, Field
from pathlib import Path


class DocumentMetadata(BaseModel):
    source_path: str
    format: str                        # "markdown" | "pdf" | "html" | "docx" | "csv" | "text"
    ingested_at: str = ""
    char_count: int = 0
    word_count: int = 0
    content_hash: str = ""
    page_count: int | None = None      # PDF only
    extra: dict[str, str] = Field(default_factory=dict)


class Document(BaseModel):
    text: str
    metadata: DocumentMetadata

    @classmethod
    def from_path(cls, path: Path, text: str, format: str) -> "Document":
        return cls(
            text=text,
            metadata=DocumentMetadata(source_path=str(path), format=format),
        )
```

### FileDispatcher skeleton (`dispatcher.py`)

```python
from __future__ import annotations

import logging
from pathlib import Path

from doc_ingestion.models import Document

logger = logging.getLogger(__name__)


class LoadError(Exception):
    """Raised when a loader cannot process a file."""


class FileDispatcher:
    """Route files to the correct loader by extension."""

    def __init__(self) -> None:
        from doc_ingestion.loaders.markdown import MarkdownLoader
        from doc_ingestion.loaders.csv_loader import CsvLoader
        from doc_ingestion.loaders.pdf import PdfLoader
        from doc_ingestion.loaders.html import HtmlLoader
        from doc_ingestion.loaders.docx import DocxLoader

        self._registry: dict[str, object] = {
            ".md": MarkdownLoader(),
            ".csv": CsvLoader(),
            ".pdf": PdfLoader(),
            ".html": HtmlLoader(),
            ".htm": HtmlLoader(),
            ".docx": DocxLoader(),
            ".txt": MarkdownLoader(),   # treat plain text same as markdown
        }

    def process(self, docs: list[Document]) -> list[Document]:
        """Expects Document.metadata.source_path to be a valid file path."""
        results: list[Document] = []
        for doc in docs:
            path = Path(doc.metadata.source_path)
            loader = self._registry.get(path.suffix.lower())
            if loader is None:
                logger.warning("No loader for extension %s — skipping %s", path.suffix, path)
                continue
            try:
                loaded = loader.load(path)  # type: ignore[attr-defined]
                results.append(loaded)
            except LoadError as exc:
                logger.error("Failed to load %s: %s", path, exc)
            except Exception as exc:
                logger.error("Unexpected error loading %s: %s", path, exc, exc_info=True)
        return results
```

### PdfLoader skeleton (`loaders/pdf.py`)

```python
from __future__ import annotations

import logging
from pathlib import Path

from doc_ingestion.models import Document
from doc_ingestion.dispatcher import LoadError

logger = logging.getLogger(__name__)


class PdfLoader:
    def load(self, path: Path) -> Document:
        try:
            import pypdf  # lazy import — optional dependency

            reader = pypdf.PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            text = "\n".join(pages)
            doc = Document.from_path(path, text, format="pdf")
            doc.metadata.page_count = len(reader.pages)
            return doc
        except Exception as exc:
            raise LoadError(f"pypdf failed on {path}: {exc}") from exc
```

### IngestionPipeline (`pipeline.py` addition)

```python
class IngestionPipeline:
    """End-to-end pipeline: discover files → load → clean → enrich → dedup."""

    def __init__(self, input_dir: Path) -> None:
        self._input_dir = input_dir
        self._stages: list[PipelineStage] = [
            FileDispatcher(),
            CleaningStage(),
            MetadataEnricher(),
            HashDeduplicator(),
        ]

    def run(self) -> list[Document]:
        """Discover all supported files and run the full pipeline."""
        extensions = {".md", ".csv", ".pdf", ".html", ".htm", ".docx", ".txt"}
        stub_docs = [
            Document(
                text="",
                metadata=DocumentMetadata(source_path=str(p), format="unknown"),
            )
            for p in self._input_dir.rglob("*")
            if p.suffix.lower() in extensions
        ]
        logger.info("Discovered %d files in %s", len(stub_docs), self._input_dir)
        return Pipeline(self._stages).run(stub_docs)
```

### CLI entry point (`__main__.py`)

```python
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from doc_ingestion.pipeline import IngestionPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents from a directory.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-json", type=Path, default=Path("summary.json"))
    args = parser.parse_args()

    pipeline = IngestionPipeline(input_dir=args.input_dir)
    docs = pipeline.run()

    summary = {
        "total_documents": len(docs),
        "formats": {},
        "documents": [],
    }
    for doc in docs:
        fmt = doc.metadata.format
        summary["formats"][fmt] = summary["formats"].get(fmt, 0) + 1
        summary["documents"].append({
            "source": doc.metadata.source_path,
            "format": fmt,
            "char_count": doc.metadata.char_count,
            "content_hash": doc.metadata.content_hash[:8],
        })

    args.output_json.write_text(json.dumps(summary, indent=2))
    logger.info("Summary written to %s", args.output_json)


if __name__ == "__main__":
    main()
```

## Exercises

These are your build tasks — complete them in order to finish the project.

1. **Implement all loaders**: Fill in `MarkdownLoader`, `CsvLoader`, `HtmlLoader`, and `DocxLoader` following the same pattern as `PdfLoader`. Each must: read the file, return a `Document`, raise `LoadError` on failure, and log at `DEBUG` level.

2. **Implement `MetadataEnricher`**: Create `enrichment.py` with a `MetadataEnricher` class satisfying `PipelineStage`. It must add `ingested_at` (use `datetime.now(UTC).isoformat()`), `char_count`, `word_count`, and `content_hash` to every document's metadata in-place.

3. **Wire `IngestionPipeline`**: Complete `pipeline.py` so `IngestionPipeline.run()` produces the correct output for a directory containing one file of each supported format. Add logging for file-discovery count.

4. **Error handling**: Ensure that a corrupt PDF or unreadable file logs an error and is excluded from results, but does not stop the pipeline. Write a test that provides a file with an unreadable path and asserts the pipeline returns an empty list (not an exception).

5. **CLI and JSON summary**: Implement `__main__.py` fully. Run it against a local `./test-docs/` directory containing at least one `.md` and one `.csv` file. Inspect the output JSON and verify `total_documents`, `formats`, and `char_count` fields are correct.

(Solutions: ../solutions/week-01/day-6.md)

## Self-check

- [ ] `MarkdownLoader`, `CsvLoader`, `PdfLoader`, `HtmlLoader`, and `DocxLoader` all return a typed `Document` and raise `LoadError` on failure.
- [ ] `TextCleaner` and `MetadataEnricher` are stage-compatible and add all required metadata fields.
- [ ] `IngestionPipeline.run()` correctly discovers, loads, cleans, and deduplicates a mixed-format directory.
- [ ] At least one bad/corrupt file is handled gracefully (logged, excluded, pipeline continues).
- [ ] `pytest -v` reports at least 5 passing tests covering loaders, enrichment, and pipeline.
