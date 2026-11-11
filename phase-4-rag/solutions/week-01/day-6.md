# Solutions — Phase 4 · Week 01 · Day 6 (Saturday Build Day)

Reference implementation for the 4-hour Multi-Format Ingestion Pipeline build day.

---

## Project Structure

```
doc_ingestion/
├── pyproject.toml
├── src/
│   └── doc_ingestion/
│       ├── __init__.py
│       ├── models.py
│       ├── loaders.py
│       ├── cleaning.py
│       ├── tables.py
│       ├── pipeline.py
│       └── cli.py
└── tests/
    └── test_pipeline.py
```

---

## `src/doc_ingestion/models.py`

```python
"""
src/doc_ingestion/models.py

Shared Pydantic models for the document ingestion pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata attached to every ingested document."""

    source_path: str = Field(description="Original file path as a string.")
    format: Literal["md", "csv", "txt", "pdf", "html", "docx", "unknown"] = Field(
        default="unknown",
        description="Detected file format.",
    )
    row_index: int | None = Field(
        default=None,
        description="For multi-record formats (CSV), the zero-based row index.",
    )
    title: str | None = Field(
        default=None,
        description="Optional document title extracted from content.",
    )
    extra: dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary additional metadata key-value pairs.",
    )


class Document(BaseModel):
    """A single unit of text ready for downstream processing (chunking, embedding)."""

    text: str = Field(description="Extracted and cleaned plain text.")
    metadata: DocumentMetadata = Field(description="Provenance and format metadata.")

    @property
    def is_empty(self) -> bool:
        """Return True if the document contains no non-whitespace text."""
        return not self.text.strip()
```

---

## `src/doc_ingestion/loaders.py`

```python
"""
src/doc_ingestion/loaders.py

Format-specific document loaders.

Optional dependencies:
- pypdf  (PDF loading)      — install with: pip install pypdf
- python-docx (DOCX loading) — install with: pip install python-docx
"""

from __future__ import annotations

import csv
import logging
from html.parser import HTMLParser
from pathlib import Path
from typing import Protocol, runtime_checkable

from doc_ingestion.models import Document, DocumentMetadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loader Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class DocumentLoader(Protocol):
    """Interface every loader must satisfy."""

    def load(self, path: Path) -> list[Document]:
        """Load *path* and return one or more Document objects."""
        ...


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

class MarkdownLoader:
    """Load a Markdown file as a single Document."""

    def load(self, path: Path) -> list[Document]:
        """Read a .md file and return a single Document.

        Args:
            path: Filesystem path to the Markdown file.

        Returns:
            Single-element list with the loaded Document.

        Raises:
            OSError: If the file cannot be read.
        """
        text = path.read_text(encoding="utf-8")
        # Use the first H1 heading as the title if present.
        title: str | None = None
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                title = stripped[2:].strip()
                break

        return [
            Document(
                text=text,
                metadata=DocumentMetadata(
                    source_path=str(path),
                    format="md",
                    title=title,
                ),
            )
        ]


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

class CSVLoader:
    """Load a CSV file — one Document per data row.

    Each row is rendered as ``key: value`` pairs separated by newlines so the
    text is human-readable and embeddable.

    Args:
        delimiter: Column delimiter (default ``,``).
    """

    def __init__(self, delimiter: str = ",") -> None:
        self._delimiter = delimiter

    def load(self, path: Path) -> list[Document]:
        """Load all rows from a CSV file.

        Args:
            path: Filesystem path to the CSV file.

        Returns:
            One Document per data row (header row becomes field names).

        Raises:
            OSError: If the file cannot be read.
        """
        documents: list[Document] = []
        with path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter=self._delimiter)
            for i, row in enumerate(reader):
                text_parts = [f"{k}: {v}" for k, v in row.items() if v is not None]
                text = "\n".join(text_parts)
                documents.append(
                    Document(
                        text=text,
                        metadata=DocumentMetadata(
                            source_path=str(path),
                            format="csv",
                            row_index=i,
                        ),
                    )
                )
        logger.debug("CSVLoader: loaded %d rows from %s", len(documents), path)
        return documents


# ---------------------------------------------------------------------------
# Plain Text
# ---------------------------------------------------------------------------

class TextLoader:
    """Load a plain-text file as a single Document."""

    def load(self, path: Path) -> list[Document]:
        """Read a .txt file.

        Args:
            path: Filesystem path to the text file.

        Returns:
            Single-element list with the loaded Document.
        """
        text = path.read_text(encoding="utf-8")
        return [
            Document(
                text=text,
                metadata=DocumentMetadata(source_path=str(path), format="txt"),
            )
        ]


# ---------------------------------------------------------------------------
# PDF (optional — requires pypdf)
# ---------------------------------------------------------------------------

try:
    from pypdf import PdfReader as _PdfReader  # type: ignore[import-untyped]

    class PDFLoader:
        """Load a PDF file — one Document per page.

        Requires ``pypdf``::

            pip install pypdf
        """

        def load(self, path: Path) -> list[Document]:
            """Extract text from every page of a PDF.

            Args:
                path: Filesystem path to the PDF file.

            Returns:
                One Document per page. Pages with no extractable text are
                included with empty text (flag with ``is_empty`` property).

            Raises:
                OSError: If the file cannot be opened.
            """
            documents: list[Document] = []
            reader = _PdfReader(str(path))
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if not text.strip():
                    logger.warning(
                        "PDFLoader: page %d of %s has no extractable text "
                        "(may be scanned image — consider OCR).",
                        page_num,
                        path,
                    )
                documents.append(
                    Document(
                        text=text,
                        metadata=DocumentMetadata(
                            source_path=str(path),
                            format="pdf",
                            row_index=page_num,
                            extra={"page": str(page_num)},
                        ),
                    )
                )
            logger.debug("PDFLoader: loaded %d pages from %s", len(documents), path)
            return documents

except ImportError:

    class PDFLoader:  # type: ignore[no-redef]
        """Stub PDFLoader — install ``pypdf`` to enable PDF support."""

        def load(self, path: Path) -> list[Document]:
            raise ImportError(
                "pypdf is required for PDF loading.\n"
                "Install it with: pip install pypdf"
            )


# ---------------------------------------------------------------------------
# HTML (stdlib only — html.parser)
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    """HTMLParser subclass that collects visible text nodes."""

    def __init__(self) -> None:
        super().__init__()
        self._skip_tags: set[str] = {"script", "style", "head", "meta", "link"}
        self._in_skip = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self._skip_tags:
            self._in_skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._skip_tags and self._in_skip > 0:
            self._in_skip -= 1

    def handle_data(self, data: str) -> None:
        if self._in_skip == 0:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)

    @property
    def text(self) -> str:
        return "\n".join(self._parts)


class HTMLLoader:
    """Load an HTML file as a single Document using stdlib html.parser."""

    def load(self, path: Path) -> list[Document]:
        """Extract visible text from an HTML file.

        Args:
            path: Filesystem path to the HTML file.

        Returns:
            Single-element list with visible text.
        """
        raw_html = path.read_text(encoding="utf-8")
        parser = _TextExtractor()
        parser.feed(raw_html)
        return [
            Document(
                text=parser.text,
                metadata=DocumentMetadata(source_path=str(path), format="html"),
            )
        ]


# ---------------------------------------------------------------------------
# DOCX (optional — requires python-docx)
# ---------------------------------------------------------------------------

try:
    import docx as _docx  # type: ignore[import-untyped]

    class DOCXLoader:
        """Load a DOCX file as a single Document.

        Requires ``python-docx``::

            pip install python-docx
        """

        def load(self, path: Path) -> list[Document]:
            """Extract paragraph text from a DOCX file.

            Args:
                path: Filesystem path to the .docx file.

            Returns:
                Single-element list with all paragraph text joined by newlines.
            """
            doc = _docx.Document(str(path))
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            return [
                Document(
                    text=text,
                    metadata=DocumentMetadata(source_path=str(path), format="docx"),
                )
            ]

except ImportError:

    class DOCXLoader:  # type: ignore[no-redef]
        """Stub DOCXLoader — install ``python-docx`` to enable DOCX support."""

        def load(self, path: Path) -> list[Document]:
            raise ImportError(
                "python-docx is required for DOCX loading.\n"
                "Install it with: pip install python-docx"
            )


# ---------------------------------------------------------------------------
# Loader registry
# ---------------------------------------------------------------------------

LOADER_REGISTRY: dict[str, DocumentLoader] = {
    ".md": MarkdownLoader(),
    ".markdown": MarkdownLoader(),
    ".csv": CSVLoader(),
    ".txt": TextLoader(),
    ".pdf": PDFLoader(),
    ".html": HTMLLoader(),
    ".htm": HTMLLoader(),
    ".docx": DOCXLoader(),
}


def get_loader(path: Path) -> DocumentLoader | None:
    """Return the appropriate loader for *path* based on its suffix.

    Args:
        path: File to inspect.

    Returns:
        A DocumentLoader instance, or None if no loader is registered for
        this file extension.
    """
    return LOADER_REGISTRY.get(path.suffix.lower())
```

---

## `src/doc_ingestion/cleaning.py`

```python
"""
src/doc_ingestion/cleaning.py

Text cleaning utilities for ingested documents.
"""

from __future__ import annotations

import logging
import re
import unicodedata

from doc_ingestion.models import Document

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")


class TextCleaner:
    """Clean and normalise the text of a Document.

    Steps (all enabled by default):
    1. Unicode NFKC normalisation — converts ligatures, full-width chars, etc.
    2. Whitespace collapse — runs of whitespace become a single space; leading/
       trailing whitespace is stripped.
    3. Control-character removal — strips ASCII control chars (0x00–0x1F) except
       tab (\\t) and newline (\\n).

    Args:
        normalize_unicode: Apply NFKC normalisation (default True).
        collapse_whitespace: Collapse runs of whitespace (default True).
        remove_control_chars: Strip ASCII control chars (default True).
    """

    def __init__(
        self,
        normalize_unicode: bool = True,
        collapse_whitespace: bool = True,
        remove_control_chars: bool = True,
    ) -> None:
        self._normalize_unicode = normalize_unicode
        self._collapse_whitespace = collapse_whitespace
        self._remove_control_chars = remove_control_chars
        # Matches ASCII control chars except \t (0x09) and \n (0x0A).
        self._ctrl_re = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")

    def clean(self, text: str) -> str:
        """Apply configured cleaning steps to *text*.

        Args:
            text: Raw input text.

        Returns:
            Cleaned text string.
        """
        if self._normalize_unicode:
            text = unicodedata.normalize("NFKC", text)
        if self._remove_control_chars:
            text = self._ctrl_re.sub("", text)
        if self._collapse_whitespace:
            # Preserve paragraph breaks (double newline) but collapse the rest.
            paragraphs = re.split(r"\n{2,}", text)
            cleaned_paragraphs = [
                _WHITESPACE_RE.sub(" ", p).strip() for p in paragraphs
            ]
            text = "\n\n".join(p for p in cleaned_paragraphs if p)
        return text

    def clean_document(self, doc: Document) -> Document:
        """Return a new Document with cleaned text; metadata is unchanged.

        Args:
            doc: Input Document.

        Returns:
            New Document with cleaned text.
        """
        cleaned_text = self.clean(doc.text)
        if len(cleaned_text) < len(doc.text):
            logger.debug(
                "TextCleaner: %s reduced from %d to %d chars.",
                doc.metadata.source_path,
                len(doc.text),
                len(cleaned_text),
            )
        return Document(text=cleaned_text, metadata=doc.metadata)
```

---

## `src/doc_ingestion/tables.py`

```python
"""
src/doc_ingestion/tables.py

Utilities for working with tabular data extracted from documents.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

TableData = list[list[str]]  # rows x columns


class TableExtractor:
    """Convert tabular data between representations.

    Typical use: a loader extracts a table as a list-of-lists (rows × columns);
    TableExtractor can render it as a Markdown table or as a list of row
    narrative sentences for embedding.
    """

    @staticmethod
    def to_markdown(
        table: TableData,
        headers: list[str] | None = None,
    ) -> str:
        """Render a list-of-lists as a GitHub-Flavored Markdown table.

        Args:
            table:   Rows of cell values (strings).  First row is used as
                     headers when *headers* is None.
            headers: Explicit column headers.  When provided, *table* should
                     contain only data rows (no header row).

        Returns:
            Markdown table string.

        Raises:
            ValueError: If *table* is empty.
        """
        if not table:
            raise ValueError("Cannot render an empty table to Markdown.")

        if headers is None:
            headers = [str(cell) for cell in table[0]]
            data_rows = table[1:]
        else:
            data_rows = table

        col_widths = [
            max(len(str(h)), max((len(str(row[i])) for row in data_rows), default=0))
            for i, h in enumerate(headers)
        ]

        def _row(cells: list[str]) -> str:
            padded = [str(c).ljust(w) for c, w in zip(cells, col_widths)]
            return "| " + " | ".join(padded) + " |"

        separator = "| " + " | ".join("-" * w for w in col_widths) + " |"
        lines = [_row(headers), separator] + [_row(row) for row in data_rows]
        return "\n".join(lines)

    @staticmethod
    def to_row_narratives(
        table: TableData,
        headers: list[str] | None = None,
        row_prefix: str = "Record",
    ) -> list[str]:
        """Convert each row to a human-readable narrative sentence.

        Useful when embedding tables: each row becomes an independent text
        chunk with all field context included.

        Args:
            table:      Rows of cell values.
            headers:    Column names; inferred from first row when None.
            row_prefix: Label prepended to each narrative (default "Record").

        Returns:
            One narrative string per data row.

        Example:
            Input:  [["Name", "Age"], ["Alice", "30"]]
            Output: ["Record 1 — Name: Alice; Age: 30"]
        """
        if not table:
            return []

        if headers is None:
            headers = [str(cell) for cell in table[0]]
            data_rows = table[1:]
        else:
            data_rows = table

        narratives: list[str] = []
        for i, row in enumerate(data_rows, start=1):
            pairs = "; ".join(
                f"{h}: {cell}" for h, cell in zip(headers, row)
            )
            narratives.append(f"{row_prefix} {i} — {pairs}")

        return narratives
```

---

## `src/doc_ingestion/pipeline.py`

```python
"""
src/doc_ingestion/pipeline.py

Core pipeline abstractions: PipelineStage Protocol, Pipeline orchestrator,
built-in stages (CleaningStage, HashDeduplicator).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Protocol, runtime_checkable

from doc_ingestion.cleaning import TextCleaner
from doc_ingestion.models import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class PipelineStage(Protocol):
    """A single processing step that transforms a list of Documents."""

    def process(self, documents: list[Document]) -> list[Document]:
        """Transform *documents* and return the result.

        Implementations must not mutate the input list in place; return a
        new list.
        """
        ...


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

class Pipeline:
    """Run a sequence of PipelineStage instances with per-document error isolation.

    Args:
        stages: Ordered list of PipelineStage instances.
    """

    def __init__(self, stages: list[PipelineStage]) -> None:
        self._stages = stages

    def run(self, documents: list[Document]) -> list[Document]:
        """Process *documents* through every stage.

        Per-document error isolation: if a stage raises on a document, that
        document is dropped and processing continues.

        Args:
            documents: Initial list of raw Documents.

        Returns:
            Documents that survived all stages.
        """
        current: list[Document] = list(documents)
        total = len(documents)

        for stage in self._stages:
            survived: list[Document] = []
            for doc in current:
                try:
                    result = stage.process([doc])
                    survived.extend(result)
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Stage %s failed on '%s': %s — skipping.",
                        type(stage).__name__,
                        doc.metadata.source_path,
                        exc,
                    )
            current = survived
            logger.info(
                "Stage %s: %d/%d documents.",
                type(stage).__name__,
                len(current),
                total,
            )

        return current


# ---------------------------------------------------------------------------
# Built-in stages
# ---------------------------------------------------------------------------

class CleaningStage:
    """Pipeline stage that applies TextCleaner to every Document.

    Args:
        cleaner: TextCleaner instance. A default instance is created if omitted.
        drop_empty: If True (default), drop documents that are empty after
                    cleaning.
    """

    def __init__(
        self,
        cleaner: TextCleaner | None = None,
        drop_empty: bool = True,
    ) -> None:
        self._cleaner = cleaner or TextCleaner()
        self._drop_empty = drop_empty

    def process(self, documents: list[Document]) -> list[Document]:
        """Clean each document; optionally remove empty results.

        Args:
            documents: Input documents.

        Returns:
            Cleaned documents, with empties removed if *drop_empty* is True.
        """
        cleaned = [self._cleaner.clean_document(doc) for doc in documents]
        if self._drop_empty:
            before = len(cleaned)
            cleaned = [d for d in cleaned if not d.is_empty]
            dropped = before - len(cleaned)
            if dropped:
                logger.info("CleaningStage: dropped %d empty documents.", dropped)
        return cleaned


class HashDeduplicator:
    """Pipeline stage that removes documents with duplicate text content.

    Uses SHA-256 of the document text as the deduplication key.  The first
    occurrence of each unique hash is kept; subsequent duplicates are dropped.

    Args:
        seen_hashes: Optional pre-populated set of already-seen hashes (useful
                     for cross-batch deduplication across pipeline runs).
    """

    def __init__(self, seen_hashes: set[str] | None = None) -> None:
        self._seen: set[str] = seen_hashes or set()

    def process(self, documents: list[Document]) -> list[Document]:
        """Drop duplicate documents.

        Args:
            documents: Input documents.

        Returns:
            Deduplicated documents.
        """
        unique: list[Document] = []
        for doc in documents:
            h = hashlib.sha256(doc.text.encode("utf-8")).hexdigest()
            if h in self._seen:
                logger.debug(
                    "HashDeduplicator: duplicate dropped — %s",
                    doc.metadata.source_path,
                )
            else:
                self._seen.add(h)
                unique.append(doc)
        return unique
```

---

## `src/doc_ingestion/cli.py`

```python
"""
src/doc_ingestion/cli.py

Command-line entry point for the document ingestion pipeline.

Usage::

    python -m doc_ingestion.cli --input-dir docs/ --output-json out.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from doc_ingestion.cleaning import TextCleaner
from doc_ingestion.loaders import get_loader
from doc_ingestion.models import Document
from doc_ingestion.pipeline import CleaningStage, HashDeduplicator, Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _discover_files(input_dir: Path) -> list[Path]:
    """Recursively find all files with a registered loader.

    Args:
        input_dir: Root directory to search.

    Returns:
        Sorted list of file paths.
    """
    supported_extensions = {".md", ".markdown", ".csv", ".txt", ".pdf", ".html", ".htm", ".docx"}
    return sorted(
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in supported_extensions
    )


def _load_documents(files: list[Path]) -> list[Document]:
    """Load all files, logging errors for individual failures.

    Args:
        files: List of file paths to load.

    Returns:
        All successfully loaded documents.
    """
    documents: list[Document] = []
    for path in files:
        loader = get_loader(path)
        if loader is None:
            logger.warning("No loader for %s — skipping.", path)
            continue
        try:
            docs = loader.load(path)
            documents.extend(docs)
            logger.debug("Loaded %d document(s) from %s.", len(docs), path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load %s: %s — skipping.", path, exc)
    return documents


def _write_summary(documents: list[Document], output_path: Path) -> None:
    """Write a JSON summary of processed documents.

    Args:
        documents: Processed Document objects.
        output_path: Destination JSON file path.
    """
    summary = {
        "total_documents": len(documents),
        "documents": [
            {
                "source_path": doc.metadata.source_path,
                "format": doc.metadata.format,
                "text_length": len(doc.text),
                "is_empty": doc.is_empty,
                "row_index": doc.metadata.row_index,
                "title": doc.metadata.title,
            }
            for doc in documents
        ],
    }
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Summary written to %s (%d documents).", output_path, len(documents))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 = success, 1 = error).
    """
    parser = argparse.ArgumentParser(
        description="Ingest documents from a directory and write a JSON summary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory to scan for documents.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("ingestion_output.json"),
        help="Path for the output JSON summary file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    args = parser.parse_args(argv)
    logging.getLogger().setLevel(args.log_level)

    if not args.input_dir.is_dir():
        logger.error("--input-dir '%s' is not a directory.", args.input_dir)
        return 1

    logger.info("Scanning %s …", args.input_dir)
    files = _discover_files(args.input_dir)
    logger.info("Found %d supported files.", len(files))

    raw_documents = _load_documents(files)
    logger.info("Loaded %d raw documents.", len(raw_documents))

    pipeline = Pipeline(
        stages=[
            CleaningStage(cleaner=TextCleaner(), drop_empty=True),
            HashDeduplicator(),
        ]
    )
    processed = pipeline.run(raw_documents)
    logger.info("Pipeline complete: %d documents remain.", len(processed))

    _write_summary(processed, args.output_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

## `tests/test_pipeline.py`

```python
"""
tests/test_pipeline.py

Pytest tests for the document ingestion pipeline.
Uses tmp_path fixture; no external dependencies required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from doc_ingestion.cleaning import TextCleaner
from doc_ingestion.loaders import CSVLoader, MarkdownLoader, TextLoader, get_loader
from doc_ingestion.models import Document, DocumentMetadata
from doc_ingestion.pipeline import CleaningStage, HashDeduplicator, Pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc(text: str, source: str = "test.txt") -> Document:
    return Document(
        text=text,
        metadata=DocumentMetadata(source_path=source, format="txt"),
    )


# ---------------------------------------------------------------------------
# Test 1: MarkdownLoader extracts title and text
# ---------------------------------------------------------------------------

class TestMarkdownLoader:
    def test_loads_title_and_text(self, tmp_path: Path) -> None:
        md_file = tmp_path / "readme.md"
        md_file.write_text("# My Title\n\nSome content here.", encoding="utf-8")

        docs = MarkdownLoader().load(md_file)

        assert len(docs) == 1
        assert docs[0].metadata.title == "My Title"
        assert "Some content here." in docs[0].text
        assert docs[0].metadata.format == "md"

    def test_no_title_when_no_h1(self, tmp_path: Path) -> None:
        md_file = tmp_path / "no_title.md"
        md_file.write_text("Just some text without a heading.", encoding="utf-8")

        docs = MarkdownLoader().load(md_file)

        assert docs[0].metadata.title is None


# ---------------------------------------------------------------------------
# Test 2: CSVLoader produces one document per row
# ---------------------------------------------------------------------------

class TestCSVLoader:
    def test_one_doc_per_row(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,age\nAlice,30\nBob,25\n", encoding="utf-8")

        docs = CSVLoader().load(csv_file)

        assert len(docs) == 2
        assert "name: Alice" in docs[0].text
        assert "age: 30" in docs[0].text
        assert docs[0].metadata.row_index == 0
        assert docs[1].metadata.row_index == 1

    def test_empty_csv_returns_no_documents(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("name,age\n", encoding="utf-8")

        docs = CSVLoader().load(csv_file)

        assert docs == []


# ---------------------------------------------------------------------------
# Test 3: TextCleaner normalises unicode and collapses whitespace
# ---------------------------------------------------------------------------

class TestTextCleaner:
    def test_unicode_normalisation(self) -> None:
        # U+FB01 LATIN SMALL LIGATURE FI → "fi" after NFKC
        dirty = "ﬁle"
        cleaned = TextCleaner().clean(dirty)
        assert cleaned == "file"

    def test_whitespace_collapse(self) -> None:
        dirty = "hello   world\n\n\nfoo   bar"
        cleaned = TextCleaner().clean(dirty)
        assert cleaned == "hello world\n\nfoo bar"

    def test_empty_input(self) -> None:
        assert TextCleaner().clean("") == ""


# ---------------------------------------------------------------------------
# Test 4: HashDeduplicator drops exact duplicates
# ---------------------------------------------------------------------------

class TestHashDeduplicator:
    def test_drops_duplicate_text(self) -> None:
        doc_a = _doc("same content", "a.txt")
        doc_b = _doc("same content", "b.txt")  # same text, different source
        doc_c = _doc("different content", "c.txt")

        dedup = HashDeduplicator()
        result = dedup.process([doc_a, doc_b, doc_c])

        assert len(result) == 2
        sources = {d.metadata.source_path for d in result}
        assert "a.txt" in sources
        assert "c.txt" in sources
        assert "b.txt" not in sources

    def test_unique_documents_all_pass(self) -> None:
        docs = [_doc(f"unique content {i}") for i in range(5)]
        result = HashDeduplicator().process(docs)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# Test 5: Pipeline error isolation — bad stage skips offending document
# ---------------------------------------------------------------------------

class TestPipelineErrorIsolation:
    def test_bad_document_skipped(self) -> None:
        good = _doc("good", "good.txt")
        bad = _doc("bad", "bad.txt")

        failing_stage = MagicMock()
        failing_stage.__class__.__name__ = "BombStage"

        def selective_fail(docs: list[Document]) -> list[Document]:
            if docs[0].metadata.source_path == "bad.txt":
                raise RuntimeError("Simulated failure")
            return docs

        failing_stage.process.side_effect = selective_fail

        pipeline = Pipeline(stages=[failing_stage])
        result = pipeline.run([good, bad])

        assert len(result) == 1
        assert result[0].metadata.source_path == "good.txt"

    def test_get_loader_returns_none_for_unknown_extension(self, tmp_path: Path) -> None:
        unknown = tmp_path / "file.xyz"
        assert get_loader(unknown) is None

    def test_cleaning_stage_drops_empty_docs(self) -> None:
        docs = [_doc("  \n\n  "), _doc("real content")]
        stage = CleaningStage(drop_empty=True)
        result = stage.process(docs)
        assert len(result) == 1
        assert result[0].text == "real content"
```

---

## Running the build-day project

```bash
# Install in editable mode with dev deps
uv pip install -e ".[dev]"

# Run the CLI
python -m doc_ingestion.cli --input-dir docs/ --output-json out.json --log-level DEBUG

# Run tests
pytest tests/ -v
```
