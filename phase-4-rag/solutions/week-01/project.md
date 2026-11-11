# Week 01 Project — Reference Solution: `doc_ingestion`

## Project File Tree

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
    └── test_ingestion.py
```

---

## `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "doc-ingestion"
version = "0.1.0"
description = "Multi-format document ingestion pipeline for RAG systems."
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.7",
]

[project.optional-dependencies]
pdf = ["pypdf>=4.0"]
docx = ["python-docx>=1.1"]
all = ["doc-ingestion[pdf,docx]"]

[project.scripts]
doc-ingestion = "doc_ingestion.cli:main"

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.4",
]

[tool.hatch.build.targets.wheel]
packages = ["src/doc_ingestion"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

---

## `src/doc_ingestion/__init__.py`

```python
"""doc_ingestion — Multi-format document ingestion pipeline."""

from doc_ingestion.models import Document, DocumentMetadata
from doc_ingestion.pipeline import CleaningStage, HashDeduplicator, Pipeline, PipelineStage

__all__ = [
    "Document",
    "DocumentMetadata",
    "Pipeline",
    "PipelineStage",
    "CleaningStage",
    "HashDeduplicator",
]
```

---

## `src/doc_ingestion/models.py`

```python
"""
src/doc_ingestion/models.py

Pydantic models for documents and their metadata.
All fields are fully typed; no bare dicts cross module boundaries.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Provenance and format information for a Document.

    Attributes:
        source_path: Original file path as a string.
        format:      Detected file format.
        row_index:   For multi-record formats (CSV), the zero-based row index.
        title:       Optional document title extracted from content.
        extra:       Arbitrary additional key-value metadata.
    """

    source_path: str = Field(description="Original file path.")
    format: Literal["md", "csv", "txt", "pdf", "html", "docx", "unknown"] = Field(
        default="unknown"
    )
    row_index: int | None = Field(default=None)
    title: str | None = Field(default=None)
    extra: dict[str, str] = Field(default_factory=dict)


class Document(BaseModel):
    """A single unit of text ready for chunking and embedding.

    Attributes:
        text:     Extracted and cleaned plain text.
        metadata: Provenance and format metadata.
    """

    text: str = Field(description="Extracted plain text.")
    metadata: DocumentMetadata = Field(description="Provenance metadata.")

    @property
    def is_empty(self) -> bool:
        """Return True when the document contains no non-whitespace text."""
        return not self.text.strip()
```

---

## `src/doc_ingestion/loaders.py`

```python
"""
src/doc_ingestion/loaders.py

Format-specific document loaders.

All stdlib loaders (Markdown, CSV, Text, HTML) have zero optional dependencies.
PDF loading requires ``pypdf`` and DOCX loading requires ``python-docx``; both
are guarded with ImportError stubs so the package works without them.
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
    """Structural interface every loader must satisfy."""

    def load(self, path: Path) -> list[Document]:
        """Load *path* and return one or more Document objects."""
        ...


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

class MarkdownLoader:
    """Load a Markdown file as a single Document.

    The first H1 heading (``# Title``) is captured as ``metadata.title``.
    """

    def load(self, path: Path) -> list[Document]:
        text = path.read_text(encoding="utf-8")
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

    Each row is rendered as ``field: value`` pairs separated by newlines.

    Args:
        delimiter: Column delimiter (default ``,``).
    """

    def __init__(self, delimiter: str = ",") -> None:
        self._delimiter = delimiter

    def load(self, path: Path) -> list[Document]:
        documents: list[Document] = []
        with path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter=self._delimiter)
            for i, row in enumerate(reader):
                text = "\n".join(f"{k}: {v}" for k, v in row.items() if v is not None)
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
        logger.debug("CSVLoader: %d rows from %s", len(documents), path)
        return documents


# ---------------------------------------------------------------------------
# Plain Text
# ---------------------------------------------------------------------------

class TextLoader:
    """Load a plain-text file as a single Document."""

    def load(self, path: Path) -> list[Document]:
        return [
            Document(
                text=path.read_text(encoding="utf-8"),
                metadata=DocumentMetadata(source_path=str(path), format="txt"),
            )
        ]


# ---------------------------------------------------------------------------
# PDF (optional — requires pypdf)
# ---------------------------------------------------------------------------

try:
    from pypdf import PdfReader as _PdfReader  # type: ignore[import-untyped]

    class PDFLoader:
        """Load a PDF — one Document per page.

        Requires ``pypdf``::

            pip install pypdf
        """

        def load(self, path: Path) -> list[Document]:
            reader = _PdfReader(str(path))
            documents: list[Document] = []
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if not text.strip():
                    logger.warning(
                        "PDFLoader: page %d of %s has no extractable text "
                        "(possibly scanned — consider OCR).",
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
            return documents

except ImportError:

    class PDFLoader:  # type: ignore[no-redef]
        """Stub — install pypdf for PDF support: ``pip install pypdf``."""

        def load(self, path: Path) -> list[Document]:
            raise ImportError(
                "pypdf is required for PDF loading.\n"
                "Install: pip install pypdf"
            )


# ---------------------------------------------------------------------------
# HTML (stdlib only)
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    """HTMLParser subclass that collects visible text."""

    _SKIP: frozenset[str] = frozenset({"script", "style", "head", "meta", "link", "noscript"})

    def __init__(self) -> None:
        super().__init__()
        self._depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self._SKIP:
            self._depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._SKIP and self._depth > 0:
            self._depth -= 1

    def handle_data(self, data: str) -> None:
        if self._depth == 0:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)

    @property
    def text(self) -> str:
        return "\n".join(self._parts)


class HTMLLoader:
    """Load an HTML file as a single Document using stdlib html.parser."""

    def load(self, path: Path) -> list[Document]:
        parser = _TextExtractor()
        parser.feed(path.read_text(encoding="utf-8"))
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
        """Stub — install python-docx for DOCX support: ``pip install python-docx``."""

        def load(self, path: Path) -> list[Document]:
            raise ImportError(
                "python-docx is required for DOCX loading.\n"
                "Install: pip install python-docx"
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
    """Return a loader for *path* based on its suffix, or None."""
    return LOADER_REGISTRY.get(path.suffix.lower())
```

---

## `src/doc_ingestion/cleaning.py`

```python
"""
src/doc_ingestion/cleaning.py

Text cleaning: Unicode normalisation and whitespace collapse.
"""

from __future__ import annotations

import logging
import re
import unicodedata

from doc_ingestion.models import Document

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")


class TextCleaner:
    """Clean and normalise document text.

    Steps applied in order:
    1. NFKC Unicode normalisation.
    2. ASCII control-character removal (keeps ``\\t`` and ``\\n``).
    3. Whitespace collapse within paragraphs (double-newline paragraphs preserved).

    Args:
        normalize_unicode:    Apply NFKC normalisation (default True).
        collapse_whitespace:  Collapse runs of whitespace (default True).
        remove_control_chars: Strip ASCII control chars (default True).
    """

    def __init__(
        self,
        normalize_unicode: bool = True,
        collapse_whitespace: bool = True,
        remove_control_chars: bool = True,
    ) -> None:
        self._normalize = normalize_unicode
        self._collapse = collapse_whitespace
        self._remove_ctrl = remove_control_chars

    def clean(self, text: str) -> str:
        """Apply configured cleaning steps to *text*.

        Args:
            text: Raw input string.

        Returns:
            Cleaned string.
        """
        if self._normalize:
            text = unicodedata.normalize("NFKC", text)
        if self._remove_ctrl:
            text = _CTRL_RE.sub("", text)
        if self._collapse:
            paragraphs = re.split(r"\n{2,}", text)
            text = "\n\n".join(
                _WHITESPACE_RE.sub(" ", p).strip()
                for p in paragraphs
                if p.strip()
            )
        return text

    def clean_document(self, doc: Document) -> Document:
        """Return a new Document with cleaned text; metadata unchanged.

        Args:
            doc: Input Document.

        Returns:
            New Document with cleaned text field.
        """
        cleaned = self.clean(doc.text)
        if len(cleaned) < len(doc.text):
            logger.debug(
                "TextCleaner: %s %d→%d chars.",
                doc.metadata.source_path,
                len(doc.text),
                len(cleaned),
            )
        return Document(text=cleaned, metadata=doc.metadata)
```

---

## `src/doc_ingestion/tables.py`

```python
"""
src/doc_ingestion/tables.py

Utilities for converting tabular data to embeddable text representations.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

TableData = list[list[str]]


class TableExtractor:
    """Convert list-of-lists table data to Markdown or row narratives.

    Typical workflow:
    1. A loader extracts a table as ``list[list[str]]``.
    2. ``to_markdown()`` produces a GFM table string for display/storage.
    3. ``to_row_narratives()`` produces one sentence per row for embedding,
       ensuring each chunk carries full column context.
    """

    @staticmethod
    def to_markdown(
        table: TableData,
        headers: list[str] | None = None,
    ) -> str:
        """Render a list-of-lists as a GitHub-Flavored Markdown table.

        Args:
            table:   Rows of cell values. If *headers* is None, the first row
                     is treated as the header row.
            headers: Explicit column headers; *table* then contains data rows only.

        Returns:
            GFM Markdown table string.

        Raises:
            ValueError: If *table* is empty.
        """
        if not table:
            raise ValueError("Cannot render an empty table.")

        if headers is None:
            headers = [str(c) for c in table[0]]
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

        sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
        return "\n".join([_row(headers), sep] + [_row(r) for r in data_rows])

    @staticmethod
    def to_row_narratives(
        table: TableData,
        headers: list[str] | None = None,
        row_prefix: str = "Record",
    ) -> list[str]:
        """Convert each row to a human-readable narrative string.

        Each narrative includes every field name and value so the chunk is
        self-contained for embedding.

        Args:
            table:      Rows of cell values.
            headers:    Column names; inferred from the first row if None.
            row_prefix: Label prepended to each narrative (default ``Record``).

        Returns:
            One narrative string per data row.

        Example:
            >>> TableExtractor.to_row_narratives([["Name","Age"],["Alice","30"]])
            ['Record 1 — Name: Alice; Age: 30']
        """
        if not table:
            return []

        if headers is None:
            headers = [str(c) for c in table[0]]
            data_rows = table[1:]
        else:
            data_rows = table

        return [
            f"{row_prefix} {i + 1} — " + "; ".join(
                f"{h}: {cell}" for h, cell in zip(headers, row)
            )
            for i, row in enumerate(data_rows)
        ]
```

---

## `src/doc_ingestion/pipeline.py`

```python
"""
src/doc_ingestion/pipeline.py

Pipeline Protocol, orchestrator, and built-in stages.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Protocol, runtime_checkable

from doc_ingestion.cleaning import TextCleaner
from doc_ingestion.models import Document

logger = logging.getLogger(__name__)


@runtime_checkable
class PipelineStage(Protocol):
    """Structural interface for a single pipeline processing step."""

    def process(self, documents: list[Document]) -> list[Document]:
        """Transform *documents* and return the result.

        Must not mutate the input list in place.
        """
        ...


class Pipeline:
    """Orchestrate a sequence of PipelineStage instances.

    Per-document error isolation: a stage failure on one document drops
    that document and processing continues with the remaining batch.

    Args:
        stages: Ordered list of PipelineStage instances.
    """

    def __init__(self, stages: list[PipelineStage]) -> None:
        self._stages = stages

    def run(self, documents: list[Document]) -> list[Document]:
        """Process *documents* through every stage.

        Args:
            documents: Initial raw documents.

        Returns:
            Documents that passed all stages without error.
        """
        current: list[Document] = list(documents)
        total = len(documents)

        for stage in self._stages:
            survived: list[Document] = []
            for doc in current:
                try:
                    survived.extend(stage.process([doc]))
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


class CleaningStage:
    """Pipeline stage: apply TextCleaner to every Document.

    Args:
        cleaner:    TextCleaner instance (default: new instance with all steps on).
        drop_empty: Drop documents that are empty after cleaning (default True).
    """

    def __init__(
        self,
        cleaner: TextCleaner | None = None,
        drop_empty: bool = True,
    ) -> None:
        self._cleaner = cleaner or TextCleaner()
        self._drop_empty = drop_empty

    def process(self, documents: list[Document]) -> list[Document]:
        cleaned = [self._cleaner.clean_document(d) for d in documents]
        if self._drop_empty:
            before = len(cleaned)
            cleaned = [d for d in cleaned if not d.is_empty]
            if (dropped := before - len(cleaned)):
                logger.info("CleaningStage: dropped %d empty documents.", dropped)
        return cleaned


class HashDeduplicator:
    """Pipeline stage: drop documents with duplicate text content.

    Uses SHA-256 of the document text as the deduplication key.

    Args:
        seen_hashes: Pre-populated set of already-seen hashes for cross-batch
                     deduplication. Mutated in place as new hashes are recorded.
    """

    def __init__(self, seen_hashes: set[str] | None = None) -> None:
        self._seen: set[str] = seen_hashes if seen_hashes is not None else set()

    def process(self, documents: list[Document]) -> list[Document]:
        unique: list[Document] = []
        for doc in documents:
            h = hashlib.sha256(doc.text.encode()).hexdigest()
            if h in self._seen:
                logger.debug("HashDeduplicator: duplicate dropped — %s", doc.metadata.source_path)
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
    uv run python -m doc_ingestion.cli --input-dir docs/ --output-json out.json
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
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".md", ".markdown", ".csv", ".txt", ".pdf", ".html", ".htm", ".docx"}
)


def _discover_files(input_dir: Path) -> list[Path]:
    return sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
    )


def _load_all(files: list[Path]) -> list[Document]:
    documents: list[Document] = []
    for path in files:
        loader = get_loader(path)
        if loader is None:
            logger.warning("No loader for %s — skipping.", path)
            continue
        try:
            docs = loader.load(path)
            documents.extend(docs)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load %s: %s — skipping.", path, exc)
    return documents


def _write_summary(documents: list[Document], output_path: Path) -> None:
    summary = {
        "total_documents": len(documents),
        "documents": [
            {
                "source_path": d.metadata.source_path,
                "format": d.metadata.format,
                "text_length": len(d.text),
                "is_empty": d.is_empty,
                "row_index": d.metadata.row_index,
                "title": d.metadata.title,
            }
            for d in documents
        ],
    }
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Output written to %s (%d documents).", output_path, len(documents))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list; defaults to sys.argv[1:].

    Returns:
        0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        description="Ingest documents from a directory and write a JSON summary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory to scan.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("ingestion_output.json"),
        help="Destination JSON file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args(argv)
    logging.getLogger().setLevel(args.log_level)

    if not args.input_dir.is_dir():
        logger.error("'%s' is not a directory.", args.input_dir)
        return 1

    logger.info("Scanning %s …", args.input_dir)
    files = _discover_files(args.input_dir)
    logger.info("Found %d supported files.", len(files))

    raw = _load_all(files)
    logger.info("Loaded %d raw documents.", len(raw))

    pipeline = Pipeline(
        stages=[
            CleaningStage(cleaner=TextCleaner(), drop_empty=True),
            HashDeduplicator(),
        ]
    )
    processed = pipeline.run(raw)
    logger.info("Pipeline complete: %d documents.", len(processed))

    _write_summary(processed, args.output_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

## `tests/test_ingestion.py`

```python
"""
tests/test_ingestion.py

Pytest tests for the doc_ingestion package.
All tests use tmp_path; no external dependencies required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from doc_ingestion.cleaning import TextCleaner
from doc_ingestion.loaders import CSVLoader, MarkdownLoader, TextLoader, get_loader
from doc_ingestion.models import Document, DocumentMetadata
from doc_ingestion.pipeline import CleaningStage, HashDeduplicator, Pipeline
from doc_ingestion.tables import TableExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc(text: str, source: str = "test.txt", fmt: str = "txt") -> Document:
    return Document(
        text=text,
        metadata=DocumentMetadata(source_path=source, format=fmt),  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Test 1: MarkdownLoader
# ---------------------------------------------------------------------------

class TestMarkdownLoader:
    def test_extracts_title_from_h1(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.md"
        f.write_text("# Hello World\n\nBody text.", encoding="utf-8")
        docs = MarkdownLoader().load(f)
        assert len(docs) == 1
        assert docs[0].metadata.title == "Hello World"
        assert docs[0].metadata.format == "md"
        assert "Body text." in docs[0].text

    def test_no_h1_title_is_none(self, tmp_path: Path) -> None:
        f = tmp_path / "plain.md"
        f.write_text("No heading here.\n", encoding="utf-8")
        docs = MarkdownLoader().load(f)
        assert docs[0].metadata.title is None

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(OSError):
            MarkdownLoader().load(tmp_path / "nonexistent.md")


# ---------------------------------------------------------------------------
# Test 2: CSVLoader
# ---------------------------------------------------------------------------

class TestCSVLoader:
    def test_one_document_per_row(self, tmp_path: Path) -> None:
        f = tmp_path / "data.csv"
        f.write_text("name,city\nAlice,Berlin\nBob,Paris\n", encoding="utf-8")
        docs = CSVLoader().load(f)
        assert len(docs) == 2
        assert "name: Alice" in docs[0].text
        assert "city: Berlin" in docs[0].text
        assert docs[0].metadata.row_index == 0
        assert docs[1].metadata.row_index == 1

    def test_header_only_csv_returns_empty_list(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.csv"
        f.write_text("name,age\n", encoding="utf-8")
        assert CSVLoader().load(f) == []


# ---------------------------------------------------------------------------
# Test 3: TextCleaner
# ---------------------------------------------------------------------------

class TestTextCleaner:
    def test_nfkc_ligature_normalisation(self) -> None:
        # U+FB01 LATIN SMALL LIGATURE FI should become "fi"
        assert TextCleaner().clean("ﬁle") == "file"

    def test_whitespace_collapse(self) -> None:
        raw = "hello   world\n\n\nfoo   bar"
        cleaned = TextCleaner().clean(raw)
        assert cleaned == "hello world\n\nfoo bar"

    def test_control_characters_removed(self) -> None:
        raw = "hello\x00world"
        assert "\x00" not in TextCleaner().clean(raw)

    def test_empty_string_returns_empty(self) -> None:
        assert TextCleaner().clean("") == ""


# ---------------------------------------------------------------------------
# Test 4: HashDeduplicator
# ---------------------------------------------------------------------------

class TestHashDeduplicator:
    def test_drops_duplicate_content(self) -> None:
        a = _doc("same text", "a.txt")
        b = _doc("same text", "b.txt")
        c = _doc("different", "c.txt")
        result = HashDeduplicator().process([a, b, c])
        assert len(result) == 2
        sources = {d.metadata.source_path for d in result}
        assert "b.txt" not in sources

    def test_all_unique_documents_pass(self) -> None:
        docs = [_doc(f"text {i}") for i in range(10)]
        assert len(HashDeduplicator().process(docs)) == 10

    def test_cross_batch_dedup_with_shared_set(self) -> None:
        seen: set[str] = set()
        batch1_result = HashDeduplicator(seen_hashes=seen).process([_doc("shared")])
        batch2_result = HashDeduplicator(seen_hashes=seen).process([_doc("shared")])
        assert len(batch1_result) == 1
        assert len(batch2_result) == 0


# ---------------------------------------------------------------------------
# Test 5: Pipeline integration + error isolation
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_full_pipeline_on_txt_file(self, tmp_path: Path) -> None:
        f = tmp_path / "article.txt"
        f.write_text("ﬁrst paragraph.\n\nSecond   paragraph.", encoding="utf-8")
        docs = TextLoader().load(f)
        pipeline = Pipeline(stages=[CleaningStage(), HashDeduplicator()])
        result = pipeline.run(docs)
        assert len(result) == 1
        assert "first paragraph." in result[0].text  # ligature normalised

    def test_bad_document_skipped_good_document_survives(self) -> None:
        good = _doc("good", "good.txt")
        bad = _doc("bad", "bad.txt")

        bomb = MagicMock()
        bomb.__class__.__name__ = "BombStage"
        bomb.process.side_effect = lambda docs: (
            (_ for _ in ()).throw(RuntimeError("boom"))
            if docs[0].metadata.source_path == "bad.txt"
            else docs
        )

        result = Pipeline(stages=[bomb]).run([good, bad])
        assert len(result) == 1
        assert result[0].metadata.source_path == "good.txt"

    def test_cleaning_stage_drops_empty_documents(self) -> None:
        docs = [_doc("  \n  "), _doc("real content")]
        result = CleaningStage(drop_empty=True).process(docs)
        assert len(result) == 1
        assert result[0].text == "real content"

    def test_get_loader_returns_none_for_unknown_extension(self, tmp_path: Path) -> None:
        assert get_loader(tmp_path / "file.xyz") is None

    def test_stages_called_in_order(self) -> None:
        """StageB must receive the output of StageA, not the original input."""
        original = _doc("original")
        transformed = _doc("transformed")

        stage_a = MagicMock()
        stage_a.__class__.__name__ = "TransformStage"
        stage_a.process.return_value = [transformed]

        stage_b = MagicMock()
        stage_b.__class__.__name__ = "PassthroughStage"
        stage_b.process.side_effect = lambda docs: docs

        Pipeline(stages=[stage_a, stage_b]).run([original])
        stage_b.process.assert_called_once_with([transformed])


# ---------------------------------------------------------------------------
# Test 6: TableExtractor
# ---------------------------------------------------------------------------

class TestTableExtractor:
    def test_to_markdown_renders_headers_and_rows(self) -> None:
        table = [["Name", "Age"], ["Alice", "30"], ["Bob", "25"]]
        md = TableExtractor.to_markdown(table)
        assert "Name" in md
        assert "Alice" in md
        assert "---" in md

    def test_to_row_narratives_format(self) -> None:
        table = [["Name", "Age"], ["Alice", "30"]]
        narratives = TableExtractor.to_row_narratives(table)
        assert narratives == ["Record 1 — Name: Alice; Age: 30"]

    def test_empty_table_raises_for_markdown(self) -> None:
        with pytest.raises(ValueError):
            TableExtractor.to_markdown([])

    def test_empty_table_returns_empty_list_for_narratives(self) -> None:
        assert TableExtractor.to_row_narratives([]) == []
```

---

## How to Run

### Setup

```bash
# Clone or unpack the project, then:
cd doc_ingestion

# Install with uv (recommended)
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
```

### Run the CLI

```bash
# Basic usage
uv run python -m doc_ingestion.cli --input-dir docs/ --output-json out.json

# With debug logging
uv run python -m doc_ingestion.cli --input-dir docs/ --output-json out.json --log-level DEBUG

# Via the installed script entry point (after pip install -e .)
doc-ingestion --input-dir docs/ --output-json out.json
```

### Run tests

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src/doc_ingestion --cov-report=term-missing

# Specific test class
pytest tests/test_ingestion.py::TestPipeline -v
```

### Install optional dependencies

```bash
# PDF support
pip install pypdf

# DOCX support
pip install python-docx

# Both
pip install "doc-ingestion[all]"
```

### Expected CLI output

```
10:22:01 [INFO] doc_ingestion.cli: Scanning docs/ …
10:22:01 [INFO] doc_ingestion.cli: Found 12 supported files.
10:22:01 [INFO] doc_ingestion.cli: Loaded 47 raw documents.
10:22:01 [INFO] doc_ingestion.pipeline: Stage CleaningStage: 45/47 documents.
10:22:01 [INFO] doc_ingestion.pipeline: Stage HashDeduplicator: 43/47 documents.
10:22:01 [INFO] doc_ingestion.cli: Pipeline complete: 43 documents.
10:22:01 [INFO] doc_ingestion.cli: Output written to out.json (43 documents).
```
