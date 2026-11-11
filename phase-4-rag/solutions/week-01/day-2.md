# Solutions — Phase 4 · Week 01 · Day 2

## Exercise 1 — Plain-text Loader

Full `TextLoader` class that mirrors the interface of the `load_markdown` worked example:

```python
import logging
import os
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Document(BaseModel):
    text: str
    metadata: dict[str, str | int | float | None]


class TextLoader:
    """Load a plain-text file into a Document.

    Args:
        encoding: File encoding, defaults to UTF-8.
        strip_whitespace: If True, strip leading/trailing whitespace from the
                          full text before storing. Default True.
    """

    def __init__(self, encoding: str = "utf-8", strip_whitespace: bool = True) -> None:
        self.encoding = encoding
        self.strip_whitespace = strip_whitespace

    def load(self, path: Path) -> Document:
        """Load *path* and return a Document.

        Raises:
            FileNotFoundError: if the file does not exist.
            IsADirectoryError: if *path* is a directory.
            UnicodeDecodeError: if the file cannot be decoded with self.encoding.
        """
        path = Path(path)  # accept str as well
        if not path.exists():
            raise FileNotFoundError(f"Text file not found: {path}")
        if path.is_dir():
            raise IsADirectoryError(f"Expected a file, got a directory: {path}")

        raw = path.read_text(encoding=self.encoding)
        text = raw.strip() if self.strip_whitespace else raw
        char_count = len(text)
        line_count = text.count("\n") + (1 if text else 0)

        logger.debug(
            "TextLoader loaded path=%s chars=%d lines=%d encoding=%s",
            path, char_count, line_count, self.encoding,
        )

        return Document(
            text=text,
            metadata={
                "source_path": str(path),
                "format": "text",
                "char_count": char_count,
                "line_count": line_count,
                "encoding": self.encoding,
                "file_size_bytes": path.stat().st_size,
            },
        )
```

**Design notes:**
- `load` is an instance method (not a module-level function) so the loader can be configured once and reused.
- `encoding` is injected at construction — avoids a runtime surprise when the corpus has ISO-8859-1 files.
- `metadata` carries `line_count` and `file_size_bytes` in addition to the `char_count` pattern from the worked example.
- All errors are standard library exceptions; no swallowing.


## Exercise 2 — Extended Document Model

`Document` extended with `loader_version` and a `from_file` classmethod that auto-dispatches by file extension:

```python
import csv
import io
import logging
import os
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Bump this whenever the loader logic changes in a breaking way.
_LOADER_VERSION = "1.0.0"


class Document(BaseModel):
    text: str
    metadata: dict[str, str | int | float | None]
    loader_version: str = Field(default=_LOADER_VERSION, description="Version of the loader that produced this document.")

    # Internal registry: extension -> loader callable
    _loaders: ClassVar[dict[str, str]] = {}

    @classmethod
    def from_file(cls, path: Path) -> "Document":
        """Create a Document from *path*, dispatching on file extension.

        Supported extensions: .txt, .md / .markdown, .csv

        Raises:
            ValueError: for unsupported extensions.
            FileNotFoundError: if path does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        logger.debug("Document.from_file path=%s suffix=%s", path, suffix)

        if suffix == ".txt":
            return cls._load_text(path)
        elif suffix in {".md", ".markdown"}:
            return cls._load_markdown(path)
        elif suffix == ".csv":
            # CSV returns multiple documents; from_file returns the first row's doc.
            # Use load_csv() directly for multi-document use.
            docs = cls._load_csv(path)
            if not docs:
                raise ValueError(f"CSV file is empty: {path}")
            logger.warning(
                "from_file on CSV returns only the first row document; use load_csv() for all rows."
            )
            return docs[0]
        else:
            raise ValueError(
                f"Unsupported file extension '{suffix}' for path {path}. "
                "Supported: .txt, .md, .markdown, .csv"
            )

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    @classmethod
    def _load_text(cls, path: Path) -> "Document":
        text = path.read_text(encoding="utf-8").strip()
        return cls(
            text=text,
            metadata={
                "source_path": str(path),
                "format": "text",
                "char_count": len(text),
            },
        )

    @classmethod
    def _load_markdown(cls, path: Path) -> "Document":
        text = path.read_text(encoding="utf-8")
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                text = parts[2].strip()
        return cls(
            text=text,
            metadata={
                "source_path": str(path),
                "format": "markdown",
                "char_count": len(text),
            },
        )

    @classmethod
    def _load_csv(cls, path: Path) -> list["Document"]:
        docs: list["Document"] = []
        with path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for i, row in enumerate(reader):
                text = " ".join(str(v) for v in row.values() if v is not None)
                docs.append(
                    cls(
                        text=text,
                        metadata={
                            "source_path": str(path),
                            "format": "csv",
                            "row_index": i,
                            "char_count": len(text),
                        },
                    )
                )
        return docs


def load_csv(path: Path, one_doc_per_row: bool = False) -> list[Document]:
    """Load a CSV file.

    Args:
        path: Path to the CSV file.
        one_doc_per_row: If True, each row becomes its own Document.
                         If False, the entire CSV is merged into a single Document.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(dict(row))

    if not rows:
        logger.warning("load_csv: empty file at %s", path)
        return []

    if one_doc_per_row:
        return [
            Document(
                text=" | ".join(f"{k}: {v}" for k, v in row.items()),
                metadata={
                    "source_path": str(path),
                    "format": "csv",
                    "row_index": i,
                    "char_count": sum(len(str(v)) for v in row.values()),
                },
            )
            for i, row in enumerate(rows)
        ]

    # Merge all rows into a single Document
    full_text = "\n".join(
        " | ".join(f"{k}: {v}" for k, v in row.items()) for row in rows
    )
    return [
        Document(
            text=full_text,
            metadata={
                "source_path": str(path),
                "format": "csv",
                "row_count": len(rows),
                "char_count": len(full_text),
            },
        )
    ]
```

**Key additions over the worked example:**
- `loader_version` defaults to the module-level `_LOADER_VERSION` constant — bump once, propagates everywhere.
- `from_file` is a `@classmethod` so it can construct `cls(...)` and is naturally subclassable.
- Extension dispatch is explicit `if/elif` rather than a registry dict — simpler to read and type-check.
- `load_csv` retains the `one_doc_per_row` flag from the original worked example signature.


## Exercise 3 — CSV Loader pytest

Full pytest suite using `StringIO`/`tmp_path` fixtures:

```python
"""Tests for load_csv and Document.from_file with CSV input."""
import csv
import io
import textwrap
from pathlib import Path

import pytest

# Adjust the import to match your project layout.
from day2_solution import Document, load_csv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_csv(tmp_path: Path, rows: list[dict[str, str]], filename: str = "test.csv") -> Path:
    """Write *rows* to a CSV file in *tmp_path* and return the path."""
    path = tmp_path / filename
    if not rows:
        path.write_text("col_a,col_b\n", encoding="utf-8")
        return path
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# load_csv — one_doc_per_row=False (default)
# ---------------------------------------------------------------------------

class TestLoadCsvMerged:
    def test_returns_single_document(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path, [{"name": "Alice", "role": "engineer"}])
        docs = load_csv(path)
        assert len(docs) == 1

    def test_document_type(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path, [{"a": "1"}, {"a": "2"}])
        doc = load_csv(path)[0]
        assert isinstance(doc, Document)

    def test_text_contains_all_values(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path, [{"city": "Berlin", "pop": "3600000"}])
        doc = load_csv(path)[0]
        assert "Berlin" in doc.text
        assert "3600000" in doc.text

    def test_metadata_row_count(self, tmp_path: Path) -> None:
        rows = [{"x": str(i)} for i in range(5)]
        path = write_csv(tmp_path, rows)
        doc = load_csv(path)[0]
        assert doc.metadata["row_count"] == 5

    def test_metadata_source_path(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path, [{"k": "v"}])
        doc = load_csv(path)[0]
        assert doc.metadata["source_path"] == str(path)

    def test_metadata_format_is_csv(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path, [{"k": "v"}])
        doc = load_csv(path)[0]
        assert doc.metadata["format"] == "csv"

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path, [])
        assert load_csv(path) == []

    def test_char_count_matches_text(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path, [{"col": "hello"}])
        doc = load_csv(path)[0]
        assert doc.metadata["char_count"] == len(doc.text)


# ---------------------------------------------------------------------------
# load_csv — one_doc_per_row=True
# ---------------------------------------------------------------------------

class TestLoadCsvPerRow:
    def test_returns_one_doc_per_row(self, tmp_path: Path) -> None:
        rows = [{"name": "Alice"}, {"name": "Bob"}, {"name": "Carol"}]
        path = write_csv(tmp_path, rows)
        docs = load_csv(path, one_doc_per_row=True)
        assert len(docs) == 3

    def test_row_index_metadata(self, tmp_path: Path) -> None:
        rows = [{"v": "x"}, {"v": "y"}]
        path = write_csv(tmp_path, rows)
        docs = load_csv(path, one_doc_per_row=True)
        assert docs[0].metadata["row_index"] == 0
        assert docs[1].metadata["row_index"] == 1

    def test_each_doc_contains_its_row_values(self, tmp_path: Path) -> None:
        rows = [{"fruit": "apple"}, {"fruit": "banana"}]
        path = write_csv(tmp_path, rows)
        docs = load_csv(path, one_doc_per_row=True)
        assert "apple" in docs[0].text
        assert "banana" in docs[1].text
        assert "apple" not in docs[1].text


# ---------------------------------------------------------------------------
# Document.from_file with CSV
# ---------------------------------------------------------------------------

class TestDocumentFromFileCsv:
    def test_from_file_csv_returns_document(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path, [{"col": "value"}])
        doc = Document.from_file(path)
        assert isinstance(doc, Document)

    def test_from_file_unsupported_extension_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "data.xyz"
        path.write_text("some data")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            Document.from_file(path)

    def test_from_file_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            Document.from_file(tmp_path / "missing.csv")

    def test_loader_version_present(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path, [{"k": "v"}])
        doc = Document.from_file(path)
        assert doc.loader_version  # non-empty string
        assert "." in doc.loader_version  # semver-like


# ---------------------------------------------------------------------------
# StringIO-based fixture (no tmp file on disk)
# ---------------------------------------------------------------------------

class TestLoadCsvFromStringIO:
    """Demonstrate using StringIO to build in-memory CSV without a real file.

    load_csv requires a Path, so we write to tmp_path — but the CSV content
    itself comes from a StringIO helper to keep test data self-contained.
    """

    @pytest.fixture()
    def csv_path(self, tmp_path: Path) -> Path:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["product", "price", "in_stock"])
        writer.writeheader()
        writer.writerows([
            {"product": "Widget", "price": "9.99", "in_stock": "true"},
            {"product": "Gadget", "price": "24.50", "in_stock": "false"},
            {"product": "Doohickey", "price": "4.00", "in_stock": "true"},
        ])
        path = tmp_path / "products.csv"
        path.write_text(buf.getvalue(), encoding="utf-8")
        return path

    def test_three_docs_per_row(self, csv_path: Path) -> None:
        docs = load_csv(csv_path, one_doc_per_row=True)
        assert len(docs) == 3

    def test_product_names_in_text(self, csv_path: Path) -> None:
        docs = load_csv(csv_path, one_doc_per_row=True)
        names = [d.text for d in docs]
        assert any("Widget" in t for t in names)
        assert any("Gadget" in t for t in names)
        assert any("Doohickey" in t for t in names)

    def test_merged_contains_all_products(self, csv_path: Path) -> None:
        doc = load_csv(csv_path)[0]
        assert "Widget" in doc.text
        assert "Gadget" in doc.text
        assert "Doohickey" in doc.text
```
