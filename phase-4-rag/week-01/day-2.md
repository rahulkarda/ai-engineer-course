# Phase 4 · Week 01 · Day 2 — Document Loaders & Formats

> Objective: Build a typed document loading layer that handles multiple file formats and emits normalized Document objects.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

## Concept

Document loading is the first stage of the RAG indexing pipeline. It sounds trivial — "just read a file" — but real documents are messy:

**Why loading is non-trivial:**

- **Binary formats** (PDF, DOCX, XLSX) require format-specific parsing libraries. You cannot `open()` them as text.
- **Encoding issues:** A UTF-8 file with a Latin-1 header, a Windows-1252 DOCX, a PDF with no declared encoding — each requires detection and normalization before you have usable text.
- **Layout complexity:** A two-column PDF extracted linearly reads as garbled interleaved sentences. Tables become noise unless handled explicitly (Day 4).
- **Metadata loss:** A naive loader returns text and discards the author, page number, section title, and document date — all critical for retrieval filtering and answer attribution.
- **Structural noise:** HTML pages contain nav bars, cookie banners, ads. PDF footers repeat on every page. DOCX style metadata leaks into content.

**The abstraction:** A document loader reads a source (file path, URL, stream) and returns a `list[Document]`. Each `Document` carries:
- `text: str` — normalized, decoded string content
- `metadata: dict[str, str | int | float]` — source, format, page, section, etc.

This abstraction decouples the loading concern from chunking, embedding, and storage. Swap loaders without touching downstream code.

---

### Key Formats at a Glance

| Format | Main challenge | Typical library |
|---|---|---|
| **PDF (text-layer)** | Multi-column layout, hyphenation, footer noise | `pypdf`, `pdfminer.six` |
| **PDF (scanned)** | No text layer — requires OCR | `pytesseract`, `unstructured` |
| **HTML** | Structure/noise separation, inline JS/CSS | `BeautifulSoup4`, `trafilatura` |
| **Markdown** | Semantic structure mostly preserved | stdlib `re`, `markdown-it-py` |
| **DOCX** | ZIP of XML, styles, tables, tracked changes | `python-docx` |
| **CSV** | Structured data — each row a "document" | stdlib `csv` |

> **When ready to handle PDF and DOCX in production:** `uv add pypdf unstructured`. The worked example below uses only stdlib — `pypdf` and `unstructured` are covered conceptually here and used in Week 02.

---

## How It Works Internally

### PDF: xref tables & content streams

A PDF is a binary container. The **cross-reference (xref) table** maps object IDs to byte offsets in the file. Text lives in **content streams** — sequences of PDF operators like `BT` (begin text), `Tj` (show string), `ET` (end text). Libraries like `pypdf` parse these streams and reconstruct character sequences. Without layout analysis, multi-column text is extracted in reading order across columns — which is why column-aware extraction is a hard problem.

### HTML: DOM traversal

HTML parsers (SAX or DOM) traverse the tag tree. Extracting "just the article" requires either a CSS selector heuristic (`article`, `main`, `.content`) or a readability algorithm (Mozilla Readability, `trafilatura`) that scores blocks by text density and discards navigation/ad nodes.

### DOCX: ZIP of XML

A `.docx` file is a ZIP archive containing XML files: `word/document.xml` (body), `word/styles.xml` (formatting), relationships, media. Paragraphs are `<w:p>` elements; text runs are `<w:r><w:t>`. You can parse it with `zipfile` + `xml.etree.ElementTree` without any third-party library — `python-docx` is a convenience wrapper.

### Encoding detection

Python's `open(path, encoding="utf-8")` raises `UnicodeDecodeError` on non-UTF-8 files. Production loaders use `chardet` or `charset-normalizer` (`uv add charset-normalizer`) to detect encoding before decoding. For files where detection is ambiguous, a fallback of `errors="replace"` prevents crashes at the cost of introducing replacement characters (`�`) — flag these in metadata.

### Chunking boundaries at the loader level

Some loaders can emit *natural chunks* rather than a single blob:
- A CSV loader can emit one `Document` per row.
- A DOCX loader can emit one `Document` per section (based on heading styles).
- An HTML loader can emit one `Document` per `<article>` or `<section>`.

This is "semantic chunking" at the loader level — it preserves document structure better than post-hoc character-count splitting (covered in Week 01 Day 5).

---

## Worked Example

```python
"""
day_2_document_loaders.py

Typed document loaders for Markdown and CSV formats.
No external dependencies — stdlib only.

Production note: for PDF use `pypdf`, for HTML use `BeautifulSoup4`,
for DOCX use `python-docx`. Install with: uv add pypdf unstructured
"""

import csv
import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core data model
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """
    Normalized output of any document loader.

    Attributes:
        text:      Decoded, loader-cleaned string content.
        metadata:  Provenance and structural metadata.
    """
    text: str
    metadata: dict[str, str | int | float] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"Document(chars={self.char_count}, source={self.metadata.get('source_path', '?')!r}, preview={preview!r})"


# ---------------------------------------------------------------------------
# Format enum
# ---------------------------------------------------------------------------

SupportedFormat = Literal["markdown", "csv", "txt", "pdf", "html", "docx"]

FORMAT_BY_EXTENSION: dict[str, SupportedFormat] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".csv": "csv",
    ".txt": "txt",
    # ".pdf": "pdf",   # requires pypdf — Week 02
    # ".html": "html", # requires BeautifulSoup4 — Week 02
    # ".docx": "docx", # requires python-docx — Week 02
}


# ---------------------------------------------------------------------------
# Markdown loader
# ---------------------------------------------------------------------------

_FRONT_MATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


def load_markdown(path: Path) -> list[Document]:
    """
    Load a Markdown file, stripping YAML front matter if present.

    Args:
        path: Absolute path to the .md file.

    Returns:
        Single-element list containing the Document.
    """
    raw = path.read_text(encoding="utf-8")
    logger.info("Loaded Markdown file: %s (%d chars)", path.name, len(raw))

    # Strip front matter (--- ... ---)
    text = _FRONT_MATTER_RE.sub("", raw).strip()
    if len(text) < len(raw):
        logger.debug("Stripped front matter from %s", path.name)

    doc = Document(
        text=text,
        metadata={
            "source_path": str(path),
            "format": "markdown",
            "char_count": len(text),
        },
    )
    return [doc]


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def load_csv(
    path: Path | None = None,
    source: io.StringIO | None = None,
    source_name: str = "<stream>",
    row_per_document: bool = True,
) -> list[Document]:
    """
    Load a CSV file or StringIO stream.

    Args:
        path:             Absolute path to the .csv file. Mutually exclusive with source.
        source:           StringIO stream (for testing). Mutually exclusive with path.
        source_name:      Label for the source when using a stream.
        row_per_document: If True, emit one Document per row. If False, concatenate
                          all rows into a single Document.

    Returns:
        List of Document objects.
    """
    if path is not None and source is not None:
        raise ValueError("Provide either path or source, not both.")
    if path is None and source is None:
        raise ValueError("Provide either path or source.")

    # Resolve reader and label
    if path is not None:
        raw = path.read_text(encoding="utf-8")
        label = str(path)
    else:
        assert source is not None
        raw = source.getvalue()
        label = source_name

    logger.info("Loading CSV: %s", label)

    reader = csv.DictReader(io.StringIO(raw))
    if reader.fieldnames is None:
        logger.warning("CSV %s has no header row — returning empty.", label)
        return []

    headers = list(reader.fieldnames)
    documents: list[Document] = []

    for row_idx, row in enumerate(reader):
        # Serialize row as "key: value; key: value; ..."
        row_text = "; ".join(f"{k}: {v}" for k, v in row.items() if v)

        doc = Document(
            text=row_text,
            metadata={
                "source_path": label,
                "format": "csv",
                "row_index": row_idx,
                "columns": ", ".join(headers),
                "char_count": len(row_text),
            },
        )

        if row_per_document:
            documents.append(doc)
        else:
            documents.append(doc)  # collected, concatenated below

    if not row_per_document and documents:
        combined_text = "\n".join(d.text for d in documents)
        combined_doc = Document(
            text=combined_text,
            metadata={
                "source_path": label,
                "format": "csv",
                "row_count": len(documents),
                "columns": ", ".join(headers),
                "char_count": len(combined_text),
            },
        )
        logger.info("CSV %s → 1 combined document (%d rows)", label, len(documents))
        return [combined_doc]

    logger.info("CSV %s → %d documents", label, len(documents))
    return documents


# ---------------------------------------------------------------------------
# Loader dispatcher
# ---------------------------------------------------------------------------

def load_document(path: str | Path, **kwargs) -> list[Document]:
    """
    Route to the correct loader based on file extension.

    Args:
        path:    Path to the document (str or Path).
        **kwargs: Passed to the underlying loader (e.g., row_per_document for CSV).

    Returns:
        List of Document objects.

    Raises:
        ValueError: If the extension is unsupported.
        FileNotFoundError: If the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Document not found: {p}")

    ext = p.suffix.lower()
    fmt = FORMAT_BY_EXTENSION.get(ext)

    if fmt is None:
        raise ValueError(
            f"Unsupported extension {ext!r}. "
            f"Supported: {list(FORMAT_BY_EXTENSION.keys())}. "
            "For PDF/HTML/DOCX run: uv add pypdf unstructured"
        )

    logger.info("Dispatching loader for format=%r, path=%s", fmt, p.name)

    match fmt:
        case "markdown":
            return load_markdown(p)
        case "csv":
            return load_csv(path=p, **kwargs)
        case "txt":
            # Simple text loader — no special handling needed
            text = p.read_text(encoding="utf-8")
            return [Document(
                text=text,
                metadata={"source_path": str(p), "format": "txt", "char_count": len(text)},
            )]
        case _:
            # Should not reach here given FORMAT_BY_EXTENSION guard above
            raise ValueError(f"No loader implemented for format: {fmt}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    import os

    # --- Markdown demo ---
    md_content = """---
title: RAG Fundamentals
author: AI Engineer Course
---

# RAG Overview

RAG grounds LLM responses in retrieved documents.
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as f:
        f.write(md_content)
        md_path = f.name

    md_docs = load_document(md_path)
    print("Markdown:", md_docs[0])
    os.unlink(md_path)

    # --- CSV demo ---
    csv_content = "title,author,year\nAttention is All You Need,Vaswani et al.,2017\nBERT,Devlin et al.,2018\n"
    csv_stream = io.StringIO(csv_content)
    csv_docs = load_csv(source=csv_stream, source_name="papers.csv", row_per_document=True)
    for doc in csv_docs:
        print("CSV row:", doc)
```

**Run it:**
```bash
python day_2_document_loaders.py
```

No dependencies. Creates a temp file, loads it, prints Document reprs, cleans up.

---

## Exercises

1. **Add a plain-text loader.** The dispatcher already has a `"txt"` case as a one-liner. Extend it: add `.log` and `.rst` as aliases for `"txt"` in `FORMAT_BY_EXTENSION`, then write a pytest unit test that loads a `.log` file from a `tmp_path` fixture and asserts `format == "txt"` in the metadata.

2. **Extend the Document model.** Add a `loader_version: str = "1.0.0"` field to `Document`. Then add a `from_file(cls, path: Path) -> "Document"` classmethod that reads a plain text file and returns a `Document` with `loader_version` set. Write a test asserting `loader_version` is present in the returned instance.

3. **Unit test the CSV loader.** Write a pytest test using a `io.StringIO` fixture (no disk I/O) that:
   - Loads a 3-row CSV with columns `name,score`.
   - Asserts `len(documents) == 3` when `row_per_document=True`.
   - Asserts `len(documents) == 1` when `row_per_document=False`.
   - Asserts `row_index` metadata is correct for the second row.

(Solutions: ../solutions/week-01/day-2.md)

---

## Self-check

- [ ] I can explain why binary formats (PDF, DOCX) cannot be loaded with `open(path).read()`.
- [ ] I can describe the `Document` abstraction and why separating text from metadata matters for downstream RAG stages.
- [ ] I can implement a new file-format loader by adding an entry to `FORMAT_BY_EXTENSION` and a handler in `load_document` — without modifying callers.
- [ ] I can explain what YAML front matter is and why stripping it before indexing improves retrieval quality.
