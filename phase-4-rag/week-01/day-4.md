# Phase 4 · Week 01 · Day 4 — Tables, Images & Metadata Handling

> Objective: Handle structured content (tables, images) and build a metadata propagation model that survives the full RAG pipeline.  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

## Concept

Days 1–3 treated documents as text streams. Real documents are not text streams — they contain tables, figures, and structured data that the "read as plain text" approach either silently drops or garbles.

### Why Tables Break RAG

Tables carry meaning in their *structure*, not their text. Consider:

```
| Product  | Q1 Revenue | Q2 Revenue |
|----------|------------|------------|
| Widget A | $1.2M      | $1.4M      |
| Widget B | $0.8M      | $0.9M      |
```

If you extract this as a plain text stream, you get: `"Product Q1 Revenue Q2 Revenue Widget A 1.2M 1.4M Widget B 0.8M 0.9M"`. A retrieval query for "Widget B Q2 revenue" may fail — the relationship between row label and column value is gone.

**Table challenges in the wild:**
- **Merged cells:** A header spanning multiple columns is represented as a single cell in DOCX XML but has no clear serialization to linear text.
- **Multi-header rows:** Two rows of headers stacked — the column meaning is the *concatenation* of both rows.
- **Implicit column context:** A table with columns `[Name, Jan, Feb, Mar]` — the month names are context; a row serialized without them loses meaning.
- **Sparse tables:** Many cells are empty; a naive serializer produces `"name: ; score: ; rank: 3"` which misleads the retriever.

### Why Images Are Hard

Images break the text assumption entirely. An image is not text — it requires a different modality:

- **Figures with captions:** The caption text is indexable, but the diagram content is not (without OCR or vision models).
- **Embedded text in diagrams:** Architecture diagrams, flowcharts, screenshots all contain text that readers rely on but is invisible to a text extractor.
- **Alt text:** HTML `<img alt="...">` and DOCX `<w:docPr descr="...">` contain author-provided descriptions — often terse but indexable.

For production table/image extraction from PDFs, use `unstructured` (`uv add unstructured`). It uses bounding-box detection and layout analysis to identify table regions and extract them as structured elements. For the worked example today, we implement table serialization directly — the format that `unstructured` would hand off to you.

### Why Metadata Matters for Retrieval Precision

A RAG system without metadata is a bag of chunks. With metadata:

- **Source filtering:** "Only answer from documents dated after 2024."
- **Section scoping:** "This question is about section 3 — retrieve only section 3 chunks."
- **Author/provenance:** "Show only documents from the legal team."
- **Page-level deduplication:** Avoid returning five chunks from the same page.

Metadata must be *propagated* — every transformation (load → clean → chunk → embed) must carry the metadata forward without silent loss. If a chunk reaches the vector store without `source_path`, you cannot trace an answer back to its source document.

---

## How It Works Internally

### PDF Table Detection: Bounding Boxes

`pdfminer.six` and `unstructured` detect tables by analyzing the *spatial layout* of text elements. Each character in a PDF has an (x, y) bounding box. Table detection heuristics:
1. Cluster text elements into rows by y-coordinate proximity.
2. Cluster columns by x-coordinate alignment.
3. If a region has consistent horizontal and vertical alignment — it's likely a table.
4. Extract as a 2D grid (list of lists).

This is why PDF table extraction is fragile — slight misalignment in the source PDF breaks the clustering.

### DOCX Tables: XML Elements

In `word/document.xml`, a table is:
```xml
<w:tbl>
  <w:tr>  <!-- table row -->
    <w:tc><w:p><w:r><w:t>Cell text</w:t></w:r></w:p></w:tc>
    <w:tc>...</w:tc>
  </w:tr>
</w:tbl>
```

Tables are explicit and structured — unlike PDF bounding-box extraction. `python-docx` gives you `table.rows[i].cells[j].text`. The challenge is merged cells (`<w:gridSpan>`) which a cell may span multiple columns.

### Caption-to-Figure Association

In DOCX, a figure is an `<w:drawing>` element; its caption is the *next* `<w:p>` element with `Caption` style. You associate them by XML sibling traversal. In PDF, caption association requires spatial proximity analysis — the paragraph closest below an image bounding box is likely its caption.

### Metadata Propagation

Each stage of the pipeline adds metadata without replacing prior metadata:

```
Document(text="...", metadata={"source_path": "report.pdf"})
    │ clean
    ▼
Document(text="...", metadata={"source_path": "report.pdf", "cleaned": True})
    │ chunk
    ▼
Chunk(text="...", metadata={"source_path": "report.pdf", "cleaned": True, "chunk_index": 3, "page": 2})
    │ embed + store
    ▼
VectorStore record: {vector: [...], metadata: {all of the above}}
```

If any stage *overwrites* `metadata` instead of *merging* it, upstream provenance is lost. This is a common pipeline bug — `doc.metadata = new_metadata` vs `doc.metadata = {**doc.metadata, **new_metadata}`.

---

## Worked Example

```python
"""
day_4_tables_images_metadata.py

Table serialization, metadata modeling, and metadata enrichment.
No external dependencies — stdlib only.

Production note: for PDF/DOCX table extraction use:
    uv add unstructured
unstructured handles bounding-box detection, merged cells,
and multi-modal extraction (images, tables) from PDFs.
"""

import logging
import os
import html.parser
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Document model (extending Day 2)
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """Normalized document with text and metadata."""
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Table serialization
# ---------------------------------------------------------------------------

# A table is a list of rows; each row is a list of cell strings (or None).
TableData = list[list[str | None]]


class TableExtractor:
    """
    Serialize a 2D table (list-of-lists) to text representations.

    Two output formats:
    - Markdown table: preserves column structure, good for LLM context.
    - Row narrative: each row as a sentence, good for embedding individual facts.
    """

    def __init__(self, empty_cell_placeholder: str = "") -> None:
        """
        Args:
            empty_cell_placeholder: String to use for None/empty cells.
                                    Use "" to skip empty cells in narratives.
        """
        self._empty = empty_cell_placeholder

    def _resolve_cell(self, cell: str | None) -> str:
        """Return cell text or placeholder for None/empty cells."""
        if cell is None or cell.strip() == "":
            return self._empty
        return cell.strip()

    def to_markdown(self, table: TableData) -> str:
        """
        Serialize table as a Markdown table string.

        The first row is treated as the header row.
        Empty cells render as the placeholder string.

        Args:
            table: list[list[str | None]] — rows x columns.

        Returns:
            Markdown table string, or empty string if table is empty.
        """
        if not table:
            logger.warning("to_markdown called with empty table.")
            return ""

        resolved = [[self._resolve_cell(cell) for cell in row] for row in table]

        # Determine column widths for alignment
        n_cols = max(len(row) for row in resolved)
        # Pad short rows
        resolved = [row + [""] * (n_cols - len(row)) for row in resolved]

        col_widths = [
            max(len(resolved[r][c]) for r in range(len(resolved)))
            for c in range(n_cols)
        ]

        def format_row(row: list[str]) -> str:
            cells = (cell.ljust(col_widths[c]) for c, cell in enumerate(row))
            return "| " + " | ".join(cells) + " |"

        lines: list[str] = []
        lines.append(format_row(resolved[0]))  # header
        lines.append("| " + " | ".join("-" * w for w in col_widths) + " |")  # separator
        for row in resolved[1:]:
            lines.append(format_row(row))

        result = "\n".join(lines)
        logger.debug("to_markdown: %d rows x %d cols", len(table), n_cols)
        return result

    def to_row_narratives(
        self,
        table: TableData,
        include_empty: bool = False,
    ) -> list[str]:
        """
        Serialize each data row as a natural-language sentence.

        Uses the first row as column headers.
        Example: row ["Widget A", "$1.2M"] with headers ["Product", "Q1 Revenue"]
                 → "Product: Widget A; Q1 Revenue: $1.2M"

        Args:
            table:         list[list[str | None]] — rows x columns.
            include_empty: If False, skip key-value pairs where the value is empty.

        Returns:
            List of sentence strings, one per data row (excluding header).
        """
        if not table or len(table) < 2:
            logger.warning("to_row_narratives: table has fewer than 2 rows (need header + data).")
            return []

        headers = [self._resolve_cell(h) or f"col_{i}" for i, h in enumerate(table[0])]
        narratives: list[str] = []

        for row in table[1:]:
            parts: list[str] = []
            for i, cell in enumerate(row):
                header = headers[i] if i < len(headers) else f"col_{i}"
                value = self._resolve_cell(cell)
                if value or include_empty:
                    parts.append(f"{header}: {value}")
            narratives.append("; ".join(parts))

        logger.debug("to_row_narratives: %d data rows → %d narratives", len(table) - 1, len(narratives))
        return narratives


# ---------------------------------------------------------------------------
# Document metadata model
# ---------------------------------------------------------------------------

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """
    Structured metadata propagated through the RAG pipeline.

    All fields are optional to accommodate partial metadata
    at different pipeline stages.
    """
    source_path: str
    page_number: int | None = None
    section_title: str | None = None
    author: str | None = None
    created_at: datetime | None = None
    format: str | None = None
    char_count: int | None = None
    custom: dict[str, str | int | float | bool] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}

    def to_flat_dict(self) -> dict[str, Any]:
        """
        Serialize to a flat dict for vector store metadata fields.
        Nested `custom` dict is merged into the top level.
        None values are excluded.
        """
        base = self.model_dump(exclude_none=True, exclude={"custom"})
        if self.created_at is not None:
            base["created_at"] = self.created_at.isoformat()
        base.update(self.custom)
        return base


# ---------------------------------------------------------------------------
# Metadata enrichment
# ---------------------------------------------------------------------------

def enrich_metadata(doc: Document, **kwargs: Any) -> Document:
    """
    Merge additional metadata into a Document without losing existing fields.

    Always merges — never overwrites the full metadata dict.
    Logs a warning if a key is being overwritten (potential pipeline bug).

    Args:
        doc:     Source Document.
        **kwargs: Additional metadata key-value pairs to merge.

    Returns:
        New Document with merged metadata (original is not mutated).
    """
    for key in kwargs:
        if key in doc.metadata:
            logger.warning(
                "enrich_metadata: overwriting existing key %r "
                "(old=%r, new=%r) — check pipeline ordering.",
                key, doc.metadata[key], kwargs[key],
            )

    merged = {**doc.metadata, **kwargs}
    enriched = Document(text=doc.text, metadata=merged)
    logger.debug("enrich_metadata: added %d keys, total metadata keys: %d", len(kwargs), len(merged))
    return enriched


# ---------------------------------------------------------------------------
# Alt-text extraction from HTML (stdlib html.parser)
# ---------------------------------------------------------------------------

class _AltTextCollector(html.parser.HTMLParser):
    """Collect alt attributes from <img> tags."""

    def __init__(self) -> None:
        super().__init__()
        self.alt_texts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "img":
            attr_dict = dict(attrs)
            alt = attr_dict.get("alt")
            if alt and alt.strip():
                self.alt_texts.append(alt.strip())


def extract_image_alt_texts(html_content: str) -> list[str]:
    """
    Extract all non-empty alt attributes from <img> tags in HTML content.

    Args:
        html_content: Raw HTML string.

    Returns:
        List of alt text strings.
    """
    collector = _AltTextCollector()
    collector.feed(html_content)
    logger.info("Extracted %d image alt texts from HTML.", len(collector.alt_texts))
    return collector.alt_texts


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Table serialization
    revenue_table: TableData = [
        ["Product",  "Q1 Revenue", "Q2 Revenue"],
        ["Widget A", "$1.2M",      "$1.4M"],
        ["Widget B", "$0.8M",      None],        # missing Q2
        ["Widget C", None,         "$2.1M"],     # missing Q1
    ]

    extractor = TableExtractor(empty_cell_placeholder="N/A")

    print("=== Markdown Table ===")
    print(extractor.to_markdown(revenue_table))

    print("\n=== Row Narratives ===")
    for narrative in extractor.to_row_narratives(revenue_table):
        print(" •", narrative)

    # 2. DocumentMetadata
    meta = DocumentMetadata(
        source_path="reports/q2-2025.pdf",
        page_number=4,
        section_title="Revenue Summary",
        author="Finance Team",
        created_at=datetime(2025, 7, 1),
        format="pdf",
        char_count=8420,
        custom={"confidentiality": "internal", "version": 2},
    )
    print("\n=== DocumentMetadata flat dict ===")
    import json
    print(json.dumps(meta.to_flat_dict(), indent=2, default=str))

    # 3. Metadata enrichment
    doc = Document(
        text="Revenue summary for Q2 2025.",
        metadata={"source_path": "reports/q2-2025.pdf", "format": "pdf"},
    )
    enriched = enrich_metadata(doc, page_number=4, section_title="Revenue Summary", chunk_index=0)
    print("\n=== Enriched metadata ===")
    print(enriched.metadata)

    # 4. Image alt text extraction
    html_sample = """
    <html><body>
      <img src="fig1.png" alt="Architecture diagram showing three-stage RAG pipeline" />
      <img src="logo.png" alt="" />
      <img src="chart.png" alt="Q2 revenue by product line" />
    </body></html>
    """
    alt_texts = extract_image_alt_texts(html_sample)
    print("\n=== Image alt texts ===")
    for alt in alt_texts:
        print(" •", alt)
```

**Run it:**
```bash
python day_4_tables_images_metadata.py
```

Requires `pydantic` (`uv add pydantic`). All other code uses stdlib.

---

## Exercises

1. **Handle missing cells in narratives.** The current `to_row_narratives` skips `None`/empty cells when `include_empty=False`. Extend `TableExtractor` so that when a cell is `None` (as opposed to `""`), the narrative marks it explicitly as `"not available"` regardless of `include_empty`. Write a pytest test with a table that has `None` in a cell and assert the narrative contains `"not available"`.

2. **JSON schema for DocumentMetadata.** Add a `to_json_schema()` classmethod to `DocumentMetadata` that returns the Pydantic JSON schema as a `dict`. This is useful when registering metadata fields with a vector store. Write a test asserting the schema contains `"source_path"` as a required property.

3. **Extract image alt text from a Document.** Write a function `extract_alt_texts_from_doc(doc: Document) -> list[str]` that checks `doc.metadata.get("format") == "html"`, then calls `extract_image_alt_texts(doc.text)`. If the format is not HTML, log a warning and return `[]`. Write a test for both the HTML path and the non-HTML path.

(Solutions: ../solutions/week-01/day-4.md)

---

## Self-check

- [ ] I can explain why table extraction from PDFs is structurally harder than from DOCX files.
- [ ] I can implement both a Markdown serialization and a row-narrative serialization of a table, and explain when each is better for RAG.
- [ ] I can implement `enrich_metadata` with the merge-not-overwrite contract and explain why overwriting metadata is a pipeline bug.
- [ ] I can use `html.parser` from stdlib to extract alt text from HTML without any external dependencies.
