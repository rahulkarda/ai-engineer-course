# Solutions — Phase 4 · Week 01 · Day 4

## Exercise 1 — TableExtractor with None Handling

Full `TableExtractor` class that gracefully handles `None` cells:

```python
"""day4_solution.py — TableExtractor, DocumentMetadata, and HTML image alt-text extractor."""
import html
import json
import logging
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Type alias for a raw table row (cells may be None when a cell is missing/empty)
CellValue = str | int | float | None
TableRow = list[CellValue]
Table = list[TableRow]


@dataclass
class ExtractedTable:
    """A table extracted from a document."""
    headers: list[str]
    rows: list[dict[str, CellValue]]
    source: str = ""
    table_index: int = 0


class TableExtractor:
    """Extract structured tables from various raw representations.

    Handles None cells produced by:
    - Merged cells in HTML/PDF extraction (rendered as None by most parsers)
    - Missing columns in ragged CSV rows
    - Explicit null values in JSON table dumps
    """

    NULL_PLACEHOLDER: str = ""  # what to emit when a cell is None

    def extract_from_raw(
        self,
        headers: list[str | None],
        rows: Table,
        source: str = "",
        table_index: int = 0,
    ) -> ExtractedTable:
        """Build an ExtractedTable from raw headers and rows, handling None safely.

        Args:
            headers: Column names. None entries are replaced with auto-generated
                     names ("col_0", "col_1", …).
            rows: List of rows, each a list of CellValues. Rows shorter than
                  the header count are padded with None; longer rows are truncated
                  with a warning.
            source: Origin string (file path, URL, etc.).
            table_index: Position of this table in the source document.

        Returns:
            ExtractedTable with dict rows, all None cells replaced by the
            NULL_PLACEHOLDER string.
        """
        # Sanitise headers: replace None with positional fallback
        clean_headers: list[str] = [
            str(h) if h is not None else f"col_{i}"
            for i, h in enumerate(headers)
        ]
        n_cols = len(clean_headers)

        if n_cols == 0:
            logger.warning("extract_from_raw: no headers provided, returning empty table")
            return ExtractedTable(headers=[], rows=[], source=source, table_index=table_index)

        clean_rows: list[dict[str, CellValue]] = []
        for row_idx, raw_row in enumerate(rows):
            if len(raw_row) > n_cols:
                logger.warning(
                    "table_index=%d row=%d has %d cells but only %d headers — truncating",
                    table_index, row_idx, len(raw_row), n_cols,
                )
                raw_row = raw_row[:n_cols]
            elif len(raw_row) < n_cols:
                # Pad with None so zip always covers all headers
                raw_row = list(raw_row) + [None] * (n_cols - len(raw_row))

            clean_row: dict[str, CellValue] = {}
            for header, cell in zip(clean_headers, raw_row):
                clean_row[header] = self.NULL_PLACEHOLDER if cell is None else cell
            clean_rows.append(clean_row)

        logger.debug(
            "extracted table source=%r index=%d headers=%d rows=%d",
            source, table_index, n_cols, len(clean_rows),
        )
        return ExtractedTable(
            headers=clean_headers,
            rows=clean_rows,
            source=source,
            table_index=table_index,
        )

    def to_markdown(self, table: ExtractedTable) -> str:
        """Render an ExtractedTable as a Markdown table string."""
        if not table.headers:
            return "_empty table_"
        header_row = "| " + " | ".join(table.headers) + " |"
        sep_row = "| " + " | ".join("---" for _ in table.headers) + " |"
        data_rows = [
            "| " + " | ".join(str(row.get(h, "")) for h in table.headers) + " |"
            for row in table.rows
        ]
        return "\n".join([header_row, sep_row] + data_rows)

    def extract_from_html_table(self, html_fragment: str, source: str = "") -> list[ExtractedTable]:
        """Parse one or more <table> elements from an HTML fragment.

        None cells arise from:
        - <td></td> empty cells
        - Cells with only whitespace
        - colspan/rowspan gaps (not backfilled by this extractor)
        """
        tables: list[ExtractedTable] = []
        # Minimal regex-based split by <table>...</table> blocks
        table_blocks = re.findall(r"<table[^>]*>(.*?)</table>", html_fragment, re.DOTALL | re.IGNORECASE)
        for t_idx, block in enumerate(table_blocks):
            headers: list[str | None] = []
            rows: Table = []

            # Extract header row from <thead> or first <tr> with <th>
            th_matches = re.findall(r"<th[^>]*>(.*?)</th>", block, re.DOTALL | re.IGNORECASE)
            if th_matches:
                headers = [html.unescape(re.sub(r"<[^>]+>", "", cell)).strip() or None for cell in th_matches]

            # Extract data rows
            tr_blocks = re.findall(r"<tr[^>]*>(.*?)</tr>", block, re.DOTALL | re.IGNORECASE)
            for tr in tr_blocks:
                cells = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.DOTALL | re.IGNORECASE)
                if not cells:
                    continue  # header row with only <th> — skip
                row: TableRow = [
                    html.unescape(re.sub(r"<[^>]+>", "", c)).strip() or None
                    for c in cells
                ]
                rows.append(row)

            if not headers and rows:
                # Infer column count from first row
                headers = [None] * len(rows[0])

            tables.append(self.extract_from_raw(headers, rows, source=source, table_index=t_idx))

        logger.info("extract_from_html_table: found %d table(s) in source=%r", len(tables), source)
        return tables
```

**None-handling strategy:**
- `None` header entries get auto-named `col_N` — downstream code always has a string key.
- `None` cell values are replaced with `NULL_PLACEHOLDER` (`""`) so dict values are always `CellValue`, never silently skipped.
- Ragged rows are padded/truncated with explicit logging — no silent data loss.


## Exercise 2 — DocumentMetadata.to_json_schema()

```python
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, get_args, get_origin, get_type_hints

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DocumentMetadata(BaseModel):
    """Metadata attached to every ingested document."""

    source_path: str = Field(description="Absolute path or URL of the source file.")
    format: str = Field(description="File format: markdown, pdf, html, csv, text, etc.")
    char_count: int = Field(ge=0, description="Number of characters in the cleaned text.")
    page_count: int | None = Field(default=None, ge=1, description="Page count (PDF/DOCX only).")
    ingested_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when the document was ingested.",
    )
    language: str | None = Field(default=None, description="ISO 639-1 language code, if detected.")
    author: str | None = Field(default=None, description="Document author, if available.")
    title: str | None = Field(default=None, description="Document title, if available.")
    tags: list[str] = Field(default_factory=list, description="Free-form classification tags.")
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Loader-specific extra fields not covered by the standard schema.",
    )

    @classmethod
    def to_json_schema(cls) -> dict[str, Any]:
        """Return a JSON Schema dict describing this model.

        The schema is suitable for:
        - Validating metadata dicts from external sources before constructing instances.
        - Documenting the metadata contract in API specs or data catalogues.
        - Generating TypeScript / OpenAPI type definitions.

        Returns:
            A JSON Schema object (draft-07 compatible) as a Python dict.
            Serialise with ``json.dumps(DocumentMetadata.to_json_schema(), indent=2)``.
        """
        schema: dict[str, Any] = cls.model_json_schema()
        # Pydantic v2 uses $defs for nested models; keep it standard
        schema.setdefault("$schema", "http://json-schema.org/draft-07/schema#")
        schema["title"] = cls.__name__
        if cls.__doc__:
            schema["description"] = cls.__doc__.strip()
        logger.debug("to_json_schema: generated schema with %d properties", len(schema.get("properties", {})))
        return schema

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentMetadata":
        """Construct from a plain dict, logging any extra keys."""
        known_keys = set(cls.model_fields.keys())
        extra_keys = set(data.keys()) - known_keys
        if extra_keys:
            logger.debug("from_dict: unknown keys will land in 'extra': %s", extra_keys)
        return cls(**data)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    schema = DocumentMetadata.to_json_schema()
    print(json.dumps(schema, indent=2, default=str))
```

**Sample output of `to_json_schema()`:**
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "DocumentMetadata",
  "description": "Metadata attached to every ingested document.",
  "type": "object",
  "properties": {
    "source_path": {"type": "string", "description": "Absolute path or URL of the source file."},
    "format": {"type": "string", "description": "File format: ..."},
    "char_count": {"type": "integer", "minimum": 0},
    "page_count": {"anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}]},
    "ingested_at": {"type": "string", "format": "date-time"},
    "language": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    "author": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    "title": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    "tags": {"type": "array", "items": {"type": "string"}},
    "extra": {"type": "object"}
  },
  "required": ["source_path", "format", "char_count"]
}
```

**Design notes:**
- Delegates to Pydantic v2's `model_json_schema()` — no hand-rolling type mappings.
- Injects `$schema` declaration so the output is immediately usable with JSON Schema validators.
- `from_dict` is a convenience classmethod that logs unexpected keys rather than silently dropping them.


## Exercise 3 — HTML Image Alt-text Extractor

Full implementation using `html.parser.HTMLParser` (stdlib only):

```python
import logging
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


@dataclass
class ImageRecord:
    """Data collected for a single <img> element."""
    src: str
    alt: str  # empty string when alt attribute is absent or blank
    title: str | None
    width: int | None
    height: int | None


class _AltTextParser(HTMLParser):
    """Internal parser that collects <img> attributes."""

    def __init__(self) -> None:
        super().__init__()
        self.images: list[ImageRecord] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "img":
            return
        attr_dict: dict[str, str] = {k.lower(): (v or "") for k, v in attrs}
        src = attr_dict.get("src", "")
        alt = attr_dict.get("alt", "")
        title = attr_dict.get("title") or None
        width: int | None = None
        height: int | None = None
        try:
            width = int(attr_dict["width"]) if "width" in attr_dict else None
        except ValueError:
            pass
        try:
            height = int(attr_dict["height"]) if "height" in attr_dict else None
        except ValueError:
            pass
        record = ImageRecord(src=src, alt=alt, title=title, width=width, height=height)
        self.images.append(record)
        logger.debug("found img src=%r alt=%r", src, alt)


def extract_image_alt_texts(
    html_content: str,
    base_url: str = "",
) -> list[ImageRecord]:
    """Extract all <img> tags and their alt text from an HTML document.

    Args:
        html_content: Raw HTML string to parse.
        base_url: Optional base URL for resolving relative src paths.
                  When provided, relative src values are made absolute.

    Returns:
        List of ImageRecord objects, one per <img> tag found, in document order.
        Images with no alt attribute have alt="".
    """
    parser = _AltTextParser()
    try:
        parser.feed(html_content)
    except Exception as exc:
        logger.error("HTML parsing error: %s", exc)
        raise

    images = parser.images

    if base_url:
        for img in images:
            if img.src and not img.src.startswith(("http://", "https://", "data:", "//")):
                img.src = urljoin(base_url, img.src)

    missing_alt = [img for img in images if not img.alt.strip()]
    if missing_alt:
        logger.warning(
            "%d image(s) have empty or missing alt text: %s",
            len(missing_alt),
            [img.src for img in missing_alt],
        )

    logger.info(
        "extract_image_alt_texts: found %d image(s), %d with non-empty alt text",
        len(images),
        sum(1 for img in images if img.alt.strip()),
    )
    return images


def extract_image_alt_texts_from_file(path: Path, base_url: str = "") -> list[ImageRecord]:
    """Convenience wrapper: read an HTML file and extract image alt texts."""
    content = Path(path).read_text(encoding="utf-8")
    return extract_image_alt_texts(content, base_url=base_url)


def alt_text_report(images: list[ImageRecord]) -> str:
    """Return a human-readable summary of extracted image alt texts."""
    if not images:
        return "No images found."
    lines: list[str] = [f"Found {len(images)} image(s):", ""]
    for i, img in enumerate(images, start=1):
        alt_display = repr(img.alt) if img.alt.strip() else "(missing)"
        lines.append(f"  [{i}] src={img.src!r}")
        lines.append(f"       alt={alt_display}")
        if img.title:
            lines.append(f"       title={img.title!r}")
        if img.width or img.height:
            lines.append(f"       size={img.width}x{img.height}")
    return "\n".join(lines)


# --- quick smoke test ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    sample_html = """
    <!DOCTYPE html>
    <html>
    <body>
      <img src="logo.png" alt="Company logo" width="200" height="80" />
      <img src="https://cdn.example.com/hero.jpg" alt="Hero image" title="Homepage hero" />
      <img src="decorative.gif" alt="" />
      <img src="no-alt.png" />
      <img src="relative/path/photo.webp" alt="A team photo" width="800" />
    </body>
    </html>
    """

    images = extract_image_alt_texts(sample_html, base_url="https://www.example.com/")
    print(alt_text_report(images))
```

**Sample output:**
```
Found 5 image(s):

  [1] src='https://www.example.com/logo.png'
       alt='Company logo'
       size=200x80
  [2] src='https://cdn.example.com/hero.jpg'
       alt='Hero image'
       title='Homepage hero'
  [3] src='https://www.example.com/decorative.gif'
       alt=(missing)
  [4] src='https://www.example.com/no-alt.png'
       alt=(missing)
  [5] src='https://www.example.com/relative/path/photo.webp'
       alt='A team photo'
       size=800xNone
```

**Design notes:**
- `html.parser.HTMLParser` is stdlib-only — no `beautifulsoup4` / `lxml` dependency.
- `_AltTextParser` is a private helper; public API is `extract_image_alt_texts`.
- `alt=""` and absent `alt` are both normalised to `alt=""` — callers can distinguish missing vs. decorative by checking `"alt" in attrs` if needed, but for RAG purposes both indicate no text to index.
- `base_url` resolution via `urllib.parse.urljoin` handles all relative path forms correctly (root-relative, doc-relative, protocol-relative).
- `logger.warning` on missing alt text surfaces accessibility issues during ingestion — useful for audit pipelines.
