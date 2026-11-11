# Solutions — Week 02 · Day 2

## Exercise answers

### Exercise 1 — Code-fence-aware separators

```python
# chunking/recursive.py  (extended separators)
CODE_FENCE_SEPARATORS = ["```", "\n\n", "\n", " ", ""]

chunks = recursive_chunks(text, chunk_size=500, separators=CODE_FENCE_SEPARATORS)
```

The `"```"` separator is tried first. `text.split("```")` alternates between text and code. If either piece exceeds `chunk_size`, it falls back to `\n\n` recursively. This guarantees code blocks are never split mid-block as long as the block itself fits within `chunk_size`.

### Exercise 2 — `code_block` metadata

```python
# chunking/markdown_aware.py  (add to _emit_section)
import re

def _is_code_block(text: str) -> bool:
    """True if the entire text is a fenced code block."""
    stripped = text.strip()
    return stripped.startswith("```") and stripped.endswith("```") and stripped.count("```") >= 2

# In _emit_section, add to metadata:
metadata = {
    "heading": heading,
    "heading_level": level,
    "code_block": _is_code_block(body),
}
```

### Exercise 3 — HTML chunker

```python
# chunking/html_chunker.py
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_HEADING_PATTERN = re.compile(r"<h[1-3][^>]*>(.*?)</h[1-3]>", re.IGNORECASE | re.DOTALL)
_TAG_PATTERN = re.compile(r"<[^>]+>")


@dataclass
class Chunk:
    text: str
    index: int
    metadata: dict = field(default_factory=dict)


def html_chunks(html: str, min_chars: int = 50) -> list[Chunk]:
    """Split HTML at h1/h2/h3 boundaries, stripping tags from the output text."""
    parts = _HEADING_PATTERN.split(html)
    # split result: [pre-first-heading, heading1_text, body1, heading2_text, body2, ...]
    chunks: list[Chunk] = []
    idx = 0
    i = 0
    while i < len(parts):
        if i == 0:
            # Content before the first heading
            body = _TAG_PATTERN.sub("", parts[i]).strip()
            if len(body) >= min_chars:
                chunks.append(Chunk(text=body, index=idx, metadata={"heading": ""}))
                idx += 1
            i += 1
        else:
            heading_text = _TAG_PATTERN.sub("", parts[i]).strip()
            body = _TAG_PATTERN.sub("", parts[i + 1]).strip() if i + 1 < len(parts) else ""
            combined = f"{heading_text}\n\n{body}".strip()
            if len(combined) >= min_chars:
                chunks.append(Chunk(text=combined, index=idx, metadata={"heading": heading_text}))
                idx += 1
            i += 2

    logger.info("html_chunks: %d chunks", len(chunks))
    return chunks
```

### Exercise 4 — Recursive chunker test

```python
# tests/unit/test_recursive.py
import pytest
from chunking.recursive import recursive_chunks

# A separator is at most 2 chars ("\n\n"), so max chunk = chunk_size + 2
MAX_EXTRA = 2


def test_no_chunk_exceeds_limit():
    text = "word " * 200  # 1000 chars with spaces
    chunks = recursive_chunks(text, chunk_size=80)
    for c in chunks:
        assert len(c.text) <= 80 + MAX_EXTRA, f"chunk too long: {len(c.text)}"


def test_paragraph_heavy_text():
    text = "\n\n".join(["A sentence here." * 5] * 10)
    chunks = recursive_chunks(text, chunk_size=200)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 200 + MAX_EXTRA


def test_empty_input():
    assert recursive_chunks("", chunk_size=100) == []


def test_single_short_text():
    text = "Short text."
    chunks = recursive_chunks(text, chunk_size=100)
    assert len(chunks) == 1
    assert chunks[0].text == text
```

### Exercise 5 — Markdown heading metadata test

```python
# tests/unit/test_markdown.py
import pytest
from chunking.markdown_aware import markdown_chunks

MD = """# Alpha

First section content here. More content.

## Beta

Second section content here.

## Gamma

Third section with a longer body that contains more words.
"""


def test_headings_propagate():
    chunks = markdown_chunks(MD)
    headings = [c.metadata["heading"] for c in chunks]
    assert "Alpha" in headings
    assert "Beta" in headings or "Gamma" in headings


def test_each_chunk_has_heading_key():
    for c in markdown_chunks(MD):
        assert "heading" in c.metadata


def test_no_chunk_exceeds_max():
    chunks = markdown_chunks(MD, max_chunk_chars=200)
    for c in chunks:
        assert len(c.text) <= 200 + 50  # small tolerance for merged text
```
