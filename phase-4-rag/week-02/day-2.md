# Phase 4 · Week 02 · Day 2 — Recursive & Structure-Aware Chunking

> Objective: Implement recursive character splitting and document-structure-aware chunking (headers, paragraphs, code blocks).  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

## Concept

Fixed-size and sentence chunking are text-agnostic. They work on any string. But real documents have **structure**: Markdown headings, HTML tags, code fences, bullet lists. Ignoring structure can split a code block in the middle of a function or split a section heading from its first paragraph.

**Recursive splitting** improves on pure fixed-size by trying a hierarchy of separators in order. Split on `\n\n` first (paragraph breaks). If any piece is still too large, split on `\n`, then on ` `, then character-by-character as a last resort. This keeps paragraph semantics intact when possible and only falls back to crude splits when necessary.

**Structure-aware chunking** goes further: parse document structure explicitly (Markdown AST, HTML DOM, docx outline) and emit chunks that align with structural boundaries — one chunk per section, one per code block, one per list. Each chunk inherits its heading as metadata, making retrieval far more precise.

## How it works internally

### Recursive splitting algorithm

```
function recursive_split(text, separators, chunk_size):
    sep = separators[0]
    pieces = text.split(sep)
    result = []
    current = ""
    for piece in pieces:
        if len(current) + len(piece) <= chunk_size:
            current += sep + piece
        else:
            if current: result.append(current.strip())
            if len(piece) > chunk_size:
                # recurse with next separator
                result += recursive_split(piece, separators[1:], chunk_size)
            else:
                current = piece
    if current: result.append(current.strip())
    return result
```

The key insight: try the *most structural* separator first. Only descend when forced.

### Structure-aware (Markdown)

Parse with a simple state machine that tracks heading level. When a new heading is encountered and the current section is long enough, flush the current section as a chunk and start a new one. This guarantees each chunk starts at a heading and ends before the next.

## Worked example

```python
# chunking/recursive.py
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DEFAULT_SEPARATORS = ["\n\n", "\n", " ", ""]


@dataclass
class Chunk:
    text: str
    index: int
    metadata: dict = field(default_factory=dict)


def _merge(pieces: list[str], separator: str, chunk_size: int) -> list[str]:
    """Merge small adjacent pieces into chunks up to chunk_size."""
    merged: list[str] = []
    current = ""
    for piece in pieces:
        candidate = (current + separator + piece).lstrip(separator) if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                merged.append(current)
            current = piece
    if current:
        merged.append(current)
    return merged


def _recursive_split(
    text: str,
    separators: list[str],
    chunk_size: int,
) -> list[str]:
    if not text:
        return []

    sep = separators[0]
    remaining_seps = separators[1:]

    if sep:
        pieces = text.split(sep)
    else:
        pieces = list(text)

    good: list[str] = []
    for piece in pieces:
        if len(piece) <= chunk_size:
            good.append(piece)
        elif remaining_seps:
            good.extend(_recursive_split(piece, remaining_seps, chunk_size))
        else:
            # No more separators — hard cut.
            for i in range(0, len(piece), chunk_size):
                good.append(piece[i : i + chunk_size])

    return _merge(good, sep, chunk_size)


def recursive_chunks(
    text: str,
    chunk_size: int = 500,
    separators: list[str] | None = None,
) -> list[Chunk]:
    """Recursively split *text* using a separator hierarchy.

    Args:
        text: Input text.
        chunk_size: Target maximum characters per chunk.
        separators: Ordered list of separators, most structural first.
                    Defaults to [double-newline, newline, space, char].

    Returns:
        Ordered list of Chunk objects.
    """
    seps = separators if separators is not None else DEFAULT_SEPARATORS
    raw = _recursive_split(text, seps, chunk_size)
    chunks = [Chunk(text=t, index=i) for i, t in enumerate(raw) if t.strip()]
    logger.info("recursive_chunks: %d chunks from %d chars", len(chunks), len(text))
    return chunks
```

```python
# chunking/markdown_aware.py
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_HEADING = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)
_CODE_FENCE = re.compile(r"^```", re.MULTILINE)


@dataclass
class Chunk:
    text: str
    index: int
    metadata: dict = field(default_factory=dict)


def markdown_chunks(
    text: str,
    min_chunk_chars: int = 100,
    max_chunk_chars: int = 1500,
) -> list[Chunk]:
    """Split Markdown at heading boundaries.

    Each chunk contains one section (heading + body). Sections shorter than
    *min_chunk_chars* are merged with the next. Sections exceeding
    *max_chunk_chars* are sub-split on paragraph breaks.

    Metadata keys set per chunk:
        heading (str): The nearest heading above the chunk.
        heading_level (int): 1–6.
    """
    lines = text.splitlines(keepends=True)
    sections: list[tuple[str, str, int]] = []  # (heading, body, level)
    current_heading = ""
    current_level = 0
    current_body: list[str] = []

    for line in lines:
        m = _HEADING.match(line)
        if m:
            if current_body or current_heading:
                sections.append((current_heading, "".join(current_body), current_level))
            current_heading = m.group(2).strip()
            current_level = len(m.group(1))
            current_body = []
        else:
            current_body.append(line)

    if current_body or current_heading:
        sections.append((current_heading, "".join(current_body), current_level))

    chunks: list[Chunk] = []
    idx = 0
    pending_heading = ""
    pending_level = 0
    pending_body = ""

    for heading, body, level in sections:
        combined = (f"# {heading}\n\n" if heading else "") + body.strip()

        if len(combined) < min_chunk_chars and pending_body:
            # Merge into pending.
            pending_body += "\n\n" + combined
            continue

        if pending_body:
            _emit_section(pending_body, pending_heading, pending_level, max_chunk_chars, chunks, idx)
            idx = len(chunks)

        pending_heading = heading
        pending_level = level
        pending_body = combined

    if pending_body:
        _emit_section(pending_body, pending_heading, pending_level, max_chunk_chars, chunks, idx)

    logger.info("markdown_chunks: %d chunks", len(chunks))
    return chunks


def _emit_section(
    body: str,
    heading: str,
    level: int,
    max_chars: int,
    chunks: list[Chunk],
    start_idx: int,
) -> None:
    """Flush a section, sub-splitting on paragraph breaks if it exceeds max_chars."""
    if len(body) <= max_chars:
        chunks.append(
            Chunk(
                text=body,
                index=start_idx,
                metadata={"heading": heading, "heading_level": level},
            )
        )
        return

    paragraphs = re.split(r"\n{2,}", body)
    current = ""
    local_idx = start_idx
    for para in paragraphs:
        if len(current) + len(para) <= max_chars:
            current = (current + "\n\n" + para).lstrip()
        else:
            if current:
                chunks.append(
                    Chunk(
                        text=current,
                        index=local_idx,
                        metadata={"heading": heading, "heading_level": level},
                    )
                )
                local_idx += 1
            current = para
    if current:
        chunks.append(
            Chunk(
                text=current,
                index=local_idx,
                metadata={"heading": heading, "heading_level": level},
            )
        )
```

```python
# Quick demo
if __name__ == "__main__":
    import logging
    from chunking.recursive import recursive_chunks
    from chunking.markdown_aware import markdown_chunks

    logging.basicConfig(level=logging.INFO)

    md = """# Introduction

RAG improves LLM grounding by injecting retrieved context.

## Why chunk?

Documents exceed context windows. Chunking splits them into
indexable pieces.

## Chunk size tradeoffs

Smaller chunks: higher precision, lower recall.
Larger chunks: lower precision, higher recall.

```python
def example():
    return "code block stays intact"
```

## Summary

Choose chunk size empirically on your data.
"""

    for c in recursive_chunks(md, chunk_size=200):
        print(f"[rec {c.index}] {repr(c.text[:60])}")

    print()
    for c in markdown_chunks(md):
        print(f"[md {c.index}] heading={c.metadata.get('heading')!r} | {repr(c.text[:60])}")
```

## Exercises

1. Extend `recursive_chunks` to accept code-fence-aware separators. Add `"```"` as a separator before `\n\n` so code blocks are never split mid-block.
2. Add a `code_block` boolean to `Chunk.metadata` in `markdown_chunks` — set it `True` when the chunk is entirely inside a fenced code block.
3. Implement a `html_chunks` function that splits an HTML string at `<h1>`/`<h2>`/`<h3>` tags, stripping tags from the text with `re.sub`.
4. Write `tests/unit/test_recursive.py` — verify that a paragraph-heavy text with some paragraphs > `chunk_size` produces no chunk exceeding `chunk_size + max_separator_length`.
5. Write `tests/unit/test_markdown.py` — test that each chunk's `metadata["heading"]` matches the nearest preceding heading in the original text.

(Solutions: ../solutions/week-02/day-2.md)

## Self-check

- [ ] I can explain why recursive splitting is superior to fixed-size for structured text.
- [ ] I can trace the recursion: given `separators=["\n\n","\n"," "]` and `chunk_size=50`, I can manually predict the output for a two-paragraph input.
- [ ] I can implement `markdown_chunks` and explain the `heading` metadata propagation.
- [ ] My tests verify both the happy path and oversized section sub-splitting.

## Supplementary links

- [LangChain RecursiveCharacterTextSplitter source](https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/character.py)
- [Unstructured.io — document parsing](https://docs.unstructured.io/)
- [LlamaIndex node parsers](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/)
