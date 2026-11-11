# Solutions — Week 02 · Day 1

## Exercise answers

### Exercise 1 — Token-based chunk size

```python
# chunking/fixed.py (extended)
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import logging

logger = logging.getLogger(__name__)
CHARS_PER_TOKEN = 4


@dataclass
class Chunk:
    text: str
    index: int
    metadata: dict = field(default_factory=dict)


def fixed_size_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    unit: Literal["chars", "tokens"] = "chars",
) -> list[Chunk]:
    if unit == "tokens":
        chunk_size = chunk_size * CHARS_PER_TOKEN
        overlap = overlap * CHARS_PER_TOKEN
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not (0 <= overlap < chunk_size):
        raise ValueError(f"overlap must be in [0, chunk_size), got {overlap}")

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    start, idx = 0, 0
    while start < len(text):
        chunks.append(Chunk(text=text[start : start + chunk_size], index=idx))
        start += step
        idx += 1
    return chunks
```

### Exercise 2 — Metadata in chunkers

```python
def fixed_size_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    source: str = "",
    page: int | None = None,
) -> list[Chunk]:
    ...
    while start < len(text):
        chunks.append(
            Chunk(
                text=text[start : start + chunk_size],
                index=idx,
                metadata={"source": source, "page": page},
            )
        )
        ...
```

Pass `source="myfile.pdf"` and `page=3` from the caller.

### Exercise 3 — ChunkerProtocol

```python
# chunking/base.py
from typing import Protocol, runtime_checkable
from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    index: int
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class ChunkerProtocol(Protocol):
    def __call__(self, text: str) -> list[Chunk]: ...
```

Verify with:

```python
from chunking.base import ChunkerProtocol
from chunking.fixed import fixed_size_chunks
import functools

wrapped = functools.partial(fixed_size_chunks, chunk_size=200)
assert isinstance(wrapped, ChunkerProtocol)  # True — satisfies structural check
```

### Exercise 4 — Unit tests for fixed chunker

```python
# tests/unit/test_fixed.py
import pytest
from chunking.fixed import fixed_size_chunks


def test_empty_string():
    result = fixed_size_chunks("", chunk_size=100, overlap=10)
    assert result == []


def test_single_chunk():
    text = "hello"
    result = fixed_size_chunks(text, chunk_size=10, overlap=0)
    assert len(result) == 1
    assert result[0].text == "hello"


def test_overlap_boundary():
    text = "a" * 100
    chunks = fixed_size_chunks(text, chunk_size=30, overlap=10)
    # chunks[0] = text[0:30], chunks[1] = text[20:50] — overlap of 10
    assert chunks[1].text == text[20:50]


def test_chunk_size_10_no_overlap():
    text = "0123456789" * 3  # 30 chars
    chunks = fixed_size_chunks(text, chunk_size=10, overlap=0)
    assert len(chunks) == 3
    assert all(len(c.text) == 10 for c in chunks)


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        fixed_size_chunks("hello", chunk_size=10, overlap=10)  # overlap == chunk_size


def test_invalid_negative_overlap_raises():
    with pytest.raises(ValueError):
        fixed_size_chunks("hello", chunk_size=10, overlap=-1)
```

### Exercise 5 — Average chunk size comparison

```python
import statistics

text = "word " * 1000  # 5,000 chars

from chunking.fixed import fixed_size_chunks
from chunking.sentence import sentence_chunks

fixed = fixed_size_chunks(text, chunk_size=200, overlap=20)
sent = sentence_chunks(text, max_chars=200, sentence_overlap=1)

print(f"fixed  mean={statistics.mean(len(c.text) for c in fixed):.0f}  "
      f"stdev={statistics.stdev(len(c.text) for c in fixed):.0f}")
print(f"sent   mean={statistics.mean(len(c.text) for c in sent):.0f}  "
      f"stdev={statistics.stdev(len(c.text) for c in sent):.0f}")
```

**Expected finding:** Fixed produces near-uniform sizes (low stdev). Sentence chunking produces variable sizes (higher stdev) because sentence lengths vary — but each chunk ends at a natural boundary.
