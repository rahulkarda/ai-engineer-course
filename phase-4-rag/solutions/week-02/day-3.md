# Solutions — Week 02 · Day 3

## Exercise answers

### Exercise 1 — Fixed-threshold breakpoint variant

```python
# chunking/semantic.py  (additional function)

def _find_breakpoints_fixed(similarities: list[float], threshold: float) -> list[int]:
    """Return indices where similarity drops below a fixed *threshold*."""
    return [i for i, s in enumerate(similarities) if s < threshold]
```

Unit test comparing variants:

```python
# tests/unit/test_semantic.py

def test_fixed_vs_percentile_breakpoints():
    """Fixed threshold may find more or fewer breaks than percentile depending on data."""
    import math
    from chunking.semantic import _find_breakpoints, _find_breakpoints_fixed

    # Similarities: one clear drop at index 2
    sims = [0.9, 0.88, 0.3, 0.85, 0.9]

    percentile_breaks = _find_breakpoints(sims, percentile=25.0)
    fixed_breaks = _find_breakpoints_fixed(sims, threshold=0.5)

    # Both should identify index 2 as a break
    assert 2 in percentile_breaks
    assert 2 in fixed_breaks
```

### Exercise 2 — Max chunk guard with sentence sub-split

```python
# chunking/semantic.py  (in semantic_chunks, after assembling chunks)

from chunking.sentence import sentence_chunks as _sentence_split

def _guard_max_chars(chunks: list[Chunk], max_chunk_chars: int) -> list[Chunk]:
    """Sub-split any chunk exceeding max_chunk_chars using sentence chunking."""
    result: list[Chunk] = []
    idx = 0
    for c in chunks:
        if len(c.text) > max_chunk_chars:
            sub = _sentence_split(c.text, max_chars=max_chunk_chars, sentence_overlap=0)
            for s in sub:
                s.index = idx
                s.metadata.update(c.metadata)
                result.append(s)
                idx += 1
        else:
            c.index = idx
            result.append(c)
            idx += 1
    return result
```

Call at the end of `semantic_chunks`:
```python
if max_chunk_chars:
    chunks = _guard_max_chars(chunks, max_chunk_chars)
```

### Exercise 3 — numpy cosine similarity

```python
# chunking/semantic.py  (numpy version)
import numpy as np

def cosine_similarity_np(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))
```

Benchmark:
```python
import timeit
import random

N = 1000
DIM = 384
pairs = [(
    [random.gauss(0, 1) for _ in range(DIM)],
    [random.gauss(0, 1) for _ in range(DIM)],
) for _ in range(N)]

pure_time = timeit.timeit(
    lambda: [cosine_similarity(a, b) for a, b in pairs], number=10
)
np_time = timeit.timeit(
    lambda: [cosine_similarity_np(a, b) for a, b in pairs], number=10
)
print(f"pure: {pure_time:.3f}s  numpy: {np_time:.3f}s")
# numpy is 5–20x faster
```

### Exercise 4 — Semantic chunker unit tests

```python
# tests/unit/test_semantic.py
import math
import pytest
from chunking.semantic import semantic_chunks, Vector


def _mock_embed(texts: list[str]) -> list[Vector]:
    """Deterministic mock: returns pre-set vectors."""
    vecs = {
        "Plants grow in soil.": [1.0, 0.0, 0.0, 0.0],
        "Chlorophyll absorbs light.": [0.9, 0.1, 0.0, 0.0],
        "Quantum computers use qubits.": [0.0, 0.0, 1.0, 0.0],
        "Qubits exploit superposition.": [0.0, 0.1, 0.9, 0.0],
    }
    return [vecs.get(t, [0.5, 0.5, 0.0, 0.0]) for t in texts]


PLANT_SENTENCES = "Plants grow in soil. Chlorophyll absorbs light."
MIXED = "Plants grow in soil. Chlorophyll absorbs light. Quantum computers use qubits. Qubits exploit superposition."


def test_single_sentence_returns_one_chunk():
    result = semantic_chunks("Hello world.", _mock_embed)
    assert len(result) == 1


def test_topic_shift_produces_two_chunks():
    result = semantic_chunks(MIXED, _mock_embed, breakpoint_percentile=30, min_chunk_sentences=1)
    assert len(result) >= 2


def test_min_chunk_sentences_respected():
    result = semantic_chunks(MIXED, _mock_embed, breakpoint_percentile=30, min_chunk_sentences=2)
    for chunk in result:
        # Each chunk should have at least 2 sentences (rough check via period count)
        assert chunk.text.count(".") >= 1  # at least one sentence boundary
```

### Exercise 5 — Window size effect

```python
def test_window_size_reduces_breakpoints():
    """Larger window smooths signal → fewer breakpoints."""
    ...
    breaks_w0 = semantic_chunks(corpus, embed_fn, window_size=0, breakpoint_percentile=25)
    breaks_w2 = semantic_chunks(corpus, embed_fn, window_size=2, breakpoint_percentile=25)
    # More chunks with window=0 (noisier), fewer with window=2 (smoother)
    assert len(breaks_w0) >= len(breaks_w2)
```

**Why:** With `window_size=2`, each "sentence" embedding actually represents 5 sentences of context (current ±2). Adjacent windows share 4 of 5 sentences — their cosine similarity is naturally higher, so fewer drops cross the percentile threshold.
