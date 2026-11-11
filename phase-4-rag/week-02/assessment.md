# Week 02 Assessment — Chunking & Embeddings

## Conceptual (interview-style)

Answer each question without code. Be precise.

1. Explain the precision-recall tradeoff of chunk size. What happens to retrieval quality as you increase chunk size from 100 chars to 5,000 chars?

2. Describe the recursive splitting algorithm. Given separators `["\n\n", "\n", " ", ""]` and `chunk_size=100`, trace through the processing of a text that has one 150-character paragraph with no blank lines.

3. What is mean pooling in the context of transformer embeddings? Why is it preferred over using only the `[CLS]` token for sentence-level embeddings?

4. You normalise an embedding vector before storing it. Why? What property does normalisation enable?

5. A colleague proposes using Euclidean distance for vector retrieval. Under what condition is Euclidean distance equivalent to cosine distance? What goes wrong if the vectors are not unit-normalised?

6. Describe two failure modes of semantic chunking. For each, suggest a mitigation.

7. Your text-embedding-3-large job costs $5. You switch to text-embedding-3-small. Estimate the new cost (use the prices from Day 5). Show your reasoning.

8. What is `sentence_overlap` in sentence chunking? Give a concrete example showing how it prevents a cross-boundary sentence from being inaccessible.

---

## Coding challenge

**Specification:** Implement a `ChunkerFactory` class that constructs any of the five chunker types from a configuration dictionary. Then use it to run a strategy comparison.

**Acceptance criteria:**

```python
from rag_pipeline.chunking.factory import ChunkerFactory
from rag_pipeline.chunking.base import ChunkerProtocol

factory = ChunkerFactory(provider=mock_provider)

chunker = factory.create({
    "strategy": "fixed",
    "chunk_size": 300,
    "overlap": 30,
})
assert isinstance(chunker, ChunkerProtocol)

chunker2 = factory.create({
    "strategy": "sentence",
    "max_chars": 400,
    "sentence_overlap": 1,
})
assert isinstance(chunker2, ChunkerProtocol)

chunker3 = factory.create({"strategy": "semantic", "breakpoint_percentile": 25.0})
assert isinstance(chunker3, ChunkerProtocol)

# Unknown strategy raises ValueError
import pytest
with pytest.raises(ValueError, match="Unknown strategy"):
    factory.create({"strategy": "magic"})
```

The factory must support strategies: `"fixed"`, `"sentence"`, `"recursive"`, `"markdown"`, `"semantic"`.

Additional requirements:

- `ChunkerFactory` takes an `EmbeddingProvider` at construction (needed for `SemanticChunker`).
- All configuration keys not relevant to a strategy are silently ignored (no error for extra keys).
- Write at least 5 unit tests in `tests/unit/test_factory.py`.

---

## Rubric (pass ≥ 75%)

| Dimension | Weight | What to look for |
|---|---|---|
| Independent implementation | 40% | `ChunkerFactory` correctly constructs all 5 strategies; `SemanticChunker` receives provider; `ValueError` on unknown strategy |
| Conceptual understanding | 25% | Q1–Q8 show clear mental models; Q4 mentions dot-product efficiency; Q7 calculation is correct |
| Debugging & problem-solving | 20% | Tests cover the error path; candidate can explain why a strategy-comparison run shows different `adj_sim` values |
| Code quality & testing | 15% | Type hints on all methods; `logging` used; ≥ 5 unit tests; no real API calls in tests |

**Pass gate:** ≥ 75 points total, AND the coding challenge tests must pass.

Answer key: ../solutions/week-02/assessment.md
