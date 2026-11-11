# Week 04 Assessment — NLP, Embeddings & Tokenization

## Conceptual (interview-style)

Answer each question in 3–6 sentences. Write in your own words.

1. **BoW vs TF-IDF**: Explain the mathematical difference between a bag-of-words vector and a TF-IDF vector for the same document. Why does TF-IDF generally improve classification performance?

2. **BPE algorithm**: Walk through the first three BPE merge steps for the corpus `["aab aac", "aab aab", "bbc bbc bbc"]`. Show the pair counts and the resulting vocabulary after each step.

3. **OOV handling**: A customer sends the word `"tokenizability"` to a production system that was trained with a fixed word-level vocabulary of 30,000 words. Explain what happens at inference time and how BPE resolves this problem.

4. **Contextual vs static embeddings**: Give a concrete sentence where the static word2vec vector for a word would be misleading, and explain how a BERT-style contextual embedding avoids the problem.

5. **Cosine vs Euclidean**: Two documents both discuss machine learning. Document A uses the word "learning" 100 times (it is very long); Document B uses it 10 times (short document). Why does cosine similarity correctly identify them as similar while Euclidean distance on raw count vectors would not?

6. **WordPiece vs BPE**: What is the key algorithmic difference in how WordPiece selects which pair to merge next? In what situation would WordPiece produce a better segmentation than BPE?

7. **Sentence embeddings**: Describe the Siamese network training setup used by Sentence-BERT. What task is it trained on, and why does that task produce better sentence embeddings than simply using BERT's [CLS] token?

8. **API mocking**: A colleague says "we can't write unit tests for the semantic search engine because the embedding API calls are external." Explain how `unittest.mock.patch` resolves this and describe what the mock should return.

---

## Coding challenge

### Challenge A — BPE encoder/decoder (required)

Implement a `BPETokenizer` class from scratch (no external NLP libraries). It must:

1. Accept a corpus (`list[str]`) and `num_merges: int` in a `fit` method.
2. Implement `encode(text: str) -> list[str]` and `decode(tokens: list[str]) -> str`.
3. Pass the following assertions:

```python
corpus = [
    "low lower lowest",
    "newer newest news",
    "old older oldest",
]
tok = BPETokenizer()
tok.fit(corpus, num_merges=15)

# encode produces a non-empty list
assert tok.encode("lower") != []

# decode round-trips
assert tok.decode(tok.encode("low")) == "low"

# unknown words degrade gracefully (no exception)
assert tok.encode("zzz") != []

# multi-word round-trip
assert tok.decode(tok.encode("older low")) == "older low"
```

4. Include a `pytest` test file with at least 6 tests.
5. Use type hints throughout. Use `logging`, not `print`.

### Challenge B — Semantic search (required)

Implement a `SemanticSearchEngine` class that:

1. Accepts an `EmbeddingProvider` at construction time (duck-typed Protocol).
2. Exposes `build_index(documents: list[str])` and `search(query: str, top_k: int) -> list[tuple[str, float]]`.
3. Uses cosine similarity for ranking.
4. Passes the following test (you write the rest):

```python
import numpy as np
from unittest.mock import MagicMock

mock = MagicMock()
# Make embed return unit vectors of dim 4
rng = np.random.default_rng(99)
mock.embed.side_effect = lambda texts: (
    lambda m: m / np.linalg.norm(m, axis=1, keepdims=True)
)(rng.standard_normal((len(texts), 4)).astype(np.float32))

engine = SemanticSearchEngine(mock)
docs = ["cats sleep", "dogs run", "birds fly"]
engine.build_index(docs)
results = engine.search("pets", top_k=2)

assert len(results) == 2
assert all(-1.01 <= score <= 1.01 for _, score in results)
```

5. Include at least 5 `pytest` tests that mock the embedding provider.

---

## Rubric (pass ≥ 75%)

| Dimension | Weight | What is assessed |
|---|---|---|
| Independent implementation | 40% | BPE + search engine written without copying reference code; both pass acceptance tests |
| Conceptual understanding | 25% | Conceptual answers demonstrate genuine understanding, not surface-level recall |
| Debugging & problem-solving | 20% | Edge cases handled (empty input, unknown words, zero-norm vectors, unsaved index); tests cover them |
| Code quality & testing | 15% | Type hints, logging, docstrings, test count ≥ 11 total, no bare `print`, no hardcoded secrets |

**Pass gate:** total score ≥ 75%.  
**Distinction:** ≥ 90% — requires MMR search variant or SentencePiece-style unigram pruning described correctly in comments.

Answer key: ../solutions/week-04/assessment.md
