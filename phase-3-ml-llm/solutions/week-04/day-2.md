# Solutions — Week 04 · Day 2

## Exercise 1 — Extend EMBEDDINGS with five new words

```python
EMBEDDINGS["prince"]   = np.array([0.85, 0.15, 0.0, 0.8])
EMBEDDINGS["princess"] = np.array([0.85, 0.85, 0.0, 0.8])
EMBEDDINGS["boy"]      = np.array([0.05, 0.1,  0.0, 0.7])
EMBEDDINGS["girl"]     = np.array([0.05, 0.85, 0.0, 0.7])
EMBEDDINGS["horse"]    = np.array([0.0,  0.0,  0.0, 0.25])

nn = nearest_neighbours("prince", EMBEDDINGS)
# Expected: [(king, ~0.98), (queen, ~0.96), (princess, ~0.95)] — royalty-related
```

---

## Exercise 2 — Euclidean distance vs cosine similarity disagreement

```python
def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))
```

**Concrete disagreement example:**

Scale `EMBEDDINGS["king"]` by 10:

```python
big_king = EMBEDDINGS["king"] * 10
small_king = EMBEDDINGS["king"]
cosine = cosine_similarity(big_king, small_king)   # ≈ 1.0 (same direction)
euclidean = euclidean_distance(big_king, small_king)  # ≈ 9× magnitude — large distance
```

Cosine sees them as identical (same direction). Euclidean sees them as far apart (different magnitude). This is why cosine is preferred for comparing text documents where term frequency differs by document length.

---

## Exercise 3 — PPMI from co-occurrence matrix

```python
def compute_ppmi(cooc: np.ndarray) -> np.ndarray:
    """Positive Pointwise Mutual Information from a co-occurrence matrix."""
    total = cooc.sum()
    if total == 0:
        return np.zeros_like(cooc)
    row_sums = cooc.sum(axis=1, keepdims=True)  # P(word_i)
    col_sums = cooc.sum(axis=0, keepdims=True)  # P(word_j)
    # P(i, j) / (P(i) * P(j))
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(cooc * total / (row_sums @ col_sums + 1e-10) + 1e-10)
    ppmi = np.maximum(pmi, 0)
    return ppmi

cooc, vocab = build_cooccurrence(SENTENCES)
ppmi = compute_ppmi(cooc)
print("PPMI matrix shape:", ppmi.shape)
print("Non-zero entries:", (ppmi > 0).sum())
```

---

## Exercise 4 — 3CosMul variant

```python
def analogy_3cosmul(
    a: str, b: str, c: str, vocab: dict[str, np.ndarray]
) -> str:
    """3CosMul: argmax cos(d,b) * cos(d,c) / (cos(d,a) + epsilon)."""
    exclude = {a, b, c}
    best_word, best_score = "", -2.0
    for word, vec in vocab.items():
        if word in exclude:
            continue
        score = (
            (cosine_similarity(vocab[b], vec) + 1) *
            (cosine_similarity(vocab[c], vec) + 1) /
            (cosine_similarity(vocab[a], vec) + 1 + 1e-6)
        )
        if score > best_score:
            best_score, best_word = score, word
    return best_word

result = analogy_3cosmul("king", "man", "woman", EMBEDDINGS)
# On toy embeddings the result should still be "queen"
```

With the 4-D toy vocabulary the two approaches usually agree. On real word vectors, 3CosMul reduces false positives from high-frequency words.

---

## Exercise 5 — pytest test suite

```python
import numpy as np
import pytest
from day2_embeddings_cosine import cosine_similarity, cosine_similarity_matrix

def test_identical_vectors_score_one() -> None:
    v = np.array([1.0, 2.0, 3.0])
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

def test_orthogonal_vectors_score_zero() -> None:
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert abs(cosine_similarity(a, b)) < 1e-6

def test_opposite_vectors_score_minus_one() -> None:
    v = np.array([1.0, 0.5, -0.3])
    assert abs(cosine_similarity(v, -v) + 1.0) < 1e-6

def test_zero_vector_returns_zero() -> None:
    v = np.array([1.0, 2.0, 3.0])
    z = np.zeros(3)
    assert cosine_similarity(v, z) == 0.0

def test_cosine_similarity_matrix_symmetric() -> None:
    mat = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    sim = cosine_similarity_matrix(mat)
    assert np.allclose(sim, sim.T, atol=1e-6)

def test_cosine_similarity_matrix_diagonal_ones() -> None:
    mat = np.random.default_rng(0).standard_normal((5, 4)).astype(np.float32)
    sim = cosine_similarity_matrix(mat)
    assert np.allclose(np.diag(sim), 1.0, atol=1e-5)

@pytest.mark.parametrize("dim", [2, 10, 128])
def test_cosine_range(dim: int) -> None:
    rng = np.random.default_rng(dim)
    a = rng.standard_normal(dim)
    b = rng.standard_normal(dim)
    score = cosine_similarity(a, b)
    assert -1.01 <= score <= 1.01
```
