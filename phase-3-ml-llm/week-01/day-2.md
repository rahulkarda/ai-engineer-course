# Phase 3 · Week 01 · Day 2 — Norms, Projections & Eigenvalue Intuition

> Objective: Understand vector norms, cosine similarity, and the intuition behind eigenvalues — all directly used in ML regularization, embeddings, and PCA.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

---

## Concept

### Vector Norms

A **norm** measures the "size" of a vector. Two norms matter in ML:

| Norm | Formula | NumPy | ML use |
|---|---|---|---|
| L1 (Manhattan) | `Σ|xᵢ|` | `np.linalg.norm(x, 1)` | Sparse solutions, Lasso |
| L2 (Euclidean) | `√(Σxᵢ²)` | `np.linalg.norm(x)` | Ridge, weight decay, distance |

**L2 norm squared** = `x · x`. Appears in Ridge regression loss: `loss = MSE + λ‖w‖₂²`

### Cosine Similarity

Dot product ÷ product of norms:

```
cos(θ) = (a · b) / (‖a‖ · ‖b‖)
```

Range: `[-1, 1]`. Value 1 → same direction. 0 → orthogonal. -1 → opposite.  
Used everywhere word/sentence embeddings are compared.

### Projections

Scalar projection of `b` onto `a`: `proj = (a · b) / ‖a‖`  
Vector projection: `proj_vec = ((a · b) / (a · a)) * a`

**ML relevance:** attention mechanisms compute a weighted sum of projections. PCA finds the directions that maximize projected variance.

### Eigenvalues — Intuition Only

For a square matrix `A`, an **eigenvector** `v` satisfies:  
`A @ v = λ * v`

`A` just scales `v` by scalar `λ` (the **eigenvalue**) — it doesn't rotate it.

**Why it matters:**
- Covariance matrix eigenvectors = principal components (PCA)
- Large eigenvalue = high-variance direction in data
- `λ < 1` in weight matrices → gradients shrink (vanishing gradient signal)
- `λ > 1` → gradients grow (exploding gradients)

You don't need to compute eigenvalues by hand. You need to know what they mean.

---

## How It Works Internally

`np.linalg.norm` computes L2 via a single BLAS `dnrm2` call — numerically stable (avoids overflow for large values by scaling).

`np.linalg.eig` uses the LAPACK `dgeev` routine (QR algorithm). For symmetric matrices (covariance), `np.linalg.eigh` is faster and guarantees real eigenvalues.

Cosine similarity in embedding search: libraries like FAISS normalize vectors first, then use inner product — equivalent to cosine similarity but faster because the norm divide happens once offline.

---

## Worked Example

```python
"""
day2_norms_projections_eigen.py
Norms, cosine similarity, projections, eigenvalue intuition.
"""
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def demo_norms() -> None:
    x: np.ndarray = np.array([3.0, -4.0, 0.0])

    l1: float = float(np.linalg.norm(x, 1))   # 3 + 4 + 0 = 7
    l2: float = float(np.linalg.norm(x))       # sqrt(9+16) = 5
    linf: float = float(np.linalg.norm(x, np.inf))  # max |xᵢ| = 4

    log.info("L1 norm : %.4f", l1)
    log.info("L2 norm : %.4f", l2)
    log.info("Linf norm: %.4f", linf)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def demo_cosine() -> None:
    king: np.ndarray  = np.array([1.0, 0.9, 0.1])  # toy word vectors
    queen: np.ndarray = np.array([0.9, 1.0, 0.2])
    car: np.ndarray   = np.array([0.0, 0.1, 1.0])

    log.info("cos(king, queen) = %.4f", cosine_similarity(king, queen))
    log.info("cos(king, car)   = %.4f", cosine_similarity(king, car))


def vector_projection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Project b onto a."""
    return (np.dot(a, b) / np.dot(a, a)) * a


def demo_projection() -> None:
    a: np.ndarray = np.array([1.0, 0.0])  # x-axis
    b: np.ndarray = np.array([3.0, 4.0])

    proj: np.ndarray = vector_projection(a, b)
    log.info("Projection of b onto a: %s", proj)  # [3. 0.] — x-component only


def demo_eigenvalues() -> None:
    # Build a simple 2x2 covariance-like matrix
    rng = np.random.default_rng(42)
    data: np.ndarray = rng.standard_normal((100, 2))
    data[:, 1] *= 3.0  # stretch second axis — high variance there

    cov: np.ndarray = np.cov(data.T)  # (2, 2)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)  # eigh for symmetric

    log.info("Covariance matrix:\n%s", cov)
    log.info("Eigenvalues: %s", eigenvalues)
    log.info("Eigenvectors:\n%s", eigenvectors)
    log.info("Largest eigenvalue direction (PC1): %s", eigenvectors[:, -1])


if __name__ == "__main__":
    demo_norms()
    demo_cosine()
    demo_projection()
    demo_eigenvalues()
```

---

## Exercises

1. Write a function `l2_normalize(v: np.ndarray) -> np.ndarray` that returns the unit vector. Assert the result has L2 norm ≈ 1.0 (within 1e-6).
2. Given embeddings for five words (random 8-dim vectors), find the pair with highest cosine similarity. Return both words and the score.
3. Compute the projection of `[2, 3, 1]` onto `[1, 0, 0]`, `[0, 1, 0]`, and `[0, 0, 1]`. What do you notice?
4. Create a (3, 3) diagonal matrix `D = diag([2, 5, 0.1])`. Compute its eigenvalues analytically (what do you expect?), then verify with `np.linalg.eig`.
5. Regularization question: if a weight vector `w = [10, -10, 0.01]`, compute both L1 and L2 penalties. Which norm penalizes the near-zero weight more heavily, relative to the large weights? What does this tell you about which penalty encourages sparsity?

(Solutions: ../solutions/week-01/day-2.md)

---

## Self-check

- [ ] I can compute L1 and L2 norms by hand for a small vector
- [ ] I can implement cosine similarity without looking it up
- [ ] I can explain what an eigenvector is in plain language (no math jargon)
- [ ] I understand why high eigenvalues in weight matrices relate to exploding gradients

---

## Supplementary

- [NumPy `linalg` reference](https://numpy.org/doc/stable/reference/routines.linalg.html)
- [Cosine similarity in embeddings — Pinecone blog](https://www.pinecone.io/learn/vector-similarity/)
- [Setosa.io PCA visualizer](https://setosa.io/ev/principal-component-analysis/) — eigenvalue intuition, interactive
