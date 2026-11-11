# Solutions — Phase 3 · Week 01 · Day 2

## Exercise 1 — L2 normalize

```python
import numpy as np

def l2_normalize(v: np.ndarray) -> np.ndarray:
    """Return the unit vector in the direction of v."""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        raise ValueError("Cannot normalize near-zero vector")
    return v / norm

v = np.array([3.0, 4.0])
u = l2_normalize(v)
assert abs(np.linalg.norm(u) - 1.0) < 1e-6, f"norm={np.linalg.norm(u)}"
```

## Exercise 2 — Highest cosine similarity pair

```python
import numpy as np

rng   = np.random.default_rng(42)
words = [f"word_{i}" for i in range(5)]
vecs  = rng.standard_normal((5, 8))

# L2-normalize all vectors
norms = np.linalg.norm(vecs, axis=1, keepdims=True)
normed = vecs / norms  # (5, 8)

# Cosine similarity matrix = normed @ normed.T
sim_matrix = normed @ normed.T  # (5, 5)
np.fill_diagonal(sim_matrix, -1.0)  # exclude self-pairs

i, j = np.unravel_index(sim_matrix.argmax(), sim_matrix.shape)
print(f"Most similar: {words[i]} & {words[j]}, score={sim_matrix[i, j]:.4f}")
```

## Exercise 3 — Projection onto axes

```python
def vector_projection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.dot(a, b) / np.dot(a, a)) * a

b = np.array([2.0, 3.0, 1.0])
axes = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])]
projs = [vector_projection(ax, b) for ax in axes]
# Results: [2, 0, 0], [0, 3, 0], [0, 0, 1]
# Observation: projecting onto the standard basis axes extracts individual components.
# The three projections sum to b — they're an orthogonal decomposition.
```

## Exercise 4 — Diagonal eigenvalues

For a diagonal matrix `D = diag([2, 5, 0.1])`, the eigenvalues ARE the diagonal entries. Each standard basis vector `eᵢ` is an eigenvector with eigenvalue `Dᵢᵢ`.

```python
D = np.diag([2.0, 5.0, 0.1])
eigenvalues, eigenvectors = np.linalg.eig(D)
# eigenvalues should be [2.0, 5.0, 0.1] (order may vary)
assert set(np.round(eigenvalues, 6)) == {2.0, 5.0, 0.1}
```

## Exercise 5 — L1 vs L2 sparsity

```python
w = np.array([10.0, -10.0, 0.01])
l1_penalty = np.linalg.norm(w, 1)   # 10 + 10 + 0.01 = 20.01
l2_penalty = np.linalg.norm(w) ** 2  # 100 + 100 + 0.0001 = 200.0001
```

**Analysis:** The L1 penalty treats the near-zero weight `0.01` proportionally to its magnitude — it only adds `0.01` to the total penalty. To minimize L1, the optimizer will push small weights all the way to zero (sparse solution), while large weights remain. L2 penalizes the square, so large weights dominate; small weights get relatively little pressure. This is why L1 encourages sparsity and L2 encourages small-but-nonzero weights.
