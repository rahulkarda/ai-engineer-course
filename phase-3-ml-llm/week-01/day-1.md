# Phase 3 · Week 01 · Day 1 — Vectors, Matrices & Matrix Multiplication

> Objective: Build fluent NumPy intuition for vectors and matrices — the language every ML model speaks.  ·  Time: ~1 hr  ·  Prereqs: Python basics, NumPy installed (`uv add numpy`)

---

## Concept

### Vectors

A **vector** is an ordered list of numbers. In ML it represents one data point, one word embedding, one weight row — anything with multiple numeric features.

```
x = [1.5, 0.0, -2.3]   # 3-dimensional feature vector
```

Two key operations:

| Operation | Formula | ML use |
|---|---|---|
| Dot product | `a · b = Σ aᵢbᵢ` | Similarity, linear layer |
| Scaling | `αx` | Learning rate step |

### Matrices

A **matrix** is a 2-D array of numbers, shape `(rows, cols)`.  
In ML: weight matrix `W` of shape `(n_out, n_in)`, batch of examples `X` of shape `(batch, features)`.

### Matrix Multiplication

`C = A @ B` where `A` is `(m, n)` and `B` is `(n, p)` → `C` is `(m, p)`.

Each entry: `Cᵢⱼ = Σ_k Aᵢk * Bkⱼ` — a dot product between row `i` of `A` and column `j` of `B`.

**Rule of thumb:** inner dimensions must match. `(3, 4) @ (4, 2) → (3, 2)`. ✓

---

## How It Works Internally

NumPy stores arrays as contiguous C blocks in memory. `@` dispatches to BLAS (Basic Linear Algebra Subprograms) — hand-tuned Fortran/C routines that exploit CPU SIMD instructions. This is why NumPy matmul is 100–1000× faster than a Python loop.

Shape inference: NumPy tracks `ndarray.shape` as a tuple of ints. Broadcasting rules extend shapes automatically along size-1 dimensions — critical for adding a bias vector to a batch of outputs.

---

## Worked Example

```python
"""
day1_vectors_matrices.py
Vectors, matrices, and matmul in NumPy.
"""
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def demo_vectors() -> None:
    a: np.ndarray = np.array([1.0, 2.0, 3.0])
    b: np.ndarray = np.array([4.0, 5.0, 6.0])

    dot: float = float(np.dot(a, b))         # 1*4 + 2*5 + 3*6 = 32
    scaled: np.ndarray = 0.5 * a             # [0.5, 1.0, 1.5]
    added: np.ndarray = a + b                # element-wise: [5, 7, 9]

    log.info("dot(a, b) = %s", dot)
    log.info("0.5 * a  = %s", scaled)
    log.info("a + b    = %s", added)


def demo_matrices() -> None:
    # Weight matrix W: 4 outputs, 3 inputs
    W: np.ndarray = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [1.0, 1.1, 1.2],
    ])  # shape (4, 3)

    x: np.ndarray = np.array([1.0, 0.0, -1.0])  # shape (3,)

    # Forward pass of one linear layer: y = W @ x
    y: np.ndarray = W @ x  # shape (4,)
    log.info("W shape : %s", W.shape)
    log.info("x shape : %s", x.shape)
    log.info("y = W@x : %s  shape %s", y, y.shape)


def demo_batched_matmul() -> None:
    # Simulate a linear layer: batch of 5 examples, 3 features each → 4 outputs
    # Convention: X is (batch, in_features), W.T is (in_features, out_features)
    X: np.ndarray = np.random.default_rng(42).standard_normal((5, 3))
    W: np.ndarray = np.random.default_rng(0).standard_normal((4, 3))
    b: np.ndarray = np.zeros(4)  # bias

    # Y = X @ W.T + b   → shape (5, 4)
    Y: np.ndarray = X @ W.T + b
    log.info("X shape : %s", X.shape)
    log.info("W.T shape: %s", W.T.shape)
    log.info("Y shape : %s", Y.shape)  # (5, 4)


if __name__ == "__main__":
    demo_vectors()
    demo_matrices()
    demo_batched_matmul()
```

Run with: `python day1_vectors_matrices.py`

### Key shape patterns to memorize

| Expression | Input shapes | Output shape | ML meaning |
|---|---|---|---|
| `np.dot(a, b)` | `(n,), (n,)` | scalar | similarity |
| `W @ x` | `(m, n), (n,)` | `(m,)` | single-sample linear layer |
| `X @ W.T` | `(B, n), (m, n)` | `(B, m)` | batched linear layer |
| `A @ B` | `(m, n), (n, p)` | `(m, p)` | layer stacking |

---

## Exercises

1. Create a `(3, 3)` identity matrix using `np.eye`. Verify that `I @ v == v` for any vector `v`.
2. Implement a function `linear_forward(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray` that computes `X @ W.T + b` with shape assertions.
3. Create two random matrices `A (4, 5)` and `B (5, 2)`. Compute `C = A @ B` and verify `C.shape == (4, 2)`.
4. Explain in one sentence why `A @ B != B @ A` in general. Give a concrete shape example where `B @ A` would raise an error.
5. A neural network embedding layer maps 10,000 vocabulary tokens to 128-dim vectors. What is the shape of the embedding matrix? Write NumPy code to look up embeddings for token indices `[42, 7, 999]`.

(Solutions: ../solutions/week-01/day-1.md)

---

## Self-check

- [ ] I can explain what the dot product computes geometrically (angle + magnitude)
- [ ] I can state the shape rule for matrix multiplication from memory
- [ ] I can write a batched linear layer forward pass in NumPy without reference
- [ ] I understand why `X @ W.T` is used instead of `X @ W` for a batch of inputs

---

## Supplementary

- [NumPy linear algebra docs](https://numpy.org/doc/stable/reference/routines.linalg.html)
- [3Blue1Brown — Essence of Linear Algebra (YouTube)](https://www.3blue1brown.com/topics/linear-algebra) — visualizations
- [The Matrix Calculus You Need For Deep Learning (Parr & Howard)](https://explained.ai/matrix-calculus/)
