# Solutions — Phase 3 · Week 01 · Day 1

## Exercise 1 — Identity matrix

```python
import numpy as np

I = np.eye(3)
v = np.array([1.0, 2.0, 3.0])
result = I @ v
assert np.allclose(result, v), f"Expected {v}, got {result}"
```

## Exercise 2 — `linear_forward`

```python
def linear_forward(
    X: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """
    X: (batch, in_features)
    W: (out_features, in_features)
    b: (out_features,)
    Returns: (batch, out_features)
    """
    assert X.shape[1] == W.shape[1], (
        f"Feature dim mismatch: X has {X.shape[1]}, W has {W.shape[1]}"
    )
    assert W.shape[0] == b.shape[0], (
        f"Output dim mismatch: W {W.shape[0]}, b {b.shape[0]}"
    )
    return X @ W.T + b
```

## Exercise 3 — Shape verification

```python
rng = np.random.default_rng(0)
A = rng.standard_normal((4, 5))
B = rng.standard_normal((5, 2))
C = A @ B
assert C.shape == (4, 2), f"Got {C.shape}"
```

## Exercise 4 — Why A@B ≠ B@A

Matrix multiplication is not commutative because the inner-dimension rule means swapping order changes shapes. For `A (3, 4)` and `B (4, 2)`: `A @ B` is valid (→ `(3, 2)`), but `B @ A` would be `(4, 2) @ (3, 4)` — inner dims `2 ≠ 3`, which raises `ValueError`.

## Exercise 5 — Embedding lookup

Embedding matrix shape: `(10000, 128)` — one 128-dim row per vocabulary token.

```python
vocab_size = 10_000
embed_dim  = 128
rng        = np.random.default_rng(1)

# Initialize embeddings (in practice: learned via backprop)
E: np.ndarray = rng.standard_normal((vocab_size, embed_dim))

# Lookup by integer indices
indices = np.array([42, 7, 999])
embeddings = E[indices]  # shape (3, 128)
assert embeddings.shape == (3, embed_dim)
```
