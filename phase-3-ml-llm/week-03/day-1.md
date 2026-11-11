# Phase 3 · Week 03 · Day 1 — Perceptrons, MLP Architecture, and Activation Functions

> Objective: Build intuition for neural network structure by computing a forward pass through a 2-layer MLP using only Python and NumPy.  ·  Time: ~1 hr  ·  Prereqs: [[phase-3-ml-llm/week-02/day-4]]

---

## Concept

A **neural network** is a function approximator built from layers of simple computational units called **neurons**. Each neuron computes a weighted sum of its inputs and passes the result through a nonlinear **activation function**. Stack enough of these in sequence and the network can approximate any continuous function (Universal Approximation Theorem).

### The Perceptron

The original perceptron (Rosenblatt, 1958) is the atom of neural computation:

```
output = activation(w · x + b)
```

- `x` — input vector, shape `(n,)`
- `w` — weight vector, shape `(n,)` — learned during training
- `b` — bias scalar — shifts the activation threshold
- `activation` — a nonlinear function applied element-wise

A single perceptron draws a linear decision boundary. Multiple perceptrons in parallel form a **layer**. Stacking layers creates a **Multi-Layer Perceptron (MLP)**.

### MLP Architecture

```
Input layer  →  Hidden layer 1  →  Hidden layer 2  →  Output layer
   (x)             (h1)               (h2)               (y_hat)
```

Each layer `l` has:
- Weight matrix `W[l]` — shape `(n_out, n_in)`
- Bias vector `b[l]` — shape `(n_out,)`
- Pre-activation (logit): `z[l] = W[l] @ a[l-1] + b[l]`
- Post-activation: `a[l] = activation(z[l])`

The input `x` is `a[0]`. The final layer's output is the network's prediction.

### Activation Functions

Activations introduce **nonlinearity** — without them, stacked linear transformations collapse to a single linear map regardless of depth.

| Function | Formula | Use case |
|---|---|---|
| Sigmoid | `1 / (1 + exp(-z))` | Binary output, gates |
| Tanh | `(exp(z) - exp(-z)) / (exp(z) + exp(-z))` | Hidden layers (zero-centered) |
| ReLU | `max(0, z)` | Default hidden activation — fast, sparse |
| GELU | `z * Phi(z)` approx `0.5 * z * (1 + tanh(sqrt(2/pi) * (z + 0.044715 * z^3)))` | Transformers, BERT, GPT |
| Softmax | `exp(z_i) / sum(exp(z_j))` | Multiclass output — converts logits to probabilities |

**Why ReLU is dominant**: it does not saturate for positive inputs (no vanishing gradient), is cheap to compute, and induces sparsity. Sigmoid and tanh saturate at both ends — gradients near zero slow learning.

**GELU** is a smooth approximation to ReLU that allows small negative values through probabilistically, which helps with transformer training.

**Softmax** is not used in hidden layers — it normalizes across all units, destroying local feature representations.

---

## How it works internally

### Memory layout

For a batch of `B` samples through a layer `(n_in → n_out)`:
- `x` has shape `(B, n_in)`
- `W` has shape `(n_out, n_in)`
- `z = x @ W.T + b` has shape `(B, n_out)` — broadcasting adds `b` across the batch

NumPy stores arrays row-major (C order). Matrix multiplication `x @ W.T` is a sequence of dot products between each row of `x` and each row of `W`. For large matrices this maps to optimized BLAS routines (OpenBLAS/MKL). For tiny toy networks NumPy's overhead dominates — irrelevant for learning purposes.

### Why biases matter

Without biases, every hyperplane the network learns passes through the origin. Biases give the activation function freedom to shift horizontally, letting neurons fire (activate) even when all inputs are zero. Practically: a layer without bias has strictly less expressive power.

### Numerical stability of softmax

Naively computing `exp(z_i) / sum(exp(z_j))` overflows for large `z`. The standard trick subtracts the max:

```
softmax(z) = exp(z - max(z)) / sum(exp(z - max(z)))
```

This does not change the result mathematically (factors cancel) but keeps values in `[0, 1]` range.

---

## Worked example

```python
"""
day_1_mlp.py — Manual 2-layer MLP forward pass with NumPy.
No ML framework. Shows weights, biases, activations step by step.
"""

import logging
from typing import Callable

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)


# --- Activation functions ---

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def tanh_act(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def gelu(z: np.ndarray) -> np.ndarray:
    """GELU approximation used by most transformers."""
    return 0.5 * z * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * z**3)))


def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over last axis."""
    shifted = z - z.max(axis=-1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / exp_z.sum(axis=-1, keepdims=True)


# --- MLP building blocks ---

def linear(
    x: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Compute z = x @ W.T + b.

    Args:
        x: Input, shape (batch, n_in).
        W: Weight matrix, shape (n_out, n_in).
        b: Bias vector, shape (n_out,).

    Returns:
        Pre-activation logits, shape (batch, n_out).
    """
    return x @ W.T + b


def forward_mlp(
    x: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    hidden_act: Callable[[np.ndarray], np.ndarray] = relu,
    output_act: Callable[[np.ndarray], np.ndarray] = softmax,
) -> dict[str, np.ndarray]:
    """Full forward pass through a 2-layer MLP.

    Returns intermediate values for inspection.
    """
    # Layer 1: hidden
    z1 = linear(x, W1, b1)           # (batch, n_hidden)
    a1 = hidden_act(z1)               # (batch, n_hidden)

    # Layer 2: output
    z2 = linear(a1, W2, b2)          # (batch, n_out)
    a2 = output_act(z2)              # (batch, n_out)

    return {"z1": z1, "a1": a1, "z2": z2, "a2": a2}


# --- Main ---

def main() -> None:
    rng = np.random.default_rng(seed=42)

    # Toy problem: classify 2D points into 3 classes
    n_in: int = 2
    n_hidden: int = 4
    n_out: int = 3
    batch: int = 3

    # Random inputs — shape (batch, n_in)
    x = rng.standard_normal((batch, n_in)).astype(np.float64)
    log.info("Input x:\n%s", x)

    # He-style initialization for ReLU hidden layer
    W1 = rng.standard_normal((n_hidden, n_in)) * np.sqrt(2.0 / n_in)
    b1 = np.zeros(n_hidden)

    # Xavier-style initialization for output layer
    W2 = rng.standard_normal((n_out, n_hidden)) * np.sqrt(1.0 / n_hidden)
    b2 = np.zeros(n_out)

    log.info("W1 shape: %s, b1 shape: %s", W1.shape, b1.shape)
    log.info("W2 shape: %s, b2 shape: %s", W2.shape, b2.shape)

    cache = forward_mlp(x, W1, b1, W2, b2, hidden_act=relu, output_act=softmax)

    log.info("Pre-activation hidden z1:\n%s", cache["z1"])
    log.info("Post-activation hidden a1 (ReLU):\n%s", cache["a1"])
    log.info("Pre-activation output z2:\n%s", cache["z2"])
    log.info("Predicted probabilities a2 (softmax):\n%s", cache["a2"])

    # Verify softmax sums to 1 per sample
    row_sums = cache["a2"].sum(axis=1)
    log.info("Softmax row sums (should be ~1.0): %s", row_sums)

    # Manual trace for first sample, first hidden neuron
    log.info("--- Manual trace: sample 0, hidden neuron 0 ---")
    z1_00 = W1[0] @ x[0] + b1[0]
    a1_00 = max(0.0, z1_00)
    log.info("z1[0,0] = %.6f, a1[0,0] = %.6f", z1_00, a1_00)
    log.info("Matches cache z1[0,0] = %.6f", cache["z1"][0, 0])


if __name__ == "__main__":
    main()
```

Run with: `python day_1_mlp.py`

Expected output (values vary by seed but structure is fixed):
- `z1`: negative and positive values (unbounded)
- `a1`: zeros where `z1 < 0` (ReLU sparsity visible)
- `a2`: each row sums to 1.0 (valid probability distribution)

---

## Exercises

1. **Conceptual**: Explain why a 3-layer MLP with linear activations is equivalent to a 1-layer MLP. What does this imply about the role of nonlinearity?

2. **Coding**: Modify `forward_mlp` to accept a list of `(W, b, activation)` tuples so it supports arbitrary depth. Verify the output matches the original for `depth=2`.

3. **Activation comparison**: Run all five activation functions (`sigmoid`, `tanh_act`, `relu`, `gelu`, `softmax`) on the same input vector `z = np.linspace(-3, 3, 13)` and log the outputs. Observe where each saturates.

4. **Numerical stability**: Remove the `max(axis=-1)` subtraction from `softmax`. Construct an input `z` large enough to cause `nan` or `inf`. Confirm the stable version handles it correctly.

5. **Initialization**: Change `W1` to all-zeros and `W2` to all-zeros. Run the forward pass and observe the output. Explain why all neurons produce identical activations and why this is called the **symmetry problem**.

(Solutions: ../solutions/week-03/day-1.md)

---

## Self-check

- [ ] I can explain what a weight matrix and bias vector represent geometrically.
- [ ] I can state the shape of `z = W @ x + b` given layer dimensions without looking it up.
- [ ] I can implement `relu`, `sigmoid`, and numerically stable `softmax` from memory in NumPy.
- [ ] I can trace a single input through a 2-layer MLP by hand (small integers) and match the code output.
- [ ] I can explain why activation functions must be nonlinear.

---

## Supplementary Links

- Rosenblatt (1958) perceptron paper overview: https://en.wikipedia.org/wiki/Perceptron
- Universal Approximation Theorem (plain language): https://towardsdatascience.com/the-universal-approximation-theorem-explained-c0649dff92c8
- Activation function zoo with visual plots: https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
- GELU paper (Hendrycks & Gimpel, 2016): https://arxiv.org/abs/1606.08415
- NumPy broadcasting rules: https://numpy.org/doc/stable/user/basics.broadcasting.html
