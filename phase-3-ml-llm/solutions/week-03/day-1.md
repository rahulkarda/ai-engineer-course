# Solutions — Week 03 · Day 1

## Exercise 1 — Implement ReLU and Sigmoid from scratch

```python
import math
import logging
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)

ArrayLike = Union[float, np.ndarray]


def relu(x: ArrayLike) -> ArrayLike:
    """Rectified Linear Unit: max(0, x)."""
    return np.maximum(0.0, x)


def relu_derivative(x: ArrayLike) -> ArrayLike:
    """Derivative of ReLU: 1 if x > 0 else 0."""
    return (np.array(x) > 0).astype(float)


def sigmoid(x: ArrayLike) -> ArrayLike:
    """Sigmoid: 1 / (1 + exp(-x)).

    Numerically stable: use -|x| to avoid overflow.
    """
    x = np.array(x, dtype=float)
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def sigmoid_derivative(x: ArrayLike) -> ArrayLike:
    """Derivative of sigmoid: s(x) * (1 - s(x))."""
    s = sigmoid(x)
    return s * (1.0 - s)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_vals = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    logger.info("x          : %s", test_vals)
    logger.info("relu(x)    : %s", relu(test_vals))
    logger.info("sigmoid(x) : %s", np.round(sigmoid(test_vals), 4))
```

Key points:
- ReLU clips negatives to 0 — gradient exactly 0 for x < 0 (dead neuron risk), exactly 1 for x > 0.
- Numerically stable sigmoid avoids `exp(-large_positive)` overflow by splitting on the sign of x.
- Both accept scalars or NumPy arrays (ArrayLike).

---

## Exercise 2 — Forward pass through a 2-layer MLP (manual)

Network: input x ∈ R^2, hidden layer 3 units (ReLU), output 1 unit (sigmoid).

```
x  = [0.5, -0.2]

W1 = [[0.1, -0.2],   shape (3, 2)
      [0.4,  0.3],
      [-0.1, 0.5]]
b1 = [0.0, 0.1, -0.1]

W2 = [[0.6, -0.3, 0.8]]   shape (1, 3)
b2 = [0.0]
```

Step-by-step:

```python
import numpy as np

x  = np.array([0.5, -0.2])

W1 = np.array([[0.1, -0.2],
               [0.4,  0.3],
               [-0.1, 0.5]])
b1 = np.array([0.0, 0.1, -0.1])

W2 = np.array([[0.6, -0.3, 0.8]])
b2 = np.array([0.0])

# Layer 1
z1 = W1 @ x + b1
# z1 = [0.1*0.5 + (-0.2)*(-0.2) + 0.0,
#        0.4*0.5 +  0.3*(-0.2) + 0.1,
#       -0.1*0.5 +  0.5*(-0.2) - 0.1]
#     = [0.05+0.04, 0.20-0.06+0.1, -0.05-0.10-0.1]
#     = [0.09, 0.24, -0.25]
a1 = np.maximum(0.0, z1)
# a1 = [0.09, 0.24, 0.0]   (ReLU zeroes -0.25)

# Layer 2
z2 = W2 @ a1 + b2
# z2 = 0.6*0.09 + (-0.3)*0.24 + 0.8*0.0
#     = 0.054 - 0.072 = -0.018
a2 = 1.0 / (1.0 + np.exp(-z2))
# a2 ≈ sigmoid(-0.018) ≈ 0.4955

print(f"z1={np.round(z1,4)}, a1={np.round(a1,4)}")
print(f"z2={np.round(z2,4)}, a2={np.round(a2,4)}")
```

Expected output:
```
z1=[ 0.09  0.24 -0.25], a1=[0.09 0.24 0.  ]
z2=[-0.018], a2=[0.4955]
```

---

## Exercise 3 — Vanishing gradient with sigmoid

Explanation:

The sigmoid derivative is s(x)(1 - s(x)), which peaks at 0.25 when x = 0 and
approaches 0 for large |x|.

In a network with L layers, the gradient of the loss w.r.t. W1 contains the
product of L sigmoid derivatives:

    dL/dW1 ∝ s'(z_L) * s'(z_{L-1}) * ... * s'(z_1)

Each factor ≤ 0.25. For L = 10 layers: 0.25^10 ≈ 9.5e-7. The gradient
shrinks exponentially — weights in early layers barely update.

ReLU fixes this: its derivative is 1 for x > 0, so multiplying L ReLU
derivatives gives 1 (no shrinkage) as long as units remain active. This is
the primary reason deep networks switched from sigmoid to ReLU hidden units.

Numerical demonstration:

```python
import numpy as np

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv(x: float) -> float:
    s = sigmoid(x)
    return s * (1.0 - s)

depths = [1, 5, 10, 20]
x_vals = [0.0, 2.0, 5.0]   # 0 = peak, 2 = mild saturation, 5 = deep saturation

for x in x_vals:
    for L in depths:
        grad = sigmoid_deriv(x) ** L
        print(f"x={x:4.1f}, L={L:2d}: gradient = {grad:.2e}")
```

---

## Exercise 4 — Implement Softmax

```python
import numpy as np
from typing import Union

ArrayLike = Union[float, np.ndarray]


def softmax(x: ArrayLike, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax.

    Subtract max before exp to prevent overflow.
    Works on batches: shape (..., C).
    """
    x = np.array(x, dtype=float)
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


# Verify properties
logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print(f"probs       : {np.round(probs, 4)}")   # [0.6590, 0.2424, 0.0986]
print(f"sums to 1   : {np.isclose(probs.sum(), 1.0)}")   # True
print(f"all >= 0    : {(probs >= 0).all()}")              # True

# Batch version
batch = np.array([[2.0, 1.0, 0.1],
                  [0.5, 0.5, 0.5]])
print(f"batch probs:\n{np.round(softmax(batch), 4)}")
```

Note: numerical stability — subtracting max(x) does not change the output
(it cancels in numerator and denominator) but prevents `exp(large)` overflow.

---

## Exercise 5 — Compare activation outputs numerically

```python
import logging
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])

logger.info("%-12s %8s %8s %8s", "x", "ReLU", "Sigmoid", "Tanh")
logger.info("-" * 42)
for xi, r, s, t in zip(x, relu(x), sigmoid(x), tanh(x)):
    logger.info("%-12.1f %8.4f %8.4f %8.4f", xi, r, s, t)

logger.info("")
logger.info("Softmax (all 5 inputs as one vector): %s", np.round(softmax(x), 4))
```

Key observations:
- ReLU: unbounded above, hard zero below — sparse activations.
- Sigmoid: squashes to (0, 1) — useful for binary output probability.
- Tanh: squashes to (-1, 1) — zero-centered, better gradient flow than sigmoid
  but still saturates.
- Softmax: turns a vector of reals into a probability distribution — used only
  at the output layer for multi-class classification.
```
