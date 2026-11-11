# Phase 3 · Week 01 · Assessment

> Week 1 — Math Essentials for ML  ·  Pass gate: 75%  ·  Time: ~1 hr

---

## Conceptual (interview-style) — 25% weight

Answer in 3–5 sentences each. Precision matters.

1. What is the dot product of two vectors, and what geometric property does it measure? Give one concrete ML use case.

2. Explain the difference between L1 and L2 regularization in terms of the norms they penalize. Which encourages sparsity and why?

3. What is a gradient? How does it differ from a derivative? Why does gradient descent step in the *negative* gradient direction?

4. State the chain rule and explain how it enables backpropagation in a 3-layer neural network. Walk through the chain for `loss → sigmoid → linear`.

5. What is cross-entropy loss? Why is it preferred over mean squared error for classification tasks? What does minimizing CE correspond to statistically?

6. Define "eigenvalue" without using linear algebra jargon. Describe one situation in ML where eigenvalues matter.

7. What is Bayes' theorem? Identify the prior, likelihood, and posterior in the context of a spam classifier.

---

## Coding Challenge — 40% weight

### Spec

Implement a **two-layer neural network** for binary classification in NumPy — from scratch, no autograd libraries.

Architecture:
- Layer 1: linear → ReLU, hidden size 8
- Layer 2: linear → sigmoid, output size 1
- Loss: binary cross-entropy

### Requirements

```python
# Your implementation must satisfy this interface:
import numpy as np

class TwoLayerNet:
    def __init__(self, input_dim: int, hidden_dim: int, seed: int = 42) -> None:
        """Initialize weights with He initialization for ReLU layers."""
        ...

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Return sigmoid output probabilities, shape (N,)."""
        ...

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Binary cross-entropy loss."""
        ...

    def backward(self, X: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
        """
        Compute gradients via backprop.
        Returns dict with keys: 'W1', 'b1', 'W2', 'b2'.
        """
        ...

    def train_step(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01
    ) -> float:
        """One GD update. Returns current loss."""
        ...

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions."""
        ...
```

### Acceptance criteria

1. Loss decreases over 500 training steps on a synthetic linearly-separable dataset
2. Final accuracy ≥ 85% on the training set
3. `backward` gradients match numerical gradients (max diff < 1e-4) for at least `W1`, `b1`
4. He initialization is used for `W1` (scale = `sqrt(2 / input_dim)`)

### Tests to include

```python
# tests/test_two_layer_net.py — write and run these
def test_forward_shape():     # output shape (N,)
def test_loss_positive():     # loss > 0
def test_backward_keys():     # returns correct keys
def test_gradient_check():    # numerical vs analytic, max diff < 1e-4
def test_loss_decreases():    # loss[499] < loss[0]
```

---

## Debugging Challenge — 20% weight

The following implementation has **three bugs**. Find and fix them. State each bug and its location.

```python
import numpy as np

def softmax_ce_loss(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Compute mean cross-entropy loss over a batch."""
    # Bug hunt: 3 bugs hidden below

    # Softmax
    exp_z = np.exp(logits)                           # Line A
    probs = exp_z / exp_z.sum()                      # Line B

    # Log probabilities
    log_probs = np.log(probs)                        # Line C

    # Gather correct class log-probs
    n = logits.shape[0]
    correct_log_probs = log_probs[y_true]            # Line D

    # Mean loss
    loss = np.sum(correct_log_probs) / n             # Line E
    return float(loss)


# Test case
logits  = np.array([[2.0, 1.0, 0.1], [0.5, 2.1, -1.0]])
y_true  = np.array([0, 1])
print(softmax_ce_loss(logits, y_true))  # Should be approximately 0.41
```

For each bug: (a) name the line, (b) describe what's wrong, (c) write the fix.

---

## Code Quality — 15% weight

Your submission will be evaluated on:

| Criterion | Description |
|---|---|
| Type hints | All functions fully annotated |
| Logging | `logging` used, no bare `print` |
| Docstrings | Each class/function has a one-line docstring |
| No magic numbers | Named constants for dims, learning rates, etc. |
| Test coverage | ≥ 10 passing pytest tests across the project |

---

## Rubric (pass ≥ 75%)

| Dimension | Weight | Points available |
|---|---|---|
| Independent implementation (coding challenge) | 40% | 40 |
| Conceptual understanding (written answers) | 25% | 25 |
| Debugging & problem-solving | 20% | 20 |
| Code quality & testing | 15% | 15 |
| **Total** | **100%** | **100** |

**Pass threshold: 75 / 100**

---

Answer key: ../solutions/week-01/assessment.md
