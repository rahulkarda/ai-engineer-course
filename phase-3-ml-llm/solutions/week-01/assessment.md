# Solutions — Phase 3 · Week 01 · Assessment

## Conceptual Answers

### 1. Dot product

The dot product `a · b = Σ aᵢbᵢ` computes the product of the magnitudes and the cosine of the angle between the vectors: `a · b = ‖a‖ ‖b‖ cos(θ)`. It measures alignment — large positive means same direction, zero means orthogonal, negative means opposite. ML use case: a linear layer computes `W @ x` which is a dot product between each weight row and the input, producing an activation score for each neuron.

### 2. L1 vs L2 regularization

L1 adds `λ Σ|wᵢ|` to the loss; L2 adds `λ Σwᵢ²`. L1 encourages **sparsity** — the subgradient of `|w|` at `w ≈ 0` is constant (±1), so the optimizer pushes small weights all the way to zero. L2's gradient `2λw` shrinks proportionally to magnitude, so small weights get small pressure and never quite reach zero. Use L1 when you want few active features (Lasso); L2 when you want all features small (Ridge).

### 3. Gradient vs. derivative

A derivative `f'(x)` is a scalar measuring rate of change when input is 1-D. A gradient `∇f(x)` is a vector collecting partial derivatives for every input dimension — it generalizes the derivative to multi-dimensional inputs. Gradient descent steps in the **negative** gradient direction because the gradient points toward steepest ascent; we want to descend (decrease loss).

### 4. Chain rule & backprop

For composed functions `f(g(x))`: `d(f∘g)/dx = f'(g(x)) · g'(x)`. For `loss → sigmoid → linear`: we compute `∂loss/∂sigmoid_output`, multiply by `∂sigmoid/∂z` (sigmoid_grad), then multiply by `∂z/∂W = X` to get `∂loss/∂W`. Backprop traverses the computation graph in reverse, accumulating these products at each node using the chain rule — no new math, just systematic application.

### 5. Cross-entropy vs. MSE for classification

Cross-entropy `H(p, q) = -Σ p log q` measures how many bits are needed to encode the true distribution using the predicted one. For classification, minimizing CE is equivalent to maximum likelihood estimation — we maximize the log probability assigned to correct labels. MSE penalizes the *numerical distance* between predicted probability and label (0 or 1), which is less appropriate: a prediction of 0.51 for the correct class gets low MSE but high CE, correctly signaling lack of confidence.

### 6. Eigenvalue in plain language

An eigenvector is a special direction for a matrix that, when you apply the matrix transformation, only gets stretched or shrunk — not rotated. The eigenvalue is the stretch factor. In ML: the covariance matrix's eigenvectors are the directions of maximum variance in the data (principal components). A large eigenvalue means high variance in that direction.

### 7. Bayes' theorem for spam

`P(spam | email) = P(email | spam) * P(spam) / P(email)`

- **Prior** `P(spam)`: fraction of spam emails before seeing content (~20%)
- **Likelihood** `P(email | spam)`: probability of this word pattern given it's spam
- **Posterior** `P(spam | email)`: updated spam probability after seeing the words

---

## Coding Challenge — Reference Solution

```python
"""
assessment_solution.py — Two-layer neural network from scratch.
"""
import logging
from dataclasses import dataclass, field
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

CLIP_VAL: float = 500.0


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -CLIP_VAL, CLIP_VAL)))


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0.0).astype(float)


@dataclass
class TwoLayerNet:
    input_dim:  int
    hidden_dim: int
    seed:       int = 42

    W1: np.ndarray = field(init=False)
    b1: np.ndarray = field(init=False)
    W2: np.ndarray = field(init=False)
    b2: np.ndarray = field(init=False)

    # Cache for backprop
    _cache: dict = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        # He init for ReLU
        scale1 = np.sqrt(2.0 / self.input_dim)
        scale2 = np.sqrt(2.0 / self.hidden_dim)
        self.W1 = rng.standard_normal((self.hidden_dim, self.input_dim)) * scale1
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = rng.standard_normal((1, self.hidden_dim)) * scale2
        self.b2 = np.zeros(1)

    def forward(self, X: np.ndarray) -> np.ndarray:
        z1 = X @ self.W1.T + self.b1          # (N, H)
        a1 = relu(z1)                          # (N, H)
        z2 = a1 @ self.W2.T + self.b2         # (N, 1)
        a2 = sigmoid(z2).flatten()             # (N,)
        self._cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
        return a2

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        a2 = self.forward(X)
        eps = 1e-9
        return float(-np.mean(y * np.log(a2 + eps) + (1 - y) * np.log(1 - a2 + eps)))

    def backward(self, X: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
        _ = self.forward(X)  # populate cache
        a2 = self._cache["a2"]
        a1 = self._cache["a1"]
        z1 = self._cache["z1"]
        n  = X.shape[0]

        # Output layer
        d_a2  = (a2 - y) / n                         # (N,)
        dz2   = d_a2.reshape(-1, 1)                   # (N, 1)
        dW2   = dz2.T @ a1                            # (1, H)
        db2   = dz2.sum(axis=0)                       # (1,)

        # Hidden layer
        da1   = dz2 @ self.W2                         # (N, H)
        dz1   = da1 * relu_grad(z1)                   # (N, H)
        dW1   = dz1.T @ X                             # (H, D)
        db1   = dz1.sum(axis=0)                       # (H,)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def train_step(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01) -> float:
        grads = self.backward(X, y)
        self.W1 -= lr * grads["W1"]
        self.b1 -= lr * grads["b1"]
        self.W2 -= lr * grads["W2"]
        self.b2 -= lr * grads["b2"]
        return self.loss(X, y)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.forward(X) >= threshold).astype(int)


def make_binary_data(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X0  = rng.normal([-2, -2], 0.5, (150, 2))
    X1  = rng.normal([+2, +2], 0.5, (150, 2))
    X   = np.vstack([X0, X1]).astype(float)
    y   = np.array([0.0] * 150 + [1.0] * 150)
    return X, y


def demo() -> None:
    X, y = make_binary_data()
    net  = TwoLayerNet(input_dim=2, hidden_dim=8)
    losses: list[float] = []

    for step in range(500):
        loss = net.train_step(X, y, lr=0.05)
        losses.append(loss)
        if step % 50 == 0:
            acc = float(np.mean(net.predict(X) == y.astype(int)))
            log.info("step %3d | loss=%.4f | acc=%.4f", step, loss, acc)

    final_acc = float(np.mean(net.predict(X) == y.astype(int)))
    log.info("Final accuracy: %.4f", final_acc)
    assert losses[-1] < losses[0], "Loss did not decrease"
    assert final_acc >= 0.85, f"Accuracy too low: {final_acc:.4f}"


if __name__ == "__main__":
    demo()
```

---

## Debugging Challenge — Bug Fixes

**Bug 1 — Line A: No numerical stability**  
`np.exp(logits)` overflows for large logits.  
Fix: `exp_z = np.exp(logits - logits.max(axis=1, keepdims=True))`

**Bug 2 — Line B: Softmax sums along wrong axis**  
`exp_z.sum()` sums all elements instead of per-row.  
Fix: `probs = exp_z / exp_z.sum(axis=1, keepdims=True)`

**Bug 3 — Line E: Missing negation**  
Cross-entropy is *negative* log-likelihood; summing without negation gives negative loss.  
Fix: `loss = -np.sum(correct_log_probs) / n`

```python
def softmax_ce_loss_fixed(logits: np.ndarray, y_true: np.ndarray) -> float:
    exp_z     = np.exp(logits - logits.max(axis=1, keepdims=True))  # Bug 1 fixed
    probs     = exp_z / exp_z.sum(axis=1, keepdims=True)             # Bug 2 fixed
    log_probs = np.log(probs + 1e-9)
    n         = logits.shape[0]
    correct_log_probs = log_probs[np.arange(n), y_true]
    loss      = -np.sum(correct_log_probs) / n                       # Bug 3 fixed
    return float(loss)

logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.1, -1.0]])
y_true = np.array([0, 1])
print(softmax_ce_loss_fixed(logits, y_true))  # ≈ 0.41
```
