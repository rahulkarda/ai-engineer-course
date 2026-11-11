# Phase 3 · Week 03 · Day 4 — PyTorch Tensors and Autograd

> Objective: Understand what PyTorch tensors are, how the autograd computation graph works, and verify it computes the same gradients as Day 3's manual backprop.  ·  Time: ~1 hr  ·  Prereqs: [[week-03/day-3]]

---

## Concept

Day 3 showed that computing gradients by hand is mechanical but tedious. **PyTorch autograd** automates exactly this: it builds a computation graph during the forward pass and traverses it in reverse during `.backward()` to accumulate gradients.

The key insight: autograd is just the chain rule, implemented automatically.

### What a Tensor Is

A `torch.Tensor` is a multi-dimensional array (like NumPy ndarray) with two extra capabilities:

1. **Device awareness** — can live on CPU or GPU; all ops dispatch to the right kernel automatically.
2. **Gradient tracking** — if `requires_grad=True`, PyTorch records every operation on the tensor into a computation graph. Calling `.backward()` on any scalar derived from it computes `d(scalar)/d(tensor)` and accumulates into `.grad`.

```python
import torch

x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x ** 2 + 3 * x        # operations recorded
z = y.sum()                # scalar
z.backward()               # backprop
# x.grad = dz/dx = 2x + 3 = [7.0, 9.0]
```

### The Computation Graph

Every operation creates a **node** in a directed acyclic graph (DAG). Leaf nodes are tensors with `requires_grad=True`. Interior nodes are function objects (e.g., `MulBackward`, `AddBackward`). Edges represent data flow.

```
x ──> **2 ──> y ──> sum ──> z
       ↑
   grad_fn=PowBackward0
```

Calling `z.backward()` traverses from `z` back to `x`, multiplying local Jacobians at each step — exactly the chain rule.

By default the graph is **freed after** `.backward()`. Call with `retain_graph=True` to keep it (needed for multiple backward passes).

### Tensor Operations

**Creation:**
```python
torch.zeros(3, 4)           # all zeros
torch.ones(2, 2)
torch.randn(3, 3)           # standard normal
torch.tensor([1.0, 2.0])    # from Python list
torch.from_numpy(arr)       # from NumPy, shares memory
```

**Indexing (same as NumPy):**
```python
t[0]          # first row
t[:, 1]       # second column
t[t > 0]      # boolean mask
```

**Broadcasting:** PyTorch follows NumPy rules. A `(3,)` bias vector added to a `(B, 3)` activation broadcasts correctly without explicit expansion.

**Device:**
```python
t = t.to("cpu")        # explicit CPU (default)
# t = t.to("cuda")     # GPU — not needed for this course
```

### .grad, .grad_fn, and .requires_grad

| Attribute | Meaning |
|---|---|
| `t.requires_grad` | True if PyTorch tracks ops on this tensor |
| `t.grad_fn` | The function that created this tensor (None for leaves) |
| `t.grad` | Accumulated gradient after `.backward()` — only set on leaf tensors |
| `t.is_leaf` | True if user-created (not the result of an op) |

**Gotcha**: `.grad` accumulates across multiple `.backward()` calls. Always zero gradients between training steps: `optimizer.zero_grad()` or `t.grad.zero_()`.

---

## How it works internally

### Dynamic computation graph (define-by-run)

Unlike static graph frameworks (early TensorFlow), PyTorch builds the graph **as Python runs**. Each tensor op instantiates a `Function` subclass (e.g., `ReluBackward`, `MmBackward`) and attaches it to the output tensor as `grad_fn`. This makes debugging natural — you can inspect the graph with `t.grad_fn`.

### Backward pass implementation

Calling `loss.backward()`:
1. Sets `loss.grad = 1.0` (seed gradient).
2. Calls `loss.grad_fn.backward(1.0)` — the last function in the graph.
3. Each function receives `grad_output`, computes `grad_input = Jacobian.T @ grad_output`, and calls `.backward()` on its inputs recursively.
4. Leaf tensor gradients are accumulated into `.grad`.

This is topological traversal of the DAG — exactly the manual backprop from Day 3.

### Memory: when to use `torch.no_grad()`

During inference (evaluation), you do not need gradients. Wrapping code in:

```python
with torch.no_grad():
    output = model(x)
```

disables graph construction — no `grad_fn` objects are created, reducing memory by roughly 50% for typical networks.

### `.detach()` vs `torch.no_grad()`

- `t.detach()` returns a new tensor that shares data but is severed from the graph — gradients do not flow through it.
- `torch.no_grad()` context: nothing inside builds a graph at all.

Use `.detach()` for things like target networks in RL. Use `no_grad()` for inference/evaluation.

---

## Worked example

```python
"""
day_4_autograd.py — PyTorch tensors and autograd.

Install: uv add torch

Demonstrates:
1. Basic tensor operations and gradient tracking.
2. Same 2-layer MLP from Day 3 implemented in PyTorch.
3. Autograd gradients verified against Day 3 manual backprop.
"""

import logging

import numpy as np
import torch
import torch.nn.functional as F  # relu, softmax, cross_entropy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)


# ---- Part 1: Tensor basics ----

def demo_tensors() -> None:
    log.info("=== Tensor Basics ===")

    # Creation
    t_zeros = torch.zeros(2, 3)
    t_rand = torch.randn(2, 3)
    log.info("zeros:\n%s", t_zeros)
    log.info("randn:\n%s", t_rand)

    # From NumPy — shares memory
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t_from_np = torch.from_numpy(arr)
    log.info("from_numpy: %s  dtype=%s", t_from_np, t_from_np.dtype)

    # To NumPy — shares memory when on CPU
    t_back = t_from_np.numpy()
    log.info("back to numpy: %s  type=%s", t_back, type(t_back))

    # Shapes and indexing
    t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    log.info("shape: %s  ndim: %d", t.shape, t.ndim)
    log.info("row 0: %s", t[0])
    log.info("col 2: %s", t[:, 2])
    log.info("element [1,3]: %s", t[1, 3])

    # Broadcasting
    bias = torch.tensor([1.0, 2.0, 3.0, 4.0])
    t_biased = t + bias  # (3,4) + (4,) — broadcasts correctly
    log.info("After bias broadcast:\n%s", t_biased)


# ---- Part 2: Autograd basics ----

def demo_autograd() -> None:
    log.info("=== Autograd Basics ===")

    # Simple: y = x^2 + 3x, dy/dx = 2x + 3
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    y = x ** 2 + 3.0 * x
    z = y.sum()
    z.backward()

    analytical = 2.0 * x.detach() + 3.0
    log.info("x: %s", x.data)
    log.info("x.grad: %s  (expected: %s)", x.grad, analytical)

    # grad_fn chain
    log.info("z.grad_fn: %s", z.grad_fn)
    log.info("y.grad_fn: %s", y.grad_fn)
    log.info("x.is_leaf: %s", x.is_leaf)

    # No grad context
    with torch.no_grad():
        y_inference = x ** 2 + 3.0 * x
    log.info("y_inference.requires_grad: %s  (should be False)", y_inference.requires_grad)


# ---- Part 3: 2-layer MLP matching Day 3 ----

def mlp_forward(
    x: torch.Tensor,
    W1: torch.Tensor,
    b1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    """Forward pass: same architecture as Day 3.

    x   -> linear(W1, b1) -> relu -> linear(W2, b2) -> logits
    """
    z1 = x @ W1.t() + b1       # (batch, n_hidden)
    a1 = F.relu(z1)             # (batch, n_hidden)
    z2 = a1 @ W2.t() + b2      # (batch, n_out)
    return z2                   # return logits, not probabilities


def demo_mlp_autograd() -> None:
    log.info("=== MLP Autograd ===")

    torch.manual_seed(7)
    rng = np.random.default_rng(seed=7)

    n_in, n_hidden, n_out, batch = 4, 8, 3, 30

    # Create parameters with requires_grad=True
    W1 = torch.randn(n_hidden, n_in, requires_grad=True) * (2.0 / n_in) ** 0.5
    b1 = torch.zeros(n_hidden, requires_grad=True)
    W2 = torch.randn(n_out, n_hidden, requires_grad=True) * (1.0 / n_hidden) ** 0.5
    b2 = torch.zeros(n_out, requires_grad=True)

    # Random input + one-hot targets
    x = torch.randn(batch, n_in)
    labels = torch.randint(0, n_out, (batch,))

    # Forward
    logits = mlp_forward(x, W1, b1, W2, b2)
    log.info("logits shape: %s", logits.shape)

    # Loss: F.cross_entropy takes raw logits + integer labels (no one-hot needed)
    # It internally applies log_softmax then NLL — numerically stable
    loss = F.cross_entropy(logits, labels)
    log.info("Loss: %.6f", loss.item())

    # Backward — one call computes all gradients
    loss.backward()

    log.info("dL/dW1 shape: %s  norm: %.6f", W1.grad.shape, W1.grad.norm().item())
    log.info("dL/db1 shape: %s  norm: %.6f", b1.grad.shape, b1.grad.norm().item())
    log.info("dL/dW2 shape: %s  norm: %.6f", W2.grad.shape, W2.grad.norm().item())
    log.info("dL/db2 shape: %s  norm: %.6f", b2.grad.shape, b2.grad.norm().item())


# ---- Part 4: Cross-verify autograd vs manual backprop ----

def demo_gradient_comparison() -> None:
    """Run Day 3 manual backprop and PyTorch autograd on identical weights + data.

    They should agree to floating-point precision.
    """
    log.info("=== Autograd vs Manual Backprop Comparison ===")

    seed = 42
    rng = np.random.default_rng(seed=seed)
    torch.manual_seed(seed)

    n_in, n_hidden, n_out, batch = 3, 5, 2, 8

    # Generate identical data in both frameworks
    x_np = rng.standard_normal((batch, n_in)).astype(np.float64)
    labels_np = rng.integers(0, n_out, size=batch)

    # Build shared weight arrays (NumPy)
    W1_np = rng.standard_normal((n_hidden, n_in)) * 0.1
    b1_np = np.zeros(n_hidden)
    W2_np = rng.standard_normal((n_out, n_hidden)) * 0.1
    b2_np = np.zeros(n_out)

    # ---- Manual backprop (Day 3 logic) ----
    def relu_np(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    def softmax_np(z: np.ndarray) -> np.ndarray:
        shifted = z - z.max(axis=1, keepdims=True)
        e = np.exp(shifted)
        return e / e.sum(axis=1, keepdims=True)

    # Forward
    z1 = x_np @ W1_np.T + b1_np
    a1 = relu_np(z1)
    z2 = a1 @ W2_np.T + b2_np

    # One-hot
    y_onehot = np.zeros((batch, n_out))
    y_onehot[np.arange(batch), labels_np] = 1.0

    # Backward
    delta2 = (softmax_np(z2) - y_onehot) / batch
    dW2_manual = delta2.T @ a1
    db2_manual = delta2.sum(axis=0)
    da1 = delta2 @ W2_np
    delta1 = da1 * (z1 > 0).astype(np.float64)
    dW1_manual = delta1.T @ x_np
    db1_manual = delta1.sum(axis=0)

    # ---- PyTorch autograd ----
    # Use float32 to match PyTorch default — convert manually
    W1_t = torch.tensor(W1_np, dtype=torch.float64, requires_grad=True)
    b1_t = torch.tensor(b1_np, dtype=torch.float64, requires_grad=True)
    W2_t = torch.tensor(W2_np, dtype=torch.float64, requires_grad=True)
    b2_t = torch.tensor(b2_np, dtype=torch.float64, requires_grad=True)

    x_t = torch.tensor(x_np, dtype=torch.float64)
    labels_t = torch.tensor(labels_np, dtype=torch.long)

    z1_t = x_t @ W1_t.t() + b1_t
    a1_t = F.relu(z1_t)
    z2_t = a1_t @ W2_t.t() + b2_t

    loss_t = F.cross_entropy(z2_t, labels_t)
    loss_t.backward()

    # ---- Compare ----
    pairs = [
        ("dW1", dW1_manual, W1_t.grad.numpy()),
        ("db1", db1_manual, b1_t.grad.numpy()),
        ("dW2", dW2_manual, W2_t.grad.numpy()),
        ("db2", db2_manual, b2_t.grad.numpy()),
    ]

    for name, manual, auto in pairs:
        max_abs_diff = np.abs(manual - auto).max()
        status = "MATCH" if max_abs_diff < 1e-8 else "MISMATCH"
        log.info("%s [%s]: max |manual - autograd| = %.2e", name, status, max_abs_diff)


# ---- Training loop with autograd ----

def demo_training_loop() -> None:
    log.info("=== Training Loop with Autograd ===")

    torch.manual_seed(0)
    n_in, n_hidden, n_out, batch = 4, 16, 3, 50

    W1 = torch.randn(n_hidden, n_in, requires_grad=True) * (2.0 / n_in) ** 0.5
    b1 = torch.zeros(n_hidden, requires_grad=True)
    W2 = torch.randn(n_out, n_hidden, requires_grad=True) * (1.0 / n_hidden) ** 0.5
    b2 = torch.zeros(n_out, requires_grad=True)

    x = torch.randn(batch, n_in)
    labels = torch.randint(0, n_out, (batch,))

    lr = 0.05
    params = [W1, b1, W2, b2]

    losses: list[float] = []
    for epoch in range(300):
        # Forward
        logits = mlp_forward(x, W1, b1, W2, b2)
        loss = F.cross_entropy(logits, labels)

        # Backward
        # Must zero gradients before each backward — they accumulate by default
        for p in params:
            if p.grad is not None:
                p.grad.zero_()

        loss.backward()

        # Update — use no_grad to avoid tracking the update itself
        with torch.no_grad():
            for p in params:
                p -= lr * p.grad  # type: ignore[operator]

        losses.append(loss.item())

        if (epoch + 1) % 50 == 0 or epoch == 0:
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                acc = (preds == labels).float().mean().item()
            log.info("Epoch %4d  loss=%.4f  accuracy=%.2f", epoch + 1, loss.item(), acc)

    log.info("Loss decreased: %s", losses[-1] < losses[0])


# ---- Main ----

def main() -> None:
    demo_tensors()
    demo_autograd()
    demo_mlp_autograd()
    demo_gradient_comparison()
    demo_training_loop()


if __name__ == "__main__":
    main()
```

---

## Exercises

1. **Conceptual**: Explain the difference between `t.detach()` and wrapping code in `torch.no_grad()`. Give a concrete use case for each.

2. **Coding**: Implement the Day 3 training loop using `torch.optim.SGD` instead of the manual `p -= lr * p.grad` update. Verify it produces the same loss trajectory. Hint: `optimizer = torch.optim.SGD(params, lr=0.05)` then `optimizer.zero_grad()` and `optimizer.step()`.

3. **Gradient accumulation bug**: Remove the `p.grad.zero_()` call from the training loop. Run for 10 epochs and log `W1.grad.norm()` each epoch. Explain what you observe and why this is a common bug.

4. **Graph inspection**: After the forward pass (before `loss.backward()`), log `logits.grad_fn`, `logits.grad_fn.next_functions`, and trace the graph two levels deep. Draw the computation graph on paper.

5. **dtype precision**: Run `demo_gradient_comparison()` with `dtype=torch.float32` instead of `float64`. Report the new `max |manual - autograd|` values. Explain why they differ.

(Solutions: ../solutions/week-03/day-4.md)

---

## Self-check

- [ ] I can explain what `requires_grad`, `.grad`, `.grad_fn`, and `.is_leaf` mean and how they relate.
- [ ] I can write a complete training loop in PyTorch (forward, loss, zero_grad, backward, step) from memory.
- [ ] I can explain why `p.grad.zero_()` must be called before each backward pass.
- [ ] I can verify that autograd gradients match manual backprop gradients to floating-point precision.
- [ ] I can explain the performance and memory implications of `torch.no_grad()` and when to use it.

---

## Setup

Install PyTorch (CPU-only):

```
uv add torch
```

No CUDA or GPU required. All examples run on CPU.

---

## Supplementary Links

- PyTorch Autograd mechanics (official): https://pytorch.org/docs/stable/notes/autograd.html
- Autograd tutorial: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
- PyTorch tensor overview: https://pytorch.org/docs/stable/tensors.html
- Karpathy "The spelled-out intro to neural networks and backpropagation": https://www.youtube.com/watch?v=VMj-3S1tku0
- PyTorch internals blog post (Edward Yang): http://blog.ezyang.com/2019/05/pytorch-internals/
