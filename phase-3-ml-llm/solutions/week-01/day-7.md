# Solutions — Phase 3 · Week 01 · Day 7

## Reflection Model Answers

### 1. Most surprising or counter-intuitive concept

Many learners find the chain rule's simplicity surprising — the same mechanical rule underpins all of backpropagation in billion-parameter models. Another common "aha": cross-entropy being exactly maximum likelihood estimation — the loss function follows directly from probability theory, not a heuristic choice.

### 2. Gradient descent in 3 sentences

A loss surface is like a hilly landscape. Gradient descent measures which direction is uphill (the gradient) and takes a small step downhill (subtracts a fraction of the gradient). Repeat until you reach a flat valley — the minimum loss, where the model best fits the data.

### 3. Open questions to investigate

- How do adaptive optimizers (Adam, AdaGrad) improve on vanilla GD?
- Why does He initialization matter for ReLU networks?
- What is the relationship between batch normalization and gradient stability?

---

## Project Tips

### He Initialization

```python
W1 = rng.standard_normal((hidden_dim, input_dim)) * np.sqrt(2.0 / input_dim)
```

Keeps activation variance stable through ReLU layers by accounting for the fact that ReLU zeroes out half the activations.

### Two-Layer Backprop Walkthrough

Forward pass:
```
z1 = X @ W1.T + b1          # (B, H)
a1 = ReLU(z1)               # (B, H)
z2 = a1 @ W2.T + b2         # (B, 1)
a2 = sigmoid(z2).flatten()  # (B,)
loss = binary_CE(a2, y)
```

Backward pass (chain rule):
```
# Output layer
d_loss  = (a2 - y) / B               # (B,) — BCE gradient
dz2     = d_loss * sigmoid_grad(z2)   # (B, 1)
dW2     = dz2.T @ a1                  # (1, H)
db2     = dz2.sum(axis=0)             # (1,)

# Hidden layer
da1     = dz2 @ W2                    # (B, H)
dz1     = da1 * relu_grad(z1)         # (B, H)
dW1     = dz1.T @ X                   # (H, D)
db1     = dz1.sum(axis=0)             # (H,)
```

### Common Mistakes

1. Forgetting to divide gradient by batch size — leads to exploding loss
2. Using `sigmoid` output (not logits) as input to cross-entropy — numerical instability
3. Shape mismatch in backward pass — always log shapes during debugging
4. Not clipping sigmoid input — `exp(-large)` underflows to zero, causing `nan`
