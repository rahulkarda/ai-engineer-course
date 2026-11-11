# Week 03 Project — Neural Network from Scratch and PyTorch Comparison

> Duration: weekend (~4-6 hr total)  ·  Language: Python 3.12+  ·  Prereqs: [[day-5]], [[day-6]], [[day-7]]

---

## Overview

Implement a 2-layer neural network twice: once in pure NumPy with manual backpropagation, and once using PyTorch's `nn.Module`. Train both on the same toy dataset and compare their behavior. This project is the synthesis of the entire week.

---

## Objectives

1. Internalize backpropagation by implementing it without automatic differentiation.
2. Map every manual operation to its PyTorch equivalent.
3. Verify correctness via a numerical gradient check.
4. Write a pytest test suite that exercises both implementations.
5. Produce a comparison log showing both models reach equivalent final accuracy.

---

## Dataset

Use the spiral dataset from Day 6: two interleaved spiral arms, 200 samples per class, `seed=42`. Normalize the input features to zero mean and unit standard deviation using training set statistics only (compute mean/std on train, apply to both train and test).

Split: 80% train, 20% test. No validation set needed for this project.

---

## Part 1 — NumPy implementation

File: `src/numpy_net.py`

**Requirements**:

- Class `TwoLayerNet(in_dim, hidden, out_dim, seed)` with:
  - `forward(X) -> logits` caching intermediate values for backprop.
  - `bce_loss(logits, y) -> float` using numerically stable log.
  - `backward(y) -> dict[str, ndarray]` returning gradients for all 4 parameter arrays.
  - `step(grads, lr)` applying SGD update.
  - `predict(X) -> ndarray` returning binary predictions.
- Training function `train(model, X, y, lr, epochs, log_every) -> list[float]` returning loss history.
- He initialization for weights; zero initialization for biases.
- All functions fully type-annotated using `numpy.typing.NDArray`.
- Use `logging` throughout; no `print` statements.

**Gradient check requirement**: Implement `gradient_check(model, X, y, eps=1e-5) -> float` that returns the maximum relative difference between numerical and analytical gradients across all parameters. This value must be below `1e-5` for the implementation to be considered correct.

---

## Part 2 — PyTorch implementation

File: `src/pytorch_net.py`

**Requirements**:

- Class `TwoLayerMLP(in_dim, hidden, out_dim)` using `nn.Module`, `nn.Linear`, `nn.ReLU`.
- `train_epoch(model, loader, optimizer, criterion) -> float` returning average train loss.
- `evaluate(model, loader, criterion) -> tuple[float, float]` returning `(avg_loss, accuracy)`.
- Full training loop in `train(cfg) -> nn.Module` function.
- `Config` dataclass with: `hidden`, `lr`, `weight_decay`, `epochs`, `batch_size`, `seed`.
- Device-agnostic: use `best_device()` from Day 7 pattern.
- Use `BCEWithLogitsLoss`.
- Use `Adam` optimizer.
- Log train loss and test accuracy every 10 epochs using `logging`.

---

## Part 3 — Comparison

File: `src/compare.py`

Write a script that:

1. Trains the NumPy implementation for 3000 epochs (SGD, `lr=0.01`, full-batch).
2. Trains the PyTorch implementation for 3000 epochs (SGD, `lr=0.01`, full-batch, no weight decay).
3. Logs final training accuracy and test accuracy for both.
4. Logs the ratio of final losses: `numpy_loss / pytorch_loss`. This should be close to 1.0.

Both should reach > 90% test accuracy. Minor differences (< 5 percentage points) are acceptable and expected due to floating-point ordering and initialization differences.

---

## Test requirements

File: `tests/test_numpy_net.py` and `tests/test_pytorch_net.py`

### `test_numpy_net.py` — required test functions

| Test | What it verifies |
|------|-----------------|
| `test_forward_output_shape` | `model.forward(X)` returns shape `(N, out_dim)` |
| `test_loss_is_finite` | Initial loss is finite and positive |
| `test_backward_grad_shapes` | Each gradient has the same shape as its parameter |
| `test_gradient_check_passes` | `gradient_check` returns value below `1e-5` |
| `test_xor_reaches_100_accuracy` | Model achieves 100% accuracy on XOR dataset after 5000 epochs |
| `test_predict_returns_binary` | `predict(X)` returns only 0.0 and 1.0 values |

### `test_pytorch_net.py` — required test functions

| Test | What it verifies |
|------|-----------------|
| `test_forward_output_shape` | Output shape `(N, 1)` for input `(N, 2)` |
| `test_initial_loss_near_log2` | Initial loss within 0.5 of `log(2)` |
| `test_training_reduces_loss` | Loss after 20 epochs < initial loss |
| `test_eval_mode_deterministic` | `model.eval()` forward pass returns same result on two calls |
| `test_spiral_accuracy_above_90pct` | Test accuracy > 0.90 after full training |

Run all tests with:

```bash
pytest tests/ -v
```

---

## Project structure

```
week-03-project/
  src/
    numpy_net.py
    pytorch_net.py
    compare.py
  tests/
    test_numpy_net.py
    test_pytorch_net.py
  pyproject.toml        (or requirements.txt)
  .gitignore
```

Minimum `pyproject.toml` dependencies:

```toml
[project]
requires-python = ">=3.12"
dependencies = [
    "torch",
    "numpy",
    "pytest",
]
```

---

## Acceptance criteria

- [ ] Both `numpy_net.py` and `pytorch_net.py` exist and are runnable without error.
- [ ] `python src/compare.py` runs without error and logs final accuracy for both implementations.
- [ ] Both implementations reach > 90% test accuracy on the spiral dataset.
- [ ] `gradient_check` passes (max relative error < 1e-5).
- [ ] All 11 required tests pass: `pytest tests/ -v` exits 0.
- [ ] No `print` statements in `src/`; logging used throughout.
- [ ] All functions and class methods have type hints.
- [ ] `He initialization` used for weights in both implementations.
- [ ] NumPy implementation uses only NumPy — no `torch` imports in `numpy_net.py`.

---

## Deliverables checklist

- [ ] `src/numpy_net.py` with `TwoLayerNet`, `train`, `gradient_check`
- [ ] `src/pytorch_net.py` with `TwoLayerMLP`, `Config`, `train_epoch`, `evaluate`, `train`
- [ ] `src/compare.py` that runs both and logs comparison
- [ ] `tests/test_numpy_net.py` with all 6 required tests
- [ ] `tests/test_pytorch_net.py` with all 5 required tests
- [ ] All tests passing: `pytest tests/ -v`
- [ ] `compare.py` output shows both models > 90% test accuracy

---

## Hints

- For the gradient check, perturb each scalar parameter element by element. A nested loop is fine for a small network.
- For the PyTorch full-batch comparison, set `batch_size = len(train_dataset)` to match NumPy's behavior.
- He initialization: `weight = N(0, sqrt(2 / fan_in))`. PyTorch's `nn.Linear` uses Kaiming uniform by default, not Kaiming normal — for a fair comparison, manually re-initialize PyTorch weights with `torch.nn.init.kaiming_normal_`.
- To make `test_eval_mode_deterministic` meaningful, set the model to `model.train()` first (with a dropout layer, if you add one), confirm the output differs, then set `model.eval()` and confirm it does not.
- The spiral dataset is not linearly separable. A hidden layer of at least 32 units is needed to reach 90% accuracy.
