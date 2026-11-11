# Phase 3 · Week 03 · Day 7 — GPU Basics, Numerical Stability, and Debugging Checklist

> Objective: Move models and tensors across devices, apply numerical stability best practices, and use a systematic debugging checklist.  ·  Time: ~30 min read + ~3 hr project time  ·  Prereqs: [[week-03/day-5]], [[week-03/day-6]]

---

## Concept

The last pillar before applying neural networks to real tasks: knowing where computation runs, why it can silently go wrong numerically, and how to diagnose problems methodically.

**Device management**: PyTorch tensors and modules live on a device. Moving them is explicit. Writing device-agnostic code means selecting the best available device at startup and calling `.to(device)` on models and tensors before they interact.

**Numerical stability**: Floating-point arithmetic is finite. Certain operations — `log(0)`, large softmax inputs, very small or very large activations — produce `nan` or `inf` silently and corrupt training. PyTorch provides numerically stable fused operations (e.g., `BCEWithLogitsLoss`, `CrossEntropyLoss`) that should be preferred over manual composition.

**Debugging**: Most PyTorch bugs fall into a small set of recurring categories. A checklist catches them faster than reading stack traces.

---

## How it works internally

### Device strings

| String | Hardware |
|--------|----------|
| `"cpu"` | CPU (always available) |
| `"cuda"` | NVIDIA GPU (requires CUDA drivers + torch built with CUDA) |
| `"cuda:0"` | First NVIDIA GPU explicitly |
| `"mps"` | Apple Silicon GPU (macOS 12.3+, PyTorch 1.12+) |

`torch.device` wraps the string. `tensor.device` returns its current device. Arithmetic between tensors on different devices raises a `RuntimeError` immediately.

### `.to(device)` for modules vs. tensors

- `model.to(device)` — moves all registered parameters and buffers in-place. The model object itself is mutated; no need to reassign.
- `tensor.to(device)` — returns a new tensor on the target device. The original is unchanged. You must reassign: `x = x.to(device)`.

### Why `BCEWithLogitsLoss` instead of `sigmoid` + `BCELoss`

`BCELoss` computes `log(p)` where `p = sigmoid(logit)`. When `logit` is very negative, `sigmoid(logit)` rounds to `0.0` in float32, and `log(0)` is `-inf`. `BCEWithLogitsLoss` uses the log-sum-exp trick internally:

```
log sigmoid(z) = z - log(1 + exp(z))
               = -log(1 + exp(-z))      (when z >= 0)
```

The fused form never evaluates `sigmoid` separately, avoiding the zero-log issue.

### Gradient flow and NaN propagation

A single `nan` in a gradient infects all downstream parameters via the chain rule. After it appears, every subsequent parameter update produces `nan` and the loss becomes `nan` too. The debugger's first job is to find the first layer where `nan` appeared.

---

## Worked example

```python
# day7_device_stability.py
"""
Device-agnostic training, numerical stability patterns, and debug utilities.
Runs on CPU; MPS/CUDA paths are shown but not required.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def best_device() -> torch.device:
    """Return the best available device in priority order: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info("Using device: %s", device)
    return device


# ---------------------------------------------------------------------------
# Device-agnostic model definition
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_features: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def device_agnostic_example() -> None:
    device = best_device()

    model = MLP(4, 16)
    model.to(device)   # moves parameters in-place; no reassignment needed

    X = torch.randn(8, 4)
    y = torch.randint(0, 2, (8, 1)).float()

    # Tensors must be on the same device as the model
    X = X.to(device)
    y = y.to(device)

    logits = model(X)
    loss = nn.BCEWithLogitsLoss()(logits, y)
    log.info("Loss on %s: %.4f", device, loss.item())

    # Move back to CPU for NumPy conversion
    result = logits.detach().cpu().numpy()
    log.info("Logits shape on CPU: %s", result.shape)


# ---------------------------------------------------------------------------
# Numerical stability demo
# ---------------------------------------------------------------------------

def stability_demo() -> None:
    """Show the difference between naive and stable log-softmax."""
    large_logits = torch.tensor([1000.0, 1001.0, 999.0])

    # Naive: softmax first then log
    probs_naive = torch.exp(large_logits) / torch.exp(large_logits).sum()
    log_probs_naive = torch.log(probs_naive)
    log.info("Naive log-softmax: %s  (nan = %s)", log_probs_naive, torch.isnan(log_probs_naive).any().item())

    # Stable: PyTorch built-in
    log_probs_stable = torch.log_softmax(large_logits, dim=0)
    log.info("Stable log-softmax: %s", log_probs_stable)

    # BCE: naive vs. fused
    extreme_logit = torch.tensor([-80.0])
    label         = torch.tensor([1.0])

    prob_naive = torch.sigmoid(extreme_logit)       # rounds to 0 in float32
    loss_naive = -(label * torch.log(prob_naive + 1e-8)).mean()
    loss_fused = nn.BCEWithLogitsLoss()(extreme_logit, label)
    log.info("Naive BCE loss: %.6f   Fused BCE loss: %.6f", loss_naive.item(), loss_fused.item())


# ---------------------------------------------------------------------------
# NaN detection hook
# ---------------------------------------------------------------------------

def register_nan_hooks(model: nn.Module) -> None:
    """
    Register forward hooks on every layer.
    Logs a warning if any output contains nan or inf.
    Useful during debugging; remove for production.
    """
    def make_hook(name: str):
        def hook(module: nn.Module, input: tuple, output: torch.Tensor) -> None:
            if torch.isnan(output).any() or torch.isinf(output).any():
                log.warning("NaN/Inf detected in output of layer '%s'", name)
        return hook

    for name, module in model.named_modules():
        if name:  # skip the top-level container
            module.register_forward_hook(make_hook(name))


# ---------------------------------------------------------------------------
# Debugging checklist (programmatic checks)
# ---------------------------------------------------------------------------

def run_debug_checks(
    model: nn.Module,
    X_sample: torch.Tensor,
    y_sample: torch.Tensor,
) -> None:
    """
    Run a set of common sanity checks before a full training run.
    Logs warnings for anything suspicious.
    """
    log.info("--- Debug pre-flight checks ---")

    # 1. Parameter count
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Trainable parameters: %d", n_params)

    # 2. Forward pass without NaN
    model.eval()
    with torch.no_grad():
        out = model(X_sample)
    if torch.isnan(out).any():
        log.warning("FAIL: forward pass produces NaN — check initialization.")
    else:
        log.info("PASS: forward pass clean.")

    # 3. Loss is finite
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(out, y_sample)
    if not math.isfinite(loss.item()):
        log.warning("FAIL: loss is not finite (%s) — check targets/labels.", loss.item())
    else:
        log.info("PASS: initial loss finite = %.4f", loss.item())

    # 4. Expected initial loss for binary classification
    # With random weights: loss should be near log(2) ~ 0.693
    expected = math.log(2)
    if abs(loss.item() - expected) > 1.0:
        log.warning(
            "WARN: initial loss %.4f far from expected %.4f — check output head or data scale.",
            loss.item(), expected,
        )

    # 5. Gradients exist after backward
    model.train()
    loss = criterion(model(X_sample), y_sample)
    loss.backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    if no_grad:
        log.warning("FAIL: no gradient for layers: %s — possible disconnected graph.", no_grad)
    else:
        log.info("PASS: all parameters have gradients.")

    # 6. Gradient magnitudes (dead neurons / exploding gradients)
    for name, param in model.named_parameters():
        if param.grad is not None:
            gnorm = param.grad.norm().item()
            if gnorm == 0.0:
                log.warning("WARN: zero gradient for '%s' — possible dead neurons.", name)
            elif gnorm > 10.0:
                log.warning("WARN: large gradient norm %.2f for '%s' — possible explosion.", gnorm, name)

    log.info("--- Debug checks complete ---")


def main() -> None:
    device_agnostic_example()
    stability_demo()

    # Pre-flight check demo
    model = MLP(4, 16)
    X = torch.randn(16, 4)
    y = torch.randint(0, 2, (16, 1)).float()
    run_debug_checks(model, X, y)


if __name__ == "__main__":
    main()
```

---

## Common debugging checklist

Use this before assuming the architecture is wrong:

| # | Check | How to verify |
|---|-------|---------------|
| 1 | Loss starts near expected value | `log(n_classes)` for random weights |
| 2 | Inputs are normalized | `X.mean()`, `X.std()` — should be ~0, ~1 |
| 3 | `.train()` called before training loop | Dropout off during eval is a silent bug |
| 4 | `zero_grad()` called before `backward()` | Gradients accumulate otherwise |
| 5 | Loss uses logits, not probabilities | Use `BCEWithLogitsLoss`, not `BCELoss(sigmoid(logits))` |
| 6 | No NaN in any layer output | Register forward hooks (see `register_nan_hooks`) |
| 7 | Gradients exist for all parameters | Check `.grad is None` after first backward |
| 8 | Tensors and model on same device | `RuntimeError: Expected all tensors to be on the same device` |
| 9 | Labels are correct dtype | `float32` for BCE, `int64` for CrossEntropy |
| 10 | Learning rate not too high | Try `lr=1e-3` with Adam as a safe default |

---

## Exercises

1. **Device string probe**: Write a function `device_info() -> dict[str, str]` that returns a dict with keys `"best"`, `"cuda_available"`, `"mps_available"`, and `"torch_version"`. Log the result at startup of any training script.

2. **Stable softmax**: Implement `stable_softmax(x: torch.Tensor) -> torch.Tensor` using the max-subtraction trick (`x - x.max()`). Verify it matches `torch.softmax` for both normal and large-value inputs.

3. **NaN injection and detection**: Manually set one weight to `float('nan')` after initialization. Use `register_nan_hooks` to identify which layer first produces `nan` in the forward pass. Then use `torch.nn.utils.clip_grad_norm_` after `backward()` and observe whether it prevents NaN from spreading.

4. **Pre-flight harness**: Wrap `run_debug_checks` into a `pytest` test that runs on the `MLP(4, 16)` model with random data and asserts that all checks pass. This becomes a reusable regression test for any new model.

(Solutions: ../solutions/week-03/day-7.md)

---

## Self-check

- [ ] I can select the best available device and write a training loop that runs without code changes on CPU, CUDA, or MPS.
- [ ] I can explain why `BCEWithLogitsLoss` is numerically safer than `BCELoss(sigmoid(logits))`.
- [ ] I can run through the debugging checklist on a new model before a training run.
- [ ] I know how to register a forward hook to detect NaN/Inf in any layer.

---

## Supplementary links

- PyTorch device docs: https://pytorch.org/docs/stable/tensor_attributes.html#torch.device
- Numerical stability in deep learning: https://d2l.ai/chapter_appendix-mathematics/numerical-stability-and-init.html
- `torch.nn.utils.clip_grad_norm_`: https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
- PyTorch debugging guide: https://pytorch.org/docs/stable/notes/debugging.html
- MPS backend for Apple Silicon: https://pytorch.org/docs/stable/notes/mps.html
