# Solutions — Week 03 · Day 4

## Exercise 1 — Tensor operations

```python
import logging
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Creation
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.ones(3)
c = torch.zeros(2, 3)
d = torch.randn(3, 3)

# Arithmetic (element-wise)
logger.info("a + b = %s", a + b)
logger.info("a * 2 = %s", a * 2)
logger.info("a @ b = %s", a @ b)   # dot product = 6.0

# Reshaping
x = torch.arange(12, dtype=torch.float32)
logger.info("x shape         : %s", x.shape)   # [12]
x_mat = x.reshape(3, 4)
logger.info("reshaped (3,4)  : %s", x_mat.shape)
x_T = x_mat.T
logger.info("transposed (4,3): %s", x_T.shape)

# Indexing
logger.info("row 0   : %s", x_mat[0])
logger.info("col 2   : %s", x_mat[:, 2])
logger.info("slice   : %s", x_mat[1:, 2:])

# Reduction
logger.info("sum     : %s", x_mat.sum())
logger.info("mean    : %s", x_mat.mean())
logger.info("max     : %s", x_mat.max())
logger.info("argmax  : %s", x_mat.argmax())   # flat index

# Type casting
x_int = torch.tensor([1, 2, 3])
logger.info("int64 dtype : %s", x_int.dtype)
x_f = x_int.float()
logger.info("float32 dtype: %s", x_f.dtype)

# NumPy interop
import numpy as np
arr = np.array([4.0, 5.0, 6.0])
t = torch.from_numpy(arr)             # shares memory
logger.info("from numpy : %s", t)
arr2 = t.numpy()                      # back to numpy
logger.info("to numpy   : %s", arr2)
```

---

## Exercise 2 — Autograd backward pass matching manual gradient

```python
import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Simple: L = (w*x - y)^2 / 2
# dL/dw = (w*x - y) * x

x_val, y_val, w_val = 2.0, 3.0, 1.5

# PyTorch autograd
x = torch.tensor(x_val)
y = torch.tensor(y_val)
w = torch.tensor(w_val, requires_grad=True)

L = 0.5 * (w * x - y) ** 2
L.backward()
autograd_dw = w.grad.item()
logger.info("Autograd dL/dw  : %.6f", autograd_dw)

# Manual calculation
manual_dw = (w_val * x_val - y_val) * x_val
logger.info("Manual   dL/dw  : %.6f", manual_dw)
logger.info("Match           : %s", np.isclose(autograd_dw, manual_dw))

# ---
# Two-layer example: L = sigmoid(W2 @ relu(W1 @ x + b1) + b2)
# Verify autograd vs numerical gradient

torch.manual_seed(0)
W1 = torch.randn(4, 2, requires_grad=True)
b1 = torch.zeros(4, requires_grad=True)
W2 = torch.randn(1, 4, requires_grad=True)
b2 = torch.zeros(1, requires_grad=True)
x_in = torch.tensor([0.5, -0.2])
y_target = torch.tensor([1.0])

z1 = W1 @ x_in + b1
a1 = torch.relu(z1)
z2 = W2 @ a1 + b2
a2 = torch.sigmoid(z2)
loss = torch.nn.functional.binary_cross_entropy(a2, y_target)
loss.backward()

# Numerical check for W1[0, 0]
eps = 1e-5
with torch.no_grad():
    W1_p = W1.clone(); W1_p[0, 0] += eps
    z1_p = W1_p @ x_in + b1
    a2_p = torch.sigmoid(W2 @ torch.relu(z1_p) + b2)
    loss_p = torch.nn.functional.binary_cross_entropy(a2_p, y_target).item()

    W1_m = W1.clone(); W1_m[0, 0] -= eps
    z1_m = W1_m @ x_in + b1
    a2_m = torch.sigmoid(W2 @ torch.relu(z1_m) + b2)
    loss_m = torch.nn.functional.binary_cross_entropy(a2_m, y_target).item()

num_grad = (loss_p - loss_m) / (2 * eps)
auto_grad = W1.grad[0, 0].item()
logger.info("W1[0,0] autograd : %.6f", auto_grad)
logger.info("W1[0,0] numerical: %.6f", num_grad)
logger.info("Match            : %s", abs(auto_grad - num_grad) < 1e-5)
```

---

## Exercise 3 — requires_grad exploration

```python
import logging
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# 1. Default: no grad tracking
x = torch.tensor([1.0, 2.0, 3.0])
logger.info("default requires_grad: %s", x.requires_grad)   # False

# 2. Enable tracking
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
logger.info("after requires_grad=True: %s", x.requires_grad)   # True

# 3. Leaf vs non-leaf
y = x * 2          # non-leaf (computed from leaf x)
logger.info("y.is_leaf        : %s", y.is_leaf)        # False
logger.info("y.requires_grad  : %s", y.requires_grad)  # True (inherits)

# 4. torch.no_grad() — disable for inference
with torch.no_grad():
    z = x * 3
logger.info("z.requires_grad inside no_grad : %s", z.requires_grad)   # False

# 5. detach() — cut from graph
w = x.detach()
logger.info("detached requires_grad: %s", w.requires_grad)   # False

# 6. grad accumulation — zero before each step!
loss1 = (x ** 2).sum()
loss1.backward()
logger.info("grad after loss1     : %s", x.grad)   # [2, 4, 6]

loss2 = (x ** 2).sum()
loss2.backward()
logger.info("grad accumulated     : %s", x.grad)   # [4, 8, 12]  (added!)

x.grad.zero_()
loss3 = (x ** 2).sum()
loss3.backward()
logger.info("grad after zero_()   : %s", x.grad)   # [2, 4, 6]  (correct)

# 7. in-place ops on leaf tensors with grad raise errors
try:
    x += 1   # RuntimeError
except RuntimeError as e:
    logger.info("in-place error: %s", e)
```

Key rules:
- `requires_grad=True` on leaf tensors (parameters).
- Call `optimizer.zero_grad()` (or `param.grad.zero_()`) before each backward.
- Use `torch.no_grad()` for inference to save memory and compute.
- `detach()` creates a new tensor that shares storage but is excluded from the graph.

---

## Exercise 4 — Custom autograd Function

```python
import logging
import torch
from torch.autograd import Function

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


class LeakyReLUFunction(Function):
    """Custom autograd function for Leaky ReLU.

    Forward:  f(x) = x if x > 0 else alpha * x
    Backward: f'(x) = 1 if x > 0 else alpha
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx,
                x: torch.Tensor,
                alpha: float = 0.01) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return torch.where(x > 0, x, alpha * x)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx,
                 grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = torch.where(x > 0,
                                 grad_output,
                                 alpha * grad_output)
        return grad_input, None   # None for alpha (not a tensor)


# Usage as a module
class LeakyReLU(torch.nn.Module):
    def __init__(self, alpha: float = 0.01) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return LeakyReLUFunction.apply(x, self.alpha)


# Test
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
fn = LeakyReLU(alpha=0.1)
y = fn(x)
logger.info("forward: %s", y.detach())

loss = y.sum()
loss.backward()
logger.info("gradient: %s", x.grad)
# Expected: [0.1, 0.1, 0.1, 1.0, 1.0]  (x=0 treated as positive)

# Compare with PyTorch built-in
x2 = x.detach().clone().requires_grad_(True)
y2 = torch.nn.functional.leaky_relu(x2, negative_slope=0.1)
y2.sum().backward()
logger.info("PyTorch built-in grad: %s", x2.grad)
logger.info("Match: %s", torch.allclose(x.grad, x2.grad))


# Second custom function: Clipped ReLU (demonstrates why custom functions matter)
class ClippedReLUFunction(Function):
    """ReLU clipped to [0, max_val]. Gradient = 1 inside, 0 outside."""

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx,
                x: torch.Tensor,
                max_val: float = 6.0) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.max_val = max_val
        return x.clamp(0.0, max_val)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx,
                 grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (x,) = ctx.saved_tensors
        mask = (x > 0) & (x < ctx.max_val)
        return grad_output * mask.float(), None


x3 = torch.tensor([-1.0, 1.0, 3.0, 7.0], requires_grad=True)
y3 = ClippedReLUFunction.apply(x3, 6.0)
y3.sum().backward()
logger.info("ClippedReLU forward : %s", y3.detach())   # [0, 1, 3, 6]
logger.info("ClippedReLU gradient: %s", x3.grad)       # [0, 1, 1, 0]
```
