# Solutions — Day 4: Positional Encodings

## Exercise 1 — First 8 PE dimensions for positions 0–31

```python
"""Print sinusoidal PE values for the first 8 dims, positions 0-31."""
import math
import torch

d_model = 64
positions = torch.arange(32).float()

for dim_pair in range(4):  # 4 pairs → dims 0,1,2,3,4,5,6,7
    freq = 1.0 / (10000 ** (2 * dim_pair / d_model))
    sin_vals = torch.sin(positions * freq)
    cos_vals = torch.cos(positions * freq)
    print(f"\nDim {2*dim_pair} (sin, freq={freq:.6f}):")
    print(" ".join(f"{v:.2f}" for v in sin_vals[:16].tolist()))
    print(f"Dim {2*dim_pair+1} (cos):")
    print(" ".join(f"{v:.2f}" for v in cos_vals[:16].tolist()))
```

**Pattern:** Lower-indexed dimensions oscillate rapidly (high frequency — captures local position). Higher-indexed dimensions oscillate slowly (low frequency — captures global position). Together they provide a unique binary-like encoding for each position, analogous to how a clock has fast seconds and slow hours hands.

---

## Exercise 2 — Linear relation PE(t+k) = M · PE(t)

For d=4 and k=1, show `PE(t+1) = M · PE(t)`:

```
PE(t, 0) = sin(t · ω₀),  PE(t, 1) = cos(t · ω₀)
PE(t, 2) = sin(t · ω₁),  PE(t, 3) = cos(t · ω₁)

where ω₀ = 1/10000^(0/4) = 1,  ω₁ = 1/10000^(2/4)
```

The rotation identity: `sin(t + k) = sin(t)cos(k) + cos(t)sin(k)`

So the 2×2 block for each frequency pair:
```
[PE(t+k, 2i)  ]   [ cos(ω_i · k)   sin(ω_i · k) ] [PE(t, 2i)  ]
[PE(t+k, 2i+1)] = [-sin(ω_i · k)   cos(ω_i · k) ] [PE(t, 2i+1)]
```

M is block-diagonal with 2×2 rotation matrices. It is independent of t — only depends on k.

```python
import torch, math

def get_pe(t: int, d: int = 4) -> torch.Tensor:
    pe = torch.zeros(d)
    for i in range(d // 2):
        freq = 1.0 / (10000 ** (2 * i / d))
        pe[2 * i] = math.sin(t * freq)
        pe[2 * i + 1] = math.cos(t * freq)
    return pe

k = 1
for t in range(5):
    pe_t = get_pe(t)
    pe_t1 = get_pe(t + k)
    # Build M via least squares (should be exact since it's a rotation)
    # Just verify the rotation property for dim 0,1:
    w0 = 1.0
    M_block = torch.tensor([
        [math.cos(w0 * k),  math.sin(w0 * k)],
        [-math.sin(w0 * k), math.cos(w0 * k)],
    ])
    rotated = M_block @ pe_t[:2]
    diff = (rotated - pe_t1[:2]).abs().max().item()
    print(f"t={t}: max diff = {diff:.8f}")
```

---

## Exercise 3 — RelativePositionBias

```python
import torch
import torch.nn as nn


class RelativePositionBias(nn.Module):
    """
    Learnable scalar bias per relative distance, added to attention scores.
    For positions i and j: bias = embedding[clip(j - i, -max_dist, max_dist) + max_dist]
    """

    def __init__(self, max_dist: int = 16) -> None:
        super().__init__()
        self.max_dist = max_dist
        # Table size: 2*max_dist + 1 (negative, zero, positive offsets)
        self.embedding = nn.Embedding(2 * max_dist + 1, 1)

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Returns bias matrix of shape (seq_len, seq_len).
        bias[i, j] = learned scalar for relative distance j-i.
        """
        i = torch.arange(seq_len).unsqueeze(1)   # (T, 1)
        j = torch.arange(seq_len).unsqueeze(0)   # (1, T)
        rel = (j - i).clamp(-self.max_dist, self.max_dist) + self.max_dist
        return self.embedding(rel).squeeze(-1)    # (T, T)


# Shape test
def test_relative_position_bias_shape():
    T = 10
    rpb = RelativePositionBias(max_dist=8)
    bias = rpb(T)
    assert bias.shape == (T, T), f"Expected ({T}, {T}), got {bias.shape}"
    # Diagonal should use index max_dist (zero offset)
    assert bias[0, 0].item() == rpb.embedding(torch.tensor([8])).item()
```

---

## Exercise 4 — Graceful handling beyond max_seq_len

If `T > max_seq_len`, `torch.arange(T)` produces position indices that exceed the embedding table size, causing an `IndexError: index out of range`.

**Graceful handling options:**

1. **Truncate (safest):** Only add PE to the first `max_seq_len` positions; the remainder get zero PE. Simple but loses all position information for the tail.

2. **Interpolate:** Linearly rescale position indices: `positions = positions * (max_seq_len - 1) / (T - 1)`. Degrades gracefully.

3. **Raise with a clear error message** at model definition time: "Sequence length T=%d exceeds max_seq_len=%d. Re-initialise with a larger max_seq_len."

4. **Switch to sinusoidal PE** which has no hard ceiling — a good design argument for sinusoidal at inference time.

For production, option 3 (fail loudly) is preferred during development; option 1 or sinusoidal PE for deployment flexibility.
