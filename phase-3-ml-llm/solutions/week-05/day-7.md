# Solutions — Phase 3 · Week 05 · Day 7 — Project Submission Day

## This is a project-only day

Day 7 of Week 05 has no new exercises. The session is dedicated to finalising the attention
library project and sitting the week assessment.

See `solutions/week-05/project.md` for the complete reference implementation of all six
components in `attention_lib.py` and the full test suite.

---

## Project submission self-check

Work through every item before marking the project done.

| # | Check | How to verify |
|---|-------|---------------|
| 1 | `scaled_dot_product_attention` returns correct shapes | `output.shape == (..., n, d_v)` and `weights.shape == (..., n, m)` |
| 2 | Attention weights sum to 1 along key dimension | `weights.sum(dim=-1).allclose(ones)` |
| 3 | Causal mask zeros upper triangle in weights | After applying mask, `weights[..., i, j] == 0` for all `j > i` |
| 4 | `MultiHeadAttention` output shape | `(B, n, d_model)` |
| 5 | `MultiHeadAttention` weights shape | `(B, num_heads, n, m)` |
| 6 | Cross-attention works (`n != m`) | Pass `query` of seq len 3, `key`/`value` of seq len 7 — no error |
| 7 | `SinusoidalPositionalEncoding` is deterministic | Two instances with same args produce identical PE |
| 8 | All T position vectors are unique | `pe[i] != pe[j]` for `i != j` |
| 9 | `FeedForward` uses GELU and preserves shape | Input shape == output shape; `GELU` visible in module |
| 10 | `TransformerBlock` is Pre-LN with two residual connections | `x = x + attn(ln1(x))`, `x = x + ffn(ln2(x))` |
| 11 | `causal_mask` upper triangle is True | `mask[i, j] == True` for `j > i`; diagonal and below are False |
| 12 | All 12 required tests pass | `pytest tests/test_attention_lib.py -v` exits 0 |
| 13 | `.backward()` propagates through `TinyGPT` | All parameter gradients non-None after one forward + backward |
| 14 | No `print` statements | `grep -r "print(" week-05/` returns nothing |
| 15 | All function signatures type-annotated | Confirmed by reading each signature |

---

## Reflection — model answers

**Hardest concept this week**: Most learners find the head-split / head-merge mechanics the
trickiest. The key insight: we do NOT process heads in a Python loop. We reshape QKV from
`(B, T, d_model)` to `(B, num_heads, T, d_head)` and run a single batched matrix multiply.
The reshape and transpose operations are the hard part.

**Most common implementation mistake**: Forgetting to transpose correctly.
`q @ k.transpose(-2, -1)` computes `(B, H, T_q, d_k) @ (B, H, d_k, T_k) = (B, H, T_q, T_k)`.
Using `k.T` instead transposes ALL dimensions (including batch/head), producing the wrong shape.

**Pre-LN vs Post-LN**: In Post-LN (original transformer) the LayerNorm sits after the residual
addition. Pre-LN places it before the sub-layer: `x = x + sublayer(LN(x))`. Pre-LN trains
more stably because the residual stream stays in a well-conditioned range regardless of layer depth.

---

## Day 6 integration test — expected log output

After completing the project, running `python day6_integration.py` should log:

```
INFO TinyGPT forward pass OK — logits shape: torch.Size([2, 16, 65])
INFO Loss: <some finite value>
INFO .backward() OK — all gradients non-None
```

If you see a shape error, check that `causal_mask` is registered as a buffer (not a parameter)
and that `mask[:T, :T]` is correctly sliced before `masked_fill`.

---

## Assessment

See `assessment.md` for the graded assessment (complete without opening any lesson file).
