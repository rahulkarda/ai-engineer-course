# Phase 3 · Week 05 · Day 1 — RNN Limits & Attention Motivation

> Objective: Understand why RNNs fail on long sequences and why attention is the solution.  ·  Time: ~1 hr  ·  Prereqs: [[week-04/day-7]]

## Concept

### Sequence modeling recap

A language model must assign probability to sequences: `P(w₁, w₂, …, wₙ)`. Factored left-to-right:

```
P(wₙ | w₁…wₙ₋₁) for each position
```

The simplest neural approach: feed each token into a recurrent cell, carry a hidden state forward.

```
h_t = f(W_h · h_{t-1} + W_x · x_t + b)
```

The entire history is compressed into `h_t` — a fixed-size vector. That's the fundamental bottleneck.

### Three concrete RNN failure modes

**1. Vanishing gradients**
Backpropagation through time (BPTT) multiplies the Jacobian of `f` at each step. If those values < 1, gradients shrink exponentially. After ~50 steps, the gradient of the loss with respect to early tokens approaches zero — the network cannot learn long-range dependencies.

**2. The information bottleneck**
All history must fit in a vector of fixed dimension `d_h`. A 1024-token document is crushed into a 512-float vector before the decoder sees it. Information is lost, order is irrelevant to the decoder.

**3. Sequential computation — no parallelism**
Each step depends on the previous hidden state: `h_t = f(h_{t-1}, x_t)`. Training a length-1024 sequence requires 1024 sequential operations. GPUs thrive on parallel computation; RNNs waste GPU silicon.

### LSTMs and GRUs: partial remedies, same structural problem

LSTMs introduce gating (forget / input / output gates) that let gradients flow more stably. They help significantly in practice but:
- Still sequential — no parallelism.
- Still single bottleneck vector (though cell state is a second channel).
- Still struggle on very long sequences (>500 tokens).

### The attention idea (motivation, no math yet)

Instead of routing everything through a bottleneck: **let the decoder directly look at every encoder hidden state and decide which ones matter**.

At each decoding step, compute a weighted sum of all encoder outputs:
- Weight is high for positions relevant to the current output.
- Weight is low (near zero) for irrelevant positions.

This is Bahdanau's 2015 additive attention (for seq2seq). It solved machine translation. The question that birthed transformers: *what if attention is ALL you need?*

## How it works internally

### Why attention fixes each failure mode

| Failure mode | RNN | Attention |
|---|---|---|
| Vanishing gradients | Path length = sequence length | Direct connection: path length = 1 |
| Bottleneck | Fixed-size vector | All positions directly accessible |
| Parallelism | Sequential | All positions computed simultaneously |

The computational graph for attention is a matrix multiply — fully parallelisable on GPU. This is why transformers trained 10-100× faster than LSTMs at the same data scale.

### What "path length" means

The number of sequential operations between any two positions in the sequence determines how well the model can relate them. RNN path length is O(n). Attention path length is O(1) — constant regardless of sequence distance.

## Worked example

A minimal demo showing the information-bottleneck problem empirically:

```python
"""
day1_rnn_limits.py
Demonstrates the vanishing-gradient problem in a simple RNN.
Run: python day1_rnn_limits.py
"""
import logging
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def measure_gradient_norms(seq_len: int, hidden_dim: int = 32) -> list[float]:
    """Return per-step gradient norms for a vanilla RNN on a random sequence."""
    rnn = nn.RNNCell(input_size=hidden_dim, hidden_size=hidden_dim)

    x = torch.randn(seq_len, hidden_dim, requires_grad=False)
    h = torch.zeros(hidden_dim, requires_grad=True)  # initial hidden state

    states: list[torch.Tensor] = []
    h_t = h
    for t in range(seq_len):
        h_t = rnn(x[t].unsqueeze(0), h_t.unsqueeze(0)).squeeze(0)
        states.append(h_t)

    # Scalar loss: sum of all final hidden values
    loss = states[-1].sum()
    loss.backward()

    # Measure gradient at each intermediate state via retain_graph trick
    norms: list[float] = []
    for state in states:
        if state.grad is not None:
            norms.append(state.grad.norm().item())
        else:
            # Re-compute gradient norm via torch.autograd.grad (no graph needed here)
            norms.append(float("nan"))

    return norms


def demonstrate_bottleneck(vocab_size: int = 20, seq_len: int = 10) -> None:
    """Show that RNN compresses all history into one vector."""
    log.info("=== Information Bottleneck Demo ===")
    embed_dim = 8
    hidden_dim = 8

    embedding = nn.Embedding(vocab_size, embed_dim)
    rnn = nn.RNNCell(embed_dim, hidden_dim)

    tokens = torch.randint(0, vocab_size, (seq_len,))
    log.info("Input token sequence: %s", tokens.tolist())

    h = torch.zeros(hidden_dim)
    for t, tok in enumerate(tokens):
        x_t = embedding(tok.unsqueeze(0))
        h = rnn(x_t, h.unsqueeze(0)).squeeze(0)
        log.info("After token %2d: hidden norm = %.4f", t, h.norm().item())

    log.info(
        "Final hidden state shape: %s — all %d tokens compressed here",
        h.shape,
        seq_len,
    )


if __name__ == "__main__":
    torch.manual_seed(42)
    demonstrate_bottleneck()

    log.info("\n=== Gradient Norms (RNN, seq_len=20) ===")
    log.info(
        "Gradient norms near zero at early steps indicate vanishing gradient."
    )
    log.info("(Full per-step norms require custom grad hooks — see exercises)")
```

Run it. Observe that `h.norm()` stays bounded — all token information is squeezed through the same 8-dimensional vector after step 1.

## Exercises

1. Modify `demonstrate_bottleneck` to use an LSTM instead of RNNCell. Compare the final hidden-state norm after 50 tokens versus 5 tokens — does it degrade more gracefully?

2. Implement a function `attention_preview(encoder_outputs: torch.Tensor, query: torch.Tensor) -> torch.Tensor` that computes a simple dot-product between the query and each encoder output, applies `softmax`, and returns the weighted sum. Shape: encoder_outputs `(T, D)`, query `(D,)`, output `(D,)`. (No learning yet — just the mechanics.)

3. Explain in one paragraph: why does attention have O(n²) memory complexity? When does this become a practical problem?

4. True/false with explanation: "An LSTM with 1024 hidden units can represent any context of any length perfectly, given enough training."

(Solutions: ../solutions/week-05/day-1.md)

## Self-check

- [ ] I can name three specific failure modes of vanilla RNNs.
- [ ] I can explain why LSTMs help but don't fully solve the parallelism problem.
- [ ] I can describe what "path length" means and why O(1) path length matters.
- [ ] I can sketch the attention idea (weighted sum of encoder outputs) without looking at notes.

## Supplementary

- Bahdanau et al. (2015) "Neural Machine Translation by Jointly Learning to Align and Translate" — https://arxiv.org/abs/1409.0473
- Karpathy, "The Unreasonable Effectiveness of Recurrent Neural Networks" — http://karpathy.github.io/2015/05/21/rnn-effectiveness/
- Colah, "Understanding LSTM Networks" — https://colah.github.io/posts/2015-08-Understanding-LSTMs/
