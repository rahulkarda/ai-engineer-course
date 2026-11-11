# Solutions — Day 1: RNN Limits & Attention Motivation

## Exercise 1 — LSTM vs RNN on long sequences

```python
"""solutions/week-05/day1_ex1.py"""
import logging
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def compare_rnn_lstm(seq_lengths: list[int] = [5, 20, 50], hidden_dim: int = 32) -> None:
    """Compare final hidden-state norm for RNN and LSTM across sequence lengths."""
    torch.manual_seed(42)
    embed_dim = 8
    vocab_size = 20

    embedding = nn.Embedding(vocab_size, embed_dim)

    for seq_len in seq_lengths:
        tokens = torch.randint(0, vocab_size, (seq_len,))
        x_seq = embedding(tokens)  # (seq_len, embed_dim)

        # RNNCell
        rnn_cell = nn.RNNCell(embed_dim, hidden_dim)
        h_rnn = torch.zeros(hidden_dim)
        for t in range(seq_len):
            h_rnn = rnn_cell(x_seq[t].unsqueeze(0), h_rnn.unsqueeze(0)).squeeze(0)

        # LSTMCell
        lstm_cell = nn.LSTMCell(embed_dim, hidden_dim)
        h_lstm = torch.zeros(1, hidden_dim)
        c_lstm = torch.zeros(1, hidden_dim)
        for t in range(seq_len):
            h_lstm, c_lstm = lstm_cell(x_seq[t].unsqueeze(0), (h_lstm, c_lstm))

        log.info(
            "seq_len=%3d  RNN h_norm=%.4f  LSTM h_norm=%.4f",
            seq_len, h_rnn.norm().item(), h_lstm.squeeze(0).norm().item()
        )


if __name__ == "__main__":
    compare_rnn_lstm()
```

**Key observation:** LSTM norms degrade more slowly with sequence length than vanilla RNN norms, but both converge — the bottleneck problem remains.

---

## Exercise 2 — Preview attention function

```python
def attention_preview(
    encoder_outputs: torch.Tensor,  # (T, D)
    query: torch.Tensor,            # (D,)
) -> torch.Tensor:                  # (D,) context vector
    import torch.nn.functional as F
    # Dot product of query with each encoder output
    scores = torch.mv(encoder_outputs, query)  # (T,)
    weights = F.softmax(scores, dim=0)          # (T,) sums to 1
    context = (weights.unsqueeze(1) * encoder_outputs).sum(dim=0)  # (D,)
    return context
```

---

## Exercise 3 — O(n²) memory argument

Attention computes a score matrix `Q Kᵀ` of shape `(n, m)`. For self-attention n = m = T, so the matrix has T² entries. Each entry is a float32 (4 bytes). For T=4096 and batch=32: `32 × 4096² × 4 bytes ≈ 2 GB` just for the score matrix. This becomes a practical limitation around T=2048–4096 on consumer GPUs and motivates efficient attention variants (FlashAttention, Sparse Attention).

---

## Exercise 4 — LSTM perfection claim

**False.** An LSTM with 1024 hidden units cannot represent arbitrary-length contexts perfectly:
1. Information bottleneck: all history still fits in 1024 floats.
2. Gradients still vanish over very long sequences (mitigated but not eliminated by gating).
3. "Enough training" cannot compensate for a representational capacity ceiling — if the context requires more bits than the hidden state can store, no amount of training helps.
