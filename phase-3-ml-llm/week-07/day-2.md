# Phase 3 · Week 07 · Day 2 — Tokenization Recap & Context Windows

> Objective: Understand how BPE tokenization works and why context windows are finite and costly.  ·  Time: ~1 hr  ·  Prereqs: [[week-07/day-1]]

## Concept

**Tokenization** converts raw text into integer IDs that a model can process. The model never sees characters or words directly — only token IDs and their embeddings.

### Byte-Pair Encoding (BPE)

BPE builds a vocabulary from a corpus by iteratively merging the most frequent adjacent token pair:

1. Start with a character-level vocabulary (or byte-level for robustness to any unicode).
2. Count all adjacent pair frequencies across the corpus.
3. Merge the most frequent pair into a new token. Add it to the vocabulary.
4. Repeat until the vocabulary reaches the target size (e.g., 50,257 for GPT-2, 100,256 for GPT-4o).

At inference, the learned merge rules are applied in order to encode new text. Unknown characters can always fall back to byte-level tokens, so BPE with a byte fallback has zero out-of-vocabulary tokens.

Example merge sequence:
```
Corpus: "low lower lowest"
Step 0 vocab: {l, o, w, e, r, s, t, ' '}
Pair 'l','o' is most frequent → merge to 'lo'
Pair 'lo','w' is most frequent → merge to 'low'
...
```

### SentencePiece and WordPiece

- **WordPiece** (BERT): similar to BPE but uses a likelihood-based merge criterion (maximizes LM likelihood of the corpus) instead of raw frequency. Produces subwords prefixed with `##` for continuation.
- **SentencePiece** (T5, LLaMA): treats whitespace as an ordinary character; trains BPE or unigram LM directly on raw bytes/chars without pre-tokenization. Language-agnostic — works for Chinese, Arabic, etc.

### Context Window

The context window is the maximum number of tokens a transformer can process in a single forward pass. Common sizes:
- GPT-2: 1,024 tokens
- GPT-3: 4,096 tokens
- LLaMA-3: 8,192 tokens (base), 128k with rope scaling
- Claude 3.5 Sonnet: 200k tokens

**Why is it limited?**
Attention is the bottleneck. In standard full attention, every token attends to every other token, making the compute cost O(n^2) and memory cost O(n^2) in the sequence length n. Doubling the context quadruples compute and memory. Architectural variants (flash attention, sparse attention, linear attention) reduce the constant but the scaling remains the dominant constraint.

**Consequences of exceeding the context window:**
- Hard truncation: the model cannot process the input at all.
- "Lost in the middle" effect: even within the context window, models often underweight information in the middle of a long context compared to the beginning or end.

### Position Encodings

Transformers are permutation-equivariant — without position information, "cat sat mat" and "mat sat cat" are identical. Position encodings inject order.

**Absolute sinusoidal (Vaswani et al., 2017):**
Each position index i maps to a fixed vector of sine/cosine values at different frequencies. The encoding is added to the token embedding. No parameters to learn; generalizes reasonably to positions seen during training.

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Learned absolute:**
Each position index has a learned embedding. Simple and effective but cannot generalize beyond the maximum training length.

**RoPE (Rotary Position Embedding):**
Instead of adding positional vectors to token embeddings, RoPE multiplies the query and key vectors by a rotation matrix that depends on position. The dot product Q·K then naturally encodes *relative* position — the angle between Q at position m and K at position n depends only on (m - n). This allows better length generalization and is the basis for techniques like YaRN and LongRoPE that extend context at inference time.

## How it works internally

**Why BPE compression matters:**
A model with a 4k token context processes more raw text if tokens are longer. "tokenization" as one token vs. "t","o","k","e","n","i","z","a","t","i","o","n" as 12 tokens — the vocabulary controls the information density per token.

**Why RoPE generalizes:**
Standard absolute encodings add a position-specific bias to embeddings. The model memorizes patterns for "position 1," "position 2," etc. RoPE encodes relative distance geometrically (rotation). Attention scores depend on cos((m-n)θ) for the relative position (m-n), which the model can extrapolate more smoothly to unseen distances.

**Tokenizer vocabulary affects model behavior:**
Number representations are tokenized character by character in many tokenizers. "1234567" becomes ["1","234","567"] or similar — arithmetic is harder because the model cannot operate on place values directly. Specialized tokenizers for code and math split differently.

## Worked example

```python
"""
Minimal BPE tokenizer implemented from scratch in pure Python.
Demonstrates corpus-level vocabulary building and encoding.
Run: python day-2-example.py
Dependencies: none (pure Python 3.12+)
"""

import logging
import re
from collections import Counter, defaultdict
from typing import Iterator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Corpus pre-processing
# ---------------------------------------------------------------------------

def pretokenize(text: str) -> list[str]:
    """
    Split text into words; represent each word as space-separated characters
    with a special end-of-word marker '</w>'.
    This matches the original BPE paper convention.
    """
    words = re.findall(r"\b\w+\b", text.lower())
    return [" ".join(list(word) + ["</w>"]) for word in words]


def build_vocab(pretokenized: list[str]) -> Counter:
    """Count word-form frequencies in the pre-tokenized corpus."""
    return Counter(pretokenized)


# ---------------------------------------------------------------------------
# Core BPE algorithm
# ---------------------------------------------------------------------------

def get_pair_counts(vocab: Counter) -> Counter:
    """Count frequency of every adjacent symbol pair across all word forms."""
    pairs: Counter = Counter()
    for word_form, freq in vocab.items():
        symbols = word_form.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def merge_pair(vocab: Counter, pair: tuple[str, str]) -> Counter:
    """
    Merge the given pair in all word forms.
    Returns a new Counter with the merged vocabulary.
    """
    merged_token = "".join(pair)
    # Build a regex that matches the exact pair (with space between, bounded)
    bigram = re.escape(" ".join(pair))
    pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    new_vocab: Counter = Counter()
    for word_form, freq in vocab.items():
        new_form = pattern.sub(merged_token, word_form)
        new_vocab[new_form] += freq
    return new_vocab


def train_bpe(
    corpus: str,
    num_merges: int,
) -> tuple[list[tuple[str, str]], Counter]:
    """
    Train BPE on corpus for num_merges steps.
    Returns: (list of merge rules, final vocab)
    """
    pretokenized = pretokenize(corpus)
    vocab = build_vocab(pretokenized)

    logger.info("Initial vocabulary (%d word forms):", len(vocab))
    for form, cnt in vocab.most_common(5):
        logger.info("  %r x%d", form, cnt)

    merge_rules: list[tuple[str, str]] = []

    for step in range(num_merges):
        pair_counts = get_pair_counts(vocab)
        if not pair_counts:
            logger.info("No more pairs to merge at step %d", step)
            break
        best_pair = pair_counts.most_common(1)[0][0]
        freq = pair_counts[best_pair]
        merge_rules.append(best_pair)
        vocab = merge_pair(vocab, best_pair)
        logger.info(
            "Step %2d: merge %r + %r -> %r (freq=%d)",
            step + 1, best_pair[0], best_pair[1], "".join(best_pair), freq,
        )

    return merge_rules, vocab


# ---------------------------------------------------------------------------
# BPE encoding for new text
# ---------------------------------------------------------------------------

def apply_bpe(word: str, merge_rules: list[tuple[str, str]]) -> list[str]:
    """
    Encode a single word using trained BPE merge rules (in order).
    Returns list of subword tokens.
    """
    # Start as character sequence with </w> end marker
    symbols = list(word) + ["</w>"]
    for pair in merge_rules:
        merged = "".join(pair)
        i = 0
        new_symbols: list[str] = []
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                new_symbols.append(merged)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        symbols = new_symbols
    return symbols


def encode(text: str, merge_rules: list[tuple[str, str]]) -> list[str]:
    """Tokenize a text string using trained BPE merge rules."""
    tokens: list[str] = []
    for word in re.findall(r"\b\w+\b", text.lower()):
        tokens.extend(apply_bpe(word, merge_rules))
    return tokens


# ---------------------------------------------------------------------------
# Context window cost demonstration
# ---------------------------------------------------------------------------

def context_window_cost(seq_len: int, d_model: int, bytes_per_float: int = 2) -> dict[str, int]:
    """
    Compute approximate memory costs for a transformer at a given sequence length.
    attention_matrix: O(n^2) per layer per head — the core bottleneck.
    kv_cache: 2 * n_layers * seq_len * d_model (computed separately in day-3)
    """
    attention_matrix_bytes = seq_len * seq_len * bytes_per_float
    return {
        "seq_len": seq_len,
        "attention_matrix_bytes": attention_matrix_bytes,
        "attention_matrix_MB": attention_matrix_bytes // (1024 ** 2),
    }


# ---------------------------------------------------------------------------
# RoPE angle demonstration
# ---------------------------------------------------------------------------

def rope_angle(pos: int, dim_index: int, d_model: int = 64) -> tuple[float, float]:
    """
    Compute the RoPE rotation angle for a given position and dimension pair.
    In practice this is applied as a 2D rotation matrix to (q, k) vectors.
    """
    import math
    theta = 1.0 / (10000 ** (2 * dim_index / d_model))
    return math.cos(pos * theta), math.sin(pos * theta)


def main() -> None:
    corpus = (
        "low lower lowest "
        "new newer newest "
        "old older oldest "
        "low lower low newer newer newest"
    )

    logger.info("=== Training BPE on small corpus ===")
    merge_rules, final_vocab = train_bpe(corpus, num_merges=10)

    logger.info("\nFinal vocabulary:")
    for form, cnt in final_vocab.most_common():
        logger.info("  %r x%d", form, cnt)

    logger.info("\n=== Encoding new words ===")
    test_words = ["low", "lower", "newest", "unknown"]
    for word in test_words:
        tokens = apply_bpe(word, merge_rules)
        logger.info("  %r -> %s", word, tokens)

    logger.info("\n=== Context Window Memory Cost (O(n^2)) ===")
    for n in [512, 1024, 4096, 16384, 128000]:
        cost = context_window_cost(n, d_model=4096)
        logger.info(
            "  n=%7d | attention matrix = %6d MB",
            cost["seq_len"], cost["attention_matrix_MB"],
        )

    logger.info("\n=== RoPE rotation angles (pos 0 vs pos 100, dim 0) ===")
    for pos in [0, 1, 100, 1000]:
        cos_val, sin_val = rope_angle(pos, dim_index=0, d_model=64)
        logger.info("  pos=%5d | cos=%.4f, sin=%.4f", pos, cos_val, sin_val)


if __name__ == "__main__":
    main()
```

Expected output (abridged):
```
INFO: Step  1: merge 'e' + 'r' -> 'er' (freq=6)
INFO: Step  2: merge 'er' + '</w>' -> 'er</w>' (freq=6)
...
INFO:   'low' -> ['low</w>']
INFO:   'lower' -> ['low', 'er</w>']
INFO:   'newest' -> ['new', 'est</w>'] (if 'est' merged)
```

Note: pure Python, no dependencies. Run with `python day-2-example.py`.

## Exercises

1. **BPE by hand**: Take the corpus `"aababcabc"`. Treat each character as an initial token. Manually run 3 BPE merge steps, showing the pair counts at each step and the vocabulary after each merge.

2. **Context window math**: A model has a context window of 8,192 tokens. The average English word is 4.7 characters and tokenizes to approximately 1.3 tokens. (a) How many words fit in the context window? (b) A novel chapter averages 4,000 words. How many full chapters fit? (c) If context window is doubled to 16,384, how does attention memory cost change (in multiples)?

3. **Tokenizer analysis**: Without running any code, reason about why the string `"1 + 1 = 2"` might tokenize differently from `"1+1=2"` in a BPE tokenizer trained on web text. What implications does this have for arithmetic tasks?

(Solutions: ../solutions/week-07/day-2.md)

## Self-check

- [ ] I can describe the three steps of one BPE merge iteration: count pairs, find maximum, apply merge.
- [ ] I can explain why attention is O(n^2) in sequence length and what this means for doubling context size.
- [ ] I can distinguish absolute sinusoidal, learned absolute, and RoPE positional encodings by their mechanism and generalization properties.
- [ ] I can implement a minimal BPE training loop in Python without reference.

## Supplementary reading

- Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units" (original BPE paper): https://arxiv.org/abs/1508.07909
- Kudo and Richardson, "SentencePiece": https://arxiv.org/abs/1808.06226
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding": https://arxiv.org/abs/2104.09864
- Transformer positional encoding visual explainer: https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
