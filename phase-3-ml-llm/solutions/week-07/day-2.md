# Solutions — Phase 3 · Week 07 · Day 2

## Exercise 1 — Manual BPE: 3 Merge Steps

**Initial corpus:**

```
"low low low low"   -> ["l", "o", "w"] x 4
"lower lower newer" -> ["l", "o", "w", "e", "r"] x 2  +  ["n", "e", "w", "e", "r"] x 1
```

Start by treating each character as a separate token and adding a word-boundary marker. Standard BPE convention appends `</w>` to the last character of each word.

**Initial vocabulary and word representations:**

```
"low"   (freq=6): l o w</w>
"lower" (freq=2): l o w e r</w>
"newer" (freq=1): n e w e r</w>
```

Note: "low" appears 4 times in sentence 1 and is part of "lower" x2 and "newer" x1. The word "lower" does not contain "low" as a standalone token — BPE operates on word boundaries separately.

Word frequencies:
```
l o w</w>    : 4  (from "low low low low")
l o w e r</w>: 2  (from "lower lower")
n e w e r</w>: 1  (from "newer")
```

**Step 1 — Count all adjacent pairs:**

From `l o w</w>` (count 4): pairs (l,o)=4, (o,w</w>)=4
From `l o w e r</w>` (count 2): pairs (l,o)=2, (o,w)=2, (w,e)=2, (e,r</w>)=2
From `n e w e r</w>` (count 1): pairs (n,e)=1, (e,w)=1, (w,e)=1, (e,r</w>)=1

Total pair counts:
```
(l, o)    : 4 + 2 = 6
(o, w</w>): 4
(o, w)    : 2
(w, e)    : 2 + 1 = 3
(e, r</w>): 2 + 1 = 3
(n, e)    : 1
(e, w)    : 1
```

**Most frequent pair: (l, o) with count 6. Merge: l + o -> lo**

Updated word representations:
```
lo w</w>    : 4
lo w e r</w>: 2
n e w e r</w>: 1
```

Vocabulary adds: `lo`

---

**Step 2 — Recount pairs with updated representations:**

From `lo w</w>` (count 4): pairs (lo,w</w>)=4
From `lo w e r</w>` (count 2): pairs (lo,w)=2, (w,e)=2, (e,r</w>)=2
From `n e w e r</w>` (count 1): pairs (n,e)=1, (e,w)=1, (w,e)=1, (e,r</w>)=1

Total pair counts:
```
(lo, w</w>): 4
(lo, w)    : 2
(w, e)     : 2 + 1 = 3
(e, r</w>) : 2 + 1 = 3
(n, e)     : 1
(e, w)     : 1
```

**Tie between (w,e) and (e,r</w>) both at 3. BPE breaks ties by first occurrence in pair list (implementation-dependent). Taking (w, e) — count 3. Merge: w + e -> we**

Updated word representations:
```
lo w</w>     : 4
lo we r</w>  : 2
n e we r</w> : 1
```

Vocabulary adds: `we`

---

**Step 3 — Recount pairs:**

From `lo w</w>` (count 4): pairs (lo,w</w>)=4
From `lo we r</w>` (count 2): pairs (lo,we)=2, (we,r</w>)=2
From `n e we r</w>` (count 1): pairs (n,e)=1, (e,we)=1, (we,r</w>)=1

Total pair counts:
```
(lo, w</w>) : 4
(lo, we)    : 2
(we, r</w>) : 2 + 1 = 3
(n, e)      : 1
(e, we)     : 1
```

**Most frequent pair: (lo, w</w>) with count 4. Merge: lo + w</w> -> low</w>**

Updated word representations:
```
low</w>      : 4
lo we r</w>  : 2
n e we r</w> : 1
```

Vocabulary adds: `low</w>`

---

**Final vocabulary after 3 merges:**

```
Base characters: l, o, w, e, r, n, </w>
Added by merge 1: lo
Added by merge 2: we
Added by merge 3: low</w>
```

The token `low</w>` now represents the complete word "low". The word "lower" is encoded as `lo we r</w>` and "newer" as `n e we r</w>`.

---

## Exercise 2 — Context Window Math

**Setup:**
- Context window: 4096 tokens
- Prompt length: 3800 tokens
- Response length: 500 tokens

**What happens:**

At the start of generation, the model processes the 3800-token prompt in the prefill phase. This fits within the 4096-token context window (3800 < 4096 with 296 tokens of headroom).

As the model generates tokens autoregressively, the total sequence length grows:
```
After token  1: 3800 + 1  = 3801  (within 4096)
After token  2: 3800 + 2  = 3802  (within 4096)
...
After token 296: 3800 + 296 = 4096 (exactly at limit)
After token 297: 4096 + 1  = 4097 (EXCEEDS context window)
```

At token 297, the total sequence length would exceed 4096. At this point, most implementations apply one of two strategies:

1. **Hard truncation / stop generation**: the model refuses to generate further tokens.
2. **Sliding window / eviction**: the oldest tokens are evicted from the context. The most common approach truncates from the beginning of the prompt (not from the end, since recent context is more relevant).

**How many prompt tokens are visible at the end of generation:**

If the model uses a sliding window and generates all 500 tokens, at the end:
```
Total tokens at end = 3800 (prompt) + 500 (response) = 4300
Tokens that exceed the window = 4300 - 4096 = 204
```

These 204 overflow tokens are taken from the beginning of the prompt (since we keep the most recent context). At the end of generation, the model can see:

```
Visible prompt tokens = 3800 - 204 = 3596
Visible response tokens = 500 (all response tokens are visible, being most recent)
```

**Practical implication:** The first 204 tokens of the original prompt are no longer in context when the model generates the 500th response token. If those tokens contained critical information (e.g., system instructions, initial context), the model's responses in the latter half of generation may degrade in quality. This is why RAG systems and long-document summarizers chunk inputs well below the context limit, and why frontier models at 128K+ context windows provide meaningful headroom.

---

## Exercise 3 — Tokenizer: encode/decode with Unknown Token Handling

```python
"""
Simple tokenizer with encode/decode and UNK handling.

Vocabulary is a dict mapping token string -> int id.
Encode: longest-match greedy tokenization (for subword vocabularies).
Decode: id -> token string lookup.

Dependencies: none (stdlib only)
"""

import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

UNK_TOKEN: str = "[UNK]"
UNK_ID: int = 0


class SimpleTokenizer:
    """
    Character-level and subword tokenizer with unknown token support.

    Args:
        vocab: dict mapping token string to integer id.
               Must include an [UNK] entry (id=0 by convention).

    Design notes:
    - encode() uses greedy longest-match (maxmunch): at each position, try
      the longest vocabulary token that matches the current text prefix.
      Falls back to single character; if that's also not in vocab, emits UNK.
    - decode() maps ids back to tokens. Unknown ids (not in inverse vocab) are
      represented as [UNK].
    - This is a simplified BPE-style tokenizer. Real tokenizers (HuggingFace
      tokenizers) use pre-tokenization (whitespace/punctuation splitting) before
      subword encoding, and handle byte-fallback for all Unicode codepoints.
    """

    def __init__(self, vocab: dict[str, int]) -> None:
        if UNK_TOKEN not in vocab:
            raise ValueError(
                f"Vocabulary must contain '{UNK_TOKEN}' token. "
                f"Add {{'{UNK_TOKEN}': {UNK_ID}}} to your vocab dict."
            )
        self.vocab: dict[str, int] = vocab
        # Inverse vocabulary: id -> token string
        self.inv_vocab: dict[int, str] = {v: k for k, v in vocab.items()}
        # Sort tokens by length descending for greedy longest-match
        self._sorted_tokens: list[str] = sorted(
            (t for t in vocab if t != UNK_TOKEN),
            key=len,
            reverse=True,
        )
        logger.debug("Tokenizer initialized with %d tokens", len(vocab))

    def encode(self, text: str) -> list[int]:
        """
        Encode text to a list of token ids.

        Uses greedy longest-match: at each character position, find the
        longest vocabulary token that matches. If nothing matches (not even
        a single character), emits UNK_ID.

        Args:
            text: input string to tokenize

        Returns:
            list of integer token ids
        """
        ids: list[int] = []
        pos: int = 0
        while pos < len(text):
            matched: bool = False
            # Try longest token first (greedy maxmunch)
            for token in self._sorted_tokens:
                if text[pos:pos + len(token)] == token:
                    ids.append(self.vocab[token])
                    pos += len(token)
                    matched = True
                    break
            if not matched:
                # Character not in vocabulary: emit UNK
                logger.debug("Unknown character '%s' at position %d -> [UNK]", text[pos], pos)
                ids.append(UNK_ID)
                pos += 1
        return ids

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token ids back to a string.

        Unknown ids (not in inverse vocab) are replaced with [UNK].

        Args:
            ids: list of integer token ids

        Returns:
            decoded string (tokens concatenated)
        """
        tokens: list[str] = []
        for token_id in ids:
            if token_id in self.inv_vocab:
                tokens.append(self.inv_vocab[token_id])
            else:
                logger.debug("Unknown id %d -> [UNK]", token_id)
                tokens.append(UNK_TOKEN)
        return "".join(tokens)

    def vocab_size(self) -> int:
        """Return total vocabulary size including special tokens."""
        return len(self.vocab)


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

def build_example_vocab() -> dict[str, int]:
    """
    Build a small example vocabulary with subword tokens.

    In production: vocab is loaded from a tokenizer.json file (HuggingFace)
    or a .vocab file generated during BPE training.
    """
    tokens = [
        UNK_TOKEN,   # id=0, always first
        "[PAD]",     # id=1
        "[BOS]",     # id=2
        "[EOS]",     # id=3
        # Subword tokens (simulating a BPE vocab)
        "hel", "lo", "hello", " ", "world", "!", "wor", "ld",
        "he", "l", "o", "w", "r", "d",
        # Common English characters
        "a", "b", "c", "e", "f", "g", "h", "i", "j", "k",
        "m", "n", "p", "q", "s", "t", "u", "v", "x", "y", "z",
        ",", ".", "?",
    ]
    return {token: idx for idx, token in enumerate(tokens)}


def main() -> None:
    vocab = build_example_vocab()
    tokenizer = SimpleTokenizer(vocab)

    logger.info("Vocab size: %d", tokenizer.vocab_size())

    # Test 1: normal encode/decode roundtrip
    text = "hello world"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    logger.info("Input:   '%s'", text)
    logger.info("Token ids: %s", ids)
    tokens = [tokenizer.inv_vocab.get(i, UNK_TOKEN) for i in ids]
    logger.info("Tokens:  %s", tokens)
    logger.info("Decoded: '%s'", decoded)
    logger.info("Roundtrip OK: %s", text == decoded)

    logger.info("")

    # Test 2: unknown character (emoji, non-ASCII) -> UNK
    text_with_unk = "hi!"
    ids_unk = tokenizer.encode(text_with_unk)
    decoded_unk = tokenizer.decode(ids_unk)
    logger.info("Input with partial unknowns: '%s'", text_with_unk)
    logger.info("Token ids: %s", ids_unk)
    logger.info("Decoded:   '%s'", decoded_unk)

    logger.info("")

    # Test 3: longest-match greedy - "hello" should prefer the 5-char token
    # over the sequence he+l+l+o
    text3 = "hello"
    ids3 = tokenizer.encode(text3)
    tokens3 = [tokenizer.inv_vocab.get(i, UNK_TOKEN) for i in ids3]
    logger.info("Greedy longest-match for 'hello': %s", tokens3)
    # Expected: ['hello'] not ['hel', 'l', 'o'] because 'hello' is in vocab


if __name__ == "__main__":
    main()
```

Expected output (approximate):
```
INFO: Vocab size: 47
INFO: Input:   'hello world'
INFO: Token ids: [6, 7, 8, ...]
INFO: Tokens:  ['hello', ' ', 'world']
INFO: Decoded: 'hello world'
INFO: Roundtrip OK: True
INFO: Greedy longest-match for 'hello': ['hello']
```

---

## Exercise 4 — Sinusoidal Position Encodings

**Formula:**

For position `pos` and dimension index `i` in a model of hidden size `d_model`:

```
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

Even dimensions use sine; odd dimensions use cosine. The base 10000 was chosen empirically in the original "Attention Is All You Need" paper.

The denominator term is often written as:

```
omega_i = 1 / 10000^(2i / d_model)
```

so `PE(pos, 2i) = sin(pos * omega_i)` and `PE(pos, 2i+1) = cos(pos * omega_i)`.

**Computation for d_model=512:**

For dimension 0 (i=0, even index -> sine):

```
omega_0 = 1 / 10000^(0 / 512) = 1 / 10000^0 = 1 / 1 = 1.0

PE(pos=0, dim=0) = sin(0 * 1.0) = sin(0) = 0.0

PE(pos=1, dim=0) = sin(1 * 1.0) = sin(1) = 0.84147...
```

**Summary table:**

| pos | dim | formula | value |
|-----|-----|---------|-------|
| 0   | 0   | sin(0 / 10000^0) = sin(0)   | 0.000000 |
| 1   | 0   | sin(1 / 10000^0) = sin(1)   | 0.841471 |

**Intuition:**

- At dim=0 (highest frequency, omega=1.0): two adjacent positions differ by sin(1) - sin(0) ≈ 0.84 — maximum positional discrimination at close range.
- At dim=510 (i=255, lowest frequency): omega = 1/10000^(510/512) ≈ 1/9776 ≈ 1.02e-4. Two adjacent positions differ by approximately 1.02e-4 — almost identical, but positions far apart (e.g., 0 vs 5000) differ meaningfully.
- The multi-frequency encoding ensures that every pair of positions has a unique combined encoding across all dimensions, while also providing a smooth relative position signal for attention to exploit.

**Python verification:**

```python
import math
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def sinusoidal_pe(max_len: int, d_model: int) -> np.ndarray:
    """
    Compute sinusoidal position encoding matrix.

    Returns shape (max_len, d_model).
    """
    pe = np.zeros((max_len, d_model), dtype=np.float32)
    positions = np.arange(max_len)[:, np.newaxis]       # (max_len, 1)
    # dimension indices for even positions: i = 0, 1, 2, ..., d_model/2 - 1
    i_indices = np.arange(0, d_model, 2)               # (d_model/2,)
    # omega_i = 1 / 10000^(2i / d_model)
    omega = 1.0 / np.power(10000.0, i_indices / d_model)  # (d_model/2,)
    angles = positions * omega                          # (max_len, d_model/2)
    pe[:, 0::2] = np.sin(angles)    # even dims: sine
    pe[:, 1::2] = np.cos(angles)    # odd dims:  cosine
    return pe


def main() -> None:
    d_model = 512
    pe = sinusoidal_pe(max_len=10, d_model=d_model)

    logger.info("PE(pos=0, dim=0) = %.6f  [expected: 0.000000]", pe[0, 0])
    logger.info("PE(pos=1, dim=0) = %.6f  [expected: 0.841471]", pe[1, 0])
    logger.info("PE(pos=0, dim=1) = %.6f  [expected: 1.000000 (cos(0))]", pe[0, 1])
    logger.info("PE(pos=1, dim=1) = %.6f  [expected: cos(1) = 0.540302]", pe[1, 1])

    # Verify: each position has a unique encoding
    # Dot products of normalized encodings should decrease with distance
    norms = np.linalg.norm(pe, axis=1, keepdims=True)
    pe_norm = pe / (norms + 1e-8)
    sim_0_1 = float(pe_norm[0] @ pe_norm[1])
    sim_0_5 = float(pe_norm[0] @ pe_norm[5])
    sim_0_9 = float(pe_norm[0] @ pe_norm[9])
    logger.info("Cosine similarity pos(0,1)=%.3f, pos(0,5)=%.3f, pos(0,9)=%.3f",
                sim_0_1, sim_0_5, sim_0_9)
    logger.info("Similarity decreases with distance: %s",
                sim_0_1 > sim_0_5 > sim_0_9)


if __name__ == "__main__":
    main()
```

Expected output:
```
INFO: PE(pos=0, dim=0) = 0.000000  [expected: 0.000000]
INFO: PE(pos=1, dim=0) = 0.841471  [expected: 0.841471]
INFO: PE(pos=0, dim=1) = 1.000000  [expected: 1.000000 (cos(0))]
INFO: PE(pos=1, dim=1) = 0.540302  [expected: cos(1) = 0.540302]
```
