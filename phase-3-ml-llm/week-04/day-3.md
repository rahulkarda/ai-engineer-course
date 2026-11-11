# Phase 3 · Week 04 · Day 3 — Tokenisation Overview & Implementing Tiny BPE

> Objective: Understand tokenisation strategies and implement Byte-Pair Encoding from scratch.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

## Concept

**Tokenisation** converts a raw string into a sequence of discrete units (tokens) that a model processes. It sits at the absolute boundary between text and numbers.

Four strategies on a spectrum from coarse to fine:

| Strategy | Token unit | Vocab size | OOV? | Examples |
|---|---|---|---|---|
| Whitespace | Full word | Large (>50k) | Yes | early NLTK |
| Character | Single char | ~256 | Never | char-RNNs |
| Subword | Sub-word piece | ~32–64k | Never | GPT / BERT |
| Byte | Raw byte | 256 | Never | GPT-4 fallback |

**Subword tokenisation** is the modern default because it:
- Eliminates out-of-vocabulary (OOV) by splitting rare words into known pieces.
- Keeps common words as single tokens (efficiency).
- Represents morphology: `un + happy + ness` → learnable components.

**Byte-Pair Encoding (BPE)** is the most widely used subword algorithm (GPT-2, GPT-3, GPT-4, RoBERTa, many others).

## How it works internally

### BPE algorithm

BPE was originally a data-compression algorithm; Sennrich et al. (2016) adapted it for NLP.

**Training phase:**

1. Start with a character-level vocabulary. Represent each word as a sequence of characters with a special end-of-word marker: `"low" → ['l', 'o', 'w', '</w>']`.
2. Count all adjacent symbol pairs in the corpus.
3. Merge the most frequent pair into a new symbol: `('l', 'o') → 'lo'`.
4. Repeat for `num_merges` iterations.

After training, the merge rules are saved as an ordered list.

**Inference phase:**

Apply merge rules in training order to tokenise new text:
1. Split word into characters.
2. Apply each merge rule left-to-right if the pair exists.
3. Final segments are the token sequence.

**Why frequency-first greedy works:**  
The most frequent pair contributes most to encoding efficiency (fewer tokens for the same corpus). Repeated merging effectively learns the most compressive vocabulary for the training data.

### Vocabulary sizes in practice

| Model | Tokeniser | Vocab size |
|---|---|---|
| GPT-2 | BPE | 50,257 |
| GPT-4 | tiktoken BPE | 100,277 |
| BERT | WordPiece | 30,522 |
| LLaMA 3 | SentencePiece BPE | 128,256 |

## Worked example

```python
"""day3_tiny_bpe.py — Byte-Pair Encoding implemented from scratch."""

import logging
import re
from collections import Counter, defaultdict
from typing import NamedTuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

Word = tuple[str, ...]          # sequence of symbols, e.g. ('l', 'o', 'w', '</w>')
Vocab = dict[Word, int]         # word -> frequency
MergeRule = tuple[str, str]     # pair to merge, e.g. ('e', 's</w>')
MergeList = list[MergeRule]


# ---------------------------------------------------------------------------
# 1. Build initial character-level vocabulary
# ---------------------------------------------------------------------------

END_OF_WORD = "</w>"


def build_initial_vocab(corpus: list[str]) -> Vocab:
    """
    Tokenise corpus into words, split each word into characters,
    append end-of-word marker, and count frequencies.
    """
    word_freqs: Counter[str] = Counter()
    for sentence in corpus:
        for word in sentence.lower().split():
            word = re.sub(r"[^a-z']", "", word)
            if word:
                word_freqs[word] += 1

    vocab: Vocab = {}
    for word, freq in word_freqs.items():
        # split into characters + end marker on last char
        chars = tuple(list(word[:-1]) + [word[-1] + END_OF_WORD])
        vocab[chars] = freq
    return vocab


# ---------------------------------------------------------------------------
# 2. Count adjacent symbol pairs
# ---------------------------------------------------------------------------

def get_pair_counts(vocab: Vocab) -> Counter[MergeRule]:
    """Count frequency of every adjacent symbol pair across all words."""
    counts: Counter[MergeRule] = Counter()
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            counts[(word[i], word[i + 1])] += freq
    return counts


# ---------------------------------------------------------------------------
# 3. Apply a single merge rule
# ---------------------------------------------------------------------------

def apply_merge(vocab: Vocab, pair: MergeRule) -> Vocab:
    """
    Merge all occurrences of `pair` in every word.
    ('l', 'o') → 'lo'
    """
    merged = "".join(pair)
    new_vocab: Vocab = {}
    for word, freq in vocab.items():
        new_word: list[str] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_vocab[tuple(new_word)] = freq
    return new_vocab


# ---------------------------------------------------------------------------
# 4. BPE training loop
# ---------------------------------------------------------------------------

class BPETokenizer:
    """Tiny BPE tokeniser — train on a corpus, encode new text."""

    def __init__(self, num_merges: int = 20) -> None:
        self.num_merges = num_merges
        self.merges: MergeList = []
        self.vocab: Vocab = {}

    def fit(self, corpus: list[str]) -> "BPETokenizer":
        """Learn BPE merge rules from corpus."""
        self.vocab = build_initial_vocab(corpus)
        logger.info(
            "Initial vocab: %d unique words, %d unique symbols",
            len(self.vocab),
            len({sym for word in self.vocab for sym in word}),
        )

        for step in range(self.num_merges):
            pair_counts = get_pair_counts(self.vocab)
            if not pair_counts:
                logger.info("No more pairs to merge at step %d", step)
                break
            best_pair = pair_counts.most_common(1)[0][0]
            self.merges.append(best_pair)
            self.vocab = apply_merge(self.vocab, best_pair)
            logger.debug(
                "Step %02d: merged %s + %s  (freq %d)",
                step + 1, best_pair[0], best_pair[1],
                pair_counts[best_pair],
            )

        n_symbols = len({sym for word in self.vocab for sym in word})
        logger.info("Training complete. Merge rules: %d, Symbol vocab: %d", len(self.merges), n_symbols)
        return self

    def _tokenise_word(self, word: str) -> list[str]:
        """Apply learned merge rules to a single word."""
        # Initial split into characters + end marker
        symbols: list[str] = list(word[:-1]) + [word[-1] + END_OF_WORD]
        for pair in self.merges:
            i = 0
            new_symbols: list[str] = []
            while i < len(symbols):
                if (
                    i < len(symbols) - 1
                    and symbols[i] == pair[0]
                    and symbols[i + 1] == pair[1]
                ):
                    new_symbols.append("".join(pair))
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols

    def encode(self, text: str) -> list[str]:
        """Tokenise a string into BPE tokens."""
        tokens: list[str] = []
        for word in text.lower().split():
            word = re.sub(r"[^a-z']", "", word)
            if word:
                tokens.extend(self._tokenise_word(word))
        return tokens


# ---------------------------------------------------------------------------
# 5. Demo
# ---------------------------------------------------------------------------

CORPUS: list[str] = [
    "low lower lowest",
    "newer newest news",
    "wider widest wide",
    "old older oldest",
    "new new new newer newer newest newest newest",
    "low low low lower lower lowest",
]


if __name__ == "__main__":
    bpe = BPETokenizer(num_merges=15)
    bpe.fit(CORPUS)

    logger.info("Merge rules learned: %s", bpe.merges)

    test_cases = [
        "low",
        "lower",
        "lowest",
        "newest",
        "unknown",   # word not seen in training
    ]
    for word in test_cases:
        tokens = bpe.encode(word)
        logger.info("encode(%r) -> %s", word, tokens)
```

Run: `python day3_tiny_bpe.py`
Deps: stdlib only (no numpy needed)

### Sample output walkthrough

After ~15 merges on this corpus, you should see:
- `"low"` → `['low</w>']` (single token — frequent enough to merge fully)
- `"lower"` → `['low', 'er</w>']` or similar
- `"unknown"` → individual characters + `</w>` (degrades gracefully)

## Exercises

1. Add logging at `INFO` level (not just `DEBUG`) so each merge step prints the merged pair and its frequency. Run with 20 merges and inspect the pattern — do prefixes or suffixes merge first on this corpus?
2. Modify `BPETokenizer` to save/load merge rules to/from a JSON file using `pathlib.Path`. Add a `from_file(path)` class method.
3. Add a `decode(tokens: list[str]) -> str` method that reverses the `</w>` markers and reconstructs the original word sequence (spaces between words, strip `</w>` at the end of each word).
4. Test the edge case: what happens when `encode` receives an empty string? A string with only punctuation? Fix any crashes.
5. Write a `pytest` parametrised test: for each word in `["low", "lower", "lowest", "newer"]`, verify that `encode` returns a non-empty list and that joining the tokens (minus `</w>`) reconstructs the original word characters.

(Solutions: ../solutions/week-04/day-3.md)

## Self-check
- [ ] I can explain the BPE training loop step by step.
- [ ] I understand what the end-of-word marker is for and why it matters.
- [ ] I can implement `get_pair_counts`, `apply_merge`, and the training loop from memory.
- [ ] I can explain how BPE handles out-of-vocabulary words at inference time.
- [ ] I know the vocabulary sizes used by GPT-2, GPT-4, and BERT.

## Supplementary links
- [Neural Machine Translation with Rare Words (Sennrich et al. 2016)](https://arxiv.org/abs/1508.07909)
- [tiktoken (OpenAI BPE library)](https://github.com/openai/tiktoken)
- [Hugging Face tokenisers summary](https://huggingface.co/docs/transformers/tokenizer_summary)
- [MinBPE by Andrej Karpathy](https://github.com/karpathy/minbpe)
