# Phase 3 · Week 04 · Day 4 — Subword Tokenisation: WordPiece, SentencePiece & HF Tokenizers

> Objective: Understand WordPiece and SentencePiece; use Hugging Face tokenizers library confidently.  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

## Concept

Day 3 built BPE from scratch. Today covers two more subword algorithms and the production library that implements all three.

### WordPiece (BERT family)

WordPiece is similar to BPE but chooses merges by **maximising the likelihood of the training corpus** under a language model, not raw pair frequency.

The merge score for pair (A, B) is:

```
score(A, B) = freq(AB) / (freq(A) * freq(B))
```

This rewards pairs that form a new unit unlikely to occur by chance — i.e. pairs that carry **mutual information**. The result tends to produce more linguistically motivated splits: `"embedding"` → `["em", "##bed", "##ding"]`.

Note the `##` prefix: WordPiece marks continuation pieces with `##` to indicate "this segment continues the previous word" (versus a word-initial piece).

### SentencePiece

BPE and WordPiece need pre-tokenised input (whitespace-split words). **SentencePiece** works directly on raw Unicode characters, treating whitespace as just another character (represented as `▁`).

Benefits:
- Language-agnostic — handles Chinese, Arabic, Thai (no spaces between words).
- Fully invertible: `encode → decode` is lossless.
- Can use BPE or **unigram language model** (ULM) under the hood.

**Unigram LM algorithm:**
1. Start with a large vocabulary of subword candidates.
2. Train a unigram language model `P(x) = ∏ P(piece_i)` using the EM algorithm.
3. Prune the vocabulary: remove pieces whose removal causes the smallest loss increase.
4. Repeat until target vocabulary size is reached (top-down pruning, unlike BPE's bottom-up merging).

| Algorithm | Direction | Criterion | Used by |
|---|---|---|---|
| BPE | Bottom-up merges | Frequency | GPT-2/3/4, RoBERTa |
| WordPiece | Bottom-up merges | Likelihood gain | BERT, DistilBERT |
| SentencePiece (BPE) | Bottom-up | Frequency on raw chars | LLaMA, T5 |
| SentencePiece (Unigram) | Top-down pruning | Likelihood | ALBERT, mT5 |

### Hugging Face `tokenizers`

`tokenizers` is a Rust-backed Python library with a four-component pipeline:

```
Normaliser → Pre-tokeniser → Model (BPE/WordPiece/Unigram) → Post-processor
```

The `transformers` library wraps `tokenizers` with `AutoTokenizer` for model-specific defaults.

## How it works internally

### Special tokens

Every tokeniser adds task-specific special tokens:
- BERT: `[CLS]` (start), `[SEP]` (separator/end), `[PAD]`, `[UNK]`, `[MASK]`
- GPT-2: `<|endoftext|>`
- LLaMA: `<s>` (BOS), `</s>` (EOS), `<unk>`

These are never split — they map to reserved IDs.

### Token IDs, attention masks, and offsets

`tokenizer(text, return_offsets_mapping=True)` returns:
- `input_ids`: list of integer token IDs
- `attention_mask`: 1 for real tokens, 0 for padding
- `offset_mapping`: `(start, end)` character positions in the original string — essential for NER

### Truncation and padding

LLMs have a context window (`max_length`). `tokenizer(text, truncation=True, max_length=512, padding="max_length")` handles both automatically.

## Worked example

```python
"""day4_hf_tokenizers.py — Using Hugging Face tokenizers library."""

import logging
import os

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NOTE: This file requires the `transformers` and `tokenizers` packages.
# Install: uv add transformers tokenizers
#
# The examples below use pre-trained tokenizer configs loaded from the HF hub
# (no model weights downloaded — tokenizer only).
# They require an internet connection on first run; subsequent runs use cache.
# ---------------------------------------------------------------------------


def demo_bert_tokenizer() -> None:
    """BERT WordPiece tokenizer exploration."""
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    logger.info("BERT vocab size: %d", tokenizer.vocab_size)
    logger.info("Special tokens: %s", tokenizer.all_special_tokens)

    texts = [
        "The quick brown fox.",
        "Unbelievably, tokenization is fascinating!",
        "embeddings",
        "embed ##ding ##s",   # manually pre-split to see how ## works
    ]

    for text in texts:
        enc = tokenizer(text)
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])
        logger.info("Text: %r  →  tokens: %s", text, tokens)

    # Batched with padding
    batch = tokenizer(
        ["Hello world.", "A longer sentence with more tokens here."],
        padding=True,
        truncation=True,
        max_length=16,
        return_tensors="np",
    )
    logger.info("Batched input_ids shape: %s", batch["input_ids"].shape)
    logger.info("Attention masks:\n%s", batch["attention_mask"])


def demo_gpt2_tokenizer() -> None:
    """GPT-2 BPE tokenizer exploration."""
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    logger.info("GPT-2 vocab size: %d", tokenizer.vocab_size)

    text = "Tokenization is fun! Byte-Pair Encoding merges frequent pairs."
    enc = tokenizer(text)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])
    logger.info("GPT-2 tokens: %s", tokens)
    logger.info("Token IDs:    %s", enc["input_ids"])

    # Decoding back
    decoded = tokenizer.decode(enc["input_ids"])
    logger.info("Decoded: %r", decoded)

    # Note: GPT-2 uses 'Ġ' (U+0120) for space-initial tokens
    for tok_id, tok in zip(enc["input_ids"], tokens):
        if "Ġ" in tok:
            logger.debug("  Space-initial token: %r (id=%d)", tok, tok_id)


def demo_offset_mapping() -> None:
    """Show character-level offsets — useful for NER span extraction."""
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    text = "Paris is a beautiful city."
    enc = tokenizer(text, return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])

    logger.info("%-20s  %-10s  %s", "Token", "Offset", "Span text")
    for tok, (start, end) in zip(tokens, enc["offset_mapping"]):
        span = text[start:end] if start != end else ""
        logger.info("%-20s  (%d, %-3d)  %r", tok, start, end, span)


def demo_tokenizers_train() -> None:
    """
    Train a small BPE tokenizer from scratch using the tokenizers library.
    Does not require an internet connection.
    """
    from tokenizers import Tokenizer  # noqa: PLC0415
    from tokenizers.models import BPE  # noqa: PLC0415
    from tokenizers.pre_tokenizers import Whitespace  # noqa: PLC0415
    from tokenizers.trainers import BpeTrainer  # noqa: PLC0415

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=100,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    )

    # Train on a tiny in-memory corpus (normally pass file paths)
    corpus = [
        "the cat sat on the mat",
        "the dog sat on the floor",
        "a quick brown fox jumps over the lazy dog",
        "tokenization splits words into subwords",
        "byte pair encoding is a greedy algorithm",
    ]
    tokenizer.train_from_iterator(corpus, trainer=trainer)

    enc = tokenizer.encode("the cat jumped over the fox")
    logger.info("Custom BPE tokens: %s", enc.tokens)
    logger.info("Custom BPE ids:    %s", enc.ids)


if __name__ == "__main__":
    try:
        demo_bert_tokenizer()
        demo_gpt2_tokenizer()
        demo_offset_mapping()
    except Exception as exc:  # noqa: BLE001
        logger.warning("HF Hub demo skipped (may need internet): %s", exc)

    # Offline demo — always runs
    demo_tokenizers_train()
```

Deps: `uv add transformers tokenizers numpy`
First run downloads tokenizer configs (~1 MB, no model weights).

## Exercises

1. Compare BERT and GPT-2 tokenisations of the same 10 sentences. For which sentence type does BERT produce more tokens? For which does GPT-2? Hypothesise why.
2. `AutoTokenizer.from_pretrained("xlm-roberta-base")` loads a SentencePiece tokenizer. Tokenise a short English sentence and a short German sentence. Notice the `▁` prefix and explain what it represents.
3. Use `return_offsets_mapping=True` with BERT on `"New York City is great."` and extract the character spans for every token that is NOT a special token (`[CLS]`, `[SEP]`).
4. The `demo_tokenizers_train` function uses a tiny 5-sentence corpus. Increase it to 20 sentences (any English text), set `vocab_size=200`, and inspect the learned vocabulary. List the 5 most surprising merge rules.
5. Write a `pytest` test (using `unittest.mock.patch`) that verifies `demo_bert_tokenizer` calls `AutoTokenizer.from_pretrained` exactly once and that the mock tokenizer's `vocab_size` attribute is read. This tests the function's interface without hitting the network.

(Solutions: ../solutions/week-04/day-4.md)

## Self-check
- [ ] I can explain how WordPiece's merge criterion differs from BPE's.
- [ ] I know what `##` means in BERT tokens.
- [ ] I can explain why SentencePiece is language-agnostic.
- [ ] I can use `AutoTokenizer` to encode, decode, and extract offset mappings.
- [ ] I can train a custom BPE tokenizer with the `tokenizers` library.

## Supplementary links
- [WordPiece paper (Schuster & Nakamura 2012)](https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/37842.pdf)
- [SentencePiece paper (Kudo & Richardson 2018)](https://arxiv.org/abs/1808.06226)
- [HF tokenizers documentation](https://huggingface.co/docs/tokenizers/)
- [Summary of tokenizers — HF Transformers docs](https://huggingface.co/docs/transformers/tokenizer_summary)
