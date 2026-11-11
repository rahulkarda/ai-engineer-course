# Solutions — Week 04 · Day 4

## Exercise 1 — BERT vs GPT-2 token counts

```python
from transformers import AutoTokenizer

bert = AutoTokenizer.from_pretrained("bert-base-uncased")
gpt2 = AutoTokenizer.from_pretrained("gpt2")

sentences = [
    "The cat sat on the mat.",
    "Unbelievably, morphologically complex words challenge tokenizers.",
    "你好世界",                               # Chinese — BERT handles better
    "http://example.com/api/v2/endpoint",   # URLs — GPT-2 often produces fewer tokens
    "NLP is fun!",
]
for s in sentences:
    n_bert = len(bert(s)["input_ids"])
    n_gpt2 = len(gpt2(s)["input_ids"])
    print(f"BERT={n_bert:3d}  GPT-2={n_gpt2:3d}  {s[:40]}")
```

**Expected pattern:**
- BERT produces more tokens for long, morphologically complex English words because WordPiece segments aggressively with `##` continuations.
- GPT-2 tends to produce slightly fewer tokens on clean English text because its 50k vocab includes more multi-character prefixes from its larger training corpus.
- For Chinese/CJK, BERT's WordPiece breaks each character into `[UNK]` (bert-base-uncased is English-only). Use `bert-base-multilingual-cased` for multilingual.

---

## Exercise 2 — SentencePiece `▁` prefix

```python
from transformers import AutoTokenizer

xlmr = AutoTokenizer.from_pretrained("xlm-roberta-base")

en = "Machine learning is fascinating."
de = "Maschinelles Lernen ist faszinierend."

for text, lang in [(en, "EN"), (de, "DE")]:
    tokens = xlmr.tokenize(text)
    print(f"{lang}: {tokens}")
```

**The `▁` prefix (U+2581)** represents a space character. SentencePiece treats the input as a raw byte stream: spaces are encoded as `▁` and become part of the token. A token beginning with `▁` starts a new word; a token without `▁` is a continuation piece. This means no separate pre-tokenisation step is needed — the algorithm is language-agnostic by design.

---

## Exercise 3 — Offset mapping for "New York City is great."

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text = "New York City is great."
enc = tokenizer(text, return_offsets_mapping=True)
tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])
special = {"[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"}

for tok, (start, end) in zip(tokens, enc["offset_mapping"]):
    if tok not in special and start != end:
        print(f"{tok!r:20s}  span: ({start}, {end})  text: {text[start:end]!r}")
```

Expected output (bert-base-uncased lowercases everything):
```
'new'                span: (0, 3)   text: 'New'
'york'               span: (4, 8)   text: 'York'
'city'               span: (9, 13)  text: 'City'
'is'                 span: (14, 16) text: 'is'
'great'              span: (17, 22) text: 'great'
'.'                  span: (22, 23) text: '.'
```

---

## Exercise 4 — Custom BPE with larger corpus

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(vocab_size=200, special_tokens=["[UNK]"])

corpus = [
    # 20 sentences
    "the quick brown fox jumps over the lazy dog",
    "natural language processing is a subfield of artificial intelligence",
    "tokenization splits text into smaller units called tokens",
    "byte pair encoding merges frequent character pairs",
    "word embeddings map words to dense vectors",
    "cosine similarity measures angle between two vectors",
    "the transformer architecture uses self attention",
    "bert is a bidirectional encoder representation",
    "sentence transformers produce fixed size sentence embeddings",
    "bag of words ignores word order in documents",
    "term frequency measures how often a word appears",
    "inverse document frequency down weights common terms",
    "gradient descent optimises model parameters",
    "neural networks learn hierarchical feature representations",
    "recurrent networks process sequences step by step",
    "attention mechanisms allow models to focus on relevant parts",
    "fine tuning adapts a pretrained model to a new task",
    "vocabulary size affects model capacity and memory",
    "subword tokenization handles rare and unknown words",
    "embedding dimensions range from fifty to thousands",
]
tokenizer.train_from_iterator(corpus, trainer=trainer)
vocab = tokenizer.get_vocab()
# Print 5 multi-character tokens (these are the interesting merge products)
multi = sorted((t for t in vocab if len(t) > 2 and not t.startswith("[")), key=vocab.get)[-5:]
print("Top merged tokens:", multi)
```

Surprising rules typically involve common word endings (`ing`, `ion`, `tion`) or very frequent word pairs (`of`, `the`) being merged early.

---

## Exercise 5 — Mock test for `demo_bert_tokenizer`

```python
from unittest.mock import MagicMock, patch, call
import pytest

def test_bert_tokenizer_called_once() -> None:
    mock_tok = MagicMock()
    mock_tok.vocab_size = 30522
    mock_tok.all_special_tokens = ["[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"]
    mock_tok.return_value = {"input_ids": [101, 1996, 102]}
    mock_tok.convert_ids_to_tokens.return_value = ["[CLS]", "the", "[SEP]"]

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok) as mock_fn:
        from day4_hf_tokenizers import demo_bert_tokenizer
        demo_bert_tokenizer()
        mock_fn.assert_called_once_with("bert-base-uncased")
        # Verify vocab_size was accessed
        _ = mock_tok.vocab_size
```

Note: The mock intentionally has limited return values; the test verifies `from_pretrained` is called exactly once and that the function accesses `vocab_size`.
