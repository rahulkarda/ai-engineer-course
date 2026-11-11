# Solutions — Week 04 · Day 3

## Exercise 1 — INFO logging for each merge step

Add to the `fit` method, change `logger.debug` to `logger.info`:

```python
logger.info(
    "Step %02d: merged ('%s', '%s')  freq=%d",
    step + 1, best_pair[0], best_pair[1],
    pair_counts[best_pair],
)
```

**Pattern observation**: On the provided corpus, suffixes like `er</w>`, `est</w>`, `s</w>` tend to merge *after* their preceding character has merged with the root. Shared roots (`new`, `low`, `old`) form first, then the suffixes attach. This is consistent with BPE learning morphological structure implicitly from frequency.

---

## Exercise 2 — Save/load merge rules to/from JSON

```python
import json
from pathlib import Path

class BPETokenizer:
    # ... existing code ...

    def save(self, path: Path | str) -> None:
        """Persist merge rules to JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {"merges": list(self.merges)}
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Saved %d merge rules to %s", len(self.merges), p)

    @classmethod
    def from_file(cls, path: Path | str) -> "BPETokenizer":
        """Restore tokenizer from JSON file."""
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        tok = cls(num_merges=len(data["merges"]))
        tok.merges = [tuple(pair) for pair in data["merges"]]
        logger.info("Loaded %d merge rules from %s", len(tok.merges), p)
        return tok
```

---

## Exercise 3 — `decode` method

```python
def decode(self, tokens: list[str]) -> str:
    """Reconstruct original text from BPE tokens."""
    words: list[str] = []
    current: list[str] = []
    for tok in tokens:
        if tok.endswith(END_OF_WORD):
            current.append(tok[:-len(END_OF_WORD)])
            words.append("".join(current))
            current = []
        else:
            current.append(tok)
    if current:
        words.append("".join(current))
    return " ".join(words)
```

Test:
```python
bpe = BPETokenizer(num_merges=15)
bpe.fit(CORPUS)
assert bpe.decode(bpe.encode("low newer")) == "low newer"
```

---

## Exercise 4 — Edge case fixes

```python
def encode(self, text: str) -> list[str]:
    """Tokenise a string into BPE tokens. Returns [] for empty/punctuation."""
    tokens: list[str] = []
    for raw_word in text.lower().split():
        word = re.sub(r"[^a-z']+", "", raw_word)
        if word:  # skip empty after stripping
            tokens.extend(self._tokenise_word(word))
    return tokens
```

- Empty string: `text.lower().split()` returns `[]` → outer loop never runs → `[]` returned.
- Punctuation-only: `re.sub(r"[^a-z']+", "", "!!!??")` → `""` → `if word:` guard skips it.

---

## Exercise 5 — Parametrised pytest tests

```python
import pytest
from day3_tiny_bpe import BPETokenizer, END_OF_WORD, CORPUS

@pytest.fixture
def bpe() -> BPETokenizer:
    tok = BPETokenizer(num_merges=15)
    tok.fit(CORPUS)
    return tok

@pytest.mark.parametrize("word,expected_chars", [
    ("low",    "low"),
    ("lower",  "lower"),
    ("lowest", "lowest"),
    ("newer",  "newer"),
])
def test_encode_round_trips_word(bpe: BPETokenizer, word: str, expected_chars: str) -> None:
    tokens = bpe.encode(word)
    assert tokens, f"encode({word!r}) returned empty list"
    # Join token strings (strip END marker) and compare to original
    reconstructed = "".join(t.replace(END_OF_WORD, "") for t in tokens)
    assert reconstructed == expected_chars

def test_encode_empty_returns_empty(bpe: BPETokenizer) -> None:
    assert bpe.encode("") == []

def test_encode_punctuation_only_returns_empty(bpe: BPETokenizer) -> None:
    assert bpe.encode("!!! ???") == []
```
