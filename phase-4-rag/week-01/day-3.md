# Phase 4 · Week 01 · Day 3 — Text Extraction & Cleaning

> Objective: Build a configurable text cleaning pipeline that normalizes raw extracted text before chunking.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

## Concept

Raw text extracted from documents is rarely clean. Before you can chunk, embed, and index it, you must normalize it. Skipping this step silently degrades retrieval quality — dirty text produces noisy embeddings, and noisy embeddings produce poor retrieval scores.

**Common artifacts in raw extracted text:**

| Artifact | Example | Source |
|---|---|---|
| Repeated headers/footers | `"Page 4 of 12"` every N lines | PDF page breaks |
| Mid-sentence line breaks | `"This is an im-\nportant concept."` | PDF column wrap |
| Ligatures | `ﬁ`, `ﬂ`, `ﬀ` (fi, fl, ff as single characters) | PDF font encoding |
| Control characters | `\x00`, `\x0c` (form feed), `\r` | Binary extraction artifacts |
| Encoding artifacts | `â€™` instead of `'` (UTF-8 read as Latin-1) | Mojibake |
| Excessive whitespace | `"word     word"`, `"\n\n\n\n"` | Paragraph markers, HTML |
| Zero-width characters | `​`, `﻿` (BOM) | Copy-paste from web, DOCX |

**The extraction pipeline:**

```
raw bytes
    │
    ▼ decode (detect encoding, fallback to UTF-8 with replace)
decoded string
    │
    ▼ structural cleaning (control chars, BOM, ligatures)
structurally clean string
    │
    ▼ semantic cleaning (headers/footers, hyphenation, whitespace)
normalized text
    │
    ▼ ready for chunking
```

**Why cleaning before chunking matters:**

- A repeated footer `"Confidential — ACME Corp"` appearing in every chunk poisons embeddings — every chunk looks like it's "about" ACME Corp.
- Hyphenated line breaks (`"infor-\nmation"`) split a word across a chunk boundary — the word may not appear in search results.
- Excessive blank lines inflate chunk character counts without adding semantic content, wasting context window tokens.
- Inconsistent whitespace makes regex-based boundary detection unreliable.

---

## How It Works Internally

### Unicode Normalization: NFC vs NFKC

Unicode allows the same *visual* character to be encoded multiple ways. `é` can be a single codepoint (`U+00E9`) or two codepoints (`e` + combining accent `U+0301`). Comparing or hashing these produces different results despite identical appearance.

`unicodedata.normalize()` offers four forms:
- **NFC** (Canonical Decomposition + Canonical Composition): Combines base + combining marks into a single codepoint. Good for storage, round-trips safely.
- **NFKC** (Compatibility Decomposition + Canonical Composition): Additionally maps compatibility characters — ligatures (`ﬁ` → `fi`), superscripts (`²` → `2`), fullwidth variants (`ａ` → `a`). **Use NFKC for NLP** — it makes text consistent for tokenization and embedding.

```python
import unicodedata
unicodedata.normalize("NFKC", "ﬁnd")  # → "find"
unicodedata.normalize("NFKC", "café")  # → "café" (stable)
```

### Regex Patterns for Common Artifacts

Control characters occupy Unicode categories `Cc` (control) and `Cf` (format). The `unicodedata.category()` function identifies them. A fast regex approach:

```python
import re
# Remove ASCII control chars except tab and newline
re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)
```

BOM (`﻿`) and zero-width space (`​`) are format characters not caught by the above — handle explicitly.

### Whitespace Normalization and Token Cost

An LLM tokenizer treats `"word  word"` (2 spaces) as multiple tokens vs `"word word"` (1 space) as fewer tokens. In a 100,000-char document with aggressive whitespace, normalization can reduce token count by 5–15% — directly reducing embedding and generation cost. A single `re.sub(r" {2,}", " ", text)` pass pays for itself immediately.

### Sentence Boundary Detection

Naive `text.split(".")` fails on:
- Abbreviations (`Mr.`, `U.S.A.`, `3.14`)
- Decimal numbers (`price is $1.99`)
- URLs (`https://example.com/path.html`)

Production-grade sentence tokenizers (spaCy, NLTK Punkt) use trained models. For the cleaning stage, you don't need sentence tokenization — that belongs in the chunker. Cleaning only needs to ensure sentences are not *split mid-word* by line breaks.

---

## Worked Example

```python
"""
day_3_text_cleaner.py

Configurable text cleaning pipeline — stdlib only.
Designed to run on raw extracted text before chunking.
"""

import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TextCleanerConfig:
    """
    Configure which cleaning steps are active.

    All steps are enabled by default. Disable individual steps for
    formats where they cause regressions (e.g., disable hyphenation
    fix for code-heavy documents).
    """
    normalize_unicode: bool = True        # NFKC normalization + ligature expansion
    strip_control_chars: bool = True      # Remove non-printable control characters
    strip_zero_width: bool = True         # Remove BOM and zero-width chars
    collapse_whitespace: bool = True      # Normalize runs of spaces; normalize line endings
    remove_header_footer_patterns: bool = True  # Remove repeated noise lines
    fix_hyphenated_linebreaks: bool = False     # "infor-\nmation" → "information" (off by default)
    header_footer_patterns: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Cleaner
# ---------------------------------------------------------------------------

class TextCleaner:
    """
    Multi-step text cleaning pipeline.

    Usage:
        config = TextCleanerConfig(
            header_footer_patterns=["Page \\d+ of \\d+", "Confidential"]
        )
        cleaner = TextCleaner(config)
        clean_text = cleaner.clean(raw_text)
    """

    # Control characters: 0x00–0x08, 0x0B–0x0C, 0x0E–0x1F, 0x7F
    # Preserve: 0x09 (tab), 0x0A (LF), 0x0D (CR — normalized later)
    _CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")

    # Zero-width and BOM characters
    _ZERO_WIDTH_RE = re.compile(r"[﻿​‌‍⁠­]")

    # Hyphenated line break: word- newline word → wordword
    _HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")

    # Multiple spaces (preserve single spaces and newlines)
    _MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")

    # Multiple blank lines → at most two
    _MULTI_BLANK_LINE_RE = re.compile(r"\n{3,}")

    def __init__(self, config: TextCleanerConfig | None = None) -> None:
        self._config = config or TextCleanerConfig()
        self._compiled_hf_patterns: list[re.Pattern[str]] = [
            re.compile(p, re.IGNORECASE)
            for p in self._config.header_footer_patterns
        ]
        logger.debug(
            "TextCleaner initialized: %d header/footer patterns",
            len(self._compiled_hf_patterns),
        )

    # ------------------------------------------------------------------
    # Individual steps (each takes and returns str)
    # ------------------------------------------------------------------

    def _normalize_unicode(self, text: str) -> str:
        """NFKC normalization — expands ligatures, normalizes compatibility chars."""
        return unicodedata.normalize("NFKC", text)

    def _strip_control_chars(self, text: str) -> str:
        """Remove ASCII control characters (keep tab, LF; normalize CR to LF)."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return self._CONTROL_CHAR_RE.sub("", text)

    def _strip_zero_width(self, text: str) -> str:
        """Remove BOM and zero-width Unicode characters."""
        return self._ZERO_WIDTH_RE.sub("", text)

    def _fix_hyphenated_linebreaks(self, text: str) -> str:
        """
        Join hyphenated line breaks: 'infor-\\nmation' → 'information'.

        Only joins when both sides of the hyphen are word characters,
        avoiding false positives on bullet lists and compound words.
        """
        return self._HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)

    def _remove_header_footer_patterns(self, text: str) -> str:
        """Remove lines that match any configured header/footer pattern."""
        if not self._compiled_hf_patterns:
            return text
        lines = text.split("\n")
        clean_lines: list[str] = []
        removed = 0
        for line in lines:
            if any(p.search(line) for p in self._compiled_hf_patterns):
                removed += 1
            else:
                clean_lines.append(line)
        if removed:
            logger.debug("Removed %d header/footer lines", removed)
        return "\n".join(clean_lines)

    def _collapse_whitespace(self, text: str) -> str:
        """
        Normalize whitespace:
        - Collapse runs of spaces/tabs to a single space.
        - Collapse 3+ consecutive newlines to 2 (paragraph separator).
        - Strip leading/trailing whitespace.
        """
        text = self._MULTI_SPACE_RE.sub(" ", text)
        text = self._MULTI_BLANK_LINE_RE.sub("\n\n", text)
        return text.strip()

    # ------------------------------------------------------------------
    # Pipeline entry point
    # ------------------------------------------------------------------

    def clean(self, text: str) -> str:
        """
        Run the configured cleaning pipeline on raw text.

        Steps run in order: unicode → control chars → zero-width →
        hyphenation → headers/footers → whitespace.

        Args:
            text: Raw extracted text string.

        Returns:
            Cleaned, normalized text string.
        """
        cfg = self._config
        original_len = len(text)

        if cfg.normalize_unicode:
            text = self._normalize_unicode(text)

        if cfg.strip_control_chars:
            text = self._strip_control_chars(text)

        if cfg.strip_zero_width:
            text = self._strip_zero_width(text)

        if cfg.fix_hyphenated_linebreaks:
            text = self._fix_hyphenated_linebreaks(text)

        if cfg.remove_header_footer_patterns:
            text = self._remove_header_footer_patterns(text)

        if cfg.collapse_whitespace:
            text = self._collapse_whitespace(text)

        final_len = len(text)
        reduction_pct = 100 * (1 - final_len / original_len) if original_len else 0
        logger.info(
            "Cleaning complete: %d → %d chars (%.1f%% reduction)",
            original_len, final_len, reduction_pct,
        )
        return text


# ---------------------------------------------------------------------------
# Inline tests (pytest-style assertions — run as: python day_3_text_cleaner.py)
# ---------------------------------------------------------------------------

def _run_assertions() -> None:
    """Spot-check cleaning steps. Run as part of demo."""

    # NFKC: ligature expansion
    cleaner = TextCleaner()
    assert "fi" in cleaner.clean("ﬁnd"), "Ligature ﬁ should expand to 'fi'"

    # Control character removal
    assert "\x00" not in cleaner.clean("hello\x00world"), "Null byte should be removed"

    # BOM removal
    assert "﻿" not in cleaner.clean("﻿text"), "BOM should be removed"

    # Whitespace collapse
    result = cleaner.clean("a   b\n\n\n\nc")
    assert "   " not in result, "Multiple spaces should collapse"
    assert "\n\n\n" not in result, "3+ blank lines should collapse to 2"

    # Header/footer removal
    cfg = TextCleanerConfig(header_footer_patterns=[r"Page \d+ of \d+"])
    cleaner_hf = TextCleaner(cfg)
    result_hf = cleaner_hf.clean("Intro text.\nPage 3 of 10\nMore content.")
    assert "Page 3 of 10" not in result_hf, "Footer pattern should be removed"

    # Hyphenated linebreak (opt-in)
    cfg_hyph = TextCleanerConfig(fix_hyphenated_linebreaks=True)
    cleaner_hyph = TextCleaner(cfg_hyph)
    result_hyph = cleaner_hyph.clean("infor-\nmation")
    assert "information" in result_hyph, "Hyphenated linebreak should be joined"

    logger.info("All assertions passed.")


if __name__ == "__main__":
    _run_assertions()

    # Benchmark on synthetic data
    import random
    import string

    large_text = (
        "ﬁrst paragraph with a ligature.\nPage 1 of 100\n"
        + " ".join(random.choices(string.ascii_letters, k=500_000))
        + "\nPage 100 of 100\n"
    )

    cfg = TextCleanerConfig(header_footer_patterns=[r"Page \d+ of \d+"])
    cleaner = TextCleaner(cfg)

    start = time.perf_counter()
    result = cleaner.clean(large_text)
    elapsed = time.perf_counter() - start

    logger.info("Benchmark: cleaned %.1f MB in %.3f s", len(large_text) / 1e6, elapsed)
    print(f"Output length: {len(result):,} chars")
```

**Run it:**
```bash
python day_3_text_cleaner.py
```

Runs assertions and a synthetic benchmark. No dependencies.

---

## Exercises

1. **Hyphenated line break fix.** The `fix_hyphenated_linebreaks` step currently only joins when *both* sides of the hyphen are `\w`. Extend it to also handle the case where the left side ends with a *capitalized* word character, so `"Infor-\nmation"` → `"Information"`. Write a pytest test with `@pytest.mark.parametrize` covering the original case, the capital case, and a *false positive* (a line ending in a bullet `-` followed by a newline, which should NOT be joined).

2. **Parametrized pytest suite.** Write a pytest test using `@pytest.mark.parametrize` with three scenarios:
   - Ligature in input → ligature expanded in output.
   - BOM character at start → removed in output.
   - Footer pattern `r"CONFIDENTIAL"` matches and is removed from a multi-line string.

3. **Benchmark logging.** Extend the `clean()` method to log processing *throughput* in MB/s (chars per second / 1e6), not just total time. Wrap this in a `if logger.isEnabledFor(logging.DEBUG)` guard so it only runs when debug logging is active (avoids a `time.perf_counter()` call in production).

(Solutions: ../solutions/week-01/day-3.md)

---

## Self-check

- [ ] I can explain the difference between NFC and NFKC normalization and why NFKC is preferred for NLP preprocessing.
- [ ] I can identify at least four categories of text artifacts and describe their source (format, encoding, extraction method).
- [ ] I can implement a configurable cleaning pipeline using the step-function pattern — each step takes a `str` and returns a `str`.
- [ ] I can explain why cleaning before chunking (rather than after) improves retrieval quality and reduces token cost.
