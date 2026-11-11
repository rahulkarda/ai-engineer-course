# Solutions — Phase 4 · Week 01 · Day 3

## Exercise 1 — Hyphenated Line Break Fix

New `fix_hyphenated_line_breaks` step method added to `TextCleaner`:

```python
"""day3_solution.py — TextCleaner with hyphenated line-break fix."""
import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CleaningStats:
    """Counters collected during a cleaning run."""
    chars_before: int = 0
    chars_after: int = 0
    hyphen_joins: int = 0


class TextCleaner:
    """Pipeline of text-cleaning steps.

    Each step is a method that takes a string and returns a string.
    Call :meth:`clean` to run the full pipeline.
    """

    def normalize_unicode(self, text: str) -> str:
        """Normalize to NFC and replace common lookalike characters."""
        text = unicodedata.normalize("NFC", text)
        # Smart quotes → straight quotes
        text = text.replace("‘", "'").replace("’", "'")
        text = text.replace("“", '"').replace("”", '"')
        # Em-dash / en-dash → hyphen-minus for consistency downstream
        text = text.replace("—", "--").replace("–", "-")
        return text

    def strip_control_chars(self, text: str) -> str:
        """Remove ASCII control characters (except newline and tab)."""
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    def collapse_whitespace(self, text: str) -> str:
        """Replace runs of spaces/tabs with a single space; normalise blank lines."""
        # Collapse horizontal whitespace
        text = re.sub(r"[ \t]+", " ", text)
        # Collapse 3+ consecutive newlines to exactly 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def remove_headers_footers(self, text: str, marker: str = "---") -> str:
        """Remove a repeated page header/footer block delimited by *marker*."""
        escaped = re.escape(marker)
        pattern = rf"^{escaped}.*?{escaped}\s*"
        return re.sub(pattern, "", text, flags=re.DOTALL | re.MULTILINE)

    def fix_hyphenated_line_breaks(self, text: str) -> str:
        """Re-join words split across lines with a trailing hyphen.

        PDF extraction often produces:

            "Infor-\\nmation retrieval"

        This step collapses those into:

            "Information retrieval"

        The pattern matches:
        - A word character sequence
        - Immediately followed by a hyphen
        - Immediately followed by a newline (with optional trailing space)
        - Followed by the continuation word

        Only soft hyphens used for line-breaking are removed.  True compound
        words whose hyphen falls at a line boundary are indistinguishable here,
        so this is a best-effort heuristic.

        Returns:
            Cleaned text with hyphenated line-breaks joined.
        """
        # Pattern: word-char(s) + hyphen + optional spaces + newline + optional leading spaces + word-char(s)
        pattern = re.compile(r"(\w)-[ \t]*\n[ \t]*(\w)")
        fixed, n_subs = pattern.subn(r"\1\2", text)
        if n_subs:
            logger.debug("fix_hyphenated_line_breaks: joined %d hyphenated break(s)", n_subs)
        return fixed

    def clean(self, text: str) -> tuple[str, CleaningStats]:
        """Run the full cleaning pipeline and return (cleaned_text, stats).

        Pipeline order:
        1. fix_hyphenated_line_breaks  — must come before collapse_whitespace
        2. normalize_unicode
        3. strip_control_chars
        4. collapse_whitespace
        5. remove_headers_footers
        """
        stats = CleaningStats(chars_before=len(text))

        # Count joins before cleaning so we can report them
        before_join = text
        text = self.fix_hyphenated_line_breaks(text)
        stats.hyphen_joins = len(re.findall(r"(\w)-[ \t]*\n[ \t]*(\w)", before_join))

        text = self.normalize_unicode(text)
        text = self.strip_control_chars(text)
        text = self.collapse_whitespace(text)
        text = self.remove_headers_footers(text)

        stats.chars_after = len(text)
        logger.info(
            "TextCleaner.clean: chars %d → %d (reduction %.1f%%), hyphen_joins=%d",
            stats.chars_before,
            stats.chars_after,
            100 * (1 - stats.chars_after / max(stats.chars_before, 1)),
            stats.hyphen_joins,
        )
        return text, stats
```

**Implementation notes:**
- The step is added as a method — same pattern as existing steps — so it slots naturally into any pipeline order.
- The regex `(\w)-[ \t]*\n[ \t]*(\w)` handles trailing spaces and leading indentation on the continuation line.
- The step is placed *before* `collapse_whitespace` so newlines are still present when the pattern runs.
- `CleaningStats.hyphen_joins` gives callers observability on how aggressive the fix was.


## Exercise 2 — Parametrize pytest

Full parametrized test covering three distinct hyphenation scenarios:

```python
"""Tests for TextCleaner.fix_hyphenated_line_breaks."""
import pytest

from day3_solution import TextCleaner


@pytest.fixture()
def cleaner() -> TextCleaner:
    return TextCleaner()


@pytest.mark.parametrize(
    "input_text, expected, description",
    [
        # Scenario 1: Basic mid-word PDF hyphenation
        (
            "infor-\nmation retrieval is hard",
            "information retrieval is hard",
            "basic mid-word hyphenation",
        ),
        # Scenario 2: Multiple hyphenated breaks in one text
        (
            "doc-\nument pro-\ncessing is pow-\nerful",
            "document processing is powerful",
            "multiple hyphenated breaks",
        ),
        # Scenario 3: No hyphenation — text should be unchanged
        (
            "This text has no\nhyphenated line breaks at all.",
            "This text has no\nhyphenated line breaks at all.",
            "no hyphenation — text unchanged",
        ),
    ],
    ids=["basic", "multiple", "no_change"],
)
def test_fix_hyphenated_line_breaks(
    cleaner: TextCleaner,
    input_text: str,
    expected: str,
    description: str,
) -> None:
    result = cleaner.fix_hyphenated_line_breaks(input_text)
    assert result == expected, f"Scenario '{description}' failed.\nGot:      {result!r}\nExpected: {expected!r}"


# Additional edge-case tests (not parametrized — distinct setup logic)

def test_fix_preserves_true_hyphens_not_at_line_break(cleaner: TextCleaner) -> None:
    """Compound words with hyphens NOT at a line boundary must survive."""
    text = "state-of-the-art system\nwell-known author"
    result = cleaner.fix_hyphenated_line_breaks(text)
    assert "state-of-the-art" in result
    assert "well-known" in result


def test_fix_handles_trailing_spaces_before_newline(cleaner: TextCleaner) -> None:
    """Trailing spaces between the hyphen and the newline should be consumed."""
    text = "hyph-   \ncontinuation"
    result = cleaner.fix_hyphenated_line_breaks(text)
    assert result == "hyphcontinuation"


def test_fix_handles_leading_spaces_after_newline(cleaner: TextCleaner) -> None:
    """Leading indent on the continuation line should be consumed."""
    text = "hyph-\n   continuation"
    result = cleaner.fix_hyphenated_line_breaks(text)
    assert result == "hyphcontinuation"


def test_clean_pipeline_includes_hyphen_fix(cleaner: TextCleaner) -> None:
    """End-to-end: clean() should apply the hyphen fix within the full pipeline."""
    text = "doc-\nument’s structure is inter-\nesting."
    cleaned, stats = cleaner.clean(text)
    assert "document" in cleaned
    assert "interesting" in cleaned
    # Smart quote should also be normalised
    assert "’" not in cleaned
```


## Exercise 3 — Benchmark

Benchmark script using `time.perf_counter`, logging elapsed time per step and total:

```python
"""benchmark_cleaning.py — Benchmark TextCleaner steps on synthetic corpora.

Usage:
    python benchmark_cleaning.py

Adjust CORPUS_SIZE and REPEAT to suit your machine.
"""
import logging
import random
import string
import time
from typing import Callable

from day3_solution import TextCleaner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("benchmark")


# ---------------------------------------------------------------------------
# Synthetic corpus generator
# ---------------------------------------------------------------------------

def _random_word(min_len: int = 3, max_len: int = 12) -> str:
    length = random.randint(min_len, max_len)
    return "".join(random.choices(string.ascii_lowercase, k=length))


def _build_corpus(
    n_words: int,
    hyphen_rate: float = 0.05,
    control_char_rate: float = 0.01,
) -> str:
    """Generate a synthetic document of *n_words* words.

    Args:
        n_words: Number of words in the generated text.
        hyphen_rate: Fraction of words that are broken across lines with a hyphen.
        control_char_rate: Fraction of words followed by a random control char.
    """
    words: list[str] = []
    for i in range(n_words):
        word = _random_word()
        if random.random() < hyphen_rate and len(word) > 4:
            # Split the word in the middle with a hyphen-newline
            mid = len(word) // 2
            words.append(word[:mid] + "-\n" + word[mid:])
        elif random.random() < control_char_rate:
            words.append(word + "\x07")  # bell character
        else:
            words.append(word)
    return " ".join(words)


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

def _time_step(
    step_fn: Callable[[str], str],
    text: str,
    repeat: int,
) -> tuple[float, str]:
    """Run *step_fn* on *text* *repeat* times; return (avg_ms, final_result)."""
    result = text
    start = time.perf_counter()
    for _ in range(repeat):
        result = step_fn(text)
    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / repeat) * 1000
    return avg_ms, result


def benchmark(
    corpus_sizes: list[int] | None = None,
    repeat: int = 10,
) -> None:
    """Run the benchmark over several corpus sizes.

    Args:
        corpus_sizes: Word counts to benchmark. Defaults to [10_000, 50_000, 100_000].
        repeat: Number of repetitions per step per corpus size (for averaging).
    """
    if corpus_sizes is None:
        corpus_sizes = [10_000, 50_000, 100_000]

    cleaner = TextCleaner()

    steps: list[tuple[str, Callable[[str], str]]] = [
        ("fix_hyphenated_line_breaks", cleaner.fix_hyphenated_line_breaks),
        ("normalize_unicode", cleaner.normalize_unicode),
        ("strip_control_chars", cleaner.strip_control_chars),
        ("collapse_whitespace", cleaner.collapse_whitespace),
        ("remove_headers_footers", cleaner.remove_headers_footers),
    ]

    for n_words in corpus_sizes:
        corpus = _build_corpus(n_words)
        corpus_chars = len(corpus)
        logger.info(
            "--- corpus: %d words / %d chars ---", n_words, corpus_chars
        )

        total_ms = 0.0
        for step_name, step_fn in steps:
            avg_ms, _ = _time_step(step_fn, corpus, repeat)
            total_ms += avg_ms
            logger.info(
                "  step=%-35s avg=%.2f ms  (chars/ms=%.0f)",
                step_name,
                avg_ms,
                corpus_chars / max(avg_ms, 0.001),
            )

        # Time the full pipeline via clean()
        start = time.perf_counter()
        for _ in range(repeat):
            cleaned, _ = cleaner.clean(corpus)
        pipeline_ms = ((time.perf_counter() - start) / repeat) * 1000
        reduction_pct = 100 * (1 - len(cleaned) / corpus_chars)

        logger.info(
            "  step=%-35s avg=%.2f ms  (chars/ms=%.0f)  reduction=%.1f%%",
            "FULL PIPELINE (clean)",
            pipeline_ms,
            corpus_chars / max(pipeline_ms, 0.001),
            reduction_pct,
        )
        logger.info(
            "  sum-of-steps=%.2f ms  pipeline=%.2f ms  overhead=%.2f ms",
            total_ms, pipeline_ms, pipeline_ms - total_ms,
        )


if __name__ == "__main__":
    random.seed(42)
    benchmark(corpus_sizes=[10_000, 50_000, 100_000], repeat=10)
```

**Sample output (indicative, hardware-dependent):**
```
2026-11-11 10:00:00 INFO benchmark — --- corpus: 10000 words / 74382 chars ---
2026-11-11 10:00:00 INFO benchmark —   step=fix_hyphenated_line_breaks      avg=2.31 ms  (chars/ms=32199)
2026-11-11 10:00:00 INFO benchmark —   step=normalize_unicode               avg=0.48 ms  (chars/ms=154963)
2026-11-11 10:00:00 INFO benchmark —   step=strip_control_chars             avg=1.12 ms  (chars/ms=66413)
2026-11-11 10:00:00 INFO benchmark —   step=collapse_whitespace             avg=0.93 ms  (chars/ms=79981)
2026-11-11 10:00:00 INFO benchmark —   step=remove_headers_footers          avg=0.04 ms  (chars/ms=1859550)
2026-11-11 10:00:00 INFO benchmark —   step=FULL PIPELINE (clean)           avg=5.12 ms  (chars/ms=14527)  reduction=3.1%
2026-11-11 10:00:00 INFO benchmark —   sum-of-steps=4.88 ms  pipeline=5.12 ms  overhead=0.24 ms
```

**Design notes:**
- `time.perf_counter()` — highest-resolution monotonic timer; correct for wall-clock benchmarking on a single thread.
- `repeat` averages out OS scheduling jitter for short runs.
- `chars/ms` throughput metric makes it easy to compare steps of different complexity.
- Corpus generator injects realistic noise (hyphenated breaks, control chars) to exercise all steps meaningfully.
- Run with `python -O` to disable debug-level logging for the cleanest timing.
