# Week 01 Assessment — Python Fundamentals

> Time: ~1 hr  ·  Pass threshold: 75%  ·  Covers: Days 1–7

---

## Conceptual (interview-style)

Answer in your own words. Aim for 2–4 sentences each.

1. **`is` vs `==`**: What is the difference between `is` and `==` in Python? Construct a concrete example where they give different results.

2. **Virtual environments**: Why do virtual environments exist? What specific problem do they solve? What goes wrong if you skip them?

3. **Name binding**: Explain Python's name binding model. What happens in memory when you write `x = 42`? What happens when you then write `x = "hello"`?

4. **Mutability**: What does "mutable" mean? Name three mutable built-in types and three immutable built-in types. Why does mutability matter when you pass objects to functions?

5. **Short-circuit evaluation**: What is short-circuit evaluation in `and`/`or`? Write a practical one-liner that uses it to provide a default value safely.

6. **LEGB rule**: What is the LEGB rule? In what order does Python resolve a name? Give a short code snippet that demonstrates the difference between local and global scope.

7. **`logging` vs `print`**: Why should production code use `logging` instead of `print`? Name three specific advantages.

8. **Execution model**: Walk through what happens step-by-step when you run `python script.py`. Start from the shell command and end at the first line of user code executing.

---

## Coding challenge — `word_stats.py`

Write a complete, runnable Python script `word_stats.py` that meets all the requirements below.

### Specification

**Input:** A filename passed as the first command-line argument.

**Processing:**
- Read the file contents.
- Tokenise into words: split on whitespace, strip leading/trailing punctuation from each token (you may use `str.strip` with a punctuation string, or `str.translate`).
- Compute:
  - Total word count (including duplicates)
  - Unique word count (case-insensitive)
  - Top 5 most frequent words with their counts (case-insensitive; if fewer than 5 unique words exist, show all)

**Output:** Write results to a file named `<input_stem>_stats.txt` (e.g., input `sample.txt` → output `sample_stats.txt`).

Example output file:
```
Word stats for: sample.txt
Total words   : 142
Unique words  : 87
Top 5:
  the         : 12
  and         :  8
  of          :  7
  to          :  6
  a           :  5
```

**Logging:** Log at `DEBUG` level when the file is opened, when tokenisation completes (log word count), and when the output file is written.

### Acceptance criteria

- [ ] Script runs with `python word_stats.py sample.txt`
- [ ] Produces a correctly formatted `sample_stats.txt`
- [ ] Handles `FileNotFoundError`: prints a friendly message to stderr, exits with code 1, does not crash with a traceback
- [ ] Handles empty file: writes stats showing zeros, does not crash
- [ ] Uses `pathlib.Path` for all file operations
- [ ] Uses f-strings for all formatted output
- [ ] Type hints on every function signature
- [ ] `logging.basicConfig` configured at `DEBUG` level
- [ ] Passes `ruff check word_stats.py` with no errors

### Constraints

- Standard library only (`collections`, `string`, `pathlib`, `logging`, `sys` are all fine)
- No global mutable state — put logic inside functions
- `main()` entry point guarded by `if __name__ == "__main__":`
- Single file, ≤ 80 lines

---

## Rubric (pass >= 75%)

| Dimension | Weight | What earns full marks |
|---|---|---|
| Independent implementation | 40% | All required functions present, logic correct, no copy-paste from scaffold |
| Conceptual understanding | 25% | Conceptual answers are accurate, use correct terminology, show own understanding |
| Debugging & problem-solving | 20% | Error handling is correct (FileNotFoundError, empty file, bad argv); errors go to stderr |
| Code quality | 15% | Type hints throughout, logging not print, pathlib for files, ruff passes |

**Scoring guide:**
- 90–100% — Ready to move to Week 2 immediately
- 75–89% — Pass; review any section below 70% before continuing
- 60–74% — Revise Days 1–6 on the weak topics, then retake
- < 60% — Repeat the week before proceeding

Answer key: ../../solutions/week-01/assessment.md
