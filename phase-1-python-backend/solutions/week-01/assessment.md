# Answer Key — Week 01 Assessment

> Instructor/self-check reference. Do not open until you have submitted your own answers.

---

## Part 1 — Conceptual (interview-style)

**Q1. What is the difference between `is` and `==` in Python?**

`==` tests **value equality** — whether two objects represent the same value.  
`is` tests **identity** — whether two names point to the exact same object in memory (same `id()`).

```python
a: list[int] = [1, 2, 3]
b: list[int] = [1, 2, 3]

print(a == b)   # True  — same contents
print(a is b)   # False — different list objects

c = a
print(a is c)   # True  — c is another name for the same object
```

Common trap: small integers (typically -5 to 256) and interned strings are cached by CPython, so `x = 5; y = 5; x is y` returns `True` — but this is an implementation detail, not guaranteed behavior. Always use `==` for value comparison.

---

**Q2. Why use a virtual environment? What problem does it solve?**

A virtual environment is an isolated Python installation scoped to a single project. It solves **dependency conflicts**: if Project A requires `requests==2.28` and Project B requires `requests==2.31`, installing both in the system Python is impossible — one will overwrite the other. With separate venvs each project has its own `site-packages` directory and its own pinned versions.

```bash
python -m venv .venv          # create
source .venv/bin/activate     # activate (macOS/Linux)
pip install requests==2.28    # goes into this venv only
```

---

**Q3. Walk through what happens when Python executes `x = 42`.**

1. The integer object `42` is created (or retrieved from the small-int cache for -5–256) in heap memory.
2. Python looks up the name `"x"` in the current **namespace** (a dictionary mapping names to objects).
3. The namespace entry for `"x"` is set to point to that integer object.

Python variables are **names** (references), not typed memory slots. The object owns its type; the name has no type at all. You can rebind `x = "hello"` immediately and the old int object's reference count drops by 1.

---

**Q4. What does "mutable" mean? Give three mutable and three immutable types.**

**Mutable** — the object's internal state can be changed after creation without creating a new object.

| Mutable | Immutable |
|---------|-----------|
| `list` | `int` |
| `dict` | `str` |
| `set` | `tuple` |

```python
# Mutable — in-place change, same object id
lst: list[int] = [1, 2]
original_id = id(lst)
lst.append(3)
print(id(lst) == original_id)  # True — same object

# Immutable — "change" creates new object
s: str = "hello"
original_id = id(s)
s += " world"
print(id(s) == original_id)    # False — new string object
```

---

**Q5. What is short-circuit evaluation? Give a practical use case.**

Python's `and` / `or` operators stop evaluating as soon as the result is determined:
- `and` returns the first **falsy** value, or the last value if all are truthy.
- `or` returns the first **truthy** value, or the last value if all are falsy.

```python
# Safe attribute access — if user is None, user.name is never evaluated
user = get_user_or_none()
display_name: str = user and user.name or "Guest"

# Default value pattern
config_timeout: int | None = None
timeout: int = config_timeout or 30   # 30 if config_timeout is None or 0
```

Practical importance: prevents `AttributeError` / `TypeError` on `None` without an explicit `if` check.

---

**Q6. What is the LEGB rule?**

Python resolves names by searching scopes in this order:

1. **L**ocal — inside the current function
2. **E**nclosing — any enclosing function scopes (for closures / nested functions)
3. **G**lobal — the module's top-level namespace
4. **B**uilt-in — Python's built-in names (`len`, `print`, `range`, …)

First match wins. If no scope contains the name, `NameError` is raised.

```python
x: int = 10           # Global

def outer() -> None:
    x: int = 20       # Enclosing

    def inner() -> None:
        print(x)      # Finds x=20 in Enclosing; never reaches Global

    inner()

outer()               # prints 20
```

---

**Q7. Why use `logging` instead of `print` in production code?**

| Feature | `logging` | `print` |
|---------|-----------|---------|
| Severity levels | DEBUG / INFO / WARNING / ERROR / CRITICAL | None |
| Filter at runtime | `--log-level WARNING` silences DEBUG/INFO | Must remove/comment out |
| Output destination | Configurable (file, stderr, remote, …) | stdout only |
| Structured output | Timestamps, module names, line numbers | Raw text |
| Production disable | Set root level to WARNING | Delete every print |

Example:
```python
import logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s: %(message)s")

logging.debug("This won't appear at WARNING level")  # filtered out
logging.warning("Disk space low: %d MB remaining", 42)  # appears
```

---

**Q8. Describe the four stages Python goes through when running `python script.py`.**

1. **Tokenize** — The source file is read as bytes and broken into tokens (keywords, identifiers, operators, literals, newlines).
2. **Parse** — The token stream is validated against Python's grammar and assembled into an **Abstract Syntax Tree** (AST) — a tree structure representing the program's logic.
3. **Compile** — The AST is compiled to **bytecode** (`.pyc` files cached in `__pycache__/`). Bytecode is a compact, platform-independent instruction set for the CPython VM.
4. **Execute** — The **CPython virtual machine** (PVM) interprets bytecode instructions one at a time using an evaluation stack.

You can inspect any of these stages:
```python
import ast, dis, py_compile

tree = ast.parse("x = 1 + 2")
print(ast.dump(tree, indent=2))   # see the AST

def add(a, b): return a + b
dis.dis(add)                       # see the bytecode
```

---

## Part 2 — Coding Challenge: `word_stats.py`

**Reference implementation:**

```python
"""word_stats.py — count and rank words in a text file."""
import logging
import string
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")


def tokenize(text: str) -> list[str]:
    """Lowercase the text, strip punctuation, split into words."""
    translator = str.maketrans("", "", string.punctuation)
    return text.lower().translate(translator).split()


def analyze(filepath: Path) -> dict[str, object]:
    """Read a file and return word statistics."""
    logging.debug("Reading %s", filepath)
    text: str = filepath.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("File is empty or contains only whitespace")
    words: list[str] = tokenize(text)
    counts: Counter[str] = Counter(words)
    return {
        "total": len(words),
        "unique": len(counts),
        "top5": counts.most_common(5),
    }


def write_stats(stats: dict[str, object], out_path: Path) -> None:
    """Write a plain-text summary to out_path."""
    lines: list[str] = [
        f"Total words: {stats['total']}",
        f"Unique words: {stats['unique']}",
        "Top 5 words:",
    ]
    for word, count in stats["top5"]:  # type: ignore[misc]
        lines.append(f"  {word}: {count}")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logging.debug("Wrote stats to %s", out_path)


def main() -> None:
    if len(sys.argv) != 2:
        logging.error("Usage: python word_stats.py <filename>")
        sys.exit(1)

    filepath = Path(sys.argv[1])
    if not filepath.exists():
        logging.error("File not found: %s", filepath)
        sys.exit(1)

    try:
        stats = analyze(filepath)
    except ValueError as e:
        logging.error("%s", e)
        sys.exit(1)

    out_path = filepath.with_name(filepath.stem + "_stats.txt")
    write_stats(stats, out_path)
    logging.info(
        "Total=%d  Unique=%d  Output=%s",
        stats["total"],
        stats["unique"],
        out_path,
    )


if __name__ == "__main__":
    main()
```

**Acceptance criteria checklist:**

- [ ] Reads the filename from `sys.argv[1]` — no hardcoded paths
- [ ] Handles missing file with a user-friendly error (exit code 1)
- [ ] Handles empty file gracefully (no crash)
- [ ] Strips punctuation before counting (e.g., `"hello,"` == `"hello"`)
- [ ] Comparison is case-insensitive (`"The"` == `"the"`)
- [ ] Writes `<stem>_stats.txt` alongside the input file
- [ ] Output contains total count, unique count, top 5 words with counts
- [ ] Uses `logging` — no bare `print` for diagnostics
- [ ] All functions have type annotations and docstrings
- [ ] `main()` guarded by `if __name__ == "__main__"`

---

## Part 3 — Rubric

| Dimension | Weight | Full marks criteria |
|-----------|--------|---------------------|
| Independent implementation | 40% | All acceptance criteria pass; no unexplained copy-paste |
| Conceptual understanding | 25% | Conceptual answers show genuine reasoning, not recitation |
| Debugging & problem-solving | 20% | Edge cases handled (empty file, missing file, bad input); correct exit codes |
| Code quality | 15% | Type hints, docstrings, `logging`, no bare `print`, functions ≤ 30 lines |

**Pass threshold: 75%**

**Score guide:**

| Score | Meaning |
|-------|---------|
| 90–100% | Ready for Week 2 |
| 75–89% | Pass — review weak areas before proceeding |
| 60–74% | Borderline — redo the coding challenge before moving on |
| < 60% | Repeat Week 1 deep-practice day before advancing |
