# Phase 1 · Week 01 · Day 6 — I/O, Debugging, and the Python Execution Model

> Objective: Read/write files with pathlib, debug with logging and pdb, and understand how Python runs your code  ·  Time: ~4 hr (Saturday deep practice)  ·  Prereqs: [[day-5]]

---

## Part 1 — I/O (~45 min)

### `input()` and why it always returns `str`

```python
raw: str = input("Enter a number: ")   # always str
n: int = int(raw)                      # convert explicitly
```

Python cannot know what type you want, so `input()` gives you the raw text. Always validate and convert before use.

---

### File I/O with `open()`

**Always use a context manager (`with`)** — it guarantees the file is closed even if an exception occurs.

```python
# Read entire file as string
with open("data.txt", "r", encoding="utf-8") as f:
    content: str = f.read()

# Read line by line (memory-efficient for large files)
with open("data.txt", "r", encoding="utf-8") as f:
    for line in f:
        print(line.rstrip("\n"))

# Write (overwrites existing file)
with open("output.txt", "w", encoding="utf-8") as f:
    f.write("Hello\n")

# Append
with open("log.txt", "a", encoding="utf-8") as f:
    f.write("new entry\n")

# Binary read
with open("image.png", "rb") as f:
    data: bytes = f.read()
```

Modes:
| Mode | Meaning |
|------|---------|
| `"r"` | Read text (default) |
| `"w"` | Write text, truncate |
| `"a"` | Append text |
| `"rb"` | Read binary |
| `"wb"` | Write binary |

Always pass `encoding="utf-8"` for text files — the default varies by platform.

---

### `pathlib.Path` — the modern way

`pathlib` gives you an object-oriented API that works identically on Windows, macOS, and Linux.

```python
from pathlib import Path

p: Path = Path("data/results.txt")

p.exists()              # True/False
p.parent                # Path("data")
p.stem                  # "results"
p.suffix                # ".txt"
p.name                  # "results.txt"

# Read/write (context manager handled internally)
text: str = p.read_text(encoding="utf-8")
p.write_text("content", encoding="utf-8")

# Create directories
p.parent.mkdir(parents=True, exist_ok=True)

# Iterate over files
for f in Path(".").glob("*.txt"):
    print(f)
```

Prefer `pathlib.Path` over `os.path` in all new code.

---

### Reading/writing CSV manually

```python
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV with header row into list of dicts."""
    lines: list[str] = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    headers: list[str] = [h.strip() for h in lines[0].split(",")]
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        values: list[str] = [v.strip() for v in line.split(",")]
        rows.append(dict(zip(headers, values)))
    return rows
```

---

## Part 2 — Debugging (~45 min)

### Reading a traceback

```
Traceback (most recent call last):
  File "main.py", line 12, in <module>    # outermost call
    result = divide(10, 0)
  File "main.py", line 7, in divide       # innermost frame <-- start here
    return a / b
ZeroDivisionError: division by zero       # exception type and message
```

**Rule**: Start at the **innermost frame** (bottom of the traceback). That's where the error actually occurred. Then read upward to understand how you got there.

---

### Common beginner errors

| Error | Cause | Fix |
|-------|-------|-----|
| `NameError: name 'x' is not defined` | Used variable before assignment | Check spelling, assignment order |
| `TypeError: unsupported operand type` | Wrong types for operator | Check and convert types |
| `IndexError: list index out of range` | Index >= len(list) | Check bounds |
| `KeyError: 'name'` | Key not in dict | Use `.get()` or check first |
| `ValueError: invalid literal for int()` | Non-numeric string to `int()` | Validate input |
| `IndentationError` | Inconsistent spaces/tabs | Use 4 spaces, no tabs |

---

### `logging` — use this instead of `print`

`print` is for user-facing output. `logging` is for developer/operational output.

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

logger.debug("detailed trace")     # only shown at DEBUG level
logger.info("normal operation")    # INFO and above
logger.warning("something odd")    # WARNING and above
logger.error("something failed")   # ERROR and above
logger.critical("fatal error")     # CRITICAL
```

**Why not `print`?**
- Can route to files, syslog, cloud log services without changing code.
- Each message carries level, timestamp, module name.
- Can silence entire modules by setting their log level.
- In production, `print` goes to stdout with no metadata.

---

### `pdb` — the interactive debugger

```bash
python -m pdb script.py
```

Or drop a breakpoint inline:
```python
breakpoint()   # Python 3.7+ — drops into pdb at this line
```

Key commands:

| Command | Action |
|---------|--------|
| `l` | List source around current line |
| `n` | Next line (step over) |
| `s` | Step into function call |
| `p expr` | Print value of expression |
| `pp expr` | Pretty-print |
| `c` | Continue until next breakpoint |
| `q` | Quit |
| `b 15` | Set breakpoint at line 15 |
| `h` | Help |

---

## Part 3 — Python Execution Model (~30 min)

### How Python runs a `.py` file

```
.py source file
     |
     v
[Tokenizer]  — splits source into tokens (keywords, names, operators, literals)
     |
     v
[Parser]     — builds an Abstract Syntax Tree (AST)
     |
     v
[Compiler]   — compiles AST to bytecode (.pyc files in __pycache__)
     |
     v
[CPython VM] — interprets bytecode instructions one at a time
```

You can inspect bytecode with the `dis` module:

```python
import dis

def add(a: int, b: int) -> int:
    return a + b

dis.dis(add)
# LOAD_FAST  0 (a)
# LOAD_FAST  1 (b)
# BINARY_OP  0 (+)
# RETURN_VALUE
```

Bytecode is not machine code — it runs on the CPython virtual machine, not directly on your CPU. That's why Python is slower than C but portable and easy to introspect.

---

### LEGB scope rule

When Python sees a name, it searches these scopes in order:

1. **L**ocal — current function body
2. **E**nclosing — any enclosing function (for nested functions)
3. **G**lobal — module level
4. **B**uilt-in — `builtins` module (`len`, `print`, `range`, ...)

```python
x: int = 10          # Global

def outer() -> None:
    x: int = 20      # Enclosing (for inner)

    def inner() -> None:
        x: int = 30  # Local
        print(x)     # 30 — Local wins

    inner()
    print(x)         # 20 — Local of outer

outer()
print(x)             # 10 — Global unchanged
```

**`global`** declares that an assignment inside a function targets the module-level name:

```python
counter: int = 0

def increment() -> None:
    global counter
    counter += 1
```

**`nonlocal`** targets the enclosing (non-global) scope:

```python
def make_counter():
    count: int = 0
    def increment() -> None:
        nonlocal count
        count += 1
    return increment
```

Overusing `global` is a code smell — prefer passing values as arguments and returning results.

---

## Worked example

```python
"""
Word frequency counter: reads a text file, counts words,
logs progress, writes results. Demonstrates pathlib, logging,
context managers, and string processing together.
"""
import logging
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


def count_words(path: Path) -> dict[str, int]:
    """Return word -> count mapping for a text file."""
    if not path.exists():
        logger.error("File not found: %s", path)
        return {}

    logger.info("Reading %s", path)
    text: str = path.read_text(encoding="utf-8")

    # Normalise and split
    words: list[str] = text.lower().split()
    # Strip punctuation from each word
    cleaned: list[str] = [w.strip(".,!?;:\"'()-") for w in words]

    freq: dict[str, int] = {}
    for word in cleaned:
        if word:
            freq[word] = freq.get(word, 0) + 1

    logger.info("Found %d unique words", len(freq))
    return freq


def write_results(freq: dict[str, int], out_path: Path) -> None:
    """Write sorted word frequencies to a file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Sort by frequency descending, then alphabetically
    sorted_items = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))

    lines: list[str] = [f"{word},{count}" for word, count in sorted_items]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Results written to %s", out_path)


if __name__ == "__main__":
    input_path = Path("sample.txt")
    output_path = Path("results/word_freq.csv")

    # Create a sample file for demo purposes
    if not input_path.exists():
        input_path.write_text(
            "the quick brown fox jumps over the lazy dog "
            "the dog barked at the fox",
            encoding="utf-8",
        )

    freq = count_words(input_path)
    write_results(freq, output_path)

    # Print top 5
    top5 = sorted(freq.items(), key=lambda kv: -kv[1])[:5]
    for word, count in top5:
        logger.info("  %s: %d", word, count)
```

---

## Exercises

Ten exercises integrating everything from this week. These are harder — expect 10–15 minutes each.

1. Write a script that reads a file path from `sys.argv[1]`, reads the file, and prints its line count. Handle `FileNotFoundError` gracefully with a clear error message.

2. Rewrite exercise 1 using `pathlib.Path` and `logging` instead of `print`. Log the line count at `INFO` level and errors at `ERROR` level.

3. Write a function `above_average(numbers: list[float]) -> list[float]` that returns a new list containing only the values above the mean. Include a type hint. Test it with `[1.0, 2.0, 9.0, 4.0, 5.0]`.

4. Write a one-line list comprehension that extracts all words longer than 4 characters from the string `"the quick brown fox jumps over the lazy dog"`.

5. Fix the following broken code snippets and explain each error in a comment:
   ```python
   # Snippet A
   for i in range(10) print(i)

   # Snippet B
   x = input("Enter a number: ")
   print(x + 1)
   ```

6. Add `breakpoint()` to a short script of your choice (at least 5 lines). Run it, use `n` to step through each line, `p` to inspect variables, and write down the exact commands you used.

7. Write a function `celsius_to_fahrenheit(c: float) -> float` that converts Celsius to Fahrenheit (`F = c * 9/5 + 32`). Call it with 0, 100, and -40 and print each result.

8. Using only string methods (no regex, no `import`), extract the domain from an email address like `"user@example.com"`. Your function signature: `def get_domain(email: str) -> str`.

9. Given `lines: list[str] = ["hello", "world", "quit", "ignored"]`, write a `while` loop that processes each line by index, stops when it encounters `"quit"`, and accumulates all preceding lines into a `result` list.

10. Write a script that: (a) creates `output.txt` and writes the numbers 1–100 one per line, (b) reads it back, (c) prints only the lines whose integer value is divisible by 7. Use `pathlib.Path` throughout.

(Solutions: ../solutions/week-01/day-6.md)

## Self-check

- [ ] I can explain why `input()` always returns a string.
- [ ] I can open, read, write, and append files using `with open(...)` and `pathlib.Path`.
- [ ] I can read a Python traceback and identify the root cause.
- [ ] I can explain the five logging levels and why `logging` is better than `print` for debugging.
- [ ] I can use `pdb` to set breakpoints and inspect state.
- [ ] I can describe the source → tokenize → parse → compile → execute pipeline.
- [ ] I can apply the LEGB rule to resolve a name lookup question.

---

**Supplementary:**
- https://docs.python.org/3.12/library/pdb.html
- https://docs.python.org/3.12/library/logging.html
- https://docs.python.org/3.12/library/pathlib.html
