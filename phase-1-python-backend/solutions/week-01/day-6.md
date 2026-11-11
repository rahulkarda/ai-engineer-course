# Solutions — Phase 1 · Week 01 · Day 6 — I/O, Debugging & Execution Model

> Answer key for the Saturday deep-practice session. Attempt all exercises before opening.

---

## Exercise 1 — File line counter (basic)

```python
import sys


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <filename>")
        sys.exit(1)
    filepath: str = sys.argv[1]
    try:
        with open(filepath) as f:
            lines: list[str] = f.readlines()
        print(f"Line count: {len(lines)}")
    except FileNotFoundError:
        print(f"Error: file not found: {filepath}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

**Key point:** `sys.argv[0]` is always the script name; arguments start at index 1.

---

## Exercise 2 — File line counter (pathlib + logging)

```python
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def count_lines(path: Path) -> int:
    """Return number of lines in a text file."""
    lines: list[str] = path.read_text(encoding="utf-8").splitlines()
    logging.debug("Read %d lines from %s", len(lines), path)
    return len(lines)


def main() -> None:
    if len(sys.argv) != 2:
        logging.error("Usage: %s <filename>", sys.argv[0])
        sys.exit(1)
    p = Path(sys.argv[1])
    if not p.exists():
        logging.error("File not found: %s", p)
        sys.exit(1)
    logging.info("Line count: %d", count_lines(p))


if __name__ == "__main__":
    main()
```

**Why prefer `pathlib`:** `Path` objects carry type information, support `/` concatenation, and integrate cleanly with `read_text()` / `write_text()`. `logging` lets you flip verbosity via `--log-level` without touching the code.

---

## Exercise 3 — Values above average

```python
def above_average(numbers: list[float]) -> list[float]:
    """Return elements strictly greater than the mean."""
    if not numbers:
        return []
    avg: float = sum(numbers) / len(numbers)
    return [x for x in numbers if x > avg]


# Verification
assert above_average([1, 2, 3, 4, 5]) == [4, 5]
assert above_average([]) == []
assert above_average([7, 7, 7]) == []  # nothing above the mean when all equal
```

**Why guard against empty list:** `sum([]) / len([])` raises `ZeroDivisionError`. Always handle the empty-collection edge case first.

---

## Exercise 4 — Words longer than 4 characters

```python
sentence: str = "The quick brown fox jumps over the lazy dog"
long_words: list[str] = [w for w in sentence.split() if len(w) > 4]
print(long_words)  # ['quick', 'brown', 'jumps']
```

**Note:** "over" has 4 characters and is excluded (`> 4`, not `>= 4`). "lazy" has 4 characters — also excluded. Double-check boundary conditions when the spec says "longer than N".

---

## Exercise 5 — Fix broken code

**Bug 1:**
```python
# Broken
for i in range(10) print(i)

# Fixed — missing colon terminates the for-statement header
for i in range(10):
    print(i)
```

**Bug 2:**
```python
# Broken
x = input("num: ")
print(x + 1)  # TypeError: can only concatenate str (not "int") to str

# Fixed — input() always returns str; convert before arithmetic
x_str: str = input("num: ")
x: int = int(x_str)
print(x + 1)
```

**Root cause:** `input()` returns a string regardless of what the user types. Type conversion is always the programmer's responsibility.

---

## Exercise 6 — pdb walkthrough

```
python -m pdb myscript.py
```

| Command | Meaning |
|---------|---------|
| `l` (list) | Show current source context (11 lines around current line) |
| `n` (next) | Execute the next line; **step over** function calls |
| `s` (step) | Execute the next line; **step into** function calls |
| `p expr` | Evaluate and print an expression (`p my_list`) |
| `pp expr` | Pretty-print (useful for dicts/lists) |
| `b 42` | Set breakpoint at line 42 |
| `c` (continue) | Run until the next breakpoint |
| `q` (quit) | Exit the debugger |

**VS Code alternative:** Set a breakpoint by clicking the gutter, then press F5. The Debug toolbar gives the same step-over / step-into controls visually.

---

## Exercise 7 — celsius_to_fahrenheit

```python
def celsius_to_fahrenheit(c: float) -> float:
    """Convert Celsius to Fahrenheit: F = C × 9/5 + 32."""
    return c * 9 / 5 + 32


# Assertions double as documentation
assert celsius_to_fahrenheit(0) == 32.0    # water freezes
assert celsius_to_fahrenheit(100) == 212.0  # water boils
assert celsius_to_fahrenheit(-40) == -40.0  # scales coincide at -40
assert celsius_to_fahrenheit(37) == 98.6    # body temperature
```

**Why `9 / 5` not `9 // 5`:** Integer floor division `9 // 5 == 1`, not `1.8`. Always use true division (`/`) for fractional coefficients.

---

## Exercise 8 — Extract email domain

```python
def extract_domain(email: str) -> str:
    """Return the domain part of an email address."""
    parts: list[str] = email.split("@")
    if len(parts) != 2:
        raise ValueError(f"Invalid email address: {email!r}")
    return parts[1]


print(extract_domain("user@example.com"))    # example.com
print(extract_domain("a.b.c@mail.co.uk"))   # mail.co.uk
```

**Why validate `len(parts) != 2`:** A valid email has exactly one `@`. Malformed inputs like `"nodomain"` or `"two@@at"` would silently return wrong results without this guard.

---

## Exercise 9 — Simulated stdin accumulator

```python
lines: list[str] = ["hello", "world", "quit", "ignored"]
accumulated: list[str] = []

for line in lines:
    if line == "quit":
        break          # stop processing — "ignored" is never seen
    accumulated.append(line)

print(accumulated)  # ['hello', 'world']
```

**Real-world pattern:** Replace the list with `sys.stdin` to process a live stream:
```python
import sys
for line in sys.stdin:
    line = line.rstrip("\n")
    if line == "quit":
        break
    accumulated.append(line)
```

---

## Exercise 10 — Write numbers 1–100, read back multiples of 7

```python
from pathlib import Path

out: Path = Path("output.txt")

# Write
out.write_text("\n".join(str(i) for i in range(1, 101)), encoding="utf-8")

# Read and filter
for line in out.read_text(encoding="utf-8").splitlines():
    n: int = int(line)
    if n % 7 == 0:
        print(n, end=" ")
# 7 14 21 28 35 42 49 56 63 70 77 84 91 98
```

**Why `range(1, 101)`:** `range` is exclusive at the stop value; `range(1, 101)` produces 1 through 100 inclusive.

---

## Bonus — Execution model walkthrough

When you run `python script.py`:

1. **Tokenizer** — reads source bytes, produces a token stream (`NAME`, `OP`, `NUMBER`, …)
2. **Parser** — builds an Abstract Syntax Tree (AST) from the token stream
3. **Compiler** — transforms the AST into bytecode; caches it as `__pycache__/script.cpython-312.pyc`
4. **CPython VM** — executes bytecode instructions one by one on the evaluation stack

You can inspect the bytecode yourself:
```python
import dis
import py_compile

# Compile a small function
def add(a: int, b: int) -> int:
    return a + b

dis.dis(add)
# LOAD_FAST 0 (a)
# LOAD_FAST 1 (b)
# BINARY_OP  (+)
# RETURN_VALUE
```
