# Phase 1 · Week 01 · Day 5 — Strings, Slicing, and Formatting

> Objective: Master Python string operations, slicing syntax, and modern formatting with f-strings  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

## Concept

### String internals

A Python `str` is an **immutable sequence of Unicode code points** (not bytes). Each character maps to a Unicode code point; Python stores it internally in one of three compact formats (Latin-1, UCS-2, or UCS-4) depending on the highest code point present.

Because strings are immutable, every method that "changes" a string actually returns a **new string**. The original is untouched.

```python
s: str = "hello"
s.upper()   # returns "HELLO" — s is still "hello"
```

**String interning**: Python may reuse the same object for small or identifier-like strings as an optimization. You can observe it with `is`, but never rely on it — use `==` for equality.

---

### Indexing and slicing

Strings are sequences, so standard sequence indexing applies.

```
 H  e  l  l  o
 0  1  2  3  4    (positive indices)
-5 -4 -3 -2 -1   (negative indices — count from end)
```

```python
s: str = "Hello"
s[0]        # "H"
s[-1]       # "o"
s[1:4]      # "ell"   — start inclusive, stop exclusive
s[::2]      # "Hlo"   — every other character
s[::-1]     # "olleH" — reverse
```

Slice syntax: `s[start:stop:step]`. Omitted parts default to start=0, stop=len, step=1.

---

### Common string methods

```python
s = "  Hello, World!  "

s.strip()           # "Hello, World!"    — removes leading/trailing whitespace
s.lstrip()          # "Hello, World!  "  — left only
s.rstrip()          # "  Hello, World!"  — right only
s.lower()           # "  hello, world!  "
s.upper()           # "  HELLO, WORLD!  "
s.replace("World", "Python")   # "  Hello, Python!  "
s.strip().split(",")           # ["Hello", " World!"]
", ".join(["a", "b", "c"])     # "a, b, c"
s.startswith("  H")            # True
s.endswith("!  ")              # True
s.find("World")                # 9  (index of first match, -1 if not found)
s.count("l")                   # 3
```

---

### String formatting

**Old style (avoid):**
```python
"Hello, %s! You are %d." % ("Alice", 30)
```

**`.format()` (Python 3, still common):**
```python
"Hello, {}! Score: {:.2f}".format("Alice", 92.5)
```

**f-strings (preferred, Python 3.6+):**
```python
name: str = "Alice"
score: float = 92.5
f"Hello, {name}! Score: {score:.2f}"   # "Hello, Alice! Score: 92.50"
```

**`=` specifier (Python 3.8+) — great for debugging:**
```python
x: int = 42
f"{x=}"   # "x=42"
```

You can put any expression inside `{}`:
```python
f"{'hello'.upper()}"   # "HELLO"
f"{2 ** 10}"           # "1024"
```

---

### Multi-line and raw strings

```python
multi: str = """Line one
Line two
Line three"""

path: str = r"C:\Users\name\file.txt"  # raw: backslashes not treated as escapes
```

---

### Bytes and encoding

```python
s: str = "café"
b: bytes = s.encode("utf-8")    # b'caf\xc3\xa9'
s2: str  = b.decode("utf-8")    # "café"
```

UTF-8 is the default and the right choice for almost all cases.

## How it works internally

Strings in CPython are `PyUnicodeObject`. When you create a string, Python inspects the maximum code point:

- All characters ≤ U+00FF → **Latin-1** (1 byte/char).
- All characters ≤ U+FFFF → **UCS-2** (2 bytes/char).
- Otherwise → **UCS-4** (4 bytes/char).

This keeps memory tight for ASCII-heavy code. Indexing is O(1) because all slots are fixed width within a given string.

**Interning**: CPython automatically interns short strings that look like identifiers (letters, digits, underscores). `sys.intern(s)` forces interning for any string, useful for repeated dictionary key lookups. Still use `==` for equality; `is` is an identity check, not a value check.

**Immutability consequence**: building a string with `+=` in a loop creates N intermediate objects. Use `"".join(parts)` instead for large concatenations — it's O(n) total vs O(n²).

## Worked example

```python
"""
Parse a CSV-like line, clean each field, reformat with f-strings.
"""


def parse_record(line: str) -> dict[str, str | float]:
    """Parse 'name, score, category' CSV line into a typed dict."""
    parts: list[str] = line.split(",")

    name: str     = parts[0].strip().title()        # clean + capitalize
    score: float  = float(parts[1].strip())
    category: str = parts[2].strip().lower()

    return {"name": name, "score": score, "category": category}


def format_record(record: dict[str, str | float]) -> str:
    """Render the record as a readable string."""
    name: str     = str(record["name"])
    score: float  = float(record["score"])
    category: str = str(record["category"])

    # f-string: .2f for 2 decimal places, <20 for left-align in 20 chars
    return f"{name:<20} | {score:6.2f} | {category}"


if __name__ == "__main__":
    raw_lines: list[str] = [
        "  alice smith ,  88.5, math ",
        "BOB JONES,  91.0,  science",
        " carol ,77, math",
    ]

    for line in raw_lines:
        rec = parse_record(line)
        print(format_record(rec))
    # Alice Smith          |  88.50 | math
    # Bob Jones            |  91.00 | science
    # Carol                |  77.00 | math
```

## Exercises

1. Given `s = "  Hello, World!  "`, strip whitespace, convert to lowercase, and split on `","`.
2. Extract the word `"World"` from `s = "Hello, World!"` using slicing (find the start and end indices manually first).
3. Write an f-string that formats a `name: str` and a `score: float` to 2 decimal places, producing output like `"Alice: 92.50"`.
4. Reverse the string `"Python"` using slicing. No loops, no built-in `reversed()`.
5. Encode `"café"` to UTF-8 bytes and decode it back. Print both the `bytes` object and the recovered `str`.

(Solutions: ../solutions/week-01/day-5.md)

## Self-check

- [ ] I can explain why `s.upper()` doesn't modify `s`.
- [ ] I can use negative indices and `s[start:stop:step]` confidently.
- [ ] I can choose between `.format()` and f-strings and explain the `=` specifier.
- [ ] I know why `"".join(parts)` is preferred over `+=` in loops.
- [ ] I can encode a string to UTF-8 bytes and decode it back.

---

**Supplementary:** https://docs.python.org/3.12/library/stdtypes.html#text-sequence-type-str
