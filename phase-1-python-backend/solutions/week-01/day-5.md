# Solutions — Phase 1 · Week 01 · Day 5 — Strings, Slicing & Formatting

> Answer key. Do not open until you have attempted the exercises.

---

## Exercise 1 — Strip, lower, split

```python
s = "  Hello, World!  "
result: list[str] = s.strip().lower().split(",")
print(result)  # ['hello', ' world!']
```

**Why it works:** Method calls chain left-to-right. `strip()` removes leading/trailing whitespace first, then `lower()` downcases the entire string, then `split(",")` cuts at every comma. The space before "world" remains because `split(",")` only splits on commas, not spaces.

---

## Exercise 2 — Slice "World" from "Hello, World!"

```python
s: str = "Hello, World!"
# Index:  0123456789...
# H=0, e=1, l=2, l=3, o=4, ,=5, (space)=6, W=7, o=8, r=9, l=10, d=11
word: str = s[7:12]
print(word)  # World
```

**Why it works:** Python slices are `[start:stop]` — inclusive at start, exclusive at stop. Character at index 12 is `!`, so `s[7:12]` captures indices 7–11 which spell "World". You can also use `s.find("World")` to discover the index programmatically.

---

## Exercise 3 — f-string with float formatting

```python
name: str = "Alice"
score: float = 92.5
formatted: str = f"{name}: {score:.2f}"
print(formatted)  # Alice: 92.50
```

**Why it works:** The format spec `:.2f` means fixed-point notation with exactly 2 decimal places. Python's f-string mini-language is a subset of the older `format()` spec. The trailing `0` is added automatically to satisfy the 2-decimal requirement.

---

## Exercise 4 — Reverse a string with slice

```python
s: str = "Python"
reversed_s: str = s[::-1]
print(reversed_s)  # nohtyP
```

**Why it works:** The third slice argument is the step. `[::-1]` means "start from the end, move backward by 1 each time." This idiom works for any sequence (list, tuple, bytes) — not just strings.

---

## Exercise 5 — UTF-8 encode/decode

```python
s: str = "café"
encoded: bytes = s.encode("utf-8")
print(encoded)         # b'caf\xc3\xa9'
decoded: str = encoded.decode("utf-8")
print(decoded)         # café
print(len(s))          # 4  (4 Unicode code points)
print(len(encoded))    # 5  (5 bytes — "é" takes 2)
```

**Why it works:** Unicode strings are abstract sequences of code points. UTF-8 is a variable-width encoding: ASCII characters (U+0000–U+007F) are 1 byte each; "é" (U+00E9) encodes as 2 bytes: `0xC3 0xA9`. This is why `len(s) != len(encoded)` for non-ASCII strings. Always specify `encoding="utf-8"` explicitly when opening files to avoid platform-dependent defaults.

---

## Bonus — String immutability demonstration

```python
s: str = "hello"
# s[0] = "H"  # raises TypeError: 'str' object does not support item assignment

# Correct way — build a new string:
s = "H" + s[1:]
print(s)  # Hello
```

Strings are immutable in Python. Every "modification" creates a new string object and rebinds the name.
