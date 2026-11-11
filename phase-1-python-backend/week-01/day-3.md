# Phase 1 · Week 01 · Day 3 — Operators, Expressions, Mutable vs Immutable Objects

> Objective: Master Python's operator set, understand short-circuit evaluation, and internalize the mutable/immutable distinction that governs aliasing bugs.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

## Concept

### Arithmetic Operators

| Operator | Name | Example | Result |
|----------|------|---------|--------|
| `+` | Addition | `3 + 2` | `5` |
| `-` | Subtraction | `7 - 4` | `3` |
| `*` | Multiplication | `3 * 4` | `12` |
| `/` | True division | `7 / 2` | `3.5` (always float) |
| `//` | Floor division | `7 // 2` | `3` (truncates toward −∞) |
| `%` | Modulus | `7 % 2` | `1` (remainder) |
| `**` | Exponentiation | `2 ** 10` | `1024` |

`/` always returns a `float`. `//` returns an `int` when both operands are `int`, and a `float` when either is a `float`.

```python
10 // 3    # 3
-10 // 3   # -4  (floor, not truncate toward zero)
10 % 3     # 1
```

---

### Comparison Operators

```python
==   !=   <   >   <=   >=
```

These always return `bool`. Comparisons can be chained:
```python
1 < x < 10   # equivalent to (1 < x) and (x < 10) — evaluated efficiently
```

---

### Logical Operators and Short-Circuit Evaluation

`and` and `or` do **not** return `True`/`False` — they return one of their operands.

- `a and b` — evaluates `a`. If falsy, returns `a` immediately (short-circuits, never evaluates `b`). If truthy, evaluates and returns `b`.
- `a or b` — evaluates `a`. If truthy, returns `a` immediately. If falsy, evaluates and returns `b`.
- `not a` — returns `True` if `a` is falsy, `False` otherwise. Always returns a `bool`.

```python
None and None.upper()   # None — short-circuits, .upper() never called
0 or "default"          # "default" — 0 is falsy, returns second operand
"hello" or "default"    # "hello" — truthy, returns first operand
```

Short-circuit evaluation is not just an optimization — it is how you safely guard against `None` dereferences.

---

### Identity vs Equality

| Operator | Checks | Example |
|----------|--------|---------|
| `==` | Value equality | `[1,2] == [1,2]` → `True` |
| `is` | Object identity (same object in memory) | `[1,2] is [1,2]` → `False` |

Two objects can be **equal** (same value) without being **identical** (same object).

Use `is` only for `None`, `True`, `False`, and enum singletons:
```python
if value is None:   # correct
if value == None:   # works, but triggers linter warning
```

---

### Augmented Assignment

```python
x = 10
x += 3    # x = x + 3  →  13
x -= 1    # 12
x *= 2    # 24
x //= 5   # 4
x **= 3   # 64
```

For **mutable** objects, augmented assignment mutates in-place:
```python
items = [1, 2]
items += [3]   # mutates items in-place (same object)
```

For **immutable** objects, it rebinds the name to a new object:
```python
s = "hello"
s += " world"   # creates a new str, rebinds s
```

---

### Mutable vs Immutable

**Immutable** types cannot be changed after creation. Operations that appear to "modify" them actually create new objects.

| Type | Immutable? |
|------|-----------|
| `int`, `float`, `bool`, `complex` | Yes |
| `str` | Yes |
| `tuple` | Yes |
| `frozenset` | Yes |
| `list` | **No** — mutable |
| `dict` | **No** — mutable |
| `set` | **No** — mutable |

**The aliasing trap** — mutable objects:
```python
a = [1, 2, 3]
b = a          # b and a point to the SAME list object
b.append(4)
print(a)       # [1, 2, 3, 4] — a was mutated through b
```

**The same trap does not exist for immutable objects:**
```python
x = "hello"
y = x
y += " world"  # creates a new string, rebinds y — x is unchanged
print(x)       # "hello"
```

---

### copy vs deepcopy

```python
import copy

original = [1, [2, 3], 4]

shallow = copy.copy(original)
# New outer list, but inner lists are still shared references

deep = copy.deepcopy(original)
# Fully independent copy — inner lists are new objects too

shallow[1].append(99)
print(original)   # [1, [2, 3, 99], 4] — inner list shared!

deep[1].append(88)
print(original)   # [1, [2, 3, 99], 4] — unchanged — deep copy is independent
```

Use `copy.copy()` when the object contains only primitives (no nested mutables). Use `copy.deepcopy()` when the object contains nested mutable containers.

## How it works internally

**Immutability in CPython** means the object's internal state cannot change after `__init__`. A `str` object stores its characters in a fixed-length C array. There is no method that modifies this array in-place — every string "modification" allocates a new `str` object and returns it.

This immutability guarantee is what allows CPython to safely intern strings and small integers (see Day 2): if they could be mutated, sharing them would be catastrophic.

**Mutable objects** like `list` are backed by a dynamic C array. `list.append()` writes to this array in-place (or reallocates if capacity is exceeded). The object's address in memory stays the same; only its contents change. This is why two names pointing to the same list both "see" the mutation — they share the address, and the object at that address changed.

**`and`/`or` returning operands (not booleans)** is a deliberate design: it allows idiomatic patterns like `x = config.get("key") or "default"` without requiring an if-else. The CPython bytecode uses `JUMP_IF_FALSE_OR_POP` / `JUMP_IF_TRUE_OR_POP` instructions to implement this without evaluating both operands.

## Worked example

```python
# day3_operators.py — aliasing trap and copy semantics
import copy
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# --- Arithmetic ---
log.info("17 // 5 = %d", 17 // 5)   # 3
log.info("17 %% 5 = %d", 17 % 5)    # 2
log.info("2 ** 10 = %d", 2 ** 10)   # 1024

# --- Short-circuit ---
value: str | None = None
result: str | None = value and value.upper()   # safe — never calls .upper() on None
log.info("short-circuit result: %r", result)   # None

fallback: str = value or "default"
log.info("or fallback: %r", fallback)          # "default"

# --- Aliasing trap ---
original: list[int] = [1, 2, 3]
alias: list[int] = original        # same object
alias.append(4)
log.info("original after alias.append(4): %s", original)  # [1, 2, 3, 4]

# --- Shallow copy ---
original2: list[list[int]] = [[1, 2], [3, 4]]
shallow: list[list[int]] = copy.copy(original2)
shallow[0].append(99)
log.info("original2 after shallow copy mutation: %s", original2)  # [[1, 2, 99], [3, 4]]

# --- Deep copy ---
original3: list[list[int]] = [[1, 2], [3, 4]]
deep: list[list[int]] = copy.deepcopy(original3)
deep[0].append(99)
log.info("original3 after deep copy mutation: %s", original3)     # [[1, 2], [3, 4]] unchanged

# --- Immutable str: item assignment fails ---
word: str = "hello"
try:
    word[0] = "H"  # type: ignore
except TypeError as exc:
    log.info("TypeError: %s", exc)

# Correct approach: string slicing
corrected: str = "H" + word[1:]
log.info("corrected string: %s", corrected)  # Hello
```

Run:
```bash
python3 day3_operators.py
```

## Exercises

1. In the REPL or a script, evaluate `17 // 5`, `17 % 5`, and `2 ** 10`. Write the results and explain why `17 // 5` equals `3` rather than `3.4`.

2. Write `x: str | None = None; result = x and x.upper()`. Log `result`. Explain in one sentence why no `AttributeError` is raised even though `None` has no `.upper()` method.

3. Create `a: list[int] = [1, 2, 3]` and `b = a`. Append `4` to `b`. Log both `a` and `b` and confirm they are identical. Fix the aliasing by replacing `b = a` with `b = copy.copy(a)` and show `a` is unaffected.

4. Try `"hello"[0] = "H"` in the REPL. Record the exact error. Then produce the string `"Hello"` from `"hello"` using string slicing (no loops).

5. Create two string variables with identical content via separate literals:
   ```python
   s1 = "hello world " * 1000
   s2 = "hello world " * 1000
   ```
   Check `s1 == s2` and `s1 is s2`. Explain the result.

(Solutions: ../solutions/week-01/day-3.md)

## Self-check

- [ ] I can predict the result of `//`, `%`, and `**` without running Python
- [ ] I can explain short-circuit evaluation and use it to guard `None` dereferences
- [ ] I know the difference between `is` and `==` and when to use each
- [ ] I can identify the aliasing trap and fix it with `copy.copy()` or `copy.deepcopy()`
- [ ] I know which built-in types are mutable and which are immutable

---

## Supplementary

- [Python 3.12 expressions reference](https://docs.python.org/3.12/reference/expressions.html)
- [Python 3.12 operator precedence table](https://docs.python.org/3.12/reference/expressions.html#operator-precedence)
- [`copy` module docs](https://docs.python.org/3.12/library/copy.html)
