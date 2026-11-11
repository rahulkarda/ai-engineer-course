# Solutions — Phase 1 · Week 01 · Day 2 — Variables, Primitive Types, Type Conversion

---

## Exercise 1 — Print each value's type and identity

```python
import logging

logging.basicConfig(level=logging.DEBUG, format="%(message)s")

x: int = 42
y: float = 3.14
z: bool = True
n: None = None

for name, val in [("x", x), ("y", y), ("z", z), ("n", n)]:
    logging.debug("%s  type=%-6s  id=%d", name, type(val).__name__, id(val))
```

**Expected output:**
```
x  type=int     id=4303342368
y  type=float   id=4385621744
z  type=bool    id=4299472192
n  type=NoneType  id=4299271600
```

Why it works: `type(val).__name__` returns the class name as a string. `id(val)` returns the object's memory address (CPython implementation). Every Python object has three attributes at runtime: type, identity (id), and value.

---

## Exercise 2 — Truthiness of common values

Run each in a REPL (`python -i`) and reason through the output:

```python
>>> bool(0)
False       # zero integer → falsy

>>> bool("")
False       # empty string → falsy

>>> bool([])
False       # empty list → falsy

>>> bool(None)
False       # None is always falsy

>>> bool(0.0)
False       # zero float → falsy

>>> bool(-1)
True        # any non-zero number → truthy

>>> bool("0")
True        # non-empty string → truthy (contains the character "0")

>>> bool([0])
True        # non-empty list → truthy (even if it contains a falsy element)
```

Why it works: Python calls `__bool__()` (or `__len__()` if `__bool__` is absent) on an object to determine its truth value. Numbers return False only for zero; containers return False only when empty; `None` always returns False.

---

## Exercise 3 — Type conversion chain

```python
s: str = "42"

# str → int
i: int = int(s)         # 42
print(type(i), i)       # <class 'int'> 42

# int → float
f: float = float(i)     # 42.0
print(type(f), f)       # <class 'float'> 42.0

# float → str
back: str = str(f)      # "42.0"
print(type(back), back) # <class 'str'> 42.0
```

Note: `int("42.0")` raises `ValueError` — you must go through `float` first:
```python
int(float("42.0"))  # 42
```

Why it works: Python's built-in type constructors (`int()`, `float()`, `str()`, `bool()`) are explicit conversion functions. They do not silently coerce — if the conversion is impossible they raise a `ValueError` or `TypeError`, which is safer than silent data loss.

---

## Exercise 4 — Integer interning explanation

```python
# Small integers are cached (CPython implementation detail)
a: int = 5
b: int = 5
print(a is b)    # True — same object (cached)
print(id(a), id(b))  # same id

# Large integers are NOT cached
x: int = 1000
y: int = 1000
print(x is y)    # False (usually) — different objects
print(id(x), id(y))  # different ids
```

**Why it works:** CPython pre-allocates integer objects for the range -5 to 256 at interpreter startup and reuses them. Every variable bound to `5` points to the same singleton. For integers outside this range, a fresh object is allocated each time a literal appears in a statement.

**Critical point:** This is a CPython implementation detail — other Python runtimes (PyPy, Jython) may behave differently. Never use `is` to compare integer values; always use `==`.

---

## Exercise 5 — Type hint annotations

```python
# Variables with explicit type annotations
name: str = "Alice"
age: int = 30
score: float | None = None   # Python 3.10+ union syntax

# Annotations on their own (no value yet)
label: str
count: int

# Verify annotations are stored (not enforced at runtime)
import __annotations__  # not needed — just illustrative

# mypy / pyright will reject this at analysis time:
# name = 42   # error: Incompatible types in assignment (str vs int)
```

Why it works: Type annotations in Python are syntactic metadata. At runtime `name: str = "Alice"` behaves identically to `name = "Alice"` — the annotation is stored in `__annotations__` but Python does not enforce it. Static type checkers (mypy, pyright, Pylance) read the annotations and flag mismatches before you run the code.

---

## Self-check answers

- **What are the five built-in scalar types?** `int`, `float`, `bool`, `str`, `NoneType`.
- **Why is `bool` a subclass of `int`?** `True == 1` and `False == 0` — this is the historical C heritage. `isinstance(True, int)` returns `True`.
- **When does `int(x)` raise?** When `x` is a string that cannot be parsed as an integer (e.g., `"3.14"`, `"hello"`).
