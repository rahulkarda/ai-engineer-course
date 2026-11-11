# Phase 1 · Week 01 · Day 2 — Variables, Primitive Types, Type Conversion

> Objective: Understand Python's name→object model, primitive types, type introspection, conversion rules, and type hints.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

## Concept

### Variables: Names, Not Slots

In C or Java, a variable is a named memory slot — the value lives at a fixed address and the name directly addresses that slot.

Python works differently: **a variable is a name that points to an object**. The object lives on the heap; the name is a binding in a namespace dictionary.

```python
x = 42       # creates int object 42, binds name 'x' to it
x = "hello"  # creates str object, rebinds 'x' to the new object
             # the int 42 is now unreferenced (eligible for GC)
```

This means:
- A variable can be rebound to any type at any time.
- Multiple names can point to the same object (aliasing).
- The object's identity (its address) is fixed; the name's binding can change.

---

### Primitive Types

| Type | Example | Notes |
|------|---------|-------|
| `int` | `42`, `-7`, `0` | Arbitrary precision — no overflow |
| `float` | `3.14`, `-0.5` | IEEE 754 double (64-bit) |
| `bool` | `True`, `False` | Subclass of `int`; `True == 1`, `False == 0` |
| `NoneType` | `None` | Singleton; represents absence of value |

```python
x: int = 42
y: float = 3.14
flag: bool = True
value: None = None
```

`bool` is a subclass of `int`, so `True + True == 2` is valid (though rarely intentional).

---

### Type Introspection

Three built-ins for inspecting objects at runtime:

```python
type(42)           # <class 'int'>
type(3.14)         # <class 'float'>
isinstance(True, int)   # True  (bool is a subclass of int)
isinstance(True, bool)  # True
id(42)             # integer: the object's memory address
```

Use `isinstance()` over `type()` for type checks — it respects inheritance.

---

### Type Conversion

**Explicit conversion** (you call a constructor):

```python
int("42")        # 42      — string to int
int(3.99)        # 3       — truncates, does NOT round
float("3.14")    # 3.14
float(42)        # 42.0
str(42)          # "42"
bool(0)          # False
bool(1)          # True
```

**Implicit conversion** is minimal in Python. The one common case: mixing `int` and `float` in arithmetic produces a `float`:
```python
1 + 2.0   # 3.0  (int promoted to float)
```

**Pitfalls:**
```python
int("3.14")          # ValueError — can't parse a decimal string as int
int(float("3.14"))   # 3 — convert to float first, then truncate
```

---

### Truthiness (Boolean Context)

Every Python object has a boolean value. Objects that evaluate to `False` in a boolean context:

| Value | Type |
|-------|------|
| `False` | bool |
| `0` | int |
| `0.0` | float |
| `""` | str (empty string) |
| `[]` | list (empty) |
| `{}` | dict (empty) |
| `()` | tuple (empty) |
| `None` | NoneType |

Everything else is truthy. This rule is consistent: **empty/zero/absent = falsy**.

---

### Type Hints

Type hints are annotations that document what type a variable or function expects. They are checked by static analysis tools (mypy, pyright, your IDE) but **not enforced at runtime** by default.

```python
name: str = "Alice"
age: int = 30
score: float = 98.5
label: str | None = None  # Python 3.10+ union syntax

def greet(name: str) -> str:
    return f"Hello, {name}"
```

Type hints exist to help tooling catch mistakes before you run the code.

## How it works internally

**CPython int interning:** For small integers (typically -5 to 256), CPython pre-allocates and caches the objects at interpreter startup. Every name bound to `5` points to the same cached object. For large integers, each assignment creates a new object. This is why:

```python
x = 5; y = 5
x is y   # True  — same cached object

x = 1000; y = 1000
x is y   # False in most contexts — two separate objects
```

This is an implementation detail of CPython, not part of the Python language spec. Never write logic that depends on it.

**Memory layout:** An `int` object in CPython is not just the number — it carries a reference count, a pointer to its type object (`int`), and the digit array. A Python `int` uses roughly 28 bytes for a small integer; a C `int` uses 4. The tradeoff is flexibility (arbitrary precision, dynamic typing) vs memory efficiency.

**None is a singleton:** There is exactly one `None` object in a Python process. Every `x = None` assignment binds `x` to that same object. This is why `x is None` is the idiomatic check (not `x == None`).

## Worked example

```python
# day2_types.py — type introspection and conversion
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def inspect_object(label: str, value: object) -> None:
    """Log the type, id, and bool value of any object."""
    log.info("%s: value=%r  type=%s  id=%d  bool=%s",
             label, value, type(value).__name__, id(value), bool(value) if value is not None else False)

# Primitive types
x: int = 42
y: float = 3.14
flag: bool = True
nothing: None = None

inspect_object("x", x)
inspect_object("y", y)
inspect_object("flag", flag)
inspect_object("nothing", nothing)

# int interning: small integers share identity
a: int = 5
b: int = 5
log.info("a is b (small int, %d): %s", a, a is b)  # True

c: int = 1000
d: int = 1000
log.info("c is d (large int, %d): %s", c, c is d)  # likely False

# Conversion pitfalls
raw: str = "3.14"
# int(raw)              # raises ValueError
as_float: float = float(raw)
as_int: int = int(as_float)   # truncates to 3
log.info("'%s' -> float -> int: %d", raw, as_int)

# Truthiness
falsy_values: list[object] = [0, 0.0, "", [], {}, (), None, False]
for val in falsy_values:
    log.info("bool(%r) = %s", val, bool(val))
```

Run:
```bash
python3 day2_types.py
```

## Exercises

1. Write a script that creates one variable of each primitive type (`int`, `float`, `bool`, `None`). For each, log its value, `type()`, and `id()`.

2. In the Python REPL, evaluate `bool(0)`, `bool("")`, `bool([])`, `bool(None)`, and `bool(0.0)`. Record each result. Which surprised you?

3. Write three lines that convert the string `"42"` to `int`, then from that `int` to `float`, then from that `float` back to `str`. Log the type and value at each step.

4. Open the REPL and run:
   ```python
   x = 5; y = 5; print(x is y)
   x = 1000; y = 1000; print(x is y)
   ```
   Research and write two sentences explaining why the results differ. Mention "interning" and "CPython implementation detail" in your answer.

5. Annotate three variables of different types with type hints. Install `mypy` (`pip install mypy`) and run `mypy your_script.py`. Confirm it reports no errors. Intentionally use the wrong type and see what error mypy produces.

(Solutions: ../../solutions/week-01/day-2.md)

## Self-check

- [ ] I can explain why Python variables are names (bindings) rather than memory slots
- [ ] I can use `type()`, `isinstance()`, and `id()` to introspect any object
- [ ] I know which values are falsy and can predict boolean coercion outcomes
- [ ] I can convert between `int`, `float`, `str`, and `bool` without hitting common pitfalls
- [ ] I can write type-annotated variables and run mypy on them

---

## Supplementary

- [Python 3.12 built-in types](https://docs.python.org/3.12/library/stdtypes.html)
- [PEP 526 — Variable annotations](https://peps.python.org/pep-0526/)
- [mypy documentation](https://mypy.readthedocs.io/)
