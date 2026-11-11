# Solutions — Phase 1 · Week 01 · Day 3 — Operators, Expressions, Mutability

---

## Exercise 1 — Arithmetic operators

```python
# Floor division: discard the fractional part, keep the integer quotient
print(17 // 5)   # 3

# Modulo: remainder after floor division
print(17 % 5)    # 2

# Exponentiation: 2 raised to the power of 10
print(2 ** 10)   # 1024

# Verify the floor-division identity: a == (a // b) * b + (a % b)
assert 17 == (17 // 5) * 5 + (17 % 5)  # True
```

Why it works: `//` and `%` are paired operators. `17 // 5` gives `3`; `17 % 5` gives `2` because `3 * 5 = 15`, and `17 - 15 = 2`. The identity `a == (a // b) * b + (a % b)` always holds.

---

## Exercise 2 — Short-circuit evaluation

```python
x: str | None = None

# 'and' short-circuits: if left side is falsy, right side is never evaluated
result = x and x.upper()
# x is None → falsy → Python returns None immediately
# x.upper() is NEVER called → no AttributeError
print(result)   # None

# Same with 'or': if left side is truthy, right side is never evaluated
default: str = x or "fallback"
print(default)  # "fallback"

# A non-None value flows through
y: str | None = "hello"
result2 = y and y.upper()
print(result2)  # "HELLO"
```

Why it works: Python's `and` returns the first falsy value it encounters, or the last value if all are truthy. `or` returns the first truthy value, or the last value if all are falsy. Neither operator always returns `True`/`False` — they return one of the operands. This makes `x and x.method()` a safe guard pattern.

---

## Exercise 3 — Aliasing trap with lists

```python
import copy

# --- The trap ---
a: list[int] = [1, 2, 3]
b = a                # b is NOT a copy; b is another name for the SAME object
b.append(4)
print(a)             # [1, 2, 3, 4] — mutated through b!
print(a is b)        # True — same id

# --- Shallow copy fix ---
a = [1, 2, 3]
c = copy.copy(a)     # or: a[:]  or list(a)
c.append(4)
print(a)             # [1, 2, 3] — unchanged
print(a is c)        # False — different objects

# --- Deep copy for nested structures ---
nested: list[list[int]] = [[1, 2], [3, 4]]
shallow = copy.copy(nested)
shallow[0].append(99)
print(nested[0])     # [1, 2, 99] — inner list still shared!

deep = copy.deepcopy(nested)
deep[0].append(88)
print(nested[0])     # [1, 2, 99] — inner list now independent
```

Why it works: Assignment in Python binds a name to an object — it never copies the object. `b = a` makes `b` point to the same list. Shallow copy creates a new list with the same element references; deep copy recursively copies every nested object.

---

## Exercise 4 — String immutability

```python
s: str = "hello"

# Attempt to mutate — raises TypeError
try:
    s[0] = "H"
except TypeError as e:
    print(f"Error: {e}")
# Error: 'str' object does not support item assignment

# Fix 1: string concatenation
s = "H" + s[1:]
print(s)   # "Hello"

# Fix 2: str.replace
s = "hello"
s = s.replace("h", "H", 1)   # replace only the first occurrence
print(s)   # "Hello"

# Fix 3: convert to list, mutate, rejoin
chars: list[str] = list("hello")
chars[0] = "H"
s = "".join(chars)
print(s)   # "Hello"
```

Why it works: Strings in Python are immutable — their internal byte buffer cannot be modified after creation. Each "modification" creates a new string object. The old object becomes eligible for garbage collection if no names reference it.

---

## Exercise 5 — `==` vs `is` for strings

```python
# Short strings: CPython interns them (reuses the same object)
a: str = "hello"
b: str = "hello"
print(a == b)   # True  — same value
print(a is b)   # True  — likely interned (CPython caches simple identifiers)

# Strings with spaces or special chars: interning is not guaranteed
a = "hello world"
b = "hello world"
print(a == b)   # True  — always correct
print(a is b)   # True OR False — implementation detail, do not rely on this

# Dynamically constructed strings are never interned
a = "hel" + "lo"  # constant folding may or may not intern
b = "hel"
b += "lo"          # definitely a new object
print(a == b)   # True
print(a is b)   # False (usually)
```

**Rule:** Use `==` to compare string values. Use `is` only to test identity with singletons: `None`, `True`, `False`.

Why it works: `is` compares object identity (memory address). String interning is a CPython optimization that is not part of the Python language spec — code that relies on `is` for string equality will break unpredictably across implementations, Python versions, or runtime contexts.

---

## Self-check answers

- **What is object aliasing?** Two or more names bound to the same object in memory.
- **Why are immutable objects safer to share?** Because no reference can mutate them — you can pass a `str` or `int` anywhere without fear of side effects.
- **When should you use `copy.deepcopy`?** When you have nested mutable containers (lists of lists, dicts of lists) and need a fully independent clone.
