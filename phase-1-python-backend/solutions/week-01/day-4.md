# Solutions — Phase 1 · Week 01 · Day 4 — Conditionals and Loops

---

## Exercise 1 — Print even numbers with a for loop

```python
for i in range(1, 21):       # 1 through 20 inclusive
    if i % 2 == 0:
        print(i)
# Output: 2 4 6 8 10 12 14 16 18 20 (each on its own line)
```

Why it works: `range(1, 21)` generates integers from 1 up to but not including 21. The `%` (modulo) operator returns the remainder — even numbers have remainder 0 when divided by 2.

---

## Exercise 2 — List comprehension version

```python
evens: list[int] = [i for i in range(1, 21) if i % 2 == 0]
print(evens)
# [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
```

Why it works: A list comprehension is syntactic sugar for a `for` loop that appends to a list. The optional `if` clause acts as a filter — only elements satisfying the condition are included. Comprehensions are generally faster than equivalent append loops because the list's internal growth logic is handled in C.

---

## Exercise 3 — Password retry loop

```python
CORRECT_PASSWORD: str = "secret"

while True:
    pwd: str = input("Enter password: ")
    if pwd == CORRECT_PASSWORD:
        print("Access granted.")
        break
    print("Wrong password, try again.")
```

Why it works: `while True` creates an infinite loop. The only way out is the `break` statement, which executes only when the password matches. This pattern is called a "loop-and-a-half" — you need at least one iteration before you can evaluate the exit condition.

Improvement — limit attempts:
```python
MAX_ATTEMPTS: int = 3
CORRECT_PASSWORD: str = "secret"

for attempt in range(1, MAX_ATTEMPTS + 1):
    pwd: str = input(f"Enter password (attempt {attempt}/{MAX_ATTEMPTS}): ")
    if pwd == CORRECT_PASSWORD:
        print("Access granted.")
        break
else:
    # The for-else clause runs only if the loop completed WITHOUT a break
    print("Too many failed attempts. Locked out.")
```

---

## Exercise 4 — Enumerate a string

```python
for i, ch in enumerate("Python"):
    print(f"{i}: {ch}")
```

**Output:**
```
0: P
1: y
2: t
3: h
4: o
5: n
```

Why it works: `enumerate(iterable, start=0)` wraps any iterable and yields `(index, value)` tuples. This avoids the error-prone pattern of maintaining a counter variable manually (`i = 0; i += 1`).

With a custom start index:
```python
for i, ch in enumerate("Python", start=1):
    print(f"{i}: {ch}")
# 1: P  2: y  3: t ...
```

---

## Exercise 5 — Zip two lists

```python
names: list[str] = ["Alice", "Bob", "Carol"]
scores: list[int] = [90, 85, 78]

for name, score in zip(names, scores):
    print(f"{name}: {score}")
```

**Output:**
```
Alice: 90
Bob: 85
Carol: 78
```

Why it works: `zip(a, b)` lazily yields `(a[i], b[i])` tuples until the shorter iterable is exhausted. It does not copy data — it produces pairs on demand. Use `zip(strict=True)` (Python 3.10+) if you want a `ValueError` when the lengths differ.

Bonus — convert to a dict:
```python
grade_book: dict[str, int] = dict(zip(names, scores))
print(grade_book)
# {'Alice': 90, 'Bob': 85, 'Carol': 78}
```

---

## Self-check answers

- **What is the difference between `break` and `continue`?**
  `break` exits the loop entirely. `continue` skips the rest of the current iteration and goes back to the loop condition check.

- **When does the `for-else` clause run?**
  Only when the loop completes normally — i.e., without hitting a `break`. Useful for "search and not found" patterns.

- **What does `range(5, 0, -1)` produce?**
  `[5, 4, 3, 2, 1]` — the third argument is the step. A negative step counts down.
