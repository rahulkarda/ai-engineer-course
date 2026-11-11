# Phase 1 · Week 01 · Day 4 — Conditionals and Loops

> Objective: Control program flow with conditionals, while/for loops, and list comprehensions  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

## Concept

### Conditionals

`if / elif / else` branches execution based on a boolean expression. Python uses indentation (4 spaces) to delimit blocks — no braces.

```python
score: int = 72

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "F"
```

**Ternary expression** — one-liner for simple branches:

```python
label: str = "pass" if score >= 70 else "fail"
```

**Nested conditionals** work but get hard to read past two levels. Prefer early returns or `elif` chains.

---

### `while` Loops

Repeat a block as long as a condition is `True`.

```python
count: int = 0
while count < 5:
    count += 1
```

- `break` — exit the loop immediately.
- `continue` — skip the rest of this iteration, re-check condition.
- `else` clause — runs **only if the loop exited normally** (condition became False), not via `break`. Useful for "did we find it?" patterns.

```python
target: int = 7
i: int = 0
while i < 10:
    if i == target:
        print("found")
        break
    i += 1
else:
    print("not found")   # runs only if break was never hit
```

**Avoiding infinite loops**: ensure the loop variable changes each iteration, or use `break` with a guard.

---

### `for` Loops

Iterate over any **iterable** — list, string, range, dict, file lines, etc.

```python
fruits: list[str] = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)
```

`range(start, stop, step)` generates integers lazily:

```python
for i in range(0, 10, 2):   # 0, 2, 4, 6, 8
    print(i)
```

`enumerate` adds an index:

```python
for idx, fruit in enumerate(fruits, start=1):
    print(f"{idx}. {fruit}")
```

`zip` pairs two iterables element-by-element:

```python
names: list[str] = ["Alice", "Bob"]
scores: list[int] = [90, 85]
for name, score in zip(names, scores):
    print(f"{name}: {score}")
```

---

### Loop Control

| Keyword | Effect |
|---------|--------|
| `break` | Exit loop immediately |
| `continue` | Skip rest of body, go to next iteration |
| `pass` | No-op placeholder — keeps syntax valid |

---

### List Comprehensions

Concise alternative to `for` + `append`:

```python
# Explicit loop
evens: list[int] = []
for n in range(1, 21):
    if n % 2 == 0:
        evens.append(n)

# List comprehension — same result
evens: list[int] = [n for n in range(1, 21) if n % 2 == 0]
```

Form: `[expression for item in iterable if condition]` — the `if` part is optional.

## How it works internally

Python's `for` loop uses the **iterator protocol**:

1. Python calls `iter(obj)` on the iterable, which invokes `obj.__iter__()` and returns an **iterator** object.
2. On each loop step, Python calls `next(iterator)`, which invokes `iterator.__next__()` and returns the next value.
3. When exhausted, `__next__` raises `StopIteration`. Python catches it silently and exits the loop.

```python
fruits = ["apple", "banana"]
it = iter(fruits)         # calls fruits.__iter__()
print(next(it))           # "apple"   — calls it.__next__()
print(next(it))           # "banana"
next(it)                  # raises StopIteration
```

This is why `for` works on strings, files, generators, custom objects — anything implementing `__iter__` / `__next__`.

`while` is simpler: the CPython VM evaluates the condition expression each iteration and branches to the loop body or past it.

## Worked example

```python
"""
Worked example: number guessing game (while), then list filtering (comprehension vs explicit).
"""
import random


def guessing_game() -> None:
    secret: int = random.randint(1, 10)
    attempts: int = 0
    max_attempts: int = 5

    while attempts < max_attempts:
        raw: str = input(f"Guess (1-10), {max_attempts - attempts} left: ")
        guess: int = int(raw)
        attempts += 1

        if guess < secret:
            print("Too low.")
        elif guess > secret:
            print("Too high.")
        else:
            print(f"Correct in {attempts} attempt(s)!")
            break
    else:
        # Runs only if we exhausted all attempts without break
        print(f"Out of guesses. Answer was {secret}.")


def above_average_explicit(numbers: list[float]) -> list[float]:
    """Return values above the mean — explicit loop."""
    if not numbers:
        return []
    avg: float = sum(numbers) / len(numbers)
    result: list[float] = []
    for n in numbers:
        if n > avg:
            result.append(n)
    return result


def above_average_comprehension(numbers: list[float]) -> list[float]:
    """Same logic as a list comprehension."""
    if not numbers:
        return []
    avg: float = sum(numbers) / len(numbers)
    return [n for n in numbers if n > avg]


if __name__ == "__main__":
    data: list[float] = [3.0, 7.5, 2.0, 9.1, 5.0, 6.3]
    print(above_average_explicit(data))       # [7.5, 9.1, 6.3]
    print(above_average_comprehension(data))  # [7.5, 9.1, 6.3]
    guessing_game()
```

## Exercises

1. Write a `for` loop that prints only even numbers from 1 to 20.
2. Rewrite exercise 1 as a list comprehension that collects those numbers into a list.
3. Write a `while` loop that repeatedly asks the user for a password until they enter `"secret"`. Use `break` to exit. Print `"Access granted"` when correct.
4. Use `enumerate` to print each character of `"Python"` with its index, e.g. `0: P`.
5. Use `zip` to pair `["Alice", "Bob"]` with `[90, 85]` and print `"Alice: 90"` etc.

(Solutions: ../../solutions/week-01/day-4.md)

## Self-check

- [ ] I can explain the difference between `break` and `continue`.
- [ ] I can explain the `while...else` clause and when it fires.
- [ ] I can describe the iterator protocol (`__iter__`, `__next__`, `StopIteration`).
- [ ] I can rewrite an explicit `for` + `append` loop as a list comprehension.
- [ ] I can use `enumerate` and `zip` without looking them up.

---

**Supplementary:** https://docs.python.org/3.12/tutorial/controlflow.html
