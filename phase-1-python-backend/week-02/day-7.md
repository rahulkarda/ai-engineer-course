# Phase 1 · Week 02 · Day 7 — Build Day: Financial Transaction Processor

> Objective: Build the week project from scratch, run the assessment, and reflect.  ·  Time: ~3 hr project + 1 hr assessment  ·  Prereqs: [[day-6]]

## Overview

Today you build the **Financial Transaction Processor** — a self-contained Python module that uses every skill from this week. Then you sit the week assessment.

**Split your time**: 3 hours on the project, then switch to `assessment.md`.

---

## Project brief

See `project.md` for the full specification. Build the module, write the tests, and verify everything passes before starting the assessment.

---

## Assessment

See `assessment.md`. Time-box to 1 hour. Work independently.

---

## Reflection checklist

After completing both:

- [ ] My project runs end-to-end without errors on the provided sample data
- [ ] All tests pass (`pytest tests/` exits 0)
- [ ] I did not look up syntax for: list comprehensions, dict comprehensions, `*args`/`**kwargs`, `sorted()` with a key, `json.dump`/`json.load`, or basic exception handling
- [ ] I needed to look up: (record what you looked up — honest self-assessment)
- [ ] Concepts I want to revisit: (list them)
- [ ] One thing I found genuinely surprising this week:

---

## Week 2 review: mental model check

Answer these three questions in plain English before moving to Week 3. No code allowed.

1. When would you choose a `set` over a `list` for storing transaction IDs? What would you lose?
2. Explain to a colleague what `*args` and `**kwargs` do, without using the words "positional" or "keyword".
3. A co-worker says "I caught a bare `Exception` so nothing can crash my script." What is wrong with this, and what would you do instead?

---

## Next

Week 3 covers **object-oriented Python**: classes, `__dunder__` methods, inheritance, `dataclasses`, and `Pydantic` models. The transaction processor you built this week will be the domain we evolve with OOP patterns.
