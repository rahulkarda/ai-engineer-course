# Phase 3 · Week 05 · Day 7 — Project Day + Assessment

> Objective: Submit the week's project (attention library + tests) and complete the assessment.  ·  Time: ~4 hr (3 hr project + 1 hr assessment)  ·  Prereqs: [[day-6]]

## Schedule

| Block | Time | Activity |
|---|---|---|
| Project polish | 90 min | Finalise attention library, ensure all tests pass, add missing coverage |
| Project submission | 30 min | Review [[project]] requirements, package files |
| Assessment | 60 min | Complete [[assessment]] independently |

## Project checklist

Before marking the project done, verify:

- [ ] `attention_lib.py` implements all 6 components in the spec.
- [ ] All tests in `tests/test_attention_lib.py` pass (`pytest -v`).
- [ ] `TinyGPT` forward pass runs on CPU without error.
- [ ] `.backward()` propagates through `TinyGPT` (gradients non-None).
- [ ] No hardcoded secrets or absolute paths. No `print` — use `logging`.
- [ ] Type hints on all function signatures.

## Assessment: open book? No.

Complete `assessment.md` without opening any lesson file. This is not a test of whether you can copy — it tests whether you can build.

## Reflection exercise

Write in your notes (not for submission — for yourself):

1. What was the hardest concept this week?
2. What implementation mistake cost you the most time?
3. What would you teach differently if you were explaining transformers to a colleague?

---

See [[project]] for the full project specification.
See [[assessment]] for the assessment questions.
