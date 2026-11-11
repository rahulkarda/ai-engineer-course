# Week 04 Assessment — Advanced Python Internals

## Conceptual (interview-style)

1. Explain the difference between an iterable and an iterator. What does `__iter__` return vs `__next__`? Give an example where they are the same object.

2. What is a generator function, and how does it differ from a regular function that returns a list? Describe two situations where a generator is preferable for memory or performance reasons.

3. Explain closures: what is a free variable, and how does Python's late-binding behavior affect closures in loops? Show a common bug and its fix.

4. How do decorators work under the hood? What is the difference between a decorator defined as a function and one defined as a class with `__call__`? Why should you use `functools.wraps`?

5. Describe the context manager protocol (`__enter__` / `__exit__`). When would you use `contextlib.contextmanager` instead of a full class? What happens to the exception if `__exit__` returns `True`?

6. What is the difference between a shallow copy and a deep copy? Give an example where `copy.copy()` produces surprising results and `copy.deepcopy()` is required.

7. Explain Python's reference counting and the role of the cyclic garbage collector. When does reference counting alone fail to free memory? How can you detect and break reference cycles?

8. Describe Python's import system: what happens when you write `import foo.bar.baz`? What is the difference between absolute and relative imports? What does `__all__` control?

9. What is the difference between `TypeVar`, `Generic`, and `TypedDict`? When would you use each? What does `bound=` on a `TypeVar` accomplish?

10. Compare Pydantic v2's `model_validator` (mode `before` vs `after`) with `field_validator`. When would you choose one over the other? How does Pydantic v2's `model_config = ConfigDict(strict=True)` change validation behavior?

---

## Coding challenge

### Build a Typed Transaction Processing Pipeline

**Context:** You are building the core of a financial data pipeline. Raw transaction records arrive as unvalidated dicts from an external source. You must validate, filter, transform, and export them using the Python internals covered this week.

**Spec:**

Implement a `TransactionPipeline` class in a single module `pipeline.py` (no external dependencies beyond `pydantic` and the standard library).

```
TransactionPipeline(records: list[dict])
    .validate()       -> TransactionPipeline          # validate with Pydantic, collect errors
    .filter(**kwargs) -> TransactionPipeline          # generator-based filtering
    .transform(fn)    -> TransactionPipeline          # apply a transform generator
    .export(path)     -> None                         # write JSON; use a context manager
```

**Required elements:**

- A Pydantic v2 `Transaction` model with: `id: str`, `amount: Decimal`, `currency: str` (2-char ISO code), `merchant: str`, `status: Literal["pending", "approved", "rejected"]`, `timestamp: datetime`. Add at least one `field_validator` and one `model_validator`.
- A `TransactionStatus` enum used as the type for `status`.
- Generator functions for filtering (by amount range, currency, status) and for transforming records (e.g., normalise merchant name to title case). Each generator must `yield` one item at a time — no intermediate lists.
- A timing decorator (`@timed`) that logs elapsed time for `validate()`, `filter()`, and `transform()` to `logging` (not `print`).
- A context manager (either class or `@contextmanager`) used inside `export()` to open/close the output file and log success or failure.
- Type annotations throughout — use `TypeVar` or `Generic` if your pipeline is parameterised.
- A `ValidationReport` `TypedDict` with keys `valid_count`, `error_count`, `errors` (list of dicts).

**Acceptance criteria:**

- [ ] `TransactionPipeline(records).validate()` returns `self`; invalid records are collected (not raised) and accessible via `.report: ValidationReport`.
- [ ] Chaining `.filter(min_amount=10, currency="USD", status="approved")` narrows results using generators — no full list materialised until `.export()`.
- [ ] `.transform(fn)` applies `fn` lazily via a generator.
- [ ] `.export("out.json")` writes a valid JSON array; the file is always closed even if serialisation raises.
- [ ] `@timed` decorator preserves the wrapped function's `__name__` and `__doc__`.
- [ ] All public methods have type annotations.
- [ ] At least 6 pytest tests covering: valid record round-trip, invalid record captured in report, filter by each criterion, transform applied, export creates valid JSON, timed decorator preserves metadata.
- [ ] No `print` statements; use `logging.getLogger(__name__)`.

---

## Rubric (pass >= 75%)

| Dimension | Weight |
|---|---|
| Independent implementation | 40% |
| Conceptual understanding | 25% |
| Debugging & problem-solving | 20% |
| Code quality & testing | 15% |

Answer key: ../solutions/week-04/assessment.md
