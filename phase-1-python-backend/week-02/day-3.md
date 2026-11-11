# Phase 1 · Week 02 · Day 3 — Functions, Arguments, and Scope

> Objective: Define functions with all argument forms; understand Python's scoping rules (LEGB); use closures deliberately.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

## Concept

### Functions as first-class objects
In Python, functions are objects. They can be assigned to variables, passed as arguments, returned from other functions, and stored in collections. This underpins higher-order patterns used throughout the course.

### Argument forms
Python supports five distinct argument styles, and they can be mixed in one signature in a fixed order:

```
def f(pos_only, /, normal, *, kw_only, **kwargs): ...
```

| Style | Syntax | Passed as |
|-------|--------|-----------|
| Positional-only | before `/` | position only |
| Regular | no marker | position or keyword |
| Keyword-only | after `*` or `*args` | keyword only |
| `*args` | `*name` | collects extra positional into tuple |
| `**kwargs` | `**name` | collects extra keyword into dict |

### Scope — LEGB rule
Python resolves a name in this order:
1. **L**ocal — inside the current function
2. **E**nclosing — enclosing function scopes (closures), inner-to-outer
3. **G**lobal — module top level
4. **B**uilt-in — `builtins` module (`len`, `range`, …)

`global x` and `nonlocal x` are explicit override declarations. Without them, assignment in a function creates a *local* binding — it never mutates an outer variable.

### Default argument gotcha
Mutable default arguments are evaluated **once** at function definition, not per call. Always use `None` and create the mutable inside the function:

```python
# WRONG — all callers share the same list object
def add_tag(tags: list[str] = []) -> list[str]:
    tags.append("new")
    return tags

# RIGHT
def add_tag(tags: list[str] | None = None) -> list[str]:
    if tags is None:
        tags = []
    tags.append("new")
    return tags
```

## How it works internally

**Call stack**: each function call creates a *frame* object on the call stack. The frame holds the local namespace (a dict of `name → object`). When the function returns, the frame is popped and the local namespace is discarded.

**Closures**: when an inner function references a name from an enclosing scope, Python captures a *cell object* — a shared reference. Both the inner function and the enclosing scope see the same cell. Modifying the variable in one scope is visible in the other.

**`*args` / `**kwargs`**: at the call site, Python constructs a tuple (for `*args`) or a dict (for `**kwargs`) before the frame is entered. These are ordinary objects by the time your function body runs.

```python
import dis

def add(a: int, b: int) -> int:
    return a + b

dis.dis(add)   # shows LOAD_FAST for local variables
```

## Worked example

```python
# functions_scope.py
from __future__ import annotations
import logging
from typing import Any

log = logging.getLogger(__name__)


# --- Basic type-annotated function ---
def calculate_fee(amount: float, rate: float = 0.025) -> float:
    """Return the processing fee for an amount."""
    return round(amount * rate, 2)


# --- *args: accept variable number of amounts ---
def total_amount(*amounts: float) -> float:
    """Sum any number of transaction amounts."""
    return sum(amounts)


# --- **kwargs: flexible metadata attachment ---
def enrich_transaction(tx_id: str, **metadata: Any) -> dict[str, Any]:
    """Create a transaction dict with mandatory id and optional metadata."""
    return {"id": tx_id, **metadata}


# --- Keyword-only arguments (require name at call site) ---
def apply_discount(
    amount: float,
    *,                          # everything after * is keyword-only
    discount: float,
    currency: str = "USD",
) -> dict[str, float | str]:
    return {
        "original": amount,
        "discounted": round(amount * (1 - discount), 2),
        "currency": currency,
    }


# Usage
print(calculate_fee(1_000.0))             # → 25.0
print(calculate_fee(1_000.0, rate=0.01))  # → 10.0
print(total_amount(10.0, 20.0, 30.0))     # → 60.0
print(enrich_transaction("T1", status="success", region="EU"))
# apply_discount(500.0, 0.1)  ← TypeError: positional not allowed
print(apply_discount(500.0, discount=0.1, currency="EUR"))


# --- Closure: factory that captures config ---
def make_fee_calculator(rate: float) -> callable:
    """Return a fee calculator fixed at a given rate."""
    def calculator(amount: float) -> float:
        return round(amount * rate, 2)   # 'rate' is captured in a cell
    return calculator

standard_fee = make_fee_calculator(0.025)
premium_fee  = make_fee_calculator(0.015)

log.debug("Standard fee for 1000: %s", standard_fee(1_000.0))
log.debug("Premium fee for 1000: %s",  premium_fee(1_000.0))


# --- Scope demo ---
threshold: float = 500.0   # global (module-level)

def is_high_value(amount: float) -> bool:
    # reads global 'threshold' via LEGB — no 'global' declaration needed for reads
    return amount > threshold

def update_threshold(new_value: float) -> None:
    global threshold          # declaration required to *assign* to global
    threshold = new_value
```

## Exercises

1. **Variadic sum**: Write `weighted_average(*values: float, weight: float = 1.0) -> float` that returns the weighted mean of all values (each value × weight, divided by count). Make `weight` keyword-only.

2. **Flexible enrichment**: Write `build_record(record_id: str, record_type: str, **fields: Any) -> dict` that always includes `id`, `type`, and `created_at` (use `datetime.now().isoformat()`) plus any extra keyword arguments.

3. **Closure factory**: Write `make_threshold_filter(min_amount: float, max_amount: float) -> callable`. The returned function should accept a `list[dict]` of transactions and return only those with `amount` in `[min_amount, max_amount]`.

4. **Scope exercise**: The following code has a bug — identify it and explain why it fails, then fix it without using `global`:

   ```python
   count = 0
   def increment() -> None:
       count += 1   # bug
   increment()
   ```

5. **Default argument trap**: Fix this function and add a test that demonstrates the original bug fails:

   ```python
   def attach_tags(tx: dict, tags: list[str] = []) -> dict:
       tags.append("processed")
       tx["tags"] = tags
       return tx
   ```

6. **Argument forwarding**: Write `log_and_call(fn: callable, *args: Any, **kwargs: Any) -> Any` that logs `"Calling {fn.__name__} with args={args} kwargs={kwargs}"` and then calls `fn(*args, **kwargs)`, returning its result.

(Solutions: ../solutions/week-02/day-3.md)

## Self-check
- [ ] I can list the five argument forms in their required order
- [ ] I can explain the LEGB lookup chain with a concrete example
- [ ] I can explain why mutable defaults are dangerous and the correct pattern
- [ ] I can write a closure that captures a variable from an enclosing scope
- [ ] I can use `*args` and `**kwargs` to forward arguments transparently

## Supplementary links
- [Python function definitions — reference](https://docs.python.org/3/reference/compound_stmts.html#function-definitions)
- [Execution model — naming and binding](https://docs.python.org/3/reference/executionmodel.html)
- [PEP 3102 — keyword-only arguments](https://peps.python.org/pep-3102/)
