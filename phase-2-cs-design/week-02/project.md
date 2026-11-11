# Week 02 Project — Build a Production LRU Cache

> Estimated time: 4–6 hr  ·  Deliver: `lru_cache_prod.py` + `tests/`

---

## Overview

Extend the basic LRU cache from Day 7 into a production-quality component. The result will be a self-contained Python module with full test coverage, thread safety, optional TTL expiry, and documented complexity guarantees.

This project is the capstone for Week 02. It integrates:
- Data structure knowledge (dict + DLL, O(1) operations)
- Software engineering practices (type hints, logging, threading, pytest)
- Performance reasoning (Big-O analysis of every public method)

---

## Specification

### Module: `lru_cache_prod.py`

Implement a class `LRUCache` with the following interface. **No implementation goes here** — spec only.

#### Constructor

```python
def __init__(self, capacity: int) -> None:
```

- `capacity`: maximum number of entries. Must be ≥ 1; raise `ValueError` otherwise.

#### `get`

```python
def get(self, key: str) -> Any | None:
```

- Return the value associated with `key`, or `None` if not present or TTL expired.
- Mark the key as most recently used (even if it expires immediately after — the recency update happens before TTL is checked is acceptable behaviour; document your choice).
- Thread-safe.

#### `put`

```python
def put(self, key: str, value: Any, ttl_seconds: float | None = None) -> None:
```

- Insert or update `key → value`.
- `ttl_seconds`: if provided and `> 0`, the entry expires that many seconds after this `put` call. Expired entries behave as if absent (return `None` on `get`, are eligible for eviction before non-expired entries).
- If `key` already exists: update value and TTL, move to front.
- If at capacity: evict the LRU non-expired entry. If all entries are expired, evict the oldest expired entry.
- Thread-safe.

#### `delete`

```python
def delete(self, key: str) -> bool:
```

- Remove `key` from the cache immediately.
- Return `True` if the key existed and was deleted, `False` if it was not present.
- Thread-safe.

#### `stats`

```python
def stats(self) -> dict[str, int]:
```

Return a snapshot dict with exactly these integer keys:

```python
{
    "hits":      int,   # total successful get() calls (key present, not expired)
    "misses":    int,   # total get() calls that returned None
    "evictions": int,   # total entries evicted due to capacity overflow
    "size":      int,   # current number of live (non-expired) entries
}
```

- Thread-safe.
- Counters never reset (no `clear_stats` method needed).

---

## Implementation requirements

### Data structure

Use a **dict + doubly linked list** (same structure as Day 7). Do not use `collections.OrderedDict`, `functools.lru_cache`, or any third-party library that implements LRU.

### Thread safety

Wrap all public methods with a single `threading.Lock`. Acquire it at the top of each public method and release on exit (use a `with self._lock:` block). The lock must be reentrant-safe for `get` and `put` (they may call internal helpers that also touch shared state — keep all internal helpers non-locking).

### TTL implementation

Store the expiry timestamp (`time.monotonic() + ttl_seconds`) on the node. In `get`, after finding the node, check `time.monotonic() > node.expires_at` — if expired, remove the node and return `None` (counts as a miss, not an eviction).

### Logging

- Log at `DEBUG` for every cache hit/miss.
- Log at `INFO` when an eviction occurs (include the evicted key).
- Log at `WARNING` when a TTL-expired entry is silently removed during `get`.
- Never log values (may contain PII).

### Type hints

All methods, internal helpers, and the `_Node` class must be fully typed. Use `from __future__ import annotations` for forward references.

---

## Test suite specification

Create `tests/test_lru_cache_prod.py` with pytest. Minimum required test functions (implement all of them):

### Unit tests

| Test name | What it verifies |
|---|---|
| `test_get_returns_none_on_empty` | `get` on empty cache → `None` |
| `test_put_and_get_basic` | Round-trip: put then get returns correct value |
| `test_eviction_evicts_lru_entry` | Capacity overflow evicts the correct (LRU) key |
| `test_update_existing_key_moves_to_front` | Updating an existing key changes its recency |
| `test_delete_existing_key` | `delete` returns `True` and key is gone |
| `test_delete_missing_key` | `delete` returns `False` for unknown key |
| `test_stats_hits_and_misses` | `stats()["hits"]` and `stats()["misses"]` count correctly |
| `test_stats_evictions` | `stats()["evictions"]` increments on capacity eviction |
| `test_ttl_expiry_returns_none` | Entry with TTL=0.01 s returns `None` after expiry |
| `test_ttl_not_yet_expired` | Entry with TTL=100 s returns value immediately after put |
| `test_capacity_one` | Capacity-1 cache evicts on every new key |
| `test_invalid_capacity_raises` | `LRUCache(0)` and `LRUCache(-1)` raise `ValueError` |

### Integration / concurrency test

| Test name | What it verifies |
|---|---|
| `test_concurrent_put_get` | 4 threads each doing 500 put+get pairs; no exception, final size ≤ capacity |

Use `threading.Thread` directly — no third-party concurrency helpers.

---

## Complexity analysis

Add a docstring or comment block at the top of `lru_cache_prod.py` with a complexity table:

```
Operation  | Time (avg) | Time (worst) | Space
-----------|------------|--------------|------
get        | O(1)       | O(1)         | O(1)
put        | O(1)       | O(1)         | O(1)
delete     | O(1)       | O(1)         | O(1)
stats      | O(1)       | O(1)         | O(1)
__init__   | O(1)       | O(1)         | O(n) where n = capacity
```

For each O(1) claim, write a one-sentence justification inline in the method docstring referencing which data structure property makes it constant-time.

---

## Deliverables

```
week-02-project/
├── lru_cache_prod.py       # production LRUCache implementation
└── tests/
    └── test_lru_cache_prod.py   # full pytest suite
```

All tests must pass: `pytest tests/ -v`

No implementation goes in this file — see `solutions/week-02/project.md` for the reference implementation.

---

## Stretch goals (optional)

These are not graded but are good practice:

1. **`clear()`** — evict all entries, reset stats. Thread-safe.
2. **`__iter__`** — iterate over `(key, value)` pairs in MRU order (snapshot; not live).
3. **Lazy expiry sweep** — add a background thread (daemon) that sweeps expired entries every N seconds. Benchmark the impact on `stats()["size"]` accuracy.
4. **`max_memory_bytes` parameter** — track approximate memory usage with `sys.getsizeof`; evict when memory threshold is exceeded in addition to capacity.
