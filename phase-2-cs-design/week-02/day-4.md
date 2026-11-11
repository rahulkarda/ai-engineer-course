# Phase 2 · Week 02 · Day 4 — Hashing Patterns: Deduplication, Grouping, Frequency Counting

> Objective: Apply hash-based data structures to solve deduplication, grouping, and frequency problems in O(n) time.  ·  Time: ~1 hr  ·  Prereqs: [[phase-2-cs-design/week-01/day-1]]

---

## Concept

A **hash table** maps keys to values in O(1) average time for insert, lookup, and delete. Python exposes two hash-based built-ins:

- `dict` — key → value mapping
- `set` — unordered collection of unique keys

Three canonical patterns built on top of these structures:

### 1. Deduplication

Remove duplicates from a sequence while preserving (or discarding) order.

```
seen = set()
for item in sequence:
    if item not in seen:
        seen.add(item)
        yield item
```

- `set` membership test is O(1) average → full pass is **O(n)** time, O(n) space.
- Python `dict` preserves insertion order (3.7+), `set` does not — use `dict.fromkeys()` when order matters.

### 2. Grouping (bucket by key)

Partition items into groups where each group shares a computed key.

```
from collections import defaultdict

groups: dict[K, list[V]] = defaultdict(list)
for item in sequence:
    key = compute_key(item)
    groups[key].append(item)
```

Classic uses: group anagrams (sorted chars as key), group by date, group by category.

### 3. Frequency Counting

Count occurrences of each element.

```
from collections import Counter

freq = Counter(sequence)          # O(n)
freq.most_common(k)               # top-k in O(n log k)
```

`Counter` is a `dict` subclass; missing keys return 0, not `KeyError`.

---

## How it works internally

### Hash functions

Python calls `__hash__()` on a key to produce an integer, then maps it to a slot via `slot = hash(key) % table_size`. The built-in hash for integers is the integer itself (mod a large prime); for strings it is a polynomial rolling hash seeded with a random salt per-process (hash randomisation, PEP 456).

### Collision resolution

CPython's `dict` uses **open addressing with pseudo-random probing** (not chaining). When two keys hash to the same slot, the prober walks a sequence derived from the hash until an empty slot is found. This keeps the table in one contiguous block — better cache behaviour than separate chaining.

### Load factor and resizing

CPython resizes when the table exceeds 2/3 full. Each resize doubles the table and rehashes all entries — O(n) cost amortised to O(1) per insertion. This is why repeated `dict` insertions are O(1) *amortised*, not O(1) worst-case.

### Why `set` vs `frozenset`

`set` is mutable and unhashable — you cannot use it as a dict key. `frozenset` is immutable and hashable, so it can serve as a key. This matters when the "group key" is itself a set (e.g., canonical form of an anagram).

### `collections.defaultdict`

Wraps a `dict` with a factory callable. On missing-key access it calls the factory (e.g., `list`, `int`, `set`) and inserts the result. Avoids repeated `if key not in d: d[key] = []` boilerplate.

---

## Worked example

```python
"""
hashing_patterns.py — deduplication, grouping, frequency counting
Python 3.12+
"""
import logging
from collections import Counter, defaultdict
from typing import Iterable

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── 1. Deduplication ────────────────────────────────────────────────────────

def deduplicate_ordered(items: Iterable[str]) -> list[str]:
    """
    Remove duplicates, preserve first-seen order.
    Time: O(n)  Space: O(n)
    """
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def deduplicate_fast(items: Iterable[str]) -> list[str]:
    """
    dict.fromkeys preserves insertion order in Python 3.7+.
    Equivalent to deduplicate_ordered but one built-in call.
    """
    return list(dict.fromkeys(items))


# ── 2. Grouping ─────────────────────────────────────────────────────────────

def group_anagrams(words: list[str]) -> list[list[str]]:
    """
    Group words that are anagrams of each other.
    Key: tuple of sorted characters — two words share a key iff they're anagrams.
    Time: O(n · k log k)  where k = avg word length   Space: O(n·k)
    """
    buckets: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for word in words:
        key = tuple(sorted(word))      # canonical form: ('a','e','t') for 'eat'/'tea'/'ate'
        buckets[key].append(word)
    return list(buckets.values())


def group_by_length(words: list[str]) -> dict[int, list[str]]:
    """Partition words by character length. O(n) time and space."""
    groups: dict[int, list[str]] = defaultdict(list)
    for word in words:
        groups[len(word)].append(word)
    return dict(groups)


# ── 3. Frequency counting ───────────────────────────────────────────────────

def top_k_frequent(items: list[str], k: int) -> list[str]:
    """
    Return the k most frequent items.
    Counter.most_common(k) uses a heap internally → O(n + k log n).
    """
    freq = Counter(items)
    return [item for item, _ in freq.most_common(k)]


def first_non_repeating(s: str) -> str | None:
    """
    Return first character that appears exactly once, or None.
    Two-pass: count frequencies, then scan for first with count == 1.
    Time: O(n)  Space: O(alphabet size)
    """
    freq = Counter(s)
    for ch in s:
        if freq[ch] == 1:
            return ch
    return None


def two_sum(nums: list[int], target: int) -> tuple[int, int] | None:
    """
    Return indices (i, j) where nums[i] + nums[j] == target, or None.
    Classic hash-table pattern: store complement → index.
    Time: O(n)  Space: O(n)
    """
    seen: dict[int, int] = {}          # value → index
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return None


# ── Demo ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    words = ["eat", "tea", "tan", "ate", "nat", "bat"]
    log.info("group_anagrams: %s", group_anagrams(words))

    sentence = "the quick brown fox jumps over the lazy dog"
    tokens = sentence.split()
    log.info("top_3_frequent: %s", top_k_frequent(tokens, 3))
    log.info("first_non_repeating 'aabbcd': %s", first_non_repeating("aabbcd"))

    nums = [2, 7, 11, 15]
    log.info("two_sum target=9: %s", two_sum(nums, 9))

    raw = ["apple", "banana", "apple", "cherry", "banana", "apple"]
    log.info("deduplicate_ordered: %s", deduplicate_ordered(raw))
    log.info("group_by_length: %s", group_by_length(words))
```

**Key observations:**

- `group_anagrams` uses a mutable `defaultdict(list)` — safe because we never use the key itself as a set member.
- `two_sum` is the canonical O(n) hash trade-off: pay O(n) space to avoid O(n²) nested scan.
- `first_non_repeating` runs two O(n) passes — still O(n) total; never O(n²).

---

## Exercises

1. **Intersection of two lists** — Given `a: list[int]` and `b: list[int]`, return a list of elements that appear in both (no duplicates in output). Target: O(n + m) time. Do not use `set.intersection` directly — implement it manually with a `set`.

2. **Isomorphic strings** — Two strings `s` and `t` are isomorphic if the characters in `s` can be replaced to get `t` (consistent, bijective mapping). Write `is_isomorphic(s: str, t: str) -> bool` in O(n) time. Example: `("egg", "add") → True`, `("foo", "bar") → False`.

3. **Word frequency from file lines** — Given `lines: list[str]` (each line is raw text), return a `dict[str, int]` mapping each lowercase word to its total count across all lines. Strip punctuation using `str.strip(".,!?;:")`. Target: O(total characters).

4. **Longest consecutive sequence** — Given an unsorted list of integers, find the length of the longest sequence of consecutive integers (e.g., `[100, 4, 200, 1, 3, 2] → 4` for sequence `1,2,3,4`). Use a `set` to achieve O(n) time — do not sort.

5. **Subarray sum equals k** — Given `nums: list[int]` and target `k: int`, return the count of contiguous subarrays whose elements sum to `k`. Target: O(n) using a prefix-sum hash map.

*(Solutions: ../solutions/week-02/day-4.md)*

---

## Self-check

- [ ] I can explain why `dict` lookup is O(1) average and O(n) worst case.
- [ ] I can implement deduplication with order-preservation without reaching for `set(list())`.
- [ ] I can use `defaultdict` to group items without an explicit missing-key guard.
- [ ] I can solve two-sum in O(n) and explain the complement → index pattern.
- [ ] I can use `Counter.most_common` and know its time complexity.

---

## Supplementary links

- [Python `collections` docs — Counter, defaultdict](https://docs.python.org/3/library/collections.html)
- [CPython dict implementation notes](https://github.com/python/cpython/blob/main/Objects/dictobject.c)
- [Hash table visualiser — VisuAlgo](https://visualgo.net/en/hashtable)
