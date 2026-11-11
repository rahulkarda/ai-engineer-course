# Phase 2 · Week 02 · Day 3 — Hash Tables Internals

> Objective: Understand open addressing, load factor, and collision resolution, and implement a hash map from scratch.  ·  Time: ~1 hr  ·  Prereqs: [[phase-2-cs-design/week-02/day-2]]

---

## Concept

A **hash table** maps keys to values in O(1) average time by converting each key to a bucket index via a **hash function**.

```
key  ──hash()──▶  index  ──▶  bucket  ──▶  value
"cat"  →  h("cat") % N  →  bucket[3]  →  "meow"
```

The challenge: hash functions map an infinite key space to a finite set of buckets, so **collisions** (two keys landing on the same index) are inevitable. Collision resolution is the core design dimension.

### Collision resolution strategies

**1. Separate chaining** — each bucket holds a list (or linked list) of all key-value pairs that hash there. Lookup walks the list. O(1) average, O(n) worst.

**2. Open addressing** — all entries live in the array itself; on collision, probe to the next available slot.

| Probing scheme | Formula | Problem |
|---|---|---|
| Linear probing | `(h + i) % N` | Clustering |
| Quadratic probing | `(h + i²) % N` | Secondary clustering |
| Double hashing | `(h + i·h₂(k)) % N` | Best distribution |

**Load factor** `α = n / N` (items stored / bucket count) controls performance:
- At α < 0.7 (open addressing): O(1) average probe count.
- At α ≥ 0.7: probe chains grow rapidly — **resize** (typically double N and rehash all entries).

Python's built-in `dict` uses open addressing with pseudo-random probing (`(5*i + 1 + perturb) % N`) and resizes at α = 2/3.

### Hash function properties

A good hash function for hash tables should be:
1. **Deterministic** — same key always → same hash.
2. **Uniform** — distributes keys evenly across buckets.
3. **Fast** — O(1) to compute.
4. **Avalanche effect** — small key change → large hash change.

Python's `hash()` satisfies all four for built-in types. Custom classes must implement `__hash__` and `__eq__` consistently.

---

## How it works internally

### CPython dict (3.6+)

CPython uses a **compact dict** layout (introduced in 3.6):
- Two arrays: an **indices** array (compact, holds slot numbers) and an **entries** array (holds actual key/value/hash triples in insertion order).
- On lookup: compute `h = hash(key)`, check `indices[h % len(indices)]` for a slot, read the entry, compare hashes then keys.
- Probing formula: `i = (5*i + 1 + perturb) % N` where `perturb >>= PERTURB_SHIFT` each step — mixes more bits of the original hash, preventing long clustering chains.
- Resize triggers at 2/3 load, doubling capacity. Tombstones ("dummy" entries) handle deleted slots.

### Why keys must be hashable (immutable)

Hash tables cache the hash value. If a key's value changed after insertion, `hash(key)` would differ from the stored hash and lookups would fail. Python enforces this by requiring keys to be immutable (or at least hash-stable).

### Birthday paradox and collision probability

By the birthday paradox, in a table with N slots, inserting √N keys already gives a ~50% chance of a collision. This is why a good hash function and a good load-factor policy both matter.

### Tombstone problem in open addressing

When a key is deleted, you cannot simply zero the slot: that would break probe chains for keys inserted after the deleted one. Solutions:
- Mark the slot as a **tombstone** — it is "logically empty" but a probe passes through it. Tombstones accumulate and slow lookups.
- **Robin Hood hashing** — on insertion, displace entries that are "rich" (close to their home slot) to give space to "poor" entries (far from home), reducing variance in probe lengths.

---

## Worked example

```python
"""
hash_table.py — open-addressing hash map with linear probing and rehashing
Python 3.12+
"""
from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Iterator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

_INITIAL_CAPACITY = 8
_LOAD_FACTOR_THRESHOLD = 0.7


class _Slot(Enum):
    EMPTY = auto()
    TOMBSTONE = auto()


type _Entry[K, V] = tuple[K, V] | _Slot


class HashMap[K, V]:
    """
    Open-addressing hash map with linear probing.

    Invariants:
    - Capacity is always a power of two (fast modulo via bitmask).
    - Load factor stays below _LOAD_FACTOR_THRESHOLD.
    - Deleted slots are marked TOMBSTONE, not EMPTY, to preserve probe chains.
    """

    def __init__(self, initial_capacity: int = _INITIAL_CAPACITY) -> None:
        self._capacity: int = self._next_power_of_two(initial_capacity)
        self._buckets: list[_Entry[K, V]] = [_Slot.EMPTY] * self._capacity
        self._size: int = 0           # live entries
        self._tombstones: int = 0     # deleted (tombstone) entries

    # -- public API ---------------------------------------------------------

    def put(self, key: K, value: V) -> None:
        """Insert or update key. O(1) average. Rehashes if load factor exceeded."""
        if (self._size + 1) / self._capacity >= _LOAD_FACTOR_THRESHOLD:
            self._rehash(self._capacity * 2)

        idx = self._probe_for_insert(key)
        entry = self._buckets[idx]
        if entry is _Slot.EMPTY or entry is _Slot.TOMBSTONE:
            if entry is _Slot.TOMBSTONE:
                self._tombstones -= 1
            self._size += 1
        # else: updating an existing key — size unchanged
        self._buckets[idx] = (key, value)

    def get(self, key: K, default: V | None = None) -> V | None:
        """Return value for key, or default if not found. O(1) average."""
        idx = self._probe_for_lookup(key)
        if idx is None:
            return default
        entry = self._buckets[idx]
        assert isinstance(entry, tuple)
        return entry[1]

    def delete(self, key: K) -> bool:
        """Remove key. Returns True if key existed. O(1) average."""
        idx = self._probe_for_lookup(key)
        if idx is None:
            return False
        self._buckets[idx] = _Slot.TOMBSTONE
        self._size -= 1
        self._tombstones += 1
        # Clean up if tombstones accumulate (optional — keeps table healthy)
        if self._tombstones > self._capacity // 4:
            self._rehash(self._capacity)  # same size, removes tombstones
        return True

    def __contains__(self, key: object) -> bool:
        return self._probe_for_lookup(key) is not None  # type: ignore[arg-type]

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[K]:
        for entry in self._buckets:
            if isinstance(entry, tuple):
                yield entry[0]

    def items(self) -> Iterator[tuple[K, V]]:
        for entry in self._buckets:
            if isinstance(entry, tuple):
                yield entry

    def load_factor(self) -> float:
        return self._size / self._capacity

    # -- internal probing ---------------------------------------------------

    def _home(self, key: K) -> int:
        """Primary hash index (home slot). O(1)."""
        return hash(key) & (self._capacity - 1)   # fast % for power-of-two

    def _probe_for_insert(self, key: K) -> int:
        """
        Find the slot to write to.
        Returns the slot of an existing entry with same key (update),
        or the first EMPTY/TOMBSTONE slot encountered (insert).
        """
        home = self._home(key)
        first_tombstone: int | None = None
        for i in range(self._capacity):
            idx = (home + i) & (self._capacity - 1)   # linear probe
            entry = self._buckets[idx]
            if entry is _Slot.EMPTY:
                return first_tombstone if first_tombstone is not None else idx
            if entry is _Slot.TOMBSTONE:
                if first_tombstone is None:
                    first_tombstone = idx   # remember for potential insert
            elif entry[0] == key:
                return idx                  # update existing key
        # Table is full of tombstones — should not happen if resize works
        assert first_tombstone is not None, "Table unexpectedly full"
        return first_tombstone

    def _probe_for_lookup(self, key: K) -> int | None:
        """
        Find the slot of an existing key.
        Returns the index, or None if not found.
        Must skip tombstones (they do not break the probe chain).
        """
        home = self._home(key)
        for i in range(self._capacity):
            idx = (home + i) & (self._capacity - 1)
            entry = self._buckets[idx]
            if entry is _Slot.EMPTY:
                return None          # genuine empty slot — key was never here
            if entry is _Slot.TOMBSTONE:
                continue             # skip — probe chain is not broken
            if entry[0] == key:
                return idx
        return None

    # -- rehashing ----------------------------------------------------------

    def _rehash(self, new_capacity: int) -> None:
        """
        Create a new backing array and reinsert all live entries.
        Called on resize or tombstone cleanup. O(n).
        """
        log.debug(
            "Rehashing: capacity %d → %d, size=%d, tombstones=%d",
            self._capacity, new_capacity, self._size, self._tombstones,
        )
        old_buckets = self._buckets
        self._capacity = self._next_power_of_two(new_capacity)
        self._buckets = [_Slot.EMPTY] * self._capacity
        self._size = 0
        self._tombstones = 0
        for entry in old_buckets:
            if isinstance(entry, tuple):
                self.put(entry[0], entry[1])   # reinsert — no tombstones now

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _next_power_of_two(n: int) -> int:
        """Return the smallest power of two >= n."""
        p = 1
        while p < n:
            p <<= 1
        return p

    def __repr__(self) -> str:
        pairs = ", ".join(f"{k!r}: {v!r}" for k, v in self.items())
        return f"HashMap({{{pairs}}})"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    hm: HashMap[str, int] = HashMap()
    words = ["apple", "banana", "cherry", "date", "elderberry",
             "fig", "grape", "honeydew"]
    for i, word in enumerate(words):
        hm.put(word, i)
        log.info("put(%r, %d) → load=%.2f", word, i, hm.load_factor())

    log.info("get('banana')=%s", hm.get("banana"))    # 1
    log.info("get('mango')=%s",  hm.get("mango"))     # None

    hm.put("banana", 999)                              # update
    log.info("After update: get('banana')=%s", hm.get("banana"))  # 999

    hm.delete("date")
    log.info("After delete('date'): 'date' in hm = %s", "date" in hm)  # False

    log.info("Size: %d", len(hm))
    log.info("All keys: %s", sorted(hm))
```

**Walkthrough — inserting "apple" into an 8-slot table:**

1. `hash("apple") & 7` → say index 3.
2. `buckets[3]` is `EMPTY` → write `("apple", 0)` there.
3. `size = 1`, `load = 1/8 = 0.125`.

**Walkthrough — collision between "banana" and "cherry" (both hash to 5):**

1. `hash("banana") & 7` = 5 → `buckets[5]` = EMPTY → write "banana".
2. `hash("cherry") & 7` = 5 → `buckets[5]` occupied → probe to 6 → EMPTY → write "cherry".
3. Lookup "cherry": probe 5 (not cherry), probe 6 (cherry!) → found.

---

## Exercises

1. **Double hashing:** Modify `HashMap` to use double hashing instead of linear probing. The second hash function should be `h2(key) = 1 + (hash(key) >> 3) % (capacity - 1)`. The probe sequence becomes `(home + i * h2(key)) % capacity`. Does this reduce clustering? Verify by checking the distribution of occupied slots.

2. **Word frequency counter:** Using `HashMap[str, int]`, implement `word_frequencies(text: str) -> HashMap[str, int]` that counts word occurrences (case-insensitive, strip punctuation). Compare performance against Python's `collections.Counter` for a 10 000-word input.

3. **Consistent hashing ring:** Implement a simplified consistent hashing ring with `add_node(node_id)`, `remove_node(node_id)`, and `get_node(key) -> str`. Use a sorted list of hash-ring positions and bisect for lookup. Explain why consistent hashing minimises remapping when a node is added or removed.

4. **Hashable class:** Define a `Point` dataclass with fields `x: float` and `y: float`. Make it usable as a `HashMap` key by implementing `__hash__` and `__eq__`. Verify that two `Point(1.0, 2.0)` instances hash identically and that `Point` cannot be used as a key after mutating (hint: use `frozen=True` in the dataclass or add a `__setattr__` guard).

5. **Anagram groups:** Given a list of strings, group them by anagram using a `HashMap[str, list[str]]` where the key is the sorted characters of each word. E.g., `["eat","tea","tan","ate","nat","bat"]` → `[["eat","tea","ate"],["tan","nat"],["bat"]]`.

*(Solutions: ../solutions/week-02/day-3.md)*

---

## Self-check

- [ ] I can explain what a load factor is and why a hash table resizes at 0.7 (or 2/3).
- [ ] I can trace a 5-step linear probe sequence for both insert and lookup, including tombstones.
- [ ] I can explain why tombstones are necessary and what problem they create over time.
- [ ] I can describe the difference between separate chaining and open addressing in terms of cache efficiency.
- [ ] I can implement a minimal `HashMap` with `put`, `get`, `delete`, and automatic rehashing from memory.

---

## Supplementary links

- [CPython dict implementation (GitHub)](https://github.com/python/cpython/blob/main/Objects/dictobject.c)
- [Raymond Hettinger — The mighty dictionary (PyCon)](https://www.youtube.com/watch?v=C4Kc8xzcA68)
- [Visualgo — Hash Table](https://visualgo.net/en/hashtable)
- [Open Addressing vs. Chaining — empirical analysis](https://codecapsule.com/2013/11/11/robin-hood-hashing/)
