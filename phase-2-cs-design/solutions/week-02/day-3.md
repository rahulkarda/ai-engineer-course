# Solutions — Phase 2 · Week 02 · Day 3

> Hash Tables Internals — open addressing, load factor, collision resolution

---

## Exercise 1 — Double hashing

**Key insight:** Double hashing uses a second hash to vary the probe step size per key, virtually eliminating primary and secondary clustering. The second hash must be coprime with the table capacity — easiest guarantee: capacity is a power of two and `h2` is always odd.

```python
"""
solution_day3_ex1.py — HashMap with double hashing
Python 3.12+
"""
from __future__ import annotations

import logging
from collections import Counter
from hash_table import HashMap, _Slot, _LOAD_FACTOR_THRESHOLD  # re-use base

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


class DoubleHashMap[K, V](HashMap[K, V]):
    """
    HashMap using double hashing instead of linear probing.

    Probe sequence: idx = (home + i * step) % capacity
    where step = h2(key) = 1 + (hash(key) >> 3) % (capacity - 1)

    step is always >= 1 and odd for power-of-two capacity,
    guaranteeing the probe visits all slots before repeating.
    """

    def _step(self, key: K) -> int:
        """Second hash function — step size for double hashing. Always odd."""
        raw = 1 + (hash(key) >> 3) % (self._capacity - 1)
        return raw | 1   # force odd to ensure full coverage of power-of-two table

    def _probe_for_insert(self, key: K) -> int:
        home = self._home(key)
        step = self._step(key)
        first_tombstone: int | None = None
        for i in range(self._capacity):
            idx = (home + i * step) & (self._capacity - 1)
            entry = self._buckets[idx]
            if entry is _Slot.EMPTY:
                return first_tombstone if first_tombstone is not None else idx
            if entry is _Slot.TOMBSTONE:
                if first_tombstone is None:
                    first_tombstone = idx
            elif entry[0] == key:
                return idx
        assert first_tombstone is not None
        return first_tombstone

    def _probe_for_lookup(self, key: K) -> int | None:
        home = self._home(key)
        step = self._step(key)
        for i in range(self._capacity):
            idx = (home + i * step) & (self._capacity - 1)
            entry = self._buckets[idx]
            if entry is _Slot.EMPTY:
                return None
            if entry is _Slot.TOMBSTONE:
                continue
            if entry[0] == key:
                return idx
        return None


def _slot_distribution(hm: HashMap) -> dict[str, int]:
    """Count EMPTY, TOMBSTONE, and LIVE slots."""
    counts: Counter[str] = Counter()
    for entry in hm._buckets:  # noqa: SLF001
        if entry is _Slot.EMPTY:
            counts["EMPTY"] += 1
        elif entry is _Slot.TOMBSTONE:
            counts["TOMBSTONE"] += 1
        else:
            counts["LIVE"] += 1
    return dict(counts)


if __name__ == "__main__":
    keys = [f"key_{i}" for i in range(50)]

    linear_hm: HashMap[str, int] = HashMap()
    double_hm: DoubleHashMap[str, int] = DoubleHashMap()

    for i, k in enumerate(keys):
        linear_hm.put(k, i)
        double_hm.put(k, i)

    log.info("Linear probing distribution: %s", _slot_distribution(linear_hm))
    log.info("Double hashing distribution: %s", _slot_distribution(double_hm))

    # Verify correctness
    for k in keys:
        assert double_hm.get(k) == linear_hm.get(k), f"Mismatch for {k}"
    log.info("Correctness check passed.")
```

**Why double hashing reduces clustering:**
- Linear probing: consecutive occupied slots cluster, making probe chains grow long.
- Double hashing: each key uses its own step size, spreading probes across the array instead of piling up in contiguous regions.

---

## Exercise 2 — Word frequency counter

```python
"""
solution_day3_ex2.py — word frequency using HashMap vs Counter
Python 3.12+
"""
from __future__ import annotations

import logging
import re
import time
from collections import Counter

from hash_table import HashMap

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def word_frequencies(text: str) -> HashMap[str, int]:
    """
    Count occurrences of each word (case-insensitive, no punctuation).
    O(n) average time.
    """
    hm: HashMap[str, int] = HashMap()
    for raw_word in re.split(r"\s+", text):
        word = re.sub(r"[^a-z]", "", raw_word.lower())
        if word:
            current = hm.get(word, 0)
            hm.put(word, (current or 0) + 1)
    return hm


def _benchmark(text: str) -> None:
    t0 = time.perf_counter()
    hm = word_frequencies(text)
    t1 = time.perf_counter()
    counter = Counter(
        re.sub(r"[^a-z]", "", w.lower())
        for w in re.split(r"\s+", text)
        if re.sub(r"[^a-z]", "", w.lower())
    )
    t2 = time.perf_counter()
    log.info("HashMap:        %.4f s, unique words=%d", t1 - t0, len(hm))
    log.info("Counter:        %.4f s, unique words=%d", t2 - t1, len(counter))
    # Verify counts match
    for k, v in counter.items():
        assert hm.get(k) == v, f"Mismatch for '{k}': hm={hm.get(k)}, counter={v}"
    log.info("Counts match.")


if __name__ == "__main__":
    import urllib.request
    # Use a public domain text (~10k words)
    url = "https://www.gutenberg.org/files/1342/1342-0.txt"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            text = resp.read(200_000).decode("utf-8", errors="ignore")
        log.info("Downloaded ~%d chars", len(text))
        _benchmark(text)
    except Exception as exc:
        log.warning("Network unavailable (%s); using local sample.", exc)
        sample = " ".join(["the quick brown fox jumps over the lazy dog"] * 1000)
        _benchmark(sample)
```

---

## Exercise 3 — Consistent hashing ring

**Key insight:** Each node is mapped to one or more positions on a 0..2³²−1 ring. A key is assigned to the first node clockwise from its hash position. Adding/removing a node only remaps keys in the arc it covers — roughly `1/N` of all keys, versus `N-1/N` with a simple modulo scheme.

```python
"""
solution_day3_ex3.py — consistent hashing ring
Python 3.12+
"""
from __future__ import annotations

import bisect
import hashlib
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _ring_hash(value: str) -> int:
    """Deterministic 32-bit ring position for a string."""
    return int(hashlib.md5(value.encode()).hexdigest(), 16) % (2**32)


class ConsistentHashRing:
    """
    Consistent hashing ring with virtual nodes for better distribution.

    Each real node is represented by `replicas` virtual nodes spread
    around the ring. This reduces hot-spots when node count is small.
    """

    def __init__(self, replicas: int = 100) -> None:
        self._replicas = replicas
        self._ring: dict[int, str] = {}        # position → node_id
        self._sorted_positions: list[int] = []

    def add_node(self, node_id: str) -> None:
        """Add a node (and its virtual replicas) to the ring. O(R log R)."""
        for i in range(self._replicas):
            pos = _ring_hash(f"{node_id}#{i}")
            self._ring[pos] = node_id
            bisect.insort(self._sorted_positions, pos)
        log.debug("Added node %r (%d replicas)", node_id, self._replicas)

    def remove_node(self, node_id: str) -> None:
        """Remove a node and its virtual replicas. O(R log R)."""
        for i in range(self._replicas):
            pos = _ring_hash(f"{node_id}#{i}")
            if pos in self._ring:
                del self._ring[pos]
                idx = bisect.bisect_left(self._sorted_positions, pos)
                self._sorted_positions.pop(idx)

    def get_node(self, key: str) -> str:
        """
        Return the node responsible for key.
        Finds the first ring position >= hash(key). O(log N·R).
        """
        if not self._ring:
            raise RuntimeError("No nodes in ring")
        pos = _ring_hash(key)
        idx = bisect.bisect_left(self._sorted_positions, pos)
        # Wrap around: if past the end, use the first node
        if idx == len(self._sorted_positions):
            idx = 0
        return self._ring[self._sorted_positions[idx]]


if __name__ == "__main__":
    ring = ConsistentHashRing(replicas=50)
    for node in ["node-A", "node-B", "node-C"]:
        ring.add_node(node)

    keys = [f"user:{i}" for i in range(20)]
    mapping_before = {k: ring.get_node(k) for k in keys}
    log.info("Before adding node-D:")
    for k, n in list(mapping_before.items())[:6]:
        log.info("  %s → %s", k, n)

    ring.add_node("node-D")
    mapping_after = {k: ring.get_node(k) for k in keys}
    remapped = sum(1 for k in keys if mapping_before[k] != mapping_after[k])
    log.info("After adding node-D: %d/%d keys remapped (~%.0f%%)",
             remapped, len(keys), 100 * remapped / len(keys))
    # Expect ~25% remapped (1/4 of keys), not 75% (as with naive mod N)
```

---

## Exercise 4 — Hashable `Point` dataclass

```python
"""
solution_day3_ex4.py — hashable frozen Point dataclass
Python 3.12+
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from hash_table import HashMap

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


@dataclass(frozen=True)    # frozen=True: __hash__ auto-generated, __setattr__ raises
class Point:
    """
    Immutable 2D point. Usable as a dict/HashMap key.

    frozen=True:
    - Generates __hash__ from all fields automatically.
    - Makes __setattr__ raise FrozenInstanceError → mutation impossible.
    - Two Point(1.0, 2.0) instances are equal and hash identically.
    """
    x: float
    y: float


if __name__ == "__main__":
    p1 = Point(1.0, 2.0)
    p2 = Point(1.0, 2.0)
    p3 = Point(3.0, 4.0)

    log.info("p1 == p2: %s", p1 == p2)          # True
    log.info("hash(p1) == hash(p2): %s", hash(p1) == hash(p2))   # True
    log.info("p1 == p3: %s", p1 == p3)          # False

    hm: HashMap[Point, str] = HashMap()
    hm.put(p1, "origin-ish")
    hm.put(p3, "further out")

    # Lookup using a new object with same values
    log.info("get(Point(1,2)): %s", hm.get(Point(1.0, 2.0)))  # origin-ish
    log.info("get(Point(3,4)): %s", hm.get(Point(3.0, 4.0)))  # further out

    # Attempt mutation — should raise
    try:
        p1.x = 99.0  # type: ignore[misc]
    except Exception as exc:
        log.info("Mutation blocked: %s", exc)

    # Also works as a built-in dict key
    builtin: dict[Point, str] = {p1: "A", p3: "B"}
    log.info("dict lookup: %s", builtin[Point(1.0, 2.0)])   # A
```

**Why `frozen=True` and not just manually defining `__hash__`?**

A mutable dataclass (or class) whose hash is based on mutable fields is dangerous: inserting into a dict then mutating the key leaves the key in the wrong bucket. `frozen=True` prevents mutation at runtime, making the hash guarantee permanent.

---

## Exercise 5 — Anagram groups

**Key insight:** The canonical key for an anagram group is the sorted tuple of its characters. Two words are anagrams if and only if their sorted-character tuples are identical. Use this as the HashMap key.

```python
"""
solution_day3_ex5.py — anagram grouping via HashMap
Python 3.12+
"""
from __future__ import annotations

import logging
from hash_table import HashMap

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def group_anagrams(words: list[str]) -> list[list[str]]:
    """
    Group words by anagram. O(n · k log k) where k = max word length.

    Key: sorted characters as a tuple (hashable, unlike list).
    Value: list of words sharing that canonical key.
    """
    hm: HashMap[tuple[str, ...], list[str]] = HashMap()
    for word in words:
        key: tuple[str, ...] = tuple(sorted(word))
        existing = hm.get(key)
        if existing is None:
            hm.put(key, [word])
        else:
            existing.append(word)   # mutates in place — no re-put needed
    return list(hm.get(k) for k in hm)  # type: ignore[misc]


if __name__ == "__main__":
    words = ["eat", "tea", "tan", "ate", "nat", "bat"]
    groups = group_anagrams(words)
    # Sort for deterministic output
    sorted_groups = sorted(sorted(g) for g in groups)
    log.info("Input:  %s", words)
    log.info("Groups: %s", sorted_groups)
    # Expected (order within groups and between groups may vary):
    # [['ate', 'eat', 'tea'], ['bat'], ['nat', 'tan']]

    # Edge cases
    log.info("Empty input: %s", group_anagrams([]))
    log.info("Single word: %s", group_anagrams(["hello"]))
    log.info("All same:    %s", group_anagrams(["abc", "bca", "cab"]))
```

**Why `tuple(sorted(word))` and not `''.join(sorted(word))`?**

Both work as HashMap keys here. The `tuple` form is marginally clearer in intent (an ordered collection of characters) and generalises to unicode where joining might behave unexpectedly. Either choice is correct.
