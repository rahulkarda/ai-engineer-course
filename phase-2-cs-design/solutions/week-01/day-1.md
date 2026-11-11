# Solutions — Day 1: Big-O, Big-Theta, Big-Omega

## Exercise 1 answers

### (a) `sum_matrix` — n×n matrix
- **Time: O(n²)** — two nested loops each running n times → n² iterations total.
- **Space: O(1)** — only `total` accumulator, no auxiliary storage.

### (b) `count_bits`
- **Time: O(log n)** — the loop runs once per bit in n. An integer n has ⌊log₂n⌋ + 1 bits.
- **Space: O(1)** — only counter variable.

### (c) `fibonacci` (naive recursive)
- **Time: O(2ⁿ)** — recurrence T(n) = T(n-1) + T(n-2) + O(1). Each call branches into two, forming a tree of height n. Total nodes ≈ 2ⁿ.
- **Space: O(n)** — maximum call stack depth is n (leftmost branch).

---

## Exercise 2 — Master Theorem

`T(n) = 4T(n/2) + n`

- a = 4, b = 2, f(n) = n → n^c = n^1 → c = 1
- log_b(a) = log₂(4) = 2
- 2 > 1 → Case 1 of Master Theorem
- **T(n) = Θ(n²)**

---

## Exercise 3 — Rank by growth

From smallest (slowest growing) to largest (fastest growing):

`1 < log n < n < n log n < n² < n³ < 2ⁿ < n!`

---

## Exercise 4 — O(n) duplicate detection

```python
def has_duplicate_fast(items: list[int]) -> bool:
    """O(n) time, O(n) space — hash set."""
    seen: set[int] = set()
    for val in items:
        if val in seen:
            return True
        seen.add(val)
    return False
```

**Space trade-off:** O(n) space for the set instead of O(1). Classic time-space trade-off.

---

## Exercise 5 — True/False

**a) False.** "Big-O algorithms are always slower" is wrong. O notation hides constants. For small n, an O(n²) algorithm with constant 1 beats an O(n log n) algorithm with constant 1,000,000. The asymptotic claim only kicks in for "sufficiently large n."

**b) True.** Θ is symmetric: if f(n) ≤ c₁·g(n) and f(n) ≥ c₂·g(n) for large n, then g(n) is sandwiched between f(n)/c₁ and f(n)/c₂ — symmetric bounds.

**c) True.** f(n) = 5 satisfies f(n) ≤ 5·1 for all n ≥ 1. The constant c=5 witnesses the O(1) membership.
