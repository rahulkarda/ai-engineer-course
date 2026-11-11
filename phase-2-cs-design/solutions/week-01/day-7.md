# Solutions — Day 7: Week Project + Assessment Day

No formal exercise solutions for Day 7 — see assessment.md solution and project.md solution.

## Day 7 reflection answers

**1. Complexity analogy (merge sort for a non-programmer):** Sorting a stack of 1,000 cards by splitting into two piles, sorting each pile, then merging is faster than comparing every card to every other. Each split halves the problem; each merge is one pass — so you do "log 1000 ≈ 10 passes" of "1000 comparisons each" = 10,000 steps vs the brute-force million.

**2. Production incident example:** A `list.insert(0, x)` in a loop processing 1M records — O(n²) means 10¹² operations. With a 1GHz machine doing 10⁹ operations/sec this would take ~1000 seconds where `append` would finish in milliseconds. Alternatively: string concatenation in a loop (`s = s + new_chunk`) has the same problem.

**3. Relationship between amortized append and sliding window O(n):** Identical accounting argument. In append: each element "pays" for its own future copy during resize. In sliding window: each element "pays" for its own future removal from the window. In both cases the aggregate cost of all "expensive" events (resize / left-pointer advance) is bounded by the total number of elements = n.
