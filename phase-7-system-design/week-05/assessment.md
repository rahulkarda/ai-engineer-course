# Week 05 Assessment — Classic System Design Problems

> Time: ~1 hr  ·  Pass: ≥ 75%  ·  Open-book on diagrams; closed-book on algorithm choices

---

## Conceptual (interview-style)

Answer each question as you would in a live interview: concisely, with justification.

1. A URL shortener receives 1 000 write requests/s. You pre-generate 1 M short codes every hour. Is the pool sufficient? Show your math.

2. Explain the **boundary burst problem** in fixed-window rate limiting. Give a concrete numerical example. What algorithm eliminates this problem?

3. Your news feed system uses fanout-on-write for all users. A celebrity with 20 M followers posts once per minute. Describe the consequence and the solution.

4. In a chat system, a message is acknowledged by the server (server returns `message_id`) but the client crashes before receiving the ack. The user restarts the app. What happens? Walk through the recovery flow step by step.

5. Compare geohash and quadtree for spatial indexing. Name one scenario where quadtree outperforms geohash.

6. A typeahead trie has 500 K words. The top-K cache at every ancestor node is maintained on every insert. What is the time complexity of a single insert for a word of length L with top-K maintained at each node? Is this acceptable?

7. Your distributed rate limiter uses Redis. The Redis primary fails and the replica has not yet been promoted. You have two choices: (a) fail open (allow all requests) or (b) fail closed (deny all requests). Which do you choose for a public API and why?

8. A notification system must send at most 1 marketing email per user per day. The system runs across 50 pods. Using only Redis, design the deduplication mechanism. Include the Redis command(s), key structure, and TTL.

---

## Coding Challenge

### Part 1 — Sliding Window Counter Rate Limiter

Implement a sliding window counter rate limiter in Python backed by Redis. Requirements:

- `class SlidingWindowLimiter` with method `allow(key: str, limit: int, window_secs: int) -> bool`.
- Uses two Redis keys: current window counter and previous window counter.
- Implements the formula: `estimated = current_count + previous_count × (1 − elapsed_fraction)`.
- All Redis operations must be atomic (use a Lua script or pipeline).
- Fails open if Redis raises `ConnectionError`.
- Full type annotations. `logging` not `print`.

**Acceptance criteria**:
- `pytest` passes with ≥ 6 test cases:
  - First request allowed.
  - Requests at limit boundary (allowed and denied).
  - Window transition: previous window contributes proportionally.
  - Redis unavailable → fails open.
  - Different keys are independent.
  - Idempotent: calling `allow` N times with same key gives deterministic results.

### Part 2 — Typeahead Trie Extension

Extend the `Trie` class from Day 5 to support a `delete(word: str) -> bool` method that:
- Removes the word from the trie.
- Returns `True` if the word existed, `False` otherwise.
- Correctly updates the top-K cache at all ancestor nodes (remove the word from any node's top-K where it appeared).
- Does not remove nodes that are still prefixes of other words.

**Acceptance criteria**:
- `pytest` passes with ≥ 4 test cases covering: delete existing word, delete non-existent word, delete word that is prefix of another (other word still findable), verify top-K updates after delete.

---

## Rubric (pass ≥ 75%)

| Dimension | Weight | Notes |
|---|---|---|
| Independent implementation | 40% | Code runs and tests pass without referring to worked examples |
| Conceptual understanding | 25% | Answers in Section 1 show genuine reasoning, not memorized phrases |
| Debugging & problem-solving | 20% | Tests include sad paths; edge cases handled |
| Code quality & testing | 15% | Type hints, logging, no secrets, ruff passes |

### Scoring detail

- Each conceptual question: 5 pts (7 × 5 + Q8 = 40 pts, weighted to 25%).
- Part 1 implementation + tests: 35 pts.
- Part 2 implementation + tests: 25 pts.

Answer key: ../solutions/week-05/assessment.md
