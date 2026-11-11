# Solutions — Week 10 · Day 4: Design Docs, RFCs & Code Review Communication

## Exercise 2 — Code review comment rewrites

### Original: "This is wrong."

**Well-formed rewrite:**
> **blocking:** The `user_id` lookup on line 34 queries the database inside the loop — this will execute N queries for N items in the batch. At the p99 batch size we see in production (~5000 items), this will cause serious latency. Move the lookup outside the loop and fetch all user records in a single query before processing begins. Similar pattern is already used in `batch_processor.py` lines 89–92.

---

### Original: "Please add tests."

**Well-formed rewrite:**
> **important:** The `compute_discount()` function has no test coverage. This function has three conditional branches (lines 22, 28, 34) that handle edge cases (negative balance, zero-quantity order, expired promotion code). Without tests, regressions in these branches are likely during future refactors. Please add unit tests for at least the three branch conditions before merge — you can follow the pattern in `tests/unit/test_pricing.py`.

---

### Original: "I would do this differently."

**Well-formed rewrite:**
> **suggestion:** I'd consider using `dataclasses.dataclass` here instead of a plain dict for the event payload (line 47). The current dict has no type enforcement, and I've seen `KeyError` issues in similar patterns when the schema evolved. A dataclass gives you type hints, IDE autocompletion, and makes the schema explicit at a glance. Not blocking — the current approach works — but worth considering if you're open to it.

---

## Exercise 3 — Design doc critique (worked example review comments)

**Review comment 1 — question:**
> **question:** Section "Open Questions" — item 1 asks about Redis eviction policy. Has the Infra team confirmed they can answer this by Nov 8? Is there a fallback plan if they can't? I ask because eviction policy affects both correctness (can stale features be served?) and the failure behavior under peak load, which seems like a blocker to finalizing the architecture.

**Review comment 2 — suggestion:**
> **suggestion:** The "Proposed solution" section doesn't address the failure mode when Redis is unavailable. The doc mentions a fallback to the nightly batch store on cache miss, but doesn't describe what happens if Redis becomes unavailable for more than a few minutes (e.g., the batch store falls further behind during that window). Consider adding a sentence on the failure mode and whether the fallback is automatic or requires an operator action.

**Review comment 3 — important:**
> **important:** The success metrics section measures feature staleness P99 and model serving latency, but doesn't include a metric for the fraud detection outcome itself (false negative rate or dollar impact). The Problem section justifies the project with an $200K/year fraud loss estimate — if we're not measuring that after launch, we can't confirm the project delivered ROI. Recommend adding one business-outcome metric to the success criteria.

**Single most important structural gap:** The design doc has no "Alternatives Considered" section in a meaningful sense — the two alternatives listed (DynamoDB and hourly batch) were rejected with one-line reasons that don't address the decision criteria. The strongest design docs show the decision criteria explicitly (latency, cost, operational burden) and score each alternative against them. Without this, reviewers cannot tell whether the analysis was thorough or whether the proposed solution was post-hoc rationalized.

---

## Exercise 4 — Non-goals explanation

**Value of explicit Non-goals:**

Explicit non-goals prevent two types of scope creep. First, they prevent reviewers from assuming the design doc addresses something it doesn't — if you don't say "we are not migrating historical data," someone will ask why you haven't. Second, they force the author to think through what they are deliberately deferring, which often surfaces scope decisions that haven't been consciously made. A non-goals section that is hard to write usually means the scope hasn't been fully decided.
