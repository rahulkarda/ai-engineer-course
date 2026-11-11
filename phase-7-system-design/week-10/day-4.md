# Phase 7 · Week 10 · Day 4 — Design Docs, RFCs & Code Review Communication

> Objective: Write a one-page technical design doc and communicate code review feedback that is specific, actionable, and constructive.  ·  Time: ~1 hr  ·  Prereqs: [[week-10/day-3]]

## Concept

### Design docs and RFCs

A **design doc** (sometimes called RFC — Request for Comments) is a written proposal for a non-trivial technical change. Its purpose is to:

1. Force the author to think through edge cases before writing code.
2. Create a forum for async feedback without requiring everyone in a meeting.
3. Produce a durable record of *why* a decision was made (more valuable than the decision itself).

Design docs are distinct from implementation tickets (those specify *what*); they exist to argue *why* and record the tradeoffs.

**When to write one:**
- The change affects more than one team's systems.
- The decision is hard to reverse (schema changes, API contracts, infra migrations).
- Reasonable engineers would disagree on the right approach.
- The change introduces new security or reliability risk.

**When not to write one:**
- Single-engineer bug fixes.
- Changes fully contained within your own service with no external contract change.
- Refactors with no behavior change.

### The one-page design doc template

A one-page design doc is not literally one page — it is a document that can be read and understood in under 10 minutes. The canonical sections:

```
Title: [Verb phrase: "Migrate Feature Store to Streaming Architecture"]
Author: [Name], Date: [YYYY-MM-DD], Status: [Draft | In Review | Approved | Implemented]

## Problem
[2–4 sentences. What is broken or missing? What is the cost of inaction?]

## Goals
[Bulleted list, 3–5 items. Each goal is testable or measurable.]

## Non-goals
[Explicit list of what this doc does NOT address. Prevents scope creep.]

## Proposed solution
[The mechanism. Include a simple diagram if needed. No longer than half a page.]

## Alternatives considered
[2–3 alternatives, each with one sentence on why they were rejected.]

## Open questions
[Numbered list of unresolved items. Who owns each? By when?]

## Success metrics
[How will you know it worked? Named metrics, targets, and timeline.]
```

### Code review communication

Code review is a communication act, not just a technical act. The way feedback is phrased determines whether the review improves the code or damages the relationship — or both.

**The three failure modes of code review comments:**

1. **Vague:** "This could be better." — The author has no path forward.
2. **Prescriptive without explanation:** "Use a list here." — The author doesn't know why, so they can't generalize.
3. **Rude or condescending:** "Why would you do it this way?" — Signals disrespect, poisons future collaboration.

**The well-formed review comment has four parts:**

1. **Observation** — What you see (no judgment).
2. **Why it matters** — The principle or risk at stake.
3. **Suggestion** — One concrete path forward (not the only path).
4. **Qualifier** — Is this blocking, or a nit you can take or leave?

## How it works internally

### Review comment taxonomy

Use explicit labels so the author knows the severity:

| Label | Meaning |
|-------|---------|
| **blocking:** | Must be addressed before merge — correctness or security issue |
| **important:** | Strong recommendation; discuss if you disagree |
| **suggestion:** | Take-it-or-leave-it improvement |
| **nit:** | Minor style point; author's call |
| **question:** | Genuinely asking, not criticizing |
| **praise:** | Explicitly positive — name what you learned or appreciated |

Praise is not optional. Studies of code review culture show that reviews with zero positive comments depress future PR submission rates.

### Design doc review vs. code review

| Dimension | Design doc review | Code review |
|-----------|------------------|-------------|
| Timing | Before coding starts | After coding is done |
| Audience | Broader; cross-team | Usually just the team |
| What to critique | Problem framing, alternatives, scope, risk | Correctness, style, testability, security |
| Tone | Exploratory; "have you considered..." | Specific; labeled by severity |

## Worked example

### Design doc — one-page example

```markdown
Title: Replace Nightly Batch Feature Computation with Kafka Streaming
Author: Priya Mehta, Date: 2024-11-01, Status: In Review

## Problem
Our fraud detection model reads features computed during a nightly batch job.
The maximum feature staleness is 24 hours. Analysis of Q3 fraud incidents shows
that 18% of missed fraud events occurred within the staleness window. The cost
is estimated at ~$200K/year in undetected fraud.

## Goals
- Reduce feature staleness from 24 hours to ≤ 30 seconds for all real-time fraud signals.
- Maintain current model serving latency (P95 ≤ 120ms).
- Allow rollback to batch system within 2 hours if streaming system fails.

## Non-goals
- Changing the model architecture or retraining the model.
- Migrating historical feature data.
- Serving features to non-fraud use cases.

## Proposed solution
Deploy a Kafka consumer group (one consumer per feature family: transaction, account,
device) that processes events from the existing transaction Kafka topic. Features are
written to Redis with a 48-hour TTL. The model serving layer reads from Redis; on
cache miss it falls back to the nightly batch store (guaranteeing no regression).

## Alternatives considered
- **DynamoDB instead of Redis:** Higher latency (~10ms vs. ~0.3ms), higher cost at our
  read volume. Rejected.
- **Sync the batch job more frequently (hourly):** Infrastructure cost of running the
  full batch pipeline hourly is prohibitive; does not meet the 30-second goal.

## Open questions
1. What is the Redis eviction policy under peak load? Owner: Infra team. Needed by: Nov 8.
2. Do we need feature versioning for the Redis store? Owner: ML team. Needed by: Nov 10.

## Success metrics
- Feature staleness P99 ≤ 30s in production (measured via feature audit log).
- Zero increase in model serving P95 latency.
- Measured in week 2 post-launch via Datadog dashboard.
```

---

### Code review comment examples

**Vague (bad):**
> "This function is too long."

**Well-formed (good):**
> **suggestion:** This function is 80 lines and handles both validation and persistence. Consider splitting at line 42 — the validation logic could be a separate `validate_payment_request()` function. This makes the persistence path easier to test in isolation and aligns with the single-responsibility principle already used in `order_service.py`.

**Prescriptive without explanation (bad):**
> "Use a dict instead."

**Well-formed (good):**
> **important:** The linear scan over `allowed_users` (line 17) is O(n) on every request. At our current user base (~50K users), this is measurable latency. Consider converting to a `set` — lookup becomes O(1). One-line change: `allowed_users: set[str] = set(config.ALLOWED_USERS)`.

## Exercises

1. **Write a design doc.** Choose a real or invented technical change (e.g., "Add a rate-limiting layer to our ML inference API"). Write a one-page design doc using the template above. It should include at least 2 alternatives considered and 2 open questions.

2. **Code review comment rewrite.** Rewrite the following comments to be well-formed, with observation, why-it-matters, suggestion, and label:
   - "This is wrong."
   - "Please add tests."
   - "I would do this differently."

3. **Design doc critique.** Review the worked example design doc above. Write three comments you would leave in a design doc review: one question, one suggestion, and one thing you would call out as a potential gap.

4. **Non-goals exercise.** For your design doc from exercise 1, add a "Non-goals" section. Why is explicitly stating non-goals valuable? Write two sentences on this.

(Solutions: ../solutions/week-10/day-4.md)

## Self-check

- [ ] I can name the six sections of the one-page design doc template
- [ ] I know when to write a design doc vs. when it is overkill
- [ ] I can label a code review comment with the correct severity tier
- [ ] I can rewrite a vague code review comment to be specific, principled, and actionable
- [ ] I have written at least one full design doc in the exercises

## Supplementary links

- [Google Design Doc Template (public version)](https://www.industrialempathy.com/posts/design-docs-at-google/)
- [Squarespace RFC process](https://engineering.squarespace.com/blog/2019/the-power-of-yes-and)
- [The Code Review Pyramid (Gunnar Morling)](https://www.morling.dev/blog/the-code-review-pyramid/)
