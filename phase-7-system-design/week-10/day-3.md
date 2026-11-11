# Phase 7 · Week 10 · Day 3 — Communicating Technical Decisions (Technical & Non-Technical Audiences)

> Objective: Learn audience-calibrated communication for technical decisions, and practice rewriting the same decision for engineers and executives.  ·  Time: ~1 hr  ·  Prereqs: [[week-10/day-2]]

## Concept

The same technical decision must be communicated differently depending on who is in the room. This is not dumbing things down — it is translating between value systems:

| Audience | Their primary concern | Language that lands |
|----------|----------------------|---------------------|
| **Engineers (your team)** | Correctness, tradeoffs, implementability | Precise, mechanism-level, acknowledges complexity |
| **Engineers (other teams)** | Impact on their systems, migration cost | Interface contract, backward compat, timeline |
| **Engineering manager / director** | Timeline, risk, headcount, delivery confidence | Milestones, dependencies, risk mitigation |
| **Product / business stakeholders** | User impact, revenue, competitive advantage | User-visible behavior, metrics, market timing |
| **Executives (CTO, VP)** | Strategic alignment, cost, irreversibility | One-pager, tradeoffs at portfolio level |
| **Customers (FDE context)** | Does it solve my problem, is it reliable | Outcome language, no jargon, concrete example |

The critical skill is not having six different voices — it is knowing which concerns matter to which audience and leading with those.

### The decision communication arc

For any significant technical decision, you need three artifacts:

1. **The design doc / RFC** — full technical detail, for engineers. (Thursday's topic.)
2. **The executive summary** — half a page, for non-technical stakeholders.
3. **The verbal briefing** — 2–3 minutes, for meetings.

All three describe the same decision. The difference is depth and vocabulary.

## How it works internally

### The BLUF principle (Bottom Line Up Front)

Borrowed from military communication: state your conclusion first, then support it. Most technical writers do the opposite — they build up context and announce the conclusion at the end. This wastes the audience's time and buries the signal.

**Wrong order (conclusion last):**
> "We evaluated PostgreSQL, DynamoDB, and Redis for this use case. Each has different consistency models. Given our read/write ratio, PostgreSQL's transactional guarantees, and our team's existing expertise, we recommend PostgreSQL."

**BLUF order (conclusion first):**
> "We recommend PostgreSQL. Rationale: our workload is 90% reads, our team already operates it, and we need ACID transactions. DynamoDB and Redis don't meet the consistency requirement."

Same content, half the cognitive load.

### The three-layer explanation model

For technical decisions explained to non-technical audiences:

**Layer 1 — What (the outcome, user-visible):**  
"The search results will appear in 300ms instead of 4 seconds."

**Layer 2 — Why it matters (business / user value):**  
"This matches the speed of our main competitor and reduces search abandonment by an estimated 20%."

**Layer 3 — How (just enough mechanism):**  
"We pre-compute the rankings offline instead of recalculating on every request."

Executives need all three. Customers need 1 and 2. Engineers need 1, 2, and full depth on 3.

### The analogy tool

When introducing a mechanism an audience cannot visualize, use an analogy followed immediately by the point of divergence:

> "Think of our feature store like a library card catalog — you look up a book by title (customer ID) and immediately get the shelf location (feature vector). Unlike a real catalog, ours updates in real time and has millisecond lookup guarantees."

The analogy creates a mental model. The divergence prevents oversimplification from becoming misinformation.

## Worked example

**Decision:** Migrate from a monolithic ML pipeline to a streaming feature computation system.

### Version 1 — for engineers (design doc intro paragraph)

> We are replacing the nightly batch feature computation job (`ml-pipeline/nightly_features.py`) with a Kafka-backed streaming processor that computes features within 30 seconds of the triggering event. The batch system introduces up to 24 hours of feature staleness, which is degrading model accuracy on time-sensitive predictions (see attached A/B results). The new architecture uses a Kafka consumer group (one consumer per feature family), a Redis feature store, and a promotion gate to prevent stale features from reaching the serving layer. No changes are required to the model serving contract.

### Version 2 — for a business stakeholder (email or Slack summary)

> We're changing how our ML system gets its data from "updated once a night" to "updated within 30 seconds." This matters because our fraud detection model was making decisions based on information that could be a day old. Early tests show this change reduces false negatives by 18%, which directly translates to ~$200K/year in prevented fraud losses. Engineering work is 6 weeks; no user-visible changes during the migration.

### Version 3 — verbal briefing for an executive (2-minute version)

> "We have a problem where our fraud model is working off day-old data. We found out that 18% of missed fraud cases happened in the window where the model didn't have the freshest information. The fix is to move from a nightly batch update to a real-time feed — data refreshes within 30 seconds instead of 24 hours. Our estimate is $200K/year in prevented losses, and it takes 6 weeks to build. The main risk is the infrastructure change, which we've mitigated by building a rollback to the batch system. We're asking for 2 engineering headcount for those 6 weeks and are ready to start Monday."

**What changed across versions:**
- Version 1 names files, systems, and architecture components.
- Version 2 names no technical systems; uses user-visible outcomes and a money number.
- Version 3 opens with the problem (not the solution), uses "we," and ends with a clear ask.

## Exercises

1. **Rewrite exercise.** Below is a technical explanation for engineers. Rewrite it for a product manager (no technical jargon, outcome-first, include a business impact number you invent plausibly):
   > "We're switching the embedding layer from a 768-dimensional dense vector to a 1536-dimensional sparse-dense hybrid using BM25 + MRL. The change increases recall@10 by 12 points on our internal eval set and adds approximately 40ms to P95 latency due to the dual-pass retrieval."

2. **Analogy construction.** Pick one of these concepts and write a two-sentence analogy + divergence for a non-technical audience: (a) vector similarity search, (b) rate limiting, (c) eventual consistency.

3. **BLUF practice.** Take a recent technical email or Slack message you have written (or invent one) that buries the conclusion. Rewrite it BLUF style. How many words shorter is the rewrite?

4. **Stakeholder mapping.** For a current or recent project: list the 4–5 stakeholders and what each one cares about most. Write one sentence per stakeholder describing what you would lead with in a status update.

(Solutions: ../solutions/week-10/day-3.md)

## Self-check

- [ ] I can identify the primary concern for each of the six audience types
- [ ] I can apply the BLUF principle to any technical write-up
- [ ] I can write the same decision at three levels of technical depth
- [ ] I can construct a two-sentence analogy with a stated divergence
- [ ] I can deliver a 2-minute verbal briefing of a technical decision to an executive

## Supplementary links

- [Amazon's writing culture — working backwards press release](https://www.allthingsdistributed.com/2006/11/working_backwards.html)
- [The Pyramid Principle — Barbara Minto (summary)](https://www.barbaraminto.com/textbook.html)
- [Google Technical Writing Course](https://developers.google.com/tech-writing)
