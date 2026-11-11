# Phase 7 · Week 10 · Day 6 (Saturday) — Build: Personal Story Bank + Design Doc

> Objective: Produce a complete, polished personal story bank (8–10 entries) and a one-page technical design doc ready for the week project submission.  ·  Time: ~4 hr  ·  Prereqs: [[week-10/day-5]]

## Overview

Saturday is synthesis day. The goal is not to learn new concepts — it is to produce the two core deliverables for the week project with real depth and personal specificity. Allocate:

- **2 hours** — Complete and polish your story bank (8–10 STAR entries, full competency coverage).
- **1.5 hours** — Write the one-page technical design doc from scratch (real or invented scenario).
- **30 minutes** — Self-review against the rubrics in `project.md`.

Work in a private document, not in these course files. The deliverables are yours — personal, authentic, not shareable. The templates here are scaffolding.

---

## Part 1 — Complete your story bank (2 hours)

### Warm-up (15 min): Story inventory review

Review the story inventory you drafted earlier in the week. Apply these filters:

1. Remove any story where the outcome is entirely attributed to the team ("we delivered") with no specific "I" action.
2. Remove any story where you cannot state the result in one concrete metric or observable outcome.
3. Flag stories older than 5 years — keep only if they are the *only* story for that competency.

You should have 6–12 candidates. If fewer than 8, spend this warm-up time brainstorming 2–3 more from memory — focus on events you felt proud of, events where something went wrong, and events where you had to convince someone.

### Core work (75 min): Write full STAR entries

Use the template from Day 2. For each story:

**A: Situation** — Write 2–3 sentences. Test: Does a stranger with no company context understand the stakes?

**T: Task** — Write exactly 1 sentence starting with "I was responsible for..." or "My goal was..."

**A: Action** — Write 3–5 bullet points. Each starts with an active past-tense verb (built, designed, proposed, scheduled, wrote, identified, escalated, negotiated). Test: Could a reader extract your specific contribution from someone who was observing the same situation?

**R: Result** — Write 1–2 sentences. Must include: one quantified outcome (time, money, percentage, scale, or adoption), and one learning/reflection sentence.

### Polish pass (30 min): Dual-use mapping

For each story, add a "reframe line" — a single sentence that shifts the opening of the story to highlight a secondary competency. Example:

> Primary: "I want to tell you about a time I dealt with technical ambiguity..."  
> Secondary reframe (for leadership): "I want to tell you about a time I had to make a directional call for my team without all the information..."

Same facts, different opening emphasis. Write these reframes now while the stories are fresh.

---

## Part 2 — Write a one-page technical design doc (1.5 hours)

### Choose your scenario (10 min)

Pick a scenario that is realistic for your career level. Options:

**Scenario A — Real:** A design decision you actually made in the last 2 years that had non-trivial tradeoffs. Anonymize company/product names if needed.

**Scenario B — Invented:** Choose from:
- "Add a rate-limiting layer to an LLM inference API serving 100K requests/day"
- "Migrate a monolithic retrieval pipeline to a distributed RAG architecture"
- "Design an evaluation framework for a multi-agent customer support system"
- "Add real-time feature freshness to a batch-trained fraud detection model"

### Write the doc (80 min)

Use the template from Day 4:

```
Title: [Verb phrase]
Author: [Your name], Date: [Today], Status: Draft

## Problem
## Goals
## Non-goals
## Proposed solution
## Alternatives considered
## Open questions
## Success metrics
```

**Problem section target:** 3–4 sentences. Include a quantified cost of inaction.

**Goals section target:** 4–6 bulleted, measurable goals. "Reduce" is weak; "Reduce X from Y to Z by date" is strong.

**Proposed solution target:** 300–400 words. Include one architecture diagram as an ASCII or Mermaid block if helpful.

**Alternatives target:** 2 alternatives minimum. Each gets 2–3 sentences: what it is, why it was rejected.

### Architecture diagram (optional, within solution section)

ASCII is fine:

```
[Transaction Events]
        |
        v
[Kafka Topic: transactions]
        |
        v
[Consumer Group: feature-processors]
   |          |          |
   v          v          v
[account  [device   [merchant
 features]  features]  features]
        \       |       /
         v      v      v
         [Redis Feature Store]
               |
               v
        [Model Serving API]
```

### Self-review pass (10 min)

Before moving on, check:
- [ ] Problem has a quantified cost of inaction.
- [ ] Every goal is testable (you could write a pass/fail criterion).
- [ ] Non-goals explicitly exclude at least 2 things someone might reasonably expect.
- [ ] Alternatives are genuine alternatives, not strawmen.
- [ ] Open questions have named owners and dates.

---

## Part 3 — Self-review against project rubric (30 min)

Read `project.md` to see the full submission requirements and rubric. For each rubric dimension, write one sentence summarizing how your deliverables address it.

Rubric dimensions for this non-coding week:
- Story quality (40%) — Are your stories specific, personal, and measurable?
- Structure adherence (25%) — Do all stories follow STAR? Does the design doc use the template?
- Communication clarity (20%) — Are the best-audience version of each artifact clear and jargon-appropriate?
- Self-awareness and reflection (15%) — Do your result sections have genuine learning, not just positive outcomes?

---

## Exercises

The following are embedded in the work above. No separate exercise set — this day is about output, not more prompts.

(Solutions: ../solutions/week-10/day-6.md)

## Self-check

- [ ] I have 8–10 complete STAR entries in my private story bank
- [ ] Every story has a quantified result and a learning sentence
- [ ] I have dual-use reframe lines for at least 4 stories
- [ ] I have a complete one-page design doc in the template format
- [ ] I have self-reviewed against the project rubric and addressed any gaps

## Supplementary links

- [Levels.fyi behavioral interview prep examples](https://www.levels.fyi/blog/behavioral-interview-prep.html)
- [GitHub Engineering Blog — RFC examples](https://github.blog/engineering/)
- [Increment Magazine — On writing](https://increment.com/documentation/)
