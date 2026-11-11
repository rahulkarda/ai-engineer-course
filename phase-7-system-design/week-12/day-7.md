# Phase 7 · Week 12 · Day 7 — Final Course Assessment, Reflection & Next Steps

> Objective: Complete a comprehensive self-assessment across all interview types, reflect on growth across the full course, and build a concrete post-course action plan.  ·  Time: ~3 hr  ·  Prereqs: All prior days; capstone doc from [[day-6]]

## Concept

Day 7 is the closing loop of the 18-month course. It has three components:

1. **Final mock self-assessment** (1.5 hr): timed mini-loops across all 5 interview types. Self-score against rubrics.
2. **Course reflection** (30 min): structured retrospective on growth, persistent gaps, and learning style.
3. **Next-steps planning** (1 hr): job search pipeline, skill maintenance cadence, and stretch goals.

## How it works internally

### Final mock self-assessment structure

Complete each mini-round in the time stated. Score yourself after each.

**Round 1 — Coding (25 min)**

Prompt: Implement a thread-safe, bounded channel (FIFO queue) with blocking `send(item)` and `receive() -> item`. Items block if channel is full/empty respectively. Include a `close()` method that causes pending receivers to unblock and return `None`.

Scoring rubric:

| Criterion | Points |
|---|---|
| Correct blocking behaviour (uses Condition, not busy-wait) | 25 |
| Thread-safe close propagation | 20 |
| Handles edge: send after close | 15 |
| Type hints + docstrings | 15 |
| 3+ pytest cases covering concurrent access | 25 |
| **Total** | **100** |

Pass: 75+

---

**Round 2 — LLD (20 min)**

Prompt: Design a plugin system for an AI assistant where plugins can: register themselves, subscribe to events (`on_message`, `on_tool_call`, `on_response`), and be enabled/disabled per user. Produce a Mermaid class diagram and define the `Plugin` Pydantic base model.

Scoring rubric:

| Criterion | Points |
|---|---|
| Plugin base class with event hooks | 25 |
| Registry with enable/disable per-user | 25 |
| Event dispatch is decoupled from plugin impl | 20 |
| Pydantic model for Plugin config | 15 |
| Extensibility: adding a new event needs 0 existing code changes | 15 |
| **Total** | **100** |

Pass: 75+

---

**Round 3 — HLD (30 min)**

Prompt: Design a notification service that sends emails, SMS, and push notifications for an e-commerce platform (10 M users, peak 50K notifications/s, delivery guarantee required).

Scoring rubric:

| Criterion | Points |
|---|---|
| Requirements + API contract before architecture | 20 |
| Named architecture diagram with all components | 25 |
| Delivery guarantee mechanism (idempotency + at-least-once) | 20 |
| Capacity estimate (queue depth, worker count) | 20 |
| Failure modes + mitigations | 15 |
| **Total** | **100** |

Pass: 75+

---

**Round 4 — Behavioral (15 min)**

Prompt: "Tell me about the most technically complex project you've built. What was the hardest decision you made and what was the impact?"

Scoring rubric:

| Criterion | Points |
|---|---|
| STAR structure clear (each phase identifiable) | 25 |
| Technical depth: specific numbers, technology, challenges | 25 |
| Individual contribution vs. team contribution distinguished | 20 |
| Quantified result | 20 |
| Lesson/growth demonstrated | 10 |
| **Total** | **100** |

Pass: 75+

---

**Round 5 — Capstone design-doc review (20 min)**

Prompt: Present your capstone design doc (Day 6) as if defending it to a panel. Identify: (a) the decision you are least confident in, (b) the scaling assumption most likely to be wrong, (c) what you would redesign with 6 more months.

Scoring rubric:

| Criterion | Points |
|---|---|
| Doc completeness (all 11 sections present) | 20 |
| Architecture diagram correct and legible | 15 |
| Honest identification of weakest decision | 20 |
| Scaling assumption critique is specific and justified | 20 |
| Redesign proposal is concrete and feasible | 25 |
| **Total** | **100** |

Pass: 75+

---

### Overall pass gate

Sum all 5 rounds. Weight: Coding 20%, LLD 20%, HLD 25%, Behavioral 15%, Capstone 20%.

Weighted score = (R1×0.20 + R2×0.20 + R3×0.25 + R4×0.15 + R5×0.20).

**Pass threshold: 75/100 weighted.**

---

### Course reflection (30 min)

Write answers to these five questions. Be honest. No one grades the reflection.

1. Which phase of the course produced the most durable skill change? What specifically shifted?
2. Which topic do you still reach for external resources for, rather than first principles? That is your current weakest area.
3. What was your biggest misconception at the start that the course corrected?
4. What study method worked best for you (spaced repetition, projects, mock interviews, teaching others)?
5. If you were starting over, what would you do in the first 30 days differently?

---

### Next-steps action plan (1 hr)

**Job search pipeline**

```
Week 1–2:  Polish top 3 projects on GitHub. Write READMEs with architecture diagrams.
Week 2–4:  Apply to 10 target companies. Tailor cover letter to each.
Week 3+:   2 mock interview sessions per week until first offer received.
           Track: company | stage | date | feedback in a simple table.
```

**Skill maintenance cadence**

```
Weekly:    1 coding problem (medium or hard) — LeetCode / HackerRank
Monthly:   1 full HLD mock (solo, timed, written)
Quarterly: 1 new AI/ML system to study and write a 1-page design critique
Ongoing:   Follow 3 engineering blogs (Stripe, Cloudflare, Anthropic)
```

**Stretch goals (pick 2)**

- Publish a technical blog post on your capstone design
- Contribute to an open-source AI project (LangGraph, Qdrant, Haystack)
- Build and deploy a public AI tool (side project)
- Pursue a cloud certification (AWS ML Specialty, GCP Professional ML Engineer)
- Mentor one other engineer going through a similar journey

## Exercises

The exercises for Day 7 are the five mock rounds above. Complete them in order, timed, before reading the solutions.

After scoring, fill out your Course Completion Record:

```
Name: ____________________________
Date: ____________________________

Round scores:
  Coding:       ___/100
  LLD:          ___/100
  HLD:          ___/100
  Behavioral:   ___/100
  Capstone:     ___/100

Weighted score: ___/100
Pass/Fail:      ___

Weakest area: _____________________
Next action:  _____________________
```

(Solutions: ../solutions/week-12/day-7.md)

## Self-check
- [ ] I have completed all 5 mock rounds timed
- [ ] I have an honest score for each round
- [ ] I have identified my weakest area and a concrete plan to address it
- [ ] I have written a post-course action plan with specific dates

## Supplementary
- [AI Engineer Foundation — Community](https://www.latent.space/)
- [The Pragmatic Programmer — Hunt & Thomas](https://pragprog.com/titles/tpp20/the-pragmatic-programmer-20th-anniversary-edition/)
- [Staff Engineer — Will Larson](https://staffeng.com/book)

---

*Congratulations on completing the 18-month AI Engineer course. You have the tools. Ship something.*
