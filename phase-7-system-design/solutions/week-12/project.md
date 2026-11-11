# Solutions — Week 12 Project: Capstone Design Doc + Mock Interview Self-Assessment

## Component A: Capstone Design Document — Grading Guide

The full model capstone design document is provided in `solutions/week-12/day-6.md`. Use it as the calibration baseline.

### Section-by-section grading notes

**Executive summary (250+ words)**
- Pass: names the system, the approach (RAG + LangGraph + self-hosted inference), the top 3 trade-offs, and expected outcomes with at least one metric.
- Fail: generic description ("we built an AI assistant using modern technologies").

**Requirements**
- Pass: functional requirements as a numbered list; non-functional requirements as a table with specific SLAs matching the given numbers (800 QPS, p50 < 1.5 s, etc.).
- Fail: missing the given scale numbers, or only functional requirements listed.

**Architecture diagram**
- Pass: named services with directional data flows; shared infrastructure listed separately; inference cluster visible.
- Fail: boxes with no labels, or a diagram that omits the vector DB or the LLM inference layer.

**Component deep-dives**
- Pass: 6 components minimum; each has a technology choice with one-line rationale; not just a list of features.
- Fail: 3 components or fewer; rationale missing ("we used Qdrant because it's good").

**Capacity planning**
- Pass: shows arithmetic. Example: "30K DAU × 16 queries/day = 480K/day. Peak 10×: 800 QPS. At 700 tokens avg output and 3K tokens/s per GPU: need 187 parallel streams → 4× A100."
- Fail: "it will scale horizontally."

**Security**
- Pass: mentions OIDC auth, tenant isolation mechanism (namespace or RLS), prompt injection defence, PII handling.
- Fail: only mentions "authentication required."

**Cost model**
- Pass: itemised table with monthly totals; LLM cost based on token math; monthly total stated.
- Fail: no numbers, or only one line item.

---

## Component B: Mock Interview Self-Assessment — Model Report Structure

```markdown
# Mock Interview Self-Assessment

**Name**: [Your Name]
**Date**: 2026-11-__
**Cohort**: Phase 7, Week 12 (Capstone)

---

## Summary Scorecard

| Round | Score (/100) | Pass/Fail (≥75) |
|---|---|---|
| Coding (Bounded Channel) | 82 | Pass |
| LLD (Plugin System) | 78 | Pass |
| HLD (Notification Service) | 71 | Fail |
| Behavioral | 85 | Pass |
| Architecture Critique | 90 | Pass |
| **Weighted total** | **80.9** | **Pass** |

Weighted calculation:
(82×0.20) + (78×0.12) + (71×0.13) + (85×0.08) + (90×0.20) + (capstone×0.20) + (code_quality×0.07)

---

## Per-Round Reflection

### Coding
Score: 82/100

What went well: Correct blocking behaviour using Condition (not busy-wait). All 4 test cases written and passing.

What failed: Missed send-after-close edge case in first draft. Added after re-reading the spec.

Root cause of gap: Habit of implementing the happy path first without reading edge cases until end.

Fix: Explicitly write edge cases to test before implementing — test-first even in interviews.

---

### LLD
Score: 78/100

What went well: Plugin base class with event hooks correct; registry design clean.

What failed: Forgot to specify that EventDispatcher calls plugins in registration order. Extensibility criterion partially missed — adding a new event still requires adding the method to the Plugin base class.

Root cause: Didn't think through "what does adding a new event require?" in design phase.

Fix: Practice the extensibility question explicitly: "If requirements change in X way, how many files change?"

---

### HLD
Score: 71/100 (Fail)

What went well: Requirements table complete; API contract correct.

What failed: Capacity estimate missing worker count calculation. Idempotency mechanism described but not how the idempotency key is generated and checked. Only 1 failure mode instead of 2.

Root cause: Ran out of time on capacity estimate because I spent too long on the architecture diagram.

Fix: Time-box each section strictly (use a timer). Architecture diagram gets 8 minutes maximum — move on even if imperfect.

---

### Behavioral
Score: 85/100

What went well: Both STAR answers had clear structure; result was quantified in both.

What failed: Prompt B answer used "we" in the Action section twice. Individual contribution less clear than it should be.

Root cause: Default to team framing when describing collaborative work.

Fix: Audio-record my STAR answers and count "we" vs. "I." Replace every "we" in the Action section with "I."

---

### Architecture Critique
Score: 90/100

What went well: Identified all 5 flaws, categorised correctly, proposed specific fixes.

What failed: Tenant isolation fix was mentioned but not explained at the level of "how does RLS work."

Root cause: Familiar with the concept but haven't implemented Postgres RLS personally.

Fix: Implement a working Postgres RLS example this week.

---

## Overall Reflection

Strongest area: Architecture critique — pattern recognition is strong after 12 weeks of design practice.

Weakest area: HLD capacity estimation — consistently running out of time on math because diagram takes too long.

One concrete action in the next 2 weeks: Complete 5 timed HLD mocks (45 min each). In each, allocate max 8 min to diagram and start capacity math by minute 30.

---

## Course Completion Record

Weak areas identified:
1. HLD time management (diagram vs. math trade-off)
2. Postgres RLS implementation depth
3. "We" vs. "I" discipline in behavioral answers

Next 30 days action:
- 5 timed HLD mocks (Saturdays, 45 min each)
- Build and deploy one working Postgres RLS example
- Audio-record 3 behavioral answers; eliminate "we" from Action sections
- Apply to 5 target companies with tailored materials
```

---

## Grading rubric applied to self-assessment report

| Criterion | Pass indicator |
|---|---|
| All 5 rounds scored | Yes/No |
| Scores are calibrated (not all 100/100) | Honest variation visible |
| Root cause identified per failure | Specific, not "I need more practice" |
| Fix is concrete and time-bounded | "Do X by Y date", not "study more" |
| Overall reflection identifies 1 specific action | Named, with a date or count |

**Common failures in self-assessment reports**:
- Scores are inflated (all 85+/100) — interview performance is routinely lower than self-assessed. If your scores are all above 85, re-run the rounds with a stricter rubric.
- Root cause is generic: "I need to study more HLD" → insufficient. Should be: "I spent too long on diagrams and ran out of time for capacity math. Fix: 8-minute diagram timer."
- No concrete action — "improve behavioral answers" is not an action. "Record 3 STAR answers this week and count we/I ratio" is.
