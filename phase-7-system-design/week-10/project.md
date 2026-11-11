# Week 10 Project — Personal Story Bank, Design Doc & Behavioral Responses

> Deliverables due: End of Day 7 (Sunday)  ·  Estimated time: 6–8 hours across Sat/Sun

## Overview

This week's project is a personal communication toolkit — the artifacts a senior engineer needs to enter any technical interview or stakeholder conversation fully prepared. There are three deliverables:

1. **A personal story bank** — 8–10 complete STAR stories, covering all core behavioral competencies.
2. **A one-page technical design doc** — Written from scratch, using the standard template.
3. **Five written behavioral responses** — Answers to the five prompts below, in full STAR format.

These deliverables are personal. Do not share them publicly; they will contain real career experiences. Work in a private document or local markdown file.

---

## Deliverable 1 — Personal Story Bank

### Requirements

Write 8–10 complete STAR entries. Each entry must follow this template exactly:

```
## Story N: [Title — one sentence describing the event]

**Competencies (primary, secondary):** [e.g., Conflict (P), Leadership (S)]
**Role at the time:** [Your title]
**Date range:** [Approximate year/quarter]

### Situation
[2–3 sentences. Stakes and context. No company confidential detail required.]

### Task
[1 sentence starting with "My responsibility was..." or "I was tasked with..."]

### Action
- [Active verb] [specific step 1]
- [Active verb] [specific step 2]
- [Active verb] [specific step 3]
- [Active verb] [optional step 4]

### Result
[1–2 sentences: one quantified outcome + one learning sentence]

### Dual-use reframe
[1 sentence: how to open this story for the secondary competency]
```

### Coverage requirement

Your 8–10 stories must collectively include at least one Primary (P) coverage for each of:

| Competency | Covered by Story # |
|-----------|-------------------|
| Leadership | |
| Conflict resolution | |
| Failure / learning | |
| Impact / results | |
| Ambiguity | |
| Collaboration | |
| Mentoring / growing others | |
| Technical communication | |

Fill in the Story # column as you write. If any row is blank when you reach 8 stories, write an additional story to fill the gap — do not skip a competency.

### Quality criteria

- **Specificity:** A stranger with no context can understand the stakes.
- **Ownership:** "I" not "we" throughout the Action section.
- **Quantification:** Every Result section has at least one number (time, money, %, scale, adoption, lines of code, incidents prevented — any real metric).
- **Honesty:** The failure story must describe a real mistake with real consequences.
- **Recency:** At least 5 of your 8 stories are from the last 3 years.

---

## Deliverable 2 — One-Page Technical Design Doc

### Requirements

Write a complete design doc for a real or invented technical scenario using this template:

```markdown
# [Design Doc Title: Verb Phrase]

**Author:** [Your Name]  
**Date:** [YYYY-MM-DD]  
**Status:** Draft  

## Problem
[3–4 sentences. What is broken or missing? Quantified cost of inaction.]

## Goals
- [Measurable goal 1]
- [Measurable goal 2]
- [Measurable goal 3]
- [Measurable goal 4 (optional)]

## Non-goals
- [Explicitly out of scope item 1]
- [Explicitly out of scope item 2]
- [Explicitly out of scope item 3 (optional)]

## Proposed Solution
[300–400 words. The mechanism. Include a simple ASCII or Mermaid diagram if it clarifies the architecture.]

## Alternatives Considered
### Alternative 1: [Name]
[2–3 sentences: what it is, why rejected]

### Alternative 2: [Name]
[2–3 sentences: what it is, why rejected]

## Open Questions
1. [Question.] Owner: [Name]. Needed by: [Date].
2. [Question.] Owner: [Name]. Needed by: [Date].

## Success Metrics
- [Metric 1: named metric, target value, measurement method]
- [Metric 2: named metric, target value, measurement method]
```

### Quality criteria

- **Problem section** names a specific system and a quantified cost of inaction.
- **Goals** are all testable — you can write a binary pass/fail criterion for each.
- **Non-goals** are genuine in-scope-looking exclusions, not trivially obvious ones.
- **Alternatives** are real alternatives you actually considered (or would have), not strawmen.
- **Open questions** have named human owners, not "TBD."
- **Success metrics** name measurement methods, not just metrics.

---

## Deliverable 3 — Five Written Behavioral Responses

Write full STAR responses to these five prompts. Each response: 200–350 words, from your own experience.

**Prompt 1 — Ambiguity:**  
"Tell me about a time you had to make an important technical decision without all the information you needed."

**Prompt 2 — Conflict:**  
"Describe a situation where you disagreed with a technical direction and how you handled it."

**Prompt 3 — Failure:**  
"Tell me about a professional mistake or failure you are genuinely proud of how you handled."

**Prompt 4 — Impact:**  
"Describe your highest-impact technical contribution in the last 2 years."

**Prompt 5 — Mentoring:**  
"Tell me about a time you helped a colleague or team member grow."

For each response, label the four STAR sections explicitly in the margin.

---

## Submission checklist

- [ ] Story bank: 8–10 entries, all four STAR elements present in each
- [ ] Story bank: 8 competency columns each have at least one Primary story
- [ ] Story bank: every Result has a quantified outcome and a learning sentence
- [ ] Story bank: dual-use reframe written for at least 4 stories
- [ ] Design doc: all seven sections complete
- [ ] Design doc: problem section has quantified cost of inaction
- [ ] Design doc: at least 2 alternatives with rejection reasons
- [ ] Design doc: open questions have named owners and dates
- [ ] Behavioral responses: all 5 written, 200–350 words each
- [ ] Behavioral responses: all use "I" framing
- [ ] Behavioral responses: each has a quantified result and a learning sentence

---

## Rubric (pass ≥ 75%)

| Dimension | Weight | Pass criterion |
|-----------|--------|---------------|
| Story quality | 40% | All 8+ stories are specific, personally owned, quantified, and competency-mapped |
| Structure adherence | 25% | All stories follow STAR; design doc uses template; 5 responses are labeled and complete |
| Communication clarity | 20% | Stories are jargon-appropriate for an interview context; design doc is clear to a senior engineer unfamiliar with your system |
| Self-awareness & reflection | 15% | Result sections show genuine learning; failure story takes full ownership; reflection prompts show real self-assessment |

**Note:** This rubric adapts the course's standard 40/25/20/15 weighting (normally: independent implementation / conceptual understanding / debugging / code quality) for a non-coding week. "Story quality" replaces "independent implementation" as the primary production dimension; "self-awareness" replaces "code quality" as the signal for craft and discipline.

---

## Example project structure (for reference)

```
project-week-10/
├── story-bank.md          # 8–10 STAR entries (PRIVATE)
├── design-doc.md          # one-page technical design doc
└── behavioral-responses.md  # 5 written STAR responses (PRIVATE)
```

Answer key and worked examples: `../solutions/week-10/project.md`
