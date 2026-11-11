# Phase 7 · Week 10 · Day 2 — Building Your Story Bank (Competency Coverage)

> Objective: Build a complete, well-balanced story bank covering all core behavioral competencies with full STAR entries.  ·  Time: ~1 hr  ·  Prereqs: [[week-10/day-1]]

## Concept

A story bank is not a collection of favorite war stories — it is a deliberately constructed inventory designed to give you at least one strong response to any behavioral question. The goal is **competency coverage with story flexibility**.

### The eight core competency areas

| Competency | What interviewers probe for |
|------------|---------------------------|
| **Leadership** | Driving change without formal authority, aligning stakeholders, making hard calls |
| **Conflict** | Disagreeing constructively, navigating interpersonal or cross-team friction |
| **Failure** | Taking ownership of a mistake, demonstrating resilience and learning |
| **Impact** | Moving a metric, shipping a product, reducing cost or risk at scale |
| **Ambiguity** | Operating with incomplete requirements, making defensible decisions under uncertainty |
| **Collaboration** | Working across teams, elevating others, giving credit |
| **Mentoring / growing others** | Coaching, knowledge transfer, building team capability |
| **Technical communication** | Explaining complex ideas to non-technical audiences, writing for clarity |

A bank of 8–10 stories, each mapped to 2–3 competencies, gives you full coverage with room for the interviewer to probe.

### Story selection principles

**Recency:** Stories from the last 2–3 years carry more weight. Use older stories only for competencies where you have no recent example.

**Reversibility:** Stories where you made a decision that later needed reversing are gold — they show judgment *and* humility.

**Scope appropriateness:** Match story scale to the role level. An IC3 can tell a story about improving a team's testing culture; a staff engineer should tell a story about changing an org's direction.

**Specificity beats impressiveness.** A story with exact numbers, dates, and named decisions at a small company is more credible than a vague story about "saving" a major product at a large company.

## How it works internally

### The story matrix

Build a two-axis matrix:

- Rows: your 8–10 stories (one per row)
- Columns: the 8 competency areas + "role level fit" + "most recent use"

Mark each cell with P (primary), S (secondary), or blank. Primary means the story's central arc is about that competency. Secondary means you can foreground that angle if asked.

A healthy matrix has:
- Every competency column has at least one P.
- No story covers more than 3 competencies as P (too diffuse = too complicated).
- At least 5 stories are from the last 3 years.

### Failure stories deserve special attention

Failure stories are the highest-signal questions in a senior interview. Interviewers use them to filter for:

1. **Ownership:** Did you accept responsibility or deflect to circumstances?
2. **Learning:** Did the failure produce a durable change in behavior?
3. **Scale:** Was the failure significant enough to demonstrate real stakes?

A mediocre failure story: "We missed a deadline because the requirements were unclear."  
A strong failure story: "I shipped a model to production without adequately testing distribution shift. It degraded silently for three weeks before a customer reported it. I then built the monitoring layer that would have caught it."

The difference: specificity, personal ownership ("I shipped"), concrete consequence, and a tangible fix.

## Worked example

### Full story bank entry — Conflict competency

**Title:** Cross-team API versioning disagreement  
**Competencies:** Conflict (P), Leadership (S), Technical communication (S)

**Situation:**  
Our ML platform team was about to ship a breaking change to the feature store API. The downstream team (data scientists with 15 active pipelines depending on the API) pushed back hard in Slack, escalating to VP level and asking to block the release.

**Task:**  
I was the tech lead for the platform team. I needed to get the API change shipped within two weeks (hard infra deadline) while preserving the relationship with the DS team and not creating a production incident.

**Action:**  
- Scheduled a 45-minute joint working session — explicitly not a debate, framed as "design the migration together." I sent the agenda 24 hours ahead with the non-negotiables labeled clearly (the deadline, the core schema change) and the open questions labeled as open.
- In the session, I invited the DS team lead to list their actual pain points rather than their proposed solution (blocking the release). Three of four concerns were solvable with a compatibility shim I hadn't considered.
- Wrote a two-page migration guide and offered to pair-program the first pipeline migration with them, completing it in a shared session that same week.
- Sent a written summary to both VPs before the escalation meeting, which was then cancelled as resolved.

**Result:**  
Shipped on schedule. Zero DS pipelines broke. The DS team lead later cited this interaction as the reason they started engaging earlier on future platform changes, reducing our average API-change lead time from 6 weeks to 2 weeks. I learned that escalations almost always contain a hidden negotiating position — the first step is to separate stated positions from actual needs.

---

### Story bank entry template (fill in for each of your stories)

```
Title: [one sentence]
Competencies: [Primary], [Secondary], [Secondary]
Role: [your title at the time]
Company/team context: [1 sentence, no confidential detail]
Date range: [approximate year/quarter]

S: [2–3 sentences — stakes and context]
T: [1 sentence — your specific responsibility]
A:
  - [active verb] [specific step 1]
  - [active verb] [specific step 2]
  - [active verb] [specific step 3]
  - [active verb] [optional step 4]
R: [metric or outcome] + [what you learned / what you would do differently]

Dual-use reframe for [secondary competency]: [1 sentence — how to pivot the opening]
```

## Exercises

1. **Complete your story matrix.** Using the template above, write full STAR entries for at least 6 stories. For each, note: which competency is primary, and what is the single strongest sentence in the Result section?

2. **Write a failure story.** This is the one competency most people avoid. Write a STAR story where the outcome was negative, where you owned it explicitly, and where the result section focuses on the systemic change you made afterward. Target length: 250–300 words.

3. **Coverage audit.** Map your completed stories to the 8 competency columns. Which column has no P-coverage? Draft a short story (even a thin one) to fill that gap.

4. **30-second version.** Take your strongest story. Compress it to a 30-second version (≈70 words) by keeping only the most memorable Situation sentence, the core Action verb, and the headline Result number. When would you use this version vs. the full version?

(Solutions: ../solutions/week-10/day-2.md)

## Self-check

- [ ] I have 6–10 complete STAR entries written out
- [ ] My failure story takes full ownership without blaming circumstances
- [ ] I can map every story to at least one primary competency
- [ ] I have no competency column with zero coverage
- [ ] I can deliver any story in a 30-second summary or a 2.5-minute full version

## Supplementary links

- [Holloway Guide to Technical Interviewing](https://www.holloway.com/g/technical-interviewing)
- [Julia Evans — Asking good questions in interviews](https://jvns.ca/blog/good-questions/)
- [StaffEng.com — Stories from staff engineering promotion packets](https://staffeng.com/stories/)
