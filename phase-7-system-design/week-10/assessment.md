# Week 10 Assessment — Behavioral Interviews & Technical Communication

> Time: ~1 hr  ·  Pass threshold: 75%

## Rubric adaptation note

This is a non-coding week. The standard rubric (40% independent implementation / 25% conceptual / 20% debugging / 15% code quality) is adapted as follows:

| Dimension | Weight | Adapted meaning |
|-----------|--------|----------------|
| Story quality | 40% | STAR completeness, specificity, quantified outcomes, personal ownership |
| Structure adherence | 25% | STAR format followed; design doc uses template; communication is audience-calibrated |
| Communication clarity | 20% | Technical concepts explained at correct depth; BLUF used where appropriate; review comments are labeled and well-formed |
| Self-awareness & reflection | 15% | Stories include learning sentences; failure story shows ownership; reflection prompts answered with genuine insight |

**Pass gate: ≥ 75% overall.**

---

## Part 1 — Conceptual questions (25% — structure adherence)

Answer each question in 2–4 sentences. Write your answers before checking the answer key.

1. What are the four elements of the STAR method, and what is the most common failure mode for each?

2. You are about to present a technical architecture decision to three audiences in sequence: (a) your engineering team, (b) a product manager, (c) the CTO. What is the one thing you would change between each version, and why?

3. What is the BLUF principle and why does it improve technical communication?

4. Name three explicit labels used in a well-structured code review, and explain what each signals to the author.

5. What is the difference between a design doc and an implementation ticket? When is each appropriate?

6. Describe the SBI feedback model. What does using it prevent compared to trait-based feedback?

7. What are three behavioral signals that distinguish an effective tech lead from a strong individual contributor?

8. In FDE (forward-deployed engineering) contexts, what is the most important behavioral competency that differs from a standard software engineering role, and why?

---

## Part 2 — Behavioral mock interview (40% — story quality)

Write full STAR responses (written format, 200–350 words each) to the following five prompts. Treat this as a real interview — write from your own experience.

**Prompt 1 — Ambiguity:**  
"Tell me about a time you had to make an important technical decision without all the information you needed. How did you decide, and what happened?"

**Prompt 2 — Conflict:**  
"Describe a situation where you disagreed with a technical direction — from your manager, a peer, or a cross-team stakeholder. How did you handle it?"

**Prompt 3 — Failure:**  
"Tell me about a professional mistake or failure you are genuinely proud of how you handled. What happened, and what changed as a result?"

**Prompt 4 — Impact:**  
"Describe your highest-impact technical contribution in the last 2 years. How do you measure it?"

**Prompt 5 — Mentoring / growing others:**  
"Tell me about a time you helped a colleague or team member grow. What was your approach and what was the outcome?"

### Scoring guide for Part 2 (per response, 8 points max):

| Criterion | Points |
|-----------|--------|
| All four STAR elements present and clearly labeled | 2 |
| "I" framing throughout (not "we") | 1 |
| At least one specific quantified outcome in Result | 2 |
| Learning / reflection sentence in Result | 1 |
| Total time ≤ 2.5 min when read aloud (~350 words max) | 1 |
| No vague language ("improved," "helped," "we delivered") | 1 |

Max: 8 points × 5 responses = 40 points.

---

## Part 3 — Design doc review (20% — communication clarity)

Read the following partial design doc and write three review comments. Each comment must be labeled (blocking / important / suggestion / question), include an observation, a reason, and a suggested action.

**Design doc excerpt:**

> **Title:** Add Caching to ML Inference API  
> **Author:** Dev Team, Date: 2024-10-15, Status: Draft
>
> **Problem:** The ML inference API is slow. Users are complaining. We need to add caching.
>
> **Goals:**  
> - Make the API faster  
> - Reduce load on the model server  
>
> **Proposed solution:**  
> Add a Redis cache in front of the model. Cache responses for 1 hour. This will make things faster.
>
> **Alternatives considered:**  
> - None considered at this time.

Write your three review comments below. Then answer: what is the single most important structural gap in this design doc?

---

## Part 4 — Communication rewrite (15% — self-awareness & reflection)

**Task:** The following is a real Slack message that buries the conclusion and uses jargon for a non-technical audience.

> "Hey team — so we ran the ablation study on the retrieval component last week. The MRL vs. standard dense embedding comparison showed a 12-point delta in recall@10 with a 40ms P95 regression. Given that the production SLA is 120ms and we're currently at 80ms P95, the 40ms headroom is exactly at the limit. We probably shouldn't ship the MRL approach in this release but maybe revisit after the infrastructure upgrade."

Rewrite this message for a product manager who needs to make a go/no-go shipping decision. Your rewrite should:
- Lead with the conclusion (BLUF).
- State the user-visible impact.
- Explain the risk in non-technical terms.
- End with a clear recommended action.

Target: 80–120 words.

Then answer in 2–3 sentences: What did you have to sacrifice from the technical message to make it work for a PM, and why is that sacrifice appropriate?

---

## Answer key

Answer key: ../solutions/week-10/assessment.md
