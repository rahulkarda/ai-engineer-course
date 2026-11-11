# Solutions — Week 10 · Day 1: The STAR Method & Story Bank Structure

## Exercise 2 — STAR critique

Original:
> "I once had to debug a hard production issue. It took a few days but I eventually found the problem and fixed it. The team was relieved."

**Weakest element:** Action (A). The entire action is described as "I eventually found the problem and fixed it" — zero specificity about what the person actually did.

**Specific information missing:**
- What was the production issue? (What system, what symptom, what impact on users?)
- What debugging steps were taken? (Logs, traces, reproducing locally, bisecting?)
- What was the root cause?
- How many days — and what happened during those days?
- What was the fix — a code change, a config change, a rollback?
- What is the quantified result? (Downtime minutes, affected users, revenue impact, incident severity?)
- What did the engineer learn?

**Restructured version:**

> **S:** Two hours after deploying a new recommendation model to production, our homepage load time increased from 120ms to 4.2s. The on-call alert fired at 2 a.m. UTC; 40,000 active users were affected.
>
> **T:** I was the on-call ML engineer. My job was to diagnose and resolve the regression without rolling back — the new model had just passed a three-week QA cycle.
>
> **A:**
> - Pulled distributed traces from Datadog and identified that 100% of the latency was concentrated in the recommendation service, not the home page API.
> - Isolated the change to the new model's inference path by comparing trace flame graphs before and after deploy.
> - Identified that the new model's embedding lookup was performing a full table scan instead of using the vector index — the index had not been recreated after a schema migration.
> - Rebuilt the index in a blue/green environment and promoted it without a model rollback.
>
> **R:** Latency returned to 118ms within 45 minutes of the index rebuild. I added an index health check to the deployment pipeline so the same failure mode is automatically caught before promotion. The incident was downgraded from P1 to P2 due to the fast resolution.

---

## Exercise 3 — Competency coverage checklist (reference)

Eight competencies to check off:
1. Leadership — driving change, aligning stakeholders
2. Conflict — handling disagreement constructively
3. Failure — ownership, resilience
4. Impact — moved a metric
5. Ambiguity — incomplete requirements
6. Collaboration — cross-team, elevating others
7. Mentoring — coaching, knowledge transfer
8. Technical communication — bridging technical and non-technical

If your 10 events don't cover all 8, the missing ones are the most important new stories to develop from memory or upcoming experiences.

---

## Exercise 4 — Dual-use reframe worked example

**Original "impact" opening:**  
"I want to tell you about a project where I redesigned the batch inference system and reduced cost by 60%."

**Reframed for "leadership":**  
"I want to tell you about a time I pushed for a technical direction that my team was skeptical of, made the case, built the prototype to prove it out, and then led the migration."

**What changes:** The opening sentence foregrounds the interpersonal and decision-making arc (skepticism, persuasion, proof-of-concept) rather than the outcome (cost reduction). The facts of the story are unchanged — the same actions produced the same result — but the opening primes the interviewer to listen for leadership signals.
