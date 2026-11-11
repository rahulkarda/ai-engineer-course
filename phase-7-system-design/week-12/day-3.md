# Phase 7 · Week 12 · Day 3 — Behavioral + Project Deep-Dive Mock Loop

> Objective: Run a full behavioral and project deep-dive mock loop; practice STAR delivery and technical storytelling.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

## Concept

Behavioral and project deep-dive rounds test judgment, collaboration, and growth — not just technical skill. They account for 1–2 out of 5–6 rounds at most top companies. Weak behavioral performance eliminates otherwise strong technical candidates.

**Behavioral round** (30–40 min): 3–5 STAR-format questions, usually mapped to company leadership principles. Interviewers probe for: ownership, impact, conflict resolution, failure mode, learning.

**Project deep-dive** (45 min): you walk through one project end-to-end. The interviewer probes technical decisions, your individual contribution, what failed, and what you'd do differently.

## How it works internally

### STAR framework (non-negotiable structure)

```
S — Situation  (2-3 sentences, context only, no padding)
T — Task       (your specific role/responsibility)
A — Action     (what YOU did — not "we", not passive voice)
R — Result     (quantified impact; what you learned)
```

**Common STAR failure modes**

| Failure | Fix |
|---|---|
| Result is vague ("it went well") | Always quantify: latency %, cost $, user adoption %, bugs caught |
| Task blurs with action | Separate what you were asked to do vs. what you chose to do |
| "We" throughout | Replace every "we" with "I" unless describing team context |
| No learning in failure stories | End every negative story with what you changed afterward |

### Project deep-dive structure (45 min map)

```
0–5   min: Project overview — problem, your role, tech stack
5–15  min: Architecture walk-through — diagram, key decisions
15–25 min: Hardest technical challenge — specific, not generic
25–35 min: What failed / what you'd change — honest critique
35–45 min: Impact measurement — how you knew it worked
```

Interviewers listen for: **individual ownership**, **technical depth on specifics**, **honest failure analysis**.

## Worked example

### Behavioral mock prompt: "Tell me about a time you disagreed with a technical decision."

**Model STAR answer**

**Situation**: On an ML platform team, the tech lead proposed storing all LLM conversation history in a single Postgres JSONB column, arguing it was "simple and working."

**Task**: I was responsible for the retrieval layer and had forecast 10× growth in user sessions over the next quarter.

**Action**: I wrote a 2-page design doc benchmarking JSONB retrieval (full table scan on unindexed JSON) against a dedicated `sessions` table with GIN index. I ran load tests showing p99 query time would degrade from 40 ms to 4 s at projected scale. I presented the results in the next architecture review, not as "you're wrong" but as "here's what happens in 3 months — I'd like us to agree on a migration plan now."

**Result**: We migrated to a normalized schema before the growth event. Query p99 stayed at 35 ms through the 8× user increase. The tech lead later cited the doc format in onboarding new engineers. I learned that data-backed framing (not opinion) wins technical disagreements.

---

### Project deep-dive mock prompt: "Walk me through your multi-agent RAG platform."

**Model answer (condensed)**

**Overview (2 min)**: Built an enterprise Q&A platform over internal policy documents. Solo architecture, 3-person implementation. Stack: FastAPI + LangGraph agents + Qdrant + `claude-sonnet-5-5`.

**Architecture (5 min)**:
```
Ingestion: S3 → PDF parser → chunker → embedder → Qdrant
Query:     User → orchestrator agent → retriever tool → reranker
                                     → synthesizer node → response
Eval:      Nightly RAGAS run → dashboard alert if faithfulness < 0.85
```

**Hardest challenge**: Handling multi-hop questions (Q that requires combining 3+ documents). Initial retrieval returned top-5 per sub-query independently; context window overflow at 128K tokens. Fixed by: query decomposition sub-agent → per-hop retrieval → cross-hop deduplication → rerank unified pool to top-12.

**What failed**: Initial chunking used fixed 512-token windows. Tables were split mid-row, destroying retrieval quality. Migrated to semantic chunking (sentence-transformer similarity threshold) — faithfulness score improved from 0.71 to 0.89.

**Impact**: Replaced 2 FTE of manual policy lookup. 94% user satisfaction in 2-week pilot. Avg response time 1.8 s p50.

## Exercises

1. **STAR practice**: Write a full STAR answer (minimum 4 paragraphs) for this prompt: "Describe a time when you had to learn a new technology quickly under deadline pressure." Quantify the result.

2. **Anti-pattern fix**: Rewrite this weak answer to make it STAR-compliant and impactful:
   > "We built an agent that answered customer questions. It worked really well and everyone was happy with it. I helped with some of the coding."

3. **Project deep-dive prep**: Pick your strongest project. Write a 5-slide outline (one bullet per section: overview, architecture, hardest challenge, failure, impact). For each bullet, identify the quantified proof point you will cite.

4. **Mock drill**: Record yourself (voice memo) answering: "Tell me about a time you handled a production incident caused by your code." Play it back. Count how many times you said "we" vs "I". Score your Result: specific/quantified (3), vague positive (1), no result (0).

5. **Failure story construction**: Write a STAR for a real failure. Constraint: the Result section must include (a) the mistake acknowledged clearly, (b) the fix, (c) what process change you made, (d) evidence it worked.

(Solutions: ../solutions/week-12/day-3.md)

## Self-check
- [ ] I can deliver a STAR answer in under 3 minutes without rambling
- [ ] Every result I cite is quantified
- [ ] I have 6 prepared STAR stories covering: impact, disagreement, failure, learning, leadership, collaboration
- [ ] I can diagram my strongest project from memory in 5 min

## Supplementary
- [Jeff Bezos Leadership Principles — Amazon](https://www.amazon.jobs/content/en/our-workplace/leadership-principles)
- [Grokking the Behavioral Interview — Educative](https://www.educative.io/courses/grokking-the-behavioral-interview)
