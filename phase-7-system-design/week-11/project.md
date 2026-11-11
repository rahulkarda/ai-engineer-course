# Week 11 Project — Project Deep-Dive Packet

> Pass gate: 75/100  ·  Estimated time: 6-8 hours total (build session in Day 6 + polish)

---

## Project Brief

Produce a complete, interview-ready deep-dive packet for one of your four capstone projects. The packet must demonstrate that you understand what you built at the depth required to discuss it in a 45-minute technical interview with a senior engineer or a hiring manager at a company that builds AI systems.

This is a documentation and narrative artifact, not a code deliverable. However, it must be grounded in the actual code and architecture of the project — vague or generic descriptions that could apply to any project will not pass.

**Choose your capstone:**
- Python backend platform (FastAPI, SQL, testing, async)
- Advanced RAG system (LLMs, embeddings, retrieval, vector DB, evaluation)
- Multi-agent operations platform (agent architecture, MCP, A2A, LangGraph, security)
- Production AI platform (distributed execution, observability, Docker, k8s, CI/CD, cost)

Selection guidance: choose the project you understand most deeply and have the most specific things to say about. If you are targeting a specific role (AI/ML Engineer, FDE, MLOps), choose the project that best maps to that role's core concerns. See Day 6 for full selection criteria.

---

## Deliverable Components

The complete packet contains 8 components. All are required. Minimum specifications are listed per component.

---

### Component 1: Architecture Diagram

**Format:** Mermaid flowchart or clean ASCII diagram. No third-party diagram tools required.

**Minimum requirements:**
- All major system components labeled
- All external services identified (LLM APIs, vector databases, cloud storage, message queues)
- All data flows shown as directed arrows with labels
- System boundaries drawn (internal vs. external)
- At least one "unhappy path" or failure boundary shown (e.g., retry logic, fallback path, error state)

**Format specification:**
If using Mermaid, embed in a fenced code block with `mermaid` language tag. If using ASCII, align columns consistently and use `-->` for data flow, `---` for containment boundaries.

**Scoring weight:** 10 points

---

### Component 2: Decision Log

**Format:** Markdown table with 6 columns

| Decision | Context | Options Considered | Chosen | Rationale | Outcome |
|---|---|---|---|---|---|

**Minimum requirements:**
- 5 decisions minimum; 7+ for full credit
- Every row has all 6 columns populated
- At least one decision that did not work out as expected (outcome column shows this)
- At least one decision involving an explicit cost or performance tradeoff
- Decisions must be specific to your project — not generic best practices restated as decisions

**What counts as a decision:** any point where you chose between two or more real alternatives that would have produced meaningfully different systems. Technology choice, architecture pattern, data model design, API design, deployment approach, chunking strategy, agent topology, evaluation approach — all count.

**What does not count:** "I decided to use Python" (no real alternative), "I decided to write tests" (not a design decision).

**Scoring weight:** 15 points

---

### Component 3: Tradeoff Analysis Table

**Format:** Markdown table with 5 columns

| Dimension | Chosen Approach | Alternative | Why Chosen | What You Gave Up |
|---|---|---|---|---|

**Minimum requirements:**
- 4 rows minimum; 6+ for full credit
- Covers at least 2 of these dimension categories: consistency model, data storage, compute/cost, developer experience, operational complexity, latency vs. throughput, build vs. buy
- "What You Gave Up" column must be honest — "nothing significant" is not accepted as a legitimate entry for any real architectural tradeoff

**Scoring weight:** 10 points

---

### Component 4: Scaling Story

**Format:** Three narrative sections (prototype, first production deployment or projected, final state or projected final state)

**Minimum requirements:**
- Prototype section: what it handled (users/requests/data), infrastructure, and the specific failure mode that limited it
- Production section: what changed architecturally, the new infrastructure, and the new performance envelope
- Final section: current or projected steady-state, at least 2 concrete numbers (latency, throughput, cost per request, dataset size), and known limits ("this would break at X")
- If the project was never deployed to production: write a "projected production" section that makes explicit what architectural changes would be required and at what load thresholds

**What counts as a number:** specific measurements (p95 latency = 340ms), specific dataset sizes (50,000 documents), cost estimates with stated assumptions ($0.002 per query at 1,000 queries/day), error rates, throughput figures.

**Scoring weight:** 15 points

---

### Component 5: 10 Deep-Dive Q&As with Model Answers

**Format:** Numbered list. Each entry: Q: [question] / A: [answer]

**Minimum requirements:**
- Exactly 10 Q&As
- Questions must be specific to your project — not generic ("what is RAG?") but project-grounded ("why did you use recursive character splitting instead of semantic splitting in your RAG system?")
- Answers: 3-6 sentences each
- Cover all of these categories (distribute 10 questions across them):
  - Technology choice justification (why X over Y)
  - Internal workings (how does [component] actually work)
  - Failure handling (what happens when [component] fails)
  - Scaling and performance (what would break at 10x)
  - Testing and quality (how did you verify correctness)
  - Cost model (what does this cost to run)
  - Retrospective (what would you change)
  - Onboarding (how long to get a new engineer productive)
- At least one Q&A must address a mistake or a failure
- Answers must sound like spoken interview answers, not documentation

**Scoring weight:** 20 points

---

### Component 6: 2-Minute Pitch Script

**Format:** Written prose script, intended to be read aloud in 110-130 seconds

**Structure (required):**
1. Hook — state the user problem (not the technical challenge), 10-15 seconds
2. System overview — major components and data flow, 25-35 seconds
3. Key challenge — the hardest technical problem, what you tried, what worked, 25-35 seconds
4. Result — what the system does now, one specific number, 15-20 seconds
5. Reflection — what you would do differently and what you learned, 20-25 seconds

**Minimum requirements:**
- Follows the 5-section structure above
- Opens with the user problem, not a technology name
- Includes at least one specific number in the Result section
- Is written as spoken language (contractions OK, colloquial phrasing OK) — not as technical documentation

**Scoring weight:** 10 points

---

### Component 7: Updated Resume Bullets for This Project

**Format:** Resume-ready text block, as it would appear in a Projects section

**Required elements:**
- Project name (as it appears on GitHub)
- One-line description of what the project does
- Technologies list (4-8 items)
- 3-4 impact bullets

**Bullet requirements:**
- Each bullet starts with a past-tense action verb
- At least 2 of 3 bullets contain a number
- No "responsible for", "helped with", or "worked on" constructions
- Each bullet can stand alone — a reader who sees only that bullet understands what you did

**Scoring weight:** 10 points

---

### Component 8: GitHub README Draft

**Format:** Markdown document suitable for use as the repository README

**Minimum sections (all required):**
- Project name and one-line description (header)
- Architecture overview — 1-2 paragraphs or a diagram
- Key technical decisions — 3-5 bullet points summarizing the most important choices
- Setup and usage instructions — specific enough that a developer can clone and run the project
- What's interesting about this project — 1 paragraph explaining what makes this worth reading

**Minimum requirements:**
- Written for someone who has never seen the code
- Setup instructions include environment variable setup (reference to .env.example)
- Does not use the phrase "this project" more than twice
- Has at least one diagram or screenshot (Mermaid diagram counts)

**Scoring weight:** 10 points

---

## Evaluation Criteria

| Component | Points |
|---|---|
| Architecture diagram (coverage, labels, data flows, boundaries) | 10 |
| Decision log (5+ entries, all fields, honesty of outcome column) | 15 |
| Tradeoff analysis (4+ rows, honest "what you gave up") | 10 |
| Scaling story (prototype → production, 2+ concrete numbers) | 15 |
| 10 deep-dive Q&As (specificity, coverage, spoken-language quality) | 20 |
| 2-minute pitch script (structure, number, spoken register) | 10 |
| Resume bullets (verbs, numbers, no passive constructions) | 10 |
| GitHub README draft (completeness, audience awareness, setup instructions) | 10 |
| **Total** | **100** |

**Pass gate: 75/100**

Partial credit applies within each component. A component that is present but generic (could apply to any project) scores 40-60% of its allocated points. A component that is specific, honest, and grounded in the actual project scores 80-100%.

---

## Submission Instructions

Submit as a single Markdown file named `deep-dive-packet-[project-name].md` (e.g., `deep-dive-packet-rag-system.md`). Place it in:

```
phase-7-system-design/week-11/submissions/
```

All 8 components must be present in the file. Use H2 headers (`##`) for each component, matching the names used in this brief.

**Before submitting, verify:**
- [ ] All 8 components are present and labeled
- [ ] Decision log has 5+ rows with all 6 columns
- [ ] Scaling story has at least 2 concrete numbers
- [ ] Q&A section has exactly 10 entries
- [ ] Pitch script is between 100-150 words (approximates 110-130 seconds spoken)
- [ ] Resume bullets section has no bullets starting with "Responsible for", "Helped", or "Worked on"
- [ ] README draft has a setup section with environment variable instructions

---

## Reference

Model solutions: ../solutions/week-11/project.md
