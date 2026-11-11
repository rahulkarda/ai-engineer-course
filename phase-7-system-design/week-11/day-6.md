# Phase 7 · Week 11 · Day 6 — Build Session: Project Deep-Dive Packet (4-hour sprint)

> Objective: Produce a complete, interview-ready deep-dive packet for one capstone project  ·  Time: ~4 hr  ·  Prereqs: [[day-1]], [[day-2]], [[day-3]], [[day-4]], [[day-5]]

---

## Concept

A deep-dive packet is a structured artifact that makes your project interview-ready. It forces you to think through the project at the depth an interviewer will probe. It also serves as a reference you can review the night before any interview.

This session is a directed build. You are not writing new code. You are producing documentation that proves you understand what you built.

---

## The Deep-Dive Packet Components

A complete packet contains 7 components:

| Component | Format | Purpose |
|---|---|---|
| Architecture diagram | Mermaid or ASCII | Visual anchor for the conversation |
| Decision log | Table (5+ entries) | Shows deliberate choices |
| Tradeoff analysis | Table | Shows engineering judgment |
| Scaling story | Narrative + numbers | Shows production thinking |
| 10 Q&As | Q + model answer | Interview rehearsal |
| 2-minute pitch | Script | Verbal delivery practice |
| (Bonus) Retrospective note | 1 paragraph | Self-awareness signal |

---

## Choosing Which Capstone to Deep-Dive

Deep-dive the project that satisfies the most of these criteria:

1. **You understand it most deeply.** The project where you made the most decisions, hit the most real problems, and can speak to specific line-level choices.
2. **It matches the role you are targeting.** RAG system → AI/ML Engineer. Multi-agent ops → FDE or AI Platform. Production platform → MLOps or Platform Engineer. Backend → Backend-adjacent AI roles.
3. **It has the most interesting tradeoffs.** The project where there was no obviously correct answer — where you weighed options and chose.
4. **You have numbers for it.** Performance metrics, latency figures, dataset sizes, cost estimates. Interviews that end with "I'm not sure exactly" on every number leave a weak impression.

If two projects tie, choose the more complex one. Complexity generates more interview surface area.

---

## How It Works Internally

### Architecture Diagram Conventions for AI Systems

An AI system diagram must show:

- **Data sources** — where inputs originate (user, database, file store, external API)
- **Processing components** — models, retrievers, agents, tools, chains
- **External services** — LLM APIs, vector databases, message queues, storage
- **Data flows** — directional arrows with labels (query, embedding, chunk, response)
- **Boundaries** — what is internal (your service), what is external (third-party)

Mermaid flowchart skeleton for an AI system:

```
graph LR
    User --> API[FastAPI Gateway]
    API --> Orchestrator[Agent Orchestrator]
    Orchestrator --> LLM[LLM API]
    Orchestrator --> Retriever[Vector Retriever]
    Retriever --> VDB[(Vector DB)]
    Orchestrator --> Tools[Tool Executor]
    Tools --> ExtAPI[External APIs]
    API --> DB[(Postgres)]
```

Label every arrow. If an arrow has no label, you do not fully understand the data flow.

### Decision Log Format

| Decision | Context | Options Considered | Chosen | Rationale | Outcome |
|---|---|---|---|---|---|
| Short name | What situation forced this decision | Option A, Option B, Option C | Option chosen | Why this option over others | What actually happened — did the decision hold? |

Minimum 5 decisions. Strong logs have 7-10. Include decisions that did not work out — those are the most valuable to discuss in interviews.

Example entry:

| Decision | Context | Options Considered | Chosen | Rationale | Outcome |
|---|---|---|---|---|---|
| Chunking strategy for RAG | Long documents caused retrieval misses with fixed-size chunks | Fixed-size (512 tokens), semantic splitting, recursive character split | Recursive character split with 20% overlap | Preserved sentence boundaries; overlap reduced context loss at chunk edges | Retrieval precision improved; overlap increased index size by ~18% |

### Tradeoff Analysis Table

Separate from the decision log. This table summarizes the key architectural tradeoffs at the system level.

| Dimension | Chosen Approach | Alternative | Why Chosen | What You Gave Up |
|---|---|---|---|---|
| Consistency model | Eventual (async writes) | Strong (sync writes) | Latency requirement | Visibility lag in reads |
| Embedding model | Hosted API | Self-hosted | Development speed | Per-call cost at scale |
| Agent state | In-memory (LangGraph) | Persistent (Redis/DB) | Prototype speed | Recovery on crash |

### Scaling Story Template

Structure: prototype → first production deployment → current/final state.

**Prototype**
- What it handled: [N users / N requests per minute / N documents]
- Infrastructure: [local / single container / minimal cloud]
- What broke first: [the specific failure mode that forced the next step]

**First production deployment**
- What changed: [the specific architectural change made to address the prototype failure]
- Infrastructure: [Docker / k8s / managed service]
- New numbers: [throughput, latency, error rate]
- What broke next: [next bottleneck discovered]

**Final state**
- Architecture: [key components]
- Numbers: [p50/p95 latency, throughput, cost per request if known]
- Remaining known limits: [what would break at 10x current load]

If your project never reached true production, write the scaling story as "prototype → projected production": what architectural changes would be required, and at what load threshold would each change become necessary.

---

## Step-by-Step Build Plan (4-Hour Session)

### Hour 1: Architecture Diagram + Decision Log

**0:00 – 0:20 — Draft architecture diagram**
Open a blank document. Draw the components from memory first — no looking at code. Components you cannot place from memory are components you do not understand well enough. Then verify against the actual code and correct.

Self-check:
- [ ] Every external service (LLM API, vector DB, message broker, cloud storage) is on the diagram
- [ ] Every data flow arrow is labeled

**0:20 – 1:00 — Write the decision log**
Start with the 3 hardest decisions you remember making. For each, work backwards: what was the context, what did you consider, what did you choose, why, and what happened. Then add 2-3 smaller decisions to reach at least 5 entries.

Self-check:
- [ ] At least one decision that did not work out as expected is included
- [ ] Every decision has all 6 fields populated

---

### Hour 2: Tradeoff Analysis + Scaling Story

**1:00 – 1:30 — Tradeoff analysis table**
Identify 4-6 architectural dimensions where you made a choice between two real alternatives. Fill the table. Be specific: "faster development" is weak; "reduced initial setup from ~2 days to ~2 hours by avoiding self-hosted infrastructure" is strong.

Self-check:
- [ ] "What you gave up" column is honest — not "nothing significant"
- [ ] At least one row acknowledges a tradeoff you would re-evaluate

**1:30 – 2:00 — Scaling story**
Write the three-section scaling narrative: prototype, first production, final state (or projected). Include specific numbers — even estimates with stated assumptions are better than vague descriptions.

Self-check:
- [ ] Prototype failure mode is identified with specificity ("the SQLite connection pool exhausted at ~50 concurrent requests, causing 503s")
- [ ] Final state has at least 2 concrete performance numbers

---

### Hour 3: 10 Q&As

**2:00 – 3:00 — Write 10 anticipated deep-dive Q&As**
Anticipate the questions a strong interviewer will ask about your project. Write model answers. Each answer should be 3-6 sentences.

Question categories to cover (pick 10 across these):
- "Why did you choose [technology X]?"
- "How does [core component] work?"
- "What would break first under 10x load?"
- "If you were starting over, what would you change?"
- "How did you test this?"
- "How did you handle [failure mode]?"
- "What's the cost model at production scale?"
- "How would you add [feature X] without breaking existing behavior?"
- "What monitoring do you have in place?"
- "How long would it take to onboard a new engineer to this codebase?"

Self-check:
- [ ] At least one Q&A covers a failure or a mistake
- [ ] At least one Q&A covers cost or resource consumption

---

### Hour 4: 2-Minute Pitch + Review

**3:00 – 3:30 — Write and rehearse the 2-minute pitch**
The 2-minute pitch is a verbal script, not a slide. It answers: what does this system do, why does it matter, what was hard, what did you learn?

Pitch structure:
1. **Hook** (15 sec): What problem does this solve? State it as the user's problem, not as a technical challenge.
2. **System overview** (30 sec): What did you build? Name the major components and how data flows through them.
3. **Key challenge** (30 sec): What was the hardest technical problem? What did you try? What worked?
4. **Result** (20 sec): What does the system do now? Include a number.
5. **Reflection** (25 sec): What would you do differently? What did you learn that you did not expect?

Write it out in full. Then read it aloud and time it. Edit until it lands in 110-130 seconds.

Self-check:
- [ ] Pitch opens with the user problem, not the technology
- [ ] Pitch includes one specific number in the result section

**3:30 – 4:00 — Review and gap-fill**
Read through the complete packet. Identify:
- Sections with vague language ("it was fast", "it scaled well") → replace with numbers or explicit acknowledgment that you don't have the number
- Decisions listed without a "why" → fill in
- Q&As that dodge the hard part → rewrite to be direct

Self-check:
- [ ] Every answer in the Q&A section, if read aloud, sounds like something you would say in an interview — not like documentation
- [ ] The packet would let someone else (who didn't build it) understand the key design choices in 15 minutes

---

## Exercises

This session is the build. The exercises are embedded in the build phases above as self-check items. Complete each self-check before moving to the next phase.

At session end, complete these two synthesis tasks:

1. Read your 2-minute pitch aloud to yourself (or record it). Identify the one sentence that sounds least natural. Rewrite it until it does.

2. Look at your decision log. Pick the decision you are least confident about — the one you would most want to revisit. Write one paragraph explaining why you made it anyway and what evidence would have changed your choice.

(Solutions: ../solutions/week-11/day-6.md)

---

## Self-check (session complete)

- [ ] Architecture diagram exists and covers all major components and data flows
- [ ] Decision log has 5+ entries with all 6 fields populated
- [ ] Tradeoff analysis table has 4+ rows with honest "what you gave up" entries
- [ ] Scaling story includes specific numbers for prototype and final/projected state
- [ ] 10 Q&As written with model answers that sound like spoken answers, not documentation
- [ ] 2-minute pitch written, timed between 110-130 seconds, and revised at least once
