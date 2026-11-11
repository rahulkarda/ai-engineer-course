# Phase 7 · Week 11 · Day 1 — The Project Deep-Dive Interview Structure

> Objective: Build a repeatable 5-part structure for narrating any portfolio project in a technical interview.  ·  Time: ~1 hr  ·  Prereqs: none

## Concept

The project deep-dive is a distinct interview format. It is not system design (whiteboard from scratch), not behavioral (STAR stories), and not a coding screen. The interviewer hands you a project you already built and asks you to defend it. They want to see how you think, not just what you shipped.

Three signals interviewers are extracting:

1. **Depth over breadth** — Can you go three levels deep on any decision? Surface-level "I used LangGraph because it supports graphs" fails. "I chose LangGraph over raw function chaining because I needed cycle detection and conditional branching for retry loops — and the state graph gave me a serializable checkpoint I could resume after tool failures" passes.
2. **Judgment** — Did you make conscious tradeoffs, or did you just copy a tutorial?
3. **Ownership** — Do you know what you would change, and why? Saying "it's perfect" is a red flag.

## How it works internally

Interviewers score project deep-dives on a rubric, not a vague "good vibes" sense. Most rubrics decompose into:

- **Technical accuracy** — are the things you claim architecturally true?
- **Decision quality** — were your choices reasonable given your constraints?
- **Self-awareness** — do you know the limits of what you built?
- **Communication** — can you adapt the explanation to a non-expert then go deep when pushed?

The scoring weight is skewed toward the middle of the conversation — the "why" questions — not the opening summary. Interviewers front-load easy questions ("what does it do?") as warmup, then drill. Depth is scored on the drill, not the summary.

## The 5-Part Structure

Use this structure for every project. Practice until it feels natural, not scripted.

### Part 1 — Context (90 seconds)

State the problem, the constraints, and why a standard solution was insufficient.

Template:
> "The project was [one sentence]. The core constraint was [technical or business constraint]. Off-the-shelf solutions failed because [specific gap]."

### Part 2 — Architecture (2 minutes)

Walk the data flow, not the component inventory. Say what moves where and why.

Template:
> "At a high level: [input] enters the system, [component A] does [specific transformation], hands off to [component B] which [does what], and the result is [output format]. The key boundary in the system is between [X and Y] because [reason]."

### Part 3 — Decisions (3 minutes, the core)

Pick 3 decisions that were non-obvious. For each: what you chose, what you rejected, and why.

Template:
> "Decision: [what you chose]. I considered [alternative]. I went with [choice] because [constraint-driven reason]. The tradeoff was [what you gave up]."

### Part 4 — Tradeoffs (1 minute)

Name what the current design sacrifices. Be precise, not vague.

Template:
> "The biggest limitation is [specific weakness]. It matters when [concrete scenario]. Solving it would require [rough approach]."

### Part 5 — Reflection (1 minute)

What you'd do differently with more time or better information. This is not self-flagellation — it's engineering judgment.

Template:
> "If I rebuilt this today, I'd change [one thing] because [what you learned]. I'd keep [one thing] because [it held up well]."

## STAR for Systems

Standard STAR (Situation, Task, Action, Result) maps to systems work with minor adaptation:

| STAR element | System deep-dive mapping |
|---|---|
| Situation | Project context, scale, team constraints |
| Task | The specific technical problem you owned |
| Action | Your architectural and implementation decisions |
| Result | Measurable outcomes: latency, cost, accuracy, reliability |

The key difference from behavioral STAR: the Action section expands to 3x the size, and every claim in Result needs a number or a comparison.

## Worked Example — Multi-Agent Operations Platform

**Context (Part 1):**
> "The project was a production multi-agent system where autonomous agents coordinate via tool calls to execute multi-step workflows. The constraint was reliability: LLM-based agents fail non-deterministically, so the system had to survive tool failures, hallucinated outputs, and partial execution without losing state. A single-agent + retry loop wasn't sufficient because different failure modes needed different recovery strategies."

**Architecture (Part 2):**
> "Tasks enter as structured requests, a supervisor agent decomposes them into subtasks and routes to specialist agents — a retrieval agent, a code execution agent, and an external API agent. Each agent is a LangGraph node with a typed state object. State is checkpointed to Redis after every node transition. The key system boundary is between the supervisor and the specialist agents: the supervisor manages orchestration logic; specialists have no awareness of other agents. This separation let us swap specialist implementations without touching orchestration."

**Decisions (Part 3):**
> "Decision 1: LangGraph for orchestration. I considered Celery task graphs and raw asyncio chains. LangGraph won because the conditional edge API let me express retry logic as graph branches, and the built-in checkpointer gave me resume-from-failure without custom state serialization.
>
> Decision 2: MCP for tool exposure. I could have hard-coded tool schemas per agent. MCP let external tools be registered dynamically — agents discovered capabilities at runtime rather than at build time. Tradeoff: MCP adds a network hop and a schema validation step; for latency-sensitive tools that was a penalty we accepted.
>
> Decision 3: typed Pydantic state vs. raw dict. Forced strict schemas on every node boundary. Tradeoff: more upfront schema work, but caught 80% of integration bugs at definition time rather than at runtime."

**Tradeoffs (Part 4):**
> "The biggest limitation is the supervisor becoming a bottleneck under high parallelism. Right now the supervisor is a single LangGraph node — it serializes subtask dispatch. At scale that's a throughput ceiling. Fixing it requires moving to a distributed supervisor pattern, which we deprioritized because single-tenant latency met our SLA."

**Reflection (Part 5):**
> "If I rebuilt it today, I'd design the agent state schema first and the graph second — we refactored state types three times because we didn't anticipate what the checkpointer needed. I'd keep the MCP integration; runtime tool discovery has paid off every time we added a new data source."

## Exercises

1. Write the Context and Architecture sections (Parts 1 and 2) for your Python Backend Platform capstone. Keep Context under 90 seconds when spoken aloud. Record yourself and trim.

2. For the Advanced RAG system, identify 3 non-obvious architectural decisions. For each, write one sentence stating what you rejected and why. Do not use "simpler" or "easier" as justification — use specific technical constraints.

3. A common interview trap: "Why didn't you use [technology the interviewer clearly prefers]?" Draft a 3-sentence response for the following: "Why didn't you use a traditional search engine instead of a vector database for your RAG system?"

4. Write the Reflection section (Part 5) for the Production AI Platform capstone. Identify one thing you would change and one you would keep, both with concrete technical justification. Avoid vague answers like "better monitoring" — name the specific missing piece.

(Solutions: ../solutions/week-11/day-1.md)

## Self-check

- [ ] I can explain what distinguishes a project deep-dive from a system design interview
- [ ] I can state the 5-part structure from memory and give the time allocation for each
- [ ] I can describe what interviewers score and where in the conversation depth is most evaluated
- [ ] I can walk through the Multi-agent platform worked example and adapt it to my own project

## Supplementary

- [Designing Data-Intensive Applications — Chapter 1 (Reliability, Scalability, Maintainability)](https://dataintensive.net/) — the vocabulary of tradeoffs that interviewers expect you to use fluently
- [Staff Engineer: Leadership Beyond the Management Track — Will Larson](https://staffeng.com/book) — how senior engineers narrate technical decisions; directly applicable to deep-dive interviews
- [The Pragmatic Engineer — Engineering Interviews series](https://blog.pragmaticengineer.com/engineering-interview-preparation/) — practitioner breakdown of how top-tier companies score technical interviews
