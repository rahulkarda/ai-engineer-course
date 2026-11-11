# Week 11 Assessment — Answer Key

---

## Part 1: Conceptual Questions (Model Answers)

**Q1: What is the fundamental difference between a project deep-dive interview and a system design interview?**

System design: candidate solves a hypothetical problem in real time. No prior context. Interviewer evaluates structured thinking under pressure.

Project deep-dive: candidate explains something they already built. Interviewer has full information disadvantage — you know the outcome, they don't. Evaluation shifts from "can you think through a problem" to "do you understand your own decisions and their consequences." The hard part is not describing what you built — it's articulating why you made each choice and what it cost you.

---

**Q2: List the 5 parts of the deep-dive interview structure in order.**

1. Context and problem statement (why this project existed, what constraints were real)
2. Architecture decision (the key design choice, with alternatives and rationale)
3. The hard part (the unexpected problem, how you diagnosed it, how you solved it)
4. Results (quantified outcomes against the original goal)
5. Reflection (what you'd do differently, and why)

---

**Q3: What is a decision log and why is it valuable in an interview?**

A decision log is a written record of significant architectural or design choices: what the options were, what was chosen, why, and what was sacrificed. It is valuable in an interview because: (a) it forces you to have thought through the reasoning before you're in the room, (b) it gives you a ready source of "the hard part" stories, and (c) it demonstrates engineering maturity — candidates who document their decisions are operating at a higher level than those who just shipped code.

---

**Q4: What does the tradeoff triangle represent, and how do you use it in a deep-dive?**

The tradeoff triangle represents the three dimensions that can't all be simultaneously maximized in most AI systems: latency, cost, and quality. Using it in a deep-dive means: (a) naming explicitly where your system sits on each axis, (b) explaining which dimension you optimized for and which you sacrificed, (c) showing that the constraint was deliberate and justified — not the result of not knowing better.

---

**Q5: What are 3 signals that an interviewer wants deeper technical detail?**

1. They echo a word from your answer as a question ("vector database?") — this is an invitation, not confusion
2. They ask "how did you decide" rather than "what did you use" — signals they want decision reasoning, not the name
3. They go quiet and let you keep talking — in a deep-dive, silence means "keep going," not "move on"

---

**Q6: What makes a resume bullet "strong" for an AI/ML Engineer role?**

A strong bullet has: (a) a specific active verb (engineered, designed, implemented — not "worked on"), (b) a named mechanism (not just "implemented caching" but "implemented query-result Redis cache with TTL-based invalidation"), and (c) a quantified outcome or verifiable quality signal (MRR@5 of 0.71, 60% cost reduction, 94% test coverage). Bullets that lack numbers and mechanisms are indistinguishable from tutorials; bullets with both are evidence of actual production work.

---

**Q7: What is the FDE Clarify→Constrain→Prototype→Communicate framework, and when do you use it?**

It is a structured approach to vague, open-ended customer requests common in Forward-Deployed Engineer roles.
- Clarify: ask questions to understand the actual problem before proposing a solution
- Constrain: explicitly scope the first deliverable — what is in, what is out, what success looks like
- Prototype: build something fast and falsifiable that answers the core question
- Communicate: translate the prototype's findings into business terms for non-technical stakeholders

Use it when: a customer gives you a vague request ("we want AI for X"), you're asked to scope a new project, or you're in an FDE-style interview where the interviewer gives you an underspecified scenario.

---

**Q8: What role does portfolio ordering play in GitHub's pinned repos, and what's the right ordering principle?**

Pinned repos are the first thing a recruiter or hiring manager sees. They function as your interview pitch — first impression, first conversation topic. Ordering principle: most differentiated and technically rich project first, not most recently built or alphabetically. The multi-agent platform leads because it's rare (most candidates don't have multi-agent + security + production tooling) and generates the most interesting technical conversation. Least differentiated (Python backend) goes last among pinned — it supports the story, doesn't lead it.

---

## Part 2: Complete Deep-Dive Narrative — Multi-Agent Operations Platform

**Architecture Walkthrough:**

"The platform has four components: a LangGraph orchestration layer that defines agent workflows as typed state graphs, an MCP (Model Context Protocol) server that acts as the centralized tool gateway, a FastAPI REST API as the external entry point, and a Redis-backed task queue for async workflow execution.

When a request arrives at the API, it's enqueued and routed to the LangGraph executor. The executor spawns agents per the graph definition — typically a researcher, planner, and specialist (coder or writer depending on task type). Agents read from and write to shared typed state (Pydantic models). Tool calls are proxied through the MCP server, which validates per-agent permissions, enforces rate limits, and logs every call. Results flow back through state to the next graph node until the terminal state is reached or a human-review checkpoint is hit."

**3 Key Decisions with Tradeoffs:**

Decision 1 — MCP gateway for all tool calls:
"We centralized all tool execution through a single MCP server rather than letting agents call tools directly. Trade-off: ~20ms latency overhead per tool call, plus a new SPOF. Benefit: single audit surface, centralized credential management, consistent rate limiting. In a security-sensitive context, the trade-off is clearly worth it. The 20ms overhead is invisible in workflows where tool calls take 200ms–30s."

Decision 2 — LangGraph over custom orchestration:
"LangGraph's explicit graph structure and typed state were worth the learning curve. We considered custom orchestration (a for-loop over agent calls), which would have been faster to prototype but impossible to instrument cleanly. LangGraph's checkpointing also gave us fault tolerance — resume from failure rather than restart from scratch — which was critical for long-running workflows."

Decision 3 — Optimistic locking for state management:
"Distributed locks (Redis SETNX) were our first instinct for concurrent state safety. We moved to optimistic locking after testing showed conflict rate < 2% — at that rate, the deadlock risk of distributed locks wasn't worth it. Optimistic locking adds version tokens to reads and validates them on writes; stale writes fail fast and trigger re-planning. Simpler, no deadlock risk, handles the actual access pattern."

**Scaling Story:**

Prototype: 2 agents, single script, 1 workflow at a time, no production tooling
Alpha: 3 agents + LangGraph, basic MCP gateway, 5 concurrent workflows, no auth
Beta: 5 agent types + full MCP gateway with auth, 20 concurrent workflows, Dockerized, FastAPI
Production: 50 concurrent workflows, p95 < 800ms orchestration overhead, k8s deployment, full observability, 99.5% SLA

**"What I'd Do Differently" Reflection:**

"The platform works. The architecture decisions held up. Two things I'd change with what I know now:

First: OpenTelemetry from day one, not week 8. We couldn't understand why a 10-step workflow was slow until we had traces. Retrofitting took 2 weeks and 3 refactors. Tracing is not an add-on for agent systems — it's the diagnostic foundation.

Second: typed state (Pydantic) from the first prototype. We started with untyped dicts in the alpha and migrated when bugs piled up. Migration was painful. Typed state in agent systems isn't premature optimization; it's the only way to catch type mismatches before they silently corrupt downstream outputs."

**5 Q&As:**

Q: How do you prevent agents from accessing tools they shouldn't use?
A: MCP gateway validates per-agent scope on every tool call. Agents register with a scope (list of allowed tool namespaces). Calls outside scope return 403. Scope is set in agent configuration, not by the agent itself.

Q: What happens when an agent fails midway through a workflow?
A: LangGraph checkpointer saves state after each node. Transient failures retry with backoff. Persistent failures route to error-handling node for simplified retry. If that fails, human-review queue. 80%+ of failures in testing were transient API errors that resolved on first retry.

Q: How do you validate agent outputs are correct?
A: Three layers: Pydantic schema enforcement at each state transition, consistency checks by downstream agents, optional human-in-the-loop checkpoints at configurable steps. No general "is this correct" evaluator — task-specific validators (run the code, check the format) where applicable.

Q: Why not use CrewAI which has a simpler API?
A: CrewAI abstracts agent internals in ways that made debugging hard. When a workflow failed, we couldn't see the state at each step. LangGraph's explicit graph structure made state at every node inspectable and instrumentable. For production systems, debuggability is worth the steeper learning curve.

Q: How did you test concurrent state correctness?
A: Load test with 50 concurrent workflows deliberately targeting the same shared state sections. Instrumented version token conflict rate. Ran 1,000 workflow iterations, manually audited 100 samples for state correctness. Zero silent corruptions post-optimistic-locking implementation. Conflict rate 1.8%, all resolved via re-planning.

---

## Part 3: Model Rewrites for Resume/Narrative Tasks

**Weak bullet → strong bullet:**

Before: "Worked on the RAG system and improved retrieval quality"
After: "Improved retrieval MRR@5 from 0.63 to 0.71 by adding cross-encoder re-ranking stage; validated improvement on 200-query held-out eval set."

Before: "Used Docker and Kubernetes to deploy the AI platform"
After: "Deployed distributed inference service on Kubernetes with HPA scaling (3–8 replicas under load); configured Prometheus/Grafana observability stack capturing per-request latency, cost, and model usage."

**Weak "tell me about yourself" → strong:**

Before: "I've been learning AI engineering and built some projects including RAG systems and agent platforms."

After: "I've spent the last six months building production AI systems across four projects — a RAG pipeline with measurable retrieval metrics, a multi-agent orchestration platform with production security tooling, a distributed inference service, and a FastAPI backend with 94% test coverage. I focus on the gap between 'model works in a notebook' and 'system behaves reliably at scale' — that's where I think the interesting engineering is."

---

## Part 4: Reflection Guidance (What Good Answers Look Like)

**What to look for in a strong reflection:**

1. Specificity: "I'd add tracing earlier" is acceptable. "I'd instrument OpenTelemetry on the MCP gateway and each agent node from day one because retrofitting required modifying every agent's execution wrapper" is excellent. Specificity demonstrates lived experience, not generic advice.

2. Honesty without self-undermining: The best reflections acknowledge real limitations without suggesting the system is broken. Frame changes as "better sequencing" or "what I learned," not "the architecture was wrong."

3. Learning curve vs. design error: Distinguish between "I didn't know this yet" (normal, acceptable) and "I knew but chose wrong" (must own explicitly). Most reflections should be the first kind.

4. Absence of "I'd rewrite everything": Reflections that say "I'd start over with a different framework" raise red flags — they imply the shipped system was a mistake, not a learning step.

---

## Scoring Guide

**Part 1 — Conceptual Questions (25 points, ~3 pts each)**

Full credit: Answer addresses the specific question, gives a concrete example or mechanism, and doesn't just restate the question. Answers of 3-5 sentences with at least one specific example earn full credit.

Partial credit: Correct but generic ("decision logs help you remember your choices" without explaining the interview value). Deduct 1 point.

No credit: Definition only, no application. "A decision log is a record of decisions" earns 0.

**Part 2 — Deep-Dive Narrative (40 points)**

Architecture walkthrough (10 pts): Names all major components, explains how data flows between them, identifies at least one design choice in the description (not just component names). Deduct 3 pts if no data flow explanation. Deduct 5 pts if architecture is described as a list of technologies with no explanation of how they interact.

3 key decisions (15 pts, 5 pts each): Each decision must have: what was decided, what the alternative was, why this choice over the alternative, and what was sacrificed. Deduct 2 pts if no alternative mentioned. Deduct 3 pts if no trade-off articulated.

Scaling story (5 pts): Must include concrete numbers at prototype and production. "I scaled it up" with no numbers = 0. Prototype + 2 intermediate stages + production numbers = full credit.

Reflection (5 pts): Specific change, specific reason, no self-undermining. Generic "I'd do it better" = 0.

5 Q&As (5 pts): Each Q&A must have a direct answer with at least one specific technical detail. "I'd use locks" = 1/2 credit. "I'd use optimistic locking because conflict rate was 2% and deadlock risk outweighed the overhead" = full credit.

**Part 3 — Resume/Narrative Rewrites (20 points)**

Full credit per bullet (4 pts): Active verb + specific mechanism + quantified outcome. All three present = 4 pts. Missing quantified outcome = 2 pts. Missing mechanism = 1 pt.

**Part 4 — Reflection (15 points)**

Scored on specificity and honesty. Full credit: names specific change, explains why with a concrete example from the project, acknowledges what was sacrificed in the original choice without undermining the system's validity. Generic reflections score 50%. Reflections that only say positive things score 30%.
