# Week 11 Day 5 Solutions — FDE Profile + Deep-Dive Talking Points

## Exercise 1: Clarify→Constrain→Prototype→Communicate for "AI for Support Tickets"

**Prompt:** Apply the FDE framework (Clarify → Constrain → Prototype → Communicate) to the scenario: "Customer wants AI for their support tickets."

**Model answer:**

**Scenario:** Customer (mid-size SaaS company, ~500 employees) says in a discovery call: "We want AI to help with our support tickets."

---

**Stage 1 — CLARIFY (5–10 questions, first 20 min of engagement)**

Don't build anything yet. Understand the actual problem:

- "What's the pain right now — tickets taking too long to resolve, agents spending time on repetitive questions, or something else?"
- "What does your current ticket flow look like? Human triaged, or auto-routed?"
- "What's the ticket volume — per day, per week?"
- "What does a typical ticket look like? Short questions, long technical descriptions, both?"
- "Who reads the AI output — the customer directly, or your support agent first?"
- "Do you have historical ticket data with resolutions? How much?"
- "What integrations exist? Zendesk, Intercom, custom system?"
- "What's 'success' — faster resolution time, fewer escalations, lower headcount, customer satisfaction score?"

**What you're learning:** Is this a deflection problem (reduce ticket volume), an efficiency problem (help agents answer faster), or a routing problem (get tickets to the right team)? The answer changes the architecture completely.

---

**Stage 2 — CONSTRAIN (scope the first deliverable)**

After clarifying: "Based on what you've told me, you're getting 200 tickets/day, 60% are the same 15 questions, and agents spend 40% of their time writing the same responses. Here's what I'd scope for a 4-week pilot:

- In scope: RAG-based answer suggestion to agents (not customer-facing yet) for the top 20 ticket categories
- Out of scope: full auto-reply, ticket routing, escalation logic
- Success metric: reduce average handle time for in-scope tickets by 30%
- Data needed: 6 months of resolved tickets with agent responses

Not in scope yet: customer-facing chatbot. That requires safety testing and edge case handling we'd do in Phase 2 once the agent-assist is validated."

**Why constrain this way:** Agent-assist is lower risk than customer-facing — wrong answer goes to a human who can catch it. It's also faster to validate (just measure AHT). Customer-facing chatbot in Week 1 is how pilots fail.

---

**Stage 3 — PROTOTYPE (build something falsifiable in the time available)**

Week 1 prototype:
- Ingest 3 months of resolved tickets into Qdrant
- Simple RAG retriever: ticket query → top-3 historical answers → claude-sonnet-5-5 summarizes and suggests response
- Wrapped in a Slack slash command (`/suggest [ticket text]`) — no UI work, immediate usability for agents
- Eval: 50 test tickets, human judge rates suggestion quality on 1-5 scale

What the prototype proves or disproves:
- Is historical ticket data dense enough for useful retrieval?
- Do agents actually use the suggestions (usage = proxy for quality)?
- What's the failure mode — hallucinations, irrelevant retrievals, formatting issues?

---

**Stage 4 — COMMUNICATE (translate to non-technical stakeholders)**

"We built a tool that reads your historical resolved tickets and suggests answers to agents when new similar tickets come in. Think of it as a search engine over your own team's expertise, but instead of returning links, it drafts the reply.

Week 1 results: agents rated 76% of suggestions as 'useful or better.' Average handle time for the 20 ticket categories dropped from 8 minutes to 5.5 minutes — 31% reduction.

Risks to flag: the system is as good as the historical data. Tickets where the right answer has changed (e.g., new product features) may surface stale information. We need a feedback loop where agents flag outdated suggestions.

Next step: expand to 40 ticket categories, add the feedback loop, then evaluate whether customer-facing auto-response is warranted."

---

## Exercise 2: 5-Bullet Talking Points for the Multi-Agent Platform

**Prompt:** Write 5-bullet talking points for the multi-agent operations platform using the 5-bullet rule: what / why hard / key decision / result / would do differently.

**Model answer:**

**Bullet 1 — What:**
"Built a production platform for multi-agent LLM workflows: multiple specialized agents (researcher, planner, coder) collaborating on complex tasks via LangGraph orchestration, with a Model Context Protocol server as the centralized tool gateway."

**Bullet 2 — Why hard:**
"Two problems made this hard that don't exist in single-agent systems: first, concurrent state modification — two agents reading and writing shared workflow state simultaneously produced silent corruptions. Second, credential isolation — each agent shouldn't be able to access tools or data outside its scope, but in most frameworks, all agents share the same tool registry."

**Bullet 3 — Key decision:**
"Centralized MCP server for all tool calls, with per-agent credential scoping. Every tool invocation routes through the gateway, which enforces permissions, rate limits, and audit logging. Trade-off: ~20ms added latency per tool call. Worth it: reduced the security audit surface from N agents to 1 gateway, and made credential rotation trivial."

**Bullet 4 — Result:**
"Platform handles 50 concurrent agent workflows at p95 under 800ms. Zero credential leaks in security testing. Audit log captures every tool call with agent identity, input, and output — took a 2-hour security review from 2 days to 2 hours."

**Bullet 5 — Would do differently:**
"Instrument distributed traces from day one. We retrofitted OpenTelemetry after the core was built — it worked, but cost two weeks of refactoring we hadn't planned. Tracing is not an observability add-on for agent systems; it's the only way to understand why a 10-step workflow produced the wrong result."

---

## Exercise 3: Explain LangGraph to a Non-Technical Executive

**Prompt:** Rewrite a technical explanation of LangGraph for a non-technical executive.

**Technical version (before):**
"LangGraph is a framework for building stateful, multi-actor LLM applications. It models agent workflows as directed acyclic graphs with typed state transitions, supporting cycles for iterative reasoning and conditional edges for dynamic routing between nodes."

**Non-technical version (after):**
"Think of LangGraph as a flowchart engine for AI tasks. You define the steps an AI should take — 'first research, then plan, then write code, then test' — and the rules for what happens at each step: when to move forward, when to loop back and try again, and when to stop. Unlike a simple script that runs top to bottom, LangGraph tracks where you are in the process even if the AI needs to pause, ask a question, or redo a step. That makes it possible to build AI workflows that behave predictably — not AI that just improvises until it either succeeds or gets lost."

**What changed:**
- Replaced "stateful, multi-actor" with "tracks where you are in the process"
- Replaced "directed acyclic graph" with "flowchart"
- Replaced "conditional edges" with "rules for what happens at each step"
- Added a contrast with "simple script" — gives non-technical listeners a reference point
- Ended with the business value ("behave predictably") not the technical property

---

## Exercise 4: Which Portfolio Project Best Positions for Which Role

**Prompt:** Identify which of the 4 portfolio projects best positions a candidate for: (a) FDE role, (b) MLOps role, (c) AI Platform role.

**Model answer:**

**FDE (Forward-Deployed Engineer) role:**
**Best project: Multi-Agent Operations Platform**

Why: FDE roles evaluate customer-facing problem-solving + deep technical ability. The multi-agent platform demonstrates systems thinking (architecture under real constraints), ability to scope and sequence (MCP gateway before full agent system), and security awareness (credential isolation, audit logging). These map to FDE core competencies: scope ambiguous problems, build under constraints, make security-conscious architectural choices, explain complex systems to customers.

Supporting evidence to mention: how you'd adapt the platform's tool-gateway pattern to a customer with N integration points, what the onboarding story looks like, how you'd tune agent behavior per-customer.

**MLOps role:**
**Best project: Production AI Platform**

Why: MLOps roles care about reliability, observability, deployment automation, and cost control — not just "does the model work." The Production AI Platform directly demonstrates: Kubernetes orchestration, Prometheus/Grafana observability, CI/CD pipelines, inference cost optimization, HPA scaling. It answers the core MLOps interview question — "how do you run ML in production?" — with concrete evidence.

Supporting evidence to mention: the cost reduction story (baseline → with caching → with routing), the alerting strategy, how you'd handle model version rollbacks.

**AI Platform role:**
**Best project: Advanced RAG System + Production AI Platform together**

Why: AI Platform roles (building the infrastructure other engineers use to deploy AI) care about scalability, API design, evaluation rigor, and abstraction quality. The RAG system shows evaluation framework thinking (MRR@5, held-out eval set, re-ranker ablation). The Production Platform shows the infrastructure layer. Combined, they show you can build systems that are both rigorous (eval) and production-ready (infra).

Supporting evidence to mention: how the RAG system's ingestion pipeline could serve as a platform primitive, how the eval framework could be generalized to other retrieval tasks, what the API contract between ingestion/retrieval/generation looks like.

**Summary table:**

| Role | Lead project | Supporting project | Key signal to emphasize |
|---|---|---|---|
| FDE | Multi-Agent Platform | RAG System | Scope + security + customer story |
| MLOps | Production AI Platform | Python Backend | Reliability + cost + observability |
| AI Platform | RAG System | Production AI Platform | Eval rigor + API design + scalability |
