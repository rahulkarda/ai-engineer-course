# Week 11 Day 1 Solutions — The Project Deep-Dive Interview Structure

## Exercise 1: Apply the 5-Part Deep-Dive Structure to a Capstone

**Prompt:** Map one of your capstone projects to the 5-part deep-dive structure.

**Model answer using the Multi-Agent Operations Platform:**

**Part 1 — Context & Problem (60 sec)**
"Our goal was a production-grade multi-agent orchestration platform. The core problem: existing agent frameworks had no separation between orchestration logic and tool execution, which made scaling and security auditing impossible. We had no budget for commercial solutions, so we built on LangGraph + FastAPI."

**Part 2 — Architecture Decision (90 sec)**
"Key decision: use a centralized MCP server as the tool gateway rather than letting each agent call tools directly. This gave us a single audit surface, rate-limiting, and credential isolation. Trade-off was added latency (~20ms per tool call) and a new SPOF, which we mitigated with Redis-backed task queues."

**Part 3 — The Hard Part (90 sec)**
"Hardest problem: agent state drift under concurrent execution. Two agents modifying shared state produced race conditions LangGraph's checkpointer didn't prevent. We added optimistic locking at the state layer — agents read a version token, and writes fail-fast if the token is stale, triggering re-planning."

**Part 4 — Results (30 sec)**
"Platform handles 50 concurrent agent workflows at p95 < 800ms. Security audit surface reduced to one gateway. Zero credential leaks in 6 months of testing."

**Part 5 — Reflection (30 sec)**
"Would instrument distributed traces from day one — we retrofitted OpenTelemetry and it cost two weeks. That was the biggest time sink."

---

## Exercise 2: What "Judgment" Means in a Deep-Dive vs. System Design

**Prompt:** Describe the difference between how judgment is evaluated in a system design interview versus a project deep-dive interview.

**Model answer:**

System design: judgment = making sound decisions under time pressure with invented constraints. Interviewer provides the problem. Candidate has no skin in the game.

Project deep-dive: judgment = defending real decisions you already made, including the ones that didn't work. Interviewer has full information asymmetry — you know what actually happened, they don't.

Specific differences:

| Dimension | System Design | Project Deep-Dive |
|---|---|---|
| Constraints | Hypothetical | Real, lived |
| Stakes | None — you can change direction freely | You shipped it; you own the consequences |
| What fails | Not considering key trade-offs | Not knowing why you made your own choices |
| What earns credit | Structured thinking under pressure | Self-awareness + honest reflection |
| Red flag | Skipping non-functional requirements | "I just used what the tutorial used" |

Judgment in a deep-dive is demonstrated by: explaining why you chose option A over B (not just that you chose A), acknowledging what the choice cost you, and showing you'd update that decision with new information.

---

## Exercise 3: 2-Minute "Tell Me About a Project" Using STAR-for-Systems

**Prompt:** Write a 2-minute scripted answer to "tell me about your most technically complex project" using the STAR-for-Systems format.

**Model answer:**

**Situation:** "I built a production multi-agent operations platform as part of an intensive AI engineering program. The problem was real: agent frameworks available in open source had no production-grade tooling for security, observability, or concurrent state management."

**Task:** "My goal was a platform where multiple LLM agents could collaborate on complex workflows — code generation, research, planning — with full audit trails and no credential leakage between agents or users."

**Architecture decision (replacing 'Action' in standard STAR):** "Core design choice: a Model Context Protocol server as the sole gateway for all tool calls. Every agent routes through it. This gave us centralized rate-limiting, scoped credentials, and a single point for logging. The downside was latency — we accepted ~20ms overhead per tool invocation because the security gain justified it."

**Hard part:** "Race conditions in shared agent state. Two agents reading and writing the same workflow state concurrently produced silent corruptions. Fixed with optimistic locking: state reads return a version token, writes validate it, and stale writes trigger re-planning. This added complexity but made the system provably safe under concurrency."

**Result:** "50 concurrent workflows, p95 under 800ms, zero secret leaks in testing. The platform became the reference architecture for the other AI agents I built in the program."

**Reflection:** "If I restarted, I'd add distributed tracing on day one. We retrofitted OpenTelemetry after the fact — it worked, but cost two weeks we could have avoided."

*Total: approximately 110 seconds at normal speaking pace.*

---

## Exercise 4: 3 Signals That an Interviewer Wants Deeper Technical Detail

**Prompt:** List 3 behavioral signals that tell you an interviewer wants you to go deeper technical during a project deep-dive.

**Model answer:**

**Signal 1 — They repeat a word from your answer as a question.**
You say "we used a vector database." They say: "Vector database?" — with no follow-up. That's not confusion; it's an invitation. Go one level deeper: why that DB, what embedding model, what indexing strategy, what query latency.

**Signal 2 — They ask "how did you decide" rather than "what did you use."**
"What did you use for orchestration?" = they want the name.
"How did you decide on that orchestration approach?" = they want the decision process, alternatives considered, and trade-off reasoning. Switch into decision-log mode.

**Signal 3 — They go quiet and let you keep talking.**
In most interview contexts, silence after your answer means they're satisfied and about to move on. In a deep-dive, a technically engaged interviewer who stops asking follow-ups and just listens as you go deeper is signaling: keep going, you're on the right track. If they wanted a different direction they'd redirect. Silence here means depth is welcomed.

**Bonus signal:** They write something down. Interviewers taking notes mid-answer are tracking specifics they want to probe — a metric you mentioned, a technology name, a specific design choice. Expect a follow-up on whatever they wrote.
