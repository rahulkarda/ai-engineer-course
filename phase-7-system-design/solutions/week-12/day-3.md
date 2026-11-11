# Solutions — Week 12 · Day 3: Behavioral + Project Deep-Dive Mock Loop

## Exercise 1 — STAR: Learning new technology under deadline

### Model answer

**Situation**: Three weeks before the demo of our enterprise RAG platform, the PM confirmed the client required streaming responses — text appearing token-by-token in the UI, not a 5-second wait. None of the team had implemented server-sent events (SSE) with LangGraph before.

**Task**: I owned the backend query service. The deadline was fixed. I had to integrate SSE into an existing FastAPI service, wire it through the LangGraph agent's streaming output, and test it against the frontend in 5 days.

**Action**: I spent day 1 reading the Anthropic streaming API docs and FastAPI SSE docs end-to-end, then wrote a spike: a standalone FastAPI endpoint that streamed `claude-sonnet-5-5` responses via `EventSourceResponse`. Day 2 I mapped LangGraph's `astream_events` API to the SSE format — each `on_llm_new_token` event maps to one `data: {...}` SSE frame. Day 3 I integrated it into the existing endpoint without breaking the non-streaming fallback. Day 4 was joint testing with the frontend engineer and bug fixing (chunked encoding headers, CORS for EventSource). Day 5 was load testing: 50 concurrent SSE connections with 4K-token responses, measuring memory leak risk.

**Result**: Streaming was live in production 2 days before the client demo. The client called it "a completely different experience." Response perceived latency dropped from 4.8 s (time to first byte) to 280 ms. I wrote an internal tutorial that 3 other engineers used in subsequent projects. Key learning: spikes are non-negotiable when learning a new integration under deadline — a working spike reduces estimate uncertainty from ±100% to ±20%.

---

## Exercise 2 — Anti-pattern rewrite

### Original (weak)
> "We built an agent that answered customer questions. It worked really well and everyone was happy with it. I helped with some of the coding."

### Improved (STAR-compliant)

**Situation**: Our customer support team was spending 4 hours/day answering repetitive policy questions that were already documented in 3,000 internal PDFs.

**Task**: I was responsible for designing the retrieval pipeline — the component that fetches relevant document chunks and feeds them to the LLM.

**Action**: I implemented a two-stage retrieval pipeline: dense retrieval via BGE-m3 embeddings in Qdrant (top-20), followed by a cross-encoder reranker that re-scored and filtered to top-5 chunks. I chose this over pure BM25 because our policy docs used domain-specific terminology that BM25 struggled with. I also built the offline eval harness using RAGAS, which let us iterate on chunk size and reranker threshold in 2 days instead of manual spot-checking.

**Result**: The agent handled 68% of incoming support tickets without human escalation (measured over 4 weeks). Support team time on repetitive questions dropped from 4 hr/day to 40 min/day. RAGAS faithfulness score was 0.91. I learned that evaluation infrastructure built before deployment, not after, is what makes iteration fast.

---

## Exercise 4 — Mock drill self-scoring guide

**Scoring production incident answer**

| Criterion | Score |
|---|---|
| I/we ratio: mostly "I" (< 20% "we") | 3 |
| Specific description of the bug and my cause | 3 |
| Specific steps I took during the incident | 3 |
| Quantified impact (downtime, affected users, revenue) | 3 |
| Fix described concretely (not "we fixed it") | 3 |
| Process change implemented afterward | 3 |
| Evidence it worked (e.g., 0 recurrences in 6 months) | 3 |
| Result section present and specific | 3 |
| Total | 24 |

Scale: 22–24 = strong; 16–21 = adequate; < 16 = needs work.

**Common issues in failure story answers**:
- Passive voice: "the system crashed" → replace with "my change to the chunking batch size caused the embedding worker to OOM"
- No process change: every failure story must end with a systemic fix, not just "I fixed the bug"
- Vague result: "it was better" → "zero recurrences over the next 6 months; we added the metric to the weekly reliability review"

---

## Exercise 5 — Failure STAR construction guide

**Minimum elements for a credible failure story**

1. **Mistake stated without hedging**: "I deployed without reading the migration script. It dropped a NOT NULL constraint that 3 downstream services depended on."
2. **Immediate fix**: "I rolled back the migration within 8 minutes using our blue-green deployment rollback procedure."
3. **Process change**: "I added a mandatory migration review checklist to our PR template — 5 questions that surface schema contract changes."
4. **Evidence it worked**: "We've run 47 migrations since; zero production schema incidents. The checklist takes 5 minutes per PR."

**Tone calibration**: Own it fully, early. "I made a mistake" — not "there was an issue." Interviewers give credit for honest ownership.
