# Week 12 Assessment — Capstone Mock Interview & Design-Doc Review

> Comprehensive multi-type mock assessment. Pass threshold: 75% weighted.  
> Rubric adapted from the course standard (40% implementation → distributed across 5 interview dimensions).  
> Time: ~3 hr

## Adaptation note

The standard weekly rubric (40% independent impl, 25% conceptual, 20% debugging, 15% code quality) is adapted for a capstone assessment spanning 5 interview types. The four dimensions are mapped across the mock rounds as follows:

| Standard dimension | Capstone mapping |
|---|---|
| Independent implementation (40%) | Coding round (20%) + Capstone doc completeness (20%) |
| Conceptual understanding (25%) | LLD round (12%) + HLD round (13%) |
| Debugging & problem-solving (20%) | Architecture critique section (20%) |
| Code quality & testing (15%) | Behavioral round (8%) + Code quality within coding (7%) |

---

## Section 1 — Coding Round (20% of total)

**Time limit: 30 minutes. No external resources.**

Implement a priority queue that supports:
- `push(item: T, priority: int) -> None`
- `pop() -> T` (returns highest-priority item; ties broken by insertion order — FIFO)
- `peek() -> T | None`
- `__len__() -> int`

Requirements:
- Generic type `T`
- Thread-safe
- `pop()` on empty queue raises `QueueEmptyError` (define this exception)
- Full type hints
- Minimum 4 pytest cases: empty queue error, priority ordering, tie-breaking by insertion order, thread-safety (two concurrent threads push; verify order)

Acceptance criteria:
- All 4 pytest cases pass
- No busy-wait in thread safety implementation
- `push` and `pop` are O(log n)

---

## Section 2 — LLD Round (12% of total)

**Time limit: 20 minutes.**

Design a session manager for a multi-tenant AI assistant. It must:
- Store conversation history per user per session
- Support session expiry (TTL-based)
- Support listing active sessions for a user
- Be injectable (usable with Redis or in-memory backends interchangeably)

Deliverables:
1. Mermaid class diagram
2. Pydantic model for `Session` (fields: id, user_id, tenant_id, messages, created_at, expires_at)
3. `SessionStore` abstract base class with 4 methods
4. One concrete `InMemorySessionStore` implementation sketch (pseudocode OK for method bodies)

---

## Section 3 — HLD Round (13% of total)

**Time limit: 35 minutes.**

Design an AI-powered code review assistant that:
- Integrates with GitHub (webhook on PR creation)
- Analyses changed files using an LLM
- Posts inline comments on the PR
- Supports multiple programming languages
- Handles 500 PRs/day at peak, with review completion p95 < 5 min

Deliverables:
1. Requirements table (functional + non-functional)
2. High-level architecture diagram (ASCII or Mermaid)
3. API contract for the webhook handler
4. Data model for `ReviewJob`
5. Capacity estimate (queue depth, worker count, token cost)
6. Two failure modes and their mitigations

---

## Section 4 — Architecture Critique (20% of total)

**Time limit: 20 minutes.**

Analyse the following design description and answer the questions below.

> A company's AI assistant pipeline: user query arrives via REST → synchronous call to embedding service (separate microservice, no circuit breaker) → synchronous call to Qdrant → top-50 chunks concatenated into prompt → synchronous call to LLM (no streaming, 60s timeout) → response returned. All services share one Postgres instance for logging. Embeddings are re-generated at query time for every request (not cached). No retry logic anywhere. Tenant isolation is enforced by filtering on `tenant_id` in every SQL query.

Questions:
1. List 5 architectural flaws, each categorised as correctness, scale, or operational.
2. For each flaw, propose a specific fix (one sentence each).
3. Which flaw would you fix first given an on-call incident at 3am? Why?
4. Is the tenant isolation approach (`WHERE tenant_id = ?`) sufficient? What is the risk and the correct pattern?

---

## Section 5 — Behavioral Round (8% of total)

**Time limit: 15 minutes. Write your answers.**

Answer both prompts using STAR format. Minimum 3 paragraphs per answer.

**Prompt A**: "Describe a time you improved system reliability or performance with data, not instinct."

**Prompt B**: "Tell me about a time you had to balance technical debt reduction with feature delivery. What did you decide and what happened?"

---

## Section 6 — Capstone Design Doc Review (20% of total + 7% code quality)

Submit your capstone design document from Day 6.

Self-review it against this checklist. Score 1 point per checked item (max 27).

**Completeness (11 points)**
- [ ] Executive summary present
- [ ] Functional requirements listed
- [ ] Non-functional requirements with specific SLAs
- [ ] Architecture diagram (Mermaid or ASCII)
- [ ] At least 5 component deep-dives
- [ ] Data model with field types
- [ ] Capacity planning with numbers
- [ ] Observability section (metrics, traces, logs named)
- [ ] Security section (auth, PII, prompt injection)
- [ ] Cost model with monthly total
- [ ] Roadmap with 3 phases

**Quality (8 points)**
- [ ] Every technology choice has a one-line rationale
- [ ] At least 3 explicit trade-offs documented
- [ ] No section uses vague adjectives without numbers
- [ ] AI-specific concerns covered (eval, feedback loop, model versioning)
- [ ] Multi-tenancy addressed
- [ ] Failure modes addressed in at least 2 components
- [ ] Security section mentions prompt injection specifically
- [ ] Cost model includes LLM token cost

**Defensibility (8 points)**
- [ ] I can identify my weakest architectural decision
- [ ] I can justify the database choice under cross-examination
- [ ] I can explain the inference serving strategy
- [ ] I can defend the chunking/retrieval approach
- [ ] I can state the scaling bottleneck and its fix
- [ ] I can propose a concrete v2 improvement
- [ ] I can critique my own security model
- [ ] I have a clear answer to "how would you test this end-to-end?"

**Score: ___ / 27 (convert to 100 scale: score × 3.7)**

---

## Rubric (pass ≥ 75%)

| Dimension | Weight | Sections |
|---|---|---|
| Independent implementation | 40% | Coding (20%) + Capstone completeness (20%) |
| Conceptual understanding | 25% | LLD (12%) + HLD (13%) |
| Debugging & problem-solving | 20% | Architecture critique |
| Code quality & behavioral communication | 15% | Behavioral (8%) + code quality in coding (7%) |

**Weighted score formula**:

```
score = (coding_score × 0.20)
      + (lld_score × 0.12)
      + (hld_score × 0.13)
      + (critique_score × 0.20)
      + (behavioral_score × 0.08)
      + (capstone_score × 0.20)
      + (code_quality_score × 0.07)
```

Pass gate: weighted score ≥ 75.

Answer key: ../solutions/week-12/assessment.md
