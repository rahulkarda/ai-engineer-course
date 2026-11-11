# Phase 7 · Week 11 · Day 2 — Turning Capstones into Interview Narratives: Decisions & Tradeoffs

> Objective: Extract the highest-signal decisions from each portfolio project and frame each as a structured tradeoff story.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

## Concept

Interviewers do not care what you built. They care what you decided and why. Two engineers who ship identical systems can give completely different interview signals: one recites the architecture, the other defends their choices under constraint. The second one gets the offer.

Decision logging is the practice of capturing not just what you chose, but what you rejected and what forced the choice. In an interview, this is your primary evidence of engineering judgment. You cannot reconstruct it on the fly. You prepare it.

## How it works internally

When an interviewer asks "why did you choose X?", they are running a diagnostic. Possible outcomes:

- **Informed choice**: "I chose X over Y because [specific constraint] made Y unsuitable. The cost of X was [named tradeoff], which was acceptable because [business or technical reason]." — This signals senior-level judgment.
- **Default choice**: "X is the standard tool for this." — Neutral. Shows you follow conventions.
- **Naive choice**: "I didn't consider anything else." — Red flag for senior roles.
- **Retroactive rationalization**: You didn't think about it at build time, but you're constructing a reason now. Interviewers can usually detect this because the reason doesn't connect to your actual constraints.

Build your decision log from your actual build process. Every time you paused during the course and chose one approach over another, that is a decision worth capturing.

## The Tradeoff Triangle for AI Systems

AI systems operate under three competing constraints. Every significant decision moves you on this triangle:

```
              Quality
             /       \
            /         \
          Latency --- Cost
```

- **Latency vs. Quality**: Larger models, deeper retrieval, multi-step chains all improve quality and increase latency. The choice is not binary — it is about acceptable degradation per user scenario.
- **Cost vs. Quality**: Embedding more documents, running re-rankers, chaining reasoning steps all cost tokens. Quality improvements have diminishing returns; you need to know where the cliff is.
- **Latency vs. Cost**: Caching reduces both, but cache invalidation for dynamic knowledge systems is a real problem. Batch processing reduces cost and increases latency.

Every decision in your portfolio projects can be placed on this triangle. Practice doing it out loud.

## Decision Log — All 4 Portfolio Projects

### Project 1: Python Backend Platform (FastAPI, SQL, testing, async)

| Decision | Rejected alternative | Driving constraint | Tradeoff accepted |
|---|---|---|---|
| Async FastAPI over Flask | Flask's sync model | Concurrent LLM calls needed cooperative multitasking | Async debugging is harder; stack traces less readable |
| Alembic migrations over raw SQL | Manual schema scripts | CI/CD required deterministic, repeatable migrations | Added dependency; migration files need careful ordering |
| pytest with factory fixtures over mocks | Direct mock patching | Test data consistency across 50+ tests | Fixture setup time per test run |
| Pydantic v2 for request/response validation | Manual validation | Type safety at API boundary with no runtime overhead | Pydantic v2 migration was a breaking change from v1 |

### Project 2: Advanced RAG System (LLMs, embeddings, retrieval, vector DB, evaluation)

| Decision | Rejected alternative | Driving constraint | Tradeoff accepted |
|---|---|---|---|
| Recursive character chunking over fixed-size | Fixed 512-token windows | Documents with variable structure (code + prose mixed) | Chunk size variance affects retrieval precision |
| Hybrid retrieval (BM25 + dense) over dense-only | Single vector search | Dense search alone missed exact-match technical terms | Two retrieval paths to maintain and tune |
| Cross-encoder re-ranking over score threshold | Top-k by cosine similarity | Bi-encoder scores don't compare across documents | Re-ranking adds ~200ms latency per query |
| RAGAS for eval over LLM-as-judge ad hoc | Manual spot checks | Needed reproducible, automated eval in CI pipeline | RAGAS requires reference answers; hard to source for novel queries |

### Project 3: Multi-Agent Operations Platform (LangGraph, MCP, A2A, security)

| Decision | Rejected alternative | Driving constraint | Tradeoff accepted |
|---|---|---|---|
| LangGraph over raw asyncio | Custom async orchestration | Needed built-in checkpointing and conditional branching | LangGraph's graph compilation step adds startup latency |
| MCP for tool exposure over hard-coded schemas | Per-agent tool registration | Dynamic tool discovery for runtime extensibility | Network hop per tool call; added schema validation overhead |
| Supervisor-specialist separation | Monolithic agent | Failure isolation: specialist failures shouldn't corrupt orchestration | Cross-agent communication adds serialization cost |
| A2A protocol for inter-agent messaging | Direct function calls | Agents needed to run in separate processes and nodes | Protocol overhead; more complex debugging |

### Project 4: Production AI Platform (Docker, k8s, observability, CI/CD, cost)

| Decision | Rejected alternative | Driving constraint | Tradeoff accepted |
|---|---|---|---|
| Kubernetes over Docker Compose for production | Compose is simpler | Auto-scaling and self-healing required for production SLA | k8s operational complexity; steep learning curve |
| OpenTelemetry for traces over custom logging | Ad hoc structured logs | Vendor-neutral instrumentation; exportable to any backend | OTEL adds SDK overhead; trace sampling config is non-trivial |
| Cost tagging per model call over aggregate billing | Single cost dashboard | Per-feature cost attribution needed for product decisions | Requires discipline in tagging at every LLM call site |
| Blue-green deployment over rolling updates | Simpler rollout | LLM-based features need instant rollback; rolling fails mid-inference | Doubles infrastructure cost during deployment windows |

## The "What Would You Do Differently?" Question

This question has two failure modes:

1. **"Nothing, it was well-designed."** — Signals lack of self-awareness. No project is perfect.
2. **"Everything was a mess."** — Undermines your own work and suggests poor judgment during the build.

The correct approach: identify one decision made under constraint that you'd revisit with better information, then pivot to what you'd keep and why.

Structure:
> "I'd change [specific decision] because [what I learned that I didn't know at build time]. The symptom that made this clear was [observable problem]. I'd keep [other decision] because [it proved out under load / turned out to be the right abstraction / aged well as requirements changed]."

This framing shows you can learn from experience without rewriting history.

## Scaling Story: From Prototype to Production

Interviewers at growth-stage companies often ask: "How would this scale to 10x/100x load?" You need concrete numbers, not hand-waving.

Template for each project:
1. **Current scale**: "This handles N requests/day, M documents indexed, P concurrent agents."
2. **First bottleneck**: "At 10x, [specific component] becomes the constraint because [why]."
3. **Mitigation**: "I'd address it by [specific architectural change], which buys [rough scale ceiling]."
4. **Next bottleneck**: "After that, the limit shifts to [next component]."

You don't need to solve all the way to Google scale. Naming the first two bottlenecks and their mitigations is sufficient for most AI engineer roles.

## Worked Example — Advanced RAG System Decision Log

**Scenario**: Interviewer asks, "Walk me through the most consequential architectural decision in your RAG system."

**Response:**

> "The most consequential decision was the retrieval architecture — specifically, whether to use dense-only retrieval or a hybrid BM25 + dense approach.
>
> I started with dense retrieval using a sentence-transformer model against a Qdrant vector store. In early evaluation with RAGAS, faithfulness scores were good but answer relevancy was lower than expected. Root cause: exact technical terms — function names, error codes, specific configuration keys — were getting semantically diluted in the embedding space. A query for 'ConnectionError 503' was returning documents about general network resilience instead of the specific error handler.
>
> I added BM25 as a parallel retrieval path using rank fusion to merge the two result sets before re-ranking. RAGAS answer relevancy improved from 0.71 to 0.84 on the evaluation set.
>
> The tradeoff: two retrieval systems to maintain, and the fusion weights (alpha for BM25 vs. dense score weighting) required empirical tuning — they're not theoretically derivable. I spent a day tuning them on a held-out eval set.
>
> What I'd do differently: I'd instrument the retrieval step with per-query logging from the start, so I could see which path was contributing for which query types. I inferred the exact-match problem from eval metrics; direct retrieval tracing would have found it faster.
>
> What I'd keep: the RAGAS pipeline in CI. Having an automated eval gate meant I could see regression within a pull request rather than in production."

## Exercises

1. Write a 3-row decision log for your Python Backend Platform. For each row: decision, rejected alternative, driving constraint, accepted tradeoff. Be specific about constraints — "it was simpler" is not a constraint.

2. Apply the tradeoff triangle to two decisions from your Advanced RAG system. For each, state which two vertices of the triangle were in tension and how you resolved the tension.

3. Draft a "what I'd do differently" answer for the Multi-Agent Operations Platform. Follow the structure: what to change, what prompted it, what to keep, why. Keep it under 90 seconds when spoken.

4. For your Production AI Platform, state the scaling story at two levels: first bottleneck at 10x, second bottleneck after that. Give concrete numbers (even rough estimates) for current scale.

(Solutions: ../solutions/week-11/day-2.md)

## Self-check

- [ ] I can explain why decision logging matters more than architecture description in an interview
- [ ] I can place any AI system decision on the latency/cost/quality triangle and name the tension
- [ ] I have a populated decision log (at least 3 rows) for each of the 4 portfolio projects
- [ ] I can answer "what would you do differently?" without undermining my own work

## Supplementary

- [Documenting Architecture Decisions — Michael Nygard](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions) — the original ADR format; read this before your interview to internalize the decision-log vocabulary
- [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — the paper behind the eval framework used in Project 2; useful for depth questions on RAG evaluation
- [Production Readiness Review checklist — Dropbox Engineering](https://dropbox.tech/infrastructure/athena-our-journey-to-a-unified-metrics-platform) — example of how production systems get reviewed; adapt the checklist to build your scaling story
