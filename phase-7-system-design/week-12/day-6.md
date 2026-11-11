# Phase 7 · Week 12 · Day 6 — Capstone System-Design Document (4–5 hr deep session)

> Objective: Write a complete, portfolio-grade capstone system-design document for a large-scale enterprise AI assistant platform.  ·  Time: ~4–5 hr  ·  Prereqs: [[day-1]] through [[day-5]]

## Concept

The capstone design document is the single most important deliverable of this course. It demonstrates that you can synthesise every phase into a coherent, production-grade architecture. A strong capstone doc:

- Stands alone — a reader with senior-engineer background can follow it without asking you questions.
- Cites specific numbers — no vague "scale it horizontally."
- Shows trade-offs explicitly — states what was given up and why.
- Covers the full AI stack: data, models, inference, eval, observability, security, cost.
- Is defensible in a 45-minute design review.

## How it works internally

**Document structure (follow this exactly)**

```
1. Executive summary (1 page)
2. Problem statement & requirements (1–2 pages)
3. Architecture overview (1 page + diagram)
4. Component deep-dives (1 page each × 5–6 components)
5. Data model (0.5 page)
6. Scalability & capacity planning (1 page)
7. Observability & eval (1 page)
8. Security & compliance (0.5 page)
9. Cost model (0.5 page)
10. Trade-offs & alternatives considered (1 page)
11. Roadmap (0.5 page)
```

**Time budget for this session**

```
0:00–0:30   Requirements & non-functional (step 2)
0:30–1:00   Architecture overview + diagram (step 3)
1:00–2:30   Component deep-dives (step 4)
2:30–3:00   Data model + capacity math (steps 5+6)
3:00–3:30   Observability + security (steps 7+8)
3:30–4:00   Cost model + trade-offs + roadmap (steps 9–11)
4:00–4:30   Executive summary + polish
```

## Worked example — partial excerpt

The full capstone document is the Day 6 project artefact. Below is the architecture overview section as a worked model to calibrate your depth.

**System**: Enterprise AI Assistant Platform (EAAP) — a multi-tenant, multi-modal AI assistant for 50,000 enterprise employees.

**Architecture overview excerpt**

```
                  ┌─────────────────────────────────────────────┐
                  │              API Gateway (Kong)             │
                  │    Auth (OIDC) · Rate limit · WAF           │
                  └──────────────────┬──────────────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
       ┌──────▼──────┐        ┌──────▼──────┐       ┌──────▼──────┐
       │ Chat Service│        │Ingest Service│       │Admin Service│
       │ (FastAPI)   │        │  (FastAPI)   │       │  (FastAPI)  │
       └──────┬──────┘        └──────┬──────┘       └─────────────┘
              │                      │
       ┌──────▼──────┐        ┌──────▼──────┐
       │Orchestrator │        │  Chunker    │
       │(LangGraph)  │        │  + Embedder │
       └──────┬──────┘        └──────┬──────┘
              │                      │
       ┌──────▼──────┐        ┌──────▼──────┐
       │  LLM Gateway│        │ Vector DB   │
       │(claude-     │        │  (Qdrant)   │
       │ sonnet-5-5) │        │             │
       └─────────────┘        └─────────────┘

Shared infrastructure (all services):
  PostgreSQL (metadata) · Redis (cache + sessions) · S3 (documents)
  OpenTelemetry → Tempo · Prometheus → Grafana · Loki (logs)
```

**Key design decision excerpt**:

> The orchestrator uses LangGraph with typed state (`AgentState` as a Pydantic model) to enforce that every node returns well-typed data. This prevents silent state corruption across multi-step reasoning chains. The alternative — a simple loop with a dict — was rejected because it makes debugging production failures significantly harder.

## The Day 6 Project (your work)

Your deliverable is a complete capstone system-design document. Use the system defined in the Week Project (`project.md`): an enterprise AI assistant platform.

**Minimum section requirements**

| Section | Min length | Key elements |
|---|---|---|
| Executive summary | 200 words | Problem, approach, key trade-offs, expected outcomes |
| Requirements | functional + NFR table | Scale numbers, latency SLAs, availability target |
| Architecture overview | diagram + 300 words | Named services, data flows, shared infra |
| Component deep-dives | 6 components × 200 words | Tech choice + rationale for each |
| Data model | Entity-relationship list | Key fields, indexes, relationships |
| Capacity planning | Back-of-envelope math | QPS, storage, cost ballpark |
| Observability & eval | 200 words | Metrics, traces, RAGAS or equivalent |
| Security | 150 words | Auth, PII, prompt injection, tenant isolation |
| Cost model | Table | Per-component monthly estimate |
| Trade-offs | 3 major trade-offs | What you chose vs. alternative and why |
| Roadmap | 3 phases | Phase name, deliverables, timeline |

## Exercises

There are no separate exercises for Day 6 — **the document IS the exercise**. Write it. Use the time budget above. Do not skip sections.

**Submission checklist**
- [ ] All 11 sections present
- [ ] Architecture diagram included (Mermaid or ASCII)
- [ ] Capacity math shows specific numbers (not "it scales")
- [ ] Every technology choice has one sentence of rationale
- [ ] Security section mentions: OIDC auth, tenant isolation, prompt injection defence
- [ ] Cost model has a monthly total
- [ ] At least 3 explicit trade-offs documented

(Solutions: ../solutions/week-12/day-6.md — a complete model capstone document)

## Self-check
- [ ] I can write a complete system-design doc in 4–5 hours unaided
- [ ] My doc is defensible in a 45-minute live review
- [ ] Every section has concrete numbers, not adjectives
- [ ] I have explained why I made each major technology choice

## Supplementary
- [Google's Site Reliability Engineering book (free)](https://sre.google/sre-book/table-of-contents/)
- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)
- [The Staff Engineer's Path — Tanya Reilly](https://www.oreilly.com/library/view/the-staff-engineers/9781098118723/)
