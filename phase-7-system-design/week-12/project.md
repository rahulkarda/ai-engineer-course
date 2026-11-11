# Week 12 Project — Capstone: Enterprise AI Assistant Platform

> Portfolio-grade capstone project. Combine a complete system-design document with a full mock interview self-assessment.  
> Time: ~6–7 hr total (Day 6: 4–5 hr design doc · Day 7: 1–2 hr self-assessment)  
> Prereqs: All weeks 1–11; all days 1–5 of this week.

## Overview

The Week 12 project has two components that together form your course capstone portfolio artefact:

**Component A — Capstone System-Design Document**  
A complete, portfolio-grade design document for an Enterprise AI Assistant Platform (EAAP). This is the most important document you will produce in this course. It should be something you are willing to share in a technical interview or attach to a job application.

**Component B — Mock Interview Self-Assessment Report**  
A structured self-assessment across all five interview types (coding, LLD, HLD, behavioral, architecture critique), with scores, honest reflection, and a targeted improvement plan.

---

## Component A: Capstone System-Design Document

### System to design

**Enterprise AI Assistant Platform (EAAP)** — a multi-tenant AI assistant for 50,000 enterprise employees at a Fortune 500 company. The platform enables employees to query internal documents, generate code, summarise meetings, and complete tasks via tool use. It must be deployed on-premises (data never leaves the corporate network) and comply with SOC 2 Type II.

### Hard requirements for the document

**Scale targets** (use these numbers; do not invent others)

| Metric | Target |
|---|---|
| Users | 50,000 active employees |
| Daily active users | 30,000 |
| Peak concurrent sessions | 5,000 |
| Queries per second (peak) | 800 QPS |
| Document corpus | 20 million documents, 500 GB |
| Response latency (p50) | < 1.5 s |
| Response latency (p99) | < 5 s |
| Availability | 99.9% (< 8.7 hr/year downtime) |
| Data residency | On-premises (no cloud LLM APIs) |

**Mandatory capabilities**

1. Conversational Q&A over internal documents (RAG)
2. Code generation and explanation
3. Meeting summarisation (async, document upload)
4. Tool use (calendar, ticketing system, HR system integration)
5. Multi-tenant isolation (department-level)
6. Admin controls (enable/disable capabilities per department)
7. Full audit log of all interactions (compliance)
8. PII detection and redaction in inputs and outputs
9. Prompt injection defence

**Mandatory document sections**

Follow the 11-section structure from Day 6. All sections are required. Minimum word counts:

| Section | Min words |
|---|---|
| Executive summary | 250 |
| Requirements | 300 (table format) |
| Architecture overview | 400 + diagram |
| Component deep-dives (6 components) | 250 each |
| Data model | 200 |
| Capacity planning | 300 (must include math) |
| Observability & eval | 250 |
| Security & compliance | 300 |
| Cost model | 200 (must include monthly total) |
| Trade-offs | 300 (minimum 3 trade-offs) |
| Roadmap | 200 (3 phases with timelines) |

**Mandatory components to address**

- LLM inference: self-hosted (vLLM or TGI on GPU) — justify model choice between claude-equivalent open-source and proprietary on-prem options
- Embedding pipeline: freshness SLA, re-index strategy, approximate nearest-neighbour index
- Orchestration: LangGraph with typed state; explain why
- Session management: multi-tenant, TTL-based
- Guardrails: placement, implementation, latency impact
- Eval harness: RAGAS or equivalent, offline + online monitoring
- Audit log: schema, retention, query patterns
- Admin service: capability flags, per-department config

### Submission format

A single Markdown file saved at:
```
phase-7-system-design/week-12/capstone-design-doc.md
```

Include all Mermaid/ASCII diagrams inline.

---

## Component B: Mock Interview Self-Assessment Report

### Structure

Complete all five mini-rounds from Day 7 (or from the Assessment). For each round:

1. Write your answer (timed — no extensions).
2. Score yourself using the provided rubric.
3. Write 2–3 sentences of honest self-critique.
4. Identify the specific skill gap the score reveals.

**Self-assessment report format**

Save as:
```
phase-7-system-design/week-12/mock-interview-self-assessment.md
```

Include:

```markdown
# Mock Interview Self-Assessment — [Your Name] — [Date]

## Summary scorecard
| Round | Score (/100) | Pass/Fail |
|---|---|---|
| Coding | | |
| LLD | | |
| HLD | | |
| Behavioral | | |
| Architecture critique | | |
| Weighted total | | |

## Per-round reflection
### Coding
Score: ___
What went well: ...
What failed: ...
Root cause of gap: ...
Fix: ...

[repeat for each round]

## Overall reflection
Strongest area: ...
Weakest area: ...
One concrete action I will take in the next 2 weeks: ...
```

---

## Grading rubric

| Dimension | Weight | Criteria |
|---|---|---|
| Independent implementation (design doc completeness) | 40% | All 11 sections present, all hard requirements addressed, all mandatory components covered |
| Conceptual understanding | 25% | Architecture choices justified; trade-offs documented; AI-specific concerns (eval, feedback loop) addressed |
| Debugging & problem-solving (architecture critique) | 20% | Identifies own weak decisions; proposes specific improvements; cost model is realistic |
| Code quality & communication (self-assessment report) | 15% | Self-scores are calibrated; reflection is honest and specific; improvement plan is actionable |

**Pass gate: 75% weighted.**

**Exceptional work criteria (for portfolio differentiation)**

- Cost model includes sensitivity analysis (e.g., what if QPS doubles?)
- Security section includes a threat model table
- Roadmap includes migration strategy from an existing system
- At least one component deep-dive includes a working code sketch (Python 3.12+, typed)
- Self-assessment mock rounds were completed fully timed

---

## Deliverables checklist

- [ ] `capstone-design-doc.md` — all 11 sections, min word counts met
- [ ] `mock-interview-self-assessment.md` — all 5 rounds scored with reflection
- [ ] Architecture diagram present in design doc
- [ ] Capacity math visible (not hidden, actual numbers)
- [ ] Cost model has a monthly total
- [ ] Trade-offs section has at least 3 trade-offs with alternatives named
- [ ] Self-assessment identifies weakest area with a specific fix

---

## Reference implementations

Model solutions are in `../solutions/week-12/project.md`. Do not read until you have submitted both components.
