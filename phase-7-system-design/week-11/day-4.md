# Phase 7 · Week 11 · Day 4 — GitHub & Portfolio Polish

> Objective: Transform your GitHub profile and portfolio repositories into a credible, navigable technical showcase for engineering hiring managers.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]], [[day-3]]

## Concept

A GitHub profile is reviewed in two passes. The first pass (30 seconds) establishes signal: Does this person write real code? Are they active? Do their projects look intentional? The second pass (5–10 minutes) digs into one or two repositories. If the first pass fails, the second never happens.

The good news: most engineers' GitHub profiles fail the first pass on fixable issues — pinned repos with no READMEs, last commit two years ago, no profile README. These are polish problems, not substance problems. You have the substance from the course. This lesson is about surfacing it.

## What Hiring Managers Look For

Based on what engineering managers consistently report reviewing:

1. **Pinned repositories** — the six pinned repos are the portfolio. If they are not pinned, the profile looks inactive even if the work exists.
2. **README quality on pinned repos** — the first signal of communication skill. A missing or stub README suggests either the project is incomplete or the engineer doesn't think about others reading their code.
3. **Commit history** — not frequency, but recency and spread. A repo with 200 commits in one weekend is suspicious. Organic development looks different from a tutorial paste.
4. **Activity graph** — recent activity signals that you are actively building, not coasting. Even documentation commits count.
5. **Repo naming and organization** — `project-1-final-v3-actual` signals disorganization. Descriptive names (`rag-evaluation-pipeline`, `multi-agent-ops-platform`) signal intentionality.

What hiring managers do not heavily weight: follower count, stars on your own repos, language distribution chart.

## The 5-Part README Formula for AI/ML Projects

Every pinned repository needs a README that answers five questions in order:

### Part 1 — What it does (2–3 sentences)
State the problem and solution. No background lore. The first sentence should be something a hiring manager can quote to a colleague.

Example:
> "Production multi-agent orchestration platform where autonomous LLM agents coordinate via structured tool calls to execute multi-step workflows. Built on LangGraph with MCP tool integration and Redis-backed checkpointing for fault-tolerant execution. Handles supervisor-specialist routing with typed Pydantic state at every boundary."

### Part 2 — Why it is interesting / what makes it non-trivial (3–5 bullets)
This is where you differentiate from tutorial projects. Name the technical problems you solved that a beginner wouldn't encounter.

Example:
> - Conditional graph branching for per-failure-type retry strategies (not a single global retry)
> - Dynamic tool discovery at runtime via MCP — agents are not coupled to tool implementations
> - State checkpointing after every node transition enables resume-from-failure without replay
> - A2A inter-agent messaging with schema validation; agents run in separate processes

### Part 3 — Architecture diagram
A single diagram showing the main components and data flow. Does not need to be elaborate. A draw.io export, a Mermaid diagram in the README, or even a clean ASCII diagram is acceptable. Its presence signals that you think in systems.

### Part 4 — Quickstart (exact commands, 5 lines or fewer)
If a reviewer cannot run the project in under 5 minutes, most will not try. The quickstart must be literally copy-pasteable.

```
git clone https://github.com/yourname/multi-agent-ops-platform
cd multi-agent-ops-platform
cp .env.example .env  # add your API key
pip install -e ".[dev]"
pytest tests/unit  # should pass without external services
```

### Part 5 — Results / metrics
The most commonly missing section. Include at least one number: benchmark results, RAGAS scores, latency measurements, cost measurements, load test output. A screenshot of a passing eval dashboard is worth 200 words.

Example:
> | Metric | Value |
> |---|---|
> | Tool call failure recovery rate | 94% automated |
> | Supervisor routing accuracy | 0.91 F1 on 50-task eval set |
> | P95 end-to-end latency (5-step workflow) | 4.2s |
> | Checkpoint overhead | <8ms per node transition |

## The GitHub Profile README

The profile README (`github.com/yourname` shows it) is a 150-word personal brand statement. Most engineers either skip it or write a generic "I love coding!" placeholder.

Effective profile README structure:
1. **One-sentence identity** — role, domain, what you build
2. **Current focus** — 1–2 sentences on what you are working on or learning
3. **Pinned project callout** — name one project and one number from it
4. **Contact / availability** — if you are open to opportunities, say so directly

Example:
> AI engineer specializing in LLM orchestration, multi-agent systems, and production RAG pipelines. Currently building distributed agent platforms with LangGraph and MCP.
>
> Recent work: a multi-agent ops platform with 94% automated failure recovery across tool call failures; an evaluation-driven RAG system reaching 0.84 RAGAS answer relevancy on a 500-document technical corpus.
>
> Previously: [prior role or education, one line].
>
> Open to senior AI engineer roles in applied LLM products. Contact: [email or LinkedIn].

## Pinned Repo Strategy for AI Engineers

You have 6 pin slots. Suggested allocation for this course's portfolio:

| Slot | Project | Why this order |
|---|---|---|
| 1 | Multi-Agent Operations Platform | Highest technical complexity; most differentiated |
| 2 | Advanced RAG System | High industry demand; concrete eval metrics |
| 3 | Production AI Platform | Shows production-readiness and ops maturity |
| 4 | Python Backend Platform | Shows fundamentals; relevant for any backend-adjacent AI role |
| 5 | Any significant side project | Signals genuine interest beyond coursework |
| 6 | Profile README repository | Keeps the profile README visible in the pin row |

Order rationale: put your most complex, most differentiated project first. Hiring managers scan left-to-right on pin rows. The first two pins are what get clicked.

## Commit Message Quality as a Signal

Hiring managers and senior engineers who review repos often look at commit history. Commit messages signal discipline and communication. Two patterns that read badly:

- `fix`, `update`, `stuff`, `wip`, `asdf` — signals no discipline
- One massive commit per project (`initial commit` with 3,000 lines) — signals tutorial paste rather than iterative development

Conventional commits format (from the course) reads well:
```
feat(retrieval): add BM25 sparse retrieval alongside dense vectors
fix(checkpointing): handle Redis connection timeout on node transition
docs(readme): add architecture diagram and benchmark results
test(rag): add integration test for hybrid retrieval with 50-query eval set
```

Each commit message tells a micro-story. A reviewer scrolling the history sees an engineer who works incrementally and communicates their intent.

If your current repos have poor commit history, you cannot rewrite it without force-pushing — which creates its own red flags. Instead, create a clean branch or repo for the portfolio version and commit it organically as you polish it.

## Demo Artifacts

For each pinned repo, consider adding:

- **Architecture diagram** — in `/docs/architecture.png` or inline in README via Mermaid
- **Benchmark screenshot** — RAGAS eval output, load test results, cost dashboard
- **Eval dashboard screenshot** — if you built a dashboard for the Production AI Platform, screenshot it and include it in the README

These artifacts do two things: they prove your numbers are real, and they give a visual anchor that reviewers remember when they scan multiple profiles.

## Worked Example — README Outline for Multi-Agent Operations Platform

```markdown
# Multi-Agent Operations Platform

Production LLM agent orchestration with fault-tolerant execution,
dynamic tool discovery, and typed state management.

## Why this is non-trivial
- LangGraph state graph with conditional branching for typed failure recovery
- MCP integration: agents discover tools at runtime, not at compile time
- Supervisor-specialist architecture isolates failure blast radius
- Redis checkpointing: workflows resume from last successful node, not from start
- A2A protocol: agents run as independent processes, communicate via schema-validated messages

## Architecture
[diagram here — Mermaid or PNG]

## Quickstart
git clone ...
cp .env.example .env
pip install -e ".[dev]"
pytest tests/unit

## Results
| Metric | Value |
|---|---|
| Automated recovery from tool failures | 94% |
| Supervisor routing F1 | 0.91 |
| P95 latency, 5-step workflow | 4.2s |

## Key technical decisions
- [Link to ADRs or brief inline decision log]

## Tech stack
Python 3.12, LangGraph, MCP, Redis, FastAPI, pytest, Docker
```

This outline is navigable in 30 seconds and gives a hiring manager everything they need to decide whether to read the code.

## Exercises

1. Audit your current GitHub profile against the 5-point hiring manager checklist (pinned repos, README quality, commit history, recent activity, naming). Write a prioritized fix list with a time estimate for each item.

2. Write Part 1 and Part 2 of the 5-part README for your Advanced RAG System. Part 1: 2–3 sentences. Part 2: 4–5 non-trivial technical bullets. Do not describe what the project is — describe what problems you solved that a beginner wouldn't face.

3. Write a GitHub profile README for yourself following the 150-word structure above. Include at least one metric from a portfolio project.

4. Review the commit history of one of your course project repositories. Identify 3 commits with weak messages. Write the conventional-commit-format version of each, as if you were retroactively describing the change.

(Solutions: ../solutions/week-11/day-4.md)

## Self-check

- [ ] I can list the 5 things engineering hiring managers look for in a GitHub profile and explain why each matters
- [ ] I can write a complete 5-part README outline for any portfolio project from memory
- [ ] I have a draft GitHub profile README under 150 words with at least one metric
- [ ] I understand the pinned repo ordering strategy and can justify the ranking for my own projects

## Supplementary

- [GitHub Profile README examples — abhisheknaiidu/awesome-github-profile-readme](https://github.com/abhisheknaiidu/awesome-github-profile-readme) — curated gallery of effective profile READMEs; useful for calibrating tone and layout
- [Mermaid diagram syntax for README architecture diagrams](https://mermaid.js.org/syntax/flowchart.html) — render architecture diagrams inline in GitHub Markdown with no image hosting required
- [Conventional Commits specification](https://www.conventionalcommits.org/en/v1.0.0/) — the commit message standard used throughout this course; read the spec to understand the full type vocabulary
